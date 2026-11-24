//! Flash Next's MTP head split over the two ranks (spec 090's scheme B, `ZSPARK_RANKS_HEAD=split`): each rank
//! holds half of the head as it holds half of a main layer (`model.Mtp.split`), and a pass is computed by both,
//! as TensorFold's `mtp.py` computes it under a world of 2. Three gathers a pass: the attention's partials and
//! the MoE's, each added rank 0 first and rounded once as a main layer's are (`ranks_forward.zig`), and the
//! scored row, each rank's part of the draft ids (TensorFold gathers each rank's 32 best there; the row is 156
//! KiB and leaves the choice, on the device and read back, what one rank's is). Each rank's `mtp.Head` chooses
//! the drafts from that row, the same on both (spec 095): in a reply's rounds rank 1 drafts as rank 0 does and
//! is told nothing (`session/pair.zig`'s `Rounds`). Elsewhere (a prompt's windows, a reply under a constraint)
//! rank 0's head runs the passes and tells rank 1 each (`Op.head`), which runs it on its half. A told pass
//! carries the head's position, so a chain's entries dropped on rank 0, a `reset` and a `rewind` reach rank 1's
//! cache with the next pass and need no operation of their own.
//!
//! Nothing here chooses a reply's token: a draft is verified by the trunk, so the split head's bits may differ
//! from the whole head's (the partials' sum is rounded once) and the reply is the two ranks' serial one.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const cache_file = @import("../../decode/graphs.zig");
const forms = @import("../../decode/forms.zig");
const ranks_file = @import("../../ranks/ranks.zig");
const forward = @import("forward.zig");
const hc = @import("blocks/hc.zig");
const moe = @import("blocks/moe.zig");
const mtp = @import("mtp.zig");
const partials = @import("partials.zig");
const ranks_forward = @import("ranks_forward.zig");
const share_file = @import("share.zig");

const world: usize = share_file.world;

/// `Op.head`'s flags.
pub const scored: u32 = 1;
pub const launched: u32 = 2;

/// The split head over `held` ranks in one process.
pub fn Held(comptime held: usize) type {
    const H = ranks_forward.Held(held);
    return struct {
        pub const Error = H.Error || Allocator.Error;

        const Half = struct { state: mtp.State, b: mtp.Buffers };

        pub const Split = struct {
            m: H.Model,
            /// The trunk's buffers: a window's final streams are the first rows of each rank's `h`, whole there.
            trunk: *const H.Buffers,
            /// Each rank's cache and scratch: rank 0's `mtp.Head`'s own where it is here, else `own`'s.
            state: [held]*mtp.State,
            b: [held]*mtp.Buffers,
            own: [held]?Half,
            /// fp32 [world, rows, hidden]: each rank's partial of the attention's branch, and of the MoE.
            branch: cuda.Buffer,
            moe_parts: cuda.Buffer,
            /// bf16 [rows, hidden]: the branch's sum, rounded once.
            branch_bf16: cuda.Buffer,
            /// fp32 [rows, hidden]: the MoE's sum before its write-back rounds it.
            moe_sum: cuda.Buffer,
            /// bf16 [1, the widest rank's ids]: what a rank other than 0 scores, before it is laid in the gathered
            /// row at its columns (a linear writes to an aligned address, which a part's first column is not).
            parts: [held]cuda.Buffer,
            /// Null launches each kernel from the host.
            graphs: ?Graphs,
            rows: usize,
            capacity: usize,

            /// In place: `head`, the first held rank's, is given `self` to run its passes, and both stay where
            /// they are. `m.lead` and `m.gather` are set already. `graphs` false launches every kernel.
            pub fn init(self: *Split, gpa: Allocator, m: H.Model, trunk: *const H.Buffers, head: ?*mtp.Head, rows: usize, capacity: usize, graphs: bool) !void {
                std.debug.assert(!m.alone and m.gather.serves(held, world));
                for (m.ranks) |one| std.debug.assert(one.mtp.?.split != null and one.mtp.?.draft != null);
                self.m = m;
                self.trunk = trunk;
                self.rows = rows;
                self.capacity = capacity;
                // a failed init leaks what it made: the process is about to stop on that error
                for (0..held) |i| {
                    if (i == 0) if (head) |lead| {
                        self.own[i] = null;
                        self.state[i] = &lead.state;
                        self.b[i] = &lead.b;
                        continue;
                    };
                    self.own[i] = .{ .state = try .init(m.ranks[i], capacity), .b = try .init(gpa, m.ranks[i], rows, capacity) };
                    self.state[i] = &self.own[i].?.state;
                    self.b[i] = &self.own[i].?.b;
                }
                const hidden = m.ranks[0].shape.hidden;
                self.branch = try .init(.f32, &.{ world, rows, hidden });
                self.moe_parts = try .init(.f32, &.{ world, rows, hidden });
                self.branch_bf16 = try .init(.bf16, &.{ rows, hidden });
                self.moe_sum = try .init(.f32, &.{ rows, hidden });
                // the last rank's part may be an id short of its columns: what is gathered there is zeros
                for (self.b) |b| try b.draft_logits.?.zero();
                for (&self.parts) |*part| {
                    part.* = try .init(.bf16, &.{ 1, m.ranks[0].mtp.?.split.?.widest() });
                    try part.zero();
                }
                self.graphs = if (graphs) try .init(gpa) else null;
                if (head) |lead| lead.split = .{ .context = self, .pass = passed };
            }

            pub fn deinit(self: *Split) void {
                if (self.graphs) |*graphs| graphs.deinit();
                inline for (.{ "branch", "moe_parts", "branch_bf16", "moe_sum" }) |name| @field(self, name).deinit();
                for (&self.parts) |*part| part.deinit();
                for (&self.own) |*half| if (half.*) |*one| {
                    one.b.deinit();
                    one.state.deinit();
                };
                self.* = undefined;
            }

            /// `mtp.Head`'s pass: rank 1 is told where rank 0 tells it (`m.lead`), then every rank here runs it.
            /// The head advances its own position after it.
            fn passed(context: *anyopaque, own: ?usize, tokens: []const i32, score: bool, wide: bool) anyerror!void {
                const self: *Split = @ptrCast(@alignCast(context));
                const position = self.state[0].position.host;
                const flags = (if (score) scored else 0) | (if (wide) launched else 0);
                if (self.m.lead) |lead| try lead.tell(@backingInt(ranks_forward.Op.head), &.{ @intCast(position), if (own) |row| @intCast(row + 1) else 0, flags }, tokens);
                for (1..held) |i| try self.stand(i, position);
                try self.run(own, tokens, score, wide);
                for (1..held) |i| try self.state[i].position.advance(tokens.len);
            }

            /// A pass rank 0 told (`session/mirror.zig`): every rank here from the head's `position`, rank 1's
            /// own head standing by.
            pub fn follow(self: *Split, position: usize, own: ?usize, tokens: []const i32, score: bool, wide: bool) Error!void {
                std.debug.assert(self.m.lead == null);
                for (0..held) |i| try self.stand(i, position);
                try self.run(own, tokens, score, wide);
                for (0..held) |i| try self.state[i].position.advance(tokens.len);
            }

            /// Rank `i`'s cache at the entries rank 0's head holds: fewer than its own after a chain's drafts
            /// were dropped, or after a `reset` or a `rewind` there. The entries below stay, as rank 0's do.
            fn stand(self: *Split, i: usize, position: usize) cuda.Error!void {
                if (self.state[i].position.host == position) return;
                // a launch that reads the position may still be running
                try cuda.synchronize();
                try self.state[i].position.set(position);
            }

            fn run(self: *Split, own: ?usize, tokens: []const i32, score: bool, wide: bool) Error!void {
                const rows = tokens.len;
                std.debug.assert(rows >= 1 and rows <= self.rows and (own == null or (rows == 1 and own.? < self.rows)));
                // The pass's gathers are on this thread's stream, or a graph's, and the trunk's window before it
                // on another: it is over before these are enqueued, here as on the other rank, so the ranks'
                // gathers meet in one order.
                try cuda.synchronize();
                for (0..held) |i| try mtp.stage(self.m.ranks[i], self.b[i], tokens);
                const graphs: ?*Graphs = if (!wide) if (self.graphs) |*g| g else null else null;
                if (graphs) |g| try g.run(self, own, rows, score) else try self.compute(own, rows, null, score);
                // And over before the trunk's next window is enqueued, on every rank: rank 0 reads a scored
                // pass's choice, which waits for it, and rank 1 reads nothing.
                try cuda.synchronize();
            }

            /// The device's part, launches only: `mtp.compute`, the layer as `ranks_forward.launch` runs a main
            /// one. `own`: the row of each rank's own streams the pass reads, null the trunk's first `rows`.
            fn compute(self: *Split, own: ?usize, rows: usize, bound: ?usize, score: bool) H.Error!void {
                const m = self.m;
                if (m.gather == .nccl) std.debug.assert(m.gather.nccl.rank == m.first);
                const shape = m.ranks[0].shape;
                var sites: [held]forward.Site = undefined;
                for (0..held) |i| {
                    const b = self.b[i];
                    const streams = if (own) |row| b.layer.h.span(row, 1) else self.trunk.ranks[i].h.rows(rows);
                    try mtp.front(m.ranks[i], b, streams, rows);
                    sites[i] = .{ .attention = .{ .cache = self.state[i].cache, .position = self.state[i].position, .bound = bound } };
                    try hc.writeback(b.layer.h.rows(rows), b.layer.pss.view, .none, shape.streams);
                }
                for (0..held) |i| {
                    const b = &self.b[i].layer;
                    try forward.readout(m.ranks[i].mtp.?.layer.attn_hc, b.*, rows, shape, b.inject_attn);
                }
                for (0..held) |i| {
                    const b = &self.b[i].layer;
                    const part = ranks_file.partial(self.branch.view, m.first + i, rows);
                    _ = try forward.runBlock(m.ranks[i], m.ranks[i].mtp.?.layer, 0, sites[i], b, rows, null, b.mixed.rows(rows), part);
                }
                const branch = ranks_file.gathered(self.branch.view, rows);
                try m.gather.run(branch);
                try partials.roundSum(branch, self.branch_bf16.rows(rows));
                for (0..held) |i| {
                    const b = &self.b[i].layer;
                    const landing: hc.Pending = .{ .branch = .{ .branch = self.branch_bf16.rows(rows), .inject = b.inject_attn.view } };
                    try hc.writeback(b.h.rows(rows), b.pss.view, landing, shape.streams);
                }
                const slots = shape.top + 1;
                for (0..held) |i| {
                    const b = &self.b[i].layer;
                    const layer = m.ranks[i].mtp.?.layer;
                    try forward.readout(layer.mlp_hc, b.*, rows, shape, b.inject_mlp);
                    try moe.forward(layer.moe, b.moe, b.mixed.rows(rows), rows, forms.none);
                    try partials.fold(b.moe.y.rows(rows * slots), b.moe.shared_y.rows(rows), b.moe.weights.rows(rows), ranks_file.partial(self.moe_parts.view, m.first + i, rows));
                }
                const folded = ranks_file.gathered(self.moe_parts.view, rows);
                try m.gather.run(folded);
                for (0..held) |i| {
                    const b = &self.b[i].layer;
                    const sum: hc.Pending = .{ .ranks_sum = .{ .gathered = folded, .inject = b.inject_mlp.view, .into = self.moe_sum.rows(rows) } };
                    try hc.writeback(b.h.rows(rows), b.pss.view, sum, shape.streams);
                    try forward.readout(m.ranks[i].mtp.?.mixer, b.*, rows, shape, null);
                }
                if (!score) return;
                // Each rank scores its part of the draft ids into its columns of one row: the first held
                // rank's, where every rank is here, and each its own, gathered, where one is.
                const row = self.b[0].draft_logits.?;
                const widest = m.ranks[0].mtp.?.split.?.widest();
                for (0..held) |i| {
                    const part = m.ranks[i].mtp.?.draft.?;
                    const to = if (m.first + i == 0) row.view.ptr else self.parts[i].view.ptr;
                    try part.linear.forward(self.b[i].layer.mixed.span(rows - 1, 1), .init(to.?, .bf16, &.{ 1, part.ids.len }));
                }
                switch (m.gather) {
                    .loopback => for (0..held) |i| {
                        if (m.first + i != 0) try row.copyAt((m.first + i) * widest * 2, self.parts[i], widest * 2);
                    },
                    // rank 0's part is in place, as `ranks.Gather.run` sends a partial; another's from its own
                    .nccl => |comm| try comm.allGather(if (m.first == 0) cuda.View.init(row.view.ptr.?, .bf16, &.{ 1, widest }) else self.parts[0].view, row.view),
                }
                // each rank chooses alike: what its choice needs of the row, as one rank's head leaves it
                try mtp.pick(mtp.scoredView(m.ranks[0].mtp.?, self.b[0].*), self.b[0].picked.view);
            }
        };

        /// `Split.compute` as captured graphs, keyed as `mtp.Graphs`: the rows, the context's bucket, where the
        /// streams are read from and whether the last row is scored. Over NCCL a graph holds its gathers among
        /// the kernels, so both ranks capture and launch the same graphs in step.
        pub const Graphs = struct {
            cache: cache_file.Cache(Key),

            const Key = struct { rows: usize, context: usize, own: usize, score: bool };

            pub fn init(gpa: Allocator) cuda.Error!Graphs {
                return .{ .cache = try .init(gpa) };
            }

            /// The graphs hold pointers into the split's models, states and buffers: drop them first.
            pub fn deinit(self: *Graphs) void {
                self.cache.deinit();
                self.* = undefined;
            }

            fn run(self: *Graphs, split: *Split, own: ?usize, rows: usize, score: bool) Error!void {
                const context = forward.Graphs.bucket(split.b[0].layer, split.state[0].position.host + rows);
                const key: Key = .{ .rows = rows, .context = context, .own = if (own) |row| row + 1 else 0, .score = score };
                return self.cache.run(key, Split.compute, .{ split, own, rows, context, score });
            }
        };
    };
}
