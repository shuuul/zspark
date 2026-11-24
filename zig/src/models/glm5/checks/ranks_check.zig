//! `zspark ranks-check random rank=R at=ADDR:PORT [seed=N]`: a rank a process over NCCL against both ranks in
//! one process, bit for bit (spec 045 WS-02). Both ranks make the tests' random GLM from one seed
//! (`random.zig`), join through the control channel, and run the same windows twice: in loopback in this
//! process, which is the reference for two devices, and as this rank's own share with the other's partials
//! gathered over NCCL. Each way launched, stopping at every gather to read, and through captured graphs; then
//! the MTP layer's steps, and a random DFlash2 drafter's passes and drafts both ways
//! (`ranks_check_drafter.zig`, spec 045 WS-09). Everything this rank holds is compared: both ranks' gathered partials of every
//! branch, its residual rows after every layer, its taps, final rows and half of the logits, its KDA states
//! and latents. The first difference is named and the command ends with 1, on both ranks. The gathered
//! buffers are filled before every window, with other bytes on the two ways, so a rank's rows read from where
//! this window wrote nothing differ too.
//!
//! The windows are a prompt chunk of the model's most rows (KDA's wide chain), serial steps on both sides of
//! the KDA states (a captured graph launched again), verify windows cut short, and a window after them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const channel_file = @import("../../../ranks/channel.zig");
const check = @import("../../../ranks/check.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const held_file = @import("../layers/ranks_forward.zig");
const layout = @import("../layers/layout.zig");
const mtp = @import("../mtp/mtp.zig");
const pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const random_file = @import("random.zig");
const drafter_check = @import("ranks_check_drafter.zig");
const trace_file = @import("buffer_trace.zig");
const whole = @import("ranks_whole.zig");
const ranks_states = @import("ranks_states.zig");
const Trace = trace_file.Trace;
const atEnd = trace_file.atEnd;
const differs = trace_file.differs;

const world = layout.world;
const fx = random_file.fx;
const capacity = 256;
/// The layers whose output the buffers keep, as a drafter would ask.
const tap_layers = [_]u32{ 1, 3, 4 };
/// A window's rows and how many of them are kept, from an empty sequence.
const plan = [_][2]usize{ .{ fx.max_rows, fx.max_rows }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 6, 3 }, .{ 9, 9 }, .{ 1, 1 }, .{ 16, 4 }, .{ 33, 33 }, .{ 2, 2 } };
/// The MTP layer after them: the last window's rows absorbed, then drafts chained from its own rows.
const mtp_chain = 3;
/// The drafter's context at most, and the trace's first step of its rounds.
const drafter_capacity = 64;
const drafter_step = plan.len + 2 + mtp_chain;

/// The ids the plan's last window ran: what the main model's taps are of when the drafter's rounds begin.
fn lastWindow(tokens: []const i32) []const i32 {
    var at: usize = 0;
    for (plan[0 .. plan.len - 1]) |window| at += window[1];
    return tokens[at..][0..plan[plan.len - 1][0]];
}

pub const usage =
    \\usage: zspark ranks-check random rank=0|1 at=ADDR:PORT [seed=N] [states]
    \\       zspark ranks-check DIR reference=REF [layers=L] [drafter=DIR] [kv=fp8] [rank=0|1 at=ADDR:PORT]
    \\  rank 1 listens at ADDR:PORT (start it first), rank 0 connects there. Both run the tests' random GLM in
    \\  loopback and as their own share over NCCL, and compare every bit a rank holds; 1 on a difference.
    \\  With DIR the checkpoint runs against the vLLM container's reference in REF and against its own serial
    \\  steps and replies (ranks_vllm.zig): the whole model a rank a Spark, its first layers also with both ranks
    \\  in a process.
    \\  Needs a build with -Dnccl=PREFIX. tools/two_ranks.sh ranks-check random starts both.
    \\
;

/// The model over `held` ranks, with what a sequence and a forward need, and the MTP layer.
fn Rig(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        m: H.Model,
        state: H.State,
        b: H.Buffers,
        pair: mtp.PairOf(held),
        /// Which of the held ranks is the one this process is.
        mine: usize,
        /// What the gathered buffers hold before every window and step, the size of one: a small number's
        /// bytes, another with both ranks here than with one, so a row read that this window did not write,
        /// or that no gather brought, is other bits on the two ways and not what the last window happened to
        /// leave. No NaN: one that reaches the DSA's selection is an index out of its rows.
        stale: []const u8,

        fn init(arena: Allocator, gpa: Allocator, model: *const Random, first: usize, mine: usize, gather: ?nccl.Comm) !Self {
            var m: H.Model = try .loadFrom(arena, gpa, model.source, &model.c, first, model.rows());
            errdefer m.deinit();
            if (gather) |comm| m.gather = .{ .nccl = comm };
            var state: H.State = try .init(m, capacity);
            errdefer state.deinit();
            var b: H.Buffers = try .init(m, capacity);
            errdefer b.deinit();
            try b.setTaps(&tap_layers);
            const stale = try arena.alloc(u8, b.attn.view.count() * @sizeOf(f32));
            @memset(stale, if (held == world) 0x3b else 0x3c);
            return .{ .m = m, .state = state, .b = b, .pair = try .init(arena, gpa, model.source, &model.c, m, capacity, model.rows()), .mine = mine, .stale = stale };
        }

        fn spoil(self: *Self) !void {
            try cuda.synchronize();
            try self.b.attn.upload(self.stale);
            try self.b.ffn.upload(self.stale);
        }

        fn deinit(self: *Self) void {
            self.pair.deinit();
            self.b.deinit();
            self.state.deinit();
            self.m.deinit();
        }

        const Watch = struct {
            rig: *Self,
            trace: *Trace,

            pub fn at(self: *const Watch, point: held_file.Point, layer: usize) !void {
                const rows = self.trace.rows;
                switch (point) {
                    .attention => try self.trace.take(.attention, layer, held_file.gatheredOf(self.rig.m.ranks[0], self.rig.b.attn, rows)),
                    .layer => {
                        try self.trace.take(.mlp, layer, held_file.gatheredOf(self.rig.m.ranks[0], self.rig.b.ffn, rows));
                        try self.trace.take(.streams, layer, self.rig.b.ranks[self.rig.mine].x.rows(rows));
                    },
                }
            }
        };

        /// The plan's windows from an empty sequence: launched, read at every gather, or through `graphs`.
        fn play(self: *Self, trace: *Trace, tokens: []const i32, graphs: ?*H.Graphs) !void {
            try self.state.reset();
            var at: usize = 0;
            for (plan, 0..) |window, index| {
                const rows = window[0];
                const ids = tokens[at..][0..rows];
                trace.step = index;
                trace.rows = rows;
                try self.spoil();
                if (graphs) |captured| {
                    try captured.forward(self.m, &self.state, &self.b, ids);
                } else {
                    try H.stage(&self.b, ids);
                    try H.computeWatched(self.m, &self.state, &self.b, rows, self.state.pos + rows, .all, &Watch{ .rig = self, .trace = trace });
                }
                const own = self.b.ranks[self.mine];
                // the last layer's gathered partials, whichever way, in the window's dtype
                try trace.take(.attention, null, held_file.gatheredOf(self.m.ranks[0], self.b.attn, rows));
                try trace.take(.mlp, null, held_file.gatheredOf(self.m.ranks[0], self.b.ffn, rows));
                try trace.take(.streams, null, own.x.rows(rows));
                try trace.take(.taps, null, own.taps.?.rows(rows));
                try trace.take(.hidden, null, own.hidden.rows(rows));
                try trace.take(.final, null, own.final.rows(rows));
                try trace.take(.logits, null, own.logits.rows(rows));
                try H.commit(self.m, &self.state, self.b, ids, window[1]);
                at += window[1];
            }
            trace.step = plan.len;
            trace.rows = at;
            const sequence = self.state.ranks[self.mine];
            const chains = sequence.kda.?;
            for (0..chains.layers) |index| try trace.take(.kda, null, chains.committed(index));
            const cache = sequence.dsa.?;
            for (0..cache.layers) |index| {
                var kept = cache.layer(index);
                kept.size[0] = @intCast(at);
                try trace.take(.latents, null, kept);
            }
        }

        /// The MTP layer after `play`: the last window's rows absorbed in one step, then chained drafts.
        fn draft(self: *Self, trace: *Trace, tokens: []const i32) !void {
            const absorbed = plan[plan.len - 1][0];
            try self.pair.setLen(0);
            self.pair.drafted = 0;
            for (0..1 + mtp_chain) |index| {
                const rows: usize = if (index == 0) absorbed else 1;
                trace.step = plan.len + 1 + index;
                trace.rows = rows;
                const ids = tokens[if (index == 0) 0 else absorbed + index - 1..][0..rows];
                try self.spoil();
                try self.pair.step(self.m, &self.b, if (index == 0) .{ .main = 0 } else .own, ids, true);
                const b = self.pair.b[self.mine];
                try trace.take(.mtp_attention, null, held_file.gathered(self.b.attn, rows));
                try trace.take(.mtp_mlp, null, held_file.gathered(self.b.ffn, rows));
                try trace.take(.mtp_stream, null, b.x.rows(rows));
                try trace.take(.mtp_final, null, b.final.view);
                try trace.take(.mtp_logits, null, b.logits.view);
                try self.pair.setLen(self.pair.len() + rows);
            }
        }
    };
}

/// The random model's tensors, its config and a sequence of ids, from a seed: the same on both ranks.
const Random = struct {
    pool: cuda.Scratch,
    c: @import("../layers/config.zig").Config,
    source: random_file.Source,
    tokens: []i32,

    fn init(self: *Random, arena: Allocator, gpa: Allocator, seed: u64) !void {
        self.pool = .init(gpa);
        var prng: std.Random.DefaultPrng = .init(seed);
        const random = prng.random();
        self.c = try fx.config(arena);
        self.source = .empty;
        try fx.putModel(arena, random, &self.source);
        var total: usize = fx.max_rows;
        for (plan) |window| total += window[1];
        self.tokens = try arena.alloc(i32, total);
        for (self.tokens) |*id| id.* = random.intRangeLessThan(i32, 0, fx.vocab);
    }

    fn rows(self: *const Random) cuda.Scratch.Rows {
        return .{ .pool = @constCast(&self.pool), .count = fx.max_rows };
    }

    /// The headers as a checkpoint has them: every tensor's name, type and shape, by name.
    fn headers(self: *const Random, arena: Allocator) !channel_file.Named {
        const names = try arena.alloc([]const u8, self.source.count());
        var keys = self.source.keyIterator();
        for (names) |*name| name.* = keys.next().?.*;
        std.mem.sort([]const u8, names, {}, struct {
            fn before(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.lessThan(u8, a, b);
            }
        }.before);
        var parts: std.ArrayList([]const u8) = .empty;
        for (names) |name| {
            const tensor = self.source.get(name).?;
            try parts.append(arena, name);
            try parts.append(arena, @tagName(tensor.dtype));
            try parts.append(arena, std.mem.sliceAsBytes(tensor.shape));
        }
        return .{ .digest = channel_file.digest(parts.items), .label = try std.fmt.allocPrint(arena, "a random GLM, {d} tensors", .{names.len}) };
    }
};

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    // a checkpoint's directory: the whole model's check
    if (words.len >= 1 and !std.mem.eql(u8, words[0], "random") and std.mem.indexOfScalar(u8, words[0], '=') == null) return whole.run(init, stdout, words);
    var place: join_file.Place = .{};
    var seed: u64 = 45;
    var random = false;
    var states = false;
    for (words) |word| {
        if (place.take(word)) continue;
        if (cli.value(word, "seed=")) |text| {
            seed = std.fmt.parseInt(u64, text, 10) catch return 2;
        } else if (std.mem.eql(u8, word, "random")) {
            random = true;
        } else if (std.mem.eql(u8, word, "states")) {
            states = true;
        } else {
            std.log.err("ranks-check: '{s}' is not `random`, and a checkpoint's directory comes first", .{word});
            return 2;
        }
    }
    if (!random or !place.complete()) return 2;
    if (!nccl.linked) {
        std.log.err("ranks-check: this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return 1;
    }
    const rank = place.rank.?;
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    _ = try cuda.probe();

    var model: Random = undefined;
    try model.init(arena, gpa, seed);
    defer model.pool.deinit();
    const tokens = model.tokens;

    const joining = cuda.monotonic();
    var joined: join_file.Joined = undefined;
    const options = try std.fmt.allocPrint(arena, "ranks-check random seed={d}{s}", .{ seed, if (states) " states" else "" });
    try joined.open(init.io, gpa, place, try model.headers(arena), options);
    defer joined.close();
    const channel = &joined.channel;
    try stdout.print("rank {d} of {d}: zspark {s}{s}, NCCL {d}, {s} of seed {d}, joined in {d:.0} ms\n", .{ rank, world, cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "", try nccl.version(), (try model.headers(arena)).label, seed, @as(f64, @floatFromInt(cuda.monotonic() - joining)) / 1e6 });
    try stdout.flush();
    if (states) return ranks_states.run(gpa, arena, stdout, channel, joined.comm, rank, model.source, &model.c, model.rows(), tokens);

    var failure: ?[]const u8 = null;
    var drafter_model: drafter_check.Random = try .init(arena, gpa, seed, &tap_layers);
    defer drafter_model.deinit(gpa);
    const last = lastWindow(tokens);

    // the reference: both ranks here, their partials written in place
    var launched: Trace = .{ .gpa = gpa };
    defer launched.deinit();
    var through: Trace = .{ .gpa = gpa };
    defer through.deinit();
    var drafted: Trace = .{ .gpa = gpa };
    defer drafted.deinit();
    var passed: Trace = .{ .gpa = gpa };
    defer passed.deinit();
    var passed_through: Trace = .{ .gpa = gpa };
    defer passed_through.deinit();
    {
        var both: Rig(world) = try .init(arena, gpa, &model, 0, rank, null);
        defer both.deinit();
        try both.play(&launched, tokens, null);
        try both.draft(&drafted, tokens);
        var drafter: drafter_check.Rig(world) = undefined;
        try drafter.init(arena, gpa, &drafter_model, both.m, &both.b, &model.pool, drafter_capacity, rank);
        defer drafter.deinit();
        try drafter.run(&passed, drafter_step, last, null);
        var graphs: held_file.Held(world).Graphs = try .init(gpa);
        defer graphs.deinit();
        try both.play(&through, tokens, &graphs);
        var passes: dflash2_pass.Graphs = try .init(gpa);
        defer passes.deinit();
        try drafter.run(&passed_through, drafter_step, last, &passes);
        // the reference agrees with itself before anything is held to it
        var ends = try launched.only(gpa, atEnd);
        defer ends.deinit();
        if (try differs(arena, &ends, &through, "launched", "through graphs")) |text| failure = try std.fmt.allocPrint(arena, "the loopback itself: {s}", .{text});
        var pass_ends = try passed.only(gpa, atEnd);
        defer pass_ends.deinit();
        if (try differs(arena, &pass_ends, &passed_through, "launched", "through graphs")) |text| failure = failure orelse try std.fmt.allocPrint(arena, "the loopback itself: {s}", .{text});
        const first = launched.marks.items[0];
        const partials = launched.of(first);
        if (std.mem.eql(u8, partials[0 .. partials.len / world], partials[partials.len / world ..]) or std.mem.allEqual(u8, partials, 0))
            failure = failure orelse "the loopback itself: the ranks' partials of the first layer are the same, or zero";
        try stdout.print("loopback: {d} windows of 1 to {d} rows launched, {d} buffers read, {d:.1} MiB; through {d} graphs, the windows' ends the same\n", .{ plan.len, fx.max_rows, launched.marks.items.len, @as(f64, @floatFromInt(launched.bytes.items.len)) / (1024.0 * 1024.0), graphs.cache.count() });
        try stdout.print("loopback: the drafter's {d} block passes launched, {d} buffers read, {d:.1} MiB; through {d} graphs, the passes' ends the same\n", .{ 1 + drafter_check.rounds, passed.marks.items.len, @as(f64, @floatFromInt(passed.bytes.items.len)) / (1024.0 * 1024.0), 1 + passes.updates.count() });
        try stdout.flush();
    }
    channel.barrier(1) catch |err| return channel.gone(rank, err);

    // this rank's share alone, the other's partials through NCCL. Both ranks run every step whatever they
    // find: one that stopped would leave the other inside a gather.
    {
        var own: Rig(1) = try .init(arena, gpa, &model, rank, 0, joined.comm);
        defer own.deinit();
        var got: Trace = .{ .gpa = gpa };
        defer got.deinit();
        try own.play(&got, tokens, null);
        if (try check.held(arena, stdout, channel, rank, 2, &failure, "launched", &launched, &got)) |code| return code;

        var got_drafted: Trace = .{ .gpa = gpa };
        defer got_drafted.deinit();
        try own.draft(&got_drafted, tokens);
        if (try check.held(arena, stdout, channel, rank, 3, &failure, "the MTP layer's steps", &drafted, &got_drafted)) |code| return code;

        var drafter: drafter_check.Rig(1) = undefined;
        try drafter.init(arena, gpa, &drafter_model, own.m, &own.b, &model.pool, drafter_capacity, 0);
        defer drafter.deinit();
        var got_passed: Trace = .{ .gpa = gpa };
        defer got_passed.deinit();
        try drafter.run(&got_passed, drafter_step, last, null);
        if (try check.held(arena, stdout, channel, rank, 4, &failure, "the drafter's passes and drafts, launched", &passed, &got_passed)) |code| return code;

        var graphs: held_file.Held(1).Graphs = try .init(gpa);
        defer graphs.deinit();
        var got_through: Trace = .{ .gpa = gpa };
        defer got_through.deinit();
        try own.play(&got_through, tokens, &graphs);
        if (try check.held(arena, stdout, channel, rank, 5, &failure, "through captured graphs", &through, &got_through)) |code| return code;

        var passes: dflash2_pass.Graphs = try .init(gpa);
        defer passes.deinit();
        var got_passed_through: Trace = .{ .gpa = gpa };
        defer got_passed_through.deinit();
        try drafter.run(&got_passed_through, drafter_step, last, &passes);
        if (try check.held(arena, stdout, channel, rank, 6, &failure, "the drafter's passes and drafts, through captured graphs", &passed_through, &got_passed_through)) |code| return code;
    }

    return check.end(arena, stdout, channel, rank, failure, 7, "");
}
