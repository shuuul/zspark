//! How the ranks' partials meet (spec 045 WS-02). A gathered buffer is used tight: fp32 [world, rows, dims] for
//! a window of `rows` rows, a rank's partial at `rank * rows * dims`, whatever the buffer was made for (or bf16 in
//! its first half, `as`, where a model's window rounds its partials as vLLM's, spec 116 WS-04). A rank
//! writes its partial there (`partial`), the `Gather` brings the others', and every reader gets the same view
//! (`gathered`) whether the ranks are one process or two: nothing is copied either way.
//!
//! ```mermaid
//! flowchart LR
//!   P0["rank 0's partial<br/>at 0"] --> G["gathered fp32 [world, rows, dims]"]
//!   P1["rank 1's partial<br/>at rows * dims"] --> G
//!   G -- "added rank 0 first" --> R["hc.post, residualAdd, branch"]
//! ```
//!
//! A window split by rows for its glue (spec 106 WS-02, after TensorFold 1.0.12's `rank_scatter.zig`) brings
//! less: each rank receives only its own rows (`share`) of the others' partials, at the same places
//! (`Gather.scatter`), sums those rows (`sumRows`), and later shares the rows it made (`Gather.shareRows`).

const std = @import("std");
const cuda = @import("../cuda/cuda.zig");
const lead_file = @import("lead.zig");
const nccl = @import("nccl.zig");

pub const Error = nccl.Error || lead_file.Error || cuda.Error;

/// The most ranks the split form's exchanges are written for (`Gather.scatter`).
const max_world = 8;

/// The first `rows` rows of every rank in `buffer`, fp32 or bf16 [world, rows or more, dims]: what a reader is given.
pub fn gathered(buffer: cuda.View, rows: usize) cuda.View {
    std.debug.assert(partials(buffer) and rows >= 1 and rows <= buffer.size[1]);
    return .init(buffer.ptr.?, buffer.dtype, &.{ @intCast(buffer.size[0]), rows, @intCast(buffer.size[2]) });
}

/// Where `rank` writes its `rows` rows of `buffer`: [rows, dims] in its dtype, `rank * rows * dims` values in.
pub fn partial(buffer: cuda.View, rank: usize, rows: usize) cuda.View {
    std.debug.assert(partials(buffer) and rank < buffer.size[0] and rows >= 1 and rows <= buffer.size[1]);
    const dims: usize = @intCast(buffer.size[2]);
    const base: [*]u8 = @ptrCast(buffer.ptr.?);
    return .init(@ptrCast(base + rank * rows * dims * buffer.dtype.size()), buffer.dtype, &.{ rows, dims });
}

/// A gathered buffer made fp32 [world, rows, dims] read as `dtype` of that shape: bf16 takes its first half.
pub fn as(buffer: cuda.View, dtype: cuda.DType) cuda.View {
    std.debug.assert(buffer.dtype == .f32 and partials(buffer) and (dtype == .f32 or dtype == .bf16));
    return .init(buffer.ptr.?, dtype, &.{ @intCast(buffer.size[0]), @intCast(buffer.size[1]), @intCast(buffer.size[2]) });
}

fn partials(buffer: cuda.View) bool {
    return (buffer.dtype == .f32 or buffer.dtype == .bf16) and buffer.ndim == 3 and buffer.row_stride == 0;
}

/// A window's rows split by rank for its glue (spec 106 WS-02): each rank `cut_rows` rows at a time, so a rank's
/// first row starts every row-indexed buffer on 128 bytes or more (a read-out's inject gates, 8 bytes a row, are
/// the narrowest). Rank 0 takes the first share, rounded up; the last the rest, possibly none.
pub const cut_rows = 16;

/// Rank `rank`'s rows [first, end) of a window of `rows`.
pub fn share(rows: usize, world: usize, rank: usize) [2]usize {
    const each = std.mem.alignForward(usize, (rows + world - 1) / world, cut_rows);
    const first = @min(rows, rank * each);
    return .{ first, @min(rows, first + each) };
}

/// The rows rank `rank` glues of a `gathered` window, fp32 [world, rows, dims], summed over the ranks, rank 0
/// first, into `into` [count, dims] (bf16, rounded once, or fp32): `qwen4/partial.cu`'s sum of those rows, the
/// same bits (`common/rank_sum.cu`).
pub fn sumRows(gathered_rows: cuda.View, first: usize, into: cuda.View) cuda.Error!void {
    std.debug.assert(into.dtype == .bf16 or into.dtype == .f32);
    return cuda.launched(zs_rank_sum(&gathered_rows, @intCast(first), &into, @intFromBool(into.dtype == .bf16)));
}

extern fn zs_rank_sum(gathered: *const cuda.View, first: i64, into: *const cuda.View, to_bf16: i64) c_int;

/// A stream for `Gather.scatterBegin` and the event its exchange ends at: one a forward's buffers, over NCCL.
pub const Beside = struct {
    side: cuda.Side,
    done: cuda.Event,

    pub fn init() cuda.Error!Beside {
        var side: cuda.Side = try .init();
        errdefer side.deinit();
        return .{ .side = side, .done = try .init() };
    }

    pub fn deinit(self: *Beside) void {
        self.done.deinit();
        self.side.deinit();
        self.* = undefined;
    }
};

/// What carries the partials between the ranks once each has written its own.
pub const Gather = union(enum) {
    /// One process holds every rank: each wrote in place, and nothing is left to do.
    loopback,
    /// A rank a process: an all-gather in place (`sendbuff == recvbuff + rank * sendcount`, `nccl.h`), on
    /// this thread's stream, so a captured graph holds it among the kernels around it.
    nccl: nccl.Comm,

    /// `view` is `gathered`'s: after it every rank's partial is there, on every rank.
    pub fn run(self: Gather, view: cuda.View) Error!void {
        switch (self) {
            .loopback => {},
            .nccl => |comm| {
                std.debug.assert(view.size[0] == comm.world);
                try comm.allGather(partial(view, comm.rank, @intCast(view.size[1])), view);
            },
        }
    }

    /// `run`'s split form: after it, each rank's rows (`share`) of every rank's partial are in `view`, where
    /// `run` would put them, and no other rows are brought. Each rank sends a peer the peer's rows of its own
    /// partial (NCCL send and receive, not `ncclReduceScatter`, which adds in its ring's order), so a rank
    /// receives half of what `run` brings at two ranks. In loopback every partial is in place already.
    pub fn scatter(self: Gather, view: cuda.View) Error!void {
        switch (self) {
            .loopback => {},
            .nccl => |comm| {
                var moves: [max_world]nccl.Comm.Move = undefined;
                try comm.exchange(view.dtype, scattered(comm, view, &moves));
            },
        }
    }

    /// `scatter` begun on `beside`'s stream after this thread's launches so far (spec 116 WS-03): it sends
    /// what this rank made of the others' rows while this thread goes on with its own rows' arithmetic, and
    /// `scatterJoin` holds this thread's launches after it until the others' rows are in. Nothing in between
    /// may write the rows sent or read or write the rows received. Null: `scatter` itself.
    pub fn scatterBegin(self: Gather, view: cuda.View, beside: ?*const Beside) Error!void {
        switch (self) {
            .loopback => {},
            .nccl => |comm| {
                const by = beside orelse return self.scatter(view);
                var moves: [max_world]nccl.Comm.Move = undefined;
                try by.side.follow();
                try comm.exchangeOn(by.side.handle, view.dtype, scattered(comm, view, &moves));
                try by.done.recordOn(by.side.handle);
            },
        }
    }

    pub fn scatterJoin(self: Gather, beside: ?*const Beside) Error!void {
        switch (self) {
            .loopback => {},
            .nccl => if (beside) |by| try by.done.join(),
        }
    }

    /// `scatter`'s moves: to each peer the peer's rows of this rank's partial, from it this rank's rows of its.
    fn scattered(comm: nccl.Comm, view: cuda.View, moves: *[max_world]nccl.Comm.Move) []const nccl.Comm.Move {
        std.debug.assert(view.size[0] == comm.world and partials(view) and comm.world <= max_world);
        const rows: usize = @intCast(view.size[1]);
        const dims: usize = @intCast(view.size[2]);
        const base: [*]u8 = @ptrCast(view.ptr.?);
        const mine = share(rows, comm.world, comm.rank);
        var count: usize = 0;
        for (0..comm.world) |peer| {
            if (peer == comm.rank) continue;
            const theirs = share(rows, comm.world, peer);
            moves[count] = .{
                .peer = peer,
                .send = @ptrCast(base + ((comm.rank * rows + theirs[0]) * dims) * view.dtype.size()),
                .send_count = (theirs[1] - theirs[0]) * dims,
                .recv = @ptrCast(base + ((peer * rows + mine[0]) * dims) * view.dtype.size()),
                .recv_count = (mine[1] - mine[0]) * dims,
            };
            count += 1;
        }
        return moves[0..count];
    }

    /// The rows each rank made of `view` [rows, ...] (`share`) brought to every rank, in place: after the split
    /// glue, what the next block reads whole. Over NCCL only; in loopback the caller copies between its buffers.
    pub fn shareRows(self: Gather, view: cuda.View) Error!void {
        const comm = self.nccl;
        std.debug.assert(comm.world <= max_world and view.row_stride == 0);
        const rows: usize = @intCast(view.size[0]);
        const row = view.count() / rows;
        const width = view.dtype.size();
        const base: [*]u8 = @ptrCast(view.ptr.?);
        const mine = share(rows, comm.world, comm.rank);
        var moves: [max_world]nccl.Comm.Move = undefined;
        var count: usize = 0;
        for (0..comm.world) |peer| {
            if (peer == comm.rank) continue;
            const theirs = share(rows, comm.world, peer);
            moves[count] = .{
                .peer = peer,
                .send = @ptrCast(base + mine[0] * row * width),
                .send_count = (mine[1] - mine[0]) * row,
                .recv = @ptrCast(base + theirs[0] * row * width),
                .recv_count = (theirs[1] - theirs[0]) * row,
            };
            count += 1;
        }
        try comm.exchange(view.dtype, moves[0..count]);
    }

    /// Whether a process holding `held` ranks of `world` may run over this.
    pub fn serves(self: Gather, held: usize, world: usize) bool {
        return switch (self) {
            .loopback => held == world,
            .nccl => |comm| held == 1 and comm.world == world,
        };
    }
};

/// Words rank 0 says to every rank inside the device's stream, gathered over NCCL as a layer's partials are:
/// u32 [world, `width`], each rank's part at its own place, rank 0's what is read. A few microseconds on the
/// link where a frame of the control channel took some 120 us to wake a follower (spec 095 WS-00, spec 144
/// WS-07). Flash Next's pair says a round's words with it (`session/pair_rounds.zig`'s `Words`), and its pool
/// its operations while a round runs (`Lead.over`, `pool/hall_ranks.zig`).
pub const Said = struct {
    buffer: cuda.Buffer,
    comm: nccl.Comm,
    width: usize,

    pub fn init(comm: nccl.Comm, width: usize) cuda.Error!Said {
        return .{ .buffer = try .init(.u32, &.{ comm.world, width }), .comm = comm, .width = width };
    }

    pub fn deinit(self: *Said) void {
        self.buffer.deinit();
        self.* = undefined;
    }

    /// Rank 0's `words`, at most `width`: written by launches that carry them (`uploadLaunched`), so the host
    /// waits for nothing and may change its copy at once, then gathered. The rest of its part keeps what it
    /// held.
    pub fn say(self: Said, words: []const u32) (cuda.Error || nccl.Error)!void {
        std.debug.assert(words.len <= self.width);
        var at: usize = 0;
        while (at < words.len) : (at += 32) {
            const part = words[at..@min(words.len, at + 32)];
            try self.buffer.uploadLaunched((self.comm.rank * self.width + at) * 4, std.mem.sliceAsBytes(part));
        }
        try self.gathered();
    }

    /// Rank 0's first `out.len` words, once they have come.
    pub fn hear(self: Said, out: []u32) (cuda.Error || nccl.Error)!void {
        std.debug.assert(out.len <= self.width);
        try self.gathered();
        try self.buffer.downloadAt(0, std.mem.sliceAsBytes(out));
    }

    fn gathered(self: Said) (cuda.Error || nccl.Error)!void {
        const all = self.buffer.view;
        const at: [*]u8 = @ptrCast(all.ptr.?);
        try self.comm.allGather(.init(@ptrCast(at + self.comm.rank * self.width * 4), .u32, &.{self.width}), all);
    }

    /// `Lead.over`'s: rank 0's frames said here.
    pub fn over(self: *Said) lead_file.Lead.Over {
        return .{ .context = self, .width = self.width, .say = sayOver };
    }

    fn sayOver(context: *anyopaque, words: []const u32) anyerror!void {
        const self: *Said = @ptrCast(@alignCast(context));
        return self.say(words);
    }
};
