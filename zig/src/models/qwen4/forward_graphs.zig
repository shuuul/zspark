//! The forward's step as captured graphs: one launch a window in place of one a kernel, one graph for each
//! window size, each side the DeltaNet states can be on and each bucket of context. `forward.zig` names it as
//! its own `Graphs`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const cache_file = @import("../../decode/graphs.zig");
const forward_file = @import("forward.zig");
const model = @import("model.zig");

/// The step as captured graphs, as TensorFold's `Graphs`: one launch a window in place of one a kernel. A graph
/// holds its launches' arguments, so there is one for each window size, each side the DeltaNet states can be on,
/// and each bucket of context; the first window of a kind is captured, then launched as every later one is. The
/// kernels and their arguments are `forward`'s, so are the bits.
pub const Graphs = struct {
    cache: cache_file.Cache(Key),

    const Key = struct { rows: usize, current: u1, context: usize };

    pub fn init(gpa: Allocator) cuda.Error!Graphs {
        return .{ .cache = try .init(gpa) };
    }

    /// The graphs hold pointers into the model, state and buffers they were captured with: drop them first.
    pub fn deinit(self: *Graphs) void {
        self.cache.deinit();
        self.* = undefined;
    }

    /// The context a window ending at `end` is launched for: a power of two from 8192, so a sequence crosses
    /// few of them, and at most the caches' capacity. Zero for a model without attention.
    pub fn bucket(b: forward_file.Buffers, end: usize) usize {
        const scratch = b.attention orelse return 0;
        return @min(scratch.capacity, @max(8192, std.math.ceilPowerOfTwoAssert(usize, end)));
    }

    /// `forward`, always with the same `m`, `state` and `b`.
    pub fn forward(self: *Graphs, m: model.Model, state: *forward_file.State, b: *forward_file.Buffers, tokens: []const i32) (forward_file.Error || Allocator.Error)!void {
        const rows = tokens.len;
        try forward_file.stage(m, state, b, tokens);
        const context = bucket(b.*, state.pos + rows);
        // an image's rows are copied from features a graph could not hold: the same launches, eager
        if (b.image_rows > 0) return forward_file.compute(m, state, b, rows, if (b.attention != null) context else null, .all, null);
        const key: Key = .{ .rows = rows, .current = if (state.current.len > 0) state.current[0] else 0, .context = context };
        return self.cache.run(key, forward_file.compute, .{ m, state, b, rows, if (b.attention != null) context else null, .all, null });
    }

    /// Whether the graph `forward` would launch for a window of `rows` rows from where `state` stands is
    /// captured: such a window then costs no capture, and may be a prompt's last as well as a reply's.
    pub fn holds(self: *const Graphs, state: forward_file.State, b: forward_file.Buffers, rows: usize) bool {
        if (rows > b.scoredRows() or forward_file.widerThanKept(state, b, rows)) return false;
        if (state.image) |image| if (image.touches(state.pos, rows)) return false;
        // a window that lands the states first finds them on the other side
        const side: u1 = if (state.current.len > 0) state.current[0] ^ @intFromBool(forward_file.lands(state, b, rows)) else 0;
        return self.cache.contains(.{ .rows = rows, .current = side, .context = bucket(b, state.pos + rows) });
    }

    /// The buckets a sequence in `b` crosses, shallowest first: the first for `after` null, else the one past
    /// `after`, and null past the last, which is the caches' capacity.
    pub fn bucketAfter(b: forward_file.Buffers, after: ?usize) ?usize {
        const context = after orelse return bucket(b, 1);
        const scratch = b.attention orelse return null;
        return if (context >= scratch.capacity) null else bucket(b, context + 1);
    }

    /// Captures the graph `forward` would for a window of `rows` rows with the DeltaNet states on side
    /// `current`, over the bucket `context`, unless it is held. A capture records its launches and runs none,
    /// so no buffer and no state is written: the sequence goes on as if this had not been called.
    pub fn prepare(self: *Graphs, m: model.Model, state: *forward_file.State, b: *forward_file.Buffers, rows: usize, current: u1, context: usize) (forward_file.Error || Allocator.Error)!void {
        std.debug.assert(rows >= 1 and rows <= b.scoredRows() and rows <= m.max_rows);
        const key: Key = .{ .rows = rows, .current = if (state.current.len > 0) current else 0, .context = context };
        if (self.cache.contains(key)) return;
        // `compute` reads the side from the state; every layer's is the same one, as `commit` flips them together
        const stood: u1 = if (state.current.len > 0) state.current[0] else 0;
        for (state.current) |*side| side.* = key.current;
        defer for (state.current) |*side| {
            side.* = stood;
        };
        _ = try self.cache.capture(key, forward_file.compute, .{ m, state, b, rows, if (b.attention != null) context else null, .all, null });
    }

    /// `prepare` for every window of 1 to `rows` rows on either state side over `context`: what a reply's
    /// rounds of up to `rows - 1` drafts ask for while the sequence ends inside that bucket.
    pub fn prepareWindows(self: *Graphs, m: model.Model, state: *forward_file.State, b: *forward_file.Buffers, rows: usize, context: usize) (forward_file.Error || Allocator.Error)!void {
        for (1..rows + 1) |window| {
            for ([_]u1{ 0, 1 }) |current| try self.prepare(m, state, b, window, current, context);
        }
    }
};
