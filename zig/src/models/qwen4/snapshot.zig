//! `Snapshot`: a sequence's committed state at one position, kept to return to, with the three ways one is
//! taken (`take`, `again` over held buffers, `ahead` of a wide window). `forward.zig` names it as its own.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const cuda = @import("../../cuda/cuda.zig");
const gdn = @import("blocks/gdn.zig");
const model = @import("model.zig");
const ple = @import("blocks/ple.zig");
const state_windows = @import("blocks/state.zig");
const forward = @import("forward.zig");
const Buffers = forward.Buffers;
const State = forward.State;
const stashRows = forward.stashRows;
const widerThanKept = forward.widerThanKept;

/// A sequence's committed state at one position, to return to: what `State` holds that later rows change in
/// place. The attention caches are not in it: their rows below the position stay as they are, and a row never
/// reads one at or past its own position + 1.
pub const Snapshot = struct {
    pos: usize,
    /// Each DeltaNet layer's committed state and conv inputs.
    recurrent: []cuda.Buffer,
    conv: []cuda.Buffer,
    /// Each PLE layer's conv tail and n-gram history.
    tails: []cuda.Buffer,
    history: [][]u32,
    gpa: Allocator,
    /// Where the device buffers above are views of one allocation (`cuda.cut`, a pool's kept state:
    /// `hall_keep.fresh`), that allocation, which `deinit` frees in their place; else none, each its own.
    block: cuda.Buffer = .{ .view = .{} },

    pub fn take(gpa: Allocator, state: State) !Snapshot {
        try cuda.synchronize();
        var self: Snapshot = .{ .pos = state.pos, .recurrent = &.{}, .conv = &.{}, .tails = &.{}, .history = &.{}, .gpa = gpa };
        errdefer self.deinit();
        self.recurrent = try copies(gpa, state.recurrent.len);
        self.conv = try copies(gpa, state.conv.len);
        self.tails = try copies(gpa, state.ple.len);
        for (self.recurrent, self.conv, state.recurrent, state.conv, 0..) |*recurrent, *conv, pair, source, linear| {
            recurrent.* = try likeOf(pair[0]);
            try forward.committed(state, linear, recurrent.*);
            conv.* = try copyOf(source);
        }
        for (self.tails, state.ple) |*tail, branch| tail.* = try copyOf(branch.tail);
        const history = try gpa.alloc([]u32, state.ple.len);
        for (history) |*h| h.* = &.{};
        self.history = history;
        for (history, state.ple) |*h, branch| h.* = try gpa.dupe(u32, branch.history);
        return self;
    }

    /// `take`, written over a snapshot of the same model that is no longer wanted, whose buffers it keeps: a
    /// state's buffers made anew are 6 ms of the prompt that waits behind them (spec 066).
    pub fn retake(self: *Snapshot, state: State) !void {
        try cuda.synchronize();
        self.pos = state.pos;
        try self.reserve(state);
        for (self.recurrent, self.conv, state.conv, 0..) |recurrent, conv, source, linear| {
            try forward.committed(state, linear, recurrent);
            try conv.copyFrom(source);
        }
        // the histories of the state it was go, as in `branches`
        const gpa = self.gpa;
        for (self.history) |h| gpa.free(h);
        gpa.free(self.history);
        self.history = &.{};
        const history = try gpa.alloc([]u32, state.ple.len);
        for (history) |*h| h.* = &.{};
        self.history = history;
        for (self.tails, history, state.ple) |tail, *h, branch| {
            try tail.copyFrom(branch.tail);
            h.* = try gpa.dupe(u32, branch.history);
        }
    }

    /// The state after the first `keep` rows of the window `tokens` that the last `forward` ran and that is not
    /// committed yet: what `commit` with `keep` and then `take` would give, with `state` left as it stands, so
    /// the window is still committed whole and a prompt is not cut where a snapshot is wanted. The DeltaNet
    /// states are the window's chain replayed to that row (`gdn.replay`, the bits of the serial steps), the
    /// conv inputs and PLE tails copies slid as `commit` slides the state's.
    ///
    /// For a window of `kept` rows or fewer: a wider one says what it keeps before it runs (`ahead`).
    pub fn within(gpa: Allocator, m: model.Model, state: State, b: Buffers, tokens: []const i32, keep: usize) !Snapshot {
        var self: Snapshot = .{ .pos = state.pos + keep, .recurrent = &.{}, .conv = &.{}, .tails = &.{}, .history = &.{}, .gpa = gpa };
        errdefer self.deinit();
        self.recurrent = try copies(gpa, state.recurrent.len);
        self.conv = try copies(gpa, state.conv.len);
        self.tails = try copies(gpa, state.ple.len);
        try self.fill(m, state, b, tokens, keep);
        return self;
    }

    /// `within`, written over a snapshot of the same model that is no longer wanted: its device buffers are
    /// kept, so a state taken again and again while a reply is decoded (spec 036) allocates nothing. The
    /// snapshot it was is gone, whether this fails or not.
    pub fn again(self: *Snapshot, m: model.Model, state: State, b: Buffers, tokens: []const i32, keep: usize) !void {
        self.pos = state.pos + keep;
        try self.fill(m, state, b, tokens, keep);
    }

    /// What `within` holds, into the buffers the snapshot has or, where it has none yet, new ones.
    fn fill(self: *Snapshot, m: model.Model, state: State, b: Buffers, tokens: []const i32, keep: usize) !void {
        const rows = tokens.len;
        std.debug.assert(keep >= 1 and keep <= rows and !widerThanKept(state, b, rows));
        try cuda.synchronize();
        const dn = m.shape.delta_net;
        for (self.recurrent, self.conv, state.recurrent, state.current, state.conv, state.scratch, b.projections) |*recurrent, *conv, pair, current, source, scratch, projection| {
            if (recurrent.view.ptr == null) recurrent.* = try likeOf(pair[current]);
            if (keep < rows or !forward.stores(rows)) {
                // from the committed side, over the rows it is behind by and then the window's
                try gdn.replay(pair[current].view, scratch, state.behind + keep, recurrent.view);
            } else {
                // the window wrote the state after all its rows on the other side
                try recurrent.copyFrom(pair[1 - current]);
            }
            if (conv.view.ptr == null) conv.* = try likeOf(source);
            try conv.copyFrom(source);
            try state_windows.shiftWindows(conv.view, .{ .view = projection.view, .layer = 0, .row = dn.projection(), .rows = rows }, keep);
        }
        try self.branches(state, b, tokens, keep);
    }

    /// The PLE tails and histories after the first `keep` rows of the window `tokens`, which has run.
    fn branches(self: *Snapshot, state: State, b: Buffers, tokens: []const i32, keep: usize) !void {
        const gpa = self.gpa;
        // a snapshot written over (`again`) has the histories of the state it was
        for (self.history) |h| gpa.free(h);
        gpa.free(self.history);
        self.history = &.{};
        const history = try gpa.alloc([]u32, state.ple.len);
        for (history) |*h| h.* = &.{};
        self.history = history;
        for (self.tails, history, state.ple, b.ple) |*tail, *h, branch, buffers| {
            if (tail.view.ptr == null) tail.* = try likeOf(branch.tail);
            try tail.copyFrom(branch.tail);
            h.* = try gpa.dupe(u32, branch.history);
            var kept: ple.State = .{ .tail = tail.*, .history = h.*, .eos = branch.eos };
            try ple.commit(&kept, buffers, @ptrCast(tokens), keep);
        }
    }

    /// `within` for a window wider than the rows each layer keeps, in two parts. Before the window runs: a
    /// snapshot at its row `keep`, whose DeltaNet states and conv inputs the forward writes once it is among
    /// `state.wanted` (each layer replays its chain to that row before the next layer takes the scratch). After
    /// the forward and before the commit, `finish` takes the PLE's part.
    pub fn ahead(gpa: Allocator, state: State, keep: usize) !Snapshot {
        std.debug.assert(keep >= 1);
        var self: Snapshot = .{ .pos = state.pos + keep, .recurrent = &.{}, .conv = &.{}, .tails = &.{}, .history = &.{}, .gpa = gpa };
        errdefer self.deinit();
        self.recurrent = try copies(gpa, state.recurrent.len);
        self.conv = try copies(gpa, state.conv.len);
        self.tails = try copies(gpa, state.ple.len);
        return self;
    }

    /// `ahead`, written over a snapshot of the same model that is no longer wanted, whose buffers it keeps.
    pub fn aheadOver(self: *Snapshot, state: State, keep: usize) void {
        std.debug.assert(keep >= 1);
        self.pos = state.pos + keep;
    }

    /// Every device buffer a snapshot of `state` holds, allocated where it is not yet: for `again` to
    /// write into, by a caller that has time before the window's results are in.
    pub fn reserve(self: Snapshot, state: State) cuda.Error!void {
        for (0..self.recurrent.len) |linear| try self.reach(state, linear);
        for (self.tails, state.ple) |*tail, branch| {
            if (tail.view.ptr == null) tail.* = try likeOf(branch.tail);
        }
    }

    /// Layer `linear`'s buffers of a snapshot taken `ahead`, allocated when the forward reaches the layer: the
    /// device is then computing the layers before it, and an allocation made before the window ran would be
    /// time nothing overlapped (6 ms of a prompt's first token for the 36 layers).
    pub fn reach(self: Snapshot, state: State, linear: usize) cuda.Error!void {
        if (self.recurrent[linear].view.ptr != null) return;
        self.recurrent[linear] = try likeOf(state.recurrent[linear][0]);
        self.conv[linear] = try likeOf(state.conv[linear]);
    }

    /// Layer `linear`'s state buffer as the scratch of a chain of `rows` rows, while the state is not
    /// written yet: k, g, beta and v one after another (`stashRows` says how many rows fit). A view of the
    /// buffer, not to be freed.
    pub fn stash(self: Snapshot, state: State, linear: usize, rows: usize) gdn.Scratch {
        const wide = state.wide.?;
        const nk: usize = @intCast(wide.k.view.size[1]);
        const nv: usize = @intCast(wide.v.view.size[1]);
        std.debug.assert(rows <= stashRows(state));
        const base: [*]u8 = @ptrCast(self.recurrent[linear].view.ptr.?);
        const k_bytes = rows * nk * gdn.key_dim * 4;
        const g_bytes = rows * nv * 4;
        return .{
            .k = .{ .view = .init(base, .f32, &.{ rows, nk, gdn.key_dim }) },
            .g = .{ .view = .init(base + k_bytes, .f32, &.{ rows, nv }) },
            .beta = .{ .view = .init(base + k_bytes + g_bytes, .f32, &.{ rows, nv }) },
            .v = .{ .view = .init(base + k_bytes + 2 * g_bytes, .bf16, &.{ rows, nv, gdn.value_dim }) },
        };
    }

    /// The second part of `ahead`, with the window's `tokens`.
    pub fn finish(self: *Snapshot, state: State, b: Buffers, tokens: []const i32) !void {
        std.debug.assert(self.pos > state.pos and self.pos <= state.pos + tokens.len);
        try cuda.synchronize();
        try self.branches(state, b, tokens, self.pos - state.pos);
    }

    /// `state` back at `pos`, as it was when the snapshot was taken.
    pub fn restore(self: Snapshot, state: *State) cuda.Error!void {
        try cuda.synchronize();
        for (self.recurrent, self.conv, state.recurrent, state.current, state.conv) |recurrent, conv, pair, current, dest| {
            try pair[current].copyFrom(recurrent);
            try dest.copyFrom(conv);
        }
        for (self.tails, self.history, state.ple) |tail, history, *branch| {
            try branch.tail.copyFrom(tail);
            @memcpy(branch.history, history);
        }
        try state.position.set(self.pos);
        state.pos = self.pos;
        state.behind = 0;
    }

    pub fn bytes(self: Snapshot) usize {
        var total: usize = 0;
        for ([_][]cuda.Buffer{ self.recurrent, self.conv, self.tails }) |list| for (list) |buffer| {
            if (buffer.view.ptr != null) total += buffer.bytes();
        };
        return total;
    }

    pub fn deinit(self: *Snapshot) void {
        const one = self.block.view.ptr != null;
        if (one) self.block.deinit();
        for ([_][]cuda.Buffer{ self.recurrent, self.conv, self.tails }) |list| {
            if (!one) for (list) |*buffer| if (buffer.view.ptr != null) buffer.deinit();
            self.gpa.free(list);
        }
        for (self.history) |h| self.gpa.free(h);
        self.gpa.free(self.history);
        self.* = undefined;
    }

    /// `count` buffers not yet allocated, which `deinit` skips.
    pub fn copies(gpa: Allocator, count: usize) ![]cuda.Buffer {
        const list = try gpa.alloc(cuda.Buffer, count);
        for (list) |*buffer| buffer.* = .{ .view = .{} };
        return list;
    }

    fn copyOf(source: cuda.Buffer) cuda.Error!cuda.Buffer {
        var copy = try likeOf(source);
        errdefer copy.deinit();
        try copy.copyFrom(source);
        return copy;
    }

    /// A buffer of `source`'s type and shape, not written.
    pub fn likeOf(source: cuda.Buffer) cuda.Error!cuda.Buffer {
        var shape: [6]usize = undefined;
        const ndim: usize = @intCast(source.view.ndim);
        for (shape[0..ndim], source.view.size[0..ndim]) |*dim, size| dim.* = @intCast(size);
        return .init(source.view.dtype, shape[0..ndim]);
    }
};
