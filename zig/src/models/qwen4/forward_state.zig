//! What a Flash Next forward runs in and what it leaves (spec 104 WS-06 moved them out of `forward.zig`): a
//! window's scratch (`Buffers`) and one sequence's committed state (`State`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const cuda = @import("../../cuda/cuda.zig");
const placeholders = @import("../../decode/placeholders.zig");
const gdn = @import("blocks/gdn.zig");
const hc = @import("blocks/hc.zig");
const model = @import("model.zig");
const moe = @import("blocks/moe.zig");
const nvfp4 = @import("../../weights/nvfp4.zig");
const ple = @import("blocks/ple.zig");
const taps = @import("taps.zig");
const forward = @import("forward.zig");
const Snapshot = forward.Snapshot;

/// Scratch for windows of up to `rows` rows.
pub const Buffers = struct {
    rows: usize,
    /// int32 [rows]
    ids: cuda.Buffer,
    /// bf16 [rows, streams * hidden]: the residual streams
    h: cuda.Buffer,
    /// fp32 [rows, hidden / 256, streams]
    pss: cuda.Buffer,
    /// bf16 [rows, streams * hidden]
    normed: cuda.Buffer,
    /// fp32 [rows, low + streams]: a layer's down projection with its inject rows
    down: cuda.Buffer,
    /// fp32 [rows, low]: the mixer's
    down_mixer: cuda.Buffer,
    /// bf16 [rows, low]
    act: cuda.Buffer,
    /// bf16 [rows, streams]: the attention-side read-out's inject gates
    inject_attn: cuda.Buffer,
    /// bf16 [rows, streams]: the MLP-side read-out's
    inject_mlp: cuda.Buffer,
    /// bf16 [rows, streams * hidden]
    up: cuda.Buffer,
    /// bf16 [rows, hidden]
    mixed: cuda.Buffer,
    /// bf16 [kept, projection width] for each DeltaNet layer: `commit` reads their conv channels
    projections: []cuda.Buffer,
    /// The rows of a window that each DeltaNet layer keeps its own projection of: a window of that many rows or
    /// fewer can be committed in part, or have a state inside it taken, after it ran (`commit`,
    /// `Snapshot.within`). The rows that can be scored, three at least.
    kept: usize,
    /// bf16 [rows, projection width]: every DeltaNet layer's projection of a window wider than `kept`, which is
    /// committed whole and says before it runs which states inside it are wanted (`State.wanted`); null where
    /// no window is wider. Each layer's own at a prompt's 2048 rows was 2.3 GiB (spec 033).
    projection_wide: ?cuda.Buffer,
    /// bf16 [rows, value heads * 128]
    delta_out: cuda.Buffer,
    /// fp32 [rows, value heads * 4]: sums the DeltaNet kernel writes and nothing reads
    delta_sums: cuda.Buffer,
    /// fp32 [rows, key heads, 128]: a wide window's queries, between its front end and its chain (`gdn.Chain.work`)
    delta_q: cuda.Buffer,
    /// bf16 [rows, hidden]
    branch: cuda.Buffer,
    moe: moe.Buffers,
    /// One for each layer with a PLE, in layer order.
    ple: []ple.Buffers,
    /// Each PLE's rows for the window `expected`, gathered while the window before it runs (`expect`).
    ahead: []ple.Ahead,
    /// The window expected after the next one staged, or none.
    expected: []const i32 = &.{},
    /// int32 [rows, 2]: the staged window's image rows (spec 019), each a row of the window and the row of
    /// `State.image`'s features it takes; `image_rows` of them, none for a window of text.
    image_pairs: cuda.Buffer,
    image_rows: usize = 0,
    /// The attention layers' scratch, sized for sequences of up to `capacity` positions; null without such layers.
    /// One serves every layer, a block leaving nothing there that the next reads, and so the MTP head's layer
    /// too: its buffers borrow the main model's (`initBeside`).
    attention: ?attention.Scratch,
    /// `attention` is another `Buffers`', which frees it and must outlive these.
    attention_lent: bool,
    /// bf16 [scored, vocab]: the rows a window's logits are asked of, `rows` or fewer (`Score`)
    logits: cuda.Buffer,
    /// With a DFlash2 drafter: the completed-layer taps, rising, and bf16 [rows, taps x hidden] their stream
    /// means side by side (`setTaps`). Without one nothing is launched for them.
    tap_layers: []const u32 = &.{},
    taps: ?cuda.Buffer = null,
    tap_mean: taps.Mean = .f32_acc,
    gpa: Allocator,

    /// `capacity`: the longest sequence, in positions, these buffers serve.
    pub fn init(gpa: Allocator, m: model.Model, rows: usize, capacity: usize) !Buffers {
        return initFor(gpa, m, m.layers, rows, rows, capacity);
    }

    /// Scratch for `layers` of `m`'s sizes: the model's own, or the MTP head's one. Windows of up to `scored`
    /// rows can have every row's logits; a prompt's wider ones have the last row's or none.
    pub fn initFor(gpa: Allocator, m: model.Model, layers: []const model.Layer, rows: usize, scored: usize, capacity: usize) !Buffers {
        return initBeside(gpa, m, layers, rows, scored, capacity, null);
    }

    /// `initFor`, the attention's scratch being `beside`'s where that one serves the same rows and capacity
    /// (spec 033: 740 MiB of a server's head buffers, most of it the indexer's scores of every block).
    pub fn initBeside(gpa: Allocator, m: model.Model, layers: []const model.Layer, rows: usize, scored: usize, capacity: usize, beside: ?*const Buffers) !Buffers {
        std.debug.assert(scored >= 1 and scored <= rows);
        const s = m.shape;
        var linear: usize = 0;
        for (layers) |layer| linear += @intFromBool(layer.block == .delta_net);
        var self: Buffers = undefined;
        self.rows = rows;
        self.gpa = gpa;
        // a failed init leaks what it allocated so far: the process is about to stop on that error
        self.ids = try .init(.i32, &.{rows});
        self.image_pairs = try .init(.i32, &.{ rows, 2 });
        self.image_rows = 0;
        self.h = try .init(.bf16, &.{ rows, s.wide() });
        self.pss = try .init(.f32, &.{ rows, s.hidden / hc.block, s.streams });
        self.normed = try .init(.bf16, &.{ rows, s.wide() });
        self.down = try .init(.f32, &.{ rows, s.low + s.streams });
        self.down_mixer = try .init(.f32, &.{ rows, s.low });
        self.act = try .init(.bf16, &.{ rows, s.low });
        self.inject_attn = try .init(.bf16, &.{ rows, s.streams });
        self.inject_mlp = try .init(.bf16, &.{ rows, s.streams });
        self.up = try .init(.bf16, &.{ rows, s.wide() });
        self.mixed = try .init(.bf16, &.{ rows, s.hidden });
        self.kept = keptRows(rows, scored);
        self.projections = try gpa.alloc(cuda.Buffer, linear);
        for (self.projections) |*projection| projection.* = try .init(.bf16, &.{ self.kept, s.delta_net.projection() });
        self.projection_wide = if (linear > 0 and rows > self.kept) try .init(.bf16, &.{ rows, s.delta_net.projection() }) else null;
        self.delta_out = try .init(.bf16, &.{ rows, s.delta_net.out() });
        self.delta_sums = try .init(.f32, &.{ rows, s.delta_net.out() / 32 });
        self.delta_q = try .init(.f32, &.{ rows, s.delta_net.key_heads, gdn.key_dim });
        self.branch = try .init(.bf16, &.{ rows, s.hidden });
        self.moe = try .init(rows, s.top, s.routed, s.width, s.hidden, moe.Weights.quantizesAny(layers));
        // a share's head scores half the vocabulary; on one rank `head.n` is `s.vocab`
        self.logits = try .init(.bf16, &.{ scored, m.head.n });
        const lent: ?attention.Scratch = if (m.attention == null) null else if (beside) |other| lent: {
            const scratch = other.attention orelse break :lent null;
            break :lent if (scratch.rows == rows and scratch.capacity == capacity) scratch else null;
        } else null;
        self.attention_lent = lent != null;
        self.attention = if (m.attention) |shared| lent orelse try .initBeside(shared.shape, shared.head, rows, capacity) else null;
        var branches: usize = 0;
        for (layers) |layer| branches += @intFromBool(layer.ple != null);
        self.ple = try gpa.alloc(ple.Buffers, branches);
        self.ahead = try gpa.alloc(ple.Ahead, branches);
        self.expected = &.{};
        branches = 0;
        for (layers) |layer| if (layer.ple) |branch| {
            self.ple[branches] = try .init(gpa, branch.weights.shape, rows);
            self.ahead[branches] = try .init(gpa, branch.weights.shape, rows);
            branches += 1;
        };
        self.tap_layers = &.{};
        self.taps = null;
        self.tap_mean = .f32_acc;
        return self;
    }

    /// The device bytes `initBeside` asks, with nothing made. `lent`: the attention's scratch is another
    /// `Buffers`' (`beside`'s, of these rows and this capacity) and is not asked again.
    pub fn bytesFor(m: model.Model, layers: []const model.Layer, rows: usize, scored: usize, capacity: usize, lent: bool) usize {
        const of = cuda.Buffer.bytesOf;
        const s = m.shape;
        var linear: usize = 0;
        for (layers) |layer| linear += @intFromBool(layer.block == .delta_net);
        const kept = keptRows(rows, scored);
        var bytes = of(.i32, &.{rows}) + of(.i32, &.{ rows, 2 }) + 3 * of(.bf16, &.{ rows, s.wide() }) + of(.f32, &.{ rows, s.hidden / hc.block, s.streams });
        bytes += of(.f32, &.{ rows, s.low + s.streams }) + of(.f32, &.{ rows, s.low }) + of(.bf16, &.{ rows, s.low }) + 2 * of(.bf16, &.{ rows, s.streams });
        bytes += 2 * of(.bf16, &.{ rows, s.hidden });
        bytes += linear * of(.bf16, &.{ kept, s.delta_net.projection() });
        if (linear > 0 and rows > kept) bytes += of(.bf16, &.{ rows, s.delta_net.projection() });
        bytes += of(.bf16, &.{ rows, s.delta_net.out() }) + of(.f32, &.{ rows, s.delta_net.out() / 32 }) + of(.f32, &.{ rows, s.delta_net.key_heads, gdn.key_dim });
        bytes += moe.Buffers.bytesFor(rows, s.top, s.routed, s.width, s.hidden, moe.Weights.quantizesAny(layers));
        bytes += of(.bf16, &.{ scored, m.head.n });
        if (m.attention) |shared| if (!lent) {
            bytes += attention.Scratch.bytesBeside(shared.shape, shared.head, rows, capacity);
        };
        for (layers) |layer| if (layer.ple) |branch| {
            bytes += ple.Buffers.bytesFor(branch.weights.shape, rows);
        };
        return bytes;
    }

    /// The widest window whose rows can all be scored.
    pub fn scoredRows(self: Buffers) usize {
        return @intCast(self.logits.view.size[0]);
    }

    /// Every forward from here on leaves the streams' mean after each of `layers` in `taps`. Call before a
    /// graph is captured over these buffers; `layers` outlives them. Decode path allocates nothing.
    pub fn setTaps(self: *Buffers, layers: []const u32, mean: taps.Mean) cuda.Error!void {
        std.debug.assert(self.taps == null and layers.len > 0);
        const hidden_dim: usize = @intCast(self.mixed.view.size[1]);
        self.taps = try .init(.bf16, &.{ self.rows, layers.len * hidden_dim });
        self.tap_layers = layers;
        self.tap_mean = mean;
    }

    /// Device bytes `setTaps` asks for `n` completed-layer means of `hidden_dim` over windows of `rows`.
    pub fn tapBytes(rows: usize, hidden_dim: usize, n: usize) usize {
        return cuda.Buffer.bytesOf(.bf16, &.{ rows, n * hidden_dim });
    }

    /// Tap `slot`'s columns of the first `rows` rows: [rows, hidden] strided over the taps beside it.
    pub fn tap(self: Buffers, slot: usize, rows: usize) cuda.View {
        const held = self.taps.?.view;
        const hidden_dim: usize = @intCast(self.mixed.view.size[1]);
        const base: [*]u8 = @ptrCast(held.ptr.?);
        var view: cuda.View = .init(@ptrCast(base + slot * hidden_dim * 2), .bf16, &.{ rows, hidden_dim });
        view.row_stride = held.size[1];
        return view;
    }

    pub fn deinit(self: *Buffers) void {
        if (self.taps) |*one| one.deinit();
        inline for (.{ "ids", "image_pairs", "h", "pss", "normed", "down", "down_mixer", "act", "inject_attn", "inject_mlp", "up", "mixed", "delta_out", "delta_sums", "delta_q", "branch", "logits" }) |name| @field(self, name).deinit();
        for (self.projections) |*projection| projection.deinit();
        self.gpa.free(self.projections);
        if (self.projection_wide) |*projection| projection.deinit();
        self.moe.deinit();
        for (self.ahead) |*branch| branch.deinit(self.gpa);
        self.gpa.free(self.ahead);
        for (self.ple) |*branch| branch.deinit(self.gpa);
        self.gpa.free(self.ple);
        if (!self.attention_lent) if (self.attention) |*scratch| scratch.deinit();
        self.* = undefined;
    }
};

/// The rows each DeltaNet layer keeps of a window, for buffers of `rows` rows of which `scored` can be scored:
/// a window that may be kept in part has every row scored, and the conv inputs are three rows.
pub fn keptRows(rows: usize, scored: usize) usize {
    return @min(rows, @max(scored, gdn.taps - 1));
}

/// An image prompt's rows (spec 019): what the tower made for its placeholders, which the forward writes over
/// their embeddings in every stream, before the PLE (`decode/placeholders.zig`). Lent: the caller's until the state
/// lets it go (`State.image` null again, at the last row's commit or `reset`).
pub const Image = placeholders.Placeholders;

/// One sequence's committed state. A round of several streams keeps one `rounds.Slot` a stream and one
/// `rounds.Shared` scratch; this packing is the one-stream path, and what a session holds.
pub const State = struct {
    /// Positions committed so far.
    pos: usize = 0,
    /// For each DeltaNet layer, two fp32 [value heads, 128, 128]: `current` picks the committed one, and a
    /// window writes the other. With rows `behind` it is not the state at `pos`: read that by `committed`.
    recurrent: [][2]cuda.Buffer,
    current: []u1,
    /// For each DeltaNet layer, bf16 [3, conv channels]: the conv inputs of the last three committed rows.
    conv: []cuda.Buffer,
    /// What replaying a prefix of the last window needs, for each DeltaNet layer: `kept` rows of it.
    scratch: []gdn.Scratch,
    /// The rows at the front of each `scratch` that the committed side of `recurrent` does not hold yet
    /// (spec 089 WS-03). A window kept in part is not replayed by its `commit`: its kept rows wait here, the
    /// next window's chain replays them on the state it loads anyway and saves its own rows after them, and
    /// the state is read and written once a round, not twice. The state at `pos` is then the committed side
    /// replayed over these rows (`committed`); a window that cannot go on from them (`lands`) has them
    /// replayed first (`land`). The device has the count beside the position (`attention.Position.put`).
    behind: usize = 0,
    /// The rows of a window each layer keeps (`Buffers.kept`, which a session's buffers must agree with).
    kept: usize,
    /// The same for a window wider than `kept`, one for every layer: the wide kernels work in it, and a replay
    /// to a row of such a window runs right after the layer's chain, before the next layer writes over it.
    /// Null where no window is wider. Each layer's own at 2048 rows was 1.5 GiB (spec 033).
    wide: ?gdn.Scratch,
    /// For each DeltaNet layer, bf16 [1, 3, conv channels]: the conv inputs after the last window wider than
    /// `kept`, which `commit` copies. Empty without `wide`.
    conv_next: []cuda.Buffer,
    /// The states wanted inside the next window wider than `kept` (`Snapshot.ahead`), lowest first: the
    /// forward writes each one's DeltaNet states and conv inputs. The caller's, and it empties the list after
    /// that forward.
    wanted: []const Snapshot = &.{},
    /// With `wanted`, by a caller that `defers` them: the forward leaves their DeltaNet states to `settle`,
    /// which the caller runs once the window is committed and its logits are read, and each layer's chain
    /// writes what a replay needs into the highest wanted state's own buffer (`Snapshot.stash`) instead of
    /// the scratch every layer shares. The replays are then time after a reply's first token and not before
    /// it: 2 ms of a 23-row window on NVFP4 with one state wanted 21 rows in (spec 041).
    stashed: bool = false,
    /// The n-gram history and conv tail of each layer with a PLE, in layer order.
    ple: []ple.State,
    /// Each attention layer's keys, values and indexer blocks, in layer order.
    caches: []attention.Cache,
    /// `pos` on the device, which every attention layer reads.
    position: attention.Position,
    /// The rotary positions the attention layers read beside `position` (spec 019): zeros for text, an image
    /// prompt's three axes and delta from `attach` to `reset`.
    rope: attention.Rope,
    /// An image prompt's features while rows of its placeholders are still to run (`attach`); `commit` drops it
    /// once the last is committed, as TensorFold's `image_rows.finish` ends them with the prompt.
    image: ?Image = null,
    gpa: Allocator,

    /// `capacity`: the positions the attention caches hold.
    pub fn init(gpa: Allocator, m: model.Model, max_rows: usize, capacity: usize) !State {
        return initKeeping(gpa, m, max_rows, max_rows, capacity);
    }

    /// `init` for buffers of which `scored` rows can be scored (`Buffers.initFor`): each layer keeps
    /// `keptRows` of a window, and wider windows share one scratch.
    pub fn initKeeping(gpa: Allocator, m: model.Model, max_rows: usize, scored: usize, capacity: usize) !State {
        const dn = m.shape.delta_net;
        var linear: usize = 0;
        for (m.layers) |layer| linear += @intFromBool(layer.block == .delta_net);
        const kept = keptRows(max_rows, scored);
        const wide = linear > 0 and max_rows > kept;
        var self: State = .{
            .recurrent = try gpa.alloc([2]cuda.Buffer, linear),
            .current = try gpa.alloc(u1, linear),
            .conv = try gpa.alloc(cuda.Buffer, linear),
            .scratch = try gpa.alloc(gdn.Scratch, linear),
            .kept = kept,
            .wide = if (wide) try .init(max_rows, dn) else null,
            .conv_next = try gpa.alloc(cuda.Buffer, if (wide) linear else 0),
            .ple = undefined,
            .caches = undefined,
            .position = try .init(),
            .rope = try .init(if (m.attention != null) capacity else 0),
            .gpa = gpa,
        };
        var attending: usize = 0;
        for (m.layers) |layer| attending += @intFromBool(layer.block == .attention);
        self.caches = try gpa.alloc(attention.Cache, attending);
        for (self.caches) |*cache| cache.* = try .init(m.attention.?.shape, capacity);
        for (self.recurrent, self.current, self.conv, self.scratch) |*pair, *current, *conv, *scratch| {
            for (pair) |*buffer| {
                buffer.* = try .init(.f32, &.{ dn.value_heads, gdn.value_dim, gdn.key_dim });
                try buffer.zero();
            }
            current.* = 0;
            conv.* = try .init(.bf16, &.{ 1, gdn.taps - 1, dn.conv() });
            try conv.zero();
            scratch.* = try .init(kept, dn);
        }
        for (self.conv_next) |*conv| conv.* = try .init(.bf16, &.{ 1, gdn.taps - 1, dn.conv() });
        var branches: usize = 0;
        for (m.layers) |layer| branches += @intFromBool(layer.ple != null);
        self.ple = try gpa.alloc(ple.State, branches);
        branches = 0;
        for (m.layers) |layer| if (layer.ple) |branch| {
            self.ple[branches] = try .init(gpa, branch.weights.shape, branch.eos);
            branches += 1;
        };
        return self;
    }

    /// Back to before the sequence's first row, as `init` leaves it, for the next sequence. The attention caches
    /// keep their rows: no row reads one at or past its own position + 1 (`attention.Position.advance`).
    pub fn reset(self: *State) cuda.Error!void {
        for (self.recurrent, self.current, self.conv) |pair, *current, conv| {
            for (pair) |buffer| try buffer.zero();
            current.* = 0;
            try conv.zero();
        }
        for (self.ple) |*branch| try branch.reset();
        try self.position.set(0);
        try self.rope.clear();
        self.image = null;
        self.pos = 0;
        self.behind = 0;
    }

    /// An image prompt from its first row (TensorFold 0.6.5's `image_rows.attach`; its `prefill` refuses an
    /// image prompt that resumes, decode.py:354): `positions` [prompt rows, 3], each row's t, h and w, and
    /// `delta`, which every row past them adds to its own position (`Rope`); `image` the placeholders'
    /// features and positions, lent until the last of them is committed. A text prompt attaches nothing.
    pub fn attach(self: *State, image: Image, positions: []const [3]i32, delta: i32) (error{InvalidImage} || cuda.Error)!void {
        if (self.pos != 0 or positions.len > self.rope.capacity or image.rows.len == 0 or
            image.features.size[0] != image.rows.len) return error.InvalidImage;
        for (image.rows, 0..) |row, i| {
            if (row >= positions.len or (i > 0 and row <= image.rows[i - 1])) return error.InvalidImage;
        }
        try self.rope.set(positions, delta);
        self.image = image;
    }

    pub fn deinit(self: *State) void {
        for (self.recurrent, self.conv, self.scratch) |*pair, *conv, *scratch| {
            for (pair) |*buffer| buffer.deinit();
            conv.deinit();
            scratch.deinit();
        }
        if (self.wide) |*scratch| scratch.deinit();
        for (self.conv_next) |*conv| conv.deinit();
        self.gpa.free(self.conv_next);
        self.gpa.free(self.recurrent);
        self.gpa.free(self.current);
        self.gpa.free(self.conv);
        self.gpa.free(self.scratch);
        for (self.ple) |*branch| branch.deinit(self.gpa);
        self.gpa.free(self.ple);
        for (self.caches) |*cache| cache.deinit();
        self.gpa.free(self.caches);
        self.position.deinit();
        self.rope.deinit();
        self.* = undefined;
    }
};
