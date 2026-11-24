//! What a Nemotron sequence carries from one window to the next, and the scratch a window's forward writes
//! (`cuda_state.zig`'s `Buffers`, split: the sequence's own from the shared scratch). `State` is one
//! sequence's committed state: its position, the Mamba layers' states (`mamba.State`, each with its meta),
//! every attention layer's KV cache, and the MTP head's own cache and position. `Buffers` is window scratch
//! shared by the layers: the rows' streams (`h`/`x`/`y`), the norms' group sums, the projections' products,
//! the experts' activations and the logits — allocated once for the most rows a window or a prompt chunk
//! runs, never kept between calls.
//!
//! A serve's audio clips are the state's `audio`: the perception's features laid over the prompt's
//! placeholder rows as the forward writes their embeddings (`decode/placeholders.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const config_file = @import("config.zig");
const placeholders = @import("../../decode/placeholders.zig");
const mamba_file = @import("mamba.zig");
const attention_file = @import("attention.zig");
const moe_file = @import("moe.zig");
const model_file = @import("model.zig");
const experts_file = @import("../../weights/experts.zig");
const ranks_file = @import("ranks.zig");

/// A sequence's committed state: what a window starts from and a `commit` advances.
pub const State = struct {
    /// The allocator the slices are freed with (`deinit`).
    gpa: Allocator,
    /// Tokens committed, the next row's position. The only field `commit` writes besides the layers'.
    pos: usize = 0,
    /// The prompt's audio rows and what the perception made for them (`serve`'s clips, a
    /// transcription's one), lent by the caller until `reset`; `forward` lays the features over
    /// the rows' token embeddings (`decode/placeholders.zig`, as `qwen3_asr`'s `audio`).
    audio: ?placeholders.Placeholders = null,
    /// The sequence's room: pos never exceeds it (the caches and the mamba states' windows are sized by it).
    capacity: usize,
    /// The position word the attention layers' prep and attend read, int32 [1] on the device: pos at each
    /// window's launch (kept in step by `commit`).
    position: cuda.Buffer,
    /// The Mamba layers' states, in layer order among the mamba blocks only; `meta[i]` is layer mamba-index
    /// i's own meta row.
    mamba: []mamba_file.State,
    mamba_meta: []mamba_file.Meta,
    /// The attention layers' KV caches, in layer order among the attention blocks only.
    attention: []attention_file.Cache,
    /// The MTP head's state: its attention's KV cache and its own position (rows absorbed and drafted).
    mtp_cache: ?attention_file.Cache = null,
    mtp_pos: usize = 0,

    /// `initS` on the whole shapes (`Shapes.of` of a null share — a serial run's).
    pub fn init(gpa: Allocator, c: *const config_file.Config, capacity: usize) (cuda.Error || Allocator.Error)!State {
        return initS(gpa, c, capacity, .of(c.backbone, null));
    }

    /// `init` on the given mixer shapes: the two-ranks run's halves (`ranks.Shapes`), the whole
    /// `Shape.of(c)`'s in serial.
    pub fn initS(gpa: Allocator, c: *const config_file.Config, capacity: usize, shapes: ranks_file.Shapes) (cuda.Error || Allocator.Error)!State {
        const b = c.backbone;
        const mshape = shapes.mamba;
        const ashape = shapes.attention;
        var position: cuda.Buffer = try .init(.i32, &.{1});
        errdefer position.deinit();
        try position.zero();
        const nm = b.count(.mamba);
        const na = b.count(.attention);
        const mamba = try gpa.alloc(mamba_file.State, nm);
        errdefer gpa.free(mamba);
        var made_m: usize = 0;
        errdefer for (mamba[0..made_m]) |*st| st.deinit();
        for (mamba) |*st| {
            st.* = try .init(mshape);
            made_m += 1;
        }
        const meta = try gpa.alloc(mamba_file.Meta, nm);
        errdefer gpa.free(meta);
        @memset(meta, .{});
        const attention = try gpa.alloc(attention_file.Cache, na);
        errdefer gpa.free(attention);
        var made_a: usize = 0;
        errdefer for (attention[0..made_a]) |*cache| cache.deinit();
        for (attention) |*cache| {
            cache.* = try .init(ashape, capacity);
            made_a += 1;
        }
        var mtp_cache: ?attention_file.Cache = null;
        if (b.mtp_layers != 0) mtp_cache = try .init(ashape, capacity);
        return .{
            .gpa = gpa,
            .capacity = capacity,
            .position = position,
            .mamba = mamba,
            .mamba_meta = meta,
            .attention = attention,
            .mtp_cache = mtp_cache,
        };
    }

    /// A new sequence over the same buffers: pos and the meta rows zero, the layers' states zeroed; the
    /// attention caches need no clear (a row's reads are bounded by its position, which the zeroed meta
    /// gives).
    pub fn reset(state: *State) cuda.Error!void {
        state.pos = 0;
        state.mtp_pos = 0;
        state.audio = null;
        for (state.mamba, state.mamba_meta) |*st, *meta| {
            try st.zero();
            meta.* = .{};
        }
        return state.position.zero();
    }

    pub fn deinit(state: *State) void {
        state.position.deinit();
        for (state.mamba) |*st| st.deinit();
        for (state.attention) |*cache| cache.deinit();
        if (state.mtp_cache) |*cache| cache.deinit();
        state.gpa.free(state.mamba);
        state.gpa.free(state.mamba_meta);
        state.gpa.free(state.attention);
        state.* = undefined;
    }
};

/// A window's scratch: made once for `rows` (the most a window or a prompt chunk runs) and the model's
/// shapes, shared by every layer. Nothing in it is kept between calls; each call overwrites what it reads.
pub const Buffers = struct {
    rows: usize,
    /// int32 [rows]: the window's token ids.
    ids: cuda.Buffer,
    /// bf16 [rows, hidden]: the residual stream (the embeddings, then each block's output added in).
    h: cuda.Buffer,
    /// A second residual row where a norm's h and the next input must differ (the add-RMSNorm's h is the
    /// residual's new value: h = bf16(x + delta)).
    hn: cuda.Buffer,
    /// bf16 [rows, hidden]: the block-norm's output, the mixers' input.
    y: cuda.Buffer,
    /// fp32 [rows, hidden / 64]: the norms' group sums (nothing reads them; the kernels write them).
    xs: cuda.Buffer,
    /// bf16 [rows, hidden]: a dense layer's product (o_proj, out_proj) before its residual add.
    delta: cuda.Buffer,
    /// bf16 [rows, mamba_proj]: a Mamba block's in_proj rows.
    proj: cuda.Buffer,
    /// bf16 [rows, mamba_inner]: a Mamba mixer's normed output.
    gate: cuda.Buffer,
    /// bf16 [rows, hidden]: an attention layer's o_proj output, the residual's delta.
    att: cuda.Buffer,
    /// bf16 [scored, vocab]: the rows' logits (scored ≤ rows).
    logits: cuda.Buffer,
    /// int32 [rows, 2]: the window's audio rows and their features' rows (`placeholders.within`).
    pairs: cuda.Buffer,
    /// The vocab head's bf16 [vocab, hidden] is what `head` is: logits bf16 [scored, vocab].
    scored: usize,
    /// The mixers' shared scratch: one a kind (mamba's conv/scan parts, attention's q/qkv and chunk
    /// partials, MoE's router slices, plan, activations and slot rows).
    mamba_sc: mamba_file.Scratch,
    attn_sc: attention_file.Scratch,
    moe_sc: moe_file.Scratch,
    /// i32 [4]: the mamba meta row a window uploads then reads (each layer's own `state.mamba_meta` is the
    /// host copy; the kernels read this word after `Meta.set` writes it).
    meta: cuda.Buffer,
    /// i32 [4]: the MTP head's own meta word.
    mtp_meta: cuda.Buffer,

    /// With a DFlash2 drafter: the completed-layer taps, rising, and bf16 [rows, taps x hidden] the rows'
    /// residual streams after each side by side (`setTaps`; `forward.zig`'s `capture`). Without one nothing
    /// is launched for them.
    tap_layers: []const u32 = &.{},
    taps: ?cuda.Buffer = null,

    /// `rows` the most a window or a prompt chunk holds; `capacity` the sequence's, which the attention
    /// partials and the caches are sized by.
    pub fn init(c: *const config_file.Config, rows: usize, scored: usize, capacity: usize) cuda.Error!Buffers {
        return initS(c, rows, scored, capacity, .of(c.backbone, null), c.backbone.vocab);
    }

    /// `init` on the given mixer shapes and `vocab` logits rows: a rank's halves and half-vocabulary at two
    /// ranks (`ranks.Shapes`, `vocab / world`), the whole shapes and vocabulary without one.
    pub fn initS(c: *const config_file.Config, rows: usize, scored: usize, capacity: usize, shapes: ranks_file.Shapes, vocab: usize) cuda.Error!Buffers {
        const b = c.backbone;
        const mshape = shapes.mamba;
        const ashape = shapes.attention;
        const eshape = shapes.moe;
        const r = @max(rows, attention_file.window_rows);
        var b2: Buffers = undefined;
        b2.rows = r;
        b2.scored = scored;
        b2.tap_layers = &.{};
        b2.taps = null;
        b2.ids = try .init(.i32, &.{r});
        errdefer b2.ids.deinit();
        b2.pairs = try .init(.i32, &.{ r, 2 });
        errdefer b2.pairs.deinit();
        b2.h = try .init(.bf16, &.{ r, b.hidden });
        errdefer b2.h.deinit();
        b2.hn = try .init(.bf16, &.{ r, b.hidden });
        errdefer b2.hn.deinit();
        b2.y = try .init(.bf16, &.{ r, b.hidden });
        errdefer b2.y.deinit();
        b2.xs = try .init(.f32, &.{ r, b.hidden / 64 });
        errdefer b2.xs.deinit();
        b2.delta = try .init(.bf16, &.{ r, b.hidden });
        errdefer b2.delta.deinit();
        b2.proj = try .init(.bf16, &.{ r, mshape.proj() });
        errdefer b2.proj.deinit();
        b2.gate = try .init(.bf16, &.{ r, mshape.xd });
        errdefer b2.gate.deinit();
        b2.att = try .init(.bf16, &.{ r, b.hidden });
        errdefer b2.att.deinit();
        b2.logits = try .init(.bf16, &.{ scored, vocab });
        errdefer b2.logits.deinit();
        b2.mamba_sc = try .init(mshape, r);
        errdefer b2.mamba_sc.deinit();
        b2.attn_sc = try .init(ashape, r, capacity);
        errdefer b2.attn_sc.deinit();
        b2.moe_sc = try .init(eshape, r, true);
        errdefer b2.moe_sc.deinit();
        b2.meta = try .init(.i32, &.{4});
        errdefer b2.meta.deinit();
        b2.mtp_meta = try .init(.i32, &.{4});
        errdefer b2.mtp_meta.deinit();
        return b2;
    }

    /// Every forward from here on leaves the residual stream after each of `layers` (zero-based layer ids,
    /// the drafter's `config.taps`) in `taps`. Call before a graph is captured over these buffers; `layers`
    /// outlives them. Decode path allocates nothing.
    pub fn setTaps(b: *Buffers, layers: []const u32) cuda.Error!void {
        std.debug.assert(b.taps == null and layers.len > 0);
        const hidden_dim: usize = @intCast(b.h.view.size[1]);
        b.taps = try .init(.bf16, &.{ b.rows, layers.len * hidden_dim });
        b.tap_layers = layers;
    }

    /// Device bytes `setTaps` asks for `n` taps of `hidden_dim` over windows of `rows`.
    pub fn tapBytes(rows: usize, hidden_dim: usize, n: usize) usize {
        return cuda.Buffer.bytesOf(.bf16, &.{ rows, n * hidden_dim });
    }

    /// Tap `slot`'s columns of the first `rows` rows: [rows, hidden] strided over the taps beside it.
    pub fn tap(b: Buffers, slot: usize, rows: usize) cuda.View {
        const held = b.taps.?.view;
        const hidden_dim: usize = @intCast(b.h.view.size[1]);
        const base: [*]u8 = @ptrCast(held.ptr.?);
        var view: cuda.View = .init(@ptrCast(base + slot * hidden_dim * 2), .bf16, &.{ rows, hidden_dim });
        view.row_stride = held.size[1];
        return view;
    }

    pub fn scoredRows(b: Buffers) usize {
        return b.scored;
    }

    pub fn deinit(b: *Buffers) void {
        if (b.taps) |*one| one.deinit();
        b.ids.deinit();
        b.pairs.deinit();
        b.h.deinit();
        b.hn.deinit();
        b.y.deinit();
        b.xs.deinit();
        b.delta.deinit();
        b.proj.deinit();
        b.gate.deinit();
        b.att.deinit();
        b.logits.deinit();
        b.mamba_sc.deinit();
        b.attn_sc.deinit();
        b.moe_sc.deinit();
        b.meta.deinit();
        b.mtp_meta.deinit();
        b.* = undefined;
    }
};
