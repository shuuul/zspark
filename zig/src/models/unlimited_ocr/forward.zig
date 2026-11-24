//! Unlimited-OCR-2's language model forward: one window of rows after a sequence's state, as `decode/draft.zig`'s
//! `Decoder` asks (`draft.zig`). Semantics from the checkpoint's modeling_deepseekv2.py (DeepseekV2DecoderLayer,
//! DeepseekV2MoE, MoEGate; Apache-2.0, read 2026-10-25) and its DASH attention with Triton off (spec 147 WS-05)
//! under `infer`'s bf16 autocast. A layer:
//!
//! 1. h += o(attention(rope(qkv(norm(h))))), MHA of 10 heads of 128 under DASH: q and k turned by page2d RoPE at
//!    the rows' coordinates (`zs_attn_prep_page2d`, `Model.page2d`), the router's `project`, `prep`, `kbar` and
//!    `route` over the state's index keys and k_bar (`router.zig`), then the attend over the causal window of 128
//!    keys and the 12 blocks it chose (`dash_attention.zig`; the step kernel to `dash_step_rows` rows, the tile
//!    kernel above: the same bits). Its scores and probabilities fp32 (FlashAttention's, vLLM's), where HF's
//!    eager attention rounds both to bf16;
//! 2. h += mlp(norm(h)): layer 0 a dense SwiGLU MLP; layers 1 on the gate's bf16 logits, their fp32 softmax and
//!    top 6 unnormalized (`moe.cu`), the routed experts' SwiGLU MLPs (`kernels/bf16/experts.cu`), their sum
//!    weighted in fp32 and rounded, plus the fused shared MLP's output, unscaled.
//!
//! The window's rows at positions pos .. pos + n - 1 write their keys and values at those rows of the caches
//! before any row reads them, so a window that is never committed leaves nothing a later one reads. A row's
//! bits are its own: the linears' K slices follow the weight's shape, attention's key tiles go in index order,
//! an expert's pair its own row of the product (the row_rule test).
//!
//! Image rows: the state's `image` lays precomputed embeddings (WS-03's tower and projector, or a fixture's)
//! over the rows whose positions it covers, after the token embeddings, so a prompt runs through the decoder's
//! windows as text does; several pictures (the pages, a `<PAGE>` between) are runs of one features buffer
//! (`Image.runs`). `forwardEmbeds` takes a whole window's embeddings instead.
//!
//! A row's coordinates come from the prompt whole (`State.paged`, `pages.plan`) and the reply's tokens
//! (`pages.Track`): `stage` writes them with the window's words, `commit` carries the track over the rows kept.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../weights/bf16.zig");
const config_file = @import("config.zig");
const cuda = @import("../../cuda/cuda.zig");
const dash_attention = @import("dash_attention.zig");
const experts_file = @import("../../weights/experts.zig");
const glue = @import("../../decode/glue.zig");
const graphs_file = @import("../../decode/graphs.zig");
const model_file = @import("model.zig");
const pages = @import("pages.zig");
const router_file = @import("router.zig");

pub const Model = model_file.Model;
const Config = config_file.Config;

pub const Error = cuda.Error || pages.Error || error{ WindowTooWide, ContextFull };

/// The most rows a DASH window takes the step kernel for (`dash_attention.zig`: about 8, by cost alone).
pub const dash_step_rows = 8;

/// Precomputed embeddings laid over rows `first` .. `first + features.size[0] - 1` of the sequence, or over
/// `runs`, each its rows of `features` in turn.
pub const Image = struct {
    first: usize = 0,
    /// bf16 [rows, hidden], held by the caller while the state runs over them
    features: cuda.Buffer,
    /// Several pictures' rows, in the order of `features`' rows, the caller's; empty: one run at `first`.
    runs: []const Run = &.{},

    pub const Run = struct { first: usize, rows: usize };

    /// The `i`th run and its first row of `features`.
    fn run(self: Image, i: usize, from: *usize) Run {
        if (self.runs.len == 0) return .{ .first = self.first, .rows = @intCast(self.features.view.size[0]) };
        if (i > 0) from.* += self.runs[i - 1].rows;
        return self.runs[i];
    }

    fn count(self: Image) usize {
        return @max(1, self.runs.len);
    }
};

/// One sequence: its position, the caches (a layer's keys and values bf16 [capacity, heads, head_dim]) and DASH's
/// state beside them.
pub const State = struct {
    pos: usize = 0,
    image: ?Image = null,
    keys: []cuda.Buffer,
    values: []cuda.Buffer,
    capacity: usize,
    /// Filled by `init`'s `initDash`.
    dash: Dash = undefined,
    gpa: Allocator,

    /// What DASH keeps of a sequence beside the caches, all by position: a rejected row's are written again by the
    /// next window at its position (k_bar's block by whichever window holds its last row), so nothing rolls back.
    pub const Dash = struct {
        /// a layer's index keys, bf16 [capacity rounded up to a block, 128] (k_bar's kernel reads whole blocks)
        idx_k: []cuda.Buffer,
        /// a layer's block summaries, fp32 [capacity / 32, 128]: whole, so a route's shared memory and a graph's
        /// arguments are the same at every position
        kbar: []cuda.Buffer,
        /// int32 [capacity / 32, 2]: a block's image pages (`pages.Pages`), the prompt's, -1 after
        pages: cuda.Buffer,
        /// host [capacity]: the prompt's rows' coordinates (`paged`), then the staged window's
        coords: []pages.Coords,
        /// host [capacity / 32] and [capacity + 1]: `paged`'s room for the block pages and `Track.starts`
        blocks: []pages.Pages,
        starts: []i32,
        /// the reply's state machine after the last row kept past the prompt; null before `paged`
        track: ?pages.Track = null,
        prompt: usize = 0,
        tokens: pages.Tokens,
        /// What a row's RoPE page adds where it is past the instruction's (page > 0): a part's first page in the
        /// document (`documents.pageBase`, SGLang's `_shift_page_coords`); its query and block pages stay the part's.
        base: i32 = 0,
    };

    pub fn init(gpa: Allocator, config: Config, capacity: usize) !State {
        var state = try initCaches(gpa, config, capacity);
        errdefer state.deinitCaches();
        state.dash = try initDash(gpa, config, config.dash, capacity);
        return state;
    }

    fn initDash(gpa: Allocator, config: Config, dash: config_file.Dash, capacity: usize) !Dash {
        // the attend's and the route's bounds (`dash_attention.cu`, `router.cu`)
        if (capacity < router_file.block or capacity > 524_288) return error.WindowTooWide;
        const blocks = (capacity + router_file.block - 1) / router_file.block;
        const idx_k = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(idx_k);
        const kbar = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(kbar);
        var made: usize = 0;
        errdefer for (idx_k[0..made], kbar[0..made]) |*k, *b| {
            k.deinit();
            b.deinit();
        };
        for (idx_k, kbar) |*k, *b| {
            k.* = try .init(.bf16, &.{ blocks * router_file.block, router_file.dim });
            b.* = cuda.Buffer.init(.f32, &.{ blocks, router_file.dim }) catch |err| {
                k.deinit();
                return err;
            };
            made += 1;
            try k.zero();
            try b.zero();
        }
        var page_buffer: cuda.Buffer = try .init(.i32, &.{ blocks, 2 });
        errdefer page_buffer.deinit();
        const coords = try gpa.alloc(pages.Coords, capacity);
        errdefer gpa.free(coords);
        const block_pages = try gpa.alloc(pages.Pages, blocks);
        errdefer gpa.free(block_pages);
        const starts = try gpa.alloc(i32, capacity + 1);
        return .{
            .idx_k = idx_k,
            .kbar = kbar,
            .pages = page_buffer,
            .coords = coords,
            .blocks = block_pages,
            .starts = starts,
            .tokens = .{ .image = @intCast(dash.image_token), .page = @intCast(dash.page_token), .per_page = dash.page_image_tokens },
        };
    }

    fn initCaches(gpa: Allocator, config: Config, capacity: usize) !State {
        const keys = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(cuda.Buffer, config.layers);
        errdefer gpa.free(values);
        var made: usize = 0;
        errdefer for (keys[0..made], values[0..made]) |*k, *v| {
            k.deinit();
            v.deinit();
        };
        for (keys, values) |*k, *v| {
            k.* = try .init(.bf16, &.{ capacity, config.heads, config.head_dim });
            v.* = cuda.Buffer.init(.bf16, &.{ capacity, config.heads, config.head_dim }) catch |err| {
                k.deinit();
                return err;
            };
            made += 1;
        }
        return .{ .keys = keys, .values = values, .capacity = capacity, .gpa = gpa };
    }

    fn deinitCaches(self: *State) void {
        for (self.keys, self.values) |*k, *v| {
            k.deinit();
            v.deinit();
        }
        self.gpa.free(self.keys);
        self.gpa.free(self.values);
    }

    pub fn deinit(self: *State) void {
        self.deinitCaches();
        const d = &self.dash;
        for (d.idx_k, d.kbar) |*k, *b| {
            k.deinit();
            b.deinit();
        }
        self.gpa.free(d.idx_k);
        self.gpa.free(d.kbar);
        d.pages.deinit();
        self.gpa.free(d.coords);
        self.gpa.free(d.blocks);
        self.gpa.free(d.starts);
        self.* = undefined;
    }

    /// A new sequence in the same caches.
    pub fn reset(self: *State) void {
        self.pos = 0;
        self.image = null;
        self.dash.track = null;
        self.dash.prompt = 0;
        self.dash.base = 0;
    }

    /// The whole prompt's ids, before its first window. Its rows' coordinates and its blocks' image pages come
    /// from the prompt whole (`pages.plan`); a reply's rows from `pages.Track` after it.
    pub fn paged(self: *State, prompt: []const i32) Error!void {
        const d = &self.dash;
        if (prompt.len == 0 or prompt.len > self.capacity) return error.ContextFull;
        const blocks = (prompt.len + router_file.block - 1) / router_file.block;
        @memset(d.blocks, pages.none);
        d.track = null;
        d.track = try pages.plan(prompt, d.tokens, router_file.block, d.coords[0..prompt.len], d.blocks[0..blocks], d.starts);
        for (d.coords[0..prompt.len]) |*c| c.page = shifted(c.page, d.base);
        d.prompt = prompt.len;
        try d.pages.upload(std.mem.sliceAsBytes(d.blocks));
    }

    /// `paged` for a part of a long document whose pages start at the document's page `base` (`Dash.base`).
    pub fn pagedFrom(self: *State, prompt: []const i32, base: i32) Error!void {
        self.dash.base = base;
        return self.paged(prompt);
    }

    /// A reused part's text rows, laid at the state's position after its prompt (`parts.zig`): the reply goes on
    /// from its first page with them behind it, as SGLang's `_splice_text` carries it (`TextPlan.coord`), so the
    /// forced `<PAGE>` after them starts the part's second page. The rows' keys stay as the previous part wrote them.
    pub fn carried(self: *State, rows: usize) void {
        self.pos += rows;
        if (self.dash.track) |*t| {
            t.page = 1;
            t.markers = t.prompt_pages + 1;
        }
    }
};

/// What a forward writes and nothing keeps.
pub const Buffers = struct {
    rows: usize,
    scored: usize,
    /// int32 [1 + rows]: the window's first position, then its tokens. The kernels read the position here, so a
    /// captured window serves every step (`stage`).
    ids: cuda.Buffer,
    /// bf16 [rows, hidden]: the residual stream
    h: cuda.Buffer,
    /// bf16 [rows, hidden]: a norm's output
    x: cuda.Buffer,
    qkv: cuda.Buffer,
    q: cuda.Buffer,
    attn: cuda.Buffer,
    /// bf16 [rows, hidden]: a branch's output before its add
    branch: cuda.Buffer,
    /// bf16 [rows, 2 widest]: a dense MLP's gate and up; act [rows, widest]
    gu: cuda.Buffer,
    act: cuda.Buffer,
    /// bf16 [rows, experts], int32 and fp32 [rows, top]: the router's logits, picks and weights (the last MoE
    /// layer's, after a forward)
    router: cuda.Buffer,
    picks: cuda.Buffer,
    weights: cuda.Buffer,
    plan: experts_file.Plan,
    /// bf16 [rows top, width] in plan order, and bf16 [rows top, hidden] at each pair
    expert_act: cuda.Buffer,
    expert_out: cuda.Buffer,
    /// fp32 [rows, heads, slots, 130] and int32 [rows, heads]: the step kernel's attention chunks and their
    /// counter (`dash_attention.cu`), the counters zero between calls
    parts: cuda.Buffer,
    done: cuda.Buffer,
    /// A DASH window's router rows.
    dash: Dash,
    /// bf16 [rows, hidden]: the final norm's rows
    final: cuda.Buffer,
    /// bf16 [scored, vocab]
    logits: cuda.Buffer,

    pub const Dash = struct {
        /// bf16 [rows, 256]: the index q and k projections; idx_q bf16 [rows, 128]
        qk: cuda.Buffer,
        idx_q: cuda.Buffer,
        /// int32 and fp32 [rows, read_blocks]: a row's chosen blocks (-1 an empty slot) and their bias
        ids: cuda.Buffer,
        bias: cuda.Buffer,
        /// int32 [rows, 3]: the window's coordinates (`pages.Coords`), staged with its words
        coords: cuda.Buffer,

        fn init(rows: usize, dash: config_file.Dash) !Dash {
            var list: [5]cuda.Buffer = undefined;
            var made: usize = 0;
            errdefer for (list[0..made]) |*buffer| buffer.deinit();
            const shapes = [_]struct { cuda.DType, []const usize }{
                .{ .bf16, &.{ rows, 2 * router_file.dim } },
                .{ .bf16, &.{ rows, router_file.dim } },
                .{ .i32, &.{ rows, dash.read_blocks } },
                .{ .f32, &.{ rows, dash.read_blocks } },
                .{ .i32, &.{ rows, 3 } },
            };
            for (shapes, 0..) |shape, i| {
                list[i] = try .init(shape[0], shape[1]);
                made += 1;
            }
            return .{ .qk = list[0], .idx_q = list[1], .ids = list[2], .bias = list[3], .coords = list[4] };
        }

        fn deinit(self: *Dash) void {
            inline for (.{ "qk", "idx_q", "ids", "bias", "coords" }) |name| @field(self, name).deinit();
        }
    };

    pub fn init(config: Config, rows: usize, scored: usize) !Buffers {
        const c = config;
        const chunks = dash_attention.slots(c.dash.read_blocks, c.dash.local_blocks * c.dash.block);
        const widest = @max(c.intermediate, c.sharedWidth());
        var list: [18]cuda.Buffer = undefined;
        var made: usize = 0;
        errdefer for (list[0..made]) |*buffer| buffer.deinit();
        const shapes = [_]struct { cuda.DType, []const usize }{
            .{ .i32, &.{1 + rows} },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, 3 * c.hidden } },
            .{ .bf16, &.{ rows, c.heads, c.head_dim } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ rows, 2 * widest } },
            .{ .bf16, &.{ rows, widest } },
            .{ .bf16, &.{ rows, c.experts } },
            .{ .i32, &.{ rows, c.top } },
            .{ .f32, &.{ rows, c.top } },
            .{ .bf16, &.{ rows * c.top, c.expert_width } },
            .{ .bf16, &.{ rows * c.top, c.hidden } },
            .{ .bf16, &.{ rows, c.hidden } },
            .{ .bf16, &.{ scored, c.vocab } },
            .{ .f32, &.{ rows * c.heads * chunks, c.head_dim + 2 } },
            .{ .i32, &.{rows * c.heads} },
        };
        for (shapes, 0..) |shape, i| {
            list[i] = try .init(shape[0], shape[1]);
            made += 1;
        }
        try list[17].zero();
        var dash: Dash = try .init(rows, c.dash);
        errdefer dash.deinit();
        const plan: experts_file.Plan = try .init(rows, c.top, c.experts);
        return .{
            .dash = dash,
            .rows = rows,
            .scored = scored,
            .ids = list[0],
            .h = list[1],
            .x = list[2],
            .qkv = list[3],
            .q = list[4],
            .attn = list[5],
            .branch = list[6],
            .gu = list[7],
            .act = list[8],
            .router = list[9],
            .picks = list[10],
            .weights = list[11],
            .plan = plan,
            .expert_act = list[12],
            .expert_out = list[13],
            .final = list[14],
            .logits = list[15],
            .parts = list[16],
            .done = list[17],
        };
    }

    pub fn deinit(self: *Buffers) void {
        inline for (.{ "ids", "h", "x", "qkv", "q", "attn", "branch", "gu", "act", "router", "picks", "weights", "expert_act", "expert_out", "final", "logits", "parts", "done" }) |name| @field(self, name).deinit();
        self.dash.deinit();
        self.plan.deinit();
        self.* = undefined;
    }

    pub fn scoredRows(self: Buffers) usize {
        return self.scored;
    }
};

/// The forward's windows as captured graphs (WS-07): one launch a window in place of some 230 kernels a step. A
/// graph holds its launches' arguments, so there is one a window size and `Score`; the position is a device word
/// `stage` writes before each launch. A window that touches the image's rows copies them from features a graph
/// could not hold, and runs eager. The kernels and their arguments are `forward`'s, and so are the bits.
pub const Graphs = struct {
    held: Held,

    const Held = graphs_file.Graphs(Key, Graphed);
    const Key = struct { rows: usize, score: Score };

    /// `decode/graphs.zig`'s `capture` over `compute`.
    const Graphed = struct {
        pub const Model = forward_file.Model;
        pub const State = forward_file.State;
        pub const Buffers = forward_file.Buffers;
        pub const Error = forward_file.Error;

        pub fn compute(m: forward_file.Model, state: *forward_file.State, b: *forward_file.Buffers, rows: usize, key: Key) forward_file.Error!void {
            return forward_file.compute(m, state.*, b, rows, key.score);
        }
    };

    pub fn init(gpa: Allocator) cuda.Error!Graphs {
        return .{ .held = try .init(gpa) };
    }

    /// The graphs hold pointers into the model, state and buffers they were captured with: drop them first.
    pub fn deinit(self: *Graphs) void {
        self.held.deinit();
        self.* = undefined;
    }

    /// `forward`, always with the same `m`, `state` and `b`.
    pub fn forward(self: *Graphs, m: Model, state: *State, b: *Buffers, tokens: []const i32) (Error || Allocator.Error)!void {
        return self.forwardScoring(m, state, b, tokens, .all);
    }

    pub fn forwardScoring(self: *Graphs, m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) (Error || Allocator.Error)!void {
        const n = tokens.len;
        try fits(m, state, b, n, score);
        try stage(state.*, b, tokens);
        if (touchesImage(state.*, n)) return compute(m, state.*, b, n, score);
        const graph = try self.held.capture(m, state, b, n, .{ .rows = n, .score = score });
        return graph.launch(self.held.cache.stream().*);
    }

    /// Captures the graph of a window of `rows` rows scored as `score` from where `state` stands, unless it is
    /// held. A capture records its launches and runs none: the sequence goes on as if this had not been called.
    pub fn prepare(self: *Graphs, m: Model, state: *State, b: *Buffers, rows: usize, score: Score) (Error || Allocator.Error)!void {
        _ = try self.held.capture(m, state, b, rows, .{ .rows = rows, .score = score });
    }

    /// Whether the graph `forward` would launch for a window of `rows` rows from where `state` stands is
    /// captured.
    pub fn holds(self: *const Graphs, state: State, b: Buffers, rows: usize) bool {
        if (rows == 0 or rows > b.scored or touchesImage(state, rows)) return false;
        return self.held.cache.contains(.{ .rows = rows, .score = .all });
    }
};

const forward_file = @This();

pub const Score = enum { none, last, all };

/// `tokens` as one window after the state, every row scored.
pub fn forward(m: Model, state: *State, b: *Buffers, tokens: []const i32) Error!void {
    return forwardScoring(m, state, b, tokens, .all);
}

pub fn forwardScoring(m: Model, state: *State, b: *Buffers, tokens: []const i32, score: Score) Error!void {
    try fits(m, state, b, tokens.len, score);
    try stage(state.*, b, tokens);
    return compute(m, state.*, b, tokens.len, score);
}

/// The window's words: its first position, then its tokens (`Buffers.ids`), in the stream's order; its rows'
/// coordinates too.
pub fn stage(state: State, b: *Buffers, tokens: []const i32) Error!void {
    try stageCoords(state, b, tokens.len, tokens);
    return stageWords(state, b, tokens);
}

/// The window's coordinates into `b.dash.coords`: a prompt row's from `State.paged`, a row past the prompt from
/// the track after the rows kept, carried over the window's tokens before it (so a verify window's rows are its
/// serial steps'). The window's are written at its positions in the host's `coords`, past the prompt's alone.
fn stageCoords(state: State, b: *Buffers, n: usize, tokens: []const i32) Error!void {
    const d = &state.dash;
    var track = d.track orelse return error.Unpaged;
    const window = d.coords[state.pos..][0..n];
    for (window, state.pos..) |*c, p| {
        if (p < d.prompt) continue;
        // a window given as embeddings has no tokens to carry the track over
        if (tokens.len != n) return error.Unpaged;
        c.* = track.advance(tokens[p - state.pos]);
        c.page = shifted(c.page, d.base);
    }
    return b.dash.coords.uploadLaunched(0, std.mem.sliceAsBytes(window));
}

/// A RoPE page with the part's base (`State.Dash.base`): the instruction's bucket 0 keeps 0.
fn shifted(page: i32, base: i32) i32 {
    return if (page > 0) page + base else page;
}

/// The block summaries (k_bar) of the blocks whose last row is among `rows` rows from `at`, every layer, from the
/// index keys there: rows copied into the caches rather than computed (`parts.zig`), as SGLang's `_apply_segment`
/// refreshes `kbar` for the blocks it copied.
pub fn summarize(m: Model, state: State, b: *Buffers, at: usize, rows: usize) Error!void {
    if (rows == 0) return;
    var where = state;
    where.pos = at;
    try stageWords(where, b, &.{});
    const words = b.ids.span(0, 1);
    const d = &state.dash;
    for (m.layers, 0..) |l, i| try router_file.kbar(l.router, d.idx_k[i].view, words, rows, d.kbar[i].view);
}

fn stageWords(state: State, b: *Buffers, tokens: []const i32) Error!void {
    var word: [1]i32 = .{@intCast(state.pos)};
    try b.ids.uploadLaunched(0, std.mem.sliceAsBytes(&word));
    return b.ids.uploadAt(@sizeOf(i32), std.mem.sliceAsBytes(tokens));
}

/// The launches of a staged window: its embeddings, every layer, the final norm and the logits `score` names.
pub fn compute(m: Model, state: State, b: *Buffers, n: usize, score: Score) Error!void {
    try place(m, state, b, n);
    return chain(m, state, b, n, score);
}

/// The layers over `b.h` with each norm in the add before it: the first layer's input norm, then every layer's adds
/// carry the norm that reads them (the post-attention norm, the next layer's input norm, the final norm), the same
/// bits as `layer` after `layer` and `finish`.
fn chain(m: Model, state: State, b: *Buffers, n: usize, score: Score) Error!void {
    try glue.rmsnorm(b.h.rows(n), m.layers[0].input_norm.view, b.x.rows(n), m.config.eps);
    for (0..m.layers.len) |i| try layerChained(m, i, state, b, n);
    try head(m, b, n, score);
}

fn touchesImage(state: State, n: usize) bool {
    const image = state.image orelse return false;
    var from: usize = 0;
    for (0..image.count()) |i| {
        const r = image.run(i, &from);
        if (@max(r.first, state.pos) < @min(r.first + r.rows, state.pos + n)) return true;
    }
    return false;
}

/// The window's rows of `b.h`: the tokens' embeddings, the state's image rows laid over those it covers.
pub fn embed(m: Model, state: State, b: *Buffers, tokens: []const i32) Error!void {
    try stage(state, b, tokens);
    return place(m, state, b, tokens.len);
}

fn place(m: Model, state: State, b: *Buffers, n: usize) Error!void {
    try cuda.launched(cuda.kernels.zs_embed(&b.ids.span(1, n), &m.embed.view, &b.h.rows(n), 1));
    if (state.image) |image| {
        var from: usize = 0;
        for (0..image.count()) |i| {
            const r = image.run(i, &from);
            const first = @max(r.first, state.pos);
            const end = @min(r.first + r.rows, state.pos + n);
            if (first < end) try b.h.copyRows(first - state.pos, image.features, from + first - r.first, end - first);
        }
    }
}

/// A window of `embeds` bf16 [n, hidden] given whole (a fixture's, or every row an image's), after the state.
pub fn forwardEmbeds(m: Model, state: *State, b: *Buffers, embeds: cuda.Buffer, score: Score) Error!void {
    const n: usize = @intCast(embeds.view.size[0]);
    try fits(m, state, b, n, score);
    try stageCoords(state.*, b, n, &.{});
    try stageWords(state.*, b, &.{});
    try b.h.copyRows(0, embeds, 0, n);
    return run(m, state, b, n, score);
}

fn fits(m: Model, state: *State, b: *Buffers, n: usize, score: Score) Error!void {
    if (n == 0 or n > b.rows or n > m.max_rows or (score == .all and n > b.scored)) return error.WindowTooWide;
    if (state.pos + n > state.capacity) return error.ContextFull;
}

/// Every layer over the window's `n` rows in `b.h`, then the final norm and the logits `score` names.
pub fn run(m: Model, state: *State, b: *Buffers, n: usize, score: Score) Error!void {
    return chain(m, state.*, b, n, score);
}

/// The final norm into `b.final`, then the head: every row's logits, the last row's into row 0, or none.
pub fn finish(m: Model, b: *Buffers, n: usize, score: Score) Error!void {
    try glue.rmsnorm(b.h.rows(n), m.norm.view, b.final.rows(n), m.config.eps);
    return head(m, b, n, score);
}

fn head(m: Model, b: *Buffers, n: usize, score: Score) Error!void {
    switch (score) {
        .none => {},
        .last => try m.head.forward(b.final.span(n - 1, 1), b.logits.rows(1)),
        .all => try m.head.forward(b.final.rows(n), b.logits.rows(n)),
    }
}

/// Layer `i` over rows 0 .. n - 1 of `b.h`, at positions `state.pos` on (the words `stage` wrote), whole: the
/// input norm first, the last add into `b.h` alone.
pub fn layer(m: Model, i: usize, state: State, b: *Buffers, n: usize) Error!void {
    const c = m.config;
    const l = m.layers[i];
    const h = b.h.rows(n);
    const x = b.x.rows(n);
    try glue.rmsnorm(h, l.input_norm.view, x, c.eps);
    try attendLayer(m, i, state, b, n, x);
    try l.o.forward(b.attn.rows(n), b.branch.rows(n));
    try glue.residualNormBf16(h, b.branch.rows(n), l.post_norm.view, x, c.eps);
    try mlpRows(m, l, h, x, b, n, null, null);
}

/// The layer `i`'s share of `chain`: the input norm is in `b.x` already, and the layer's last add writes the next
/// norm's rows, the next layer's into `b.x` or the final norm's into `b.final`.
fn layerChained(m: Model, i: usize, state: State, b: *Buffers, n: usize) Error!void {
    const c = m.config;
    const l = m.layers[i];
    const h = b.h.rows(n);
    const x = b.x.rows(n);
    const last = i + 1 == m.layers.len;
    const next_w = if (last) m.norm.view else m.layers[i + 1].input_norm.view;
    const next_out = if (last) b.final.rows(n) else x;
    try attendLayer(m, i, state, b, n, x);
    try l.o.forward(b.attn.rows(n), b.branch.rows(n));
    try glue.residualNormBf16(h, b.branch.rows(n), l.post_norm.view, x, c.eps);
    try mlpRows(m, l, h, x, b, n, &next_w, &next_out);
}

/// The qkv product and DASH's step 1 to `b.attn` from `b.qkv` and the input norm's rows `x`: page2d RoPE into q
/// and the caches, the router's picks, then the attend.
fn attendLayer(m: Model, i: usize, state: State, b: *Buffers, n: usize, x: cuda.View) Error!void {
    const c = m.config;
    const l = m.layers[i];
    try l.qkv.forward(x, b.qkv.rows(n));
    const kc = state.keys[i].view;
    const vc = state.values[i].view;
    const at = b.ids.span(0, 1);
    const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(c.head_dim)));
    const dash = c.dash;
    const d = &state.dash;
    const w = &b.dash;
    const rope = m.page2d;
    const coords = w.coords.rows(n);
    try cuda.launched(zs_attn_prep_page2d(&b.qkv.rows(n), &rope.inv.view, &at, &coords, @intCast(rope.n_tok), @intCast(rope.page_mod), &b.q.rows(n), &kc, &vc));
    try router_file.project(l.router, x, w.qk.rows(n));
    try router_file.prep(l.router, rope, c.eps, w.qk.rows(n), at, coords, w.idx_q.rows(n), d.idx_k[i].view);
    try router_file.kbar(l.router, d.idx_k[i].view, at, n, d.kbar[i].view);
    try router_file.route(.of(dash, i), w.idx_q.rows(n), d.kbar[i].view, at, coords, d.pages.view, w.ids.rows(n), w.bias.rows(n), null);
    const form: dash_attention.Form = if (n <= dash_step_rows) .step else .tile;
    const scratch: dash_attention.Scratch = .{ .parts = b.parts.view, .done = b.done.view };
    try dash_attention.attend(form, b.q.rows(n), kc, vc, at, w.ids.rows(n), w.bias.rows(n), dash.local_blocks * dash.block, scale, scratch, b.attn.rows(n));
}

/// The layer's MLP: the dense MLP, or the gate's picks, the routed experts' sum over the shared MLP's, and the
/// residual add that writes `out` (or `h` where `out` is null, the unchained layer's).
fn mlpRows(m: Model, l: model_file.Layer, h: cuda.View, x: cuda.View, b: *Buffers, n: usize, next_w: ?*const cuda.View, next_out: ?*const cuda.View) Error!void {
    const c = m.config;
    switch (l.mlp) {
        .dense => |dense_mlp| {
            try mlp(b, dense_mlp, x, n, b.branch.rows(n));
            if (next_w) |w| try glue.residualNormBf16(h, b.branch.rows(n), w.*, next_out.?.*, c.eps) else try glue.residualAdd(h, b.branch.rows(n));
        },
        .experts => |e| {
            const picks = b.picks.rows(n);
            const weights = b.weights.rows(n);
            try e.router.forward(x, b.router.rows(n));
            const plan = b.plan;
            if (n == 1) {
                // one row's plan is its picks in order: the route writes it
                try cuda.launched(zs_unlimited_ocr_moe_route(&b.router.rows(n), &picks, &weights, &plan.members.view, &plan.items.view, &plan.counts.view));
            } else {
                try cuda.launched(zs_unlimited_ocr_moe_route(&b.router.rows(n), &picks, &weights, null, null, null));
                try plan.routeAs(picks, n, .{ .item = bf16.Experts.item, .tiles = 4 });
            }
            const pairs = n * c.top;
            try bf16.Experts.up(x, e.gu.view, plan.members.view, plan.items.view, plan.counts.view, c.top, b.expert_act.rows(pairs));
            try bf16.Experts.down(b.expert_act.rows(pairs), e.down.view, plan.members.view, plan.items.view, plan.counts.view, b.expert_out.rows(pairs));
            try mlp(b, e.shared, x, n, b.branch.rows(n));
            try cuda.launched(zs_unlimited_ocr_moe_combine(&b.expert_out.rows(pairs), &weights, &b.branch.rows(n), &h, next_w, next_out, c.eps));
        },
    }
}

/// out = down(bf16(bf16(silu(gate x)) * up x)), each product bf16.
fn mlp(b: *Buffers, d: model_file.Dense, x: cuda.View, n: usize, out: cuda.View) Error!void {
    const gu = flat(b.gu, n, 2 * d.width);
    const act = flat(b.act, n, d.width);
    try d.gu.forward(x, gu);
    try glue.swiglu(gu, act, std.math.floatMax(f32));
    try d.down.forward(act, out);
}

/// `buffer`'s first `n * cols` elements as a contiguous [n, cols].
fn flat(buffer: cuda.Buffer, n: usize, cols: usize) cuda.View {
    std.debug.assert(n * cols <= buffer.view.count());
    return .init(buffer.view.ptr.?, buffer.view.dtype, &.{ n, cols });
}

/// Makes the window's first `keep` rows part of the sequence: their keys and values are in the caches already
/// (their index keys too, and the track goes over those past the prompt).
pub fn commit(_: Model, state: *State, _: Buffers, tokens: []const i32, keep: usize) Error!void {
    std.debug.assert(keep >= 1 and keep <= tokens.len);
    if (state.dash.track) |*track| for (tokens[0..keep], state.pos..) |token, p| {
        if (p >= state.dash.prompt) _ = track.advance(token);
    };
    state.pos += keep;
}

/// Row `row`'s logits, bf16 bits a token, after the device's launches.
pub fn logitsRow(b: Buffers, row: usize, out: []u16) Error!void {
    return cuda.download(b.logits.span(row, 1), std.mem.sliceAsBytes(out));
}

/// The window's first `rows` final rows.
pub fn streams(b: Buffers, rows: usize) cuda.View {
    return b.final.rows(rows);
}

pub extern fn zs_attn_prep_page2d(qkv: *const cuda.View, inv: *const cuda.View, pos: *const cuda.View, coords: *const cuda.View, n_tok: i64, page_mod: i64, q: *const cuda.View, kc: *const cuda.View, vc: *const cuda.View) c_int;
pub extern fn zs_unlimited_ocr_moe_route(logits: *const cuda.View, picks: *const cuda.View, weights: *const cuda.View, members: ?*const cuda.View, items: ?*const cuda.View, counts: ?*const cuda.View) c_int;
pub extern fn zs_unlimited_ocr_moe_combine(y: *const cuda.View, weights: *const cuda.View, shared: *const cuda.View, h: *const cuda.View, w: ?*const cuda.View, out: ?*const cuda.View, eps: f64) c_int;
