//! Flash Next's multi-token-prediction head on the device. It reads a row's final streams from the main model and
//! the token chosen after it, through one attention layer with a cache of its own, and scores the token after
//! that: a draft. Drafts choose nothing (`draft.zig` keeps only what the main model chooses), so this file's
//! arithmetic is compared with TensorFold's by tolerance.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("blocks/attention.zig");
const confidence = @import("../../decode/confidence.zig");
const cuda = @import("../../cuda/cuda.zig");
const cache_file = @import("../../decode/graphs.zig");
const draft = @import("draft.zig");
const forward = @import("forward.zig");
const hc = @import("blocks/hc.zig");
const masks_file = @import("../../decode/masks.zig");
const model = @import("model.zig");
const rule_file = @import("mtp_rule.zig");
const sampling = @import("../../decode/sampling.zig");
const picking = @import("../../decode/pick.zig");

/// out bf16 [rows, wide] = x over its row's RMS, times scale fp32 [wide]; `wide` in blocks of 256.
pub fn rmsnorm(x: cuda.View, scale: cuda.View, out: cuda.View, eps: f32) cuda.Error!void {
    return cuda.launched(zs_qwen4_mtp_rmsnorm(&x, &scale, &out, eps));
}

/// out bf16 [rows, streams * dims]: e bf16 [rows, dims] added to each stream of hs bf16 [rows * streams, dims].
pub fn addStreams(e: cuda.View, hs: cuda.View, out: cuda.View, streams: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_mtp_add_streams(&e, &hs, &out, @intCast(streams)));
}

/// A scored row's start on the device, and its words on the host (`zig/src/decode/pick.zig`).
pub const pick = picking.pick;
pub const Picked = picking.Picked;

extern fn zs_qwen4_mtp_rmsnorm(x: *const cuda.View, scale: *const cuda.View, out: *const cuda.View, eps: f64) c_int;
extern fn zs_qwen4_mtp_add_streams(e: *const cuda.View, hs: *const cuda.View, out: *const cuda.View, streams: i64) c_int;

/// The model the head's own layer runs on: `m`, or beside a share (`model.Mtp.whole`, rank 0 of two) `m` with
/// the whole model's sizes and the head's own attention in place of the rank's halves. What the head reads of
/// it besides is whole on a share as it stands: the embedding, the vocabulary, the scratch pool. A rank's
/// half of a split head (`model.Mtp.split`) is of the share's own sizes: `m`.
pub fn over(m: model.Model) model.Model {
    const whole = m.mtp.?.whole orelse return m;
    var out = m;
    out.shape = whole.shape;
    out.attention = whole.attention;
    return out;
}

/// One sequence's entries in the head's attention cache: entry i is position i's final streams with the token
/// at i + 1. TensorFold's `State.mtp_*`.
pub const State = struct {
    cache: attention.Cache,
    /// Entries held: one for each position absorbed, then `drafted` of chained drafts.
    position: attention.Position,
    drafted: usize = 0,
    /// The sequence's rotary positions (spec 019): entry i rotates as the main model's position i, as
    /// TensorFold's head reads the same `image_positions` and `rope_delta` by its own `mtp_pos`. A copy of the
    /// main state's (`Head.attach`), zeros for text.
    rope: attention.Rope,

    pub fn init(m: model.Model, capacity: usize) cuda.Error!State {
        var cache: attention.Cache = try .init(over(m).attention.?.headShape(), capacity);
        errdefer cache.deinit();
        var rope: attention.Rope = try .init(capacity);
        errdefer rope.deinit();
        return .{ .cache = cache, .position = try .init(), .rope = rope };
    }

    pub fn deinit(self: *State) void {
        self.cache.deinit();
        self.position.deinit();
        self.rope.deinit();
        self.* = undefined;
    }

    /// Drops the chained drafts' entries: the next rows absorbed are written over them before any row reads
    /// them, as a window's dropped rows are in the main model's caches.
    pub fn trim(self: *State) cuda.Error!void {
        if (self.drafted == 0) return;
        try self.position.set(self.position.host - self.drafted);
        self.drafted = 0;
    }
};

/// Scratch for head windows of up to `rows` rows.
pub const Buffers = struct {
    /// The layer's and the mixer's scratch. `layer.h` holds the head's streams, which a chained draft reads
    /// back; `layer.logits`' first row the last row's logits.
    layer: forward.Buffers,
    /// bf16 [rows, hidden]: the next tokens' embeddings
    e: cuda.Buffer,
    /// bf16 [rows, hidden]
    e_normed: cuda.Buffer,
    /// bf16 [rows, hidden]: `fc_embedding`'s output
    eo: cuda.Buffer,
    /// bf16 [rows, streams * hidden]: the input streams over their row's RMS
    hn: cuda.Buffer,
    /// bf16 [rows * streams, hidden]: `fc_hidden`'s output, a stream a row
    hs: cuda.Buffer,
    /// bf16 [1, draft vocabulary]: the last row's logits from the draft head, where the model has one. Split
    /// over two ranks, [1, ranks * the widest rank's ids]: each rank's columns from `Split.widest` times its
    /// rank, so the ids' columns are the first of it, in order (`mtp_ranks.zig` gathers them).
    draft_logits: ?cuda.Buffer,
    /// u32 [`Picked.wordsFor` the scored logits]: what a draft's choice reads of the scored row
    picked: cuda.Buffer,

    pub fn init(gpa: Allocator, m: model.Model, rows: usize, capacity: usize) !Buffers {
        return initBeside(gpa, m, rows, capacity, null);
    }

    /// `init`, the layer's attention computing in `beside`'s scratch, the main model's buffers, where those
    /// serve the same rows and capacity (`forward.Buffers.initBeside`); they must outlive these. Beside a
    /// share nothing is lent: the trunk's scratch is sized for a rank's half of the heads.
    pub fn initBeside(gpa: Allocator, m: model.Model, rows: usize, capacity: usize, beside: ?*const forward.Buffers) !Buffers {
        const head = m.mtp.?;
        const own = over(m);
        const s = own.shape;
        var b: Buffers = .{
            // the head scores a window's last row alone
            .layer = try .initBeside(gpa, own, &.{head.layer}, rows, 1, capacity, if (head.whole == null) beside else null),
            .e = try .init(.bf16, &.{ rows, s.hidden }),
            .e_normed = try .init(.bf16, &.{ rows, s.hidden }),
            .eo = try .init(.bf16, &.{ rows, s.hidden }),
            .hn = try .init(.bf16, &.{ rows, s.wide() }),
            .hs = try .init(.bf16, &.{ rows * s.streams, s.hidden }),
            .draft_logits = if (head.draft != null) try .init(.bf16, &.{ 1, draftColumns(head) }) else null,
            .picked = try .init(.u32, &.{Picked.wordsFor(if (head.scored()) |ids| ids.len else s.vocab)}),
        };
        errdefer b.deinit();
        // `choiceOnDevice` downloads every word and `pick_emit` writes only those `Picked.read` reads: zeroed
        // once, the rest are defined (initcheck, spec 103 WS-01)
        try b.picked.zero();
        return b;
    }

    /// The columns of `draft_logits`.
    fn draftColumns(head: model.Mtp) usize {
        return if (head.split) |split| split.widest() * @import("share.zig").world else head.draft.?.ids.len;
    }

    /// The device bytes `initBeside` asks, with nothing made; `lent` as `forward.Buffers.bytesFor`'s.
    pub fn bytesFor(m: model.Model, rows: usize, capacity: usize, lent: bool) usize {
        const of = cuda.Buffer.bytesOf;
        const head = m.mtp.?;
        const own = over(m);
        const s = own.shape;
        var bytes = forward.Buffers.bytesFor(own, &.{head.layer}, rows, 1, capacity, lent and head.whole == null);
        bytes += 3 * of(.bf16, &.{ rows, s.hidden }) + of(.bf16, &.{ rows, s.wide() }) + of(.bf16, &.{ rows * s.streams, s.hidden });
        if (head.draft != null) bytes += of(.bf16, &.{ 1, draftColumns(head) });
        return bytes + of(.u32, &.{Picked.wordsFor(if (head.scored()) |ids| ids.len else s.vocab)});
    }

    pub fn deinit(self: *Buffers) void {
        self.layer.deinit();
        inline for (.{ "e", "e_normed", "eo", "hn", "hs", "picked" }) |name| @field(self, name).deinit();
        if (self.draft_logits) |*logits| logits.deinit();
        self.* = undefined;
    }
};

/// Runs rows whose main-model final streams are `streams` bf16 [rows, streams * hidden] and whose following
/// tokens are `tokens`, as entries `state.position` onward. Leaves the head's own streams in `b.layer.h` and,
/// with `score`, the last row's logits: over the draft vocabulary in `b.draft_logits` where the model has a
/// draft head, else over the vocabulary in `b.layer.logits`' first row; and in `b.picked` what `pick` finds in
/// them. The caller advances
/// `state.position` by what it keeps. `streams` may be rows of `b.layer.h`: they are read before it is written.
pub fn run(m: model.Model, state: *State, b: *Buffers, streams: cuda.View, tokens: []const i32, score: bool) forward.Error!void {
    return runTapped(m, state, b, streams, tokens, score, null);
}

/// `run` with `tap` called at the layer's points (layer 0), the final streams and the mixer's read-out.
pub fn runTapped(m: model.Model, state: *State, b: *Buffers, streams: cuda.View, tokens: []const i32, score: bool, tap: ?forward.Tap) forward.Error!void {
    try stage(m, b, tokens);
    return compute(m, state, b, streams, tokens.len, null, score, tap);
}

/// The host's part: the next tokens' ids.
pub fn stage(m: model.Model, b: *Buffers, tokens: []const i32) cuda.Error!void {
    std.debug.assert(tokens.len >= 1 and tokens.len <= b.layer.rows and tokens.len <= m.mtp.?.max_rows);
    return b.layer.ids.uploadAt(0, std.mem.sliceAsBytes(tokens));
}

/// The device's part, launches only, as `forward.compute` is: TensorFold's `mtp_compute`.
fn compute(m: model.Model, state: *State, b: *Buffers, streams: cuda.View, rows: usize, bound: ?usize, score: bool, tap: ?forward.Tap) forward.Error!void {
    const head = m.mtp.?;
    const own = over(m);
    std.debug.assert(head.split == null);
    try front(m, b, streams, rows);
    const site: forward.Site = .{ .attention = .{ .cache = state.cache, .position = state.position, .bound = bound, .rope = state.rope.view() } };
    const pending = try forward.layerForward(own, head.layer, 0, site, &b.layer, rows, .none, true, tap);
    try forward.finish(own, head.mixer, 1, &b.layer, rows, pending, tap);
    if (!score) return;
    const last = b.layer.mixed.span(rows - 1, 1);
    if (head.draft) |d| try d.linear.forward(last, b.draft_logits.?.view) else try m.head.forward(last, b.layer.logits.rows(1));
    // and what a draft's choice needs of them, so the host reads a kilobyte a draft and not the row
    try pick(scoredView(head, b.*), b.picked.view);
}

/// What the head makes of its inputs before its layer, into `b.layer.h`: whole on every rank of a split head.
pub fn front(m: model.Model, b: *Buffers, streams: cuda.View, rows: usize) cuda.Error!void {
    const head = m.mtp.?;
    const s = over(m).shape;
    std.debug.assert(streams.size[0] == rows and streams.size[1] == s.wide());
    const e = b.e.rows(rows);
    const e_normed = b.e_normed.rows(rows);
    const eo = b.eo.rows(rows);
    try hc.embed(b.layer.ids.rows(rows), m.embed.view, e, 1);
    try rmsnorm(e, head.norm_embedding.view, e_normed, s.eps);
    try head.fc_embedding.forward(e_normed, eo);
    // one norm over a row's streams together, then `fc_hidden` on each stream as a row of its own
    try rmsnorm(streams, head.norm_hidden.view, b.hn.rows(rows), s.eps);
    var each = b.hn.rows(rows);
    each.size[0] = @intCast(rows * s.streams);
    each.size[1] = @intCast(s.hidden);
    const hs = b.hs.rows(rows * s.streams);
    try head.fc_hidden.forward(each, hs);
    try addStreams(eo, hs, b.layer.h.rows(rows), s.streams);
}

/// The logits `compute` scores: the draft ids' columns, or the first row of the vocabulary's.
pub fn scoredView(head: model.Mtp, b: Buffers) cuda.View {
    const logits = b.draft_logits orelse return b.layer.logits.rows(1);
    return .init(logits.view.ptr.?, .bf16, &.{ 1, head.scored().?.len });
}

/// `run` as captured graphs, as `forward.Graphs`: one for each window size, bucket of context, place the input
/// streams are read from (the main model's buffer, or a row of the head's own for a chained draft) and whether
/// the last row is scored.
pub const Graphs = struct {
    cache: cache_file.Cache(Key),

    const Key = struct { rows: usize, context: usize, streams: usize, score: bool };

    pub fn init(gpa: Allocator) cuda.Error!Graphs {
        return .{ .cache = try .init(gpa) };
    }

    /// The graphs hold pointers into the model, state and buffers they were captured with: drop them first.
    pub fn deinit(self: *Graphs) void {
        self.cache.deinit();
        self.* = undefined;
    }

    /// `run`, always with the same `m`, `state` and `b`.
    pub fn run(self: *Graphs, m: model.Model, state: *State, b: *Buffers, streams: cuda.View, tokens: []const i32, score: bool) (forward.Error || Allocator.Error)!void {
        const rows = tokens.len;
        try stage(m, b, tokens);
        const context = forward.Graphs.bucket(b.layer, state.position.host + rows);
        const key: Key = .{ .rows = rows, .context = context, .streams = @intFromPtr(streams.ptr.?), .score = score };
        return self.cache.run(key, compute, .{ m, state, b, streams, rows, context, score, null });
    }

    /// Captures the graph `run` would for `rows` rows read from `streams` over the bucket `context`, unless it
    /// is held. As `forward.Graphs.prepare`: nothing is launched, so nothing is written.
    pub fn prepare(self: *Graphs, m: model.Model, state: *State, b: *Buffers, streams: cuda.View, rows: usize, context: usize, score: bool) (forward.Error || Allocator.Error)!void {
        const key: Key = .{ .rows = rows, .context = context, .streams = @intFromPtr(streams.ptr.?), .score = score };
        _ = try self.cache.capture(key, compute, .{ m, state, b, streams, rows, context, score, null });
    }
};

/// A row's probabilities on the host, which GLM's head takes too: `zig/src/decode/confidence.zig`.
pub const probability = confidence.probability;
pub const normalizer = confidence.normalizer;
pub const Normalizer = confidence.Normalizer;

/// How a draft is chosen from a row the head scored, and while it pays: `mtp_rule.zig`.
pub const Paying = rule_file.Paying;
pub const Rule = rule_file.Rule;

/// The MTP head as a `draft.Drafter`, TensorFold's `absorb` and `draft`: it absorbs the rows the main model
/// kept, then chains drafts, each from the last one's streams, each chosen as the main model will choose at
/// that position (the same sampling and key) from the head's logits.
pub const minus_infinity = confidence.minus_infinity;

pub const Head = struct {
    m: model.Model,
    state: State,
    b: Buffers,
    /// Null launches each kernel from the host.
    graphs: ?Graphs,
    sampler: *sampling.Sampler,
    settings: sampling.Sampling,
    /// A chain stops before a draft the head gives less than this, but for its first, which is kept and ends
    /// it: TensorFold's rule for this family, with its `CONFIDENCE` of 0.7. 0, the default, leaves the chain
    /// to `chain`.
    confidence: f32 = 0,
    /// A chain stops before a draft that takes the product of its own drafts' probabilities below this, its
    /// first again kept and ending it: what the head gives the whole chain, where `confidence` is what it
    /// gives one draft. A row of the window is only worth its time if every draft before it is right too,
    /// which is what the product says and one draft's probability does not, so this rule can chain to the
    /// window's limit where TensorFold's needs its `DEPTH` of 6. 0 leaves the chain to `confidence`; with
    /// both 0 a chain goes as far as it is asked. 0.65 with 15 drafts a window was chosen by replies timed
    /// on each checkpoint (spec 028): against 0.7 a draft and 6, code decodes 6% faster on the EXL3 pack and
    /// 14% on NVFP4, and prose verifies no more rows than it did. The head is surer than it is right (a
    /// chain it gives 0.65 is kept 1 time in 3 on prose and 1 in 2 on code), and a row pays for itself from
    /// about 1 in 4 on prose and 2 in 5 on code, so a lower floor verifies rows for nothing.
    chain: f32 = default_chain,
    /// A chain ends with a draft that is one of these: the reply stops there (`draft.Decoder.eos`), and rows
    /// after it would be verified for nothing.
    ends: []const u32 = &.{},
    /// One row's logits on the host, over the vocabulary. With a draft head the tokens it does not score stay
    /// at -inf, so the sampler reads the row as it reads the main model's and keys a token by its own id.
    logits: []u16,
    /// [draft vocabulary] the draft head's logits as downloaded.
    columns: []u16,
    whole: Normalizer,
    /// A draft the constraint leaves free is chosen from what the device found in its row (`Picked`) where
    /// that settles the choice, and from the row itself, read back, where it does not. False always reads the
    /// row, as every draft did before spec 028: the tests hold the two against each other.
    on_device: bool = true,
    /// `confidence_margin`, a field so that a test can widen it until every probability is at the threshold.
    margin: f64 = confidence_margin,
    /// The rank `b.picked` is asked to, which the settings need: 0 before the first, and where the device
    /// cannot serve them (`sampling.leadingNeeded`: no cut at all, a `top_k` past `Picked.most` or the columns).
    asked: u32 = 0,
    asked_for: ?sampling.Sampling = null,
    /// Drafts chosen each way.
    chosen: struct { device: usize = 0, host: usize = 0 } = .{},
    /// The last call's own choices in order, each with the probability the head gave it (0 where neither
    /// `confidence` nor `chain` is set and none is computed): the drafts it handed over and, last, the one that ended the chain if it was
    /// not handed over. What `zspark generate ... rounds` prints, to set a head's chains beside TensorFold's.
    trail: [draft.forward_max_rows + 1]Considered = undefined,
    trailed: usize = 0,
    /// The cache is full: the head has stopped absorbing, and so drafting.
    full: bool = false,
    /// The reply's `Paying` where the head drafts by it (`fresh`), and the last round's chain, its drafts and
    /// those the head scored, until the next call says how many tokens the round gave.
    paying: ?Paying = null,
    last_chain: ?Paying.Told = null,
    /// Told the chain's drafts so far each time the head goes on from them to the next: whoever stages the
    /// round's window may start on their rows while the pass runs (`forward.ask`).
    picked: ?struct { context: *anyopaque, tell: *const fn (context: *anyopaque, drafts: []const i32) void } = null,
    /// The checkpoint's (`model.Mtp.costs`); null drafts by `chain`, as `generate chains` and the tests ask.
    costs: ?Paying.Costs,
    /// The head split over two ranks (`mtp_ranks.zig`): a pass runs there, on every rank, in place of `run`.
    /// `own` is the row of the head's own streams it reads, null the main model's first rows.
    split: ?struct { context: *anyopaque, pass: *const fn (context: *anyopaque, own: ?usize, tokens: []const i32, score: bool, wide: bool) anyerror!void } = null,
    gpa: Allocator,

    pub const Options = struct {
        /// The most rows absorbed at once: the main model's window.
        rows: usize,
        /// Positions the cache holds; drafting stops there.
        capacity: usize,
        graphs: bool = true,
        /// The main model's buffers, whose attention scratch the head's layer then computes in
        /// (`Buffers.initBeside`); they outlive the head.
        beside: ?*const forward.Buffers = null,
    };

    /// `m` has a head (`m.mtp`). The sampler is the main model's: drafts are keyed as its choices are.
    pub fn init(gpa: Allocator, m: model.Model, sampler: *sampling.Sampler, settings: sampling.Sampling, options: Options) !Head {
        var state: State = try .init(m, options.capacity);
        errdefer state.deinit();
        var b: Buffers = try .initBeside(gpa, m, options.rows, options.capacity, options.beside);
        errdefer b.deinit();
        const logits = try gpa.alloc(u16, m.shape.vocab);
        errdefer gpa.free(logits);
        @memset(logits, minus_infinity);
        const columns = try gpa.alloc(u16, if (m.mtp.?.scored()) |ids| ids.len else 0);
        errdefer gpa.free(columns);
        var whole: Normalizer = try .init(gpa);
        errdefer whole.deinit(gpa);
        return .{ .m = m, .state = state, .b = b, .graphs = if (options.graphs) try .init(gpa) else null, .sampler = sampler, .settings = settings, .logits = logits, .columns = columns, .whole = whole, .costs = m.mtp.?.costs, .gpa = gpa };
    }

    pub fn deinit(self: *Head) void {
        if (self.graphs) |*graphs| graphs.deinit();
        self.gpa.free(self.logits);
        self.gpa.free(self.columns);
        self.whole.deinit(self.gpa);
        self.b.deinit();
        self.state.deinit();
        self.* = undefined;
    }

    /// Before a new sequence's first row, with the sampling its drafts are chosen by.
    pub fn reset(self: *Head, settings: sampling.Sampling) cuda.Error!void {
        try self.state.position.set(0);
        try self.state.rope.clear();
        self.state.drafted = 0;
        self.full = false;
        self.settings = settings;
    }

    /// After `reset`, an image prompt's rotary positions: the main state's (`forward.State.attach`), copied.
    pub fn attach(self: *Head, rope: attention.Rope) cuda.Error!void {
        std.debug.assert(self.state.position.host == 0);
        return self.state.rope.copyFrom(rope);
    }

    /// Back to a sequence's position `position`, which the head absorbed: its entries below it stay, so the
    /// token after the last of them must be the sequence's still.
    pub fn rewind(self: *Head, position: usize, settings: sampling.Sampling) cuda.Error!void {
        try self.state.trim();
        std.debug.assert(position <= self.state.position.host and !self.full);
        try self.state.position.set(position);
        self.settings = settings;
    }

    /// A reply starts: it drafts by `costs` (`Paying`), or by `chain` where null.
    pub fn fresh(self: *Head) void {
        self.paying = if (self.costs) |c| .begin(c) else null;
        self.last_chain = null;
    }

    pub fn drafter(self: *Head) draft.Drafter {
        return .{ .context = self, .draft = drafts };
    }

    /// Captures every graph `drafts` can ask for over the bucket `context`, launching nothing. `main` is the
    /// main model's streams, its buffers' `h`. The keys, for each r of 1 to `draft.forward_max_rows`: r rows
    /// of `main` absorbed, the last scored or not (a round's kept rows, a prompt's last windows, a reply's
    /// end), and a chained draft's one row read from row r - 1 of the head's own streams. They do not follow
    /// a reply's depth: a prompt's last window is as wide as its rows, and one wider than these launches its
    /// kernels, as does the draft chained from its last row (`step`).
    pub fn prepare(self: *Head, main: cuda.Buffer, context: usize) (forward.Error || Allocator.Error)!void {
        const graphs = if (self.graphs) |*g| g else return;
        const widest = @min(draft.forward_max_rows, self.b.layer.rows, self.m.mtp.?.max_rows, @as(usize, @intCast(main.view.size[0])));
        for (1..widest + 1) |rows| {
            for ([_]bool{ true, false }) |score| try graphs.prepare(self.m, &self.state, &self.b, main.rows(rows), rows, context, score);
            try graphs.prepare(self.m, &self.state, &self.b, self.b.layer.h.span(rows - 1, 1), 1, context, true);
        }
    }

    /// `wide`: the rows are, or the row follows, a window wider than a reply's. `own`: `streams` is that row
    /// of the head's own.
    fn step(self: *Head, streams: cuda.View, own: ?usize, tokens: []const i32, score: bool, wide: bool) !void {
        // a prompt's windows are as wide as they come: a graph for each width, and one for the draft chained
        // from each such window's last row, would be launched once
        const graphs: ?*Graphs = if (!wide) if (self.graphs) |*g| g else null else null;
        if (score and self.on_device) try self.ask();
        if (self.split) |split| {
            try split.pass(split.context, own, tokens, score, wide);
        } else if (graphs) |g| try g.run(self.m, &self.state, &self.b, streams, tokens, score) else try run(self.m, &self.state, &self.b, streams, tokens, score);
        try self.state.position.advance(tokens.len);
    }

    /// The scored row's logits into `self.logits`. Returns the logits the head scored, in the order of their
    /// ids: the whole row, or a draft head's columns.
    fn fetchLogits(self: *Head) cuda.Error![]const u16 {
        const ids = self.m.mtp.?.scored() orelse {
            try forward.logitsRow(self.b.layer, 0, self.logits);
            return self.logits;
        };
        try cuda.synchronize();
        try self.b.draft_logits.?.downloadAt(0, std.mem.sliceAsBytes(self.columns));
        for (ids, self.columns) |id, bits| self.logits[id] = bits;
        return self.columns;
    }

    /// The head's fields that are the rule's (`Rule`), as one value.
    pub fn rule(self: *const Head) Rule {
        return .{ .confidence = self.confidence, .chain = self.chain, .paying = self.paying, .on_device = self.on_device, .margin = self.margin };
    }

    pub const Considered = draft.Chain.Considered;

    pub const default_chain = 0.65;

    /// How near `confidence` a draft's probability from the device's sum may come and still be compared with
    /// it: the sum is the host's within 1e-5 relatively (fp32 terms and adds for the host's fp64), so a
    /// probability this far from the threshold is on the host's side of it. A chain's product gathers each of
    /// its drafts' error, so it is held as many margins off `chain` as it has drafts, and relatively: a
    /// product may be small where a probability worth comparing is not.
    pub const confidence_margin = 1e-4;

    /// Before a row is scored: `b.picked` asked to the rank the settings need.
    fn ask(self: *Head) cuda.Error!void {
        if (self.asked_for) |settings| if (std.meta.eql(settings, self.settings)) return;
        const columns = if (self.m.mtp.?.scored()) |ids| ids.len else self.m.shape.vocab;
        const rank = Rule.rank(self.settings, columns, self.m.shape.vocab);
        // a launch that reads the rank may still be running: the settings change between sequences
        try cuda.synchronize();
        try Picked.ask(self.b.picked, @max(rank, 1));
        self.asked = rank;
        self.asked_for = self.settings;
    }

    /// The draft at `position` from `b.picked` (`Rule.onDevice`); null too where the device was not asked.
    fn choiceOnDevice(self: *Head, position: u64, before: f64, trail: []const draft.Chain.Considered) cuda.Error!?Rule.Choice {
        if (self.asked == 0) return null;
        try cuda.synchronize();
        var raw: [Picked.words]u32 = undefined;
        try self.b.picked.downloadAt(0, std.mem.sliceAsBytes(&raw));
        return self.rule().onDevice(&raw, self.m.mtp.?.scored(), self.m.shape.vocab, self.sampler, self.settings, position, before, trail);
    }

    /// `draft.Drafter.draft`. Under a constraint (`guide`) each draft is first asked of it. A token it forces is
    /// the draft, and the head is neither scored nor run for it until a later draft needs the head to stand
    /// after it: a run of forced tokens that fills `out` costs the absorbing step alone. Where it allows several
    /// tokens the head chooses among those, its other logits at -inf, as the main model will under the same
    /// mask, and a draft of its own that ends the reply is taken back, so none of its drafts is cut before the
    /// window. Its confidence in a draft stays the probability it gave the token before the mask. TensorFold's drafter is not told of the grammar; its drafts are cut after the fact.
    fn drafts(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
        const self: *Head = @ptrCast(@alignCast(context));
        if (self.full) return 0;
        try self.state.trim();
        // every position is absorbed in order, so the head's entries are the sequence's positions
        std.debug.assert(self.state.position.host + tokens.len == position);
        if (position > self.state.cache.capacity) {
            self.full = true;
            return 0;
        }
        self.trailed = 0;
        if (self.last_chain) |last| if (self.paying) |*paying| paying.done(tokens.len, last);
        self.last_chain = null;
        var chain: draft.Chain = try .begin(out, guide);
        try self.step(streams, null, tokens, chain.scores(), tokens.len > draft.forward_max_rows);
        // the row of the head's streams that the next draft it runs reads
        var last = tokens.len - 1;
        while (true) switch (try chain.step(guide)) {
            .halt => break,
            .run => {
                if (self.state.position.host == self.state.cache.capacity) break;
                if (self.picked) |picked| picked.tell(picked.context, out[0..chain.count]);
                try self.step(self.b.layer.h.span(last, 1), last, out[chain.ran..][0..1], true, last >= draft.forward_max_rows);
                self.state.drafted += 1;
                if (self.paying) |*paying| paying.counts.levels += 1;
                last = 0;
                chain.ran += 1;
            },
            .choose => {
                // a mask is laid over the row itself; a free draft needs the row's start alone
                var choice: ?Rule.Choice = if (chain.next == .free and self.on_device) try self.choiceOnDevice(position + 1 + chain.count, chain.product, chain.trail[0..chain.own]) else null;
                self.chosen.device += @intFromBool(choice != null);
                if (choice == null) {
                    const row = try self.fetchLogits();
                    const among: ?[]const u32 = switch (chain.next) {
                        .among => |mask| mask,
                        else => null,
                    };
                    choice = try self.rule().onHost(row, self.logits, among, &self.whole, self.sampler, self.settings, position + 1 + chain.count, chain.product, chain.trail[0..chain.own]) orelse break;
                    self.chosen.host += 1;
                }
                try chain.take(guide, choice.?.token, choice.?.p, choice.?.low, choice.?.closing or std.mem.indexOfScalar(u32, self.ends, @intCast(choice.?.token)) != null);
            },
        };
        // the chain's trail outlives the call, for whoever prints it
        @memcpy(self.trail[0..chain.trailed], chain.trail[0..chain.trailed]);
        self.trailed = chain.trailed;
        if (out.len > 0) self.last_chain = Paying.told(chain);
        return chain.count;
    }
};
