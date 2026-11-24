//! GLM-5.3-Flash's MTP layer, the drafter its checkpoint carries: layer `layers` of the checkpoint, a DSA and a
//! MoE with plain residuals over one stream, fed the main model's final normed row and the next token's
//! embedding, scored by a four-bit copy of the main model's head where the caller made one (`PairOf.heads`;
//! TensorFold's `draft_head`) and by the head itself where not. TensorFold's `mtp.py` (`mtp_compute`) and
//! `decode.py` (`absorb`, `draft`).
//!
//! For a row: `x = eh [norm_e(embed(next)) | norm_h(hidden)]`, the embedding zero at the sequence's first
//! position; `x += dsa(norm_in(x))` over the layer's own latents, one a position absorbed; `x += moe(norm_post(x))`;
//! the logits are the head's of `norm(x)`, which is also the hidden row the next chained draft reads. A scored
//! row's start is found on the device, each rank's of its half of the vocabulary (`zig/src/decode/pick.zig`), and
//! gathered: a draft is chosen from the halves' joined start where that settles it (`HeadOf.fromPicks`) and
//! from the row read back where not. TensorFold reads 28 candidates a rank and weighs a draft among them;
//! here the start settles the token the whole row gives and carries the whole row's sum.
//!
//! A rank holds the projection and the norms whole and its share of the DSA and the MoE, so the steps are
//! phases with the ranks' partials gathered between them, as the forward's (`forward.zig`); `step` runs them
//! for both ranks in one process, beside `loopback.zig`. Drafts choose nothing (`zig/src/decode/draft.zig`), so
//! nothing here is held to the row rule: a step is the same bits for the same entries, which is all a
//! repeated reply needs.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../../weights/bf16.zig");
const confidence = @import("../../../decode/confidence.zig");
const cuda = @import("../../../cuda/cuda.zig");
const draft = @import("../../../decode/draft.zig");
const masks_file = @import("../../../decode/masks.zig");
const numeric = @import("../../../weights/numeric.zig");
const picking = @import("../../../decode/pick.zig");
const sampling = @import("../../../decode/sampling.zig");
const Config = @import("../layers/config.zig").Config;
const dsa = @import("../layers/dsa.zig");
const hc = @import("../layers/hc.zig");
const layout = @import("../layers/layout.zig");
const lead_file = @import("../../../ranks/lead.zig");
const held_file = @import("../layers/ranks_forward.zig");
const mlp = @import("../layers/mlp.zig");
const model = @import("../layers/model.zig");
const share = @import("../layers/share.zig");
pub const Linear = @import("../../../weights/linear.zig").Linear;
const Picked = picking.Picked;

const world = layout.world;
pub const LoadError = model.LoadError;

/// A rank's share of the MTP layer.
pub const Weights = struct {
    /// bf16 [dims]: the embedding's norm, the hidden row's, the layer's two and the head's
    enorm: cuda.Buffer,
    hnorm: cuda.Buffer,
    in_norm: cuda.Buffer,
    post_norm: cuda.Buffer,
    norm: cuda.Buffer,
    /// [dims, 2 dims], whole on each rank
    eh: bf16.Linear,
    attention: dsa.Weights,
    moe: mlp.Moe,
    eps: f32,
    /// Whose share this is.
    rank: usize,

    /// The layer after the model's last, as `rank` holds it; `arena` and `gpa` as `model.Model.load`'s.
    pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!Weights {
        const p = try std.fmt.allocPrint(arena, layout.prefix ++ "layers.{d}.", .{c.layers});
        var self: Weights = undefined;
        self.eps = @floatCast(c.eps);
        self.rank = rank;
        const norms = .{ .{ "enorm", "enorm.weight" }, .{ "hnorm", "hnorm.weight" }, .{ "in_norm", "input_layernorm.weight" }, .{ "post_norm", "post_attention_layernorm.weight" }, .{ "norm", "shared_head.norm.weight" } };
        var done: usize = 0;
        errdefer inline for (norms, 0..) |one, index| {
            if (index < done) @field(self, one[0]).deinit();
        };
        inline for (norms) |one| {
            @field(self, one[0]) = try model.upload(arena, source, try std.mem.concat(arena, u8, &.{ p, one[1] }), rank, .BF16);
            done += 1;
        }
        const eh_name = try std.mem.concat(arena, u8, &.{ p, "eh_proj.weight" });
        const eh = try share.get(arena, source, eh_name, rank) orelse {
            std.log.err("{s}: not in the checkpoint", .{eh_name});
            return error.MissingTensor;
        };
        if (eh.dtype != .BF16 or eh.data.len != c.hidden * 2 * c.hidden * 2) return error.UnsupportedShape;
        self.eh = try .init(eh.data, c.hidden, 2 * c.hidden, rows);
        errdefer self.eh.deinit();
        self.attention = try .load(arena, source, try std.mem.concat(arena, u8, &.{ p, "self_attn." }), c, rank, rows);
        errdefer self.attention.deinit();
        self.moe = try .load(arena, gpa, source, c, c.layers, rank, rows);
        return self;
    }

    pub fn deinit(self: *Weights) void {
        inline for (.{ "enorm", "hnorm", "in_norm", "post_norm", "norm" }) |name| @field(self, name).deinit();
        self.eh.deinit();
        self.attention.deinit();
        self.moe.deinit();
        self.* = undefined;
    }
};

/// What a step of up to `rows` rows writes on a rank. The DSA's and the MoE's scratch is the main model's:
/// a step runs between its forwards and reads none of what they leave there.
pub const Buffers = struct {
    rows: usize,
    /// int32 [rows]: the next tokens
    ids: cuda.Buffer,
    /// bf16 [rows, dims], [rows, 2 dims]: the embeddings, and both norms side by side
    me: cuda.Buffer,
    cat: cuda.Buffer,
    /// bf16 [rows, dims]: the layer's stream, and a norm of it
    x: cuda.Buffer,
    normed: cuda.Buffer,
    /// bf16 [1, dims] and [1, vocab / world]: the last row's hidden row and its logits
    final: cuda.Buffer,
    logits: cuda.Buffer,
    /// u32 [world, `Picked.wordsFor` the rank's rows of the vocabulary]: the scored row's start in each rank's
    /// half, a rank's at its place (`picked`, a view of it); the others' are brought by a gather, and with both
    /// ranks in one process each reads its own buffer's.
    picks: cuda.Buffer,
    picked: cuda.Buffer,

    pub fn init(m: model.Model, rows: usize) cuda.Error!Buffers {
        var self: Buffers = undefined;
        self.rows = rows;
        const shapes = .{ .{ "me", m.dims }, .{ "cat", 2 * m.dims }, .{ "x", m.dims }, .{ "normed", m.dims } };
        self.ids = try .init(.i32, &.{rows});
        errdefer self.ids.deinit();
        var done: usize = 0;
        errdefer inline for (shapes, 0..) |one, index| {
            if (index < done) @field(self, one[0]).deinit();
        };
        inline for (shapes) |one| {
            @field(self, one[0]) = try .init(.bf16, &.{ rows, one[1] });
            done += 1;
        }
        self.final = try .init(.bf16, &.{ 1, m.dims });
        errdefer self.final.deinit();
        self.logits = try .init(.bf16, &.{ 1, m.head.outputs() });
        errdefer self.logits.deinit();
        const words = Picked.wordsFor(m.head.outputs());
        self.picks = try .init(.u32, &.{ world, words });
        errdefer self.picks.deinit();
        try self.picks.zero();
        const base: [*]u8 = @ptrCast(self.picks.view.ptr.?);
        self.picked = .{ .view = .init(@ptrCast(base + m.rank * words * @sizeOf(u32)), .u32, &.{words}) };
        try Picked.ask(self.picked, 1);
        return self;
    }

    pub fn deinit(self: *Buffers) void {
        inline for (.{ "ids", "me", "cat", "x", "normed", "final", "logits", "picks" }) |name| @field(self, name).deinit();
        self.* = undefined;
    }
};

/// The first phase on a rank: `hidden` bf16 [rows, dims] and the staged next tokens into the layer's stream,
/// and the DSA over the rank's heads at the cache's position into `partial` fp32 [rows, dims]. `first`: the
/// window starts the sequence, whose first embedding is zero.
pub fn attention(m: model.Model, w: Weights, cache: dsa.Cache, b: *Buffers, scratch: dsa.Buffers, hidden: cuda.View, first: bool, partial: cuda.View) dsa.Error!void {
    const rows: usize = @intCast(hidden.size[0]);
    const me = b.me.rows(rows);
    try cuda.launched(cuda.kernels.zs_embed(&b.ids.rows(rows), &m.embed.view, &me, 1));
    if (first) try cuda.zeroLaunched(b.me.rows(1));
    const cat = b.cat.rows(rows);
    var left = cat;
    left.size[1] = @intCast(m.dims);
    left.row_stride = @intCast(2 * m.dims);
    var right: cuda.View = .init(@ptrCast(@as([*]u8, @ptrCast(cat.ptr.?)) + m.dims * 2), .bf16, &.{ rows, m.dims });
    right.row_stride = @intCast(2 * m.dims);
    try hc.rmsnorm(me, w.enorm.view, left, w.eps);
    try hc.rmsnorm(hidden, w.hnorm.view, right, w.eps);
    const x = b.x.rows(rows);
    try w.eh.forward(cat, x);
    const normed = b.normed.rows(rows);
    try hc.rmsnorm(x, w.in_norm.view, normed, w.eps);
    try w.attention.forward(cache, 0, scratch, normed, partial);
}

/// The DSA's residual from `gathered` fp32 [world, rows, dims], then the MoE's partial.
pub fn moe(w: Weights, b: *Buffers, scratch: mlp.Moe.Buffers, rows: usize, gathered: cuda.View, partial: cuda.View) cuda.Error!void {
    const x = b.x.rows(rows);
    try hc.residualAdd(x, gathered);
    const normed = b.normed.rows(rows);
    try hc.rmsnorm(x, w.post_norm.view, normed, w.eps);
    try w.moe.forward(scratch, normed, partial, 0);
}

/// The MoE's residual and the last row's hidden row (`b.final`), which a chained draft reads.
pub fn finish(w: Weights, b: *Buffers, rows: usize, gathered: cuda.View) cuda.Error!void {
    try hc.residualAdd(b.x.rows(rows), gathered);
    try hc.rmsnorm(b.x.span(rows - 1, 1), w.norm.view, b.final.view, w.eps);
}

/// The rank's logits of the hidden row from `head`, its rows of the vocabulary (the main model's, or a
/// four-bit copy of them), and the row's start in them, to the rank `b.picked` is asked to.
pub fn score(head: Linear, b: *Buffers) cuda.Error!void {
    try head.forward(b.final.view, b.logits.view);
    try picking.pick(b.logits.view, b.picked.view);
}

/// The layers, entries and buffers of the ranks one process holds: both (`Pair`), or its own (spec 045).
pub fn PairOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        const main_model = held_file.Held(held);

        weights: [held]Weights,
        /// A latent a position absorbed, then one a chained draft; the same rows on both ranks.
        caches: [held]dsa.Cache,
        b: [held]Buffers,
        /// Of the caches' entries, the chained drafts'.
        drafted: usize = 0,
        /// The main model's (`ranks_forward.zig`): `step`, `setLen` and `wholeRow` are told before they run.
        lead: ?*lead_file.Lead = null,
        /// Each rank's four-bit copy of its rows of the vocabulary (`decode/dflash2/weights.zig`'s `draftHead`), the
        /// caller's: what a step scores with. Null scores with the main model's own rows.
        heads: ?[held]Linear = null,
        /// The rank a scored row's start is asked to (`Picked.ask`), which the reply's sampling needs, and the
        /// one the buffers hold: a step that scores tells it, so both ranks cut their halves at the same rank.
        ask: u32 = 1,
        asked: u32 = 1,

        /// `capacity` entries, steps of up to `rows` rows.
        pub fn init(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, m: main_model.Model, capacity: usize, rows: cuda.Scratch.Rows) LoadError!Self {
            var self: Self = .{ .weights = undefined, .caches = undefined, .b = undefined, .lead = m.lead };
            var done: usize = 0;
            errdefer for (0..done) |index| {
                self.weights[index].deinit();
                self.caches[index].deinit();
                self.b[index].deinit();
            };
            for (m.ranks, 0..) |rank, index| {
                self.weights[index] = try .load(arena, gpa, source, c, rank.rank, rows);
                errdefer self.weights[index].deinit();
                self.caches[index] = try self.weights[index].attention.cacheFor(1, capacity);
                errdefer self.caches[index].deinit();
                self.b[index] = try .init(rank, rows.count);
                done += 1;
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            for (&self.weights, &self.caches, &self.b) |*w, *cache, *b| {
                w.deinit();
                cache.deinit();
                b.deinit();
            }
            self.* = undefined;
        }

        /// Entries held.
        pub fn len(self: Self) usize {
            return self.caches[0].pos;
        }

        pub fn setLen(self: *Self, entries: usize) held_file.Error!void {
            if (self.lead) |lead| try lead.tell(@backingInt(held_file.Op.mtp_len), &.{@intCast(entries)}, &.{});
            for (&self.caches) |*cache| try cache.setPos(entries);
        }

        /// Drops the chained drafts' entries: the next rows absorbed are written over them.
        pub fn trim(self: *Self) held_file.Error!void {
            if (self.drafted == 0) return;
            try self.setLen(self.len() - self.drafted);
            self.drafted = 0;
        }

        /// A rank's hidden rows for a step: rows `from` on of the main model's final rows, or the layer's own.
        pub const Hidden = union(enum) { main: usize, own };

        /// `tokens.len` rows as entries `len()` onward: row i is the hidden row `hidden` names and the token
        /// after it. With `score` each rank's `logits` hold the last row's, and its `picks` the start of every
        /// rank's half of that row, to the rank `ask`. The entries are kept by `setLen`.
        /// `main` is the main model's buffers, whose final rows and scratch a step reads and computes in, and
        /// whose gathered buffers take the layer's partials, `m.gather` between the phases.
        pub fn step(self: *Self, m: main_model.Model, main: *main_model.Buffers, hidden: Hidden, tokens: []const i32, scored: bool) held_file.Error!void {
            const rows = tokens.len;
            try cuda.bound(rows >= 1 and rows <= self.b[0].rows and self.len() + rows <= self.caches[0].capacity, "glm5 mtp: {d} rows (buffers of {d}) at {d} of a capacity of {d}", .{ rows, self.b[0].rows, self.len(), self.caches[0].capacity });
            const first = self.len() == 0;
            if (self.lead) |lead| {
                const digest = held_file.words(try main_model.picks(main.*));
                try lead.tell(@backingInt(held_file.Op.mtp_step), &switch (hidden) {
                    .main => |from| [5]u32{ 0, @intCast(from), if (scored) self.ask else 0, digest[0], digest[1] },
                    .own => [5]u32{ 1, 0, if (scored) self.ask else 0, digest[0], digest[1] },
                }, tokens);
            }
            // the picks in the main model's scratch are this step's from here
            main.window = rows;
            const attn = held_file.gathered(main.attn, rows);
            const ffn = held_file.gathered(main.ffn, rows);
            for (&self.b) |*b| try b.ids.uploadLaunched(0, std.mem.sliceAsBytes(tokens));
            for (m.ranks, self.weights, self.caches, &self.b, main.ranks) |rank, w, cache, *b, scratch| {
                const rows_in = switch (hidden) {
                    .main => |from| scratch.final.span(from, rows),
                    .own => b.final.view,
                };
                try attention(rank, w, cache, b, scratch.dsa.?, rows_in, first, held_file.partial(main.attn, rank.rank, rows));
            }
            try m.gather.run(attn);
            for (m.ranks, self.weights, &self.b, main.ranks) |rank, w, *b, scratch|
                try moe(w, b, scratch.moe.?, rows, attn, held_file.partial(main.ffn, rank.rank, rows));
            try m.gather.run(ffn);
            for (self.weights, &self.b) |w, *b| try finish(w, b, rows, ffn);
            if (!scored) return;
            std.debug.assert(self.ask >= 1 and self.ask <= Picked.most);
            if (self.asked != self.ask) {
                for (self.b) |b| try Picked.ask(b.picked, self.ask);
                self.asked = self.ask;
            }
            for (m.ranks, &self.b, 0..) |rank, *b, index| try score(if (self.heads) |heads| heads[index] else rank.head, b);
            // a rank's start of its half is 4 KiB: every rank's on every rank, as the partials are
            switch (m.gather) {
                .loopback => {},
                .nccl => |comm| try comm.allGather(self.b[0].picked.view, self.b[0].picks.view),
            }
        }

        /// What the last scoring step found of the row in each rank's half of the vocabulary, rank 0's first:
        /// `Picked`'s words. Nothing is told: the step gathered them on every rank.
        pub fn picked(self: Self, out: *[world][Picked.words]u32) cuda.Error!void {
            try cuda.synchronize();
            for (out, 0..) |*raw, rank| {
                const from = self.b[if (held == world) rank else 0].picks;
                try from.downloadAt(rank * @as(usize, @intCast(from.view.size[1])) * @sizeOf(u32), std.mem.sliceAsBytes(raw));
            }
        }

        /// With the weights and what a step needs made elsewhere (a pack's: `pack.Loaded.mtp`).
        pub fn of(weights: [held]Weights, m: main_model.Model, capacity: usize) cuda.Error!Self {
            var self: Self = .{ .weights = weights, .caches = undefined, .b = undefined, .lead = m.lead };
            var done: usize = 0;
            errdefer for (0..done) |index| {
                self.caches[index].deinit();
                self.b[index].deinit();
            };
            for (m.ranks, 0..) |rank, index| {
                self.caches[index] = try weights[index].attention.cacheFor(1, capacity);
                errdefer self.caches[index].deinit();
                self.b[index] = try .init(rank, m.max_rows);
                done += 1;
            }
            return self;
        }

        /// The scored row's logits over the vocabulary, wherever the ranks are: `logitsRow` with all of them
        /// here, the halves gathered through `main` with one (`ranks_forward.zig`'s `gatherRow`).
        pub fn wholeRow(self: Self, main: main_model.Buffers, out: []u16) held_file.Error!void {
            if (self.lead) |lead| try lead.tell(@backingInt(held_file.Op.mtp_row), &.{}, &.{});
            if (held < world) return main_model.gatherRow(main, self.b[0].logits.view, out);
            return self.logitsRow(out);
        }

        /// The scored row's logits as bf16 bits [vocab], rank 0's half and then rank 1's; a process that holds
        /// one rank writes that rank's half alone.
        pub fn logitsRow(self: Self, out: []u16) cuda.Error!void {
            try cuda.synchronize();
            const half = out.len / world;
            for (self.b, self.weights) |b, w| try b.logits.download(std.mem.sliceAsBytes(out[w.rank * half ..][0..half]));
        }
    };
}

/// The layer drafting for several sequences at once (`hall.zig`): a seat a slot, and a step over several.
pub const SeatsOf = @import("mtp_seats.zig").SeatsOf;

/// Both ranks' layers, entries and buffers in one process.
pub const Pair = PairOf(world);

/// The MTP layer as a `draft.Drafter`, TensorFold's `absorb` and `draft`: it absorbs the rows the main model
/// kept, then chains drafts, each from its own last hidden row, each chosen as the main model will choose at
/// that position (the same sampling and key) from the layer's logits.
pub fn HeadOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        const main_model = held_file.Held(held);

        m: main_model.Model,
        pair: *PairOf(held),
        /// The main model's buffers: its final rows are what `draft`'s `streams` names.
        main: *main_model.Buffers,
        sampler: *sampling.Sampler,
        settings: sampling.Sampling,
        /// A chain stops before a draft that takes the product of its drafts' probabilities below this, but for
        /// its first, which is kept and ends it: TensorFold's rule (`decode.py:draft`), with the 0.35 its engine
        /// gives a greedy reply's MTP. 0 chains as far as it is asked.
        confidence: f32 = default_confidence,
        /// A chain ends with a draft that is one of these: the reply stops there.
        ends: []const u32 = &.{},
        /// The most drafts of a round: the room a caller asks is cut to it, as the DFlash2 drafter's
        /// `depthOf` cuts its own. The room's whole 15 as it has always drafted.
        cap: usize = default_cap,
        /// One row's logits on the host, and its sum.
        logits: []u16,
        whole: confidence.Normalizer,
        /// A draft the constraint leaves free is chosen from what the device found of its row (`fromPicks`)
        /// where that settles the choice, and from the row itself, read back, where it does not. False always
        /// reads the row, as every draft did before spec 068: the tests hold the two against each other.
        on_device: bool = true,
        /// How near its threshold a product may be before the row is read: the device's sum is fp32 terms
        /// added in fp32 and counts each half to its own largest, within 2e-5 of the host's.
        margin: f64 = confidence_margin,
        /// Drafts chosen each way.
        chosen: struct { device: usize = 0, host: usize = 0 } = .{},
        joined: picking.Joined = .{},
        /// The cache is full: the head has stopped absorbing, and so drafting.
        full: bool = false,
        /// The last chain's own choices with the probability the head gave each (`draft.Chain.trail`), kept
        /// past the call for whoever records it (`generate ... chains`).
        trail: [draft.forward_max_rows + 1]draft.Chain.Considered = undefined,
        trailed: usize = 0,
        gpa: Allocator,

        pub const default_confidence = 0.35;
        pub const default_cap = 15;
        pub const confidence_margin = 1e-4;

        /// The sampler is the main model's: drafts are keyed as its choices are.
        pub fn init(gpa: Allocator, m: main_model.Model, pair: *PairOf(held), main: *main_model.Buffers, sampler: *sampling.Sampler, settings: sampling.Sampling, vocab: usize) !Self {
            const logits = try gpa.alloc(u16, vocab);
            errdefer gpa.free(logits);
            return .{ .m = m, .pair = pair, .main = main, .sampler = sampler, .settings = settings, .logits = logits, .whole = try .init(gpa), .gpa = gpa };
        }

        pub fn deinit(self: *Self) void {
            self.gpa.free(self.logits);
            self.whole.deinit(self.gpa);
            self.* = undefined;
        }

        /// Before a new sequence's first row, with the sampling its drafts are chosen by.
        pub fn reset(self: *Self, settings: sampling.Sampling) held_file.Error!void {
            try self.pair.setLen(0);
            self.pair.drafted = 0;
            self.full = false;
            self.settings = settings;
        }

        pub fn drafter(self: *Self) draft.Drafter {
            return .{ .context = self, .draft = drafts };
        }

        /// The rank the halves' starts are asked to for the reply's sampling: what settles its choice
        /// (`sampling.leadingNeeded`), or 1 where the device cannot serve it (no cut at all, a temperature
        /// over 1, a `top_k` past `Picked.most`) and every row is read.
        pub fn rankAsked(self: Self) u32 {
            const needed = sampling.leadingNeeded(self.settings, self.logits.len) orelse return 1;
            return if (needed <= @min(Picked.most, self.logits.len / world)) @intCast(needed) else 1;
        }

        const Choice = struct { token: i32, p: f64, low: bool };

        /// The draft at `position` from the halves' starts of the scored row, as the sampler and the
        /// normalizer give it from the whole row; null where the joined start does not settle it
        /// (`Sampler.chooseLeading`: a nucleus whose cut the start does not bound, a zero on top of a greedy
        /// row, more equals at a cut than a pick holds) or the chain's product is at the threshold.
        fn fromPicks(self: *Self, raw: *const [world][Picked.words]u32, position: u64, chain: *const draft.Chain) ?Choice {
            var halves: [world]Picked = undefined;
            for (&halves, raw) |*half, *words| half.* = .read(words);
            const row = picking.joined(&halves, self.logits.len / world, &self.joined) orelse return null;
            const token = self.sampler.chooseLeading(row, position, self.settings) orelse return null;
            if (self.confidence <= 0) return .{ .token = token, .p = 1, .low = false };
            const sum = row.sum orelse return null;
            const at = std.mem.findScalar(u32, row.ids, @intCast(token)).?;
            const p = confidence.fromLeading(sampling.bitsOf(row.keys[at]), sampling.bitsOf(row.keys[0]), sum);
            const through = chain.product * p;
            if (@abs(through - self.confidence) <= self.margin * @as(f64, @floatFromInt(chain.own + 1)) * self.confidence) return null;
            return .{ .token = token, .p = p, .low = through < self.confidence };
        }

        /// `draft.Drafter.draft`. Under a constraint each draft is first asked of it, as Flash Next's head asks
        /// (`draft.Chain`): a forced token is the draft, and among several the head chooses with its other logits
        /// at -inf. TensorFold's drafter is not told of the grammar; its drafts are cut after the fact.
        fn drafts(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
            const self: *Self = @ptrCast(@alignCast(context));
            self.trailed = 0;
            if (self.full) return 0;
            try self.pair.trim();
            // every position is absorbed in order, so the head's entries are the sequence's positions
            std.debug.assert(streams.ptr == self.main.ranks[0].final.view.ptr and self.pair.len() + tokens.len == position);
            if (position > self.pair.caches[0].capacity) {
                self.full = true;
                return 0;
            }
            self.pair.ask = self.rankAsked();
            var chain: draft.Chain = try .begin(out[0..@min(out.len, self.cap)], guide);
            // a window of the main model's is at most the rows a step takes: both are the model's `max_rows`
            try self.pair.step(self.m, self.main, .{ .main = 0 }, tokens, chain.scores());
            try self.pair.setLen(position);
            while (true) switch (try chain.step(guide)) {
                .halt => break,
                .run => {
                    if (self.pair.len() == self.pair.caches[0].capacity) break;
                    try self.pair.step(self.m, self.main, .own, out[chain.ran..][0..1], true);
                    try self.pair.setLen(self.pair.len() + 1);
                    self.pair.drafted += 1;
                    chain.ran += 1;
                },
                .choose => if (!try self.choose(Own{ .pair = self.pair, .main = self.main }, position, guide, &chain)) break,
            };
            @memcpy(self.trail[0..chain.trailed], chain.trail[0..chain.trailed]);
            self.trailed = chain.trailed;
            return chain.count;
        }

        /// The pair's last scored row, as `choose` reads one.
        const Own = struct {
            pair: *PairOf(held),
            main: *main_model.Buffers,

            pub fn picked(self: Own, out: *[world][Picked.words]u32) cuda.Error!void {
                return self.pair.picked(out);
            }

            pub fn whole(self: Own, out: []u16) held_file.Error!void {
                return self.pair.wholeRow(self.main.*, out);
            }
        };

        /// The chain's next draft, the head's to choose, from the scored `row` (its `picked` and `whole`: the
        /// pair's last step's, or one of a step of several sequences', `mtp_seats.zig`), by `settings` and
        /// `ends`; `position` is the sequence's before the chain. False where the constraint leaves the head
        /// no token: the chain ends as it stands.
        pub fn choose(self: *Self, row: anytype, position: usize, guide: ?masks_file.Guide, chain: *draft.Chain) !bool {
            // a mask is laid over the whole row
            if (self.on_device and chain.next == .free) {
                var raw: [world][Picked.words]u32 = undefined;
                try row.picked(&raw);
                if (self.fromPicks(&raw, position + 1 + chain.count, chain)) |one| {
                    self.chosen.device += 1;
                    try chain.take(guide, one.token, one.p, one.low, std.mem.indexOfScalar(u32, self.ends, @intCast(one.token)) != null);
                    return true;
                }
            }
            self.chosen.host += 1;
            try row.whole(self.logits);
            const judged = self.confidence > 0;
            const sum = if (judged) self.whole.of(self.logits) else 0;
            switch (chain.next) {
                .among => |mask| {
                    masks_file.write(mask, self.logits);
                    if (std.mem.indexOfNone(u16, self.logits, &.{confidence.minus_infinity}) == null) return false;
                },
                else => {},
            }
            const token = try self.sampler.choose(self.logits, position + 1 + chain.count, self.settings);
            const p: f64 = if (judged) confidence.fromSum(self.logits[@intCast(token)], sum) else 1;
            const low = judged and chain.product * p < self.confidence;
            try chain.take(guide, token, p, low, std.mem.indexOfScalar(u32, self.ends, @intCast(token)) != null);
            return true;
        }
    };
}

/// Both ranks in one process.
pub const Head = HeadOf(world);
