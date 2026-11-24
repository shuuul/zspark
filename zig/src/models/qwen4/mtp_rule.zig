//! How a draft is chosen from a row the head scored (spec 057's rule) and while it pays (spec 061's cost
//! model): one rule for a session's head (`mtp.Head`) and for each seat of a round (`hall_engine.zig`),
//! which differ in how the head's rows are launched and not in this.

const std = @import("std");
const confidence = @import("../../decode/confidence.zig");
const draft = @import("draft.zig");
const masks_file = @import("../../decode/masks.zig");
const mtp = @import("mtp.zig");
const numeric = @import("../../weights/numeric.zig");
const sampling = @import("../../decode/sampling.zig");
const paying_file = @import("../../decode/paying.zig");
const Head = mtp.Head;
const Normalizer = mtp.Normalizer;
const Picked = mtp.Picked;
const minus_infinity = mtp.minus_infinity;

/// How a draft is chosen from a row the head scored, and whether the head gave it too little to draft on
/// from: one rule for a session's head (`Head`) and for each seat of a round (`hall_engine.zig`), which differ
/// in how the head's rows are launched and not in this (spec 057). `Head`'s fields of these names say what
/// each is and how it was chosen.
/// A draft kept while it pays (spec 061), TensorFold 1.0.2's `DepthRule` (`zig/src/core/draft_depth.zig`) at
/// its power of 2: a chain's draft after its first is verified while the square of the chain's probability
/// with it is at least the tokens a round's cost the stream's recent rounds earned, times what its row adds;
/// the chain goes on from a draft only while that also covers the head's level for the next. So a stream
/// whose rounds keep little (reasoning) drafts on where one that keeps much does not. The rate follows the
/// rounds by their modelled cost and never a clock: two runs of a request draft alike.
///
/// Costs are in a one-row window's time. A departure from TensorFold, which times sixteen windows five
/// times at load: `row` and `level` are constants a storage, fitted to recorded rounds (spec 061), since
/// those windows would add 3 to 5 s to a load. Over fourteen recorded replies the rule models 7.2% on
/// reasoning on the retired NVFP4 checkpoint, prose, code and calls within 1%.
pub const Paying = struct {
    /// What a drafted row adds to its window, and a level of the head.
    row: f64,
    level: f64,
    /// Tokens a round and its cost, over the recent rounds: each new one weighs `weight`.
    tokens: f64 = 2,
    cost: f64,
    /// Of a chain's drafts after its first, how many the recent rounds kept and how many the squares of
    /// their chains' probabilities said they would: the head is kept half as often as it says on prose, and
    /// more often than it says on reasoning (spec 061). Their ratio, within `trust_range`, multiplies the
    /// square a draft is judged by; zspark's own, TensorFold's rule reading the probability as it is.
    got: f64 = 1,
    said: f64 = 1,
    /// The round the stream shares, where its drafts are priced by it (`Round`); null prices it alone.
    round: ?Round.Price = null,
    /// The checkpoint's calibration where it has one (`Landing`): a chain then pays by it too.
    landing: ?Landing = null,
    counts: Counts = .{},

    /// What a draft the head gives `p` lands with, the chain's drafts before it kept: a logistic on the
    /// logit of `p` with an intercept a level, fitted a checkpoint on recorded chains (spec 085; vLLM's
    /// `OnlineAcceptanceEstimator` is this shape, fitted as it serves). Where a checkpoint has one, a draft
    /// pays where its chain pays by its probability squared, as without one, and by the product of these,
    /// no trust on it, both: the head says too much of a deep draft on code, and the calibration only cuts.
    /// In the square's place, with a trust of its own, it drew more on prose and was timed slower there.
    pub const Landing = struct {
        slope: f64,
        /// By the head's own drafts before this one; the last is five and more. The retired EXL3 packs had
        /// one each (tag retired-checkpoints-2026-10-21); no checkpoint read today has one.
        levels: [6]f64,

        pub fn of(self: Landing, p: f64, own: usize) f64 {
            const held = std.math.clamp(p, 1e-6, 1 - 1e-6);
            return 1 / (1 + @exp(-(self.slope * @log(held / (1 - held)) + self.levels[@min(own, self.levels.len - 1)])));
        }

        /// A chain of `trail` then a draft given `p`.
        pub fn through(self: Landing, trail: []const draft.Chain.Considered, p: f64) f64 {
            var product = self.of(p, trail.len);
            for (trail, 0..) |one, own| product *= self.of(one.p, own);
            return product;
        }
    };

    /// The depths told apart, a chain's own drafts before a level: 1, 2, 3, and 4 and more together.
    pub const deep = 4;

    /// A reply's levels as its rounds were told of them, for its line in the log: by depth the levels run
    /// for the chain, the drafts they scored that the rule took, and those the main model kept; `levels`
    /// the head's chained passes in the reply's rounds, whoever chained in them. Depth k is a chain standing
    /// at k drafts of its own, the level's draft being its k + 1-th; the last entry is 4 and more. A reply's
    /// last chain is in none, nor a chain a constraint forced tokens into.
    pub const Counts = struct {
        levels: usize = 0,
        passes: [deep]usize = @splat(0),
        taken: [deep]usize = @splat(0),
        kept: [deep]usize = @splat(0),

        pub fn line(self: Counts, buffer: []u8) []const u8 {
            if (self.levels == 0) return "";
            return std.fmt.bufPrint(buffer, "; {d} chained passes, by depth {any} for it, {any} taken, {any} kept", .{ self.levels, self.passes, self.taken, self.kept }) catch "";
        }
    };

    fn bucket(standing: usize) usize {
        return @min(@max(standing, 1), deep) - 1;
    }

    const trust_weight = 0.05;
    const trust_range: [2]f64 = .{ 0.25, 1 };

    /// A chain as a round is told of it: its drafts, those the head scored, and what the head's own after
    /// the first were said to keep.
    /// `own` of them the head's.
    pub const Told = struct { drafts: usize, scored: usize, said: f64, own: usize = 0 };

    pub fn told(chain: draft.Chain) Told {
        var said: f64 = 0;
        var through: f64 = 1;
        var out: Told = .{ .drafts = chain.count, .scored = chain.trailed, .said = 0, .own = chain.own };
        for (chain.trail[0..chain.own], 0..) |one, i| {
            through *= one.p;
            if (i > 0) said += through * through;
        }
        if (chain.own > 1) out.said = said;
        return out;
    }

    /// What a drafter gives the rule, in a lone stream's one-row round: a drafted row, a level of the head
    /// (0 for a drafter that drafts a block in one pass), and what a round of several seats changes of them.
    pub const Costs = struct {
        row: f64,
        level: f64,
        /// Of a seat's one-row round, the part a round of several seats pays once. 0, and no `rows`: the
        /// storage has no round of several seats, and its seats are priced alone (`Round.price`).
        shared: f64 = 0,
        /// The checkpoint's calibration (`Landing`); null judges a chain by its probability squared.
        landing: ?Landing = null,
        /// What a row costs by the round's rows, against `row`: the first band at or above them. None: `row`.
        rows: []const Band = &.{},

        pub const Band = struct { upto: usize, of_row: f64 };

        pub fn rowAt(self: Costs, rows: usize) f64 {
            for (self.rows) |band| if (rows <= band.upto) return band.of_row;
            return if (self.rows.len > 0) self.rows[self.rows.len - 1].of_row else 1;
        }
    };
    /// The retired NVFP4 checkpoint's window of one row 51 ms, a row more 3.3, a chained draft 1.63
    /// (`generate` at `context=8192`, spec 061), which FP4-Local drafts by, its own not fitted. It has no
    /// round of several seats: its seats are priced alone. One was fitted from a traced load of one to four
    /// replies of prose (404 rounds of 2 to 24 rows: 52.1 ms, 3.5 a seat and 2.2 a row, so 0.90 paid once,
    /// and half a row past 16 rows) and gained nothing at four streams: 80.5 tok/s of prose against 80.2,
    /// inside the loads' spread, and lists level (spec 077 WS-03). The retired EXL3 packs' costs had a round
    /// (spec 075; tag retired-checkpoints-2026-10-21).
    pub const published: Costs = .{ .row = 0.065, .level = 0.032 };

    /// The round a pool's seats share (spec 077), as TensorFold's shared rounds weigh a draft
    /// (`core/lanes/depth.zig` `budgets`) and vLLM's and SGLang's adaptive verification do: against the
    /// whole round's tokens and cost, not the stream's own. The decoding seats' `tokens` and `cost` are
    /// summed, the part of a one-row round that the seats pay once taken off for each seat past the first,
    /// and a row is priced at the round's rows, a prompt chunk's beside the replies' among them. Nothing
    /// else crosses seats: the trust stays each reply's. One seat alone is priced as without it, to the bit
    /// (the sums are its own, nothing is taken off, the first band is `row`). Departures from TensorFold: the
    /// costs are constants a storage where it times rows at load and reads a clock each round, and the rows
    /// are those of the round just run, the next one's being what the drafts chosen here make.
    pub const Round = struct {
        tokens: f64 = 0,
        cost: f64 = 0,
        seats: usize = 0,

        /// What a seat's floors read of its round.
        pub const Price = struct {
            /// Tokens a unit of cost, the round's.
            rate: f64,
            /// A row at the round's rows, against the seat's `row`.
            of_row: f64,
        };

        pub fn add(self: *Round, seat: Paying) void {
            self.tokens += seat.tokens;
            self.cost += seat.cost;
            self.seats += 1;
        }

        /// The price of a draft in this round of `rows` rows; null for a round of no seat, and for a storage
        /// that has no round of several seats: each seat is then priced alone.
        pub fn price(self: Round, costs: Costs, rows: usize) ?Price {
            if (self.seats == 0 or (costs.shared == 0 and costs.rows.len == 0)) return null;
            const once = costs.shared * @as(f64, @floatFromInt(self.seats - 1));
            return .{ .rate = self.tokens / (self.cost - once), .of_row = costs.rowAt(rows) };
        }
    };

    /// Before a reply's first round: the rate of a round that keeps one draft.
    pub fn begin(costs: Costs) Paying {
        return .{ .row = costs.row, .level = costs.level, .cost = 1 + costs.row + costs.level, .landing = costs.landing };
    }

    pub fn rate(self: Paying) f64 {
        return self.tokens / self.cost;
    }

    /// A round gave `tokens` tokens, its window drafted from `chain`.
    pub fn done(self: *Paying, tokens: usize, chain: Told) void {
        const cost = 1 + self.row * @as(f64, @floatFromInt(chain.drafts)) + self.level * @as(f64, @floatFromInt(chain.scored));
        self.tokens = paying_file.Rate.average(self.tokens, @floatFromInt(tokens));
        self.cost = paying_file.Rate.average(self.cost, cost);
        if (chain.said > 0) {
            // the token the window's last kept row chose, and the first draft, are not drafts after the first
            const kept: f64 = @floatFromInt(@min(tokens -| 2, chain.drafts - 1));
            self.got += trust_weight * (kept - self.got);
            self.said += trust_weight * (chain.said - self.said);
        }
        // the head's choice `at` past its first came of a level run for a chain standing at `at`; a chain
        // with a constraint's tokens in it does not say which rows its own drafts were
        if (chain.drafts != chain.own) return;
        for (1..@max(chain.scored, 1)) |at| {
            const taken = at < chain.own;
            const landed = taken and tokens >= at + 2;
            const into = bucket(at);
            self.counts.passes[into] += 1;
            self.counts.taken[into] += @intFromBool(taken);
            self.counts.kept[into] += @intFromBool(landed);
        }
    }

    /// What a chain's probability must reach for a draft after the first to be verified, and for the chain
    /// to go on from a draft.
    pub fn floors(self: Paying) [2]f64 {
        const rate_of, const row = if (self.round) |round| .{ round.rate, self.row * round.of_row } else .{ self.rate(), self.row };
        const r = rate_of / std.math.clamp(self.got / self.said, trust_range[0], trust_range[1]);
        return .{ @sqrt(r * row), @sqrt(r * (row + self.level)) };
    }

    /// Under `landing`: whether a chain that lands with `landed` is too little for its newest draft's row
    /// (`low`), or enough for it and not for another level (`closing`).
    pub fn pays(self: Paying, landed: f64) struct { low: bool, closing: bool } {
        const rate_of, const row = if (self.round) |round| .{ round.rate, self.row * round.of_row } else .{ self.rate(), self.row };
        return .{ .low = landed < rate_of * row, .closing = landed < rate_of * (row + self.level) };
    }

    /// The stream as its round prices it; with null, alone.
    pub fn within(self: Paying, round: ?Round.Price) Paying {
        var priced = self;
        priced.round = round;
        return priced;
    }
};

pub const Rule = struct {
    confidence: f32 = 0,
    chain: f32 = Head.default_chain,
    /// In place of `chain` where it is set: the stream's `Paying` as it stands before this round.
    paying: ?Paying = null,
    on_device: bool = true,
    margin: f64 = Head.confidence_margin,

    pub const Choice = struct {
        token: i32,
        /// The head gave the token, or its chain with it, too little.
        low: bool,
        /// What it gave it; 0 where no threshold is set (`judges`).
        p: f32 = 0,
        /// The draft is kept and the chain ends with it (`paying`: another level would not pay).
        closing: bool = false,
    };

    /// A draft's probability is needed: a threshold is set.
    pub fn judges(self: Rule) bool {
        return self.confidence > 0 or self.chain > 0 or self.paying != null;
    }

    /// The head gives the draft `p`, and `through` its chain with it, `own` drafts of its own before it: too
    /// little to draft on from (`low`), or enough for this draft and not for another (`closing`).
    pub fn little(self: Rule, p: f64, through: f64, own: usize) struct { low: bool, closing: bool } {
        if (p < self.confidence) return .{ .low = true, .closing = false };
        const paying = self.paying orelse return .{ .low = through < self.chain, .closing = false };
        const floors = paying.floors();
        if (own > 0 and through < floors[0]) return .{ .low = true, .closing = false };
        return .{ .low = false, .closing = through < floors[1] };
    }

    /// `little` of a draft given `p` after the head's own `trail`, whose product is `before`; and by the
    /// checkpoint's calibration where the stream has one, the lower of the two.
    fn weigh(self: Rule, p: f64, before: f64, trail: []const draft.Chain.Considered) struct { low: bool, closing: bool } {
        if (p >= self.confidence) if (self.paying) |paying| if (paying.landing) |landing| {
            const found = paying.pays(landing.through(trail, p));
            const squared = self.little(p, before * p, trail.len);
            if (squared.low or (trail.len > 0 and found.low)) return .{ .low = true, .closing = false };
            return .{ .low = false, .closing = found.closing or squared.closing };
        };
        const found = self.little(p, before * p, trail.len);
        return .{ .low = found.low, .closing = found.closing };
    }

    /// `through` is so near a threshold that the device's sum and the host's may fall on either side of it.
    /// Under a calibration a margin of a probability near 1 is much of its logit, so the chain is weighed with
    /// its newest draft a margin under and a margin over (the trail's are numbers both sides read alike), and
    /// is near where the two differ, as it is by the square's own margins.
    fn near(self: Rule, p: f64, before: f64, trail: []const draft.Chain.Considered) bool {
        const own = trail.len;
        const through = before * p;
        const scale = self.margin * @as(f64, @floatFromInt(own + 1));
        if (self.paying) |paying| {
            if (paying.landing != null and !std.meta.eql(self.weigh(p * (1 - self.margin), before, trail), self.weigh(p * (1 + self.margin), before, trail))) return true;
            for (paying.floors()) |floor| if (@abs(through - floor) <= scale * floor) return true;
            return false;
        }
        return self.chain > 0 and @abs(through - self.chain) <= scale * self.chain;
    }

    /// The rank a scored row's `Picked` is asked to for `settings`, the row having `columns` logits of a
    /// vocabulary of `vocab`; 0 where the device cannot serve them (`sampling.leadingNeeded`: no cut at all, a
    /// `top_k` past `Picked.most` or the columns), and the buffer is then asked to 1.
    pub fn rank(settings: sampling.Sampling, columns: usize, vocab: usize) u32 {
        return Picked.rankFor(settings, columns, vocab);
    }

    /// The draft at `position` from `raw`, what `pick` found in the scored row, as the sampler and
    /// `normalizer` give it from the whole row; null where the row's start does not settle it
    /// (`Sampler.chooseLeading`: a nucleus whose cut the start does not bound, a `top_k` past the kernel's candidates, a zero on top of
    /// a greedy row, a row that is not finite) or the probability is at a threshold. `ids` is a draft
    /// head's vocabulary, null where the row is the whole one; `before` the product of the chain's own
    /// drafts before this one, `trail` those drafts.
    pub fn onDevice(self: Rule, raw: *const [Picked.words]u32, ids: ?[]const u32, vocab: usize, sampler: *sampling.Sampler, settings: sampling.Sampling, position: u64, before: f64, trail: []const draft.Chain.Considered) ?Choice {
        const picked: Picked = .read(raw);
        if (picked.count == 0) return null;
        // a draft head's columns are its ids ascending, so the columns' order is the ids'
        var tokens: [Picked.slots]u32 = undefined;
        for (tokens[0..picked.count], picked.columns[0..picked.count]) |*id, column| id.* = if (ids) |known| known[column] else column;
        const leading: sampling.Leading = .{ .keys = picked.keys[0..picked.count], .ids = tokens[0..picked.count], .smallest = picked.smallest, .len = vocab, .sum = picked.sum };
        const token = sampler.chooseLeading(leading, position, settings) orelse return null;
        if (!self.judges()) return .{ .token = token, .low = false };
        const at = std.mem.findScalar(u32, leading.ids, @intCast(token)).?;
        // GLM's head (`glm5/mtp/mtp.zig`) takes the same expression through the same function
        const p = confidence.fromLeading(sampling.bitsOf(picked.keys[at]), sampling.bitsOf(picked.largest), picked.sum);
        if (self.confidence > 0 and @abs(p - self.confidence) <= self.margin) return null;
        if (self.near(p, before, trail)) return null;
        const found = self.weigh(p, before, trail);
        return .{ .token = token, .low = found.low, .p = @floatCast(p), .closing = found.closing };
    }

    /// The draft at `position` from the row read back. `logits` is the row over the vocabulary, -inf where
    /// the head scored nothing, and `row` the logits it scored (a draft head's columns, or `logits`). Under
    /// `among`, a constraint's mask, the head chooses among what it leaves, its other logits at -inf, as the
    /// main model will under the same mask; null where it scored none of those. The head's confidence is
    /// what it gave the token before the mask: a mask that leaves little of what the head expected makes
    /// its choice among the rest look sure when it is a guess, and a chain drafted on from a guess is rows
    /// computed for nothing. `product` is the chain's before this draft, of the head's own drafts `trail`.
    pub fn onHost(self: Rule, row: []const u16, logits: []u16, among: ?[]const u32, whole: *Normalizer, sampler: *sampling.Sampler, settings: sampling.Sampling, position: u64, product: f64, trail: []const draft.Chain.Considered) !?Choice {
        const sum = if (self.judges()) whole.of(row) else 0;
        if (among) |mask| {
            masks_file.write(mask, logits);
            // the head may score none of the tokens the constraint takes here
            if (std.mem.indexOfNone(u16, logits, &.{minus_infinity}) == null) return null;
        }
        const token = try sampler.choose(logits, position, settings);
        const p: f32 = if (self.judges()) @floatCast(confidence.fromSum(logits[@intCast(token)], sum)) else 0;
        if (!self.judges()) return .{ .token = token, .low = false, .p = p };
        const found = self.weigh(p, product, trail);
        return .{ .token = token, .low = found.low, .p = p, .closing = found.closing };
    }
};
