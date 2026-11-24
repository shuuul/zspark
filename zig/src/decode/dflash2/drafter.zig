//! DFlash2 as a `draft.Drafter` over the ranks a process holds (`DrafterOf(held, Target)`), TensorFold's
//! `dflash_decode` around `Drafter.add_taps` and `propose`: the rows the main model kept give their taps to the
//! context, then one block pass proposes the tokens after the newest, a position's candidates merged over the
//! ranks and chained on the host (`chain.zig`). The candidates are read from the pair's gathered `tops`, every
//! rank's, so a rank a process chains from the bits both ranks in one do.
//!
//! What the main model gives is `Target.Of(held)` (`pass.zig`'s header), the drafter's `target`: its taps
//! (`taps`, `tapLayers`), each rank's embedding and rows of the vocabulary (`mains`), and what it owns
//! (`deinit`). GLM's is its held pair; Flash Next's its one-rank model or its held pair, with the vocabulary
//! split over the drafter's two ranks (`models/qwen4/dflash2/drafter.zig`).
//!
//! Departures from TensorFold: under a constraint each draft is first asked of it, as the MTP head asks
//! (`draft.Chain`), a forced token taking a place of the block and a mask leaving its candidates only, where
//! TensorFold's drafter is not told of the grammar; and a context past the capacity stops the drafting
//! where TensorFold raises.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const masks_file = @import("../masks.zig");
const sampling = @import("../sampling.zig");
const numeric = @import("../../weights/numeric.zig");
const draft = @import("../draft.zig");
const chain_file = @import("chain.zig");
const config_file = @import("config.zig");
const pass = @import("pass.zig");
const weights_file = @import("weights.zig");

const world = config_file.world;

pub fn DrafterOf(comptime held: usize, comptime Target: type) type {
    return struct {
        const Self = @This();

        c: *const config_file.Config,
        pair: *pass.PairOf(held, Target),
        /// The main model's side: its taps, its embedding and its rows of the vocabulary (`pass.zig`'s header).
        target: Target.Of(held),
        books: chain_file.Books,
        settings: sampling.Sampling,
        /// The most drafts of a round, and the product of the picks' confidences a chain stops under, its first
        /// pick kept. The default is `fnc7:0.2` in the kit's words, and no longer TensorFold mainline's
        /// `fc5:0.3`: on GLM's whole model the same twelve replies came 3.8% faster by it (spec 067 WS-12; the
        /// kit's own `fnc7:0.3` 1.8%).
        cap: usize = default_cap,
        confidence: f64 = default_confidence,
        /// The product is of the picks' noise-aware confidences (`chain.Pick.noisy`): the kit's `fncN:P`. Only
        /// where a chain stops changes; the picks are the same. False: the product of their probabilities.
        noisy: bool = true,
        /// A chain ends with a draft that is one of these: the reply stops there.
        ends: []const u32 = &.{},
        /// Each rank's four-bit copy of its rows of the vocabulary (`weights.draftHead`), the caller's; without,
        /// the target's own rows (`Target.Of.mains`).
        heads: ?[held]weights_file.Linear = null,
        /// The bf16 rows `heads` were made of where the main head is not BF16 (FP4-Local's MXFP8, widened at
        /// load): what a pick is scored against again (`pass.rescore`). Null: the target's own BF16 rows.
        widened: ?[held]cuda.View = null,
        /// The context is past its capacity: nothing more is absorbed, and nothing drafted.
        full: bool = false,
        /// Block passes run.
        passes: usize = 0,
        /// The last chain's picks with the probability each was given (`draft.Chain.trail`), kept past the call
        /// for whoever records it (`generate ... chains`).
        trail: [draft.forward_max_rows + 1]draft.Chain.Considered = undefined,
        trailed: usize = 0,
        /// `chain.Pick.noisy` of each of them.
        noised: [draft.forward_max_rows + 1]f64 = undefined,
        /// The projected rows as read back.
        selected: []u16,
        /// Each rank's largest of the block's positions [world][block - 1][top] as read back, and the block's
        /// merged candidates.
        half_values: []f32,
        half_tokens: []i32,
        tokens: []u32,
        values: []f64,
        projected: []f64,
        gpa: Allocator,

        const default_cap = 7;
        pub const default_confidence = 0.2;

        /// `settings` is the reply's sampling: a sampled reply's drafts are noised by the key of their place.
        /// `target` moves here: the drafter frees it (`deinit`), on an error here too.
        pub fn init(gpa: Allocator, c: *const config_file.Config, pair: *pass.PairOf(held, Target), target: Target.Of(held), books: weights_file.Codebooks, settings: sampling.Sampling) Allocator.Error!Self {
            var owned = target;
            errdefer owned.deinit();
            std.debug.assert(std.mem.eql(u32, owned.tapLayers(), c.taps) and c.top_k <= chain_file.max_top);
            const rows = c.block - 1;
            var self: Self = .{ .c = c, .pair = pair, .target = owned, .books = .{ .predecessor = books.predecessor, .successor = books.successor, .rank = books.rank }, .settings = settings, .selected = &.{}, .half_values = &.{}, .half_tokens = &.{}, .tokens = &.{}, .values = &.{}, .projected = &.{}, .gpa = gpa };
            errdefer inline for (.{ "selected", "half_values", "half_tokens", "tokens", "values", "projected" }) |name| self.gpa.free(@field(self, name));
            self.selected = try gpa.alloc(u16, rows * c.selector_rank);
            self.half_values = try gpa.alloc(f32, world * (c.block - 1) * c.top_k);
            self.half_tokens = try gpa.alloc(i32, world * (c.block - 1) * c.top_k);
            self.tokens = try gpa.alloc(u32, rows * c.top_k);
            self.values = try gpa.alloc(f64, rows * c.top_k);
            self.projected = try gpa.alloc(f64, rows * c.selector_rank);
            return self;
        }

        pub fn deinit(self: *Self) void {
            self.target.deinit();
            inline for (.{ "selected", "half_values", "half_tokens", "tokens", "values", "projected" }) |name| self.gpa.free(@field(self, name));
            self.* = undefined;
        }

        /// Before a new sequence's first row, with the sampling its drafts are chosen by.
        pub fn reset(self: *Self, settings: sampling.Sampling) void {
            self.pair.reset();
            self.full = false;
            self.settings = settings;
        }

        pub fn drafter(self: *Self) draft.Drafter {
            return .{ .context = self, .draft = drafts };
        }

        pub fn mains(self: Self) [held]pass.Main {
            return self.target.mains(self.heads, self.widened);
        }

        /// The main model's last window's taps on each held rank.
        pub fn taps(self: Self) [held]cuda.Buffer {
            return self.target.taps();
        }

        /// The block pass after `pending`, and its first `depth` positions' candidates on the host
        /// (`Drafter.candidates`).
        pub fn candidates(self: *Self, pending: i32, depth: usize) (pass.GatherError(Target) || Allocator.Error || error{PeerGone})!chain_file.Candidates {
            try self.pair.block(self.mains(), pending);
            return self.read(.{ .tops = self.pair.tops, .selected = self.pair.ranks[0].selected, .rows = self.c.block - 1, .first = 0 }, depth);
        }

        /// Where a block's candidates lie after a pass: `tops` every rank's largest and their tokens, tight
        /// for the pass's `rows` draft rows (a rank's values, then its tokens), `selected` the first rank's
        /// projected rows; the block's own are rows `first` onward of each.
        pub const Source = struct { tops: cuda.Buffer, selected: cuda.Buffer, rows: usize, first: usize };

        /// A block's first `depth` positions' candidates on the host, from where its pass left them.
        pub fn read(self: *Self, s: Source, depth: usize) cuda.Error!chain_file.Candidates {
            const c = self.c;
            const top = c.top_k;
            std.debug.assert(depth >= 1 and depth <= c.block - 1 and s.first + c.block - 1 <= s.rows);
            try cuda.synchronize();
            self.passes += 1;
            // a rank's largest are picked on the device (`kernels.top`): 2 ms of reading the logits back saved
            const per = (c.block - 1) * top;
            // from the gathered table, a rank's values and then its tokens: every rank's, wherever they ran
            for (0..world) |r| {
                const at = (r * 2 * s.rows + s.first) * top * @sizeOf(f32);
                try s.tops.downloadAt(at, std.mem.sliceAsBytes(self.half_values[r * per ..][0 .. depth * top]));
                try s.tops.downloadAt(at + s.rows * top * @sizeOf(f32), std.mem.sliceAsBytes(self.half_tokens[r * per ..][0 .. depth * top]));
            }
            for (0..depth) |d| {
                var halves: [world]chain_file.Half = undefined;
                for (&halves, 0..) |*one, r| one.* = .{ .values = self.half_values[r * per + d * top ..][0..top], .tokens = self.half_tokens[r * per + d * top ..][0..top] };
                chain_file.merge(&halves, self.tokens[d * top ..][0..top], self.values[d * top ..][0..top]);
            }
            // the projected rows are the same bits on every rank; the first here
            const rank_of = c.selector_rank;
            try s.selected.downloadAt(s.first * rank_of * 2, std.mem.sliceAsBytes(self.selected[0 .. depth * rank_of]));
            for (self.projected[0 .. depth * rank_of], self.selected[0 .. depth * rank_of]) |*wide, bits| wide.* = numeric.f32FromBf16(bits);
            return .{ .tokens = self.tokens[0 .. depth * top], .values = self.values[0 .. depth * top], .projected = self.projected[0 .. depth * rank_of], .top = top };
        }

        /// `draft.Drafter.draft`.
        fn drafts(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
            _ = streams;
            const self: *Self = @ptrCast(@alignCast(context));
            self.trailed = 0;
            if (self.full) return 0;
            // every position is absorbed in order, so the context's rows are the sequence's positions
            std.debug.assert(self.pair.context_end + tokens.len == position);
            if (position > self.pair.capacity) {
                self.full = true;
                return 0;
            }
            try self.pair.addTaps(self.taps(), 0, tokens.len);

            const depth = self.depthOf(out.len);
            const pending = tokens[tokens.len - 1];
            var chain: draft.Chain = try .begin(out[0..depth], guide);
            // the block runs once a draft is the drafter's to choose: a chain of forced tokens costs no pass
            if (try advance(&chain, guide)) try self.choices(&chain, guide, try self.candidates(pending, depth), pending, position);
            @memcpy(self.trail[0..chain.trailed], chain.trail[0..chain.trailed]);
            self.trailed = chain.trailed;
            return chain.count;
        }

        /// The drafts of a round at most, of `room` asked.
        pub fn depthOf(self: Self, room: usize) usize {
            return @min(room, self.cap, self.c.block - 1);
        }

        /// Takes what the constraint forces of `chain`; whether its next draft is the drafter's to choose, for
        /// which the block's candidates are wanted. A block holds every position already: nothing is run for a
        /// draft before the next.
        pub fn advance(chain: *draft.Chain, guide: ?masks_file.Guide) !bool {
            while (true) switch (try chain.step(guide)) {
                .halt => return false,
                .run => chain.ran += 1,
                .choose => return true,
            };
        }

        /// The rest of `chain`, which stands at a draft of the drafter's, from its block's candidates `made`,
        /// by `settings` and `ends`: `pending` is the block's first token, at `position`.
        pub fn choices(self: *Self, chain: *draft.Chain, guide: ?masks_file.Guide, made: chain_file.Candidates, pending: i32, position: usize) !void {
            const judged = self.confidence > 0;
            const noise: ?chain_file.Noise = if (self.settings.temperature > 0) .{ .seed = self.settings.seed, .temperature = self.settings.temperature } else null;
            while (true) {
                const prev: u32 = @intCast(if (chain.count == 0) pending else chain.out[chain.count - 1]);
                const allowed: ?[]const u32 = switch (chain.next) {
                    .among => |mask| mask,
                    else => null,
                };
                const one = chain_file.pick(self.books, made, chain.count, prev, position + 1, noise, allowed, judged) orelse return;
                const p = if (self.noisy) one.noisy else one.p;
                const low = judged and chain.product * p < self.confidence;
                if (chain.trailed < self.noised.len) self.noised[chain.trailed] = one.noisy;
                try chain.take(guide, @intCast(one.token), p, low, std.mem.indexOfScalar(u32, self.ends, one.token) != null);
                if (!try advance(chain, guide)) return;
            }
        }
    };
}
