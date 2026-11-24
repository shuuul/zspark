//! Nemotron-3.5-Transcribe's DFlash2 drafter over the target's completed-layer taps. Both drafter
//! weight ranks can run on one GPU; each proposes over its own half of the vocabulary and scores
//! those picks against the corresponding stored BF16 rows. The target chooses every emitted token.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const shared = @import("../../../decode/dflash2/drafter.zig");
const shared_pass = @import("../../../decode/dflash2/pass.zig");
const weights_file = @import("../../../decode/dflash2/weights.zig");
const model_file = @import("../model.zig");
const state_file = @import("../state.zig");
const pass = @import("pass.zig");

pub fn DrafterOf(comptime held: usize) type {
    return shared.DrafterOf(held, pass.Target);
}

/// Both drafter ranks in one process.
pub const Drafter = DrafterOf(shared_pass.world);

/// Each drafter rank's proposing head over its vocabulary half, stored BF16 or a four-bit copy.
/// A whole target's head is split here; a TP2 target already holds its own half.
pub fn draftHeads(comptime held: usize, arena: Allocator, gpa: Allocator, m: model_file.Model, storage: weights_file.LinearStorage, rows: cuda.Scratch.Rows) ![held]weights_file.Linear {
    const half = m.config.backbone.vocab / shared_pass.world;
    const whole = m.head.n == m.config.backbone.vocab;
    if ((whole and held != shared_pass.world) or (!whole and (held != 1 or m.head.n != half))) return error.UnsupportedShape;
    const bytes = try gpa.alloc(u8, m.head.n * m.head.k * 2);
    defer gpa.free(bytes);
    try cuda.synchronize();
    try m.head.weight.download(bytes);
    var heads: [held]weights_file.Linear = undefined;
    var made: usize = 0;
    errdefer for (heads[0..made]) |*one| one.deinit();
    for (&heads, 0..) |*one, rank| {
        const first = if (whole) rank * half * m.head.k * 2 else 0;
        const part = bytes[first..][0 .. half * m.head.k * 2];
        one.* = try .init(arena, gpa, &.{part}, m.head.k, storage, rows);
        made += 1;
    }
    return heads;
}
/// The main model the drafter reads: its embedding, its rows of the vocabulary, and its buffers' taps.
/// `Shares` is what a two-rank split holds: one rank's `Model`, its `Buffers` and which rank it is —
/// the rank's `m.head` is its half of the vocabulary, so the pair gathers the picks (each drafter
/// weight rank takes a candidate rank of the two).
pub fn Of(comptime held: usize) type {
    return struct {
        const Self = @This();

        const Ranks = union(enum) {
            /// The whole backbone on one rank: both drafter ranks read its embedding/taps, and each
            /// scores its own half of the vocabulary.
            whole: struct {
                m: model_file.Model,
                main: *state_file.Buffers,
            },
            /// `held` ranks of the two-rank split (WS-06): rank `i`'s rank is `shares[i].rank`, its
            /// `m.head` already that rank's half.
            shares: [held]Share,
        };

        ranks: Ranks,

        pub const Share = struct {
            /// Which of the pair's two ranks: the head's rows start at `rank * (vocab / world)`.
            rank: usize,
            m: model_file.Model,
            b: *state_file.Buffers,
        };

        /// One rank, the whole model.
        pub fn whole(m: model_file.Model, main: *state_file.Buffers) Self {
            return .{ .ranks = .{ .whole = .{ .m = m, .main = main } } };
        }

        /// The held ranks of the split.
        pub fn heldBy(shares: [held]Share) Self {
            return .{ .ranks = .{ .shares = shares } };
        }

        fn rank(self: Self, r: usize) struct { m: model_file.Model, b: *state_file.Buffers } {
            return switch (self.ranks) {
                .whole => |one| .{ .m = one.m, .b = one.main },
                .shares => |shares| .{ .m = shares[r].m, .b = shares[r].b },
            };
        }

        pub fn tapLayers(self: Self) []const u32 {
            return self.rank(0).b.tap_layers;
        }

        /// Each rank's proposer and the same vocabulary half's stored rows for rescoring.
        pub fn mains(self: Self, heads: ?[held]weights_file.Linear, widened: ?[held]cuda.View) [held]shared_pass.Main {
            var out: [held]shared_pass.Main = undefined;
            for (&out, 0..) |*one, r| {
                const one_rank = self.rank(r);
                const half = one_rank.m.config.backbone.vocab / shared_pass.world;
                const stored = switch (self.ranks) {
                    .whole => one_rank.m.head.weight.view.span(r * half, half),
                    .shares => one_rank.m.head.weight.view,
                };
                std.debug.assert(self.ranks != .whole or heads != null);
                one.* = .{
                    .embed = one_rank.m.embed.view,
                    .head = if (heads) |all| all[r] else .{ .bf16 = one_rank.m.head },
                    .rows = if (widened) |rows| rows[r] else stored,
                };
            }
            return out;
        }

        /// The last window's taps: each rank's own buffers' (the whole model's one on both of a
        /// one-rank target; the completed residual is the same on every rank it is computed on).
        pub fn taps(self: Self) [held]cuda.Buffer {
            var out: [held]cuda.Buffer = undefined;
            for (&out, 0..) |*one, r| one.* = self.rank(r).b.taps.?;
            return out;
        }

        /// Nothing: the model and its buffers are the caller's.
        pub fn deinit(self: *Self) void {
            _ = self;
        }
    };
}
