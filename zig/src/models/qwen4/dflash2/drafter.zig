//! Qwen3.8 Flash Next's DFlash2 drafter: `decode/dflash2/drafter.zig` over Qwen's target (`Of(held)`). The
//! target is one `Model` (not a held pair): both drafter weight ranks still run on one GPU, and the target's
//! embedding table and its vocabulary rows are split across those ranks for scoring (`one`). Over a held pair
//! (`held`) each rank already holds its half of `lm_head` and the whole embedding; taps come from that rank's
//! buffers. Capture of taps is the target's `Buffers.setTaps`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const shared = @import("../../../decode/dflash2/drafter.zig");
const shared_pass = @import("../../../decode/dflash2/pass.zig");
const weights_file = @import("../../../decode/dflash2/weights.zig");
const forward_file = @import("../forward.zig");
const model_file = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const pass = @import("pass.zig");
const bf16 = @import("../../../weights/bf16.zig");

const world = shared_pass.world;

pub fn DrafterOf(comptime held: usize) type {
    return shared.DrafterOf(held, pass.Target);
}

/// Both ranks in one process.
pub const Drafter = DrafterOf(world);

/// Each rank's owned four-bit proposer, from the target's stored BF16 head.
pub fn splitHeads(arena: Allocator, gpa: Allocator, m: model_file.Model, rows: cuda.Scratch.Rows) ![world]weights_file.Linear {
    std.debug.assert(m.shape.vocab % world == 0 and m.head.count == 1);
    const hidden = m.shape.hidden;
    const half = m.shape.vocab / world;
    const linear = switch (m.head.runs[0].linear) {
        .bf16 => |one| one,
        else => return error.UnsupportedShape,
    };
    try cuda.synchronize();
    const bytes = try gpa.alloc(u8, half * hidden * 2);
    defer gpa.free(bytes);
    var heads: [world]weights_file.Linear = undefined;
    var made: usize = 0;
    errdefer for (heads[0..made]) |*one| one.deinit();
    for (&heads, 0..) |*one, rank| {
        try linear.weight.downloadAt(rank * half * hidden * 2, bytes);
        one.* = try .init(arena, gpa, &.{bytes}, hidden, .four, rows);
        made += 1;
    }
    return heads;
}

/// A rank's owned four-bit proposer, from its already-halved `lm_head`.
pub fn shareHead(arena: Allocator, gpa: Allocator, m: model_file.Model, rows: cuda.Scratch.Rows) !weights_file.Linear {
    std.debug.assert(m.head.count == 1);
    const linear = switch (m.head.runs[0].linear) {
        .bf16 => |one| one,
        else => return error.UnsupportedShape,
    };
    try cuda.synchronize();
    const bytes = try gpa.alloc(u8, linear.n * linear.k * 2);
    defer gpa.free(bytes);
    try linear.weight.download(bytes);
    return .init(arena, gpa, &.{bytes}, linear.k, .four, rows);
}

/// Each held rank's half of the vocabulary, already cut on that share.
pub fn splitHeadsFromShares(comptime held: usize, arena: Allocator, gpa: Allocator, ranks: [held]model_file.Model, rows: cuda.Scratch.Rows) ![held]weights_file.Linear {
    var heads: [held]weights_file.Linear = undefined;
    var made: usize = 0;
    errdefer for (heads[0..made]) |*one| one.deinit();
    for (&heads, ranks) |*one, rank| {
        one.* = try shareHead(arena, gpa, rank, rows);
        made += 1;
    }
    return heads;
}

/// Borrow a vocabulary span. Only the model frees its weight and the pool frees the partials.
/// Halving n can change splitK; match a freshly initialized half's plan and row capacity.
pub fn borrowHead(linear: bf16.Linear, first: usize, n: usize, rows: cuda.Scratch.Rows) !weights_file.Linear {
    const sk = bf16.splitK(n, linear.k);
    return .{ .bf16 = .{
        .weight = .{ .view = linear.weight.span(first, n) },
        .part = if (sk > 1) try rows.pool.lend(.part, .f32, &.{ sk, rows.count, n }) else null,
        .scratch = rows.pool,
        .n = n,
        .k = linear.k,
        .sk = sk,
        .max_rows = rows.count,
    } };
}

fn storedHead(m: model_file.Model) !bf16.Linear {
    std.debug.assert(m.head.count == 1);
    return switch (m.head.runs[0].linear) {
        .bf16 => |one| one,
        else => error.UnsupportedShape,
    };
}

/// The main model the drafter reads, and the halves of its vocabulary it proposes by.
pub fn Of(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        /// One-rank target; over a held pair (`held`), its first rank.
        m: model_file.Model,
        /// The one-rank target's buffers; over a held pair, its first rank's.
        main: *forward_file.Buffers,
        /// Held ranks; null on the one-rank path.
        held: ?H.Model = null,
        /// Held taps; null on the one-rank path.
        held_main: ?*H.Buffers = null,
        /// Borrowed vocabulary spans; their weights and scratch outlive this target.
        borrowed: [held]weights_file.Linear,

        /// The one-rank target `m` and its buffers.
        pub fn one(gpa: Allocator, m: model_file.Model, main: *forward_file.Buffers) !Self {
            comptime std.debug.assert(held == world);
            _ = gpa;
            std.debug.assert(m.shape.vocab % world == 0);
            const linear = try storedHead(m);
            const half = m.shape.vocab / world;
            var borrowed: [held]weights_file.Linear = undefined;
            for (&borrowed, 0..) |*head, rank| head.* = try borrowHead(linear, rank * half, half, .{ .pool = m.scratch, .count = shared_pass.batch });
            return .{ .m = m, .main = main, .borrowed = borrowed };
        }

        /// Over the ranks a process holds: each rank's own embedding, its already-halved head, and its taps.
        pub fn heldBy(gpa: Allocator, m: H.Model, main: *H.Buffers) !Self {
            comptime std.debug.assert(held == 1 or held == world);
            _ = gpa;
            var borrowed: [held]weights_file.Linear = undefined;
            for (&borrowed, m.ranks) |*head, rank| {
                const linear = try storedHead(rank);
                head.* = try borrowHead(linear, 0, linear.n, .{ .pool = rank.scratch, .count = shared_pass.batch });
            }
            return .{ .m = m.ranks[0], .main = &main.ranks[0], .held = m, .held_main = main, .borrowed = borrowed };
        }

        pub fn deinit(self: *Self) void {
            _ = self;
        }

        pub fn tapLayers(self: Self) []const u32 {
            return self.main.tap_layers;
        }

        /// Each rank's embedding, `heads` or its borrowed half, and the rows a pick is scored against again.
        pub fn mains(self: Self, heads: ?[held]weights_file.Linear, widened: ?[held]cuda.View) [held]shared_pass.Main {
            var out: [held]shared_pass.Main = undefined;
            for (&out, 0..) |*slot, r| {
                const head = if (heads) |all| all[r] else self.borrowed[r];
                const rows = if (widened) |wide| wide[r] else self.borrowed[r].bf16.weight.view;
                const embed = if (self.held) |held_m| held_m.ranks[r].embed.view else self.m.embed.view;
                slot.* = .{ .embed = embed, .head = head, .rows = rows };
            }
            return out;
        }

        /// The last window's taps on each held rank, or the one-rank target's copy on both.
        pub fn taps(self: Self) [held]cuda.Buffer {
            var out: [held]cuda.Buffer = undefined;
            if (self.held_main) |b| {
                for (&out, b.ranks) |*slot, rank| slot.* = rank.taps.?;
            } else {
                const tapped = self.main.taps.?;
                for (&out) |*slot| slot.* = tapped;
            }
            return out;
        }
    };
}
