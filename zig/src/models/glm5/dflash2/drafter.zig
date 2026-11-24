//! GLM-5.3-Flash's DFlash2 drafter: `decode/dflash2/drafter.zig` over GLM's held pair (`Of(held)`), the
//! drafter's target: each rank's taps of its own buffers (`Buffers.setTaps` with the config's), its embedding
//! and its rows of the vocabulary.

const cuda = @import("../../../cuda/cuda.zig");
const shared = @import("../../../decode/dflash2/drafter.zig");
const shared_pass = @import("../../../decode/dflash2/pass.zig");
const weights_file = @import("../../../decode/dflash2/weights.zig");
const held_file = @import("../layers/ranks_forward.zig");
const pass = @import("pass.zig");

pub fn DrafterOf(comptime held: usize) type {
    return shared.DrafterOf(held, pass.Target);
}

/// Both ranks in one process.
pub const Drafter = DrafterOf(shared_pass.world);

/// The main model the drafter reads: its ranks (`m`) and their buffers, whose taps the kept rows' are.
pub fn Of(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        m: H.Model,
        main: *H.Buffers,

        pub fn tapLayers(self: Self) []const u32 {
            return self.main.ranks[0].tap_layers;
        }

        /// Each rank's embedding, `heads` or its own head, and the rows a pick is scored against again.
        pub fn mains(self: Self, heads: ?[held]weights_file.Linear, widened: ?[held]cuda.View) [held]shared_pass.Main {
            var out: [held]shared_pass.Main = undefined;
            for (&out, self.m.ranks, 0..) |*one, rank, r|
                one.* = .{ .embed = rank.embed.view, .head = if (heads) |all| all[r] else rank.head, .rows = if (widened) |rows| rows[r] else rank.head.bf16.weight.view };
            return out;
        }

        pub fn taps(self: Self) [held]cuda.Buffer {
            var out: [held]cuda.Buffer = undefined;
            for (&out, self.main.ranks) |*one, rank| one.* = rank.taps.?;
            return out;
        }

        /// Nothing: the main model is the session's.
        pub fn deinit(self: *Self) void {
            _ = self;
        }
    };
}
