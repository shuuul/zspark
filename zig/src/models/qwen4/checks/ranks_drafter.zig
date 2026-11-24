//! `ranks-check DIR reference=REF drafter=DIR`'s DFlash2 half (spec 126 WS-01, the owner's 2026-10-24 word):
//! the stored drafter over the ranks a process holds, loaded each rank from the directory both Sparks hold
//! (no copy crosses the link) and given to the reference path's pair — rank 0 drafts by it (`Drafts.given`),
//! rank 1 follows its passes (`Pair.flash`, `mirror.zig`) — as GLM's `glm5/checks/ranks_check_drafter.zig`
//! (spec 045 WS-09) gives GLM's check its random one beside the random model. The drafter only proposes: the
//! target chooses every token, and the check's drafted reply is the serial one, bit for bit.
//!
//! What a reply costs the caller, as the loopback reference does it (`tests/models/qwen4/dflash2/
//! ranks_loopback_test.zig`): `d.reset(settings)` before a reply that starts from an empty pair, the settings
//! the reply's are. The two-process NCCL run is `Given(1)`; the loopback test's `Given(2)` is its reference.

const std = @import("std");
const Allocator = std.mem.Allocator;
const pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const rig = @import("../dflash2/rig.zig");
const pair_file = @import("../session/pair.zig");

pub fn Given(comptime held: usize) type {
    return struct {
        const Self = @This();

        arena_state: std.heap.ArenaAllocator,
        source: rig.Source,
        flash: rig.Rig(held),
        /// The pair the drafter was given to; its `flash` is cleared before the drafter is freed.
        pair: *pair_file.Of(held),

        /// In place over `pair`, which must not have run a window yet: the drafter of `opened` loaded (each
        /// rank its own share, stored BF16 as the file has it, or `.mixed` with the four-bit head where not
        /// `stored`), `initHeld` beside the pair's model and buffers,
        /// and given to the pair — `pair.flash`, so rank 1 replays the passes rank 0 tells. `eager`: the
        /// drafter's passes are launched one by one, as an eager pair's are; `serve` asks it, its pair's
        /// windows captured: captured at two ranks the passes took nothing off a round (spec 135 WS-01).
        pub fn init(gpa: Allocator, opened: *const rig.Opened, pair: *pair_file.Of(held), stored: bool, eager: bool) !*Self {
            comptime std.debug.assert(held == 1 or held == dflash2_pass.world);
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.pair = pair;
            self.arena_state = .init(gpa);
            errdefer self.arena_state.deinit();
            self.source = .{ .checkpoint = &opened.checkpoint.? };
            const ranks: [held]usize = if (held == 1) .{pair.m.first} else .{ 0, 1 };
            self.flash = undefined;
            try self.flash.load(self.arena_state.allocator(), gpa, opened, &self.source, ranks, pair.m.ranks[0].scratch, stored, null);
            errdefer self.flash.unload();
            const heads = try rig.draftHeadsHeld(held, self.arena_state.allocator(), gpa, pair.m, stored, .{ .pool = pair.m.ranks[0].scratch, .count = dflash2_pass.batch });
            try self.flash.initHeld(gpa, opened, heads, null, pair.m, &pair.b, pair.capacity, .{ .seed = 0 }, eager or pair.graphs == null);
            pair.flash = &self.flash.d;
            return self;
        }

        pub fn deinit(self: *Self, gpa: Allocator) void {
            self.pair.flash = null;
            self.flash.deinit();
            self.arena_state.deinit();
            gpa.destroy(self);
        }
    };
}
