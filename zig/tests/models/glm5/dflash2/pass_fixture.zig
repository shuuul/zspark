//! What `pass_test.zig` and `pass_checkpoint_test.zig` both call: a bf16 value, and a pair's drafts read back.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const glm5_testing = @import("../testing.zig");
const pass = @import("../../../../src/models/glm5/dflash2/pass.zig");

pub fn value(bits: u16) f32 {
    return numeric.f32FromBf16(bits);
}

pub fn drafts(arena: Allocator, pair: pass.Pair) ![]u16 {
    var all: std.ArrayList(u16) = .empty;
    for (pair.ranks) |rank| for ([_]cuda.Buffer{ rank.logits, rank.selected, rank.x }) |buffer| try all.appendSlice(arena, try glm5_testing.fetch(u16, arena, buffer));
    return all.items;
}
