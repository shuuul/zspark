//! `decode/dflash2/config.zig`'s tests: Qwen's stored window, published sizes, and rising taps.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../../src/decode/dflash2/config.zig");
const rig = @import("../../../../src/models/qwen4/dflash2/rig.zig");
const drafter_testing = @import("../../../../src/models/qwen4/checks/random_drafter.zig");
const taps = @import("../../../../src/models/qwen4/taps.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;

const fixture = drafter_testing.fixture;

fn expectRefused(expected: Error, from: []const u8, to: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fixture, from));
    const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture, from, to);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, Config.parseStored(arena.allocator(), text, &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message, needle) == null) {
        std.debug.print("\"{s}\" is not in \"{s}\"\n", .{ needle, diagnostic.message });
        return error.TestExpectedEqual;
    }
}

test "a Qwen drafter's config: stored window, a rank's halves, the taps' width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parseStored(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(usize, 256), c.hidden);
    try testing.expectEqual(@as(?usize, 64), c.window);
    try testing.expect(!c.causal);
    try testing.expectEqualSlices(u32, &.{ 0, 2 }, c.taps);
    try testing.expectEqual(@as(usize, 512), c.tapWidth());
    try testing.expectEqual(@as(u32, 317), c.mask);
    try testing.expectEqual(@as(?[]const u8, null), try c.mismatch(arena.allocator(), 256, 320, 3));
    try testing.expect(std.mem.indexOf(u8, (try c.mismatch(arena.allocator(), 256, 320, 4)).?, "3 layers") != null);
}

test "the published Flash Next DFlash2 sizes: hidden 2560, taps 5/15/25/35/45, stored window 2048" {
    const p = rig.published;
    try testing.expectEqual(@as(usize, 2560), p.hidden);
    try testing.expectEqual(@as(usize, 5), p.layers);
    try testing.expectEqual(@as(usize, 32), p.heads);
    try testing.expectEqual(@as(usize, 8), p.kv_heads);
    try testing.expectEqual(@as(usize, 128), p.head_dim);
    try testing.expectEqual(@as(usize, 7680), p.intermediate);
    try testing.expectEqual(@as(?usize, 2048), p.window);
    try testing.expectEqual(@as(usize, 8), p.block);
    try testing.expectEqual(@as(usize, 16), p.group);
    try testing.expectEqual(@as(usize, 256), p.selector_rank);
    try testing.expectEqual(@as(usize, 16), p.top_k);
    try testing.expectEqual(@as(usize, 248320), p.vocab);
    try testing.expectEqual(@as(u32, 248077), p.mask);
    try testing.expectEqualSlices(u32, &p.taps, &taps.layers);
    try testing.expectEqual(@as(usize, 48), p.target_layers);
}

test "asymmetric taps: swapping 5/15 is refused as not rising" {
    try expectRefused(error.InvalidConfig, "[0,2]", "[2,0]", "rising layers");
    try expectRefused(error.InvalidConfig, "[0,2]", "[15,5]", "rising layers");
}
test "what a Qwen drafter's config is refused for" {
    try expectRefused(error.UnsupportedModel, "DFlash2DraftModel", "Qwen3ForCausalLM", "not a DFlash2 drafter");
    try expectRefused(error.UnsupportedModel, "\"conv_kernel_size\":2", "\"conv_kernel_size\":3", "two taps");
    try expectRefused(error.UnsupportedModel, "\"head_dim\":128", "\"head_dim\":64", "heads of 128");
    try expectRefused(error.InvalidConfig, "\"mask_token_id\":317", "\"mask_token_id\":320", "not in the vocabulary");
}
