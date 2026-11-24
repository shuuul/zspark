//! `config.zig`'s tests.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/decode/dflash2/config.zig");
const drafter_testing = @import("../../../src/models/glm5/checks/random_drafter.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;
const Error = config_file.Error;

const fixture = drafter_testing.fixture;

/// The fixture with `from` (once in it) replaced by `to` is refused with `expected` and a message holding `needle`.
fn expectRefused(expected: Error, from: []const u8, to: []const u8, needle: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fixture, from));
    const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture, from, to);
    var diagnostic: Diagnostic = .{};
    try testing.expectError(expected, Config.parse(arena.allocator(), text, &diagnostic));
    if (std.mem.indexOf(u8, diagnostic.message, needle) == null) {
        std.debug.print("\"{s}\" is not in \"{s}\"\n", .{ needle, diagnostic.message });
        return error.TestExpectedEqual;
    }
}

test "a drafter's config: a rank's halves, the taps' width, the kernel's and the ring" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parse(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(usize, 512), c.hidden);
    try testing.expectEqual(@as(usize, 2), c.rankHeads());
    try testing.expectEqual(@as(usize, 1), c.rankKvHeads());
    try testing.expectEqual(@as(usize, 128), c.rankWidth());
    try testing.expectEqual(@as(?usize, 96), c.window);
    try testing.expect(!c.causal);
    try testing.expectEqualSlices(u32, &.{ 1, 3 }, c.taps);
    try testing.expectEqual(@as(usize, 1024), c.tapWidth());
    try testing.expectEqual(@as(usize, 128), c.kernelWidth());
    try testing.expectEqual(@as(u32, 120), c.mask);
    // 96 + 4 + 64 rows in whole tiles of 64
    try testing.expectEqual(@as(usize, 192), c.ringRows());
    try testing.expectEqual(@as(u64, 2 * 2 * 1 * 192 * 128 * 2), c.contextBytes(192));
    try testing.expectEqual(@as(?[]const u8, null), try c.mismatch(arena.allocator(), 512, 128, 4));
    try testing.expect(std.mem.indexOf(u8, (try c.mismatch(arena.allocator(), 512, 128, 5)).?, "4 layers") != null);
}

test "the published drafter's ring is 2176 rows: TensorFold's draft_ring_rows(2047, 8)" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    var c = try Config.parse(arena.allocator(), fixture, &diagnostic);
    c.window = 2047;
    c.block = 8;
    try testing.expectEqual(@as(usize, 2176), c.ringRows());
}

test "what a drafter's config is refused for" {
    try expectRefused(error.UnsupportedModel, "DFlash2DraftModel", "Qwen3ForCausalLM", "not a DFlash2 drafter");
    try expectRefused(error.UnsupportedModel, "\"conv_kernel_size\":2", "\"conv_kernel_size\":3", "two taps");
    try expectRefused(error.UnsupportedModel, "\"head_dim\":128", "\"head_dim\":64", "heads of 128");
    try expectRefused(error.UnsupportedModel, "\"is_causal\":false", "\"is_causal\":true", "both ways");
    try expectRefused(error.UnsupportedModel, "\"is_causal\":false,", "", "both ways");
    try expectRefused(error.InvalidConfig, "\"num_key_value_heads\":2", "\"num_key_value_heads\":1", "each take half");
    try expectRefused(error.InvalidConfig, "\"intermediate_size\":256", "\"intermediate_size\":255", "each take half");
    try expectRefused(error.InvalidConfig, "\"num_attention_heads\":4", "\"num_attention_heads\":6", "hidden 512");
    try expectRefused(error.InvalidConfig, "\"conv_group_size\":16", "\"conv_group_size\":24", "does not divide");
    try expectRefused(error.InvalidConfig, "\"block_size\":4", "\"block_size\":1", "block_size is 1");
    try expectRefused(error.InvalidConfig, "[1,3]", "[3,1]", "rising layers");
    try expectRefused(error.InvalidConfig, "[1,3]", "[1,4]", "rising layers");
    try expectRefused(error.InvalidConfig, "\"mask_token_id\":120", "\"mask_token_id\":128", "not in the vocabulary");
    try expectRefused(error.InvalidConfig, "\"sliding_window\":97,", "", "no sliding_window");
    try expectRefused(error.InvalidConfig, "\"rope_theta\":10000.0", "\"theta\":1", "no rope_theta");
}
