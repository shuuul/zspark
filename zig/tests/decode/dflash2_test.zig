//! `decode/dflash2/`'s tests: the stored window versus GLM's exclusive one, and the ring wrap.
//! Capture of taps is not here.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../src/decode/dflash2/config.zig");
const ring = @import("../../src/decode/dflash2/ring.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;

/// A two-layer drafter of the published shapes, sliding_window 97 / 2048 as the cases ask.
const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","hidden_size":512,"vocab_size":128,
    \\"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":128,
    \\"intermediate_size":256,"rms_norm_eps":1e-05,"rope_parameters":{"rope_theta":10000.0},
    \\"sliding_window":97,"is_causal":false,"num_target_layers":4,
    \\"dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":120,
    \\"selector_rank":64,"selector_top_k":8,"target_layer_ids":[1,3]}}
;

test "exclusive window is sliding_window less one; stored keeps the written value" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const exclusive = try Config.parse(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(?usize, 96), exclusive.window);
    const stored = try Config.parseStored(arena.allocator(), fixture, &diagnostic);
    try testing.expectEqual(@as(?usize, 97), stored.window);
    try testing.expectEqual(exclusive.hidden, stored.hidden);
    try testing.expectEqual(exclusive.block, stored.block);
}

test "Qwen's stored 2048 and GLM's exclusive 2047 both ring to 2176 rows" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const text = try std.mem.replaceOwned(u8, arena.allocator(), fixture, "\"sliding_window\":97", "\"sliding_window\":2048");
    var glm = try Config.parse(arena.allocator(), text, &diagnostic);
    glm.block = 8;
    try testing.expectEqual(@as(?usize, 2047), glm.window);
    try testing.expectEqual(@as(usize, 2176), glm.ringRows());
    var qwen = try Config.parseStored(arena.allocator(), text, &diagnostic);
    qwen.block = 8;
    try testing.expectEqual(@as(?usize, 2048), qwen.window);
    try testing.expectEqual(@as(usize, 2176), qwen.ringRows());
}

test "the ring wraps in two runs; a flat context is one run from the start" {
    const wrapped = ring.wrap(8, true, 10, 5);
    try testing.expectEqual(@as(usize, 5), wrapped.slot);
    try testing.expectEqual(@as(usize, 3), wrapped.run);
    const one = ring.wrap(8, true, 6, 4);
    try testing.expectEqual(@as(usize, 2), one.slot);
    try testing.expectEqual(@as(usize, 4), one.run);
    const flat = ring.wrap(16, false, 11, 5);
    try testing.expectEqual(@as(usize, 6), flat.slot);
    try testing.expectEqual(@as(usize, 5), flat.run);
}

test "shape rings when the window's rows are smaller than capacity plus a block" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try Config.parse(arena.allocator(), fixture, &diagnostic);
    // ringRows is 192; a short capacity does not ring (192 is not < 64 + 4)
    const small = ring.shape(&c, 64, true);
    try testing.expect(!small.ring);
    try testing.expectEqual(@as(usize, 64 + 4), small.cap);
    const large = ring.shape(&c, 4096, true);
    try testing.expect(large.ring);
    try testing.expectEqual(c.ringRows(), large.cap);
    const asked_off = ring.shape(&c, 4096, false);
    try testing.expect(!asked_off.ring);
    try testing.expectEqual(@as(usize, 4096 + 4), asked_off.cap);
}
