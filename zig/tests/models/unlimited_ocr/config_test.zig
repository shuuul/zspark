//! `config.zig`'s tests on Unlimited-OCR-2's own config.json (`fixtures/unlimited_ocr/config_dash.json`, spec
//! 147 WS-02), the refusal of a config without `sparse_attention_config` (the retired v1's, spec 170) and of what
//! DASH's config may say that zspark does not read.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/models/unlimited_ocr/config.zig");
const Config = config_file.Config;
const Diagnostic = config_file.Diagnostic;

const fx = struct {
    const dash = @embedFile("fixtures/unlimited_ocr/config_dash.json");
};

fn parse(arena: std.mem.Allocator, text: []const u8, diagnostic: *Diagnostic) !Config {
    return Config.parse(arena, text, diagnostic);
}

/// v2's config with `from` (which must occur once) replaced by `to`.
fn edited(arena: std.mem.Allocator, from: []const u8, to: []const u8) ![]const u8 {
    try testing.expectEqual(@as(usize, 1), std.mem.count(u8, fx.dash, from));
    return std.mem.replaceOwned(u8, arena, fx.dash, from, to);
}

test "unlimited_ocr config: a config without sparse_attention_config is the retired v1's and is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var diagnostic: Diagnostic = .{};
    // v1's whole object removed, or null, or not an object
    const missing = parse(a, "{\"model_type\": \"unlimited-ocr\", \"language_config\": {}}", &diagnostic);
    try testing.expectError(error.UnsupportedModel, missing);
    const nulled = try std.mem.replaceOwned(u8, a, fx.dash, "\"sparse_attention_config\": {", "\"sparse_attention_config\": null, \"x\": {");
    try testing.expectError(error.UnsupportedModel, parse(a, nulled, &diagnostic));
    const number = try std.mem.replaceOwned(u8, a, fx.dash, "\"sparse_attention_config\": {", "\"sparse_attention_config\": 1, \"x\": {");
    try testing.expectError(error.InvalidConfig, parse(a, number, &diagnostic));
}

test "unlimited_ocr config: v2 is told by sparse_attention_config and read whole" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: Diagnostic = .{};
    const c = try parse(arena.allocator(), fx.dash, &diagnostic);
    try testing.expectEqual(@as(usize, 12), c.layers);
    try testing.expectEqual(@as(usize, 10), c.heads);
    try testing.expectEqual(@as(usize, 64), c.experts);
    try testing.expectEqual(@as(usize, 131072), c.max_positions);
    const d = c.dash;
    try testing.expectEqual(@as(usize, 32), d.block);
    try testing.expectEqual(@as(usize, 4), d.local_blocks);
    try testing.expectEqual(@as(usize, 12), d.read_blocks);
    try testing.expectEqual(@as(usize, 16), d.candidates);
    try testing.expectEqual(@as(usize, 128), d.index_dim);
    try testing.expectEqual(@as(usize, 12), d.scale.len);
    for (d.scale, d.floor) |scale, floor| {
        try testing.expectEqual(@as(f64, 128.0), scale);
        try testing.expectEqual(@as(f64, 1e-4), floor);
    }
    try testing.expectEqual(@as(f64, 1.0), d.sigma);
    try testing.expectEqual(config_file.Dash.Stage0.head_cls, d.stage0);
    try testing.expectEqual(config_file.Dash.Routing.entmax_topk_perlayer, d.routing);
    try testing.expect(d.image_only and d.image_buckets);
    try testing.expectEqual(@as(u32, 128815), d.image_token);
    try testing.expectEqual(@as(u32, 128827), d.page_token);
    try testing.expectEqual(@as(usize, 273), d.page_image_tokens);
    try testing.expectEqual(@as(f64, 10000.0), d.rope.tok_theta);
    try testing.expectEqual(@as(f64, 256.0), d.rope.page_theta);
    try testing.expectEqual(@as(usize, 1), d.rope.page_pairs);
    try testing.expectEqual(@as(usize, 2), d.rope.page_mod);
    try testing.expectEqual(config_file.Page2d.TokFreq.resample, d.rope.tok_freq);
    // without the per-layer lists, the scalars (HF's `_per_layer_val`)
    const scalars = try parse(arena.allocator(), try edited(arena.allocator(), "\"dash_scale_per_layer\"", "\"x_dash_scale_per_layer\""), &diagnostic);
    for (scalars.dash.scale) |scale| try testing.expectEqual(@as(f64, 128.0), scale);
}

test "unlimited_ocr config: what DASH's config may say that zspark does not read is refused" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]struct { from: []const u8, to: []const u8, err: anyerror, says: []const u8 }{
        .{ .from = "\"dash_stage0\": \"head_cls\"", .to = "\"dash_stage0\": \"indexer\"", .err = error.UnsupportedModel, .says = "dash_stage0 \"indexer\"" },
        .{ .from = "\"dash_router\": \"entmax_topk_perlayer\"", .to = "\"dash_router\": \"softmax_topk\"", .err = error.UnsupportedModel, .says = "dash_router" },
        .{ .from = "\"sparse_main_rope_mode\": \"page2d\"", .to = "\"sparse_main_rope_mode\": \"1d\"", .err = error.UnsupportedModel, .says = "sparse_main_rope_mode \"1d\"" },
        .{ .from = "\"sparse_rope_mode\": \"page2d\"", .to = "\"sparse_rope_mode\": \"1d\"", .err = error.UnsupportedModel, .says = "sparse_rope_mode \"1d\"" },
        .{ .from = "\"sparse_rope_tok_freq_mode\": \"resample\"", .to = "\"sparse_rope_tok_freq_mode\": \"truncate\"", .err = error.UnsupportedModel, .says = "sparse_rope_tok_freq_mode" },
        .{ .from = "\"sparse_main_rope_tok_freq_mode\": \"resample\"", .to = "\"x\": 0", .err = error.UnsupportedModel, .says = "sparse_main_rope_tok_freq_mode \"truncate\"" },
        .{ .from = "\"sparse_rope_tok_theta\": 10000.0", .to = "\"x\": 0", .err = error.UnsupportedModel, .says = "rope_theta" },
        .{ .from = "\"sparse_rope_page_pairs\": 1", .to = "\"sparse_rope_page_pairs\": 64", .err = error.UnsupportedModel, .says = "page_pairs 64" },
        .{ .from = "\"training_stage\": \"dash\"", .to = "\"training_stage\": \"sparse\"", .err = error.UnsupportedModel, .says = "training_stage \"sparse\"" },
        .{ .from = "\"sparse_attention_freq\": [\n      1,", .to = "\"sparse_attention_freq\": [\n      0,", .err = error.UnsupportedModel, .says = "sparse_attention_freq[0]" },
        .{ .from = "\"sparse_num_index_heads\": 1", .to = "\"sparse_num_index_heads\": 2", .err = error.UnsupportedModel, .says = "sparse_num_index_heads" },
        .{ .from = "\"dash_topk_blocks\": 12", .to = "\"dash_topk_blocks\": 17", .err = error.InvalidConfig, .says = "17 read of 16" },
        .{ .from = "\"sparse_index_dim\": 128", .to = "\"sparse_index_dim\": 64", .err = error.UnsupportedModel, .says = "sparse_index_dim 64" },
    };
    for (cases) |case| {
        var diagnostic: Diagnostic = .{};
        const text = try edited(a, case.from, case.to);
        try testing.expectError(case.err, parse(a, text, &diagnostic));
        testing.expect(std.mem.indexOf(u8, diagnostic.message, case.says) != null) catch |err| {
            std.debug.print("{s} -> {s}: \"{s}\"\n", .{ case.from, case.to, diagnostic.message });
            return err;
        };
    }
}
