//! `modelopt.zig`'s tests: a map's entries as ModelOpt writes them, the entry covering a module, each storage
//! read as stored with its refusals, and a rank's cut of each: both ranks' parts joined against the whole, the
//! csf_32x2_v1 window's scales against the whole plane flashinfer's encoder made (`csf_blocks_cases.bin`).

const std = @import("std");
const json = std.json;
const modelopt = @import("../../src/weights/modelopt.zig");
const safetensors = @import("../../src/weights/safetensors.zig");
const csf_blocks = @import("../../src/weights/csf_blocks.zig");
const csf_blocks_test = @import("csf_blocks_test.zig");
const Tensor = safetensors.Tensor;
const Split = @import("../../src/weights/share.zig").Split;

const Source = std.StringHashMapUnmanaged(Tensor);

const Diagnostic = struct {
    message: []const u8 = "",

    pub fn set(self: *Diagnostic, arena: std.mem.Allocator, comptime format: []const u8, args: anytype) void {
        self.message = std.fmt.allocPrint(arena, format, args) catch "out of memory";
    }
};

fn entryOf(arena: std.mem.Allocator, text: []const u8) !modelopt.QuantizedLayer {
    const value = try json.parseFromSliceLeaky(json.Value, arena, text, .{});
    return modelopt.entry("m", value.object);
}

test "a map entry: group_size or weight_block_size, the csf encodings, and what is no entry" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const block = try entryOf(arena, "{\"quant_algo\":\"FP8_PB_WO\",\"weight_block_size\":[128,128]}");
    try std.testing.expectEqual(modelopt.Algo.fp8_block, block.algo);
    try std.testing.expectEqual(@as(?u32, 128), block.group_size);
    const mx = try entryOf(arena, "{\"quant_algo\":\"MXFP8\",\"weight_block_size\":[1,32]}");
    try std.testing.expectEqual(modelopt.Algo.mxfp8, mx.algo);
    try std.testing.expectEqual(@as(?u32, 32), mx.group_size);
    const experts = try entryOf(arena, "{\"group_size\":16,\"quant_algo\":\"NVFP4\",\"weight_scale_encoding\":\"csf_32x2_v1\"}");
    try std.testing.expectEqual(modelopt.ScaleEncoding.csf32, experts.scale_encoding);
    try std.testing.expectEqual(modelopt.Scheme.nvfp4, modelopt.Scheme.of(experts).?);
    const a16 = try entryOf(arena, "{\"group_size\":16,\"quant_algo\":\"w4a16_nvfp4\"}");
    try std.testing.expectEqual(modelopt.Algo.nvfp4, a16.algo);
    try std.testing.expectEqual(modelopt.ScaleEncoding.plain, a16.scale_encoding);
    // W4A16 keeps its activations 16-bit; NVFP4's are quantized where a model runs them so
    try std.testing.expect(a16.a16 and !experts.a16);
    try std.testing.expectEqual(modelopt.ScaleEncoding.csf, (try entryOf(arena, "{\"quant_algo\":\"NVFP4\",\"weight_scale_encoding\":\"csf\"}")).scale_encoding);
    try std.testing.expectEqual(@as(?u32, null), (try entryOf(arena, "{\"quant_algo\":\"MXFP8\"}")).group_size);
    try std.testing.expectError(error.UnknownAlgo, entryOf(arena, "{\"quant_algo\":\"INT4_AWQ\"}"));
    try std.testing.expectError(error.UnknownEncoding, entryOf(arena, "{\"quant_algo\":\"NVFP4\",\"weight_scale_encoding\":\"zip\"}"));
    try std.testing.expectError(error.InvalidEntry, entryOf(arena, "{\"group_size\":16}"));
    try std.testing.expectError(error.InvalidEntry, entryOf(arena, "{\"quant_algo\":\"MXFP8\",\"weight_block_size\":[32]}"));
    // the storages a linear is read in, and those it is not
    try std.testing.expectEqual(modelopt.Scheme.bf16, modelopt.Scheme.of(null).?);
    try std.testing.expectEqual(modelopt.Scheme.fp8, modelopt.Scheme.of(try entryOf(arena, "{\"quant_algo\":\"FP8\"}")).?);
    try std.testing.expectEqual(@as(?modelopt.Scheme, null), modelopt.Scheme.of(try entryOf(arena, "{\"quant_algo\":\"FP8\",\"group_size\":16}")));
    try std.testing.expectEqual(@as(?modelopt.Scheme, null), modelopt.Scheme.of(try entryOf(arena, "{\"quant_algo\":\"FP8_PB_WO\",\"group_size\":64}")));
    try std.testing.expectEqual(@as(?modelopt.Scheme, null), modelopt.Scheme.of(try entryOf(arena, "{\"quant_algo\":\"NVFP4\",\"weight_scale_encoding\":\"csf\"}")));
}

test "the entry covering a module: its own name or one beneath it, at a dot" {
    const layers = [_]modelopt.QuantizedLayer{
        .{ .name = "model.language_model.layers.4.mlp.experts", .algo = .nvfp4, .group_size = 16 },
        .{ .name = "model.language_model.layers.45.mlp.experts", .algo = .nvfp4, .group_size = 16 },
        .{ .name = "model.language_model.layers.4.mlp.shared_experts.gate_proj", .algo = .fp8_block, .group_size = 128 },
    };
    try std.testing.expectEqual(@as(?usize, 0), modelopt.findIndex(&layers, "model.language_model.layers.4.mlp.experts.7.up_proj"));
    try std.testing.expectEqual(@as(?usize, 1), modelopt.findIndex(&layers, "model.language_model.layers.45.mlp.experts.0.down_proj"));
    try std.testing.expectEqual(@as(?usize, 2), modelopt.findIndex(&layers, "model.language_model.layers.4.mlp.shared_experts.gate_proj"));
    try std.testing.expectEqual(@as(?usize, null), modelopt.findIndex(&layers, "model.language_model.layers.4.mlp.gate"));
    try std.testing.expectEqual(@as(?usize, null), modelopt.findIndex(&layers, "model.language_model.layers.4.mlp.experts_extra"));
    try std.testing.expectEqual(modelopt.Algo.fp8_block, modelopt.find(&layers, "model.language_model.layers.4.mlp.shared_experts.gate_proj").?.algo);
}

fn put(arena: std.mem.Allocator, source: *Source, random: std.Random, name: []const u8, dtype: safetensors.DType, shape: []const u64) !void {
    var tensor: Tensor = .{ .dtype = dtype, .shape = try arena.dupe(u64, shape), .data = &.{} };
    const data = try arena.alloc(u8, @intCast(tensor.count() * dtype.size()));
    random.bytes(data);
    tensor.data = data;
    try source.put(arena, name, tensor);
}

/// A linear of each storage, [rows, cols], under `module`. `four` stores an FP8_PB_WO scale 4-D.
fn putLinear(arena: std.mem.Allocator, source: *Source, random: std.Random, module: []const u8, scheme: modelopt.Scheme, rows: u64, cols: u64, four: bool) !void {
    const name = struct {
        fn of(a: std.mem.Allocator, m: []const u8, leaf: []const u8) ![]const u8 {
            return std.fmt.allocPrint(a, "{s}.{s}", .{ m, leaf });
        }
    }.of;
    switch (scheme) {
        .bf16 => try put(arena, source, random, try name(arena, module, "weight"), .BF16, &.{ rows, cols }),
        .fp8 => {
            try put(arena, source, random, try name(arena, module, "weight"), .F8_E4M3, &.{ rows, cols });
            try put(arena, source, random, try name(arena, module, "weight_scale"), .F32, &.{});
            try put(arena, source, random, try name(arena, module, "input_scale"), .F32, &.{});
        },
        .mxfp8 => {
            try put(arena, source, random, try name(arena, module, "weight"), .F8_E4M3, &.{ rows, cols });
            try put(arena, source, random, try name(arena, module, "weight_scale"), .U8, &.{ rows, cols / 32 });
        },
        .fp8_block => {
            try put(arena, source, random, try name(arena, module, "weight"), .F8_E4M3, &.{ rows, cols });
            try put(arena, source, random, try name(arena, module, "weight_scale"), .F32, if (four) &.{ rows / 128, 1, cols / 128, 1 } else &.{ rows / 128, cols / 128 });
        },
        .nvfp4 => {
            try put(arena, source, random, try name(arena, module, "weight"), .U8, &.{ rows, cols / 2 });
            try put(arena, source, random, try name(arena, module, "weight_scale"), .F8_E4M3, &.{ rows, cols / 16 });
            try put(arena, source, random, try name(arena, module, "weight_scale_2"), .F32, &.{});
        },
    }
}

/// Both ranks' halves of `whole` by `split` joined: the rows one after the other, or each row's two halves.
pub fn expectJoined(whole: Tensor, halves: [2]Tensor, split: Split) !void {
    if (split == .row) {
        try std.testing.expectEqual(whole.data.len, halves[0].data.len + halves[1].data.len);
        try std.testing.expectEqualSlices(u8, whole.data[0..halves[0].data.len], halves[0].data);
        try std.testing.expectEqualSlices(u8, whole.data[halves[0].data.len..], halves[1].data);
        return;
    }
    const outer: usize = @intCast(whole.shape[0]);
    const inner = whole.data.len / outer;
    for (0..outer) |o| for (halves, 0..) |half, rank| {
        try std.testing.expectEqualSlices(u8, whole.data[o * inner + rank * inner / 2 ..][0 .. inner / 2], half.data[o * inner / 2 ..][0 .. inner / 2]);
    };
}

/// Both ranks' cuts of `whole` by `split` (`row` or `col`) joined against it: the weight, a scale, the block scales
/// csf windows decode; `weight_scale_2` and the streams the same bytes on both.
pub fn expectHalves(arena: std.mem.Allocator, whole: modelopt.Linear, halves: [2]modelopt.Linear, split: Split) !void {
    const by_rows = split == .row or split == .vocab;
    for (halves) |half| {
        try std.testing.expectEqual(if (by_rows) whole.rows / 2 else whole.rows, half.rows);
        try std.testing.expectEqual(if (by_rows) whole.cols else whole.cols / 2, half.cols);
    }
    const as: Split = if (by_rows) .row else .col;
    try expectJoined(whole.weight, .{ halves[0].weight, halves[1].weight }, as);
    if (whole.scale) |scale| try expectJoined(scale, .{ halves[0].scale.?, halves[1].scale.? }, as);
    if (whole.scale_2) |scale_2| for (halves) |half| try std.testing.expectEqual(scale_2.data.ptr, half.scale_2.?.data.ptr);
    if (whole.csf != null) {
        const shape: []const u64 = &.{ whole.rows, whole.cols / 16 };
        const all: Tensor = .{ .dtype = .F8_E4M3, .shape = shape, .data = try whole.blockScales(arena) };
        var parts: [2]Tensor = undefined;
        for (&parts, halves) |*part, half| {
            try std.testing.expectEqual(whole.csf.?.streams.codes.ptr, half.csf.?.streams.codes.ptr);
            part.* = .{ .dtype = .F8_E4M3, .shape = &.{ half.rows, half.cols / 16 }, .data = try half.blockScales(arena) };
        }
        try expectJoined(all, parts, as);
    }
}

test "each storage read as stored: its parts' dtypes and shapes, FP8_PB_WO's 4-D scale as 2-D, and the refusals" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(107);
    var source: Source = .empty;
    var diagnostic: Diagnostic = .{};
    const mx: modelopt.QuantizedLayer = .{ .name = "mx", .algo = .mxfp8, .group_size = 32 };
    const block: modelopt.QuantizedLayer = .{ .name = "fp8", .algo = .fp8_block, .group_size = 128 };
    const plain: modelopt.QuantizedLayer = .{ .name = "fp4", .algo = .nvfp4, .group_size = 16 };
    try putLinear(arena, &source, prng.random(), "bf", .bf16, 64, 96, false);
    try putLinear(arena, &source, prng.random(), "mx", .mxfp8, 64, 128, false);
    try putLinear(arena, &source, prng.random(), "fp8", .fp8_block, 256, 384, true);
    try putLinear(arena, &source, prng.random(), "fp4", .nvfp4, 64, 64, false);

    const bf = try modelopt.read(arena, &source, "bf", null, 64, 96, &diagnostic);
    try std.testing.expectEqual(modelopt.Scheme.bf16, bf.scheme);
    try std.testing.expectEqual(@as(usize, 1), bf.held(.row).count);
    const read_mx = try modelopt.read(arena, &source, "mx", mx, 64, 128, &diagnostic);
    try std.testing.expectEqualSlices(u64, &.{ 64, 4 }, read_mx.scale.?.shape);
    const read_block = try modelopt.read(arena, &source, "fp8", block, 256, 384, &diagnostic);
    try std.testing.expectEqualSlices(u64, &.{ 2, 3 }, read_block.scale.?.shape);
    try std.testing.expectEqual(source.get("fp8.weight_scale").?.data.ptr, read_block.scale.?.data.ptr);
    const read_fp4 = try modelopt.read(arena, &source, "fp4", plain, 64, 64, &diagnostic);
    try std.testing.expectEqual(@as(usize, 3), read_fp4.held(.col).count);
    try std.testing.expectEqual(source.get("fp4.weight_scale").?.data.ptr, (try read_fp4.blockScales(arena)).ptr);

    // the config's shape, not the stored one; a part missing; a scale's dtype; inputs not whole blocks
    try std.testing.expectError(error.UnexpectedTensor, modelopt.read(arena, &source, "bf", null, 96, 64, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message, "bf.weight") != null);
    try std.testing.expectError(error.MissingTensor, modelopt.read(arena, &source, "none", null, 64, 96, &diagnostic));
    try std.testing.expectError(error.UnexpectedTensor, modelopt.read(arena, &source, "fp8", block, 256, 256, &diagnostic));
    try std.testing.expectError(error.UnexpectedTensor, modelopt.read(arena, &source, "mx", mx, 64, 120, &diagnostic));
    const scale = source.get("mx.weight_scale").?;
    try source.put(arena, "mx.weight_scale", .{ .dtype = .F8_E4M3, .shape = scale.shape, .data = scale.data });
    try std.testing.expectError(error.UnexpectedTensor, modelopt.read(arena, &source, "mx", mx, 64, 128, &diagnostic));
    try std.testing.expect(std.mem.indexOf(u8, diagnostic.message, "mx.weight_scale") != null);
    // per-tensor FP8: e4m3 weight and one fp32 scale, its `input_scale` kept and not read
    try putLinear(arena, &source, prng.random(), "p8", .fp8, 64, 128, false);
    const p8 = try modelopt.read(arena, &source, "p8", .{ .name = "p8", .algo = .fp8, .group_size = null }, 64, 128, &diagnostic);
    try std.testing.expectEqual(modelopt.Scheme.fp8, p8.scheme);
    try std.testing.expectEqual(@as(usize, 4), p8.scale_2.?.data.len);
    try std.testing.expectEqual(@as(usize, 4), p8.input_scale.?.data.len);
}

test "a rank's cut of each storage: both ranks' parts joined are the whole, on whole scale blocks" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(5);
    var source: Source = .empty;
    var diagnostic: Diagnostic = .{};
    const cases = [_]struct { []const u8, modelopt.Scheme, ?modelopt.QuantizedLayer, u64, u64 }{
        .{ "bf", .bf16, null, 64, 96 },
        .{ "mx", .mxfp8, .{ .name = "mx", .algo = .mxfp8, .group_size = 32 }, 64, 128 },
        .{ "fp8", .fp8_block, .{ .name = "fp8", .algo = .fp8_block, .group_size = 128 }, 256, 512 },
        .{ "fp4", .nvfp4, .{ .name = "fp4", .algo = .nvfp4, .group_size = 16 }, 64, 64 },
    };
    for (cases) |case| {
        const module, const scheme, const layer, const rows, const cols = case;
        try putLinear(arena, &source, prng.random(), module, scheme, rows, cols, true);
        const whole = try modelopt.read(arena, &source, module, layer, rows, cols, &diagnostic);
        for ([_]Split{ .row, .col }) |split| {
            var halves: [2]modelopt.Linear = undefined;
            for (&halves, 0..) |*half, rank| half.* = try modelopt.cut(arena, whole, split, rank, 2, &diagnostic);
            try std.testing.expectEqual(if (split == .row) rows / 2 else rows, halves[1].rows);
            try std.testing.expectEqual(if (split == .row) cols else cols / 2, halves[1].cols);
            try expectJoined(whole.weight, .{ halves[0].weight, halves[1].weight }, split);
            if (whole.scale) |scale| try expectJoined(scale, .{ halves[0].scale.?, halves[1].scale.? }, split);
            if (whole.scale_2) |scale_2| try std.testing.expectEqual(scale_2.data.ptr, halves[1].scale_2.?.data.ptr);
            // a rank holds half of the cut parts and all of the rest
            const held = whole.held(split);
            try std.testing.expect(held.rank * 2 >= held.bytes and held.rank < held.bytes);
        }
        try std.testing.expectEqual(whole.weight.data.ptr, (try modelopt.cut(arena, whole, .whole, 1, 2, &diagnostic)).weight.data.ptr);
    }
    // half of 192 inputs is not whole MXFP8 blocks of 32 a rank; half of 384 not whole FP8 blocks of 128
    try putLinear(arena, &source, prng.random(), "mx_odd", .mxfp8, 64, 96, false);
    const odd = try modelopt.read(arena, &source, "mx_odd", .{ .name = "mx_odd", .algo = .mxfp8, .group_size = 32 }, 64, 96, &diagnostic);
    try std.testing.expectError(error.Indivisible, modelopt.cut(arena, odd, .col, 0, 2, &diagnostic));
    try putLinear(arena, &source, prng.random(), "fp8_odd", .fp8_block, 384, 384, false);
    const odd_block = try modelopt.read(arena, &source, "fp8_odd", .{ .name = "fp8_odd", .algo = .fp8_block, .group_size = 128 }, 384, 384, &diagnostic);
    try std.testing.expectError(error.Indivisible, modelopt.cut(arena, odd_block, .row, 0, 2, &diagnostic));
    try std.testing.expectError(error.Indivisible, modelopt.cut(arena, odd_block, .col, 1, 2, &diagnostic));
}

test "csf_32x2_v1 scales: the streams whole on both ranks, each decoding its rows or columns of the plane" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var prng: std.Random.DefaultPrng = .init(9);
    const random = prng.random();
    const found = try csf_blocks_test.fixture(arena);
    // [256, 68]: 1,088 inputs, a rank's 544 whole blocks of 16 both ways
    const plane = found.planes[7];
    try std.testing.expectEqual(csf_blocks.Shape{ .rows = 256, .cols = 68 }, plane.shape);
    const rows = plane.shape.rows;
    const cols = plane.shape.cols * 16;
    var source: Source = .empty;
    try put(arena, &source, random, "e.weight", .U8, &.{ rows, cols / 2 });
    try put(arena, &source, random, "e.weight_scale_2", .F32, &.{});
    try put(arena, &source, random, "e.input_scale", .F32, &.{});
    const streams = [_]struct { []const u8, safetensors.DType, []const u8, u64 }{
        .{ csf_blocks.bases_suffix, .U8, plane.streams.bases, 1 },
        .{ csf_blocks.codes_suffix, .U8, plane.streams.codes, 1 },
        .{ csf_blocks.page_bases_suffix, .U32, plane.streams.page_bases, 4 },
        .{ csf_blocks.offsets_suffix, .U16, plane.streams.offsets, 2 },
        .{ csf_blocks.exceptions_suffix, .U16, plane.streams.exceptions, 2 },
    };
    for (streams) |stream| try source.put(arena, try std.fmt.allocPrint(arena, "e.weight_scale{s}", .{stream[0]}), .{
        .dtype = stream[1],
        .shape = try arena.dupe(u64, &.{stream[2].len / stream[3]}),
        .data = stream[2],
    });
    var diagnostic: Diagnostic = .{};
    const layer: modelopt.QuantizedLayer = .{ .name = "e", .algo = .nvfp4, .group_size = 16, .scale_encoding = .csf32 };
    const whole = try modelopt.read(arena, &source, "e", layer, rows, cols, &diagnostic);
    try std.testing.expect(whole.scale == null and whole.input_scale != null);
    try std.testing.expectEqualSlices(u8, plane.logical, try whole.blockScales(arena));
    // weight, weight_scale_2, five streams and the input scale; a rank holds the streams whole, no input scale
    const held = whole.held(.row);
    try std.testing.expectEqual(@as(usize, 8), held.count);
    try std.testing.expectEqual(held.bytes - 4 - whole.weight.data.len / 2, held.rank);
    for ([_]Split{ .row, .col }) |split| {
        var halves: [2]modelopt.Linear = undefined;
        for (&halves, 0..) |*half, rank| half.* = try modelopt.cut(arena, whole, split, rank, 2, &diagnostic);
        try expectJoined(whole.weight, .{ halves[0].weight, halves[1].weight }, split);
        const scales: [2]Tensor = .{
            .{ .dtype = .F8_E4M3, .shape = &.{ halves[0].rows, halves[0].cols / 16 }, .data = try halves[0].blockScales(arena) },
            .{ .dtype = .F8_E4M3, .shape = &.{ halves[1].rows, halves[1].cols / 16 }, .data = try halves[1].blockScales(arena) },
        };
        try expectJoined(.{ .dtype = .F8_E4M3, .shape = &.{ rows, cols / 16 }, .data = plane.logical }, scales, split);
        try std.testing.expectEqual(plane.streams.codes.ptr, halves[1].csf.?.streams.codes.ptr);
    }
    // a rank's rows need not be whole blocks of 32, as vLLM's ranges: 16 of 256, rank 3's half a block
    const sixteenth = try modelopt.cut(arena, whole, .row, 3, 16, &diagnostic);
    try std.testing.expectEqualSlices(u8, plane.logical[48 * 68 .. 64 * 68], try sixteenth.blockScales(arena));
    try std.testing.expectError(error.Indivisible, modelopt.cut(arena, whole, .row, 0, 3, &diagnostic));
}

test "a single-format export: its quant_algo, its group, the modules it excludes as vLLM matches them" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16's config.json and hf_quant_config.json, as ModelOpt 0.47.0 wrote them
    const ignore =
        \\["thinker.audio_tower*","thinker.lm_head","thinker.model.embed_tokens","thinker.model.layers.0*",
        \\"thinker.model.layers.1.*","thinker.model.layers.26*","thinker.model.layers.27*"]
    ;
    const flat = "{\"config_groups\":{\"group_0\":{\"weights\":{\"dynamic\":false,\"num_bits\":4,\"type\":\"float\",\"group_size\":16},\"targets\":[\"Linear\"]}},\"ignore\":" ++ ignore ++ ",\"quant_algo\":\"W4A16_NVFP4\",\"quant_method\":\"modelopt\"}";
    const nested = "{\"producer\":{\"name\":\"modelopt\"},\"quantization\":{\"quant_algo\":\"W4A16_NVFP4\",\"kv_cache_quant_algo\":null,\"group_size\":16,\"exclude_modules\":" ++ ignore ++ "}}";
    for ([_][]const u8{ flat, nested }) |text| {
        const value = try json.parseFromSliceLeaky(json.Value, arena, text, .{});
        const q = try modelopt.uniform(arena, value.object);
        try std.testing.expectEqual(modelopt.Algo.nvfp4, q.layer.algo);
        try std.testing.expect(q.layer.a16);
        try std.testing.expectEqual(@as(?u32, 16), q.layer.group_size);
        try std.testing.expectEqual(@as(usize, 7), q.exclude.len);
        for ([_][]const u8{
            "thinker.audio_tower.layers.3.fc1",        "thinker.lm_head",                       "thinker.model.layers.0.mlp.up_proj",
            "thinker.model.layers.1.self_attn.q_proj", "thinker.model.layers.26.mlp.down_proj", "thinker.model.layers.27.self_attn.o_proj",
        }) |module| try std.testing.expect(q.of(module) == null);
        // `layers.1.*` is no prefix of layers 10 to 19, `layers.0*` none of layers 2 to 25
        for ([_][]const u8{ "thinker.model.layers.13.mlp.gate_proj", "thinker.model.layers.2.self_attn.v_proj", "thinker.model.layers.25.mlp.down_proj" }) |module| {
            const layer = q.of(module) orelse return error.TestUnexpectedResult;
            try std.testing.expectEqualStrings(module, layer.name);
            try std.testing.expectEqual(modelopt.Scheme.nvfp4, modelopt.Scheme.of(layer).?);
        }
    }
    const mixed = try json.parseFromSliceLeaky(json.Value, arena, "{\"quant_algo\":\"MIXED_PRECISION\",\"quantized_layers\":{}}", .{});
    try std.testing.expectError(error.Mixed, modelopt.uniform(arena, mixed.object));
    try std.testing.expect(modelopt.wildcard("a*b?d", "a.x.bcd") and !modelopt.wildcard("a*b?d", "a.x.bd") and modelopt.wildcard("*", ""));
}
