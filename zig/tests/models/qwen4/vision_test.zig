//! `vision.zig`'s tests (spec 019 WS-02; `tools/check.sh qwen4 "tower"`): the tower against TensorFold 0.6.5's
//! (`$ZSPARK_REFERENCE_VISION`, `tools/reference/vision_export.py`, exported from the retired NVFP4 checkpoint's
//! BF16 tower) and its row rule across images, its kernels against `vision_ref.zig`, the config's check; the
//! quantized tower is FP4-Local's (WS-08, `$ZSPARK_FLASH_NEXT_FP4_LOCAL`). The BF16 tower's own tests are at tag
//! retired-checkpoints-2026-10-21.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const vision = @import("../../../src/models/qwen4/vision.zig");
const vision_ref = @import("../../../src/models/qwen4/vision_ref.zig");
const nvfp4 = @import("../../../src/weights/nvfp4.zig");
const numeric_fixture = @import("../../weights/numeric_fixture.zig");
const randomBf16 = numeric_fixture.randomBf16;
const Grid = vision.Grid;
const Tower = vision.Tower;

/// `zspark compare`'s bound, each row within 2e-3 relative rms of the reference's: the embedding, the first
/// block's input and the merger's norm hold it (5.5e-4, 6.2e-4, 2.6e-4 at worst, `bands`).
const bound = 2e-3;

/// A block from the reference's own input: its worst row measured 1.1e-2 (`bands`, block 26), most blocks 2 to
/// 5e-3; `tiny`'s are the reference's bits but at blocks 5 and 24 (1.8e-3, 4.5e-4). Not `zspark compare`'s 2e-3:
/// the reference's attention is PyTorch's flash kernel, whose key blocks and sums are not zspark's, and the
/// residual stream carries columns in the thousands where a bf16 step is 16 (block 26: 3,600 against 3,072).
/// An f64 emulation of block 0 on PyTorch's roundings is 3.0e-3 from the reference on `bands` and 2.4e-3 on
/// `tiny`, where this tower gives its bits: no arithmetic but the flash kernel's own holds 2e-3 per block.
const block_bound = 1.5e-2;

/// The merger's linears from the reference's own input (2.9e-3 fc1, 3.7e-3 the features, `bands`): the
/// reference's own fc1 is 2.4e-3 from the product in f64 on `bands` (54 rows) and 2e-4 on `tiny` (2 rows), so
/// its GEMM is the less exact one there (cuBLAS, PyTorch's reduced-precision reductions allowed, by the shape).
const merger_bound = 5e-3;

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

fn bf16s(gpa: std.mem.Allocator, tensor: safetensors.Tensor) ![]u16 {
    std.debug.assert(tensor.dtype == .BF16);
    const out = try gpa.alloc(u16, tensor.data.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), tensor.data);
    return out;
}

fn f32s(gpa: std.mem.Allocator, tensor: safetensors.Tensor) ![]f32 {
    std.debug.assert(tensor.dtype == .F32);
    const out = try gpa.alloc(f32, tensor.data.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), tensor.data);
    return out;
}

/// A comparison's rows: the worst row's relative rms, the median row's and the whole tensor's.
const Spread = struct {
    worst: f64,
    median: f64,
    whole: f64,

    fn of(gpa: std.mem.Allocator, mine: []const u16, want: []const u16, width: usize) !Spread {
        const rows = mine.len / width;
        const each = try gpa.alloc(f64, rows);
        defer gpa.free(each);
        var diff_all: f64 = 0;
        var ref_all: f64 = 0;
        for (each, 0..) |*e, row| {
            var diff: f64 = 0;
            var ref: f64 = 0;
            for (mine[row * width ..][0..width], want[row * width ..][0..width]) |a, b| {
                const x: f64 = numeric.f32FromBf16(a);
                const y: f64 = numeric.f32FromBf16(b);
                diff += (x - y) * (x - y);
                ref += y * y;
            }
            e.* = if (ref > 0) @sqrt(diff / ref) else @sqrt(diff);
            diff_all += diff;
            ref_all += ref;
        }
        std.mem.sort(f64, each, {}, std.sort.asc(f64));
        return .{ .worst = each[rows - 1], .median = each[rows / 2], .whole = @sqrt(diff_all / ref_all) };
    }

    fn print(self: Spread, what: []const u8, image: []const u8) void {
        std.debug.print("vision {s} {s}: worst row {e:.3}, median row {e:.3}, whole {e:.3}\n", .{ image, what, self.worst, self.median, self.whole });
    }
};

/// The worst row's relative rms of `mine` from `reference`, rows of `width`.
fn worstRow(mine: []const u16, want: []const u16, width: usize) f64 {
    std.debug.assert(mine.len == want.len);
    var worst: f64 = 0;
    var row: usize = 0;
    while (row * width < mine.len) : (row += 1) {
        var diff: f64 = 0;
        var ref: f64 = 0;
        for (mine[row * width ..][0..width], want[row * width ..][0..width]) |a, b| {
            const x: f64 = numeric.f32FromBf16(a);
            const y: f64 = numeric.f32FromBf16(b);
            diff += (x - y) * (x - y);
            ref += y * y;
        }
        worst = @max(worst, if (ref > 0) @sqrt(diff / ref) else @sqrt(diff));
    }
    return worst;
}

fn worstRowF64(mine: []const u16, want: []const f64, width: usize) f64 {
    var worst: f64 = 0;
    var row: usize = 0;
    while (row * width < mine.len) : (row += 1) {
        var diff: f64 = 0;
        var ref: f64 = 0;
        for (mine[row * width ..][0..width], want[row * width ..][0..width]) |a, y| {
            const x: f64 = numeric.f32FromBf16(a);
            diff += (x - y) * (x - y);
            ref += y * y;
        }
        worst = @max(worst, if (ref > 0) @sqrt(diff / ref) else @sqrt(diff));
    }
    return worst;
}

const Loaded = struct {
    checkpoint: safetensors.Checkpoint,
    tower: Tower,
    arena: std.heap.ArenaAllocator,

    /// The tower of the checkpoint `variable` names (`ZSPARK_FLASH_NEXT_FP4_LOCAL`'s MXFP8 and NVFP4 one); null when
    /// it or the references are not set.
    fn openFrom(variable: [*:0]const u8, max_patches: usize) !?Loaded {
        const path = env(variable) orelse return null;
        _ = env("ZSPARK_REFERENCE_VISION") orelse return null;
        var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, testing.io, .cwd(), path);
        errdefer checkpoint.close();
        var arena: std.heap.ArenaAllocator = .init(testing.allocator);
        errdefer arena.deinit();
        var diagnostic: qwen4_config.Diagnostic = .{};
        const tower = Tower.load(testing.allocator, arena.allocator(), &checkpoint, max_patches, &diagnostic) catch |err| {
            std.debug.print("vision: {s}\n", .{diagnostic.message});
            return err;
        };
        return .{ .checkpoint = checkpoint, .tower = tower, .arena = arena };
    }

    fn close(self: *Loaded) void {
        self.tower.deinit();
        self.checkpoint.close();
        self.arena.deinit();
    }
};

fn reference(name: []const u8) !safetensors.File {
    const dir = env("ZSPARK_REFERENCE_VISION").?;
    var buffer: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buffer, "{s}/{s}.safetensors", .{ dir, name });
    return safetensors.File.open(testing.allocator, testing.io, .cwd(), path);
}

fn download(gpa: std.mem.Allocator, view: cuda.View) ![]u16 {
    const out = try gpa.alloc(u16, view.count());
    try cuda.download(view, std.mem.sliceAsBytes(out));
    return out;
}

/// Set by `expectWithin` on a point past its bound: a test reports every point, then fails.
var past_bound = false;

fn expectWithin(what: []const u8, image: []const u8, worst: f64, limit: f64) !void {
    if (worst > limit) {
        std.debug.print("vision {s} {s}: worst row {e:.3}, above {e:.1}\n", .{ image, what, worst, limit });
        past_bound = true;
    }
}

fn passed() !void {
    defer past_bound = false;
    if (past_bound) return error.TestExpectedApproxEqRel;
}

const images = [_]struct { name: []const u8, grid: Grid }{
    .{ .name = "bands", .grid = .{ .t = 1, .h = 12, .w = 18 } },
    .{ .name = "tiny", .grid = .{ .t = 1, .h = 2, .w = 4 } },
};

/// End to end from `pixel_values` the bounds are the measured ones, beside the tower's own sensitivity: one bf16
/// step in one pixel value of 1,000 moves `shapes`' features by 0.19 worst row, 2.4e-2 median, 4.5e-2 whole
/// (the spread test); this tower against TensorFold's: `shapes` 0.51, 2.8e-2, 7.2e-2; `bands` 7.7e-2, 2.3e-2,
/// 2.6e-2; `tiny` 2.5e-2. The blocks' differences above, carried through 27 blocks, grow as that step does.
const end_to_end = struct {
    const worst = 0.6;
    const median = 4e-2;
    const whole = 0.1;
};

// One tower of the default patch budget for both (spec 133 WS-02).
test "tower: a quantized tower's image features are the same bits alone, beside another and in another order, and near TensorFold's BF16 tower's, image by image" {
    var loaded = (try Loaded.openFrom("ZSPARK_FLASH_NEXT_FP4_LOCAL", vision.max_patches_default)) orelse return error.SkipZigTest;
    defer loaded.close();
    try expectRowRule(&loaded);
    try expectNearBf16(&loaded);
}

/// `pair`'s two images alone, together and swapped: each image's features the same bits every time.
fn expectRowRule(loaded: *Loaded) !void {
    const gpa = testing.allocator;
    var processor = try reference("pair.processor");
    defer processor.close();
    const pixels = try f32s(gpa, processor.get("pixel_values").?);
    defer gpa.free(pixels);
    const bands = images[0].grid;
    const tiny = images[1].grid;
    const bands_pixels = pixels[0 .. bands.patches() * vision.patch_values];
    const tiny_pixels = pixels[bands_pixels.len..];
    const swapped = try gpa.alloc(f32, pixels.len);
    defer gpa.free(swapped);
    @memcpy(swapped[0..tiny_pixels.len], tiny_pixels);
    @memcpy(swapped[tiny_pixels.len..], bands_pixels);

    const runs = [_]struct { pixels: []const f32, grids: []const Grid }{
        .{ .pixels = bands_pixels, .grids = &.{bands} },
        .{ .pixels = tiny_pixels, .grids = &.{tiny} },
        .{ .pixels = pixels, .grids = &.{ bands, tiny } },
        .{ .pixels = swapped, .grids = &.{ tiny, bands } },
    };
    var outs: [runs.len][]u16 = undefined;
    for (runs, &outs) |run, *out| {
        const rows = Tower.featureRows(run.grids);
        var features: cuda.Buffer = try .init(.bf16, &.{ rows, vision.out_hidden });
        defer features.deinit();
        try loaded.tower.encode(gpa, run.pixels, run.grids, features.view);
        out.* = try download(gpa, features.view);
    }
    defer for (outs) |out| gpa.free(out);
    const bands_n = bands.patches() / 4 * vision.out_hidden;
    const tiny_n = tiny.patches() / 4 * vision.out_hidden;
    try testing.expectEqualSlices(u16, outs[0], outs[2][0..bands_n]);
    try testing.expectEqualSlices(u16, outs[1], outs[2][bands_n..]);
    try testing.expectEqualSlices(u16, outs[1], outs[3][0..tiny_n]);
    try testing.expectEqualSlices(u16, outs[0], outs[3][tiny_n..]);
}

// The tower of 1,024 patches this test loads second serves the blocks' check after it (spec 133 WS-02).
test "tower: runs of whole images split by the patch budget give the bits of one run; and a quantized tower's blocks from TensorFold's BF16 inputs, the quantization's spread a block" {
    const gpa = testing.allocator;
    // a budget of 216 patches puts bands (216) and tiny (8) in two runs
    var loaded = (try Loaded.openFrom("ZSPARK_FLASH_NEXT_FP4_LOCAL", 216)) orelse return error.SkipZigTest;
    defer loaded.close();
    var processor = try reference("pair.processor");
    defer processor.close();
    var file = try reference("pair.tower");
    defer file.close();
    const pixels = try f32s(gpa, processor.get("pixel_values").?);
    defer gpa.free(pixels);
    const grids = [_]Grid{ images[0].grid, images[1].grid };
    var features: cuda.Buffer = try .init(.bf16, &.{ Tower.featureRows(&grids), vision.out_hidden });
    defer features.deinit();
    try loaded.tower.encode(gpa, pixels, &grids, features.view);
    const split = try download(gpa, features.view);
    defer gpa.free(split);
    loaded.tower.deinit();
    var diagnostic: qwen4_config.Diagnostic = .{};
    loaded.tower = try Tower.load(gpa, loaded.arena.allocator(), &loaded.checkpoint, 1024, &diagnostic);
    try loaded.tower.encode(gpa, pixels, &grids, features.view);
    const whole = try download(gpa, features.view);
    defer gpa.free(whole);
    try testing.expectEqualSlices(u16, whole, split);
    // an image above the budget is refused
    loaded.tower.max_patches = 100;
    try testing.expectError(error.UnsupportedVision, loaded.tower.encode(gpa, pixels, &grids, features.view));
    loaded.tower.max_patches = 1024;
    try blocksFromBf16(&loaded);
}

test "tower kernels: the attention by image near the host's in f64, every image's rows its own" {
    const gpa = testing.allocator;
    // a lone patch, a part of a tile, a tile and two keys, whole tiles (one, two), a ragged last tile of a ragged
    // last key step
    const segments = [_]usize{ 1, 5, 130, 128, 256, 200 };
    var p: usize = 0;
    for (segments) |n| p += n;
    var prng: std.Random.DefaultPrng = .init(19);
    const qkv = try gpa.alloc(u16, p * 3 * vision.hidden);
    defer gpa.free(qkv);
    randomBf16(prng.random(), qkv, 1.5);
    var qkv_buffer: cuda.Buffer = try .init(.bf16, &.{ p, 3 * vision.hidden });
    defer qkv_buffer.deinit();
    try qkv_buffer.upload(std.mem.sliceAsBytes(qkv));
    var out: cuda.Buffer = try .init(.bf16, &.{ p, vision.hidden });
    defer out.deinit();
    var grids: [segments.len]Grid = undefined;
    // the tiles as frames of n patches: `writeTiles` cuts by h * w alone
    for (segments, &grids) |n, *grid| grid.* = .{ .t = 1, .h = 1, .w = n };
    const count = vision.tileCount(&grids);
    const tiles = try gpa.alloc([4]i32, count);
    defer gpa.free(tiles);
    vision.writeTiles(&grids, tiles);
    var tiles_buffer: cuda.Buffer = try .init(.i32, &.{ count, 4 });
    defer tiles_buffer.deinit();
    try tiles_buffer.upload(std.mem.sliceAsBytes(tiles));
    try vision.attention(qkv_buffer.view, tiles_buffer.view, out.view);
    const got = try download(gpa, out.view);
    defer gpa.free(got);
    const want = try gpa.alloc(f64, p * vision.hidden);
    defer gpa.free(want);
    const scores = try gpa.alloc(f64, p);
    defer gpa.free(scores);
    vision_ref.attention(qkv, &segments, want, scores);
    const e = worstRowF64(got, want, vision.hidden);
    std.debug.print("vision attention: worst row {e:.3} of the host's f64\n", .{e});
    try testing.expect(e < 1e-2);

    // the last image alone, at the start of the call: its rows' bits as among the others
    const last = segments[segments.len - 1];
    const from = p - last;
    var alone: cuda.Buffer = try .init(.bf16, &.{ last, 3 * vision.hidden });
    defer alone.deinit();
    try alone.upload(std.mem.sliceAsBytes(qkv[from * 3 * vision.hidden ..]));
    var alone_out: cuda.Buffer = try .init(.bf16, &.{ last, vision.hidden });
    defer alone_out.deinit();
    const one = [_]Grid{grids[segments.len - 1]};
    const one_tiles = try gpa.alloc([4]i32, vision.tileCount(&one));
    defer gpa.free(one_tiles);
    vision.writeTiles(&one, one_tiles);
    var one_buffer: cuda.Buffer = try .init(.i32, &.{ one_tiles.len, 4 });
    defer one_buffer.deinit();
    try one_buffer.upload(std.mem.sliceAsBytes(one_tiles));
    try vision.attention(alone.view, one_buffer.view, alone_out.view);
    const alone_got = try download(gpa, alone_out.view);
    defer gpa.free(alone_got);
    try testing.expectEqualSlices(u16, got[from * vision.hidden ..], alone_got);
}

test "tower kernels: the linear with its bias and GELU near the host's, a ragged K, a row's bits its own" {
    const gpa = testing.allocator;
    const m = 150;
    const n = 136;
    const k = 4304;
    var prng: std.Random.DefaultPrng = .init(23);
    const random = prng.random();
    const x = try gpa.alloc(u16, m * k);
    defer gpa.free(x);
    const w = try gpa.alloc(u16, n * k);
    defer gpa.free(w);
    const bias = try gpa.alloc(u16, n);
    defer gpa.free(bias);
    randomBf16(random, x, 1);
    randomBf16(random, w, 0.02);
    randomBf16(random, bias, 0.5);
    var linear: vision.Linear = .{ .weight = .{ .bf16 = try .init(.bf16, &.{ n, k }) }, .bias = try .init(.bf16, &.{n}) };
    defer linear.weight.bf16.deinit();
    defer linear.bias.deinit();
    try linear.weight.bf16.upload(std.mem.sliceAsBytes(w));
    try linear.bias.upload(std.mem.sliceAsBytes(bias));
    var xb: cuda.Buffer = try .init(.bf16, &.{ m, k });
    defer xb.deinit();
    try xb.upload(std.mem.sliceAsBytes(x));
    var out: cuda.Buffer = try .init(.bf16, &.{ m, n });
    defer out.deinit();
    const want = try gpa.alloc(f64, m * n);
    defer gpa.free(want);
    for ([_]vision.Act{ .none, .gelu_tanh, .gelu_erf }) |act| {
        try linear.forward(xb.view, out.view, act);
        const got = try download(gpa, out.view);
        defer gpa.free(got);
        vision_ref.linear(x, w, bias, m, n, k, act, want);
        const e = worstRowF64(got, want, n);
        std.debug.print("vision linear {t}: worst row {e:.3} of the host's f64\n", .{ act, e });
        try testing.expect(e < 1e-2);
        // the last 3 rows alone: the same bits
        try linear.forward(xb.span(m - 3, 3), out.rows(3), act);
        const few = try download(gpa, out.rows(3));
        defer gpa.free(few);
        try testing.expectEqualSlices(u16, got[(m - 3) * n ..], few);
    }
}

test "tower: the config's vision_config is checked, the published one taken" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const published =
        \\{"vision_config":{"deepstack_visual_indexes":[],"depth":27,"hidden_act":"gelu_pytorch_tanh","hidden_size":1152,
        \\"in_channels":3,"intermediate_size":4304,"model_type":"qwen4_exp_vision","num_heads":16,
        \\"num_position_embeddings":2304,"out_hidden_size":2560,"patch_size":16,"spatial_merge_size":2,"temporal_patch_size":2}}
    ;
    var diagnostic: qwen4_config.Diagnostic = .{};
    try vision.checkConfig(a, published, &diagnostic);
    const wider = try std.mem.replaceOwned(u8, a, published, "\"depth\":27", "\"depth\":32");
    try testing.expectError(error.UnsupportedVision, vision.checkConfig(a, wider, &diagnostic));
    try testing.expect(std.mem.indexOf(u8, diagnostic.message, "depth") != null);
    const other = try std.mem.replaceOwned(u8, a, published, "qwen4_exp_vision", "siglip");
    try testing.expectError(error.UnsupportedVision, vision.checkConfig(a, other, &diagnostic));
    try testing.expectError(error.UnsupportedVision, vision.checkConfig(a, "{\"model_type\":\"qwen4_exp\"}", &diagnostic));
}

/// A quantized tower (MXFP8 attention, NVFP4 MLP and merger, as a ModelOpt mixed-precision checkpoint stores
/// them) against the BF16 tower of the same model and against TensorFold's features, from the same
/// `pixel_values`: the quantization's spread, beside the BF16 tower's own against TensorFold (`end_to_end`).
const quantized = struct {
    /// Features end to end against the BF16 tower's, measured (spec 019 WS-08, 2026-10-18): `shapes` 1.86 worst
    /// row, 0.31 median, 0.35 whole; `bands` 0.64, 0.28, 0.25; `tiny` (one row) 0.62. Against TensorFold's they
    /// are the same (1.40, 0.31, 0.32 on `shapes`): it is the checkpoint's own quantization, not the kernels', a
    /// stored weight decoded on the host being 2.7% from the BF16 one as MXFP8 and 9.5% as NVFP4, and a block
    /// from TensorFold's BF16 input 6 to 9% from its output (the next test) against 2 to 5e-3 for the BF16
    /// tower. The bounds are those maxima with a quarter more.
    const worst = 2.4;
    const median = 0.8;
    const whole = 0.8;
};

/// A quantized tower's features near TensorFold's BF16 tower's, image by image.
fn expectNearBf16(mixed: *Loaded) !void {
    const gpa = testing.allocator;
    const cases = [_]struct { processor: []const u8, tower: []const u8, grids: []const Grid, alone: []const []const u8 }{
        .{ .processor = "shapes.processor", .tower = "shapes.tower", .grids = &.{.{ .t = 1, .h = 16, .w = 16 }}, .alone = &.{"shapes.alone"} },
        .{ .processor = "pair.processor", .tower = "pair.tower", .grids = &.{ images[0].grid, images[1].grid }, .alone = &.{ "bands.alone", "tiny.alone" } },
    };
    for (cases) |case| {
        var processor = try reference(case.processor);
        defer processor.close();
        var file = try reference(case.tower);
        defer file.close();
        const pixels = try f32s(gpa, processor.get("pixel_values").?);
        defer gpa.free(pixels);
        const rows = Tower.featureRows(case.grids);
        var features: cuda.Buffer = try .init(.bf16, &.{ rows, vision.out_hidden });
        defer features.deinit();
        try mixed.tower.encode(gpa, pixels, case.grids, features.view);
        const got = try download(gpa, features.view);
        defer gpa.free(got);
        var at: usize = 0;
        for (case.grids, case.alone) |grid, alone| {
            const want = try bf16s(gpa, file.get(alone).?);
            defer gpa.free(want);
            const n = grid.patches() / vision.merge_unit * vision.out_hidden;
            // the retired NVFP4 checkpoint's BF16 tower, which these bounds were measured against, is TensorFold's
            // within 2e-3 a row (tag retired-checkpoints-2026-10-21)
            const to_bf16 = try Spread.of(gpa, got[at..][0..n], want, vision.out_hidden);
            to_bf16.print("quantized tower's features against TensorFold's (BF16)", alone);
            try expectWithin("quantized features against TensorFold's BF16 tower's, worst row", alone, to_bf16.worst, quantized.worst);
            try expectWithin("quantized features against TensorFold's BF16 tower's, median row", alone, to_bf16.median, quantized.median);
            try expectWithin("quantized features against TensorFold's BF16 tower's, whole", alone, to_bf16.whole, quantized.whole);
            at += n;
        }
    }
    try passed();
}

/// A quantized tower's blocks from TensorFold's BF16 inputs, on a tower of 1,024 patches: the quantization's
/// spread a block, printed.
fn blocksFromBf16(loaded: *Loaded) !void {
    const gpa = testing.allocator;
    std.debug.assert(loaded.tower.max_patches == 1024);
    const tower = &loaded.tower;
    for (images) |image| {
        var file = try reference(try std.fmt.allocPrint(loaded.arena.allocator(), "{s}.tower", .{image.name}));
        defer file.close();
        const p = image.grid.patches();
        try tower.begin(gpa, &.{image.grid});
        var worst: f64 = 0;
        var worst_at: usize = 0;
        var median: f64 = 0;
        var name: [64]u8 = undefined;
        for (0..vision.depth) |i| {
            const input_name = if (i == 0) "blocks.0.in" else try std.fmt.bufPrint(&name, "blocks.{d}.out", .{i - 1});
            const input = try bf16s(gpa, file.get(input_name).?);
            defer gpa.free(input);
            try tower.h.uploadAt(0, std.mem.sliceAsBytes(input));
            tower.pending = false;
            try tower.block(i);
            try tower.settle();
            const got = try download(gpa, tower.h.rows(p));
            defer gpa.free(got);
            const want = try bf16s(gpa, file.get(try std.fmt.bufPrint(&name, "blocks.{d}.out", .{i})).?);
            defer gpa.free(want);
            const spread = try Spread.of(gpa, got, want, vision.hidden);
            if (spread.worst > worst) {
                worst = spread.worst;
                worst_at = i;
            }
            median = @max(median, spread.median);
        }
        const last = try bf16s(gpa, file.get("blocks.26.out").?);
        defer gpa.free(last);
        try tower.h.uploadAt(0, std.mem.sliceAsBytes(last));
        tower.pending = false;
        var features: cuda.Buffer = try .init(.bf16, &.{ p / vision.merge_unit, vision.out_hidden });
        defer features.deinit();
        try tower.merge(features.view);
        const got = try download(gpa, features.view);
        defer gpa.free(got);
        const want = try bf16s(gpa, file.get("features").?);
        defer gpa.free(want);
        const merger = try Spread.of(gpa, got, want, vision.out_hidden);
        std.debug.print("vision {s} quantized: worst block {e:.3} (block {d}), worst median row of a block {e:.3}; the merger from blocks.26.out: worst row {e:.3}, median {e:.3}, whole {e:.3}\n", .{ image.name, worst, worst_at, median, merger.worst, merger.median, merger.whole });
    }
}

/// e4m3 bytes at random, NaN (0x7f, 0xff) left out.
fn randomE4m3(random: std.Random, out: []u8) void {
    for (out) |*b| {
        b.* = random.int(u8);
        while (b.* & 0x7f == 0x7f) b.* = random.int(u8);
    }
}

test "tower kernels: an MXFP8 and an NVFP4 linear with bias and GELU near the host's, a padded K and width, a row's bits its own" {
    const gpa = testing.allocator;
    const m = 150;
    var prng: std.Random.DefaultPrng = .init(29);
    const random = prng.random();
    // NVFP4: K 4,304 padded to 4,352 (zero codes and scales, zero inputs past 4,304, as fc2 reads fc1's output),
    // and 200 outputs written to 256 columns (zero past 200, as fc1 writes for fc2)
    const cases = [_]struct { format: vision.Format, n: usize, width: usize, k: usize, kp: usize }{
        .{ .format = .mxfp8, .n = 136, .width = 136, .k = 1152, .kp = 1152 },
        .{ .format = .nvfp4, .n = 200, .width = 256, .k = 4304, .kp = 4352 },
    };
    for (cases) |case| {
        const n = case.n;
        const kp = case.kp;
        const x = try gpa.alloc(u16, m * kp);
        defer gpa.free(x);
        randomBf16(random, x, 1);
        for (0..m) |r| @memset(x[r * kp + case.k ..][0 .. kp - case.k], 0);
        // the weight's values in bf16 (exact), for the host, and its stored codes and scales
        const w = try gpa.alloc(u16, case.width * kp);
        defer gpa.free(w);
        @memset(w, 0);
        const bias = try gpa.alloc(u16, case.width);
        defer gpa.free(bias);
        randomBf16(random, bias, 0.5);
        @memset(bias[n..], 0);
        var scale: f64 = 1;
        var linear: vision.Linear = undefined;
        switch (case.format) {
            .mxfp8 => {
                const codes = try gpa.alloc(u8, n * kp);
                defer gpa.free(codes);
                randomE4m3(random, codes);
                const exponents = try gpa.alloc(u8, n * kp / 32);
                defer gpa.free(exponents);
                for (exponents) |*e| e.* = 112 + random.uintLessThan(u8, 12);
                for (0..n) |c| for (0..kp) |i| {
                    const v = numeric.f32FromE4m3(codes[c * kp + i]) * std.math.pow(f32, 2, @as(f32, @floatFromInt(exponents[c * kp / 32 + i / 32])) - 127);
                    w[c * kp + i] = numeric.bf16FromF32(v);
                };
                linear = .{ .weight = .{ .mxfp8 = try .init(gpa, codes, exponents, n, kp) }, .bias = try .init(.bf16, &.{case.width}) };
            },
            .nvfp4 => {
                const codes = try gpa.alloc(u8, n * kp / 2);
                defer gpa.free(codes);
                const scales = try gpa.alloc(u8, n * kp / 16);
                defer gpa.free(scales);
                for (codes) |*c| c.* = random.int(u8);
                // e4m3 scales between 2^-6 and 2^2, positive
                for (scales) |*s| s.* = 0x08 + random.uintLessThan(u8, 0x40);
                for (0..n) |c| {
                    @memset(codes[c * kp / 2 + case.k / 2 ..][0 .. (kp - case.k) / 2], 0);
                    @memset(scales[c * kp / 16 + case.k / 16 ..][0 .. (kp - case.k) / 16], 0);
                }
                for (0..n) |c| for (0..kp) |i| {
                    const byte = codes[c * kp / 2 + i / 2];
                    const code = if (i % 2 == 0) byte & 0xf else byte >> 4;
                    w[c * kp + i] = numeric.bf16FromF32(nvfp4.e2m1[code] * numeric.f32FromE4m3(scales[c * kp / 16 + i / 16]));
                };
                scale = 0.0123;
                linear = .{ .weight = .{ .nvfp4 = try .init(gpa, codes, scales, n, kp, scale) }, .bias = try .init(.bf16, &.{case.width}) };
            },
            .bf16 => unreachable,
        }
        defer linear.deinit();
        try linear.bias.upload(std.mem.sliceAsBytes(bias));
        var xb: cuda.Buffer = try .init(.bf16, &.{ m, kp });
        defer xb.deinit();
        try xb.upload(std.mem.sliceAsBytes(x));
        var out: cuda.Buffer = try .init(.bf16, &.{ m, case.width });
        defer out.deinit();
        const want = try gpa.alloc(f64, m * case.width);
        defer gpa.free(want);
        for ([_]vision.Act{ .none, .gelu_tanh, .gelu_erf }) |act| {
            nvfp4.biased_tile = null;
            try linear.forward(xb.view, out.view, act);
            const got = try download(gpa, out.view);
            defer gpa.free(got);
            vision_ref.linearScaled(x, w, scale, bias, m, case.width, kp, act, want);
            const e = worstRowF64(got, want, case.width);
            std.debug.print("vision linear {t} {t}: worst row {e:.3} of the host's f64\n", .{ case.format, act, e });
            try testing.expect(e < 1e-2);
            // the padded columns exact zeros
            for (0..m) |r| for (got[r * case.width + n .. (r + 1) * case.width]) |v| try testing.expectEqual(@as(u16, 0), v);
            // the last 3 rows alone, and every tile: the same bits
            for ([_]u3{ 0, 4, 5 }) |tile| {
                nvfp4.biased_tile = tile;
                try linear.forward(xb.span(m - 3, 3), out.rows(3), act);
                const few = try download(gpa, out.rows(3));
                defer gpa.free(few);
                try testing.expectEqualSlices(u16, got[(m - 3) * case.width ..], few);
                try linear.forward(xb.view, out.view, act);
                const all = try download(gpa, out.view);
                defer gpa.free(all);
                try testing.expectEqualSlices(u16, got, all);
            }
            nvfp4.biased_tile = null;
        }
    }
}

test "tower: a quantized tower is read by its tensors' formats" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    var diagnostic: qwen4_config.Diagnostic = .{};
    if (env("ZSPARK_FLASH_NEXT_FP4_LOCAL")) |path| {
        var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, testing.io, .cwd(), path);
        defer checkpoint.close();
        const a = arena.allocator();
        try testing.expectEqual(vision.Format.mxfp8, try vision.Format.of(a, &checkpoint, "model.visual.blocks.0.attn.qkv", 3 * vision.hidden, vision.hidden, &diagnostic));
        try testing.expectEqual(vision.Format.mxfp8, try vision.Format.of(a, &checkpoint, "model.visual.blocks.26.attn.proj", vision.hidden, vision.hidden, &diagnostic));
        try testing.expectEqual(vision.Format.nvfp4, try vision.Format.of(a, &checkpoint, "model.visual.blocks.3.mlp.linear_fc1", vision.intermediate, vision.hidden, &diagnostic));
        try testing.expectEqual(vision.Format.nvfp4, try vision.Format.of(a, &checkpoint, "model.visual.blocks.3.mlp.linear_fc2", vision.hidden, vision.intermediate, &diagnostic));
        try testing.expectEqual(vision.Format.nvfp4, try vision.Format.of(a, &checkpoint, "model.visual.merger.linear_fc2", vision.out_hidden, vision.merged, &diagnostic));
        // a shape that is not the tower's is refused in words
        try testing.expectError(error.UnsupportedVision, vision.Format.of(a, &checkpoint, "model.visual.blocks.0.attn.qkv", vision.hidden, vision.hidden, &diagnostic));
        try testing.expect(std.mem.indexOf(u8, diagnostic.message, "F8_E4M3") != null);
    }
}

// `ZSPARK_VISION_PROFILE` names square grids by their side in patches (`16,64,128`: 256, 4,096 and 16,384
// patches, a 256 x 256 image, a 1024 x 1024 one and TensorFold's largest), each encoded from random pixels in
// four passes, the first a warm-up, on the quantized tower (`ZSPARK_FLASH_NEXT_FP4_LOCAL`); each
// pass's wall time is printed. What `nsys` is given:
//
//   ZSPARK_VISION_PROFILE=16,64,128 nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none TEST
test "tower profile: the tower's passes at square grids, for a trace" {
    const sides = env("ZSPARK_VISION_PROFILE") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    // each checkpoint set, one after the other: the BF16 tower, then the quantized one
    for ([_][*:0]const u8{"ZSPARK_FLASH_NEXT_FP4_LOCAL"}) |variable| {
        var loaded = (try Loaded.openFrom(variable, vision.max_patches_default)) orelse continue;
        defer loaded.close();
        std.debug.print("vision profile: {s}\n", .{variable});
        try profile(gpa, &loaded, sides);
    }
}

fn profile(gpa: std.mem.Allocator, loaded: *Loaded, sides: []const u8) !void {
    var prng: std.Random.DefaultPrng = .init(5);
    var it = std.mem.tokenizeScalar(u8, sides, ',');
    while (it.next()) |text| {
        const side = try std.fmt.parseInt(usize, text, 10);
        const grids = [_]Grid{.{ .t = 1, .h = side, .w = side }};
        const pixels = try gpa.alloc(f32, grids[0].patches() * vision.patch_values);
        defer gpa.free(pixels);
        for (pixels) |*v| v.* = prng.random().float(f32) * 2 - 1;
        var features: cuda.Buffer = try .init(.bf16, &.{ Tower.featureRows(&grids), vision.out_hidden });
        defer features.deinit();
        for (0..4) |pass| {
            const start: std.Io.Timestamp = .now(testing.io, .awake);
            try loaded.tower.encode(gpa, pixels, &grids, features.view);
            try cuda.synchronize();
            const ms = @as(f64, @floatFromInt(start.untilNow(testing.io, .awake).toNanoseconds())) / 1e6;
            std.debug.print("vision profile {d} x {d} ({d} patches) pass {d}: {d:.2} ms\n", .{ side, side, grids[0].patches(), pass, ms });
        }
    }
}
