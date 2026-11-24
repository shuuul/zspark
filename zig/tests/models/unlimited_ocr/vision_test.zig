//! Unlimited-OCR-2's tower against its HF fixtures (`export_v2.py`): image rows from `prefill.embeds`,
//! selected by `seq_mask`, and each image/crop's bits alone or beside others. Uses
//! `$ZSPARK_UNLIMITED_OCR_2` and `$ZSPARK_REFERENCE_UNLIMITED_OCR_2`; no retired v1 fixture is read.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const vision = @import("../../../src/models/unlimited_ocr/vision.zig");

const Size = vision.Size;
const Tower = vision.Tower;

/// Whole image-feature cosine to HF; the shared tower's original acceptance floor.
const cosine_bound = 0.999;

/// `lm_features`' worst content row's cosine, at least: 0.968 measured (gundam; small 0.994, base 0.998), where
/// the whole is 0.9998. One row moves with the sums' order (0.979 with an earlier relpos kernel).
const row_bound = 0.95;

fn env(name: []const u8) ?[]const u8 {
    return testing.environ.getPosix(name);
}

const Loaded = struct {
    checkpoint: safetensors.Checkpoint,
    tower: Tower,
    arena: std.heap.ArenaAllocator,

    fn open(capacity: Tower.Capacity) !?Loaded {
        const path = env("ZSPARK_UNLIMITED_OCR_2") orelse return null;
        _ = env("ZSPARK_REFERENCE_UNLIMITED_OCR_2") orelse return null;
        var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, testing.io, .cwd(), path);
        errdefer checkpoint.close();
        var arena: std.heap.ArenaAllocator = .init(testing.allocator);
        errdefer arena.deinit();
        var diagnostic: vision.Diagnostic = .{};
        const tower = Tower.load(testing.allocator, arena.allocator(), &checkpoint, capacity, &diagnostic) catch |err| {
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
    var buffer: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buffer, "{s}/{s}.safetensors", .{ env("ZSPARK_REFERENCE_UNLIMITED_OCR_2").?, name });
    return safetensors.File.open(testing.allocator, testing.io, .cwd(), path);
}

fn get(file: *const safetensors.File, name: []const u8) !safetensors.Tensor {
    return file.get(name) orelse {
        std.debug.print("vision: the reference lacks {s}\n", .{name});
        return error.TestUnexpectedResult;
    };
}

fn bits(tensor: safetensors.Tensor) []const u16 {
    std.debug.assert(tensor.dtype == .BF16);
    return @alignCast(std.mem.bytesAsSlice(u16, tensor.data));
}

const Agreement = struct {
    cosine: f64,
    max_abs: f64,

    fn of(mine: []const f64, want: []const f64) Agreement {
        var dot: f64 = 0;
        var a2: f64 = 0;
        var b2: f64 = 0;
        var worst: f64 = 0;
        for (mine, want) |a, b| {
            dot += a * b;
            a2 += a * a;
            b2 += b * b;
            worst = @max(worst, @abs(a - b));
        }
        return .{ .cosine = dot / @sqrt(a2 * b2), .max_abs = worst };
    }
};

/// Set when image-feature agreement falls below a floor; report all cases before failing.
var under_bound = false;

fn passed() !void {
    defer under_bound = false;
    if (under_bound) return error.TestExpectedApproxEqRel;
}

/// One page's views in the OCR2 fixture (global views and crop grids are batched by page).
fn imageOf(ref: *const safetensors.File, page: usize) !vision.Image {
    const grid = try get(ref, "images_spatial_crop");
    const w: usize = @intCast(std.mem.readInt(i32, grid.data[page * 8 ..][0..4], .little));
    const h: usize = @intCast(std.mem.readInt(i32, grid.data[page * 8 + 4 ..][0..4], .little));
    const per_global = 3 * Size.global.side * Size.global.side;
    var crops: []const u16 = &.{};
    if (ref.get("views.crops")) |c| {
        var at: usize = 0;
        for (0..page) |p| {
            const gw: usize = @intCast(std.mem.readInt(i32, grid.data[p * 8 ..][0..4], .little));
            const gh: usize = @intCast(std.mem.readInt(i32, grid.data[p * 8 + 4 ..][0..4], .little));
            at += gw * gh;
        }
        const per_crop = 3 * Size.crop.side * Size.crop.side;
        crops = bits(c)[at * per_crop ..][0 .. w * h * per_crop];
    }
    return .{ .global = bits(try get(ref, "views.global"))[page * per_global ..][0..per_global], .crops = crops, .width = w, .height = h };
}

fn imageFeatures(gpa: std.mem.Allocator, ref: *const safetensors.File) ![]u16 {
    const mask = try get(ref, "seq_mask");
    const embeds = bits(try get(ref, "prefill.embeds"));
    try testing.expectEqual(mask.data.len * vision.lm_hidden, embeds.len);
    var rows: usize = 0;
    for (mask.data) |v| rows += @intFromBool(v != 0);
    const out = try gpa.alloc(u16, rows * vision.lm_hidden);
    var at: usize = 0;
    for (mask.data, 0..) |v, row| {
        if (v == 0) continue;
        @memcpy(out[at..][0..vision.lm_hidden], embeds[row * vision.lm_hidden ..][0..vision.lm_hidden]);
        at += vision.lm_hidden;
    }
    return out;
}

/// `lm_features` against the reference's: the newlines and the separator to the bit, each content row by its
/// cosine, the whole by its cosine.
fn expectFeatures(gpa: std.mem.Allocator, case: []const u8, mine_bits: []const u16, want_bits: []const u16, newline: []const u16, separator: []const u16) !void {
    const width = vision.lm_hidden;
    const rows = want_bits.len / width;
    try testing.expectEqual(want_bits.len, mine_bits.len);
    var marks: usize = 0;
    var low: f64 = 1;
    const mine = try gpa.alloc(f64, mine_bits.len);
    defer gpa.free(mine);
    const want = try gpa.alloc(f64, want_bits.len);
    defer gpa.free(want);
    for (mine, mine_bits) |*v, b| v.* = numeric.f32FromBf16(b);
    for (want, want_bits) |*v, b| v.* = numeric.f32FromBf16(b);
    for (0..rows) |r| {
        const w = want_bits[r * width ..][0..width];
        const m = mine_bits[r * width ..][0..width];
        if (std.mem.eql(u16, w, newline) or std.mem.eql(u16, w, separator)) {
            try testing.expectEqualSlices(u16, w, m);
            marks += 1;
            continue;
        }
        low = @min(low, Agreement.of(mine[r * width ..][0..width], want[r * width ..][0..width]).cosine);
    }
    const whole = Agreement.of(mine, want);
    // the hash of the bits, for a change that claims them unchanged
    std.debug.print("vision {s} lm_features: {d} rows, {d} newlines and separators to the bit, row cosine {d:.6} at worst, whole {d:.6}, max abs {e:.3}, bits {x:0>16}\n", .{ case, rows, marks, low, whole.cosine, whole.max_abs, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(mine_bits)) });
    try testing.expect(std.mem.eql(u16, want_bits[(rows - 1) * width ..], separator));
    if (!(whole.cosine >= cosine_bound and low >= row_bound)) under_bound = true;
}

test "vision: OCR2 image features near HF and image rows bit exact alone or beside others" {
    var loaded = (try Loaded.open(.{})) orelse return error.SkipZigTest;
    defer loaded.close();
    const gpa = testing.allocator;
    const tower = &loaded.tower;
    const newline = bits(loaded.checkpoint.get("model.image_newline").?);
    const separator = bits(loaded.checkpoint.get("model.view_seperator").?);
    var refs: [2]safetensors.File = undefined;
    const cases = [_][]const u8{ "one_page", "two_pages" };
    var opened: usize = 0;
    defer for (refs[0..opened]) |*r| r.close();
    for (cases, 0..) |case, i| {
        refs[i] = try reference(case);
        opened += 1;
    }
    for (cases, &refs, 1..) |case, *ref, count| {
        var images: [2]vision.Image = undefined;
        var rows: usize = 0;
        for (images[0..count], 0..) |*image, p| {
            image.* = try imageOf(ref, p);
            rows += image.rows();
        }
        const want = try imageFeatures(gpa, ref);
        defer gpa.free(want);
        var out: cuda.Buffer = try .init(.bf16, &.{ rows, vision.lm_hidden });
        defer out.deinit();
        try tower.encode(gpa, images[0..count], out.view);
        const mine = try gpa.alloc(u16, rows * vision.lm_hidden);
        defer gpa.free(mine);
        try out.download(std.mem.sliceAsBytes(mine));
        try expectFeatures(gpa, case, mine, want, newline, separator);
    }
    {
        const images = [_]vision.Image{ try imageOf(&refs[0], 0), try imageOf(&refs[1], 0), try imageOf(&refs[1], 1) };
        var total: usize = 0;
        for (images) |image| total += image.rows();
        var out: cuda.Buffer = try .init(.bf16, &.{ total, vision.lm_hidden });
        defer out.deinit();
        try tower.encode(gpa, &images, out.view);
        const together = try gpa.alloc(u16, total * vision.lm_hidden);
        defer gpa.free(together);
        try out.download(std.mem.sliceAsBytes(together));
        var at: usize = 0;
        for (images) |image| {
            var alone: cuda.Buffer = try .init(.bf16, &.{ image.rows(), vision.lm_hidden });
            defer alone.deinit();
            try tower.encode(gpa, &.{image}, alone.view);
            const mine = try gpa.alloc(u16, image.rows() * vision.lm_hidden);
            defer gpa.free(mine);
            try alone.download(std.mem.sliceAsBytes(mine));
            try testing.expectEqualSlices(u16, mine, together[at * vision.lm_hidden ..][0..mine.len]);
            at += image.rows();
        }
    }
    // a crop alone is the bits of the same crop in a run of six
    {
        const image = try imageOf(&refs[0], 0);
        const per = 3 * Size.crop.side * Size.crop.side;
        const rows = Size.crop.outTokens() * vision.lm_hidden;
        try tower.putPixels(Size.crop, image.crops);
        try tower.runSam(Size.crop, image.cropCount());
        try tower.runClip(Size.crop, image.cropCount());
        const run = try tower.runProject(Size.crop, image.cropCount());
        const all = try gpa.alloc(u16, run.count());
        defer gpa.free(all);
        try cuda.download(run, std.mem.sliceAsBytes(all));
        for ([_]usize{ 0, image.cropCount() - 1 }) |i| {
            try tower.putPixels(Size.crop, image.crops[i * per ..][0..per]);
            try tower.runSam(Size.crop, 1);
            try tower.runClip(Size.crop, 1);
            const one = try tower.runProject(Size.crop, 1);
            const mine = try gpa.alloc(u16, rows);
            defer gpa.free(mine);
            try cuda.download(one, std.mem.sliceAsBytes(mine));
            try testing.expectEqualSlices(u16, all[i * rows ..][0..rows], mine);
        }
    }
    try passed();
}
