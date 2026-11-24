//! An image prompt made ready for the tower and the decoder from decoded RGB, on the host: what TensorFold
//! 0.6.5's `vision/qwen_processing.py` (`QwenImageProcessor.prepare`, `media_positions`, `_processor_options`)
//! and `vision/images.py` (`ImageInput.content_hash`) do, with transformers 5.18's `Qwen2VLImageProcessor`
//! under them (`smart_resize`, the resize, the patches). Spec 019 WS-01. The shapes and names are TensorFold
//! 1.0.12's Zig (`core/qwen_vl/geometry.zig`, `rope.zig`); the values are 0.6.5's, which they are not:
//!
//! - The resize is torchvision's, not Pillow's: transformers 5.18's processor is a torch backend, and on the CPU
//!   uint8 RGB goes through torch's antialiased bicubic on integers (`picture.resample`'s `.torch`). 1.0.12 uses
//!   `.pillow`, which differs by 1 on a few bytes.
//! - The normalization is torchvision's fused one, `(f32(x) - mean / rescale) / (std / rescale)` in f32, which
//!   is `(x - 127.5) / 127.5` here. 1.0.12's table rounds `x * rescale` first: 128 of the 256 values differ.
//! - The least pixels an image is given are `factor`² (1,024: 0.6.5's `prepare`, the processor having no
//!   `min_pixels` once transformers moves it into `size`), not the checkpoint's `shortest_edge` 65,536: no
//!   image is refused for being small; a 1 x 1 picture becomes a 2 x 2 patch grid, one token.
//! - An image's hash is 0.6.5's `content_hash`, sha256 of `tensorfold-rgb-v1\0`, the width and height
//!   big-endian and the RGB, in hex; 1.0.12's is 62 bits of the encoded bytes and the grid.
//!
//! Refusals carry 0.6.5's words. A decoded picture's limits (8,192 a side, 16 Mi pixels; 32 Mi a request) are
//! the decoder's (`zig/src/image/`, `picture.fits`) and the request path's (spec 019 WS-04), not this file's.
//! 1.0.12's `picture.identity` is not 0.6.5's `content_hash`, so the hash is written here.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Value = std.json.Value;
const picture = @import("picture");
const Tokenizer = @import("tokenizer.zig").Tokenizer;

/// `prepare`'s refusals: `ContextLength` is OpenAI's `context_length_exceeded`, the rest a request's error.
pub const Error = error{ OutOfMemory, BadRequest, ContextLength };

/// Why an image prompt or a checkpoint's settings were refused: TensorFold 0.6.5's `ValueError` text.
pub const Problem = struct {
    buffer: [320]u8 = undefined,
    len: usize = 0,

    pub fn message(p: *const Problem) []const u8 {
        return p.buffer[0..p.len];
    }

    fn fail(p: *Problem, err: anytype, comptime format: []const u8, args: anytype) @TypeOf(err) {
        var writer: Io.Writer = .fixed(&p.buffer);
        writer.print(format, args) catch {};
        p.len = writer.end;
        return err;
    }
};

/// What the image path takes from a checkpoint: config.json's tower geometry and marker ids, and the processor
/// files' normalization (`_processor_options`).
pub const Settings = struct {
    patch: u32,
    temporal: u32,
    merge: u32,
    image_token: u32,
    video_token: ?u32,
    vision_start: u32,
    vision_end: u32,
    rescale: f64 = 1.0 / 255.0,
    mean: [3]f32 = @splat(0.5),
    std: [3]f32 = @splat(0.5),

    /// Image sides are multiples of this many pixels.
    pub fn factor(s: Settings) u32 {
        return s.patch * s.merge;
    }

    /// A patch row's values: channel by frame by row by column.
    pub fn patchWidth(s: Settings) u32 {
        return 3 * s.temporal * s.patch * s.patch;
    }

    /// config.json, preprocessor_config.json and processor_config.json from `dir`; null for a checkpoint whose
    /// config has no `vision_config`.
    pub fn load(arena: Allocator, io: Io, dir: Io.Dir, problem: *Problem) (Error || error{InvalidSettings})!?Settings {
        const limit: Io.Limit = .limited(1 << 22);
        const config = dir.readFileAlloc(io, "config.json", arena, limit) catch return problem.fail(error.InvalidSettings, "config.json could not be read", .{});
        var texts: [2]?[]const u8 = .{ null, null };
        for ([_][]const u8{ "preprocessor_config.json", "processor_config.json" }, &texts) |name, *text| {
            text.* = dir.readFileAlloc(io, name, arena, limit) catch |err| switch (err) {
                error.FileNotFound => null,
                else => return problem.fail(error.InvalidSettings, "{s} could not be read", .{name}),
            };
        }
        return parse(arena, config, texts[0], texts[1], problem);
    }

    /// The settings from the files' texts (`preprocessor` read whole, `processor`'s `image_processor`, later
    /// keys over earlier ones, as 0.6.5 merges them).
    pub fn parse(arena: Allocator, config_json: []const u8, preprocessor_json: ?[]const u8, processor_json: ?[]const u8, problem: *Problem) error{ OutOfMemory, InvalidSettings }!?Settings {
        const bad = error.InvalidSettings;
        const doc = std.json.parseFromSliceLeaky(Value, arena, config_json, .{}) catch return problem.fail(bad, "config.json is not JSON", .{});
        if (doc != .object) return problem.fail(bad, "config.json is not an object", .{});
        const vc = doc.object.get("vision_config") orelse return null;
        if (vc != .object or vc.object.count() == 0) return null;
        const kind = doc.object.get("model_type") orelse Value.null;
        if (kind != .string or !(std.mem.eql(u8, kind.string, "qwen3_5") or std.mem.eql(u8, kind.string, "qwen4_exp"))) {
            return problem.fail(bad, "Image preprocessing currently supports Qwen3.5/3.8 dense and Flash Next multimodal checkpoints only", .{});
        }
        var s: Settings = .{ .patch = 0, .temporal = 0, .merge = 0, .image_token = 0, .video_token = null, .vision_start = 0, .vision_end = 0 };
        inline for (.{ .{ "patch_size", "patch" }, .{ "temporal_patch_size", "temporal" }, .{ "spatial_merge_size", "merge" } }) |kv| {
            @field(s, kv[1]) = positive(vc.object.get(kv[0])) orelse return problem.fail(bad, "vision_config's " ++ kv[0] ++ " must be a positive integer", .{});
        }
        if (vc.object.get("in_channels")) |v| if (positive(v) != 3) return problem.fail(bad, "vision_config's in_channels must be 3", .{});
        inline for (.{ .{ "image_token_id", "image_token" }, .{ "vision_start_token_id", "vision_start" }, .{ "vision_end_token_id", "vision_end" } }) |kv| {
            @field(s, kv[1]) = positive(doc.object.get(kv[0])) orelse return problem.fail(bad, "config.json's " ++ kv[0] ++ " must be a positive integer", .{});
        }
        if (doc.object.get("video_token_id")) |v| s.video_token = positive(v);
        var raw: std.json.ObjectMap = .empty;
        for ([_]?[]const u8{ preprocessor_json, processor_json }, 0..) |text, i| if (text) |t| {
            const p = std.json.parseFromSliceLeaky(Value, arena, t, .{}) catch return problem.fail(bad, "an image processor file is not JSON", .{});
            if (p != .object) return problem.fail(bad, "an image processor file is not an object", .{});
            const ip = if (i == 0) p else p.object.get("image_processor") orelse continue;
            if (ip != .object) return problem.fail(bad, "processor_config.json's image_processor is not an object", .{});
            var it = ip.object.iterator();
            while (it.next()) |e| try raw.put(arena, e.key_ptr.*, e.value_ptr.*);
        };
        inline for (.{ .{ "patch_size", s.patch }, .{ "temporal_patch_size", s.temporal }, .{ "merge_size", s.merge } }) |kv| {
            if (raw.get(kv[0])) |v| if (positive(v) != kv[1]) return problem.fail(bad, "Image processor " ++ kv[0] ++ " disagrees with the vision tower", .{});
        }
        inline for (.{ .{ "image_mean", "mean" }, .{ "image_std", "std" } }) |kv| if (raw.get(kv[0])) |v| {
            if (v != .array or v.array.items.len != 3) return problem.fail(bad, "Image processor " ++ kv[0] ++ " must be three numbers", .{});
            for (v.array.items, &@field(s, kv[1])) |x, *out| out.* = @floatCast(number(x) orelse return problem.fail(bad, "Image processor " ++ kv[0] ++ " must be three numbers", .{}));
        };
        if (raw.get("rescale_factor")) |v| s.rescale = number(v) orelse 0;
        // transformers fuses the rescale into the normalization only when both are on, which every Qwen checkpoint
        // ships; the other three ways are not written
        inline for (.{ "do_rescale", "do_normalize" }) |key| if (raw.get(key)) |v| if (v != .bool or !v.bool) {
            return problem.fail(bad, "Image processor " ++ key ++ " must be true: only the fused rescale and normalization is written", .{});
        };
        for (s.std) |x| if (!(x > 0)) return problem.fail(bad, "Image processor image_std must be positive", .{});
        if (!(s.rescale > 0)) return problem.fail(bad, "Image processor rescale_factor must be positive", .{});
        return s;
    }

    /// The tokenizer's text for `image_token`, checked to encode back to it (0.6.5's `__init__`).
    pub fn imageText(s: Settings, tokenizer: *const Tokenizer, problem: *Problem) error{InvalidSettings}![]const u8 {
        const text = tokenizer.tokenBytes(s.image_token);
        if (text.len == 0) return problem.fail(error.InvalidSettings, "The local tokenizer does not define the checkpoint's image token", .{});
        if (tokenizer.addedId(text) != s.image_token) return problem.fail(error.InvalidSettings, "The local tokenizer's image token does not match the checkpoint", .{});
        return text;
    }
};

fn positive(v: ?Value) ?u32 {
    const x = v orelse return null;
    if (x != .integer or x.integer <= 0 or x.integer > std.math.maxInt(u32)) return null;
    return @intCast(x.integer);
}

fn number(v: Value) ?f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

pub const Size = struct { h: u32, w: u32 };

/// transformers' `smart_resize`: sides multiples of `factor` (Python's round, halves to even), the area within
/// the bounds; a side past 200 times the other refused with its words.
pub fn smartResize(height: u32, width: u32, factor: u32, min_pixels: u64, max_pixels: u64, problem: *Problem) error{BadRequest}!Size {
    const h: f64 = @floatFromInt(height);
    const w: f64 = @floatFromInt(width);
    const ratio = @max(h, w) / @min(h, w);
    if (ratio > 200) {
        var repr: [32]u8 = undefined;
        return problem.fail(error.BadRequest, "absolute aspect ratio must be smaller than 200, got {s}", .{pyFloat(&repr, ratio)});
    }
    const f: f64 = @floatFromInt(factor);
    var hb = roundEven(h / f) * f;
    var wb = roundEven(w / f) * f;
    if (hb * wb > @as(f64, @floatFromInt(max_pixels))) {
        const beta = @sqrt(h * w / @as(f64, @floatFromInt(max_pixels)));
        hb = @max(f, @floor(h / beta / f) * f);
        wb = @max(f, @floor(w / beta / f) * f);
    } else if (hb * wb < @as(f64, @floatFromInt(min_pixels))) {
        const beta = @sqrt(@as(f64, @floatFromInt(min_pixels)) / (h * w));
        hb = @ceil(h * beta / f) * f;
        wb = @ceil(w * beta / f) * f;
    }
    return .{ .h = @intFromFloat(hb), .w = @intFromFloat(wb) };
}

/// Python's round: a half goes to the even neighbour.
pub fn roundEven(x: f64) f64 {
    const r = @round(x);
    return if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r;
}

/// Python's `repr` of a float of at most 16 digits before the point: the shortest digits, `.0` when whole.
pub fn pyFloat(buffer: *[32]u8, x: f64) []const u8 {
    const text = std.fmt.bufPrint(buffer, "{d}", .{x}) catch return "nan";
    if (std.mem.findAny(u8, text, ".en") != null) return text;
    buffer[text.len] = '.';
    buffer[text.len + 1] = '0';
    return buffer[0 .. text.len + 2];
}

/// How a request asks for an image: `low` holds it to 256 tokens.
pub const Detail = enum { auto, low, high };

/// One decoded picture: RGB rows, three bytes a pixel.
pub const Image = struct {
    width: u32,
    height: u32,
    rgb: []const u8,
    detail: Detail = .auto,

    /// A decoded picture (`zig/src/image/`'s `decode`) as the request's detail asks for it; it keeps `rgb`.
    pub fn of(rgb: picture.Rgb, detail: Detail) Image {
        return .{ .width = std.math.cast(u32, rgb.width) orelse 0, .height = std.math.cast(u32, rgb.height) orelse 0, .rgb = rgb.pixels, .detail = detail };
    }

    /// 0.6.5's `ImageInput` check.
    pub fn valid(img: Image) bool {
        return img.width > 0 and img.height > 0 and img.rgb.len == @as(u64, img.width) * img.height * 3;
    }

    /// 0.6.5's `content_hash`: sha256 of `tensorfold-rgb-v1\0`, the width and the height big-endian, the RGB.
    pub fn contentHash(img: Image) [64]u8 {
        var h: std.crypto.hash.sha2.Sha256 = .init(.{});
        h.update("tensorfold-rgb-v1\x00");
        var sides: [8]u8 = undefined;
        std.mem.writeInt(u32, sides[0..4], img.width, .big);
        std.mem.writeInt(u32, sides[4..8], img.height, .big);
        h.update(&sides);
        h.update(img.rgb);
        return std.fmt.bytesToHex(h.finalResult(), .lower);
    }
};

/// What a request's images share: `max_visual_tokens` (`--vision-image-tokens`, 4,096), each image at most
/// `max_image_tokens` (`TOKENS_PER_IMAGE`, 4,096), and the expanded prompt at most `max_prompt_tokens`.
pub const Budget = struct {
    max_visual_tokens: u32 = 4096,
    max_image_tokens: u32 = 4096,
    max_prompt_tokens: ?u32 = null,
};

/// An image's pixel bounds, as 0.6.5's `prepare` hands them to the processor.
pub fn bounds(s: Settings, images: usize, detail: Detail, budget: Budget) [2]u64 {
    const per = @as(u64, s.factor()) * s.factor();
    var limit: u64 = budget.max_visual_tokens / images;
    if (budget.max_image_tokens != 0) limit = @min(limit, budget.max_image_tokens);
    const per_image = limit * per;
    const cap = if (detail == .low) @min(per_image, 256 * per) else per_image;
    return .{ @min(per, cap), cap };
}

/// `img` resized as the processor resizes it: its new size and RGB, which the caller frees.
pub fn resized(a: Allocator, s: Settings, img: Image, images: usize, budget: Budget, problem: *Problem) Error!struct { Size, []u8 } {
    const b = bounds(s, images, img.detail, budget);
    const size = try smartResize(img.height, img.width, s.factor(), b[0], b[1], problem);
    const out = try picture.resample.resize(a, .torch, .of(img.rgb, img.width, img.height), size.w, size.h);
    return .{ size, out };
}

/// The normalized value of each byte on each channel: torchvision's fused rescale and normalization.
pub fn table(s: Settings) [3][256]f32 {
    var lut: [3][256]f32 = undefined;
    const inverse: f32 = @floatCast(1.0 / s.rescale);
    for (&lut, 0..) |*row, c| {
        const mean = s.mean[c] * inverse;
        const sd = s.std[c] * inverse;
        for (row, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(i)) - mean) / sd;
    }
    return lut;
}

/// Resized RGB (`w` x `h`, multiples of the factor) as patch rows in merge order (block row, block column, row,
/// column), each [channel, frame, y, x] with the one frame repeated, appended to `out`.
pub fn patchify(s: Settings, lut: *const [3][256]f32, rgb: []const u8, size: Size, out: []f32) void {
    const P = s.patch;
    const M = s.merge;
    const gh = size.h / P;
    const gw = size.w / P;
    const width = s.patchWidth();
    std.debug.assert(out.len == @as(usize, gh) * gw * width);
    var n: usize = 0;
    for (0..gh / M) |by| for (0..gw / M) |bx| for (0..M) |my| for (0..M) |mx| {
        const y0 = (by * M + my) * P;
        const x0 = (bx * M + mx) * P;
        const dst = out[n * width ..][0..width];
        for (0..3) |c| for (0..s.temporal) |t| for (0..P) |dy| for (0..P) |dx| {
            dst[((c * s.temporal + t) * P + dy) * P + dx] = lut[c][rgb[((y0 + dy) * size.w + x0 + dx) * 3 + c]];
        };
        n += 1;
    };
}

/// One image's rows from `at`: `rows` x `cols` merged places, all at position `base` plus their row and column.
pub const Span = struct {
    at: u32,
    rows: u32,
    cols: u32,
    base: i32,

    pub fn end(s: Span) u32 {
        return s.at + s.rows * s.cols;
    }
};

/// The marker ids around and in an image's rows.
pub const Ids = struct { start: u32, image: u32, end: u32, video: ?u32 = null };

/// 0.6.5's `media_positions` (transformers' `get_rope_index`) for images: text counts up on all three axes; an
/// image sits at the next position with its rows and columns on h and w; the text after it resumes past its
/// larger side. A row past every image is at its index plus `shift` (decode rows, drafts, later turns).
pub const Layout = struct {
    spans: []Span = &.{},
    shift: i32 = 0,

    /// The layout of `tokens` for `grids` (t, h, w in patches, prompt order), every placeholder run checked.
    pub fn init(a: Allocator, tokens: []const u32, grids: []const [3]u32, ids: Ids, merge: u32, problem: *Problem) Error!Layout {
        const bad = error.BadRequest;
        if (ids.video) |v| if (std.mem.findScalar(u32, tokens, v) != null) return problem.fail(bad, "Video inputs are not supported by the Qwen image frontend", .{});
        const spans = try a.alloc(Span, grids.len);
        errdefer a.free(spans);
        var used: usize = 0;
        var cursor: usize = 0;
        var next: i64 = 0;
        while (std.mem.findScalarPos(u32, tokens, cursor, ids.image)) |begin| {
            if (used >= grids.len) return problem.fail(bad, "The prompt contains image tokens without a corresponding image", .{});
            const g = grids[used];
            if (g[0] != 1 or g[1] == 0 or g[2] == 0 or merge == 0 or g[1] % merge != 0 or g[2] % merge != 0) {
                return problem.fail(bad, "An image grid must contain one frame and merge-aligned positive dimensions", .{});
            }
            const rows = g[1] / merge;
            const cols = g[2] / merge;
            const stop = begin + @as(usize, rows) * cols;
            if (begin == 0 or tokens[begin - 1] != ids.start or stop >= tokens.len or tokens[stop] != ids.end or
                !std.mem.allEqual(u32, tokens[begin..stop], ids.image))
            {
                return problem.fail(bad, "Image placeholders must match the processed image grid exactly", .{});
            }
            const base = next + @as(i64, @intCast(begin - cursor));
            spans[used] = .{ .at = @intCast(begin), .rows = rows, .cols = cols, .base = @intCast(base) };
            used += 1;
            next = base + @max(rows, cols);
            cursor = stop;
        }
        if (used < grids.len) return problem.fail(bad, "Image grid has no matching image tokens in the prompt", .{});
        return .{ .spans = spans, .shift = @intCast(next - @as(i64, @intCast(cursor))) };
    }

    pub fn deinit(l: *Layout, a: Allocator) void {
        a.free(l.spans);
        l.* = .{};
    }

    /// Row `p`'s (t, h, w): an image row's place in its grid, any other row its index shifted past earlier images.
    pub fn position(l: *const Layout, p: u32) [3]i32 {
        var shift: i32 = 0;
        for (l.spans) |s| {
            if (p < s.at) break;
            const k = p - s.at;
            if (k < s.rows * s.cols) return .{ s.base, s.base + @as(i32, @intCast(k / s.cols)), s.base + @as(i32, @intCast(k % s.cols)) };
            shift = s.base + @as(i32, @intCast(@max(s.rows, s.cols))) - @as(i32, @intCast(s.end()));
        }
        const t = @as(i32, @intCast(p)) + shift;
        return .{ t, t, t };
    }

    /// The positions of rows [0, n) on the three axes, as 0.6.5's `position_ids` [3, 1, n].
    pub fn fill(l: *const Layout, out: [3][]i32) void {
        for (0..out[0].len) |p| {
            const x = l.position(@intCast(p));
            for (out, x) |axis, v| axis[p] = v;
        }
    }

    /// Whether rows [first, first + n) hold an image row.
    pub fn holds(l: *const Layout, first: u32, n: u32) bool {
        for (l.spans) |s| if (s.at < first + n and first < s.end()) return true;
        return false;
    }
};

/// An image prompt ready for the tower and the decoder. Every slice is the allocator's that `prepare` was given.
pub const Prepared = struct {
    ids: []u32,
    /// [rows, `patchWidth`] f32: every image's patch rows, in prompt order.
    pixel_values: []f32,
    /// (t, h, w) in patches, an image a row.
    grids: [][3]u32,
    layout: Layout,
    hashes: [][64]u8,

    pub fn deinit(p: *Prepared, a: Allocator) void {
        a.free(p.ids);
        a.free(p.pixel_values);
        a.free(p.grids);
        p.layout.deinit(a);
        a.free(p.hashes);
    }

    /// The decoder's offset for rows past the prompt (0.6.5's `rope_delta`).
    pub fn delta(p: *const Prepared) i32 {
        return p.layout.shift;
    }

    /// The image tokens' count.
    pub fn visualTokens(p: *const Prepared) usize {
        var n: usize = 0;
        for (p.layout.spans) |s| n += s.rows * s.cols;
        return n;
    }
};

/// 0.6.5's `QwenImageProcessor.prepare` for images: each image marker of `rendered` (the template's output, the
/// image token's text once an image) expanded to the image's tokens, the result tokenized, its positions.
pub fn prepare(a: Allocator, s: Settings, tokenizer: *const Tokenizer, rendered: []const u8, images: []const Image, budget: Budget, problem: *Problem) Error!Prepared {
    const bad = error.BadRequest;
    if (images.len == 0 or budget.max_visual_tokens < 1) return problem.fail(bad, "Image preprocessing needs images and a positive visual-token budget", .{});
    for (images) |img| if (!img.valid()) return problem.fail(bad, "image input requires RGB bytes, matching dimensions and valid detail", .{});
    const marker = s.imageText(tokenizer, problem) catch return bad;
    if (std.mem.count(u8, rendered, marker) != images.len) return problem.fail(bad, "The rendered prompt must contain exactly one image marker for each image", .{});
    if (images.len > budget.max_visual_tokens) return problem.fail(bad, "The image count exceeds the visual-token budget", .{});
    const sizes = try a.alloc(Size, images.len);
    defer a.free(sizes);
    var rows: usize = 0;
    var tokens: usize = 0;
    for (images, sizes) |img, *size| {
        const b = bounds(s, images.len, img.detail, budget);
        size.* = try smartResize(img.height, img.width, s.factor(), b[0], b[1], problem);
        const patches = @as(usize, size.h / s.patch) * (size.w / s.patch);
        rows += patches;
        tokens += patches / (s.merge * s.merge);
    }
    if (tokens > budget.max_visual_tokens) return problem.fail(bad, "Processed images exceed the visual-token budget; reduce image resolution or count", .{});

    var text: std.ArrayList(u8) = .empty;
    defer text.deinit(a);
    var rest = rendered;
    for (sizes) |size| {
        const at = std.mem.find(u8, rest, marker).?;
        try text.appendSlice(a, rest[0..at]);
        const count = @as(usize, size.h / s.factor()) * (size.w / s.factor());
        try text.ensureUnusedCapacity(a, marker.len * count);
        for (0..count) |_| text.appendSliceAssumeCapacity(marker);
        rest = rest[at + marker.len ..];
    }
    try text.appendSlice(a, rest);
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(a);
    tokenizer.encode(a, text.items, &ids) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidUtf8 => return problem.fail(bad, "the rendered prompt is not UTF-8", .{}),
    };
    if (budget.max_prompt_tokens) |most| if (ids.items.len > most) {
        return problem.fail(error.ContextLength, "This server's maximum context length is {d} tokens: the expanded image prompt has {d} tokens, which exceeds the context window; reduce image resolution or prompt length", .{ most, ids.items.len });
    };

    const grids = try a.alloc([3]u32, images.len);
    errdefer a.free(grids);
    for (sizes, grids) |size, *g| g.* = .{ 1, size.h / s.patch, size.w / s.patch };
    var layout = try Layout.init(a, ids.items, grids, .{ .start = s.vision_start, .image = s.image_token, .end = s.vision_end, .video = s.video_token }, s.merge, problem);
    errdefer layout.deinit(a);

    const pixels = try a.alloc(f32, rows * s.patchWidth());
    errdefer a.free(pixels);
    const hashes = try a.alloc([64]u8, images.len);
    errdefer a.free(hashes);
    const lut = table(s);
    var row: usize = 0;
    for (images, sizes, hashes) |img, size, *hash| {
        const rgb = try picture.resample.resize(a, .torch, .of(img.rgb, img.width, img.height), size.w, size.h);
        defer a.free(rgb);
        const n = @as(usize, size.h / s.patch) * (size.w / s.patch) * s.patchWidth();
        patchify(s, &lut, rgb, size, pixels[row..][0..n]);
        row += n;
        hash.* = img.contentHash();
    }
    return .{ .ids = try ids.toOwnedSlice(a), .pixel_values = pixels, .grids = grids, .layout = layout, .hashes = hashes };
}
