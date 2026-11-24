//! Unlimited-OCR-2's host preprocessing of a picture and its prompt, as the checkpoint's own HF code builds a
//! request (`modeling_unlimitedocr.py` `infer` and `_uocr_build_inputs`, vLLM's processor as its doc says; spec
//! 147 WS-06): the views a vision tower reads, the image tokens' layout and the prompt's ids.
//!
//! - `pagePrompt` writes `<PAGE><image>` a page before the instruction ("Multi page parsing." where none is
//!   given); `preparePages` cuts every `<image>` of it, one picture each, gundam for one picture and base for
//!   several unless told (`Mode`), with at most 16 tiles (`max_tiles`). The prompt is not stripped.
//! - Gundam, a picture: both sides at most 640 give the global view alone; any other picture is cut by
//!   `dynamic_preprocess` (2 to 16 tiles of 640, `find_closest_aspect_ratio`'s tie-break) as well. The global view
//!   is `ImageOps.pad` to 1024 on gray 127. The tiles are the whole picture resized to (640 x w, 640 x h) and cut
//!   in rows; there is no thumbnail.
//! - Base, any number of pages: each page is the global view alone, its 273 tokens one after the other at its
//!   `<image>`.
//! - A view is `ToTensor` then `Normalize(0.5, 0.5)` then `.to(bfloat16)`, channels first.
//! - Ids: BOS, the prompt's pieces, the image tokens (id 128815) after each.
//!
//! The resize and the pad are Pillow's, bit for bit (`image/resize.zig`). The pictures are `image.decode`'s: an
//! alpha channel is composited over white as vLLM's Pillow loader does, where the HF code's `convert("RGB")`
//! drops it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const resize_mod = @import("../../image/resize.zig");
const text_tokenizer = @import("../../text/tokenizer.zig");

pub const bos_id: u32 = 0;
pub const image_token_id: u32 = 128815;
pub const image_tag = "<image>";
/// The global view's side (`base_size`) and a tile's (`image_size`).
pub const global_side = 1024;
pub const tile_side = 640;
/// `ImageOps.pad`'s color: `int(0.5 * 255)`.
pub const pad_fill: u8 = 127;
pub const channels = 3;
const min_tiles = 2;
/// The most tiles a picture is cut into in gundam mode (`_uocr_build_inputs`, spec 147).
pub const max_tiles = 16;
/// `<PAGE>` (`sparse_attention_config.page_token_id`): a page's marker in the prompt and reply.
pub const page_token_id: u32 = 128827;
pub const page_tag = "<PAGE>";
/// v2's instruction where a request gives none (`infer`'s default prompt).
pub const pages_instruction = "Multi page parsing.";
/// The tokens of a view of `global_side`: `((1024 / 16 / 4) + 1)` per row, 16 rows, one separator.
pub const global_tokens = 273;
/// A tile's queries a side: `ceil((640 / 16) / 4)`.
const tile_queries = 10;

pub const Mode = enum {
    /// One picture, cut into tiles beside its global view.
    gundam,
    /// One or more pages, each a global view alone.
    base,
};

/// A decoded picture: RGB rows, three bytes a pixel.
pub const Picture = struct { width: usize, height: usize, rgb: []const u8 };

/// `images_spatial_crop`: the tiles across and down.
pub const Grid = struct { w: u32, h: u32 };

pub const Error = error{ ImagePlaceholder, PageCount, EmptyPicture } || Allocator.Error || text_tokenizer.EncodeError;

/// The tokens a picture of this grid takes in the prompt.
pub fn imageTokens(grid: Grid) usize {
    const tiles: usize = if (grid.w > 1 or grid.h > 1) @as(usize, tile_queries) * grid.h * (tile_queries * grid.w + 1) else 0;
    return global_tokens + tiles;
}

const Ratio = struct { w: u32, h: u32 };

/// `dynamic_preprocess`'s candidates, `(i, j)` with `2 <= i * j <= max_tiles`, in the order of area.
/// Python sorts a set of them; ties of area keep the set's order there and the lexicographic one here,
/// which no choice can reach: two
/// ratios of one area are equally far from the picture's only where the picture is midway between them, and a
/// ratio of another area then lies nearer.
const ratios = ratios: {
    @setEvalBranchQuota(100_000);
    var all: [max_tiles * max_tiles]Ratio = undefined;
    var n = 0;
    for (1..max_tiles + 1) |i| for (1..max_tiles + 1) |j| {
        if (i * j < min_tiles or i * j > max_tiles) continue;
        all[n] = .{ .w = i, .h = j };
        n += 1;
    };
    var sorted: [n]Ratio = all[0..n].*;
    std.mem.sort(Ratio, &sorted, {}, struct {
        fn less(_: void, a: Ratio, b: Ratio) bool {
            return a.w * a.h < b.w * b.h;
        }
    }.less);
    break :ratios sorted;
};

/// `find_closest_aspect_ratio(width / height, ratios, width, height, 640)` over the ratios of at most `most` tiles.
pub fn closestGrid(width: usize, height: usize, most: u32) Grid {
    const aspect = @as(f64, @floatFromInt(width)) / @as(f64, @floatFromInt(height));
    const area: f64 = @floatFromInt(width * height);
    var best_diff = std.math.inf(f64);
    var best: Ratio = .{ .w = 1, .h = 1 };
    for (ratios) |r| {
        if (r.w * r.h > most) break;
        const diff = @abs(aspect - @as(f64, @floatFromInt(r.w)) / @as(f64, @floatFromInt(r.h)));
        if (diff < best_diff) {
            best_diff = diff;
            best = r;
        } else if (diff == best_diff) {
            if (area > 0.5 * tile_side * tile_side * @as(f64, @floatFromInt(r.w)) * @as(f64, @floatFromInt(r.h))) best = r;
        }
    }
    return .{ .w = best.w, .h = best.h };
}

/// The grid `infer` cuts a picture into in gundam mode, of at most `max_tiles` tiles: 1 x 1 (the global view
/// alone) where both sides are at most 640.
pub fn gundamGrid(width: usize, height: usize) Grid {
    if (width <= tile_side and height <= tile_side) return .{ .w = 1, .h = 1 };
    return closestGrid(width, height, max_tiles);
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}

/// `Normalize((0.5,) * 3, (0.5,) * 3)(ToTensor()(view)).to(bfloat16)`: `rgb` is `side x side` HWC bytes, `out`
/// the three channel planes. float32 throughout, as torch's: `x / 255`, minus 0.5, over 0.5, rounded to nearest even.
pub fn normalize(rgb: []const u8, out: []u16) void {
    const plane = out.len / channels;
    std.debug.assert(rgb.len == out.len);
    for (0..plane) |p| for (0..channels) |c| {
        const x: f32 = @floatFromInt(rgb[p * channels + c]);
        out[c * plane + p] = bf16((x / 255.0 - 0.5) / 0.5);
    };
}

/// What the vision tower reads and what the language model is given.
pub const Prepared = struct {
    /// The prompt's ids and, for each, whether it is an image token (`images_seq_mask`).
    ids: []u32,
    seq_mask: []bool,
    /// One grid for each picture (`images_spatial_crop`).
    grids: []Grid,
    /// The global views, one for each picture: `[n][1024][1024][3]` bytes, then normalized `[n][3][1024][1024]` bf16.
    global_u8: []u8,
    global: []u16,
    /// The tiles of every picture in order, each row-major: `[tiles][640][640][3]`, then `[tiles][3][640][640]`.
    crops_u8: []u8,
    crops: []u16,

    pub fn deinit(self: *Prepared, gpa: Allocator) void {
        gpa.free(self.ids);
        gpa.free(self.seq_mask);
        gpa.free(self.grids);
        gpa.free(self.global_u8);
        gpa.free(self.global);
        gpa.free(self.crops_u8);
        gpa.free(self.crops);
        self.* = undefined;
    }
};

/// One picture's views under `grid`: its global view normalized into `global` (bf16 [3, 1024, 1024]) and its tiles,
/// row-major, into `crops` (bf16 [w * h, 3, 640, 640], empty where the grid is 1 x 1); the bytes before they were
/// normalized into `global_u8` and `crops_u8` where given. What `prepare` makes of each picture, and `serve` of a
/// request's (`served.zig`).
pub fn views(gpa: Allocator, p: Picture, grid: Grid, global: []u16, crops: []u16, global_u8: ?[]u8, crops_u8: ?[]u8) Allocator.Error!void {
    const view = global_side * global_side * channels;
    const tile = tile_side * tile_side * channels;
    std.debug.assert(global.len == view);
    const padded = try resize_mod.pad(gpa, p.rgb, p.width, p.height, global_side, global_side, pad_fill);
    defer gpa.free(padded);
    if (global_u8) |out| @memcpy(out, padded);
    normalize(padded, global);
    if (grid.w == 1 and grid.h == 1) return;
    // `dynamic_preprocess`: the picture to (640 w, 640 h), cut into rows of tiles
    const gw: usize = grid.w;
    std.debug.assert(crops.len == gw * grid.h * tile);
    const whole = try resize_mod.resize(gpa, p.rgb, p.width, p.height, tile_side * gw, tile_side * grid.h);
    defer gpa.free(whole);
    const bytes = try gpa.alloc(u8, tile);
    defer gpa.free(bytes);
    for (0..gw * grid.h) |t| {
        const out = if (crops_u8) |all| all[t * tile ..][0..tile] else bytes;
        const x0 = (t % gw) * tile_side;
        const y0 = (t / gw) * tile_side;
        for (0..tile_side) |y| {
            const src = whole[((y0 + y) * gw * tile_side + x0) * channels ..][0 .. tile_side * channels];
            @memcpy(out[y * tile_side * channels ..][0 .. tile_side * channels], src);
        }
        normalize(out, crops[t * tile ..][0..tile]);
    }
}

/// The prompt for `pages` pictures as `infer` writes it: `prompt`, or `pages_instruction` where it is empty, with
/// `<PAGE><image>` a page before it where it holds no `<image>`; one `<image>` among several pages becomes
/// `<PAGE><image>` a page; as many `<image>` as pages are kept as they stand. Any other count is
/// `error.ImagePlaceholder`, `infer`'s ValueError. In `gpa`.
pub fn pagePrompt(gpa: Allocator, prompt: []const u8, pages: usize) Error![]u8 {
    const text = if (prompt.len == 0 and pages > 0) pages_instruction else prompt;
    if (text.len == 0) return error.ImagePlaceholder;
    const marks = std.mem.count(u8, text, image_tag);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    if (pages > 0 and marks == 0) {
        for (0..pages) |_| try out.appendSlice(gpa, page_tag ++ image_tag);
        try out.appendSlice(gpa, text);
    } else if (pages > 1 and marks == 1) {
        const at = std.mem.indexOf(u8, text, image_tag).?;
        try out.appendSlice(gpa, text[0..at]);
        for (0..pages) |_| try out.appendSlice(gpa, page_tag ++ image_tag);
        try out.appendSlice(gpa, text[at + image_tag.len ..]);
    } else if (marks == pages) {
        try out.appendSlice(gpa, text);
    } else return error.ImagePlaceholder;
    return out.toOwnedSlice(gpa);
}

/// The views, grids and ids, `_uocr_build_inputs`: `prompt` (`pagePrompt`'s, unstripped) cut at its `<image>`s,
/// one a picture in order, each piece encoded (`<PAGE>` the added token) and each picture's tokens after it. `mode`
/// null is the processor's own choice: gundam for one picture, base for several; a picture of at most 640 a side is
/// its global view alone in either.
pub fn preparePages(gpa: Allocator, tokenizer: *const text_tokenizer.Tokenizer, mode: ?Mode, prompt: []const u8, pictures: []const Picture) Error!Prepared {
    if (pictures.len == 0) return error.PageCount;
    for (pictures) |p| if (p.width == 0 or p.height == 0 or p.rgb.len != p.width * p.height * channels) return error.EmptyPicture;
    if (std.mem.count(u8, prompt, image_tag) != pictures.len) return error.ImagePlaceholder;
    const pieces = try gpa.alloc([]const u8, pictures.len + 1);
    defer gpa.free(pieces);
    var it = std.mem.splitSequence(u8, prompt, image_tag);
    for (pieces) |*piece| piece.* = it.next().?;
    const grids = try gpa.alloc(Grid, pictures.len);
    const cropping = (mode orelse (if (pictures.len == 1) Mode.gundam else Mode.base)) == .gundam;
    for (pictures, grids) |p, *g| g.* = if (cropping) gundamGrid(p.width, p.height) else .{ .w = 1, .h = 1 };
    return build(gpa, tokenizer, pieces, pictures, grids);
}

/// The views of `pictures` under `grids` (which it takes, freed on an error), and the ids: BOS, then for each
/// picture `pieces[i]` encoded and the picture's tokens, then the last piece.
fn build(gpa: Allocator, tokenizer: *const text_tokenizer.Tokenizer, pieces: []const []const u8, pictures: []const Picture, grids: []Grid) Error!Prepared {
    std.debug.assert(pieces.len == pictures.len + 1 and grids.len == pictures.len);
    errdefer gpa.free(grids);
    var tiles: usize = 0;
    for (grids) |g| {
        if (g.w > 1 or g.h > 1) tiles += g.w * g.h;
    }

    const view = global_side * global_side * channels;
    const tile = tile_side * tile_side * channels;
    const global_u8 = try gpa.alloc(u8, pictures.len * view);
    errdefer gpa.free(global_u8);
    const global = try gpa.alloc(u16, pictures.len * view);
    errdefer gpa.free(global);
    const crops_u8 = try gpa.alloc(u8, tiles * tile);
    errdefer gpa.free(crops_u8);
    const crops = try gpa.alloc(u16, tiles * tile);
    errdefer gpa.free(crops);

    var cut: usize = 0;
    for (pictures, grids, 0..) |p, g, n| {
        const count: usize = if (g.w == 1 and g.h == 1) 0 else @as(usize, g.w) * g.h;
        try views(gpa, p, g, global[n * view ..][0..view], crops[cut * tile ..][0 .. count * tile], global_u8[n * view ..][0..view], crops_u8[cut * tile ..][0 .. count * tile]);
        cut += count;
    }

    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var mask: std.ArrayList(bool) = .empty;
    defer mask.deinit(gpa);
    try ids.append(gpa, bos_id);
    for (pieces, 0..) |piece, n| {
        try tokenizer.encode(gpa, piece, &ids);
        if (n == grids.len) break;
        try mask.appendNTimes(gpa, false, ids.items.len - mask.items.len);
        const rows = imageTokens(grids[n]);
        try ids.appendNTimes(gpa, image_token_id, rows);
        try mask.appendNTimes(gpa, true, rows);
    }
    try mask.appendNTimes(gpa, false, ids.items.len - mask.items.len);
    const seq_mask = try mask.toOwnedSlice(gpa);
    errdefer gpa.free(seq_mask);
    return .{
        .ids = try ids.toOwnedSlice(gpa),
        .seq_mask = seq_mask,
        .grids = grids,
        .global_u8 = global_u8,
        .global = global,
        .crops_u8 = crops_u8,
        .crops = crops,
    };
}
