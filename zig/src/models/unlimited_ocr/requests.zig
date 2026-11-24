//! What a request gives Unlimited-OCR-2 before its reply, on the host (no device): the chat template, the images
//! (`Offer`, TensorFold's `engine_api.Vision`) and the rules each row of the reply obeys (`Rules`). `served.zig`
//! serves through it, `draft.zig`'s `greedy` follows its rules.
//!
//! - Template: the checkpoint has none; zspark's is vLLM's for `unlimited-ocr` with the page prompt of the
//!   checkpoint's `infer` (`chat_template_v2.jinja`: `<PAGE><image>` a page before the instruction, "Multi page
//!   parsing." where none is given).
//! - Images: a request may hold up to `v2_images` (SGLang's wheel serves a request of several, the
//!   decision of 2026-10-27), cut as the HF processor cuts them (`_uocr_build_inputs`, vLLM's auto policy): one
//!   picture in gundam mode (at most 16 tiles), several in base mode, each page its global view alone, padded
//!   straight to 1,024. The host hands `prepare_asked` one image at a time with the request's image count and the
//!   mode it names (the vendored `images.zig` `expandAsked`, `fields.imageMode`: SGLang's `images_config`,
//!   `crop_mode` or `image_mode`, and `mm_processor_kwargs`, spec 147 WS-08). A request that names none keeps
//!   HF's rule, whose fixtures (spec 147 WS-01) are base; SGLang's own default, gundam for every page, is the
//!   mode a request names.
//! - Rules (`Rules`): the no-repeat n-gram rule (`decode/ngram.zig`) then the PageCountGuard
//!   (`decode/page_guard.zig`), applied in that order as v2's `_uocr_generate` does. The n-gram rule reads the
//!   reply alone (`_uocr_no_repeat_ngram`, as vLLM and SGLang's `SGLANG_UOCR_NGRAM_OUTPUT_ONLY`). Neither field
//!   asked is the image rule (35 over 128), a window of 1,024 where the prompt holds several pages (the README's
//!   request, the fixtures'). The guard counts the prompt's `<PAGE>`s, its cap the request's `max_page_tokens` or
//!   20,000; a request that ignores the end token has none.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const images = @import("../../server/images.zig");
const ngram = @import("../../decode/ngram.zig");
const page_guard = @import("../../decode/page_guard.zig");
const masks = @import("../../decode/masks.zig");
const preprocess = @import("preprocess.zig");

/// zspark's template for the checkpoint, which has none (the file's doc comment).
pub const template = @embedFile("chat_template_v2.jinja");

/// The pages a request, and their rows: each page's share holds its largest gundam grid (16 tiles, 1 x 16: 2,033
/// rows), as a request may ask gundam for every page (`images_config`); a document past the context is cut into
/// parts (`documents.zig`).
pub const v2_images = 128;
pub const page_tokens = preprocess.imageTokens(.{ .w = 1, .h = preprocess.max_tiles });
pub const image_tokens = v2_images * page_tokens;

/// What `prepare` hands `reply` in a `PreparedImage`'s pixels: the global view, then the tiles.
pub const global_values = preprocess.global_side * preprocess.global_side * preprocess.channels;
pub const tile_values = preprocess.tile_side * preprocess.tile_side * preprocess.channels;

/// The host's `Info.vision`.
pub const Offer = struct {
    var here: u8 = 0; // the Vision's ctx, unread

    pub fn vision() api.Vision {
        return .{
            .ctx = &here,
            .prepare = prepareOpaque,
            .prepare_asked = askedOpaque,
            .image_token = preprocess.image_token_id,
            .markers = &.{},
            .image_tokens = image_tokens,
            .max_images = v2_images,
            .explain = explain,
        };
    }

    fn prepareOpaque(_: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        _ = max_tokens;
        return prepare(a, bytes, try modeOf(.{ .images = 1 }));
    }

    pub const ModeError = error{ ImageModeUnknown, ImageModeUnserved };

    /// A refusal's words: SGLang's for a mode it does not know (without the value, which `explain` is not given),
    /// zspark's for one SGLang serves and zspark does not; else the server's (`images.explain`).
    pub fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.ImageModeUnknown => "Unknown image_mode. Supported: auto, tiny, small, base, large, gundam",
            error.ImageModeUnserved => "image_mode: this server cuts pages in gundam (auto) or base mode alone; tiny, small and large are not served",
            else => images.explain(err),
        };
    }

    fn askedOpaque(_: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32, asked: api.Asked) anyerror!api.PreparedImage {
        _ = max_tokens;
        return prepare(a, bytes, try modeOf(asked));
    }

    /// The mode a picture is cut in: the mode the request names (`images_config`, SGLang's `_resolve_mode_v2`:
    /// stripped, lower case, `auto` gundam), else HF's, gundam for a picture alone and base for several. zspark
    /// serves gundam and base; SGLang's tiny, small and large are refused.
    pub fn modeOf(asked: api.Asked) ModeError!preprocess.Mode {
        const named = asked.mode orelse return if (asked.images <= 1) .gundam else .base;
        const word = std.mem.trim(u8, named, " \t\n\r\x0b\x0c");
        var buf: [8]u8 = undefined;
        if (word.len > buf.len) return error.ImageModeUnknown;
        const key = std.ascii.lowerString(&buf, word);
        if (std.mem.eql(u8, key, "auto") or std.mem.eql(u8, key, "gundam")) return .gundam;
        if (std.mem.eql(u8, key, "base")) return .base;
        for ([_][]const u8{ "tiny", "small", "large" }) |m| if (std.mem.eql(u8, key, m)) return error.ImageModeUnserved;
        return error.ImageModeUnknown;
    }

    /// The picture in `bytes` as the tower's views in `mode`, in `a`. A picture's rows are its grid's: the host
    /// refuses one that takes more than its budget.
    pub fn prepare(a: Allocator, bytes: []const u8, mode: preprocess.Mode) (images.PrepareError || Allocator.Error)!api.PreparedImage {
        var rgb = try images.open(a, bytes);
        defer rgb.deinit(a);
        if (rgb.width == 0 or rgb.height == 0) return error.Invalid;
        const grid: preprocess.Grid = switch (mode) {
            .gundam => preprocess.gundamGrid(rgb.width, rgb.height),
            .base => .{ .w = 1, .h = 1 },
        };
        const tiles: usize = if (grid.w == 1 and grid.h == 1) 0 else @as(usize, grid.w) * grid.h;
        const views = try a.alloc(u16, global_values + tiles * tile_values);
        defer a.free(views);
        const picture: preprocess.Picture = .{ .width = rgb.width, .height = rgb.height, .rgb = rgb.pixels };
        try preprocess.views(a, picture, grid, views[0..global_values], views[global_values..], null, null);
        const pixels = try a.alloc(f32, views.len);
        for (pixels, views) |*p, v| p.* = @bitCast(@as(u32, v) << 16);
        return .{
            .pixels = pixels,
            .gh = grid.h,
            .gw = grid.w,
            .tokens = @intCast(preprocess.imageTokens(grid)),
            .hash = std.hash.Wyhash.hash(0, rgb.pixels) ^ (@as(u64, grid.w) << 32 | grid.h),
        };
    }
};

/// The n-gram rule a request's `no_repeat` asks, or null for none: neither field given is `default`; a size alone
/// takes `default`'s window, and a window alone is no rule (vLLM's `validate_params`).
pub fn ruleOf(asked: api.NoRepeat, default: ngram.Rule) ?ngram.Rule {
    if (asked.size == null and asked.window != null) return null;
    const size = asked.size orelse default.size;
    const window = asked.window orelse default.window;
    if (size == 0) return null;
    const rule: ngram.Rule = .{ .size = size, .window = window, .whitelist = asked.whitelist };
    return if (rule.valid()) rule else null;
}

/// SGLang's `_tail_has_repeat`: the reply's last `size` tokens repeat `count` times back to back, for a `size` from
/// `max(1, min_size)` to `max_size`.
pub fn tailRepeats(reply: []const u32, rule: api.TailRepeat) bool {
    var size: usize = @max(1, rule.min_size);
    while (size <= rule.max_size) : (size += 1) {
        if (size * rule.min_count > reply.len) return false;
        const tail = reply[reply.len - size ..];
        var m: usize = 1;
        while (m < rule.min_count) : (m += 1) {
            if (!std.mem.eql(u32, reply[reply.len - size * (m + 1) ..][0..size], tail)) break;
        } else return true;
    }
    return false;
}

/// What a reply's rows obey, in order: the tail rule (only the end token where it holds), the n-gram rule, the guard.
pub const Rules = struct {
    /// Read against the reply alone (`_uocr_no_repeat_ngram`), the image rule or several pages' page rule.
    ngram: ?ngram.Rule,
    guard: ?page_guard.Guard = null,
    /// The periodic tail rule (SGLang's `custom_params` `rep_*`): on where `max_size > 0` and `min_count >= 2`.
    tail: api.TailRepeat = .{},
    eos: u32 = 0,

    /// The rules of a request: `prompt` its ids (i32 or u32), `eos` the config's end token, empty `ends` a
    /// request that ignores it.
    pub fn of(request: api.Request, prompt: anytype, eos: u32) Rules {
        const pages = page_guard.pagesIn(prompt, preprocess.page_token_id);
        const cap = request.max_page_tokens orelse page_guard.Guard.default_cap;
        return .{
            .ngram = ruleOf(request.no_repeat, if (pages > 1) ngram.Rule.page else ngram.Rule.image),
            .guard = if (request.eos.len == 0) null else .init(prompt, preprocess.page_token_id, eos, cap),
            .tail = request.tail_repeat,
            .eos = eos,
        };
    }

    /// The rules laid on the row after `seen`, the prompt's `prompt_len` ids then the reply's, its logits bf16 bits.
    pub fn apply(self: *Rules, seen: []const u32, prompt_len: usize, logits: []u16) void {
        const reply = seen[prompt_len..];
        if (self.tail.max_size > 0 and self.tail.min_count >= 2 and self.eos < logits.len and tailRepeats(reply, self.tail)) {
            @memset(logits, masks.minus_infinity);
            logits[self.eos] = 0;
            return;
        }
        if (self.ngram) |r| _ = ngram.apply(r, reply, &.{}, logits);
        if (self.guard) |*g| _ = g.apply(reply, logits);
    }
};
