//! What a model that reads images offers TensorFold's host (`engine_api.Vision`, spec 019 WS-04): an image
//! part's bytes to the tower's input, between the host's render and its encode. The host (the vendored
//! `server/images.zig`, a module of the same name) takes the part's bytes from a `data:` URL or, under
//! `--vision-urls`, a public HTTPS URL, and hands them to `prepare` with the image's token budget; it then writes
//! the image's tokens in place of the template's one placeholder and gives the request the prepared pixels
//! (`Request.images`). Here the bytes are decoded (`image/`: the format sniffed, an animation's first frame, the
//! orientation, alpha over white), resized and patchified as WS-01's `text/vision.zig` does for TensorFold 0.6.5.
//! Both read the same `image_pad` ids: the token is a special added one, so expanding it after the encode gives
//! the ids `vision.prepare`'s re-encode gives.
//!
//! A refusal is vLLM's 400 (`vllm/multimodal/media/image.py` `load_bytes`: `Failed to load image: ...`), in
//! static words `explain` gives the host: the formats `image/` does not decode are named in it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const image = @import("../image/image.zig");
const processor = @import("../text/vision.zig");

/// 0.6.5's budgets (`--vision-image-tokens`, `--vision-max-images`): a request's image tokens and its images.
pub const image_tokens = 4096;
pub const max_images = 4;

/// A checkpoint's image settings as the host reads them; it lives as long as the server and changes not.
pub const Offer = struct {
    settings: processor.Settings,
    lut: [3][256]f32,
    markers: [3]u32,

    pub fn init(settings: processor.Settings) Offer {
        return .{ .settings = settings, .lut = processor.table(settings), .markers = .{ settings.vision_start, settings.image_token, settings.vision_end } };
    }

    /// The host's `Info.vision`; `o` outlives it.
    pub fn vision(o: *const Offer) api.Vision {
        return .{
            .ctx = @constCast(o),
            .prepare = prepareOpaque,
            .image_token = o.settings.image_token,
            .markers = &o.markers,
            .image_tokens = image_tokens,
            .max_images = max_images,
            .explain = explain,
        };
    }

    fn prepareOpaque(ctx: *anyopaque, a: Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
        const o: *const Offer = @ptrCast(@alignCast(ctx));
        return o.prepare(a, bytes, max_tokens);
    }

    /// The image's patches at most `max_tokens` tokens (the host's share of the request's budget, a low
    /// detail's 256 already in it), in `a`.
    pub fn prepare(o: *const Offer, a: Allocator, bytes: []const u8, max_tokens: u32) PrepareError!api.PreparedImage {
        var rgb = try open(a, bytes);
        defer rgb.deinit(a);
        const img: processor.Image = .of(rgb, .auto);
        if (!img.valid()) return error.Invalid;
        const s = o.settings;
        var problem: processor.Problem = .{};
        const size, const resized = processor.resized(a, s, img, 1, .{ .max_visual_tokens = max_tokens, .max_image_tokens = image_tokens }, &problem) catch |e| return switch (e) {
            error.OutOfMemory => error.OutOfMemory,
            else => error.AspectRatio, // smartResize's one refusal
        };
        defer a.free(resized);
        const gh = size.h / s.patch;
        const gw = size.w / s.patch;
        const pixels = try a.alloc(f32, @as(usize, gh) * gw * s.patchWidth());
        errdefer a.free(pixels);
        processor.patchify(s, &o.lut, resized, size, pixels);
        const hex = img.contentHash();
        return .{ .pixels = pixels, .gh = gh, .gw = gw, .tokens = gh * gw / (s.merge * s.merge), .hash = std.fmt.parseInt(u64, hex[0..16], 16) catch unreachable };
    }
};

pub const PrepareError = image.Error || error{ Unidentified, AspectRatio, Tiff, Avif, Heic, Ico, Jpeg2000, Qoi, Dds, PnmText, PsdMode };

/// An image part's bytes decoded, or refused as vLLM's loader refuses them (`explain`'s words): every model's
/// `prepare` (Flash Next's `Offer`, Unlimited-OCR's `served.zig`). In `a`.
pub fn open(a: Allocator, bytes: []const u8) PrepareError!image.Rgb {
    if (gap(bytes)) |e| return e;
    if (image.sniff(bytes) == .unknown) return error.Unidentified;
    return image.decode(a, bytes);
}

/// A format vLLM's Pillow opens and `image/` does not, by its first bytes: refused by name (spec 019's gaps). PNM
/// in binary (P5, P6), BMP, TGA and a PSD of RGB are stb_image's (`image/stb.zig`).
fn gap(bytes: []const u8) ?PrepareError {
    const starts = std.mem.startsWith;
    if (starts(u8, bytes, "II*\x00") or starts(u8, bytes, "MM\x00*")) return error.Tiff;
    if (bytes.len >= 12 and std.mem.eql(u8, bytes[4..8], "ftyp")) {
        const brand = bytes[8..12];
        if (std.mem.eql(u8, brand, "avif") or std.mem.eql(u8, brand, "avis")) return error.Avif;
        if (std.mem.eql(u8, brand, "heic") or std.mem.eql(u8, brand, "heix") or std.mem.eql(u8, brand, "mif1") or std.mem.eql(u8, brand, "msf1")) return error.Heic;
    }
    if (starts(u8, bytes, "\x00\x00\x01\x00") or starts(u8, bytes, "\x00\x00\x02\x00")) return error.Ico; // ICO, CUR
    if (starts(u8, bytes, "\x00\x00\x00\x0cjP  \r\n\x87\n") or starts(u8, bytes, "\xff\x4f\xff\x51")) return error.Jpeg2000;
    if (starts(u8, bytes, "qoif")) return error.Qoi;
    if (starts(u8, bytes, "DDS ")) return error.Dds;
    // stb_image reads PNM's binary P5 and P6 alone, and a PSD of RGB colour (mode 3)
    if (bytes.len >= 3 and bytes[0] == 'P' and bytes[1] >= '1' and bytes[1] <= '4' and std.ascii.isWhitespace(bytes[2])) return error.PnmText;
    if (starts(u8, bytes, "8BPS") and bytes.len >= 26 and std.mem.readInt(u16, bytes[24..26], .big) != 3) return error.PsdMode;
    return null;
}

const decoded = "send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM";

/// vLLM's words for a refusal; null: the host's (out of memory: a 500).
pub fn explain(err: anyerror) ?[]const u8 {
    return switch (err) {
        error.Unidentified => "Failed to load image: cannot identify image file",
        error.Invalid => "Failed to load image: image file is truncated or damaged",
        error.TooManyBytes => "Failed to load image: image is empty or exceeds the encoded byte limit of 10 MB",
        error.Oversized => "Image dimensions exceed the maximum of 8192 pixels a side or 16777216 pixels",
        error.CmykJpeg => "Failed to load image: CMYK JPEG images are not decoded by this server; " ++ decoded,
        error.NoJpegLibrary, error.JpegLibraryVersion => "Failed to load image: image file is truncated or damaged",
        error.AspectRatio => "absolute aspect ratio must be smaller than 200",
        error.Tiff => "Failed to load image: TIFF images are not decoded by this server; " ++ decoded,
        error.Avif => "Failed to load image: AVIF images are not decoded by this server; " ++ decoded,
        error.Heic => "Failed to load image: HEIC images are not decoded by this server; " ++ decoded,
        error.Ico => "Failed to load image: ICO images are not decoded by this server; " ++ decoded,
        error.Jpeg2000 => "Failed to load image: JPEG 2000 images are not decoded by this server; " ++ decoded,
        error.Qoi => "Failed to load image: QOI images are not decoded by this server; " ++ decoded,
        error.Dds => "Failed to load image: DDS images are not decoded by this server; " ++ decoded,
        error.PnmText => "Failed to load image: PBM, PGM and PPM images in text or 1-bit form (P1 to P4) are not decoded by this server; " ++ decoded,
        error.PsdMode => "Failed to load image: PSD images in another colour mode than RGB are not decoded by this server; " ++ decoded,
        else => null,
    };
}
