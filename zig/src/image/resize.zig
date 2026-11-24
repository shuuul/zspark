//! Pillow 12.3's `Image.resize(size, BICUBIC)` and `ImageOps.pad` for an 8-bit RGB picture, bit for bit (spec 137
//! WS-02). The checkpoint's HF code resizes with Pillow (`dynamic_preprocess`'s `image.resize`, `infer`'s
//! `ImageOps.pad`), whose filter is not torch's bicubic (`text/vision.zig`'s): `libImaging/Resample.c`'s two
//! passes, horizontal then vertical, each pass a row of 22-bit fixed-point coefficients (`PRECISION_BITS`) from
//! the Keys cubic (a = -0.5, support 2) widened by the shrink factor, normalized to sum 1 in double, then rounded
//! to int; each output a sum of `u8 * i32` plus half, shifted right by 22 and clipped to a byte. The vertical pass
//! reads only the source rows the horizontal pass has made, so the order of the two cannot change a bit.

const std = @import("std");
const Allocator = std.mem.Allocator;

const precision_bits = 32 - 8 - 2;
const support = 2.0;
const channels = 3;

/// One axis' coefficients: output `i` is `taps[i * ksize ..][0..len[i]]` over the source from `first[i]`.
const Axis = struct {
    ksize: usize,
    first: []i32,
    len: []i32,
    taps: []i32,

    fn deinit(self: Axis, gpa: Allocator) void {
        gpa.free(self.first);
        gpa.free(self.len);
        gpa.free(self.taps);
    }
};

/// Pillow's `bicubic_filter`.
fn bicubic(x_in: f64) f64 {
    const a = -0.5;
    const x = @abs(x_in);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}

/// `precompute_coeffs` and `normalize_coeffs_8bpc` for the whole source axis (`box` is the whole picture).
fn axis(gpa: Allocator, in_size: usize, out_size: usize) Allocator.Error!Axis {
    const scale = @as(f64, @floatFromInt(in_size)) / @as(f64, @floatFromInt(out_size));
    const filterscale = @max(scale, 1.0);
    const sup = support * filterscale;
    const ksize: usize = @as(usize, @intFromFloat(@ceil(sup))) * 2 + 1;
    const first = try gpa.alloc(i32, out_size);
    errdefer gpa.free(first);
    const len = try gpa.alloc(i32, out_size);
    errdefer gpa.free(len);
    const taps = try gpa.alloc(i32, out_size * ksize);
    errdefer gpa.free(taps);
    const w = try gpa.alloc(f64, ksize);
    defer gpa.free(w);
    const inv = 1.0 / filterscale;
    const size: i64 = @intCast(in_size);
    for (0..out_size) |xx| {
        const center = (@as(f64, @floatFromInt(xx)) + 0.5) * scale;
        // C's (int) truncates toward zero; the clamps make a negative one 0 as well
        const xmin: i64 = @max(0, @as(i64, @intFromFloat(center - sup + 0.5)));
        const xmax: i64 = @min(size, @as(i64, @intFromFloat(center + sup + 0.5))) - xmin;
        var ww: f64 = 0.0;
        const n: usize = @intCast(@max(xmax, 0));
        for (0..n) |x| {
            const v = bicubic((@as(f64, @floatFromInt(@as(i64, @intCast(x)) + xmin)) - center + 0.5) * inv);
            w[x] = v;
            ww += v;
        }
        if (ww != 0.0) for (w[0..n]) |*v| {
            v.* /= ww;
        };
        const row = taps[xx * ksize ..][0..ksize];
        @memset(row, 0);
        for (w[0..n], row[0..n]) |v, *t| {
            const scaled = v * @as(f64, 1 << precision_bits);
            t.* = if (v < 0) @intFromFloat(-0.5 + scaled) else @intFromFloat(0.5 + scaled);
        }
        first[xx] = @intCast(xmin);
        len[xx] = @intCast(n);
    }
    return .{ .ksize = ksize, .first = first, .len = len, .taps = taps };
}

fn clip8(sum: i32) u8 {
    return @intCast(std.math.clamp(sum >> precision_bits, 0, 255));
}

/// `src` (`sw` x `sh`, RGB rows) resized to `dw` x `dh` as `Image.resize((dw, dh), Image.BICUBIC)`; the caller
/// frees it. A picture of the size asked for is copied, as Pillow does.
pub fn resize(gpa: Allocator, src: []const u8, sw: usize, sh: usize, dw: usize, dh: usize) Allocator.Error![]u8 {
    std.debug.assert(src.len == sw * sh * channels and sw > 0 and sh > 0 and dw > 0 and dh > 0);
    if (sw == dw and sh == dh) return gpa.dupe(u8, src);
    const out = try gpa.alloc(u8, dw * dh * channels);
    errdefer gpa.free(out);
    const vert = try axis(gpa, sh, dh);
    defer vert.deinit(gpa);
    // the source rows the vertical pass reads: the horizontal pass makes only those
    const row0: usize = @intCast(vert.first[0]);
    const row1: usize = @intCast(vert.first[dh - 1] + vert.len[dh - 1]);

    // horizontal pass, `dw` x (row1 - row0), into `mid` (all of the source's columns when the width is kept)
    var mid: []const u8 = src[row0 * sw * channels .. row1 * sw * channels];
    var mid_w = sw;
    var owned: ?[]u8 = null;
    defer if (owned) |m| gpa.free(m);
    if (dw != sw) {
        const horiz = try axis(gpa, sw, dw);
        defer horiz.deinit(gpa);
        const made = try gpa.alloc(u8, dw * (row1 - row0) * channels);
        owned = made;
        for (row0..row1) |y| {
            const line = src[y * sw * channels ..][0 .. sw * channels];
            const dst = made[(y - row0) * dw * channels ..][0 .. dw * channels];
            for (0..dw) |xx| {
                const k = horiz.taps[xx * horiz.ksize ..][0..horiz.ksize];
                const at: usize = @intCast(horiz.first[xx]);
                const n: usize = @intCast(horiz.len[xx]);
                var s0: i32 = 1 << (precision_bits - 1);
                var s1 = s0;
                var s2 = s0;
                for (k[0..n], 0..) |c, x| {
                    const px = line[(at + x) * channels ..][0..channels];
                    s0 += @as(i32, px[0]) * c;
                    s1 += @as(i32, px[1]) * c;
                    s2 += @as(i32, px[2]) * c;
                }
                dst[xx * channels + 0] = clip8(s0);
                dst[xx * channels + 1] = clip8(s1);
                dst[xx * channels + 2] = clip8(s2);
            }
        }
        mid = made;
        mid_w = dw;
    }

    // vertical pass
    if (dh == sh) {
        @memcpy(out, mid);
        return out;
    }
    const row_len = mid_w * channels;
    const acc = try gpa.alloc(i32, row_len);
    defer gpa.free(acc);
    for (0..dh) |yy| {
        const k = vert.taps[yy * vert.ksize ..][0..vert.ksize];
        const at: usize = @as(usize, @intCast(vert.first[yy])) - row0;
        const n: usize = @intCast(vert.len[yy]);
        @memset(acc, 1 << (precision_bits - 1));
        for (k[0..n], 0..) |c, y| {
            const line = mid[(at + y) * row_len ..][0..row_len];
            for (acc, line) |*a, v| a.* += @as(i32, v) * c;
        }
        for (out[yy * row_len ..][0..row_len], acc) |*o, a| o.* = clip8(a);
    }
    return out;
}

/// Python's `round` of a float: halves go to the even integer.
fn roundHalfEven(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    const i: i64 = @intFromFloat(f);
    if (d < 0.5) return i;
    if (d > 0.5) return i + 1;
    return if (@mod(i, 2) == 0) i else i + 1;
}

/// `ImageOps.pad(image, (dw, dh), BICUBIC, color=(fill, fill, fill))` centered: the picture contained in the
/// size at its aspect ratio (`ImageOps.contain`), then pasted centered on `fill`. The caller frees it.
pub fn pad(gpa: Allocator, src: []const u8, sw: usize, sh: usize, dw: usize, dh: usize, fill: u8) Allocator.Error![]u8 {
    var cw = dw;
    var ch = dh;
    const im_ratio = @as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(sh));
    const dest_ratio = @as(f64, @floatFromInt(dw)) / @as(f64, @floatFromInt(dh));
    if (im_ratio != dest_ratio) {
        if (im_ratio > dest_ratio) {
            ch = @intCast(roundHalfEven(@as(f64, @floatFromInt(sh)) / @as(f64, @floatFromInt(sw)) * @as(f64, @floatFromInt(dw))));
        } else {
            cw = @intCast(roundHalfEven(@as(f64, @floatFromInt(sw)) / @as(f64, @floatFromInt(sh)) * @as(f64, @floatFromInt(dh))));
        }
    }
    const inner = try resize(gpa, src, sw, sh, cw, ch);
    if (cw == dw and ch == dh) return inner;
    defer gpa.free(inner);
    const out = try gpa.alloc(u8, dw * dh * channels);
    errdefer gpa.free(out);
    @memset(out, fill);
    const x0: usize = if (cw != dw) @intCast(roundHalfEven(@as(f64, @floatFromInt(dw - cw)) * 0.5)) else 0;
    const y0: usize = if (cw == dw) @intCast(roundHalfEven(@as(f64, @floatFromInt(dh - ch)) * 0.5)) else 0;
    for (0..ch) |y| {
        @memcpy(out[((y0 + y) * dw + x0) * channels ..][0 .. cw * channels], inner[y * cw * channels ..][0 .. cw * channels]);
    }
    return out;
}
