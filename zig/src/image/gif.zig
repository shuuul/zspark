//! A GIF's first frame as Pillow 12.3 loaded it (`GifImagePlugin.py`, `libImaging/GifDecode.c`), read the way
//! TensorFold 0.6.5's `_decode` read a picture but with GIF among the formats and no frame count: vLLM and
//! SGLang take an animation's first frame, and so does zspark (spec 019's Decisions of 2026-10-13).
//!
//! What Pillow does and this follows: the canvas is the logical screen, grown to hold the first frame; it starts
//! as the frame's transparent index where the frame has one, else index 0, and the frame's pixels are written
//! over it whole. The colours are the frame's table (its own, else the global one) unless that table is the
//! identity grey ramp, when the indices are grey levels themselves (mode L); an index past the table is black.
//! A transparent index is alpha 0, so white once over white. The LZW stream is GifDecode.c's: a code past the
//! next free one, or an end code or the data's end before the frame is full, refuses the picture, as Pillow's
//! load did on a file that size. A GIF has no orientation tag Pillow reads.

const std = @import("std");
const Allocator = std.mem.Allocator;
const picture = @import("picture");

pub fn decode(a: Allocator, data: []const u8) picture.Error!picture.Samples {
    var r: Reader = .{ .data = data };
    const head = r.take(13) orelse return error.Invalid;
    if (!std.mem.startsWith(u8, head, "GIF87a") and !std.mem.startsWith(u8, head, "GIF89a")) return error.Invalid;
    var width: usize = std.mem.readInt(u16, head[6..8], .little);
    var height: usize = std.mem.readInt(u16, head[8..10], .little);
    var global: ?[]const u8 = null;
    if (head[10] & 0x80 != 0) {
        const table = r.take(@as(usize, 3) << @intCast((head[10] & 7) + 1)) orelse return error.Invalid;
        if (needed(table)) global = table;
    }
    const f = try frame(&r);
    width = @max(width, f.x + f.w);
    height = @max(height, f.y + f.h);
    try picture.fits(width, height);
    if (f.w == 0 or f.h == 0) return error.Invalid; // Pillow's setimage: a tile that holds no pixel
    if (f.bits > 12) return error.Invalid;
    const table: ?[]const u8 = if (f.local) |local| (if (needed(local)) local else null) else global;
    const canvas = try a.alloc(u8, width * height);
    defer a.free(canvas);
    @memset(canvas, f.transparent orelse 0);
    try lzw(&r, f, canvas, width);
    const channels: usize = if (f.transparent != null) 4 else 3;
    const px = try a.alloc(u8, width * height * channels);
    for (canvas, 0..) |index, i| {
        const out = px[i * channels ..][0..channels];
        if (table) |t| {
            const at = @as(usize, index) * 3;
            if (at + 3 <= t.len) @memcpy(out[0..3], t[at..][0..3]) else @memset(out[0..3], 0);
        } else @memset(out[0..3], index);
        if (channels == 4) out[3] = if (index == f.transparent.?) 0 else 255;
    }
    return .{ .width = width, .height = height, .channels = channels, .data = px };
}

/// Pillow's `_is_palette_needed`: a table is a palette unless entry i is (i, i, i) for every i.
fn needed(table: []const u8) bool {
    var i: usize = 0;
    while (i + 3 <= table.len) : (i += 3) {
        const v = i / 3;
        if (table[i] != v or table[i + 1] != v or table[i + 2] != v) return true;
    }
    return false;
}

const Reader = struct {
    data: []const u8,
    at: usize = 0,

    /// `n` bytes, or null where fewer are left (where a short read failed Pillow's parsing).
    fn take(r: *Reader, n: usize) ?[]const u8 {
        if (r.data.len - r.at < n) return null;
        defer r.at += n;
        return r.data[r.at..][0..n];
    }

    /// The bytes up to `n`, as many as are left.
    fn upTo(r: *Reader, n: usize) []const u8 {
        const got = @min(n, r.data.len - r.at);
        defer r.at += got;
        return r.data[r.at..][0..got];
    }

    fn byte(r: *Reader) ?u8 {
        return if (r.take(1)) |b| b[0] else null;
    }

    /// Pillow's `data()`: a sub-block, null at its terminator or the data's end; short where the data ends.
    fn block(r: *Reader) ?[]const u8 {
        const n = r.byte() orelse return null;
        if (n == 0) return null;
        return r.upTo(n);
    }

    /// Pillow's `while self.data(): pass`.
    fn skipBlocks(r: *Reader) void {
        while (r.block()) |b| if (b.len == 0) break;
    }
};

const Frame = struct {
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    interlace: bool,
    local: ?[]const u8,
    transparent: ?u8,
    bits: u8,
};

/// Pillow's `_seek(0)`: the extensions before the first image descriptor, then the descriptor, its table and the
/// LZW code size; `r` is left at the frame's first sub-block.
fn frame(r: *Reader) picture.Error!Frame {
    var transparent: ?u8 = null;
    var s = r.byte() orelse return error.Invalid; // no frame: Pillow's open found no image
    if (s == ';') return error.Invalid;
    while (true) {
        switch (s) {
            '!' => {
                const label = r.byte() orelse return error.Invalid;
                const first = r.block();
                if (label == 254) { // a comment: its sub-blocks, and nothing after a first terminator
                    var b = first;
                    while (b) |got| {
                        if (got.len == 0) break;
                        b = r.block();
                    }
                } else {
                    if (first) |b| {
                        if (label == 249) { // the graphic control extension
                            if (b.len == 0 or (b[0] & 1 != 0 and b.len < 4) or b.len < 3) return error.Invalid;
                            if (b[0] & 1 != 0) transparent = b[3];
                        }
                        if (label == 255 and std.mem.startsWith(u8, b, "NETSCAPE2.0")) _ = r.block();
                    }
                    r.skipBlocks();
                }
            },
            ',' => {
                const d = r.take(9) orelse return error.Invalid;
                const flags = d[8];
                var local: ?[]const u8 = null;
                if (flags & 0x80 != 0) local = r.upTo(@as(usize, 3) << @intCast((flags & 7) + 1));
                if (local) |t| if (t.len % 3 != 0) return error.Invalid; // Pillow's palette check indexed past it
                const bits = r.byte() orelse return error.Invalid;
                return .{
                    .x = std.mem.readInt(u16, d[0..2], .little),
                    .y = std.mem.readInt(u16, d[2..4], .little),
                    .w = std.mem.readInt(u16, d[4..6], .little),
                    .h = std.mem.readInt(u16, d[6..8], .little),
                    .interlace = flags & 0x40 != 0,
                    .local = local,
                    .transparent = transparent,
                    .bits = bits,
                };
            },
            else => {}, // any other byte is passed over
        }
        s = r.byte() orelse return error.Invalid;
        if (s == ';') return error.Invalid;
    }
}

const table_size = 1 << 12;

/// GifDecode.c over the rest of the data: done when the frame's last row is written; refused otherwise.
fn lzw(r: *Reader, f: Frame, canvas: []u8, stride: usize) picture.Error!void {
    const clear: u32 = @as(u32, 1) << @intCast(f.bits);
    const end = clear + 1;
    var link: [table_size]u16 = undefined;
    var suffix: [table_size]u8 = undefined;
    var buffer: [table_size]u8 = undefined;
    var next: u32 = 0;
    var size: u5 = 0;
    var mask: u32 = 0;
    var first = true; // the first code after a clear
    var last_code: u32 = 0;
    var last_data: u8 = 0;
    var bit_buffer: u32 = 0;
    var bit_count: u5 = 0;
    var block_left: usize = 0;
    var out: Out = .{ .canvas = canvas, .stride = stride, .f = f };
    var restart = true;
    while (true) {
        if (restart) {
            next = clear + 2;
            size = @intCast(f.bits + 1);
            mask = (@as(u32, 1) << size) - 1;
            first = true;
            restart = false;
        }
        while (bit_count < size) {
            if (block_left == 0) {
                // a whole sub-block or nothing: a block that runs past the data is the truncated file Pillow refused
                const n = r.byte() orelse return error.Invalid;
                if (r.data.len - r.at < n) return error.Invalid;
                block_left = n;
                continue;
            }
            bit_buffer |= @as(u32, r.byte().?) << bit_count;
            bit_count += 8;
            block_left -= 1;
        }
        var c = bit_buffer & mask;
        bit_buffer >>= size;
        bit_count -= size;
        if (c == clear) {
            if (!first) restart = true;
            continue;
        }
        if (c == end) return error.Invalid; // the frame stopped short: Pillow then read past the data
        if (first) {
            if (c > clear) return error.Invalid;
            last_code = c;
            last_data = @truncate(c);
            first = false;
            if (out.put(&.{last_data})) return;
            continue;
        }
        const this = c;
        if (c > next) return error.Invalid;
        var at: usize = table_size;
        if (c == next) {
            at -= 1;
            buffer[at] = last_data;
            c = last_code;
        }
        while (c >= clear) {
            if (at == 0 or c >= table_size) return error.Invalid;
            at -= 1;
            buffer[at] = suffix[c];
            c = link[c];
        }
        last_data = @truncate(c);
        if (next < table_size) {
            suffix[next] = last_data;
            link[next] = @intCast(last_code);
            if (next == mask and size < 12) {
                size += 1;
                mask = (@as(u32, 1) << size) - 1;
            }
            next += 1;
        }
        last_code = this;
        if (out.put(&.{last_data})) return;
        if (out.put(buffer[at..])) return;
    }
}

/// GifDecode.c's writer: rows of the frame's box in order, or in the four interlaced passes.
const Out = struct {
    canvas: []u8,
    stride: usize,
    f: Frame,
    x: usize = 0,
    y: usize = 0,
    pass: u8 = 0,

    /// Writes `bytes`; true once the last row is full (the decoder's end).
    fn put(o: *Out, bytes: []const u8) bool {
        for (bytes) |v| {
            o.canvas[(o.f.y + o.y) * o.stride + o.f.x + o.x] = v;
            o.x += 1;
            if (o.x < o.f.w) continue;
            o.x = 0;
            if (!o.f.interlace) {
                o.y += 1;
                if (o.y >= o.f.h) return true;
                continue;
            }
            o.y += ([_]usize{ 8, 8, 4, 2 })[o.pass];
            while (o.y >= o.f.h) {
                if (o.pass == 3) return true;
                o.pass += 1;
                o.y = ([_]usize{ 0, 4, 2, 1 })[o.pass];
            }
        }
        return false;
    }
};
