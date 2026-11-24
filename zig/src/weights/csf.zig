//! NVFP4 block scales stored compressed (`weight_scale_encoding: "csf"`, codec
//! `byte-window4-fixed-stream-u24-exceptions/1`), decoded on the host to the e4m3 bytes `nvfp4.zig` takes.
//! A linear's `weight_scale` [rows, cols] (cols = inputs / 16) is stored as two tensors:
//!
//! - `weight_scale.nvfp4_csf_fixed`, U8 [rows / 16, 16 (1 + cols / 2)]: for each slab of 16 rows, one base byte a
//!   row, then each row's cols / 2 bytes of 4-bit codes, two a byte, the even column's in the low nibble. A
//!   scale's byte is its row's base plus its code.
//! - `weight_scale.nvfp4_csf_exceptions`, U32 [count]: words whose low 24 bits are a flat `row * cols + col` and
//!   whose high 8 bits are the scale's byte there, overriding the fixed stream.
//!
//! Lossless: the decoded bytes are the uncompressed checkpoint's. Read from vLLM PR 60646
//! (`model_executor/model_loader/csf_scales.py`, open upstream; the card's containers apply it, and their b12x
//! loader decodes at load the same way unless `VLLM_CSF_RESIDENT` keeps the streams on the device). Host only.

const std = @import("std");

pub const fixed_suffix = ".nvfp4_csf_fixed";
pub const exceptions_suffix = ".nvfp4_csf_exceptions";

/// Rows a slab holds, each with its base byte.
pub const slab = 16;
/// The highest base a row may have (PR 60646 refuses more: base + 15 must stay a byte below 0xff).
pub const max_base = 240;

pub const Error = error{InvalidCsf};

/// The scale matrix's shape, from the fixed stream's.
pub const Shape = struct {
    rows: u64,
    cols: u64,

    /// Null where `dims` is no fixed stream's shape.
    pub fn ofFixed(dims: []const u64) ?Shape {
        if (dims.len != 2 or dims[1] % slab != 0 or dims[1] < 2 * slab) return null;
        return .{ .rows = dims[0] * slab, .cols = (dims[1] / slab - 1) * 2 };
    }

    /// The fixed stream's shape for a scale matrix of `rows` and `cols`.
    pub fn fixedDims(self: Shape) [2]u64 {
        return .{ self.rows / slab, slab * (1 + self.cols / 2) };
    }

    pub fn count(self: Shape) u64 {
        return self.rows * self.cols;
    }
};

/// Whether `rows` x `cols` scales can be stored so: whole slabs, codes in whole bytes.
pub fn fits(rows: u64, cols: u64) bool {
    return rows % slab == 0 and cols % 2 == 0 and cols > 0 and rows * cols <= 1 << 24;
}

/// A window of a scale matrix: rows `[row, row + rows)` and columns `[col, col + cols)`, as a rank of two holds
/// an expert's (its rows of gate and up, its columns of down). `col` and `cols` are even: codes in whole bytes.
pub const Window = struct {
    row: u64,
    rows: u64,
    col: u64,
    cols: u64,

    pub fn whole(shape: Shape) Window {
        return .{ .row = 0, .rows = shape.rows, .col = 0, .cols = shape.cols };
    }

    pub fn count(self: Window) u64 {
        return self.rows * self.cols;
    }
};

/// Decodes into `out` (rows x cols bytes, row-major e4m3) the scales stored as `fixed` and `exceptions` (the U32
/// words' little-endian bytes).
pub fn decode(shape: Shape, fixed: []const u8, exceptions: []const u8, out: []u8) Error!void {
    return decodeWindow(shape, fixed, exceptions, .whole(shape), out);
}

/// `decode`'s bytes in `window` alone, into `out` (window.rows x window.cols), reading only the window's rows of
/// the fixed stream: vLLM's `CsfSource.select` (the card's b12x loader, for a tensor-parallel rank's experts).
/// The bases and exceptions outside the window are not checked.
pub fn decodeWindow(shape: Shape, fixed: []const u8, exceptions: []const u8, window: Window, out: []u8) Error!void {
    const rows: usize = @intCast(shape.rows);
    const cols: usize = @intCast(shape.cols);
    const half = cols / 2;
    if (!fits(shape.rows, shape.cols) or fixed.len != rows * (1 + half) or exceptions.len % 4 != 0)
        return error.InvalidCsf;
    if (window.row + window.rows > shape.rows or window.col + window.cols > shape.cols or window.col % 2 != 0 or
        window.cols % 2 != 0 or out.len != window.count())
        return error.InvalidCsf;
    const first: usize = @intCast(window.col / 2);
    const width: usize = @intCast(window.cols);
    for (0..@intCast(window.rows)) |i| {
        const at: usize = @intCast(window.row + i);
        const block = fixed[at / slab * slab * (1 + half) ..][0 .. slab * (1 + half)];
        const r = at % slab;
        const base = block[r];
        if (base > max_base) return error.InvalidCsf;
        const codes = block[slab + r * half ..][first..][0 .. width / 2];
        const row = out[i * width ..][0..width];
        for (codes, 0..) |byte, c| {
            row[2 * c] = base + (byte & 0xf);
            row[2 * c + 1] = base + (byte >> 4);
        }
    }
    var i: usize = 0;
    while (i < exceptions.len) : (i += 4) {
        const word = std.mem.readInt(u32, exceptions[i..][0..4], .little);
        const at = word & 0xff_ffff;
        if (at >= rows * cols) return error.InvalidCsf;
        const row = at / cols;
        const col = at % cols;
        if (row < window.row or row >= window.row + window.rows or col < window.col or col >= window.col + window.cols) continue;
        out[@intCast((row - window.row) * window.cols + (col - window.col))] = @intCast(word >> 24);
    }
}
