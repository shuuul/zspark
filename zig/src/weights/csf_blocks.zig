//! NVFP4 block scales stored compressed in blocks (`weight_scale_encoding: "csf_32x2_v1"`, GLM-5.3-Flash
//! FP4-Local's routed experts), decoded on the host to the e4m3 bytes `nvfp4.zig` takes. Lossless: the decoded
//! bytes are the uncompressed checkpoint's. Not `csf.zig`'s encoding (Flash Next's, slabs of 16 rows and U32
//! exceptions), though the window a rank decodes is the same `csf.Window`.
//!
//! A linear's `weight_scale` [rows, cols] (cols = inputs / 16, rows a multiple of 32) is cut into blocks of 32
//! rows by 2 columns (one at an odd last column), visited by tiles of 4 block rows by 2 block columns, a tile's
//! blocks column by column; 32 tiles make a page. It is stored as five tensors, `weight_scale.csf32_*`:
//!
//! - `bases`, U8 [rows]: a row's base byte, at most 240;
//! - `codes`, U8 [rows * cols / 2]: each block's 4-bit offsets in visit order, at its local positions
//!   `2 * row + col` (those of its columns), two a byte, the lower position in the low nibble. A scale's byte is
//!   its row's base plus its offset;
//! - `exceptions`, U16 [count]: a block's records in visit order, sorted by position: the low byte a local
//!   position, the high byte the scale's byte there, overriding the offset;
//! - `offsets`, U16 [blocks + pages]: for each page, its blocks' record prefixes from 0, and its end;
//! - `page_bases`, U32 [pages]: twice the records before a page's first block.
//!
//! Read from flashinfer's b12x loader as the card's vLLM container ships it
//! (`flashinfer/experimental/b12x/loader/_csf_blocks.py`: `block_order`, `CsfBlockPlane.validate`,
//! `logical_csf_components`); vLLM's `nvfp4_csf.py` stages a tensor-parallel rank's rows or columns of it. Host
//! only.

const std = @import("std");
const csf = @import("csf.zig");

pub const encoding = "csf_32x2_v1";
/// A block's rows and columns, a tile's block rows and columns, a page's tiles.
pub const block_rows = 32;
pub const block_cols = 2;
pub const tile_rows = 4;
pub const tile_cols = 2;
pub const page_tiles = 32;
/// The highest base: base + 15 stays a byte.
pub const max_base = 240;

pub const bases_suffix = ".csf32_bases";
pub const codes_suffix = ".csf32_codes";
pub const page_bases_suffix = ".csf32_page_bases";
pub const offsets_suffix = ".csf32_offsets";
pub const exceptions_suffix = ".csf32_exceptions";

pub const Window = csf.Window;
pub const Error = error{InvalidCsf};

/// The scale matrix's shape.
pub const Shape = struct {
    rows: u64,
    cols: u64,

    /// From the bases' and the codes' element counts; null where they make no plane.
    pub fn ofStreams(bases: u64, codes: u64) ?Shape {
        if (bases == 0 or bases % block_rows != 0 or (2 * codes) % bases != 0 or codes == 0) return null;
        return .{ .rows = bases, .cols = 2 * codes / bases };
    }

    pub fn fits(self: Shape) bool {
        return self.rows > 0 and self.rows % block_rows == 0 and self.cols > 0;
    }

    pub fn blockRows(self: Shape) u64 {
        return self.rows / block_rows;
    }

    pub fn blockCols(self: Shape) u64 {
        return (self.cols + block_cols - 1) / block_cols;
    }

    pub fn blocks(self: Shape) u64 {
        return self.blockRows() * self.blockCols();
    }

    pub fn tiles(self: Shape) u64 {
        return ceil(self.blockRows(), tile_rows) * ceil(self.blockCols(), tile_cols);
    }

    pub fn pages(self: Shape) u64 {
        return ceil(self.tiles(), page_tiles);
    }

    pub fn codeBytes(self: Shape) u64 {
        return self.rows * self.cols / 2;
    }

    pub fn count(self: Shape) u64 {
        return self.rows * self.cols;
    }
};

fn ceil(a: u64, b: u64) u64 {
    return (a + b - 1) / b;
}

/// The five tensors' bytes as stored, little-endian.
pub const Streams = struct {
    bases: []const u8,
    codes: []const u8,
    page_bases: []const u8,
    offsets: []const u8,
    exceptions: []const u8,

    fn offset(self: Streams, at: u64) u64 {
        return std.mem.readInt(u16, self.offsets[@intCast(2 * at)..][0..2], .little);
    }

    fn pageBase(self: Streams, page: u64) u64 {
        return std.mem.readInt(u32, self.page_bases[@intCast(4 * page)..][0..4], .little);
    }

    fn record(self: Streams, at: u64) u16 {
        return std.mem.readInt(u16, self.exceptions[@intCast(2 * at)..][0..2], .little);
    }
};

/// One block as the codes and the exceptions hold it.
const Block = struct {
    /// Its first row and column of the plane, and its columns (1 or 2).
    row: u64,
    col: u64,
    width: u64,
    /// Where its offsets start in `codes`, and its records in `exceptions`.
    code: u64,
    first: u64,
    records: u64,
};

/// The blocks in visit order (`block_order`), with what `offsets` and `page_bases` say of each.
const Walk = struct {
    shape: Shape,
    streams: Streams,
    tile_row: u64 = 0,
    tile_col: u64 = 0,
    /// Within the tile: the block column and row.
    col: u64 = 0,
    row: u64 = 0,
    /// The visit's index, its tile, the code bytes before it.
    index: u64 = 0,
    tile: u64 = 0,
    code: u64 = 0,

    fn next(self: *Walk) ?Block {
        const block_rows_n = self.shape.blockRows();
        const block_cols_n = self.shape.blockCols();
        if (self.tile_row * tile_rows >= block_rows_n) return null;
        const r0 = self.tile_row * tile_rows;
        const c0 = self.tile_col * tile_cols;
        const rows: u64 = @min(tile_rows, block_rows_n - r0);
        const cols: u64 = @min(tile_cols, block_cols_n - c0);
        const br = r0 + self.row;
        const bc = c0 + self.col;
        const page = self.tile / page_tiles;
        const at = self.index + page;
        const width: u64 = @min(block_cols, self.shape.cols - block_cols * bc);
        const block: Block = .{
            .row = br * block_rows,
            .col = bc * block_cols,
            .width = width,
            .code = self.code,
            .first = self.streams.pageBase(page) / 2 + self.streams.offset(at),
            .records = self.streams.offset(at + 1) -% self.streams.offset(at),
        };
        self.code += block_rows * width / 2;
        self.index += 1;
        self.row += 1;
        if (self.row == rows) {
            self.row = 0;
            self.col += 1;
            if (self.col == cols) {
                self.col = 0;
                self.tile += 1;
                self.tile_col += 1;
                if (self.tile_col * tile_cols >= block_cols_n) {
                    self.tile_col = 0;
                    self.tile_row += 1;
                }
            }
        }
        return block;
    }
};

/// `CsfBlockPlane.validate`: the streams' lengths, the bases, each page's prefixes from 0 over one run of
/// records, a block's records no more than its scales, sorted, unique and on its columns. A plane `decode`
/// reads is checked so first; nothing past these is read.
pub fn validate(shape: Shape, streams: Streams) Error!void {
    if (!shape.fits()) return error.InvalidCsf;
    const pages = shape.pages();
    if (streams.bases.len != shape.rows or streams.codes.len != shape.codeBytes() or streams.page_bases.len != 4 * pages or
        streams.offsets.len != 2 * (shape.blocks() + pages) or streams.exceptions.len % 2 != 0)
        return error.InvalidCsf;
    for (streams.bases) |base| if (base > max_base) return error.InvalidCsf;
    var walk: Walk = .{ .shape = shape, .streams = streams };
    var records: u64 = 0;
    var page: u64 = 0;
    var blocks: u64 = 0;
    while (true) {
        const tile = walk.tile;
        const block = walk.next() orelse break;
        if (tile / page_tiles == page) {
            // a page's first block: its prefix is 0, its base the records before it
            if (streams.offset(blocks + page) != 0 or streams.pageBase(page) != 2 * records) return error.InvalidCsf;
            page += 1;
        }
        blocks += 1;
        if (block.records > block_rows * block.width or block.first != records) return error.InvalidCsf;
        records += block.records;
        if (records > streams.exceptions.len / 2) return error.InvalidCsf;
        var last: ?u16 = null;
        for (block.first..block.first + block.records) |at| {
            const local = streams.record(at) & 0xff;
            if (local >= block_rows * block_cols or local % 2 >= block.width) return error.InvalidCsf;
            if (last) |before| if (local <= before) return error.InvalidCsf;
            last = local;
        }
    }
    if (records != streams.exceptions.len / 2) return error.InvalidCsf;
}

/// Decodes into `out` (rows x cols bytes, row-major e4m3) the scales the streams hold.
pub fn decode(shape: Shape, streams: Streams, out: []u8) Error!void {
    return decodeWindow(shape, streams, .{ .row = 0, .rows = shape.rows, .col = 0, .cols = shape.cols }, out);
}

/// `decode`'s bytes in `window` alone, into `out` (window.rows x window.cols): a rank's rows of a gate or up
/// projection, or its columns of a down projection. Any window inside the plane; the whole plane is
/// validated, and only the blocks it meets are decoded.
pub fn decodeWindow(shape: Shape, streams: Streams, window: Window, out: []u8) Error!void {
    try validate(shape, streams);
    if (window.row + window.rows > shape.rows or window.col + window.cols > shape.cols or out.len != window.count())
        return error.InvalidCsf;
    var walk: Walk = .{ .shape = shape, .streams = streams };
    while (walk.next()) |block| {
        if (block.row + block_rows <= window.row or block.row >= window.row + window.rows or
            block.col + block.width <= window.col or block.col >= window.col + window.cols) continue;
        const codes = streams.codes[@intCast(block.code)..][0..@intCast(block_rows * block.width / 2)];
        for (0..block_rows) |r| {
            const row = block.row + r;
            if (row < window.row or row >= window.row + window.rows) continue;
            const base = streams.bases[@intCast(row)];
            for (0..@intCast(block.width)) |c| {
                const col = block.col + c;
                if (col < window.col or col >= window.col + window.cols) continue;
                // the block's fields are its local positions on its columns, in order
                const field = r * block.width + c;
                const offset = codes[field / 2] >> @intCast(4 * (field % 2)) & 0xf;
                out[@intCast((row - window.row) * window.cols + col - window.col)] = base + offset;
            }
        }
        for (block.first..block.first + block.records) |at| {
            const record = streams.record(at);
            const row = block.row + (record & 0xff) / 2;
            const col = block.col + (record & 0xff) % 2;
            if (row < window.row or row >= window.row + window.rows or col < window.col or col >= window.col + window.cols) continue;
            out[@intCast((row - window.row) * window.cols + col - window.col)] = @intCast(record >> 8);
        }
    }
}
