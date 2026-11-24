//! A kept state's files by block (spec 101 WS-03), as TensorFold 1.0.12 keeps its own (`core/rows_store.zig`,
//! `rows_chain.zig`): a state's rows lie in blocks of `block_tokens` positions that every state over the same
//! tokens shares, and its own parts with the rows past its last whole block in its own file. Host only and no
//! model's: a part is bytes, and what a row of it is the model says by `Part.per`.
//!
//! The bytes a state is (`learned.zig`'s image: a header, its tokens, the table of its parts, the parts in
//! order) stay as they were, a part whole; `Layout` maps them onto the files. A part of rows (`per` > 0: a row
//! every `per` positions, `at / per` rows below the state's position `at`) gives block k its rows of positions
//! [256 k, 256 k + 256), the parts' rows one after another; what is past the last whole block, and every part
//! that is not rows, goes in the state's file in the parts' order.
//!
//! A block is named by SHA-256 chained over the identity, the rows' layout and the tokens through it, as
//! TensorFold names its own, vLLM a block (`kv_offload/file_mapper.py`) and SGLang a page (`get_hash_str`).
//! Two departures. The lookahead is 1 for every part: the head's cache's entry p - 1 is absorbed with token
//! p (`models/qwen4/session/learnt.zig`), so block k's name covers token 256 (k + 1) too, and a block is whole
//! where that token is the state's (256 (k + 1) <= at). No decoded spans and no chunk starts: TensorFold names
//! them because its rows can differ by the window that wrote them; here a row's bits are its tokens' alone
//! (the row rule), so the tokens are the whole name. A block file holds its full name and is checked by it, and
//! a state's file its tokens, compared on a read: the names of its blocks come from tokens known to be its.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Sha = std.crypto.hash.sha2.Sha256;

/// Positions a block holds: 256, TensorFold's, a quarter of a pool extent's (`extents.positions`). A test
/// lowers it; a block's name holds it.
pub var block_tokens: u32 = 256;
pub const lookahead = 1;

/// A part of a state's image: its bytes, and the positions a row of it stands for (0: not rows).
pub const Part = extern struct { bytes: u64, per: u32 = 0, pad: u32 = 0 };

pub const Name = [32]u8;
/// A name's first 16 bytes: its file's stem.
pub const Stem = [16]u8;

pub fn stem(name: Name) Stem {
    return name[0..16].*;
}

pub const block_magic: [8]u8 = "zsblock\x01".*;

/// A block file's first bytes; its rows follow.
pub const BlockHeader = extern struct { magic: [8]u8, identity: u64, name: Name, bytes: u64 };

/// `STEM.blk`, and `STEM.blk.part` while it is written.
pub fn blockFile(name: Name) [36]u8 {
    return stemFile(stem(name));
}

pub fn stemFile(one: Stem) [36]u8 {
    var out: [36]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{s}.blk", .{std.fmt.bytesToHex(one, .lower)}) catch unreachable;
    return out;
}

pub fn blockPart(name: Name) [41]u8 {
    var out: [41]u8 = undefined;
    _ = std.fmt.bufPrint(&out, "{s}.blk.part", .{std.fmt.bytesToHex(stem(name), .lower)}) catch unreachable;
    return out;
}

/// The stem a block file's name gives, null where it is no block's.
pub fn stemIn(file: []const u8) ?Stem {
    if (file.len != 36 or !std.mem.endsWith(u8, file, ".blk")) return null;
    var out: Stem = undefined;
    _ = std.fmt.hexToBytes(&out, file[0..32]) catch return null;
    return out;
}

/// A run of the parts' bytes in one file: from byte `at` of the parts, `len` bytes, at `offset` of block
/// `block`'s rows, or of the state file's own bytes where `block` is null.
pub const Segment = struct { at: u64, len: u64, block: ?u32, offset: u64 };

pub const Layout = struct {
    /// In the parts' order, back to back.
    segments: []Segment,
    /// Each whole block's name.
    names: []Name,
    /// For each part of rows, the index of its segment in block 0: block k's is that plus k.
    rows: []u32,
    /// A block's rows, the state file's own bytes, every part's.
    block_bytes: u64,
    own_bytes: u64,
    total: u64,

    /// The layout of a state of `parts` over `tokens` (its position is the last of them) for `identity`.
    pub fn init(gpa: Allocator, identity: u64, parts: []const Part, tokens: []const i32) !Layout {
        const at = tokens.len -| 1;
        var seed = Sha.init(.{});
        seed.update("zspark rows 1");
        words(&seed, &.{ @truncate(identity), @truncate(identity >> 32), block_tokens, lookahead });
        var rowed: usize = 0;
        for (parts) |part| if (rowBytes(part, at)) |row| {
            words(&seed, &.{ part.per, @truncate(row), @truncate(row >> 32) });
            rowed += 1;
        };
        const whole: usize = if (rowed > 0) at / block_tokens else 0;
        var segments: std.ArrayList(Segment) = .empty;
        errdefer segments.deinit(gpa);
        const rows = try gpa.alloc(u32, if (whole > 0) rowed else 0);
        errdefer gpa.free(rows);
        var self: Layout = .{ .segments = &.{}, .names = &.{}, .rows = rows, .block_bytes = 0, .own_bytes = 0, .total = 0 };
        var j: usize = 0;
        for (parts) |part| {
            const row = if (whole > 0) rowBytes(part, at) else null;
            if (row) |bytes| {
                const run = block_tokens / part.per * bytes;
                rows[j] = @intCast(segments.items.len);
                j += 1;
                for (0..whole) |k| try segments.append(gpa, .{ .at = self.total + k * run, .len = run, .block = @intCast(k), .offset = self.block_bytes });
                self.block_bytes += run;
                const tail = part.bytes - whole * run;
                if (tail > 0) try segments.append(gpa, .{ .at = self.total + whole * run, .len = tail, .block = null, .offset = self.own_bytes });
                self.own_bytes += tail;
            } else if (part.bytes > 0) {
                try segments.append(gpa, .{ .at = self.total, .len = part.bytes, .block = null, .offset = self.own_bytes });
                self.own_bytes += part.bytes;
            }
            self.total += part.bytes;
        }
        const names = try gpa.alloc(Name, whole);
        errdefer gpa.free(names);
        var chain = seed.finalResult();
        for (names, 0..) |*name, k| {
            const hi = (k + 1) * block_tokens;
            var h = Sha.init(.{});
            h.update(&chain);
            h.update(std.mem.sliceAsBytes(tokens[k * block_tokens .. hi]));
            chain = h.finalResult();
            var b = Sha.init(.{});
            b.update(&chain);
            b.update(std.mem.sliceAsBytes(tokens[hi..][0..lookahead]));
            name.* = b.finalResult();
        }
        self.names = names;
        self.segments = try segments.toOwnedSlice(gpa);
        return self;
    }

    pub fn deinit(self: *Layout, gpa: Allocator) void {
        gpa.free(self.segments);
        gpa.free(self.names);
        gpa.free(self.rows);
        self.* = undefined;
    }

    /// A block file's bytes.
    pub fn blockBytes(self: *const Layout) u64 {
        return @sizeOf(BlockHeader) + self.block_bytes;
    }

    /// Block k's segment of the j-th part of rows.
    pub fn slice(self: *const Layout, j: usize, k: usize) Segment {
        return self.segments[self.rows[j] + k];
    }

    /// The segment that holds byte `at` of the parts (`at` < `total`).
    pub fn locate(self: *const Layout, at: u64) Segment {
        var lo: usize = 0;
        var hi: usize = self.segments.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (self.segments[mid].at <= at) lo = mid else hi = mid;
        }
        return self.segments[lo];
    }
};

/// A part's bytes a row, where it is rows that blocks can hold.
fn rowBytes(part: Part, at: usize) ?u64 {
    if (part.per == 0 or block_tokens % part.per != 0) return null;
    const rows = at / part.per;
    if (rows == 0 or part.bytes % rows != 0) return null;
    return part.bytes / rows;
}

fn words(h: *Sha, list: []const u32) void {
    for (list) |x| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, x, .little);
        h.update(&b);
    }
}
