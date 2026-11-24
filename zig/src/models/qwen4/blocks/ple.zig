//! Flash Next's n-gram embedding branch (PLE), on the one layer that carries it. Each token hashes with the tokens
//! before it into `heads` rows of a table kept memory-mapped on the host; the rows, side by side, are the token's
//! embedding. Its value projection, gated per stream by the key projection against the stream, goes through a
//! dilated convolution over the sequence's last rows and is added to the streams.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../../weights/bf16.zig");
const cuda = @import("../../../cuda/cuda.zig");
const Drive = @import("../../../weights/drive.zig").Drive;
const ahead_file = @import("../../../weights/ahead.zig");
const face = @import("../../../weights/face.zig");
const Face = face.Face;
const qwen4_config = @import("../config.zig");
const layout = @import("../layout.zig");
const numeric = @import("../../../weights/numeric.zig");
const nvfp4 = @import("../../../weights/nvfp4.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const shards_file = @import("../../../weights/shards.zig");
const Shards = shards_file.Shards;
const state = @import("state.zig");
const Config = qwen4_config.Config;
const Ngram = qwen4_config.Ngram;

/// The sizes of a PLE.
pub const Shape = struct {
    /// Values of a stream.
    hidden: usize,
    streams: usize,
    /// Table rows a token gathers.
    heads: usize,
    /// Values of a table row.
    dims: usize,
    /// The convolution's taps, `dilation` rows apart.
    taps: usize,
    dilation: usize,
    /// Tokens of the longest n-gram.
    ngram: usize,

    pub fn of(config: Config) Shape {
        return .{
            .hidden = config.hidden,
            .streams = config.streams,
            .heads = config.ngramHeads(),
            .dims = config.ngramDims(),
            .taps = config.ple_kernel,
            .dilation = config.ngram_size,
            .ngram = config.ngram_size,
        };
    }

    pub fn wide(self: Shape) usize {
        return self.streams * self.hidden;
    }

    /// Values of a token's embedding: its rows side by side.
    pub fn embedding(self: Shape) usize {
        return self.heads * self.dims;
    }

    /// Rows before a row that its convolution reads.
    pub fn tailRows(self: Shape) usize {
        return (self.taps - 1) * self.dilation;
    }
};

/// Table rows [tokens.len, heads] for `tokens` that follow `history`, the `ngram_size - 1` tokens before them.
/// A token mixes with the tokens before it back to the last `eos`, which stands for every earlier one: head
/// block g - 2 hashes the g-gram ending at the token. As TensorFold's cuda/ngram.py `NGram.ids`.
pub fn ngramIds(ngram: Ngram, eos: u32, history: []const u32, tokens: []const u32, out: []u64) void {
    const n = ngram.multipliers.len;
    const heads = ngram.head_sizes.len;
    const per = heads / (n - 1);
    std.debug.assert(history.len == n - 1 and out.len == tokens.len * heads);
    // tokens between the last eos and the one at hand
    var run: usize = 0;
    for (history) |token| run = if (token == eos) 0 else run + 1;
    for (tokens, 0..) |token, i| {
        var mixed: i64 = 0;
        for (0..n) |p| {
            const back: i64 = if (p > run) eos else if (p <= i) tokens[i - p] else history[history.len - (p - i)];
            mixed ^= back *% ngram.multipliers[p];
            if (p == 0) continue;
            for ((p - 1) * per..p * per) |head| {
                const size: i64 = @intCast(ngram.head_sizes[head]);
                out[i * heads + head] = @as(u64, @intCast(@mod(mixed, size))) + ngram.head_offsets[head];
            }
        }
        run = if (token == eos) 0 else run + 1;
    }
}

/// An n-gram table on the host: its shards stay in the checkpoint's mapping, and a lookup copies the rows asked
/// for, so the pages read are the pages faulted in. As TensorFold's host_table.py `BF16Table` and `NVFP4Table`.
pub const Table = struct {
    kind: Kind,
    /// Each shard's rows as stored, [rows, dims] (nvfp4: two E2M1 codes a byte [rows, dims/2], the lower value
    /// in the low nibble), in row order.
    shards: []const []const u8,
    /// nvfp4: each shard's e4m3 block scales [rows, dims/16]; empty otherwise.
    blocks: []const []const u8 = &.{},
    /// The first row of each shard, then the table's row count (`Shards.starts`).
    starts: []const u64,
    dims: usize,
    /// What the embedding multiplies the gathered rows by: a bf16 table's `weight_scale`; 1 for an NVFP4 one,
    /// whose rows are decoded with it (`global`).
    scale: f32,
    /// nvfp4: the table's `weight_scale_2`, which a decoded value is multiplied by before its one rounding.
    global: f32 = 1.0,
    /// The drive the shards' file is on, told of every gather so that it stays awake between a reply's
    /// rounds (`weights/drive.zig`); the model's, and null where none is kept (a table that is no file's).
    drive: ?*Drive = null,
    /// [shards]: each shard's file and where its bytes lie in it (`readFrom`); empty for a table that is no
    /// file's. Owned by the table `readFrom` was called on, as `shards` is.
    files: []const File = &.{},

    pub const Kind = enum { bf16, nvfp4 };
    /// A shard's file (its O_DIRECT descriptor, which advice reads ahead through the page cache all the same)
    /// and the offsets of its rows and block scales in it; `fd` -1 where it is not known.
    pub const File = struct { fd: std.posix.fd_t = -1, rows: u64 = 0, blocks: u64 = 0 };
    pub const Error = error{ UnsupportedScheme, InvalidCheckpoint } || Allocator.Error;

    /// 16-bit words `gather` writes a row: its bf16 values.
    pub fn stride(self: Table) usize {
        return self.dims;
    }

    /// Bytes of a stored row.
    fn rowBytes(self: Table) usize {
        return switch (self.kind) {
            .bf16 => self.dims * 2,
            .nvfp4 => self.dims / 2,
        };
    }

    /// Shard bounds over `starts`: stored row width and `shards.len`.
    pub fn index(self: Table) Shards {
        return .{ .starts = self.starts, .width = self.rowBytes(), .count = self.shards.len };
    }

    /// `table`'s shards must all be present; they must outlive the result.
    pub fn init(gpa: Allocator, table: layout.Table) Error!Table {
        const kind: Kind = switch (table.scheme) {
            .bf16 => .bf16,
            .nvfp4 => .nvfp4,
            else => {
                std.log.err("the n-gram table is stored as {t}; zspark gathers bf16 and NVFP4 tables", .{table.scheme});
                return error.UnsupportedScheme;
            },
        };
        if (kind == .nvfp4 and table.dims % 16 != 0) return error.InvalidCheckpoint;
        const width: usize = if (kind == .nvfp4) 1 else 2;
        const columns: u64 = if (kind == .nvfp4) table.dims / 2 else table.dims;
        const shards = try gpa.alloc([]const u8, table.shards.len);
        errdefer gpa.free(shards);
        const blocks = try gpa.alloc([]const u8, if (kind == .nvfp4) table.shards.len else 0);
        errdefer gpa.free(blocks);
        const starts = try gpa.alloc(u64, table.shards.len + 1);
        errdefer gpa.free(starts);
        starts[0] = 0;
        for (table.shards, shards, 0..) |shard, *bytes, i| {
            const weight = shard.weight;
            if (weight.shape.len != 2 or weight.shape[1] != columns or weight.data.len != weight.count() * width) {
                std.log.err("n-gram shard {d} is {t} {any}; rows of {d} {t} values are read", .{ i, weight.dtype, weight.shape, table.dims, kind });
                return error.InvalidCheckpoint;
            }
            bytes.* = weight.data;
            // rows are read one here, one there: reading ahead of a fault would fetch pages no row needs
            advise(weight.data, std.os.linux.MADV.RANDOM);
            if (kind == .nvfp4) {
                const block = shard.scale orelse {
                    std.log.err("n-gram shard {d} is NVFP4 without its block scales", .{i});
                    return error.InvalidCheckpoint;
                };
                if (block.dtype != .F8_E4M3 or block.data.len != weight.shape[0] * (table.dims / 16)) {
                    std.log.err("n-gram shard {d}'s block scales are {t} {any}; e4m3 [{d}, {d}] are read", .{ i, block.dtype, block.shape, weight.shape[0], table.dims / 16 });
                    return error.InvalidCheckpoint;
                }
                blocks[i] = block.data;
                advise(block.data, std.os.linux.MADV.RANDOM);
            }
            starts[i + 1] = starts[i] + weight.shape[0];
            // a lookup touches a row here and a row there: read-ahead would only evict useful pages
            const page = std.heap.pageSize();
            const from = std.mem.alignBackward(usize, @intFromPtr(weight.data.ptr), page);
            const span = @intFromPtr(weight.data.ptr) + weight.data.len - from;
            if (span > 0) std.posix.madvise(@ptrFromInt(from), span, std.posix.MADV.RANDOM) catch {};
        }
        if (starts[table.shards.len] != table.rows) {
            std.log.err("the n-gram shards hold {d} rows, expected {d}", .{ starts[table.shards.len], table.rows });
            return error.InvalidCheckpoint;
        }
        var scale: f32 = 1.0;
        if (table.scale) |tensor| scale = switch (tensor.dtype) {
            .F32 => @bitCast(std.mem.readInt(u32, tensor.data[0..4], .little)),
            .BF16 => numeric.f32FromBf16(std.mem.readInt(u16, tensor.data[0..2], .little)),
            else => return error.InvalidCheckpoint,
        };
        var self: Table = .{ .kind = kind, .shards = shards, .blocks = blocks, .starts = starts, .dims = table.dims, .scale = scale };
        if (kind == .nvfp4) {
            // TensorFold's NVFP4Table: bf16(fp32(code x block scale) x weight_scale_2), the embedding's scale 1
            self.global = scale;
            self.scale = 1.0;
        }
        return self;
    }

    pub fn deinit(self: *Table, gpa: Allocator) void {
        gpa.free(self.shards);
        gpa.free(self.blocks);
        gpa.free(self.starts);
        if (self.files.len > 0) gpa.free(self.files);
        self.* = undefined;
    }

    pub fn rows(self: Table) u64 {
        return self.index().rows();
    }

    /// Ids from which a gather is cut into parts, each on a thread of its own: 128 tokens' rows, about 3 ms on
    /// one core.
    pub const part_ids = 2048;
    /// The most parts a gather is cut into, its caller's thread one of them.
    pub const most_parts = 8;

    /// Rows `ids` as bf16 bits into `out` [ids.len, dims]. A wide gather (a prompt's window: 2,048 tokens'
    /// 32,768 rows took about 50 ms on one core, spec 149 WS-05) is cut into contiguous parts gathered side by
    /// side, the caller's thread one of them; a row's bytes are its own id's whichever thread writes them.
    pub fn gather(self: Table, ids: []const u64, out: []u16) error{RowOutOfRange}!void {
        const columns = self.stride();
        std.debug.assert(out.len == ids.len * columns);
        if (self.drive) |drive| drive.touch();
        const parts = @min(most_parts, ids.len / part_ids);
        if (parts < 2) return self.gatherPart(ids, out);
        const Part = struct {
            fn run(table: Table, part: []const u64, into: []u16, failed: *std.atomic.Value(bool)) void {
                table.gatherPart(part, into) catch failed.store(true, .monotonic);
            }
        };
        var failed: std.atomic.Value(bool) = .init(false);
        var threads: [most_parts - 1]?std.Thread = @splat(null);
        const per = std.math.divCeil(usize, ids.len, parts) catch unreachable;
        var from = per;
        for (&threads) |*thread| {
            if (from >= ids.len) break;
            const to = @min(from + per, ids.len);
            // a thread that cannot be started leaves its part to this one
            thread.* = std.Thread.spawn(.{}, Part.run, .{ self, ids[from..to], out[from * columns .. to * columns], &failed }) catch null;
            if (thread.* == null) Part.run(self, ids[from..to], out[from * columns .. to * columns], &failed);
            from = to;
        }
        Part.run(self, ids[0..per], out[0 .. per * columns], &failed);
        for (threads) |held| if (held) |thread| thread.join();
        if (failed.load(.monotonic)) return error.RowOutOfRange;
    }

    /// Where each shard's bytes lie in its file, from `checkpoint`'s files: its rows read ahead through the
    /// file's descriptor (`readAhead`). A shard no file holds keeps advising its pages through the mapping.
    pub fn readFrom(self: *Table, gpa: Allocator, checkpoint: *const safetensors.Checkpoint) Allocator.Error!void {
        const files = try gpa.alloc(File, self.shards.len);
        for (files, self.shards, 0..) |*file, rows_bytes, i| {
            file.* = .{};
            const held = checkpoint.fileOf(rows_bytes) orelse continue;
            if (held.direct < 0) continue;
            const blocks_at: u64 = if (self.kind == .nvfp4) at: {
                const blocks = checkpoint.fileOf(self.blocks[i]) orelse continue;
                if (blocks != held) continue;
                break :at @intFromPtr(self.blocks[i].ptr) - @intFromPtr(held.map.ptr);
            } else 0;
            file.* = .{ .fd = held.direct, .rows = @intFromPtr(rows_bytes.ptr) - @intFromPtr(held.map.ptr), .blocks = blocks_at };
        }
        if (self.files.len > 0) gpa.free(self.files);
        self.files = files;
    }

    /// `gather` of `ids` on the calling thread.
    fn gatherPart(self: Table, ids: []const u64, out: []u16) error{RowOutOfRange}!void {
        const columns = self.stride();
        // The table is far larger than what is kept in memory beside the model, so most rows are still on the
        // disk. Every row's read is asked for before the first is waited on: the disk serves them together,
        // where faulting them in one after another cost a step 6.7 ms a row.
        for (ids) |id| try self.readAhead(id);
        for (ids, 0..) |id, i| {
            const bytes = try self.row(id);
            const into = out[i * columns ..][0..columns];
            switch (self.kind) {
                .bf16 => @memcpy(std.mem.sliceAsBytes(into), bytes),
                .nvfp4 => {
                    const blocks = try self.blockRow(id);
                    for (blocks, 0..) |block, b| {
                        // a code times an e4m3 scale is exact in fp32; the table's scale rounds once, bf16 once
                        const scale = numeric.f32FromE4m3(block);
                        for (bytes[b * 8 ..][0..8], 0..) |byte, pair| {
                            const at = into[b * 16 + 2 * pair ..][0..2];
                            at[0] = numeric.bf16FromF32(nvfp4.e2m1[byte & 0xf] * scale * self.global);
                            at[1] = numeric.bf16FromF32(nvfp4.e2m1[byte >> 4] * scale * self.global);
                        }
                    }
                },
            }
        }
    }

    /// Asks for row `id`'s pages to be read in. Through the file's descriptor where it is known
    /// (`posix_fadvise`): an `madvise` takes the process's mapping lock, and 32,768 rows' took 27 ms of a
    /// 2,048-token window's 40 and did not shrink on eight threads; the file's advice took 7 on eight (spec 149
    /// WS-05). Advice only: the bytes are read through the mapping either way.
    fn readAhead(self: Table, id: u64) error{RowOutOfRange}!void {
        const at = try self.locate(id);
        if (self.files.len > 0 and self.files[at.shard].fd >= 0) {
            const file = self.files[at.shard];
            const width = self.rowBytes();
            _ = std.os.linux.fadvise(file.fd, @intCast(file.rows + at.row * width), @intCast(width), std.os.linux.POSIX_FADV.WILLNEED);
            if (self.kind == .nvfp4) {
                const blocks = self.dims / 16;
                _ = std.os.linux.fadvise(file.fd, @intCast(file.blocks + at.row * blocks), @intCast(blocks), std.os.linux.POSIX_FADV.WILLNEED);
            }
            return;
        }
        advise(try self.row(id), std.os.linux.MADV.WILLNEED);
        if (self.kind == .nvfp4) advise(try self.blockRow(id), std.os.linux.MADV.WILLNEED);
    }

    /// An NVFP4 row's block scales as stored.
    fn blockRow(self: Table, id: u64) error{RowOutOfRange}![]const u8 {
        const at = try self.locate(id);
        const width = self.dims / 16;
        return self.blocks[at.shard][at.row * width ..][0..width];
    }

    /// Row `id` as stored.
    fn row(self: Table, id: u64) error{RowOutOfRange}![]const u8 {
        const at = try self.locate(id);
        const width = self.rowBytes();
        return self.shards[at.shard][at.row * width ..][0..width];
    }

    /// The shard holding row `id` and the row's index in it.
    fn locate(self: Table, id: u64) error{RowOutOfRange}!shards_file.At {
        return self.index().locate(id);
    }
};

/// Advice on the pages `bytes` touches. A refusal is ignored: it is advice, and memory that is no file's (a
/// test's table) has nothing to read ahead.
fn advise(bytes: []const u8, advice: u32) void {
    if (bytes.len == 0) return;
    const page = std.heap.pageSize();
    const start = std.mem.alignBackward(usize, @intFromPtr(bytes.ptr), page);
    const end = std.mem.alignForward(usize, @intFromPtr(bytes.ptr) + bytes.len, page);
    _ = std.os.linux.madvise(@ptrFromInt(start), end - start, advice);
}

/// A layer's PLE weights on the device.
pub const Weights = struct {
    shape: Shape,
    /// The norms' epsilon.
    eps: f32,
    /// What the staged rows are multiplied by (`Table.scale`).
    scale: f32,
    /// [streams * hidden, heads * dims]: BF16
    key: Face,
    /// [hidden, heads * dims]
    value: Face,
    /// fp32 [streams * hidden], as scales (gamma)
    norm_key: cuda.Buffer,
    /// fp32 [streams * hidden]
    norm_query: cuda.Buffer,
    /// fp32 [streams * hidden]
    norm_conv: cuda.Buffer,
    /// bf16 [streams * hidden, taps], the oldest tap first: the checkpoint's [channels, 1, taps] as stored
    conv: cuda.Buffer,

    /// What `init` reads from the checkpoint, on the host.
    pub const Host = struct {
        /// bf16 [streams * hidden, heads * dims], row-major bytes
        key: []const u8,
        /// bf16 [hidden, heads * dims]
        value: []const u8,
        /// [streams * hidden] each, as scales (gamma)
        norm_key: []const f32,
        norm_query: []const f32,
        norm_conv: []const f32,
        /// bf16 [streams * hidden, taps]
        conv: []const u8,
    };

    pub const Error = error{ UnsupportedScheme, UnsupportedShape } || Allocator.Error || cuda.Error;

    /// The weights of `ple` from the checkpoint's mapping. `centred`: the norms are stored as gamma - 1
    /// (`Layout.norms_centred`). `scale` is the table's (`Table.scale`). Calls take up to `max_rows` rows.
    pub fn init(gpa: Allocator, config: Config, ple: layout.Ple, centred: bool, scale: f32, scratch: cuda.Scratch.Rows) Error!Weights {
        const shape: Shape = .of(config);
        for ([_]layout.Linear{ ple.key, ple.value }) |linear| {
            if (linear.scheme != .bf16 or linear.weight.dtype != .BF16) {
                std.log.err("{s}: stored as {t} ({t}); the PLE's projections are read as BF16", .{ linear.name, linear.scheme, linear.weight.dtype });
                return error.UnsupportedScheme;
            }
        }
        const norms = try gpa.alloc(f32, 3 * shape.wide());
        defer gpa.free(norms);
        for ([_]layout.Vector{ ple.norm_key, ple.norm_query, ple.norm_conv }, 0..) |vector, v| {
            std.debug.assert(vector.len() == shape.wide());
            const offset: f32 = if (centred and vector.gamma) 1.0 else 0.0;
            for (norms[v * shape.wide() ..][0..shape.wide()], 0..) |*value, i| value.* = offset + vector.value(i);
        }
        return initHost(shape, @floatCast(config.eps), scale, .{
            .key = ple.key.weight.data,
            .value = ple.value.weight.data,
            .norm_key = norms[0..shape.wide()],
            .norm_query = norms[shape.wide()..][0..shape.wide()],
            .norm_conv = norms[2 * shape.wide() ..],
            .conv = ple.conv.tensor.data,
        }, scratch);
    }

    pub fn initHost(shape: Shape, eps: f32, scale: f32, host: Host, scratch: cuda.Scratch.Rows) (error{ UnsupportedShape, OutOfMemory } || cuda.Error)!Weights {
        const wide = shape.wide();
        var self: Weights = undefined;
        self.shape = shape;
        self.eps = eps;
        self.scale = scale;
        self.key = try .init(host.key, wide, shape.embedding(), scratch);
        errdefer self.key.deinit();
        self.value = try .init(host.value, shape.hidden, shape.embedding(), scratch);
        errdefer self.value.deinit();
        self.norm_key = try uploaded(.f32, &.{wide}, std.mem.sliceAsBytes(host.norm_key));
        errdefer self.norm_key.deinit();
        self.norm_query = try uploaded(.f32, &.{wide}, std.mem.sliceAsBytes(host.norm_query));
        errdefer self.norm_query.deinit();
        self.norm_conv = try uploaded(.f32, &.{wide}, std.mem.sliceAsBytes(host.norm_conv));
        errdefer self.norm_conv.deinit();
        self.conv = try uploaded(.bf16, &.{ wide, shape.taps }, host.conv);
        return self;
    }

    pub fn deinit(self: *Weights) void {
        self.key.deinit();
        self.value.deinit();
        self.norm_key.deinit();
        self.norm_query.deinit();
        self.norm_conv.deinit();
        self.conv.deinit();
        self.* = undefined;
    }
};

pub fn uploaded(dtype: cuda.DType, shape: []const usize, bytes: []const u8) cuda.Error!cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

/// Staging and scratch for windows of up to `rows` rows. `nrow` is what `commit` slides the tail over.
pub const Buffers = struct {
    rows: usize,
    shape: Shape,
    /// [rows, heads]: the window's table rows
    ids: []u64,
    /// bf16 bits [rows * heads, dims]: the gathered rows before they go to the device.
    host: []u16,
    /// bf16 [rows * heads, dims]: the gathered rows, a token's heads in order
    staged: cuda.Buffer,
    /// bf16 [rows, heads * dims]: the staged rows times a bf16 table's scale; unused when the scale is 1.
    emb: cuda.Buffer,
    /// bf16 [rows, streams * hidden]
    keys: cuda.Buffer,
    /// bf16 [rows, hidden]
    vals: cuda.Buffer,
    /// bf16 [rows, streams * hidden]
    gated: cuda.Buffer,
    /// fp32 [rows, streams]: each stream's gated squares summed
    pss: cuda.Buffer,
    /// bf16 [rows, streams * hidden]: the rows the convolution read, which the tail keeps
    nrow: cuda.Buffer,

    pub fn init(gpa: Allocator, shape: Shape, rows: usize) (Allocator.Error || cuda.Error)!Buffers {
        var self: Buffers = undefined;
        self.rows = rows;
        self.shape = shape;
        self.ids = try gpa.alloc(u64, rows * shape.heads);
        errdefer gpa.free(self.ids);
        self.host = try gpa.alloc(u16, rows * shape.embedding());
        errdefer gpa.free(self.host);
        self.staged = try .init(.bf16, &.{ rows * shape.heads, shape.dims });
        errdefer self.staged.deinit();
        self.emb = try .init(.bf16, &.{ rows, shape.embedding() });
        errdefer self.emb.deinit();
        self.keys = try .init(.bf16, &.{ rows, shape.wide() });
        errdefer self.keys.deinit();
        self.vals = try .init(.bf16, &.{ rows, shape.hidden });
        errdefer self.vals.deinit();
        self.gated = try .init(.bf16, &.{ rows, shape.wide() });
        errdefer self.gated.deinit();
        self.pss = try .init(.f32, &.{ rows, shape.streams });
        errdefer self.pss.deinit();
        self.nrow = try .init(.bf16, &.{ rows, shape.wide() });
        return self;
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(shape: Shape, rows: usize) usize {
        const of = cuda.Buffer.bytesOf;
        return of(.bf16, &.{ rows * shape.heads, shape.dims }) + of(.bf16, &.{ rows, shape.embedding() }) + 3 * of(.bf16, &.{ rows, shape.wide() }) +
            of(.bf16, &.{ rows, shape.hidden }) + of(.f32, &.{ rows, shape.streams });
    }

    pub fn deinit(self: *Buffers, gpa: Allocator) void {
        gpa.free(self.ids);
        gpa.free(self.host);
        self.staged.deinit();
        self.emb.deinit();
        self.keys.deinit();
        self.vals.deinit();
        self.gated.deinit();
        self.pss.deinit();
        self.nrow.deinit();
        self.* = undefined;
    }

    /// Copies the first `rows` tokens' gathered rows in `host`, each of `stride` words (`Table.stride`), to the
    /// device. The copy is not ordered against launches on another stream: the previous forward must have finished.
    pub fn upload(self: Buffers, rows: usize, stride: usize) cuda.Error!void {
        std.debug.assert(rows <= self.rows and stride <= self.shape.dims);
        const part: cuda.Buffer = .{ .view = .init(self.staged.view.ptr.?, .u16, &.{ rows * self.shape.heads, stride }) };
        return part.upload(std.mem.sliceAsBytes(self.host[0 .. rows * self.shape.heads * stride]));
    }
};

/// A window's table rows gathered before its turn, by a thread of its own while the device works: a prompt's
/// next window while the one before it runs (spec 087: gathered after it, the rows left the device idle 73 to
/// 95 ms a window of 2,048 not in the page cache), a reply's next round as its tokens become known, while the
/// head drafts (spec 089: a row's read waits 0.5 to 1.5 ms on the disk whatever the rows asked with it, and
/// nothing once asked a head pass before). Rows are their ids' alone, so they are handed over for the very
/// ids they were gathered for and for nothing else: a window nobody expected, or expected wrongly, is
/// gathered as before. N-gram ids and history stay here; the thread, the generation words and the handover
/// are `weights/ahead.zig`. The table's mapping must outlive the worker.
pub const Ahead = struct {
    const Worker = ahead_file.Ahead(Table);

    /// The asking thread's own: the ids of the window it is asked about, before they are compared and published.
    wanted: []u64,
    /// The asking thread's own: the history the window asked about follows.
    history: []u32,
    worker: Worker,
    /// A gather failed, a row out of range: the thread's to write, read once it is idle.
    failed: std.atomic.Value(bool) = .init(false),
    /// Windows `give` and `take` served rows to.
    served: usize = 0,

    pub fn init(gpa: Allocator, shape: Shape, rows: usize) Allocator.Error!Ahead {
        const wanted = try gpa.alloc(u64, rows * shape.heads);
        errdefer gpa.free(wanted);
        const history = try gpa.alloc(u32, shape.ngram - 1);
        errdefer gpa.free(history);
        return .{
            .wanted = wanted,
            .history = history,
            .worker = try .init(gpa, rows * shape.heads, shape.dims),
        };
    }

    /// `self` must not move once it has been asked: the thread holds its address.
    pub fn deinit(self: *Ahead, gpa: Allocator) void {
        self.worker.deinit(gpa);
        gpa.free(self.wanted);
        gpa.free(self.history);
        self.* = undefined;
    }

    /// How many of `ids` are asked for.
    pub fn held(self: *const Ahead) usize {
        return self.worker.held();
    }

    /// Asks for the rows of `next`, the window that follows `run` when all of `run` is kept; `run` follows
    /// `sequence`'s committed tokens. Where the rows held are those of `next`'s first tokens (a round's
    /// window asked again with one more draft), only the new ones are asked for; otherwise what was held is
    /// gone. A thread that cannot be started leaves nothing held.
    pub fn start(self: *Ahead, ngram: Ngram, table: Table, sequence: State, run: []const u32, next: []const u32) void {
        const want = next.len * ngram.head_sizes.len;
        if (next.len == 0 or want > self.worker.ids.len) {
            self.worker.drop();
            self.failed.store(false, .release);
            return;
        }
        std.debug.assert(sequence.history.len == self.history.len and self.worker.host.len >= want * table.stride());
        @memcpy(self.history, sequence.history);
        historyAfter(self.history, run, run.len);
        ngramIds(ngram, sequence.eos, self.history, next, self.wanted[0..want]);
        var have = self.held();
        const more = have > 0 and have <= want and table.shards.ptr == self.worker.source.shards.ptr and
            std.mem.eql(u64, self.worker.ids[0..have], self.wanted[0..have]);
        if (more and have == want) return;
        if (!more) {
            // the thread is idle from here until the count is published: the table is its to read after
            self.worker.adopt(table);
            self.failed.store(false, .release);
            have = 0;
        }
        // rows the thread was asked for stay as they are; it reads the new ones only once the count says so
        self.worker.ask(have, self.wanted[have..want]);
    }

    /// The rows held for the first of `ids`, `Table.gather`'s bytes for them: all of `ids` or, where fewer
    /// rows are held, as many, if the ids are the same in order; else none. A row's ids are its own tokens'
    /// and those before it, so the first rows of a window asked for are a shorter window's rows too. They
    /// stay until the next `start`.
    pub fn take(self: *Ahead, ids: []const u64) []const u16 {
        const rows = self.worker.take(ids);
        self.failed.store(self.worker.failed.load(.acquire), .release);
        if (rows.len != 0) self.served += 1;
        return rows;
    }

    /// `take` without waiting on the thread: the rows it has gathered so far for the first of `ids`, the same
    /// bytes, and the rest left to the caller's own gather (a pool round's stage, `rounds_stage.Ahead`).
    pub fn ready(self: *Ahead, ids: []const u64) []const u16 {
        const rows = self.worker.ready(ids);
        if (rows.len != 0) self.served += 1;
        return rows;
    }

    /// `take` with no copy: `host`, a `Buffers.host`, becomes the memory that holds the rows, and what it was
    /// is gathered into next. Returns how many of `ids` its rows are; 0, and nothing changed, where the rows
    /// held are not theirs. Rows handed over are held no more: the memory taken in their place is not
    /// theirs, so the same window run again (a prompt sent twice, from the same snapshot) gathers its own.
    pub fn give(self: *Ahead, ids: []const u64, host: *[]u16) usize {
        const n = self.worker.give(ids, host);
        self.failed.store(self.worker.failed.load(.acquire), .release);
        if (n != 0) self.served += 1;
        return n;
    }
};

/// What a committed sequence keeps for its next rows.
pub const State = struct {
    /// bf16 [1, (taps - 1) * dilation, streams * hidden]: the last rows the convolution read, oldest first;
    /// zeros before the sequence's first row
    tail: cuda.Buffer,
    /// The last `ngram_size - 1` committed tokens, oldest first; `eos` before the sequence's first.
    history: []u32,
    /// The token that resets the n-grams (`Config.ple_eos`).
    eos: u32,

    pub fn init(gpa: Allocator, shape: Shape, eos: u32) (Allocator.Error || cuda.Error)!State {
        const history = try gpa.alloc(u32, shape.ngram - 1);
        errdefer gpa.free(history);
        var self: State = .{ .tail = try .init(.bf16, &.{ 1, shape.tailRows(), shape.wide() }), .history = history, .eos = eos };
        errdefer self.tail.deinit();
        try self.reset();
        return self;
    }

    /// The device bytes `init` asks, with nothing made: the tail.
    pub fn bytesFor(shape: Shape) usize {
        return cuda.Buffer.bytesOf(.bf16, &.{ 1, shape.tailRows(), shape.wide() });
    }

    pub fn deinit(self: *State, gpa: Allocator) void {
        self.tail.deinit();
        gpa.free(self.history);
        self.* = undefined;
    }

    /// Back to before the sequence's first row.
    pub fn reset(self: *State) cuda.Error!void {
        @memset(self.history, self.eos);
        return self.tail.zero();
    }
};

/// The first `rows` rows of `buffer` as a view.
pub fn leading(buffer: cuda.Buffer, rows: usize) cuda.View {
    var view = buffer.view;
    view.size[0] = @intCast(rows);
    return view;
}

/// The host's work before a forward: the table rows of `tokens`, which follow the sequence's committed tokens,
/// gathered and copied to the device. See `Buffers.upload` for when it may run.
pub fn stage(ngram: Ngram, table: Table, sequence: State, buffers: Buffers, tokens: []const u32) (error{RowOutOfRange} || cuda.Error)!void {
    try gather(ngram, table, sequence, buffers, tokens);
    return buffers.upload(tokens.len, table.stride());
}

/// `stage` without the copy to the device: the table rows of `tokens` into `buffers.host`. It touches nothing
/// the device reads, so it may run while the previous forward's launches still do; `Buffers.upload` follows.
pub fn gather(ngram: Ngram, table: Table, sequence: State, buffers: Buffers, tokens: []const u32) error{RowOutOfRange}!void {
    const shape = buffers.shape;
    std.debug.assert(tokens.len <= buffers.rows and table.dims == shape.dims);
    const ids = buffers.ids[0 .. tokens.len * shape.heads];
    ngramIds(ngram, sequence.eos, sequence.history, tokens, ids);
    return table.gather(ids, buffers.host[0 .. ids.len * table.stride()]);
}

/// `gather`, but that rows `ahead` holds for this very window, or for its first tokens, are taken and not read
/// again: the same bytes in `buffers.host`, whose memory may then be what `ahead`'s was.
pub fn gatherAfter(ngram: Ngram, table: Table, sequence: State, buffers: *Buffers, ahead: *Ahead, tokens: []const u32) error{RowOutOfRange}!void {
    const shape = buffers.shape;
    std.debug.assert(tokens.len <= buffers.rows and table.dims == shape.dims);
    const ids = buffers.ids[0 .. tokens.len * shape.heads];
    ngramIds(ngram, sequence.eos, sequence.history, tokens, ids);
    const given = ahead.give(ids, &buffers.host);
    return table.gather(ids[given..], buffers.host[given * table.stride() .. ids.len * table.stride()]);
}

/// Adds the n-gram branch of the staged rows to h bf16 [rows, streams * hidden], in place. The streams must hold
/// what was pending for them. `buffers.nrow` keeps the rows for `commit`.
pub fn forward(weights: Weights, buffers: Buffers, sequence: State, h: cuda.View, rows: usize) cuda.Error!void {
    const shape = weights.shape;
    std.debug.assert(rows >= 1 and rows <= buffers.rows and h.size[0] == rows);
    const staged = leading(buffers.staged, rows * shape.heads);
    // a token's rows lie side by side as they were staged, so with no scale they are the embedding
    var emb = cuda.View.init(staged.ptr.?, .bf16, &.{ rows, shape.embedding() });
    if (weights.scale != 1.0) {
        emb = leading(buffers.emb, rows);
        try cuda.launched(cuda.kernels.zs_qwen4_ple_embed(&staged, &emb, @intCast(shape.heads), weights.scale));
    }
    const keys = leading(buffers.keys, rows);
    const vals = leading(buffers.vals, rows);
    const gated = leading(buffers.gated, rows);
    const pss = leading(buffers.pss, rows);
    const nrow = leading(buffers.nrow, rows);
    try weights.key.forward(emb, keys);
    try weights.value.forward(emb, vals);
    try cuda.launched(cuda.kernels.zs_qwen4_ple_gate(&keys, &vals, &h, &weights.norm_key.view, &weights.norm_query.view, &gated, &pss, weights.eps, @intCast(shape.streams)));
    try cuda.launched(zs_qwen4_ple_conv(&gated, &pss, &weights.norm_conv.view, &sequence.tail.view, &weights.conv.view, &h, &nrow, weights.eps, @intCast(shape.streams), @intCast(shape.dilation)));
}

/// `history`, a sequence's last tokens, after the first `keep` of `tokens` follow it.
pub fn historyAfter(history: []u32, tokens: []const u32, keep: usize) void {
    if (keep >= history.len) {
        @memcpy(history, tokens[keep - history.len .. keep]);
    } else {
        std.mem.copyForwards(u32, history[0 .. history.len - keep], history[keep..]);
        @memcpy(history[history.len - keep ..], tokens[0..keep]);
    }
}

/// Keeps the first `keep` rows of the window `tokens` that `forward` ran: they enter the history, and the tail
/// slides to end at the last of them.
pub fn commit(sequence: *State, buffers: Buffers, tokens: []const u32, keep: usize) cuda.Error!void {
    std.debug.assert(keep >= 1 and keep <= tokens.len and tokens.len <= buffers.rows);
    historyAfter(sequence.history, tokens, keep);
    if (sequence.tail.view.size[1] == 0) return;
    const wide = buffers.shape.wide();
    return state.shiftWindows(sequence.tail.view, .{ .view = leading(buffers.nrow, tokens.len), .layer = tokens.len * wide, .row = wide, .rows = tokens.len }, keep);
}

extern fn zs_qwen4_ple_conv(gated: *const cuda.View, pss: *const cuda.View, norm_conv: *const cuda.View, tail: *const cuda.View, conv: *const cuda.View, h: *const cuda.View, nrow: *const cuda.View, eps: f64, streams: i64, dilation: i64) c_int;
