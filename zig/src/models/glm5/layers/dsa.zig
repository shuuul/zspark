//! GLM-5.3-Flash's DSA (DeepSeek sparse attention) over a rank's heads, its dense rows: the low-rank
//! projections, a token's latent written to the layer's cache, and the attention over the latents with kv_b's
//! key blocks taken into the query and its value blocks applied to the result. TensorFold's `forward.py`
//! `dsa_block` and `_dsa_latent`, `latent.py`; the kernels are `zig/kernels/glm5/mla.cu` and `mla_proj.cu`.
//!
//! Row r of a window stands at position pos + r and reads the cache's rows 0 .. pos + r, its own among them,
//! which the same forward writes: a row's bits depend on its own input and the rows before it alone. With an
//! indexer, a row past the dense limit (`index_topk + index_kpool - 1`) then attends to its selected tokens
//! and overwrites that dense result; a row still inside the limit keeps it (its list's count is 0). The
//! indexer launches on every window, so a graph captured below the limit is the one replayed past it.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../../weights/bf16.zig");
const face = @import("../../../weights/face.zig");
const Face = face.Face;
const Part = face.Part;
const cuda = @import("../../../cuda/cuda.zig");
const Config = @import("config.zig").Config;
const hc = @import("hc.zig");
const index_launch = @import("index_launch.zig");
const host_index = @import("index.zig");
const nccl = @import("../../../ranks/nccl.zig");
const share = @import("share.zig");

/// A latent's width (`kv_lora_rank`), which the attention is written for.
pub const latent = 512;

/// How a cache holds a token's latent (`--kv-dtype`, spec 107 WS-03): bf16, or vLLM's GLM record (`fp8_ds_mla`):
/// 512 E4M3 codes and 4 fp32 scales, one a group of 128 values, its largest |value| / 448 (`mla.cuh`). A record is
/// its own token's alone, written by the window that holds the token and read as codes by every row after it.
pub const Storage = enum {
    bf16,
    fp8,

    pub fn parse(word: []const u8) ?Storage {
        return std.meta.stringToEnum(Storage, word);
    }

    /// What `serve`, `chat` and `generate` hold without `--kv-dtype`: FP8 on FP4-Local, as the card's vLLM
    /// (`--kv-cache-dtype fp8`; the owner, 2026-10-18).
    pub fn stored(c: *const Config) Storage {
        _ = c;
        return .fp8;
    }

    /// A token's bytes a layer.
    pub fn width(self: Storage) usize {
        return switch (self) {
            .bf16 => latent * 2,
            .fp8 => fp8_record,
        };
    }
};

/// An FP8 record: the codes, then the groups' scales.
pub const fp8_record = latent + latent / 128 * 4;
/// Keys a block of the attention takes: the cache's rows are attended in chunks of this many positions.
pub const chunk = 512;

pub const LoadError = error{ MissingTensor, UnsupportedShape } || Allocator.Error || cuda.Error;

/// The forward's: the launches' and a split's exchange.
pub const Error = cuda.Error || nccl.Error;

/// A layer's tensors, whole or a rank's shares: the projections as their checkpoint stores them (BF16, FP8_PB_WO
/// or MXFP8 in FP4-Local), the norms and `kv_b` (which `mla_proj.cu` reads itself) bf16 bytes.
pub const Parts = struct {
    q_a: Part,
    kv_a: Part,
    q_norm: []const u8,
    kv_norm: []const u8,
    q_b: Part,
    kv_b: []const u8,
    o: Part,

    /// The checkpoint's leaf of each field, in the fields' order.
    pub const leaves = [_][]const u8{
        "q_a_proj.weight",       "kv_a_proj_with_mqa.weight", "q_a_layernorm.weight",
        "kv_a_layernorm.weight", "q_b_proj.weight",           "kv_b_proj.weight",
        "o_proj.weight",
    };
};

/// A DSA layer's weights for `heads` heads.
pub const Weights = struct {
    /// [q_a | kv_a, dims]: the query's low rank and the latent, before their norms, in one product a row.
    proj: Face,
    /// bf16 [q_rank] and [512]
    q_norm: cuda.Buffer,
    kv_norm: cuda.Buffer,
    /// [heads key_dim, q_rank]
    q_b: Face,
    /// bf16 [heads, key_dim + value_dim, 512]: a head's key rows, then its value rows, as the checkpoint has them.
    kv_b: cuda.Buffer,
    /// [dims, heads value_dim]
    o: Face,
    heads: usize,
    q_rank: usize,
    key_dim: usize,
    value_dim: usize,
    eps: f32,
    /// key_dim ^ -1/2, on the scores.
    scale: f64,
    /// The indexer, when the layer has one. Without it every row stays dense.
    index: ?Index = null,

    /// `scratch` holds what a projection is packed in while it is.
    pub fn init(scratch: Allocator, parts: Parts, c: *const Config, rows: cuda.Scratch.Rows) LoadError!Weights {
        const dims = c.hidden;
        const per_head = (c.key_dim + c.value_dim) * latent * 2;
        if (c.kv_rank != latent or c.key_dim == 0 or c.key_dim % 16 != 0 or c.key_dim > 256 or c.value_dim == 0 or c.value_dim > 256) return error.UnsupportedShape;
        const heads = parts.kv_b.len / per_head;
        if (heads == 0 or parts.kv_b.len != heads * per_head) return error.UnsupportedShape;
        if (!share.shaped(parts.q_a, c.q_rank, dims) or !share.shaped(parts.kv_a, latent, dims) or parts.q_norm.len != c.q_rank * 2 or parts.kv_norm.len != latent * 2) return error.UnsupportedShape;
        if (!share.shaped(parts.q_b, heads * c.key_dim, c.q_rank) or !share.shaped(parts.o, dims, heads * c.value_dim)) return error.UnsupportedShape;

        var proj = try share.faceOf(scratch, &.{ parts.q_a, parts.kv_a }, dims, rows);
        errdefer proj.deinit();
        var q_b = try share.faceOf(scratch, &.{parts.q_b}, c.q_rank, rows);
        errdefer q_b.deinit();
        var o = try share.faceOf(scratch, &.{parts.o}, heads * c.value_dim, rows);
        errdefer o.deinit();
        var q_norm: cuda.Buffer = try .init(.bf16, &.{c.q_rank});
        errdefer q_norm.deinit();
        try q_norm.upload(parts.q_norm);
        var kv_norm: cuda.Buffer = try .init(.bf16, &.{latent});
        errdefer kv_norm.deinit();
        try kv_norm.upload(parts.kv_norm);
        var kv_b: cuda.Buffer = try .init(.bf16, &.{ heads, c.key_dim + c.value_dim, latent });
        errdefer kv_b.deinit();
        try kv_b.upload(parts.kv_b);
        return .{ .proj = proj, .q_norm = q_norm, .kv_norm = kv_norm, .q_b = q_b, .kv_b = kv_b, .o = o, .heads = heads, .q_rank = c.q_rank, .key_dim = c.key_dim, .value_dim = c.value_dim, .eps = @floatCast(c.eps), .scale = 1.0 / @sqrt(@as(f64, @floatFromInt(c.key_dim))), .index = null };
    }

    /// The DSA layer under `prefix` (`...layers.N.self_attn.`) of `source` (anything with `get(name)`) as `rank`
    /// holds it: half of the heads, the low-rank projections whole.
    pub fn load(arena: Allocator, source: anytype, prefix: []const u8, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!Weights {
        var parts: Parts = undefined;
        inline for (comptime std.meta.fieldNames(Parts), Parts.leaves) |field, leaf| {
            const name = try std.mem.concat(arena, u8, &.{ prefix, leaf });
            const module = name[0 .. name.len - ".weight".len];
            if (comptime @FieldType(Parts, field) == Part) {
                @field(parts, field) = try share.partOf(arena, source, c, module, rank);
            } else if (comptime std.mem.eql(u8, field, "kv_b")) {
                @field(parts, field) = try share.bf16Weight(arena, source, c, module, rank);
            } else {
                const tensor = try share.get(arena, source, name, rank) orelse {
                    std.log.err("{s}: not in the checkpoint", .{name});
                    return error.MissingTensor;
                };
                if (tensor.dtype != .BF16) return error.UnsupportedShape;
                @field(parts, field) = tensor.data;
            }
        }
        var self = try init(arena, parts, c, rows);
        errdefer self.deinit();
        self.index = try Index.load(arena, source, prefix, c, rank, rows);
        return self;
    }

    pub fn deinit(self: *Weights) void {
        self.proj.deinit();
        self.q_b.deinit();
        self.o.deinit();
        self.q_norm.deinit();
        self.kv_norm.deinit();
        self.kv_b.deinit();
        if (self.index) |*index| index.deinit();
        self.* = undefined;
    }

    /// The window x bf16 [rows, dims] at the cache's position through the layer: out fp32 [rows, dims], the
    /// rank's partial of the block. The rows' latents are written to `layer`'s cache at pos .. pos + rows - 1;
    /// the position does not move (`Cache.commit`).
    pub fn forward(self: Weights, cache: Cache, layer: usize, buffers: Buffers, x: cuda.View, out: cuda.View) Error!void {
        return self.forwardAt(cache, layer, buffers, x, out, cache.pos + @as(usize, @intCast(x.size[0])));
    }

    /// `forward` with its launches sized for `positions` keys, from the window's end to the capacity: the same
    /// bits, and a captured graph serves every window that ends under them.
    pub fn forwardAt(self: Weights, cache: Cache, layer: usize, buffers: Buffers, x: cuda.View, out: cuda.View, positions: usize) Error!void {
        try self.attendAt(cache, layer, buffers, x, positions);
        try self.emit(buffers, @intCast(x.size[0]), out);
    }

    /// `forwardAt` with `split` (`attendAtSplit`).
    pub fn forwardAtSplit(self: Weights, cache: Cache, layer: usize, buffers: Buffers, x: cuda.View, out: cuda.View, positions: usize, split: ?nccl.Comm) Error!void {
        try self.attendAtSplit(cache, layer, buffers, x, positions, split);
        try self.emit(buffers, @intCast(x.size[0]), out);
    }

    /// `forwardAt` but its last step (`emit`, or `emitRows` a part of the rows at a time, spec 116 WS-03).
    pub fn attendAt(self: Weights, cache: Cache, layer: usize, buffers: Buffers, x: cuda.View, positions: usize) Error!void {
        return self.attendAtSplit(cache, layer, buffers, x, positions, null);
    }

    /// `attendAt` with `split`: a prompt chunk's comm (`indexRowsSplit`); null, and decode's and loopback's,
    /// runs the whole window.
    pub fn attendAtSplit(self: Weights, cache: Cache, layer: usize, buffers: Buffers, x: cuda.View, positions: usize, split: ?nccl.Comm) Error!void {
        const rows: usize = @intCast(x.size[0]);
        try self.project(buffers, x);
        try self.indexProject(buffers, x);
        try self.write(cache, layer, buffers, 0, rows, positions);
        try self.indexRowsSplit(cache, layer, buffers, 0, rows, positions, split);
        try absorb(buffers.q.rows(rows), self.kv_b.view, self.key_dim, buffers.qa.rows(rows));
        try self.attend(cache, layer, buffers, 0, rows, positions);
    }

    /// The most keys a row attends in a context of `capacity`: with an indexer the list's width, which is the
    /// dense limit too, so the chunks' partials stop there whatever the context.
    pub fn attended(self: Weights, capacity: usize) usize {
        const index = self.index orelse return capacity;
        return @min(capacity, index.kept * index_pool + index_pool - 1);
    }

    /// A cache whose indexer storage matches this layer, or a dense cache when the layer has none.
    pub fn cacheFor(self: Weights, layers: usize, capacity: usize) cuda.Error!Cache {
        return Cache.holding(layers, capacity, if (self.index) |index| index.dim else null);
    }

    /// `forwardAt`'s first step over a window's rows, whichever sequences they are of: the queries
    /// (`buffers.q`) and the latents (`buffers.lat`). A row's are its own.
    pub fn project(self: Weights, buffers: Buffers, x: cuda.View) cuda.Error!void {
        const rows: usize = @intCast(x.size[0]);
        std.debug.assert(rows >= 1 and rows <= buffers.rows and buffers.heads == self.heads);
        const low = buffers.low.rows(rows);
        try self.proj.forward(x, low);
        // the query's low rank and the latent are columns of the product's rows
        var q_low = low;
        q_low.size[1] = @intCast(self.q_rank);
        q_low.row_stride = @intCast(self.q_rank + latent);
        var kv_low: cuda.View = .init(@ptrCast(@as([*]u8, @ptrCast(low.ptr.?)) + self.q_rank * 2), .bf16, &.{ rows, latent });
        kv_low.row_stride = @intCast(self.q_rank + latent);
        const q_normed = buffers.q_normed.rows(rows);
        const lat = buffers.lat.rows(rows);
        try hc.rmsnorm(q_low, self.q_norm.view, q_normed, self.eps);
        try hc.rmsnorm(kv_low, self.kv_norm.view, lat, self.eps);
        try self.q_b.forward(q_normed, buffers.q.rows(rows));
    }

    /// The projected rows `first` .. `first + rows - 1`, one sequence's, as latents of `layer` at `cache`'s
    /// position onward.
    pub fn write(self: Weights, cache: Cache, layer: usize, buffers: Buffers, first: usize, rows: usize, positions: usize) cuda.Error!void {
        _ = self;
        std.debug.assert(rows >= 1 and first + rows <= buffers.rows);
        std.debug.assert(cache.pos + rows <= positions and positions <= cache.capacity);
        const lat = buffers.lat.span(first, rows);
        const keys = cache.layer(layer);
        var stored = cache.indexer;
        const table: ?*const cuda.View = if (stored) |*one| &one.table.view else null;
        try cuda.launched(zs_glm5_mla_write(&lat, &keys, &cache.pos_dev.view, table, @intCast(positions)));
    }

    /// The same rows' absorbed queries (`buffers.qa`) over `cache`'s latents, theirs among them, into the same
    /// rows of `buffers.ol`. With an indexer, after `indexRows`: a row its list is positive for attends the
    /// listed latents alone.
    pub fn attend(self: Weights, cache: Cache, layer: usize, buffers: Buffers, first: usize, rows: usize, positions: usize) cuda.Error!void {
        std.debug.assert(rows >= 1 and first + rows <= buffers.rows and cache.pos + rows <= positions and positions <= cache.capacity);
        const keys = cache.layer(layer);
        const qa = buffers.qa.span(first, rows);
        const ol = buffers.ol.span(first, rows);
        var stored = cache.indexer;
        const table: ?*const cuda.View = if (stored) |*one| &one.table.view else null;
        var list: ?[2]cuda.View = null;
        if (self.index != null) {
            const scratch = buffers.indexer.?;
            list = .{ scratch.tokens.span(first, rows), scratch.counts.span(first, rows) };
        }
        try attendRows(qa, keys, cache.pos_dev.view, table, .{ buffers.po.view, buffers.pm.view, buffers.pl.view }, ol, list, self.scale, positions);
    }

    /// `forwardAt`'s last step over a window's rows: kv_b's value rows and `o`, into out fp32 [rows, dims].
    pub fn emit(self: Weights, buffers: Buffers, rows: usize, out: cuda.View) cuda.Error!void {
        std.debug.assert(out.dtype == .f32 or out.dtype == .bf16); // bf16: spec 116 WS-04
        const o = buffers.o.rows(rows);
        try expand(buffers.ol.rows(rows), self.kv_b.view, self.key_dim, o);
        try self.o.forward(o, out);
    }

    /// `emit` of the rows [first, first + count) alone, into those rows of out fp32 [rows, dims]: a row's bits
    /// are the whole's.
    pub fn emitRows(self: Weights, buffers: Buffers, first: usize, count: usize, out: cuda.View) cuda.Error!void {
        std.debug.assert(out.dtype == .f32 or out.dtype == .bf16); // bf16: spec 116 WS-04
        const o = buffers.o.span(first, count);
        try expand(buffers.ol.span(first, count), self.kv_b.view, self.key_dim, o);
        try self.o.forward(o, out.span(first, count));
    }

    /// The indexer's products over the window's rows: the key, the gate, the head weights and the query.
    /// A row's are its own. Nothing, without an indexer.
    pub fn indexProject(self: Weights, buffers: Buffers, x: cuda.View) cuda.Error!void {
        const index = self.index orelse return;
        const scratch = buffers.indexer.?;
        const rows: usize = @intCast(x.size[0]);
        try index.wk.forward(x, scratch.raw.rows(rows));
        try index.gate.forward(x, scratch.gate.rows(rows));
        try index.weights.forward(x, scratch.weights.rows(rows));
        var query = scratch.query.rows(rows);
        query.ndim = 2;
        query.size[1] = @intCast(index.heads * index.dim);
        try index.wq_b.forward(buffers.q_normed.rows(rows), query);
    }

    /// Rows `first` .. `first + rows - 1` written into `layer`'s index cache, pooled, scored and selected.
    /// `positions` sizes the launches, as `forwardAt`'s does. Nothing, without an indexer.
    pub fn indexRows(self: Weights, cache: Cache, layer: usize, buffers: Buffers, first: usize, rows: usize, positions: usize) Error!void {
        return self.indexRowsSplit(cache, layer, buffers, first, rows, positions, null);
    }

    /// `indexRows` with `split` (a prompt chunk's comm over `world` ranks, `index.split_from`): this rank
    /// scores and selects its block of the rows alone — `per` a rank, rank r's [r * per, r * per + per) — and
    /// the blocks trade their lists over it so every rank holds every row's list before the attention. The
    /// same kernels on the same rows, so a row's list is the full indexer's, bit for bit.
    pub fn indexRowsSplit(self: Weights, cache: Cache, layer: usize, buffers: Buffers, first: usize, rows: usize, positions: usize, split: ?nccl.Comm) Error!void {
        const index = self.index orelse return;
        const scratch = buffers.indexer.?;
        const stored = cache.indexer.?;
        std.debug.assert(rows >= 1 and first + rows <= buffers.rows and cache.pos + rows <= positions and positions <= cache.capacity);
        const split_comm = if (split) |comm| if (host_index.splitAt(cache.pos, host_index.split_from, comm.world)) comm else null else null;
        const per = if (split_comm) |comm| (rows + comm.world - 1) / comm.world else rows;
        const lo = if (split_comm) |comm| @min(rows, comm.rank * per) else 0;
        const hi = @min(rows, lo + per);
        const raw = scratch.raw.span(first, rows);
        const gate = scratch.gate.span(first, rows);
        const keys = stored.plane(stored.keys, layer, cache.span);
        const gates = stored.plane(stored.gates, layer, cache.span);
        try index_launch.write(&raw, &gate, &index.ln_w.view, &index.ln_b.view, &keys, &gates, &stored.table.view, &cache.pos_dev.view, rows, positions);
        const codes = stored.codesOf(layer, cache.pooled_rows);
        const scales = stored.scalesOf(layer, cache.pooled_rows);
        try index_launch.pool(&keys, &gates, &stored.table.view, &index.ape.view, &codes, &scales, &cache.pos_dev.view, rows, positions);
        if (hi > lo) {
            const query = scratch.query.span(first + lo, hi - lo);
            const weights = scratch.weights.span(first + lo, hi - lo);
            const q8 = scratch.q8.span(first + lo, hi - lo);
            const wq = scratch.wq.span(first + lo, hi - lo);
            try index_launch.query(&query, &weights, &q8, &wq);
            // the block's first row stands at cache.pos + lo: the kernels read its device copy, which a
            // split's own word carries where the cache's stands at the window's
            const pos = if (split_comm != null and lo > 0) at: {
                const word: i32 = @intCast(cache.pos + lo);
                try scratch.pos_lo.uploadLaunched(0, std.mem.asBytes(&word));
                break :at scratch.pos_lo.view;
            } else cache.pos_dev.view;
            const scores = scratch.scores.span(first + lo, hi - lo);
            try index_launch.score(&q8, &wq, &codes, &scales, &pos, &scores, positions);
            const tokens = scratch.tokens.span(first + lo, hi - lo);
            const counts = scratch.counts.span(first + lo, hi - lo);
            try index_launch.select(&scores, &pos, &tokens, &counts, index.kept, positions);
        }
        const comm = split_comm orelse return;
        // every block's lists and counts to every rank, where the full indexer's rows lie
        const width: usize = @intCast(scratch.tokens.view.size[1]);
        const base: [*]u8 = @ptrCast(scratch.tokens.view.ptr.?);
        const counts_base: [*]u8 = @ptrCast(scratch.counts.view.ptr.?);
        var moves: [7]nccl.Comm.Move = undefined;
        var count: usize = 0;
        for (0..comm.world) |peer| {
            if (peer == comm.rank) continue;
            const plo = @min(rows, peer * per);
            const phi = @min(rows, plo + per);
            moves[count] = .{
                .peer = peer,
                .send = @ptrCast(base + (first + lo) * width * @sizeOf(i32)),
                .send_count = (hi - lo) * width,
                .recv = @ptrCast(base + (first + plo) * width * @sizeOf(i32)),
                .recv_count = (phi - plo) * width,
            };
            count += 1;
        }
        try comm.exchange(.i32, moves[0..count]);
        count = 0;
        for (0..comm.world) |peer| {
            if (peer == comm.rank) continue;
            const plo = @min(rows, peer * per);
            const phi = @min(rows, plo + per);
            moves[count] = .{
                .peer = peer,
                .send = @ptrCast(counts_base + (first + lo) * @sizeOf(i32)),
                .send_count = hi - lo,
                .recv = @ptrCast(counts_base + (first + plo) * @sizeOf(i32)),
                .recv_count = phi - plo,
            };
            count += 1;
        }
        try comm.exchange(.i32, moves[0..count]);
    }
};

/// The indexer's weights. The products are tensor-core GEMMs, the head weights' written fp32 as the card's vLLM
/// projects them; the norm, the pool, the coding of the pooled keys and the queries, the scores and the list are
/// `index.cu`, the host model's arithmetic. `index_kpool` is 4, which that kernel is written for, and the tail
/// of an unfinished pool is always kept. The key a power of two from 16 to 128 values: the rotation's points.
pub const Index = struct {
    wk: Face,
    /// The gate's product is fp32, rounded to bf16 where it is stored.
    gate: bf16.Linear,
    weights: Face,
    wq_b: Face,
    /// bf16 [dim], and bf16 [4, dim]
    ln_w: cuda.Buffer,
    ln_b: cuda.Buffer,
    ape: cuda.Buffer,
    heads: usize,
    dim: usize,
    /// Pools a sparse row keeps.
    kept: usize,

    fn load(arena: Allocator, source: anytype, prefix: []const u8, c: *const Config, rank: usize, rows: cuda.Scratch.Rows) LoadError!Index {
        if (c.index_pool != index_pool or c.index_top < c.index_pool or c.index_top % c.index_pool != 0 or !c.index_tail) return error.UnsupportedShape;
        if (c.index_dim < 16 or c.index_dim > 128 or !std.math.isPowerOfTwo(c.index_dim) or c.index_heads == 0 or c.index_heads > 32) return error.UnsupportedShape;
        const wk_part = try storedPart(arena, source, c, prefix, "indexer.wk", rank);
        const gate_bytes = try take(arena, source, prefix, "indexer.index_kpool_compress_gate", rank);
        const weights_part = try storedPart(arena, source, c, prefix, "indexer.weights_proj", rank);
        const wq_part = try storedPart(arena, source, c, prefix, "indexer.wq_b", rank);
        const ln_w_bytes = try take(arena, source, prefix, "indexer.k_norm.weight", rank);
        const ln_b_bytes = try take(arena, source, prefix, "indexer.k_norm.bias", rank);
        const ape_bytes = try take(arena, source, prefix, "indexer.index_kpool_compress_ape", rank);
        const dim = c.index_dim;
        const heads = c.index_heads;
        if (!share.shaped(wk_part, dim, c.hidden) or gate_bytes.len != dim * c.hidden * 2 or !share.shaped(weights_part, heads, c.hidden)) return error.UnsupportedShape;
        if (!share.shaped(wq_part, heads * dim, c.q_rank) or ln_w_bytes.len != dim * 2 or ln_b_bytes.len != dim * 2 or ape_bytes.len != index_pool * dim * 2) return error.UnsupportedShape;
        var wk = try share.faceOf(arena, &.{wk_part}, c.hidden, rows);
        errdefer wk.deinit();
        var gate: bf16.Linear = try .init(gate_bytes, dim, c.hidden, rows);
        errdefer gate.deinit();
        var weights = try share.faceOf(arena, &.{weights_part}, c.hidden, rows);
        errdefer weights.deinit();
        var wq_b = try share.faceOf(arena, &.{wq_part}, c.q_rank, rows);
        errdefer wq_b.deinit();
        var ln_w: cuda.Buffer = try .init(.bf16, &.{dim});
        errdefer ln_w.deinit();
        try ln_w.upload(ln_w_bytes);
        var ln_b: cuda.Buffer = try .init(.bf16, &.{dim});
        errdefer ln_b.deinit();
        try ln_b.upload(ln_b_bytes);
        var ape: cuda.Buffer = try .init(.bf16, &.{ index_pool, dim });
        errdefer ape.deinit();
        try ape.upload(ape_bytes);
        return .{ .wk = wk, .gate = gate, .weights = weights, .wq_b = wq_b, .ln_w = ln_w, .ln_b = ln_b, .ape = ape, .heads = heads, .dim = dim, .kept = c.index_top / c.index_pool };
    }

    fn deinit(self: *Index) void {
        self.wk.deinit();
        self.gate.deinit();
        self.weights.deinit();
        self.wq_b.deinit();
        self.ln_w.deinit();
        self.ln_b.deinit();
        self.ape.deinit();
        self.* = undefined;
    }
};

/// The linear `prefix ++ module` as `rank` holds it, in its stored format.
fn storedPart(arena: Allocator, source: anytype, c: *const Config, prefix: []const u8, module: []const u8, rank: usize) LoadError!Part {
    return share.partOf(arena, source, c, try std.mem.concat(arena, u8, &.{ prefix, module }), rank);
}

fn take(arena: Allocator, source: anytype, prefix: []const u8, leaf: []const u8, rank: usize) LoadError![]const u8 {
    const name = try std.mem.concat(arena, u8, &.{ prefix, leaf });
    const tensor = try share.get(arena, source, name, rank) orelse {
        std.log.err("{s}: not in the checkpoint", .{name});
        return error.MissingTensor;
    };
    if (tensor.dtype != .BF16) return error.UnsupportedShape;
    return tensor.data;
}

/// Tokens a pool gathers. `index.cu` is written for four, and 512 is a multiple of it.
const index_pool = 4;

/// q bf16 [rows, heads key_dim] -> qa bf16 [rows, heads, 512]: kv_b's key rows (bf16 [heads, key_dim + value_dim,
/// 512]) taken into the queries, a product a head on the tensor cores. key_dim is a multiple of 16.
pub fn absorb(q: cuda.View, kv_b: cuda.View, key_dim: usize, qa: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_mla_absorb(&q, &kv_b, @intCast(key_dim), &qa));
}

/// qa bf16 [rows, heads, 512], row r at position pos + r, over `cache`'s latents into out; `parts` are the
/// chunks' partials (`Buffers`' po, pm, pl). With `list` (tokens int32 [rows, width], counts int32 [rows]) a row
/// whose count is positive attends those tokens alone, in list order; a count of 0 is a dense row. `table` maps
/// an extent of 512 positions onto `cache`.
pub fn attendRows(qa: cuda.View, cache: cuda.View, pos: cuda.View, table: ?*const cuda.View, parts: [3]cuda.View, out: cuda.View, list: ?[2]cuda.View, scale: f64, positions: usize) cuda.Error!void {
    const tokens: ?*const cuda.View = if (list) |*one| &one[0] else null;
    const counts: ?*const cuda.View = if (list) |*one| &one[1] else null;
    try cuda.launched(zs_glm5_mla_attend(&qa, &cache, &pos, table, &parts[0], &parts[1], &parts[2], &out, tokens, counts, scale, @intCast(positions)));
}

/// `expand` up to 16 rows asks for kv_b's rows before it loads them (`ZSPARK_GLM_MLA_ASK`, spec 073 WS-05): below
/// zero every row of a block before its first step, above it the step that many ahead, 0 the base kernel. The
/// same bits. Said before a step's graph is captured.
pub fn setAsk(steps: i32) void {
    zs_glm5_mla_ask(steps);
}

/// An FP8 cache's chunks with the queries and the softmax's weights in FP8 as vLLM's (`ZSPARK_GLM_MLA_Q8`, spec 114
/// WS-05, `mla_q8.cu`), or bf16 queries over the codes (`false`): not the same bits. Said before a graph is
/// captured.
pub fn setQ8(on: bool) void {
    zs_glm5_mla_q8(@intFromBool(on));
}

/// From `window_rows` rows (spec 116 WS-01) an attend launch runs a block a row, its chunks in order and each
/// folded into the row's output as the merge folds it, with no partials but the running output in po; fewer rows
/// (steps, drafts, verify windows) keep a block a chunk and the merge. The same bits. `Buffers` sizes po for it.
pub const window_rows: usize = 128;

/// For tests alone, to hold the window form against the chunk blocks (0 never); `window_rows` puts it back.
/// Said before a graph is captured.
pub fn setWindow(rows: usize) void {
    zs_glm5_mla_window(@intCast(@min(rows, std.math.maxInt(c_int))));
}

/// ol bf16 [rows, heads, 512] -> o bf16 [rows, heads value_dim]: kv_b's value rows applied to the attention.
pub fn expand(ol: cuda.View, kv_b: cuda.View, key_dim: usize, o: cuda.View) cuda.Error!void {
    return cuda.launched(zs_glm5_mla_expand(&ol, &kv_b, @intCast(key_dim), &o));
}

/// A sequence's latents: one row a position and DSA layer, the same on both ranks.
pub const Cache = struct {
    layers: usize,
    capacity: usize,
    /// Rows allocated a layer: `capacity`, or the next multiple of 512 once an indexer is held, so a token's
    /// extent is whole. A stream over a shared `Store` names the store's rows, which are every slot's.
    span: usize,
    /// Pooled-key rows a layer. The store's rows and these are the same count when the cache owns its keys;
    /// a shared store keeps the pooled keys contiguous for this stream alone.
    pooled_rows: usize = 0,
    /// bf16 [layers, span, 512], or FP8 records u8 [layers, span, 528] (`storage`). Freed by `deinit` when
    /// `owns` is set; a stream's view of a `Store` does not.
    rows: cuda.Buffer,
    storage: Storage = .bf16,
    owns: bool = true,
    /// The rows committed, and the same on the device (int32 [1]) where the kernels read it.
    pos: usize = 0,
    pos_dev: cuda.Buffer,
    /// Keys, gates and pooled keys' codes and scales, when the sequence's layers have an indexer.
    indexer: ?Indexer = null,

    pub fn init(layers: usize, capacity: usize) cuda.Error!Cache {
        return holding(layers, capacity, null);
    }

    /// `init`, and the indexer's rows when `dim` is set. The extent table is the identity: logical position
    /// `p` is physical row `p`, which is what a contiguous cache already is.
    pub fn holding(layers: usize, capacity: usize, dim: ?usize) cuda.Error!Cache {
        return holdingAs(.bf16, layers, capacity, dim);
    }

    /// `holding` with its latents in `storage`.
    pub fn holdingAs(storage: Storage, layers: usize, capacity: usize, dim: ?usize) cuda.Error!Cache {
        std.debug.assert(layers >= 1 and capacity >= 1);
        const span = if (dim != null) ((capacity + index_launch.extent - 1) / index_launch.extent) * index_launch.extent else capacity;
        var rows = try rowsOf(storage, layers, span);
        errdefer rows.deinit();
        var pos_dev: cuda.Buffer = try .init(.i32, &.{1});
        errdefer pos_dev.deinit();
        var self: Cache = .{ .layers = layers, .capacity = capacity, .span = span, .rows = rows, .storage = storage, .pos_dev = pos_dev };
        if (dim) |width| {
            self.indexer = try Indexer.init(layers, span, width);
            self.pooled_rows = span / index_pool;
        }
        errdefer if (self.indexer) |*indexer| indexer.deinit();
        try self.setPos(0);
        return self;
    }

    /// One rank's bytes for `layers` caches of `capacity` positions. `dim` null is the latent alone.
    pub fn bytes(layers: usize, capacity: usize, dim: ?usize) usize {
        return bytesAs(.bf16, layers, capacity, dim);
    }

    /// `bytes` with the latents in `storage`.
    pub fn bytesAs(storage: Storage, layers: usize, capacity: usize, dim: ?usize) usize {
        const span = if (dim != null) ((capacity + index_launch.extent - 1) / index_launch.extent) * index_launch.extent else capacity;
        var n = layers * span * storage.width() + @sizeOf(i32);
        if (dim) |width| n += Indexer.bytes(layers, span, width);
        return n;
    }

    pub fn deinit(self: *Cache) void {
        if (self.owns) self.rows.deinit();
        self.pos_dev.deinit();
        if (self.indexer) |*indexer| indexer.deinit();
        self.* = undefined;
    }

    /// The sequence holds `pos` positions: rows past them are not read.
    pub fn setPos(self: *Cache, pos: usize) cuda.Error!void {
        std.debug.assert(pos <= self.capacity);
        self.pos = pos;
        const value: i32 = @intCast(pos);
        try self.pos_dev.uploadLaunched(0, std.mem.asBytes(&value));
    }

    /// The sequence ends at `pos`. The pool that holds that token is unfinished: its pooled key (codes and
    /// scale) is zero and is written again when its fourth token arrives. Rows below `pos` stay.
    pub fn cut(self: *Cache, pos: usize) cuda.Error!void {
        try self.setPos(pos);
        const indexer = &(self.indexer orelse return);
        const pools = self.pooled_rows;
        const from = pos / index_pool;
        if (from >= pools) return;
        for (0..self.layers) |index| {
            const codes = indexer.codesOf(index, pools);
            const scales = indexer.scalesOf(index, pools);
            try cuda.zeroLaunched(.init(@ptrCast(@as([*]u8, @ptrCast(codes.ptr.?)) + from * indexer.dim), .u8, &.{(pools - from) * indexer.dim}));
            try cuda.zeroLaunched(.init(@ptrCast(@as([*]u8, @ptrCast(scales.ptr.?)) + from * @sizeOf(f32)), .f32, &.{pools - from}));
        }
    }

    /// Keeps the first `keep` rows of the last window: the rest are written over by the next.
    pub fn commit(self: *Cache, keep: usize) cuda.Error!void {
        return self.setPos(self.pos + keep);
    }

    /// The extent table, when the cache has one: int32 [extents], logical extent to the store's.
    pub fn extents(self: Cache) ?cuda.View {
        const indexer = self.indexer orelse return null;
        return indexer.table.view;
    }

    /// The extent table becomes `ids`. Each id is an extent of this cache, and the length is the table's.
    pub fn retarget(self: *Cache, ids: []const i32) cuda.Error!void {
        const indexer = &(self.indexer orelse unreachable);
        const places = self.span / index_launch.extent;
        std.debug.assert(ids.len == indexer.table.view.count());
        for (ids) |id| std.debug.assert(id >= 0 and @as(usize, @intCast(id)) < places);
        try indexer.table.upload(std.mem.sliceAsBytes(ids));
    }

    /// DSA layer `index`'s rows, bf16 [span, 512], or u8 [span, 528] of FP8 records.
    pub fn layer(self: Cache, index: usize) cuda.View {
        std.debug.assert(index < self.layers);
        const base: [*]u8 = @ptrCast(self.rows.view.ptr.?);
        const at: *anyopaque = @ptrCast(base + index * self.span * self.storage.width());
        return switch (self.storage) {
            .bf16 => .init(at, .bf16, &.{ self.span, latent }),
            .fp8 => .init(at, .u8, &.{ self.span, fp8_record }),
        };
    }

    /// `self` begins as `source` stood at `n` positions: the latents, keys and gates under each cache's
    /// own extents (a stream's table maps a logical extent onto the shared store's), and the first
    /// `n / index_pool` pooled keys, which a stream keeps contiguous (`Store.stream`). `n` is a position
    /// `source` reached, so what the copies read is committed. The unfinished pool at `n` is cut.
    pub fn copyPrefix(self: *Cache, source: *const Cache, n: usize) cuda.Error!void {
        std.debug.assert(self.layers == source.layers and n <= source.pos and n <= self.capacity);
        const indexer = &(self.indexer orelse {
            for (0..self.layers) |index| {
                const dst: cuda.Buffer = .{ .view = self.layer(index) };
                const src: cuda.Buffer = .{ .view = source.layer(index) };
                try dst.copyRows(0, src, 0, n);
            }
            try self.cut(n);
            return;
        });
        const from = &(source.indexer orelse unreachable);
        var src_table: [64]i32 = undefined;
        var dst_table: [64]i32 = undefined;
        const table_len = indexer.table.view.count();
        std.debug.assert(table_len <= src_table.len and from.table.view.count() == table_len);
        try from.table.download(std.mem.sliceAsBytes(src_table[0..table_len]));
        try indexer.table.download(std.mem.sliceAsBytes(dst_table[0..table_len]));
        const want = (n + index_launch.extent - 1) / index_launch.extent;
        for (0..want) |extent| {
            const s = @as(usize, @intCast(src_table[extent])) * index_launch.extent;
            const d = @as(usize, @intCast(dst_table[extent])) * index_launch.extent;
            const rows = @min(index_launch.extent, n - extent * index_launch.extent);
            for (0..self.layers) |index| {
                const dst: cuda.Buffer = .{ .view = self.layer(index) };
                const src: cuda.Buffer = .{ .view = source.layer(index) };
                try dst.copyRows(d, src, s, rows);
                const d_keys: cuda.Buffer = .{ .view = indexer.plane(indexer.keys, index, self.span) };
                const s_keys: cuda.Buffer = .{ .view = from.plane(from.keys, index, source.span) };
                const d_gates: cuda.Buffer = .{ .view = indexer.plane(indexer.gates, index, self.span) };
                const s_gates: cuda.Buffer = .{ .view = from.plane(from.gates, index, source.span) };
                try d_keys.copyRows(d, s_keys, s, rows);
                try d_gates.copyRows(d, s_gates, s, rows);
            }
        }
        // the pooled keys' `n / index_pool` first pools, every layer's: a stream's own pools are its
        // sequence's in order (`Store.stream`), the extent table is the latents' alone
        const pools = n / index_pool;
        try indexer.codes.copyAt(0, from.codes, pools * indexer.dim * self.layers);
        try indexer.scales.copyAt(0, from.scales, pools * @sizeOf(f32) * self.layers);
        try self.cut(n);
    }
};

fn rowsOf(storage: Storage, layers: usize, span: usize) cuda.Error!cuda.Buffer {
    return switch (storage) {
        .bf16 => .init(.bf16, &.{ layers, span, latent }),
        .fp8 => .init(.u8, &.{ layers, span, fp8_record }),
    };
}

/// One rank's latents, keys and gates for every extent of a pool (`decode/extents.zig`). A stream's `Cache`
/// is a view of it plus that stream's table and its own pooled keys, which stay contiguous in pool order.
pub const Store = struct {
    layers: usize,
    span: usize,
    dim: usize,
    rows: cuda.Buffer,
    storage: Storage,
    keys: cuda.Buffer,
    gates: cuda.Buffer,

    /// `places` extents, the pool's, shared by the streams whose tables name them, the latents in `storage`.
    pub fn init(storage: Storage, layers: usize, places: usize, dim: usize) cuda.Error!Store {
        std.debug.assert(layers >= 1 and places >= 1 and dim >= 1);
        const span = places * index_launch.extent;
        var rows = try rowsOf(storage, layers, span);
        errdefer rows.deinit();
        var keys: cuda.Buffer = try .init(.bf16, &.{ layers, span, dim });
        errdefer keys.deinit();
        var gates: cuda.Buffer = try .init(.bf16, &.{ layers, span, dim });
        errdefer gates.deinit();
        return .{ .layers = layers, .span = span, .dim = dim, .rows = rows, .storage = storage, .keys = keys, .gates = gates };
    }

    pub fn deinit(self: *Store) void {
        self.rows.deinit();
        self.keys.deinit();
        self.gates.deinit();
        self.* = undefined;
    }

    /// A stream of `capacity` positions whose logical extents are `ids`, each an extent of this store.
    pub fn stream(self: *const Store, capacity: usize, ids: []const i32) cuda.Error!Cache {
        std.debug.assert(capacity >= 1 and ids.len >= 1);
        const places = self.span / index_launch.extent;
        for (ids) |id| std.debug.assert(id >= 0 and @as(usize, @intCast(id)) < places);
        const pooled_rows = ids.len * (index_launch.extent / index_pool);
        var pos_dev: cuda.Buffer = try .init(.i32, &.{1});
        errdefer pos_dev.deinit();
        var codes: cuda.Buffer = try .init(.u8, &.{ self.layers, pooled_rows, self.dim });
        errdefer codes.deinit();
        var scales: cuda.Buffer = try .init(.f32, &.{ self.layers, pooled_rows });
        errdefer scales.deinit();
        var table: cuda.Buffer = try .init(.i32, &.{ids.len});
        errdefer table.deinit();
        try table.upload(std.mem.sliceAsBytes(ids));
        var cache: Cache = .{
            .layers = self.layers,
            .capacity = capacity,
            .span = self.span,
            .pooled_rows = pooled_rows,
            .rows = self.rows,
            .storage = self.storage,
            .owns = false,
            .pos_dev = pos_dev,
            .indexer = .{ .dim = self.dim, .keys = self.keys, .gates = self.gates, .owns_keys = false, .codes = codes, .scales = scales, .table = table },
        };
        try cache.setPos(0);
        return cache;
    }
};

/// One sequence's index rows: keys and gates bf16 [layers, span, dim] through an extent table, pooled
/// keys contiguous in pool order as vLLM stores them (`index.cu`): e4m3 codes u8 [layers, pools, dim] and a
/// scale fp32 [layers, pools] a pool, `dim + 4` bytes.
const Indexer = struct {
    dim: usize,
    keys: cuda.Buffer,
    gates: cuda.Buffer,
    owns_keys: bool = true,
    codes: cuda.Buffer,
    scales: cuda.Buffer,
    /// int32 [extents], identity.
    table: cuda.Buffer,

    fn init(layers: usize, span: usize, dim: usize) cuda.Error!Indexer {
        std.debug.assert(span % index_launch.extent == 0 and dim >= 1);
        var keys: cuda.Buffer = try .init(.bf16, &.{ layers, span, dim });
        errdefer keys.deinit();
        var gates: cuda.Buffer = try .init(.bf16, &.{ layers, span, dim });
        errdefer gates.deinit();
        var codes: cuda.Buffer = try .init(.u8, &.{ layers, span / index_pool, dim });
        errdefer codes.deinit();
        var scales: cuda.Buffer = try .init(.f32, &.{ layers, span / index_pool });
        errdefer scales.deinit();
        const extents = span / index_launch.extent;
        var table: cuda.Buffer = try .init(.i32, &.{extents});
        errdefer table.deinit();
        var at: usize = 0;
        var ids: [64]i32 = undefined;
        while (at < extents) {
            const n = @min(ids.len, extents - at);
            for (ids[0..n], 0..) |*one, i| one.* = @intCast(at + i);
            try table.uploadAt(at * @sizeOf(i32), std.mem.sliceAsBytes(ids[0..n]));
            at += n;
        }
        return .{ .dim = dim, .keys = keys, .gates = gates, .codes = codes, .scales = scales, .table = table };
    }

    fn bytes(layers: usize, span: usize, dim: usize) usize {
        const extents = span / index_launch.extent;
        return layers * span * dim * 2 * 2 + layers * (span / index_pool) * (dim + @sizeOf(f32)) + extents * @sizeOf(i32);
    }

    fn deinit(self: *Indexer) void {
        if (self.owns_keys) {
            self.keys.deinit();
            self.gates.deinit();
        }
        self.codes.deinit();
        self.scales.deinit();
        self.table.deinit();
        self.* = undefined;
    }

    /// Layer `index`'s plane, bf16 [rows, dim].
    fn plane(self: Indexer, buffer: cuda.Buffer, index: usize, rows: usize) cuda.View {
        const base: [*]u8 = @ptrCast(buffer.view.ptr.?);
        return .init(@ptrCast(base + index * rows * self.dim * 2), .bf16, &.{ rows, self.dim });
    }

    /// Layer `index`'s pooled keys' codes, u8 [pools, dim], and their scales, fp32 [pools].
    fn codesOf(self: Indexer, index: usize, pools: usize) cuda.View {
        const base: [*]u8 = @ptrCast(self.codes.view.ptr.?);
        return .init(@ptrCast(base + index * pools * self.dim), .u8, &.{ pools, self.dim });
    }

    fn scalesOf(self: Indexer, index: usize, pools: usize) cuda.View {
        const base: [*]u8 = @ptrCast(self.scales.view.ptr.?);
        return .init(@ptrCast(base + index * pools * @sizeOf(f32)), .f32, &.{pools});
    }
};

/// A context of `need` device bytes fits `free`, with `cuda.reserve` kept.
pub fn admits(free: usize, need: usize) bool {
    return need + cuda.reserve <= free;
}

/// What a forward of up to `rows` rows over up to `chunks` chunks of keys writes on its way, a layer at a time.
pub const Buffers = struct {
    rows: usize,
    heads: usize,
    chunks: usize,
    /// bf16 [rows, q_rank + 512], [rows, q_rank], [rows, 512], [rows, heads key_dim]
    low: cuda.Buffer,
    q_normed: cuda.Buffer,
    lat: cuda.Buffer,
    q: cuda.Buffer,
    /// bf16 [rows, heads, 512]: the queries over the latent, and the attention's result
    qa: cuda.Buffer,
    ol: cuda.Buffer,
    /// bf16 [rows, heads value_dim]
    o: cuda.Buffer,
    /// fp32 [parted, chunks, heads, 512] and [parted, chunks, heads] twice (`parted`): the chunks' partials of a
    /// launch under `window_rows` rows; po is a window's running output, [rows, heads, 512], from there on
    po: cuda.Buffer,
    pm: cuda.Buffer,
    pl: cuda.Buffer,
    /// The indexer's scratch, when the layer has one. Scores cover every pool of the capacity, so a graph's
    /// bucket and an eager window launch the same grid.
    indexer: ?IndexScratch = null,

    /// The rows `po`, `pm` and `pl` hold of `chunks` chunks: a launch's under `window_rows` rows, and at least
    /// `rows` rows of one chunk, a window's running output.
    pub fn parted(rows: usize, chunks: usize) usize {
        return @max(@min(rows, window_rows - 1), (rows + chunks - 1) / chunks);
    }

    /// The bytes of `po`, `pm` and `pl` for a forward of `rows` rows of at most `keys` keys each
    /// (`Weights.attended`).
    pub fn scratchBytes(heads: usize, rows: usize, keys: usize) usize {
        const chunks = (keys + chunk - 1) / chunk;
        const held = parted(rows, chunks);
        return held * chunks * heads * latent * @sizeOf(f32) + 2 * held * chunks * heads * @sizeOf(f32);
    }

    /// For `weights`' shapes and a cache of `capacity` rows.
    pub fn init(weights: Weights, capacity: usize, rows: usize) cuda.Error!Buffers {
        const heads = weights.heads;
        const chunks = (weights.attended(capacity) + chunk - 1) / chunk;
        var self: Buffers = .{ .rows = rows, .heads = heads, .chunks = chunks, .low = undefined, .q_normed = undefined, .lat = undefined, .q = undefined, .qa = undefined, .ol = undefined, .o = undefined, .po = undefined, .pm = undefined, .pl = undefined, .indexer = null };
        self.low = try .init(.bf16, &.{ rows, weights.q_rank + latent });
        errdefer self.low.deinit();
        self.q_normed = try .init(.bf16, &.{ rows, weights.q_rank });
        errdefer self.q_normed.deinit();
        self.lat = try .init(.bf16, &.{ rows, latent });
        errdefer self.lat.deinit();
        self.q = try .init(.bf16, &.{ rows, heads * weights.key_dim });
        errdefer self.q.deinit();
        self.qa = try .init(.bf16, &.{ rows, heads, latent });
        errdefer self.qa.deinit();
        self.ol = try .init(.bf16, &.{ rows, heads, latent });
        errdefer self.ol.deinit();
        self.o = try .init(.bf16, &.{ rows, heads * weights.value_dim });
        errdefer self.o.deinit();
        const held = parted(rows, chunks);
        self.po = try .init(.f32, &.{ held, chunks, heads, latent });
        errdefer self.po.deinit();
        self.pm = try .init(.f32, &.{ held, chunks, heads });
        errdefer self.pm.deinit();
        self.pl = try .init(.f32, &.{ held, chunks, heads });
        errdefer self.pl.deinit();
        if (weights.index) |index| self.indexer = try .init(index, capacity, rows);
        return self;
    }

    pub fn deinit(self: *Buffers) void {
        inline for (.{ "low", "q_normed", "lat", "q", "qa", "ol", "o", "po", "pm", "pl" }) |name| @field(self, name).deinit();
        if (self.indexer) |*indexer| indexer.deinit();
        self.* = undefined;
    }
};

/// What the indexer's forward writes on its way, a layer at a time.
pub const IndexScratch = struct {
    raw: cuda.Buffer,
    /// fp32 [rows, dim]
    gate: cuda.Buffer,
    /// The head weights, fp32 [rows, heads]
    weights: cuda.Buffer,
    /// bf16 [rows, heads, dim]
    query: cuda.Buffer,
    /// The queries coded, u8 [rows, heads, dim], and the weights with their scales, fp32 [rows, heads]
    q8: cuda.Buffer,
    wq: cuda.Buffer,
    /// fp32 [rows, pools]
    scores: cuda.Buffer,
    tokens: cuda.Buffer,
    counts: cuda.Buffer,
    /// int32 [1]: the window's position plus a split block's first row, for the score and select launches
    /// (`index.split_from`). Never written where no rank splits.
    pos_lo: cuda.Buffer,

    fn init(index: Index, capacity: usize, rows: usize) cuda.Error!IndexScratch {
        const pools = @max(capacity / index_pool, 1);
        const width = index.kept * index_pool + index_pool - 1;
        var self: IndexScratch = undefined;
        self.raw = try .init(.bf16, &.{ rows, index.dim });
        errdefer self.raw.deinit();
        self.gate = try .init(.f32, &.{ rows, index.dim });
        errdefer self.gate.deinit();
        self.weights = try .init(.f32, &.{ rows, index.heads });
        errdefer self.weights.deinit();
        self.query = try .init(.bf16, &.{ rows, index.heads, index.dim });
        errdefer self.query.deinit();
        self.q8 = try .init(.u8, &.{ rows, index.heads, index.dim });
        errdefer self.q8.deinit();
        self.wq = try .init(.f32, &.{ rows, index.heads });
        errdefer self.wq.deinit();
        self.scores = try .init(.f32, &.{ rows, pools });
        errdefer self.scores.deinit();
        self.tokens = try .init(.i32, &.{ rows, width });
        errdefer self.tokens.deinit();
        self.counts = try .init(.i32, &.{rows});
        errdefer self.counts.deinit();
        self.pos_lo = try .init(.i32, &.{1});
        return self;
    }

    fn deinit(self: *IndexScratch) void {
        inline for (.{ "raw", "gate", "weights", "query", "q8", "wq", "scores", "tokens", "counts", "pos_lo" }) |name| @field(self, name).deinit();
        self.* = undefined;
    }
};

extern fn zs_glm5_mla_absorb(q: *const cuda.View, kv_b: *const cuda.View, dk: i64, out: *const cuda.View) c_int;
extern fn zs_glm5_mla_ask(steps: c_int) void;
extern fn zs_glm5_mla_q8(on: c_int) void;
extern fn zs_glm5_mla_window(rows: c_int) void;
extern fn zs_glm5_mla_expand(ol: *const cuda.View, kv_b: *const cuda.View, dk: i64, out: *const cuda.View) c_int;
extern fn zs_glm5_mla_write(lat: *const cuda.View, cache: *const cuda.View, pos: *const cuda.View, table: ?*const cuda.View, context: i64) c_int;
extern fn zs_glm5_mla_attend(qa: *const cuda.View, cache: *const cuda.View, pos: *const cuda.View, table: ?*const cuda.View, po: *const cuda.View, pm: *const cuda.View, pl: *const cuda.View, out: *const cuda.View, tokens: ?*const cuda.View, counts: ?*const cuda.View, scale: f64, context: i64) c_int;
