//! Fixtures two or more of `zig/src/models/qwen4/`'s test files share, a struct a fixture with its imports inside.
//! `attending`: an attention block of random weights, its shapes, a run of windows over caches in order or
//! behind a table, and the host's readers, for `attention_test.zig`, `attention_selection_test.zig` and
//! `attention_table_test.zig`; it is `testing_attending.zig`.
//! `random_model`: the small model of DeltaNet layers on random weights, with a head and a drafter where asked,
//! which every other model test loads, and the two settings a test sets around the models it makes
//! (`caches`, `head_caches`).
//! `tiny_model`: one DeltaNet layer with a tiny PLE and one attention layer, for `rounds_test.zig`.
//! `sessions`: a sink that counts its pieces and the deepest window, for the four `session_*test.zig` files.
//! `checkpoints`: the opener of the Flash Next checkpoint (`ZSPARK_FLASH_NEXT_FP4_LOCAL`) every checkpoint test
//! begins with, and the count of layers up to its first attention layer.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");

pub const attending = @import("testing_attending.zig");

pub const random_model = struct {
    /// How the head's attention of `testModelRows` keeps its caches: a test sets it around the models it makes.
    pub var caches: attention.Storage = .bf16;

    /// How that head holds its own cache where it is not `caches` (`model.Model.Attention.head`).
    pub var head_caches: ?attention.Storage = null;

    const forward_file = @import("../../../src/models/qwen4/forward.zig");
    const experts = @import("../../../src/weights/experts.zig");
    const face = @import("../../../src/weights/face.zig");
    const numeric = @import("../../../src/weights/numeric.zig");
    const nvfp4 = @import("../../../src/weights/nvfp4.zig");
    const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
    const gdn = @import("../../../src/models/qwen4/blocks/gdn.zig");
    const model = @import("../../../src/models/qwen4/model.zig");
    const moe = @import("../../../src/models/qwen4/blocks/moe.zig");

    pub const fx = struct {
        pub const shape: model.Shape = .{ .hidden = 256, .streams = 4, .low = 64, .eps = 1e-6, .vocab = 320, .top = 2, .routed = 8, .width = 64 };
        pub const layers = 3;
        const max_rows = 4;
    };

    pub fn randomBf16(arena: Allocator, random: std.Random, count: usize, spread: f32) ![]const u8 {
        const bits = try arena.alloc(u16, count);
        for (bits) |*b| b.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
        return std.mem.sliceAsBytes(bits);
    }

    pub fn randomFloats(arena: Allocator, random: std.Random, count: usize, base: f32, spread: f32) ![]f32 {
        const out = try arena.alloc(f32, count);
        for (out) |*v| v.* = base + random.floatNorm(f32) * spread;
        return out;
    }

    fn randomConnection(arena: Allocator, random: std.Random, inject: bool, rows: cuda.Scratch.Rows) !model.HyperConnection {
        const s = fx.shape;
        const wide = s.wide();
        const scale = try randomFloats(arena, random, wide, 1.0, 0.1);
        return .init(
            try randomBf16(arena, random, s.low * wide, 0.05),
            if (inject) try randomBf16(arena, random, s.streams * wide, 0.05) else null,
            try randomBf16(arena, random, wide * s.low, 0.2),
            scale,
            s.low,
            rows,
        );
    }

    pub fn randomMatrix(arena: Allocator, random: std.Random, n: usize, k: usize) !experts.Matrix {
        const count = fx.shape.routed;
        const codes = try arena.alloc(u8, count * n * k / 2);
        random.bytes(codes);
        const block_scales = try arena.alloc(u8, count * n * k / 16);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        return .{
            .codes = try experts.Matrix.parts(arena, codes, count),
            .block_scales = try experts.Matrix.parts(arena, block_scales, count),
            .scales = try randomFloats(arena, random, count, 0.02, 0.0),
        };
    }

    fn randomMoe(gpa: Allocator, arena: Allocator, random: std.Random, staging: *experts.Staging, rows: cuda.Scratch.Rows) !moe.Weights {
        const s = fx.shape;
        return .{
            .router = try .init(try randomBf16(arena, random, (s.routed + 1) * s.hidden, 0.3), s.routed + 1, s.hidden, rows),
            .stored = .{ .nvfp4 = .{
                .routed = .{ .nvfp4 = try .init(gpa, staging, s.routed, s.width, s.hidden, try randomMatrix(arena, random, s.width, s.hidden), try randomMatrix(arena, random, s.width, s.hidden), try randomMatrix(arena, random, s.hidden, s.width)) },
                .shared_gate_up = try .init(try randomBf16(arena, random, 2 * s.width * s.hidden, 0.06), 2 * s.width, s.hidden, rows),
                .shared_down = try .init(try randomBf16(arena, random, s.hidden * s.width, 0.1), s.hidden, s.width, rows),
            } },
        };
    }

    fn deviceFloats(values: []const f32) !cuda.Buffer {
        var buffer: cuda.Buffer = try .init(.f32, &.{values.len});
        errdefer buffer.deinit();
        try buffer.upload(std.mem.sliceAsBytes(values));
        return buffer;
    }

    /// `testModel` with an MTP head on random weights, for tests in mtp.zig: its attention layer selects blocks
    /// from position 20 on.
    pub fn testModelWithHead(gpa: Allocator, arena: Allocator) !model.Model {
        return testModelDrafting(gpa, arena, null);
    }

    /// `testModelWithHead`, its drafts scored by a draft head over `draft_vocab` (ascending) where that is given.
    pub fn testModelDrafting(gpa: Allocator, arena: Allocator, draft_vocab: ?[]const u32) !model.Model {
        return testModelRows(gpa, arena, draft_vocab, fx.max_rows);
    }

    /// `testModelDrafting` whose windows take up to `rows` rows: a prompt's are wider than a reply's.
    pub fn testModelRows(gpa: Allocator, arena: Allocator, draft_vocab: ?[]const u32, max_rows: usize) !model.Model {
        var head_bytes: []const u8 = undefined;
        var m = try testModelKeeping(gpa, arena, &head_bytes, max_rows);
        errdefer m.deinit();
        const rows = m.scratch.rows(max_rows);
        const s = fx.shape;
        var prng: std.Random.DefaultPrng = .init(43);
        const random = prng.random();
        const shape: attention.Shape = .{ .dims = s.hidden, .heads = 6, .kv_heads = 2, .head_dim = 32, .index_heads = 2, .index_dim = 64, .rotary_dim = 16, .rope_theta = 1e7, .budget = 16, .ratio = 4, .eps = s.eps, .kv = caches };
        const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(s.hidden)));
        var staging: experts.Staging = .{};
        defer staging.deinit(gpa);
        // a failed init leaks what it made so far: the test is about to fail on that error
        m.mtp = .{
            .norm_embedding = try deviceFloats(try randomFloats(arena, random, s.hidden, 1.0, 0.1)),
            .norm_hidden = try deviceFloats(try randomFloats(arena, random, s.wide(), 1.0, 0.1)),
            .fc_embedding = try .init(try randomBf16(arena, random, s.hidden * s.hidden, spread), s.hidden, s.hidden, rows),
            .fc_hidden = try .init(try randomBf16(arena, random, s.hidden * s.hidden, spread), s.hidden, s.hidden, rows.times(s.streams)),
            .layer = .{
                .attn_hc = try randomConnection(arena, random, true, rows),
                .mlp_hc = try randomConnection(arena, random, true, rows),
                .block = .{ .attention = try .init(shape, .{
                    .q = try randomBf16(arena, random, shape.heads * 2 * shape.head_dim * s.hidden, spread),
                    .k = try randomBf16(arena, random, shape.kv_heads * shape.head_dim * s.hidden, spread),
                    .v = try randomBf16(arena, random, shape.kv_heads * shape.head_dim * s.hidden, spread),
                    .index_qk = try randomBf16(arena, random, (shape.index_heads + 1) * shape.index_dim * s.hidden, spread),
                    .o = try randomBf16(arena, random, s.hidden * shape.out(), 0.1),
                    .q_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
                    .k_norm = try randomFloats(arena, random, shape.head_dim, 1.0, 0.1),
                    .index_q_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
                    .index_k_norm = try randomFloats(arena, random, shape.index_dim, 1.0, 0.1),
                }, rows) },
                .moe = try randomMoe(gpa, arena, random, &staging, rows),
            },
            .mixer = try randomConnection(arena, random, false, rows),
            .draft = if (draft_vocab) |ids| try .init(gpa, .{ .bf16 = head_bytes }, s.vocab, s.hidden, ids) else null,
            .max_rows = max_rows,
        };
        m.attention = .{ .shape = shape, .rotary = try .init(shape), .head = head_caches orelse caches };
        return m;
    }

    /// A small model of DeltaNet layers on random weights, for tests here and in draft.zig.
    pub fn testModel(gpa: Allocator, arena: Allocator) !model.Model {
        var head_bytes: []const u8 = undefined;
        return testModelKeeping(gpa, arena, &head_bytes, fx.max_rows);
    }

    /// `testModel` whose windows run `max_rows` rows: a pool's rounds, with no head.
    pub fn testModelOfRows(gpa: Allocator, arena: Allocator, max_rows: usize) !model.Model {
        var head_bytes: []const u8 = undefined;
        return testModelKeeping(gpa, arena, &head_bytes, max_rows);
    }

    /// `testModel`; `head_bytes` is left the head's bf16 weight [vocab, hidden], in `arena`.
    fn testModelKeeping(gpa: Allocator, arena: Allocator, head_bytes: *[]const u8, max_rows: usize) !model.Model {
        const s = fx.shape;
        const dn = s.delta_net;
        // a failed init leaks what it made so far, the pool included: the test is about to fail on that error
        const pool: *cuda.Scratch = try .create(gpa);
        const rows = pool.rows(max_rows);
        var prng: std.Random.DefaultPrng = .init(42);
        const random = prng.random();
        var embed: cuda.Buffer = try .init(.bf16, &.{ s.vocab, s.hidden });
        try embed.upload(try randomBf16(arena, random, s.vocab * s.hidden, 1.0));
        const layers = try gpa.alloc(model.Layer, fx.layers);
        for (layers) |*layer| {
            var staging: experts.Staging = .{};
            defer staging.deinit(gpa);
            layer.* = .{
                .attn_hc = try randomConnection(arena, random, true, rows),
                .mlp_hc = try randomConnection(arena, random, true, rows),
                .block = .{ .delta_net = try .init(.{
                    .projection = &.{
                        try randomBf16(arena, random, dn.conv() * s.hidden, 0.06),
                        try randomBf16(arena, random, dn.out() * s.hidden, 0.06),
                        try randomBf16(arena, random, dn.value_heads * s.hidden, 0.06),
                        try randomBf16(arena, random, dn.value_heads * s.hidden, 0.06),
                    },
                    .conv = try randomBf16(arena, random, dn.conv() * gdn.taps, 0.5),
                    .a_log = try randomFloats(arena, random, dn.value_heads, -1.0, 0.5),
                    .dt_bias = try randomFloats(arena, random, dn.value_heads, 0.0, 1.0),
                    .norm = try randomBf16(arena, random, gdn.value_dim, 1.0),
                    .out = try randomBf16(arena, random, s.hidden * dn.out(), 0.02),
                }, s, rows) },
                .moe = try randomMoe(gpa, arena, random, &staging, rows),
            };
        }
        head_bytes.* = try randomBf16(arena, random, s.vocab * s.hidden, 0.1);
        return .{
            .shape = s,
            .embed = embed,
            .layers = layers,
            .mixer = try randomConnection(arena, random, false, rows),
            .head = try .init(head_bytes.*, s.vocab, s.hidden, rows),
            .max_rows = max_rows,
            .scratch = pool,
            .gpa = gpa,
        };
    }
};

pub const tiny_model = struct {
    const experts = @import("../../../src/weights/experts.zig");
    const numeric = @import("../../../src/weights/numeric.zig");
    const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
    const gdn = @import("../../../src/models/qwen4/blocks/gdn.zig");
    const model = @import("../../../src/models/qwen4/model.zig");
    const moe = @import("../../../src/models/qwen4/blocks/moe.zig");
    const ple = @import("../../../src/models/qwen4/blocks/ple.zig");

    pub const tiny = struct {
        pub const shape: model.Shape = .{ .hidden = 256, .streams = 4, .low = 64, .eps = 1e-6, .vocab = 320, .top = 2, .routed = 8, .width = 64 };
        pub const max_rows = 40;
        pub const capacity = 3072;
        const ple_shape: ple.Shape = .{ .hidden = 256, .streams = 4, .heads = 2, .dims = 32, .taps = 2, .dilation = 1, .ngram = 2 };
        pub const attn: attention.Shape = .{ .dims = 256, .heads = 4, .kv_heads = 2, .head_dim = 32, .index_heads = 2, .index_dim = 64, .rotary_dim = 16, .rope_theta = 1e7, .budget = 2048, .ratio = 4, .eps = 1e-6 };
    };

    fn randomBf16(arena: Allocator, random: std.Random, count: usize, spread: f32) ![]const u8 {
        const bits = try arena.alloc(u16, count);
        for (bits) |*entry| entry.* = numeric.bf16FromF32(random.floatNorm(f32) * spread);
        return std.mem.sliceAsBytes(bits);
    }

    fn randomFloats(arena: Allocator, random: std.Random, count: usize, base: f32, spread: f32) ![]f32 {
        const out = try arena.alloc(f32, count);
        for (out) |*value| value.* = base + random.floatNorm(f32) * spread;
        return out;
    }

    fn randomConnection(arena: Allocator, random: std.Random, inject: bool, rows: cuda.Scratch.Rows) !model.HyperConnection {
        const s = tiny.shape;
        const wide = s.wide();
        return .init(
            try randomBf16(arena, random, s.low * wide, 0.05),
            if (inject) try randomBf16(arena, random, s.streams * wide, 0.05) else null,
            try randomBf16(arena, random, wide * s.low, 0.2),
            try randomFloats(arena, random, wide, 1.0, 0.1),
            s.low,
            rows,
        );
    }

    fn randomMatrix(arena: Allocator, random: std.Random, n: usize, k: usize) !experts.Matrix {
        const count = tiny.shape.routed;
        const codes = try arena.alloc(u8, count * n * k / 2);
        random.bytes(codes);
        const block_scales = try arena.alloc(u8, count * n * k / 16);
        for (block_scales) |*byte| byte.* = 0x28 + random.uintLessThan(u8, 0x18);
        return .{
            .codes = try experts.Matrix.parts(arena, codes, count),
            .block_scales = try experts.Matrix.parts(arena, block_scales, count),
            .scales = try randomFloats(arena, random, count, 0.02, 0.0),
        };
    }

    fn randomMoe(gpa: Allocator, arena: Allocator, random: std.Random, staging: *experts.Staging, rows: cuda.Scratch.Rows) !moe.Weights {
        const s = tiny.shape;
        return .{
            .router = try .init(try randomBf16(arena, random, (s.routed + 1) * s.hidden, 0.3), s.routed + 1, s.hidden, rows),
            .stored = .{ .nvfp4 = .{
                .routed = .{ .nvfp4 = try .init(gpa, staging, s.routed, s.width, s.hidden, try randomMatrix(arena, random, s.width, s.hidden), try randomMatrix(arena, random, s.width, s.hidden), try randomMatrix(arena, random, s.hidden, s.width)) },
                .shared_gate_up = try .init(try randomBf16(arena, random, 2 * s.width * s.hidden, 0.06), 2 * s.width, s.hidden, rows),
                .shared_down = try .init(try randomBf16(arena, random, s.hidden * s.width, 0.1), s.hidden, s.width, rows),
            } },
        };
    }

    /// One DeltaNet layer carrying a PLE, then one attention layer. The table's bytes live in `arena`.
    pub fn tinyModel(gpa: Allocator, arena: Allocator) !model.Model {
        const s = tiny.shape;
        const dn = s.delta_net;
        const pool: *cuda.Scratch = try .create(gpa);
        const rows = pool.rows(tiny.max_rows);
        var prng: std.Random.DefaultPrng = .init(7);
        const random = prng.random();
        var embed: cuda.Buffer = try .init(.bf16, &.{ s.vocab, s.hidden });
        try embed.upload(try randomBf16(arena, random, s.vocab * s.hidden, 1.0));
        const layers = try gpa.alloc(model.Layer, 2);
        var staging: experts.Staging = .{};
        defer staging.deinit(gpa);
        const spread = 1.0 / @sqrt(@as(f32, @floatFromInt(s.hidden)));
        const ple_shape = tiny.ple_shape;
        const table_rows: usize = 12;
        const values = try arena.alloc(u16, table_rows * ple_shape.dims);
        for (values, 0..) |*entry, i| entry.* = numeric.bf16FromF32(@as(f32, @floatFromInt(i)) * 0.02 - 0.1);
        const scale: f32 = 1;
        const table: ple.Table = try .init(gpa, .{
            .scheme = .bf16,
            .shards = &.{.{ .weight = .{ .dtype = .BF16, .shape = &.{ table_rows, ple_shape.dims }, .data = std.mem.sliceAsBytes(values) } }},
            .scale = .{ .dtype = .F32, .shape = &.{1}, .data = std.mem.asBytes(&scale) },
            .rows = table_rows,
            .dims = @intCast(ple_shape.dims),
        });
        const attn = tiny.attn;
        layers[0] = .{
            .attn_hc = try randomConnection(arena, random, true, rows),
            .mlp_hc = try randomConnection(arena, random, true, rows),
            .block = .{ .delta_net = try .init(.{
                .projection = &.{
                    try randomBf16(arena, random, dn.conv() * s.hidden, 0.06),
                    try randomBf16(arena, random, dn.out() * s.hidden, 0.06),
                    try randomBf16(arena, random, dn.value_heads * s.hidden, 0.06),
                    try randomBf16(arena, random, dn.value_heads * s.hidden, 0.06),
                },
                .conv = try randomBf16(arena, random, dn.conv() * gdn.taps, 0.5),
                .a_log = try randomFloats(arena, random, dn.value_heads, -1.0, 0.5),
                .dt_bias = try randomFloats(arena, random, dn.value_heads, 0.0, 1.0),
                .norm = try randomBf16(arena, random, gdn.value_dim, 1.0),
                .out = try randomBf16(arena, random, s.hidden * dn.out(), 0.02),
            }, s, rows) },
            .moe = try randomMoe(gpa, arena, random, &staging, rows),
            .ple = .{
                .weights = try ple.Weights.initHost(ple_shape, s.eps, 1, .{
                    .key = try randomBf16(arena, random, ple_shape.wide() * ple_shape.embedding(), spread),
                    .value = try randomBf16(arena, random, ple_shape.hidden * ple_shape.embedding(), spread),
                    .norm_key = try randomFloats(arena, random, ple_shape.wide(), 1.0, 0.1),
                    .norm_query = try randomFloats(arena, random, ple_shape.wide(), 1.0, 0.1),
                    .norm_conv = try randomFloats(arena, random, ple_shape.wide(), 1.0, 0.1),
                    .conv = try randomBf16(arena, random, ple_shape.wide() * ple_shape.taps, 0.5),
                }, rows),
                .table = table,
                .ngram = .{ .head_sizes = &.{ 5, 7 }, .head_offsets = &.{ 0, 5 }, .multipliers = &.{ 1, 3 }, .rows = table_rows, .dims = ple_shape.dims },
                .eos = 0,
            },
        };
        layers[1] = .{
            .attn_hc = try randomConnection(arena, random, true, rows),
            .mlp_hc = try randomConnection(arena, random, true, rows),
            .block = .{ .attention = try .init(attn, .{
                .q = try randomBf16(arena, random, attn.heads * 2 * attn.head_dim * s.hidden, spread),
                .k = try randomBf16(arena, random, attn.kv_heads * attn.head_dim * s.hidden, spread),
                .v = try randomBf16(arena, random, attn.kv_heads * attn.head_dim * s.hidden, spread),
                .index_qk = try randomBf16(arena, random, (attn.index_heads + 1) * attn.index_dim * s.hidden, spread),
                .o = try randomBf16(arena, random, s.hidden * attn.out(), 0.1),
                .q_norm = try randomFloats(arena, random, attn.head_dim, 1.0, 0.1),
                .k_norm = try randomFloats(arena, random, attn.head_dim, 1.0, 0.1),
                .index_q_norm = try randomFloats(arena, random, attn.index_dim, 1.0, 0.1),
                .index_k_norm = try randomFloats(arena, random, attn.index_dim, 1.0, 0.1),
            }, rows) },
            .moe = try randomMoe(gpa, arena, random, &staging, rows),
        };
        return .{
            .shape = s,
            .embed = embed,
            .layers = layers,
            .mixer = try randomConnection(arena, random, false, rows),
            .head = try .init(try randomBf16(arena, random, s.vocab * s.hidden, 0.1), s.vocab, s.hidden, rows),
            .attention = .{ .shape = attn, .rotary = try .init(attn), .head = attn.kv },
            .max_rows = tiny.max_rows,
            .scratch = pool,
            .gpa = gpa,
        };
    }
};

pub const sessions = struct {
    const request_file = @import("../../../src/server/request.zig");
    const draft = @import("../../../src/models/qwen4/draft.zig");
    const session_file = @import("../../../src/models/qwen4/session/session.zig");

    /// Where the states a session took on a reply's stride stand, lowest first.
    pub fn stridePositions(session: *const session_file.Session, gpa: Allocator) ![]usize {
        const out = try gpa.alloc(usize, session.strides.items.len);
        for (out, session.strides.items) |*pos, one| pos.* = one.pos;
        return out;
    }

    /// Where a session's snapshots stand, lowest first.
    pub fn snapshotPositions(session: *const session_file.Session, gpa: Allocator) ![]usize {
        const out = try gpa.alloc(usize, session.snapshots.items.len);
        for (out, session.snapshots.items) |*pos, one| pos.* = one.pos;
        return out;
    }

    /// Counts the pieces it is sent and asks for the end after `limit` of them.
    pub const Counting = struct {
        pieces: usize = 0,
        limit: usize = std.math.maxInt(usize),

        pub fn sink(self: *Counting) request_file.Sink {
            return .{ .context = self, .writeFn = write };
        }

        fn write(context: *anyopaque, kind: request_file.Kind, text: []const u8) request_file.Sink.Signal {
            const self: *Counting = @ptrCast(@alignCast(context));
            _ = kind;
            _ = text;
            self.pieces += 1;
            return if (self.pieces >= self.limit) .stop else .more;
        }
    };

    /// The deepest a window drafts: what the tests ask for where they do not name a depth, whatever a served
    /// reply's default is.
    pub const deepest = draft.forward_max_rows - 1;
};

/// What a test on the Flash Next checkpoint begins with: its config, its files and its layout.
pub const checkpoints = struct {
    const qwen4_config = @import("../../../src/models/qwen4/config.zig");
    const layout = @import("../../../src/models/qwen4/layout.zig");
    const model = @import("../../../src/models/qwen4/model.zig");
    const safetensors = @import("../../../src/weights/safetensors.zig");
    const attention = @import("../../../src/models/qwen4/blocks/attention.zig");

    /// The files an opened checkpoint holds: `open` in place (the layout reads its checkpoint by address), `close`
    /// after. `config` and the layout are made in `arena`.
    pub const Held = struct {
        diagnostic: qwen4_config.Diagnostic = .{},
        checkpoint: safetensors.Checkpoint = undefined,

        pub fn open(self: *Held, gpa: Allocator, arena: Allocator, path: []const u8) !struct { qwen4_config.Config, layout.Layout } {
            const io = testing.io;
            var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
            defer dir.close(io);
            self.diagnostic = .{};
            const config = try qwen4_config.Config.read(arena, io, dir, &self.diagnostic);
            self.checkpoint = try .open(gpa, io, .cwd(), path);
            errdefer self.checkpoint.close();
            const found = try layout.Layout.init(arena, &config, &self.checkpoint, &self.diagnostic);
            return .{ config, found };
        }

        pub fn close(self: *Held) void {
            self.checkpoint.close();
        }
    };

    /// The layers up to and with the first attention layer, which the tests load; null where there is none.
    pub fn attentionLayers(found: layout.Layout) ?usize {
        return for (found.layers, 1..) |maybe, n| {
            if (maybe) |layer| if (layer.block == .full_attention) break n;
        } else null;
    }

    /// `m` as a load of `rows` rows has it, where `m` was loaded for more: the same weights, and scratch that
    /// is only larger; its windows, rounds and buffers `rows` wide (`Model.max_rows`), as `hall_open` narrows
    /// a pool's rounds. A check written for a load of `rows` runs so on a load several checks share (spec 133).
    pub fn narrowed(m: model.Model, rows: usize) model.Model {
        std.debug.assert(rows <= m.max_rows);
        var narrow = m;
        narrow.max_rows = rows;
        return narrow;
    }

    /// `m` as a load with `.kv = kv, .head_kv = head` has it, where `m` was loaded with other caches: the same
    /// weights, the trunk's caches stored `kv` and the head's `head` (spec 133 WS-08). A load's cache storage
    /// lands in its shapes alone (`Model.loadWith`: `attention_shape.kv`, the `Attention`s' `head`, the whole
    /// head's beside a share); no weight is read or packed by it (`attention.Weights.load` reads the shape's
    /// dims, `Mtp.load` takes no storage), and a run sizes its caches and buffers by the shape it is given.
    pub fn storing(m: model.Model, kv: attention.Storage, head: attention.Storage) model.Model {
        std.debug.assert(head == kv or head == .bf16);
        var stored = m;
        if (stored.attention) |*shared| {
            shared.shape.kv = kv;
            shared.head = head;
        }
        if (stored.mtp) |*drafter| if (drafter.whole) |*whole| {
            whole.attention.shape.kv = kv;
            whole.attention.head = head;
        };
        return stored;
    }
};

/// The serial references: a reply by one-row steps of the model alone, written without the decoder, which the
/// tests of windows, drafts, sessions and the pool hold their replies against.
pub const serial = struct {
    const forward = @import("../../../src/models/qwen4/forward.zig");
    const grammar = @import("../../../src/text/grammar.zig");
    const masks_file = @import("../../../src/decode/masks.zig");
    const model = @import("../../../src/models/qwen4/model.zig");
    const sampling = @import("../../../src/decode/sampling.zig");

    /// `count` tokens after `prompt` by one-row steps, each token keyed by its place in the sequence: what
    /// `zspark chat --serial` does, written without the decoder.
    pub fn serialReply(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling) ![]i32 {
        var state: forward.State = try .init(gpa, m, m.max_rows, 256);
        defer state.deinit();
        var b: forward.Buffers = try .init(gpa, m, m.max_rows, 256);
        defer b.deinit();
        var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
        defer sampler.deinit(gpa);
        const logits = try gpa.alloc(u16, m.shape.vocab);
        defer gpa.free(logits);
        const out = try gpa.alloc(i32, count);
        errdefer gpa.free(out);
        for (prompt) |token| {
            try forward.forward(m, &state, &b, &.{token});
            try forward.commit(m, &state, b, &.{token}, 1);
        }
        for (out, 0..) |*token, made| {
            try forward.logitsRow(b, 0, logits);
            token.* = try sampler.choose(logits, prompt.len + made, settings);
            try forward.forward(m, &state, &b, &.{token.*});
            try forward.commit(m, &state, b, &.{token.*}, 1);
        }
        return out;
    }

    /// The reply to `prompt` under `constraint` by one-row steps, written without the decoder and without `Masks`:
    /// before every token the reply so far is walked again from the constraint's start, the mask filled there, and
    /// -inf written token by token. It ends with the first of `eos`, or at `count` tokens. `gate` as `Masks.gate`.
    pub fn serialConstrained(gpa: Allocator, m: model.Model, prompt: []const i32, count: usize, settings: sampling.Sampling, constraint: grammar.Constraint, gate: ?i32, eos: []const u32) ![]i32 {
        var state: forward.State = try .init(gpa, m, m.max_rows, 256);
        defer state.deinit();
        var b: forward.Buffers = try .init(gpa, m, m.max_rows, 256);
        defer b.deinit();
        var sampler: sampling.Sampler = try .init(gpa, m.shape.vocab);
        defer sampler.deinit(gpa);
        const logits = try gpa.alloc(u16, m.shape.vocab);
        defer gpa.free(logits);
        const mask = try gpa.alloc(u32, constraint.words);
        defer gpa.free(mask);
        const out = try gpa.alloc(i32, count);
        errdefer gpa.free(out);
        for (prompt) |token| {
            try forward.forward(m, &state, &b, &.{token});
            try forward.commit(m, &state, b, &.{token}, 1);
        }
        var made: usize = 0;
        while (made < count) {
            try forward.logitsRow(b, 0, logits);
            var open = gate == null;
            var at = try constraint.start();
            for (out[0..made]) |token| {
                if (open) at = (try constraint.accept(at, @intCast(token))).? else open = token == gate.?;
            }
            if (open) {
                try constraint.fill(at, mask);
                for (logits, 0..) |*logit, token| {
                    if (!grammar.allowed(mask, @intCast(token))) logit.* = masks_file.minus_infinity;
                }
            }
            const token = try sampler.choose(logits, prompt.len + made, settings);
            out[made] = token;
            made += 1;
            if (std.mem.indexOfScalar(u32, eos, @intCast(token)) != null) break;
            try forward.forward(m, &state, &b, &.{token});
            try forward.commit(m, &state, b, &.{token}, 1);
        }
        return gpa.realloc(out, made);
    }
};
