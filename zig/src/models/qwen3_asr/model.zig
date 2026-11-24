//! Qwen3-ASR's language model on the device: its weights as the checkpoint stores them (28 layers, the final
//! norm, `lm_head`), loaded a layer at a time from `layout.zig`'s tensors: bf16, or a W4A16_NVFP4 build's NVFP4
//! linears read as stored into `qmmf`'s lane matmul (`linear.Linear.stored`, spec 140). Nothing is quantized at
//! load. A layer's q, k and v are one stacked linear, the MLP's gate and up another. The head's weight is the embedding's table too: the
//! checkpoint's `embed_tokens` is the same bytes, and only one copy is loaded (spec 138's Decision).

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../weights/bf16.zig");
const config_file = @import("config.zig");
const cuda = @import("../../cuda/cuda.zig");
const layout_file = @import("layout.zig");
const linear = @import("../../weights/linear.zig");
const safetensors = @import("../../weights/safetensors.zig");

const Config = config_file.Config;
const Layout = layout_file.Layout;

pub const Layer = struct {
    input_norm: cuda.Buffer,
    post_norm: cuda.Buffer,
    q_norm: cuda.Buffer,
    k_norm: cuda.Buffer,
    /// q, k and v stacked [(heads + 2 kv_heads) head_dim, hidden]
    qkv: linear.Linear,
    o: linear.Linear,
    /// gate and up stacked [2 intermediate, hidden]
    gu: linear.Linear,
    down: linear.Linear,

    fn deinit(self: *Layer) void {
        inline for (.{ "input_norm", "post_norm", "q_norm", "k_norm", "qkv", "o", "gu", "down" }) |name| @field(self, name).deinit();
    }
};

/// The weights, passed by value (`decode/draft.zig`'s `Model`); `deinit` once.
pub const Model = struct {
    config: Config,
    /// The most rows one forward takes.
    max_rows: usize,
    /// The most rows of logits a forward writes.
    scored: usize,
    norm: cuda.Buffer,
    /// [vocab, hidden]: the head, whose weight is also the embedding's table (`table`)
    head: bf16.Linear,
    layers: []Layer,
    /// fp32 [head_dim / 2]: RoPE's inverse frequencies, theta^(-2 d / head_dim), as HF computes them in fp32.
    inv: cuda.Buffer,
    pool: *cuda.Scratch,
    gpa: Allocator,

    /// Loads every tensor of `layout`; a layer's bytes are read once (`Checkpoint.hold`) and let go after.
    pub fn load(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, layout: Layout, config: Config, max_rows: usize, scored: usize, diagnostic: *config_file.Diagnostic) !Model {
        const pool = try cuda.Scratch.create(gpa);
        errdefer pool.destroy();
        const rows = pool.rows(max_rows);
        const layers = try gpa.alloc(Layer, config.layers);
        errdefer gpa.free(layers);
        var made: usize = 0;
        errdefer for (layers[0..made]) |*layer| layer.deinit();
        for (layers, layout.layers) |*layer, t| {
            var ranges: [4 * 7][]const u8 = undefined;
            var held = try checkpoint.hold(gpa, t.ranges(&ranges));
            defer held.release();
            var arena: std.heap.ArenaAllocator = .init(gpa);
            defer arena.deinit();
            layer.* = try loadLayer(arena.allocator(), gpa, t, config, rows, diagnostic);
            made += 1;
        }
        var norm = try vector(layout.norm.data, config.hidden);
        errdefer norm.deinit();
        var head_held = try checkpoint.hold(gpa, &.{layout.head.data});
        defer head_held.release();
        var head: bf16.Linear = try .init(layout.head.data, config.vocab, config.hidden, pool.rows(scored));
        errdefer head.deinit();
        const half = config.head_dim / 2;
        var inv: cuda.Buffer = try .init(.f32, &.{half});
        errdefer inv.deinit();
        var freq: [128]f32 = undefined;
        for (freq[0..half], 0..) |*f, d| {
            // HF: 1 / theta ** (arange(0, dim, 2).float() / dim), in fp32
            const exponent: f32 = @as(f32, @floatFromInt(2 * d)) / @as(f32, @floatFromInt(config.head_dim));
            f.* = 1.0 / std.math.pow(f32, @floatCast(config.rope_theta), exponent);
        }
        try inv.upload(std.mem.sliceAsBytes(freq[0..half]));
        return .{ .config = config, .max_rows = max_rows, .scored = scored, .norm = norm, .head = head, .layers = layers, .inv = inv, .pool = pool, .gpa = gpa };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit();
        self.gpa.free(self.layers);
        self.norm.deinit();
        self.head.deinit();
        self.inv.deinit();
        self.pool.destroy();
        self.* = undefined;
    }

    /// bf16 [vocab, hidden]: the embedding's table, the head's own weight.
    pub fn table(self: Model) cuda.View {
        return self.head.weight.view;
    }
};

fn vector(bytes: []const u8, n: usize) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(.bf16, &.{n});
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

fn loadLayer(arena: Allocator, gpa: Allocator, t: layout_file.Layer, c: Config, rows: cuda.Scratch.Rows, diagnostic: *config_file.Diagnostic) !Layer {
    var input_norm = try vector(t.input_norm.data, c.hidden);
    errdefer input_norm.deinit();
    var post_norm = try vector(t.post_norm.data, c.hidden);
    errdefer post_norm.deinit();
    var q_norm = try vector(t.q_norm.data, c.head_dim);
    errdefer q_norm.deinit();
    var k_norm = try vector(t.k_norm.data, c.head_dim);
    errdefer k_norm.deinit();
    var qkv: linear.Linear = try .stored(arena, gpa, &.{ t.q, t.k, t.v }, rows, diagnostic);
    errdefer qkv.deinit();
    var o: linear.Linear = try .stored(arena, gpa, &.{t.o}, rows, diagnostic);
    errdefer o.deinit();
    var gu: linear.Linear = try .stored(arena, gpa, &.{ t.gate, t.up }, rows, diagnostic);
    errdefer gu.deinit();
    const down: linear.Linear = try .stored(arena, gpa, &.{t.down}, rows, diagnostic);
    return .{ .input_norm = input_norm, .post_norm = post_norm, .q_norm = q_norm, .k_norm = k_norm, .qkv = qkv, .o = o, .gu = gu, .down = down };
}

/// The checkpoint at `path` read and closed: its config and the model on the device, as stored.
pub fn open(gpa: Allocator, io: std.Io, path: []const u8, max_rows: usize, scored: usize) !Model {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const config = Config.read(arena.allocator(), io, dir, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, if (diagnostic.message.len > 0) diagnostic.message else @errorName(err) });
        return err;
    };
    const quant = Config.quantization(arena.allocator(), io, dir, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, if (diagnostic.message.len > 0) diagnostic.message else @errorName(err) });
        return err;
    };
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    const layout = Layout.init(arena.allocator(), &config, &checkpoint, quant, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, diagnostic.message });
        return err;
    };
    return Model.load(gpa, &checkpoint, layout, config, max_rows, scored, &diagnostic) catch |err| {
        if (diagnostic.message.len > 0) std.log.err("{s}: {s}", .{ path, diagnostic.message });
        return err;
    };
}
