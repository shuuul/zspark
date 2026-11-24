//! Unlimited-OCR-2's language model on the device: its bf16 weights as the checkpoint stores them (embed, 12
//! layers, the final norm, `lm_head`), loaded a layer at a time from `layout.zig`'s tensors. Nothing is
//! quantized. A layer's q, k and v are one stacked linear, a dense MLP's gate and up another; a MoE layer's
//! routed experts are stacked [experts, 2 width, hidden] (gate rows, then up rows) and [experts, hidden, width]
//! for `kernels/bf16/experts.cu`. Every layer carries its DASH router (`Router`, spec 147) and the model its
//! page2d RoPE (`page2d`), which the main heads and the router's share.

const std = @import("std");
const Allocator = std.mem.Allocator;
const bf16 = @import("../../weights/bf16.zig");
const config_file = @import("config.zig");
const cuda = @import("../../cuda/cuda.zig");
const layout_file = @import("layout.zig");
const router_file = @import("router.zig");
const safetensors = @import("../../weights/safetensors.zig");

const Config = config_file.Config;
const Layout = layout_file.Layout;

/// A dense MLP: gate and up stacked [2 width, hidden], down [hidden, width].
pub const Dense = struct {
    gu: bf16.Linear,
    down: bf16.Linear,
    width: usize,

    fn deinit(self: *Dense) void {
        self.gu.deinit();
        self.down.deinit();
    }
};

pub const Experts = struct {
    /// [experts, hidden]: a row's gate logits, a bf16 product (HF's `F.linear` under bf16 autocast)
    router: bf16.Linear,
    /// bf16 [experts, 2 width, hidden]
    gu: cuda.Buffer,
    /// bf16 [experts, hidden, width]
    down: cuda.Buffer,
    shared: Dense,

    fn deinit(self: *Experts) void {
        self.router.deinit();
        self.gu.deinit();
        self.down.deinit();
        self.shared.deinit();
    }
};

/// A layer's DASH router, all bf16 as stored.
pub const Router = struct {
    /// `q_proj` and `k_proj` stacked [2 index_dim, hidden]
    qk: bf16.Linear,
    /// [index_dim]: Gemma RMSNorms' weights (`x * (1 + w)`)
    q_norm: cuda.Buffer,
    k_norm: cuda.Buffer,
    /// [index_dim]: the block summary's query
    head_cls: cuda.Buffer,

    fn deinit(self: *Router) void {
        self.qk.deinit();
        self.q_norm.deinit();
        self.k_norm.deinit();
        self.head_cls.deinit();
    }
};

pub const Layer = struct {
    input_norm: cuda.Buffer,
    post_norm: cuda.Buffer,
    /// q, k and v stacked [3 hidden, hidden]
    qkv: bf16.Linear,
    o: bf16.Linear,
    mlp: union(enum) { dense: Dense, experts: Experts },
    router: Router,

    fn deinit(self: *Layer) void {
        self.input_norm.deinit();
        self.post_norm.deinit();
        self.qkv.deinit();
        self.o.deinit();
        self.router.deinit();
        switch (self.mlp) {
            inline else => |*mlp| mlp.deinit(),
        }
    }
};

/// The weights, passed by value (`decode/draft.zig`'s `Model`); `deinit` once.
pub const Model = struct {
    config: Config,
    /// The most rows one forward takes.
    max_rows: usize,
    /// The most rows of logits a forward writes (a verify window's).
    scored: usize,
    embed: cuda.Buffer,
    norm: cuda.Buffer,
    head: bf16.Linear,
    layers: []Layer,
    /// DASH's page2d RoPE (spec 147), the attention's and the router's.
    page2d: router_file.Rope,
    pool: *cuda.Scratch,
    gpa: Allocator,

    /// Loads every tensor of `layout`; a layer's bytes are read once (`Checkpoint.hold`) and let go after.
    pub fn load(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, layout: Layout, read: Config, max_rows: usize, scored: usize) !Model {
        // the config's lists lie in the reader's arena (`open`'s, freed after the load): the model keeps its own
        var config = read;
        const scale = try gpa.dupe(f64, read.dash.scale);
        errdefer gpa.free(scale);
        const floor = try gpa.dupe(f64, read.dash.floor);
        errdefer gpa.free(floor);
        config.dash.scale = scale;
        config.dash.floor = floor;
        const pool = try cuda.Scratch.create(gpa);
        errdefer pool.destroy();
        const rows = pool.rows(max_rows);
        const layers = try gpa.alloc(Layer, config.layers);
        errdefer gpa.free(layers);
        var made: usize = 0;
        errdefer for (layers[0..made]) |*layer| layer.deinit();
        for (layers, layout.layers) |*layer, tensors| {
            var ranges: std.ArrayList([]const u8) = .empty;
            defer ranges.deinit(gpa);
            try ranges.appendSlice(gpa, &.{ tensors.q.data, tensors.k.data, tensors.v.data, tensors.o.data });
            try ranges.appendSlice(gpa, &.{ tensors.router.q.data, tensors.router.k.data });
            switch (tensors.mlp) {
                .dense => |d| try ranges.appendSlice(gpa, &.{ d.gate.data, d.up.data, d.down.data }),
                .experts => |e| {
                    for (e.gate, e.up, e.down) |g, u, d| try ranges.appendSlice(gpa, &.{ g.data, u.data, d.data });
                    try ranges.appendSlice(gpa, &.{ e.shared.gate.data, e.shared.up.data, e.shared.down.data });
                },
            }
            var held = try checkpoint.hold(gpa, ranges.items);
            defer held.release();
            layer.* = try loadLayer(tensors, config, rows);
            made += 1;
        }
        var embed: cuda.Buffer = try .init(.bf16, &.{ config.vocab, config.hidden });
        errdefer embed.deinit();
        try embed.upload(layout.embed.data);
        var norm = try vector(layout.norm.data, config.hidden);
        errdefer norm.deinit();
        var head: bf16.Linear = try .init(layout.head.data, config.vocab, config.hidden, pool.rows(scored));
        errdefer head.deinit();
        var page2d: router_file.Rope = try .init(config.dash.rope);
        errdefer page2d.deinit();
        return .{ .config = config, .max_rows = max_rows, .scored = scored, .embed = embed, .norm = norm, .head = head, .layers = layers, .page2d = page2d, .pool = pool, .gpa = gpa };
    }

    pub fn deinit(self: *Model) void {
        for (self.layers) |*layer| layer.deinit();
        self.gpa.free(self.layers);
        self.embed.deinit();
        self.norm.deinit();
        self.head.deinit();
        self.page2d.deinit();
        self.gpa.free(self.config.dash.scale);
        self.gpa.free(self.config.dash.floor);
        self.pool.destroy();
        self.* = undefined;
    }
};

fn vector(bytes: []const u8, n: usize) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(.bf16, &.{n});
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

fn dense(tensors: layout_file.Dense, width: usize, hidden: usize, rows: cuda.Scratch.Rows) !Dense {
    var gu: bf16.Linear = try .initStacked(&.{ tensors.gate.data, tensors.up.data }, hidden, rows);
    errdefer gu.deinit();
    const down: bf16.Linear = try .init(tensors.down.data, hidden, width, rows);
    return .{ .gu = gu, .down = down, .width = width };
}

fn loadLayer(tensors: layout_file.Layer, c: Config, rows: cuda.Scratch.Rows) !Layer {
    var input_norm = try vector(tensors.input_norm.data, c.hidden);
    errdefer input_norm.deinit();
    var post_norm = try vector(tensors.post_norm.data, c.hidden);
    errdefer post_norm.deinit();
    var qkv: bf16.Linear = try .initStacked(&.{ tensors.q.data, tensors.k.data, tensors.v.data }, c.hidden, rows);
    errdefer qkv.deinit();
    var o: bf16.Linear = try .init(tensors.o.data, c.hidden, c.hidden, rows);
    errdefer o.deinit();
    var dash_router: Router = try loadRouter(tensors.router, c, rows);
    errdefer dash_router.deinit();
    const mlp: @FieldType(Layer, "mlp") = switch (tensors.mlp) {
        .dense => |d| .{ .dense = try dense(d, c.intermediate, c.hidden, rows) },
        .experts => |e| experts: {
            var router: bf16.Linear = try .init(e.router.data, c.experts, c.hidden, rows);
            errdefer router.deinit();
            const w = c.expert_width;
            var gu: cuda.Buffer = try .init(.bf16, &.{ c.experts, 2 * w, c.hidden });
            errdefer gu.deinit();
            var down: cuda.Buffer = try .init(.bf16, &.{ c.experts, c.hidden, w });
            errdefer down.deinit();
            const part = w * c.hidden * 2;
            for (e.gate, e.up, e.down, 0..) |g, u, d, x| {
                try gu.uploadAt(2 * x * part, g.data);
                try gu.uploadAt(2 * x * part + part, u.data);
                try down.uploadAt(x * part, d.data);
            }
            const shared = try dense(e.shared, c.sharedWidth(), c.hidden, rows);
            break :experts .{ .experts = .{ .router = router, .gu = gu, .down = down, .shared = shared } };
        },
    };
    return .{ .input_norm = input_norm, .post_norm = post_norm, .qkv = qkv, .o = o, .mlp = mlp, .router = dash_router };
}

fn loadRouter(tensors: layout_file.Router, c: Config, rows: cuda.Scratch.Rows) !Router {
    const n = c.dash.index_dim;
    var qk: bf16.Linear = try .initStacked(&.{ tensors.q.data, tensors.k.data }, c.hidden, rows);
    errdefer qk.deinit();
    var q_norm = try vector(tensors.q_norm.data, n);
    errdefer q_norm.deinit();
    var k_norm = try vector(tensors.k_norm.data, n);
    errdefer k_norm.deinit();
    const head_cls = try vector(tensors.head_cls.data, n);
    return .{ .qk = qk, .q_norm = q_norm, .k_norm = k_norm, .head_cls = head_cls };
}

/// The checkpoint at `path` read and closed: its config and the model on the device.
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
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    const layout = Layout.init(arena.allocator(), &config, &checkpoint, &diagnostic) catch |err| {
        std.log.err("{s}: {s}", .{ path, diagnostic.message });
        return err;
    };
    return Model.load(gpa, &checkpoint, layout, config, max_rows, scored);
}
