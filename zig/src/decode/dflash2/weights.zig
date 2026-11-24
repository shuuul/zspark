//! A rank's share of the DFlash2 drafter on the device, and the selector's codebooks on the host: every
//! tensor sliced as `dflash2.py:Drafter.__init__` slices it, each linear as stored (BF16) or in NVFP4 made at
//! load (`layout.zig`'s table says which). The four-bit copy lives in memory only; nothing of the drafter is
//! written anywhere.
//!
//! WS-03's Qwen drafter imports `Weights`, `Layer`, `Linear`, `Codebooks`, `Storage`, `LoadError` and
//! `draftHead`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../cuda/cuda.zig");
const safetensors = @import("../../weights/safetensors.zig");
const linear_file = @import("../../weights/linear.zig");
const share = @import("../../weights/share.zig");
const config_file = @import("config.zig");
const layout = @import("layout.zig");
const Config = config_file.Config;

pub const LoadError = error{ MissingTensor, UnsupportedShape } || Allocator.Error || cuda.Error;

/// Which of `layout.kinds`' two columns a load follows.
pub const Storage = enum {
    /// Every linear BF16 as the file has it: what the reference check runs, and an option.
    stored,
    /// `Kind.storage`: the large linears NVFP4, the rest as stored.
    mixed,

    pub fn of(self: Storage, name: []const u8) layout.Storage {
        const kind = layout.kinds[layout.find(name).?];
        return if (self == .stored and kind.storage == .four) .bf16 else kind.storage;
    }

    /// A linear's: no linear is the host's.
    fn ofLinear(self: Storage, name: []const u8) linear_file.Storage {
        return switch (self.of(name)) {
            .bf16 => .bf16,
            .four => .four,
            .host => unreachable,
        };
    }
};

/// A rank's four-bit copy of its rows of the main model's vocabulary, `rows` bf16 [vocab / world, hidden] as the
/// main checkpoint stores them, for the drafter's logits (TensorFold's `draft_head`): made in memory, never
/// written. The main model chooses every token from its own rows.
pub fn draftHead(arena: Allocator, gpa: Allocator, rows_of: []const u8, hidden: usize, rows: cuda.Scratch.Rows) LoadError!Linear {
    return .init(arena, gpa, &.{rows_of}, hidden, .four, rows);
}

/// A linear of the drafter in either storage (`weights/linear.zig`).
pub const Linear = linear_file.Linear;
pub const LinearStorage = linear_file.Storage;

/// What a load reads its tensors through: `source` is anything with `get(name) ?safetensors.Tensor`.
fn Reader(comptime Source: type) type {
    return struct {
        arena: Allocator,
        gpa: Allocator,
        source: Source,
        rank: usize,
        storage: Storage,
        rows: cuda.Scratch.Rows,
        /// `layers.N.` or nothing.
        under: []const u8 = "",

        const Self = @This();

        /// The tensor `leaf` as the rank holds it, BF16.
        fn part(self: Self, leaf: []const u8) LoadError!safetensors.Tensor {
            const name = try std.mem.concat(self.arena, u8, &.{ self.under, leaf });
            const tensor = self.source.get(name) orelse {
                std.log.err("{s}: not in the drafter's file", .{name});
                return error.MissingTensor;
            };
            if (tensor.dtype != .BF16) return error.UnsupportedShape;
            return share.take(self.arena, tensor, layout.rule(name) orelse return error.UnsupportedShape, self.rank, config_file.world);
        }

        /// On the device in its own shape: a norm's weight, a base kernel.
        fn buffer(self: Self, leaf: []const u8, shape: []const usize) LoadError!cuda.Buffer {
            const tensor = try self.part(leaf);
            var count: usize = 1;
            for (shape) |dim| count *= dim;
            if (tensor.data.len != count * 2) return error.UnsupportedShape;
            var out: cuda.Buffer = try .init(.bf16, shape);
            errdefer out.deinit();
            try out.upload(tensor.data);
            return out;
        }

        /// The linears `leaves` of one input of `k`, stacked; the storage is the first's.
        fn linear(self: Self, leaves: []const []const u8, n: usize, k: usize) LoadError!Linear {
            var parts: [3][]const u8 = undefined;
            var total: usize = 0;
            for (leaves, parts[0..leaves.len]) |leaf, *to| {
                to.* = (try self.part(leaf)).data;
                total += to.len;
            }
            if (total != n * k * 2) return error.UnsupportedShape;
            const first = try std.mem.concat(self.arena, u8, &.{ self.under, leaves[0] });
            return .init(self.arena, self.gpa, parts[0..leaves.len], k, self.storage.ofLinear(first), self.rows);
        }
    };
}

/// A layer as a rank holds it.
pub const Layer = struct {
    in_norm: cuda.Buffer,
    post_norm: cuda.Buffer,
    /// [2, 2, hidden]: a convolution's branches and taps.
    a_base: cuda.Buffer,
    m_base: cuda.Buffer,
    /// hidden -> a row's kernel [2, 2, hidden / group], whole.
    a_kp: Linear,
    m_kp: Linear,
    /// The rank's query heads; its key heads then its value heads, for a context row and a block's alike.
    /// TensorFold holds the three stacked for a block (`qkv`) and the two again for the context (`kv`).
    q: Linear,
    kv: Linear,
    /// [head_dim]: a head's RMS norm of q and of k.
    q_norm: cuda.Buffer,
    k_norm: cuda.Buffer,
    /// [hidden, the rank's heads]: a row-parallel partial.
    o: Linear,
    /// The rank's gate rows, then its up rows; [hidden, the rank's width]: the other partial.
    gate_up: Linear,
    down: Linear,

    fn load(r: anytype, c: *const Config) LoadError!Layer {
        const d = c.hidden;
        const q_inner = c.rankHeads() * c.head_dim;
        const kv_inner = c.rankKvHeads() * c.head_dim;
        var in_norm = try r.buffer("input_layernorm.weight", &.{d});
        errdefer in_norm.deinit();
        var post_norm = try r.buffer("post_attention_layernorm.weight", &.{d});
        errdefer post_norm.deinit();
        var a_base = try r.buffer("attention_conv.base_kernel", &.{ 2, 2, d });
        errdefer a_base.deinit();
        var m_base = try r.buffer("mlp_conv.base_kernel", &.{ 2, 2, d });
        errdefer m_base.deinit();
        var a_kp = try r.linear(&.{"attention_conv.kernel_projection.weight"}, c.kernelWidth(), d);
        errdefer a_kp.deinit();
        var m_kp = try r.linear(&.{"mlp_conv.kernel_projection.weight"}, c.kernelWidth(), d);
        errdefer m_kp.deinit();
        var q = try r.linear(&.{"self_attn.q_proj.weight"}, q_inner, d);
        errdefer q.deinit();
        var kv = try r.linear(&.{ "self_attn.k_proj.weight", "self_attn.v_proj.weight" }, 2 * kv_inner, d);
        errdefer kv.deinit();
        var q_norm = try r.buffer("self_attn.q_norm.weight", &.{c.head_dim});
        errdefer q_norm.deinit();
        var k_norm = try r.buffer("self_attn.k_norm.weight", &.{c.head_dim});
        errdefer k_norm.deinit();
        var o = try r.linear(&.{"self_attn.o_proj.weight"}, d, q_inner);
        errdefer o.deinit();
        var gate_up = try r.linear(&.{ "mlp.gate_proj.weight", "mlp.up_proj.weight" }, 2 * c.rankWidth(), d);
        errdefer gate_up.deinit();
        const down = try r.linear(&.{"mlp.down_proj.weight"}, d, c.rankWidth());
        return .{
            .in_norm = in_norm,
            .post_norm = post_norm,
            .a_base = a_base,
            .m_base = m_base,
            .a_kp = a_kp,
            .m_kp = m_kp,
            .q = q,
            .kv = kv,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .o = o,
            .gate_up = gate_up,
            .down = down,
        };
    }

    fn deinit(self: *Layer) void {
        inline for (.{ "in_norm", "post_norm", "a_base", "m_base", "a_kp", "m_kp", "q", "kv", "q_norm", "k_norm", "o", "gate_up", "down" }) |field| @field(self, field).deinit();
        self.* = undefined;
    }

    fn bytes(self: Layer) u64 {
        var sum: u64 = 0;
        inline for (.{ "in_norm", "post_norm", "a_base", "m_base", "q_norm", "k_norm" }) |field| sum += @field(self, field).bytes();
        inline for (.{ "a_kp", "m_kp", "q", "kv", "o", "gate_up", "down" }) |field| sum += @field(self, field).bytes();
        return sum;
    }
};

/// A rank's share of the drafter.
pub const Weights = struct {
    /// [hidden, taps x hidden], whole: a context row from its taps.
    fc: Linear,
    hidden_norm: cuda.Buffer,
    norm: cuda.Buffer,
    /// [selector rank, hidden], whole: a block row projected for the chain.
    projection: Linear,
    layers: []Layer,
    gpa: Allocator,

    /// The share of `rank` from `source`. `arena` holds the copies a column slice and a stack take until the
    /// load is over; the caller frees it then. Calls take up to `rows.count` rows.
    pub fn load(arena: Allocator, gpa: Allocator, source: anytype, c: *const Config, rank: usize, storage: Storage, rows: cuda.Scratch.Rows) LoadError!Weights {
        var r: Reader(@TypeOf(source)) = .{ .arena = arena, .gpa = gpa, .source = source, .rank = rank, .storage = storage, .rows = rows };
        var fc = try r.linear(&.{"fc.weight"}, c.hidden, c.tapWidth());
        errdefer fc.deinit();
        var hidden_norm = try r.buffer("hidden_norm.weight", &.{c.hidden});
        errdefer hidden_norm.deinit();
        var norm = try r.buffer("norm.weight", &.{c.hidden});
        errdefer norm.deinit();
        var projection = try r.linear(&.{"candidate_selector.hidden_projection.weight"}, c.selector_rank, c.hidden);
        errdefer projection.deinit();
        const layers = try gpa.alloc(Layer, c.layers);
        errdefer gpa.free(layers);
        var loaded: usize = 0;
        errdefer for (layers[0..loaded]) |*layer| layer.deinit();
        for (layers, 0..) |*layer, index| {
            r.under = try std.fmt.allocPrint(arena, "layers.{d}.", .{index});
            layer.* = try Layer.load(r, c);
            loaded += 1;
        }
        return .{ .fc = fc, .hidden_norm = hidden_norm, .norm = norm, .projection = projection, .layers = layers, .gpa = gpa };
    }

    pub fn deinit(self: *Weights) void {
        for (self.layers) |*layer| layer.deinit();
        self.gpa.free(self.layers);
        self.fc.deinit();
        self.hidden_norm.deinit();
        self.norm.deinit();
        self.projection.deinit();
        self.* = undefined;
    }

    /// Device bytes of the weights.
    pub fn bytes(self: Weights) u64 {
        var sum: u64 = self.fc.bytes() + self.hidden_norm.bytes() + self.norm.bytes() + self.projection.bytes();
        for (self.layers) |layer| sum += layer.bytes();
        return sum;
    }
};

/// The selector's codebooks as the chain reads them: fp32 [vocab, rank] on the host, one copy for both ranks.
pub const Codebooks = struct {
    predecessor: []f32,
    successor: []f32,
    rank: usize,

    pub fn load(gpa: Allocator, source: anytype, c: *const Config) LoadError!Codebooks {
        const predecessor = try widened(gpa, source, "candidate_selector.predecessor_codebook", c);
        errdefer gpa.free(predecessor);
        return .{ .predecessor = predecessor, .successor = try widened(gpa, source, "candidate_selector.successor_codebook", c), .rank = c.selector_rank };
    }

    fn widened(gpa: Allocator, source: anytype, name: []const u8, c: *const Config) LoadError![]f32 {
        const tensor = source.get(name) orelse {
            std.log.err("{s}: not in the drafter's file", .{name});
            return error.MissingTensor;
        };
        if (tensor.dtype != .BF16 or tensor.data.len != c.vocab * c.selector_rank * 2) return error.UnsupportedShape;
        const out = try gpa.alloc(f32, c.vocab * c.selector_rank);
        // bf16 is fp32's upper half
        for (out, 0..) |*value, at| value.* = @bitCast(@as(u32, std.mem.readInt(u16, tensor.data[at * 2 ..][0..2], .little)) << 16);
        return out;
    }

    pub fn deinit(self: *Codebooks, gpa: Allocator) void {
        gpa.free(self.predecessor);
        gpa.free(self.successor);
        self.* = undefined;
    }
};
