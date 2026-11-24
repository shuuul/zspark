//! Flash Next's MoE on an NVFP4 checkpoint: the router picks `top` of the routed experts a row, which run on the
//! grouped NVFP4 kernel (`Routed`); the shared expert (BF16 or MXFP8) runs on its faces and takes the last
//! slot. The slots and their weights stay pending for the next hyper-connection's write-back.

const std = @import("std");
const bf16 = @import("../../../weights/bf16.zig");
const cuda = @import("../../../cuda/cuda.zig");
const experts = @import("../../../weights/experts.zig");
const hc = @import("hc.zig");
const Face = @import("../../../weights/face.zig").Face;

/// A layer's MoE weights on the device. `dims` is the model's hidden size, `width` an expert's.
pub const Weights = struct {
    /// [routed + 1, dims]: the router's rows, then the shared expert's gate row.
    router: bf16.Linear,
    stored: Stored,

    pub const Stored = union(enum) {
        nvfp4: Nvfp4,
    };

    /// A layer's routed experts on the plan's kernels: NVFP4, stored so or requantized at load.
    pub const Routed = union(enum) {
        nvfp4: experts.Experts,

        pub fn count(self: Routed) usize {
            return switch (self) {
                inline else => |routed| routed.count,
            };
        }

        pub fn deinit(self: *Routed) void {
            switch (self.*) {
                inline else => |*routed| routed.deinit(),
            }
        }

        fn run(self: Routed, plan: experts.Plan, picks: cuda.View, x: cuda.View, act: cuda.View, quantized: ?experts.Quantized, y: cuda.View, rows: usize, from: usize, skip: ?usize) cuda.Error!void {
            return switch (self) {
                inline else => |routed| routed.run(plan, picks, x, act, quantized, y, rows, from, skip),
            };
        }

        /// Whether the experts take W4A4 rows in a prompt's window (`model.PromptForm.card`).
        pub fn quantizes(self: Routed) bool {
            return switch (self) {
                inline else => |routed| routed.activations == .nvfp4,
            };
        }
    };

    pub const Nvfp4 = struct {
        routed: Routed,
        /// [2 width, dims]: the shared expert's gate rows, then its up rows.
        shared_gate_up: Face,
        /// [dims, width]
        shared_down: Face,
    };

    /// Whether any of `layers`' routed experts take W4A4 rows (their `Buffers` hold `quantized`).
    pub fn quantizesAny(layers: anytype) bool {
        for (layers) |layer| switch (layer.moe.stored) {
            .nvfp4 => |stored| if (stored.routed.quantizes()) return true,
        };
        return false;
    }

    pub fn deinit(self: *Weights) void {
        self.router.deinit();
        switch (self.stored) {
            .nvfp4 => |*stored| {
                stored.routed.deinit();
                stored.shared_gate_up.deinit();
                stored.shared_down.deinit();
            },
        }
        self.* = undefined;
    }
};

/// Scratch for up to `rows` rows; `y`, `shared_y` and `weights` are what the write-back reads.
pub const Buffers = struct {
    rows: usize,
    top: usize,
    /// fp32 [rows, routed + 1]
    logits: cuda.Buffer,
    /// int32 [rows, top + 1]: each slot's expert, the shared expert's id (routed) last
    picks: cuda.Buffer,
    /// fp32 [rows, top + 1]
    weights: cuda.Buffer,
    plan: experts.Plan,
    /// bf16 [rows * (top + 1), width]
    act: cuda.Buffer,
    /// fp32 [rows * (top + 1), dims]: the routed slots; a row's last slot is the shared expert's and stays unwritten
    y: cuda.Buffer,
    /// bf16 [rows, 2 width]
    shared_gate_up: cuda.Buffer,
    /// bf16 [rows, width]
    shared_act: cuda.Buffer,
    /// fp32 [rows, dims]: the shared expert's result, the row's last slot
    shared_y: cuda.Buffer,
    /// W4A4 experts' (`a4`): a pair's rows quantized for gate and up and for down
    quantized: ?experts.Quantized = null,

    pub fn init(rows: usize, top: usize, routed: usize, width: usize, dims: usize, a4: bool) cuda.Error!Buffers {
        const slots = top + 1;
        var self: Buffers = undefined;
        self.rows = rows;
        self.top = top;
        self.logits = try .init(.f32, &.{ rows, routed + 1 });
        errdefer self.logits.deinit();
        self.picks = try .init(.i32, &.{ rows, slots });
        errdefer self.picks.deinit();
        self.weights = try .init(.f32, &.{ rows, slots });
        errdefer self.weights.deinit();
        self.plan = try .init(rows, slots, routed + 1);
        errdefer self.plan.deinit();
        self.act = try .init(.bf16, &.{ rows * slots, width });
        errdefer self.act.deinit();
        self.y = try .init(.f32, &.{ rows * slots, dims });
        errdefer self.y.deinit();
        self.shared_gate_up = try .init(.bf16, &.{ rows, 2 * width });
        errdefer self.shared_gate_up.deinit();
        self.shared_act = try .init(.bf16, &.{ rows, width });
        errdefer self.shared_act.deinit();
        self.shared_y = try .init(.f32, &.{ rows, dims });
        errdefer self.shared_y.deinit();
        self.quantized = if (a4) try .init(rows * slots, dims, width) else null;
        return self;
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(rows: usize, top: usize, routed: usize, width: usize, dims: usize, a4: bool) usize {
        const of = cuda.Buffer.bytesOf;
        const slots = top + 1;
        var bytes = of(.f32, &.{ rows, routed + 1 }) + of(.i32, &.{ rows, slots }) + of(.f32, &.{ rows, slots }) + experts.Plan.bytesFor(rows, slots, routed + 1);
        bytes += of(.bf16, &.{ rows * slots, width }) + of(.f32, &.{ rows * slots, dims });
        bytes += of(.bf16, &.{ rows, 2 * width }) + of(.bf16, &.{ rows, width }) + of(.f32, &.{ rows, dims });
        // `experts.Quantized`: codes and e4m3 scales of a pair's inputs to gate and up, and to down
        if (a4) bytes += rows * slots * (dims / 2 + dims / 16 + width / 2 + width / 16);
        return bytes;
    }

    pub fn deinit(self: *Buffers) void {
        self.logits.deinit();
        self.picks.deinit();
        self.weights.deinit();
        self.plan.deinit();
        self.act.deinit();
        self.y.deinit();
        self.shared_gate_up.deinit();
        self.shared_act.deinit();
        self.shared_y.deinit();
        if (self.quantized) |*one| one.deinit();
        self.* = undefined;
    }
};

/// What `forward` left for the write-back with `inject`'s gates.
pub fn pending(buffers: Buffers, inject: cuda.View) hc.Pending {
    return .{ .slots = .{ .y = buffers.y.view, .shared = buffers.shared_y.view, .weights = buffers.weights.view, .inject = inject } };
}

/// The first `rows` rows of `buffer` as a view.
/// Routes x bf16 [rows, dims] and runs its experts: `buffers.y` holds each row's routed slots, `buffers.shared_y`
/// the shared expert's, and `buffers.weights` what each slot is worth. W4A4 experts take the rows from `from` on
/// W4A4, the rows before A16 (`experts.Experts.run`; the pool's `quantize_from`, `forward.setForm`).
pub fn forward(all: Weights, buffers: Buffers, x: cuda.View, rows: usize, from: usize) cuda.Error!void {
    std.debug.assert(rows <= buffers.rows and x.size[0] == rows);
    const slots = buffers.top + 1;
    const logits = buffers.logits.rows(rows);
    const picks = buffers.picks.rows(rows);
    // the selection sums the router's K slices as it reads them, and writes the logits
    const sliced = try all.router.slices(x, logits);
    try cuda.launched(zs_qwen4_moe_topk(&logits, &picks, &buffers.weights.view, @intCast(buffers.top), if (sliced.view.size[0] > 1) &sliced.view else null));
    const weights = switch (all.stored) {
        .nvfp4 => |stored| stored,
    };
    const shared = weights.routed.count();
    const act = buffers.act.rows(rows * slots);
    const y = buffers.y.rows(rows * slots);
    try weights.routed.run(buffers.plan, picks, x, act, buffers.quantized, y, rows, from, shared);

    const gate_up = buffers.shared_gate_up.rows(rows);
    const shared_act = buffers.shared_act.rows(rows);
    const shared_y = buffers.shared_y.rows(rows);
    try weights.shared_gate_up.forward(x, gate_up);
    try cuda.launched(zs_qwen4_moe_shared_act(&gate_up, &shared_act));
    try weights.shared_down.forward(shared_act, shared_y);
}

pub extern fn zs_qwen4_moe_topk(logits: *const cuda.View, picks: *const cuda.View, weights: *const cuda.View, top: i64, part: ?*const cuda.View) c_int;
extern fn zs_qwen4_moe_shared_act(gu: *const cuda.View, act: *const cuda.View) c_int;
