//! Flash Next's embedding and hyper-connections on the device. The residual is `streams` copies of `dims` values a
//! row; a branch is written back into every stream, then read out (norm, down, SiLU, up, mix) for the next block.

const cuda = @import("../../../cuda/cuda.zig");
const partials = @import("../partials.zig");

/// Values of a stream whose squares one entry of `pss` sums.
pub const block = 256;

/// What a write-back adds to the streams before squaring them.
pub const Pending = union(enum) {
    /// Nothing: the first read-out of a forward.
    none,
    /// bf16 [rows, dims] and its inject gates bf16 [rows, streams].
    branch: struct { branch: cuda.View, inject: cuda.View },
    /// The MoE's routed slots fp32 [rows * slots, dims] (the last slot unread), the shared expert's fp32
    /// [rows, dims] in that slot's place, the slots' weights fp32 [rows, slots], and the inject gates. Without
    /// `shared` the last slot of `y` is the shared expert's and is read.
    slots: struct { y: cuda.View, shared: ?cuda.View, weights: cuda.View, inject: cuda.View },
    /// The MoE's slots already summed, fp32 [rows, dims]: `slots`' fma chain in slot order before its rounding,
    /// as `exl3_experts`' `forwardSum` leaves it (spec 073 WS-07), and the inject gates. `slots`' bits.
    sum: struct { sum: cuda.View, inject: cuda.View },
    /// Gathered fp32 MoE partials [ranks, rows, dims], each already folded. Added rank 0 first into `into`, then
    /// `.sum`. A world of one never uses it.
    ranks_sum: struct { gathered: cuda.View, inject: cuda.View, into: cuda.View },
};

/// ids int32 [rows], table bf16 [vocab, dims] -> out bf16 [rows, streams * dims].
pub fn embed(ids: cuda.View, table: cuda.View, out: cuda.View, streams: usize) cuda.Error!void {
    return cuda.launched(cuda.kernels.zs_embed(&ids, &table, &out, @intCast(streams)));
}

/// h bf16 [rows, streams * dims] takes `pending` in place; pss fp32 [rows, dims / 256, streams] gets each
/// stream's squares, which `normed` reads.
pub fn writeback(h: cuda.View, pss: cuda.View, pending: Pending, streams: usize) cuda.Error!void {
    if (pending == .ranks_sum) {
        const g = pending.ranks_sum;
        try partials.add(g.gathered, g.into);
        return writeback(h, pss, .{ .sum = .{ .sum = g.into, .inject = g.inject } }, streams);
    }
    const s: i64 = @intCast(streams);
    return cuda.launched(switch (pending) {
        .none => zs_qwen4_hc_writeback(&h, &pss, 0, null, null, null, null, null, s),
        .branch => |b| zs_qwen4_hc_writeback(&h, &pss, 1, &b.branch, &b.inject, null, null, null, s),
        .slots => |m| zs_qwen4_hc_writeback(&h, &pss, 2, null, &m.inject, &m.y, &m.weights, if (m.shared) |*shared| shared else null, s),
        .sum => |m| zs_qwen4_hc_writeback(&h, &pss, 3, null, &m.inject, &m.sum, null, null, s),
        .ranks_sum => unreachable,
    });
}

/// normed bf16 [rows, streams * dims] = h over its stream's RMS, times scale fp32 [streams * dims].
pub fn normed(h: cuda.View, pss: cuda.View, scale: cuda.View, out: cuda.View, eps: f32, streams: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_normed(&h, &pss, &scale, &out, eps, @intCast(streams)));
}

/// `writeback` and the `normed` after it as one launch (`site`) where the two stand side by side
/// (spec 073): the same bits, a launch less a site. `ZSPARK_HC_JOINED=0` keeps the two launches.
pub var joined: bool = true;

/// The MoE's slots go by `site` up to this many rows: each of its four blocks a row sums the slots again, which
/// is the faster while they stay in L2 and, at a prompt chunk's rows, 2.7 times the two launches' time.
pub const joined_slot_rows = 8;

/// Whether `pending` over `rows` rows goes with the norm after it.
pub fn joins(pending: Pending, rows: usize) bool {
    return switch (pending) {
        .ranks_sum => false,
        .slots => joined and rows <= joined_slot_rows,
        else => joined,
    };
}

/// `writeback` of `pending`, then `normed`, in one launch: the same bits as the two.
pub fn site(h: cuda.View, pss: cuda.View, pending: Pending, scale: cuda.View, out: cuda.View, eps: f32, streams: usize) cuda.Error!void {
    const s: i64 = @intCast(streams);
    return cuda.launched(switch (pending) {
        .none => zs_qwen4_hc_site(&h, &pss, 0, null, null, null, null, null, s, &scale, &out, eps),
        .branch => |b| zs_qwen4_hc_site(&h, &pss, 1, &b.branch, &b.inject, null, null, null, s, &scale, &out, eps),
        .slots => |m| zs_qwen4_hc_site(&h, &pss, 2, null, &m.inject, &m.y, &m.weights, if (m.shared) |*shared| shared else null, s, &scale, &out, eps),
        .sum => |m| zs_qwen4_hc_site(&h, &pss, 3, null, &m.inject, &m.sum, null, null, s, &scale, &out, eps),
        .ranks_sum => unreachable,
    });
}

/// The down projection's fp32 [rows, low (+ streams)], as K slices this sums in order (`face.Face.slices`) ->
/// act bf16 [rows, low] and inject gates bf16 [rows, streams].
pub fn activate(down: cuda.Slices, act: cuda.View, inject: ?cuda.View, streams: usize, low: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_act(&down.view, &act, if (inject) |*view| view else null, @intCast(streams), @intCast(low)));
}

/// mixed bf16 [rows, dims] from the up projection and the normed streams, both bf16 [rows, streams * dims].
pub fn mix(up: cuda.View, normed_streams: cuda.View, mixed: cuda.View, streams: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_mix(&up, &normed_streams, &mixed, @intCast(streams)));
}

/// h bf16 [rows, streams * dims] -> out bf16 [rows, dims], the streams' mean. `out`'s rows may be strided (a
/// tap among the five, side by side). `mean` is GLM's f32-acc or sequential-rounded bf16 (spec 122).
pub fn streamMean(h: cuda.View, out: cuda.View, streams: usize, mean: @import("../taps.zig").Mean) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_stream_mean(&h, &out, @intCast(streams), @backingInt(mean)));
}

/// The read-out's up and `mix` at a prompt window's rows (`hc_window.cu`, spec 095): act [rows, low] times the up projection's weight
/// [streams * dims, low], both bf16, then `mix` with the normed streams, the up product never written.
/// The bits of the GEMM and of `mix`. `stages` 2 or 3 (0: 2) moves no bit.
pub fn upMixWindow(act: cuda.View, up: cuda.View, normed_streams: cuda.View, mixed: cuda.View, streams: usize, stages: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_up_mix_window(&act, &up, &normed_streams, &mixed, @intCast(streams), @intCast(stages)));
}

/// A read-out of this many rows or more whose up projection is one bf16 matmul of one K slice takes
/// `upMixWindow` in place of the GEMM and `mix` (`forward.readout`); 0 is never. The bits are the same either way.
/// From 128 rows (ncu, Flash Next's shapes, spec 095 WS-03): 74 us against 93 at 128, 127 against 154 at 256,
/// 465 against 795 at 2,048; at 64 rows 76 against 66.
pub var window_rows: usize = 128;
/// Read-outs that went so: a test's proof that its comparison was of the window launch.
pub var window_readouts: u64 = 0;

/// The read-out at a prompt window's rows in three launches where the chain is five (spec 148 WS-03, after Mia's
/// AI Lab's recipe, patch 0014): the write-back without its norm, `downWindow` and `upMixStreams`, each forming
/// the norm from the streams itself, so the normed rows are never written or read. Every output byte is the
/// chain's (`hc_fused_test.zig`). `ZSPARK_HC_FUSED=0` keeps the chain, for a comparison.
pub var fused: bool = true;
/// A read-out of this many rows or more goes so (`forward.readoutAfter`); under it the chain, the same bits. The
/// down launch's blocks walk the whole K, about 250 us at any width to 1,024 rows (nsys, Flash Next's shapes, spec
/// 148 WS-03): down and up 375 us against the chain's 442 at 512 rows, 482 against 616 at 1,024, 803 against 1,111
/// at 2,048; 319 against 205 at 256.
pub var fused_rows: usize = 512;
/// Read-outs that went so: a test's proof that its comparison was of the fused launches.
pub var fused_readouts: u64 = 0;

/// h bf16 [rows, streams * dims] normed as `normed` norms it (pss the write-back's, scale fp32 [streams * dims]),
/// times the down projection's weight bf16 [n, streams * dims] in `sk` K slices (the linear's own), the slices
/// summed in order and activated as `activate` does: act bf16 [rows, low] and, where n is low + streams, the
/// inject gates bf16 [rows, streams] (`hc_down.cu`). The bits of `normed`, the GEMM and `activate`.
pub fn downWindow(h: cuda.View, pss: cuda.View, scale: cuda.View, down: cuda.View, sk: usize, act: cuda.View, inject: ?cuda.View, eps: f32, streams: usize, low: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_down_window(&h, &pss, &scale, &down, &act, if (inject) |*view| view else null, @intCast(streams), @intCast(low), @intCast(sk), eps));
}

/// `upMixWindow` with the streams h in place of the normed rows, normed in the epilogue as `normed` norms them:
/// the bits of `normed`, the up GEMM and `mix` (`hc_window.cu`).
pub fn upMixStreams(act: cuda.View, up: cuda.View, h: cuda.View, pss: cuda.View, scale: cuda.View, mixed: cuda.View, eps: f32, streams: usize) cuda.Error!void {
    return cuda.launched(zs_qwen4_hc_up_mix_streams(&act, &up, &h, &pss, &scale, &mixed, @intCast(streams), eps));
}

extern fn zs_qwen4_hc_down_window(h: *const cuda.View, pss: *const cuda.View, scale: *const cuda.View, w: *const cuda.View, act: *const cuda.View, inject: ?*const cuda.View, streams: i64, low: i64, sk: i64, eps: f64) c_int;
extern fn zs_qwen4_hc_up_mix_streams(act: *const cuda.View, up: *const cuda.View, h: *const cuda.View, pss: *const cuda.View, scale: *const cuda.View, mixed: *const cuda.View, streams: i64, eps: f64) c_int;
extern fn zs_qwen4_hc_writeback(h: *const cuda.View, pss: *const cuda.View, mode: i64, branch: ?*const cuda.View, inject: ?*const cuda.View, y: ?*const cuda.View, wts: ?*const cuda.View, shared: ?*const cuda.View, streams: i64) c_int;
extern fn zs_qwen4_hc_site(h: *const cuda.View, pss: *const cuda.View, mode: i64, branch: ?*const cuda.View, inject: ?*const cuda.View, y: ?*const cuda.View, wts: ?*const cuda.View, shared: ?*const cuda.View, streams: i64, scale: *const cuda.View, normed: *const cuda.View, eps: f64) c_int;
extern fn zs_qwen4_hc_normed(h: *const cuda.View, pss: *const cuda.View, scale: *const cuda.View, normed: *const cuda.View, eps: f64, streams: i64) c_int;
extern fn zs_qwen4_hc_act(dn: *const cuda.View, act: *const cuda.View, inject: ?*const cuda.View, streams: i64, low: i64) c_int;
extern fn zs_qwen4_hc_mix(up: *const cuda.View, normed: *const cuda.View, mixed: *const cuda.View, streams: i64) c_int;
extern fn zs_qwen4_hc_up_mix_window(act: *const cuda.View, up: *const cuda.View, normed: *const cuda.View, mixed: *const cuda.View, streams: i64, stages: i64) c_int;
extern fn zs_qwen4_hc_stream_mean(h: *const cuda.View, out: *const cuda.View, streams: i64, mean_mode: i64) c_int;
