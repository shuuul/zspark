//! The residual norm's refusals (spec 106 WS-05, after TensorFold v1.0.12's 140ae71e): `normed` and the
//! joined site load h and store normed four bf16 values at a time and read the scale four floats at a time, so a
//! view whose start those loads cannot take is refused before anything is launched, not read wrong. zspark's
//! other norms (GLM's, the MTP head's, attention's q and k) read their rows one value at a time.
//! The entries are called directly: `cuda.launched` logs a refusal as an error, which fails a test.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../../src/cuda/cuda.zig");

extern fn zs_qwen4_hc_normed(h: *const cuda.View, pss: *const cuda.View, scale: *const cuda.View, normed: *const cuda.View, eps: f64, streams: i64) c_int;
extern fn zs_qwen4_hc_site(h: *const cuda.View, pss: *const cuda.View, mode: i64, branch: ?*const cuda.View, inject: ?*const cuda.View, y: ?*const cuda.View, wts: ?*const cuda.View, shared: ?*const cuda.View, streams: i64, scale: *const cuda.View, normed: *const cuda.View, eps: f64) c_int;
extern fn zs_last_error() [*:0]const u8;

const rows = 2;
const streams = 4;
const dims = 256;
const wide = streams * dims;

/// `buffer` from byte `bytes` on, as `shape`.
fn off(buffer: cuda.Buffer, bytes: usize, dtype: cuda.DType, shape: []const usize) cuda.View {
    return .init(@ptrCast(@as([*]u8, @ptrCast(buffer.view.ptr.?)) + bytes), dtype, shape);
}

fn expectRefused(status: c_int, says: []const u8) !void {
    try testing.expect(status != 0);
    try testing.expect(std.mem.indexOf(u8, std.mem.span(zs_last_error()), says) != null);
}

test "the residual norm refuses h, normed or scale off the start its wide loads take, alone and joined" {
    var h: cuda.Buffer = try .init(.bf16, &.{rows * wide + 8});
    defer h.deinit();
    var pss: cuda.Buffer = try .init(.f32, &.{ rows, dims / 256, streams });
    defer pss.deinit();
    var scale: cuda.Buffer = try .init(.f32, &.{wide + 4});
    defer scale.deinit();
    var normed: cuda.Buffer = try .init(.bf16, &.{rows * wide + 8});
    defer normed.deinit();
    try h.zero();
    try pss.zero();
    try scale.zero();
    const shape = [_]usize{ rows, wide };
    const h_at = off(h, 0, .bf16, &shape);
    const scale_at = off(scale, 0, .f32, &.{wide});
    const normed_at = off(normed, 0, .bf16, &shape);
    const h_off = off(h, 2, .bf16, &shape);
    const scale_off = off(scale, 4, .f32, &.{wide});
    const normed_off = off(normed, 2, .bf16, &shape);

    try testing.expectEqual(@as(c_int, 0), zs_qwen4_hc_normed(&h_at, &pss.view, &scale_at, &normed_at, 1e-6, streams));
    try expectRefused(zs_qwen4_hc_normed(&h_off, &pss.view, &scale_at, &normed_at, 1e-6, streams), "8 bytes");
    try expectRefused(zs_qwen4_hc_normed(&h_at, &pss.view, &scale_at, &normed_off, 1e-6, streams), "8 bytes");
    try expectRefused(zs_qwen4_hc_normed(&h_at, &pss.view, &scale_off, &normed_at, 1e-6, streams), "scale on 16");

    try testing.expectEqual(@as(c_int, 0), zs_qwen4_hc_site(&h_at, &pss.view, 0, null, null, null, null, null, streams, &scale_at, &normed_at, 1e-6));
    try expectRefused(zs_qwen4_hc_site(&h_off, &pss.view, 0, null, null, null, null, null, streams, &scale_at, &normed_at, 1e-6), "8 bytes");
    try expectRefused(zs_qwen4_hc_site(&h_at, &pss.view, 0, null, null, null, null, null, streams, &scale_at, &normed_off, 1e-6), "8 bytes");
    try expectRefused(zs_qwen4_hc_site(&h_at, &pss.view, 0, null, null, null, null, null, streams, &scale_off, &normed_at, 1e-6), "scale on 16");
    try cuda.synchronize();
}
