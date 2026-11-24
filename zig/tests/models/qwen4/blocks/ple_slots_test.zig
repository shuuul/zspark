//! `kernels/qwen4/ple_slots.cu`'s bound on its slot tables (spec 103): the pool's convolution reads a row's
//! segment by the slot `sid` names, and a row whose slot or segment lies outside the tables or the round is left
//! as it was, while the rows beside it keep their bits. The pool's own rounds (`pool/rounds.zig` `pleRows`) are
//! held against the serial steps in `pool/`'s tests; this launches the kernel alone.

const std = @import("std");
const cuda = @import("../../../../src/cuda/cuda.zig");
const numeric = @import("../../../../src/weights/numeric.zig");
const testing = std.testing;

extern fn zs_qwen4_ple_conv_slots(gated: *const cuda.View, pss: *const cuda.View, norm_conv: *const cuda.View, conv: *const cuda.View, h: *const cuda.View, nrow: *const cuda.View, sid: *const cuda.View, seg: *const cuda.View, ptrs: *const cuda.View, stride: i64, col: i64, eps: f64, streams: i64, dilation: i64) c_int;

const fx = struct {
    const rows = 4;
    const streams = 2;
    const dims = 64;
    const wide = streams * dims;
    const taps = 4;
    const dilation = 1;
    const tail_rows = (taps - 1) * dilation;
    const slots = 2;
    const stride = 3;
    const col = 1;
    const eps = 1e-6;
};

fn bf16s(gpa: std.mem.Allocator, random: std.Random, count: usize) ![]u16 {
    const out = try gpa.alloc(u16, count);
    for (out) |*v| v.* = numeric.bf16FromF32(random.float(f32) * 2 - 1);
    return out;
}

fn device(dtype: cuda.DType, shape: []const usize, bytes: []const u8) !cuda.Buffer {
    var buffer: cuda.Buffer = try .init(dtype, shape);
    errdefer buffer.deinit();
    try buffer.upload(bytes);
    return buffer;
}

/// One launch over the fixture's rows with the given slot tables: the rows' h after it.
fn launch(gpa: std.mem.Allocator, sid: []const i32, seg: []const i32) ![]u16 {
    var prng: std.Random.DefaultPrng = .init(103);
    const random = prng.random();
    const gated_host = try bf16s(gpa, random, fx.rows * fx.wide);
    defer gpa.free(gated_host);
    const conv_host = try bf16s(gpa, random, fx.wide * fx.taps);
    defer gpa.free(conv_host);
    const h_host = try bf16s(gpa, random, fx.rows * fx.wide);
    defer gpa.free(h_host);
    const tail_host = try bf16s(gpa, random, fx.slots * fx.tail_rows * fx.wide);
    defer gpa.free(tail_host);
    var pss_host: [fx.rows * fx.streams]f32 = undefined;
    for (&pss_host) |*v| v.* = 1 + random.float(f32);
    var nc_host: [fx.wide]f32 = undefined;
    for (&nc_host) |*v| v.* = random.float(f32) + 0.5;

    var gated = try device(.bf16, &.{ fx.rows, fx.wide }, std.mem.sliceAsBytes(gated_host));
    defer gated.deinit();
    var pss = try device(.f32, &.{ fx.rows, fx.streams }, std.mem.sliceAsBytes(&pss_host));
    defer pss.deinit();
    var nc = try device(.f32, &.{fx.wide}, std.mem.sliceAsBytes(&nc_host));
    defer nc.deinit();
    var conv = try device(.bf16, &.{ fx.wide, fx.taps }, std.mem.sliceAsBytes(conv_host));
    defer conv.deinit();
    var h = try device(.bf16, &.{ fx.rows, fx.wide }, std.mem.sliceAsBytes(h_host));
    defer h.deinit();
    var nrow: cuda.Buffer = try .init(.bf16, &.{ fx.rows, fx.wide });
    defer nrow.deinit();
    try nrow.zero();
    var tails = try device(.bf16, &.{ fx.slots * fx.tail_rows, fx.wide }, std.mem.sliceAsBytes(tail_host));
    defer tails.deinit();
    var ptrs_host: [fx.slots * fx.stride]i64 = @splat(0);
    for (0..fx.slots) |s| {
        const tail = tails.span(s * fx.tail_rows, fx.tail_rows);
        ptrs_host[s * fx.stride + fx.col] = @intCast(@intFromPtr(tail.ptr.?));
    }
    var ptrs = try device(.i64, &.{ fx.slots, fx.stride }, std.mem.sliceAsBytes(&ptrs_host));
    defer ptrs.deinit();
    var sid_buffer = try device(.i32, &.{fx.rows}, std.mem.sliceAsBytes(sid));
    defer sid_buffer.deinit();
    var seg_buffer = try device(.i32, &.{ fx.slots, 4 }, std.mem.sliceAsBytes(seg));
    defer seg_buffer.deinit();

    try cuda.launched(zs_qwen4_ple_conv_slots(&gated.view, &pss.view, &nc.view, &conv.view, &h.view, &nrow.view, &sid_buffer.view, &seg_buffer.view, &ptrs.view, fx.stride, fx.col, fx.eps, fx.streams, fx.dilation));
    const out = try gpa.alloc(u16, fx.rows * fx.wide);
    errdefer gpa.free(out);
    try h.download(std.mem.sliceAsBytes(out));
    return out;
}

fn before(gpa: std.mem.Allocator) ![]u16 {
    var prng: std.Random.DefaultPrng = .init(103);
    const random = prng.random();
    // the draws `launch` makes before h's
    const skip = try bf16s(gpa, random, fx.rows * fx.wide + fx.wide * fx.taps);
    gpa.free(skip);
    return bf16s(gpa, random, fx.rows * fx.wide);
}

fn row(values: []const u16, r: usize) []const u16 {
    return values[r * fx.wide ..][0..fx.wide];
}

test "the pool's PLE convolution leaves a row whose slot or segment is outside its tables as it was, and the rows beside it keep their bits" {
    const gpa = testing.allocator;
    const h0 = try before(gpa);
    defer gpa.free(h0);
    // two segments of two rows: slot 0 rows 0 and 1, slot 1 rows 2 and 3 (first row, rows, kept, position)
    const seg = [_]i32{ 0, 2, 2, 7, 2, 2, 2, 11 };
    const good = try launch(gpa, &.{ 0, 0, 1, 1 }, &seg);
    defer gpa.free(good);
    for (0..fx.rows) |r| try testing.expect(!std.mem.eql(u16, row(good, r), row(h0, r)));

    // row 2 names slot 5 of 2: it is left as it was, and rows 0, 1 and 3 (whose taps read row 2's own input,
    // not its output) are the launch's above
    const past_slot = try launch(gpa, &.{ 0, 0, 5, 1 }, &seg);
    defer gpa.free(past_slot);
    try testing.expectEqualSlices(u16, row(h0, 2), row(past_slot, 2));
    for ([_]usize{ 0, 1, 3 }) |r| try testing.expectEqualSlices(u16, row(good, r), row(past_slot, r));
    // a negative slot too
    const negative = try launch(gpa, &.{ 0, 0, -1, 1 }, &seg);
    defer gpa.free(negative);
    try testing.expectEqualSlices(u16, row(h0, 2), row(negative, 2));

    // slot 1's segment runs past the round's four rows (from row 3, two rows): its rows are left as they were
    const past_round = [_]i32{ 0, 2, 2, 7, 3, 2, 2, 11 };
    const past = try launch(gpa, &.{ 0, 0, 1, 1 }, &past_round);
    defer gpa.free(past);
    for ([_]usize{ 2, 3 }) |r| try testing.expectEqualSlices(u16, row(h0, r), row(past, r));
    for ([_]usize{ 0, 1 }) |r| try testing.expectEqualSlices(u16, row(good, r), row(past, r));
}
