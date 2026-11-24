//! `model.zig`'s tests.

const std = @import("std");
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const face = @import("../../../src/weights/face.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const Model = model.Model;
const faceOf = model.faceOf;
const hold = model.hold;

test "a mixed-precision checkpoint's faces: MXFP8 stacked, the stored weights, a window equal to its serial steps" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to a mixed-precision checkpoint's directory; skipped without.
    try checkFaces(std.mem.span(std.c.getenv("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest));
}

/// The first layers' projections of the hidden row as faces against the weights as the host dequantizes them,
/// and each row alone equal to the window: MXFP8 stacked into one run (the shared expert's gate and up too).
fn checkFaces(path: []const u8) !void {
    const testing = std.testing;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(testing.allocator, arena, path);
    defer opened.close();
    const k = config.hidden;
    const max_rows = 20;

    var prng: std.Random.DefaultPrng = .init(3);
    const random = prng.random();
    const x_bits = try arena.alloc(u16, max_rows * k);
    for (x_bits) |*bits| bits.* = numeric.bf16FromF32(random.floatNorm(f32));
    var x: cuda.Buffer = try .init(.bf16, &.{ max_rows, k });
    defer x.deinit();
    try x.upload(std.mem.sliceAsBytes(x_bits));

    // the first DeltaNet layers and the first attention layer
    for (found.layers[0..@min(found.layers.len, 4)]) |maybe| {
        const source = maybe orelse return error.SkipZigTest;
        const block_linears: []const layout.Linear = switch (source.block) {
            .linear_attention => |dn| &.{ dn.qkv, dn.z, dn.b, dn.a },
            .full_attention => |tensors| &.{ tensors.q, tensors.k, tensors.v, tensors.index_qk },
        };
        const shared: []const layout.Linear = &.{ source.moe.shared.gate, source.moe.shared.up };
        for ([_][]const layout.Linear{ block_linears, shared }) |linears| {
            var held = try hold(testing.allocator, found, linears);
            defer held.release();
            try checkFace(arena, linears, x, x_bits, k, max_rows);
        }
    }
}

fn checkFace(arena: std.mem.Allocator, linears: []const layout.Linear, x: cuda.Buffer, x_bits: []const u16, k: usize, max_rows: usize) !void {
    const testing = std.testing;
    {
        var scratch: cuda.Scratch = .init(testing.allocator);
        defer scratch.deinit();
        var projection = try faceOf(testing.allocator, linears, scratch.rows(max_rows));
        defer projection.deinit();
        // every part MXFP8: one stacked matmul
        try testing.expectEqual(@as(usize, 1), projection.count);
        try testing.expect(projection.runs[0].linear == .mxfp8);

        const n = projection.n;
        var out_f32: cuda.Buffer = try .init(.f32, &.{ max_rows, n });
        defer out_f32.deinit();
        try projection.forward(x.view, out_f32.view);
        var window: cuda.Buffer = try .init(.bf16, &.{ max_rows, n });
        defer window.deinit();
        try projection.forward(x.view, window.view);
        try cuda.synchronize();
        const got = try arena.alloc(f32, max_rows * n);
        try out_f32.download(std.mem.sliceAsBytes(got));
        const want_bits = try arena.alloc(u16, max_rows * n);
        try window.download(std.mem.sliceAsBytes(want_bits));

        // a few outputs of every linear against the weight as the host dequantizes it
        var first: usize = 0;
        for (linears) |linear| {
            const part = try linear.part();
            for ([_]usize{ 0, linear.rows / 2 + 1, linear.rows - 1 }) |output| for ([_]usize{ 0, max_rows - 1 }) |row| {
                var want: f64 = 0;
                var magnitude: f64 = 0;
                for (0..k) |i| {
                    const weight: f64 = switch (part) {
                        .bf16 => |bytes| numeric.f32FromBf16(std.mem.readInt(u16, bytes[(output * k + i) * 2 ..][0..2], .little)),
                        .mxfp8 => |rows| std.math.ldexp(@as(f64, numeric.f32FromE4m3(rows.codes[output * k + i])), @as(i32, rows.exponents[output * (k / 32) + i / 32]) - 127),
                        else => unreachable,
                    };
                    const term = @as(f64, numeric.f32FromBf16(x_bits[row * k + i])) * weight;
                    want += term;
                    magnitude += @abs(term);
                }
                try testing.expectApproxEqAbs(want, @as(f64, got[row * n + first + output]), magnitude * 1e-5);
            };
            first += linear.rows;
        }
        try testing.expectEqual(n, first);

        // each row alone, as a serial step takes it
        var step: cuda.Buffer = try .init(.bf16, &.{ 1, n });
        defer step.deinit();
        const got_bits = try arena.alloc(u16, n);
        for (0..max_rows) |row| {
            const base: [*]u8 = @ptrCast(x.view.ptr.?);
            try projection.forward(.init(base + row * k * 2, .bf16, &.{ 1, k }), step.view);
            try cuda.synchronize();
            try step.download(std.mem.sliceAsBytes(got_bits));
            try testing.expectEqualSlices(u16, want_bits[row * n ..][0..n], got_bits);
        }
    }
}

/// Expert `e`'s matrix `m` of `per` back out of the blocks `experts.Experts` holds on the device ([n/32, k/32,
/// per, 144] words an expert): the checkpoint's order, `codes` [n, k/2] and `block_scales` [n, k/16].
fn unpackExpert(blocks: []const u32, n: usize, k: usize, per: usize, m: usize, codes: []u8, block_scales: []u8) void {
    @memset(codes, 0);
    for (0..n / 32) |cb| for (0..k / 32) |g| {
        const block = blocks[((cb * (k / 32) + g) * per + m) * 144 ..][0..144];
        for (0..8) |gq| for (0..4) |t| for (0..4) |j| {
            const output = cb * 32 + j * 8 + gq;
            const word = block[(gq * 4 + t) * 4 + j];
            for (0..2) |h| for (0..4) |q| {
                const input = g * 32 + h * 16 + t * 4 + q;
                const code: u8 = @intCast((word >> @intCast(4 * (2 * h + q / 2 + 4 * (q % 2)))) & 0xf);
                codes[output * (k / 2) + input / 2] |= if (input % 2 == 0) code else code << 4;
            };
        };
        const scales: *const [64]u8 = @ptrCast(block[128..]);
        for (0..4) |t| for (0..2) |h| for (0..4) |j| for (0..2) |c| {
            const output = cb * 32 + j * 8 + t * 2 + c;
            block_scales[output * (k / 16) + g * 2 + h] = scales[((t * 2 + h) * 4 + j) * 2 + c];
        };
    };
}

test "FP4-Local's routed experts placed by the device from a staged part are the host's packed bytes, one Spark and both ranks" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = std.testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const testing = std.testing;
    const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
    const experts = @import("../../../src/weights/experts.zig");
    const load_file = @import("../../../src/models/qwen4/model_load.zig");
    const share_file = @import("../../../src/models/qwen4/share.zig");
    const share_layout = @import("../../../src/models/qwen4/share_layout.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const centred = found.norms_centred.?;
    const heads = share_file.Heads.of(config);
    const pool: *cuda.Scratch = try .create(gpa);
    defer pool.destroy();
    const scratch = pool.rows(4);
    // the first layer and the last, whole and each rank's: down's columns read where they lie on a rank, or its
    // half of the outputs read and the other inputs of them traded (`experts.Swap`)
    const last = found.layers.len - 1;
    for ([_]usize{ 0, last }) |i| for ([_]?u32{ null, 0, 1 }) |rank_id| for ([_]bool{ false, true }) |swapping| {
        if (swapping and rank_id == null) continue;
        const rank: ?share_file.Share = if (rank_id) |id| .of(id) else null;
        const cfg = if (rank != null) share_layout.halved(config) else config;
        const shape: model.Shape = .of(cfg);
        const attention_shape = try attention.Shape.of(&cfg);
        var reading: load_file.Reading = .init(gpa, found, found.layers.len, rank);
        reading.swapping = swapping;
        defer reading.deinit();
        const part = try reading.begin(load_file.Reading.layer(i));
        var whole = found.layers[i].?;
        if (whole.ple) |*branch| branch.table.shards = &.{};
        const read = try reading.moved(part, whole);
        var cuts: std.heap.ArenaAllocator = .init(gpa);
        defer cuts.deinit();
        // the host packs from the mapping where the stage holds half of down
        const host_read = if (swapping) whole else read;
        const host_source = if (rank) |one| try share_layout.sliceLayer(cuts.allocator(), host_read, one, heads) else host_read;
        const device_source = if (rank) |one| try share_layout.sliceLayerPlaced(cuts.allocator(), read, one, heads) else read;
        var host_staging: experts.Staging = .{};
        defer host_staging.deinit(gpa);
        var host = try load_file.loadLayer(gpa, &host_staging, host_source, shape, attention_shape, centred, scratch, false, null, null, .bf16);
        defer host.deinit(gpa);
        var device_staging: experts.Staging = .{ .place = true };
        defer device_staging.deinit(gpa);
        // the peer's side of the trade from the checkpoint's mapping, and this rank's checked against it
        var peer: SwapPeer = .{ .downs = &.{}, .rank = rank_id orelse 0 };
        if (swapping) {
            const each = found.layers[i].?.moe.experts.each;
            const downs = try arena.alloc([]const u8, each.len);
            for (each, downs) |expert, *down| down.* = expert.down.weight.data;
            peer.downs = downs;
            device_staging.swap = .{ .rank = rank_id.?, .context = &peer, .trade = SwapPeer.trade };
        }
        var placed = try load_file.loadLayer(gpa, &device_staging, device_source, shape, attention_shape, centred, scratch, false, null, rank, .bf16);
        defer placed.deinit(gpa);
        try device_staging.settle();
        reading.end();
        try testing.expect(device_staging.device != null);
        try testing.expectEqual(swapping, peer.traded);
        try testing.expect(peer.sent_right);
        const want = host.moe.stored.nvfp4.routed.nvfp4;
        const got = placed.moe.stored.nvfp4.routed.nvfp4;
        try testing.expectEqual(experts.Activations.bf16, got.activations);
        var bytes: usize = 0;
        for ([_][2]cuda.Buffer{ .{ want.up, got.up }, .{ want.down, got.down }, .{ want.up_scale, got.up_scale }, .{ want.down_scale, got.down_scale } }) |pair| {
            try testing.expectEqual(pair[0].bytes(), pair[1].bytes());
            const a = try gpa.alloc(u8, pair[0].bytes());
            defer gpa.free(a);
            const b = try gpa.alloc(u8, pair[1].bytes());
            defer gpa.free(b);
            try pair[0].download(a);
            try pair[1].download(b);
            try testing.expect(std.mem.eql(u8, a, b));
            bytes += a.len;
        }
        const whose: []const u8 = if (rank_id) |id| (if (id == 0) " rank 0" else " rank 1") else " whole";
        std.debug.print("layer {d}{s}{s}: {d} bytes placed by the device as the host packs them\n", .{ i, whose, if (swapping) ", down traded" else "", bytes });
    };
}

test "FP4-Local's MTP head read as a staged part and placed by the device is the held head's bytes, one Spark and both halves" {
    // Runs with ZSPARK_FLASH_NEXT_FP4_LOCAL set to the checkpoint's directory; skipped without.
    const path = std.testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const testing = std.testing;
    const experts = @import("../../../src/weights/experts.zig");
    const load_file = @import("../../../src/models/qwen4/model_load.zig");
    const share_file = @import("../../../src/models/qwen4/share.zig");
    const gpa = testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    try testing.expect(load_file.placesMtp(found.mtp.?));
    const pool: *cuda.Scratch = try .create(gpa);
    defer pool.destroy();
    const scratch = pool.rows(4);
    for ([_]?u32{ null, 0, 1 }) |rank_id| {
        const rank: ?share_file.Share = if (rank_id) |id| .of(id) else null;
        var reading: load_file.Reading = .init(gpa, found, found.layers.len, rank);
        reading.mtp = if (rank != null) .split else .whole;
        defer reading.deinit();
        const part = try reading.begin(reading.mtpPart());
        const staged = try reading.moved(part, found.mtp.?);
        var host_staging: experts.Staging = .{};
        defer host_staging.deinit(gpa);
        var held = try model.Mtp.load(gpa, &host_staging, config, found, scratch, false, rank, null);
        defer held.deinit(gpa);
        var device_staging: experts.Staging = .{};
        defer device_staging.deinit(gpa);
        var placed = try model.Mtp.load(gpa, &device_staging, config, found, scratch, false, rank, staged);
        defer placed.deinit(gpa);
        reading.end();
        try testing.expect(device_staging.device != null);
        const want = held.layer.moe.stored.nvfp4.routed.nvfp4;
        const got = placed.layer.moe.stored.nvfp4.routed.nvfp4;
        var bytes: usize = 0;
        for ([_][2]cuda.Buffer{ .{ want.up, got.up }, .{ want.down, got.down }, .{ want.up_scale, got.up_scale }, .{ want.down_scale, got.down_scale }, .{ held.norm_hidden, placed.norm_hidden } }) |pair| {
            try testing.expectEqual(pair[0].bytes(), pair[1].bytes());
            const a = try gpa.alloc(u8, pair[0].bytes());
            defer gpa.free(a);
            const b = try gpa.alloc(u8, pair[1].bytes());
            defer gpa.free(b);
            try pair[0].download(a);
            try pair[1].download(b);
            try testing.expect(std.mem.eql(u8, a, b));
            bytes += a.len;
        }
        const whose: []const u8 = if (rank_id) |id| (if (id == 0) " rank 0's half" else " rank 1's half") else " whole";
        std.debug.print("the MTP head{s}: {d} bytes from the stage as held\n", .{ whose, bytes });
    }
}

/// The other rank of `experts.Swap` in a test: what it would send, read from each expert's down in the mapping.
const SwapPeer = struct {
    downs: []const []const u8,
    rank: usize,
    traded: bool = false,
    sent_right: bool = true,

    fn trade(context: *const anyopaque, send: cuda.View, recv: cuda.View) cuda.Error!void {
        const self: *SwapPeer = @ptrCast(@alignCast(@constCast(context)));
        self.traded = true;
        const gpa = std.testing.allocator;
        const sent = gpa.alloc(u8, send.count()) catch return error.CudaFailure;
        defer gpa.free(sent);
        try cuda.synchronize();
        try cuda.download(send, sent);
        const given = gpa.alloc(u8, recv.count()) catch return error.CudaFailure;
        defer gpa.free(given);
        const half: usize = @intCast(send.size[1]);
        const bytes: usize = @intCast(send.size[2]);
        const other = 1 - self.rank;
        const row = 2 * bytes;
        for (self.downs, 0..) |weight, e| for (0..half) |r| {
            const at = (e * half + r) * bytes;
            if (!std.mem.eql(u8, sent[at..][0..bytes], weight[(self.rank * half + r) * row + other * bytes ..][0..bytes])) self.sent_right = false;
            @memcpy(given[at..][0..bytes], weight[(other * half + r) * row + self.rank * bytes ..][0..bytes]);
        };
        const into: cuda.Buffer = .{ .view = recv };
        try into.upload(given);
    }
};
