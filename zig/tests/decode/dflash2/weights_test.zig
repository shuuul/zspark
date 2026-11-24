//! `weights.zig`'s tests: a rank's slices against slices taken by index here, the two storages, the codebooks,
//! and the published drafter's own share where `ZSPARK_GLM_DRAFTER` names its directory.

const std = @import("std");
const testing = std.testing;
const cuda = @import("../../../src/cuda/cuda.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");
const numeric = @import("../../../src/weights/numeric.zig");
const glm5_testing = @import("../../models/glm5/testing.zig");
const config_file = @import("../../../src/decode/dflash2/config.zig");
const layout = @import("../../../src/decode/dflash2/layout.zig");
const drafter_testing = @import("../../../src/models/glm5/checks/random_drafter.zig");
const weights = @import("../../../src/decode/dflash2/weights.zig");

const rows = 5;

/// `linear` on `x` bf16 [rows, k], as fp32.
fn run(gpa: std.mem.Allocator, linear: weights.Linear, x: cuda.Buffer) ![]f32 {
    var out: cuda.Buffer = try .init(.f32, &.{ rows, linear.outputs() });
    defer out.deinit();
    try linear.forward(x.view, out.view);
    return glm5_testing.fetch(f32, gpa, out);
}

/// Row `row` of the matrix `tensor` [n, k] times `x`, in f64.
fn dot(tensor: safetensors.Tensor, row: usize, first: usize, x: []const u16) f64 {
    const k: usize = @intCast(tensor.shape[1]);
    var sum: f64 = 0;
    for (x, 0..) |bits, at| {
        const w = std.mem.readInt(u16, tensor.data[(row * k + first + at) * 2 ..][0..2], .little);
        sum += @as(f64, numeric.f32FromBf16(w)) * numeric.f32FromBf16(bits);
    }
    return sum;
}

test "a rank's linears are the file's rows and columns Drafter.__init__ slices, in both storages" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var pool: cuda.Scratch = .init(testing.allocator);
    defer pool.deinit();
    var prng: std.Random.DefaultPrng = .init(67);
    const random = prng.random();
    const c = try drafter_testing.config(arena, 512, 64, 5, &.{ 1, 3 });
    var source: drafter_testing.Source = .empty;
    try drafter_testing.put(arena, random, c, &source);

    const d = c.hidden;
    const x_host = try arena.alloc(u16, rows * d);
    glm5_testing.randomBf16(random, x_host, 1.0);
    var x = try glm5_testing.device(u16, .bf16, &.{ rows, d }, x_host);
    defer x.deinit();
    const narrow_host = try arena.alloc(u16, rows * c.rankWidth());
    glm5_testing.randomBf16(random, narrow_host, 1.0);
    var narrow = try glm5_testing.device(u16, .bf16, &.{ rows, c.rankWidth() }, narrow_host);
    defer narrow.deinit();
    const heads_host = try arena.alloc(u16, rows * c.rankHeads() * c.head_dim);
    glm5_testing.randomBf16(random, heads_host, 1.0);
    var heads = try glm5_testing.device(u16, .bf16, &.{ rows, c.rankHeads() * c.head_dim }, heads_host);
    defer heads.deinit();

    for (0..config_file.world) |rank| {
        var stored: weights.Weights = try .load(arena, testing.allocator, source, &c, rank, .stored, .{ .pool = &pool, .count = rows });
        defer stored.deinit();
        var mixed: weights.Weights = try .load(arena, testing.allocator, source, &c, rank, .mixed, .{ .pool = &pool, .count = rows });
        defer mixed.deinit();
        try testing.expect(mixed.bytes() < stored.bytes());
        const layer = stored.layers[1];
        const p = "layers.1.";
        const q_inner = c.rankHeads() * c.head_dim;
        const kv_inner = c.rankKvHeads() * c.head_dim;
        const width = c.rankWidth();
        // each output of a stored linear against the file's row (or columns) at the rank's offset
        const Case = struct { linear: weights.Linear, input: cuda.Buffer, host: []const u16, name: []const u8, row: usize, col: usize, at: usize, n: usize };
        const cases = [_]Case{
            .{ .linear = layer.q, .input = x, .host = x_host, .name = p ++ "self_attn.q_proj.weight", .row = rank * q_inner, .col = 0, .at = 0, .n = q_inner },
            .{ .linear = layer.kv, .input = x, .host = x_host, .name = p ++ "self_attn.k_proj.weight", .row = rank * kv_inner, .col = 0, .at = 0, .n = kv_inner },
            .{ .linear = layer.kv, .input = x, .host = x_host, .name = p ++ "self_attn.v_proj.weight", .row = rank * kv_inner, .col = 0, .at = kv_inner, .n = kv_inner },
            .{ .linear = layer.gate_up, .input = x, .host = x_host, .name = p ++ "mlp.gate_proj.weight", .row = rank * width, .col = 0, .at = 0, .n = width },
            .{ .linear = layer.gate_up, .input = x, .host = x_host, .name = p ++ "mlp.up_proj.weight", .row = rank * width, .col = 0, .at = width, .n = width },
            .{ .linear = layer.o, .input = heads, .host = heads_host, .name = p ++ "self_attn.o_proj.weight", .row = 0, .col = rank * q_inner, .at = 0, .n = d },
            .{ .linear = layer.down, .input = narrow, .host = narrow_host, .name = p ++ "mlp.down_proj.weight", .row = 0, .col = rank * width, .at = 0, .n = d },
            .{ .linear = layer.a_kp, .input = x, .host = x_host, .name = p ++ "attention_conv.kernel_projection.weight", .row = 0, .col = 0, .at = 0, .n = c.kernelWidth() },
            .{ .linear = layer.m_kp, .input = x, .host = x_host, .name = p ++ "mlp_conv.kernel_projection.weight", .row = 0, .col = 0, .at = 0, .n = c.kernelWidth() },
            .{ .linear = stored.projection, .input = x, .host = x_host, .name = "candidate_selector.hidden_projection.weight", .row = 0, .col = 0, .at = 0, .n = c.selector_rank },
        };
        for (cases) |case| {
            const got = try run(arena, case.linear, case.input);
            const outputs = case.linear.outputs();
            const k = case.host.len / rows;
            const tensor = source.get(case.name).?;
            for (0..rows) |r| for (0..case.n) |o| {
                const want = dot(tensor, case.row + o, case.col, case.host[r * k ..][0..k]);
                testing.expectApproxEqAbs(want, @as(f64, got[r * outputs + case.at + o]), 2e-4) catch |err| {
                    std.debug.print("{s} rank {d} row {d} output {d}\n", .{ case.name, rank, r, o });
                    return err;
                };
            };
        }
        // what the table keeps as stored is the same bits in both loads; four bits are near, not equal
        const Pair = struct { a: weights.Linear, b: weights.Linear, input: cuda.Buffer, four: bool };
        const mixed_layer = mixed.layers[1];
        const pairs = [_]Pair{
            .{ .a = layer.kv, .b = mixed_layer.kv, .input = x, .four = false },
            .{ .a = layer.a_kp, .b = mixed_layer.a_kp, .input = x, .four = false },
            .{ .a = layer.m_kp, .b = mixed_layer.m_kp, .input = x, .four = false },
            .{ .a = stored.projection, .b = mixed.projection, .input = x, .four = false },
            .{ .a = layer.q, .b = mixed_layer.q, .input = x, .four = true },
            .{ .a = layer.o, .b = mixed_layer.o, .input = heads, .four = true },
            .{ .a = layer.gate_up, .b = mixed_layer.gate_up, .input = x, .four = true },
            .{ .a = layer.down, .b = mixed_layer.down, .input = narrow, .four = true },
        };
        for (pairs) |pair| {
            try testing.expectEqual(pair.four, pair.b == .four);
            const want = try run(arena, pair.a, pair.input);
            const got = try run(arena, pair.b, pair.input);
            if (!pair.four) {
                try testing.expectEqualSlices(u32, @ptrCast(want), @ptrCast(got));
                continue;
            }
            var err: f64 = 0;
            var size: f64 = 0;
            var differing: usize = 0;
            for (want, got) |w, g| {
                err += (@as(f64, w) - g) * (@as(f64, w) - g);
                size += @as(f64, w) * w;
                differing += @intFromBool(w != g);
            }
            // E2M1 with a scale a group of 16: about a tenth of a weight's size
            try testing.expect(differing > want.len / 2);
            try testing.expect(@sqrt(err / size) < 0.15);
        }
        try testing.expect(stored.fc == .bf16 and mixed.fc == .four);
        try testing.expectEqual(c.hidden, mixed.fc.outputs());
    }

    var books: weights.Codebooks = try .load(testing.allocator, source, &c);
    defer books.deinit(testing.allocator);
    const pred = source.get("candidate_selector.predecessor_codebook").?;
    const succ = source.get("candidate_selector.successor_codebook").?;
    for ([_]usize{ 0, 17, c.vocab - 1 }) |token| for ([_]usize{ 0, 5, c.selector_rank - 1 }) |at| {
        const index = (token * c.selector_rank + at) * 2;
        try testing.expectEqual(numeric.f32FromBf16(std.mem.readInt(u16, pred.data[index..][0..2], .little)), books.predecessor[token * c.selector_rank + at]);
        try testing.expectEqual(numeric.f32FromBf16(std.mem.readInt(u16, succ.data[index..][0..2], .little)), books.successor[token * c.selector_rank + at]);
    };
}

test "the published drafter's share loads in both storages at the bytes its layout gives" {
    // Runs with ZSPARK_GLM_DRAFTER set to the drafter's directory; skipped without. Rank 1's share: 1.2 GiB
    // as stored, then 0.4 GiB with the four-bit copies made in memory.
    const path = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var diagnostic: config_file.Diagnostic = .{};
    const c = try config_file.Config.read(arena, io, dir, &diagnostic);
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, io, .cwd(), path);
    defer checkpoint.close();
    const found = try layout.Layout.init(arena, &c, &checkpoint, &diagnostic);
    var pool: cuda.Scratch = .init(testing.allocator);
    defer pool.deinit();
    const mib: f64 = 1024.0 * 1024.0;
    for ([_]weights.Storage{ .stored, .mixed }) |storage| {
        var load_arena: std.heap.ArenaAllocator = .init(testing.allocator);
        defer load_arena.deinit();
        const start: std.Io.Timestamp = .now(io, .awake);
        var w: weights.Weights = try .load(load_arena.allocator(), testing.allocator, &checkpoint, &c, 1, storage, .{ .pool = &pool, .count = 64 });
        defer w.deinit();
        const seconds = @as(f64, @floatFromInt(start.durationTo(.now(io, .awake)).toNanoseconds())) / 1e9;
        const want = found.device(false, storage == .stored);
        std.debug.print("drafter rank 1 {t}: {d:.1} MiB on the device (layout {d:.1}), {d:.1} s\n", .{ storage, @as(f64, @floatFromInt(w.bytes())) / mib, @as(f64, @floatFromInt(want)) / mib, seconds });
        // a four-bit linear pads its outputs to whole tiles of 64: nothing at these shapes
        try testing.expectApproxEqAbs(@as(f64, @floatFromInt(want)), @as(f64, @floatFromInt(w.bytes())), 0.5 * mib);
    }
    var books: weights.Codebooks = try .load(testing.allocator, &checkpoint, &c);
    defer books.deinit(testing.allocator);
    try testing.expectEqual(c.vocab * c.selector_rank, books.predecessor.len);
}
