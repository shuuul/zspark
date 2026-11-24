//! The two-rank share against the whole model, as a gate (`ranks_test.zig`'s, spec 090): one rank's share alone
//! and the whole model timed through graphs at 1 to 16 rows and at a 2,048-row prompt. Opt-in: it runs only with
//! `ZSPARK_SHARE_GATE` set to the directory its `gate.txt` goes to, and `ZSPARK_FLASH_NEXT_FP4_LOCAL`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const testing = std.testing;
const fixtures = @import("testing.zig");
const cuda = @import("../../../src/cuda/cuda.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const forward = @import("../../../src/models/qwen4/forward.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const ranks = @import("../../../src/models/qwen4/ranks_forward.zig");

const gate_window = 2048;
const gate_scored = 16;
const gate_repeats: usize = 5;
const One = ranks.Held(1);

fn replay(graph: cuda.Graph, stream: cuda.Stream) !u64 {
    try graph.launch(stream);
    try cuda.synchronize();
    const start = cuda.monotonic();
    for (0..gate_repeats) |_| try graph.launch(stream);
    try cuda.synchronize();
    return (cuda.monotonic() - start) / gate_repeats;
}

fn endCapture(stream: *cuda.Stream) void {
    if (stream.end()) |graph| {
        var unused = graph;
        unused.deinit();
    } else |_| {}
}

fn captureWhole(stream: *cuda.Stream, m: model.Model, state: *forward.State, b: *forward.Buffers, tokens: []const i32, score: forward.Score) !cuda.Graph {
    try forward.stage(m, state, b, tokens);
    try stream.begin();
    forward.device(m, state, b, tokens.len, score) catch |err| {
        endCapture(stream);
        return err;
    };
    return stream.end();
}

fn captureShare(stream: *cuda.Stream, m: One.Model, state: *One.State, b: *One.Buffers, tokens: []const i32, score: forward.Score) !cuda.Graph {
    try forward.stage(m.ranks[0], &state.ranks[0], &b.ranks[0], tokens);
    try stream.begin();
    One.launch(m, state, b, tokens.len, null, score, null) catch |err| {
        endCapture(stream);
        return err;
    };
    return stream.end();
}

fn atRows(times: []const u64, rows: f64) f64 {
    const lo: usize = @intFromFloat(@floor(rows));
    const frac = rows - @as(f64, @floatFromInt(lo));
    const a: f64 = @floatFromInt(times[lo - 1]);
    const b: f64 = @floatFromInt(times[lo]);
    return a + (b - a) * frac;
}

const Gate = struct { round: [gate_scored]u64, bare: u64, prompt: u64 };

fn timeWhole(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, tokens: []const i32) !Gate {
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = config.layers, .max_rows = gate_window, .mtp = false });
    defer m.deinit();
    var state: forward.State = try .initKeeping(gpa, m, gate_window, gate_scored, gate_window);
    defer state.deinit();
    var buffers: forward.Buffers = try .initFor(gpa, m, m.layers, gate_window, gate_scored, gate_window);
    defer buffers.deinit();
    var stream: cuda.Stream = try .init();
    defer stream.deinit();
    var out: Gate = undefined;
    for (1..gate_scored + 1) |rows| {
        var graph = try captureWhole(&stream, m, &state, &buffers, tokens[0..rows], .all);
        defer graph.deinit();
        out.round[rows - 1] = try replay(graph, stream);
        std.debug.print("whole model, {d} rows: {d:.2} ms\n", .{ rows, @as(f64, @floatFromInt(out.round[rows - 1])) / 1e6 });
    }
    var bare = try captureWhole(&stream, m, &state, &buffers, tokens[0..gate_scored], .none);
    defer bare.deinit();
    out.bare = try replay(bare, stream);
    var prompt = try captureWhole(&stream, m, &state, &buffers, tokens, .none);
    defer prompt.deinit();
    out.prompt = try replay(prompt, stream);
    std.debug.print("whole model, {d} rows without the head: {d:.2} ms; a {d}-row prompt: {d:.2} ms\n", .{ gate_scored, @as(f64, @floatFromInt(out.bare)) / 1e6, gate_window, @as(f64, @floatFromInt(out.prompt)) / 1e6 });
    return out;
}

fn timeShare(gpa: Allocator, config: qwen4_config.Config, found: layout.Layout, tokens: []const i32) !Gate {
    var held: One.Model = .{
        .ranks = .{try model.Model.loadWith(gpa, config, found, .{ .layers = config.layers, .max_rows = gate_window, .mtp = false, .share = .of(0) })},
        .max_rows = gate_window,
        .alone = true,
    };
    defer held.deinit();
    var state: One.State = try .initKeeping(gpa, held, gate_window, gate_scored, gate_window);
    defer state.deinit();
    var buffers: One.Buffers = try .initFor(gpa, held, gate_window, gate_scored, gate_window);
    defer buffers.deinit();
    var stream: cuda.Stream = try .init();
    defer stream.deinit();
    var out: Gate = undefined;
    for (1..gate_scored + 1) |rows| {
        var graph = try captureShare(&stream, held, &state, &buffers, tokens[0..rows], .all);
        defer graph.deinit();
        out.round[rows - 1] = try replay(graph, stream);
        std.debug.print("rank 0 alone, {d} rows: {d:.2} ms\n", .{ rows, @as(f64, @floatFromInt(out.round[rows - 1])) / 1e6 });
    }
    var bare = try captureShare(&stream, held, &state, &buffers, tokens[0..gate_scored], .none);
    defer bare.deinit();
    out.bare = try replay(bare, stream);
    var prompt = try captureShare(&stream, held, &state, &buffers, tokens, .none);
    defer prompt.deinit();
    out.prompt = try replay(prompt, stream);
    std.debug.print("rank 0 alone, {d} rows without the head: {d:.2} ms; a {d}-row prompt: {d:.2} ms\n", .{ gate_scored, @as(f64, @floatFromInt(out.bare)) / 1e6, gate_window, @as(f64, @floatFromInt(out.prompt)) / 1e6 });
    return out;
}

test "a share alone against the whole model, the gate" {
    const out_path = testing.environ.getPosix("ZSPARK_SHARE_GATE") orelse return error.SkipZigTest;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    var tokens: [gate_window]i32 = @splat(1);

    // The share first: about 39 GiB, then freed before the whole model's 76.
    const share_side = try timeShare(gpa, config, found, &tokens);
    try cuda.synchronize();
    const whole_side = try timeWhole(gpa, config, found, &tokens);

    const prose_s = atRows(&share_side.round, 3.7) / 1e6;
    const prose_w = atRows(&whole_side.round, 3.7) / 1e6;
    const code_s = atRows(&share_side.round, 8.5) / 1e6;
    const code_w = atRows(&whole_side.round, 8.5) / 1e6;
    const prose = prose_w / prose_s;
    const code = code_w / code_s;
    const slower = @min(prose, code);
    const go = slower >= 1.35;
    std.debug.print("gate: prose {d:.3}x ({d:.2} ms / {d:.2} ms), code {d:.3}x ({d:.2} ms / {d:.2} ms), slower {d:.3}x, {s} at 1.35; gathers not in this run\n", .{ prose, prose_w, prose_s, code, code_w, code_s, slower, if (go) "go" else "stop" });
    std.debug.print("gate: prompt {d:.3}x ({d:.2} / {d:.2} ms); 16 rows without the head {d:.2} ms whole, {d:.2} ms share\n", .{ @as(f64, @floatFromInt(whole_side.prompt)) / @as(f64, @floatFromInt(share_side.prompt)), @as(f64, @floatFromInt(whole_side.prompt)) / 1e6, @as(f64, @floatFromInt(share_side.prompt)) / 1e6, @as(f64, @floatFromInt(whole_side.bare)) / 1e6, @as(f64, @floatFromInt(share_side.bare)) / 1e6 });

    var text: [4096]u8 = undefined;
    var used: usize = 0;
    const head = "rows ms_share ms_whole\n";
    @memcpy(text[0..head.len], head);
    used = head.len;
    for (share_side.round, whole_side.round, 1..) |s, w, rows| {
        const line = try std.fmt.bufPrint(text[used..], "{d} {d:.3} {d:.3}\n", .{ rows, @as(f64, @floatFromInt(s)) / 1e6, @as(f64, @floatFromInt(w)) / 1e6 });
        used += line.len;
    }
    const tail = try std.fmt.bufPrint(text[used..], "prose_ms {d:.3} {d:.3} {d:.3}\ncode_ms {d:.3} {d:.3} {d:.3}\nslower {d:.3} {s}\nprompt_ms {d:.3} {d:.3}\nbare16_ms {d:.3} {d:.3}\n", .{
        prose_s,                                          prose_w,                                          prose,                                          code_s,                                         code_w, code, slower, if (go) "go" else "stop",
        @as(f64, @floatFromInt(share_side.prompt)) / 1e6, @as(f64, @floatFromInt(whole_side.prompt)) / 1e6, @as(f64, @floatFromInt(share_side.bare)) / 1e6, @as(f64, @floatFromInt(whole_side.bare)) / 1e6,
    });
    used += tail.len;
    var folder = try std.Io.Dir.cwd().openDir(io, out_path, .{});
    defer folder.close(io);
    var file = try folder.createFile(io, "gate.txt", .{});
    defer file.close(io);
    var wbuf: [4096]u8 = undefined;
    var writer = file.writerStreaming(io, &wbuf);
    try writer.interface.writeAll(text[0..used]);
    try writer.interface.flush();
    try testing.expect(share_side.round[0] > 0 and whole_side.round[0] > 0);
}
