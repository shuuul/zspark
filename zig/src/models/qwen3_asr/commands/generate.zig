//! `zspark generate DIR [max=N] [rows=N] [context=N] (fixture=FILE | ID..)`: a prompt through Qwen3-ASR's language
//! model and its greedy reply (`draft.greedy`), timed. The prompt is token ids, or a reference fixture's
//! (`tools/reference/qwen3_asr/`, `ZSPARK_REFERENCE_QWEN3_ASR`): its `prompt.ids`, with its audio embeddings
//! (`enc.proj2`) laid over the `<|audio_pad|>` rows from `prompt.audio_pad_first`. Audio itself (the tower) is
//! WS-03's, serving WS-05's.

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const draft = @import("../draft.zig");
const forward_file = @import("../forward.zig");
const model_file = @import("../model.zig");
const safetensors = @import("../../../weights/safetensors.zig");

pub const usage = "generate DIR [max=N] [rows=N] [context=N] (fixture=FILE | ID..)";

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    var max: usize = 4096;
    var rows: usize = 512;
    var context: usize = 0;
    var fixture: ?[]const u8 = null;
    var ids: std.ArrayList(i32) = .empty;
    for (words[1..]) |word| {
        if (cli.value(word, "max=")) |text| {
            max = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "rows=")) |text| {
            rows = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "context=")) |text| {
            context = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "fixture=")) |text| {
            fixture = text;
        } else {
            try ids.append(arena, std.fmt.parseInt(i32, word, 10) catch {
                try stdout.print("usage: zspark {s}\n", .{usage});
                return 2;
            });
        }
    }
    var features: ?cuda.Buffer = null;
    defer if (features) |*f| f.deinit();
    var audio_rows: []u32 = &.{};
    if (fixture) |path| {
        var file: safetensors.File = try .open(gpa, init.io, .cwd(), path);
        defer file.close();
        const input = file.get("prompt.ids") orelse return refuse(path, "prompt.ids");
        const first_t = file.get("prompt.audio_pad_first") orelse return refuse(path, "prompt.audio_pad_first");
        const enc = file.get("enc.proj2") orelse return refuse(path, "enc.proj2");
        for (std.mem.bytesAsSlice(i32, input.data)) |id| try ids.append(arena, id);
        const first: usize = @intCast(std.mem.bytesAsSlice(i32, first_t.data)[0]);
        const count: usize = @intCast(enc.shape[0]);
        if (first + count > ids.items.len) return refuse(path, "audio rows within the prompt");
        audio_rows = try arena.alloc(u32, count);
        for (audio_rows, 0..) |*row, i| row.* = @intCast(first + i);
        features = try .init(.bf16, &.{ count, @intCast(enc.shape[1]) });
        try features.?.upload(enc.data);
    }
    if (ids.items.len == 0) {
        try stdout.print("usage: zspark {s}\n", .{usage});
        return 2;
    }
    if (context == 0) context = ids.items.len + max;
    const start = cuda.monotonic();
    var m = try model_file.open(gpa, init.io, words[0], rows, 1);
    defer m.deinit();
    var state: forward_file.State = try .init(gpa, m.config, context);
    defer state.deinit();
    var b: forward_file.Buffers = try .init(m.config, rows, 1, context);
    defer b.deinit();
    try cuda.synchronize();
    const loaded = cuda.monotonic();
    if (features) |f| state.audio = .{ .features = f.view, .rows = audio_rows };
    const logits = try gpa.alloc(u16, m.config.vocab);
    defer gpa.free(logits);
    var reply: std.ArrayList(u32) = .empty;
    defer reply.deinit(gpa);
    var timing: draft.Timing = .{};
    try draft.greedy(gpa, m, &state, &b, ids.items, @min(max, context - ids.items.len), logits, &reply, &timing);
    const end = cuda.monotonic();
    for (reply.items) |token| try stdout.print("{d} ", .{token});
    const ms = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    const steps = reply.items.len -| 1;
    try stdout.print("\nload {d:.0} ms, prompt {d} tokens {d:.1} ms, {d} tokens, {d:.1} tokens a second after the first, device {d} MB\n", .{
        ms(loaded - start),                                                                    ids.items.len,            ms(timing.first - loaded), reply.items.len,
        if (steps > 0) @as(f64, @floatFromInt(steps)) / (ms(end - timing.first) / 1e3) else 0, cuda.account.bytes >> 20,
    });
    return 0;
}

fn refuse(path: []const u8, what: []const u8) u8 {
    std.log.err("{s}: no {s}", .{ path, what });
    return 1;
}
