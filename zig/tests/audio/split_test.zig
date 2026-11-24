//! `zig/src/audio/split.zig` against vLLM's `split_audio` on numpy float32 (`fixtures/audio/split_cases.json`,
//! written by `tools/reference/audio_split_cases.py`, spec 142 WS-06): the pieces' ends of a generated signal,
//! made again here from the same integers (a quiet stride taken, strides that tie at zero, a quieter last stride
//! never read), and of every clip of Qwen3-ASR's WER set past 30 s where `ZSPARK_REFERENCE_QWEN3_ASR` names
//! its directory.

const std = @import("std");
const testing = std.testing;
const audio = @import("../../src/audio/audio.zig");
const split = audio.split;

const cases_json = @embedFile("../fixtures/audio/split_cases.json");

const Cases = struct {
    max_seconds: f64,
    overlap_seconds: f64,
    window: usize,
    synthetic: []const usize,
    clips: std.json.ArrayHashMap([]const usize),
};

/// `audio_split_cases.py`'s `synthetic`: 95 s of a 64-bit LCG's high 16 bits, quieter or zero over its spans.
fn synthetic(a: std.mem.Allocator) ![]f32 {
    const n = 95 * 16000;
    const v = try a.alloc(i64, n);
    var s: u64 = 0x2545F4914F6CDD1D;
    for (v) |*x| {
        s = s *% 6364136223846793005 +% 1442695040888963407;
        x.* = @as(i64, @intCast((s >> 48) & 0xFFFF)) - 32768;
    }
    const spans = [_][3]usize{ .{ 468800, 470400, 8 }, .{ 473600, 475200, 6 }, .{ 935000, 950000, 0 }, .{ 1408000, 1409600, 2 }, .{ 1416000, 1417600, 10 } };
    for (spans) |span| {
        for (v[span[0]..span[1]]) |*x| x.* = if (span[2] == 0) 0 else x.* >> @intCast(span[2]);
    }
    const out = try a.alloc(f32, n);
    for (out, v) |*o, x| o.* = @as(f32, @floatFromInt(x)) / 32768.0;
    return out;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

test "split: a long clip's pieces end where vLLM's split_audio ends them" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = try std.json.parseFromSliceLeaky(Cases, a, cases_json, .{});
    try testing.expectEqual(@as(f64, split.overlap_seconds), cases.overlap_seconds);
    try testing.expectEqual(@as(usize, split.energy_window), cases.window);

    const signal = try synthetic(a);
    try testing.expectEqualSlices(usize, cases.synthetic, try split.cuts(a, signal, audio.rate, cases.max_seconds, cases.overlap_seconds, cases.window));
    // 30 s or less is one piece
    try testing.expectEqualSlices(usize, &.{480000}, try split.cuts(a, signal[0..480000], audio.rate, 30, 1, 1600));

    const dir = env("ZSPARK_REFERENCE_QWEN3_ASR") orelse return;
    var it = cases.clips.map.iterator();
    while (it.next()) |entry| {
        errdefer std.debug.print("clip {s}\n", .{entry.key_ptr.*});
        const path = try std.fmt.allocPrint(a, "{s}/wer/clips/{s}.flac", .{ dir, entry.key_ptr.* });
        const bytes = try std.Io.Dir.cwd().readFileAlloc(testing.io, path, a, .limited(64 << 20));
        const clip = try audio.clip(a, bytes, 1200);
        try testing.expectEqualSlices(usize, entry.value_ptr.*, try split.cuts(a, clip.samples, audio.rate, cases.max_seconds, cases.overlap_seconds, cases.window));
    }
}
