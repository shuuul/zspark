//! `chain.zig` against TensorFold's `Drafter.chain` on recorded candidates (spec 067 WS-13):
//! `tools/reference/glm5_drafter_export.py` writes `NAME.chain.json`, the published drafter's merged
//! candidates and projected rows on made taps and the picks of each chain, greedy and sampled, under
//! confidences, from two anchors and two first positions. The codebooks are read from the drafter's own
//! directory and nothing of them is in the file. Needs `ZSPARK_GLM_DRAFTER` and
//! `ZSPARK_REFERENCE_GLM_DRAFTER`; skipped without. No device.

const std = @import("std");
const testing = std.testing;
const safetensors = @import("../../../src/weights/safetensors.zig");
const chain = @import("../../../src/decode/dflash2/chain.zig");

const Recorded = struct {
    top: usize,
    rank: usize,
    tokens: []const []const u32,
    values: []const []const f64,
    projected: []const []const f64,
    chains: []const struct {
        anchor: u32,
        first: u64,
        seed: ?u64,
        temperature: ?f64,
        confidence: f64,
        picks: []const u32,
    },
};

/// A codebook as the chain reads it: bf16 [vocab, rank] widened to fp32.
fn widened(arena: std.mem.Allocator, drafter: *const safetensors.Checkpoint, name: []const u8) ![]f32 {
    const tensor = drafter.get(name) orelse return error.MissingTensor;
    try testing.expect(tensor.dtype == .BF16);
    const out = try arena.alloc(f32, tensor.data.len / 2);
    for (out, 0..) |*value, at| value.* = @bitCast(@as(u32, std.mem.readInt(u16, tensor.data[at * 2 ..][0..2], .little)) << 16);
    return out;
}

test "DFlash2's chains on TensorFold's recorded candidates are TensorFold's picks, greedy and sampled" {
    const path = testing.environ.getPosix("ZSPARK_GLM_DRAFTER") orelse return error.SkipZigTest;
    const reference = testing.environ.getPosix("ZSPARK_REFERENCE_GLM_DRAFTER") orelse return error.SkipZigTest;
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var references = try std.Io.Dir.cwd().openDir(io, reference, .{});
    defer references.close(io);
    var drafter: safetensors.Checkpoint = try .open(testing.allocator, io, .cwd(), path);
    defer drafter.close();
    var books: chain.Books = .{
        .predecessor = try widened(arena, &drafter, "candidate_selector.predecessor_codebook"),
        .successor = try widened(arena, &drafter, "candidate_selector.successor_codebook"),
        .rank = 0,
    };

    var chains: usize = 0;
    var sampled: usize = 0;
    var cut: usize = 0;
    var distinct: std.AutoArrayHashMapUnmanaged(u64, void) = .empty;
    for ([_][]const u8{ "short.chain.json", "long.chain.json" }) |name| {
        const text = references.readFileAlloc(io, name, arena, .limited(1 << 26)) catch |err| switch (err) {
            error.FileNotFound => return error.SkipZigTest,
            else => return err,
        };
        const r = try std.json.parseFromSliceLeaky(Recorded, arena, text, .{ .ignore_unknown_fields = true });
        books.rank = r.rank;
        const depth = r.tokens.len;
        const tokens = try arena.alloc(u32, depth * r.top);
        const values = try arena.alloc(f64, depth * r.top);
        const projected = try arena.alloc(f64, depth * r.rank);
        for (r.tokens, r.values, r.projected, 0..) |t, v, p, d| {
            @memcpy(tokens[d * r.top ..][0..r.top], t);
            @memcpy(values[d * r.top ..][0..r.top], v);
            @memcpy(projected[d * r.rank ..][0..r.rank], p);
        }
        const c: chain.Candidates = .{ .tokens = tokens, .values = values, .projected = projected, .top = r.top };
        const out = try arena.alloc(u32, depth);
        for (r.chains, 0..) |one, index| {
            const noise: ?chain.Noise = if (one.seed) |seed| .{ .seed = seed, .temperature = one.temperature.? } else null;
            const kept = chain.choose(books, c, one.anchor, one.first, noise, one.confidence, false, out);
            testing.expectEqualSlices(u32, one.picks, out[0..kept]) catch |err| {
                std.debug.print("{s}, chain {d}: anchor {d}, first {d}, seed {?d}, temperature {?d}, confidence {d}\n", .{ name, index, one.anchor, one.first, one.seed, one.temperature, one.confidence });
                return err;
            };
            chains += 1;
            sampled += @intFromBool(noise != null);
            cut += @intFromBool(kept < depth);
            try distinct.put(arena, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(out[0..kept])), {});
        }
    }
    std.debug.print("DFlash2's chain against TensorFold's: {d} chains ({d} sampled, {d} ended by the confidence), {d} distinct\n", .{ chains, sampled, cut, distinct.count() });
    // the recording says something: sampled chains, chains the confidence ends, and more than a few outcomes
    try testing.expect(sampled > 0 and cut > 0 and cut < chains and distinct.count() >= 16);
}
