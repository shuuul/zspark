//! `ranks-check DIR reference=REF`: the whole model, one share a Spark (spec 050 WS-07). Both ranks load
//! every layer, run `primes` with each block fed the reference's input, and rank 0 then checks a window of
//! that prompt against its serial steps and the greedy replies against TensorFold's decode path. A two-rank
//! reply is not the one-rank reply: the fp32 partials are added and rounded once.
//!
//! Rank 0 loads the MTP head whole beside its share (spec 090), and each reply is decoded again with the head
//! drafting, greedy and sampled: the two-rank serial reply token for token, or the check fails. Rank 1 is
//! given the same options and loads nothing for the head. `drafter=DIR` (spec 126 WS-01) puts the stored
//! DFlash2 drafter in the head's place (`ranks_drafter.zig`), both ranks loading their own share of DIR.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const channel_file = @import("../../../ranks/channel.zig");
const cuda = @import("../../../cuda/cuda.zig");
const nccl = @import("../../../ranks/nccl.zig");
const sampling = @import("../../../decode/sampling.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const qwen4_config = @import("../config.zig");
const compare = @import("../compare.zig");
const forward_file = @import("../forward.zig");
const layout = @import("../layout.zig");
const model_file = @import("../model.zig");
const mtp = @import("../mtp.zig");
const held_file = @import("../ranks_forward.zig");
const rig = @import("../dflash2/rig.zig");
const ranks_drafter = @import("ranks_drafter.zig");
const pair_file = @import("../session/pair.zig");

const H = held_file.Held(1);
/// A reply's drafts a window at most, as `chat` and `serve` ask unless told.
const depth = @import("../../../decode/draft.zig").forward_max_rows - 1;
const Comparison = compare.Comparison;
const rows = 32;
const room = 128;
/// One rank's bound is 2e-3. A share's partials are summed and rounded once, and that gap accumulates over
/// the 48 layers: the measured worst on the published checkpoint is the branch, 3.2e-3.
const relative_bound = 4e-3;
const picks_bound = 8;
const pick_gap_bound = 1e-5;

const Case = struct {
    name: []const u8,
    /// How many of TensorFold's decode-path tokens the one-rank reply matches. A shorter prefix is a failure.
    least: usize,
};

const cases = [_]Case{
    .{ .name = "sky", .least = 64 },
    .{ .name = "primes", .least = 18 },
};

pub fn run(
    gpa: Allocator,
    io: std.Io,
    stdout: *std.Io.Writer,
    channel: *channel_file.Channel,
    comm: nccl.Comm,
    rank: usize,
    config: qwen4_config.Config,
    found: layout.Layout,
    layers: usize,
    reference_path: []const u8,
    given: ?*const rig.Opened,
) !u8 {
    var references = try std.Io.Dir.cwd().openDir(io, reference_path, .{});
    defer references.close(io);
    var reference: safetensors.File = try .open(gpa, io, references, "primes.serial.f32.safetensors");
    defer reference.close();
    const ids = reference.get("ids") orelse return error.MissingReference;
    const prompt = try gpa.alloc(i32, @intCast(ids.count()));
    defer gpa.free(prompt);
    @memcpy(std.mem.sliceAsBytes(prompt), ids.data);
    if (prompt.len > rows) return error.UnsupportedShape;

    const loading = cuda.monotonic();
    const draft_ids = try model_file.DraftHead.defaultIds(gpa);
    defer gpa.free(draft_ids);
    const built: H.Model = .{
        .ranks = .{try model_file.Model.loadWith(gpa, config, found, .{ .layers = layers, .max_rows = rows, .mtp = true, .draft_vocab = draft_ids, .share = .of(@intCast(rank)), .head_split = model_file.ranks_head == .split })},
        .first = rank,
        .max_rows = rows,
    };
    const pair = try pair_file.Pair.createScoring(gpa, rank, built, comm, channel, rows, rows, room, true);
    defer pair.destroy();
    // the drafter beside the share, both ranks loading their own of it (rank 1 replays the passes rank 0 tells)
    var drafter: ?*ranks_drafter.Given(1) = null;
    defer if (drafter) |d| d.deinit(gpa);
    if (given) |opened| {
        drafter = try ranks_drafter.Given(1).init(gpa, opened, pair, rig.drafter_stored, false);
        try stdout.print("rank {d}: DFlash2 beside the share, its taps after layers {any}\n", .{ rank, opened.config.taps });
        try stdout.flush();
    }
    try stdout.print("rank {d}: {d} layers loaded in {d:.1} s; {d:.2} GiB on the device, {s}\n", .{ rank, layers, units.seconds(cuda.monotonic() - loading), units.gib(cuda.account.bytes), if (pair.split != null) "half of the MTP head beside the share" else if (pair.head != null) "the MTP head whole beside the share" else "no MTP head" });
    // what `serve` would ask for the head's scratch at its defaults, with nothing made
    if (pair.head != null) try stdout.print("rank {d}: the head's scratch for 512 rows and 262144 positions would be {d:.2} GiB\n", .{ rank, units.gib(mtp.Buffers.bytesFor(pair.m.ranks[0], 512, 262144, false)) });
    try stdout.flush();

    const partial = blk: {
        var name: [96]u8 = undefined;
        break :blk reference.get(try Comparison.plain(&name, .h_in, layers)) != null;
    };
    var comparison: Comparison = .{ .reference = &reference, .slots = pair.m.ranks[0].shape.top + 1, .force = true, .partial = partial, .gpa = gpa };
    defer comparison.deinit();
    channel.barrier(10) catch |err| return gone(channel, rank, err);
    H.stageAll(pair.m, &pair.state, &pair.b, prompt) catch |err| return fail(channel, rank, err);
    H.launch(pair.m, &pair.state, &pair.b, prompt.len, null, .none, comparison.tap()) catch |err| return fail(channel, rank, err);
    channel.barrier(11) catch |err| return gone(channel, rank, err);
    if (bounds(stdout, rank, comparison)) |text| {
        channel.fail(text);
        return 1;
    }
    channel.barrier(12) catch |err| return gone(channel, rank, err);

    if (rank == 0) {
        if (window(gpa, pair, prompt)) |text| {
            std.log.err("rank 0: {s}", .{text});
            channel.fail(text);
            return 1;
        }
        try stdout.print("rank 0: the prompt's window matches its serial steps\n", .{});
        try stdout.flush();
        for (cases) |case| {
            if (greedy(gpa, io, stdout, pair, references, case, drafter)) |text| {
                std.log.err("rank 0: {s}", .{text});
                channel.fail(text);
                return 1;
            }
        }
        pair.finish() catch |err| return fail(channel, rank, err);
    } else {
        _ = pair.follow() catch |err| return fail(channel, rank, err);
        try stdout.print("rank 1: followed the window and the replies\n", .{});
        try stdout.flush();
    }
    channel.barrier(13) catch |err| return gone(channel, rank, err);
    try stdout.print("rank {d}: the whole model on two ranks is within the bounds\n", .{rank});
    return 0;
}

/// The points the two-rank launch hands the tap. A rank's own half (the projection, the block's output, the
/// logits) is not among them: the reference is the whole model's.
fn bounds(stdout: *std.Io.Writer, rank: usize, comparison: Comparison) ?[]const u8 {
    var worst: f64 = 0;
    var worst_point: forward_file.Point = .embed_h;
    var seen: usize = 0;
    var outside = false;
    for (std.enums.values(forward_file.Point)) |point| {
        const stats = comparison.stats.get(point);
        if (stats.layers == 0) continue;
        seen += 1;
        stdout.print("rank {d}: {t}: relative rms {e:.3} at layer {d}, max abs {e:.3}, {d} of {d} differ\n", .{ rank, point, stats.relative, stats.worst_layer, stats.max_abs, stats.mismatched, stats.elements }) catch return "the report was not written";
        if (point == .moe_picks) continue;
        if (stats.relative >= relative_bound) outside = true;
        if (stats.relative > worst) {
            worst = stats.relative;
            worst_point = point;
        }
    }
    if (seen == 0) return "no block was compared";
    if (comparison.picks_differing > picks_bound or comparison.widest_pick_gap >= pick_gap_bound) {
        std.log.err("rank {d}: {d} picks differ, widest gap {e:.3}", .{ rank, comparison.picks_differing, comparison.widest_pick_gap });
        return "the picks are outside the bound";
    }
    stdout.print("rank {d}: {d} points, worst relative rms {e:.3} ({t}); {d} picks differ, widest gap {e:.3}\n", .{ rank, seen, worst, worst_point, comparison.picks_differing, comparison.widest_pick_gap }) catch return "the report was not written";
    stdout.flush() catch return "the report was not written";
    if (outside) return "a block is outside the bound";
    return null;
}

/// The prompt as one window and as one token a step. The streams of a row are the serial step's.
pub fn window(gpa: Allocator, pair: *pair_file.Pair, prompt: []const i32) ?[]const u8 {
    pair.reset() catch |err| return oops(err);
    H.forwardScoring(pair.m, &pair.state, &pair.b, prompt, .none) catch |err| return oops(err);
    const saved = gpa.alloc(u64, prompt.len) catch return "out of memory";
    defer gpa.free(saved);
    for (saved, 0..) |*hash, row| hash.* = rowHash(gpa, pair.b.ranks[0].h, row) catch return "the window's streams were not read";
    pair.reset() catch |err| return oops(err);
    for (prompt, 0..) |token, row| {
        const step = [1]i32{token};
        H.forwardScoring(pair.m, &pair.state, &pair.b, &step, .none) catch |err| return oops(err);
        const got = rowHash(gpa, pair.b.ranks[0].h, 0) catch return "a serial step's streams were not read";
        if (got != saved[row]) {
            std.debug.print("rank 0: row {d} of the window is not the serial step\n", .{row});
            return "the window's streams are not the serial step's";
        }
        H.commit(pair.m, &pair.state, pair.b, &step, 1) catch |err| return oops(err);
    }
    return null;
}

fn greedy(gpa: Allocator, io: std.Io, stdout: *std.Io.Writer, pair: *pair_file.Pair, references: std.Io.Dir, case: Case, drafter: ?*ranks_drafter.Given(1)) ?[]const u8 {
    var name: [64]u8 = undefined;
    const file = std.fmt.bufPrint(&name, "{s}.f32.json", .{case.name}) catch return "the case name does not fit";
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const text = references.readFileAlloc(io, file, arena, .limited(1 << 24)) catch return "the greedy reference was not read";
    const summary = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{}) catch return "the greedy reference is not json";
    const prompt_ids = summary.object.get("ids").?.array.items;
    const expected_ids = summary.object.get("greedy_decode_path").?.array.items;
    const prompt = arena.alloc(i32, prompt_ids.len) catch return "out of memory";
    const expected = arena.alloc(i32, expected_ids.len) catch return "out of memory";
    for (prompt, prompt_ids) |*token, id| token.* = @intCast(id.integer);
    for (expected, expected_ids) |*token, id| token.* = @intCast(id.integer);
    pair.reset() catch |err| return oops(err);
    const before = pair.tokens.items.len;
    const settings: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 };
    const serial_start = cuda.monotonic();
    _ = pair.reply(prompt, settings, &.{}, .{}, .{}, null, expected.len, null, null) catch |err| return oops(err);
    const serial_s = units.seconds(cuda.monotonic() - serial_start);
    const got = arena.dupe(i32, pair.tokens.items[before..]) catch return "out of memory";
    if (pair.head != null or drafter != null) {
        const sampled: sampling.Sampling = .{ .seed = 11, .temperature = 0.8, .top_k = 40, .top_p = 0.95, .min_p = 0 };
        const whom = if (drafter != null) "DFlash2" else "the MTP head";
        for ([_]sampling.Sampling{ settings, sampled }) |each| {
            var serial: []const i32 = got;
            if (!each.greedy()) {
                pair.reset() catch |err| return oops(err);
                const from = pair.tokens.items.len;
                _ = pair.reply(prompt, each, &.{}, .{}, .{}, null, expected.len, null, null) catch |err| return oops(err);
                serial = arena.dupe(i32, pair.tokens.items[from..]) catch return "out of memory";
            }
            pair.reset() catch |err| return oops(err);
            const from = pair.tokens.items.len;
            const start = cuda.monotonic();
            // the drafter's context from empty, as the pair's reply is (`Drafter.reset`)
            if (drafter) |d| d.flash.d.reset(each);
            const by: pair_file.Drafts = if (drafter) |d| .{ .given = d.flash.d.drafter() } else .{ .depth = depth };
            const done = pair.reply(prompt, each, &.{}, by, .{}, null, expected.len, null, null) catch |err| return oops(err);
            const seconds = units.seconds(cuda.monotonic() - start);
            const drafted = pair.tokens.items[from..];
            const how = if (each.greedy()) "greedy" else "sampled";
            if (!std.mem.eql(i32, drafted, serial)) {
                std.debug.print("rank 0: {s} {s}, the drafted reply is not the serial one: {d} tokens against {d}\n", .{ case.name, how, drafted.len, serial.len });
                return "the drafted reply is not the two ranks' serial one";
            }
            stdout.print("rank 0: {s} {s}, drafted by {s}, equals serial: {d} tokens in {d} rounds, {d:.2} tokens a round, {d} of {d} drafts kept; prompt and reply in {d:.2} s", .{ case.name, how, whom, drafted.len, done.stats.rounds, @as(f64, @floatFromInt(drafted.len)) / @as(f64, @floatFromInt(@max(done.stats.rounds, 1))), done.stats.accepted, done.stats.drafted, seconds }) catch return "the report was not written";
            if (each.greedy()) stdout.print(" (serial {d:.2} s; a launch a kernel, no timing)", .{serial_s}) catch return "the report was not written";
            stdout.print("\n", .{}) catch return "the report was not written";
            stdout.flush() catch return "the report was not written";
        }
    }
    var same: usize = 0;
    while (same < got.len and same < expected.len and got[same] == expected[same]) same += 1;
    if (same < got.len and same < expected.len) {
        std.debug.print("rank 0: {s} greedy, the first {d} of {d} tokens are TensorFold's decode path's (one rank matches {d}); token {d} is {d} here and {d} there. Two ranks add the fp32 partials and round once.\n", .{ case.name, same, expected.len, case.least, same, got[same], expected[same] });
    } else std.debug.print("rank 0: {s} greedy, all {d} tokens are TensorFold's decode path's\n", .{ case.name, same });
    return null;
}

fn rowHash(gpa: Allocator, buffer: cuda.Buffer, row: usize) !u64 {
    const width = buffer.bytes() / @as(usize, @intCast(buffer.view.size[0]));
    const bytes = try gpa.alloc(u8, width);
    defer gpa.free(bytes);
    try buffer.downloadAt(row * width, bytes);
    return std.hash.Wyhash.hash(0, bytes);
}

pub fn oops(err: anyerror) []const u8 {
    std.log.err("the whole model: {t}", .{err});
    return "the whole model failed";
}

pub fn fail(channel: *channel_file.Channel, rank: usize, err: anyerror) u8 {
    std.log.err("rank {d}: {t} {s}", .{ rank, err, channel.said() });
    channel.fail("the whole model failed");
    return 1;
}

pub fn gone(channel: *channel_file.Channel, rank: usize, err: channel_file.Error) u8 {
    if (err == error.PeerFailed) std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
    return 1;
}
