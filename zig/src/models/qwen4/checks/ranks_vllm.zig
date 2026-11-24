//! `ranks-check DIR reference=REF` where REF is the vLLM container's (spec 105 WS-04: `NAME.json` and
//! `score.json` from `tools/reference/vllm_reference.py`, no TensorFold export): the whole model, one share a
//! Spark, held to vLLM as one Spark is (`vllm_reference.zig`, `compare_vllm_test.zig`). Rank 0 checks a prompt's
//! window against its serial steps; each case's greedy reply, drafted by the MTP head as `chat` drafts, against
//! the two ranks' serial reply bit for bit, and a sampled reply the same way; the greedy reply against vLLM's,
//! leaving it only where the two ranks' serial choice along vLLM's own tokens leaves it, at a tie of vLLM's
//! (`tie`); then the score text in windows of 16, its losses against vLLM's two runs. Rank 1 follows.
//! `drafter=DIR` (spec 126 WS-01) puts the stored DFlash2 drafter in the head's place (`ranks_drafter.zig`),
//! both ranks loading their own share of DIR: the drafted replies are still the serial ones, bit for bit.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const channel_file = @import("../../../ranks/channel.zig");
const cuda = @import("../../../cuda/cuda.zig");
const nccl = @import("../../../ranks/nccl.zig");
const sampling = @import("../../../decode/sampling.zig");
const qwen4_config = @import("../config.zig");
const layout = @import("../layout.zig");
const model_file = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const rig = @import("../dflash2/rig.zig");
const ranks_drafter = @import("ranks_drafter.zig");
const pair_file = @import("../session/pair.zig");
const ranks_whole = @import("ranks_whole.zig");
const vllm = @import("vllm_reference.zig");

const H = held_file.Held(1);
/// A reply's drafts a window at most, as `chat` and `serve` ask unless told.
const depth = @import("../../../decode/draft.zig").forward_max_rows - 1;
const rows = 32;
/// The score text's 5,336 tokens, and room for the longest reply after its prompt. Both ranks make the same.
const capacity = 8192;
/// The score text's window, as `zspark score` and one Spark's test run it.
const score_window = 16;

const greedy_settings = vllm.greedy;
const sampled_settings = vllm.sampled;

/// Whether `reference_path` holds vLLM's files rather than TensorFold's export.
pub fn holds(io: std.Io, reference_path: []const u8) bool {
    return vllm.holds(io, reference_path, "primes.serial.f32.safetensors");
}

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
    const loading = cuda.monotonic();
    const draft_ids = try model_file.DraftHead.defaultIds(gpa);
    defer gpa.free(draft_ids);
    const built: H.Model = .{
        .ranks = .{try model_file.Model.loadWith(gpa, config, found, .{ .layers = layers, .max_rows = rows, .mtp = given == null, .draft_vocab = draft_ids, .share = .of(@intCast(rank)), .head_split = model_file.ranks_head == .split })},
        .first = rank,
        .max_rows = rows,
    };
    const pair = try pair_file.Pair.createScoring(gpa, rank, built, comm, channel, rows, rows, capacity, true);
    defer pair.destroy();
    // the drafter beside the share, both ranks loading their own of it (rank 1 replays the passes rank 0 tells)
    var drafter: ?*ranks_drafter.Given(1) = null;
    defer if (drafter) |d| d.deinit(gpa);
    if (given) |opened| {
        drafter = try ranks_drafter.Given(1).init(gpa, opened, pair, rig.drafter_stored, false);
        try stdout.print("rank {d}: DFlash2 beside the share, {d:.2} GiB on the device with it, its taps after layers {any}\n", .{ rank, units.gib(cuda.account.bytes), opened.config.taps });
        try stdout.flush();
    }
    try stdout.print("rank {d}: {d} layers loaded in {d:.1} s; {d:.2} GiB on the device, {s}\n", .{ rank, layers, units.seconds(cuda.monotonic() - loading), units.gib(cuda.account.bytes), if (pair.split != null) "half of the MTP head beside the share" else if (pair.head != null) "the MTP head whole beside the share" else "no MTP head" });
    try stdout.flush();
    channel.barrier(10) catch |err| return ranks_whole.gone(channel, rank, err);

    if (rank == 0) {
        if (checks(gpa, io, stdout, pair, references, drafter)) |text| {
            std.log.err("rank 0: {s}", .{text});
            channel.fail(text);
            return 1;
        }
        pair.finish() catch |err| return ranks_whole.fail(channel, rank, err);
    } else {
        _ = pair.follow() catch |err| return ranks_whole.fail(channel, rank, err);
        try stdout.print("rank 1: followed the window, the replies and the score text\n", .{});
        try stdout.flush();
    }
    channel.barrier(13) catch |err| return ranks_whole.gone(channel, rank, err);
    try stdout.print("rank {d}: the whole model on two ranks is within vLLM's bounds\n", .{rank});
    return 0;
}

fn checks(gpa: Allocator, io: std.Io, stdout: *std.Io.Writer, pair: *pair_file.Pair, references: std.Io.Dir, drafter: ?*ranks_drafter.Given(1)) ?[]const u8 {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var first_prompt: ?[]const i32 = null;
    for (vllm.cases) |case| {
        var name_buffer: [64]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buffer, "{s}.json", .{case.name}) catch return "the case name does not fit";
        const text = references.readFileAlloc(io, name, arena, .limited(1 << 26)) catch return "vLLM's reply was not read";
        const theirs = vllm.reply(arena, text) catch return "vLLM's reply is not its json";
        if (first_prompt == null) {
            first_prompt = theirs.prompt;
            if (ranks_whole.window(gpa, pair, theirs.prompt)) |failed| return failed;
            stdout.print("rank 0: {s}'s prompt as one window is its serial steps\n", .{case.name}) catch return "the report was not written";
        }
        if (reply(arena, stdout, pair, case, theirs, drafter)) |failed| return failed;
    }
    return score(arena, io, stdout, pair, references);
}

/// A case: drafted greedy and sampled replies their serial ones, the greedy one against vLLM's.
fn reply(arena: Allocator, stdout: *std.Io.Writer, pair: *pair_file.Pair, case: vllm.Case, theirs: vllm.Reply, drafter: ?*ranks_drafter.Given(1)) ?[]const u8 {
    const expected = theirs.runs[0].tokens;
    if (theirs.prompt.len > rows or expected.len < case.least) return "vLLM's case does not fit the check";
    // vLLM's two runs agree, or its own reply is not one to hold zspark to
    for (theirs.runs[1..]) |other| if (!std.mem.eql(i32, other.tokens, expected)) return "vLLM's runs of a case differ";
    const whom = if (drafter != null) "DFlash2" else "the MTP head";
    var greedy: []const i32 = &.{};
    for ([_]sampling.Sampling{ greedy_settings, sampled_settings }) |settings| {
        const serial = once(arena, pair, theirs.prompt, settings, .{}, expected.len) catch |err| return ranks_whole.oops(err);
        // the drafter's context from empty, as the pair's reply is (`Drafter.reset`)
        if (drafter) |d| d.flash.d.reset(settings);
        // DFlash2 drafts in the head's place, as `serve --drafter` has it: no MTP head loaded, the prompt told
        // once and rank 1's drafter absorbing its windows (spec 135 WS-02)
        const by: pair_file.Drafts = .{ .depth = depth };
        const start = cuda.monotonic();
        const drafted = once(arena, pair, theirs.prompt, settings, by, expected.len) catch |err| return ranks_whole.oops(err);
        const seconds = units.seconds(cuda.monotonic() - start);
        const how = if (settings.greedy()) "greedy" else "sampled";
        if (!std.mem.eql(i32, drafted.tokens, serial.tokens)) {
            std.debug.print("rank 0: {s} {s}, the drafted reply is not the serial one: {d} tokens against {d}\n", .{ case.name, how, drafted.tokens.len, serial.tokens.len });
            return "the drafted reply is not the two ranks' serial one";
        }
        stdout.print("rank 0: {s} {s}, drafted by {s}, equals serial: {d} tokens in {d} rounds, {d} of {d} drafts kept; prompt and reply in {d:.2} s\n", .{ case.name, how, whom, drafted.tokens.len, drafted.rounds, drafted.accepted, drafted.drafted, seconds }) catch return "the report was not written";
        if (settings.greedy()) greedy = serial.tokens;
    }
    // along vLLM's reply: the two ranks leave it where their serial choice after vLLM's tokens does
    const along = forced(pair, theirs.prompt, expected) catch |err| return ranks_whole.oops(err);
    var same: usize = 0;
    while (same < greedy.len and same < expected.len and greedy[same] == expected[same]) same += 1;
    stdout.print("rank 0: vLLM {s}: the first {d} of {d} tokens are vLLM's; after its tokens {d} of {d} choices are the two ranks' own (one Spark: at least {d})\n", .{ case.name, same, greedy.len, along.agree, expected.len, case.least }) catch return "the report was not written";
    if ((along.first orelse greedy.len) != same) return "the reply leaves vLLM's elsewhere than the serial choice along vLLM's tokens";
    // the two ranks' bits are not one Spark's (a partial a rank, added in rank order): they may leave
    // vLLM's reply before one Spark does, but only at one of vLLM's ties
    if (along.first) |at| {
        const want = vllm.logprobOf(theirs, at, expected[at]) orelse return "vLLM's token is not among its own top";
        const mine = vllm.logprobOf(theirs, at, greedy[at]) orelse return "the two ranks' token is not among vLLM's top";
        stdout.print("rank 0: vLLM {s}: at token {d} vLLM's {d} at {d:.4}, the two ranks' {d} at {d:.4}\n", .{ case.name, at, expected[at], want, greedy[at], mine }) catch return "the report was not written";
        if (want - mine > vllm.tie) return "the reply leaves vLLM's at no tie of vLLM's";
    } else if (same < case.least) return "the reply leaves vLLM's earlier than one Spark's";
    stdout.flush() catch return "the report was not written";
    return null;
}

const Once = struct { tokens: []const i32, rounds: usize, accepted: usize, drafted: usize };

fn once(arena: Allocator, pair: *pair_file.Pair, prompt: []const i32, settings: sampling.Sampling, by: pair_file.Drafts, max_tokens: usize) !Once {
    try pair.reset();
    const from = pair.tokens.items.len;
    const done = try pair.reply(prompt, settings, &.{}, by, .{}, null, max_tokens, null, null);
    return .{ .tokens = try arena.dupe(i32, pair.tokens.items[from..]), .rounds = done.stats.rounds, .accepted = done.stats.accepted, .drafted = done.stats.drafted };
}

const Forced = struct { agree: usize = 0, first: ?usize = null };

/// The two ranks' greedy choice at each of vLLM's tokens, fed vLLM's tokens before it: a step a token.
fn forced(pair: *pair_file.Pair, prompt: []const i32, theirs: []const i32) !Forced {
    try pair.reset();
    try H.forwardScoring(pair.m, &pair.state, &pair.b, prompt, .last);
    try H.logitsRow(pair.b, 0, pair.logits);
    try H.commit(pair.m, &pair.state, pair.b, prompt, prompt.len);
    var result: Forced = .{};
    for (theirs, 0..) |want, i| {
        const token = try pair.sampler.choose(pair.logits, prompt.len + i, greedy_settings);
        if (token == want) result.agree += 1 else if (result.first == null) result.first = i;
        if (i + 1 == theirs.len) break;
        const step = [1]i32{want};
        try H.forwardScoring(pair.m, &pair.state, &pair.b, &step, .last);
        try H.logitsRow(pair.b, 0, pair.logits);
        try H.commit(pair.m, &pair.state, pair.b, &step, 1);
    }
    return result;
}

/// The score text in windows of `score_window`, every row's logits gathered, against vLLM's prompt logprobs.
fn score(arena: Allocator, io: std.Io, stdout: *std.Io.Writer, pair: *pair_file.Pair, references: std.Io.Dir) ?[]const u8 {
    const text = references.readFileAlloc(io, "score.json", arena, .limited(1 << 26)) catch return "vLLM's score was not read";
    const scored = vllm.scored(arena, text) catch return "vLLM's score is not its json";
    if (scored.runs.len < 2 or scored.prompt.len > capacity) return "vLLM's score does not fit the check";
    const tokens = scored.prompt;
    pair.reset() catch |err| return ranks_whole.oops(err);
    var sums: vllm.Sums = .{};
    var at: usize = 0;
    while (at < tokens.len) {
        const count = @min(score_window, tokens.len - at);
        const chunk = tokens[at..][0..count];
        H.forwardScoring(pair.m, &pair.state, &pair.b, chunk, .all) catch |err| return ranks_whole.oops(err);
        for (0..count) |r| {
            const next = at + r + 1;
            if (next >= tokens.len) break;
            H.logitsRow(pair.b, r, pair.logits) catch |err| return ranks_whole.oops(err);
            sums.add(scored, pair.logits, next);
        }
        H.commit(pair.m, &pair.state, pair.b, chunk, count) catch |err| return ranks_whole.oops(err);
        at += count;
    }
    stdout.print("rank 0: ", .{}) catch return "the report was not written";
    sums.print(stdout) catch return "the report was not written";
    stdout.flush() catch return "the report was not written";
    if (!sums.within(vllm.nll_apart)) return "the score text is outside vLLM's bounds";
    return null;
}
