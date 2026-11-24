//! `zspark generate`: a prompt's ids through the whole model and N tokens decoded, timed; with the MTP head's
//! drafts, DFlash2 (`drafter=DIR`), serially, or many replies on one load.

const std = @import("std");
const qwen4_layout = @import("../layout.zig");
const cuda = @import("../../../cuda/cuda.zig");
const qwen4_attention = @import("../blocks/attention.zig");
const qwen4_draft = @import("../draft.zig");
const decode_draft = @import("../../../decode/draft.zig");
const copied = @import("../../../decode/copied.zig");
const qwen4_forward = @import("../forward.zig");
const qwen4_model = @import("../model.zig");
const qwen4_mtp = @import("../mtp.zig");
const rig = @import("../dflash2/rig.zig");
const pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const sampling = @import("../../../decode/sampling.zig");
const cli = @import("../../../cli.zig");
const join_file = @import("../../../ranks/join.zig");
const common = @import("common.zig");
const generate_ranks = @import("generate_ranks.zig");
const score = @import("score.zig");

/// Runs the prompt `ids` through the whole model in windows, then decodes `count` tokens greedily, one a step,
/// past end tokens. Prints each prompt row's most likely next token, then the decoded ids and their rate.
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const count = try std.fmt.parseInt(usize, argv[1], 10);
    const arguments = argv[2..];
    const arena = init.arena.allocator();
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, path, .raise)) return 1;
    // `rank=R at=ADDR:PORT`: a rank a Spark (`generate_ranks.zig`)
    var place: join_file.Place = .{};
    var rest: std.ArrayList([]const u8) = .empty;
    for (arguments) |word| {
        if (!place.take(word)) try rest.append(arena, word);
    }
    if (place.rank != null or place.at != null) {
        if (!place.complete()) return error.InvalidArgument;
        return generate_ranks.generate(init, stdout, &opened, place, count, rest.items);
    }
    const config = opened.config;
    const found = opened.layout;
    // `rows=N` first: buffers and scratch for windows of N rows, as a session's are for a prompt's (512), where
    // 16 is this command's own
    const wider = if (arguments.len > 0) cli.value(arguments[0], "rows=") else null;
    const widened = wider != null;
    const window = if (wider) |given| try std.fmt.parseInt(usize, given, 10) else 16;
    if (window < 16) return error.InvalidArgument;
    // then, before the storage and the mode: `rounds`, a line a round with the head's chain, and `follow=FILE`, a
    // reply's token ids that take the place of the model's choices (spec 028: two programs' drafts on one text)
    var ids = arguments[@intFromBool(widened)..];
    var follow: ?[]const i32 = null;
    var drafting: Drafting = .{};
    // `cases=FILE`: many replies on one load (spec 037). Each line of FILE is a reply's own `N [OPTION..] ID..`,
    // with `name=LABEL` among the options; the state and the head start again before each. `score=FILE` scores
    // that file's ids after the cases, on this load (spec 049); `score-each=FILE` with each token's loss too.
    var cases: ?[]const u8 = null;
    var score_path: ?[]const u8 = null;
    var score_each = false;
    while (ids.len > 0) : (ids = ids[1..]) {
        if (cli.value(ids[0], "cases=")) |file| {
            cases = try std.Io.Dir.cwd().readFileAlloc(init.io, file, arena, .unlimited);
        } else if (cli.value(ids[0], "score=")) |file| {
            score_path = file;
        } else if (cli.value(ids[0], "score-each=")) |file| {
            score_path = file;
            score_each = true;
        } else if (!try drafting.take(init.io, arena, ids[0], &follow)) break;
    }
    if (cases == null and ids.len == 0) return error.InvalidArgument;
    const named = if (ids.len > 0) qwen4_attention.Storage.parse(ids[0]) else null;
    const caches = qwen4_layout.storageFor(config, 1, 1, named);
    const mode = if (ids.len > @intFromBool(named != null)) ids[@intFromBool(named != null)] else "";
    const eager = std.mem.eql(u8, mode, "eager");
    const serial = eager or std.mem.eql(u8, mode, "serial");
    const lazy = std.mem.eql(u8, mode, "lazy");
    if (cases != null and serial) return error.InvalidArgument;
    const prompt = try cli.ids(arena, ids[@as(usize, @intFromBool(serial or lazy)) + @intFromBool(named != null) ..]);

    var d_opened: rig.Opened = .{};
    defer d_opened.close(init.io);
    if (drafting.drafter) |drafter_dir| if (!try rig.enter(init, &d_opened, drafter_dir, &config, config.layers)) return 1;

    const load_start: std.Io.Timestamp = .now(init.io, .awake);
    var model: qwen4_model.Model = try .loadWith(init.gpa, config, found, try common.loadOptions(arena, config, window, !serial, caches));
    defer model.deinit();
    var state: qwen4_forward.State = try .init(init.gpa, model, window, drafting.context);
    defer state.deinit();
    var buffers: qwen4_forward.Buffers = try .init(init.gpa, model, window, drafting.context);
    defer buffers.deinit();
    var graphs: ?qwen4_forward.Graphs = if (eager) null else try .init(init.gpa);
    defer if (graphs) |*g| g.deinit();
    const vocab = model.shape.vocab;
    if (cases) |text| {
        var report_load = true;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            var words = std.mem.tokenizeAny(u8, line, " \t\r");
            const first = words.next() orelse continue;
            if (first[0] == '#') continue;
            var one = drafting;
            var followed = follow;
            var tokens: std.ArrayList(i32) = .empty;
            while (words.next()) |word| {
                if (tokens.items.len == 0 and try one.take(init.io, arena, word, &followed)) continue;
                try tokens.append(arena, try std.fmt.parseInt(i32, word, 10));
            }
            try stdout.print("== {s}\n", .{one.name});
            try stdout.flush();
            try state.reset();
            const count_case = try std.fmt.parseInt(usize, first, 10);
            const status = if (one.plain)
                try generatePlain(init, stdout, model, &state, &buffers, &graphs.?, tokens.items, count_case, window)
            else
                try generateDrafted(init, stdout, model, &state, &buffers, &graphs.?, tokens.items, count_case, !lazy, followed, if (report_load) load_start else null, one);
            report_load = false;
            if (status != 0) return status;
            try stdout.flush();
        }
        if (score_path) |file| _ = try score.scoreFile(init, stdout, model, file, score_each);
        return 0;
    }
    if (drafting.drafter != null and !std.mem.eql(u8, mode, "serial")) return generateFlash(init, stdout, model, &state, &buffers, if (graphs) |*g| g else null, prompt, count, !lazy and !eager, follow, null, drafting, &d_opened);
    if (model.mtp != null) return generateDrafted(init, stdout, model, &state, &buffers, &graphs.?, prompt, count, !lazy, follow, null, drafting);

    try stdout.writeAll("prompt argmax:");
    var next: i32 = 0;
    var at: usize = 0;
    while (at < prompt.len) {
        const rows = @min(window, prompt.len - at);
        try qwen4_forward.forward(model, &state, &buffers, prompt[at..][0..rows]);
        for (0..rows) |r| {
            next = try qwen4_forward.argmax(init.gpa, buffers, vocab, r);
            try stdout.print(" {d}", .{next});
        }
        try qwen4_forward.commit(model, &state, buffers, prompt[at..][0..rows], rows);
        at += rows;
    }
    try stdout.writeAll("\ngreedy:");
    try stdout.flush();
    const start: std.Io.Timestamp = .now(init.io, .awake);
    const margins = try arena.alloc(f32, count);
    for (margins) |*margin| {
        try stdout.print(" {d}", .{next});
        const step = [1]i32{next};
        if (graphs) |*g| {
            try g.forward(model, &state, &buffers, &step);
        } else try qwen4_forward.forward(model, &state, &buffers, &step);
        const best = try qwen4_forward.top(init.gpa, buffers, vocab, 0);
        next = best.token;
        margin.* = best.margin;
        try qwen4_forward.commit(model, &state, buffers, &step, 1);
    }
    try stdout.writeAll("\nmargin of the token after each:");
    for (margins) |margin| try stdout.print(" {d:.2}", .{margin});
    const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("\n{d} tokens in {d:.2} s: {d:.1} tok/s (serial, {s}, argmax on the host)\n", .{ count, seconds, @as(f64, @floatFromInt(count)) / seconds, if (eager) "one launch a kernel" else "a captured graph a step" });
    return 0;
}
/// What `generate` is told of its drafts before the mode.
const Drafting = struct {
    /// `rounds`: a line a round with the head's chain.
    listed: bool = false,
    /// `chains`: no draft is verified. Every round is one row, and the head's chain from that row is printed
    /// to the window's depth whatever it gives each draft, so that any rule for ending a chain can be played
    /// over the reply afterwards (`tools/reference/draft_policy.py`).
    chains: bool = false,
    depth: usize = common.draft_depth,
    confidence: ?f32 = null,
    /// `sample=SEED,TEMPERATURE,TOP_K,TOP_P`: the reply is sampled with these, not greedy.
    sampling: sampling.Sampling = .{ .seed = 0, .temperature = 0, .top_k = 0, .top_p = 1, .min_p = 0 },
    /// `name=LABEL`: what a case of `cases=FILE` is printed under.
    name: []const u8 = "",
    /// `plain`: one token a step, the head not asked. The graphs a drafted case prepared are replayed.
    plain: bool = false,
    /// `copies`: drafts copied from the context where the head begins the same (`decode/copied.zig`).
    copies: bool = false,
    context: usize = 4096,
    /// `drafter=DIR`: DFlash2 in the MTP head's place (`dflash2/rig.zig`).
    drafter: ?[]const u8 = null,
    /// The drafter's linears BF16 as the file has them.
    drafter_stored: bool = false,

    /// Whether `word` is one of the options, which is then taken; `follow=FILE`'s tokens go to `follow`.
    fn take(self: *Drafting, io: std.Io, arena: std.mem.Allocator, word: []const u8, follow: *?[]const i32) !bool {
        if (std.mem.eql(u8, word, "rounds")) {
            self.listed = true;
        } else if (std.mem.eql(u8, word, "chains")) {
            self.chains = true;
        } else if (cli.value(word, "depth=")) |given| {
            self.depth = try std.fmt.parseInt(usize, given, 10);
            if (self.depth >= qwen4_draft.forward_max_rows) return error.InvalidArgument;
        } else if (cli.value(word, "confidence=")) |given| {
            self.confidence = try std.fmt.parseFloat(f32, given);
        } else if (cli.value(word, "context=")) |given| {
            self.context = try std.fmt.parseInt(usize, given, 10);
            if (self.context < 512 or self.context % 512 != 0) return error.InvalidArgument;
        } else if (cli.value(word, "name=")) |given| {
            self.name = try arena.dupe(u8, given);
        } else if (std.mem.eql(u8, word, "copies")) {
            self.copies = true;
        } else if (std.mem.eql(u8, word, "plain")) {
            self.plain = true;
        } else if (cli.value(word, "sample=")) |given| {
            var parts = std.mem.splitScalar(u8, given, ',');
            self.sampling.seed = try std.fmt.parseInt(@TypeOf(self.sampling.seed), parts.next() orelse return error.InvalidArgument, 10);
            self.sampling.temperature = try std.fmt.parseFloat(f64, parts.next() orelse return error.InvalidArgument);
            self.sampling.top_k = try std.fmt.parseInt(@TypeOf(self.sampling.top_k), parts.next() orelse return error.InvalidArgument, 10);
            self.sampling.top_p = try std.fmt.parseFloat(f64, parts.next() orelse return error.InvalidArgument);
        } else if (cli.value(word, "follow=")) |given| {
            follow.* = try cli.readIds(io, arena, given);
        } else if (cli.value(word, "drafter=")) |given| {
            self.drafter = given;
        } else if (std.mem.eql(u8, word, "drafter_stored")) {
            self.drafter_stored = true;
        } else if (std.mem.eql(u8, word, "drafter_mixed")) {
            self.drafter_stored = false;
        } else return false;
        return true;
    }
};

/// One reply, one token a step, the head not asked. The one-row graph is the one a drafted case prepared.
fn generatePlain(init: std.process.Init, stdout: *std.Io.Writer, model: qwen4_model.Model, state: *qwen4_forward.State, buffers: *qwen4_forward.Buffers, graphs: *qwen4_forward.Graphs, prompt: []const i32, count: usize, window: usize) !u8 {
    const vocab = model.shape.vocab;
    const prompt_at: std.Io.Timestamp = .now(init.io, .awake);
    var next: i32 = 0;
    var at: usize = 0;
    while (at < prompt.len) {
        const rows = @min(window, prompt.len - at);
        try qwen4_forward.forward(model, state, buffers, prompt[at..][0..rows]);
        for (0..rows) |r| next = try qwen4_forward.argmax(init.gpa, buffers.*, vocab, r);
        try qwen4_forward.commit(model, state, buffers.*, prompt[at..][0..rows], rows);
        at += rows;
    }
    const prompt_s = @as(f64, @floatFromInt(prompt_at.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("prompt {d:.3} s\n", .{prompt_s});
    try stdout.flush();
    const start: std.Io.Timestamp = .now(init.io, .awake);
    for (0..count) |_| {
        const step = [1]i32{next};
        try graphs.forward(model, state, buffers, &step);
        const best = try qwen4_forward.top(init.gpa, buffers.*, vocab, 0);
        next = best.token;
        try qwen4_forward.commit(model, state, buffers.*, &step, 1);
    }
    const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("{d} tokens in {d:.2} s: {d:.1} tok/s (serial)\n", .{ count, seconds, @as(f64, @floatFromInt(count)) / seconds });
    return 0;
}

/// `generate` with the MTP head: greedy rounds of the newest token and the head's drafts.
fn generateDrafted(init: std.process.Init, stdout: *std.Io.Writer, model: qwen4_model.Model, state: *qwen4_forward.State, buffers: *qwen4_forward.Buffers, graphs: *qwen4_forward.Graphs, prompt: []const i32, count: usize, prepared: bool, follow: ?[]const i32, load_from: ?std.Io.Timestamp, drafting_options: Drafting) !u8 {
    const settings = drafting_options.sampling;
    const logits = try init.arena.allocator().alloc(u16, model.shape.vocab);
    var sampler: sampling.Sampler = try .init(init.gpa, logits.len);
    defer sampler.deinit(init.gpa);
    var head: qwen4_mtp.Head = try .init(init.gpa, model, &sampler, settings, .{ .rows = buffers.rows, .capacity = drafting_options.context, .beside = buffers });
    defer head.deinit();
    const listed = drafting_options.listed or drafting_options.chains;
    const chains = drafting_options.chains;
    const depth = if (chains) qwen4_draft.forward_max_rows - 1 else drafting_options.depth;
    // a chain is cut by nothing, yet every draft's probability is wanted: a threshold nothing falls under
    if (chains) head.confidence = 1e-30 else if (drafting_options.confidence) |confidence| head.confidence = confidence;
    if (chains) head.chain = 0;
    // a chain cut by nothing is not weighed against the checkpoint's costs either (`mtp_rule.Paying`)
    if (chains) head.costs = null;
    head.fresh();
    var rounds: qwen4_draft.Decoder = .{ .m = model, .state = state, .b = buffers, .graphs = graphs, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits, .follow = if (follow) |tokens| .{ .tokens = tokens, .first = prompt.len } else null };
    // a line a round, printed after the reply so that no round waits for it
    var lines: std.Io.Writer.Allocating = .init(init.arena.allocator());
    // as `Session.prepare` does for `chat` and `serve`: the graphs of this command's rounds, before the prompt
    const free_before = try cuda.memory();
    const bytes_before = cuda.account.bytes;
    if (prepared) {
        try graphs.prepareWindows(model, state, buffers, if (chains) 1 else depth + 1, qwen4_forward.Graphs.bucketAfter(buffers.*, null).?);
        try head.prepare(buffers.h, qwen4_forward.Graphs.bucketAfter(head.b.layer, null).?);
    }
    if (load_from) |start| {
        // chat's `loaded in`: the weights and this reply's graphs, not its prompt
        const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
        try stdout.print("loaded in {d:.1} s\n", .{seconds});
        try stdout.flush();
    }
    const free_after = try cuda.memory();
    const step_graphs: usize = graphs.cache.count();
    const prepared_graphs = step_graphs + head.graphs.?.cache.count();
    var chosen: [qwen4_draft.forward_max_rows]i32 = undefined;
    var drafts: [qwen4_draft.forward_max_rows]i32 = undefined;
    const prompt_at: std.Io.Timestamp = .now(init.io, .awake);
    var copies: ?copied.Copied = if (drafting_options.copies and !chains) try .init(init.gpa, head.drafter(), &.{}) else null;
    defer if (copies) |*c| c.deinit();
    if (copies) |*c| try c.begin(prompt);
    const drafter = if (copies) |*c| c.drafter() else head.drafter();
    var kept = try rounds.prompt(drafter, prompt, &chosen);
    const prompt_s = @as(f64, @floatFromInt(prompt_at.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("prompt {d:.3} s; greedy: {d}", .{ prompt_s, chosen[kept - 1] });
    try stdout.flush();
    const start: std.Io.Timestamp = .now(init.io, .awake);
    const launches_before = cuda.Graph.launches;
    const launching_before = cuda.Graph.launching;
    const reading_before = qwen4_forward.reading;
    const rows_read_before = qwen4_forward.rows_read;
    const choosing_before = decode_draft.choosing;
    var made: usize = 1;
    // seconds in the head: absorbing and chaining drafts, their choices on the host included
    var drafting: f64 = 0;
    // seconds in windows of each number of rows, and how many there were
    var windows: [qwen4_draft.forward_max_rows + 1]f64 = @splat(0);
    var sizes: [qwen4_draft.forward_max_rows + 1]usize = @splat(0);
    // seconds in the windows that captured their graph first, which the sizes above leave out
    var capturing: f64 = 0;
    while (made < count) {
        const before: std.Io.Timestamp = .now(init.io, .awake);
        const drafted = try rounds.draft(drafter, chosen[0..kept], @min(depth, count - made - 1), &drafts);
        const n = if (chains) 0 else drafted;
        drafting += @as(f64, @floatFromInt(before.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
        const verify: std.Io.Timestamp = .now(init.io, .awake);
        const captured = graphs.cache.count();
        const position = state.pos;
        kept = try rounds.round(chosen[kept - 1], drafts[0..n], &chosen);
        if (listed) {
            // the head's chain as token:probability, a `/` before one that ended it and was not drafted
            try lines.writer.print("{s} at {d}: {d} drafts, {d} kept;", .{ if (chains) "chain" else "round", position, n, kept - 1 });
            for (head.trail[0..head.trailed], 0..) |c, i| try lines.writer.print(" {s}{d}:{d:.4}", .{ if (i >= n and !chains) "/" else "", c.token, c.p });
            try lines.writer.writeAll("\n");
        }
        const took = @as(f64, @floatFromInt(verify.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
        if (graphs.cache.count() != captured) {
            capturing += took;
        } else {
            windows[n + 1] += took;
            sizes[n + 1] += 1;
        }
        for (chosen[0..kept]) |token| try stdout.print(" {d}", .{token});
        made += kept;
    }
    const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    const stats = rounds.stats;
    try stdout.print("\n{s}", .{lines.written()});
    const step_graphs_now: usize = graphs.cache.count();
    const step_capture_now: u64 = graphs.cache.capturing();
    try stdout.print("{d} tokens in {d:.2} s: {d:.1} tok/s, {d:.2} s of it drafting ({d} steps, {d} drafts, {d} of them kept; {d} step graphs, {d} head graphs)\n", .{ made - 1, seconds, @as(f64, @floatFromInt(made - 1)) / seconds, drafting, stats.rounds, stats.drafted, stats.accepted, step_graphs_now, head.graphs.?.cache.count() });
    if (copies) |c| try stdout.print("copies: found in {d} rounds, offered in {d}; {d} copied rows, {d} of them kept\n", .{ c.stats.found, c.stats.offered, c.stats.rows, c.stats.kept });
    try stdout.print("{d:.1} ms capturing step graphs, {d:.1} ms head graphs, {d} of the graphs before the prompt; device_free {d} before the graphs and {d} after, buffers {d} then {d}; {d:.2} s in the windows that captured; ms a window after, by its rows:", .{ @as(f64, @floatFromInt(step_capture_now)) / 1e6, @as(f64, @floatFromInt(head.graphs.?.cache.capturing())) / 1e6, prepared_graphs, free_before.free, free_after.free, bytes_before, cuda.account.bytes, capturing });
    for (windows, sizes, 0..) |total, n, rows| {
        if (n > 0) try stdout.print(" {d}: {d:.1} ({d})", .{ rows, 1e3 * total / @as(f64, @floatFromInt(n)), n });
    }
    const launches = cuda.Graph.launches - launches_before;
    try stdout.print("\n{d} graphs launched for the first time so far, {d:.2} ms the host's in those launches", .{ cuda.Graph.first_launches, @as(f64, @floatFromInt(cuda.Graph.first_launching)) / 1e6 });
    try stdout.print("\n{d} rows of logits copied to the host in {d:.1} ms, their tokens chosen in {d:.1} ms", .{ qwen4_forward.rows_read - rows_read_before, @as(f64, @floatFromInt(qwen4_forward.reading - reading_before)) / 1e6, @as(f64, @floatFromInt(decode_draft.choosing - choosing_before)) / 1e6 });
    const launching = @as(f64, @floatFromInt(cuda.Graph.launching - launching_before)) / 1e6;
    try stdout.print("\n{d} graph launches took the host {d:.1} ms, {d:.3} a step; drafts chosen from the device's candidates {d}, from the row read back {d}\n", .{ launches, launching, launching / @as(f64, @floatFromInt(@max(stats.rounds, 1))), head.chosen.device, head.chosen.host });
    return 0;
}

/// `generate` with DFlash2: greedy rounds of the newest token and the drafter's proposals; the target chooses.
fn generateFlash(init: std.process.Init, stdout: *std.Io.Writer, model: qwen4_model.Model, state: *qwen4_forward.State, buffers: *qwen4_forward.Buffers, graphs: ?*qwen4_forward.Graphs, prompt: []const i32, count: usize, prepared: bool, follow: ?[]const i32, load_from: ?std.Io.Timestamp, drafting_options: Drafting, opened: *const rig.Opened) !u8 {
    const settings = drafting_options.sampling;
    const arena = init.arena.allocator();
    var flash: rig.Rig(2) = undefined;
    var source: rig.Source = .{ .checkpoint = &opened.checkpoint.? };
    try flash.load(arena, init.gpa, opened, &source, .{ 0, 1 }, model.scratch, drafting_options.drafter_stored, null);
    errdefer flash.unload();
    const heads = try rig.draftHeads(2, arena, init.gpa, model, drafting_options.drafter_stored, .{ .pool = model.scratch, .count = dflash2_pass.batch });
    try flash.init(init.gpa, opened, heads, null, model, buffers, drafting_options.context, settings, graphs == null);
    defer flash.deinit();
    if (drafting_options.confidence) |confidence| flash.d.confidence = confidence;
    const listed = drafting_options.listed or drafting_options.chains;
    const chains = drafting_options.chains;
    const depth = if (chains) qwen4_draft.forward_max_rows - 1 else drafting_options.depth;
    if (chains) {
        flash.d.confidence = 1e-30;
        flash.d.cap = depth;
    }
    const logits = try arena.alloc(u16, model.shape.vocab);
    var sampler: sampling.Sampler = try .init(init.gpa, logits.len);
    defer sampler.deinit(init.gpa);
    var rounds: qwen4_draft.Decoder = .{ .m = model, .state = state, .b = buffers, .graphs = graphs, .sampler = &sampler, .settings = settings, .eos = &.{}, .logits = logits, .follow = if (follow) |tokens| .{ .tokens = tokens, .first = prompt.len } else null };
    var lines: std.Io.Writer.Allocating = .init(arena);
    if (prepared) if (graphs) |g| try g.prepareWindows(model, state, buffers, if (chains) 1 else depth + 1, qwen4_forward.Graphs.bucketAfter(buffers.*, null).?);
    if (load_from) |start| {
        const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
        try stdout.print("loaded in {d:.1} s\n", .{seconds});
        try stdout.flush();
    }
    try stdout.print("drafted by DFlash2, its taps after layers {any}\n", .{opened.config.taps});
    try stdout.flush();
    var chosen: [qwen4_draft.forward_max_rows]i32 = undefined;
    var drafts: [qwen4_draft.forward_max_rows]i32 = undefined;
    const prompt_at: std.Io.Timestamp = .now(init.io, .awake);
    const drafter = flash.d.drafter();
    var kept = try rounds.prompt(drafter, prompt, &chosen);
    const prompt_s = @as(f64, @floatFromInt(prompt_at.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    try stdout.print("prompt {d:.3} s; greedy: {d}", .{ prompt_s, chosen[kept - 1] });
    try stdout.flush();
    const start: std.Io.Timestamp = .now(init.io, .awake);
    var made: usize = 1;
    var drafting: f64 = 0;
    while (made < count) {
        const before: std.Io.Timestamp = .now(init.io, .awake);
        const drafted = try rounds.draft(drafter, chosen[0..kept], @min(depth, count - made - 1), &drafts);
        const n = if (chains) 0 else drafted;
        drafting += @as(f64, @floatFromInt(before.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
        const position = state.pos;
        kept = try rounds.round(chosen[kept - 1], drafts[0..n], &chosen);
        if (listed) {
            try lines.writer.print("{s} at {d}: {d} drafts, {d} kept;", .{ if (chains) "chain" else "round", position, n, kept - 1 });
            for (flash.d.trail[0..flash.d.trailed], 0..) |c, i| {
                try lines.writer.print(" {s}{d}:{d:.4}", .{ if (i >= n and !chains) "/" else "", c.token, c.p });
                if (i < flash.d.trailed) try lines.writer.print(":{d:.4}", .{flash.d.noised[i]});
            }
            try lines.writer.writeAll("\n");
        }
        for (chosen[0..kept]) |token| try stdout.print(" {d}", .{token});
        made += kept;
    }
    const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    const stats = rounds.stats;
    try stdout.print("\n{s}", .{lines.written()});
    try stdout.print("{d} tokens in {d:.2} s: {d:.1} tok/s, {d:.2} s of it drafting ({d} steps, {d} drafts, {d} of them kept)\n", .{ made - 1, seconds, @as(f64, @floatFromInt(made - 1)) / seconds, drafting, stats.rounds, stats.drafted, stats.accepted });
    return 0;
}
