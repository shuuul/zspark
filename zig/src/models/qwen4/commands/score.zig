//! `zspark score`: how well each token of a text is predicted from those before it.

const std = @import("std");
const qwen4_layout = @import("../layout.zig");
const cuda = @import("../../../cuda/cuda.zig");
const numeric = @import("../../../weights/numeric.zig");
const qwen4_attention = @import("../blocks/attention.zig");
const qwen4_compare = @import("../compare.zig");
const qwen4_forward = @import("../forward.zig");
const qwen4_model = @import("../model.zig");
const cli = @import("../../../cli.zig");
const common = @import("common.zig");

/// Runs `ids` through the whole model in windows of 16 rows and scores each token after the first against the
/// logits of the row before it: the mean negative log-likelihood and the share of tokens ranked first, over the
/// whole sequence and over its second half (the deeper contexts). A measure to hold an approximation against:
/// the same ids on two builds or options differ by what the approximation costs. `perturb=SIZE[@LAYER[:ROW]]`
/// then scores the ids again under a tap that adds nothing, and `seeds=N` times more with noise of SIZE of a
/// row's rms added to LAYER's `h_in`, drawn afresh each window: each run against the tapped one, so the score's
/// own spread under a rounding's worth of difference is read beside a difference between two programs.
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const ids = argv[1..];
    const arena = init.arena.allocator();
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, path, .raise)) return 1;
    const config = opened.config;
    const found = opened.layout;
    const named = qwen4_attention.Storage.parse(ids[0]);
    const kv = qwen4_layout.storageFor(config, 1, 1, named).trunk;
    var words = ids[@intFromBool(named != null)..];
    // each: each token's position, loss and whether it ranked first, a line each, before the means
    var each = false;
    var noise: ?qwen4_compare.Perturb = null;
    var seeds: usize = 1;
    while (words.len > 0) : (words = words[1..]) {
        if (std.mem.eql(u8, words[0], "each")) {
            each = true;
        } else if (cli.value(words[0], "perturb=")) |given| {
            noise = try qwen4_compare.Perturb.parse(given);
        } else if (cli.value(words[0], "seeds=")) |count| {
            seeds = try std.fmt.parseInt(usize, count, 10);
        } else break;
    }
    const tokens = try cli.ids(arena, words);

    const window = 16;
    var model: qwen4_model.Model = try .loadWith(init.gpa, config, found, .{ .layers = config.layers, .max_rows = window, .mtp = false, .kv = kv });
    defer model.deinit();
    if (each and noise != null) try stdout.writeAll("== plain\n");
    const plain = try scoreRows(init, stdout, model, tokens, each, null);
    const settings = noise orelse return 0;

    var tap: Noisy = .{ .perturb = settings, .gpa = init.gpa };
    defer tap.scratch.deinit(init.gpa);
    tap.perturb.relative = 0;
    if (each) try stdout.writeAll("== tapped\n");
    const tapped = try scoreRows(init, stdout, model, tokens, each, &tap);
    try stdout.writeAll("under the tap, no noise: ");
    _ = try against(stdout, tapped, plain);
    try stdout.print("noise of {e:.2} of a row's rms added to layer {d}'s h_in, ", .{ settings.relative, settings.layer });
    if (settings.row) |row| try stdout.print("row {d}", .{row}) else try stdout.writeAll("every row");
    try stdout.writeAll(", each window its own draw; each seed against the tapped run\n");
    const means = try arena.alloc(f64, seeds);
    for (means, 0..) |*mean, seed| {
        tap.perturb = settings;
        tap.perturb.seed = seed + 1;
        if (each) try stdout.print("== seed {d}\n", .{seed + 1});
        const noisy = try scoreRows(init, stdout, model, tokens, each, &tap);
        try stdout.print("seed {d}: {e:.3} after the bf16 rounding; ", .{ seed + 1, tap.perturb.achieved });
        mean.* = try against(stdout, noisy, tapped);
    }
    var sum: f64 = 0;
    for (means) |m| sum += m;
    const centre = sum / @as(f64, @floatFromInt(seeds));
    var squares: f64 = 0;
    for (means) |m| squares += (m - centre) * (m - centre);
    const spread = if (seeds > 1) @sqrt(squares / @as(f64, @floatFromInt(seeds - 1))) else 0;
    try stdout.print("seeds: {d}, their differences' mean {d:.5}, standard deviation {d:.5}\n", .{ seeds, centre, spread });
    return 0;
}

/// Prints `mine`'s mean against `base`'s, token by token: the mean difference, its standard error by blocks
/// of 64 tokens, the mean of a token's absolute difference and the tokens whose loss differs at all. Returns the
/// mean difference.
fn against(stdout: *std.Io.Writer, mine: []const f64, base: []const f64) !f64 {
    const block = 64;
    var total: f64 = 0;
    var absolute: f64 = 0;
    var differing: usize = 0;
    for (mine, base) |a, b| {
        total += a - b;
        absolute += @abs(a - b);
        differing += @intFromBool(a != b);
    }
    const count: f64 = @floatFromInt(mine.len);
    const mean = total / count;
    const blocks = mine.len / block;
    var squares: f64 = 0;
    for (0..blocks) |k| {
        var part: f64 = 0;
        for (mine[k * block ..][0..block], base[k * block ..][0..block]) |a, b| part += a - b;
        const d = part / block - mean;
        squares += d * d;
    }
    const se = if (blocks > 1) @sqrt(squares / @as(f64, @floatFromInt(blocks - 1)) / @as(f64, @floatFromInt(blocks))) else 0;
    try stdout.print("{s}{d:.5} ± {d:.5} (blocks of {d}), a token's |difference| {d:.4}, {d} of {d} tokens differ\n", .{ if (mean < 0) "-" else "+", @abs(mean), se, block, absolute / count, differing, mine.len });
    return mean;
}

/// The tap `score` runs under: noise on one layer's `h_in` (none when `relative` is 0), seeded a window.
const Noisy = struct {
    perturb: qwen4_compare.Perturb,
    scratch: std.ArrayList(u8) = .empty,
    gpa: std.mem.Allocator,

    fn tapOf(self: *Noisy) qwen4_forward.Tap {
        return .{ .context = self, .at = at };
    }

    fn at(context: *anyopaque, point: qwen4_forward.Point, layer: usize, view: cuda.View) anyerror!void {
        const self: *Noisy = @ptrCast(@alignCast(context));
        if (point != .h_in or layer != self.perturb.layer or self.perturb.relative == 0) return;
        try self.scratch.resize(self.gpa, view.count() * view.dtype.size());
        try cuda.synchronize();
        try cuda.download(view, self.scratch.items);
        self.perturb.apply(self.scratch.items, @intCast(view.size[0]));
        try cuda.upload(view, self.scratch.items);
    }
};

/// The ids of `path` scored on `model`, which is already loaded. Windows of 16, as `score` uses; `each`: every
/// token's loss too.
pub fn scoreFile(init: std.process.Init, stdout: *std.Io.Writer, model: qwen4_model.Model, path: []const u8, each: bool) !u8 {
    _ = try scoreRows(init, stdout, model, try cli.readIds(init.io, init.arena.allocator(), path), each, null);
    return 0;
}

/// Mean negative log-likelihood of `tokens` on `model`, windows of 16 rows, and the share ranked first; each
/// scored token's loss is returned (the arena's). `noisy`: the forward runs under that tap.
fn scoreRows(init: std.process.Init, stdout: *std.Io.Writer, model: qwen4_model.Model, tokens: []const i32, each: bool, noisy: ?*Noisy) ![]f64 {
    const window = 16;
    var state: qwen4_forward.State = try .init(init.gpa, model, window, tokens.len);
    defer state.deinit();
    var buffers: qwen4_forward.Buffers = try .init(init.gpa, model, window, tokens.len);
    defer buffers.deinit();
    const logits = try init.arena.allocator().alloc(u16, model.shape.vocab);
    const losses = try init.arena.allocator().alloc(f64, tokens.len -| 1);

    // [whole, second half]
    var nll: [2]f64 = @splat(0);
    var first: [2]usize = @splat(0);
    var scored: [2]usize = @splat(0);
    var at: usize = 0;
    while (at < tokens.len) {
        const rows = @min(window, tokens.len - at);
        if (noisy) |tap| tap.perturb.window = at;
        try qwen4_forward.forwardTapped(model, &state, &buffers, tokens[at..][0..rows], if (noisy) |tap| tap.tapOf() else null);
        for (0..rows) |r| {
            if (at + r + 1 >= tokens.len) break;
            const target: usize = @intCast(tokens[at + r + 1]);
            try qwen4_forward.logitsRow(buffers, r, logits);
            var highest = -std.math.inf(f32);
            for (logits) |bits| highest = @max(highest, numeric.f32FromBf16(bits));
            var total: f64 = 0;
            for (logits) |bits| total += @exp(@as(f64, numeric.f32FromBf16(bits) - highest));
            const own = numeric.f32FromBf16(logits[target]);
            const loss = @log(total) - @as(f64, own - highest);
            losses[at + r] = loss;
            if (each) try stdout.print("{d} {d:.6} {d}\n", .{ at + r + 1, loss, @intFromBool(own == highest) });
            for ([2]bool{ true, at + r + 1 >= tokens.len / 2 }, 0..) |counted, part| {
                if (!counted) continue;
                nll[part] += loss;
                first[part] += @intFromBool(own == highest);
                scored[part] += 1;
            }
        }
        try qwen4_forward.commit(model, &state, buffers, tokens[at..][0..rows], rows);
        at += rows;
    }
    for ([2][]const u8{ "all", "second half" }, 0..) |name, part| {
        const count: f64 = @floatFromInt(scored[part]);
        try stdout.print("{s}: {d} tokens, nll {d:.5}, ranked first {d:.4}\n", .{ name, scored[part], nll[part] / count, @as(f64, @floatFromInt(first[part])) / count });
    }
    return losses;
}
