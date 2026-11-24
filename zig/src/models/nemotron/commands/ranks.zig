//! `zspark ranks DIR [check|generate] [rank=R at=ADDR:PORT] [max=N] [rows=N] [context=N] ID..`
//! (spec 151 WS-06): the two-ranks Nemotron backbone. Without `rank=`/`at=` this process holds both
//! shares in lockstep (loopback — every rank's layer partial made before the fold reads it); with them
//! this process is one rank of two and joins the other through the control channel and NCCL
//! (`zig/src/ranks/join.zig`, rank 1 started first — `tools/two_ranks.sh` writes a rank's place after
//! the command's words). `check` runs the prompt's window through the serial model and through the
//! shares, and reports the scored rows' picks and the logits' cosine and rms gap (over NCCL both ranks
//! run it and compare a digest over the channel). `generate` decodes a greedy reply on the split path,
//! one serial row a step (the MTP drafter's rank form is the DFlash stream's).

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const join_file = @import("../../../ranks/join.zig");
const config_file = @import("../config.zig");
const model_file = @import("../model.zig");
const forward_file = @import("../forward.zig");
const state_file = @import("../state.zig");
const ranks_file = @import("../ranks.zig");
const ranks_forward = @import("../ranks_forward.zig");
const exchange = @import("../../../ranks/ranks.zig");
const experts_file = @import("../../../weights/experts.zig");
const generate = @import("generate.zig");

pub const usage = "ranks DIR [check|generate] [rank=R at=ADDR:PORT] [max=N] [rows=N] [context=N] ID..";

/// The rank shares' run state beside the gathered buffers (loopback: rank 0's and rank 1's buffers the
/// same process's; a pair: this rank's `[0]`, the peer's over NCCL). `driver` runs them in lockstep —
/// layer `index` on every held rank before the fold's gather, the loopback's proof the split is real.
const Held = struct {
    /// The rank shares' run state (loopback holds two, a pair this rank's own).
    state: [ranks_file.world]state_file.State,
    b: [ranks_file.world]state_file.Buffers,
    /// The gathered partials and logits halves.
    ranks: ranks_forward.Ranks,
    /// The two-process join, where `rank=`/`at=` named one.
    joined: ?join_file.Joined = null,
    /// The lockstep driver over the held ranks.
    driver: ranks_forward.Held,
};

/// The ranks `held` holds: two in loopback, one of a pair.
fn heldNum(held: *const Held) usize {
    return if (held.joined == null) ranks_file.world else 1;
}

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    if (words.len < 2) {
        try stdout.print("usage: zspark {s}\n", .{usage});
        return 2;
    }
    var mode: enum { check, generate } = .check;
    var max: usize = 32;
    var rows: usize = 512;
    var context: usize = 0;
    var place: join_file.Place = .{};
    var ids: std.ArrayList(i32) = .empty;
    for (words[1..]) |word| {
        if (place.take(word)) {
            continue;
        } else if (std.mem.eql(u8, word, "check")) {
            mode = .check;
        } else if (std.mem.eql(u8, word, "generate")) {
            mode = .generate;
        } else if (cli.value(word, "max=")) |text| {
            max = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "rows=")) |text| {
            rows = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "context=")) |text| {
            context = try std.fmt.parseInt(usize, text, 10);
        } else {
            try ids.append(arena, std.fmt.parseInt(i32, word, 10) catch {
                try stdout.print("usage: zspark {s}\n", .{usage});
                return 2;
            });
        }
    }
    const paired = place.rank != null or place.at != null;
    if (ids.items.len == 0 or (paired and !place.complete())) {
        try stdout.print("usage: zspark {s}\n", .{usage});
        return 2;
    }
    if (context == 0) context = ids.items.len + max + 16;
    var opened: generate.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, words[0], .raise)) return 1;
    const c = opened.config;

    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch = try cuda.Scratch.create(gpa);
    defer scratch.destroy();

    var held: Held = .{ .state = undefined, .b = undefined, .ranks = undefined, .driver = undefined };
    var joined: join_file.Joined = undefined;
    if (paired) {
        var text_buffer: [128]u8 = undefined;
        const text = std.fmt.bufPrint(&text_buffer, "{t} rows={d} context={d}", .{ mode, rows, context }) catch &text_buffer;
        try joined.open(init.io, gpa, place, try join_file.headers(arena, &opened.checkpoint.?), text);
        held.joined = joined;
    }
    defer if (held.joined) |*j| j.close();
    const gather: exchange.Gather = if (paired) .{ .nccl = joined.comm } else .loopback;
    const my_rank: usize = if (paired) place.rank.? else 0;

    // the serial model, whole — the loopback check's reference; a pair loads its share alone
    var m0: model_file.Model = undefined;
    var have0 = false;
    defer if (have0) m0.deinit();
    var m: [ranks_file.world]model_file.Model = undefined;
    var made: usize = 0;
    defer for (m[0..made]) |*one| one.deinit();
    if (paired) {
        m[0] = try model_file.Model.loadShared(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, rows, rows, .{ .rank = my_rank });
        made = 1;
    } else {
        m0 = try model_file.Model.loadShared(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, rows, rows, null);
        have0 = true;
        for (&m, 0..) |*one, rank| {
            one.* = try model_file.Model.loadShared(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, rows, rows, .{ .rank = rank });
            made += 1;
        }
    }
    try staging.settle();

    const shapes = ranks_file.Shapes.of(c.backbone, .{ .rank = my_rank });
    var made_sb: usize = 0;
    defer for (0..made_sb) |i| {
        held.state[i].deinit();
        held.b[i].deinit();
    };
    for (0..heldNum(&held)) |i| {
        held.state[i] = try state_file.State.initS(gpa, &c, context, shapes);
        held.b[i] = state_file.Buffers.initS(&c, rows, rows, context, shapes, c.backbone.vocab / ranks_file.world) catch |err| {
            held.state[i].deinit();
            return err;
        };
        made_sb += 1;
    }
    held.ranks = try ranks_forward.Ranks.init(m[0], rows, rows, my_rank, gather);
    defer held.ranks.deinit();
    held.driver = .{ .held = heldNum(&held), .m = m, .state = undefined, .b = undefined, .ranks = &held.ranks };
    for (0..heldNum(&held)) |i| {
        held.driver.state[i] = &held.state[i];
        held.driver.b[i] = &held.b[i];
    }
    var s0: state_file.State = undefined;
    var b0: state_file.Buffers = undefined;
    if (!paired) {
        s0 = try .init(gpa, &c, context);
        b0 = try .init(&c, rows, rows, context);
    }
    defer if (!paired) {
        s0.deinit();
        b0.deinit();
    };
    try cuda.synchronize();

    const logits = try gpa.alloc(u16, c.backbone.vocab);
    defer gpa.free(logits);
    const half_logits = try gpa.alloc(u16, c.backbone.vocab);
    defer gpa.free(half_logits);
    switch (mode) {
        .check => if (paired)
            return checkPair(stdout, &held, ids.items, logits)
        else
            return check(stdout, m0, &s0, &b0, &held, ids.items, logits, half_logits),
        .generate => return reply(stdout, m, &held, ids.items, max, logits),
    }
}

/// The prompt's window through the serial model and the two shares in lockstep, the scored rows'
/// logits compared: the split's fp32 partials sum in another order than the serial product's, so a
/// row's logits are not its bits — the gate is every row's pick equal (the transcript the reply takes)
/// and each row's cosine near 1, the rms gap reported.
fn check(stdout: *std.Io.Writer, m0: model_file.Model, s0: *state_file.State, b0: *state_file.Buffers, held: *Held, ids: []const i32, logits: []u16, half_logits: []u16) !u8 {
    const rows = ids.len;
    try forward_file.forwardScoring(m0, s0, b0, ids, .all);
    try held.driver.forward(ids, .all);
    var worst_cos: f64 = 1;
    var rms2: f64 = 0;
    var picks_diff: usize = 0;
    var first_pick: usize = 0;
    for (0..rows) |row| {
        try forward_file.logitsRow(b0.*, row, logits);
        try ranks_forward.logitsRow(&held.ranks, row, rows, half_logits);
        const a = argmax(logits);
        const x = argmax(half_logits);
        if (a != x) {
            picks_diff += 1;
            if (picks_diff == 1) first_pick = row;
        }
        var dot: f64 = 0;
        var aa: f64 = 0;
        var xx: f64 = 0;
        for (logits, half_logits) |one, other| {
            const av: f64 = @floatCast(@as(f32, @bitCast(@as(u32, one) << 16)));
            const xv: f64 = @floatCast(@as(f32, @bitCast(@as(u32, other) << 16)));
            dot += av * xv;
            aa += av * av;
            xx += xv * xv;
            rms2 += (av - xv) * (av - xv);
        }
        const cos = dot / @sqrt(aa * xx);
        worst_cos = @min(worst_cos, cos);
    }
    const rms = @sqrt(rms2 / @as(f64, @floatFromInt(rows * logits.len)));
    if (picks_diff == 0 and worst_cos >= 0.999) {
        try stdout.print("serial and split agree: {d} rows' picks equal, cosine {d:.6}, rms {d:.5}\n", .{ rows, worst_cos, rms });
        return 0;
    }
    try stdout.print("the split departs on row {d}: {d} rows' picks differ, cosine {d:.6}, rms {d:.5}\n", .{ first_pick, picks_diff, worst_cos, rms });
    return 1;
}

/// `check` a rank of two: this rank's share's forward, the scored rows' logits gathered and digested,
/// the digest met over the channel — the pair's bit-for-bit check is that both ranks hold the same
/// rows (the serial comparison is the loopback run's; `tools/two_ranks.sh` runs both).
fn checkPair(stdout: *std.Io.Writer, held: *Held, ids: []const i32, logits: []u16) !u8 {
    const channel = &held.joined.?.channel;
    const rows = ids.len;
    held.ranks.rank = held.ranks.gather.nccl.rank;
    try held.driver.forward(ids, .all);
    var digest = std.hash.Wyhash.init(0);
    for (0..rows) |row| {
        try ranks_forward.logitsRow(&held.ranks, row, rows, logits);
        digest.update(std.mem.sliceAsBytes(logits));
    }
    const mine = digest.final();
    // both ranks hold the same gathered rows: the digests met alike, or out of step
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, mine, .little);
    channel.request(&bytes) catch |err| return channel.gone(held.ranks.rank, err);
    const theirs: u64 = switch (channel.next() catch |err| return channel.gone(held.ranks.rank, err)) {
        .request => |got| if (got.len >= 8) std.mem.readInt(u64, got[0..8], .little) else return error.OutOfStep,
        .barrier => return error.OutOfStep,
    };
    if (theirs != mine) {
        try stdout.print("rank {d}: the gathered logits digest differs ({x:0>16} vs {x:0>16})\n", .{ held.ranks.rank, mine, theirs });
        return 1;
    }
    try stdout.print("rank {d}: the pair's gathered logits agree, {d} rows x {d} (digest {x:0>16})\n", .{ held.ranks.rank, rows, logits.len, mine });
    return 0;
}

/// A greedy reply on the split path: the prompt's window kept whole (`commit`), then a row a step
/// (`score .last`), each rank's half of the vocabulary gathered into `wide_logits` for the argmax. A
/// pair decodes the same reply on both ranks; rank 0 prints, the digest of each step met over the
/// channel so a rank that departs ends both.
fn reply(stdout: *std.Io.Writer, m: [ranks_file.world]model_file.Model, held: *Held, ids: []const i32, max: usize, wide_logits: []u16) !u8 {
    const rows = ids.len;
    try held.driver.forward(ids, .all);
    for (0..held.driver.held) |i| try forward_file.commit(held.driver.m[i], held.driver.state[i], held.driver.b[i].*, ids, rows);
    try ranks_forward.logitsRow(&held.ranks, rows - 1, rows, wide_logits);
    var token: i32 = argmax(wide_logits);
    var made: usize = 0;
    while (made < max) : (made += 1) {
        if (mRank(held) == 0) try stdout.print("{d} ", .{token});
        if (std.mem.indexOfScalar(u32, m[0].config.eos, @intCast(token)) != null) break;
        const step = [1]i32{token};
        try held.driver.forward(&step, .last);
        for (0..held.driver.held) |i| try forward_file.commit(held.driver.m[i], held.driver.state[i], held.driver.b[i].*, &step, 1);
        try ranks_forward.logitsRow(&held.ranks, 0, 1, wide_logits);
        token = argmax(wide_logits);
    }
    if (mRank(held) == 0) try stdout.print("\n", .{});
    return 0;
}

/// This process's rank of the pair, or 0 in loopback after a pass over its held ranks.
fn mRank(held: *const Held) usize {
    return switch (held.ranks.gather) {
        .loopback => 0,
        .nccl => |comm| comm.rank,
    };
}

/// The argmax of a bf16 logits row (ties the lower id).
fn argmax(logits: []const u16) i32 {
    var best: usize = 0;
    var top: f32 = -std.math.inf(f32);
    for (logits, 0..) |bits, i| {
        const v: f32 = @bitCast(@as(u32, bits) << 16);
        if (v > top) {
            top = v;
            best = i;
        }
    }
    return @intCast(best);
}
