//! `zspark generate DIR [max=N] [rows=N] [context=N] [depth=N] [serial] [mtp] [drafter=DIR] [drafter_stored] ID..`:
//! a prompt through the backbone and its reply decoded, timed. The drafter is DFlash2 by default — the
//! `~/models` checkpoint of the published drafter (`dflash2.zig`'s `defaultDir`, missing refused with the
//! tried path), or `drafter=DIR` naming another — the MTP head with `mtp`, `serial` serial steps;
//! `drafter_stored` loads the drafter's linears and proposing head BF16 as the file has them rather than
//! `.mixed` (NVFP4 copies made at load). The prompt is token ids; audio rows come from `perception.zig`
//! (WS-03) when a caller asks for them.

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const cli_opened = @import("../../../cli.zig");
const config_file = @import("../config.zig");
const layout_file = @import("../layout.zig");
const model_file = @import("../model.zig");
const forward_file = @import("../forward.zig");
const state_file = @import("../state.zig");
const draft_file = @import("../draft.zig");
const mtp_file = @import("../mtp.zig");
const rig = @import("../dflash2/rig.zig");
const dflash2_file = @import("../dflash2.zig");
const drafter_file = @import("../dflash2/drafter.zig");
const sampling = @import("../../../decode/sampling.zig");
const experts_file = @import("../../../weights/experts.zig");

pub const usage = "generate DIR [max=N] [rows=N] [context=N] [depth=N] [serial] [mtp] [drafter=DIR] [drafter_stored] ID..";

/// A checkpoint directory opened as Nemotron-3.5-Transcribe's: config, tensors, layout.
pub const Opened = cli_opened.Opened(config_file.Config, layout_file.Layout, config_file.Diagnostic);

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    var max: usize = 256;
    var rows: usize = 512;
    var context: usize = 0;
    var depth: usize = draft_file.forward_max_rows - 1;
    var serial = false;
    var mtp = false;
    var drafter_path: ?[]const u8 = null;
    var drafter_stored = false;
    var ids: std.ArrayList(i32) = .empty;
    for (words[1..]) |word| {
        if (cli.value(word, "max=")) |text| {
            max = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "rows=")) |text| {
            rows = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "context=")) |text| {
            context = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "depth=")) |text| {
            depth = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "drafter=")) |path| {
            serial = false;
            mtp = false;
            drafter_path = path;
        } else if (std.mem.eql(u8, word, "drafter_stored")) {
            drafter_stored = true;
        } else if (std.mem.eql(u8, word, "mtp")) {
            serial = false;
            mtp = true;
            drafter_path = null;
        } else if (std.mem.eql(u8, word, "serial")) {
            serial = true;
            mtp = false;
            drafter_path = null;
        } else {
            try ids.append(arena, std.fmt.parseInt(i32, word, 10) catch {
                try stdout.print("usage: zspark {s}\n", .{usage});
                return 2;
            });
        }
    }
    // DFlash2 is the default drafter (criterion 74): `mtp` selects the MTP head, `serial` neither, and
    // `drafter=DIR` names another DFlash2 checkpoint. The default's directory absent is an error, not a
    // quiet MTP run — nothing drafts silently under another head.
    if (!serial and !mtp and drafter_path == null) drafter_path = try dflash2_file.defaultDir(arena, init);
    const flash_drafter = !serial and !mtp;
    if (ids.items.len == 0) {
        try stdout.print("usage: zspark {s}\n", .{usage});
        return 2;
    }
    if (context == 0) context = ids.items.len + max;
    var opened: Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, words[0], .raise)) return 1;
    const c = opened.config;
    const start = cuda.monotonic();
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch = try cuda.Scratch.create(gpa);
    defer scratch.destroy();
    var m = try model_file.Model.load(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, draft_file.forward_max_rows, rows);
    defer m.deinit();
    try staging.settle();
    var state: state_file.State = try .init(gpa, &c, context);
    defer state.deinit();
    var b: state_file.Buffers = try .init(&c, rows, draft_file.forward_max_rows, context);
    defer b.deinit();
    var head: ?mtp_file.Head = null;
    var drafter: ?draft_file.Drafter = null;
    var drafts_ctx: mtp_file.Drafter = undefined;
    // DFlash2 in the MTP head's place: `drafter=DIR`, or the default `~/models` checkpoint of the
    // published drafter (`dflash2_file.defaultDir`) — absent, `enter`'s refusal names the tried path.
    // `mtp` picks the head explicitly instead, `serial` neither.
    var d_opened: rig.Opened = .{};
    defer d_opened.close(init.io);
    // on the heap: the rig points into itself (its drafter at its pair, the pair at its passes)
    var flash: ?*rig.Rig(2) = null;
    defer if (flash) |held| {
        held.deinit();
        gpa.destroy(held);
    };
    if (flash_drafter) {
        std.log.info("the drafter: DFlash2 at {s}", .{drafter_path.?});
        if (!try rig.enter(init, &d_opened, drafter_path.?, &c, c.backbone.types.len)) return 1;
        flash = try gpa.create(rig.Rig(2));
        var source: rig.Source = .{ .checkpoint = &d_opened.checkpoint.? };
        try flash.?.load(arena, gpa, &d_opened, &source, .{ 0, 1 }, scratch, drafter_stored, null);
        const storage: @import("../../../weights/linear.zig").Storage = if (drafter_stored) .bf16 else .four;
        const heads = try drafter_file.draftHeads(2, arena, gpa, m, storage, .{ .pool = scratch, .count = @import("../../../decode/dflash2/pass.zig").batch });
        try flash.?.init(gpa, &d_opened, heads, null, m, &b, context, .{ .seed = sampling.seedFor(ids.items, 0), .temperature = 0.0 }, false);
        drafter = flash.?.d.drafter();
    } else if (!serial) {
        // `mtp` asked (or `drafter=` named but the enter refused above): the head, where the model has one —
        // a model without it fails as `NoMtpHead`, never a quiet serial step
        head = try .init(&c, b.rows, context);
        try head.?.bind(m);
        drafts_ctx = .{ .head = &head.?, .state = &state };
        drafter = drafts_ctx.drafter();
    }
    defer if (head) |*h| h.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    const logits = try gpa.alloc(u16, c.backbone.vocab);
    defer gpa.free(logits);
    var sampler: sampling.Sampler = try .init(gpa, logits.len);
    defer sampler.deinit(gpa);
    try cuda.synchronize();
    const loaded = cuda.monotonic();
    var reply: std.ArrayList(u32) = .empty;
    defer reply.deinit(gpa);
    // the reply: the prompt's windows kept whole, then rounds of the MTP head's drafts verified (or serial
    // steps where it was not asked or has no head)
    var decoder: draft_file.Decoder = .{ .m = m, .state = &state, .b = &b, .graphs = &graphs, .sampler = &sampler, .settings = .{ .seed = sampling.seedFor(ids.items, 0), .temperature = 0.0 }, .eos = c.eos, .logits = logits };
    var chosen: [draft_file.forward_max_rows]i32 = undefined;
    var drafts: [draft_file.forward_max_rows]i32 = undefined;
    var kept = try decoder.prompt(drafter, ids.items, &chosen);
    const first = cuda.monotonic();
    // the reply is the decoder's choices: `chosen[i]` is the token after kept row i of each round, the last
    // round's `kept` the newest. Collect each token as it is made.
    try reply.append(gpa, @intCast(chosen[kept - 1]));
    var made: usize = 1;
    while (made < max) {
        const want = @min(depth, max - made);
        const n = if (drafter) |d| try decoder.draft(d, chosen[0..kept], want, &drafts) else 0;
        kept = try decoder.round(chosen[kept - 1], drafts[0..n], &chosen);
        const take = @min(kept, max - made);
        for (chosen[0..take]) |t| try reply.append(gpa, @intCast(t));
        made += kept;
    }
    const end = cuda.monotonic();
    const ms = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    for (reply.items) |t| try stdout.print("{d} ", .{t});
    try stdout.print("\nload {d:.0} ms, prompt {d} tokens {d:.1} ms, reply {d} tokens in {d:.1} ms ({d:.1} tok/s), device {d} MB\n", .{
        ms(loaded - start), ids.items.len, ms(first - loaded), reply.items.len, ms(end - first), if (reply.items.len > 0) @as(f64, @floatFromInt(reply.items.len)) / (ms(end - first) / 1e3) else 0, cuda.account.bytes >> 20,
    });
    return 0;
}
