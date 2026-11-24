//! The checks on a checkpoint's first layers that load the same prefix (spec 133 WS-02): one load a cache
//! storage, every check of that storage run on it, each with the rows its own load had
//! (`checkpoints.narrowed`). Before, each check loaded the prefix itself: ten loads for three storages. The
//! checks are their files' fixtures' (`pool/hall_keep_fixture.zig`, `pool/hall_carry_fixture.zig`,
//! `pool/hall_score_fixture.zig`, `pool/hall_fixture.zig`, `mtp_fixture.zig`, `session/learnt_fixture.zig`).
//! Spec 133 WS-08 added `draft_test.zig`'s oracle on the coded caches (`draft_fixture.zig`) and
//! `forward_checkpoint_test.zig`'s graphs, windows and snapshot on the bf16 caches (`forward_fixture.zig`), the
//! trunk alone as their loads had it: three loads fewer. The three storages are then one load, each a copy of
//! it with its caches stored so (`checkpoints.storing`): two loads fewer.
//! A load's `max_rows` past a check's sizes scratch alone; the model's weights are only read, and a check that
//! changes its `Model` changes its own copy.

const std = @import("std");
const fixtures = @import("testing.zig");
const narrowed = fixtures.checkpoints.narrowed;
const attention = @import("../../../src/models/qwen4/blocks/attention.zig");
const draft = @import("../../../src/models/qwen4/draft.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const model = @import("../../../src/models/qwen4/model.zig");
const carry_fixture = @import("pool/hall_carry_fixture.zig");
const hall_fixture = @import("pool/hall_fixture.zig");
const keep_fixture = @import("pool/hall_keep_fixture.zig");
const score_fixture = @import("pool/hall_score_fixture.zig");
const learnt_fixture = @import("session/learnt_fixture.zig");
const heads = @import("mtp_fixture.zig");
const drafted_fixture = @import("draft_fixture.zig");
const forward_fixture = @import("forward_fixture.zig");
const qwen4_config = @import("../../../src/models/qwen4/config.zig");
const sampling = @import("../../../src/decode/sampling.zig");
const tokenizer = @import("../../../src/text/tokenizer.zig");
const rig = @import("../../../src/models/qwen4/dflash2/rig.zig");
const flash_config = @import("../../../src/decode/dflash2/config.zig");
const testing = std.testing;

/// The rows the session and pool checks were written for.
const rows = 80;
/// The rows a head pass of eight seats takes (`hall_score_fixture.zig`), which its check's own load had.
const seat_rows = score_fixture.seats_most * 16 + 32;
/// The load's rows, the widest check's: `forward_fixture.firstLayers`'s prompt window of 400 rows, a pack's
/// experts on their wide kernel.
const window_rows = 400;

/// Which checks a storage runs. `windows`, `forward_fixture.firstLayers`, on the load's 400 rows.
const Checks = struct { pool: bool, scored: bool, drafted: bool, flash: bool = false, windows: bool = false };

/// The storages the load's caches are given in turn (`checkpoints.storing`: a load's storage is its shapes'
/// alone), the head's cache bf16 in each, and the checks each runs.
const storages = [_]struct { kv: attention.Storage, checks: Checks }{
    .{ .kv = .fp8, .checks = .{ .pool = true, .scored = true, .drafted = true, .flash = true } },
    .{ .kv = .fp8_nvfp4, .checks = .{ .pool = false, .scored = false, .drafted = true } },
    .{ .kv = .bf16, .checks = .{ .pool = false, .scored = false, .drafted = false, .windows = true } },
};

test "a checkpoint's first layers, one load on FP8 caches: a pool stream's kept states and what a session held when the pool opened are a serial slot's, a pass's seats scored alone, a reply the serial one through the session and the pool and drafted by its head, two streams drafted by DFlash2 in the pool each the serial reply, an oracle's drafts the serial reply, the learnt states read back; the same load on FP8 caches with NVFP4 values: a reply the serial one through the session and the pool and drafted by its head, an oracle's drafts the serial reply, the learnt states read back; on bf16 caches: MXFP8 faces and the NVFP4 table through graphs, wide windows and a snapshot, the learnt states read back" {
    const gpa = testing.allocator;
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var tokens = try carry_fixture.tinyTokens(gpa);
    defer tokens.deinit(gpa);
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var opened: fixtures.checkpoints.Held = undefined;
    const config, const found = try opened.open(gpa, arena, path);
    defer opened.close();
    const layers = fixtures.checkpoints.attentionLayers(found) orelse return error.SkipZigTest;
    var m: model.Model = try .loadWith(gpa, config, found, .{ .layers = layers, .max_rows = window_rows, .draft_vocab = try model.DraftHead.defaultIds(arena), .kv = storages[0].kv, .head_kv = layout.head_storage });
    defer m.deinit();
    try testing.expectEqual(attention.Storage.bf16, layout.head_storage);
    for (storages) |storage| {
        const stored = fixtures.checkpoints.storing(m, storage.kv, layout.head_storage);
        try onStorage(gpa, arena, stored, storage.kv, storage.checks, &opened, &config, found, layers, &tokens);
    }
}

/// The checks of `checks` on `m`, the load with its caches stored `kv`; then the oracle's drafted reply
/// (`draft_fixture.checkpointDrafted`) where the caches are coded, and the learnt states.
fn onStorage(gpa: std.mem.Allocator, arena: std.mem.Allocator, m: model.Model, kv: attention.Storage, checks: Checks, opened: *const fixtures.checkpoints.Held, config: *const qwen4_config.Config, found: layout.Layout, layers: usize, tokens: *const tokenizer.Tokenizer) !void {
    try testing.expectEqual(kv, m.attention.?.shape.kv);
    try testing.expectEqual(layout.head_storage, m.attention.?.head);
    if (checks.windows) {
        errdefer std.debug.print("caches {t}: graphs, wide windows and a snapshot\n", .{kv});
        // the trunk alone, as the check's own load had it (`.mtp = false`)
        var trunk = m;
        trunk.mtp = null;
        try forward_fixture.firstLayers(gpa, trunk, opened, config.*, found);
    }
    const m80 = narrowed(m, rows);
    if (checks.pool) {
        errdefer std.debug.print("caches {t}: a pool stream's kept states, the hand-over\n", .{kv});
        // a pool stream's kept states are a serial slot's
        try keep_fixture.keptAsSerial(gpa, m80, .{ .slots = 2, .prompt_rows = 64, .chunk = 20 }, true);
        // what a session held when the pool opened is a serial slot's states and rows
        try carry_fixture.handOver(gpa, m80, tokens, .live);
    }
    if (checks.scored) {
        errdefer std.debug.print("caches {t}: a pass's seats scored\n", .{kv});
        // a pass's seats scored in one launch are each the row scored alone, by the draft head and by the head
        // as stored, which `ZSPARK_DRAFTS_STORED` scores the drafts with; in the rows its own load had
        const seated = narrowed(m, seat_rows);
        try testing.expect(seated.mtp.?.draft != null);
        try score_fixture.scoredAlike(gpa, seated);
        var stored = seated;
        stored.mtp.?.draft = null;
        try score_fixture.scoredAlike(gpa, stored);
    }
    if (checks.drafted) {
        errdefer std.debug.print("caches {t}: through the session and the pool, the drafted reply\n", .{kv});
        // through the session and the pool: a reply the serial one before, beside and after
        try hall_fixture.sameBeforeBesideAfter(gpa, m80, tokens, .{ .kv = kv, .head = .bf16 }, null);
        // with its head: the drafted reply is the serial one
        try heads.draftedAsSerial(gpa, narrowed(m, draft.forward_max_rows));
    }
    if (checks.flash) {
        errdefer std.debug.print("caches {t}: two streams drafted by DFlash2 through the pool\n", .{kv});
        try draftedByFlash(gpa, arena, m80, config, layers, tokens);
    }
    if (kv != .bf16) {
        errdefer std.debug.print("caches {t}: the oracle's drafted reply\n", .{kv});
        // `draft_test.zig`'s: the bf16 caches' is in the card's form, `forward_checkpoint_test.zig`'s load
        try drafted_fixture.checkpointDrafted(gpa, m);
    }
    errdefer std.debug.print("caches {t}: the learnt states\n", .{kv});
    // a state written by one session is read by the next, a shared prefix's and one that left
    try learnt_fixture.readBack(m80);
    try learnt_fixture.spiltBack(m80);
}

/// `serve --drafter` on the load: the pinned DFlash2 checkpoint (`~/models/Qwen3.8-Flash-Next-DFlash2`, said
/// and passed over where absent) held to the layers loaded (`rig.fit`), and two streams through a core whose
/// session drafts by it, each the serial reply (`hall_fixture.twoDrafted`). The model's MTP head is left out,
/// as `serve` leaves it where DFlash2 drafts.
fn draftedByFlash(gpa: std.mem.Allocator, arena: std.mem.Allocator, loaded: model.Model, config: *const qwen4_config.Config, layers: usize, tokens: *const tokenizer.Tokenizer) !void {
    const home = testing.environ.getPosix("HOME") orelse return;
    const path = try std.fmt.allocPrint(arena, "{s}/models/Qwen3.8-Flash-Next-DFlash2", .{home});
    var opened: rig.Opened = .{ .path = path };
    opened.dir = std.Io.Dir.cwd().openDir(testing.io, path, .{}) catch {
        std.debug.print("{s} is absent: no DFlash2 check\n", .{path});
        return;
    };
    defer opened.close(testing.io);
    opened.config = try flash_config.Config.readStored(arena, testing.io, opened.dir.?, &opened.diagnostic);
    opened.checkpoint = try .open(gpa, testing.io, .cwd(), path);
    try testing.expect(try rig.fit(arena, &opened, config, layers));
    var m = loaded;
    m.mtp = null;
    const core = try hall_fixture.coreDrafting(gpa, m, tokens, &.{}, 4, 256, &opened);
    defer core.destroy();
    // a prefix's rows are not the whole model's: drafts are asked for whatever their confidence
    const flash = core.session.?.flash.?;
    flash.d.confidence = 0;
    flash.d.cap = flash.d.c.block - 1;
    const sampled: sampling.Sampling = .{ .seed = 5, .temperature = 0.9 };
    try hall_fixture.twoDrafted(gpa, core, m, .{ &.{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 }, &.{ 30, 31, 32, 33, 34 } }, .{ hall_fixture.greedy, sampled }, 24);
}
