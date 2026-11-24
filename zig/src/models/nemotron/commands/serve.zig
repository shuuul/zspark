//! `zspark serve DIR [--rows N] [--drafter DIR] [--drafter-stored] [--mtp] [--serial] [--cap N] [TensorFold's serve flags]`:
//! Nemotron-3.5-Transcribe kept loaded behind TensorFold's HTTP server (spec 151 WS-08), serving
//! `POST /v1/audio/transcriptions` in vLLM's shape, `/v1/models`, and `/v1/chat/completions` and
//! `/v1/completions` with `audio_url`/`input_audio` parts, as vLLM 0.32's NeMo SpeechLM plugin does: the
//! checkpoint's `nemotron3p5` template, thinking off unless a request asks for it (`--thinking`, or a
//! request's `chat_template_kwargs`), audio of up to 3,600 s (`served.max_seconds` is the plugin's bound).
//! The served name is `--name`, else the directory's basename.
//!
//! zspark's own flags: `--rows N`, a prompt window's rows (2,048); the drafter is DFlash2 by default
//! (criterion 74) — `--drafter DIR` names its checkpoint (`NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`),
//! `--drafter-stored` loads its linears BF16 as the file stores them, and absent the flag the published
//! drafter is looked up under `~/models` (`dflash2.zig`'s `defaultDir`, missing refused with the tried
//! path — never a quiet MTP run). `--mtp` selects the MTP head instead, `--serial` drafts by neither;
//! `--cap N` the most drafts of a round. Two ranks serve as Flash/GLM's `--rank 0|1 --at ADDR:PORT` (rank
//! 1 listening): rank 0 alone runs the routes and the decode, telling each windowed op over the join's
//! channel (`mirror.zig` replays them on rank 1's share — WS-06's split). TensorFold's flags as every
//! `serve` reads them (`server/command.zig`); `--context` the tokens a clip's audio rows, its prompt and
//! its reply share (`served.default_context`, the card's 65,536); `--max-queued N` the requests that may
//! wait behind the running one (64). `--parallel` is not read: one reply at a time, the decoder's state
//! and the perception being one.

const std = @import("std");
const server = @import("server");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const command = @import("../../../server/command.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const experts_file = @import("../../../weights/experts.zig");
const ranks_file = @import("../ranks.zig");
const ranks_forward = @import("../ranks_forward.zig");
const lead_file = @import("../../../ranks/lead.zig");
const held_ns = @import("../held_ns.zig");
const mirror = @import("../mirror.zig");
const config_file = @import("../config.zig");
const model_file = @import("../model.zig");
const forward_file = @import("../forward.zig");
const state_file = @import("../state.zig");
const draft_file = @import("../draft.zig");
const mtp_file = @import("../mtp.zig");
const perception_file = @import("../perception.zig");
const prompt_file = @import("../prompt.zig");
const served_file = @import("../served.zig");
const rig = @import("../dflash2/rig.zig");
const dflash2_file = @import("../dflash2.zig");
const opened_file = @import("generate.zig");
const sampling = @import("../../../decode/sampling.zig");

pub const usage = "serve DIR [--rows N] [--drafter DIR] [--drafter-stored] [--mtp] [--serial] [--cap N] [TensorFold's serve flags]";

/// A prompt window's rows unless `--rows` says.
const default_rows = 2048;

const Own = struct {
    rows: ?usize = null,
    /// The DFlash2 drafter's directory: it drafts in the MTP head's place.
    drafter: ?[]const u8 = null,
    /// No drafting head at all: serial steps.
    serial: bool = false,
    /// The MTP head drafting, the default DFlash2 declined (`--drafter`/`--drafter-stored` beside it).
    mtp: bool = false,
    /// The most drafts of a round (`Serve.depth`).
    cap: ?usize = null,
    /// The drafter's linears and proposing head loaded BF16 as stored (`rig.drafter_stored`), not
    /// `.four` copies — the stored precision the rig reads where the flag is given.
    drafter_stored: bool = false,

    const flags = [_]cli.Flag{
        .{ .name = "--rows", .field = "rows", .min = 1 },
        .{ .name = "--drafter", .field = "drafter" },
        .{ .name = "--serial", .field = "serial" },
        .{ .name = "--cap", .field = "cap", .min = 1, .max = draft_file.forward_max_rows - 1 },
        .{ .name = "--mtp", .field = "mtp" },
        .{ .name = "--drafter-stored", .field = "drafter_stored" },
    };

    const switches = [_][]const u8{ "--serial", "--mtp", "--drafter-stored" };

    fn kindOf(word: []const u8) command.Kind {
        for (flags) |each| {
            if (std.mem.eql(u8, each.name, word)) break;
        } else return .host;
        for (switches) |name| {
            if (std.mem.eql(u8, name, word)) return .own;
        }
        return .own_valued;
    }
};

pub fn run(init: std.process.Init, _: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    const path = words[0];
    // two ranks take Flash/GLM's words: `--rank 0|1 --at ADDR:PORT`, rank 1 listening; a half pair or a
    // build without NCCL is refused said (`command.ranked`)
    var place: join_file.Place = .{};
    const rest = join_file.rest(arena, words[1..], &place) catch {
        std.log.err("serve: --rank and --at each take a value", .{});
        return 2;
    };
    const pairing = command.ranked("serve", place, nccl.linked, .flags);
    if (pairing == .exit) return pairing.exit;
    const parted = try command.part(arena, rest, Own.kindOf);
    var own = cli.parse(Own, &Own.flags, .{ .quoted = false, .prefix = "serve: " }, parted.own) orelse return 2;
    rig.drafter_stored = own.drafter_stored;
    // DFlash2 is the default drafter (spec 151 criterion 74): `--mtp` selects the MTP head, `--serial`
    // neither, `--drafter DIR` names another DFlash2 checkpoint. The default's `~/models` directory absent
    // is a refusal naming the tried path — the engine never falls to MTP quietly.
    if (!own.serial and !own.mtp and own.drafter == null) own.drafter = try dflash2_file.defaultDir(arena, init);
    if (own.serial and own.drafter != null) std.log.warn("serve: --drafter with --serial: no drafter is loaded", .{});
    if (own.mtp and own.drafter != null) std.log.warn("serve: --drafter with --mtp: DFlash2 is loaded (the flag's DFlash2 wins)", .{});
    const bounded = try command.bounded(arena, parted.rest) orelse return 2;
    var args = try command.hostArgs(arena, path, bounded.rest) orelse return 2;
    // the card decodes thinking off: the template's assistant opener is closed by `<think></think>` unless a
    // request's `chat_template_kwargs` opens it — vLLM's default for this checkpoint (`prompt.zig`'s
    // transcription prompt the same). An explicit `--thinking` or `--no-thinking` is the user's, kept;
    // the flag absent, the host's default of thinking on would render `<think>` open, so it is made false.
    if (!thinkingAsked(bounded.rest)) args.thinking = false;
    command.warnUnread(args, "is not read by zspark's Nemotron-3.5-Transcribe", &.{
        .{ .name = "--prompt-cache-gib", .given = args.prompt_cache_gib != null },
        .{ .name = "--learn", .given = args.learn },
        .{ .name = "--parallel", .given = server.cli.parallelFixed(args.parallel) },
    });
    command.logVersion();
    server.log.init(io, false);
    if (pairing == .paired) return runPair(init, path, place, own, bounded, args, started);
    var startup = try command.Startup.open(arena, init.environ_map) orelse return 1;
    defer startup.deinit();

    var opened: opened_file.Opened = .{};
    defer opened.close(io);
    if (!try opened.open(init, path, .raise)) return 1;
    const c = opened.config;
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    const flash = own.drafter != null and !own.serial;
    if (flash) if (!try dflash2_file.enter(init, &drafting, own.drafter.?, &c, c.backbone.types.len)) return 1;

    // the text first: a checkpoint the server cannot tokenize fails before the load
    const text = command.loadText(gpa, io, path, arena) orelse return 1;
    defer text.deinit();
    var tok = command.loadTokenizer(gpa, &text.tok, path) orelse return 1;
    defer tok.deinit(gpa);
    const locator = tok.addedId(prompt_file.audio_locator) orelse {
        std.log.err("{s}: the tokenizer has no {s}", .{ path, prompt_file.audio_locator });
        return 1;
    };
    command.warnEnds(text.eos, c.eos);

    const context = (command.askedContext(args, 2) catch return 2) orelse served_file.default_context;
    const rows = own.rows orelse default_rows;

    const start = cuda.monotonic();
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch = try cuda.Scratch.create(gpa);
    defer scratch.destroy();
    // the drafter's bytes staged before the shares fill the memory (`rig.Early`)
    var early: ?*rig.Early = if (flash) begun: {
        const one = try gpa.create(rig.Early);
        one.* = .{ .source = .{ .checkpoint = &drafting.checkpoint.? } };
        one.begin(gpa, true, null);
        break :begun one;
    } else null;
    defer if (early) |one| {
        one.deinit(gpa);
        gpa.destroy(one);
    };
    var model = try model_file.Model.load(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, @max(rows, draft_file.forward_max_rows), rows);
    defer model.deinit();
    var perception = try perception_file.Perception.load(gpa, arena, &c, &opened.checkpoint.?, perception_chunks);
    defer perception.deinit();
    try staging.settle();
    var state: state_file.State = try .init(gpa, &c, context);
    defer state.deinit();
    var buffers: state_file.Buffers = try .init(&c, rows, draft_file.forward_max_rows, context);
    defer buffers.deinit();
    var graphs: forward_file.Graphs = try .init(gpa);
    defer graphs.deinit();
    var features: cuda.Buffer = try .init(.bf16, &.{ served_file.clipRowsCeiling(&c), c.perception.output_dim });
    defer features.deinit();
    var sampler: sampling.Sampler = try .init(gpa, c.backbone.vocab);
    defer sampler.deinit(gpa);
    var head: ?mtp_file.Head = null;
    var mtp_drafter: ?mtp_file.Drafter = null;
    var flash_rig: ?*rig.Rig(2) = null;
    if (!own.serial and !flash) {
        // `--mtp` asked: the head, where the model has one — a checkpoint without it fails as
        // `NoMtpHead`, never a quiet serial step
        head = try .init(&c, buffers.rows, context);
        try head.?.bind(model);
        mtp_drafter = .{ .head = &head.?, .state = &state };
    }
    defer if (head) |*h| h.deinit();
    if (flash) {
        const held = try gpa.create(rig.Rig(2));
        errdefer gpa.destroy(held);
        // the drafter's share from its stage, the stage then the host's again
        try held.load(arena, gpa, &drafting, try early.?.wait(), .{ 0, 1 }, scratch, rig.drafter_stored, null);
        errdefer held.unload();
        const heads = try @import("../dflash2/drafter.zig").draftHeads(2, arena, gpa, model, .four, .{ .pool = scratch, .count = @import("../../../decode/dflash2/pass.zig").batch });
        // `Rig.init` (`initRig`): taps the buffers before any graph is captured over them, and the
        // four-bit heads are the rig's from here (`pass.Target.owns_heads`)
        try held.init(gpa, &drafting, heads, null, model, &buffers, context, .{ .seed = 0, .temperature = 0 }, false);
        flash_rig = held;
        early.?.deinit(gpa);
        gpa.destroy(early.?);
        early = null;
    }
    defer if (flash_rig) |held| {
        held.deinit();
        gpa.destroy(held);
    };
    try cuda.synchronize();
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    std.log.info("Nemotron-3.5-Transcribe: {d:.2} GiB in {d:.1} s; {d} tokens of context, windows of {d} rows{s}", .{ gib, @as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e9, context, rows, if (flash) ", DFlash2 drafting" else if (own.serial) ", serial" else ", MTP drafting" });

    var serve_engine: served_file.Serve = .{
        .gpa = gpa,
        .io = io,
        .model = &model,
        .perception = &perception,
        .tok = &tok,
        .state = &state,
        .buffers = &buffers,
        .graphs = &graphs,
        .sampler = &sampler,
        .features = features,
        .locator = locator,
        .context = context,
        .most_waiting = bounded.max_queued orelse served_file.most_waiting,
        .drafter = if (flash) flash_rig.?.d.drafter() else if (mtp_drafter) |*d| d.drafter() else null,
        .flash = if (flash_rig) |held| &held.d else null,
        .depth = own.cap orelse draft_file.forward_max_rows - 1,
    };
    try serve_engine.start();
    const status = startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = serve_engine.engine(),
        .text = text.text(),
        .served = server.hub.servedName(args.name, args.model, path),
        .model_dir = path,
        .model_type = config_file.model_types[0],
        .sampling = null,
        .environ = init.environ_map,
        .started = started,
    });
    serve_engine.close();
    std.log.info("stopped", .{});
    return status;
}

/// Whether the command line named `--thinking` or `--no-thinking` (the host's `Args` cannot tell the
/// default from the flag given): a word of `rest` being either, or either written `=value`, counts.
pub fn thinkingAsked(rest: []const []const u8) bool {
    for (rest) |word| {
        if (std.mem.eql(u8, word, "--thinking") or std.mem.eql(u8, word, "--no-thinking")) return true;
        if (std.mem.startsWith(u8, word, "--thinking=") or std.mem.startsWith(u8, word, "--no-thinking=")) return true;
    }
    return false;
}

/// Encoder windows a call: the PEE's (pe_core + both contexts) rows a window, as many as the longest
/// clip's windows at a time — capped so the scratch stays small; a longer clip's windows run in groups
/// (`Perception.encode`).
const perception_chunks = 8;

/// `serve` on a rank of two (`--rank 0|1 --at ADDR:PORT`): this rank's share of the checkpoint
/// (`Model.loadShared`, `ranks.Share` halving heads, groups and expert width). `Held(1)` meets fp32
/// partials over NCCL. Rank 0 runs routes and decoding and tells each operation before it runs;
/// rank 1 replays it through `mirror.follow` until rank 0 closes.
fn runPair(init: std.process.Init, path: []const u8, place: join_file.Place, own: Own, bounded: command.Bounded, args: server.cli.Args, started: i96) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const rank = place.rank.?;
    var startup = if (rank == 0) try command.Startup.open(arena, init.environ_map) orelse return 1 else command.Startup{};
    defer startup.deinit();

    var opened: opened_file.Opened = .{};
    defer opened.close(io);
    if (!try opened.open(init, path, .raise)) return 1;
    const c = opened.config;
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    const flash = own.drafter != null and !own.serial;
    if (flash) if (!try dflash2_file.enter(init, &drafting, own.drafter.?, &c, c.backbone.types.len)) return 1;

    const text = command.loadText(gpa, io, path, arena) orelse return 1;
    defer text.deinit();
    var tok = command.loadTokenizer(gpa, &text.tok, path) orelse return 1;
    defer tok.deinit(gpa);
    const locator = tok.addedId(prompt_file.audio_locator) orelse {
        std.log.err("{s}: the tokenizer has no {s}", .{ path, prompt_file.audio_locator });
        return 1;
    };
    command.warnEnds(text.eos, c.eos);

    const context = (command.askedContext(args, 2) catch return 2) orelse served_file.default_context;
    const rows = own.rows orelse default_rows;

    // the pair: the checkpoint's name and every setting the ranks' told ops depend on are what the join
    // compares — a different DIR, drafter (its checkpoint's headers digest, the same weights under
    // another path alike), window, cap or context on the other end refuses both alike, where a silent
    // mismatch would diverge the collective sequence
    const drafter_name = if (drafting.checkpoint) |*checkpoint| named: {
        // the drafter's own headers' digest, the same weights under another path alike — a remote
        // `~/models` name resolving differently never refuses the same checkpoint
        const headers = try join_file.headers(arena, checkpoint);
        const hex = std.fmt.bytesToHex(&headers.digest, .lower);
        break :named try std.fmt.allocPrint(arena, "{s}@{s}", .{ std.fs.path.basename(own.drafter.?), &hex });
    } else if (own.serial) "serial" else "mtp";
    const named = try std.fmt.allocPrint(arena, "nemotron serve {s}:{s}:{s}:{s}:{d}:{d}:{d}", .{
        @import("std").fs.path.basename(path),
        drafter_name,
        if (own.serial) "serial" else "draft",
        if (own.drafter_stored) "stored" else "four",
        rows,
        own.cap orelse draft_file.forward_max_rows - 1,
        context,
    });
    var joined: join_file.Joined = undefined;
    try joined.open(io, gpa, place, try join_file.headers(arena, &opened.checkpoint.?), named);

    defer joined.close();
    const gather: @import("../../../ranks/ranks.zig").Gather = .{ .nccl = joined.comm };
    const start = cuda.monotonic();
    var staging: experts_file.Staging = .{};
    defer staging.deinit(gpa);
    var scratch = try cuda.Scratch.create(gpa);
    defer scratch.destroy();
    var early: ?*rig.Early = if (flash) begun: {
        const one = try gpa.create(rig.Early);
        one.* = .{ .source = .{ .checkpoint = &drafting.checkpoint.? } };
        one.begin(gpa, true, null);
        break :begun one;
    } else null;
    defer if (early) |one| {
        one.deinit(gpa);
        gpa.destroy(one);
    };
    const shapes = ranks_file.Shapes.of(c.backbone, .of(rank));
    var share_model = try model_file.Model.loadShared(arena, gpa, &c, &opened.checkpoint.?, &opened.diagnostic, &staging, scratch, @max(rows, draft_file.forward_max_rows), rows, .of(rank));
    defer share_model.deinit();
    var perception = try perception_file.Perception.load(gpa, arena, &c, &opened.checkpoint.?, perception_chunks);
    defer perception.deinit();
    try staging.settle();
    var model: held_ns.Model = .{ .rank = rank, .share = share_model, .max_rows = share_model.max_rows };
    var state: state_file.State = try .initS(gpa, &c, context, shapes);
    defer state.deinit();
    var rank_buffers: state_file.Buffers = try .initS(&c, rows, draft_file.forward_max_rows, context, shapes, c.backbone.vocab / ranks_file.world);
    defer rank_buffers.deinit();
    var shared_ranks: ranks_forward.Ranks = try .init(share_model, rank_buffers.rows, draft_file.forward_max_rows, rank, gather);
    defer shared_ranks.deinit();
    var lead: lead_file.Lead = .{ .channel = &joined.channel };
    var buffers: held_ns.Buffers = .{ .rank = &rank_buffers, .ranks = &shared_ranks, .lead = if (rank == 0) &lead else null, .rows = rank_buffers.rows };
    if (rank == 0) try buffers.initWide();
    var graphs: held_ns.Graphs = .init();
    defer graphs.deinit();
    var features: cuda.Buffer = try .init(.bf16, &.{ served_file.clipRowsCeiling(&c), c.perception.output_dim });
    defer features.deinit();
    var sampler: sampling.Sampler = try .init(gpa, c.backbone.vocab);
    defer sampler.deinit(gpa);

    var head: ?mtp_file.Head = null;
    var mtp_drafter: ?mtp_file.Drafter = null;
    var flash_rig: ?*rig.Rig(1) = null;
    if (!own.serial and !flash) {
        // `--mtp` asked: the head, where the model has one — a checkpoint without it fails as
        // `NoMtpHead`, never a quiet serial step
        head = try .initS(&c, buffers.rows, context, shapes);
        try head.?.bindShared(share_model, &shared_ranks);
        mtp_drafter = .{ .head = &head.?, .state = &state };
    }
    defer if (head) |*h| h.deinit();
    if (flash) {
        const held = try gpa.create(rig.Rig(1));
        errdefer gpa.destroy(held);
        try held.load(arena, gpa, &drafting, try early.?.wait(), .{rank}, scratch, rig.drafter_stored, null);
        errdefer held.unload();
        const heads = try @import("../dflash2/drafter.zig").draftHeads(1, arena, gpa, share_model, .four, .{ .pool = scratch, .count = @import("../../../decode/dflash2/pass.zig").batch });
        // `initRigHeld`: the buffers tapped on this rank's share, its half of `lm_head` scoring drafts
        var shares: [1]@import("../dflash2/drafter.zig").Of(1).Share = .{.{ .rank = rank, .m = share_model, .b = &rank_buffers }};
        try rig.initRigHeld(held, gpa, &drafting, heads, null, &shares, rank, gather, if (rank == 0) &lead else null, context, .{ .seed = 0, .temperature = 0 }, false);
        flash_rig = held;
        early.?.deinit(gpa);
        gpa.destroy(early.?);
        early = null;
    }
    defer if (flash_rig) |held| {
        held.deinit();
        gpa.destroy(held);
    };
    try cuda.synchronize();
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    std.log.info("rank {d}: Nemotron-3.5-Transcribe's share {d:.2} GiB in {d:.1} s; {d} tokens of context, windows of {d} rows{s}", .{ rank, gib, @as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e9, context, rows, if (flash) ", DFlash2 drafting" else if (own.serial) ", serial" else ", MTP drafting" });

    if (rank == 1) {
        // rank 0 serves alone; this rank replays each op it tells, on the same share and buffers — the
        // gather's halves meet there (no queue, no decode of its own)
        var follower: mirror.Followed = .{
            .gpa = gpa,
            .share = share_model,
            .state = &state,
            .b = &rank_buffers,
            .ranks = &shared_ranks,
            .capacity = context,
            .features = features,
            .perception = &perception,
            .mtp = if (mtp_drafter) |*d| d else null,
            .flash = if (flash_rig) |held| &held.d else null,
        };
        defer follower.deinit();
        const ran = mirror.follow(&joined.channel, &follower) catch |err| {
            if (err == error.PeerGone or err == error.PeerFailed) {
                std.log.err("rank 1: the peer is gone: {t}", .{err});
            } else {
                var buffer: [160]u8 = undefined;
                joined.channel.fail(std.fmt.bufPrint(&buffer, "rank 1 following: {t}", .{err}) catch "rank 1 following");
            }
            return 1;
        };
        std.log.info("rank 1: followed rank 0's {d} ops", .{ran});
        return 0;
    }

    var serve_engine: served_file.ServePair = .{
        .gpa = gpa,
        .io = io,
        .model = &model,
        .perception = &perception,
        .tok = &tok,
        .state = &state,
        .buffers = &buffers,
        .graphs = &graphs,
        .sampler = &sampler,
        .features = features,
        .locator = locator,
        .context = context,
        .most_waiting = bounded.max_queued orelse served_file.most_waiting,
        .drafter = if (flash) flash_rig.?.d.drafter() else if (mtp_drafter) |*d| d.drafter() else null,
        .depth = own.cap orelse draft_file.forward_max_rows - 1,
        .lead = &lead,
        .flash = if (flash_rig) |held| &held.d else null,
    };
    try serve_engine.start();
    defer serve_engine.close();
    const status = startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = serve_engine.engine(),
        .text = text.text(),
        .served = server.hub.servedName(args.name, args.model, path),
        .model_dir = path,
        .model_type = config_file.model_types[0],
        .sampling = null,
        .environ = init.environ_map,
        .started = started,
    });
    std.log.info("stopped", .{});
    return status;
}
