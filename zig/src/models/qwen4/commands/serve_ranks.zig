//! `serve.zig` on two ranks (`--rank R --at ADDR:PORT`, or `rank=R at=ADDR:PORT`; spec 050): rank 1
//! listens, loads its share and follows; rank 0 connects, loads and serves. The MTP head is not loaded on
//! a share, and one reply runs at a time, or several through the pool on both ranks (`hall_ranks.zig`;
//! spec 094 WS-03b). The flags are `serve`'s (`serve.zig`'s `OwnOptions` beside the host's); what changes
//! a rank's bits is compared when the ranks join (`join.headers`, the `alike` line).

const std = @import("std");
const server = @import("server");
const cuda = @import("../../../cuda/cuda.zig");
const qwen4_hall = @import("../pool/hall.zig");
const qwen4_hall_engine = @import("../pool/hall_engine.zig");
const qwen4_hall_open = @import("../pool/hall_open.zig");
const qwen4_hall_ranks = @import("../pool/hall_ranks.zig");
const qwen4_layout = @import("../layout.zig");
const qwen4_model = @import("../model.zig");
const qwen4_session = @import("../session/session.zig");
const served_file = @import("../served.zig");
const command = @import("../../../server/command.zig");
const segments_file = @import("../../../server/segments.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const channel_file = @import("../../../ranks/channel.zig");
const lead_file = @import("../../../ranks/lead.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("../session/pair.zig");
const ranks_drafter = @import("../checks/ranks_drafter.zig");
const qwen4_rig = @import("../dflash2/rig.zig");
const cli = @import("../../../cli.zig");
const common = @import("common.zig");
const serve_file = @import("serve.zig");

const H = held_file.Held(1);

/// One rank of two, until rank 0 is signalled. Rank 1 follows; rank 0 is the HTTP server.
pub fn serve(init: std.process.Init, path: []const u8, words: []const []const u8, place: join_file.Place) !u8 {
    const arena = init.arena.allocator();
    const io = init.io;
    const gpa = init.gpa;
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    const parted = try command.part(arena, words, serve_file.OwnOptions.kindOf);
    const own = cli.parse(serve_file.OwnOptions, &serve_file.OwnOptions.flags, .{ .quoted = false }, parted.own) orelse return 2;
    // rank 1 offers rank 0's copies by what its own lookup found (`pair_rounds.zig`): no record reaches it
    if (own.recalling()) {
        std.log.err("serve: --recall is one rank's; two ranks draft from their context alone", .{});
        return 2;
    }
    if (own.warm_turns) {
        std.log.err("serve: --warm-turns is one rank's pool's", .{});
        return 2;
    }
    const bounded = try command.bounded(arena, parted.rest) orelse return 2;
    const args = try command.hostArgs(arena, path, bounded.rest) orelse return 2;
    const parallel = command.parallelWithin(args, 1, 0, qwen4_hall.most_parallel) orelse return 2;
    const context = (command.askedContext(args, serve_file.least_context) catch return 2) orelse qwen4_hall.default_context;
    // Both ranks load their drafter share; the leader alone schedules and proposes for the shared pool.
    const flash = own.drafter != null and !(own.serial or args.no_drafts) and own.depth > 0;
    // the pool on two ranks keeps its states within `--prompt-cache-gib`, as one rank's does
    command.warnUnread(args, "is not read when Flash Next is served on two ranks", &.{
        .{ .name = "--prompt-cache-gib", .given = args.prompt_cache_gib != null and parallel == 1 },
        .{ .name = "--learn", .given = args.learn },
        .{ .name = "--spill-gib", .given = own.spill_gib != 0 },
        .{ .name = "--learn-min-free-gib", .given = own.learn_min_free_gib != (serve_file.OwnOptions{}).learn_min_free_gib },
    });
    command.logVersion();
    server.log.init(io, false);
    var startup = if (place.rank.? == 0) try command.Startup.open(arena, init.environ_map) orelse return 1 else command.Startup{};
    defer startup.deinit();

    var opened: common.Opened = .{};
    defer opened.close(io);
    if (!try opened.enter(io, path, .say) or !try opened.configure(init) or !try opened.resolve(init)) return 1;
    const config = opened.config;
    const rank = place.rank.?;
    const follows = rank != 0;
    // rank 1's command line says nothing of drafting, and what the ranks hold alike does not either, unless
    // the head is split (`common.headAlike`)
    const drafts = !(own.serial or args.no_drafts) and own.depth > 0 and !flash;
    const drafting = common.headOnRank(rank, drafts) orelse return 2;
    var flash_opened: qwen4_rig.Opened = .{};
    defer flash_opened.close(io);
    if (flash) if (!try qwen4_rig.enter(init, &flash_opened, own.drafter.?, &config, config.layers)) return 1;
    const text: ?*server.hf_text.HfText = if (follows) null else command.loadText(gpa, io, path, arena) orelse return 1;
    defer if (text) |held| held.deinit();
    var segments: ?segments_file.Segments = if (text) |held| try .init(gpa, io, held.text(), &held.tok, segments_file.default_bytes) else null;
    defer if (segments) |*kept| kept.deinit();
    if (text) |held| command.warnEnds(held.eos, config.eos);

    // One reply at a time is a pair; several are the pool on both ranks (`hall_ranks.zig`), its rows one
    // rank's for as many streams (`hall.rowsFor`). The storage is one rank's for as many streams, a share's
    // caches in it (half the KV heads).
    // A pair's windows are `--max-num-batched-tokens`'s as one Spark's lone stream's (spec 111), unset 4,096; the
    // pool's rounds on two ranks are too where it is asked (spec 149 WS-06), unset `session.promptRows` (2,048).
    // Two ranks' pool has no lone session, so its rounds are its only windows; one Spark's keeps them at 2,048
    // whatever is asked, beside its lone session's (its slots and windows 25.1 GiB at 4,096 against 13.5).
    const pooled = parallel > 1;
    const window_rows = (if (pooled) common.poolRows(config, own.max_num_batched_tokens) else common.pairRows(config, own.max_num_batched_tokens)) orelse return 2;
    const rows = if (pooled) qwen4_hall.rowsFor(window_rows, parallel) else @min(window_rows, context);
    const caches = qwen4_layout.storageFor(config, parallel, 2, own.kv);
    var streams_buffer: [24]u8 = undefined;
    const streams_text = if (pooled) try std.fmt.bufPrint(&streams_buffer, " parallel={d}", .{parallel}) else "";
    // both ranks load the drafter or neither
    const alike = try std.fmt.allocPrint(arena, "serve context={d} rows={d}{s}{s}{s}{s}", .{ context, rows, common.cachesAlike(caches), common.headAlike(drafts), streams_text, if (!flash) "" else if (qwen4_rig.drafter_stored) " drafter" else " drafter mixed" });
    // the first parts are read while the ranks join (`weights/loading.zig`); freed once the share is loaded
    var reading: qwen4_model.Reading = .init(gpa, opened.layout, config.layers, .of(@intCast(rank)));
    var reading_held = true;
    defer if (reading_held) reading.deinit();
    const load_options: qwen4_model.Model.Options = .{ .layers = config.layers, .max_rows = rows, .mtp = drafting, .draft_vocab = if (drafting) try qwen4_model.DraftHead.defaultIds(arena) else null, .share = .of(@intCast(rank)), .head_split = qwen4_model.ranks_head == .split, .kv = caches.trunk, .head_kv = caches.head, .reading = &reading };
    // each rank reads half of every routed down projection and trades the rest with the other (`Reading.swapping`),
    // and the MTP head after the head where it holds it
    reading.swapping = true;
    reading.mtp = qwen4_model.Model.mtpReading(qwen4_model.Model.drafts(opened.layout, load_options), load_options.head_split);
    reading.prefetch();
    var joined: join_file.Joined = undefined;
    cuda.phases.mark("the ranks join");
    try joined.open(io, gpa, place, try join_file.headers(arena, &opened.checkpoint.?), alike);
    defer joined.close();
    cuda.phases.mark("the ranks joined");
    const channel = &joined.channel;
    if (reading.swapping) reading.swap = .{ .rank = rank, .context = &joined.comm, .trade = nccl.Comm.trade };

    const load_start: std.Io.Timestamp = .now(io, .awake);
    const built: H.Model = .{
        .ranks = .{try qwen4_model.Model.loadWith(gpa, config, opened.layout, load_options)},
        .first = rank,
        .max_rows = rows,
    };
    cuda.phases.mark("the share is loaded");
    reading.deinit();
    reading_held = false;
    if (pooled) {
        var session_options = qwen4_hall.sessionFor(window_rows, context, own.reply_stride);
        session_options.copies = !own.no_copies and !flash;
        session_options.drafter = if (flash) &flash_opened else null;
        return servePool(init, &startup, config.model_type, .{ .built = built, .comm = joined.comm, .channel = channel, .rank = rank, .context = context, .parallel = parallel, .session = session_options, .own = own, .args = args, .max_queued = bounded.max_queued, .path = path, .dir = opened.dir.?, .vocab = config.vocab, .eos = config.eos, .text = text, .segments = if (segments) |*kept| kept else null, .started = started, .load_start = load_start, .kept_bytes = if (args.prompt_cache_gib) |gib| @intFromFloat(@min(@max(gib, 0), 1024) * (1 << 30)) else qwen4_hall.default_states_bytes });
    }
    const pair = pair_file.Pair.create(gpa, rank, built, joined.comm, channel, rows, context, false) catch |err| return rankFailed(channel, rank, err);
    defer pair.destroy();
    // before any window: the drafter taps the pair's buffers before a graph is captured over them; its own
    // passes are eager, as N1 ran them (spec 126): captured, they took nothing off a round (spec 135 WS-01)
    var given: ?*ranks_drafter.Given(1) = null;
    defer if (given) |held| held.deinit(gpa);
    if (flash) given = ranks_drafter.Given(1).init(gpa, &flash_opened, pair, qwen4_rig.drafter_stored, true) catch |err| return rankFailed(channel, rank, err);
    channel.barrier(1) catch |err| return rankGone(channel, rank, err);
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    const load_s = @as(f64, @floatFromInt(load_start.untilNow(io, .awake).toNanoseconds())) / 1e9;
    var drafts_buffer: [64]u8 = undefined;
    const halved = pair.split != null;
    const drafts_text = if (given != null) (if (follows) "follows rank 0, its share of the DFlash2 drafter beside it" else "its share of the DFlash2 drafter beside it") else if (pair.head != null) try std.fmt.bufPrint(&drafts_buffer, "{s} beside it, {d} drafts a window", .{ if (halved) "half of the MTP head" else "the MTP head", own.depth }) else if (follows) (if (halved) "follows rank 0, half of the MTP head beside it" else "follows rank 0") else "no drafts";
    std.log.info("rank {d} of 2: NCCL {d}, its share {d:.2} GiB in {d:.1} s; {d} tokens of context, {s}", .{ rank, try nccl.version(), gib, load_s, context, drafts_text });
    std.log.info("attention caches {s}, the drafting head's {s}", .{ caches.trunk.name(), caches.head.name() });

    if (follows) {
        const ran = pair.follow() catch |err| return rankFailed(channel, rank, err);
        std.log.info("rank {d}: followed rank 0's {d} operations", .{ rank, ran });
        return 0;
    }

    var ended = false;
    errdefer if (!ended) pair.finish() catch {};
    // the states a pair keeps are one rank's, and its stride through a reply the flag's
    try pair.marking(.{ .reply_stride = own.reply_stride });
    var front: command.Front = undefined;
    if (!try front.open(gpa, io, opened.dir.?, path, &text.?.tok, config.vocab, config.eos)) {
        pair.finish() catch {};
        return 1;
    }
    defer front.deinit(gpa);
    const name = server.hub.servedName(args.name, args.model, path);
    const defaults = try command.samplingDefaults(arena, io, path, args);
    var served: served_file.Served = .{
        .gpa = gpa,
        .io = io,
        .frontend = &front.frontend,
        .compiler = front.grammar.compiler(),
        .behind = .{ .leader = pair, .depth = if (pair.head == null and given == null) 0 else own.depth, .copies = !own.no_copies },
        .capacity = context,
        .vocab = @intCast(config.vocab),
        .lanes = 1,
        .max_queued = bounded.max_queued,
        .think = front.think(),
        // two ranks: the head beside rank 0's share drafts where it is loaded, or the DFlash2 each rank holds under --drafter
        .draft_sources = if (own.serial or args.no_drafts or own.depth == 0) .{} else .{ .copies = !own.no_copies, .mtp = pair.head != null and given == null, .drafter = if (given != null) .{ .kind = "DFlash2", .name = own.drafter.? } else null },
        .device_bytes = &cuda.account.bytes,
    };
    const status = startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = served.engine(),
        .text = segments.?.text(),
        .served = name,
        .model_dir = path,
        .model_type = config.model_type,
        .sampling = defaults,
        .environ = init.environ_map,
        .started = started,
    });
    served.close();
    ended = true;
    pair.finish() catch |err| return rankFailed(channel, rank, err);
    std.log.info("stopped", .{});
    return status;
}

/// What `servePool` is handed by `serve`.
const Pooled = struct {
    built: H.Model,
    comm: nccl.Comm,
    channel: *channel_file.Channel,
    rank: usize,
    context: usize,
    parallel: usize,
    session: qwen4_session.Options,
    own: serve_file.OwnOptions,
    args: server.cli.Args,
    max_queued: ?usize,
    path: []const u8,
    dir: std.Io.Dir,
    vocab: usize,
    eos: []const u32,
    text: ?*server.hf_text.HfText,
    segments: ?*segments_file.Segments,
    started: i96,
    load_start: std.Io.Timestamp,
    kept_bytes: usize,
};

/// `--parallel N` on two ranks (spec 094 WS-03b): each rank opens the pool over its share, rank 0's the hall
/// with its scheduler and the server, rank 1's following what rank 0 tells (`hall_ranks.zig`). No stream is a
/// lone session first, as one rank's are: the pool is opened at the load and the first stream is its own.
fn servePool(init: std.process.Init, startup: *command.Startup, model_type: []const u8, p: Pooled) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const channel = p.channel;
    const rank = p.rank;
    var lead: lead_file.Lead = .{ .channel = channel };
    var m = p.built;
    defer m.deinit();
    m.gather = .{ .nccl = p.comm };
    m.lead = if (rank == 0) &lead else null;
    // The split drafter and all stream contexts are allocated first. The target pool fit then counts
    // only its own buffers and taps, without a second estimate of the drafter's allocations.
    var flash: ?*qwen4_rig.Pool = if (p.session.drafter) |opened| qwen4_rig.Pool.create(arena, gpa, opened, m, p.context, p.parallel, qwen4_rig.drafter_stored) catch |err| return rankFailed(channel, rank, err) else null;
    defer if (flash) |held| held.destroy();
    const drafts: []const u8 = if (flash != null) "a split DFlash2 drafter with one context per stream" else if (m.ranks[0].mtp) |head| (if (head.split != null) "half of the MTP head" else "the MTP head") else if (rank == 0) "no drafts" else "no head";
    var link = qwen4_hall_ranks.Link.init(gpa, m, p.parallel, p.context) catch |err| return rankFailed(channel, rank, err);
    defer link.deinit();
    // The split drafter's allocations stand in `cuda.available`'s answer; the fit that follows counts only
    // the pool's own buffers and taps, without a second estimate of them. No drafter: the load asks no fit,
    // as it did where the refusal stood.
    var states = p.kept_bytes;
    if (p.session.drafter) |opened| {
        const tap_bytes = qwen4_hall_open.Rest.drafting(m.ranks[0], p.parallel, p.context, opened.config.taps.len).flash;
        const left = cuda.available(io) -| tap_bytes;
        if (p.args.prompt_cache_gib == null) states = @min(states, qwen4_hall_open.Fit.statesWithin(m.ranks[0], p.context, p.parallel, left, 0));
        const fit = qwen4_hall_open.Fit.of(m.ranks[0], p.context, p.parallel, left, 0, states);
        if (!fit.fits) {
            std.log.err("rank {d}: the pool of {d} streams of {d} tokens does not fit beside its drafter and {d} bytes of kept states: {d} bytes of extents, {d} of slots/windows/head, {d} of taps, {d} left before taps, {d} reserved", .{ rank, p.parallel, p.context, states, fit.bytes, fit.rest.bytes(), tap_bytes, left + tap_bytes, cuda.reserve });
            channel.fail("the shared target pool does not fit beside the drafter and kept states");
            return 1;
        }
    }
    const options: qwen4_hall.EngineOptions = .{
        .slots = p.parallel,
        .session = p.session,
        .pool = qwen4_hall.poolFor(p.context, p.parallel),
        .depth = p.own.depth,
        .chunk = p.own.prompt_chunk orelse m.max_rows,
        .lazy_graphs = p.own.lazy_graphs,
        // the load's captures are told through the link, after the barrier (`Link.captureFirst`)
        .capture_at_open = false,
        .states_bytes = states,
    };
    const said = struct {
        fn loaded(r: usize, at: std.Io.Timestamp, on: std.Io, context: usize, parallel: usize, head: []const u8) void {
            const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
            const load_s = @as(f64, @floatFromInt(at.untilNow(on, .awake).toNanoseconds())) / 1e9;
            std.log.info("rank {d} of 2: its share and the pool of {d} streams of {d} tokens, {d:.2} GiB of device buffers, in {d:.1} s; {s}", .{ r, parallel, context, gib, load_s, head });
        }
    };

    if (rank != 0) {
        var engine = qwen4_hall_engine.Engine.open(gpa, m.ranks[0], p.eos, options) catch |err| return rankFailed(channel, rank, err);
        defer engine.deinit();
        engine.bindPool();
        link.bind(&engine);
        if (flash) |held| {
            engine.seatRankDrafter(held) catch |err| return rankFailed(channel, rank, err);
            flash = null;
        }
        channel.barrier(1) catch |err| return rankGone(channel, rank, err);
        cuda.phases.mark("the pool is made");
        cuda.phases.mark("the pool is made");
        said.loaded(rank, p.load_start, io, p.context, p.parallel, drafts);
        const ran = qwen4_hall_ranks.follow(channel, &engine, &link) catch |err| return rankFailed(channel, rank, err);
        std.log.info("rank {d}: followed rank 0's {d} operations", .{ rank, ran });
        return 0;
    }

    const text = p.text.?;
    var frontend = cli.loadFrontend(gpa, io, p.dir, p.path, &text.tok) orelse return 1;
    defer frontend.deinit(gpa);
    var with = options;
    with.ranks = &link;
    const hall = qwen4_hall.Hall.create(gpa, io, m.ranks[0], &frontend.tokenizer, p.eos, with) catch |err| return rankFailed(channel, rank, err);
    var destroyed = false;
    defer if (!destroyed) hall.destroy();
    if (flash) |held| {
        hall.core.engine.?.seatRankDrafter(held) catch |err| return rankFailed(channel, rank, err);
        flash = null;
    }
    channel.barrier(1) catch |err| return rankGone(channel, rank, err);
    // the load's graphs, rank 1 capturing each as it is told (`hall_ranks.Link.captureFirst`); unless the
    // load was told to leave them to the step that first needs one
    if (!p.own.lazy_graphs) link.captureFirst() catch |err| return rankFailed(channel, rank, err);
    cuda.phases.mark("the pool is made");
    said.loaded(rank, p.load_start, io, p.context, p.parallel, drafts);
    var grammar: command.Grammar = undefined;
    try grammar.init(gpa, &frontend.tokenizer, p.vocab, p.eos);
    defer grammar.deinit();
    const name = server.hub.servedName(p.args.name, p.args.model, p.path);
    var served: served_file.Served = .{
        .gpa = gpa,
        .io = io,
        .frontend = &frontend,
        .compiler = grammar.compiler(),
        .behind = .{ .hall = hall, .depth = if (m.ranks[0].mtp == null and p.session.drafter == null) 0 else p.own.depth },
        .capacity = p.context,
        .vocab = @intCast(p.vocab),
        .lanes = @intCast(p.parallel),
        .max_queued = p.max_queued,
        .think = hall.think(),
        .draft_sources = .{ .copies = options.session.copies, .mtp = m.ranks[0].mtp != null and p.own.depth != 0, .drafter = if (p.session.drafter != null) .{ .kind = "DFlash2", .name = p.own.drafter.? } else null },
        .device_bytes = &cuda.account.bytes,
        // the two ranks' pool keeps its states within `--prompt-cache-gib`, as one rank's does (spec 160)
        .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = states, .explicit_budget = p.args.prompt_cache_gib != null },
    };
    const status = startup.run(gpa, io, p.args, .{
        .identity = command.identity(),
        .engine = served.engine(),
        .text = p.segments.?.text(),
        .served = name,
        .model_dir = p.path,
        .model_type = model_type,
        .sampling = try command.samplingDefaults(arena, io, p.path, p.args),
        .environ = init.environ_map,
        .started = p.started,
    });
    served.close();
    // the scheduler's thread has stopped: nothing more is told after the end
    hall.destroy();
    destroyed = true;
    held_file.told(&lead, .end, &.{}, &.{}) catch |err| return rankFailed(channel, rank, err);
    std.log.info("stopped", .{});
    return status;
}

fn rankGone(channel: *channel_file.Channel, rank: usize, err: channel_file.Error) u8 {
    if (err == error.PeerFailed) {
        std.log.err("rank {d}: the other rank failed: {s}", .{ rank, channel.said() });
    } else std.log.err("rank {d}: the control channel: {t}{s}{s}", .{ rank, err, if (channel.said().len > 0) ": " else "", channel.said() });
    return 1;
}

fn rankFailed(channel: *channel_file.Channel, rank: usize, err: anyerror) u8 {
    var buffer: [160]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "rank {d}: {t}", .{ rank, err }) catch "rank failed";
    std.log.err("{s}", .{text});
    channel.fail(text);
    return 1;
}
