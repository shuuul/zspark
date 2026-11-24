//! `zspark serve`: the checkpoint kept loaded behind TensorFold's HTTP server (spec 059). With `--rank R --at
//! ADDR:PORT` (or `rank=R at=ADDR:PORT`) it is one rank of two (spec 050), which `serve_ranks.zig` holds: rank 1
//! listens, loads its share and follows; rank 0 connects, loads and serves. The MTP head is not loaded on a
//! share. TensorFold's host
//! renders, tokenizes and speaks the three APIs; this file loads the model as `serve` always did and hands the
//! server `served.zig`'s engine. The flags are TensorFold's `serve` flags (`server.cli`), with the flags of
//! decoding that are zspark's own (`common.decode_flags`) read first. `main` is TensorFold's
//! `zig/tests/server/fake_serve.zig` and `zig/src/server/main.zig` with zspark's engine in the engine's place.
//!
//! `--drafter DIR` (spec 126, the owner's 2026-10-25 word) drafts by the DFlash2 drafter in DIR in the MTP
//! head's place for the whole load, as `generate` and `ranks-check` take it: the head is not loaded. One stream
//! is the session's drafter, several the pool's seats beside one another (`pool/hall_engine.zig`'s
//! `seatDrafter`). On two ranks each holds its share of DIR and each slot's drafter context; rank 1 follows
//! the passes rank 0 tells. Without it the MTP head drafts, as before; `--serial` drafts by neither and
//! leaves DIR unread.
//!
//! Where a flag of TensorFold's means something else here: `--parallel auto` is zspark's default of four
//! streams (TensorFold's is eight); `--no-drafts` is `--serial` (no drafting head is loaded);
//! `--prompt-cache-gib` bounds the pool's kept states and the sequences a lone session left, and where it is
//! not given they hold `hall.default_states_bytes` each, not TensorFold's 70% of memory: the pool's
//! extents are made at load and its ceiling is theirs (spec 064). `--context` is a stream's and the pool is
//! `--parallel` of them (`hall.poolFor`, spec 082): its extents' bytes and the rest's are said at the load,
//! and a pool the device cannot give is refused there (`hall_open.Fit`: the extents beside the session, then
//! the rest in its place, spec 084), where TensorFold sizes a stream's caches as it admits it. The states'
//! budget is held out of both steps, the default's as far as the pool leaves room (spec 101). `--learn` keeps the lone session's shared
//! prefixes' states on disk (`learnt.zig`), where TensorFold has it for GLM alone. `--spill-gib N` (0: off) and
//! `--spill-dir` are read here, before the host's words: TensorFold's host lists `--spill-gib` as a flag of
//! its Python engine and refuses it; here the states that leave the pool or a session go to disk under it
//! (`decode/spill.zig`, spec 074); `--learn-min-free-gib` (4) is the free disk both kinds' files leave, as
//! TensorFold's (`spill.Floor`, spec 101). `--prompt-chunk N` (1 to 4,096) is the most rows of a prompt a round of
//! the pool takes beside the replies it decodes (spec 074); unset, it takes what the replies leave of the
//! round's rows (`OwnOptions.prompt_chunk`). `--max-num-batched-tokens N` (2,048 to 4,096; on two ranks a
//! pair's, spec 111, and the pool's rounds, spec 149 WS-06) is the model's rows and a lone prompt's window, vLLM's
//! flag (spec 105 WS-07); unset, 4,096 on one rank (spec 148 WS-02) and in a pair on two (spec 149 WS-04), the
//! pool's rounds 2,048 (one Spark's whatever is asked). The three defaults are vLLM's nearest (the owner, 2026-10-16;
//! spec 101), each beside its field. `--speed-up`, `--backend`,
//! `--device` and `--segments` are parsed and not read, each with a line saying so. `--slide`,
//! `--slide-graph` and `--drafter-bits`, the host's since TensorFold 1.0.4, and `--chat-template`, its since
//! 1.0.5, are refused (`command.zig` `unserved`), as its 1.0.2 refused them; `--drafter` is zspark's own.

const std = @import("std");
const server = @import("server");
const cuda = @import("../../../cuda/cuda.zig");
const qwen4_attention = @import("../blocks/attention.zig");
const qwen4_hall = @import("../pool/hall.zig");
const qwen4_hall_open = @import("../pool/hall_open.zig");
const qwen4_layout = @import("../layout.zig");
const qwen4_learnt = @import("../session/learnt.zig");
const qwen4_model = @import("../model.zig");
const qwen4_session = @import("../session/session.zig");
const served_file = @import("../served.zig");
const command = @import("../../../server/command.zig");
const segments_file = @import("../../../server/segments.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const qwen4_rig = @import("../dflash2/rig.zig");
const flash_pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const cli = @import("../../../cli.zig");
const recall_file = @import("../../../decode/recall.zig");
const tokenizer = @import("tokenizer");
const common = @import("common.zig");
const qwen4_config = @import("../config.zig");
const qwen4_vision = @import("../vision.zig");
const text_vision = @import("../../../text/vision.zig");
const server_images = @import("../../../server/images.zig");
const serve_ranks = @import("serve_ranks.zig");

/// zspark's own flags: how it decodes, which no flag of TensorFold's says. `serve_ranks.zig` reads them too.
pub const OwnOptions = struct {
    serial: bool = false,
    no_copies: bool = false,
    lazy_graphs: bool = false,
    depth: usize = common.draft_depth,
    reply_stride: usize = (qwen4_session.Options{}).reply_stride,
    /// Null: not asked, and the storage is `serve`'s own choice for the checkpoint and the lanes.
    kv: ?qwen4_attention.Storage = null,
    /// GiB of states that left, on disk; 0: none is written or read. Off by default as vLLM's
    /// `--kv-offloading-size` (None: nothing leaves the device) and SGLang's `--enable-hierarchical-cache`
    /// (false) are (vLLM 3313825, SGLang 1291f31).
    spill_gib: f64 = 0,
    spill_dir: ?[]const u8 = null,
    /// GiB a disk keeps free under `--learn`'s and `--spill-gib`'s files: TensorFold 1.0.12's flag and
    /// default (`lanes/learned_disk.zig`), read here and not by the host (`command.zig` refuses it for GLM).
    learn_min_free_gib: f64 = 4,
    /// The rows the model is loaded for and a lone prompt's window runs (`common.promptRows`, spec 105 WS-07):
    /// vLLM's `--max-num-batched-tokens`, a step's tokens. Null: the checkpoint's, 4,096 on one rank (spec 148
    /// WS-02: a 16k cold prompt's experts read half as often as at 2,048, for about 2.6 GiB of windows) and
    /// on two ranks a pair's, 4,096 too (`common.pairRows`, spec 149 WS-04), and the pool's rounds' where asked
    /// (`common.poolRows`, 2,048 unset; spec 149 WS-06).
    max_num_batched_tokens: ?usize = null,
    /// Rows of a prompt in a round of the pool that decodes replies, at most. Null: what the replies' windows
    /// leave of the rows the model is loaded for (`hall.rowsFor`: 2,048 on FP4-Local), the round being one budget of rows that the replies take first and the prompt the rest. That
    /// is vLLM's chunked prefill (`max_num_batched_tokens`, decodes scheduled before a prefill's chunk) and
    /// SGLang's (`chunked_prefill_size`); both give 8,192 tokens a step on a device of GB10's memory (vLLM's
    /// API server at 70 GiB and more, not an A100; SGLang's 90 to 160 GB band). Not 8,192 here: a round's
    /// buffers grow with its rows (11.2 GiB of windows at 2,048 on NVFP4, spec 084) and four times that does
    /// not fit beside the defaults' pool, so the budget is the model's rows. Until spec 101 it was 512.
    prompt_chunk: ?usize = null,
    /// Recall (`decode/recall.zig`, spec 106): drafts copied from a record of earlier requests' ids, kept
    /// under `--recall-dir` or `~/.cache/zspark/recall/`, `--recall-gib` of disk; `--recall-read-only` reads
    /// it and adds nothing. TensorFold 1.0.12's flags, its defaults; read here and not by the host, so GLM's
    /// `serve` refuses them still (`command.unserved`). Off by default, as upstream.
    recall: bool = false,
    recall_dir: ?[]const u8 = null,
    recall_gib: f64 = (recall_file.Asked{}).gib,
    recall_read_only: bool = false,
    /// Warm turns (`pool/hall_warm.zig`, spec 106 WS-07): each chat reply's next turn prefilled in the
    /// pool while nothing else is live, TensorFold 1.0.12's `warm_turns`. Off by default (the owner,
    /// 2026-10-17); a pool's, so `--parallel 1` runs none.
    warm_turns: bool = false,
    /// `--drafter DIR`: the DFlash2 drafter in DIR drafts in the MTP head's place (spec 126), as `generate`'s
    /// `drafter=DIR`. Null: the MTP head drafts, the default.
    drafter: ?[]const u8 = null,

    /// `serve`'s own beside `common.decode_flags`.
    pub const flags = common.decode_flags ++ [_]cli.Flag{
        .{ .name = "--spill-gib", .field = "spill_gib" },
        .{ .name = "--spill-dir", .field = "spill_dir" },
        .{ .name = "--learn-min-free-gib", .field = "learn_min_free_gib" },
        .{ .name = "--prompt-chunk", .field = "prompt_chunk" },
        .{ .name = "--recall", .field = "recall" },
        .{ .name = "--recall-dir", .field = "recall_dir" },
        .{ .name = "--recall-gib", .field = "recall_gib" },
        .{ .name = "--recall-read-only", .field = "recall_read_only" },
        .{ .name = "--warm-turns", .field = "warm_turns" },
        .{ .name = "--drafter", .field = "drafter" },
    };

    /// Those of `flags` that take no value.
    const switches = [_][]const u8{ "--serial", "--no-copies", "--lazy-graphs", "--recall", "--recall-read-only", "--warm-turns" };

    /// Whether Recall is asked: `--recall` or a flag of its, which imply it as TensorFold's do.
    pub fn recalling(self: OwnOptions) bool {
        return self.recall or self.recall_dir != null or self.recall_read_only or self.recall_gib != (recall_file.Asked{}).gib;
    }

    /// A word of `serve`'s: one of `flags`, or TensorFold's.
    pub fn kindOf(word: []const u8) command.Kind {
        for (flags) |each| {
            if (std.mem.eql(u8, each.name, word)) break;
        } else return .host;
        for (switches) |name| {
            if (std.mem.eql(u8, name, word)) return .own;
        }
        return .own_valued;
    }
};

/// The shortest `--context` (`serve_ranks.zig` reads it too).
pub const least_context: usize = 64;

/// Loads the checkpoint once and serves it until SIGINT or SIGTERM.
pub fn run(init: std.process.Init, _: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const words = argv[1..];
    const arena = init.arena.allocator();
    var place: join_file.Place = .{};
    const rest = join_file.rest(arena, words, &place) catch {
        std.log.err("serve: --rank and --at each take a value", .{});
        return 2;
    };
    switch (command.ranked("serve", place, nccl.linked, .flags)) {
        .exit => |code| return code,
        .paired => return serve_ranks.serve(init, path, rest, place),
        .alone => {},
    }
    const io = init.io;
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    const parted = try command.part(arena, words, OwnOptions.kindOf);
    const own = cli.parse(OwnOptions, &OwnOptions.flags, .{ .quoted = false }, parted.own) orelse return 2;
    if (!command.checkSpill(own.spill_gib, own.learn_min_free_gib, own.spill_dir)) return 2;
    // `cli.Flag`'s bounds are for a field that is not optional
    if (own.prompt_chunk) |rows| if (rows < 1 or rows > qwen4_session.most_prompt_rows) {
        std.log.err("--prompt-chunk {d}: rows from 1 to {d}", .{ rows, qwen4_session.most_prompt_rows });
        return 2;
    };
    if (!(own.recall_gib >= recall_file.least_gib and own.recall_gib < 1 << 20)) {
        std.log.err("--recall-gib {d}: at least 1/128 GiB (two 4 MiB chunks)", .{own.recall_gib});
        return 2;
    }
    const bounded = try command.bounded(arena, parted.rest) orelse return 2;
    const args = try command.hostArgs(arena, path, bounded.rest) orelse return 2;
    if (own.recalling() and (own.no_copies or own.serial or args.no_drafts or own.depth == 0)) {
        std.log.err("serve: --recall drafts through the copies, and {s} drafts none", .{if (own.no_copies) "--no-copies" else "a serial reply"});
        return 2;
    }
    const parallel = command.parallelWithin(args, qwen4_hall.default_parallel, 0, qwen4_hall.most_parallel) orelse return 2;
    const context = (command.askedContext(args, least_context) catch return 2) orelse qwen4_hall.default_context;
    const serial = own.serial or args.no_drafts;
    command.warnUnread(args, "is not read by zspark yet", &.{});
    command.logVersion();
    server.log.init(io, false);
    var startup = try command.Startup.open(arena, init.environ_map) orelse return 1;
    defer startup.deinit();

    var opened: common.Opened = .{};
    defer opened.close(io);
    if (!try opened.enter(io, path, .say) or !try opened.configure(init) or !try opened.resolve(init)) return 1;
    const config = opened.config;
    const found = opened.layout;
    // `--drafter`: opened and held to the model before anything loads; the MTP head is then not loaded
    var drafting: qwen4_rig.Opened = .{};
    defer drafting.close(io);
    const flash = own.drafter != null and !serial and own.depth > 0;
    if (own.drafter != null and !flash) std.log.warn("serve: --drafter with {s}: no drafter is loaded", .{if (serial) "a serial reply" else "--depth 0"});
    if (flash) if (!try qwen4_rig.enter(init, &drafting, own.drafter.?, &config, config.layers)) return 1;
    const prompt_rows = common.promptRows(config, own.max_num_batched_tokens) orelse return 2;
    const caches = qwen4_layout.storageFor(config, parallel, 1, own.kv);
    const load_options = try common.loadOptions(arena, config, qwen4_hall.rowsFor(prompt_rows, parallel), !serial and own.depth > 0 and !flash, caches);
    // the first parts are read while the tokenizer and the template load (`Reading.prefetch`), as two ranks' during
    // their join
    var reading: qwen4_model.Reading = .init(init.gpa, found, config.layers, null);
    var reading_held = true;
    defer if (reading_held) reading.deinit();
    reading.mtp = qwen4_model.Model.mtpReading(qwen4_model.Model.drafts(found, load_options), false);
    reading.prefetch();
    const text = command.loadText(init.gpa, io, path, arena) orelse return 1;
    defer text.deinit();
    // a held turn's prompt is encoded once: the host is handed its own text with the segments' ids kept
    var segments: segments_file.Segments = try .init(init.gpa, io, text.text(), &text.tok, segments_file.default_bytes);
    defer segments.deinit();
    // tokenizer.json is read once: zspark's front end stands on the host's tokenizer, and is freed before it
    var frontend = cli.loadFrontend(init.gpa, io, opened.dir.?, path, &text.tok) orelse return 1;
    defer frontend.deinit(init.gpa);
    command.warnEnds(text.eos, config.eos);
    cuda.phases.mark("the model's load begins");
    var with_reading = load_options;
    with_reading.reading = &reading;
    var model: qwen4_model.Model = try .loadWith(init.gpa, config, found, with_reading);
    defer model.deinit();
    // its stages go before the session's buffers are made
    reading.deinit();
    reading_held = false;
    if (model.attention) |held| std.log.info("attention caches {s}, the drafting head's {s}", .{ held.shape.kv.name(), held.head.name() });
    const model_bytes = cuda.account.bytes;
    cuda.phases.mark("the model is loaded");
    if (cuda.account.trace) cuda.account.mark("model");
    var grammar: command.Grammar = undefined;
    try grammar.init(init.gpa, &frontend.tokenizer, config.vocab, config.eos);
    defer grammar.deinit();
    const name = server.hub.servedName(args.name, args.model, path);

    // `--parallel 1` is the session: contiguous caches, its kernels and graph keys. Above that the replies
    // share rounds, and the attention caches are the pool.
    var session_options = qwen4_hall.sessionFor(prompt_rows, context, own.reply_stride);
    session_options.copies = !own.no_copies;
    // the lone stream's session loads it, and the pool takes it when a second stream comes
    if (flash) session_options.drafter = &drafting;
    session_options.drafter_stored = qwen4_rig.drafter_stored;
    const kept_bytes: usize = if (args.prompt_cache_gib) |gib| @intFromFloat(@min(@max(gib, 0), 1024) * (1 << 30)) else qwen4_hall.default_states_bytes;
    session_options.parked_bytes = kept_bytes;
    // `--recall`: one record for the session and the pool, opened before either takes a request
    const recall: ?*recall_file.Store = if (own.recalling()) opened: {
        const identity: recall_file.Identity = .{ .digest = try tokenizer.identity(&text.tok, arena), .vocab = text.text().vocabSize(), .name = name };
        const asked: recall_file.Asked = .{ .dir = own.recall_dir, .gib = own.recall_gib, .read_only = own.recall_read_only };
        var why: []const u8 = "";
        break :opened try recall_file.open(init.gpa, io, arena, identity, asked, init.minimal.environ.getPosix("HOME"), &why) orelse {
            std.log.err("{s}", .{why});
            return 1;
        };
    } else null;
    defer if (recall) |store| store.close();
    session_options.recall = recall;
    // closed after the session and the pool, which write what they hold as they go (`--spill-gib`)
    var learning: ?*qwen4_learnt.Learning = null;
    defer if (learning) |held| held.close();
    // Images (spec 019 WS-04, WS-05): on one rank of FP4-Local, the tower loaded before the session and the pool,
    // so the pool's fit is said with it on the device; where it cannot be, the server serves text and an
    // image part is its 400
    var tower: ?qwen4_vision.Tower = null;
    defer if (tower) |*held| held.deinit();
    var eye: ?served_file.Eye = null;
    defer if (eye) |*held| held.deinit();
    if (config.servesImages()) images: {
        var refused: text_vision.Problem = .{};
        const settings = (text_vision.Settings.load(arena, io, opened.dir.?, &refused) catch |err| {
            std.log.warn("images: off, the checkpoint's image settings: {s}", .{if (refused.len > 0) refused.message() else @errorName(err)});
            break :images;
        }) orelse break :images;
        var diagnostic: qwen4_config.Diagnostic = .{};
        tower = qwen4_vision.Tower.load(init.gpa, arena, &opened.checkpoint.?, qwen4_vision.max_patches_default, &diagnostic) catch |err| {
            std.log.warn("images: off, the vision tower: {s}", .{if (diagnostic.message.len > 0) diagnostic.message else @errorName(err)});
            break :images;
        };
        // the pool's slots' rotary tables beside it, none for one session
        eye = try .init(init.gpa, &tower.?, settings, if (parallel > 1) parallel else 0, session_options.capacity);
        if (cuda.account.trace) cuda.account.mark("tower");
        cuda.phases.mark("the vision tower is loaded");
        std.log.info("images: on, {d} images and {d} image tokens a request at most", .{ server_images.max_images, server_images.image_tokens });
    }
    var session: ?*qwen4_session.Session = null;
    defer if (session) |held| {
        qwen4_learnt.flush(held, true);
        held.destroy();
    };
    var hall: ?*qwen4_hall.Hall = null;
    defer if (hall) |held| held.destroy();
    if (parallel <= 1) {
        if (own.warm_turns) std.log.warn("serve: --warm-turns runs in the pool; --parallel 1 runs no warm pass", .{});
        session = try qwen4_session.Session.create(init.gpa, io, model, &frontend.tokenizer, config.eos, session_options);
        if (cuda.account.trace) cuda.account.mark("session");
        if (!own.lazy_graphs) try session.?.prepare(own.depth, qwen4_session.prepared_buckets);
    } else {
        // the lone stream's windows and the model's rows are `prompt_rows` (4,096 unless asked); the pool's rounds
        // stay `session.promptRows`, whose buffers are made for every stream (13.5 GiB at 2,048 rows on
        // FP4-Local, 25.1 at 4,096: the defaults' pool would not fit), so its figures are for those
        var pooled = model;
        pooled.max_rows = qwen4_hall.rowsFor(qwen4_session.promptRows(config), parallel);
        if (own.prompt_chunk) |rows| if (rows > pooled.max_rows) {
            std.log.err("--prompt-chunk {d}: a round of {d} streams has {d} rows on this checkpoint", .{ rows, parallel, pooled.max_rows });
            return 2;
        };
        const before_session = cuda.account.bytes;
        hall = try qwen4_hall.Hall.create(init.gpa, io, model, &frontend.tokenizer, config.eos, .{
            .slots = parallel,
            .session = session_options,
            .pool = qwen4_hall.poolFor(context, parallel),
            .depth = own.depth,
            .chunk = own.prompt_chunk orelse pooled.max_rows,
            .rows = pooled.max_rows,
            .lazy_graphs = own.lazy_graphs,
            .states_bytes = kept_bytes,
            .sight = if (eye) |*held| &held.sight else null,
        });
        if (cuda.account.trace) cuda.account.mark("hall");
        cuda.phases.mark("the pool is made");
        // The pool opens whole at the second stream: whether it will fit is said now (`hall_open.Fit`),
        // its extents beside this session and the rest of it in the session's place, the states' budget
        // held out of both. A budget given (`--prompt-cache-gib`) is held whole or the load refused; the
        // default is held as far as the pool leaves room for it, as vLLM's cache is what its load leaves
        // (spec 101), and set so in the session and the pool before any request comes.
        // DFlash2 in the pool: its taps of a round's rows and a seat of the drafter's context a stream, both
        // ranks' in this process, held out of what is left (`hall_open.Rest.drafting`, `SeatsOf.bytes`)
        const flash_bytes: usize = if (hall.?.core.session.?.flash) |held| qwen4_hall_open.Rest.drafting(pooled, parallel, context, held.d.c.taps.len).flash + parallel * dflash2_pass.world * flash_pass.SeatsOf(2).bytes(&held.pair) else 0;
        if (flash_bytes > 0) std.log.info("DFlash2 in the pool: {d:.2} GiB of taps and seats, held out of what the device has left", .{@as(f64, @floatFromInt(flash_bytes)) / (1024.0 * 1024.0 * 1024.0)});
        const left = cuda.available(io) -| flash_bytes;
        const session_bytes = cuda.account.bytes - before_session;
        const given = args.prompt_cache_gib != null;
        const states = if (given) kept_bytes else @min(kept_bytes, qwen4_hall_open.Fit.statesWithin(pooled, context, parallel, left, session_bytes));
        if (states < kept_bytes) {
            hall.?.options.states_bytes = states;
            hall.?.core.options.states_bytes = states;
            hall.?.core.session.?.lot.cap = states;
        }
        const fit: qwen4_hall_open.Fit = .of(pooled, context, parallel, left, session_bytes, states);
        const gib = 1024.0 * 1024.0 * 1024.0;
        const pool_gib = @as(f64, @floatFromInt(fit.bytes)) / gib;
        const rest_gib = @as(f64, @floatFromInt(fit.rest.bytes())) / gib;
        const session_gib = @as(f64, @floatFromInt(fit.session)) / gib;
        const left_gib = @as(f64, @floatFromInt(fit.left)) / gib;
        const states_gib = @as(f64, @floatFromInt(fit.states)) / gib;
        const kept_gib = @as(f64, @floatFromInt(cuda.reserve)) / gib;
        std.log.info("the pool of {d} streams of {d} tokens: {d} positions, {d:.2} GiB of extents and {d:.2} GiB of slots, windows and the head's in the session's {d:.2}, opened whole when a second stream comes, and {d:.2} GiB of states (--prompt-cache-gib{s}); {d:.2} GiB left on the device with {d:.0} kept free", .{ parallel, context, fit.positions, pool_gib, rest_gib, session_gib, states_gib, if (given) "" else " unset", left_gib, kept_gib });
        // what vLLM says of its cache, at info; a warning where no state can be kept
        if (fit.fits and states < kept_bytes) {
            const said = "--prompt-cache-gib unset: the states hold {d:.2} GiB, what the pool leaves on the device, short of the default's {d:.2}; a shorter --context or fewer --parallel leaves more";
            const values = .{ states_gib, @as(f64, @floatFromInt(kept_bytes)) / gib };
            if (states == 0) std.log.warn(said, values) else std.log.info(said, values);
        }
        if (!fit.fits) {
            var longest: [48]u8 = undefined;
            const shorter = if (fit.context >= least_context) std.fmt.bufPrint(&longest, "--context {d} at most", .{fit.context}) catch unreachable else "no --context";
            std.log.err("--parallel {d} of --context {d} does not fit: the pool's {d} positions take {d:.2} GiB of extents, then {d:.2} GiB of slots, windows and the head's in the session's {d:.2}, beside {d:.2} GiB of states (--prompt-cache-gib), and {d:.2} GiB are left on the device with {d:.0} kept free; --parallel {d} at most at this context, or {s} for {d} streams{s}", .{ parallel, context, fit.positions, pool_gib, rest_gib, session_gib, states_gib, left_gib, kept_gib, fit.parallel, shorter, parallel, if (given) ", or a smaller --prompt-cache-gib" else "" });
            return 1;
        }
    }
    // `--learn`: the lone stream's session reads and writes a shared prefix's state, the pool reads it once
    // open (specs 064, 101). `--spill-gib`: the states that leave either go to disk, and both read them (spec 074).
    if (args.learn or own.spill_gib > 0) learning = try qwen4_learnt.open(session orelse hall.?.core.session.?, command.askedStates(init, path, if (args.learn) .{ .dir = args.learn_dir, .gib = args.learn_gib } else null, own.spill_gib, own.spill_dir, own.learn_min_free_gib));
    if (learning) |held| if (held.spill) |*kept| {
        if (session) |one| qwen4_learnt.spilling(one, &kept.store) else hall.?.core.spilling(&kept.store);
    };
    const prepared_count, const prepared_ns = if (session) |held| counted: {
        const got = held.captured();
        break :counted .{ got.count, got.nanoseconds };
    } else counted: {
        const got = hall.?.captured();
        break :counted .{ got.count, got.nanoseconds };
    };
    if (cuda.account.trace) cuda.account.mark("graphs");
    cuda.phases.mark("the session and the pool are made");
    std.log.info("{s} loaded, {d} graphs captured in {d:.0} ms ({d} tokens of context, {d} together{s})", .{
        name, prepared_count, @as(f64, @floatFromInt(prepared_ns)) / 1e6, context, parallel, if (flash) ", drafted by DFlash2" else if (model.mtp == null) ", no drafts" else "",
    });
    std.log.info("device buffers: {d} MiB in {d}, {d} MiB of them the model's and {d} MiB the session's", .{
        cuda.account.bytes >> 20, cuda.account.buffers, model_bytes >> 20, (cuda.account.bytes - model_bytes) >> 20,
    });

    var served: served_file.Served = .{
        .gpa = init.gpa,
        .io = io,
        .frontend = &frontend,
        .compiler = grammar.compiler(),
        .behind = .{ .session = session, .hall = hall, .depth = if (model.mtp == null and !flash) 0 else own.depth, .eye = if (eye) |*held| held else null },
        .capacity = session_options.capacity,
        .vocab = @intCast(config.vocab),
        .lanes = @intCast(parallel),
        .max_queued = bounded.max_queued,
        .warm_turns = own.warm_turns and hall != null,
        // a row's log probabilities need its whole vocabulary on one device: two ranks hold half each
        .logprobs = true,
        .vision = if (eye) |*held| held.offer.vision() else null,
        .think = if (session) |held| held.think else hall.?.think(),
        // under --serial/--no-drafts or --depth 0 nothing drafts; DFlash2 takes the MTP head's place under --drafter
        .draft_sources = if (serial or own.depth == 0) .{} else .{ .copies = !own.no_copies, .mtp = model.mtp != null and !flash, .drafter = if (flash) .{ .kind = "DFlash2", .name = own.drafter.? } else null },
        .device_bytes = &cuda.account.bytes,
        // the budget the states were given, in the host's terms for /memory and /metrics (spec 160)
        .prompt_cache_plan = .{ .source = .explicit, .budget_bytes = if (hall) |held| held.options.states_bytes else session_options.parked_bytes, .explicit_budget = args.prompt_cache_gib != null },
    };
    // before the session and the pool go: every request's thread has left them
    defer served.close();
    const status = startup.run(init.gpa, io, args, .{
        .identity = command.identity(),
        .engine = served.engine(),
        .text = segments.text(),
        .served = name,
        .model_dir = path,
        .model_type = config.model_type,
        .sampling = try command.samplingDefaults(arena, io, path, args),
        .environ = init.environ_map,
        .started = started,
        .recall = recall,
    });
    if (cuda.account.trace) cuda.account.mark("stopped");
    std.log.info("stopped", .{});
    return status;
}
