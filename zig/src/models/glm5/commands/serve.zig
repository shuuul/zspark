//! `zspark serve DIR --rank R --at ADDR:PORT`: GLM-5.3-Flash kept loaded on two ranks behind TensorFold's HTTP
//! server (spec 045 WS-04; Flash Next's is `../../qwen4/commands/serve.zig`, spec 059). Rank 1 is started
//! first, listens at ADDR:PORT, loads its share and follows; rank 0 connects there, loads, and serves: the
//! host renders, tokenizes and speaks the three APIs, and `served.zig`'s engine decodes on the session. Both
//! are given the same flags: what changes a rank's bits is compared when they join. SIGINT or SIGTERM on rank
//! 0 ends both in order; either rank's end ends the other.
//!
//! The whole model does not fit one process, so without a rank the command refuses it and names this form;
//! `--layers L` under the model's count serves the first layers with both ranks in the process, to try the
//! server (its replies are no text).
//!
//! The flags are TensorFold's `serve` flags (`server.cli`) and zspark's own: `--rank`, `--at` (or the words
//! `rank=R at=ADDR:PORT`, as `tools/two_ranks.sh` writes them), `--layers`, `--rows` (a prompt window's rows),
//! `--drafter DIR` (DFlash2 in the MTP layer's place), `--serial`, `--cap N` (the most drafts of a round, the
//! MTP layer's, 5 on two ranks, or DFlash2's), `--confidence F` (the product a chain stops under, 0.65 on two ranks) and
//! `--kv-dtype bf16|fp8` (the DSA caches' latents: fp8 unless asked; the MTP
//! layer's stays bf16; spec 107).
//! `--parallel N` above 1 decodes N replies at once (`../pool/hall.zig`; spec 068 WS-04), each of `--context` tokens, and is refused at the load where N slots do not fit the device; `auto` is 1.
//! `--no-drafts` is `--serial`. `--prompt-cache-gib`, `--learn`, `--speed-up`, `--backend`, `--device`
//! and `--segments` are parsed and not read, each with a line saying so. `--slide`, `--slide-graph` and
//! `--drafter-bits`, the host's since TensorFold 1.0.4, and `--chat-template`, its
//! since 1.0.5, are refused (`command.zig` `unserved`); `--drafter` is zspark's own here and never the host's.
//! `--spill-gib N` (0: off), `--spill-dir` and `--learn-min-free-gib` (4) are read here, as Flash Next's: the
//! states that leave the session go to disk on each rank (`../session/learnt.zig`, spec 108 WS-04); under
//! `--parallel` above 1 they are said not to be read, the pool keeping no states.
//! `--weights-from-peer`, on both ranks: the checkpoint is rank 0's alone. Rank 0 hands rank 1 its config and its
//! files' headers before they join, then reads rank 1's share of each part beside its own and sends it over the
//! link (`../session/pack.zig` `Line`); rank 1 opens nothing under DIR. With `--drafter DIR` on both, the drafter's
//! config and header go with the checkpoint's, and its file whole over the link at the load (`rig.Source.cross`):
//! rank 1 opens nothing under its drafter's DIR either, and nothing of it is written on either rank.

const std = @import("std");
const server = @import("server");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const glm5_layout = @import("../layers/layout.zig");
const served_file = @import("../session/served.zig");
const command = @import("../../../server/command.zig");
const segments_file = @import("../../../server/segments.zig");
const rig = @import("../dflash2/rig.zig");
const session_file = @import("../session/session.zig");
const snapshot_file = @import("../session/snapshot.zig");
const learnt = @import("../session/learnt.zig");
const dsa = @import("../layers/dsa.zig");
const common = @import("common.zig");
const config_file = @import("../layers/config.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const world = glm5_layout.world;

/// Tokens a prompt and its reply share unless `--context` says: 262,144, once the indexer and the extent
/// pool allocate the cache (spec 046; the owner, 2026-10-06). A context the device cannot hold is refused
/// at the load, on both ranks, before the cache is allocated.
pub fn defaultContext(config: anytype) usize {
    _ = config;
    return 262144;
}
/// A prompt window's rows unless `--rows` says, the same bits whatever the window. FP4-Local's 4,096 (spec 114
/// WS-08): a 16,000-token cold prompt 9.4 to 10.4 s against 10.65 to 10.8 in windows of 2,048, 4,000 tokens 2.3
/// to 2.65 against 2.62 to 2.65, every other request of the client set and decoding within the loads' spread, 7.2
/// GiB more a rank (2,048 itself 1.63 to 1.67 times 512's rate, spec 107 WS-03).
pub fn defaultRows(config: *const config_file.Config) usize {
    _ = config;
    return 4096;
}
/// The MTP layer's most drafts a round on two ranks unless `--cap` says: on the whole model a Spark a rank, the
/// held-out chains prompt runs 41.9 to 42.5 tokens/s at 5 against 38.2 to 38.6 at the head's room 15, code 27.2
/// against 26.8 (spec 099; the owner, 2026-10-17). DFlash2 keeps its own.
const two_rank_mtp_cap = 5;
/// The MTP layer's confidence on two ranks: a round there is bound by the weights' bytes, so a draft must be likelier
/// kept than the EXL3 pack's 0.35 (retired) asked to pay its step. The client set, every reply, both Sparks (spec 114 WS-02): 0.35
/// 40.95 tok/s, 0.5 41.6, 0.65 42.15, 0.8 42.2, 0.9 41.5, the same bits. DFlash2 keeps its own.
const two_rank_mtp_confidence = 0.65;
/// Replies at once at most (`--parallel`), as Flash Next's; what fits the device is asked at the load.
const most_parallel = 64;
/// A window's rows at most: its tokens and a forward's words in the control channel's widest frame.
const most_rows = 4096;
comptime {
    std.debug.assert(most_rows + 8 <= @import("../../../ranks/lead.zig").Lead.widest);
}

const Own = struct {
    layers: ?usize = null,
    rows: ?usize = null,
    serial: bool = false,
    /// `--weights-from-peer`: rank 1's share comes from rank 0's checkpoint over the link.
    from_peer: bool = false,
    /// The DFlash2 drafter's directory: it drafts in the MTP layer's place.
    drafter: ?[]const u8 = null,
    /// The most drafts of a round, and the product a chain stops under; null the drafter's own defaults.
    cap: ?usize = null,
    confidence: ?f32 = null,
    /// The DSA caches' latents (`--kv-dtype bf16|fp8`, spec 107 WS-03); null is the checkpoint's (`Storage.stored`).
    kv: ?dsa.Storage = null,
    /// GiB of states that left, on disk (`../session/learnt.zig`, spec 108 WS-04); 0: none is written or
    /// read, the default as Flash Next's (vLLM's `--kv-offloading-size`, SGLang's hierarchical cache: off).
    spill_gib: f64 = 0,
    spill_dir: ?[]const u8 = null,
    /// GiB a disk keeps free under `--spill-gib`'s files: TensorFold 1.0.12's flag and default, as Flash Next's.
    learn_min_free_gib: f64 = 4,

    const flags = [_]cli.Flag{
        .{ .name = "--layers", .field = "layers", .min = 1 },
        .{ .name = "--rows", .field = "rows", .min = 1 },
        .{ .name = "--serial", .field = "serial" },
        .{ .name = "--weights-from-peer", .field = "from_peer" },
        .{ .name = "--drafter", .field = "drafter" },
        .{ .name = "--cap", .field = "cap", .min = 1 },
        .{ .name = "--confidence", .field = "confidence", .min = 0 },
        .{ .name = "--kv-dtype", .field = "kv" },
        .{ .name = "--spill-gib", .field = "spill_gib" },
        .{ .name = "--spill-dir", .field = "spill_dir" },
        .{ .name = "--learn-min-free-gib", .field = "learn_min_free_gib" },
    };

    /// Those of `flags` that take no value.
    const switches = [_][]const u8{ "--serial", "--weights-from-peer" };

    /// A word of `serve`'s: one of `flags`, or TensorFold's.
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
    const arena = init.arena.allocator();
    const io = init.io;
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    const path = words[0];
    var place: join_file.Place = .{};
    const rest = join_file.rest(arena, words[1..], &place) catch {
        std.log.err("serve: --rank and --at each take a value", .{});
        return 2;
    };
    const parted = try command.part(arena, rest, Own.kindOf);
    const own = cli.parse(Own, &Own.flags, .{ .quoted = false, .prefix = "serve: " }, parted.own) orelse return 2;
    if (!command.checkSpill(own.spill_gib, own.learn_min_free_gib, own.spill_dir)) return 2;
    const bounded = try command.bounded(arena, parted.rest) orelse return 2;
    const args = try command.hostArgs(arena, path, bounded.rest) orelse return 2;
    // `--parallel auto` is one reply at a time, not the repository's four (spec 082 WS-02): slots launch every
    // window where the session runs captured graphs, and one reply alone among slots is not timed yet
    const parallel = command.parallelWithin(args, 1, 1, most_parallel) orelse return 2;
    const asked_context = command.askedContext(args, 64) catch return 2;
    const pairing = command.ranked("serve", place, nccl.linked, .flags);
    if (pairing == .exit) return pairing.exit;
    const paired = pairing == .paired;
    if (own.from_peer and !paired) {
        std.log.err("--weights-from-peer: rank 1's share from rank 0's checkpoint, on two ranks (--rank, --at)", .{});
        return 2;
    }
    command.warnUnread(args, "is not read by zspark's GLM-5.3-Flash yet", &.{
        .{ .name = "--prompt-cache-gib", .given = args.prompt_cache_gib != null },
        .{ .name = "--learn", .given = args.learn },
    });
    command.logVersion();
    server.log.init(io, false);

    var opened: common.Opened = .{};
    defer opened.close(io);
    // rank 1's drafter under --weights-from-peer: rank 0's, handed with its checkpoint
    var peer_drafter: ?Given = null;
    defer if (peer_drafter) |*given| given.checkpoint.close();
    if (own.from_peer and place.rank.? != 0) {
        // rank 0's config and headers, and its drafter's, before the ranks join; DIR and the drafter's are only names here
        var peer = try peerCheckpoint(init, place);
        if (!try opened.given(init, path, peer.main.config, peer.main.checkpoint)) {
            if (peer.drafter) |*given| given.checkpoint.close();
            return 1;
        }
        peer_drafter = peer.drafter;
        if ((peer_drafter != null) != (own.drafter != null)) {
            std.log.err("--weights-from-peer: rank 0 {s} a drafter and this rank {s}: give both ranks --drafter or neither", .{ if (peer_drafter != null) "sent" else "sent no", if (own.drafter != null) "has one" else "has none" });
            return 1;
        }
    } else if (!try opened.enter(io, path, .say) or !try opened.configure(init) or !try opened.resolve(init)) return 1;
    const config = &opened.config;
    const rows = own.rows orelse defaultRows(config);
    // a window's tokens go to rank 1 in one frame
    if (rows > most_rows) {
        std.log.err("--rows {d}: {d} at most", .{ rows, most_rows });
        return 2;
    }
    if (parallel > 1 and !session_file.SessionOf(1).Hall.holds(rows, parallel)) {
        std.log.err("--parallel {d}: a round of windows of {d} rows holds fewer replies; two rows a reply at least (--rows)", .{ parallel, rows });
        return 2;
    }
    const layers = own.layers orelse config.layers;
    if (layers > config.layers) {
        std.log.err("--layers {d}: the model has {d}", .{ layers, config.layers });
        return 2;
    }
    if (!paired and layers == config.layers) {
        std.log.err("GLM-5.3-Flash does not fit one process: it is served by two ranks, a Spark each. Start `zspark serve {s} --rank 1 --at ADDR:PORT` on the second, then the same with `--rank 0` and the server's flags on the first, ADDR:PORT being where rank 1 listens on the link between them", .{path});
        return 1;
    }
    const context = asked_context orelse defaultContext(config);
    // the pool keeps no states, a slot's caches alone (`../pool/hall.zig`): none leaves it to be written
    if (own.spill_gib > 0 and parallel > 1) std.log.warn("--spill-gib is not read under --parallel {d}: the replies' slots keep no states to write", .{parallel});
    const spilling = own.spill_gib > 0 and parallel == 1;
    const asked = command.askedStates(init, path, null, own.spill_gib, own.spill_dir, own.learn_min_free_gib);
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    if (own.drafter) |drafter| {
        if (peer_drafter) |given| {
            // `drafting` holds its checkpoint from here
            peer_drafter = null;
            if (!try rig.given(init, &drafting, drafter, given.config, given.checkpoint, config, layers)) return 1;
        } else if (!try rig.enter(init, &drafting, drafter, config, layers)) return 1;
    }
    if (own.from_peer and place.rank.? == 0) try handCheckpoint(init, &opened, if (own.drafter != null) &drafting else null, place);
    const options: session_file.Options = .{ .layers = layers, .rows = rows, .capacity = context, .serial = own.serial or args.no_drafts, .confidence = own.confidence orelse if (paired and own.drafter == null) two_rank_mtp_confidence else null, .cap = own.cap orelse if (paired and own.drafter == null) two_rank_mtp_cap else null, .drafter = if (own.drafter != null) &drafting else null, .parallel = parallel, .snapshots = snapshot_file.held_default, .stride = snapshot_file.stride_default, .kv = own.kv orelse .stored(config), .spill = spilling, .from_peer = own.from_peer };
    const spill: ?learnt.Asked = if (spilling) asked else null;
    return if (paired)
        serve(1, init, &opened, path, args, bounded.max_queued, place, options, spill, started)
    else
        serve(world, init, &opened, path, args, bounded.max_queued, place, options, spill, started);
}

/// What rank 0 hands rank 1 under `--weights-from-peer` (`join_file.hand`): the checkpoint's group of blobs, then
/// the drafter's where it has one (`--drafter`); a group is its `config.json`, the number of its files (8 bytes,
/// little-endian), then each file's length (8 bytes) and header, in the checkpoint's order. No tensor's bytes.
fn handCheckpoint(init: std.process.Init, opened: *common.Opened, drafter: ?*rig.Opened, place: join_file.Place) !void {
    const arena = init.arena.allocator();
    var blobs: std.ArrayList([]const u8) = .empty;
    try group(arena, init.io, opened, &blobs);
    if (drafter) |held| try group(arena, init.io, held, &blobs);
    try join_file.hand(init.io, init.gpa, place, blobs.items);
}

/// `opened`'s group of blobs (`handCheckpoint`) after `blobs`'.
fn group(arena: std.mem.Allocator, io: std.Io, opened: anytype, blobs: *std.ArrayList([]const u8)) !void {
    const checkpoint = &opened.checkpoint.?;
    try blobs.append(arena, try opened.dir.?.readFileAlloc(io, "config.json", arena, .limited(1 << 26)));
    const count = try arena.alloc(u8, 8);
    std.mem.writeInt(u64, count[0..8], checkpoint.shards.len, .little);
    try blobs.append(arena, count);
    for (checkpoint.shards) |shard| {
        const bytes = try arena.alloc(u8, 8 + shard.head.len);
        std.mem.writeInt(u64, bytes[0..8], shard.map.len, .little);
        @memcpy(bytes[8..], shard.head);
        try blobs.append(arena, bytes);
    }
}

/// A checkpoint as rank 1 holds it from rank 0's group: its config's text and its files' headers.
const Given = struct { config: []const u8, checkpoint: safetensors.Checkpoint };

/// Rank 1's side of `handCheckpoint`: the checkpoint's, and the drafter's where rank 0 sent one.
fn peerCheckpoint(init: std.process.Init, place: join_file.Place) !struct { main: Given, drafter: ?Given } {
    const arena = init.arena.allocator();
    var blobs = try join_file.handed(init.io, init.gpa, arena, place);
    var main = try ungroup(init.gpa, arena, &blobs);
    errdefer main.checkpoint.close();
    const drafter: ?Given = if (blobs.len > 0) try ungroup(init.gpa, arena, &blobs) else null;
    if (blobs.len > 0) return error.InvalidSafetensors;
    return .{ .main = main, .drafter = drafter };
}

/// The first group of `blobs`, which are left after it.
fn ungroup(gpa: std.mem.Allocator, arena: std.mem.Allocator, blobs: *[]const []const u8) !Given {
    if (blobs.len < 2 or blobs.*[1].len != 8) return error.InvalidSafetensors;
    const count = std.math.cast(usize, std.mem.readInt(u64, blobs.*[1][0..8], .little)) orelse return error.InvalidSafetensors;
    if (count < 1 or blobs.len - 2 < count) return error.InvalidSafetensors;
    const files = blobs.*[2..][0..count];
    const heads = try arena.alloc([]const u8, count);
    const lengths = try arena.alloc(u64, count);
    for (files, heads, lengths) |blob, *head, *length| {
        if (blob.len < 8) return error.InvalidSafetensors;
        length.* = std.mem.readInt(u64, blob[0..8], .little);
        head.* = blob[8..];
    }
    const config = blobs.*[0];
    blobs.* = blobs.*[2 + count ..];
    return .{ .config = config, .checkpoint = try .fromHeads(gpa, heads, lengths) };
}

fn serve(comptime held: usize, init: std.process.Init, opened: *common.Opened, path: []const u8, args: server.cli.Args, max_queued: ?usize, place: join_file.Place, options: session_file.Options, spill: ?learnt.Asked, started: i96) !u8 {
    const Session = session_file.SessionOf(held);
    const arena = init.arena.allocator();
    const io = init.io;
    const config = &opened.config;
    const follows = Session.paired and place.rank.? != 0;
    var startup = if (follows) command.Startup{} else try command.Startup.open(arena, init.environ_map) orelse return 1;
    defer startup.deinit();
    // the text is rank 0's alone, and read before the load: a checkpoint the server cannot speak fails here
    const text: ?*server.hf_text.HfText = if (follows) null else command.loadText(init.gpa, io, path, arena) orelse return 1;
    defer if (text) |held_text| held_text.deinit();
    // a held turn's prompt is encoded once: the host is handed its own text with the segments' ids kept
    var segments: ?segments_file.Segments = if (text) |held_text| try .init(init.gpa, io, held_text.text(), &held_text.tok, segments_file.default_bytes) else null;
    defer if (segments) |*kept| kept.deinit();
    if (text) |held_text| command.warnEnds(held_text.eos, config.eos);

    const session = Session.create(init.gpa, arena, io, &opened.checkpoint.?, config, place, options, "serve") catch |err| {
        std.log.err("serve: {t}", .{err});
        return 1;
    };
    defer session.destroy();
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    const seconds = @as(f64, @floatFromInt(session.load_ns)) / 1e9;
    if (Session.paired) {
        std.log.info("rank {d} of {d}: NCCL {d}, {d} of {d} layers, its share {d:.2} GiB in {d:.1} s; {d} tokens of context{s}", .{ session.first, world, try nccl.version(), options.layers, config.layers, gib, seconds, options.capacity, if (options.serial) ", no drafts" else "" });
    } else std.log.info("both ranks in this process: {d} of {d} layers, {d:.2} GiB in {d:.1} s; {d} tokens of context{s}", .{ options.layers, config.layers, gib, seconds, options.capacity, if (options.serial) ", no drafts" else "" });
    session.meet() catch |err| return common.gone(session, err);
    cuda.phases.mark("the ranks met");
    // `--spill-gib`: each rank keeps its own half of a state that left, rank 1 opening its stores when told
    session.asked = spill;

    if (session.follows()) {
        const ran = session.follow() catch |err| return common.gone(session, err);
        learnt.settle(session);
        std.log.info("rank {d}: followed rank 0's {d} operations", .{ session.first, ran });
        return 0;
    }
    if (spill) |asked| session.learning = learnt.open(session, asked) catch |err| return common.gone(session, err);

    // The grammar engine a structured request's constraint is compiled by, over zspark's own reading of the
    // host's tokenizer (read once): a request that asks for nothing reaches neither.
    var front: command.Front = undefined;
    if (!try front.open(init.gpa, io, opened.dir.?, path, &text.?.tok, config.vocab, config.eos)) return 1;
    defer front.deinit(init.gpa);
    cuda.phases.mark("the frontend and the grammar are made");

    const name = server.hub.servedName(args.name, args.model, path);
    // several replies at once: the hall's rounds on the scheduler's one decoding thread
    const Glm = served_file.Glm(held);
    const pool: ?*Glm.Pool = if (session.hall) |*hall| try Glm.Pool.create(init.gpa, io, hall) else null;
    defer if (pool) |one| {
        one.close();
        one.destroy();
    };
    var served: served_file.ServedOf(held) = .{
        .gpa = init.gpa,
        .io = io,
        .behind = .{ .session = session, .pool = pool },
        .capacity = options.capacity,
        .vocab = @intCast(config.vocab),
        .lanes = Glm.lanes(pool),
        .max_queued = max_queued,
        .frontend = &front.frontend,
        .compiler = front.grammar.compiler(),
        .think = front.think(),
        // --serial leaves the MTP layer unasked; --drafter puts DFlash2 in its place
        .draft_sources = if (options.serial) .{} else if (options.drafter != null) .{ .drafter = .{ .kind = "DFlash2" } } else .{ .mtp = true },
        .device_bytes = &cuda.account.bytes,
    };
    const status = startup.run(init.gpa, io, args, .{
        .identity = command.identity(),
        .engine = served.engine(),
        .text = segments.?.text(),
        .served = name,
        .model_dir = path,
        .model_type = config.model_type,
        .sampling = try command.samplingDefaults(arena, io, path, args),
        .environ = init.environ_map,
        .started = started,
    });
    // every request's thread has left the session before rank 1 is told the end
    served.close();
    learnt.flush(session);
    if (session.leads()) {
        const told = session.end() catch |err| return common.gone(session, err);
        std.log.info("rank 1 followed the same {d} operations", .{told});
    }
    std.log.info("stopped", .{});
    return status;
}
