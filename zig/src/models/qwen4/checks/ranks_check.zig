//! `zspark ranks-check random model=flash-next rank=R at=ADDR:PORT [seed=N]` and `zspark ranks-check DIR
//! [layers=L] rank=R at=ADDR:PORT`: Flash Next a rank a process over NCCL against both ranks in one process, bit
//! for bit (spec 050 WS-04). Both ranks make the same shares (the random model of a seed, `random.zig`, or the
//! checkpoint's first layers), join through the control channel, and run the same windows twice: in loopback in
//! this process, the reference for two devices, and as this rank's own share with the other's partials
//! gathered over NCCL; each way launched and through captured graphs. What a window leaves is compared: the
//! last layer's gathered partials of the branch and the MoE (both ranks'), this rank's streams, mixed rows and
//! half of the logits, the last row's whole logits (`logitsRow`, gathered over NCCL on one rank), and after the
//! windows every DeltaNet state and conv window, and of each attention cache the committed rows as stored
//! (keys, values, their scales, the indexer's keys). The gathered buffers are filled before every window, with
//! other bytes on the two ways, so a rank's rows that no gather brought differ too.
//!
//! `kv=bf16|fp8|fp8-nvfp4` (spec 094) is the storage of the attention caches, a rank its half of the KV heads:
//! the checkpoint's layers loaded with it, and the random model given an attention layer after its DeltaNet
//! ones (`random.Extra.caches`; without the word it has none). The head's own cache stays bf16.
//!
//! `streams` (`ranks_streams.zig`) is the other check of two ranks: several streams packed into rounds, every
//! row against the row its stream has alone, with `kv=` over the attention layer's pool of extents.
//! `hall` (`ranks_hall.zig`): the pool's engine serving them, rank 1 following, each together as alone.
//! `windows` (`ranks_windows.zig`): a prompt in windows of 2,048 rows, of 512 and serial steps, the same bits,
//! and the pair's frames carrying such windows (spec 111).
//!
//! `reference=` also takes `drafter=DIR` (spec 126 WS-01): the stored DFlash2 drafter beside the reference
//! path's pair (`ranks_drafter.zig`), each rank its own share of DIR, its drafted replies the serial ones.
//!
//! The windows are a prompt chunk of the most rows, serial steps (the DeltaNet states' two sides, a graph
//! launched again), verify windows cut short, and a window after them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const channel_file = @import("../../../ranks/channel.zig");
const check = @import("../../../ranks/check.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const trace_file = @import("../../../ranks/trace.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const qwen4_config = @import("../config.zig");
const layout = @import("../layout.zig");
const attention = @import("../blocks/attention.zig");
const model = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const share_file = @import("../share.zig");
const rig = @import("../dflash2/rig.zig");
const random_file = @import("random.zig");
const random_drafter = @import("random_drafter.zig");
const ranks_head = @import("ranks_head.zig");
const ranks_session = @import("ranks_session.zig");
const ranks_streams = @import("ranks_streams.zig");
const ranks_hall = @import("ranks_hall.zig");
const ranks_states = @import("ranks_states.zig");
const ranks_windows = @import("ranks_windows.zig");
const ranks_vllm = @import("ranks_vllm.zig");
const ranks_whole = @import("ranks_whole.zig");

const world: usize = share_file.world;
const max_rows = 16;
/// The attention caches' positions: a checkpoint's first layers', and the random model's under `kv=`.
const capacity = 4096;
/// A window's rows and how many of them are kept, from an empty sequence. A window can run more rows than it
/// keeps, so the ids have to cover the furthest `at + rows`, not the sum of the keeps.
const plan = [_][2]usize{ .{ max_rows, max_rows }, .{ 1, 1 }, .{ 1, 1 }, .{ 1, 1 }, .{ 6, 3 }, .{ 9, 9 }, .{ 1, 1 }, .{ 16, 4 }, .{ 2, 2 } };

fn tokenCount() usize {
    var at: usize = 0;
    var need: usize = 0;
    for (plan) |window| {
        need = @max(need, at + window[0]);
        at += window[1];
    }
    return need;
}

fn planTokens(arena: Allocator, seed: u64, vocab: usize) ![]i32 {
    const tokens = try arena.alloc(i32, tokenCount());
    var prng: std.Random.DefaultPrng = .init(seed +% 1);
    for (tokens) |*id| id.* = prng.random().intRangeLessThan(i32, 0, @intCast(vocab));
    return tokens;
}

pub const usage =
    \\usage: zspark ranks-check random model=flash-next rank=0|1 at=ADDR:PORT [seed=N] [kv=T] [session|states|streams|hall|windows]
    \\       zspark ranks-check DIR [layers=L] rank=0|1 at=ADDR:PORT [kv=T] [session|states|streams|hall|windows]   (Flash Next's DIR)
    \\       zspark ranks-check random model=flash-next hall drafter=random rank=0|1 at=ADDR:PORT [kv=T]
    \\       zspark ranks-check DIR hall drafter=DIR [layers=L] rank=0|1 at=ADDR:PORT
    \\       zspark ranks-check DIR reference=REF [drafter=DIR] rank=0|1 at=ADDR:PORT         (every layer, against
    \\                                                     REF: TensorFold's export, or the vLLM container's;
    \\                                                     drafter=DIR: the stored DFlash2 drafter drafts
    \\                                                     the replies, each rank its own share of DIR)
    \\  rank 1 listens at ADDR:PORT (start it first), rank 0 connects there. Both run the same shares in
    \\  loopback and as their own over NCCL, and compare every bit a rank holds; 1 on a difference. DIR runs
    \\  its first L layers (4 unless given). kv=bf16|fp8|fp8-nvfp4: the attention caches' storage, and on the
    \\  random model an attention layer to hold them (none without it; not with reference=). streams: four
    \\  streams packed into rounds, each row the row its stream has alone on the two ranks. hall: the pool's
    \\  engine serves them, each together as alone. windows: a prompt in windows of 2048 rows, of 512 and
    \\  serial steps, every row the same bits, and a pair of 2048 rows telling them.
    \\  Needs a build with -Dnccl=PREFIX. tools/two_ranks.sh starts both.
    \\
;

const What = enum {
    branch,
    moe,
    streams,
    mixed,
    logits,
    row,
    recurrent,
    conv,
    keys,
    values,
    key_scales,
    value_scales,
    index_keys,

    pub fn text(self: What) []const u8 {
        return switch (self) {
            .branch => "the last layer's gathered branch partials",
            .moe => "the last layer's gathered MoE partials",
            .streams => "the streams",
            .mixed => "the head's mixed rows",
            .logits => "the rank's half of the logits",
            .row => "the last row's whole logits",
            .recurrent => "a DeltaNet state",
            .conv => "a DeltaNet conv window",
            .keys => "an attention cache's keys",
            .values => "an attention cache's values",
            .key_scales => "an attention cache's key scales",
            .value_scales => "an attention cache's value scales",
            .index_keys => "an attention cache's indexer keys",
        };
    }

    pub fn gathered(self: What) bool {
        return self == .branch or self == .moe;
    }
};

const Trace = trace_file.Trace(What);

/// The shares over `held` ranks, with their buffers.
fn Rig(comptime held: usize) type {
    return struct {
        const Self = @This();
        const H = held_file.Held(held);

        m: H.Model,
        b: H.Buffers,
        /// Which of the held ranks is the one this process is.
        mine: usize,
        /// What the gathered buffers hold before every window, the size of one: another byte with both ranks
        /// here than with one, so a row no gather brought is other bits on the two ways.
        stale: []const u8,
        gpa: Allocator,

        fn init(arena: Allocator, gpa: Allocator, built: [held]model.Model, first: usize, mine: usize, gather: ?nccl.Comm) !Self {
            var m: H.Model = .{ .ranks = built, .first = first, .max_rows = max_rows };
            errdefer m.deinit();
            if (gather) |comm| m.gather = .{ .nccl = comm };
            var b: H.Buffers = try .init(gpa, m, max_rows, capacity);
            errdefer b.deinit();
            const stale = try arena.alloc(u8, b.branch.bytes());
            @memset(stale, if (held == world) 0x3b else 0x3c);
            return .{ .m = m, .b = b, .mine = mine, .stale = stale, .gpa = gpa };
        }

        fn deinit(self: *Self) void {
            self.b.deinit();
            self.m.deinit();
        }

        fn spoil(self: *Self) !void {
            try cuda.synchronize();
            try self.b.branch.upload(self.stale);
            try self.b.moe_parts.upload(self.stale);
        }

        /// The plan's windows from an empty sequence: launched, or through `graphs`.
        fn play(self: *Self, trace: *Trace, tokens: []const i32, graphs: ?*H.Graphs) !void {
            var state: H.State = try .init(self.gpa, self.m, max_rows, capacity);
            defer state.deinit();
            const vocab = self.m.ranks[0].head.n * world;
            const row = try self.gpa.alloc(u16, vocab);
            defer self.gpa.free(row);
            var at: usize = 0;
            for (plan, 0..) |window, index| {
                const rows = window[0];
                const ids = tokens[at..][0..rows];
                trace.step = index;
                trace.rows = rows;
                try self.spoil();
                if (graphs) |captured| {
                    try captured.forward(self.m, &state, &self.b, ids);
                } else {
                    try H.forward(self.m, &state, &self.b, ids);
                }
                const own = self.b.ranks[self.mine];
                try trace.take(.branch, null, ranks_file.gathered(self.b.branch.view, rows));
                try trace.take(.moe, null, ranks_file.gathered(self.b.moe_parts.view, rows));
                try trace.take(.streams, null, own.h.rows(rows));
                try trace.take(.mixed, null, own.mixed.rows(rows));
                try trace.take(.logits, null, own.logits.rows(rows));
                try H.logitsRow(self.b, rows - 1, row);
                try trace.put(.row, std.mem.sliceAsBytes(row), 2, vocab);
                try H.commit(self.m, &state, self.b, ids, window[1]);
                at += window[1];
            }
            trace.step = plan.len;
            trace.rows = at;
            const sequence = state.ranks[self.mine];
            for (sequence.recurrent, sequence.current, 0..) |sides, side, layer| try trace.take(.recurrent, layer, sides[side].view);
            for (sequence.conv, 0..) |window, layer| try trace.take(.conv, layer, window.view);
            // the committed rows as stored: a coded cache's codes and scales are a row's own
            for (sequence.caches, 0..) |cache, layer| {
                std.debug.assert(cache.table == null);
                try trace.take(.keys, layer, cache.k.rows(at));
                try trace.take(.values, layer, cache.v.rows(at));
                if (cache.k_scale) |scale| try trace.take(.key_scales, layer, scale.rows(at));
                if (cache.v_scale) |scale| try trace.take(.value_scales, layer, scale.rows(at));
                try trace.take(.index_keys, layer, cache.index_keys.rows(at));
            }
        }
    };
}

/// How both ranks make their shares: the random model of a seed, or a checkpoint's first layers.
const Source = union(enum) {
    random: u64,
    checkpoint: struct { config: qwen4_config.Config, found: layout.Layout, layers: usize },

    /// `drafting`: rank 0's share with the MTP head whole beside it (the random model's is random), or under
    /// `model.ranks_head == .split` every rank's with its half. `caches`: the attention caches' storage, the
    /// head's own bf16 beside it as the commands load it; null is bf16, and the random model without an
    /// attention layer.
    fn shares(self: Source, comptime held: usize, gpa: Allocator, arena: Allocator, first: usize, drafting: bool, caches: ?attention.Storage) ![held]model.Model {
        return self.sharesOf(held, gpa, arena, first, drafting, caches, max_rows);
    }

    /// `shares` for windows of up to `rows` rows.
    fn sharesOf(self: Source, comptime held: usize, gpa: Allocator, arena: Allocator, first: usize, drafting: bool, caches: ?attention.Storage, rows: usize) ![held]model.Model {
        const split = model.ranks_head == .split;
        switch (self) {
            .random => |seed| return random_file.sharesOf(held, gpa, arena, seed, first, rows, .{ .head = if (!drafting) .none else if (split) .split else .whole, .caches = caches }),
            .checkpoint => |c| {
                const ids: ?[]const u32 = if (drafting) try model.DraftHead.defaultIds(arena) else null;
                var out: [held]model.Model = undefined;
                var made: usize = 0;
                errdefer for (out[0..made]) |*one| one.deinit();
                for (&out, 0..) |*one, i| {
                    one.* = try .loadWith(gpa, c.config, c.found, .{ .layers = c.layers, .max_rows = rows, .mtp = drafting, .draft_vocab = ids, .share = .of(@intCast(first + i)), .head_split = split, .kv = caches orelse .bf16, .head_kv = layout.head_storage });
                    made += 1;
                }
                return out;
            },
        }
    }
};

/// Both ranks in this process, on the random model: the plan launched and through captured graphs, bit for
/// bit. What `run` holds the NCCL side to, checked here without a second rank. `caches` as `kv=` gives it.
pub fn loopback(gpa: Allocator, caches: ?attention.Storage) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const seed: u64 = 50;
    const tokens = try planTokens(arena, seed, random_file.fx.shape.vocab);
    var both: Rig(world) = try .init(arena, gpa, try (Source{ .random = seed }).shares(world, gpa, arena, 0, false, caches), 0, 0, null);
    defer both.deinit();
    var launched: Trace = .{ .gpa = gpa };
    defer launched.deinit();
    var through: Trace = .{ .gpa = gpa };
    defer through.deinit();
    try both.play(&launched, tokens, null);
    var graphs: held_file.Held(world).Graphs = try .init(gpa);
    defer graphs.deinit();
    try both.play(&through, tokens, &graphs);
    if (try Trace.differs(arena, &launched, &through, "launched", "through graphs")) |text| {
        std.debug.print("loopback: {s}\n", .{text});
        return error.Mismatch;
    }
}

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    var place: join_file.Place = .{};
    var seed: u64 = 50;
    var layers: usize = 4;
    var dir: ?[]const u8 = null;
    var through_session = false;
    var through_states = false;
    var through_streams = false;
    var through_hall = false;
    var through_windows = false;
    var reference: ?[]const u8 = null;
    var drafter: ?[]const u8 = null;
    var caches: ?attention.Storage = null;
    for (words, 0..) |word, index| {
        if (place.take(word)) continue;
        if (cli.value(word, "seed=")) |text| {
            seed = std.fmt.parseInt(u64, text, 10) catch return 2;
        } else if (cli.value(word, "layers=")) |text| {
            layers = std.fmt.parseInt(usize, text, 10) catch return 2;
        } else if (std.mem.eql(u8, word, "session")) {
            through_session = true;
        } else if (std.mem.eql(u8, word, "states")) {
            through_states = true;
        } else if (std.mem.eql(u8, word, "streams")) {
            through_streams = true;
        } else if (std.mem.eql(u8, word, "hall")) {
            through_hall = true;
        } else if (std.mem.eql(u8, word, "windows")) {
            through_windows = true;
        } else if (cli.value(word, "reference=")) |text| {
            reference = text;
        } else if (cli.value(word, "drafter=")) |text| {
            drafter = text;
        } else if (cli.value(word, "kv=")) |text| {
            caches = attention.Storage.parse(text) orelse {
                std.log.err("ranks-check: kv={s}: bf16, fp8 or fp8-nvfp4", .{text});
                return 2;
            };
        } else if (std.mem.eql(u8, word, "model=flash-next") or (index == 0 and std.mem.eql(u8, word, "random"))) {
            continue;
        } else if (index == 0) {
            dir = word;
        } else {
            std.log.err("ranks-check: '{s}' is not a word of Flash Next's check", .{word});
            return 2;
        }
    }
    if (!place.complete() or layers == 0) return 2;
    if (caches != null and reference != null) {
        std.log.err("ranks-check: kv= is not read with reference=", .{});
        return 2;
    }
    if (drafter != null and reference == null and !through_hall) {
        std.log.err("ranks-check: drafter= is read with reference= or hall", .{});
        return 2;
    }
    if (!nccl.linked) {
        std.log.err("ranks-check: this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return 1;
    }
    const rank = place.rank.?;
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    _ = try cuda.probe();

    var source: Source = .{ .random = seed };
    var vocab: usize = random_file.fx.shape.vocab;
    var named: channel_file.Named = .of(try std.fmt.allocPrint(arena, "a random Flash Next of seed {d}", .{seed}));
    var checkpoint: ?safetensors.Checkpoint = null;
    defer if (checkpoint) |*opened| opened.close();
    var drafting: rig.Opened = .{};
    defer drafting.close(io);
    var drafter_opened: ?*const rig.Opened = null;
    if (dir) |path| {
        var folder = try std.Io.Dir.cwd().openDir(io, path, .{});
        defer folder.close(io);
        var diagnostic: qwen4_config.Diagnostic = .{};
        const config = try qwen4_config.Config.read(arena, io, folder, &diagnostic);
        checkpoint = try .open(gpa, io, .cwd(), path);
        const found = try layout.Layout.init(arena, &config, &checkpoint.?, &diagnostic);
        if (reference != null) layers = config.layers;
        if (layers > config.layers) {
            std.log.err("ranks-check: {d} layers of the model's {d}", .{ layers, config.layers });
            return 2;
        }
        source = .{ .checkpoint = .{ .config = config, .found = found, .layers = layers } };
        vocab = config.vocab;
        named = try join_file.headers(arena, &checkpoint.?);
        // the drafter is held to the main model before anything loads (`fit`), the whole model's layers here
        if (drafter) |drafter_dir| {
            if (!try rig.enter(init, &drafting, drafter_dir, &source.checkpoint.config, layers)) return 1;
            drafter_opened = &drafting;
        }
    }
    if (dir == null and drafter != null) {
        if (!through_hall or !std.mem.eql(u8, drafter.?, "random")) return 2;
        const c = try random_drafter.config(arena, random_file.fx.shape.hidden, vocab, random_file.fx.layers, &.{ 0, 2 });
        var tensors: random_drafter.Source = .empty;
        var prng: std.Random.DefaultPrng = .init(seed +% 162);
        try random_drafter.put(arena, prng.random(), c, &tensors);
        var found: std.StringArrayHashMapUnmanaged(safetensors.Tensor) = .empty;
        var it = tensors.iterator();
        while (it.next()) |entry| try found.put(arena, entry.key_ptr.*, entry.value_ptr.*);
        drafting = .{ .path = "the rank check's random DFlash2 drafter", .config = c };
        // All tensors belong to the command arena; no files or checkpoint descriptors are owned here.
        drafting.checkpoint = .{ .shards = &.{}, .tensors = found, .gpa = arena };
        drafter_opened = &drafting;
    }
    const tokens = try planTokens(arena, seed, vocab);

    const joining = cuda.monotonic();
    var joined: join_file.Joined = undefined;
    const options = try std.fmt.allocPrint(arena, "ranks-check flash-next seed={d} layers={d}{s}{s}{s}{s}{s}{s}{s}{s}{s}", .{ seed, layers, if (through_session) " session" else "", if (through_states) " states" else "", if (through_streams) " streams" else "", if (through_hall) " hall" else "", if (through_windows) " windows" else "", if (model.ranks_head == .split) " head=split" else "", if (caches != null) " kv=" else "", if (caches) |kv| kv.name() else "", if (drafter != null) " drafter" else "" });
    try joined.open(io, gpa, place, named, options);
    defer joined.close();
    const channel = &joined.channel;
    try stdout.print("rank {d} of {d}: zspark {s}{s}, NCCL {d}, {s}, {d} layers, joined in {d:.0} ms\n", .{ rank, world, cli.stamp.commit, if (cli.stamp.dirty) " dirty" else "", try nccl.version(), named.label, if (dir != null) layers else random_file.fx.layers + @as(usize, @intFromBool(caches != null)), @as(f64, @floatFromInt(cuda.monotonic() - joining)) / 1e6 });
    try stdout.flush();

    if (reference) |ref| {
        if (dir == null) return 2;
        const checkpoint_source = source.checkpoint;
        // TensorFold's export, or the vLLM container's replies and scores where no TensorFold reads the
        // checkpoint; `drafter=` beside either, the DFlash2 drafter's drafted replies against the serial ones
        if (ranks_vllm.holds(io, ref)) return ranks_vllm.run(gpa, io, stdout, channel, joined.comm, rank, checkpoint_source.config, checkpoint_source.found, checkpoint_source.layers, ref, drafter_opened);
        return ranks_whole.run(gpa, io, stdout, channel, joined.comm, rank, checkpoint_source.config, checkpoint_source.found, checkpoint_source.layers, ref, drafter_opened);
    }
    if (through_windows) return ranks_windows.run(gpa, arena, stdout, channel, joined.comm, rank, seed, vocab, try source.sharesOf(1, gpa, arena, rank, true, caches, ranks_windows.rows));
    if (through_hall) return ranks_hall.run(gpa, arena, stdout, channel, joined.comm, rank, seed, try source.sharesOf(1, gpa, arena, rank, drafter_opened == null, caches, ranks_hall.rows), drafter_opened);
    if (through_streams) return ranks_streams.run(gpa, arena, stdout, channel, joined.comm, rank, seed, try source.sharesOf(1, gpa, arena, rank, false, caches, ranks_streams.rows));
    if (through_session) {
        if (model.ranks_head == .split) {
            const both = try source.shares(world, gpa, arena, 0, true, caches);
            const failed = try ranks_head.run(gpa, stdout, channel, joined.comm, rank, both, try source.shares(1, gpa, arena, rank, true, caches));
            if (failed != 0) return failed;
        }
        return ranks_session.run(gpa, stdout, channel, joined.comm, rank, try source.sharesOf(1, gpa, arena, rank, true, caches, ranks_session.rows));
    }
    if (through_states) return ranks_states.run(gpa, stdout, channel, joined.comm, rank, try source.shares(1, gpa, arena, rank, false, caches));

    var failure: ?[]const u8 = null;
    var launched: Trace = .{ .gpa = gpa };
    defer launched.deinit();
    var through: Trace = .{ .gpa = gpa };
    defer through.deinit();
    // the reference: both ranks here, their partials written in place
    {
        var both: Rig(world) = try .init(arena, gpa, try source.shares(world, gpa, arena, 0, false, caches), 0, rank, null);
        defer both.deinit();
        try both.play(&launched, tokens, null);
        var graphs: held_file.Held(world).Graphs = try .init(gpa);
        defer graphs.deinit();
        try both.play(&through, tokens, &graphs);
        if (try Trace.differs(arena, &launched, &through, "launched", "through graphs")) |text| failure = try std.fmt.allocPrint(arena, "the loopback itself: {s}", .{text});
        const partials = launched.of(launched.marks.items[0]);
        if (std.mem.eql(u8, partials[0 .. partials.len / world], partials[partials.len / world ..]) or std.mem.allEqual(u8, partials, 0))
            failure = failure orelse "the loopback itself: the ranks' branch partials are the same, or zero";
        try stdout.print("loopback: {d} windows of 1 to {d} rows launched, {d} buffers read, {d:.1} MiB; through {d} graphs, the same\n", .{ plan.len, max_rows, launched.marks.items.len, @as(f64, @floatFromInt(launched.bytes.items.len)) / (1024.0 * 1024.0), graphs.cache.count() });
        try stdout.flush();
    }
    channel.barrier(1) catch |err| return channel.gone(rank, err);

    // this rank's share alone, the other's partials through NCCL. Both ranks run every step whatever they
    // find: one that stopped would leave the other inside a gather.
    {
        var own: Rig(1) = try .init(arena, gpa, try source.shares(1, gpa, arena, rank, false, caches), rank, 0, joined.comm);
        defer own.deinit();
        var got: Trace = .{ .gpa = gpa };
        defer got.deinit();
        try own.play(&got, tokens, null);
        if (try check.held(arena, stdout, channel, rank, 2, &failure, "launched", &launched, &got)) |code| return code;

        var graphs: held_file.Held(1).Graphs = try .init(gpa);
        defer graphs.deinit();
        var got_through: Trace = .{ .gpa = gpa };
        defer got_through.deinit();
        try own.play(&got_through, tokens, &graphs);
        if (try check.held(arena, stdout, channel, rank, 3, &failure, "through captured graphs", &through, &got_through)) |code| return code;
    }

    return check.end(arena, stdout, channel, rank, failure, 4, "");
}
