//! GLM-5.3-Flash kept loaded for replies in turn (spec 045 WS-04): the ranks this process holds (`ranks_forward.zig`),
//! one sequence's state, the decoder over it (`draft.Namespace`) and its drafter: the MTP layer, or the DFlash2
//! drafter of `Options.drafter` (`dflash2/rig.zig`) in its place, the layer then loaded and not asked. Either
//! scores its drafts with the session's four-bit copy of the head, one a rank (`heads`). `generate`, `chat`
//! and `serve` stand on it.
//!
//! With a `Place` the process is one rank of two: it joins the other (`zig/src/ranks/join.zig`) before it loads,
//! and then rank 0 leads and rank 1 follows (`zig/src/ranks/lead.zig`, `mirror.zig`): only rank 0 calls `reply`,
//! rank 1 calls `follow` once and runs what it is told until rank 0's `end`. Each watches the connection for
//! the other's end. `Options.alike` is the check of that: both ranks decode and neither follows.
//!
//! A reply whose prompt begins with everything the caches hold runs the rest alone. With `Options.snapshots`
//! above zero, a prompt that parts from that is resumed from the latest snapshot strictly before the prompt's
//! end (`marks.zig`'s `reached`): the one at the end leaves the decoder nothing to run. Where a prompt and its reply keep
//! their snapshots is `marks.zig`'s (Flash Next's `Planner`, spec 112 WS-06): its messages' starts, where it
//! parted, two tokens before its end, and the reply's stride and end. The KDA state and the cache are cut to
//! the mark, and the rows between are run again.
//! Zero, the default, keeps the older rule and a prompt that parts starts from an empty sequence. Flash
//! Next's session is not shared (`../../AGENTS.md`).

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const channel_file = @import("../../../ranks/channel.zig");
const join_file = @import("../../../ranks/join.zig");
const lead_file = @import("../../../ranks/lead.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const loading_file = @import("../../../weights/loading.zig");
const grammar = @import("../../../text/grammar.zig");
const request_file = @import("../../../server/request.zig");
const scheduler_file = @import("../../../server/scheduler.zig");
const marking = @import("marks.zig");
const book_file = @import("../../../decode/book.zig");
const shelf_file = @import("../../../decode/shelf.zig");
const masks_file = @import("../../../decode/masks.zig");
const sampling = @import("../../../decode/sampling.zig");
const shared = @import("../../../decode/draft.zig");
const Config = @import("../layers/config.zig").Config;
const flash_pass = @import("../dflash2/pass.zig");
const rig = @import("../dflash2/rig.zig");
const draft = @import("../draft.zig");
const dsa = @import("../layers/dsa.zig");
const kda = @import("../layers/kda.zig");
const model = @import("../layers/model.zig");
const held_file = @import("../layers/ranks_forward.zig");
const layout = @import("../layers/layout.zig");
const hall_file = @import("../pool/hall.zig");
const mirror = @import("mirror.zig");
const mtp = @import("../mtp/mtp.zig");
const snapshot = @import("snapshot.zig");
const pack = @import("pack.zig");
const learnt = @import("learnt.zig");

const world = layout.world;

fn forwardScratch(rank: model.Model, rows: usize, capacity: usize) usize {
    for (rank.layers) |layer| switch (layer.attention) {
        .dsa => |weights| return dsa.Buffers.scratchBytes(weights.heads, rows, weights.attended(capacity)),
        else => {},
    };
    return 0;
}

/// `slots_n` sequences' caches and one forward's scratch. The MTP layer's cache is counted once a slot.
fn contextBytes(rank: model.Model, capacity: usize, rows: usize, slots_n: usize) usize {
    var need = model.Seq.bytes(rank, capacity) * slots_n + forwardScratch(rank, rows, capacity);
    if (rank.dsa_layers > 0) need += dsa.Cache.bytes(1, capacity, rank.indexDim()) * slots_n;
    return need;
}

/// `cache`'s extent table becomes the chain a pool gives one stream of its capacity.
fn placeCache(gpa: Allocator, cache: *dsa.Cache) !void {
    if (cache.indexer == null) return;
    const capacity = cache.capacity;
    var pool = try extents.Pool.init(gpa, .{ .extents = extents.wanted(capacity, 1), .capacity = capacity, .slots = 1 });
    defer pool.deinit();
    const chain = try pool.open();
    try pool.grow(chain, capacity);
    try cache.retarget(chain.table.items);
}

pub const Options = struct {
    /// The first layers alone, to try a command; the model's count is the model.
    layers: usize,
    /// A window's rows at most.
    rows: usize = 16,
    /// Positions a prompt and its reply share.
    capacity: usize,
    /// Every window launched, none through a captured graph.
    eager: bool = false,
    /// No drafts: the MTP layer is loaded and not asked.
    serial: bool = false,
    /// Both ranks of a pair decode, neither follows.
    alike: bool = false,
    /// A chain of drafts stops below this product of their probabilities; null is the drafter's own default
    /// (`mtp.Head.default_confidence`, `dflash2`'s `Drafter.default_confidence`).
    confidence: ?f32 = null,
    /// The most drafts of a round; null is the drafter's own default (`mtp.Head.default_cap`, `dflash2`'s
    /// `default_cap`).
    cap: ?usize = null,
    /// The DFlash2 drafter, opened and held to the model (`rig.fit`): it drafts in the MTP layer's place.
    drafter: ?*const rig.Opened = null,
    /// The drafter's linears BF16 as the file has them, not NVFP4 made at load (the rig's `.mixed`).
    drafter_stored: bool = false,
    /// Replies decoded at once: above 1 the session holds a hall of as many slots, each of `capacity`
    /// positions, and every window is launched (a round's rows are no fixed graph's).
    parallel: usize = 1,
    /// States held of one conversation. Zero keeps none: a prompt that is not the caches' prefix starts again.
    snapshots: usize = 0,
    /// A reply keeps a state each this many tokens (`marks.zig`).
    stride: usize = snapshot.stride_default,
    /// How the DSA layers' caches hold their latents (`--kv-dtype`); the MTP layer's stays bf16.
    kv: dsa.Storage = .bf16,
    /// States that leave go to disk and come back (`serve --spill-gib`, `learnt.zig`): on both ranks or neither.
    spill: bool = false,
    /// Rank 1's share is read by rank 0 and sent over the link (`serve --weights-from-peer`, `pack.Line`): rank
    /// 1's checkpoint is rank 0's headers alone (`safetensors.Checkpoint.fromHeads`). The drafter's too: rank 0
    /// sends rank 1 its file whole at its load (`rig.Source.cross`).
    from_peer: bool = false,

    /// What both ranks must hold alike of these, for the handshake.
    pub fn text(self: Options, buffer: []u8) []const u8 {
        return std.fmt.bufPrint(buffer, "layers={d} rows={d} capacity={d} eager={} serial={} alike={} c={?d} cap={?d} dstored={} parallel={d} snaps={d} stride={d} kv={s} spill={} peer={}", .{ self.layers, self.rows, self.capacity, self.eager, self.serial, self.alike, self.confidence, self.cap, self.drafter_stored, self.parallel, self.snapshots, self.stride, @tagName(self.kv), self.spill, self.from_peer }) catch buffer;
    }
};

/// Who takes a reply's tokens as rounds keep them, and may end it.
pub const Tap = struct {
    context: *anyopaque,
    /// A token of the reply, the one that ends it too.
    take: *const fn (context: *anyopaque, token: i32) void,
    /// Asked after every round: true ends the reply there.
    halt: *const fn (context: *anyopaque) bool,
    /// Once, before the prompt runs: how many of its tokens the caches held.
    held: *const fn (context: *anyopaque, cached: usize) void,
};

pub const ReplyOptions = struct {
    /// The tokens that end the reply; none, and it runs to its length.
    eos: []const u32,
    /// Whether the MTP layer drafts (never under `Options.serial`).
    drafts: bool = true,
    /// The MTP layer drafts though the session holds a DFlash2 drafter (TensorFold's policy is a request's too):
    /// from an empty sequence, and the next reply the drafter's starts from one.
    mtp: bool = false,
    /// From an empty sequence whatever the caches hold.
    fresh: bool = false,
    /// Where the prompt's messages start, ascending (`text/frontend.zig`'s `starts`), and where the prefix it
    /// shares with other conversations ends (`decode/marks.zig`'s `keptOf`): its states are kept there.
    starts: []const usize = &.{},
    kept: ?usize = null,
    tap: ?Tap = null,
    /// What the reply is held to (`decode/masks.zig`): each row's logits are masked at the row's own place on
    /// rank 0's host, where the row's halves are joined, so rank 1 hears nothing of it. `gate` is the token
    /// it begins after (`</think>`) where the prompt leaves the reply thinking.
    constraint: ?grammar.Constraint = null,
    gate: ?i32 = null,
    /// A measurement's record (`generate ... chains`, spec 067 WS-12): no draft is verified. Every round is one
    /// row, the drafter is asked for a chain as long as it can make with no confidence to stop it, and
    /// `chain` is told it. The reply is serial decoding's; what any stop rule would have drafted at each
    /// position follows from the lines.
    chains: ?Chains = null,
};

pub const Chains = struct {
    context: *anyopaque,
    /// `position` is the sequence's before the chain; `trail` the drafter's own choices in order, each with the
    /// probability it gave it; `noisy` DFlash2's noise-aware confidence of each (`decode/dflash2/chain.zig`), none
    /// from the MTP layer.
    chain: *const fn (context: *anyopaque, position: usize, trail: []const Trail, noisy: []const f64) anyerror!void,

    pub const Trail = shared.Chain.Considered;

    /// Below every chain's product: a drafter's rule still gives each pick its probability.
    pub const confidence = 1e-30;
};

pub const Finish = request_file.Finish;

/// Each buffer's view.
fn views(comptime held: usize, buffers: *const [held]cuda.Buffer) [held]cuda.View {
    var out: [held]cuda.View = undefined;
    for (&out, buffers) |*one, buffer| one.* = buffer.view;
    return out;
}

/// The most slots of `each` bytes that fit `left` bytes with `cuda.reserve` kept.
fn slotsWithin(left: usize, each: usize) usize {
    return (left -| cuda.reserve) / @max(each, 1);
}

pub const Reply = struct {
    /// The reply's tokens, in the caller's `out`: an end token is the last.
    made: usize,
    finish: Finish,
    /// The prompt's tokens that did not run.
    cached: usize,
    /// To the first token, and from it to the last.
    first_ns: u64,
    rest_ns: u64,
    stats: shared.Stats,
};

/// A session over `held` ranks in this process: `world`, or one of them beside a `Place`.
pub fn SessionOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        pub const H = held_file.Held(held);
        const Decoder = shared.Decoder(draft.Namespace(held));
        pub const paired = held < world;
        /// The ranks this process holds.
        pub const ranks_held = held;
        pub const Hall = hall_file.HallOf(held);
        const Followed = struct { slots: Hall.Slots, seats: ?mtp.SeatsOf(held), flash_seats: ?flash_pass.SeatsOf(held) };

        gpa: Allocator,
        io: std.Io,
        options: Options,
        /// The rank this process holds first: 0 where it holds both.
        first: usize,
        joined: join_file.Joined,
        pool: cuda.Scratch,
        loaded: [held]pack.Loaded,
        load_ns: u64,
        m: H.Model,
        lead: lead_file.Lead,
        watcher: lead_file.Watcher,
        watching: bool = false,
        state: H.State,
        b: H.Buffers,
        graphs: ?H.Graphs,
        sampler: sampling.Sampler,
        logits: []u16,
        /// A constraint's mask of one row, a bit a token.
        mask: []u32,
        pair: mtp.PairOf(held),
        head: mtp.HeadOf(held),
        /// Each rank's four-bit copy of its rows of the vocabulary (`rig.draftHeads`), which the MTP layer and
        /// the DFlash2 drafter score their drafts with; null where the session drafts nothing.
        heads: ?[held]mtp.Linear = null,
        /// The bf16 rows `heads` were made of, on the device, where DFlash2 drafts over a head not BF16
        /// (`rig.draftHeads`' `widened`).
        widened: ?[held]cuda.Buffer = null,
        /// The tokens whose state the caches hold, and whether the drafter has absorbed every one of them: the
        /// MTP layer, or the DFlash2 drafter where one is loaded (`marks.zig`'s `flashTo`).
        known: std.ArrayList(i32) = .empty,
        in_step: bool = true,
        /// The DFlash2 drafter where `options.drafter` names one.
        flash: rig.Rig(held) = undefined,
        /// `options.parallel` replies at once (`hall.zig`): `reply` is then not called. On the rank that
        /// follows, the sequences and the drafter's entries rank 0's hall tells of.
        hall: ?Hall = null,
        followed: ?Followed = null,
        /// States the conversation can return to, oldest first (`snapshot.zig`), and the KDA buffers of marks
        /// dropped, for the next taken: a reply allocates none once marks are held (`decode/shelf.zig`).
        marks: Marks = undefined,
        /// Marks asked of the window about to run, which writes their KDA states as it runs (`marks.zig`'s
        /// `askAhead`), until the window has run and each is taken.
        ahead: std.ArrayList(Mark) = .empty,
        /// The rules the marks are kept and gone back to by (`marks.zig`'s `booked`, `decode/book.zig`).
        book: book_file.Book = undefined,
        /// The end of the last prompt run: the marks below it are a prompt's (`learnt.zig`).
        prompt_end: ?usize = null,
        /// The states that left, on disk, and what the disk keeps free under them (`learnt.zig`); null keeps
        /// none. `learning` holds them, this session's to close.
        spilled: ?*learnt.Store = null,
        floor: ?*learnt.Floor = null,
        learning: ?*learnt.Learning = null,
        /// What `serve` asked of the stores, which rank 1 opens when told.
        asked: ?learnt.Asked = null,
        /// int32 [3]: the word a rank sends and the two it gathers, where the ranks agree on a file.
        agreed: ?cuda.Buffer = null,
        /// Rank 1: the tokens rank 0 told of a state on disk.
        told: std.ArrayList(i32) = .empty,

        pub const Mark = struct {
            pos: usize,
            ranks: [held]snapshot.Shot,
            /// The MTP cache's length, the sequence's position when the mark was taken.
            mtp: usize,
            /// The DFlash2 drafter's context at the mark, where one is loaded and its context stood there
            /// (`marks.zig`'s `flashAt`).
            flash: ?flash_pass.KeptOf(held) = null,
            /// A reply's, on its stride or at its end, outside a prompt's budget (`marks.zig`).
            reply: bool = false,

            pub fn deinit(self: *Mark) void {
                for (&self.ranks) |*shot| shot.deinit();
                if (self.flash) |*kept| kept.deinit();
                self.* = undefined;
            }

            /// `deinit`, the KDA buffers left to `spare` where it has room.
            pub fn release(self: *Mark, gpa: Allocator, spare: *shelf_file.Spares(kda.Mark)) void {
                for (&self.ranks) |*shot| if (shot.kda) |kept| {
                    spare.give(gpa, kept);
                    shot.kda = null;
                };
                self.deinit();
            }
        };

        pub const Marks = shelf_file.Shelf(Mark, kda.Mark);

        /// Joins the other rank where `place` names one (`what` and the options are held alike), loads this
        /// process's shares and meets the other rank loaded. `arena` is for the handshake's text alone.
        pub fn create(gpa: Allocator, arena: Allocator, io: std.Io, checkpoint: *const safetensors.Checkpoint, c: *const Config, place: join_file.Place, options: Options, what: []const u8) !*Self {
            std.debug.assert(options.layers >= 1 and options.layers <= c.layers and options.rows >= 1 and options.capacity >= 2);
            cuda.phases.mark("the session begins");
            _ = try cuda.probe();
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            self.gpa = gpa;
            self.io = io;
            self.options = options;
            self.spilled = null;
            self.floor = null;
            self.learning = null;
            self.asked = null;
            self.agreed = null;
            self.told = .empty;
            self.first = if (paired) place.rank.? else 0;
            self.known = .empty;
            self.marks = .{ .gpa = gpa };
            self.ahead = .empty;
            self.book = try .init(gpa, marking.bookOptions(options));
            errdefer self.book.deinit(gpa);
            self.prompt_end = null;
            self.in_step = true;
            self.watching = false;
            // rank 1's share over the link, which outlives the reader
            std.debug.assert(paired or !options.from_peer);
            var line: ?pack.Line = if (options.from_peer) try .init() else null;
            defer if (line) |*wired| wired.deinit();
            // this process's first rank's first parts are read while the ranks join
            const reading = try gpa.create(pack.Each);
            defer gpa.destroy(reading);
            reading.* = .init(gpa, checkpoint, c, self.first, 0, options.layers, true);
            // two ranks read half of each routed layer's down projections and trade the rest
            reading.swapping = paired;
            if (line) |*wired| reading.over(wired.wire());
            var still_reading = true;
            defer if (still_reading) reading.deinit();
            try reading.prefetch();
            // the drafter's bytes, staged before the shares fill the memory (`rig.Early`): read from the disk, or
            // under --weights-from-peer read by rank 0 and received by rank 1 at its load (`rig.Source.cross`)
            var early: ?*rig.Early = if (options.drafter) |opened| begun: {
                const one = try gpa.create(rig.Early);
                one.* = .{ .source = .{ .checkpoint = &opened.checkpoint.? } };
                const pin: ?safetensors.Stage.Pin = if (options.from_peer) .{ .register = cuda.hostRegister, .unregister = cuda.hostUnregister } else null;
                one.begin(gpa, !options.from_peer or self.first == 0, pin);
                break :begun one;
            } else null;
            defer if (early) |one| freeEarly(gpa, one);
            // a reader that waits for the line is let go before it is joined
            errdefer if (line) |*wired| wired.leave();
            // what changes a rank's bits or its reply, alike on both before either loads
            if (paired) {
                var buffer: [256]u8 = undefined;
                const drafted_by = if (options.drafter) |opened| try rig.name(arena, opened, (try join_file.headers(arena, &opened.checkpoint.?)).digest) else "mtp";
                const text = try std.fmt.allocPrint(arena, "{s} by {s} {s}", .{ what, drafted_by, options.text(&buffer) });
                try self.joined.open(io, gpa, place, try join_file.headers(arena, checkpoint), text);
            }
            errdefer if (paired) self.joined.close();
            if (paired) reading.comm = &self.joined.comm;
            if (line) |*wired| try wired.open(&self.joined.comm);
            cuda.phases.mark("the ranks joined");

            self.pool = .init(gpa);
            errdefer self.pool.deinit();
            const loading = cuda.monotonic();
            var done: usize = 0;
            // the MTP layers are `loaded`'s until the pair below takes them
            var taken = false;
            errdefer for (self.loaded[0..done]) |*one| {
                if (!taken) one.mtp.deinit();
                one.m.deinit();
            };
            for (&self.loaded, self.first..) |*one, rank| {
                const rows: cuda.Scratch.Rows = .{ .pool = &self.pool, .count = options.rows };
                if (rank == self.first) {
                    const got = try pack.loadFrom(gpa, reading, rows);
                    one.* = .{ .m = got.m, .mtp = got.mtp.? };
                    // its buffers go before the next rank's are read
                    still_reading = false;
                    reading.deinit();
                } else one.* = try pack.load(gpa, checkpoint, c, options.layers, rank, rows);
                done += 1;
            }
            try cuda.synchronize();
            self.load_ns = cuda.monotonic() - loading;
            loading_file.report(self.load_ns);
            pack.Sending.report();
            cuda.phases.mark("the shares are loaded");
            // the shares are `loaded`'s to free
            self.m = .{ .ranks = undefined, .max_rows = options.rows };
            for (&self.m.ranks, self.loaded) |*one, got| {
                one.* = got.m;
                one.kv = options.kv;
            }
            if (paired) self.m.gather = .{ .nccl = self.joined.comm };
            if (self.leads()) {
                self.lead = .{ .channel = &self.joined.channel };
                self.m.lead = &self.lead;
            }

            // the drafter's share, and its stage freed, before the caches and the heads are made (`rig.Early`)
            var drafter_loaded = false;
            if (options.drafter) |opened| {
                var ranks: [held]usize = undefined;
                for (&ranks, self.m.ranks) |*number, rank| number.* = rank.rank;
                try self.flash.load(arena, gpa, opened, try early.?.wait(), ranks, &self.pool, options.drafter_stored, if (options.from_peer) &self.joined.comm else null);
                drafter_loaded = true;
                freeEarly(gpa, early.?);
                early = null;
            }
            errdefer if (drafter_loaded) self.flash.unload();

            // the context's caches and a forward's scratch, against the smaller rank, before either allocates them
            const need = contextBytes(self.m.ranks[0], options.capacity, options.rows, @max(options.parallel, 1));
            const free = if (paired) try self.narrow(io) else cuda.available(io);
            if (!dsa.admits(free, need)) {
                std.log.err("--context {d} does not fit: {d} bytes are asked and {d} are free with {d} kept", .{ options.capacity, need, free, cuda.reserve });
                return error.ContextDoesNotFit;
            }

            // under a hall the sequences are its slots', and this one holds nothing
            self.state = try .init(self.m, if (options.parallel > 1) 2 else options.capacity);
            errdefer self.state.deinit();
            self.b = try .init(self.m, options.capacity);
            errdefer self.b.deinit();
            cuda.phases.mark("the caches are made");
            self.graphs = if (options.eager or options.parallel > 1) null else try .init(gpa);
            errdefer if (self.graphs) |*g| g.deinit();
            self.sampler = try .init(gpa, c.vocab);
            errdefer self.sampler.deinit(gpa);
            self.logits = try gpa.alloc(u16, c.vocab);
            errdefer gpa.free(self.logits);
            self.mask = try gpa.alloc(u32, (c.vocab + 31) / 32);
            errdefer gpa.free(self.mask);
            var layer: [held]mtp.Weights = undefined;
            for (&layer, self.loaded) |*one, got| one.* = got.mtp;
            self.pair = try .of(layer, self.m, options.capacity);
            taken = true;
            errdefer self.pair.deinit();
            // one head a rank for whichever drafts
            self.heads = null;
            self.widened = null;
            // both drafters score with a four-bit copy of the head (over FP4-Local's MXFP8 head, of it read back as
            // bf16 rows: only drafts are quantized, spec 114 WS-02); DFlash2 keeps those rows to score its picks again
            if (!options.serial or options.drafter != null)
                self.heads = try rig.draftHeads(held, arena, gpa, self.m, .{ .pool = &self.pool, .count = options.rows }, if (options.drafter != null) &self.widened else null);
            errdefer if (self.heads) |*heads| for (heads) |*one| one.deinit();
            errdefer if (self.widened) |*rows| for (rows) |*one| one.deinit();
            cuda.phases.mark("the draft heads are made");
            self.pair.heads = self.heads;
            self.head = try .init(gpa, self.m, &self.pair, &self.b, &self.sampler, .{ .seed = 0, .temperature = 0 }, c.vocab);
            errdefer self.head.deinit();
            if (options.confidence) |value| self.head.confidence = value;
            if (options.cap) |value| self.head.cap = value;
            // DFlash2 in the layer's place: its share beside the model's, the main buffers tapped before any graph
            if (options.drafter) |opened| {
                try self.flash.init(gpa, opened, self.heads.?, if (self.widened) |*rows| views(held, rows) else null, self.m, &self.b, options.capacity, .{ .seed = 0, .temperature = 0 }, options.eager or options.parallel > 1);
                if (options.confidence) |value| self.flash.d.confidence = value;
                if (options.cap) |value| self.flash.d.cap = value;
            }
            errdefer if (options.drafter != null) self.flash.deinit();
            // from here a failure frees the whole drafter, not its share alone
            drafter_loaded = false;
            self.hall = null;
            self.followed = null;
            try self.bindTables(gpa);
            if (options.parallel > 1) try self.seat(io, c.vocab);
            cuda.phases.mark("the session is made");
            return self;
        }

        /// The drafter's stage, once the device has read what was uploaded from it.
        fn freeEarly(gpa: Allocator, early: *rig.Early) void {
            cuda.synchronize() catch {};
            early.deinit(gpa);
            gpa.destroy(early);
            cuda.phases.mark("the drafter's stage is freed");
        }

        /// The smaller rank's free bytes. Both call it at the same step.
        fn narrow(self: *Self, io: std.Io) !usize {
            const mine: u64 = cuda.available(io);
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, mine, .little);
            try self.joined.channel.request(&bytes);
            const theirs: u64 = switch (try self.joined.channel.next()) {
                .request => |got| if (got.len >= 8) std.mem.readInt(u64, got[0..8], .little) else return error.OutOfStep,
                .barrier => return error.OutOfStep,
            };
            return @intCast(@min(mine, theirs));
        }

        /// Each cache's extent table is the chain `extents.Pool` gives a stream of its capacity. One slot's
        /// chain is extents 0, 1, 2, which is the identity the cache was allocated with; the upload is that chain.
        fn bindTables(self: *Self, gpa: Allocator) !void {
            for (&self.state.ranks) |*rank| if (rank.dsa) |*cache| try placeCache(gpa, cache);
            for (&self.pair.caches) |*cache| if (cache.indexer != null) try placeCache(gpa, cache);
        }

        /// The slots of `options.parallel` replies at once, where they fit what the device has left: rank 0's
        /// hall of them, or what rank 1 follows it on.
        fn seat(self: *Self, io: std.Io, vocab: usize) !void {
            const gpa = self.gpa;
            const options = self.options;
            const count = options.parallel;
            const drafting: Hall.Drafting = if (options.serial) .none else if (options.drafter != null) .{ .flash = &self.flash.d } else .{ .mtp = &self.head };
            const each = held * Hall.slotBytes(self.m.ranks[0], options.capacity, drafting);
            const scratch = forwardScratch(self.m.ranks[0], options.rows, options.capacity);
            const left = if (paired) try self.narrow(io) else cuda.available(io);
            const fit = slotsWithin(left -| scratch, each);
            std.log.info("{d} replies at once: {d:.3} GiB a slot of {d} tokens, {d:.2} GiB left on the device, {d} slots fit", .{ count, units.gib(each), options.capacity, units.gib(left), fit });
            if (count > fit) {
                std.log.err("--parallel {d} does not fit: a slot of {d} tokens takes {d:.3} GiB here and {d:.2} GiB are left with {d:.0} kept free; {d} at most, or a smaller --context", .{ count, options.capacity, units.gib(each), units.gib(left), units.gib(cuda.reserve), fit });
                return error.SlotsDoNotFit;
            }
            if (!self.follows()) {
                self.hall = try .init(gpa, self.m, &self.b, count, options.capacity, vocab, &self.sampler, self.logits, drafting);
                // the hall's rounds as graphs, a round's shape their key: the loopback's alone, for over
                // NCCL a capture must be told to the rank that follows, as the serial round's is
                self.hall.?.graphs = if (paired or options.eager) null else try .init(gpa);
                return;
            }
            var slots = try Hall.slotsFor(gpa, self.m, count, options.capacity);
            errdefer slots.deinit();
            var seats: ?mtp.SeatsOf(held) = if (drafting == .mtp) try .init(gpa, self.m, count, options.capacity) else null;
            errdefer if (seats) |*one| one.deinit();
            const contexts: ?flash_pass.SeatsOf(held) = if (drafting == .flash) try .init(gpa, &self.flash.pair, count) else null;
            self.followed = .{ .slots = slots, .seats = seats, .flash_seats = contexts };
        }

        /// Both loaded before either launches (a gather waits for the other rank, and its wait is no kernel's
        /// time); from here a rank reads no frame but an operation, and its watcher ends it if the other is
        /// gone. After `create`, once the caller has said what it loaded.
        pub fn meet(self: *Self) !void {
            if (!paired) return;
            try self.joined.channel.barrier(1);
            if (self.options.alike) return;
            self.watcher = .{ .handle = self.joined.link.stream.socket.handle, .rank = self.first, .channel = if (self.leads()) &self.joined.channel else null };
            try self.watcher.start();
            self.watching = true;
        }

        pub fn destroy(self: *Self) void {
            const gpa = self.gpa;
            if (self.watching) self.watcher.stop();
            if (self.learning) |kept| kept.close();
            if (self.agreed) |*one| one.deinit();
            self.told.deinit(gpa);
            self.known.deinit(gpa);
            marking.dropAhead(self);
            self.ahead.deinit(gpa);
            self.marks.deinit();
            self.book.deinit(gpa);
            if (self.hall) |*hall| hall.deinit();
            if (self.followed) |*f| {
                if (f.flash_seats) |*one| one.deinit();
                if (f.seats) |*one| one.deinit();
                f.slots.deinit();
            }
            if (self.options.drafter != null) self.flash.deinit();
            self.head.deinit();
            if (self.heads) |*heads| for (heads) |*one| one.deinit();
            if (self.widened) |*rows| for (rows) |*one| one.deinit();
            self.pair.deinit();
            gpa.free(self.mask);
            gpa.free(self.logits);
            self.sampler.deinit(gpa);
            if (self.graphs) |*g| g.deinit();
            self.b.deinit();
            self.state.deinit();
            for (&self.loaded) |*one| one.m.deinit();
            self.pool.deinit();
            if (paired) self.joined.close();
            gpa.destroy(self);
        }

        pub fn leads(self: *const Self) bool {
            return paired and !self.options.alike and self.first == 0;
        }

        pub fn follows(self: *const Self) bool {
            return paired and !self.options.alike and self.first != 0;
        }

        /// Rank 1's whole life after `meet`: what rank 0 tells until its `end`, then the barrier of the
        /// operations' count. A failure of this rank's is sent to rank 0 before it is returned.
        pub fn follow(self: *Self) !usize {
            if (held != 1) unreachable;
            std.debug.assert(self.follows());
            var followed: mirror.Followed = .{ .m = self.m, .state = &self.state, .b = &self.b, .graphs = if (self.graphs) |*g| g else null, .pair = &self.pair, .flash = if (self.options.drafter != null) &self.flash.d else null, .logits = self.logits, .keep = keepTold, .within = withinTold, .ahead = aheadTold, .drop = dropTold, .back = backTold, .fresh = freshTold, .spill = spillTold, .snap = self };
            if (self.followed) |*f| {
                followed.slots = &f.slots;
                if (f.seats) |*one| followed.seats = one;
                if (f.flash_seats) |*one| followed.flash_seats = one;
            }
            const ran = mirror.follow(&self.joined.channel, followed) catch |err| {
                self.unwatch();
                if (err != error.PeerFailed and err != error.PeerGone) {
                    var buffer: [160]u8 = undefined;
                    self.joined.channel.fail(std.fmt.bufPrint(&buffer, "rank {d} following: {t}", .{ self.first, err }) catch "rank 1 following");
                }
                return err;
            };
            self.unwatch();
            try self.joined.channel.barrier(ran);
            return ran;
        }

        /// Rank 0's orderly end: rank 1 is told and both meet at the operations' count, which is returned.
        pub fn end(self: *Self) !usize {
            std.debug.assert(self.leads());
            self.unwatch();
            try self.lead.tell(@backingInt(held_file.Op.end), &.{}, &.{});
            try self.joined.channel.barrier(self.lead.told);
            return self.lead.told;
        }

        fn unwatch(self: *Self) void {
            if (self.watching) self.watcher.stop();
            self.watching = false;
        }

        /// How many of `prompt`'s tokens a reply to it would not run.
        pub fn cachedOf(self: *const Self, prompt: []const i32, drafted: bool) usize {
            const known = self.known.items;
            if (known.len == 0 or known.len >= prompt.len or (drafted and !self.in_step)) return 0;
            return if (std.mem.eql(i32, prompt[0..known.len], known)) known.len else 0;
        }

        /// `cachedOf`, or, where the prompt parts from what the caches hold, the latest snapshot strictly
        /// before the prompt's end and at or before the shared prefix. A mark at the prompt's own end is not
        /// used: the decoder runs a token to choose the reply's first. The state stands at the position
        /// returned.
        fn resumeFrom(self: *Self, at: Reach) !usize {
            if (!at.back) return at.pos;
            if (self.leads()) try self.lead.tell(@backingInt(held_file.Op.back), &.{@intCast(at.pos)}, &.{});
            try marking.restoreLocal(self, at.pos);
            return at.pos;
        }

        /// Where `resumeFrom` would leave the state, and whether it goes back to a mark for it, changing nothing;
        /// `same`, how many tokens the prompt shares with the sequence the caches hold.
        const Reach = struct { pos: usize, back: bool, same: usize };

        fn reachOf(self: *Self, prompt: []const i32, drafting: bool) !Reach {
            const known = self.known.items;
            var same: usize = 0;
            while (same < known.len and same < prompt.len and known[same] == prompt[same]) same += 1;
            const live = self.cachedOf(prompt, drafting);
            if (self.options.snapshots == 0 or live != 0 or (drafting and !self.in_step)) return .{ .pos = live, .back = false, .same = same };
            const pos = try marking.reached(self, prompt.len, same) orelse return .{ .pos = 0, .back = false, .same = same };
            return .{ .pos = pos, .back = true, .same = same };
        }

        fn spillTold(context: *anyopaque, op: held_file.Op, arguments: []align(1) const u32, tokens: []const i32) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            try learnt.told(self, op, arguments, tokens);
        }

        /// Whether both ranks hold `ok`: one word gathered over NCCL, as rank 1 answers rank 0 nothing on the
        /// control channel. The same on one process.
        pub fn agree(self: *Self, ok: bool) !bool {
            if (!paired) return ok;
            const words = self.agreed orelse return error.OutOfStep;
            const mine: i32 = @intFromBool(ok);
            try words.uploadAt(0, std.mem.asBytes(&mine));
            try self.joined.comm.allGather(words.span(0, 1), words.span(1, world));
            // the gather is NCCL's, on the session's stream
            try cuda.synchronize();
            var both: [world]i32 = undefined;
            try words.downloadAt(@sizeOf(i32), std.mem.sliceAsBytes(&both));
            return std.mem.allEqual(i32, &both, 1);
        }

        fn keepTold(context: *anyopaque) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            try marking.keepLocal(self, false);
        }

        fn withinTold(context: *anyopaque, rows: usize) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            if (rows == 0 or rows > self.b.rows) return error.OutOfStep;
            try marking.keepWithinLocal(self, rows);
        }

        fn aheadTold(context: *anyopaque, rows: []align(1) const u32) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            for (rows) |one| if (one == 0 or one >= self.b.rows) return error.OutOfStep;
            try marking.askAheadLocal(self, rows);
        }

        fn dropTold(context: *anyopaque, pos: usize) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            try marking.dropLocal(self, pos);
        }

        /// Rank 0 clears its marks as it resets; a follower that kept them would thin a list out of order.
        fn freshTold(context: *anyopaque) void {
            const self: *Self = @ptrCast(@alignCast(context));
            marking.clearMarks(self);
        }

        fn backTold(context: *anyopaque, pos: usize) anyerror!void {
            const self: *Self = @ptrCast(@alignCast(context));
            try marking.restoreLocal(self, pos);
        }

        // `marks.zig`'s, by the names its `keep` calls, which `marks_test.zig`'s session of no device's answers
        pub fn takeMark(self: *Self) !Mark {
            return marking.takeMark(self);
        }

        pub fn flashTo(self: *Self, pos: usize, from: ?usize) !?usize {
            return marking.flashTo(self, pos, from);
        }

        pub fn flashBack(self: *Self, absorbed: ?usize) !void {
            return marking.flashBack(self, absorbed);
        }

        /// Retained conversation state and snapshots, including reusable snapshot buffers; decoding thread only.
        pub fn cacheSnapshot(self: *const Self) scheduler_file.Cache.Snapshot {
            var bytes: usize = 0;
            const current = self.known.items.len > 0;
            if (current) for (self.m.ranks) |rank| {
                bytes += model.Seq.bytes(rank, self.options.capacity);
            };
            for (self.marks.held.items) |mark| {
                for (mark.ranks) |shot| if (shot.kda) |kept| {
                    bytes += kda.Mark.bytes(kept.layers, kept.heads);
                };
                if (mark.flash) |kept| for (kept.ranks) |buffer| {
                    bytes += buffer.bytes();
                };
            }
            for (self.marks.spare.list.items) |kept| bytes += kda.Mark.bytes(kept.layers, kept.heads);
            return .{ .states = self.marks.held.items.len + @intFromBool(current), .bytes = bytes };
        }

        /// What a host test reads of the snapshot marks (spec 099 WS-03): how many are held, and a mark's
        /// position with each rank's shot's own position and whether its KDA chains were taken.
        pub const Marked = struct { pos: usize, ranks: [held]usize, kda: [held]bool };

        pub fn marksHeld(self: *Self) usize {
            return self.marks.held.items.len;
        }

        pub fn markAt(self: *Self, at: usize) Marked {
            const one = self.marks.held.items[at];
            var out: Marked = .{ .pos = one.pos, .ranks = undefined, .kda = undefined };
            for (&out.ranks, one.ranks, &out.kda) |*position, shot, *chains| {
                position.* = shot.pos;
                chains.* = shot.kda != null;
            }
            return out;
        }

        /// The marks a prompt's plan names (`marks.zig`), taken window by window.
        const Taking = struct {
            session: *Self,
            planned: []const usize,
            next: usize = 0,
            failed: ?anyerror = null,
        };

        /// The marks that stand in the window about to run, asked of it where it runs chunked.
        fn askWindow(context: *anyopaque, run: []const i32) Decoder.Error!void {
            const taking: *Taking = @ptrCast(@alignCast(context));
            marking.askAhead(taking.session, taking.planned, taking.next, run.len) catch |err| {
                taking.failed = err;
                return error.DraftFailed;
            };
        }

        /// The marks that stand in the window that has run and is not committed.
        fn takeWindow(context: *anyopaque, run: []const i32, _: bool) Decoder.Error!void {
            const taking: *Taking = @ptrCast(@alignCast(context));
            marking.takeIn(taking.session, taking.planned, &taking.next, run.len) catch |err| {
                taking.failed = err;
                return error.DraftFailed;
            };
        }

        /// A reply to `prompt` of at most `out.len` tokens, which must fit the capacity with it. Not on a
        /// rank that follows.
        pub fn reply(self: *Self, prompt: []const i32, settings: sampling.Sampling, o: ReplyOptions, out: []i32) !Reply {
            std.debug.assert(!self.follows() and prompt.len >= 1 and out.len >= 1 and prompt.len + out.len <= self.options.capacity);
            const flashed = self.options.drafter != null and !o.mtp;
            // the drafter that does not draft absorbs nothing
            const other = self.options.drafter != null and o.mtp;
            const drafter: ?shared.Drafter = if (self.options.serial or !o.drafts) null else if (flashed) self.flash.d.drafter() else self.head.drafter();
            const resumes = !(o.fresh or other);
            // the conversation's state that this prompt would lose goes to disk first
            const reach: Reach = if (resumes) try self.reachOf(prompt, drafter != null) else .{ .pos = 0, .back = false, .same = 0 };
            if (self.spilled != null) learnt.leave(self, reach.pos);
            var cached = if (resumes) try self.resumeFrom(reach) else 0;
            if (resumes and self.spilled != null) cached = try learnt.recall(self, prompt, cached);
            if (o.tap) |tap| tap.held(tap.context, cached);
            // until the reply is whole the caches are no prompt's prefix
            self.known.clearRetainingCapacity();
            if (cached == 0) {
                try self.state.reset();
                try self.head.reset(settings);
                if (self.options.drafter != null) self.flash.d.reset(settings);
                self.in_step = true;
                // the marks are the sequence just left; a new one starts with none
                marking.clearMarks(self);
            } else {
                self.head.settings = settings;
                if (flashed) self.flash.d.settings = settings;
            }
            std.debug.assert(self.state.pos == cached);
            self.head.ends = o.eos;
            if (flashed) self.flash.d.ends = o.eos;
            // a record's chains run to the drafter's end; the session's own rule comes back with the reply
            const rule = .{ self.head.confidence, if (flashed) self.flash.d.confidence else 0, if (flashed) self.flash.d.cap else 0, flashed and self.flash.d.noisy };
            defer if (o.chains != null) {
                self.head.confidence = rule[0];
                if (flashed) {
                    self.flash.d.confidence = rule[1];
                    self.flash.d.cap = rule[2];
                    self.flash.d.noisy = rule[3];
                }
            };
            if (o.chains != null) {
                self.head.confidence = Chains.confidence;
                if (flashed) {
                    self.flash.d.confidence = Chains.confidence;
                    self.flash.d.cap = shared.forward_max_rows;
                    self.flash.d.noisy = false;
                }
            }
            if (drafter == null or other) self.in_step = false;
            var decoder: Decoder = .{ .m = self.m, .state = &self.state, .b = &self.b, .graphs = if (self.graphs) |*g| g else null, .sampler = &self.sampler, .settings = settings, .eos = o.eos, .logits = self.logits };

            var masks: masks_file.Masks = undefined;
            if (o.constraint) |constraint| {
                masks = try .init(constraint, self.mask, o.gate);
                decoder.masks = &masks;
            }

            var chosen: [shared.forward_max_rows]i32 = undefined;
            var last: [shared.forward_max_rows]i32 = undefined;
            var drafts: [shared.forward_max_rows]i32 = undefined;
            // the prompt's marks, planned over the windows it runs and each taken inside the one that holds it
            const started = cuda.monotonic();
            // what a prompt that did not finish left asked of its window
            marking.dropAhead(self);
            const planned: []const usize = if (self.options.snapshots > 0) try marking.plan(self, decoder, chosen.len, .{ .pos = cached, .len = prompt.len, .shared = if (resumes) @max(reach.same, cached) else 0, .starts = o.starts, .kept = o.kept }) else &.{};
            defer self.gpa.free(planned);
            var taking: Taking = .{ .session = self, .planned = planned };
            if (planned.len > 0) decoder.within = .{ .context = &taking, .window = takeWindow, .before = askWindow };
            const rows = decoder.prompt(drafter, prompt, &chosen) catch |err| {
                if (taking.failed) |failed| return failed;
                return err;
            };
            std.debug.assert(taking.next == planned.len and self.ahead.items.len == 0);
            decoder.within = null;
            @memcpy(last[0..rows], chosen[0..rows]);
            out[0] = chosen[rows - 1];
            self.prompt_end = prompt.len;
            self.book.gridFrom(self.state.pos);
            const first_ns = cuda.monotonic() - started;
            if (o.tap) |tap| tap.take(tap.context, out[0]);
            var made: usize = 1;
            var finish: Finish = .length;
            // the choices after the rows the last window kept, which the drafter absorbs
            var kept: []const i32 = last[0..rows];
            while (true) {
                const whole = if (decoder.masks) |held_masks| held_masks.ended() else false;
                if (whole or std.mem.indexOfScalar(u32, o.eos, @intCast(out[made - 1])) != null) {
                    finish = .stop;
                    break;
                }
                if (made == out.len) break;
                if (o.tap) |tap| if (tap.halt(tap.context)) {
                    finish = .cut;
                    break;
                };
                const drafted = if (drafter) |d| try decoder.draft(d, kept, out.len - made - 1, &drafts) else 0;
                if (o.chains) |record| {
                    const trail = if (drafter == null) &.{} else if (flashed) self.flash.d.trail[0..self.flash.d.trailed] else self.head.trail[0..self.head.trailed];
                    try record.chain(record.context, self.state.pos, trail, if (drafter != null and flashed) self.flash.d.noised[0..trail.len] else &.{});
                }
                const n = if (o.chains != null) 0 else drafted;
                const keep = try decoder.round(out[made - 1], drafts[0..n], &chosen);
                @memcpy(out[made..][0..keep], chosen[0..keep]);
                if (o.tap) |tap| for (chosen[0..keep]) |token| tap.take(tap.context, token);
                made += keep;
                kept = out[made - keep .. made];
                if (self.options.snapshots > 0) while (self.book.onGrid(self.state.pos)) try marking.keep(self, true);
            }
            // the last window's rows into the drafter, as the next round's drafts would have taken them: a
            // prompt that goes on from here finds every position absorbed
            if (drafter) |d| _ = try decoder.draft(d, kept, 0, &drafts);
            try cuda.synchronize();
            const rest_ns = cuda.monotonic() - started - first_ns;
            try self.known.appendSlice(self.gpa, prompt);
            try self.known.appendSlice(self.gpa, out[0 .. made - 1]);
            std.debug.assert(self.known.items.len == self.state.pos);
            // the reply's end, one of its own: a prompt that parts inside the next turn's goes on from it
            if (self.book.options.grid) try marking.keep(self, true);
            return .{ .made = made, .finish = finish, .cached = cached, .first_ns = first_ns, .rest_ns = rest_ns, .stats = decoder.stats };
        }
    };
}
