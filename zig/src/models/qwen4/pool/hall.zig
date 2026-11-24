//! Several replies in one round, behind `zspark serve --parallel` (spec 018, step B9; spec 051). `--parallel 1`
//! does not build one: that path is `session.zig` alone. One live stream here is that session too (`Core.session`:
//! its state, snapshots, marks, reply stride and `settle`, its graphs captured at load), so a lone stream holds
//! the tokens `--parallel 1` holds and keeps its rate; its tokens go to the scheduler through `session.Tap`. A
//! second stream opens the shared pool, into which the session's sequence and its states are carried before
//! the session is freed (`hall_carry.zig`): the stream that was live goes on from its row. A round of several streams is the replies' windows and
//! then, while one is decoding, at most `chunk` rows of the oldest prompt still being filled. Those graphs are
//! `rounds.Graphs`, keyed by the total rows and the context bucket, and only a reply-only round under
//! `gdn.wide_rows` uses one; a prompt chunk launches directly. The head on that path launches its kernels.
//!
//! The caches are one storage a load (`layout.storageFor`): the session's and the pool's are the model's.
//!
//! An image prompt (spec 019 WS-05, `Ask.image`) runs as a lone stream through the session's own image path,
//! from empty; in the pool its slot's `attention.Rope` holds its positions and `Sight` its features, which the
//! seat whose prompt chunk reaches the image encodes again if another took the buffer since. It forks only from
//! a state at or before its first image row, and none past that row is kept (`hall_keep.allows`), as vLLM
//! keys its prefix blocks before the image.
//!
//! In the pool a fork reuses a kept state, and only when the new prompt goes on from it with the token the
//! head took in there. A stream keeps the states a session keeps (spec 054, `hall_keep.zig`): where its
//! prompt's messages start, a little before the prompt's end, below where it parted from what was held, on
//! the prompt's stride and where its last chunks ended, where the reply's reasoning ended, near each 256 tokens of the reply and at its end. The
//! trunk state, the attention rows and the head's cache are copied; the rows past it run again.
//!
//! This file holds the options and the request's types, `Core` (the session or the pool) and `Hall`; the
//! seats' and the round's types are `hall_seats.zig`'s, the pool `hall_engine.zig`'s `Engine`, the jobs and
//! the scheduler's decoder table `hall_front.zig`'s.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("../blocks/attention.zig");
const request_file = @import("../../../server/request.zig");
const copied = @import("../../../decode/copied.zig");
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const grammar = @import("../../../text/grammar.zig");
const masks_file = @import("../../../decode/masks.zig");
const model_file = @import("../model.zig");
const mtp = @import("../mtp.zig");
const prefix = @import("../../../decode/prefix.zig");
const prefix_wait = @import("../../../decode/prefix_wait.zig");
const rounds = @import("rounds.zig");
const sampling = @import("../../../decode/sampling.zig");
const sched = @import("../../../server/scheduler.zig");
const session = @import("../session/session.zig");
const chat_file = @import("../session/chat.zig");
const tokenizer = @import("../../../text/tokenizer.zig");
const hall_carry = @import("hall_carry.zig");
const hall_spill = @import("hall_spill.zig");
const hall_engine = @import("hall_engine.zig");
const hall_front = @import("hall_front.zig");
const hall_warm = @import("hall_warm.zig");
const forward = @import("../forward.zig");
const rig = @import("../dflash2/rig.zig");
const tower_file = @import("../vision.zig");
const Engine = hall_engine.Engine;
const hall_keep = @import("hall_keep.zig");
const decoder_table = hall_front.decoder_table;

pub const default_parallel: usize = 4;
pub const default_context: usize = 262144;
/// The streams a pool has at most (`Engine.openAfter`).
pub const most_parallel: usize = 64;

/// The pool's positions for what `serve` was asked (spec 082; the owner, 2026-10-12): `parallel` contexts of
/// `context` tokens, in whole extents and no fewer than `extents.lowerBound`. The defaults' is 1,048,576.
pub fn poolFor(context: usize, parallel: usize) usize {
    return extents.wanted(context, parallel) * extents.positions;
}

/// Rows the model is loaded for, its prompts run in windows of `prompt` rows (`session.promptRows`, or what
/// `--max-num-batched-tokens` asked). `--parallel 1` is a prompt window. Above that, a round may be every
/// slot's verify window plus one prompt chunk.
pub fn rowsFor(prompt: usize, parallel: usize) usize {
    if (parallel <= 1) return prompt;
    return @max(prompt, parallel * masks_file.max_rows + rounds.prompt_chunk);
}

/// The session `serve` makes for a checkpoint whose prompts run in windows of `prompt` rows: `--parallel 1`'s,
/// and above that the hall's lone stream's, so the two are one value. `prompt_rows` is said and not left to
/// its default: a model loaded for a round (`rowsFor` above one stream, wider than `promptRows` where that is narrow)
/// would otherwise run a lone stream's prompt in windows as wide as the round, not in those `--parallel 1`
/// runs it in, and `Session.plan` places a prompt's snapshots by its windows.
pub fn sessionFor(prompt: usize, context: usize, reply_stride: usize) session.Options {
    return .{ .capacity = context, .prompt_rows = prompt, .reply_stride = reply_stride };
}

pub fn address(view: cuda.View) i64 {
    return @intCast(@intFromPtr(view.ptr.?));
}

pub const Out = struct {
    ctx: *anyopaque,
    emit: *const fn (*anyopaque, usize, []const i32) void,
    end: *const fn (*anyopaque, usize, request_file.Finish) void,
    fail: *const fn (*anyopaque, usize, []const u8) void,
    /// The prompt's tokens that were not run, once a lone stream's prompt has started (`sched.Round.held`).
    held: *const fn (*anyopaque, usize, usize) void = heldNothing,
    /// What is said of a slot's reply as it ends, before `end` (`sched.Round.tail`).
    tail: *const fn (*anyopaque, usize, sched.Tail) void = tailNothing,
    /// Whether the slot's connection asked for the end, and whether a request waits to be admitted.
    cancelled: *const fn (*anyopaque, usize) bool = neverCancelled,
    waiting: *const fn (*anyopaque) bool = nobodyWaits,
    /// The rows of the slot's next tokens, for a reply that asks for log probabilities (`Ask.logprobs`), before
    /// `emit` gives the tokens; after the reply's own, the row of the token that ended it.
    rows: *const fn (*anyopaque, usize, []const request_file.LogprobRow) void = rowsNothing,

    fn rowsNothing(_: *anyopaque, _: usize, _: []const request_file.LogprobRow) void {}

    fn heldNothing(_: *anyopaque, _: usize, _: usize) void {}

    fn tailNothing(_: *anyopaque, _: usize, _: sched.Tail) void {}

    fn neverCancelled(_: *anyopaque, _: usize) bool {
        return false;
    }

    fn nobodyWaits(_: *anyopaque) bool {
        return false;
    }

    pub fn send(self: Out, slot: usize, tokens: []const i32) void {
        if (tokens.len > 0) self.emit(self.ctx, slot, tokens);
    }

    pub fn sendRows(self: Out, slot: usize, rows: []const request_file.LogprobRow) void {
        if (rows.len > 0) self.rows(self.ctx, slot, rows);
    }

    pub fn stop(self: Out, slot: usize, finish: request_file.Finish) void {
        self.end(self.ctx, slot, finish);
    }
};

/// An image prompt's (spec 019 WS-05), as the request's thread laid it out: the tower's input, the prompt
/// rows its features take and every row's three positions (`session.ImagePrompt`). The request's, lent for
/// the job; the tower runs on the decoding thread, when the prompt's rows are first run (`Sight.encode`).
pub const Picture = struct {
    /// f32 [patches, 1536] and a grid an image, as `tower_file.Tower.encode` takes them.
    pixels: []const f32,
    grids: []const tower_file.Grid,
    /// The prompt position of each feature row, ascending.
    rows: []const u32,
    /// [prompt rows, 3], and what the rows past them add to their position (`attention.Rope`).
    positions: []const [3]i32,
    delta: i32,

    /// The first placeholder row: a state at or below it is a text prompt's, bit for bit (the rows before
    /// it are text, each at its own position in all three), and none above it is kept (`hall_keep.zig`).
    pub fn first(self: Picture) usize {
        return self.rows[0];
    }
};

/// The tower and where the features of an image prompt are made, `serve`'s beside the load (spec 019): one
/// prompt's at a time, as one prompt fills at a time, and the rotary table of each of the pool's slots
/// (`rounds.Slot.rope`), lent to it at the opening.
pub const Sight = struct {
    /// Null in a test of the pool, which writes `features` itself.
    tower: ?*tower_file.Tower,
    /// bf16 [rows, width]: `images.image_tokens` by `tower_file.out_hidden` when served.
    features: cuda.Buffer,
    /// [slots], `capacity` positions each; none for one session, whose state has its own.
    ropes: []attention.Rope,
    /// Whose features `features` holds: a seat's `hall_seats.Seat.seq`; 0, none of the pool's.
    owner: u64 = 0,
    gpa: Allocator,

    pub fn init(gpa: Allocator, tower: ?*tower_file.Tower, rows: usize, width: usize, slots: usize, capacity: usize) !Sight {
        var features: cuda.Buffer = try .init(.bf16, &.{ rows, width });
        errdefer features.deinit();
        const ropes = try gpa.alloc(attention.Rope, slots);
        errdefer gpa.free(ropes);
        var made: usize = 0;
        errdefer for (ropes[0..made]) |*rope| rope.deinit();
        for (ropes) |*rope| {
            rope.* = try .init(capacity);
            made += 1;
        }
        return .{ .tower = tower, .features = features, .ropes = ropes, .gpa = gpa };
    }

    pub fn deinit(self: *Sight) void {
        for (self.ropes) |*rope| rope.deinit();
        self.gpa.free(self.ropes);
        self.features.deinit();
    }

    /// `picture`'s features through the tower, for `owner` (0: a session's), unless they are there already.
    pub fn encode(self: *Sight, gpa: Allocator, picture: Picture, owner: u64) !forward.Image {
        const rows = picture.rows.len;
        if (rows == 0 or rows > @as(usize, @intCast(self.features.view.size[0]))) return error.InvalidImage;
        const out = self.features.rows(rows);
        if (owner == 0 or self.owner != owner) if (self.tower) |tower| {
            self.owner = 0;
            try tower.encode(gpa, picture.pixels, picture.grids, out);
            self.owner = owner;
        };
        return .{ .features = out, .rows = picture.rows };
    }
};

/// What a request asks of the decoder, as its front made it. The slices are the caller's: whoever keeps one
/// past the call copies it, but for `constraint`, `mask` and `image`, which live as long as the request.
pub const Ask = struct {
    prompt: []const i32,
    settings: sampling.Sampling,
    max_tokens: usize,
    depth: usize,
    opening: usize,
    constraint: ?grammar.Constraint = null,
    /// The constraint's scratch.
    mask: []u32 = &.{},
    /// `</think>` where the prompt leaves the reply thinking.
    gate: ?i32 = null,
    /// Where the prompt's messages start (`session.ReplyOptions.starts`).
    starts: []const usize = &.{},
    /// Where the prefix other conversations share ends (`session.ReplyOptions.kept`).
    kept: ?usize = null,
    /// Set where the request waited for another prompt's fill (spec 152 WS-06): the decoding thread's.
    wait: prefix_wait.Waiting = .{},
    /// The tokens that end this reply where they are not the checkpoint's (`session.ReplyOptions.eos`).
    eos: ?[]const u32 = null,
    /// A warm pass (`server/served.zig`'s `Ask.warm`): run for the states its prompt keeps, only while nothing
    /// else is live (`hall_warm.zig`).
    warm: bool = false,
    /// Log probabilities, this many best tokens a row (`session.ReplyOptions.logprobs`).
    logprobs: ?u8 = null,
    /// An image prompt (spec 019 WS-05), the request's: it goes on from a text state at or below its first
    /// placeholder row at most, as vLLM's blocks before an image's are a text prompt's.
    image: ?Picture = null,
};

/// A reply that left the pool, for its line in the log.
pub const Report = struct {
    prompt: usize,
    held: usize,
    prompt_seconds: f64,
    /// Whether the reply's first token came, and the windows it ran.
    first: bool,
    rounds: usize,
    tokens: usize,
    seconds: f64,
    accepted: usize,
    drafted: usize,
    /// As `session.Reply.drafts`: two replies with the same one were drafted alike.
    drafts: u64,
    /// The copies of its context the reply was offered; all zero where it was offered none.
    copies: copied.Stats = .{},
    /// The head's levels in its rounds (`mtp.Paying.Counts`).
    levels: mtp.Paying.Counts = .{},
    /// States the pool's store holds and their device bytes, and the streams live beside this one.
    states: usize,
    states_bytes: usize,
    beside: usize,
};

const Captured = struct { count: usize, nanoseconds: u64 };

pub const EngineOptions = struct {
    slots: usize,
    /// Positions of the shared pool, a whole number of extents: `poolFor` of the streams and the context.
    pool: usize,
    /// The lone stream's session, as `serve --parallel 1` makes it for the same model and flags. The pool
    /// takes two of these for its own: `capacity` is the context a stream's prompt and reply share, and
    /// `prompt_rows` the widest prompt window. Its `eager` (no graphs at all) is left unset.
    session: session.Options,
    chunk: usize = rounds.prompt_chunk,
    /// The rows a round is made for where fewer than the model's (`Model.max_rows`, `openAfter`): `serve
    /// --max-num-batched-tokens` widens the lone stream's prompt windows and the model's rows, and the pool's
    /// rounds stay the rows the default loads (spec 105 WS-07). Null: the model's (two ranks' pool, whose rounds
    /// take the flag's rows, spec 149 WS-06).
    rows: ?usize = null,
    depth: usize = session.default_depth,
    /// `serve --lazy-graphs`: nothing is captured at load. The lone stream's graphs are captured as its
    /// windows first need them, and the pool's rounds launch their kernels (`Engine.eager`).
    lazy_graphs: bool = false,
    /// The open captures the pool's first graphs itself (`Engine.captureFirst`). On two ranks the load's
    /// captures are told through the link after the barrier (`Link.captureFirst`, rank 1 in `follow`),
    /// so the open skips them (`servePool` passes false on both ranks).
    capture_at_open: bool = true,
    /// Device bytes the pool's kept states hold at most, the buffers kept for the next one included: what
    /// sixteen states of both recurrent sides and a whole pooled buffer held before spec 054, so the pool's
    /// ceiling is where it was (112.0 GiB on NVFP4). A state is now one side and the pooled rows below it,
    /// 110 MiB and 832 bytes a position: 58 of them at 8,000 positions, 21 at 262,144.
    states_bytes: usize = default_states_bytes,
    /// One rank of two (`hall_ranks.zig`): the pool is opened at once, with no lone session, and captures its
    /// graphs at the load as a lone rank's does, told through the link, or as its rounds first need them
    /// under `--lazy-graphs`.
    ranks: ?*@import("hall_ranks.zig").Link = null,
    /// `serve`'s tower, where it serves images (spec 019 WS-05): `slots` ropes, the slots' own.
    sight: ?*Sight = null,
    /// How long a request waits for a fill that shares a cut of its prompt (`decode/prefix_wait.zig`;
    /// ZSPARK_PREFIX_WAIT_MS, 0 off). Null reads the environment.
    prefix_wait: ?prefix_wait.Wait = null,
};

pub const default_states_bytes: usize = 6816 << 20;

/// The one live stream's request, while the session decodes it.
const Lone = struct {
    slot: usize,
    /// The request, its prompt and its messages' starts copied.
    ask: Ask,
    /// Of `given`, the tokens the stream's client had before this admission.
    before: usize,
    /// Every token of the reply so far.
    given: std.ArrayList(i32) = .empty,
    /// The reply ended; the scheduler has yet to retire the slot.
    ended: bool = false,
    /// The scheduler was told how much of the prompt was held.
    told: bool = false,
    /// What `Session.reply` said, over every call the reply took: a call ends early when another request
    /// waits, and the next goes on from the state if that request has gone.
    done: ?session.Reply = null,

    fn deinit(self: *Lone, gpa: Allocator) void {
        gpa.free(self.ask.prompt);
        gpa.free(self.ask.starts);
        self.given.deinit(gpa);
    }
};

/// The decoder behind `Hall`, without its scheduler and its text: one live stream on a session, several on the
/// shared pool. Its calls are the decoding thread's.
pub const Core = struct {
    gpa: Allocator,
    io: std.Io,
    m: model_file.Model,
    tokens: *const tokenizer.Tokenizer,
    eos: []const u32,
    options: EngineOptions,
    /// One live stream is this session's, as `--parallel 1`'s is: it reuses a prefix by the same states.
    /// Destroyed when `engine` opens, inside it: the sequence it holds is carried into the pool first, rows
    /// and states (`migrate`).
    session: ?*session.Session = null,
    /// Set when a second stream is admitted, and for the rest of the server's life.
    engine: ?Engine = null,
    /// The session was freed and `engine` did not open. The next round fails the streams that are live.
    broken: bool = false,
    lone: ?Lone = null,
    /// `serve --spill-gib`'s store, the caller's (`spilling`): the session's until the pool opens.
    spill: ?*hall_spill.Store = null,
    /// The prompt and the reply's tokens so far, for `Session.reply`.
    sequence: std.ArrayList(i32) = .empty,
    /// The fill wait bound the pool seats use (`Ask.wait`); a lone stream fills alone, so it waits on
    /// nothing.
    prefix_wait: prefix_wait.Wait,
    /// The `Waiting` the last `admit` that returned `error.Wait` chose, for the caller to keep in its job
    /// (the `Ask` it got was a copy); null where the wait was no fill's.
    wait_offer: ?prefix_wait.Waiting = null,
    /// The requests whose line was logged, and the last line.
    reports: usize = 0,
    line: [640]u8 = undefined,
    line_len: usize = 0,
    /// Only the decoding thread reads the store; HTTP readers use this publication.
    cache: sched.Cache = .{},
    /// `Info.prompt_cache_capacity`'s smallest state the keep planner admits; `options.states_bytes` is the
    /// budget beside it, so the fit `serve` applies after `create` is what is told.
    minimum_state_bytes: usize = 0,

    pub fn create(gpa: Allocator, io: std.Io, m: model_file.Model, tokens: *const tokenizer.Tokenizer, eos: []const u32, options: EngineOptions) !*Core {
        const self = try gpa.create(Core);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .io = io, .m = m, .tokens = tokens, .eos = eos, .options = options, .prefix_wait = options.prefix_wait orelse prefix_wait.Wait.fromEnv() };
        if (options.ranks) |link| {
            // two ranks: every stream is the pool's, the first too (`hall_ranks.zig`); the load's captures
            // are told through the link, rank 0 after the barrier
            self.engine = try Engine.open(gpa, m, eos, options);
            const engine = &self.engine.?;
            engine.io = io;
            link.bind(engine);
            engine.bindPool();
            self.minimum_state_bytes = hall_keep.bytesAt(engine, 0);
            self.publishCache();
            return self;
        }
        const held = try session.Session.create(gpa, io, m, tokens, eos, options.session);
        errdefer held.destroy();
        if (!options.lazy_graphs) try held.prepare(options.depth, session.prepared_buckets);
        self.session = held;
        self.minimum_state_bytes = minimumStateBytes(&held.state);
        self.publishCache();
        return self;
    }

    /// The states that leave the session and the pool go to `store` and are read from it, before any request.
    pub fn spilling(self: *Core, store: *hall_spill.Store) void {
        self.spill = store;
        if (self.session) |held| hall_spill.learnt.spilling(held, store);
    }

    pub fn destroy(self: *Core) void {
        // what is held when the server stops goes to disk as what left did (`spill.zig`)
        if (self.engine) |*engine| hall_spill.flush(engine) else if (self.session) |held| hall_spill.learnt.flush(held, true);
        if (self.lone) |*lone| lone.deinit(self.gpa);
        self.sequence.deinit(self.gpa);
        if (self.session) |held| held.destroy();
        if (self.engine) |*engine| engine.deinit();
        self.gpa.destroy(self);
    }

    pub fn captured(self: *const Core) Captured {
        if (self.session) |held| {
            const got = held.captured();
            return .{ .count = got.count, .nanoseconds = got.nanoseconds };
        }
        if (self.engine) |*engine| return .{
            .count = engine.graphs.cache.count(),
            .nanoseconds = engine.graphs.cache.capturing(),
        };
        return .{ .count = 0, .nanoseconds = 0 };
    }

    const Admitted = struct { slot: usize, cached: usize };

    /// `ask` is the request and `sent` the tokens of its reply its client already has (a stream taken off and
    /// admitted again). `error.Wait`: no room now; `error.Never`: the prompt cannot fit.
    pub fn admit(self: *Core, ask: Ask, sent: []const i32) !Admitted {
        defer self.publishCache();
        if (self.broken) return error.Failed;
        if (ask.max_tokens < sent.len) return error.Never;
        if (ask.warm) try hall_warm.admits(self, ask);
        self.wait_offer = null;
        if (self.engine == null) {
            // a warm pass gives the session up before a second stream would open the pool
            if (self.lone) |lone| if (lone.ask.warm) return error.Wait;
            if (self.lone == null) {
                if (ask.prompt.len < 1 or ask.prompt.len + ask.max_tokens > self.options.session.capacity) return error.Never;
                const prompt = try self.gpa.dupe(i32, ask.prompt);
                errdefer self.gpa.free(prompt);
                const starts = try self.gpa.dupe(usize, ask.starts);
                errdefer self.gpa.free(starts);
                var given: std.ArrayList(i32) = .empty;
                errdefer given.deinit(self.gpa);
                try given.appendSlice(self.gpa, sent);
                var own = ask;
                own.prompt = prompt;
                own.starts = starts;
                self.lone = .{ .slot = 0, .ask = own, .before = sent.len, .given = given };
                // How much of the prompt the session holds is known when it starts: `round` tells it.
                return .{ .slot = 0, .cached = 0 };
            }
            try self.migrate();
        }
        const engine = &self.engine.?;
        const slot = for (engine.seats, 0..) |seat, i| {
            if (seat == null) break i;
        } else return error.Wait;
        // A request that waited for a fill holds it still only while the fill lives short of the cut; the
        // caller keeps the record (`Ask.wait`, `wait_offer`), this copy's writes would die with it. An
        // image prompt never waits (`Engine.begin` says why).
        var fills: [most_parallel]prefix_wait.Filler = undefined;
        var cuts: [most_parallel][prefix.prompt_marks + 1]usize = undefined;
        const fills_ = self.leaders(&fills, &cuts);
        if (ask.wait.leader != null) {
            if (prefix_wait.holds(&self.prefix_wait, &ask.wait, fills_, nowNs(self.io))) return error.Wait;
        } else if (!ask.wait.waited and ask.image == null and sent.len == 0) {
            if (prefix_wait.pick(&self.prefix_wait, fills_, ask.prompt, 0, nowNs(self.io))) |waiting| {
                self.prefix_wait.waits += 1;
                self.wait_offer = waiting;
                return error.Wait;
            }
        }
        const full = try self.gpa.alloc(i32, ask.prompt.len + sent.len);
        defer self.gpa.free(full);
        @memcpy(full[0..ask.prompt.len], ask.prompt);
        @memcpy(full[ask.prompt.len..], sent);
        var rest = ask;
        rest.prompt = full;
        rest.max_tokens = ask.max_tokens - sent.len;
        return .{ .slot = slot, .cached = try engine.begin(slot, rest, sent.len) };
    }

    /// The prompt fills the pool runs, as `prefix_wait`'s fillers: `marks` the seats' plans still to keep
    /// a state at, `reached` where each fill stands.
    fn leaders(self: *Core, out: *[most_parallel]prefix_wait.Filler, cuts: *[most_parallel][prefix.prompt_marks + 1]usize) []prefix_wait.Filler {
        const engine = &self.engine.?;
        var n: usize = 0;
        for (engine.seats) |*held| {
            const seat = &(held.* orelse continue);
            if (seat.phase != .fill or seat.warm) continue;
            // the marks the plan still names: `taken` holds only what the windows run so far kept, so the
            // rest of the prompt is played on a copy of the planner, a chunk at a time as the round runs it
            // (`hall_keep.chunk`'s stop at the first mark a window plans).
            var list = seat.list;
            var dropped = seat.dropped;
            var planner = seat.planner;
            planner.list = &list;
            planner.dropped = &dropped;
            var filled = seat.filled;
            while (planner.from < seat.prompt.len) {
                var take = @min(engine.chunk, seat.prompt.len - filled);
                if (engine.keeps.cut) {
                    var probe_list = list;
                    var probe_dropped = dropped;
                    var probe = planner;
                    probe.list = &probe_list;
                    probe.dropped = &probe_dropped;
                    probe.window(filled + take);
                    for (probe.taken()) |pos| if (pos > filled) {
                        take = pos - filled;
                        break;
                    };
                }
                planner.window(filled + take);
                filled += take;
            }
            var count: usize = 0;
            for (planner.taken()) |at| {
                if (at > seat.filled and at < seat.prompt.len) {
                    cuts[n][count] = at;
                    count += 1;
                }
            }
            out[n] = .{ .context = seat, .prompt = seat.prompt, .reached = seat.filled, .marks = cuts[n][0..count], .id = @intCast(seat.seq) };
            n += 1;
        }
        return out[0..n];
    }

    /// The wait's clock, the awake one's nanoseconds.
    fn nowNs(io: std.Io) i96 {
        return std.Io.Clock.awake.now(io).toNanoseconds();
    }

    /// What `migrate` hands `Engine.openAfter`: the session's sequence goes into the extents and the session
    /// is freed, between the extents and the rest of the pool.
    const Handing = struct {
        core: *Core,
        carried: hall_carry.Carried,
        /// The pool's extents were made; the rows were copied; the session was freed.
        made: ?std.Io.Timestamp = null,
        copied: ?std.Io.Timestamp = null,
        freed: ?std.Io.Timestamp = null,
        /// Device bytes of buffers while the session's caches and the extents both stood.
        peak: usize = 0,
        /// The session's DFlash2 drafter (`session.Options.drafter`), taken before it is freed: the pool
        /// seats it (`Engine.seatDrafter`).
        flash: ?*rig.Rig(2) = null,

        fn carry(context: *anyopaque, rooms: hall_engine.Rooms) void {
            const self: *Handing = @ptrCast(@alignCast(context));
            const core = self.core;
            const held = core.session orelse return;
            self.made = .now(core.io, .awake);
            hall_carry.take(&self.carried, held, rooms) catch |err| {
                std.log.err("carrying the session's states into the pool: {t}", .{err});
                hall_carry.drop(&self.carried, rooms);
            };
            // the copies read the caches that are freed next
            cuda.synchronize() catch {};
            self.copied = .now(core.io, .awake);
            self.peak = cuda.account.bytes;
            // the sequences it left that were not carried leave here (their states went with those that were)
            hall_spill.learnt.flush(held, false);
            self.flash = held.takeFlash();
            held.destroy();
            core.session = null;
            self.freed = .now(core.io, .awake);
        }
    };

    /// The second stream. The shared pool is opened and the session freed inside that (`Handing`): what the
    /// session held is carried into the pool first (`hall_carry.zig`), its sequence's rows and its states.
    /// The stream already live is then begun on the pool as any request is, and finds them: it goes on from
    /// the row it stood at, the one row of its last token run again. The tokens it has sent are part of its
    /// prompt, the constraint is advanced over them, and none is sent again. A session with nothing to carry
    /// leaves the stream to run from its prompt's start.
    fn migrate(self: *Core) !void {
        const lone = &self.lone.?;
        std.debug.assert(!lone.ended);
        const full = try self.gpa.alloc(i32, lone.ask.prompt.len + lone.given.items.len);
        defer self.gpa.free(full);
        @memcpy(full[0..lone.ask.prompt.len], lone.ask.prompt);
        @memcpy(full[lone.ask.prompt.len..], lone.given.items);
        std.log.info("opening the shared pool for a second stream", .{});
        const start: std.Io.Timestamp = .now(self.io, .awake);
        // `--learn`'s store, which the session read and wrote, is read by the pool
        const learned = if (self.session) |held| held.learned else null;
        var handing: Handing = .{ .core = self, .carried = .{ .gpa = self.gpa } };
        defer handing.carried.deinit();
        // a drafter the pool did not take
        defer if (handing.flash) |flash| {
            flash.deinit();
            self.gpa.destroy(flash);
        };
        self.engine = Engine.openAfter(self.gpa, self.m, self.eos, self.options, .{ .context = &handing, .call = Handing.carry }) catch |first| opened: {
            // The extents did not fit beside the session: it is freed and the pool opened empty, as it was
            // before spec 054. With the session gone already, the rest did not fit.
            const held = self.session orelse {
                self.broken = true;
                return first;
            };
            std.log.warn("the pool's extents beside the session: {t}; opening it with nothing carried", .{first});
            handing.flash = held.takeFlash();
            held.destroy();
            self.session = null;
            break :opened Engine.open(self.gpa, self.m, self.eos, self.options) catch |err| {
                self.broken = true;
                return err;
            };
        };
        const opened: std.Io.Timestamp = .now(self.io, .awake);
        const engine = &self.engine.?;
        engine.io = self.io;
        engine.bindPool();
        if (self.spill) |store| hall_spill.attach(engine, store);
        engine.recall = self.options.session.recall;
        engine.keeps.learned = learned;
        // DFlash2 drafts the pool's streams as it drafted the session's (spec 126)
        if (handing.flash) |flash| {
            handing.flash = null;
            engine.seatDrafter(flash) catch |err| {
                self.broken = true;
                return err;
            };
        }
        const carried = .{ handing.carried.rows, handing.carried.states };
        hall_carry.give(&handing.carried, engine);
        // the first prompts' states find their buffers made (`hall_keep.ready`)
        engine.readyStates();
        // its messages' starts go with it: the pool keeps its states there as the session did
        var rest = lone.ask;
        rest.prompt = full;
        rest.max_tokens = lone.ask.max_tokens - lone.given.items.len;
        const cached = engine.begin(lone.slot, rest, lone.given.items.len) catch |err| {
            self.broken = true;
            return err;
        };
        lone.deinit(self.gpa);
        self.lone = null;
        const capture = @as(f64, @floatFromInt(engine.graphs.cache.capturing())) / 1e9;
        std.log.info("shared pool is open in {d:.3} s: {d:.3} s the extents, {d:.3} s carrying {d} positions and {d} states ({d} held), {d:.3} s freeing the session, {d:.3} s the rest of the pool and {d:.3} s its graphs; the live stream goes on from {d} of {d}; {d} MiB of device buffers with the session and the extents, {d} now", .{
            between(start, .now(self.io, .awake)),  between(start, handing.made),
            between(handing.made, handing.copied),  carried[0],
            carried[1],                             engine.store.marks.items.len,
            between(handing.copied, handing.freed), @max(between(handing.freed, opened) - capture, 0),
            capture,                                cached,
            full.len,                               handing.peak >> 20,
            cuda.account.bytes >> 20,
        });
    }

    pub fn round(self: *Core, out: Out) sched.Outcome {
        defer self.publishCache();
        if (self.broken) return .failed;
        if (self.engine) |*engine| {
            const outcome = engine.round(out);
            engine.publishAhead(out);
            return outcome;
        }
        if (self.lone == null) return .ran;
        self.loneRound(out) catch |err| {
            std.log.err("decoding a reply: {t}", .{err});
            out.fail(out.ctx, self.lone.?.slot, "decoding failed");
            self.lone.?.ended = true;
        };
        return .ran;
    }

    const Taking = struct {
        core: *Core,
        out: Out,
        /// The prompt's tokens this call's reply did not run (`session.Tap.heldFn`).
        cached: usize = 0,

        fn take(context: *anyopaque, token: i32) void {
            const self: *Taking = @ptrCast(@alignCast(context));
            const lone = &self.core.lone.?;
            // room for every token was reserved before the reply began
            lone.given.appendAssumeCapacity(token);
            // What the prompt found held is told before the reply's first token, for who takes the tokens to
            // know it by then (`sched.Taker.began`); a reply that goes on after a cut says its first part's.
            if (!lone.told) {
                self.out.held(self.out.ctx, lone.slot, if (lone.done) |sum| sum.cached else self.cached);
                lone.told = true;
            }
            self.out.send(lone.slot, &.{token});
        }

        fn row(context: *anyopaque, one: *const request_file.LogprobRow) void {
            const self: *Taking = @ptrCast(@alignCast(context));
            self.out.sendRows(self.core.lone.?.slot, one[0..1]);
        }

        fn held(context: *anyopaque, cached: usize) void {
            const self: *Taking = @ptrCast(@alignCast(context));
            self.cached = cached;
        }

        fn halt(context: *anyopaque) bool {
            const self: *Taking = @ptrCast(@alignCast(context));
            const lone = &self.core.lone.?;
            // a warm pass that has run its prompt settles its states with its one token: a cut would drop them
            if (lone.ask.warm) return self.out.cancelled(self.out.ctx, lone.slot);
            return self.out.cancelled(self.out.ctx, lone.slot) or self.out.waiting(self.out.ctx);
        }
    };

    /// A lone stream's round is its reply, as `--parallel 1` decodes one: `Session.reply`, from the state the
    /// session holds, with the snapshots, the reply's stride and `settle`. It returns when the reply ends,
    /// its client hangs up or another request waits; in the last case the scheduler admits that request,
    /// which moves this stream to the pool (`migrate`), or finds it gone, and the next round goes on.
    fn loneRound(self: *Core, out: Out) !void {
        const lone = &self.lone.?;
        if (lone.ended) return;
        if (hall_warm.stepsAside(lone.ask.warm, lone.slot, out)) {
            lone.ended = true;
            return;
        }
        // Asked before the prompt is run too: a request that waits already is admitted first, and the prompt
        // then runs once, in the pool.
        if (out.cancelled(out.ctx, lone.slot) or out.waiting(out.ctx)) return;
        const held = self.session.?;
        const sent = lone.given.items.len;
        try lone.given.ensureTotalCapacity(self.gpa, lone.ask.max_tokens);
        self.sequence.clearRetainingCapacity();
        try self.sequence.appendSlice(self.gpa, lone.ask.prompt);
        try self.sequence.appendSlice(self.gpa, lone.given.items);
        var taking: Taking = .{ .core = self, .out = out };
        const none: request_file.Sink = .{ .context = undefined, .writeFn = unasked };
        // an image prompt runs from an empty state, its features made again where a call before was cut
        const image: ?session.ImagePrompt = if (lone.ask.image) |picture| .{
            .image = try (self.options.sight orelse return error.InvalidImage).encode(self.gpa, picture, 0),
            .positions = picture.positions,
            .delta = picture.delta,
        } else null;
        const done = try held.reply(self.sequence.items, lone.ask.settings, .{
            .max_tokens = lone.ask.max_tokens - sent,
            .depth = lone.ask.depth,
            .opening = lone.ask.opening,
            .constraint = lone.ask.constraint,
            .starts = lone.ask.starts,
            .kept = lone.ask.kept,
            .tap = .{ .context = &taking, .takeFn = Taking.take, .haltFn = Taking.halt, .heldFn = Taking.held, .rowFn = Taking.row },
            .sent = sent,
            .eos = lone.ask.eos,
            .logprobs = lone.ask.logprobs,
            .image = image,
        }, none);
        if (lone.done) |*sum| {
            sum.finish = done.finish;
            sum.end = done.end;
            sum.tokens += done.tokens;
            sum.seconds += done.seconds;
            sum.stats.rounds += done.stats.rounds;
            sum.stats.drafted += done.stats.drafted;
            sum.stats.accepted += done.stats.accepted;
        } else lone.done = done;
        // The request is told what its own prompt found held. A stream moved to the pool before its first
        // token ran its prompt there from the start, and is told none.
        if (!lone.told and (done.finish != .cut or lone.given.items.len > lone.before)) {
            out.held(out.ctx, lone.slot, lone.done.?.cached);
            lone.told = true;
        }
        if (done.finish == .cut) return;
        const sum = lone.done.?;
        out.tail(out.ctx, lone.slot, .{ .end = sum.end, .rounds = sum.stats.rounds, .drafted = sum.stats.drafted, .accepted = sum.stats.accepted, .prompt_seconds = sum.prompt_seconds });
        out.stop(lone.slot, done.finish);
        lone.ended = true;
    }

    pub fn unasked(_: *anyopaque, _: request_file.Kind, _: []const u8) request_file.Sink.Signal {
        return .more;
    }

    /// Returns the reply's counts as far as it ran (`sched.Decoder.Table.retire`).
    pub fn retire(self: *Core, slot: usize, why: sched.Leave, kept: usize) sched.Tail {
        defer self.publishCache();
        if (self.engine) |*engine| {
            const report = engine.retire(slot, why, kept) orelse return .{};
            const counted: sched.Tail = .{ .rounds = report.rounds, .drafted = report.drafted, .accepted = report.accepted, .prompt_seconds = if (report.first) report.prompt_seconds else null };
            var copies_text: [96]u8 = undefined;
            var levels_text: [160]u8 = undefined;
            var w: std.Io.Writer = .fixed(&self.line);
            w.print("prompt {d} tokens, {d} held, in {d:.2} s; reply {d} tokens in {d:.2} s, {d:.1} tok/s; {d} of {d} drafts kept{s}{s}; in the pool beside {d}, {d} states held in {d} MiB, {d} MiB of device buffers", .{
                report.prompt,                    report.held,
                report.prompt_seconds,            report.tokens,
                report.seconds,                   @as(f64, @floatFromInt(report.tokens)) / @max(report.seconds, 1e-9),
                report.accepted,                  report.drafted,
                report.copies.line(&copies_text), report.levels.line(&levels_text),
                report.beside,                    report.states,
                report.states_bytes >> 20,        cuda.account.bytes >> 20,
            }) catch {};
            self.said(w.buffered().len);
            return counted;
        }
        const lone = if (self.lone) |*held| held else return .{};
        if (lone.slot != slot) return .{};
        var counted: sched.Tail = .{};
        if (lone.done) |done| {
            // a reply that gave no token has run no prompt to its end
            counted = .{ .rounds = done.stats.rounds, .drafted = done.stats.drafted, .accepted = done.stats.accepted, .prompt_seconds = if (lone.given.items.len > 0) done.prompt_seconds else null };
            if (self.session) |held| {
                const text = held.replyLine(&self.line, lone.ask.prompt.len, done);
                self.said(text.len);
            }
        }
        lone.deinit(self.gpa);
        self.lone = null;
        return counted;
    }

    fn publishCache(self: *Core) void {
        const snapshot: sched.Cache.Snapshot = if (self.engine) |*engine|
            .{ .states = engine.store.marks.items.len, .bytes = engine.keeps.bytes() }
        else if (self.session) |held|
            sessionCache(held)
        else
            .{};
        self.cache.publish(snapshot);
    }

    /// The line `--parallel 1` writes for a reply (`served.zig`), for every request here too.
    fn said(self: *Core, len: usize) void {
        self.line_len = len;
        self.reports += 1;
        std.log.info("{s}", .{self.line[0..len]});
    }

    pub fn preempt(self: *Core, slot: usize) void {
        defer self.publishCache();
        if (self.engine) |*engine| return engine.preempt(slot);
        if (self.lone) |*lone| if (lone.slot == slot) {
            lone.deinit(self.gpa);
            self.lone = null;
        };
    }
};

/// Retained snapshots and parked sequences, with their reusable device storage.
pub fn sessionCache(held: *const session.Session) sched.Cache.Snapshot {
    var states: usize = held.snapshots.items.len + held.strides.items.len;
    for (held.lot.list.items) |one| states += one.states.items.len;
    return .{ .states = states, .bytes = held.snapshotBytes() + held.lot.bytes() };
}

/// The keep planner's smallest state: one recurrent side, conv inputs and PLE tails, no pooled rows.
pub fn minimumStateBytes(state: *const forward.State) usize {
    var bytes: usize = 0;
    for (state.recurrent, state.conv) |pair, conv| bytes += pair[0].bytes() + conv.bytes();
    for (state.ple) |branch| bytes += branch.tail.bytes();
    return bytes;
}

pub fn between(from: ?std.Io.Timestamp, to: ?std.Io.Timestamp) f64 {
    const a = from orelse return 0;
    const b = to orelse return 0;
    return @as(f64, @floatFromInt(a.durationTo(b).toNanoseconds())) / 1e9;
}

// -- the text side -----------------------------------------------------------------------------------------------

pub const Hall = struct {
    gpa: Allocator,
    io: std.Io,
    options: EngineOptions,
    core: *Core,
    scheduler: sched.Scheduler,
    think_open: ?i32,
    think_close: ?i32,

    pub fn create(gpa: Allocator, io: std.Io, m: model_file.Model, tokens: *const tokenizer.Tokenizer, eos: []const u32, options: EngineOptions) !*Hall {
        const self = try gpa.create(Hall);
        errdefer gpa.destroy(self);
        const open = tokens.addedId("<think>");
        const close = tokens.addedId("</think>");
        const core = try Core.create(gpa, io, m, tokens, eos, options);
        errdefer core.destroy();
        self.* = .{
            .gpa = gpa,
            .io = io,
            .options = options,
            .core = core,
            .scheduler = undefined,
            .think_open = if (open) |id| @intCast(id) else null,
            .think_close = if (close) |id| @intCast(id) else null,
        };
        self.scheduler = try sched.Scheduler.init(gpa, io, .{ .context = self, .table = &decoder_table }, .{ .slots = options.slots });
        errdefer self.scheduler.deinit();
        try self.scheduler.start();
        return self;
    }

    pub fn destroy(self: *Hall) void {
        self.scheduler.close();
        self.scheduler.deinit();
        self.core.destroy();
        self.gpa.destroy(self);
    }

    pub fn captured(self: *const Hall) Captured {
        return self.core.captured();
    }

    /// The checkpoint's thinking tags, where it has both.
    pub fn think(self: *const Hall) ?chat_file.Think {
        return .{ .open = self.think_open orelse return null, .close = self.think_close orelse return null };
    }
};
