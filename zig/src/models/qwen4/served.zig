//! Flash Next behind TensorFold's `engine_api.Engine` (spec 059): what `server/served.zig`'s engine asks of a
//! model. The session and the pool behind it are what `zspark serve` always ran, so a reply's tokens are
//! theirs: on a session a request holds `decoding` and runs `Session.reply` with a tap, in the pool it queues
//! a job and takes its tokens (`Scheduler.decode`).
//!
//! The token that ends a reply is none of zspark's reply (`session.Tap`, `sched.Tail`), so it is sent after
//! the reply's own (`Live.closing`). A request's `eos` is the reply's (`ReplyOptions.eos`).
//!
//! A prompt's marks are placed at its messages' starts, found in the ids, and `shared_prefixes` says which of
//! them a harness's prefix ends at (`decode/marks.zig`'s `keptOf`, GLM's alike). What is Flash Next's own here:
//! the reply's first window
//! (`session.openingFor`); the pool's counts in `status`. A session's tap says the prompt has run when it is
//! first asked to halt, as TensorFold's Flash Next host does at the prompt's end.
//!
//! Images (spec 019 WS-04, WS-05): where `serve` loaded the tower (FP4-Local on one rank, `Config.servesImages`),
//! a request's images (`Ask.images`, the host's `Offer`'s patches) are laid out with 0.6.5's positions (`Eye`).
//! On a session they go through the tower in its turn and the prompt runs from an empty state
//! (`ReplyOptions.image`); in the pool the job carries them (`hall.Ask.image`) and the tower runs on the
//! decoding thread when the prompt's rows do, from a text state at or below the first image row.

const std = @import("std");
const Allocator = std.mem.Allocator;
const request_file = @import("../../server/request.zig");
const sched = @import("../../server/scheduler.zig");
const shared = @import("../../server/served.zig");
const masks_file = @import("../../decode/masks.zig");
const marks_file = @import("../../decode/marks.zig");
const hall_file = @import("pool/hall.zig");
const hall_front = @import("pool/hall_front.zig");
const hall_warm = @import("pool/hall_warm.zig");
const pair_file = @import("session/pair.zig");
const session_file = @import("session/session.zig");
const api = @import("engine_api");
const cuda = @import("../../cuda/cuda.zig");
const images = @import("../../server/images.zig");
const processor = @import("../../text/vision.zig");
const tower_file = @import("vision.zig");

/// What an image prompt is made with (`Behind.eye`): the host's offer of the checkpoint's image settings, and
/// the tower with where it writes a prompt's features and the pool's slots' rotary tables (`hall.Sight`),
/// made at load.
pub const Eye = struct {
    offer: images.Offer,
    sight: hall_file.Sight,

    /// `slots` rotary tables of `capacity` positions: the pool's streams, none for one session.
    pub fn init(gpa: Allocator, tower: *tower_file.Tower, settings: processor.Settings, slots: usize, capacity: usize) !Eye {
        return .{ .offer = .init(settings), .sight = try .init(gpa, tower, images.image_tokens, tower_file.out_hidden, slots, capacity) };
    }

    pub fn deinit(self: *Eye) void {
        self.sight.deinit();
    }

    /// The request's images laid out on the request's thread: the tower's input, the rows their features take
    /// and every row's positions as 0.6.5's layout gives them, in `arena`. The tower runs where the prompt
    /// does (`hall.Sight.encode`).
    pub fn picture(self: *const Eye, arena: Allocator, ids: []const i32, given: []const api.Image, said: *request_file.Diagnostic) !hall_file.Picture {
        const s = self.offer.settings;
        const grids = try arena.alloc(tower_file.Grid, given.len);
        const shapes = try arena.alloc([3]u32, given.len);
        var values: usize = 0;
        var rows: usize = 0;
        for (given, grids, shapes) |img, *g, *shape| {
            g.* = .{ .t = 1, .h = img.gh, .w = img.gw };
            shape.* = .{ 1, img.gh, img.gw };
            values += img.pixels.len;
            rows += img.tokens;
        }
        if (rows > images.image_tokens) {
            said.set("the request's images take {d} rows; {d} at most", .{ rows, images.image_tokens });
            return error.BadRequest;
        }
        const pixels = try arena.alloc(f32, values);
        var at: usize = 0;
        for (given) |img| {
            @memcpy(pixels[at..][0..img.pixels.len], img.pixels);
            at += img.pixels.len;
        }

        const tokens = try arena.alloc(u32, ids.len);
        for (tokens, ids) |*t, id| t.* = @intCast(id);
        var problem: processor.Problem = .{};
        const layout = processor.Layout.init(arena, tokens, shapes, .{ .start = s.vision_start, .image = s.image_token, .end = s.vision_end, .video = s.video_token }, s.merge, &problem) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => {
                said.set("{s}", .{problem.message()});
                return error.BadRequest;
            },
        };
        const positions = try arena.alloc([3]i32, ids.len);
        for (positions, 0..) |*p, i| p.* = layout.position(@intCast(i));
        const feature_rows = try arena.alloc(u32, rows);
        var n: usize = 0;
        for (given) |img| for (0..img.tokens) |j| {
            feature_rows[n] = img.at + @as(u32, @intCast(j));
            n += 1;
        };
        return .{ .pixels = pixels, .grids = grids, .rows = feature_rows, .positions = positions, .delta = layout.shift };
    }

    /// A session's image prompt (`ReplyOptions.image`): the request's images through the tower now, on the
    /// session's turn; the features stay `sight`'s until the next prompt's.
    pub fn prompt(self: *Eye, gpa: Allocator, arena: Allocator, ids: []const i32, given: []const api.Image, said: *request_file.Diagnostic) !session_file.ImagePrompt {
        const laid = try self.picture(arena, ids, given, said);
        return .{ .image = try self.sight.encode(gpa, laid, 0), .positions = laid.positions, .delta = laid.delta };
    }
};

/// What `server/served.zig` asks of a model (`shared.Served`'s doc comment).
const flash_next = struct {
    /// One of the two: `--parallel 1`'s session, or the pool above it.
    pub const Behind = struct {
        session: ?*session_file.Session = null,
        hall: ?*hall_file.Hall = null,
        /// Rank 0 of two (`session/pair.zig`). Set, the session and the hall are not: one reply at a time,
        /// drafted by the head beside rank 0's share where `depth` is not 0, its prompt run from where it
        /// parts from the sequence the pair holds (spec 094).
        leader: ?*pair_file.Pair = null,
        /// The leader's drafts give way to copies from the context; a session holds its own choice.
        copies: bool = false,
        /// A reply on two ranks failed after the ranks had been told: they may have parted, and every later
        /// request is refused.
        broken: std.atomic.Value(bool) = .init(false),
        /// Drafts a window at most; a request with `drafts` false takes none.
        depth: usize,
        /// The session's images (`servesImages`, `Served.vision` its offer); null: none is asked of it.
        eye: ?*Eye = null,
        cache: sched.Cache = .{},
    };

    /// The pool's own count of who decodes and who waits; a session decodes one.
    pub fn counts(behind: *const Behind) ?sched.Counts {
        return if (behind.hall) |hall| hall.scheduler.counts() else null;
    }

    pub fn promptCache(behind: *const Behind) ?sched.Cache.Snapshot {
        return if (behind.hall) |hall| hall.core.cache.read() else behind.cache.read();
    }

    pub fn cacheCapacity(behind: *const Behind) ?api.PromptCacheCapacity {
        // the pool's budget is what the load's fit left in its options, `serve`'s writes included
        if (behind.hall) |hall| return .{ .budget_bytes = hall.core.options.states_bytes, .minimum_state_bytes = hall.core.minimum_state_bytes };
        if (behind.session) |held| return .{ .budget_bytes = held.options.parked_bytes, .minimum_state_bytes = hall_file.minimumStateBytes(&held.state) };
        return null;
    }

    /// The session and the pool are their maker's to end.
    pub fn closed(_: *Behind) void {}

    pub fn reply(live: *Live, arena: Allocator, ask: shared.Ask, said: *request_file.Diagnostic) !shared.Ended {
        const served = live.served;
        const request = live.request;
        const behind = &served.behind;
        const prompt = ask.prompt;
        const thinking = served.thinks(prompt);
        const depth: usize = if (request.drafts) behind.depth else 0;
        // tools offered and the call free take the short first window
        const opening = session_file.openingFor(request.tools_json.len != 0, request.call != null or request.structure != null, depth);
        const starts = try served.frontend.starts(arena, prompt);
        const kept = marks_file.keptOf(starts, request.shared_prefixes);

        // a warm pass is the pool's (`pool/hall_warm.zig`); `serve` asks none of a session or two ranks
        if (ask.warm and behind.hall == null) return .{ .reason = .length };
        // `serve` offers images on one rank (`Served.vision`), so the host sends two ranks none
        if (ask.images.len > 0 and (behind.eye == null or behind.leader != null)) {
            said.set("this server accepts text parts only; image, audio and video inputs are unsupported", .{});
            return error.BadRequest;
        }
        if (behind.leader) |pair| return lead(live, arena, ask, said, pair, depth, opening, starts, kept);

        if (behind.session) |held| {
            // the session's turn, in the order the requests came; a client that left while it waited ends here
            if (!live.enters()) return .{ .reason = .cancelled };
            defer live.leaves();
            defer behind.cache.publish(hall_file.sessionCache(held));
            live.begins();
            const constraint = try live.constrained(arena, thinking, said);
            const image: ?session_file.ImagePrompt = if (ask.images.len > 0) try behind.eye.?.prompt(served.gpa, arena, prompt, ask.images, said) else null;
            const done = try held.reply(prompt, ask.settings, .{
                .max_tokens = ask.max_tokens,
                .depth = depth,
                .opening = opening,
                .constraint = constraint,
                .starts = starts,
                .kept = kept,
                .tap = .{ .context = live, .takeFn = Live.tapTake, .haltFn = tapHalt, .heldFn = Live.tapHeld, .rowFn = tapRow },
                .eos = request.eos,
                .logprobs = ask.logprobs,
                .image = image,
            }, .{ .context = live, .writeFn = hall_file.Core.unasked });
            live.closing(done.finish, done.end);
            var line: [640]u8 = undefined;
            std.log.info("{s}", .{held.replyLine(&line, prompt.len, done)});
            return .{ .reason = live.reasonOf(done.finish), .stats = .{
                .rounds = done.stats.rounds,
                .drafted = done.stats.drafted,
                .accepted = done.stats.accepted,
                .prefill_seconds = done.prompt_seconds,
            } };
        }

        const hall = behind.hall.?;
        live.begins();
        // the job borrows the constraint: it is released below, before the request frees it
        const constraint = try live.constrained(arena, thinking, said);
        // an image prompt is laid out here and its tower run on the decoding thread (`hall.Sight`)
        const picture: ?hall_file.Picture = if (ask.images.len > 0) try behind.eye.?.picture(arena, prompt, ask.images, said) else null;
        const job = try hall_front.makeJob(hall, .{ .prompt = prompt, .settings = ask.settings, .max_tokens = ask.max_tokens, .thinking = thinking, .constraint = constraint, .logprobs = ask.logprobs }, depth, opening, starts, kept, request.eos, picture);
        defer hall_front.releaseJob(hall, job);
        if (ask.warm) {
            if (!hall_warm.quiet(served.io, &hall.scheduler, live)) return .{ .reason = .cancelled };
            hall_front.warmJob(job);
            // declined beside a live stream, or ended for one: the host only frees what it sent
            const done = hall.scheduler.decode(job, live.taker(), said) catch return .{ .reason = .cancelled };
            return .{ .reason = live.reasonOf(done.finish) };
        }
        const done = try hall.scheduler.decode(job, live.taker(), said);
        if (done.tail.end_row) |*end| live.row(end);
        live.closing(done.finish, done.tail.end);
        return .{ .reason = live.reasonOf(done.finish), .stats = .{
            .rounds = done.tail.rounds,
            .drafted = done.tail.drafted,
            .accepted = done.tail.accepted,
            .prefill_seconds = done.tail.prompt_seconds,
        } };
    }

    // -- a session's tap, on the request's thread ----------------------------------------------------------------

    fn tapRow(context: *anyopaque, one: *const request_file.LogprobRow) void {
        const live: *Live = @ptrCast(@alignCast(context));
        live.row(one);
    }

    /// Asked first when the prompt has run, then after each window: the prompt is told first, then `Live.tapHalt`.
    fn tapHalt(context: *anyopaque) bool {
        const live: *Live = @ptrCast(@alignCast(context));
        live.prefilled();
        return Live.tapHalt(context);
    }

    /// Rank 0's reply. Rank 1 is already following; this tells it each window and the tokens, and which states
    /// of the sequence it keeps: the pair goes on from what it ran, as a session does.
    fn lead(live: *Live, arena: Allocator, ask: shared.Ask, said: *request_file.Diagnostic, pair: *pair_file.Pair, depth: usize, opening: usize, starts: []const usize, kept: ?usize) !shared.Ended {
        const served = live.served;
        const request = live.request;
        const behind = &served.behind;
        const prompt = ask.prompt;
        if (!live.enters()) return .{ .reason = .cancelled };
        defer live.leaves();
        defer publishPair(behind, pair);
        if (Live.refusesBroken(&behind.broken, said)) return error.Failed;
        if (prompt.len < 1 or prompt.len + ask.max_tokens > pair.capacity) {
            said.set("the prompt and the reply do not fit the context", .{});
            return error.Failed;
        }
        live.begins();
        const thinking = served.thinks(prompt);
        const constraint = try live.constrained(arena, thinking, said);
        var held: ?masks_file.Masks = null;
        if (constraint) |c| {
            const mask = try arena.alloc(u32, c.words);
            held = masks_file.Masks.init(c, mask, if (thinking) served.think.?.close else null) catch |err| return broke(behind, pair, said, err);
        }
        const turn: pair_file.Turn = .{ .starts = starts, .kept = kept, .close = if (thinking) served.think.?.close else null };
        const done = pair.reply(prompt, ask.settings, request.eos, .{ .depth = depth, .opening = opening, .copies = behind.copies }, turn, if (held) |*one| one else null, ask.max_tokens, null, .{
            .context = live,
            .take = leadTake,
            .halt = tapHalt,
            .held = Live.tapHeld,
        }) catch |err| return broke(behind, pair, said, err);
        live.closing(done.finish, done.end);
        var copied_line: [64]u8 = undefined;
        var marks: [16]usize = undefined;
        std.log.info("two ranks: prompt {d} tokens, {d} held, in {d} windows; reply {d} in {d} rounds, {d} of {d} drafts kept{s}; {d} states at {any}", .{ prompt.len, done.cached, done.windows, done.tokens, done.stats.rounds, done.stats.accepted, done.stats.drafted, done.copies.line(&copied_line), pair.book.count(), if (pair.book.count() <= marks.len) pair.book.positions(&marks) else marks[0..0] });
        return .{ .reason = live.reasonOf(done.finish), .stats = .{
            .rounds = done.stats.rounds,
            .drafted = done.stats.drafted,
            .accepted = done.stats.accepted,
        } };
    }

    fn publishPair(behind: *Behind, pair: *const pair_file.Pair) void {
        var bytes: usize = 0;
        for (pair.shelf.held.items) |ranks| for (ranks) |shot| {
            bytes += shot.bytes();
        };
        for (pair.shelf.spare.list.items) |ranks| for (ranks) |shot| {
            bytes += shot.bytes();
        };
        behind.cache.publish(.{ .states = pair.shelf.held.items.len, .bytes = bytes });
    }

    fn leadTake(context: *anyopaque, token: i32) bool {
        const live: *Live = @ptrCast(@alignCast(context));
        if (live.give(token)) return true;
        return live.cancelled();
    }

    fn broke(behind: *Behind, pair: *pair_file.Pair, said: *request_file.Diagnostic, err: anyerror) error{Failed} {
        behind.broken.store(true, .release);
        said.set("{t}", .{err});
        pair.lead.channel.fail(said.message());
        return error.Failed;
    }
};

pub const Served = shared.Served(flash_next);
const Live = Served.Live;
