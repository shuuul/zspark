//! GLM-5.3-Flash behind TensorFold's `engine_api.Engine` (spec 045 WS-04): what `server/served.zig`'s engine
//! asks of a model. The session behind it (`session.zig`) is what `generate` runs, so a reply's tokens are
//! its. With one lane a request holds `decoding` for its reply, and those that wait wait there. With `serve
//! --parallel N` above 1 (spec 068 WS-04) the lanes are a hall's slots (`pool.zig`): the request's thread
//! takes its tokens from the scheduler's decoding thread, and its constraint is compiled on it.
//!
//! A required call and a JSON document are the reply's constraint, compiled by zspark's grammar engine as
//! Flash Next's are (`server/structured.zig`) and laid over rank 0's rows (`session.ReplyOptions.constraint`).
//!
//! A prompt's marks are placed at its messages' starts, found in the ids, and `shared_prefixes` says which of
//! them a harness's prefix ends at (`decode/marks.zig`'s `keptOf`), as Flash Next's (spec 112 WS-06).
//!
//! What is GLM's own here: a reply that failed leaves the ranks' state unknown, and every later request is
//! refused (`broken`). A session's tap is given the token that ends the reply among the reply's own, and it is
//! sent as the pool's and Flash Next's are (`Live.closing`); `status` counts the lanes, not the pool, which
//! takes no stream off (`preemptions` is null). A reply that failed for its constraint alone breaks nothing.

const std = @import("std");
const units = @import("../../../cuda/units.zig");
const Allocator = std.mem.Allocator;
const request_file = @import("../../../server/request.zig");
const sched = @import("../../../server/scheduler.zig");
const shared = @import("../../../server/served.zig");
const marks_file = @import("../../../decode/marks.zig");
const pool_file = @import("../pool/pool.zig");
const session_file = @import("session.zig");

pub fn ServedOf(comptime held: usize) type {
    return shared.Served(Glm(held));
}

/// What `server/served.zig` asks of a model (`shared.Served`'s doc comment), for a session of `held` ranks.
pub fn Glm(comptime held: usize) type {
    return struct {
        const Session = session_file.SessionOf(held);
        const Live = ServedOf(held).Live;
        pub const Pool = pool_file.PoolOf(held);

        pub const Behind = struct {
            session: *Session,
            /// Several replies at once (`pool.zig`) where the session holds a hall: the caller's, started
            /// before the first request and closed by `close`.
            pool: ?*Pool = null,
            /// A reply failed: the ranks may have parted, and every later request is refused.
            broken: std.atomic.Value(bool) = .init(false),
            cache: sched.Cache = .{},
        };

        /// Replies at once: the hall's slots, or the session's one.
        pub fn lanes(pool: ?*const Pool) u32 {
            return if (pool) |one| @intCast(one.hall.streams.len) else 1;
        }

        /// `status` counts the lanes.
        pub fn counts(_: *const Behind) ?sched.Counts {
            return null;
        }

        pub fn promptCache(behind: *const Behind) ?sched.Cache.Snapshot {
            return if (behind.pool) |pool| pool.cache.read() else behind.cache.read();
        }

        /// No request is left to wait for the decoding thread.
        pub fn closed(behind: *Behind) void {
            if (behind.pool) |pool| pool.close();
        }

        /// One token of the reply to the server; a stop string ends the reply at it. The token that ends the
        /// reply is the session's last, and goes as a reply's end does (the tap's other calls are `Live`'s).
        fn tapTake(context: *anyopaque, token: i32) void {
            const live: *Live = @ptrCast(@alignCast(context));
            if (std.mem.indexOfScalar(u32, live.request.eos, @intCast(token)) != null) return live.closing(.stop, token);
            Live.tapTake(context, token);
        }

        /// The reply decoded beside the others' (`pool.zig`): the constraint is compiled here, on the
        /// request's thread, and every mask of it is the decoding thread's (`../../server/AGENTS.md`).
        fn among(live: *Live, pool: *Pool, arena: Allocator, ask: shared.Ask, said: *request_file.Diagnostic) !shared.Ended {
            const served = live.served;
            const request = live.request;
            const prompt = ask.prompt;
            live.begins();
            const thinking = served.thinks(prompt);
            // the request's, freed after this returns: `decode` returns once the stream has left its slot
            const constraint = try live.constrained(arena, thinking, said);
            var job: Pool.Job = .{ .request = .{ .prompt = prompt, .settings = ask.settings, .eos = request.eos, .max_tokens = ask.max_tokens, .drafts = request.drafts, .constraint = constraint, .gate = if (thinking) served.think.?.close else null } };
            const done = try pool.decode(&job, live.taker(), said);
            live.closing(done.finish, done.tail.end);
            const seconds = done.tail.prompt_seconds orelse 0;
            std.log.info("{d} tokens of prompt ({d} held) in {d:.3} s beside others, {d} of reply; {d} rounds, {d} of {d} drafts kept", .{ prompt.len, done.cached_tokens, seconds, live.made.load(.acquire), done.tail.rounds, done.tail.accepted, done.tail.drafted });
            return .{
                .reason = live.reasonOf(done.finish),
                // null where the reply ended before its first token, as TensorFold's hosts say it
                .stats = .{ .rounds = done.tail.rounds, .drafted = done.tail.drafted, .accepted = done.tail.accepted, .prefill_seconds = done.tail.prompt_seconds },
            };
        }

        pub fn reply(live: *Live, arena: Allocator, ask: shared.Ask, said: *request_file.Diagnostic) !shared.Ended {
            const served = live.served;
            const request = live.request;
            const behind = &served.behind;
            const prompt = ask.prompt;
            if (behind.pool) |pool| return among(live, pool, arena, ask, said);
            const out = try arena.alloc(i32, ask.max_tokens);

            // the session's turn, in the order the requests came; a client that left while it waited ends here
            if (!live.enters()) return .{ .reason = .cancelled };
            defer live.leaves();
            defer behind.cache.publish(behind.session.cacheSnapshot());
            if (Live.refusesBroken(&behind.broken, said)) return error.Broken;
            live.begins();
            const thinking = served.thinks(prompt);
            const constraint = try live.constrained(arena, thinking, said);
            const starts = try served.frontend.starts(arena, prompt);
            const done = behind.session.reply(prompt, ask.settings, .{
                .starts = starts,
                .kept = marks_file.keptOf(starts, request.shared_prefixes),
                .eos = request.eos,
                .drafts = request.drafts,
                .tap = .{ .context = live, .take = tapTake, .halt = Live.tapHalt, .held = Live.tapHeld },
                .constraint = constraint,
                .gate = if (thinking) served.think.?.close else null,
            }, out) catch |err| {
                // a constraint that failed has moved no rank; anything else may have (`pool.zig` says the same)
                if (err != error.ConstraintFailed) behind.broken.store(true, .release);
                return err;
            };
            const seconds = units.seconds(done.first_ns);
            std.log.info("{d} tokens of prompt ({d} held) in {d:.3} s, {d} of reply at {d:.1} a second; {d} rounds, {d} of {d} drafts kept", .{
                prompt.len,                                                                                                              done.cached,       seconds,             done.made,
                if (done.made > 1) @as(f64, @floatFromInt(done.made - 1)) * 1e9 / @as(f64, @floatFromInt(@max(done.rest_ns, 1))) else 0, done.stats.rounds, done.stats.accepted, done.stats.drafted,
            });
            return .{
                .reason = live.reasonOf(done.finish),
                .stats = .{ .rounds = done.stats.rounds, .drafted = done.stats.drafted, .accepted = done.stats.accepted, .prefill_seconds = seconds },
            };
        }
    };
}
