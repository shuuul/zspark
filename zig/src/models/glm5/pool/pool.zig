//! A hall of GLM-5.3-Flash's replies (`hall.zig`) under the scheduler (`server/scheduler.zig`; spec 068 WS-04):
//! `serve --parallel N` above 1. The scheduler's one decoding thread admits the waiting requests into the
//! hall's slots and runs its rounds; a request's own thread takes its tokens (`Scheduler.decode`,
//! `served.zig`). Flash Next's is `qwen4`'s `hall_engine.zig`; nothing is shared (`../../AGENTS.md`).
//!
//! What the scheduler offers and this decoder never asks: a round is never short (a slot's memory is its own
//! from the load, `session.zig` refuses a count that does not fit), so no stream is taken off and run again.
//! The token that ends a reply is none of the tokens emitted: it is the reply's `Tail.end`.
//!
//! A round that fails leaves the ranks' state unknown: every live reply fails and every later one is refused,
//! as a session's.

const std = @import("std");
const Allocator = std.mem.Allocator;
const request_file = @import("../../../server/request.zig");
const scheduler_file = @import("../../../server/scheduler.zig");
const hall_file = @import("hall.zig");

pub fn PoolOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        pub const Hall = hall_file.HallOf(held);

        /// A request ready to decode: the request's thread's, read by the decoding thread while it is live.
        pub const Job = struct { request: hall_file.Request };

        /// What the pool keeps of the reply in a slot while it is live.
        const Reply = struct {
            /// The reply's end tokens, the job's.
            eos: []const u32 = &.{},
            /// When the slot was taken, and the seconds from then to its first token.
            began: i96 = 0,
            prompt_seconds: ?f64 = null,
            /// The token that ended the reply.
            end: ?i32 = null,
        };

        gpa: Allocator,
        io: std.Io,
        hall: *Hall,
        replies: []Reply,
        /// A round failed: the ranks may have parted.
        broken: std.atomic.Value(bool) = .init(false),
        scheduler: scheduler_file.Scheduler,
        cache: scheduler_file.Cache = .{},

        /// Over `hall`, whose slots are the scheduler's; the decoding thread is started. `gpa` is thread-safe.
        pub fn create(gpa: Allocator, io: std.Io, hall: *Hall) !*Self {
            const self = try gpa.create(Self);
            errdefer gpa.destroy(self);
            const replies = try gpa.alloc(Reply, hall.streams.len);
            errdefer gpa.free(replies);
            @memset(replies, .{});
            self.* = .{ .gpa = gpa, .io = io, .hall = hall, .replies = replies, .scheduler = undefined };
            self.scheduler = try .init(gpa, io, .{ .context = self, .table = &decoder }, .{ .slots = hall.streams.len });
            errdefer self.scheduler.deinit();
            try self.scheduler.start();
            return self;
        }

        /// Ends every request and the decoding thread; after it nothing of the hall is run.
        pub fn close(self: *Self) void {
            self.scheduler.close();
        }

        /// After `close`, and after every `decode` has returned.
        pub fn destroy(self: *Self) void {
            const gpa = self.gpa;
            self.scheduler.deinit();
            gpa.free(self.replies);
            gpa.destroy(self);
        }

        /// `job`'s reply to `taker`, on the caller's thread.
        pub fn decode(self: *Self, job: *Job, taker: scheduler_file.Taker, said: *request_file.Diagnostic) request_file.Error!scheduler_file.Done {
            return self.scheduler.decode(@ptrCast(job), taker, said);
        }

        // -- the decoder, on the decoding thread ------------------------------------------------------------------------

        const decoder: scheduler_file.Decoder.Table = .{ .admit = admit, .round = round, .retire = retire, .preempt = preempt };

        fn nowNs(self: *const Self) i96 {
            return std.Io.Clock.awake.now(self.io).toNanoseconds();
        }

        fn admit(context: *anyopaque, opaque_job: *scheduler_file.Job, sent: []const i32, said: *request_file.Diagnostic) scheduler_file.Admission {
            const self: *Self = @ptrCast(@alignCast(context));
            defer self.cache.publish(self.hall.cacheSnapshot());
            const job: *Job = @ptrCast(@alignCast(opaque_job));
            // no stream is taken off, so none comes back with tokens sent
            std.debug.assert(sent.len == 0);
            if (self.broken.load(.acquire)) {
                said.set("an earlier reply failed and the model's state is not known: restart the server", .{});
                return .{ .failed = error.Failed };
            }
            const taken = self.hall.admit(job.request) catch |err| {
                // a constraint that cannot start has changed nothing; anything else may have
                if (err != error.ConstraintFailed) self.broken.store(true, .release);
                said.set("the request could not be started: {t}", .{err});
                return .{ .failed = error.Failed };
            } orelse return .wait;
            self.replies[taken.slot] = .{ .eos = job.request.eos, .began = self.nowNs() };
            return .{ .admitted = .{ .slot = taken.slot, .cached_tokens = taken.cached } };
        }

        /// What a round of the hall gives a slot, passed on: the tokens but the one that ends the reply, and
        /// the reply's end with its counts.
        const Sink = struct {
            pool: *Self,
            told: *scheduler_file.Round,

            pub fn tokens(sink: *Sink, slot: usize, given: []const i32) void {
                const reply = &sink.pool.replies[slot];
                if (reply.prompt_seconds == null) reply.prompt_seconds = @as(f64, @floatFromInt(sink.pool.nowNs() - reply.began)) / std.time.ns_per_s;
                const last = given[given.len - 1];
                const ends = last >= 0 and std.mem.indexOfScalar(u32, reply.eos, @intCast(last)) != null;
                if (ends) reply.end = last;
                const own = given[0 .. given.len - @intFromBool(ends)];
                if (own.len > 0) sink.told.emit(slot, own);
            }

            pub fn ended(sink: *Sink, slot: usize, how: hall_file.Ended) void {
                const reply = sink.pool.replies[slot];
                sink.told.tail(slot, .{ .end = reply.end, .rounds = how.stats.rounds, .drafted = how.stats.drafted, .accepted = how.stats.accepted, .prompt_seconds = reply.prompt_seconds });
                sink.told.end(slot, how.finish);
            }
        };

        fn round(context: *anyopaque, told: *scheduler_file.Round) scheduler_file.Outcome {
            const self: *Self = @ptrCast(@alignCast(context));
            defer self.cache.publish(self.hall.cacheSnapshot());
            var sink: Sink = .{ .pool = self, .told = told };
            _ = self.hall.round(&sink) catch |err| {
                std.log.err("a round of {d} replies failed: {t}", .{ self.hall.live(), err });
                self.broken.store(true, .release);
                return .failed;
            };
            return .ran;
        }

        fn retire(context: *anyopaque, slot: usize, _: scheduler_file.Leave, _: usize) scheduler_file.Tail {
            const self: *Self = @ptrCast(@alignCast(context));
            defer self.cache.publish(self.hall.cacheSnapshot());
            // what the reply ran, for one its connection left before its end (a stop string, a hang-up)
            const stats = self.hall.streams[slot].stats;
            self.hall.retire(slot);
            return .{ .rounds = stats.rounds, .drafted = stats.drafted, .accepted = stats.accepted, .prompt_seconds = self.replies[slot].prompt_seconds };
        }

        fn preempt(context: *anyopaque, slot: usize) void {
            const self: *Self = @ptrCast(@alignCast(context));
            defer self.cache.publish(self.hall.cacheSnapshot());
            self.hall.retire(slot);
        }
    };
}
