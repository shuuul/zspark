//! `served.zig`'s `Live`, split from it as whole declarations (spec 171): one request's own thread, from
//! `Served.submit`'s spawn to its `finished` event. It reads the request (`reply`), waits for the session's
//! turn (`enters`, `leaves`), lends the model the request's constraint (`constrained`, under its loop guard
//! or thinking budget), and sends the reply's events (`give`, `closing`, `row`, `prefilled`, `taker`): what
//! a model's `reply` is handed, session and pool alike. What it shares of the engine — the requests' table,
//! the queue bound, the session's turn and the rates — stays `Served`'s, reached under its mutex.
//!
//! Like `served.zig` this file imports no model and not `cuda/cuda.zig`.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const request_file = @import("request.zig");
const sched = @import("scheduler.zig");
const served_file = @import("served.zig");
const grammar = @import("../text/grammar.zig");
const guard_file = @import("../decode/guard.zig");
const sampling = @import("../decode/sampling.zig");

/// `Served(M).Live`, one request's thread; `Served` is the engine's own type, `M` the model's namespace.
pub fn Live(comptime Served: type, comptime M: type) type {
    return struct {
        const Self = @This();
        const Ended = served_file.Ended;

        served: *Served,
        id: api.Id,
        /// The server's, valid until `finished` is sent.
        request: *const api.Request,
        sink: api.Sink,
        cancel: std.atomic.Value(bool) = .init(false),
        /// Under `Served.mutex`: in `Served.open`; the one behind it in the line for the session's turn.
        counted: bool = true,
        after: ?*Self = null,
        /// The scheduler its reply waits at, once `Scheduler.decode` has said so (`taker`'s `waits`).
        waits: std.atomic.Value(?*sched.Scheduler) = .init(null),
        /// The request's own thread's from here on. A stop string ended the reply.
        stopped: bool = false,
        /// The prompt's tokens that were not run, and whether the server has been told.
        cached: usize = 0,
        told: bool = false,
        /// The reply's tokens given, for `status`; and when the request's prompt could start.
        made: std.atomic.Value(u32) = .init(0),
        began_ns: i96 = 0,
        submitted_ns: i96 = 0,
        admitted_ns: i96 = 0,
        /// Published by session admission or the scheduler; a warm pass has no indicator.
        filling: std.atomic.Value(bool) = .init(false),
        /// The reply so far, kept only for a request with stop strings (`Request.stop` reads all of it).
        emitted: std.ArrayList(u32) = .empty,
        /// The row of the token `give` or `closing` sends next, for a request that asks for log probabilities.
        pending: ?request_file.LogprobRow = null,
        /// What `constrained` compiled or took, freed by `run` after the model's reply; and the arena a
        /// constraint taken from `Served.ready` came with.
        constraint: ?grammar.Constraint = null,
        lent: ?std.heap.ArenaAllocator = null,
        /// The request's loop guard or thinking budget, around `constraint`: the request's arena's.
        guard: ?*guard_file.Guard = null,

        /// The request's constraint, or null: the one `Served.refusal` compiled when the host asked it,
        /// else compiled here (`Served.constraintFor`), and around it the request's loop guard or thinking
        /// budget where it asks for one (`guarded`). Lent to the model until its `reply` returns, by
        /// when no slot and no job may still read it. Asked once a request.
        pub fn constrained(self: *Self, arena: Allocator, thinking: bool, said: *request_file.Diagnostic) request_file.Error!?grammar.Constraint {
            std.debug.assert(self.constraint == null);
            if (Served.Asked.of(self.request, thinking)) |asked| if (self.served.taken(asked)) |ready| {
                self.constraint = ready.made;
                self.lent = ready.arena;
                return self.guarded(arena, said);
            };
            self.constraint = try self.served.constraintFor(arena, self.request, thinking, said);
            return self.guarded(arena, said);
        }

        /// `constraint` as the model takes it: under the request's `loop_guard` or `think_budget` a
        /// `Guard` around it, in `arena`, else itself. The host sets the two only on a reply its template
        /// leaves thinking, with the tokens that close the block; they are followed as given, as
        /// TensorFold's engine follows them.
        fn guarded(self: *Self, arena: Allocator, said: *request_file.Diagnostic) request_file.Error!?grammar.Constraint {
            const request = self.request;
            const ask = guard_file.Ask.of(request.think_budget, request.loop_guard, request.think_close, request.think_end) orelse return self.constraint;
            const words = if (self.constraint) |held| held.words else if (self.served.compiler) |compiler| compiler.engine.words else {
                // a mask's length is the grammar engine's to say; both `serve` commands make one
                said.set("a thinking budget and the loop guard are not available: this server was started without a grammar engine", .{});
                return error.Failed;
            };
            self.guard = try guard_file.Guard.init(arena, ask, self.constraint, words, @min(@as(usize, request.max_tokens), self.served.capacity));
            return self.guard.?.constraint();
        }

        pub fn run(self: *Self) void {
            const served = self.served;
            var arena_state: std.heap.ArenaAllocator = .init(served.gpa);
            defer arena_state.deinit();
            var said: request_file.Diagnostic = .{};
            const ended: Ended = self.reply(arena_state.allocator(), &said) catch |err| failed: {
                if (said.len == 0) said.set("decoding failed: {t}", .{err});
                std.log.err("request {d}: {s}", .{ self.id, said.message() });
                break :failed .{ .reason = .failed, .message = said.message() };
            };
            // the reply is over, whatever ended it: before `forget`, which frees this request
            if (self.constraint) |made| made.deinit();
            if (self.lent) |held_arena| held_arena.deinit();
            var stats = ended.stats;
            if (self.admitted_ns > self.submitted_ns) stats.queued_seconds = @as(f64, @floatFromInt(self.admitted_ns - self.submitted_ns)) / std.time.ns_per_s;
            // the cycle the guard closed the think block on, read once nothing decodes this reply
            if (self.guard) |guard| if (guard.period()) |period| {
                stats.loop_period = period;
            };
            // before the end is sent: a client that has its reply's end finds its place free
            served.uncount(self);
            self.send(.{ .finished = .{ .reason = ended.reason, .stats = stats, .message = ended.message } });
            _ = served.peak.fetchMax(served.deviceBytes(), .acq_rel);
            served.forget(self);
        }

        /// The request read, then the model's reply.
        fn reply(self: *Self, arena: Allocator, said: *request_file.Diagnostic) !Ended {
            const served = self.served;
            const request = self.request;
            if (request.prompt.len == 0 or request.prompt.len >= served.capacity) {
                said.set("This server's maximum context length is {d} tokens; the prompt is {d}", .{ served.capacity, request.prompt.len });
                return error.BadRequest;
            }
            const prompt = try arena.alloc(i32, request.prompt.len);
            for (prompt, request.prompt) |*token, id| token.* = @intCast(id);
            // TensorFold's sampling has zspark's fields and its seed's rule (`sampling.seedFor`); none is greedy.
            const settings: sampling.Sampling = if (request.sampling) |s| .{ .seed = s.seed, .temperature = s.temperature, .top_k = s.top_k, .top_p = s.top_p, .min_p = s.min_p } else .{ .seed = 0, .temperature = 0.0 };
            // the host's pass for the next turn: one token, which `give` sends nobody
            const warm = served.warm_turns and request.background and request.max_tokens == 0;
            const max_tokens = if (warm) 1 else @min(@as(usize, request.max_tokens), served.capacity - prompt.len);
            if (max_tokens == 0 or prompt.len + max_tokens > served.capacity) return .{ .reason = .length };
            // hung up before anything was asked of the model: it neither queues nor waits for the session
            if (self.cancelled()) return .{ .reason = .cancelled };
            return M.reply(self, arena, .{ .prompt = prompt, .settings = settings, .max_tokens = max_tokens, .warm = warm, .logprobs = if (served.logprobs and !warm) request.logprobs else null, .images = if (served.vision != null) request.images else &.{} }, said);
        }

        fn send(self: *Self, event: api.Event) void {
            self.sink.event(self.sink.ctx, self.id, &event);
        }

        pub fn cancelled(self: *const Self) bool {
            return self.cancel.load(.acquire);
        }

        // -- a session's tap, on the request's thread: the calls both models hand `session.reply` ----------------

        /// The tap's `held`: the prompt's tokens the cache already held.
        pub fn tapHeld(context: *anyopaque, cached: usize) void {
            const live: *Self = @ptrCast(@alignCast(context));
            live.cached = cached;
        }

        /// The tap's `take`: one token of the reply, as `give` takes it.
        pub fn tapTake(context: *anyopaque, token: i32) void {
            const live: *Self = @ptrCast(@alignCast(context));
            _ = live.give(token);
        }

        /// The tap's `halt`: asked after each window; true once the client has left or a stop string ended the reply.
        pub fn tapHalt(context: *anyopaque) bool {
            const live: *Self = @ptrCast(@alignCast(context));
            return live.cancelled();
        }

        /// True, with the refusal said, where an earlier reply failed (`broken`) and the ranks' state is not known.
        pub fn refusesBroken(broken: *const std.atomic.Value(bool), said: *request_file.Diagnostic) bool {
            if (!broken.load(.acquire)) return false;
            said.set("an earlier reply failed and the model's state is not known: restart the server", .{});
            return true;
        }

        /// The reply is no longer wanted, and a request that waits hears it now: the line for the session's
        /// turn is woken, and the scheduler it is queued at. Under `Served.mutex`, which is taken before
        /// the scheduler's and never under it.
        pub fn cancelNow(self: *Self) void {
            self.cancel.store(true, .release);
            self.served.turned.broadcast(self.served.io);
            if (self.waits.load(.acquire)) |at| at.wake();
        }

        /// Waits for the session's turn, in the order the requests came; false when the reply stopped
        /// being wanted first, and then nothing was taken: the request ends as cancelled, at once, and the
        /// replies before it go on. True is followed by `leaves`.
        pub fn enters(self: *Self) bool {
            const served = self.served;
            const io = served.io;
            served.mutex.lockUncancelable(io);
            defer served.mutex.unlock(io);
            if (served.last) |last| last.after = self else served.first = self;
            served.last = self;
            while (!self.cancelled() and (served.busy or served.first != self)) served.turned.waitUncancelable(io, &served.mutex);
            // out of the line, wherever it stood
            var before: ?*Self = null;
            var at = served.first;
            while (at != self) : (at = at.?.after) before = at;
            if (before) |b| b.after = self.after else served.first = self.after;
            if (self.after == null) served.last = before;
            self.after = null;
            if (self.cancelled()) {
                // the one behind may be first now
                served.turned.broadcast(io);
                return false;
            }
            served.busy = true;
            self.admitted_ns = served.nowNs();
            if (!self.request.background or self.request.max_tokens != 0) self.filling.store(true, .release);
            return true;
        }

        /// The session's turn is over.
        pub fn leaves(self: *Self) void {
            const served = self.served;
            served.mutex.lockUncancelable(served.io);
            defer served.mutex.unlock(served.io);
            served.busy = false;
            served.turned.broadcast(served.io);
        }

        /// The request's prompt may start now: its rate for `status` is counted from here. A pool's request
        /// calls it before it queues, and `began` moves it to where the wait ended.
        pub fn begins(self: *Self) void {
            self.began_ns = self.served.nowNs();
        }

        /// The prompt is in the cache: said once, before the reply's first token.
        pub fn prefilled(self: *Self) void {
            if (self.told) return;
            self.told = true;
            self.filling.store(false, .release);
            self.send(.{ .prefilled = @intCast(self.cached) });
            // The prompt's rate for `status`: its tokens that ran over the time since the request held a session
            // or a pool's slot, no wait in it (TensorFold, vLLM and SGLang time the prefill alone).
            const served = self.served;
            const now = served.nowNs();
            const ran = self.request.prompt.len -| self.cached;
            if (now <= self.began_ns or ran == 0) return;
            served.mutex.lockUncancelable(served.io);
            defer served.mutex.unlock(served.io);
            served.prefill_rate = @as(f64, @floatFromInt(ran)) / (@as(f64, @floatFromInt(now - self.began_ns)) / std.time.ns_per_s);
            served.prefill_at = now;
        }

        /// One token of the reply to the server; true when a stop string ends the reply at it.
        pub fn give(self: *Self, token: i32) bool {
            // a window's tokens after the one a stop string ended at are not the reply's
            if (self.stopped) return true;
            // a warm pass's token is no one's, and its prompt is no fill (TensorFold's `lane_host_status`)
            if (self.request.background and self.request.max_tokens == 0) return false;
            self.prefilled();
            self.sendRow();
            const id: u32 = @intCast(token);
            self.send(.{ .tokens = &.{id} });
            _ = self.made.fetchAdd(1, .acq_rel);
            self.served.tallied();
            const stop = self.request.stop orelse return false;
            self.emitted.append(self.served.gpa, id) catch return false;
            if (!stop.check(stop.ctx, self.emitted.items)) return false;
            self.stopped = true;
            self.cancel.store(true, .release);
            return true;
        }

        /// The token that ended the reply, after the reply's own: the server reads the reply's end from it.
        /// It is in `status`'s rate, and no stop string reads it.
        pub fn closing(self: *Self, finish: request_file.Finish, end: ?i32) void {
            if (finish != .stop or self.stopped or self.request.background) return;
            const token = end orelse return;
            self.prefilled();
            self.sendRow();
            self.send(.{ .tokens = &.{@intCast(token)} });
            _ = self.made.fetchAdd(1, .acq_rel);
            self.served.tallied();
        }

        /// The row of the token `give` or `closing` is given next (`Ask.logprobs`): sent as its `logprobs`
        /// event, before the token's `tokens` event, as the host reads them (one row a token).
        pub fn row(self: *Self, one: *const request_file.LogprobRow) void {
            self.pending = one.*;
        }

        fn sendRow(self: *Self) void {
            const one = self.pending orelse return;
            self.pending = null;
            // TensorFold's row: zspark gives no rank (vLLM's, asked of `prompt_logprobs` alone)
            const rows = [1]api.LogprobRow{.{ .token = one.token, .logprob = one.logprob, .count = one.count, .ids = one.ids, .logprobs = one.logprobs }};
            self.send(.{ .logprobs = &rows });
        }

        pub fn reasonOf(self: *const Self, finish: request_file.Finish) api.Reason {
            return switch (finish) {
                .stop => .stop,
                .length => .length,
                .cut => if (self.stopped) .stop else .cancelled,
            };
        }

        /// This request as a pool's taker (`Scheduler.decode`), on this request's thread.
        pub fn taker(self: *Self) sched.Taker {
            return .{ .context = self, .take = take, .probe = probe, .began = began, .waits = waitsAt, .row = takerRow, .filling = if (self.request.background and self.request.max_tokens == 0) null else &self.filling };
        }

        fn waitsAt(context: *anyopaque, at: *sched.Scheduler) void {
            const self: *Self = @ptrCast(@alignCast(context));
            self.waits.store(at, .release);
        }

        fn began(context: *anyopaque, cached_tokens: usize, admitted_ns: i96) void {
            const self: *Self = @ptrCast(@alignCast(context));
            self.cached = cached_tokens;
            self.admitted_ns = admitted_ns;
            self.began_ns = @max(self.began_ns, admitted_ns);
        }

        fn take(context: *anyopaque, token: i32) request_file.Sink.Signal {
            const self: *Self = @ptrCast(@alignCast(context));
            if (self.give(token)) return .stop;
            return if (self.cancelled()) .stop else .more;
        }

        fn takerRow(context: *anyopaque, one: *const request_file.LogprobRow) void {
            const self: *Self = @ptrCast(@alignCast(context));
            self.row(one);
        }

        fn probe(context: *anyopaque) request_file.Sink.Signal {
            const self: *Self = @ptrCast(@alignCast(context));
            return if (self.cancelled()) .stop else .more;
        }
    };
}
