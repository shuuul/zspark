//! Several replies decoded at once (spec 018, step B6): a queue of requests, one decoding thread that admits
//! them into a decoder's slots and runs its rounds, and a box a request through which its own thread takes the
//! reply's tokens. Host only: it knows `request.zig` and no model. TensorFold's `cuda/scheduler.py` is the
//! shape (`submit` blocks its caller on a queue of token batches; `_loop` admits, then runs a round); where it
//! departs is said below.
//!
//! Two threads meet here:
//!
//! - A request's own (`served.zig`'s `Live`, spec 059; a connection, below): it makes the job (the
//!   constraint compiled), calls `Scheduler.decode` with it, which queues it and blocks on its stream's box,
//!   and is given each of the reply's tokens through its `Taker`; it releases the job after. Many at once.
//! - The decoding thread, and on it alone the model's `Decoder`: `admit` a job into a slot, `round` over every
//!   slot, `retire` a slot whose reply ended or was cancelled, `preempt` one whose request is taken off to
//!   run again.
//!
//! So a slow client fills its own box and holds nobody, and nothing made of the tokens is on the decoding
//! thread.
//!
//! The rules, each with a test below:
//!
//! - First come, first served, and no bound on the queue (vLLM's and SGLang's defaults). Before a round, while
//!   a slot is free, the oldest waiting request is offered to the decoder.
//! - A request the decoder cannot hold yet (`Admission.wait`: the pool is short beside the live streams)
//!   stays first in line and nothing passes it, as TensorFold's `held`. With no stream live nothing would
//!   ever make room, so it is refused as one that never fits.
//! - A request the decoder can never hold (`Admission.never`) is `error.BadRequest` with the decoder's words:
//!   a plain 400, as a prompt past the context.
//! - Cancellation is a flag. The connection sets it when its taker says stop (a stop string, a client that
//!   hung up, the server stopping) or its task is canceled; the decoding thread reads it before a round, takes
//!   a waiting request out of the queue and retires a live one. The connection takes no token after the one
//!   its taker said stop at, so what a client receives, and the usage it is told, are a lone stream's, however
//!   many rounds ran on before the flag was read (spec 018's step 0 has vLLM's and SGLang's ways).
//! - A round that the pool cannot serve (`Outcome.short`) has changed nothing; the newest admitted stream is
//!   taken off (`preempt`) and put back at the head of the queue, and the round is tried again. When it is
//!   admitted again the decoder is given the tokens it was already sent and runs them as a prompt: the client
//!   sees a pause and nothing twice, since the text never left the connection. A lone stream that is short
//!   fails.
//! - A request still waiting or being filled is given nothing, so its connection asks its taker
//!   (`Taker.probe`) every `Options.probe`, when its task is canceled, and at once when anyone calls `wake`
//!   (`served.zig`'s `cancel`, told where by `Taker.waits`): a client that hung up in the queue leaves it, and
//!   so does every request when the server stops.
//! - A request that is no longer wanted while it waits leaves the queue on its own thread, there and then
//!   (`abandon`): it is offered to no decoder, and its `decode` returns although a round is being run (spec
//!   074; TensorFold's lane host, vLLM and SGLang each take a cancelled request out of what waits). One the
//!   decoder is being offered at that moment (`.offered`) is the decoding thread's, and leaves as a live one.
//!
//! Where this departs from TensorFold's scheduler: no background streams and no yielding of a lane; a request
//! the pool cannot hold with nothing live is refused with a 400 where TensorFold answers its `NoRoom` error;
//! preemption replaces its "the newest ends with an error" (the Decisions of spec 018).
//!
//! The lock. `mutex` guards the queue, every stream's box and state, and `closing`; both conditions are the
//! scheduler's, so a stream holds no lock or condition of its own and its connection may free it the moment
//! it reads `ended`. `live` and `endings` are the decoding thread's alone. The decoder and the taker are
//! never called under the lock.
//!
//! The grammar engine needs no lock and no instances a request, by which thread does what: a constraint is
//! compiled where the job is made, on the request's thread, and compiling reads the engine's vocabulary and
//! writes nothing shared (`json_schema.Engine.compile`, `call_gate.compile`: a schema parsed, a matcher's own
//! frames); what constraints share and write (`json_schema.Vocabulary.common`, `call_gate.Strings.masks`) is
//! written by `fill` alone, and `start`, `fill` and `accept` are the decoder's, on the one decoding thread. A
//! slow compile therefore stalls no round. A constraint is freed with its job, after its stream has left its
//! slot.
//!
//! The tests are `zig/tests/server/scheduler_test.zig`, their model `scheduler_fake.zig` beside them.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const request_file = @import("request.zig");

/// A request ready to decode, as its model holds it: the request's thread makes it, the decoder reads it, and
/// the scheduler only carries it.
pub const Job = opaque {};

/// How a reply ended, for `decode`'s caller.
pub const Done = struct {
    /// `.cut` when the connection's taker asked for the end.
    finish: request_file.Finish,
    /// The reply's tokens the connection took.
    tokens: usize,
    /// The prompt's tokens the decoder did not run.
    cached_tokens: usize,
    /// What the decoder said of the reply as it ended (`Round.tail`); a reply that was cut has none.
    tail: Tail = .{},
};

/// What a decoder says of a reply that ends as `.stop` or `.length`, for a taker that passes it on (TensorFold's
/// `engine_api.Stats`, and the end token its server expects after the reply's own).
pub const Tail = struct {
    /// The token that ended the reply, where one did: it is none of the reply's tokens and is not emitted.
    end: ?i32 = null,
    /// The reply's windows, and the drafts they carried and kept.
    rounds: u64 = 0,
    drafted: u64 = 0,
    accepted: u64 = 0,
    /// Seconds from the request's admission to its first token.
    prompt_seconds: ?f64 = null,
    /// The row of `end`, for a reply that asks for log probabilities (`Round.rows`).
    end_row: ?request_file.LogprobRow = null,
};

/// Who takes a reply's tokens, on the thread that called `Scheduler.decode`.
pub const Taker = struct {
    context: *anyopaque,
    /// The reply's next token, in order, each once; `.stop` ends the reply there, and no token follows.
    take: *const fn (context: *anyopaque, token: i32) request_file.Sink.Signal,
    /// Asked every `Options.probe` while nothing comes (the queue, a prompt being filled): `.stop` when the
    /// reply is no longer wanted.
    probe: *const fn (context: *anyopaque) request_file.Sink.Signal,
    /// Told once, before the reply's first token: the prompt's tokens the decoder did not run, and when the
    /// request was first offered a slot it took (the awake clock's nanoseconds): its wait ended there.
    began: ?*const fn (context: *anyopaque, cached_tokens: usize, admitted_ns: i96) void = null,
    /// Told once, as `decode` queues the request: whose `wake` has `probe` asked now. The scheduler outlives
    /// every `decode`, so the pointer may be kept as long as the taker's owner lives.
    waits: ?*const fn (context: *anyopaque, at: *Scheduler) void = null,
    /// For a reply that asks for log probabilities: the row of the token `take` is given next, before it.
    row: ?*const fn (context: *anyopaque, row: *const request_file.LogprobRow) void = null,
    /// A served request's prefill indicator; null for background work. Written on admission and retirement.
    filling: ?*std.atomic.Value(bool) = null,
};

/// What the decoder says of a job offered to it.
pub const Admission = union(enum) {
    admitted: Admitted,
    /// It fits, but not beside the streams that are live: it stays first in line.
    wait,
    /// It can never be held (`extents.Pool.admission`'s `never`): a 400 with `said`.
    never,
    /// It cannot be started: `said` has why.
    failed: request_file.Error,

    pub const Admitted = struct {
        /// The slot it took, below `Options.slots` and free.
        slot: usize,
        /// The prompt's tokens found in a held prefix.
        cached_tokens: usize = 0,
    };
};

pub const Outcome = enum {
    /// The round ran: tokens and endings were given to the `Round`.
    ran,
    /// The pool cannot hold this round's rows, idle chains evicted: nothing was run or changed.
    short,
    /// The device failed: every live stream fails.
    failed,
};

/// Why a slot is left.
pub const Leave = enum { ended, cancelled, failed };

/// A model's decoder over slots. Every call is on the decoding thread.
pub const Decoder = struct {
    context: *anyopaque,
    table: *const Table,

    pub const Table = struct {
        /// Offers the oldest waiting job. `sent` is empty, or the tokens of the reply its client already
        /// has, for a request that was taken off (`preempt`): they are run after the prompt as a prompt, the
        /// sampler and the constraint advanced over them, and not given to the round again.
        admit: *const fn (context: *anyopaque, job: *Job, sent: []const i32, said: *request_file.Diagnostic) Admission,
        /// One round over the live slots, prompts being filled among them.
        round: *const fn (context: *anyopaque, round: *Round) Outcome,
        /// The slot's stream leaves; the slot is free when this returns. `kept` is how many of the reply's
        /// tokens its connection took before it asked for the end, when `why` is `.cancelled`: the tokens past
        /// them were decoded and never sent. Otherwise it is all that were given; a reply that ended before
        /// its connection's stop was heard is `.ended` with them all. Returns the reply's counts as far as it
        /// ran, read where `why` is `.cancelled`: no `Round.tail` said them, and a reply a stop string ended is
        /// a reply (TensorFold's lane host, vLLM and SGLang give its counts as any other's).
        retire: *const fn (context: *anyopaque, slot: usize, why: Leave, kept: usize) Tail,
        /// The slot's stream is taken off to be admitted again later: its memory is released, the slot free.
        preempt: *const fn (context: *anyopaque, slot: usize) void,
    };
};

/// What a round tells the scheduler, slot by slot.
pub const Round = struct {
    scheduler: *Scheduler,

    /// Tokens of the slot's reply, in order; they reach its connection at once.
    pub fn emit(round: *Round, slot: usize, tokens: []const i32) void {
        const self = round.scheduler;
        const stream = self.live[slot].?;
        std.debug.assert(self.endings[slot] == null);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        stream.tokens.appendSlice(self.gpa, tokens) catch {
            self.endings[slot] = .{ .finish = .cut, .fault = error.OutOfMemory };
            return;
        };
        self.changed.broadcast(self.io);
    }

    /// The rows of the slot's next tokens, before `emit` gives them, for a reply that asks for log
    /// probabilities: one a token, and after the reply's own the row of the token that ended it (`Tail.end_row`).
    pub fn rows(round: *Round, slot: usize, given: []const request_file.LogprobRow) void {
        const self = round.scheduler;
        const stream = self.live[slot].?;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        stream.rows.appendSlice(self.gpa, given) catch {
            self.endings[slot] = .{ .finish = .cut, .fault = error.OutOfMemory };
        };
    }

    /// The slot's reply ended after the tokens given: `.stop` or `.length`.
    pub fn end(round: *Round, slot: usize, finish: request_file.Finish) void {
        const self = round.scheduler;
        std.debug.assert(self.live[slot] != null);
        if (self.endings[slot] == null) self.endings[slot] = .{ .finish = finish };
    }

    /// The slot's request failed; the others go on.
    pub fn fail(round: *Round, slot: usize, fault: request_file.Error, message: []const u8) void {
        const self = round.scheduler;
        std.debug.assert(self.live[slot] != null);
        var ending: Ending = .{ .finish = .cut, .fault = fault };
        ending.said.set("{s}", .{message});
        self.endings[slot] = ending;
    }

    /// The prompt's tokens the decoder did not run, where it learns them after `admit` (a decoder whose state is
    /// looked up when the prompt starts: `hall.zig`'s lone stream). The first admission's count is the one
    /// the request is told, as `Admitted.cached_tokens`: a replay after `preempt` does not change it.
    pub fn held(round: *Round, slot: usize, cached_tokens: usize) void {
        const self = round.scheduler;
        const stream = self.live[slot].?;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (stream.admissions == 1) stream.cached_tokens = cached_tokens;
    }

    /// What the decoder says of the slot's reply, before `end`.
    pub fn tail(round: *Round, slot: usize, said: Tail) void {
        const self = round.scheduler;
        const stream = self.live[slot].?;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        stream.tail = said;
    }

    /// Whether the slot's connection has asked for the end. The scheduler retires such a slot before the next
    /// round; a decoder whose round is long (a lone stream's whole reply) asks here.
    pub fn cancelled(round: *const Round, slot: usize) bool {
        return round.scheduler.live[slot].?.cancel.load(.acquire);
    }

    /// A request is queued. A lone stream's draft loop returns here so the scheduler can admit it; the loop
    /// does not return between windows while this is false.
    pub fn waiting(round: *Round) bool {
        const self = round.scheduler;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return self.head != null;
    }
};

pub const Options = struct {
    /// The decoder's slots: `--parallel`.
    slots: usize = 1,
    /// How often a connection whose reply sends nothing asks its taker whether the reply is still wanted.
    probe: Io.Duration = .fromMilliseconds(100),
};

pub const Counts = struct {
    queued: usize,
    live: usize,
    closing: bool,
    /// Since the start.
    admitted: u64,
    rounds: u64,
    preempted: u64,
};

/// Cache accounting published by its decoding thread; status readers never enter the model's store.
pub const Cache = struct {
    states: std.atomic.Value(u64) = .init(0),
    bytes: std.atomic.Value(u64) = .init(0),

    pub const Snapshot = struct { states: u64 = 0, bytes: u64 = 0 };

    pub fn publish(self: *Cache, snapshot: Snapshot) void {
        self.bytes.store(snapshot.bytes, .release);
        self.states.store(snapshot.states, .release);
    }

    pub fn read(self: *const Cache) Snapshot {
        return .{ .states = self.states.load(.acquire), .bytes = self.bytes.load(.acquire) };
    }
};

const Ending = struct {
    finish: request_file.Finish,
    fault: ?request_file.Error = null,
    said: request_file.Diagnostic = .{},
};

/// One request from `decode`'s entry to its return, in that call's frame. Everything but `cancel` is under
/// the scheduler's mutex, or the decoding thread's alone where said.
const Stream = struct {
    job: *Job,
    /// The reply's tokens as the rounds gave them: the box. The decoding thread appends, and reads them back
    /// for a replay; the connection reads from `taken`.
    tokens: std.ArrayList(i32) = .empty,
    /// Their rows, where the reply asks for log probabilities (`Round.rows`): `rows.items[i]` is `tokens.items[i]`'s.
    rows: std.ArrayList(request_file.LogprobRow) = .empty,
    /// How many the connection has copied out.
    taken: usize = 0,
    /// How many it gave its taker before it asked for the end.
    kept: usize = 0,
    /// `.offered`: out of the queue and in no slot yet, the decoder being asked (`admit`).
    state: enum { queued, offered, live, ended } = .queued,
    cancel: std.atomic.Value(bool) = .init(false),
    /// The decoding thread's: where it is live, and when it was admitted.
    slot: usize = 0,
    sequence: u64 = 0,
    admissions: usize = 0,
    cached_tokens: usize = 0,
    admitted_ns: i96 = 0,
    filling: ?*std.atomic.Value(bool) = null,
    tail: Tail = .{},
    ending: Ending = .{ .finish = .cut },
    /// The queue's link.
    next: ?*Stream = null,
};

pub const Scheduler = struct {
    gpa: Allocator,
    io: Io,
    decoder: Decoder,
    options: Options,

    mutex: Io.Mutex = .init,
    /// For the decoding thread: a request arrived, a cancel was asked, or `close`.
    work: Io.Condition = .init,
    /// For the connections: tokens were given, or a stream ended.
    changed: Io.Condition = .init,
    head: ?*Stream = null,
    tail: ?*Stream = null,
    queued: usize = 0,
    closing: bool = false,
    live_count: usize = 0,
    /// How often `wake` was called: a `decode` that sees it move asks its taker.
    wakes: u64 = 0,

    /// `[slots]`, the decoding thread's: who is in each slot, and how a round said it ended.
    live: []?*Stream,
    endings: []?Ending,
    sequence: u64 = 0,
    admitted: std.atomic.Value(u64) = .init(0),
    rounds: std.atomic.Value(u64) = .init(0),
    preempted: std.atomic.Value(u64) = .init(0),

    thread: ?Io.Future(void) = null,

    /// `gpa` must be thread-safe. The scheduler must not move once `start` is called.
    pub fn init(gpa: Allocator, io: Io, decoder: Decoder, options: Options) Allocator.Error!Scheduler {
        std.debug.assert(options.slots >= 1);
        const live = try gpa.alloc(?*Stream, options.slots);
        errdefer gpa.free(live);
        const endings = try gpa.alloc(?Ending, options.slots);
        @memset(live, null);
        @memset(endings, null);
        return .{ .gpa = gpa, .io = io, .decoder = decoder, .options = options, .live = live, .endings = endings };
    }

    /// Starts the decoding thread.
    pub fn start(self: *Scheduler) Io.ConcurrentError!void {
        self.thread = try self.io.concurrent(run, .{self});
    }

    /// Ends every request (a waiting one fails, a live one is cut) and the decoding thread; `decode` then
    /// refuses. The server is stopped first, so that its connections are leaving.
    pub fn close(self: *Scheduler) void {
        const io = self.io;
        self.mutex.lockUncancelable(io);
        self.closing = true;
        self.work.signal(io);
        self.mutex.unlock(io);
        if (self.thread) |*thread| thread.await(io);
        self.thread = null;
    }

    /// After `close`, and after every `decode` has returned.
    pub fn deinit(self: *Scheduler) void {
        std.debug.assert(self.thread == null and self.head == null and self.live_count == 0);
        self.gpa.free(self.live);
        self.gpa.free(self.endings);
        self.* = undefined;
    }

    pub fn counts(self: *Scheduler) Counts {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return .{
            .queued = self.queued,
            .live = self.live_count,
            .closing = self.closing,
            .admitted = self.admitted.load(.acquire),
            .rounds = self.rounds.load(.acquire),
            .preempted = self.preempted.load(.acquire),
        };
    }

    /// Every request that is given nothing now (the queue, a prompt being filled) asks its taker at once
    /// whether its reply is still wanted, without waiting for `Options.probe`. From any thread.
    pub fn wake(self: *Scheduler) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.wakes +%= 1;
        self.changed.broadcast(self.io);
    }

    // -- the connection's side -----------------------------------------------------------------------------------

    /// Queues `job` and gives `taker` its reply's tokens until the reply ends, on the caller's thread
    /// (`served.zig`, spec 059). The job stays the caller's, to release after.
    pub fn decode(self: *Scheduler, job: *Job, taker: Taker, said: *request_file.Diagnostic) request_file.Error!Done {
        const io = self.io;
        var stream: Stream = .{ .job = job, .filling = taker.filling };
        defer stream.tokens.deinit(self.gpa);
        defer stream.rows.deinit(self.gpa);

        if (taker.waits) |waits| waits(taker.context, self);
        self.mutex.lockUncancelable(io);
        if (self.closing) {
            self.mutex.unlock(io);
            said.set("the server is stopping", .{});
            return error.Failed;
        }
        self.pushBack(&stream);
        self.work.signal(io);

        // The tokens given to the taker, and whether it still wants the reply.
        var given: usize = 0;
        var wanted = true;
        var batch: [64]i32 = undefined;
        var batch_rows: [64]request_file.LogprobRow = undefined;
        var probe_at = self.probeAfterNow();
        // one behind: the taker is asked once as its request queues, so a `wake` that came between `waits`
        // and here is not lost
        var woken = self.wakes -% 1;
        // The mutex is held at the top of each turn.
        while (true) {
            const have = stream.tokens.items.len - stream.taken;
            if (wanted and have != 0) {
                const n = @min(have, batch.len);
                @memcpy(batch[0..n], stream.tokens.items[stream.taken..][0..n]);
                // a round gives a token's row before the token
                const rowed = if (taker.row != null and stream.rows.items.len > stream.taken) @min(n, stream.rows.items.len - stream.taken) else 0;
                if (rowed > 0) @memcpy(batch_rows[0..rowed], stream.rows.items[stream.taken..][0..rowed]);
                stream.taken += n;
                const cached_tokens = stream.cached_tokens;
                const admitted_ns = stream.admitted_ns;
                self.mutex.unlock(io);
                if (given == 0) if (taker.began) |began| began(taker.context, cached_tokens, admitted_ns);
                for (batch[0..n], 0..) |token, i| {
                    if (i < rowed) taker.row.?(taker.context, &batch_rows[i]);
                    given += 1;
                    if (taker.take(taker.context, token) == .stop) {
                        wanted = false;
                        break;
                    }
                }
                probe_at = self.probeAfterNow();
                self.mutex.lockUncancelable(io);
                if (!wanted) self.abandon(&stream, given);
                continue;
            }
            if (stream.state == .ended) break;
            if (!wanted) {
                // The decoding thread still names the stream: it is this call's until that thread lets go.
                self.changed.waitUncancelable(io, &self.mutex);
                continue;
            }
            // a `wake` before this turn is answered before any wait
            const timed_out = if (self.wakes != woken)
                true
            else if (self.changed.waitTimeout(io, &self.mutex, .{ .deadline = probe_at })) |_|
                probe_at.durationFromNow(io).raw.nanoseconds <= 0
            else |err| switch (err) {
                // The connection's task is being ended (the server is stopping): the taker knows, and is asked
                // now, so that it is the taker that says the reply was not sent.
                error.Timeout, error.Canceled => true,
            };
            if (timed_out and wanted and stream.state != .ended and stream.tokens.items.len == stream.taken) {
                // Nothing came for a while (the queue, a prompt being filled): is the reply still wanted?
                woken = self.wakes;
                self.mutex.unlock(io);
                const signal = taker.probe(taker.context);
                probe_at = self.probeAfterNow();
                self.mutex.lockUncancelable(io);
                if (signal == .stop) {
                    wanted = false;
                    self.abandon(&stream, given);
                }
            }
        }
        const ending = stream.ending;
        const cached_tokens = stream.cached_tokens;
        const admitted_ns = stream.admitted_ns;
        var tail = stream.tail;
        // a reply its connection left has its counts and no end token
        if (!wanted) tail.end = null;
        if (tail.end != null and stream.rows.items.len > stream.tokens.items.len) tail.end_row = stream.rows.items[stream.tokens.items.len];
        self.mutex.unlock(io);
        if (given == 0 and admitted_ns != 0) if (taker.began) |began| began(taker.context, cached_tokens, admitted_ns);

        if (wanted) if (ending.fault) |fault| {
            said.* = ending.said;
            return fault;
        };
        return .{ .finish = if (wanted) ending.finish else .cut, .tokens = given, .cached_tokens = cached_tokens, .tail = tail };
    }

    fn probeAfterNow(self: *const Scheduler) Io.Clock.Timestamp {
        return .fromNow(self.io, .{ .raw = self.options.probe, .clock = .awake });
    }

    /// The connection wants no more of its reply. Under the mutex. A stream in the queue is taken out here,
    /// by its own thread: no decoder is offered it, and nothing waits for the round being run. Any other is
    /// the decoding thread's to let go, at its flag.
    fn abandon(self: *Scheduler, stream: *Stream, given: usize) void {
        stream.kept = given;
        stream.cancel.store(true, .release);
        if (stream.state == .queued) {
            self.unlink(stream);
            return self.finishQueued(stream, .{ .finish = .cut });
        }
        self.work.signal(self.io);
    }

    // -- the queue, under the mutex ------------------------------------------------------------------------------

    fn pushBack(self: *Scheduler, stream: *Stream) void {
        stream.next = null;
        if (self.tail) |tail| tail.next = stream else self.head = stream;
        self.tail = stream;
        self.queued += 1;
    }

    fn pushFront(self: *Scheduler, stream: *Stream) void {
        stream.next = self.head;
        self.head = stream;
        if (self.tail == null) self.tail = stream;
        self.queued += 1;
    }

    fn popFront(self: *Scheduler) ?*Stream {
        const stream = self.head orelse return null;
        self.head = stream.next;
        if (self.head == null) self.tail = null;
        stream.next = null;
        self.queued -= 1;
        return stream;
    }

    /// Takes a stream that is in the queue out of it.
    fn unlink(self: *Scheduler, stream: *Stream) void {
        var before: ?*Stream = null;
        var at = self.head;
        while (at != stream) : (at = at.?.next) before = at;
        if (before) |b| b.next = stream.next else self.head = stream.next;
        if (stream.next == null) self.tail = before;
        stream.next = null;
        self.queued -= 1;
    }

    /// Ends a stream that is in no slot. Its connection may free it once the mutex is released, so nothing
    /// reads it after this.
    fn finishQueued(self: *Scheduler, stream: *Stream, ending: Ending) void {
        stream.ending = ending;
        stream.state = .ended;
        self.changed.broadcast(self.io);
    }

    /// Takes the waiting requests whose connections have gone out of the queue.
    fn dropCancelled(self: *Scheduler) void {
        var before: ?*Stream = null;
        var at = self.head;
        while (at) |stream| {
            const next = stream.next;
            if (stream.cancel.load(.acquire)) {
                if (before) |b| b.next = next else self.head = next;
                if (next == null) self.tail = before;
                self.queued -= 1;
                self.finishQueued(stream, .{ .finish = .cut });
            } else before = stream;
            at = next;
        }
    }

    // -- the decoding thread -------------------------------------------------------------------------------------

    fn run(self: *Scheduler) void {
        const io = self.io;
        while (true) {
            self.mutex.lockUncancelable(io);
            while (!self.closing and self.head == null and self.live_count == 0) self.work.waitUncancelable(io, &self.mutex);
            if (self.closing) {
                while (self.popFront()) |stream| {
                    var ending: Ending = .{ .finish = .cut, .fault = error.Failed };
                    ending.said.set("the server is stopping", .{});
                    self.finishQueued(stream, ending);
                }
                self.mutex.unlock(io);
                for (self.live, 0..) |held, slot| if (held != null) self.leave(slot, .cancelled, .{ .finish = .cut });
                return;
            }
            self.dropCancelled();
            self.mutex.unlock(io);

            // A cancel is read here, before a round, and nowhere else.
            for (self.live, 0..) |held, slot| if (held) |stream| {
                if (stream.cancel.load(.acquire)) self.leave(slot, .cancelled, .{ .finish = .cut });
            };
            self.admit();
            if (self.live_count != 0) self.round();
        }
    }

    /// The slot's stream leaves the decoder, and then its connection hears the end: when a client has its
    /// reply's end, its slot is free.
    fn leave(self: *Scheduler, slot: usize, why: Leave, ending: Ending) void {
        const io = self.io;
        const stream = self.live[slot].?;
        self.mutex.lockUncancelable(io);
        const kept = if (why == .cancelled) stream.kept else stream.tokens.items.len;
        self.mutex.unlock(io);
        const counted = self.decoder.table.retire(self.decoder.context, slot, why, kept);
        if (stream.filling) |filling| filling.store(false, .release);
        self.live[slot] = null;
        self.endings[slot] = null;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.live_count -= 1;
        if (why == .cancelled) stream.tail = counted;
        stream.ending = ending;
        stream.state = .ended;
        self.changed.broadcast(io);
    }

    fn refuse(self: *Scheduler, stream: *Stream, fault: request_file.Error, said: request_file.Diagnostic, fallback: []const u8) void {
        var ending: Ending = .{ .finish = .cut, .fault = fault, .said = said };
        if (said.len == 0) ending.said.set("{s}", .{fallback});
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.finishQueued(stream, ending);
    }

    /// While a slot is free, the oldest waiting request is offered to the decoder.
    fn admit(self: *Scheduler) void {
        const io = self.io;
        while (self.live_count < self.options.slots) {
            self.mutex.lockUncancelable(io);
            self.dropCancelled();
            const waiting = self.popFront();
            // the decoding thread's from here: its connection can no longer take it out of the queue
            if (waiting) |offered| offered.state = .offered;
            self.mutex.unlock(io);
            const stream = waiting orelse return;
            var said: request_file.Diagnostic = .{};
            // before the decoder takes it: what the decoder runs of the prompt in `admit` is no wait
            const offered_ns = Io.Clock.awake.now(io).toNanoseconds();
            // The tokens are appended by this thread alone, so it reads them without the lock.
            switch (self.decoder.table.admit(self.decoder.context, stream.job, stream.tokens.items, &said)) {
                .admitted => |taken| {
                    std.debug.assert(taken.slot < self.options.slots and self.live[taken.slot] == null);
                    self.live[taken.slot] = stream;
                    self.endings[taken.slot] = null;
                    self.sequence += 1;
                    _ = self.admitted.fetchAdd(1, .acq_rel);
                    self.mutex.lockUncancelable(io);
                    defer self.mutex.unlock(io);
                    stream.slot = taken.slot;
                    stream.sequence = self.sequence;
                    if (stream.admissions == 0) {
                        stream.cached_tokens = taken.cached_tokens;
                        stream.admitted_ns = offered_ns;
                    }
                    stream.admissions += 1;
                    if (stream.tokens.items.len == 0) if (stream.filling) |filling| filling.store(true, .release);
                    stream.state = .live;
                    self.live_count += 1;
                },
                .wait => {
                    if (self.live_count == 0) {
                        // Nothing is live to make room by ending: it would wait for ever.
                        self.refuse(stream, error.BadRequest, said, "the prompt is larger than the cache pool can hold");
                        continue;
                    }
                    self.mutex.lockUncancelable(io);
                    defer self.mutex.unlock(io);
                    stream.state = .queued;
                    self.pushFront(stream);
                    return;
                },
                .never => self.refuse(stream, error.BadRequest, said, "the prompt is larger than the cache pool can ever hold"),
                .failed => |fault| self.refuse(stream, fault, said, "the request could not be started"),
            }
        }
    }

    /// The newest admitted stream.
    fn newest(self: *const Scheduler) usize {
        var found: ?usize = null;
        for (self.live, 0..) |held, slot| if (held) |stream| {
            if (found == null or stream.sequence > self.live[found.?].?.sequence) found = slot;
        };
        return found.?;
    }

    fn round(self: *Scheduler) void {
        const io = self.io;
        var told: Round = .{ .scheduler = self };
        while (true) switch (self.decoder.table.round(self.decoder.context, &told)) {
            .ran => break,
            .short => {
                const slot = self.newest();
                if (self.live_count == 1) {
                    var ending: Ending = .{ .finish = .cut, .fault = error.Failed };
                    ending.said.set("the cache pool cannot hold this reply", .{});
                    self.leave(slot, .failed, ending);
                    return;
                }
                // Taken off, to run again from the head of the queue with the tokens its client has.
                const stream = self.live[slot].?;
                self.decoder.table.preempt(self.decoder.context, slot);
                if (stream.filling) |filling| filling.store(false, .release);
                self.live[slot] = null;
                self.endings[slot] = null;
                _ = self.preempted.fetchAdd(1, .acq_rel);
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);
                self.live_count -= 1;
                stream.state = .queued;
                self.pushFront(stream);
            },
            .failed => {
                for (self.live, 0..) |held, slot| if (held != null) {
                    var ending: Ending = .{ .finish = .cut, .fault = error.Failed };
                    ending.said.set("decoding failed", .{});
                    self.leave(slot, .failed, ending);
                };
                return;
            },
        };
        _ = self.rounds.fetchAdd(1, .acq_rel);
        for (self.endings, 0..) |ended, slot| if (ended) |ending| {
            self.leave(slot, if (ending.fault != null) .failed else .ended, ending);
        };
    }
};
