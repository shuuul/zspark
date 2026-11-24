//! `Fake`, the model of the scheduler's tests: a decoder of several slots on the tiny tokenizer, with nothing
//! computed, and the text side a request's thread has around `Scheduler.decode` (`Model.generate`: a request
//! made a job, the reply's tokens made text, the result).

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const request_file = @import("../../src/server/request.zig");
const scheduler = @import("../../src/server/scheduler.zig");
const Admission = scheduler.Admission;
const Decoder = scheduler.Decoder;
const Done = scheduler.Done;
const Job = scheduler.Job;
const Leave = scheduler.Leave;
const Outcome = scheduler.Outcome;
const Round = scheduler.Round;
const Scheduler = scheduler.Scheduler;

const grammar = @import("../../src/text/grammar.zig");
const structured_fixture = @import("structured_fixture.zig");
const tokenizer_file = @import("../../src/text/tokenizer.zig");

/// A model for the scheduler's tests: a text side and a decoder of several slots on the tiny tokenizer, with
/// nothing computed. A request's reply is a script chosen by the request alone, so the reply a request
/// gets alone is known; a round gives each live stream a few of its tokens, a width that depends on the
/// stream's own position and seed and never on who is beside it. A structured request is compiled as a served
/// one is, on its connection's task, and decoded at random under its masks on the decoding
/// thread. Memory is a count of positions: a stream holds its prompt and the tokens it has run.
pub const Fake = struct {
    gpa: Allocator,
    io: Io,
    bench: structured_fixture.Bench = undefined,
    slots: []?*Held,
    /// The free replies, chosen by the request's `seed`: `seed % scripts.len`.
    scripts: []const []const u8 = &default_scripts,
    /// Positions the pool holds, and a stream at most.
    pool: usize = 1 << 20,
    capacity: usize = 1 << 16,
    /// Rounds a prompt takes before its reply's first token: its positions over this, rounded up.
    chunk: usize = 64,
    /// A stream's tokens a round at most.
    width: usize = 4,
    /// With `manual`, a round waits for `allow`.
    manual: bool = false,
    permits: usize = 0,
    mutex: Io.Mutex = .init,
    allowed: Io.Condition = .init,
    /// Fails the round after this many, once.
    fail_round: ?u64 = null,
    /// Positions of the pool that nothing live holds and nothing gives back.
    held_back: usize = 0,
    /// The prompt's held tokens are told by the stream's first round (`Round.held`) and not at admission.
    held_late: bool = false,
    /// Each token is given its row first (`Round.rows`): its own token, its position's negative as its log
    /// probability; a reply that stops is ended by `end_token`, whose row comes after the reply's own.
    logprobs: bool = false,

    /// What happened, for the tests to read once the scheduler is quiet (or under `mutex`).
    order: std.ArrayList(u64) = .empty,
    retired: std.ArrayList(Retired) = .empty,
    /// Admissions, retirements and preemptions as they came.
    log: std.ArrayList(Event) = .empty,
    generated: std.atomic.Value(u64) = .init(0),
    rounds: std.atomic.Value(u64) = .init(0),
    most_live: std.atomic.Value(usize) = .init(0),
    preempted: std.atomic.Value(u64) = .init(0),
    replayed: std.atomic.Value(u64) = .init(0),
    constrained: std.atomic.Value(u64) = .init(0),
    prepared: std.atomic.Value(u64) = .init(0),
    released: std.atomic.Value(u64) = .init(0),

    pub const Retired = struct { seed: u64, why: Leave, kept: usize, generated: usize };
    pub const Event = struct { what: enum { admitted, retired, preempted }, seed: u64 };

    pub const default_scripts = [_][]const u8{
        "one two three four five six seven eight nine ten",
        "alpha beta gamma delta",
        "The quick brown fox jumps over the old dog, twice; then END it rests. And more after.",
        "x",
        "a b c d e f g h i j k l m n o p q r s t u v w x y",
        "first line\nsecond line\nthird line END fourth line never seen by a stop",
    };

    const Held = struct {
        fake: *Fake,
        seed: u64,
        /// The prompt's positions: the request's `max_tokens`-free knob, see `promptOf`.
        prompt: usize,
        max_tokens: usize,
        /// The reply fails once it has this many tokens: the number after `f=`.
        fail_at: ?usize,
        /// Tokens of reasoning the reply opens with.
        reasoning: usize,
        /// The free reply's tokens, reasoning first; empty under a constraint.
        script: []const i32,
        constraint: ?grammar.Constraint,
        // -- the connection's --
        decoder: tokenizer_file.StreamDecoder,
        taken: usize = 0,
        // -- the decoding thread's --
        filling: usize = 0,
        told: bool = false,
        at: usize = 0,
        position: grammar.Position = undefined,
        random: std.Random.DefaultPrng,
        ending: usize = 0,
        mask: []u32 = &.{},
    };

    pub fn init(self: *Fake, gpa: Allocator, io: Io, slots: usize) !void {
        self.* = .{ .gpa = gpa, .io = io, .slots = try gpa.alloc(?*Held, slots) };
        @memset(self.slots, null);
        errdefer gpa.free(self.slots);
        try self.bench.init(gpa);
    }

    pub fn deinit(self: *Fake) void {
        self.log.deinit(self.gpa);
        self.order.deinit(self.gpa);
        self.retired.deinit(self.gpa);
        self.bench.deinit(self.gpa);
        self.gpa.free(self.slots);
    }

    /// One request of the tests: what picks its script, and what its reply must be.
    pub const Request = struct {
        /// Text the knobs are read from: `p=`, `s=`, `f=` (`numberAfter`).
        messages: std.json.Value,
        tools: ?std.json.Value = null,
        thinking: bool = true,
        max_tokens: ?usize = null,
        seed: ?u64 = null,
        structure: ?request_file.Structure = null,
        /// Where the reply's tokens and rows go as its connection takes them (`Fake.logprobs`).
        heard: ?*Heard = null,
    };

    /// The token that ends a reply under `logprobs`, none of the scripts'.
    pub const end_token: i32 = 0;

    /// What a connection took: each token and the row given before it, and the end's (`Tail.end_row`).
    pub const Heard = struct {
        tokens: std.ArrayList(i32) = .empty,
        rows: std.ArrayList(request_file.LogprobRow) = .empty,
        end_row: ?request_file.LogprobRow = null,
        /// Rows given before a token was, and not taken by one: one at most, the next token's.
        pending: usize = 0,

        pub fn deinit(self: *Heard, gpa: Allocator) void {
            self.tokens.deinit(gpa);
            self.rows.deinit(gpa);
        }
    };

    /// The row the fake gives `token` at the reply's position `at`.
    pub fn rowOf(token: i32, at: usize) request_file.LogprobRow {
        var row: request_file.LogprobRow = .{ .token = @intCast(token), .logprob = -@as(f32, @floatFromInt(at)), .count = 1 };
        row.ids[0] = @intCast(token);
        row.logprobs[0] = row.logprob;
        return row;
    }

    pub const Result = struct {
        finish: request_file.Finish,
        prompt_tokens: usize,
        /// Of them, the tokens the decoder did not run.
        cached_tokens: usize = 0,
        /// The reply's tokens its connection took, reasoning included.
        completion_tokens: usize,
        reasoning_tokens: usize,
        /// What the decoder counted of the reply (`Done.tail`).
        rounds: u64 = 0,
    };

    /// The fake behind a scheduler, as a request's thread uses the two.
    pub const Model = struct {
        fake: *Fake,
        scheduler: *Scheduler,

        /// One reply, on the caller's thread and from many at once: the job made, `Scheduler.decode` with a
        /// taker that makes each token text for `sink` and asks it with an empty piece while nothing comes,
        /// then the result; the job is released whatever happened.
        pub fn generate(self: Model, arena: Allocator, request: Request, sink: request_file.Sink, said: *request_file.Diagnostic) request_file.Error!Result {
            const job = try self.fake.prepare(arena, request, said);
            defer self.fake.release(job);
            var texts: Texts = .{ .job = job, .sink = sink, .heard = request.heard, .gpa = self.fake.gpa };
            const done = try self.scheduler.decode(job, .{ .context = &texts, .take = Texts.take, .probe = Texts.probe, .row = if (request.heard != null) Texts.row else null }, said);
            if (request.heard) |heard| heard.end_row = done.tail.end_row;
            return finish(job, done, sink);
        }

        const Texts = struct {
            job: *Job,
            sink: request_file.Sink,
            heard: ?*Heard,
            gpa: Allocator,

            fn take(context: *anyopaque, token: i32) request_file.Sink.Signal {
                const self: *Texts = @ptrCast(@alignCast(context));
                if (self.heard) |heard| {
                    heard.tokens.append(self.gpa, token) catch @panic("out of memory");
                    heard.pending = 0;
                }
                return Fake.take(self.job, token, self.sink);
            }

            fn row(context: *anyopaque, one: *const request_file.LogprobRow) void {
                const self: *Texts = @ptrCast(@alignCast(context));
                const heard = self.heard.?;
                heard.rows.append(self.gpa, one.*) catch @panic("out of memory");
                heard.pending += 1;
            }

            fn probe(context: *anyopaque) request_file.Sink.Signal {
                const self: *Texts = @ptrCast(@alignCast(context));
                return self.sink.write(.reasoning, "");
            }
        };
    };

    pub fn model(self: *Fake, over: *Scheduler) Model {
        return .{ .fake = self, .scheduler = over };
    }

    pub fn decoder(self: *Fake) Decoder {
        return .{ .context = self, .table = &decoder_table };
    }

    /// Lets `rounds` more rounds run, under `manual`.
    pub fn allow(self: *Fake, rounds: usize) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.permits += rounds;
        self.allowed.broadcast(self.io);
    }

    /// Every round from now on waits for `allow` again.
    pub fn hold(self: *Fake) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.manual = true;
        self.permits = 0;
    }

    /// Every round from now on runs unasked.
    pub fn free(self: *Fake) void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.manual = false;
        self.allowed.broadcast(self.io);
    }

    const decoder_table: Decoder.Table = .{ .admit = admit, .round = round, .retire = retire, .preempt = preempt };

    /// The number after `key` in the request's messages: `p=` the prompt's positions (8 without), `s=` the
    /// script (the request's `seed` without, or the first), `f=` where the reply fails.
    fn numberAfter(arena: Allocator, request: Request, key: []const u8) ?u64 {
        const text = std.json.Stringify.valueAlloc(arena, request.messages, .{}) catch return null;
        const at = (std.mem.indexOf(u8, text, key) orelse return null) + key.len;
        var end = at;
        while (end < text.len and std.ascii.isDigit(text[end])) end += 1;
        return std.fmt.parseInt(u64, text[at..end], 10) catch null;
    }

    fn promptOf(arena: Allocator, request: Request) usize {
        return @intCast(numberAfter(arena, request, "p=") orelse 8);
    }

    fn prepare(self: *Fake, arena: Allocator, request: Request, said: *request_file.Diagnostic) request_file.Error!*Job {
        const gpa = self.gpa;
        const prompt = promptOf(arena, request);
        if (prompt >= self.capacity) {
            said.set("This server's maximum context length is {d} tokens; the prompt is {d}", .{ self.capacity, prompt });
            return error.BadRequest;
        }
        const seed = numberAfter(arena, request, "s=") orelse request.seed orelse 0;
        // Compiled here, on the connection's task, beside whatever the decoding thread is filling.
        const constraint = try self.bench.compiler().compile(gpa, arena, .{ .tools = request.tools, .structure = request.structure }, said);
        errdefer if (constraint) |made| made.deinit();
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(gpa);
        var reasoning: usize = 0;
        if (request.thinking) {
            self.bench.tokenizer.encode(gpa, "Thinking.", &ids) catch return error.Failed;
            reasoning = ids.items.len;
        }
        if (constraint == null) {
            var rest: std.ArrayList(u32) = .empty;
            defer rest.deinit(gpa);
            self.bench.tokenizer.encode(gpa, self.scripts[seed % self.scripts.len], &rest) catch return error.Failed;
            try ids.appendSlice(gpa, rest.items);
        }
        const script = try gpa.alloc(i32, ids.items.len);
        errdefer gpa.free(script);
        for (script, ids.items) |*token, id| token.* = @intCast(id);
        const mask = try gpa.alloc(u32, if (constraint) |made| made.words else 0);
        errdefer gpa.free(mask);
        const held = try gpa.create(Held);
        errdefer gpa.destroy(held);
        held.* = .{
            .fake = self,
            .seed = seed,
            .prompt = prompt,
            .max_tokens = request.max_tokens orelse std.math.maxInt(usize),
            .fail_at = if (numberAfter(arena, request, "f=")) |at| @intCast(at) else null,
            .reasoning = reasoning,
            .script = script,
            .constraint = constraint,
            .decoder = try .init(gpa, &self.bench.tokenizer, .{}),
            .random = .init(seed *% 0x9e3779b97f4a7c15 +% 1),
            .mask = mask,
        };
        _ = self.prepared.fetchAdd(1, .acq_rel);
        return @ptrCast(held);
    }

    fn release(self: *Fake, job: *Job) void {
        const held: *Held = @ptrCast(@alignCast(job));
        if (held.constraint) |made| made.deinit();
        held.decoder.deinit(self.gpa);
        self.gpa.free(held.mask);
        self.gpa.free(held.script);
        self.gpa.destroy(held);
        _ = self.released.fetchAdd(1, .acq_rel);
    }

    fn take(job: *Job, token: i32, sink: request_file.Sink) request_file.Sink.Signal {
        const held: *Held = @ptrCast(@alignCast(job));
        const kind: request_file.Kind = if (held.taken < held.reasoning) .reasoning else .content;
        held.taken += 1;
        const piece = held.decoder.next(@intCast(token));
        if (piece.len == 0) return .more;
        return sink.write(kind, piece);
    }

    fn finish(job: *Job, done: Done, sink: request_file.Sink) Result {
        const held: *Held = @ptrCast(@alignCast(job));
        const rest = held.decoder.finish();
        if (rest.len != 0 and done.finish != .cut) _ = sink.write(if (held.taken <= held.reasoning) .reasoning else .content, rest);
        return .{
            .finish = done.finish,
            .prompt_tokens = held.prompt,
            .cached_tokens = done.cached_tokens,
            .completion_tokens = done.tokens,
            .reasoning_tokens = @min(done.tokens, held.reasoning),
            .rounds = done.tail.rounds,
        };
    }

    /// Positions the live streams hold.
    fn used(self: *const Fake) usize {
        var sum: usize = self.held_back;
        for (self.slots) |slot| if (slot) |held| {
            sum += held.prompt + held.at;
        };
        return sum;
    }

    fn admit(context: *anyopaque, job: *Job, sent: []const i32, said: *request_file.Diagnostic) Admission {
        const self: *Fake = @ptrCast(@alignCast(context));
        const held: *Held = @ptrCast(@alignCast(job));
        const needs = held.prompt + sent.len;
        if (needs > self.pool) {
            said.set("the prompt is {d} tokens and the cache pool holds {d}: it can never be held", .{ needs, self.pool });
            return .never;
        }
        // Room for a round of every stream is left, or a request taken off would be admitted to be taken off.
        var live: usize = 1;
        for (self.slots) |taken| live += @intFromBool(taken != null);
        if (self.used() + needs + live * self.width > self.pool) return .wait;
        const slot = for (self.slots, 0..) |taken, index| {
            if (taken == null) break index;
        } else unreachable;
        self.slots[slot] = held;
        // A request taken off runs its prompt and what it was sent again, as a prompt, and goes on from there.
        held.filling = (needs + self.chunk - 1) / self.chunk;
        if (sent.len != 0) {
            _ = self.replayed.fetchAdd(1, .acq_rel);
            std.debug.assert(sent.len == held.at);
        } else if (held.constraint) |constraint| {
            held.position = constraint.start() catch return .{ .failed = error.Failed };
            _ = self.constrained.fetchAdd(1, .acq_rel);
        }
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.order.append(self.gpa, held.seed) catch return .{ .failed = error.OutOfMemory };
        self.log.append(self.gpa, .{ .what = .admitted, .seed = held.seed }) catch return .{ .failed = error.OutOfMemory };
        _ = self.most_live.fetchMax(live, .acq_rel);
        return .{ .admitted = .{ .slot = slot, .cached_tokens = if (self.held_late) 0 else held.prompt / 2 } };
    }

    fn retire(context: *anyopaque, slot: usize, why: Leave, kept: usize) scheduler.Tail {
        const self: *Fake = @ptrCast(@alignCast(context));
        const held = self.slots[slot].?;
        self.slots[slot] = null;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.retired.append(self.gpa, .{ .seed = held.seed, .why = why, .kept = kept, .generated = held.at }) catch {};
        self.log.append(self.gpa, .{ .what = .retired, .seed = held.seed }) catch {};
        // the fake's count of a reply is the tokens it made
        return .{ .rounds = held.at };
    }

    fn preempt(context: *anyopaque, slot: usize) void {
        const self: *Fake = @ptrCast(@alignCast(context));
        const held = self.slots[slot].?;
        self.slots[slot] = null;
        _ = self.preempted.fetchAdd(1, .acq_rel);
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        self.log.append(self.gpa, .{ .what = .preempted, .seed = held.seed }) catch {};
    }

    /// How many tokens the stream runs at `at`: its own seed and position decide, nothing beside it.
    fn widthAt(self: *const Fake, held: *const Held) usize {
        var hash: std.hash.Wyhash = .init(held.seed);
        hash.update(std.mem.asBytes(&held.at));
        return 1 + hash.final() % self.width;
    }

    fn round(context: *anyopaque, told: *Round) Outcome {
        const self: *Fake = @ptrCast(@alignCast(context));
        const io = self.io;
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            while (self.manual and self.permits == 0) self.allowed.waitUncancelable(io, &self.mutex);
            if (self.manual) self.permits -= 1;
        }
        // What this round's rows need, asked before any is written: short changes nothing.
        var needs: usize = 0;
        for (self.slots) |taken| if (taken) |held| {
            needs += held.prompt + held.at + if (held.filling != 0) 0 else if (held.constraint != null) 1 else self.widthAt(held);
        };
        if (needs > self.pool) return .short;
        const done = self.rounds.fetchAdd(1, .acq_rel);
        if (self.fail_round) |at| if (done == at) {
            self.fail_round = null;
            return .failed;
        };
        for (self.slots, 0..) |taken, slot| if (taken) |held| {
            if (self.held_late and !held.told) {
                held.told = true;
                told.held(slot, held.prompt / 2);
            }
            if (held.filling != 0) {
                held.filling -= 1;
                continue;
            }
            if (held.fail_at) |at| if (held.at >= at) {
                told.fail(slot, error.Failed, "the scripted reply failed");
                continue;
            };
            if (held.constraint) |constraint| {
                self.masked(told, slot, held, constraint);
                continue;
            }
            const end = @min(held.script.len, held.max_tokens);
            const n = @min(self.widthAt(held), end - held.at, (held.fail_at orelse end) -| held.at);
            if (self.logprobs) {
                var rows: [16]request_file.LogprobRow = undefined;
                for (rows[0..n], held.script[held.at..][0..n], held.at..) |*row, token, at| row.* = rowOf(token, at);
                told.rows(slot, rows[0..n]);
            }
            told.emit(slot, held.script[held.at..][0..n]);
            held.at += n;
            _ = self.generated.fetchAdd(n, .acq_rel);
            if (held.at == end and self.logprobs and held.at == held.script.len) {
                told.rows(slot, &.{rowOf(end_token, held.at)});
                told.tail(slot, .{ .end = end_token });
            }
            if (held.at == end) told.end(slot, if (held.at == held.script.len) .stop else .length);
        };
        return .ran;
    }

    /// One token under the constraint: any the mask allows, a closer preferred once the reply is long enough.
    fn masked(self: *Fake, told: *Round, slot: usize, held: *Held, constraint: grammar.Constraint) void {
        const tokenizer = &self.bench.tokenizer;
        const stop = self.bench.stops[0];
        const hurry: usize = 10;
        const raw_end = "\n</parameter>\n";
        if (held.at >= held.max_tokens) return told.end(slot, .length);
        if (held.at < held.reasoning) {
            told.emit(slot, held.script[held.at..][0..1]);
            held.at += 1;
            return;
        }
        const count = held.at - held.reasoning;
        if (count == 2000) return told.fail(slot, error.Failed, "the masked reply did not end");
        const random = held.random.random();
        constraint.fill(held.position, held.mask) catch return told.fail(slot, error.Failed, "the mask could not be filled");
        var allowed_count: usize = 0;
        var chosen: ?u32 = null;
        // one of the allowed, each as likely: the k-th is taken with chance 1/k
        for (0..tokenizer.count()) |index| {
            const token: u32 = @intCast(index);
            if (!grammar.allowed(held.mask, token)) continue;
            allowed_count += 1;
            if (random.uintLessThan(usize, allowed_count) == 0) chosen = token;
        }
        var token = chosen orelse return told.fail(slot, error.Failed, "the mask allows nothing");
        if (count >= hurry) closing: for ("\"]}") |closer| for (0..tokenizer.count()) |index| {
            const candidate: u32 = @intCast(index);
            if (!grammar.allowed(held.mask, candidate)) continue;
            const bytes = tokenizer.tokenBytes(candidate);
            if (bytes.len != 1 or bytes[0] != closer) continue;
            token = candidate;
            break :closing;
        };
        if (count >= hurry) {
            var raw: usize = 0;
            var next: ?u32 = null;
            for (0..tokenizer.count()) |index| {
                const candidate: u32 = @intCast(index);
                if (!grammar.allowed(held.mask, candidate)) continue;
                const bytes = tokenizer.tokenBytes(candidate);
                if (bytes.len == 1 and std.mem.indexOfScalar(u8, "\"x\n", bytes[0]) != null) raw += 1;
                if (bytes.len != 0 and std.mem.startsWith(u8, raw_end[held.ending..], bytes)) next = candidate;
            }
            if (raw == 3 and next != null) {
                token = next.?;
                held.ending += tokenizer.tokenBytes(token).len;
            } else held.ending = 0;
            if (held.ending == raw_end.len) held.ending = 0;
        }
        if (grammar.allowed(held.mask, stop) and (count >= hurry * 40 or random.boolean())) token = stop;
        held.position = (constraint.accept(held.position, token) catch return told.fail(slot, error.Failed, "the constraint failed")) orelse
            return told.fail(slot, error.Failed, "the constraint refused a token its mask allowed");
        if (token == stop) return told.end(slot, .stop);
        told.emit(slot, &.{@intCast(token)});
        held.at += 1;
        _ = self.generated.fetchAdd(1, .acq_rel);
    }
};
