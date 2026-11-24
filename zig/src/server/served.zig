//! A model behind TensorFold's `engine_api.Engine` (specs 059, 045 WS-04; one file for both models since spec
//! 069): a request arrives as the prompt's token ids with its sampling, and the reply leaves as token ids in
//! events. TensorFold's server stands before it and does the text (the template, the tokenizer, the three
//! APIs, the stop strings, the calls), each turn over the whole conversation: it renders and encodes it for the
//! prompt, again for `history_len`, a probe for `shared_prefixes`, and since TensorFold 1.0.4 four renders of
//! a one-message conversation for `decode_spans`, plus 1.0.5's probe for `rewind_len` (unread here); the encodes are answered by
//! `segments.zig`, which the `serve` commands put between the server and its tokenizer. What decodes is the model's own, a session for one reply at a time and
//! a pool for several, so a reply's tokens are theirs.
//!
//! A request is a thread of its own from `submit` to its `finished` event: it reads the request, then runs
//! the model's `reply`, which on a session holds `decoding` and taps the reply's tokens and in a pool queues
//! a job and takes them (`Scheduler.decode`, through `Live.taker`). Events of one request therefore come from
//! one thread, in order, and `submit` and `cancel` return at once.
//! That thread is `served_live.zig`'s `Live`; this file keeps the requests' table and their admission,
//! `status`, `memory`, a request's constraint and its refusal.
//!
//! The token that ends a reply is sent as the reply's last (`Live.closing`): TensorFold's server expects it
//! after the reply's own (it counts it in the usage and closes an open call by it). It is in `status`'s rate
//! and no stop string reads it, as in TensorFold's lanes and in vLLM. A request's `eos` is the reply's: empty,
//! under `ignore_eos`, the reply runs to its length. Once a stop string has ended a reply, what a model still
//! gives of the same window is dropped (TensorFold's `Stream.commit`, SGLang's `output_ids_through_stop`).
//!
//! What a model gives is `Served`'s doc comment. Here is what no model's reply decides: the table of live
//! requests, `status` and `memory`, the refusal of a prompt the context cannot hold, the sampling, the
//! request's constraint (`constraintFor`), a token's way to the server with the stop strings (`Live.give`),
//! the `prefilled` event, and how a reply's end is named (`Live.reasonOf`).
//!
//! A request's `think_budget` and `loop_guard` (`--thinking-budget`, a request's `thinking_budget`,
//! `--loop-guard`; spec 074) are enforced as TensorFold 1.0.5's own engine enforces them, and `Info.loop_guard`
//! says so: `Live.constrained` puts `decode/guard.zig`'s `Guard` around the request's constraint, or around
//! none, with the request's `think_close` and `think_end` as the host made them, and the model decodes under
//! it as under any constraint, on a session and in a pool alike. The period of a loop that fired is the
//! reply's `Stats.loop_period`. A request that asks for neither has no guard and is decoded as before. Under
//! one a reply is drafted by the model's drafter alone: `decode/copied.zig` offers no copy to a reply that
//! has a constraint.
//!
//! What the interface asks and no model does yet: `chunks` and `history_len` are not read; of
//! `Structure`, a regex, a choice, a grammar and a structural tag are refused where xgrammar is not (below);
//! of `Stats`, `min_rows` and the prefill's widths; of `Memory`, `cache` and `standing`. Of TensorFold
//! 1.2.3's and 1.2.5's: the new `Info` fields (`readouts`, `rerank`, `benchmark_ranks`, `benchmark_group`,
//! `ledger`, `reported_context_window`, `reasoning`, `image`, `prefill_min_chunk`; `thinking_switch` and
//! `thinking_applicable` are the server's from the template and tokenizer) stay at their defaults, as
//! `REFERENCE.md`'s last rows say; a request's `readout` never reaches an engine that is no readout.
//! would print `none` under the default although the models draft); `VTable.load` stays null (`GET /v1/load`
//! then has no engine group); `Info.penalties` false (a nonzero presence/frequency penalty the host
//! refuses, since 1.2.5); `Info.decode_share` null; `Request.penalties` unread with it and `Request.loop_stop`
//! unread (`--loop-guard-stop` is `command.zig`'s `unserved`); `Structure.compiled`/`budget` are the host's own
//! (it compiles and frees them); `SubmitError.MemoryPressure` never returned; `Open`'s new fields are
//! TensorFold's `open`'s, which zspark's commands do not call. Of 1.2.3's and 1.2.5's (no spec):
//! `Info.readouts`, `rerank`, `benchmark_ranks`, `benchmark_group`, `ledger`, `reported_context_window`
//! and `image` null, `reasoning` and `prefill_min_chunk` their defaults and `VTable.submit_image` unset,
//! as `REFERENCE.md`'s last rows say; a `Request.readout` never arrives, `submit` refusing it first, and
//! `--profile`'s one value is the default `stable`. Of 1.2.8's (spec 174): `Info.ocr` null (upstream's
//! `/v1/ocr` reader is its own `Info.ocr` interface for engines that read page images; no zspark engine is
//! one, so `POST /v1/ocr` answers the host's own `bad_input` refusal), `PreparedImage.frame` and `Size` the
//! engine's to fill or leave, `chat.Input`'s `cancel`, `image_bytes` and `fill_room` the host's (set only
//! by its own routes), `Rendered.frames` passed through unread, and `Admit`, `Ledger`, `Memory` and
//! `MemorySource` re-exports now (their shapes unchanged).
//!
//!
//! Structured output since TensorFold 1.0.12 (spec 106): where the host loads xgrammar (`libtfgrammar`, and
//! the checkpoint's `tokenizer.json` and `config.json` `vocab_size` in `Setup.model_dir`), it compiles every
//! structure itself, a required call as a structural tag, and hands it on with its `compiler`;
//! `constraintFor` then decodes under xgrammar's masks (`xgrammar.zig`), after the lead the template writes
//! there, and `refusal` only checks that xgrammar's vocabulary covers the logits. Where it does not, the host
//! hands on a required call as `Request.call` and a structure uncompiled (zspark's patch of `server.zig`), and
//! zspark's own matcher serves a JSON schema and the call gate a required call, as under 1.0.5.
//!
//! What TensorFold's interface (1.0.4 to 1.2.1) asks and is left at its default, field by field, and why:
//! `REFERENCE.md` beside this file, "TensorFold's interface left at its defaults".
//!
//! A constraint that cannot be enforced (those three kinds, a schema's keyword the matcher does not read, a
//! required call's tool it cannot hold, no grammar engine) is refused before a reply exists. The host asks
//! `refusal` on the connection's thread before it sends anything (`Engine.refusal`, upstream's own since 1.2.5,
//! vendored `engine_api.zig` and `chat.zig`; spec 072): the constraint is compiled, and the client has a 400
//! with the keyword and its path, streamed or not, as vLLM and SGLang answer. A constraint that compiles is
//! kept for the request's reply (`ready`, by what the request asks, since it has no id yet), so a request
//! compiles once: a required call's gate is 0.2 ms a tool. Where the host does not ask, the same refusal is
//! still the reply's failure.
//! A prompt's token id outside the model's vocabulary is refused there too, first (spec 081): the host's tokenizer
//! gives none, and vLLM answers one with a 400 as well.
//!
//! Admission (spec 074 WS-03), as TensorFold 1.0.5's lane host has it and where it leaves a choice as vLLM
//! and SGLang do. `submit` takes every request: what waits is unbounded unless `max_queued` is set
//! (`--max-queued`; vLLM's `max_num_queued_reqs` and SGLang's `max_queued_requests`, both unset by default,
//! and TensorFold's serial host's fixed 256), and then a request beyond the `lanes` being decoded and that
//! many waiting is `error.Busy`, the host's "the engine is busy; retry shortly", a 503; since TensorFold 1.1.1 the
//! host submits a stream before its headers (#912), so a stream is refused with the 503 too (1.0.12 needed
//! zspark's patch `Engine.full`, asked before them, which 1.1.1 retired). `Busy` is otherwise only a
//! request or its thread that could not be made, as in TensorFold. A request that waits and is cancelled (its
//! client left: the host asks every 50 ms and calls `cancel`) ends at once as `cancelled` and is never
//! decoded: on a session it stands in `Live.enters`' line, which `cancel` wakes, and in a pool `cancel` wakes
//! the scheduler it waits at (`Taker.waits`, `Scheduler.wake`), which takes it out of its queue. Requests
//! take a session's turn in the order they came.
//!
//! This file imports no model and not `cuda/cuda.zig`: the device's bytes are a counter its caller names.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const request_file = @import("request.zig");
const sched = @import("scheduler.zig");
const live_file = @import("served_live.zig");
const structured = @import("structured.zig");
const xgrammar = @import("xgrammar.zig");
const grammar = @import("../text/grammar.zig");
const contract = @import("../decode/contract.zig");
const sampling = @import("../decode/sampling.zig");
const text_frontend = @import("../text/frontend.zig");

/// A request as every model's reply takes it: read, and found to fit.
pub const Ask = struct {
    /// The arena's, the request's ids.
    prompt: []i32,
    settings: sampling.Sampling,
    /// At least one, and no more than the context leaves after the prompt.
    max_tokens: usize,
    /// A warm pass (`Served.warm_turns`): the prompt run for the next turn's sake, one token that nobody is
    /// sent, and never ahead of a live request. Its states are what it is for.
    warm: bool = false,
    /// Log probabilities, with this many best tokens a row (`Request.logprobs`, where `Served.logprobs`): the
    /// model gives each token's row to `Live.row` before the token.
    logprobs: ?u8 = null,
    /// The request's images (`Request.images`, where `Served.vision`): each one's tower input and the prompt
    /// rows its features take, in order.
    images: []const api.Image = &.{},
};

/// How a request ended, for its `finished` event.
pub const Ended = struct { reason: api.Reason, stats: api.Stats = .{}, message: []const u8 = "" };

/// `Served`'s contract as declarations: a name added to its doc comment is added here.
pub const owed = [_]contract.Owed{
    .{ .name = "Behind", .is = .{ .type = .{} } },
    .{ .name = "reply", .is = .{ .function = 4 } },
    .{ .name = "counts", .is = .{ .function = 1 } },
    .{ .name = "closed", .is = .{ .function = 1 } },
};

/// The engine over the model `M`, a namespace that declares:
///
/// - `Behind`: what its replies run on (its session, its pool, what it reads of a request beside them), a
///   field of `Served` the model's command fills.
/// - `reply(live, arena, ask, said) !Ended`: one request's reply, on the request's thread, through
///   `live`: `give` each token (or `taker()` for a pool under `Scheduler.decode`), `prefilled` and `cached`
///   for the prompt, `closing` for the token that ended it, `reasonOf` for the event. A session's reply holds
///   the session's turn (`Live.enters`, `Live.leaves`); `Live.constrained` compiles what the request asks and lends it: the constraint is the
///   request's, freed on its thread once `reply` has returned, so a model keeps it no longer and frees nothing.
/// - `counts(behind) ?sched.Counts`: its pool's own count of who decodes and who waits, for `status`; null
///   where it keeps none, and `status` then counts the lanes.
/// - `closed(behind)`: called by `close` once no request is left.
/// Optional `promptCache(behind)` reads published `sched.Cache.Snapshot` accounting; `cacheCapacity(behind)`
/// gives the `api.PromptCacheCapacity` for the applied budget where the model has one, no copy held.
pub fn Served(comptime M: type) type {
    return struct {
        const Self = @This();

        gpa: Allocator,
        io: std.Io,
        /// zspark's own reading of the checkpoint's text side: the template's openers, the leads after `</think>`.
        frontend: *const text_frontend.Frontend,
        /// What makes a required call or a document the reply's constraint; null refuses such a request.
        compiler: ?structured.Compiler,
        /// The model's: its session, or its pool above it.
        behind: M.Behind,
        /// Positions a prompt and its reply share.
        capacity: usize,
        /// The model's token ids are below it (its config's `vocab_size`): a prompt with one that is not is
        /// refused (`refusal`).
        vocab: u32,
        /// Replies at once.
        lanes: u32,
        /// `<think>` and `</think>`, where the tokenizer has both: a constraint begins after the second where
        /// the prompt leaves the reply thinking.
        think: ?request_file.Think,
        /// The device buffers' bytes alive (`cuda.account.bytes`), read atomically from any thread.
        device_bytes: *const usize,

        /// How many requests may wait beyond the `lanes` being decoded; one more is `error.Busy`. Null: no
        /// bound, as TensorFold's lane host.
        max_queued: ?usize = null,
        /// `Info.warm_turns`: the host prefills each chat reply's next turn through a background request,
        /// which `reply` hands the model as a warm `Ask` (`serve --warm-turns`; TensorFold 1.0.12's).
        warm_turns: bool = false,
        /// `Info.logprobs`: the model gives each token's row (`Ask.logprobs`, `Live.row`). Flash Next on one rank.
        logprobs: bool = false,
        /// `Info.vision`: the host takes image parts and hands them over in `Ask.images` (spec 019 WS-04;
        /// `images.zig`'s `Offer`). Null: an image part is the host's 400.
        vision: ?api.Vision = null,
        /// `Info.draft_sources` (TensorFold 1.2.1): the serving line's `drafts:` text names the proposals the
        /// model offers; each command fills it (empty prints `none`). `--no-drafts`/`plain_only` still print `off`.
        draft_sources: api.DraftSources = .{},
        /// Startup retained-prefix budget, including whether the command named it explicitly.
        prompt_cache_plan: ?api.PromptCachePlan = null,

        /// Guards `live`, `closing`, `open`, the session's turn and the rates.
        mutex: std.Io.Mutex = .init,
        emptied: std.Io.Condition = .init,
        live: std.AutoHashMapUnmanaged(api.Id, *Live) = .empty,
        closing: bool = false,
        /// The requests taken whose end has not been sent: what `max_queued` counts.
        open: usize = 0,
        /// A session decodes one reply at a time, and its constraint is compiled and asked in the same turn
        /// (`Live.enters`): whether it is taken, and who waits for it, the oldest first.
        busy: bool = false,
        first: ?*Live = null,
        last: ?*Live = null,
        turned: std.Io.Condition = .init,
        /// For `status`, under `mutex`: the tokens given in each of the last `spans` quarter seconds (a span's
        /// number beside its count), and the last prompt's rate with when it ended.
        given: [spans]u32 = @splat(0),
        given_at: [spans]i64 = @splat(-1),
        prefill_rate: f64 = 0,
        prefill_at: i96 = 0,
        /// The most device bytes `memory` has seen, a reply's end included.
        peak: std.atomic.Value(u64) = .init(0),
        /// The constraints `refusal` compiled that no reply has taken yet, under `mutex`; `kept` counts them,
        /// and the oldest gives its place when all are held (its request never came: its client left).
        ready: [places]?Ready = @splat(null),
        kept: u64 = 0,

        /// What a request's constraint is made from, beside the server's own: a request that says the same is
        /// held to an equal constraint, so `refusal`'s is the reply's (`Live.constrained`).
        pub const Asked = struct {
            what: enum { call, json, json_schema },
            /// A call's tools, a schema's text: the request's.
            text: []const u8,
            thinking: bool,

            pub fn of(request: *const api.Request, thinking: bool) ?Asked {
                if (request.call != null) return .{ .what = .call, .text = request.tools_json, .thinking = thinking };
                const structure = request.structure orelse return null;
                // xgrammar's (`xgrammar.zig`): compiled by the host already, and again by the reply from its cache
                if (structure.compiler != null) return null;
                return switch (structure.kind) {
                    .json => .{ .what = .json, .text = "", .thinking = thinking },
                    .json_schema => .{ .what = .json_schema, .text = structure.text, .thinking = thinking },
                    .regex, .choice, .grammar, .structural_tag => null,
                };
            }
        };

        /// A constraint compiled ahead of its reply, with the arena it was compiled in: `text` and whatever
        /// the constraint still reads of the request's tools are there, not in the host's request.
        pub const Ready = struct {
            what: @FieldType(Asked, "what"),
            text: []const u8,
            thinking: bool,
            made: grammar.Constraint,
            arena: std.heap.ArenaAllocator,
            at: u64,

            fn is(self: *const Ready, asked: Asked) bool {
                return self.what == asked.what and self.thinking == asked.thinking and std.mem.eql(u8, self.text, asked.text);
            }

            fn deinit(self: Ready) void {
                self.made.deinit();
                self.arena.deinit();
            }
        };

        /// Constraints held between a request's refusal and its reply: a request is submitted at once after it
        /// is asked, so more than a few wait only when clients leave in between.
        const places = 8;

        /// TensorFold's status averages over two seconds (`lane_host.zig`'s `window_ns`).
        const spans = 8;
        const span_ns = 250 * std.time.ns_per_ms;

        pub fn engine(self: *Self) api.Engine {
            // here and not where `M` is given: a model's `reply` names `Live`, which is this type's
            comptime contract.hold("server/served.zig's Served (its doc comment)", M, &owed);
            return .{ .ctx = self, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory, .refusal = refusal } };
        }

        /// Ends every request and waits for their threads; `submit` then refuses. Before the session or the
        /// pool is destroyed.
        pub fn close(self: *Self) void {
            const io = self.io;
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.closing = true;
            var each = self.live.valueIterator();
            while (each.next()) |one| one.*.cancelNow();
            while (self.live.count() != 0) self.emptied.waitUncancelable(io, &self.mutex);
            self.live.deinit(self.gpa);
            for (&self.ready) |*slot| {
                if (slot.*) |one| one.deinit();
                slot.* = null;
            }
            M.closed(&self.behind);
        }

        fn info(ctx: *anyopaque) api.Info {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return .{ .name = "zspark", .choice_policy = .single, .lanes = self.lanes, .context_window = @intCast(self.capacity), .call_gates = true, .loop_guard = true, .structures = true, .reasoning = self.think != null, .warm_turns = self.warm_turns, .logprobs = self.logprobs, .vision = self.vision, .draft_sources = self.draft_sources, .prompt_cache = true, .prompt_cache_capacity = if (@hasDecl(M, "cacheCapacity")) M.cacheCapacity(&self.behind) else null, .prompt_cache_plan = self.prompt_cache_plan };
        }

        /// What of `request` this engine cannot enforce, for the host's 400 (`Engine.refusal`, asked on the
        /// connection's thread before anything is sent; TensorFold 1.2.5's `engine_refusal.of` cuts the answer at
        /// `api.refusal_bytes`). The constraint is compiled here, as its reply would compile it, and kept for the
        /// reply (`ready`), so a request compiles once. A compile writes nothing the grammar engine shares
        /// (`text/grammar.zig`): it needs no lock beside a reply being decoded. A request without a constraint is
        /// two null checks. Before either, a prompt's id outside the vocabulary is refused, whatever else the
        /// request asks.
        fn refusal(ctx: *anyopaque, request: *const api.Request, out: []u8) ?[]const u8 {
            const self: *Self = @ptrCast(@alignCast(ctx));
            // The host's tokenizer gives no such id; one that came all the same would index the embedding
            // table by it (spec 081). vLLM refuses it where it renders a prompt (`renderers/base.py`, "Token id
            // N is out of vocabulary", a 400); SGLang checks `token_ids_logprob` so, and TensorFold nothing.
            for (request.prompt, 0..) |id, at| if (id >= self.vocab)
                return std.fmt.bufPrint(out, "prompt token {d} is {d}, out of the model's vocabulary; valid range is [0, {d})", .{ at, id, self.vocab }) catch out[0..0];
            if (request.call == null and request.structure == null) return null;
            if (request.structure) |structure| if (structure.compiler) |compiler| {
                const words = if (self.compiler) |own| own.engine.words else {
                    const no_engine = "structured output is not available: this server was started without a grammar engine";
                    @memcpy(out[0..no_engine.len], no_engine);
                    return out[0..no_engine.len];
                };
                // xgrammar's vocabulary is config.json's `vocab_size`, the logits' width
                if (compiler.words < words) return std.fmt.bufPrint(out, "{s}: xgrammar's vocabulary ({d} tokens) is narrower than the model's logits ({d})", .{ structure.field, compiler.words * 32, words * 32 }) catch out[0..0];
                return null;
            };
            // an id is below the vocabulary, so below 2^31: the server's ids are the model's as they lie
            const thinking = self.thinks(@ptrCast(request.prompt));
            const asked = Asked.of(request, thinking);
            // asked again before its reply (the host prepares a request it compacts more than once)
            if (asked) |same| if (self.holds(same)) return null;
            var said: request_file.Diagnostic = .{};
            // the constraint's own arena: it may outlive the host's request, in `ready`
            var arena: std.heap.ArenaAllocator = .init(self.gpa);
            const made = self.constraintFor(arena.allocator(), request, thinking, &said) catch |err| {
                arena.deinit();
                return switch (err) {
                    // out of memory: no words in the fixed buffer the reply would give it; the reply fails as before
                    error.OutOfMemory => null,
                    error.BadRequest => blk: {
                        const message = said.message();
                        @memcpy(out[0..@min(out.len, message.len)], message[0..@min(out.len, message.len)]);
                        break :blk out[0..@min(out.len, message.len)];
                    },
                    // not the client's: left to the reply, which fails with it as before
                    error.Failed => null,
                };
            };
            // what compiles is one of `Asked`'s kinds
            self.keep(asked.?, made.?, arena);
            return null;
        }

        fn holds(self: *Self, asked: Asked) bool {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            for (&self.ready) |*slot| {
                if (slot.*) |*one| if (one.is(asked)) return true;
            }
            return false;
        }

        /// `made` and the arena it was compiled in wait for the reply of a request that asks `asked`; where
        /// they cannot be kept they are freed, and the reply compiles its own.
        fn keep(self: *Self, asked: Asked, made: grammar.Constraint, arena: std.heap.ArenaAllocator) void {
            var one: Ready = .{ .what = asked.what, .text = "", .thinking = asked.thinking, .made = made, .arena = arena, .at = 0 };
            one.text = one.arena.allocator().dupe(u8, asked.text) catch return one.deinit();
            var gone: ?Ready = null;
            {
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                var place: *?Ready = &self.ready[0];
                for (&self.ready) |*slot| {
                    const other = slot.* orelse {
                        place = slot;
                        break;
                    };
                    if (other.at < place.*.?.at) place = slot;
                }
                gone = place.*;
                self.kept += 1;
                one.at = self.kept;
                place.* = one;
            }
            if (gone) |old| old.deinit();
        }

        /// What `refusal` compiled for a request that asks `asked`, now its taker's to free.
        pub fn taken(self: *Self, asked: Asked) ?Ready {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            for (&self.ready) |*slot| {
                const one = slot.* orelse continue;
                if (!one.is(asked)) continue;
                slot.* = null;
                return one;
            }
            return null;
        }

        fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const io = self.io;
            const one = self.gpa.create(Live) catch return error.Busy;
            one.* = .{ .served = self, .id = id, .request = request, .sink = sink, .submitted_ns = self.nowNs() };
            {
                self.mutex.lockUncancelable(io);
                defer self.mutex.unlock(io);
                if (self.closing) {
                    self.gpa.destroy(one);
                    return error.Closed;
                }
                // full: the replies being decoded and as many waiting as may wait
                if (self.max_queued) |most| if (self.open >= self.lanes + most) {
                    self.gpa.destroy(one);
                    return error.Busy;
                };
                self.live.put(self.gpa, id, one) catch {
                    self.gpa.destroy(one);
                    return error.Busy;
                };
                self.open += 1;
            }
            const thread = std.Thread.spawn(.{}, Live.run, .{one}) catch {
                self.uncount(one);
                self.forget(one);
                return error.Busy;
            };
            thread.detach();
        }

        fn cancel(ctx: *anyopaque, id: api.Id) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.live.get(id)) |one| one.cancelNow();
        }

        /// `one`'s end is about to be sent: it no longer counts against `max_queued`.
        pub fn uncount(self: *Self, one: *Live) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (one.counted) self.open -= 1;
            one.counted = false;
        }

        fn status(ctx: *anyopaque, out: *api.Status, stream_tokens: []u32) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            // Read the scheduler before our mutex; cache hooks read atomics alone.
            const pool: ?sched.Counts = M.counts(&self.behind);
            const cache = if (@hasDecl(M, "promptCache")) M.promptCache(&self.behind) else null;
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            const count: u32 = @intCast(self.live.count());
            const running: u32 = if (pool) |counts| @intCast(@min(counts.live, count)) else @min(count, self.lanes);
            const now = self.nowNs();
            const span = @divFloor(now, span_ns);
            var recent: u64 = 0;
            for (self.given, self.given_at) |tokens, at| {
                if (at > span - spans) recent += tokens;
            }
            out.* = .{
                .running = running,
                .waiting = count - running,
                .decode_tokens_per_second = @as(f64, @floatFromInt(recent)) / (@as(f64, spans * span_ns) / std.time.ns_per_s),
                .prefill_tokens_per_second = if (now - self.prefill_at <= spans * span_ns) self.prefill_rate else 0,
                .preemptions = if (pool) |counts| counts.preempted else null,
                .prompt_cache = if (cache) |held| .{ .states = held.states, .bytes = held.bytes } else null,
            };
            var each = self.live.valueIterator();
            while (each.next()) |one| {
                const made = one.*.made.load(.acquire);
                out.generation_tokens += made;
                if (one.*.counted and one.*.filling.load(.acquire)) out.fill_unknown = true;
                if (out.streams < stream_tokens.len) {
                    stream_tokens[out.streams] = @intCast(one.*.request.prompt.len + made);
                    out.streams += 1;
                }
            }
        }

        pub fn deviceBytes(self: *const Self) u64 {
            return @atomicLoad(usize, self.device_bytes, .monotonic);
        }

        /// The device buffers' bytes, as `cuda.account` counts them: read without the device, from any thread.
        fn memory(ctx: *anyopaque, reset_peak: bool) ?api.Memory {
            const self: *Self = @ptrCast(@alignCast(ctx));
            const active = self.deviceBytes();
            if (reset_peak) self.peak.store(active, .release);
            return .{ .active = active, .peak = @max(self.peak.fetchMax(active, .acq_rel), active) };
        }

        pub fn nowNs(self: *const Self) i96 {
            return std.Io.Clock.awake.now(self.io).toNanoseconds();
        }

        /// A token went to a request: counted in its quarter second.
        pub fn tallied(self: *Self) void {
            const span: i64 = @intCast(@divFloor(self.nowNs(), span_ns));
            const slot: usize = @intCast(@mod(span, spans));
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.given_at[slot] != span) {
                self.given_at[slot] = span;
                self.given[slot] = 0;
            }
            self.given[slot] += 1;
        }

        /// The request's thread is done with `one`: nothing reads it after.
        pub fn forget(self: *Self, one: *Live) void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            _ = self.live.remove(one.id);
            one.emitted.deinit(self.gpa);
            self.gpa.destroy(one);
            self.emptied.broadcast(self.io);
        }

        /// Whether `prompt` leaves the reply thinking: its last `<think>` is not closed.
        pub fn thinks(self: *const Self, prompt: []const i32) bool {
            return request_file.thinks(self.think, prompt);
        }

        /// The constraint `request` asks for, or null: a required call over the offered tools, or a document.
        pub fn constraintFor(self: *const Self, arena: Allocator, request: *const api.Request, thinking: bool, said: *request_file.Diagnostic) request_file.Error!?grammar.Constraint {
            var asked: request_file.Request = .{};
            if (request.call != null) {
                // The gate's form is read from the template by `text/call_gate.zig` itself; of TensorFold's gate
                // the offered tools are what it needs, and they are the request's.
                asked.tools = std.json.parseFromSliceLeaky(std.json.Value, arena, request.tools_json, .{}) catch |err| switch (err) {
                    error.OutOfMemory => return error.OutOfMemory,
                    else => {
                        said.set("tool_choice: the offered tools could not be read", .{});
                        return error.BadRequest;
                    },
                };
                asked.structure = .{ .call = .{} };
            } else if (request.structure) |structure| switch (structure.kind) {
                .json => asked.structure = .{ .document = .{ .spec = .json_object, .field = structure.field } },
                .json_schema => asked.structure = .{ .document = .{ .spec = .{ .json_schema = structure.text }, .field = structure.field } },
                .regex, .choice, .grammar, .structural_tag => if (structure.compiler == null) {
                    said.set("{s}: structured output by {t} is not enforced by this engine without xgrammar (libtfgrammar): send a JSON schema", .{ structure.field, structure.kind });
                    return error.BadRequest;
                },
            } else return null;
            // After thinking a call and a document open as the template writes them there (specs 034, 035).
            var compiler = self.compiler orelse {
                said.set("structured output is not available: this server was started without a grammar engine", .{});
                return error.BadRequest;
            };
            if (thinking) {
                compiler.gate.lead = self.frontend.call_lead;
                compiler.lead = self.frontend.content_lead;
            }
            if (request.structure) |structure| if (structure.compiler) |library| {
                // xgrammar's, after the lead the template writes there as zspark's own are (a call's for a structural tag)
                var told: [512]u8 = undefined;
                const made = xgrammar.compile(self.gpa, self.io, library, xgrammar.kindOf(structure.kind), structure.text, compiler.engine.words, &told) catch |err| switch (err) {
                    error.OutOfMemory => |e| return e,
                    error.Grammar => {
                        said.set("{s}: the grammar cannot be enforced: {s}", .{ structure.field, std.mem.sliceTo(&told, 0) });
                        return error.BadRequest;
                    },
                };
                errdefer made.deinit();
                return try compiler.led(self.gpa, arena, made, if (structure.kind == .structural_tag) compiler.gate.lead else compiler.lead);
            };
            return compiler.compile(self.gpa, arena, asked, said);
        }

        /// One request from `submit` to its `finished` event, on its own thread (`served_live.zig`).
        pub const Live = live_file.Live(Self, M);
    };
}
