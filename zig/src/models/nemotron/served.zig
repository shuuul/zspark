//! Nemotron-3.5-Transcribe behind TensorFold's host (spec 151 WS-08): the engine of
//! `POST /v1/audio/transcriptions` (`Engine.VTable.submit_transcription`) and of chat and completions with
//! `audio_url`/`input_audio` parts, as vLLM 0.32's NeMo SpeechLM plugin serves them. A transcription's file is
//! decoded and resampled (`zig/src/audio/`), the prompt made of `prompt.zig`'s `transcription` (the
//! `nemotron3p5` render's ChatML with thinking closed), and the clip's perception rows (`perception.zig`)
//! laid over its one `<|audio|>` row's expansion (`decode/placeholders.zig`, `state.audio`). A chat is the
//! host's render of the checkpoint's template: each audio part's placeholder expanded to the clip's rows
//! (`server/audios.zig`), `prepare` decoding it as the route's file.
//!
//! The reply decodes with `draft.zig`'s `Decoder`: serial steps without a drafter, the MTP head or a DFlash2
//! rig where `serve` bound one (`Drafter`, `draft.Drafter`). Its text is read as vLLM's `nemotron_v3`
//! reasoning parser reads it (`prompt.split`/`transcript`): thinking off by default — the template writes the
//! assistant's opener closed by `<think></think>` — and on when the request asks, the reasoning then the
//! content. The reply is greedy: the card's `generation_config.json` samples nothing, and the plugin runs
//! the model's own decode (greedy) for transcription; a request's sampling is honoured where it names one.
//!
//! One request runs at a time, on this engine's one thread (the decoder's state and the perception are one);
//! the rest wait in order. A clip is at most `prompt.max_seconds` (3,600, the plugin's bound) and is never
//! cut: the PEE's online windows are the encoder's own segmentation.
//!
//! `ServePair` (`pair`) is the same engine on rank 0 of the two-rank split (WS-06): `held_ns`'s lead tells
//! each window, commit, logits row and MTP draft to rank 1 before it runs, `mirror.zig` replaying them —
//! rank 1 holds no jobs and no sampler, so nothing of a request's settings, stop strings or cancellation
//! has to cross: a stop is simply no more ops. An admitted clip's 16-kHz PCM rides the `audio` op
//! in bounded frames for rank 1's perception to encode.

const std = @import("std");
const api = @import("engine_api");
const audio = @import("../../audio/audio.zig");
const cuda = @import("../../cuda/cuda.zig");
const text_tokenizer = @import("../../text/tokenizer.zig");
const config_file = @import("config.zig");
const draft_file = @import("draft.zig");
const forward_file = @import("forward.zig");
const model_file = @import("model.zig");
const perception_file = @import("perception.zig");
const prompt_file = @import("prompt.zig");
const sampling = @import("../../decode/sampling.zig");
const shared_draft = @import("../../decode/draft.zig");
const masks_file = @import("../../decode/masks.zig");
const held_ns = @import("held_ns.zig");
const lead_file = @import("../../ranks/lead.zig");
const ranks_forward = @import("ranks_forward.zig");
const Allocator = std.mem.Allocator;

/// The card's served context, shared by the prompt and reply (the backbone config permits more).
pub const default_context = 65536;
/// Requests waiting behind the running one before `submit*` answers Busy, unless `--max-queued` says.
pub const most_waiting = 64;
/// The longest clip's audio rows (`prompt.clipRows` of `max_seconds`), at the config's own featurizer.
pub fn clipRowsCeiling(c: *const config_file.Config) usize {
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    return prompt_file.clipRows(arena.allocator(), c, @intFromFloat(prompt_file.max_seconds * @as(f64, @floatFromInt(c.perception.sample_rate)))) catch 0;
}

/// The model shapes the engine serves on: `one`'s whole model (a process's only rank), `pair`'s one
/// rank of two held through `ranks_forward.Ranks` (WS-06). Every decode call is the generic's — the
/// pair's `Decoder` gathers its partials where the one's writes them whole.
pub const one = struct {
    pub const Model = model_file.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Graphs = forward_file.Graphs;
    /// A lone rank's windows can capture; the pair's are told ops, never graphs (`held_ns.Graphs` empty).
    pub const graphed = true;
    pub const Decoder = draft_file.Decoder;
    pub const Flash = @import("dflash2/drafter.zig").DrafterOf(2);
    pub fn stateReset(s: *State) !void {
        return s.reset();
    }
    pub fn setAudio(s: *State, features: cuda.View, rows: []const u32) void {
        s.audio = .{ .features = features, .rows = rows };
    }
    pub fn audioNone(s: *State) void {
        s.audio = null;
    }
    pub fn pos(s: *const State) usize {
        return s.pos;
    }
    pub fn vocab(m: *const Model) usize {
        return m.config.backbone.vocab;
    }
    pub fn eos(m: *const Model) []const u32 {
        return m.config.eos;
    }
};
/// `one`'s rank-held form: this process's share of the pair (`held_ns` the decoder's namespace; the
/// peer's windows and kept drafts are the same bits because the request's inputs crossed the channel).
pub const pair = struct {
    pub const M = held_ns;
    pub const Model = M.Model;
    pub const State = M.State;
    pub const Buffers = M.Buffers;
    pub const Graphs = M.Graphs;
    pub const graphed = false;
    pub const Decoder = shared_draft.Decoder(M);
    pub const Flash = @import("dflash2/drafter.zig").DrafterOf(1);
    pub fn stateReset(s: *State) !void {
        return s.reset();
    }
    pub fn setAudio(s: *State, features: cuda.View, rows: []const u32) void {
        s.audio = .{ .features = features, .rows = rows };
    }
    pub fn audioNone(s: *State) void {
        s.audio = null;
    }
    pub fn pos(s: *const State) usize {
        return s.pos;
    }
    pub fn vocab(m: *const Model) usize {
        return m.share.config.backbone.vocab;
    }
    pub fn eos(m: *const Model) []const u32 {
        return m.share.config.eos;
    }
};

pub fn ServeOf(comptime P: type) type {
    return struct {
        const Self = @This();
        /// Rank 0 of two tells rank 1 each windowed op before it runs (`held_ns`'s `lead`, `mirror.zig`
        /// replaying them): the channel's `Lead` sits on the pair's buffers; what the ops do not carry — the
        /// reply's start (`reset`), a clip's PCM and prompt rows (`audio`) and the session's end —
        /// this engine tells at the points the one-rank path sets them.
        lead: ?*lead_file.Lead = null,
        /// The drafter a reply drafts by, where `serve` gave one (MTP or DFlash2); null: serial steps.
        drafter: ?draft_file.Drafter = null,
        /// `drafter` drafts by DFlash2: its pair tells the `flash_*` ops itself (`pass.zig`'s `lead`); an
        /// MTP `draft` round is this engine's to tell.
        flash: ?*P.Flash = null,

        io: std.Io,
        model: *const P.Model,
        perception: *perception_file.Perception,
        tok: *const text_tokenizer.Tokenizer,
        state: *P.State,
        buffers: *P.Buffers,
        graphs: *P.Graphs,
        sampler: *sampling.Sampler,
        /// bf16 [the longest clip's rows, output_dim]: the perception's embeddings of the running clip.
        features: cuda.Buffer,
        /// `<|audio|>`'s id (the checkpoint's `audio_locator_tag`, added token 131,072).
        locator: u32,
        context: usize,
        most_waiting: usize = most_waiting,
        /// Drafts a round asks for at most (`Decoder.depth` bounds it further).
        depth: usize = draft_file.forward_max_rows - 1,

        gpa: Allocator,
        mutex: std.Io.Mutex = .init,
        cond: std.Io.Condition = .init,
        queue: std.ArrayList(*Job) = .empty,
        running: ?*Job = null,
        closing: bool = false,
        worker: ?std.Thread = null,
        /// The reply's start told: `state.reset` was the one-rank path's (`P.stateReset`); the follower
        /// drops its open window and audio rows with it.
        fn tellReset(self: *Self) anyerror!void {
            if (self.lead) |lead| try lead.tell(@backingInt(ranks_forward.Op.reset), &.{}, &.{});
        }

        /// An admitted clip's request-owned PCM and placeholder rows. Row ids ride in chunks of a
        /// thousand; PCM follows in bounded frames, including clips larger than the channel's limit.
        fn tellAudio(self: *Self, samples: []const f32, rows: []const u32) anyerror!void {
            const lead = self.lead orelse return;
            var ids: [1000]i32 = undefined;
            var at: usize = 0;
            var first = true;
            while (at < rows.len) {
                const n = @min(ids.len, rows.len - at);
                for (rows[at..][0..n], ids[0..n]) |r, *id| id.* = @intCast(r);
                if (first) {
                    try lead.tell(@backingInt(ranks_forward.Op.audio), &.{ @intCast(rows.len), @intCast(samples.len) }, ids[0..n]);
                    first = false;
                } else try lead.tell(@backingInt(ranks_forward.Op.audio), &.{ 0, 0 }, ids[0..n]);
                at += n;
            }
            const pcm = std.mem.sliceAsBytes(samples);
            at = 0;
            while (at < pcm.len) {
                const n = @min(ranks_forward.audio_frame_bytes, pcm.len - at);
                try lead.channel.request(pcm[at..][0..n]);
                at += n;
            }
        }

        const Job = struct {
            id: api.Id,
            kind: union(enum) { transcription: *const api.TranscriptionRequest, chat: *const api.Request },
            sink: api.Sink,
            stop: std.atomic.Value(bool) = .init(false),

            /// The decoder's work, reported through the existing host statistics contract.
            stats: api.Stats = .{},
            /// A reply that failed as sent: the words of its `finished` event.
            failure: []const u8 = "",

            fn prefilled(ctx: *anyopaque) void {
                const job: *Job = @ptrCast(@alignCast(ctx));
                const event: api.Event = .{ .prefilled = 0 };
                job.sink.event(job.sink.ctx, job.id, &event);
            }
        };

        /// Starts the engine's thread; `close` ends it.
        pub fn start(self: *Self) !void {
            self.worker = try std.Thread.spawn(.{}, work, .{self});
        }

        /// Refuses what is submitted from now, cancels what waits and runs, and joins the thread once all have ended.
        pub fn close(self: *Self) void {
            {
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                self.closing = true;
                for (self.queue.items) |job| job.stop.store(true, .release);
                if (self.running) |job| job.stop.store(true, .release);
                self.cond.broadcast(self.io);
            }
            if (self.worker) |t| t.join();
            self.worker = null;
            if (self.lead) |lead| {
                lead.tell(@backingInt(ranks_forward.Op.end), &.{}, &.{}) catch |err| std.log.err("peer end: {t}", .{err});
                // rank 1 answers at the same count before either closes: without it the follower's
                // next read can lose the end frame to a closed socket (`PeerGone`, exit 1).
                lead.channel.barrier(lead.told) catch |err| std.log.err("peer end barrier: {t}", .{err});
            }
            self.queue.deinit(self.gpa);
        }

        pub fn engine(self: *Self) api.Engine {
            return .{ .ctx = self, .vtable = &vtable };
        }

        const vtable: api.Engine.VTable = .{
            .info = info,
            .submit = submit,
            .cancel = cancel,
            .status = status,
            .memory = memory,
            .submit_transcription = submitTranscription,
        };

        fn info(ctx: *anyopaque) api.Info {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return .{
                .choice_policy = .single,
                .context_window = @intCast(self.context),
                // the card's `generation_config.json` decodes greedily; a request's `sampling` is honoured
                // (`decode` hands it to the shared decoder), so `greedy_only` stays false — the host admits
                // a request's temperature
                .greedy_only = false,
                // `think_budget` and `loop_guard` are the shared lane engine's (`server/served.zig`'s)
                // enforcement; this engine would take the fields and never act — advertise neither
                .thinking_budget = false,
                .loop_guard = false,
                .transcription = .{ .max_seconds = prompt_file.max_seconds, .max_tokens = @intCast(self.context) },
                .audio = .{
                    .ctx = self,
                    .prepare = prepare,
                    .explain = explain,
                    .audio_token = self.locator,
                    // the card's `--limit-mm-per-prompt` for audio is one clip a request
                    .max_audios = 1,
                    .max_seconds = prompt_file.max_seconds,
                    // a request's clips run through the perception into `features` one after another
                    .max_tokens = @intCast(self.features.view.size[0]),
                    // VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
                    .max_bytes = 25 << 20,
                },
            };
        }

        /// Decode into the HTTP request's allocator. Only admitted jobs send these samples to the peer;
        /// refusals and cancellation need no engine-owned file cache or separate audio lifetime.
        fn prepare(ctx: *anyopaque, a: Allocator, bytes: []const u8) anyerror!api.PreparedAudio {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.prepareClip(a, bytes);
        }

        /// The clip's decoded 16-kHz samples and encoder row count, owned by the caller's allocator.
        pub fn prepareClip(self: *Self, a: Allocator, bytes: []const u8) anyerror!api.PreparedAudio {
            const clip = try audio.clip(a, bytes, prompt_file.max_seconds);
            const rows = try self.perception.rows(a, clip.samples.len);
            if (rows == 0) return error.TooShort;
            return .{ .samples = clip.samples, .seconds = clip.seconds, .tokens = @intCast(rows) };
        }

        fn explain(err: anyerror) ?[]const u8 {
            return switch (err) {
                error.TooShort => "The audio is shorter than one encoder row.",
                error.Unsupported => audio.message(error.Unsupported),
                error.Invalid => audio.message(error.Invalid),
                error.TooLong => audio.message(error.TooLong),
                else => null,
            };
        }

        /// A chat or a completion: its prompt as the host rendered it, each audio part's rows in place.
        fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.queued(.{ .id = id, .kind = .{ .chat = request }, .sink = sink });
        }

        fn submitTranscription(ctx: *anyopaque, id: api.Id, request: *const api.TranscriptionRequest, sink: api.Sink) api.TranscriptionSubmitError!void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            return self.queued(.{ .id = id, .kind = .{ .transcription = request }, .sink = sink });
        }

        fn queued(self: *Self, given: Job) error{ Closed, Busy }!void {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            if (self.closing) return error.Closed;
            // `most_waiting` counts behind the running one; an idle engine always takes its first
            if (self.queue.items.len >= self.most_waiting and !(self.queue.items.len == 0 and self.running == null)) return error.Busy;
            const job = self.gpa.create(Job) catch return error.Busy;
            job.* = given;
            self.queue.append(self.gpa, job) catch {
                self.gpa.destroy(job);
                return error.Busy;
            };
            self.cond.signal(self.io);
        }

        fn cancel(ctx: *anyopaque, id: api.Id) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            var dropped: ?*Job = null;
            {
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                if (self.running) |job| if (job.id == id) job.stop.store(true, .release);
                for (self.queue.items, 0..) |job, i| if (job.id == id) {
                    dropped = self.queue.orderedRemove(i);
                    break;
                };
            }
            if (dropped) |job| {
                const event: api.Event = .{ .finished = .{ .reason = .cancelled, .message = "", .stats = .{} } };
                job.sink.event(job.sink.ctx, job.id, &event);
                self.freeJob(job);
            }
            // rank 1 needs no cancel of its own: its ops stop arriving when this rank's job stops
        }

        fn freeJob(self: *Self, job: *Job) void {
            self.gpa.destroy(job);
        }

        fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
            const self: *Self = @ptrCast(@alignCast(ctx));
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            out.* = .{ .running = @intFromBool(self.running != null), .waiting = @intCast(self.queue.items.len) };
        }

        fn memory(_: *anyopaque, _: bool) ?api.Memory {
            return .{ .active = cuda.account.bytes };
        }

        fn work(self: *Self) void {
            while (true) {
                const job = next: {
                    self.mutex.lockUncancelable(self.io);
                    defer self.mutex.unlock(self.io);
                    while (self.queue.items.len == 0 and !self.closing) self.cond.waitUncancelable(self.io, &self.mutex);
                    if (self.queue.items.len == 0) return;
                    const job = self.queue.orderedRemove(0);
                    self.running = job;
                    break :next job;
                };
                var reason: api.Reason = .stop;
                var message: []const u8 = "";
                var arena: std.heap.ArenaAllocator = .init(self.gpa);
                defer arena.deinit();
                if (job.stop.load(.acquire)) {
                    reason = .cancelled;
                } else self.runJob(arena.allocator(), job, &reason, &message);
                {
                    self.mutex.lockUncancelable(self.io);
                    defer self.mutex.unlock(self.io);
                    self.running = null;
                }
                const event: api.Event = .{ .finished = .{ .reason = reason, .message = message, .stats = job.stats } };
                job.sink.event(job.sink.ctx, job.id, &event);
                self.freeJob(job);
            }
        }

        /// The dequeued job's decode, `reason` and `message` filled for its `finished` event.
        fn runJob(self: *Self, a: Allocator, job: *Job, reason: *api.Reason, message: *[]const u8) void {
            if (switch (job.kind) {
                .transcription => self.transcribe(a, job),
                .chat => |r| self.chat(a, job, r),
            }) |r| {
                reason.* = if (job.stop.load(.acquire)) .cancelled else r;
                if (reason.* == .failed) message.* = job.failure;
            } else |err| {
                const what = if (job.kind == .chat) "reply" else "transcription";
                reason.* = .failed;
                message.* = std.fmt.allocPrint(a, "{s} failed: {t}", .{ what, err }) catch "failed";
                std.log.err("{s} {d}: {t}", .{ what, job.id, err });
            }
        }

        fn refuse(r: *const api.TranscriptionRequest, message: []const u8, param: []const u8) Allocator.Error!api.Reason {
            r.result.refused = try r.gpa.dupe(u8, message);
            r.result.param = try r.gpa.dupe(u8, param);
            return .stop;
        }

        /// The reply to `prompt` (its audio rows already set on `state.audio`), each token of `out`: the prompt
        /// in windows, then rounds of the drafter's verified drafts or serial steps, until an end token of `eos`,
        /// `max` tokens or `job.stop`. `on_token`, as a chat's host needs, sees each token as it is chosen, the
        /// end token too. Returns whether the reply ended on an end token.
        fn decode(self: *Self, a: Allocator, job: *Job, prompt: []const i32, eos: []const u32, max: usize, out: *std.ArrayList(u32), on_token: ?Reply.TokenHook, ctx: *anyopaque, settings: ?sampling.Sampling) !bool {
            const started = cuda.monotonic();
            const chosen_settings = settings orelse sampling.Sampling{ .seed = sampling.seedFor(prompt, 0), .temperature = 0.0 };
            if (self.flash) |flash| flash.reset(chosen_settings);
            var decoder: P.Decoder = .{
                .m = self.model.*,
                .state = self.state,
                .b = self.buffers,
                .graphs = if (P.graphed) self.graphs else null,
                .sampler = self.sampler,
                .settings = chosen_settings,
                .eos = eos,
                .logits = try a.alloc(u16, P.vocab(self.model)),
            };
            defer {
                job.stats.rounds = @intCast(decoder.stats.rounds);
                job.stats.drafted = @intCast(decoder.stats.drafted);
                job.stats.accepted = @intCast(decoder.stats.accepted);
            }
            var chosen: [draft_file.forward_max_rows]i32 = undefined;
            var drafts: [draft_file.forward_max_rows]i32 = undefined;
            var told: Told = undefined;
            const enabled = switch (job.kind) {
                .chat => |request| request.drafts,
                .transcription => true,
            };
            const available = if (enabled) self.drafter else null;
            const drafter: ?draft_file.Drafter = if (available) |d| blk: {
                // rank 0 of two: an MTP draft round is an op the peer replays (`Op.draft` — `prompt`'s and
                // each round's ask alike); a DFlash2 rig tells its `flash_*` itself inside the call, nothing
                // extra here (`self.flash` keeps `d` as it is)
                if (self.flash != null or self.lead == null) break :blk d;
                told = .{ .lead = self.lead.?, .d = d, .state = self.state };
                break :blk .{ .context = &told, .draft = Told.draft };
            } else null;
            var kept = try decoder.prompt(drafter, prompt, &chosen);
            if (job.stop.load(.acquire)) return false;
            Job.prefilled(job);
            job.stats.prefill_seconds = @as(f64, @floatFromInt(cuda.monotonic() - started)) / std.time.ns_per_s;
            const first: u32 = @intCast(chosen[kept - 1]);
            try out.append(a, first);
            if (on_token) |hook| if (hook.call(ctx, first)) return false;
            if (std.mem.indexOfScalar(u32, eos, first) != null) return true;
            var last: i32 = chosen[kept - 1];
            var made: usize = 1;
            while (made < max) {
                if (job.stop.load(.acquire)) return false;
                const want = @min(self.depth, max - made);
                const n = if (drafter) |d| try decoder.draft(d, chosen[0..kept], want, &drafts) else 0;
                kept = try decoder.round(last, drafts[0..n], &chosen);
                const rows: u32 = @intCast(n + 1);
                job.stats.min_rows = if (job.stats.min_rows == 0) rows else @min(job.stats.min_rows, rows);
                const take = @min(kept, max - made);
                for (chosen[0..take]) |t| {
                    const token: u32 = @intCast(t);
                    try out.append(a, token);
                    if (on_token) |hook| if (hook.call(ctx, token)) return false;
                    if (std.mem.indexOfScalar(u32, eos, token) != null) return true;
                }
                made += kept;
                last = chosen[kept - 1];
            }
            return false;
        }

        /// An MTP `draft` told as `Op.draft` before it runs here (a DFlash2 rig tells inside its own call):
        /// the peer replays the round on its own kept window, the same `want` asks at the same `mtp_pos`.
        const Told = struct {
            lead: *lead_file.Lead,
            d: draft_file.Drafter,
            state: *P.State,

            fn draft(context: *anyopaque, streams: cuda.View, tokens: []const i32, position: usize, guide: ?masks_file.Guide, out: []i32) anyerror!usize {
                const self: *Told = @ptrCast(@alignCast(context));
                try self.lead.tell(@backingInt(ranks_forward.Op.draft), &.{ @intCast(out.len), @intCast(self.state.mtp_pos) }, tokens);
                return self.d.draft(self.d.context, streams, tokens, position, guide, out);
            }
        };

        /// One request, its answer in `job.kind.transcription.result`: a refusal (`refused`) or the reply's text.
        fn transcribe(self: *Self, a: Allocator, job: *Job) !api.Reason {
            const r = job.kind.transcription;
            const forced = prompt_file.language(r.language) catch
                return refuse(r, try prompt_file.refusal(a, r.language.?), "language");
            const clip = audio.clip(a, r.audio, prompt_file.max_seconds) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return refuse(r, audio.message(err), "file"),
            };
            r.result.seconds = clip.seconds;
            const rows_n = try self.perception.rows(a, clip.samples.len);
            if (rows_n == 0) return refuse(r, "The audio is shorter than one encoder row.", "file");

            const context_text = try prompt_file.sanitize(a, r.prompt);
            const words = try prompt_file.transcription(a, context_text, forced);
            var ids: std.ArrayList(u32) = .empty;
            self.tok.encode(a, words, &ids) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                else => return refuse(r, "prompt must be valid UTF-8 text", "prompt"),
            };
            const at = std.mem.indexOfScalar(u32, ids.items, self.locator) orelse return error.NoAudioLocator;
            const length = ids.items.len - 1 + rows_n;
            if (length + 1 > self.context)
                return refuse(r, try std.fmt.allocPrint(a, "The audio and the prompt take {d} tokens, past the context of {d}", .{ length, self.context }), "prompt");
            const built = try prompt_file.expand(a, ids.items, at, rows_n, self.locator);
            const prompt = built.prompt;
            const rows = built.rows;

            try self.perception.encode(a, clip.samples, self.features.rows(rows_n));
            try self.tellReset();
            try self.tellAudio(clip.samples, rows);
            try P.stateReset(self.state);
            defer P.audioNone(self.state);
            P.setAudio(self.state, self.features.rows(rows_n), rows);
            const limit = @min(@as(usize, r.max_tokens), self.context - length);
            if (limit == 0) return .length;
            var reply: std.ArrayList(u32) = .empty;
            const ended = try self.decode(a, job, prompt, P.eos(self.model), limit, &reply, null, undefined, null);
            var raw: std.ArrayList(u8) = .empty;
            try self.tok.decode(a, reply.items, .{ .skip_special = true }, &raw);
            r.result.text = try r.gpa.dupe(u8, prompt_file.transcript(raw.items));
            if (forced) |f| r.result.language = try r.gpa.dupe(u8, f);
            r.result.prompt_tokens = @intCast(length);
            r.result.tokens = @intCast(reply.items.len);
            return if (ended or reply.items.len < limit) .stop else .length;
        }

        /// A chat or a completion: each audio part's clip through the perception into the next rows of
        /// `features`, the host's prompt's placeholder rows taking them, then the reply, each token sent to the
        /// host as it is chosen, the end token last. The reply's text is read as `nemotron_v3` reads it: thinking
        /// off unless the template left it open — the host strips nothing here; the reasoning split is the
        /// answer's shape, not this route's to cut (vLLM returns `reasoning_content` beside `content` through the
        /// same events).
        fn chat(self: *Self, a: Allocator, job: *Job, r: *const api.Request) !api.Reason {
            if (r.prompt.len == 0 or r.prompt.len >= self.context) {
                job.failure = try std.fmt.allocPrint(a, "This server's maximum context length is {d} tokens; the prompt is {d}", .{ self.context, r.prompt.len });
                return .failed;
            }
            var rows: std.ArrayList(u32) = .empty;
            var offset: usize = 0;
            try self.tellReset();
            for (r.audios) |clip| {
                const n = try self.perception.rows(a, clip.samples.len);
                if (n != clip.tokens) return error.AudioRowsDiffer;
                try self.perception.encode(a, clip.samples, self.features.span(offset, n));
                for (0..n) |i| try rows.append(a, @intCast(clip.at + i));
                try self.tellAudio(clip.samples, rows.items[offset .. offset + n]);
                offset += n;
            }
            const prompt = try a.alloc(i32, r.prompt.len);
            for (prompt, r.prompt) |*p, id| p.* = @intCast(id);

            try P.stateReset(self.state);
            defer P.audioNone(self.state);
            if (offset > 0) P.setAudio(self.state, self.features.rows(offset), rows.items);
            const limit = @min(@as(usize, r.max_tokens), self.context - prompt.len);
            if (limit == 0) return .length;
            var reply_stream: Reply = .{ .job = job, .a = a, .request = r };
            var out: std.ArrayList(u32) = .empty;
            const ended = try self.decode(a, job, prompt, r.eos, limit, &out, .{ .ctx = &reply_stream, .call = Reply.give }, &reply_stream, if (r.sampling) |sd| .{ .seed = sd.seed, .temperature = sd.temperature, .top_k = sd.top_k, .top_p = sd.top_p, .min_p = sd.min_p } else null);
            if (reply_stream.stopped) return .stop;
            const n = out.items.len;
            return if (ended or (n > 0 and std.mem.indexOfScalar(u32, r.eos, out.items[n - 1]) != null)) .stop else .length;
        }

        /// A chat reply's tokens to the host, and its stop strings over them (`server/served.zig`'s `Live.give`).
        const Reply = struct {
            job: *Job,
            a: Allocator,
            request: *const api.Request,
            emitted: std.ArrayList(u32) = .empty,
            stopped: bool = false,

            const TokenHook = struct { ctx: *anyopaque, call: *const fn (ctx: *anyopaque, token: u32) bool };

            fn give(ctx: *anyopaque, token: u32) bool {
                const self: *Reply = @ptrCast(@alignCast(ctx));
                const tokens = [1]u32{token};
                const event: api.Event = .{ .tokens = &tokens };
                self.job.sink.event(self.job.sink.ctx, self.job.id, &event);
                // the end token ends the reply, and no stop string reads it
                if (std.mem.indexOfScalar(u32, self.request.eos, token) != null) return false;
                const stop = self.request.stop orelse return false;
                self.emitted.append(self.a, token) catch return false;
                if (!stop.check(stop.ctx, self.emitted.items)) return false;
                self.stopped = true;
                return true;
            }
        };
    };
}

/// The one-rank engine, the shape `served_test` and a single process's `serve` use.
pub const Serve = ServeOf(one);
/// The rank engine the two-rank `serve` runs on each end.
pub const ServePair = ServeOf(pair);
