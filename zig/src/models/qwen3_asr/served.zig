//! Qwen3-ASR behind TensorFold's host (spec 138 WS-05): the engine of `POST /v1/audio/transcriptions`
//! (`Engine.VTable.submit_transcription`, a zspark patch of the host). A request's file is decoded and resampled
//! (`zig/src/audio/`), its mel features run through the audio tower (`tower.zig`), and the language model decodes
//! greedily over the package's prompt with the tower's embeddings over its `<|audio_pad|>` rows
//! (`decode/placeholders.zig`); the reply's text is read as the package reads it (`transcript.zig`).
//!
//! It also answers chat and completions as vLLM serves this model (spec 150): the host renders the checkpoint's
//! template, reads an `audio_url` or `input_audio` part's bytes and asks `Info.audio`'s `prepare` to decode them,
//! and writes each part's placeholder as its rows (the host's `server/audios.zig`); `chat` runs the clips through
//! the tower and the reply greedily over the rows, its text as the model writes it.
//!
//! One request runs at a time, on this engine's one thread (the device's caches and the tower are one); the rest,
//! transcriptions and chats alike, wait in order.
//!
//! The qwen-asr package (0.0.6, `Qwen3ASRModel.transcribe`) is the reference: its prompt (the system turn kept
//! when the context is empty; vLLM leaves it out), its `parse_asr_output` and greedy decoding up to 4,096 tokens.
//! It splits audio past 1,200 s into chunks; the route refuses it instead, and takes a shorter clip whole, unless
//! `serve --max-audio-clip-s` is given: then a clip past it is cut into pieces as vLLM cuts it and the pieces
//! run at once (`transcribePieces`, spec 142 WS-06).

const std = @import("std");
const api = @import("engine_api");
const audio = @import("../../audio/audio.zig");
const cuda = @import("../../cuda/cuda.zig");
const numeric = @import("../../weights/numeric.zig");
const text_tokenizer = @import("../../text/tokenizer.zig");
const draft = @import("draft.zig");
const pieces_file = @import("pieces.zig");
const forward_file = @import("forward.zig");
const model_file = @import("model.zig");
const tower_file = @import("tower.zig");
const transcript = @import("transcript.zig");
const Allocator = std.mem.Allocator;

/// The longest clip taken: the package's `MAX_ASR_INPUT_SECONDS`, which it splits past and the route refuses.
pub const max_seconds = 1200;
/// Reply tokens unless the request names fewer: the package's `max_new_tokens` on vLLM.
pub const max_tokens = 4096;
/// Requests waiting behind the running one before `submit_transcription` answers Busy, unless `--max-queued` says.
pub const most_waiting = 64;

/// Tokens of context that hold the longest clip's audio rows, the prompt's words and a whole reply.
pub fn defaultContext() usize {
    return tower_file.tokens(max_seconds * 100) + prompt_room + max_tokens;
}

pub const Asr = struct {
    gpa: Allocator,
    io: std.Io,
    model: *const model_file.Model,
    tower: *tower_file.Tower,
    tables: *const audio.mel.Tables,
    tok: *const text_tokenizer.Tokenizer,
    state: *forward_file.State,
    buffers: *forward_file.Buffers,
    /// bf16 [the longest clip's tokens, the tower's out_dim]: the tower's embeddings of the running clip.
    features: cuda.Buffer,
    pad: u32,
    context: usize,
    most_waiting: usize = most_waiting,
    /// Set, a clip past its seconds runs in pieces; `features` then holds a chunk more a piece (`pieceRows`).
    pieces: ?Pieces = null,

    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    queue: std.ArrayList(*Job) = .empty,
    running: ?*Job = null,
    closing: bool = false,
    worker: ?std.Thread = null,

    const Job = struct {
        id: api.Id,
        kind: union(enum) { transcription: *const api.TranscriptionRequest, chat: *const api.Request },
        sink: api.Sink,
        stop: std.atomic.Value(bool) = .init(false),
        /// From the mel features to the prompt's logits (the tower's encode and the prompt), for the host's /metrics (`Stats.prefill_seconds`).
        prefill_seconds: ?f64 = null,
        /// A chat reply that failed as sent: the words of its `finished` event.
        failure: []const u8 = "",

        /// `draft.Timing.on_first`: the prompt is in, the first token next; the host stamps its first token's time.
        fn prefilled(ctx: *anyopaque) void {
            const job: *Job = @ptrCast(@alignCast(ctx));
            const event: api.Event = .{ .prefilled = 0 };
            job.sink.event(job.sink.ctx, job.id, &event);
        }
    };

    /// Starts the engine's thread; `close` ends it.
    pub fn start(self: *Asr) !void {
        self.worker = try std.Thread.spawn(.{}, work, .{self});
    }

    /// Refuses what is submitted from now, cancels what waits and runs, and joins the thread once all have ended.
    pub fn close(self: *Asr) void {
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
        self.queue.deinit(self.gpa);
    }

    pub fn engine(self: *Asr) api.Engine {
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
        const self: *Asr = @ptrCast(@alignCast(ctx));
        return .{
            .name = "zspark",
            .choice_policy = .single,
            .lanes = 1,
            .context_window = @intCast(self.context),
            // vLLM's sampling is the client's; the package's (`generation_config.json`) and the transcription
            // route's is greedy, the one this engine runs
            .greedy_only = true,
            // TensorFold 1.2.5's `/v1/models` publishes this flag: a speech model writes no reasoning block
            .reasoning = false,
            .transcription = .{ .max_seconds = max_seconds, .max_tokens = max_tokens },
            .audio = .{
                .ctx = self,
                .prepare = prepare,
                .explain = explain,
                .audio_token = self.pad,
                // vLLM's `--limit-mm-per-prompt` default for a modality the model leaves unbounded
                .max_audios = 999,
                .max_seconds = max_seconds,
                // a request's clips run through the tower into `features` one after another
                .max_tokens = @intCast(self.features.view.size[0]),
                // VLLM_MAX_AUDIO_CLIP_FILESIZE_MB
                .max_bytes = 25 << 20,
            },
        };
    }

    /// A chat part's bytes decoded and resampled as the transcription route's file (`audio.clip`).
    fn prepare(_: *anyopaque, a: Allocator, bytes: []const u8) anyerror!api.PreparedAudio {
        const clip = try audio.clip(a, bytes, max_seconds);
        const frames = audio.mel.frames(clip.samples.len);
        if (frames == 0) return error.TooShort;
        return .{ .samples = clip.samples, .seconds = clip.seconds, .tokens = @intCast(audio.mel.tokens(frames)) };
    }

    fn explain(err: anyerror) ?[]const u8 {
        return switch (err) {
            error.TooShort => "The audio is shorter than one mel frame (10 ms).",
            error.Unsupported => audio.message(error.Unsupported),
            error.Invalid => audio.message(error.Invalid),
            error.TooLong => audio.message(error.TooLong),
            else => null,
        };
    }

    /// A chat or completion: its prompt as the host rendered it, an audio part's placeholder made its rows.
    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const self: *Asr = @ptrCast(@alignCast(ctx));
        return self.queued(.{ .id = id, .kind = .{ .chat = request }, .sink = sink });
    }

    fn submitTranscription(ctx: *anyopaque, id: api.Id, request: *const api.TranscriptionRequest, sink: api.Sink) api.TranscriptionSubmitError!void {
        const self: *Asr = @ptrCast(@alignCast(ctx));
        return self.queued(.{ .id = id, .kind = .{ .transcription = request }, .sink = sink });
    }

    fn queued(self: *Asr, one: Job) error{ Closed, Busy }!void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.closing) return error.Closed;
        if (self.queue.items.len >= self.most_waiting) return error.Busy;
        const job = self.gpa.create(Job) catch return error.Busy;
        job.* = one;
        self.queue.append(self.gpa, job) catch {
            self.gpa.destroy(job);
            return error.Busy;
        };
        self.cond.signal(self.io);
    }

    fn cancel(ctx: *anyopaque, id: api.Id) void {
        const self: *Asr = @ptrCast(@alignCast(ctx));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        if (self.running) |job| if (job.id == id) job.stop.store(true, .release);
        for (self.queue.items) |job| if (job.id == id) job.stop.store(true, .release);
    }

    fn status(ctx: *anyopaque, out: *api.Status, _: []u32) void {
        const self: *Asr = @ptrCast(@alignCast(ctx));
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        out.* = .{ .running = @intFromBool(self.running != null), .waiting = @intCast(self.queue.items.len) };
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return .{ .active = cuda.account.bytes };
    }

    fn work(self: *Asr) void {
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
            } else if (switch (job.kind) {
                .transcription => self.transcribe(arena.allocator(), job),
                .chat => |r| self.chat(arena.allocator(), job, r),
            }) |r| {
                reason = if (job.stop.load(.acquire)) .cancelled else r;
                if (reason == .failed) message = job.failure;
            } else |err| {
                const what = if (job.kind == .chat) "reply" else "transcription";
                reason = .failed;
                message = std.fmt.allocPrint(arena.allocator(), "{s} failed: {t}", .{ what, err }) catch "failed";
                std.log.err("{s} {d}: {t}", .{ what, job.id, err });
            }
            {
                self.mutex.lockUncancelable(self.io);
                defer self.mutex.unlock(self.io);
                self.running = null;
            }
            const event: api.Event = .{ .finished = .{ .reason = reason, .message = message, .stats = .{ .prefill_seconds = job.prefill_seconds } } };
            job.sink.event(job.sink.ctx, job.id, &event);
            self.gpa.destroy(job);
        }
    }

    fn refuse(r: *const api.TranscriptionRequest, message: []const u8, param: []const u8) Allocator.Error!api.Reason {
        r.result.refused = try r.gpa.dupe(u8, message);
        r.result.param = try r.gpa.dupe(u8, param);
        return .stop;
    }

    /// One request, its answer in `job.kind.transcription.result`: a refusal (`refused`) or the reply's text.
    fn transcribe(self: *Asr, a: Allocator, job: *Job) !api.Reason {
        const r = job.kind.transcription;
        const forced = transcript.language(r.language) catch
            return refuse(r, try transcript.refusal(a, r.language.?), "language");
        const clip = audio.clip(a, r.audio, max_seconds) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(r, audio.message(err), "file"),
        };
        r.result.seconds = clip.seconds;
        const frames = audio.mel.frames(clip.samples.len);
        if (frames == 0) return refuse(r, "The audio is shorter than one mel frame (10 ms).", "file");
        const started = cuda.monotonic(); // the prefill counts the tower's encode, as vLLM's prefill counts its encoder

        const context = try transcript.sanitize(a, r.prompt);
        const words = try transcript.prompt(a, context, forced);
        var ids: std.ArrayList(u32) = .empty;
        self.tok.encode(a, words, &ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return refuse(r, "prompt must be valid UTF-8 text", "prompt"),
        };
        const at = std.mem.indexOfScalar(u32, ids.items, self.pad) orelse return error.NoAudioPad;
        if (self.pieces) |p| if (clip.samples.len > p.seconds * audio.rate)
            return self.transcribePieces(a, job, clip.samples, forced, ids.items, at, started);

        const n = audio.mel.tokens(frames);
        try self.encode(a, clip.samples, frames, self.features.rows(n));
        const length = ids.items.len - 1 + n;
        if (length + 1 > self.context)
            return refuse(r, try std.fmt.allocPrint(a, "The audio and the prompt take {d} tokens, past the context of {d}", .{ length, self.context }), "prompt");
        const built = try expand(a, ids.items, at, n, self.pad);
        const prompt = built.prompt;
        const rows = built.rows;

        try self.state.reset();
        defer self.state.audio = null;
        self.state.audio = .{ .features = self.features.rows(n), .rows = rows };
        const limit = @min(r.max_tokens, self.context - length);
        const logits = try a.alloc(u16, self.model.config.vocab);
        var reply: std.ArrayList(u32) = .empty;
        var timing: draft.Timing = .{ .stop = &job.stop, .on_first = .{ .ctx = job, .call = Job.prefilled } };
        try draft.greedy(a, self.model.*, self.state, self.buffers, prompt, limit, logits, &reply, &timing);
        if (timing.first > started) job.prefill_seconds = @as(f64, @floatFromInt(timing.first - started)) / std.time.ns_per_s;

        var raw: std.ArrayList(u8) = .empty;
        try self.tok.decode(a, reply.items, .{ .skip_special = true }, &raw);
        const parsed = try transcript.parse(a, raw.items, forced);
        r.result.text = try r.gpa.dupe(u8, parsed.text);
        const named = if (forced) |f| f else parsed.language;
        if (named.len > 0) r.result.language = try r.gpa.dupe(u8, named);
        r.result.prompt_tokens = @intCast(length);
        r.result.tokens = @intCast(reply.items.len);
        const ended = reply.items.len > 0 and self.model.config.isEos(reply.items[reply.items.len - 1]);
        return if (ended or reply.items.len < limit) .stop else .length;
    }

    /// A chat or a completion (spec 150), as vLLM answers one for this model: each audio part's clip through the
    /// tower into the next rows of `features`, its placeholder rows of the host's prompt taking them, then the
    /// serial greedy reply, each token given to the host as it is chosen, the end token last (as
    /// `server/served.zig`'s `Live.closing` gives it). The reply is the model's text as it is (`language X<asr_text>
    /// ...`), not read as the transcription route reads it. No clip is cut: vLLM's `--max-audio-clip-s` is the
    /// transcription route's alone.
    fn chat(self: *Asr, a: Allocator, job: *Job, r: *const api.Request) !api.Reason {
        if (r.prompt.len == 0 or r.prompt.len >= self.context) {
            job.failure = try std.fmt.allocPrint(a, "This server's maximum context length is {d} tokens; the prompt is {d}", .{ self.context, r.prompt.len });
            return .failed;
        }
        const started = cuda.monotonic(); // the prefill counts the tower's encode, as the transcription's does
        var rows: std.ArrayList(u32) = .empty;
        var offset: usize = 0;
        for (r.audios) |clip| {
            const frames = audio.mel.frames(clip.samples.len);
            std.debug.assert(audio.mel.tokens(frames) == clip.tokens);
            try self.encode(a, clip.samples, frames, self.features.span(offset, clip.tokens));
            for (0..clip.tokens) |i| try rows.append(a, @intCast(clip.at + i));
            offset += clip.tokens;
        }
        const prompt = try a.alloc(i32, r.prompt.len);
        for (prompt, r.prompt) |*p, id| p.* = @intCast(id);

        try self.state.reset();
        defer self.state.audio = null;
        if (offset > 0) self.state.audio = .{ .features = self.features.rows(offset), .rows = rows.items };
        const limit = @min(r.max_tokens, self.context - prompt.len);
        if (limit == 0) return .length;
        const logits = try a.alloc(u16, self.model.config.vocab);
        var reply: Reply = .{ .job = job, .a = a, .request = r };
        var out: std.ArrayList(u32) = .empty;
        var timing: draft.Timing = .{
            .stop = &job.stop,
            .on_first = .{ .ctx = job, .call = Job.prefilled },
            .on_token = .{ .ctx = &reply, .call = Reply.give },
            .eos = r.eos,
        };
        try draft.greedy(a, self.model.*, self.state, self.buffers, prompt, limit, logits, &out, &timing);
        if (timing.first > started) job.prefill_seconds = @as(f64, @floatFromInt(timing.first - started)) / std.time.ns_per_s;
        if (reply.stopped) return .stop;
        const n = out.items.len;
        return if (n > 0 and std.mem.indexOfScalar(u32, r.eos, out.items[n - 1]) != null) .stop else .length;
    }

    /// A chat reply's tokens to the host, and its stop strings over them (`server/served.zig`'s `Live.give`).
    const Reply = struct {
        job: *Job,
        a: Allocator,
        request: *const api.Request,
        emitted: std.ArrayList(u32) = .empty,
        stopped: bool = false,

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

    /// `samples`' mel features through the tower into `out`, the tokens of `frames` frames.
    fn encode(self: *Asr, a: Allocator, samples: []const f32, frames: usize, out: cuda.View) !void {
        const mel = try audio.mel.features(a, self.tables, samples);
        const bits = try a.alloc(u16, mel.len);
        for (bits, mel) |*b, x| b.* = numeric.bf16FromF32(x);
        try self.tower.encode(a, bits, frames, out);
    }

    /// A clip past `--max-audio-clip-s` (spec 142 WS-06): cut where vLLM cuts it (`audio/split.zig`), each piece
    /// transcribed as a request of its own with the request's prompt and language, the pieces at once
    /// (`pieces.zig`), their transcripts joined as vLLM joins them (`transcript.join`). A piece's reply is read
    /// as the package reads one (`transcript.parse`: stripped, repetitions folded) where vLLM takes the text
    /// after `<asr_text>` as it is. The usage is the request's: the pieces' prompts and replies summed.
    fn transcribePieces(self: *Asr, a: Allocator, job: *Job, samples: []const f32, forced: ?[]const u8, ids: []const u32, at: usize, started: u64) !api.Reason {
        const r = job.kind.transcription;
        const p = self.pieces.?;
        const capacity = p.lanes.capacity();
        const ends = try audio.split.cuts(a, samples, audio.rate, @floatFromInt(p.seconds), audio.split.overlap_seconds, audio.split.energy_window);
        const list = try a.alloc(pieces_file.Piece, ends.len);
        var begin: usize = 0;
        var offset: usize = 0;
        var prompt_tokens: usize = 0;
        for (list, ends) |*piece, end| {
            const part = samples[begin..end];
            begin = end;
            const frames = audio.mel.frames(part.len);
            const n = audio.mel.tokens(frames);
            const length = ids.len - 1 + n;
            if (length + 1 > capacity)
                return refuse(r, try std.fmt.allocPrint(a, "A piece's audio and the prompt take {d} tokens, past a piece's {d}", .{ length, capacity }), "prompt");
            const features = self.features.span(offset, n);
            try self.encode(a, part, frames, features);
            const built = try expand(a, ids, at, n, self.pad);
            piece.* = .{ .prompt = built.prompt, .audio = .{ .features = features, .rows = built.rows }, .limit = @min(r.max_tokens, capacity - length) };
            offset += n;
            prompt_tokens += length;
        }
        var timing: draft.Timing = .{ .stop = &job.stop, .on_first = .{ .ctx = job, .call = Job.prefilled } };
        try pieces_file.greedy(a, self.model.*, p.lanes.*, self.buffers, list, &timing);
        if (timing.first > started) job.prefill_seconds = @as(f64, @floatFromInt(timing.first - started)) / std.time.ns_per_s;

        const texts = try a.alloc([]const u8, list.len);
        var named: []const u8 = forced orelse "";
        var tokens: usize = 0;
        var reason: api.Reason = .stop;
        for (texts, list) |*text, piece| {
            var raw: std.ArrayList(u8) = .empty;
            try self.tok.decode(a, piece.reply.items, .{ .skip_special = true }, &raw);
            const parsed = try transcript.parse(a, raw.items, forced);
            text.* = parsed.text;
            if (named.len == 0) named = parsed.language;
            tokens += piece.reply.items.len;
            const n = piece.reply.items.len;
            if (n == piece.limit and !self.model.config.isEos(piece.reply.items[n - 1])) reason = .length;
        }
        r.result.text = try transcript.join(r.gpa, texts, forced);
        if (named.len > 0) r.result.language = try r.gpa.dupe(u8, named);
        r.result.prompt_tokens = @intCast(prompt_tokens);
        r.result.tokens = @intCast(tokens);
        return reason;
    }
};

/// The prompt's ids with its one `<|audio_pad|>` (at `at`) made `n` rows, and those rows' positions.
fn expand(a: Allocator, ids: []const u32, at: usize, n: usize, pad: u32) Allocator.Error!struct { prompt: []i32, rows: []u32 } {
    const prompt = try a.alloc(i32, ids.len - 1 + n);
    const rows = try a.alloc(u32, n);
    for (ids[0..at], 0..) |id, i| prompt[i] = @intCast(id);
    for (rows, 0..) |*row, i| {
        row.* = @intCast(at + i);
        prompt[at + i] = @intCast(pad);
    }
    for (ids[at + 1 ..], 0..) |id, i| prompt[at + n + i] = @intCast(id);
    return .{ .prompt = prompt, .rows = rows };
}

/// A long clip's pieces (`--max-audio-clip-s`): the longest a clip runs whole, in seconds, and the lanes its
/// pieces decode in.
pub const Pieces = struct {
    seconds: usize,
    lanes: *pieces_file.Lanes,
};

/// Positions a lane holds: a piece's audio rows, the prompt's words (`prompt_room`) and `reply_room` reply
/// tokens, rounded up to a chunk of keys.
pub fn laneCapacity(seconds: usize) usize {
    const want = tower_file.tokens(seconds * 100) + prompt_room + reply_room;
    return std.mem.alignForward(usize, want, forward_file.chunk);
}
/// What `defaultContext` keeps for the prompt's words beside the audio.
pub const prompt_room = 512;
/// A piece's reply tokens a lane holds at least: a 30 s piece's transcript is 100 to 200 tokens (WS-05's set).
pub const reply_room = 1024;
/// Pieces decoded at once unless `--parallel` says.
pub const default_lanes = 16;

/// The feature rows the longest clip's pieces take beyond its whole tokens: a cut lands at least `seconds` less
/// the overlap after the last, and a piece's last chunk, partial, takes up to a chunk's tokens more.
pub fn pieceRows(seconds: usize) usize {
    const most = max_seconds / (seconds - audio.split.overlap_seconds) + 1;
    return most * tower_file.chunk_tokens;
}
