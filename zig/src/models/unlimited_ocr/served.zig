//! Unlimited-OCR-2 behind TensorFold's `engine_api.Engine` (spec 147): what `server/served.zig`'s engine asks of
//! a model, for `zspark serve` (`commands/serve.zig`). One reply at a time, from an empty state each: no prefix
//! is shared between requests (vLLM's recipe serves it with `--no-enable-prefix-caching`), and a reply is serial
//! steps (`draft.zig`: no drafter), one captured graph a step (`forward.Graphs`).
//!
//! What a request gives before its reply is `requests.zig`'s, on the host: the template (the checkpoint has
//! none), the images (`requests.Offer`: the host hands a part's bytes to `prepare`, which cuts them as the
//! checkpoint's HF code does and says the rows the picture takes; the host writes that many `<image>` ids where
//! the template wrote one) and the rules of each row (`requests.Rules`). `reply` runs a picture's views through
//! the tower (`vision.Tower.encode`) into the request's own features and lays them over its rows
//! (`State.image`). The view's values reach `reply` as f32 (`PreparedImage.pixels`): bf16 widened, so they narrow
//! back exactly. A request's several pages are encoded together into one features buffer, laid a run a page
//! (`forward.Image.runs`), and its prompt's page coordinates come from `State.paged`.
//!
//! A reply follows the request's no-repeat n-gram rule (`Request.no_repeat`, `decode/ngram.zig`) and the
//! PageCountGuard (`Request.max_page_tokens`, `decode/page_guard.zig`), as `requests.Rules` says. Every row is
//! chosen after them (and after the request's constraint, where it asks one) by the request's sampling, greedy
//! unless asked: the checkpoint has no `generation_config.json`.
//!
//! A document of several pages whose prompt and replies outgrow the state (or whose request names
//! `doc_split_options.max_batch_pages`, or the server `ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW`) runs in parts
//! (`parts.zig`, SGLang's windows; spec 147 WS-08), unless it asks a constraint or logprobs. Else the reply is one
//! window, its tokens at most the state's room.

const std = @import("std");
const Allocator = std.mem.Allocator;
const api = @import("engine_api");
const request_file = @import("../../server/request.zig");
const sched = @import("../../server/scheduler.zig");
const shared = @import("../../server/served.zig");
const decoding = @import("../../decode/draft.zig");
const masks_file = @import("../../decode/masks.zig");
const sampling = @import("../../decode/sampling.zig");
const cuda = @import("../../cuda/cuda.zig");
const draft = @import("draft.zig");
const forward_file = @import("forward.zig");
const documents = @import("documents.zig");
const parts_file = @import("parts.zig");
const requests = @import("requests.zig");
const vision = @import("vision.zig");

/// What `server/served.zig` asks of a model (`shared.Served`'s doc comment).
const unlimited_ocr = struct {
    /// The model, its tower and the one sequence every reply runs in turn; the command's.
    pub const Behind = struct {
        model: *const forward_file.Model,
        tower: *vision.Tower,
        state: *forward_file.State,
        buffers: *forward_file.Buffers,
        sampler: *sampling.Sampler,
        /// A row's logits, bf16 bits over the vocabulary.
        logits: []u16,
        /// A long document's most pages a part where the request names none
        /// (`ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW`); null: the pages the state holds.
        page_window: ?u32 = null,
    };

    pub fn counts(_: *const Behind) ?sched.Counts {
        return null;
    }

    pub fn closed(_: *Behind) void {}

    pub fn reply(live: *Live, arena: Allocator, ask: shared.Ask, said: *request_file.Diagnostic) !shared.Ended {
        const served = live.served;
        const request = live.request;
        const behind = &served.behind;
        const m = behind.model.*;
        const state = behind.state;
        const b = behind.buffers;
        const prompt = ask.prompt;
        // `serve` asks no warm pass
        if (ask.warm) return .{ .reason = .length };
        if (!live.enters()) return .{ .reason = .cancelled };
        defer live.leaves();
        live.begins();
        const began_ns = cuda.monotonic();
        const constraint = try live.constrained(arena, false, said);
        var held: ?masks_file.Masks = null;
        if (constraint) |c| held = try masks_file.Masks.init(c, try arena.alloc(u32, c.words), null);
        var rules: requests.Rules = .of(request.*, prompt, m.config.eos);

        // the images' rows: the tower over their views, into this request's features, a run a picture
        state.reset();
        var features: ?cuda.Buffer = null;
        defer if (features) |*f| f.deinit();
        if (ask.images.len > 0) {
            const pictures = try arena.alloc(vision.Image, ask.images.len);
            const runs = try arena.alloc(forward_file.Image.Run, ask.images.len);
            var rows: usize = 0;
            for (ask.images, pictures, runs) |given, *picture, *run| {
                const tiles: usize = if (given.gw == 1 and given.gh == 1) 0 else @as(usize, given.gw) * given.gh;
                if (given.pixels.len != requests.global_values + tiles * requests.tile_values or given.at + given.tokens > prompt.len) {
                    said.set("the image's views do not match its grid", .{});
                    return error.Failed;
                }
                const views = try arena.alloc(u16, given.pixels.len);
                for (views, given.pixels) |*v, p| v.* = @intCast(@as(u32, @bitCast(p)) >> 16);
                picture.* = .{ .global = views[0..requests.global_values], .crops = views[requests.global_values..], .width = given.gw, .height = given.gh };
                if (picture.rows() != given.tokens) {
                    said.set("the image takes {d} rows, the prompt {d}", .{ picture.rows(), given.tokens });
                    return error.Failed;
                }
                run.* = .{ .first = given.at, .rows = given.tokens };
                rows += given.tokens;
            }
            features = try .init(.bf16, &.{ rows, vision.lm_hidden });
            try behind.tower.encode(served.gpa, pictures, features.?.view);
            state.image = .{ .features = features.?, .runs = runs };
        }
        // the long documents (spec 147 WS-08): parts of pages where the state does not hold the whole
        if (ask.images.len > 1) {
            if ((request.doc_split.overlap_pages orelse 0) > 0) {
                said.set("doc_split_options.overlap_pages above 0 is not served", .{});
                return error.BadRequest;
            }
            if (constraint == null and ask.logprobs == null) {
                if (try document(arena, behind, request.*, prompt, features.?, ask)) |planned|
                    return documentReply(live, arena, planned, ask, began_ns);
            }
        }
        if (prompt.len >= state.capacity) {
            said.set("the prompt is {d} tokens; the model's state holds {d}", .{ prompt.len, state.capacity });
            return error.BadRequest;
        }
        const max_tokens = @min(ask.max_tokens, state.capacity - prompt.len);
        // Unlimited-OCR-2's page coordinates (spec 147), from the prompt whole
        state.paged(prompt) catch |err| switch (err) {
            error.Unpaged => {
                said.set("Unlimited-OCR-2 reads a prompt with an image and a <PAGE> marker up to its last image token", .{});
                return error.Failed;
            },
            else => |e| return e,
        };

        // the prompt in windows of the buffers' rows, the last row's logits kept
        while (state.pos < prompt.len) {
            if (live.cancelled()) return .{ .reason = .cancelled };
            try decoding.promptWindow(draft.namespace, m, state, b, prompt);
        }
        // the steps as one captured graph (`forward.Graphs`, as `draft.greedy`): the same launches, the same bits
        var graphs: forward_file.Graphs = try .init(served.gpa);
        defer graphs.deinit();
        try graphs.prepare(m, state, b, 1, .last);
        live.prefilled();
        const prefill_ns = cuda.monotonic();

        // the reply, a step a token; `seen` is what the rules read
        var seen: std.ArrayList(u32) = .empty;
        try seen.ensureTotalCapacity(arena, prompt.len + max_tokens);
        for (prompt) |id| seen.appendAssumeCapacity(@intCast(id));
        var made: usize = 0;
        var finish: request_file.Finish = .length;
        var end: ?i32 = null;
        while (made < max_tokens) {
            const logits = behind.logits;
            try forward_file.logitsRow(b.*, 0, logits);
            rules.apply(seen.items, prompt.len, logits);
            if (held) |*masks| {
                _ = try masks.window(&.{});
                try masks.apply(0, logits);
            }
            const token = try behind.sampler.choose(logits, state.pos, ask.settings);
            if (held) |*masks| try masks.commit(1, token);
            made += 1;
            if (std.mem.indexOfScalar(u32, request.eos, @intCast(token)) != null) {
                finish = .stop;
                end = token;
                break;
            }
            if (live.give(token) or live.cancelled()) {
                finish = .cut;
                break;
            }
            if (made == max_tokens) break;
            seen.appendAssumeCapacity(@intCast(token));
            const step = [_]i32{token};
            try graphs.forwardScoring(m, state, b, &step, .last);
            try forward_file.commit(m, state, b.*, &step, 1);
        }
        live.closing(finish, end);
        const done_ns = cuda.monotonic();
        const decode_s = @as(f64, @floatFromInt(done_ns -| prefill_ns)) / 1e9;
        var image_rows: usize = 0;
        for (ask.images) |given| image_rows += given.tokens;
        std.log.info("prompt {d} tokens ({d} image rows, {d} images); reply {d} tokens, {d:.1} tokens a second", .{ prompt.len, image_rows, ask.images.len, made, if (decode_s > 0) @as(f64, @floatFromInt(made)) / decode_s else 0 });
        return .{ .reason = live.reasonOf(finish), .stats = .{ .rounds = made, .prefill_seconds = @as(f64, @floatFromInt(prefill_ns -| began_ns)) / 1e9 } };
    }
};

/// A long document's parts, where it has more than one.
const Planned = struct { doc: parts_file.Document, parts: []const documents.Part };

/// The parts of a v2 prompt of several pages (SGLang's `plan_batches`): `request.doc_split`'s most pages a part,
/// else the server's, else the pages the state holds with their replies; null where it stays one part (or its
/// pages are not the images'). The rows a page takes are SGLang's estimate of its image mode's (tiled: gundam).
fn document(arena: Allocator, behind: *unlimited_ocr.Behind, request: api.Request, prompt: []const i32, features: cuda.Buffer, ask: shared.Ask) !?Planned {
    const rows = try arena.alloc(usize, ask.images.len);
    var tiled = false;
    for (ask.images, rows) |given, *r| {
        r.* = given.tokens;
        if (given.gw != 1 or given.gh != 1) tiled = true;
    }
    const doc = try parts_file.Document.of(arena, prompt, features, rows) orelse return null;
    const per_page = documents.tokensPerPage(if (tiled) "gundam" else "base") + documents.reply_rows_per_page;
    const max_pages = request.doc_split.max_batch_pages orelse behind.page_window;
    const planned = try documents.plan(arena, @intCast(rows.len), per_page, behind.state.capacity, ask.max_tokens, max_pages) orelse return null;
    return .{ .doc = doc, .parts = planned };
}

fn documentReply(live: *Live, arena: Allocator, planned: Planned, ask: shared.Ask, began_ns: u64) !shared.Ended {
    const behind = &live.served.behind;
    var sink: Sink = .{ .live = live, .ask = ask };
    const run: parts_file.Run = .{ .model = behind.model.*, .state = behind.state, .buffers = behind.buffers, .logits = behind.logits };
    const outcome = parts_file.reply(live.served.gpa, arena, run, planned.doc, planned.parts, live.request.*, ask.max_tokens, sink.sink()) catch |err| switch (err) {
        error.PartTooLong => return .{ .reason = .length },
        else => |e| return e,
    };
    const finish: request_file.Finish = switch (outcome.finish) {
        .stop => .stop,
        .length => .length,
        .cut => .cut,
        .cancelled => return .{ .reason = .cancelled },
    };
    live.closing(finish, if (outcome.end) |e| @intCast(e) else null);
    const seconds = @as(f64, @floatFromInt(cuda.monotonic() -| began_ns)) / 1e9;
    std.log.info("prompt {d} tokens, {d} pages in {d} parts ({d} taken over); reply {d} tokens in {d:.1} s", .{ ask.prompt.len, ask.images.len, outcome.parts, outcome.reused, outcome.made, seconds });
    return .{ .reason = live.reasonOf(finish), .stats = .{ .rounds = outcome.made, .prefill_seconds = @as(f64, @floatFromInt(outcome.prefilled_ns -| began_ns)) / 1e9 } };
}

/// `parts.Sink` over the request's sampling and its client.
const Sink = struct {
    live: *Live,
    ask: shared.Ask,

    fn sink(self: *Sink) parts_file.Sink {
        return .{ .ctx = self, .choose = choose, .give = give, .cancelled = cancelled, .prefilled = prefilled };
    }
    fn choose(ctx: *anyopaque, logits: []u16, position: usize) anyerror!i32 {
        const self: *Sink = @ptrCast(@alignCast(ctx));
        return self.live.served.behind.sampler.choose(logits, position, self.ask.settings);
    }
    fn give(ctx: *anyopaque, token: i32) bool {
        const self: *Sink = @ptrCast(@alignCast(ctx));
        return self.live.give(token) or self.live.cancelled();
    }
    fn cancelled(ctx: *anyopaque) bool {
        const self: *Sink = @ptrCast(@alignCast(ctx));
        return self.live.cancelled();
    }
    fn prefilled(ctx: *anyopaque) void {
        const self: *Sink = @ptrCast(@alignCast(ctx));
        self.live.prefilled();
    }
};

pub const Served = shared.Served(unlimited_ocr);
const Live = Served.Live;
