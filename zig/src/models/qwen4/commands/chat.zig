//! `zspark chat`: one request answered as text, from messages to streamed text.

const std = @import("std");
const qwen4_layout = @import("../layout.zig");
const bf16 = @import("../../../weights/bf16.zig");
const qwen4_attention = @import("../blocks/attention.zig");
const qwen4_model = @import("../model.zig");
const held_file = @import("../ranks_forward.zig");
const qwen4_session = @import("../session/session.zig");
const session_chat = @import("../session/chat.zig");
const pair_file = @import("../session/pair.zig");
const sampling = @import("../../../decode/sampling.zig");
const masks_file = @import("../../../decode/masks.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const server_request = @import("../../../server/request.zig");
const text_frontend = @import("../../../text/frontend.zig");
const text_grammar = @import("../../../text/grammar.zig");
const json_schema = @import("../../../text/json_schema.zig");
const text_messages = @import("../../../text/messages.zig");
const tokenizer = @import("../../../text/tokenizer.zig");
const cli = @import("../../../cli.zig");
const command = @import("../../../server/command.zig");
const common = @import("common.zig");

pub const chat_usage =
    \\usage: zspark chat DIR [OPTION..] [MESSAGE]
    \\
    \\Renders the request through the checkpoint's chat template, decodes a reply and prints it as it comes.
    \\MESSAGE is one user message; --messages gives a whole conversation instead.
    \\
    \\  --system TEXT      a system message before MESSAGE
    \\  --messages FILE    the request's messages, a JSON array as the OpenAI API takes it
    \\  --tools FILE       the request's tools, a JSON array
    \\  --max-tokens N     the most reply tokens (default 512)
    \\  --schema FILE      the reply is a JSON document of this JSON Schema: every token is masked to it
    \\  --json             the reply is a JSON object (--schema with {"type": "object"})
    \\  --no-thinking      close the think block in the prompt
    \\  --effort NAME      none, minimal, low, medium, high, xhigh or max; the template hears the nearest it names
    \\  --seed N           the sampling key (default: derived from the prompt, so a prompt repeats its reply)
    \\  --temperature X    0 decodes greedily (default: the checkpoint's generation config, else 0)
    \\  --top-k N, --top-p X, --min-p X   (default: the checkpoint's generation config)
    \\  --rank N --at ADDR:PORT   one rank of two (or the words rank=N at=ADDR:PORT); rank 1 listens
    \\  --render           print the prompt's text and ids, and stop before loading the model
    \\  --eager            launch each kernel from the host instead of replaying the step's captured graph
    \\  --lazy-graphs      capture a graph when a step first needs it, not while the model loads
    \\  --serial           one token a step, without the MTP head's drafts (the reply is the same either way)
    \\  --depth N          the most drafts a step verifies, 0 to 15 (default 15)
    \\  --no-copies        draft with the MTP head alone, not with what followed an earlier occurrence of the
    \\                     reply's tail in the prompt or the reply (the reply is the same either way)
    \\  --reply-stride N   keep a state each N tokens of the reply, as `serve` does (default 256; 0: none)
    \\  --kv-dtype T       how the attention caches hold keys and values: bf16 (default), fp8, or fp8-nvfp4
    \\                     (FP8 keys, NVFP4 values); the drafting head's own cache stays bf16
    \\  --max-num-batched-tokens N   the rows a prompt's window runs, 2048 to 4096 (default 4096, one rank
    \\                     or two): 4096 reads a long prompt's experts half as often for twice the windows' buffers
    \\
    \\The reply goes to stdout; the reasoning before it, and the timings, go to stderr.
    \\
;

const ChatOptions = struct {
    system: ?[]const u8 = null,
    messages: ?[]const u8 = null,
    tools: ?[]const u8 = null,
    message: ?[]const u8 = null,
    schema: ?[]const u8 = null,
    json: bool = false,
    max_tokens: usize = 512,
    thinking: bool = true,
    effort: ?text_messages.Effort = null,
    seed: ?i64 = null,
    temperature: ?f64 = null,
    top_k: ?u32 = null,
    top_p: ?f64 = null,
    min_p: ?f64 = null,
    render: bool = false,
    eager: bool = false,
    lazy_graphs: bool = false,
    serial: bool = false,
    no_copies: bool = false,
    depth: usize = common.draft_depth,
    reply_stride: usize = (qwen4_session.Options{}).reply_stride,
    /// Null: not asked, and the storage is one stream's (`layout.storageFor`).
    kv: ?qwen4_attention.Storage = null,
    /// A prompt's window and the model's rows (`common.promptRows`, `common.pairRows` on two ranks); null: the checkpoint's.
    max_num_batched_tokens: ?usize = null,

    const flags = common.decode_flags ++ [_]cli.Flag{
        .{ .name = "--no-thinking", .field = "thinking", .on = false },
        .{ .name = "--render", .field = "render" },
        .{ .name = "--json", .field = "json" },
        .{ .name = "--eager", .field = "eager" },
        .{ .name = "--system", .field = "system" },
        .{ .name = "--messages", .field = "messages" },
        .{ .name = "--tools", .field = "tools" },
        .{ .name = "--schema", .field = "schema" },
        .{ .name = "--effort", .field = "effort" },
        .{ .name = "--max-tokens", .field = "max_tokens", .min = 1 },
        .{ .name = "--seed", .field = "seed" },
        .{ .name = "--top-k", .field = "top_k" },
        .{ .name = "--temperature", .field = "temperature" },
        .{ .name = "--top-p", .field = "top_p" },
        .{ .name = "--min-p", .field = "min_p" },
    };

    /// Null, with the reason logged, when the arguments do not parse.
    fn parse(args: []const []const u8) ?ChatOptions {
        const o = cli.parse(ChatOptions, &flags, .{ .positional = .{ .field = "message", .again = "more than one MESSAGE; quote it, or use --messages" }, .quoted = true }, args) orelse return null;
        if ((o.message == null) == (o.messages == null)) {
            std.log.err("give either MESSAGE or --messages FILE", .{});
            return null;
        }
        if (o.schema != null and o.json) {
            std.log.err("give either --schema FILE or --json", .{});
            return null;
        }
        if (o.messages != null and o.system != null) {
            std.log.err("--system goes with MESSAGE; put the system message in the --messages file", .{});
            return null;
        }
        return o;
    }
};

/// One request, from messages to streamed text: the chat template, the tokenizer, the forward, keyed sampling.
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const args = argv[1..];
    const arena = init.arena.allocator();
    var place: join_file.Place = .{};
    const words = join_file.rest(arena, args, &place) catch {
        std.log.err("chat: --rank and --at each take a value", .{});
        return 2;
    };
    switch (command.ranked("chat", place, nccl.linked, .flags)) {
        .exit => |code| return code,
        .alone, .paired => {},
    }
    const options = ChatOptions.parse(words) orelse return 2;
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, path, .say)) return 1;
    var frontend = cli.loadFrontend(init.gpa, init.io, opened.dir.?, path, null) orelse return 1;
    defer frontend.deinit(init.gpa);
    const ask: cli.ChatAsk = .{
        .system = options.system,
        .message = options.message,
        .messages_file = options.messages,
        .tools_file = options.tools,
        .thinking = options.thinking,
        .effort = options.effort,
        .render = options.render,
        .seed = options.seed,
        .temperature = options.temperature,
        .top_k = options.top_k,
        .top_p = options.top_p,
        .min_p = options.min_p,
    };
    const made = switch (try cli.chatPrompt(init, stdout, &frontend, ask, "sampling")) {
        .exit => |code| return code,
        .ready => |ready| ready,
    };
    const prompt = made.ids;
    const settings = made.settings;

    if (!try opened.configure(init)) return 1;
    const config = opened.config;
    const capacity = 4096;
    if (prompt.len + options.max_tokens > capacity) {
        std.log.err("the prompt's {d} tokens and {d} reply tokens pass the {d} this command holds", .{ prompt.len, options.max_tokens, capacity });
        return 1;
    }
    // The reply's constraint, compiled before the model is loaded: a schema that is refused costs no load. The
    // engine is the tokenizer's, as wide as the logits, and ends a document with the tokens that end a reply.
    var engine: ?json_schema.Engine = null;
    defer if (engine) |*e| e.deinit(init.gpa);
    var constraint: ?text_grammar.Constraint = null;
    defer if (constraint) |c| c.deinit();
    if (options.schema != null or options.json) {
        const spec: text_grammar.Spec = if (options.schema) |file| .{
            .json_schema = std.Io.Dir.cwd().readFileAlloc(init.io, file, arena, .limited(1 << 24)) catch |err| {
                std.log.err("{s}: {t}", .{ file, err });
                return 1;
            },
        } else .json_object;
        const compile_start: std.Io.Timestamp = .now(init.io, .awake);
        engine = try .init(init.gpa, &frontend.tokenizer, config.vocab, config.eos);
        var schema_diagnostic: text_grammar.Diagnostic = .{};
        constraint = engine.?.interface().compile(init.gpa, spec, &schema_diagnostic) catch |err| {
            std.log.err("{s}: {s}", .{ options.schema orelse "--json", if (schema_diagnostic.len > 0) schema_diagnostic.message() else @errorName(err) });
            return 1;
        };
        std.log.info("the constraint compiled in {d:.1} ms", .{@as(f64, @floatFromInt(compile_start.untilNow(init.io, .awake).toNanoseconds())) / 1e6});
    }

    if (place.complete()) return chatRanks(init, stdout, &opened, &frontend, place, options, prompt, settings, constraint);

    if (!try opened.resolve(init)) return 1;
    const found = opened.layout;
    const load_start: std.Io.Timestamp = .now(init.io, .awake);
    const prompt_rows = common.promptRows(config, options.max_num_batched_tokens) orelse return 2;
    const caches = qwen4_layout.storageFor(config, 1, 1, options.kv);
    var model: qwen4_model.Model = try .loadWith(init.gpa, config, found, try common.loadOptions(arena, config, prompt_rows, !options.serial and options.depth > 0, caches));
    defer model.deinit();
    const session: *qwen4_session.Session = try .create(init.gpa, init.io, model, &frontend.tokenizer, config.eos, .{ .capacity = capacity, .prompt_rows = prompt_rows, .eager = options.eager, .reply_stride = options.reply_stride, .copies = !options.no_copies });
    defer session.destroy();
    if (!options.lazy_graphs) try session.prepare(options.depth, qwen4_session.prepared_buckets);
    const prepared = session.captured();
    const load_seconds = @as(f64, @floatFromInt(load_start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;

    // The reasoning goes to stderr and the reply to stdout.
    var stderr_buffer: [256]u8 = undefined;
    // Streaming, not positional: std.log writes to the same file, and a redirected one must not be overwritten.
    var stderr_file: std.Io.File.Writer = .initStreaming(.stderr(), init.io, &stderr_buffer);
    var terminal: Terminal = .{ .stdout = stdout, .stderr = &stderr_file.interface };
    const reply = try session.reply(prompt, settings, .{ .max_tokens = options.max_tokens, .depth = options.depth, .constraint = constraint }, terminal.sink());
    try terminal.finish();
    std.log.info("loaded in {d:.1} s; prompt {d} tokens in {d:.2} s; reply {d} tokens in {d:.2} s, {d:.1} tok/s; {s}; {s}", .{
        load_seconds,                                                               prompt.len,
        reply.prompt_seconds,                                                       reply.tokens,
        reply.seconds,                                                              @as(f64, @floatFromInt(reply.tokens)) / reply.seconds,
        if (reply.finish == .stop) "ended by the model" else "cut at --max-tokens", if (settings.greedy()) "greedy" else "sampled",
    });
    std.log.info("{d} steps, {d} drafts, {d} of them kept{s}; {d} states kept on the way, {d} MiB held", .{ reply.stats.rounds, reply.stats.drafted, reply.stats.accepted, if (model.mtp == null) " (no MTP head)" else "", reply.taken, session.snapshotBytes() >> 20 });
    if (constraint != null) std.log.info("under the constraint: {d} rows masked in {d:.2} ms, {d:.1} us a row; {d} drafts the constraint's, {d} cut before their window", .{
        reply.masks.masked,                                                                                            @as(f64, @floatFromInt(reply.masks.nanoseconds)) / 1e6,
        @as(f64, @floatFromInt(reply.masks.nanoseconds)) / 1e3 / @as(f64, @floatFromInt(@max(reply.masks.masked, 1))), reply.masks.forced,
        reply.masks.pruned,
    });
    const captured = session.captured();
    if (!options.eager) std.log.info("{d} graphs captured in {d:.0} ms while loading, {d} in {d:.0} ms by the reply", .{ prepared.count, @as(f64, @floatFromInt(prepared.nanoseconds)) / 1e6, captured.count - prepared.count, @as(f64, @floatFromInt(captured.nanoseconds - prepared.nanoseconds)) / 1e6 });
    return 0;
}

const H = held_file.Held(1);

/// One rank of two. Rank 0 samples and prints; rank 1 follows and prints nothing. The MTP head is rank 0's,
/// whole beside its share, and drafts as on one rank (`--serial`, `--depth`, `--no-copies`).
fn chatRanks(init: std.process.Init, stdout: *std.Io.Writer, opened: *common.Opened, frontend: *text_frontend.Frontend, place: join_file.Place, options: ChatOptions, prompt: []const i32, settings: sampling.Sampling, constraint: ?text_grammar.Constraint) !u8 {
    if (!try opened.resolve(init)) return 1;
    const config = opened.config;
    const rank = place.rank.?;
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    // a prompt's windows as one Spark's are asked (spec 111), unset 4,096: the pair's frames carry them (`Lead.widest`)
    const rows = common.pairRows(config, options.max_num_batched_tokens) orelse return 2;
    // not in `alike` where the head is whole: rank 1 loads and runs nothing for it
    const drafts = !options.serial and options.depth > 0;
    const drafting = common.headOnRank(rank, drafts) orelse return 2;
    const caches = qwen4_layout.storageFor(config, 1, 1, options.kv);
    const alike = try std.fmt.allocPrint(arena, "chat rows={d} max={d} t={d} seed={d} eager={} prompt {x}{s}{s}", .{ rows, options.max_tokens, settings.temperature, settings.seed, options.eager, std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(prompt)), common.cachesAlike(caches), common.headAlike(drafts) });
    var joined: join_file.Joined = undefined;
    try joined.open(init.io, gpa, place, try join_file.headers(arena, &opened.checkpoint.?), alike);
    defer joined.close();
    const channel = &joined.channel;

    const load_start: std.Io.Timestamp = .now(init.io, .awake);
    const built: H.Model = .{
        .ranks = .{try qwen4_model.Model.loadWith(gpa, config, opened.layout, .{ .layers = config.layers, .max_rows = rows, .mtp = drafting, .draft_vocab = if (drafting) try qwen4_model.DraftHead.defaultIds(arena) else null, .share = .of(@intCast(rank)), .head_split = qwen4_model.ranks_head == .split, .kv = caches.trunk, .head_kv = caches.head })},
        .first = rank,
        .max_rows = rows,
    };
    const pair = pair_file.Pair.create(gpa, rank, built, joined.comm, channel, rows, 4096, options.eager) catch |err| return channel.failWith(rank, err);
    defer pair.destroy();
    channel.barrier(1) catch |err| return channel.gone(rank, err);
    const load_seconds = @as(f64, @floatFromInt(load_start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;

    if (rank != 0) {
        const ran = pair.follow() catch |err| return channel.failWith(rank, err);
        std.log.info("rank {d}: followed rank 0's {d} operations", .{ rank, ran });
        channel.barrier(std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(pair.tokens.items))) catch |err| return channel.gone(rank, err);
        return 0;
    }

    const open = frontend.tokenizer.addedId("<think>");
    const close = frontend.tokenizer.addedId("</think>");
    const tags: ?server_request.Think = if (open != null and close != null) .{ .open = @intCast(open.?), .close = @intCast(close.?) } else null;
    var mask_buf: []u32 = &.{};
    defer if (mask_buf.len > 0) gpa.free(mask_buf);
    var held: ?masks_file.Masks = null;
    if (constraint) |c| {
        mask_buf = try gpa.alloc(u32, c.words);
        held = try .init(c, mask_buf, if (server_request.thinks(tags, prompt)) tags.?.close else null);
    }
    var decoder: tokenizer.StreamDecoder = try .init(gpa, &frontend.tokenizer, .{ .skip_special = true });
    defer decoder.deinit(gpa);
    var stderr_buffer: [256]u8 = undefined;
    var stderr_file: std.Io.File.Writer = .initStreaming(.stderr(), init.io, &stderr_buffer);
    var terminal: Terminal = .{ .stdout = stdout, .stderr = &stderr_file.interface };
    var printing: Printing = .{
        .speaker = .{ .decoder = &decoder, .close = if (tags) |t| t.close else null, .reasoning = server_request.thinks(tags, prompt) },
        .sink = terminal.sink(),
        .eos = config.eos,
    };
    const start: std.Io.Timestamp = .now(init.io, .awake);
    const reply = pair.reply(prompt, settings, config.eos, .{ .depth = if (pair.head == null) 0 else options.depth, .copies = !options.no_copies }, .{}, if (held) |*m| m else null, options.max_tokens, null, .{ .context = &printing, .take = Printing.take }) catch |err| return channel.failWith(rank, err);
    printing.speaker.end(reply.finish == .cut, printing.sink);
    try terminal.finish();
    const seconds = @as(f64, @floatFromInt(start.untilNow(init.io, .awake).toNanoseconds())) / 1e9;
    pair.finish() catch |err| return channel.failWith(rank, err);
    channel.barrier(std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(pair.tokens.items))) catch |err| return channel.gone(rank, err);
    var drafts_buffer: [96]u8 = undefined;
    const drafts_text = if (pair.head == null) "no MTP head" else try std.fmt.bufPrint(&drafts_buffer, "{d} rounds, {d} of {d} drafts kept", .{ reply.stats.rounds, reply.stats.accepted, reply.stats.drafted });
    std.log.info("rank 0: loaded its share in {d:.1} s; prompt {d} tokens; reply {d} tokens in {d:.2} s, {d:.1} tok/s; {s}; {s}; {s}", .{
        load_seconds,                                                                                                       prompt.len,                                     printing.speaker.count, seconds, @as(f64, @floatFromInt(printing.speaker.count)) / @max(seconds, 1e-9),
        if (reply.finish == .stop) "ended by the model" else if (reply.finish == .length) "cut at --max-tokens" else "cut", if (settings.greedy()) "greedy" else "sampled", drafts_text,
    });
    return 0;
}

const Printing = struct {
    speaker: session_chat.Speaker,
    sink: server_request.Sink,
    eos: []const u32,

    fn take(context: *anyopaque, token: i32) bool {
        const self: *Printing = @ptrCast(@alignCast(context));
        if (self.speaker.counted(token, self.eos) == null) return true;
        return self.speaker.say(token, self.sink) == .stop;
    }
};

/// `chat`'s sink: content to stdout and reasoning to stderr, each piece flushed, a line break after the reasoning.
const Terminal = struct {
    stdout: *std.Io.Writer,
    stderr: *std.Io.Writer,
    last: ?server_request.Kind = null,
    failed: bool = false,

    fn sink(self: *Terminal) server_request.Sink {
        return .{ .context = self, .writeFn = write };
    }

    fn write(context: *anyopaque, kind: server_request.Kind, text: []const u8) server_request.Sink.Signal {
        const self: *Terminal = @ptrCast(@alignCast(context));
        self.put(kind, text) catch {
            self.failed = true;
            return .stop;
        };
        return .more;
    }

    fn put(self: *Terminal, kind: server_request.Kind, text: []const u8) !void {
        if (self.last == .reasoning and kind == .content) {
            try self.stderr.writeAll("\n");
            try self.stderr.flush();
        }
        self.last = kind;
        const out = if (kind == .reasoning) self.stderr else self.stdout;
        try out.writeAll(text);
        try out.flush();
    }

    fn finish(self: *Terminal) !void {
        if (self.failed) return error.WriteFailed;
        if (self.last == .reasoning) {
            try self.stderr.writeAll("\n");
            try self.stderr.flush();
        }
        try self.stdout.writeAll("\n");
        try self.stdout.flush();
    }
};
