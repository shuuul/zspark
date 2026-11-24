//! What `served_test.zig` and the tests beside it share: `Hi`, a model that decodes nothing, behind TensorFold's
//! server on a port of the loopback (`Stand`), a request's events kept for a test's thread (`Kept`), and a
//! client's side of a socket.

const std = @import("std");
const api = @import("engine_api");
const host = @import("server");
const grammar = @import("../../src/text/grammar.zig");
const request_file = @import("../../src/server/request.zig");
const sched = @import("../../src/server/scheduler.zig");
const served_file = @import("../../src/server/served.zig");
const structured = @import("../../src/server/structured.zig");
const structured_fixture = @import("structured_fixture.zig");
const text_frontend = @import("../../src/text/frontend.zig");
const Allocator = std.mem.Allocator;
const posix = std.posix;
const testing = std.testing;

/// A chat request's one user message, `x`.
pub const user = "\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]";

/// A model whose reply is `hi` unless a test says another (`Behind.says`), under whatever the request asks and
/// cut at its limit: it counts its replies and those held to a constraint.
pub const Hi = struct {
    pub const Behind = struct {
        replies: std.atomic.Value(u32) = .init(0),
        constrained: std.atomic.Value(u32) = .init(0),
        /// Of `replies`, the warm passes (`Ask.warm`), each asked one token.
        warm: std.atomic.Value(u32) = .init(0),
        /// A tiny retained prefix for the observability test, guarded by the session's turn.
        prefix_cache: bool = false,
        prefix: [512]i32 = undefined,
        prefix_len: usize = 0,
        cache: sched.Cache = .{},
        cache_capacity: ?api.PromptCacheCapacity = null,
        /// A byte a token; set before the request that reads it.
        says: []const u8 = "hi",
        /// The token that ends a reply that stops, sent after it (`Live.closing`): the text's end token.
        end: ?u32 = null,
        /// The images of the last request that had any (`Ask.images`); the last request's prompt length and
        /// its ids that are `image_token`, set before its first token.
        images: []const api.Image = &.{},
        image_rows: usize = 0,
        prompt_len: usize = 0,
        with_images: std.atomic.Value(u32) = .init(0),
        /// The last request's no-repeat n-gram rule (`Request.no_repeat`), the whitelist's first four ids.
        no_repeat: api.NoRepeat = .{},
        no_repeat_whitelist: [4]u32 = @splat(0),
        no_repeat_whitelisted: usize = 0,
        /// The last request's PageCountGuard cap (`Request.max_page_tokens`).
        max_page_tokens: ?u32 = null,
        /// The last request's SGLang fields (spec 147 WS-08): `doc_split`, `tail_repeat`, and `image_mode`'s first
        /// 16 bytes, its length null where none was named.
        doc_split: api.DocSplit = .{},
        tail_repeat: api.TailRepeat = .{},
        image_mode: [16]u8 = @splat(0),
        image_mode_len: ?usize = null,
    };

    /// The row `Hi` gives `token` at the reply's position `at` (`Ask.logprobs`): its log probability
    /// -(at + 1) / 2, then the `k` best, `token` first, each next id a half lower.
    pub fn rowOf(token: u32, at: usize, k: u8) request_file.LogprobRow {
        const own = -@as(f32, @floatFromInt(at + 1)) / 2;
        var row: request_file.LogprobRow = .{ .token = token, .logprob = own, .count = k };
        for (0..k) |j| {
            row.ids[j] = (token + @as(u32, @intCast(j))) % 256;
            row.logprobs[j] = own - @as(f32, @floatFromInt(j)) / 2;
        }
        return row;
    }
    const Live = served_file.Served(Hi).Live;

    pub fn reply(live: *Live, arena: Allocator, ask: served_file.Ask, said: *request_file.Diagnostic) !served_file.Ended {
        const behind = &live.served.behind;
        if (!live.enters()) return .{ .reason = .cancelled };
        defer live.leaves();
        live.begins();
        if (behind.prefix_cache) live.cached = std.mem.indexOfDiff(i32, ask.prompt, behind.prefix[0..behind.prefix_len]) orelse @min(ask.prompt.len, behind.prefix_len);
        _ = behind.replies.fetchAdd(1, .acq_rel);
        if (try live.constrained(arena, false, said) != null) _ = behind.constrained.fetchAdd(1, .acq_rel);
        const rule = live.request.no_repeat;
        behind.no_repeat = .{ .size = rule.size, .window = rule.window };
        behind.no_repeat_whitelisted = rule.whitelist.len;
        behind.max_page_tokens = live.request.max_page_tokens;
        behind.doc_split = live.request.doc_split;
        behind.tail_repeat = live.request.tail_repeat;
        behind.image_mode_len = if (live.request.image_mode) |m| m.len else null;
        if (live.request.image_mode) |m| @memcpy(behind.image_mode[0..@min(16, m.len)], m[0..@min(16, m.len)]);
        @memcpy(behind.no_repeat_whitelist[0..@min(4, rule.whitelist.len)], rule.whitelist[0..@min(4, rule.whitelist.len)]);
        if (ask.warm and ask.max_tokens == 1) _ = behind.warm.fetchAdd(1, .acq_rel);
        behind.image_rows = 0;
        for (ask.prompt) |id| behind.image_rows += @intFromBool(id == image_token);
        behind.prompt_len = ask.prompt.len;
        if (ask.images.len > 0) {
            live.served.gpa.free(behind.images);
            behind.images = try live.served.gpa.dupe(api.Image, ask.images);
            _ = behind.with_images.fetchAdd(1, .acq_rel);
        }
        const fits = @min(behind.says.len, ask.max_tokens);
        for (behind.says[0..fits], 0..) |byte, at| {
            if (ask.logprobs) |k| live.row(&rowOf(byte, at, k));
            _ = live.give(byte);
        }
        const finish: request_file.Finish = if (fits < behind.says.len) .length else .stop;
        if (behind.prefix_cache) {
            const n = @min(ask.prompt.len, behind.prefix.len - fits - 1);
            @memcpy(behind.prefix[0..n], ask.prompt[0..n]);
            behind.prefix[n] = '|';
            for (behind.says[0..fits], behind.prefix[n + 1 ..][0..fits]) |byte, *token| token.* = byte;
            behind.prefix_len = n + 1 + fits;
            behind.cache.publish(.{ .states = 1 });
        }
        if (behind.end) |end| {
            if (ask.logprobs) |k| live.row(&rowOf(end, fits, k));
            live.closing(finish, @intCast(end));
        }
        return .{ .reason = live.reasonOf(finish) };
    }

    pub fn counts(_: *Behind) ?sched.Counts {
        return null;
    }

    pub fn promptCache(behind: *const Behind) ?sched.Cache.Snapshot {
        return if (behind.prefix_cache) behind.cache.read() else null;
    }

    pub fn cacheCapacity(behind: *const Behind) ?api.PromptCacheCapacity {
        return behind.cache_capacity;
    }

    pub fn closed(_: *Behind) void {}
};

/// A grammar engine that counts what it is asked to compile.
pub const Counting = struct {
    inner: grammar.Engine,
    compiles: std.atomic.Value(u32) = .init(0),

    pub fn interface(self: *Counting) grammar.Engine {
        return .{ .context = self, .words = self.inner.words, .compileFn = compile };
    }

    fn compile(context: *anyopaque, gpa: Allocator, spec: grammar.Spec, diagnostic: *grammar.Diagnostic) grammar.CompileError!grammar.Constraint {
        const self: *Counting = @ptrCast(@alignCast(context));
        _ = self.compiles.fetchAdd(1, .acq_rel);
        return self.inner.compile(gpa, spec, diagnostic);
    }

    /// `bench`'s compiler over this engine.
    pub fn compiler(self: *Counting, bench: *structured_fixture.Bench) structured.Compiler {
        var made = bench.compiler();
        self.* = .{ .inner = made.engine };
        made.engine = self.interface();
        return made;
    }
};

/// The host's text side: a byte a token, every conversation the six tokens of `prompt`, no template and no
/// special token (TensorFold's `stream_preflight_test.zig`'s). With `contents`, a conversation is its
/// messages' contents, each after `|`: two that differ in their last message part there, as a template's do.
pub const Bytes = struct {
    contents: bool = false,
    /// A content part's text, and each image part `image_token`'s byte (as a template writes its placeholder).
    images: bool = false,
    /// With `contents`: each audio part `audio_token`'s byte, as Qwen3-ASR's template writes its placeholder.
    audios: bool = false,
    /// The text has its tokens' bytes (`token_bytes`, a byte a token), as the host's logprobs need, and 0 as
    /// its end token.
    bytes: bool = false,
    /// The one byte the tokenizer calls special (`Text.isSpecial`), as an added token flagged special.
    special: ?u8 = null,
    const model_text = host.model_text;

    pub fn text(self: *Bytes) model_text.Text {
        if (self.bytes) return .{ .ctx = self, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = endIds, .render = render, .template_source = templateSource, .token_bytes = tokenBytes, .special = isSpecial, .added = isAdded } };
        return .{ .ctx = self, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .special = isSpecial, .added = isAdded } };
    }

    fn isSpecial(ctx: *anyopaque, id: u32) bool {
        const self: *Bytes = @ptrCast(@alignCast(ctx));
        return if (self.special) |byte| id == byte else false;
    }

    /// `image_token`, the bytes 2 and 3 (the vision tests' start and end markers) and the special byte are
    /// added tokens, as a checkpoint's markers are: since TensorFold 1.1.4 the host checks each before a vision
    /// server starts.
    fn isAdded(ctx: *anyopaque, id: u32) bool {
        return id == image_token or id == 2 or id == 3 or isSpecial(ctx, id);
    }

    fn endIds(_: *anyopaque) []const u32 {
        return &.{0};
    }

    fn tokenBytes(_: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        return try a.dupe(u8, &.{@intCast(id)});
    }

    fn encode(_: *anyopaque, a: Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }

    fn render(ctx: *anyopaque, a: Allocator, messages: host.json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error!model_text.Rendered {
        const self: *Bytes = @ptrCast(@alignCast(ctx));
        if (!self.contents or messages != .array) return .{ .text = try a.dupe(u8, "prompt") };
        var out: std.ArrayList(u8) = .empty;
        for (messages.array) |message| {
            try out.append(a, '|');
            const content = message.get("content") orelse continue;
            if (content == .string) try out.appendSlice(a, content.string);
            if ((self.images or self.audios) and content == .array) for (content.array) |part| {
                if (part.get("text")) |t| if (t == .string) try out.appendSlice(a, t.string);
                if (part.get("image_url") != null) try out.append(a, image_token);
                if (self.audios and part.get("audio_url") != null) try out.append(a, audio_token);
            };
        }
        // every byte the template's: a byte a token has no added text to mask
        return .{ .text = try out.toOwnedSlice(a) };
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// A request's events, kept for the test's thread.
pub const Kept = struct {
    tokens: [512]u32 = undefined,
    count: usize = 0,
    period: ?u32 = null,
    reason: ?api.Reason = null,
    done: std.atomic.Value(bool) = .init(false),
    queued_seconds: f64 = 0,

    pub fn sink(self: *Kept) api.Sink {
        return .{ .ctx = self, .event = event };
    }

    fn event(ctx: *anyopaque, _: api.Id, what: *const api.Event) void {
        const self: *Kept = @ptrCast(@alignCast(ctx));
        switch (what.*) {
            .tokens => |tokens| for (tokens) |token| {
                self.tokens[self.count] = token;
                self.count += 1;
            },
            .finished => |finished| {
                self.period = finished.stats.loop_period;
                self.queued_seconds = finished.stats.queued_seconds;
                self.reason = finished.reason;
                self.done.store(true, .release);
            },
            else => {},
        }
    }

    pub fn wait(self: *Kept) !void {
        while (!self.done.load(.acquire)) try testing.io.sleep(.fromMilliseconds(1), .awake);
    }
};

/// The image placeholder of `Bytes.images` and of `Stand.Options.vision`'s settings; 2 and 3 its markers.
pub const image_token = 1;
/// The audio placeholder of `Bytes.audios`.
pub const audio_token = 4;

/// `Hi` behind TensorFold's server on a port of the loopback, with the tiny tokenizer's grammar engine.
pub const Stand = struct {
    gpa: Allocator,
    bench: structured_fixture.Bench,
    counting: Counting,
    frontend: text_frontend.Frontend,
    device_bytes: usize,
    served: served_file.Served(Hi),
    text: Bytes,
    server: *host.Server,
    listener: host.listener.Listener,
    stop: std.atomic.Value(bool),
    accepting: std.Thread,
    port: u16,

    /// What a test may change of the stand: the context, the host's configuration past the served name.
    pub const Options = struct {
        capacity: usize = 64,
        /// `serve`'s own, as `cli.zig` gives it with no `--max-tokens`: what the context leaves
        default_max_tokens: i64 = (host.cli.Args{}).max_tokens,
        compact_at: ?host.server.CompactAt = null,
        /// The checkpoint's folder: where the host's xgrammar reads `tokenizer.json` and `config.json` (none: no xgrammar)
        model_dir: []const u8 = "",
        /// `Served.warm_turns`, which the host reads as it starts.
        warm_turns: bool = false,
        /// `Served.logprobs`, with the text's tokens' bytes and an end token (`Bytes.bytes`, `Hi.Behind.end`).
        logprobs: bool = false,
        /// `Served.vision`, with the text's image parts (`Bytes.images`); `--vision-urls`.
        vision: ?api.Vision = null,
        vision_urls: bool = false,
        /// `Bytes.special`: the byte the text calls special.
        special: ?u8 = null,
        prefix_cache: bool = false,
        cache_capacity: ?api.PromptCacheCapacity = null,
        prompt_cache_plan: ?api.PromptCachePlan = null,
    };

    pub fn init(self: *Stand, gpa: Allocator) !void {
        return self.initWith(gpa, .{});
    }

    pub fn initWith(self: *Stand, gpa: Allocator, options: Options) !void {
        self.gpa = gpa;
        try self.bench.init(gpa);
        errdefer self.bench.deinit(gpa);
        // read only for a reply that thinks, and no prompt here does (`think` is null)
        self.frontend = undefined;
        self.device_bytes = 0;
        self.served = .{ .gpa = gpa, .io = testing.io, .frontend = &self.frontend, .compiler = self.counting.compiler(&self.bench), .behind = .{}, .capacity = options.capacity, .vocab = 256, .lanes = 1, .think = null, .device_bytes = &self.device_bytes, .warm_turns = options.warm_turns, .logprobs = options.logprobs, .vision = options.vision };
        self.served.behind.prefix_cache = options.prefix_cache;
        self.served.behind.cache_capacity = options.cache_capacity;
        self.served.prompt_cache_plan = options.prompt_cache_plan;
        if (options.logprobs) self.served.behind.end = 0;
        self.text = .{ .contents = options.warm_turns or options.vision != null, .bytes = options.logprobs, .images = options.vision != null, .special = options.special };
        self.server = try .init(gpa, testing.io, self.served.engine(), self.text.text(), .{ .served_name = "hi", .model_ids = &.{"hi"}, .enable_thinking = false, .keep_warm_s = 0, .default_max_tokens = options.default_max_tokens, .compact_at = options.compact_at, .model_dir = options.model_dir, .vision_urls = options.vision_urls }, null);
        errdefer self.server.deinit();
        self.listener = try .open(.{ .ip4 = .loopback(0) });
        errdefer self.listener.close();
        self.port = self.listener.port();
        self.stop = .init(false);
        self.accepting = try std.Thread.spawn(.{}, host.Server.serve, .{ self.server, self.listener, &self.stop });
    }

    pub fn deinit(self: *Stand) void {
        self.stop.store(true, .release);
        self.accepting.join();
        // a connection's thread counts itself out after its last use of the server
        while (self.server.open_connections.load(.acquire) != 0) std.Io.sleep(testing.io, .fromMilliseconds(1), .awake) catch {};
        self.listener.close();
        self.served.close();
        self.gpa.free(self.served.behind.images);
        self.server.deinit();
        self.bench.deinit(self.gpa);
    }

    /// Everything the server sends for `body` posted to `path`, to the connection's end.
    pub fn post(self: *Stand, arena: Allocator, path: []const u8, body: []const u8) ![]const u8 {
        return self.ask(arena, try posted(arena, path, body));
    }

    /// Everything the server sends for a GET of `path`.
    pub fn get(self: *Stand, arena: Allocator, path: []const u8) ![]const u8 {
        return self.ask(arena, try std.fmt.allocPrint(arena, "GET {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n", .{path}));
    }

    pub fn ask(self: *Stand, arena: Allocator, asked: []const u8) ![]const u8 {
        return heard(arena, try sent(self.port, asked));
    }
};

/// A connection to the loopback's `port` with `asked` written to it: a client that has sent its request.
pub fn sent(port: u16, asked: []const u8) !posix.socket_t {
    const made = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(made) != .SUCCESS) return error.SocketFailed;
    const fd: posix.socket_t = @intCast(made);
    errdefer _ = posix.system.close(fd);
    // a reply that never ends fails the test in ten seconds
    const timeout: posix.timeval = .{ .sec = 10, .usec = 0 };
    try posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.RCVTIMEO, std.mem.asBytes(&timeout));
    // 127.0.0.1 in network order, as the bytes lie
    const address: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = @bitCast([4]u8{ 127, 0, 0, 1 }) };
    if (posix.errno(posix.system.connect(fd, @ptrCast(&address), @sizeOf(posix.sockaddr.in))) != .SUCCESS) return error.ConnectFailed;
    var at: usize = 0;
    while (at < asked.len) {
        const wrote = posix.system.write(fd, asked[at..].ptr, asked.len - at);
        if (posix.errno(wrote) != .SUCCESS) return error.WriteFailed;
        at += @intCast(wrote);
    }
    return fd;
}

/// Everything the server sends on `fd`, to the connection's end; the connection is closed.
pub fn heard(arena: Allocator, fd: posix.socket_t) ![]const u8 {
    defer _ = posix.system.close(fd);
    var reply: std.ArrayList(u8) = .empty;
    var buffer: [4096]u8 = undefined;
    while (true) {
        const got = try posix.read(fd, &buffer);
        if (got == 0) return reply.items;
        try reply.appendSlice(arena, buffer[0..got]);
    }
}

/// The request line and headers of `body` posted to `path`, as `Stand.post` writes them.
pub fn posted(arena: Allocator, path: []const u8, body: []const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "POST {s} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {d}\r\n\r\n{s}", .{ path, body.len, body });
}

/// A refusal is the whole reply: a 400 as JSON with the API's error body, and nothing of a stream.
pub fn expectRefused(reply: []const u8, body: []const u8) !void {
    try expectStatus(reply, "HTTP/1.1 400 Bad Request\r\n", body);
}

/// A full engine's: a 503 with the same body, which a client retries.
pub fn expectFull(reply: []const u8, body: []const u8) !void {
    try expectStatus(reply, "HTTP/1.1 503 Service Unavailable\r\n", body);
}

pub fn expectStatus(reply: []const u8, line: []const u8, body: []const u8) !void {
    errdefer std.debug.print("the reply:\n{s}\n", .{reply});
    try testing.expect(std.mem.startsWith(u8, reply, line));
    try testing.expect(std.mem.indexOf(u8, reply, "Content-Type: application/json\r\n") != null);
    try testing.expect(std.mem.indexOf(u8, reply, "text/event-stream") == null);
    const head = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoBody;
    try testing.expectEqualStrings(body, reply[head + 4 ..]);
}

/// `body` read as JSON.
pub fn jsonOf(arena: Allocator, body: []const u8) !host.json.Value {
    errdefer std.debug.print("the body:\n{s}\n", .{body});
    return switch (try host.json.parse(arena, body)) {
        .ok => |value| value,
        .err => error.NotJson,
    };
}

/// The body of a reply that is `status`.
pub fn bodyOf(reply: []const u8, status: []const u8) ![]const u8 {
    errdefer std.debug.print("the reply:\n{s}\n", .{reply});
    try testing.expect(std.mem.startsWith(u8, reply, status));
    const head = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.NoBody;
    return reply[head + 4 ..];
}
