//! `zig build bench-held-turn -Doptimize=fast -- CKPT_DIR [tokens=16300] [messages=2] [turns=24] [memo=off]`:
//! what the host costs a held turn before the engine has the prompt (spec 072 WS-03). No GPU and no weights:
//! TensorFold's server over the checkpoint's own tokenizer and template (`hf_text`), and an engine that ends
//! every reply at once and notes when `submit` came.
//!
//! A conversation of about `tokens` prompt tokens over `messages` messages (the head of
//! `tools/reference/client_bench_corpus.txt`, cut by characters) is sent as a chat completion on the loopback
//! `turns` times, each turn with the reply before it and a new question appended, as a client that holds a
//! conversation does. A line a case: from the request's first byte written to `submit` (what a first token
//! waits for), and inside it the template's renders, the tokenizer's encodes and decodes (a request has three
//! renders and three encodes of the conversation: the prompt, the prompt without its generation suffix for
//! `history_len`, and the probe for the shared system prefix; and since TensorFold 1.0.4 four of a
//! one-message conversation, for `decode_spans`), then the rest (the body read and parsed, the messages normalized, the
//! chunk starts). The first turn is printed apart: nothing is kept yet. `memo=off` is TensorFold's text as it
//! is, `memo=on` what `zspark serve` hands the host, `zig/src/server/segments.zig` over it. The conversations
//! are the same from run to run, and a case's last line is a hash of every prompt's ids the engine was given:
//! the two sides print the same.
//!
//! Run it under `flock -s ~/.cache/zspark/machine.lock`, more than once, with `uptime` beside it.

const std = @import("std");
const server = @import("server");
const api = @import("engine_api");
const segments_file = @import("segments");
const Allocator = std.mem.Allocator;
const Io = std.Io;

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).toNanoseconds();
}

/// The host's text with a clock on its three working entries.
const Timed = struct {
    io: Io,
    inner: server.model_text.Text,
    render_ns: i96 = 0,
    encode_ns: i96 = 0,
    decode_ns: i96 = 0,
    renders: usize = 0,
    encodes: usize = 0,

    fn of(ctx: *anyopaque) *Timed {
        return @ptrCast(@alignCast(ctx));
    }
    fn encode(ctx: *anyopaque, a: Allocator, s: []const u8, add_special: bool) server.model_text.Error![]u32 {
        const t = of(ctx);
        const start = now(t.io);
        defer {
            t.encode_ns += now(t.io) - start;
            t.encodes += 1;
        }
        return t.inner.encode(a, s, add_special);
    }
    /// A chat's encode since TensorFold 1.1.4: the prompt with its request bytes masked.
    fn encodeMasked(ctx: *anyopaque, a: Allocator, s: []const u8, spans: []const server.model_text.Span, problem: *[]const u8) server.model_text.Error![]u32 {
        const t = of(ctx);
        const start = now(t.io);
        defer {
            t.encode_ns += now(t.io) - start;
            t.encodes += 1;
        }
        return t.inner.encodeMasked(a, s, spans, problem);
    }
    fn decode(ctx: *anyopaque, a: Allocator, ids: []const u32) Allocator.Error![]u8 {
        const t = of(ctx);
        const start = now(t.io);
        defer t.decode_ns += now(t.io) - start;
        return t.inner.decode(a, ids);
    }
    fn render(ctx: *anyopaque, a: Allocator, messages: server.json.Value, options: server.model_text.RenderOptions, problem: *[]const u8) server.model_text.Error!server.model_text.Rendered {
        const t = of(ctx);
        const start = now(t.io);
        defer {
            t.render_ns += now(t.io) - start;
            t.renders += 1;
        }
        return t.inner.render(a, messages, options, problem);
    }
    fn tokenId(ctx: *anyopaque, piece: []const u8) ?u32 {
        return of(ctx).inner.tokenId(piece);
    }
    fn tokenString(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error![]u8 {
        return of(ctx).inner.tokenString(a, id);
    }
    fn vocabSize(ctx: *anyopaque) u32 {
        return of(ctx).inner.vocabSize();
    }
    fn eosIds(ctx: *anyopaque) []const u32 {
        return of(ctx).inner.eosIds();
    }
    fn templateSource(ctx: *anyopaque) []const u8 {
        return of(ctx).inner.templateSource();
    }
    fn special(ctx: *anyopaque, id: u32) bool {
        return of(ctx).inner.isSpecial(id);
    }
    fn added(ctx: *anyopaque, id: u32) bool {
        return of(ctx).inner.isAdded(id);
    }
    fn tokenBytes(ctx: *anyopaque, a: Allocator, id: u32) Allocator.Error!?[]u8 {
        return of(ctx).inner.tokenBytes(a, id);
    }
    fn text(t: *Timed) server.model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource, .special = special, .added = added, .token_bytes = tokenBytes, .encode_masked = encodeMasked } };
    }
    fn reset(t: *Timed) void {
        t.render_ns = 0;
        t.encode_ns = 0;
        t.decode_ns = 0;
        t.renders = 0;
        t.encodes = 0;
    }
};

/// An engine with no model: a reply ends at once. It keeps when the last prompt came, how long it was, and a
/// hash of its ids (`memo=on` and `memo=off` must agree).
const Stub = struct {
    io: Io,
    at: i96 = 0,
    prompt: usize = 0,
    history: usize = 0,
    hash: u64 = 0,

    fn of(ctx: *anyopaque) *Stub {
        return @ptrCast(@alignCast(ctx));
    }
    fn info(_: *anyopaque) api.Info {
        return .{ .name = "zspark-bench", .lanes = 1, .context_window = 262144, .call_gates = true, .structures = true };
    }
    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const s = of(ctx);
        s.at = now(s.io);
        s.prompt = r.prompt.len;
        s.history = r.history_len;
        s.hash = std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(r.prompt));
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
    fn engine(s: *Stub) api.Engine {
        return .{ .ctx = s, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
};

const Message = struct { role: []const u8, content: []const u8 };
const Body = struct { model: []const u8 = "bench", messages: []const Message, max_tokens: u32 = 8, temperature: f64 = 0 };

/// One turn's figures, in milliseconds.
const Turn = struct { total: f64, render: f64, encode: f64, decode: f64, rest: f64, prompt: usize, hash: u64 };

const Bench = struct {
    gpa: Allocator,
    io: Io,
    timed: *Timed,
    stub: *Stub,
    text: server.model_text.Text,
    corpus: []const u8,
    tokens: usize,
    messages: usize,
    turns: usize,
    memo: []const u8,
    segments: *segments_file.Segments,

    var port = std.atomic.Value(u16).init(0);

    fn listening(p: u16) void {
        port.store(p, .release);
    }

    fn post(b: *Bench, body: []const u8) !Turn {
        const io = b.io;
        const address = try Io.net.IpAddress.parse("127.0.0.1", port.load(.acquire));
        var read_buffer: [1 << 14]u8 = undefined;
        var write_buffer: [1 << 14]u8 = undefined;
        b.timed.reset();
        b.stub.at = 0;
        const start = now(io);
        const stream = try address.connect(io, .{ .mode = .stream });
        defer stream.close(io);
        var writer = stream.writer(io, &write_buffer);
        var reader = stream.reader(io, &read_buffer);
        try writer.interface.print("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{body.len});
        try writer.interface.writeAll(body);
        try writer.interface.flush();
        const reply = try reader.interface.allocRemaining(b.gpa, .unlimited);
        defer b.gpa.free(reply);
        if (b.stub.at == 0 or !std.mem.startsWith(u8, reply, "HTTP/1.1 200")) {
            std.debug.print("refused: {s}\n", .{reply[0..@min(reply.len, 600)]});
            return error.Refused;
        }
        const ms = struct {
            fn of(ns: i96) f64 {
                return @as(f64, @floatFromInt(ns)) / 1e6;
            }
        }.of;
        const total = ms(b.stub.at - start);
        const render = ms(b.timed.render_ns);
        const encode = ms(b.timed.encode_ns);
        const decode = ms(b.timed.decode_ns);
        return .{ .total = total, .render = render, .encode = encode, .decode = decode, .rest = total - render - encode - decode, .prompt = b.stub.prompt, .hash = b.stub.hash };
    }

    /// A conversation whose history is `messages` messages of `chars` characters in all, then `turn` more
    /// exchanges, the last one open. `salt` makes it one no run before has sent.
    fn conversation(b: *Bench, a: Allocator, chars: usize, turn: usize, salt: u64) ![]u8 {
        var list: std.ArrayList(Message) = .empty;
        try list.append(a, .{ .role = "system", .content = try std.fmt.allocPrint(a, "You answer from the documents below. Run {x}.", .{salt}) });
        const count = if (chars == 0) 0 else b.messages;
        const each = if (count == 0) 0 else chars / count;
        for (0..count) |i| {
            const piece = b.corpus[i * each ..][0..each];
            try list.append(a, .{ .role = if (i % 2 == 0) "user" else "assistant", .content = piece });
        }
        if (count % 2 == 0) try list.append(a, .{ .role = "user", .content = "What is the first part about?" });
        for (0..turn) |i| {
            try list.append(a, .{ .role = "assistant", .content = try std.fmt.allocPrint(a, "It is about its subject, answer {d}.", .{i}) });
            try list.append(a, .{ .role = "user", .content = try std.fmt.allocPrint(a, "And part {d}, in a sentence?", .{i + 2}) });
        }
        return std.json.Stringify.valueAlloc(a, Body{ .messages = list.items }, .{});
    }

    fn line(name: []const u8, values: []Turn) void {
        const Column = struct { name: []const u8, at: usize };
        std.debug.print("| {s} | {d} | {d} |", .{ name, values[0].prompt, values.len });
        inline for (.{ Column{ .name = "total", .at = 0 }, Column{ .name = "render", .at = 1 }, Column{ .name = "encode", .at = 2 }, Column{ .name = "decode", .at = 3 }, Column{ .name = "rest", .at = 4 } }) |column| {
            var buffer: [512]f64 = undefined;
            const n = @min(values.len, buffer.len);
            for (values[0..n], buffer[0..n]) |v, *slot| slot.* = switch (column.at) {
                0 => v.total,
                1 => v.render,
                2 => v.encode,
                3 => v.decode,
                else => v.rest,
            };
            std.mem.sort(f64, buffer[0..n], {}, std.sort.asc(f64));
            std.debug.print(" {d:.2} ({d:.2} to {d:.2}) |", .{ buffer[n / 2], buffer[0], buffer[n - 1] });
        }
        std.debug.print("\n", .{});
    }

    /// The characters of the corpus that make about `tokens` tokens.
    fn charsFor(b: *Bench, a: Allocator, tokens: usize) !usize {
        const sample = b.corpus[0..@min(b.corpus.len, 40000)];
        const ids = try b.text.encode(a, sample, false);
        return @min(b.corpus.len, tokens * sample.len / ids.len);
    }

    fn run(b: *Bench) !void {
        const io = b.io;
        while (port.load(.acquire) == 0) try io.sleep(.fromMilliseconds(5), .awake);
        var arena: std.heap.ArenaAllocator = .init(b.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const long = try b.charsFor(a, b.tokens);
        var prng: std.Random.DefaultPrng = .init(72);
        std.debug.print("| case ({s}) | prompt tokens | turns | first byte to submit, ms | of it the template | the encodes | the decodes | the rest |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n", .{b.memo});
        const Case = struct { name: []const u8, chars: usize };
        for ([_]Case{ .{ .name = "held", .chars = long }, .{ .name = "short", .chars = 0 } }) |case| {
            const salt = prng.random().int(u64);
            var values: std.ArrayList(Turn) = .empty;
            var hashes: u64 = 0;
            for (0..b.turns + 1) |turn| {
                // a short conversation is a new one each time: nothing of it was sent before
                const body = try b.conversation(a, case.chars, if (case.chars == 0) 0 else turn, if (case.chars == 0) prng.random().int(u64) else salt);
                const one = try b.post(body);
                hashes = std.hash.Wyhash.hash(hashes, std.mem.asBytes(&one.hash));
                if (turn == 0) {
                    var first = [1]Turn{one};
                    line(try std.fmt.allocPrint(a, "{s}, first met", .{case.name}), &first);
                } else try values.append(a, one);
            }
            line(try std.fmt.allocPrint(a, "{s}, a turn after", .{case.name}), values.items);
            std.debug.print("ids of {s}: {x}\n", .{ case.name, hashes });
            const kept = b.segments.account();
            if (b.segments.keeps) std.debug.print("kept after {s}: {d} segments, {d} bytes; {d} answered, {d} encoded\n", .{ case.name, kept.kept, kept.bytes, kept.hits, kept.misses });
        }
    }

    fn thread(b: *Bench) void {
        b.run() catch |err| {
            std.debug.print("held_turn_bench: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
        std.process.exit(0);
    }
};

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len < 2) {
        std.debug.print("usage: held_turn_bench CKPT_DIR [tokens=16300] [messages=2] [turns=24] [memo=off]\n", .{});
        return 2;
    }
    var tokens: usize = 16300;
    var messages: usize = 2;
    var turns: usize = 24;
    var memo: []const u8 = "off";
    for (argv[2..]) |word| {
        const eq = std.mem.indexOfScalar(u8, word, '=') orelse return 2;
        const value = word[eq + 1 ..];
        if (std.mem.eql(u8, word[0..eq], "tokens")) tokens = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, word[0..eq], "messages")) messages = @max(1, try std.fmt.parseInt(usize, value, 10)) else if (std.mem.eql(u8, word[0..eq], "turns")) turns = @max(1, try std.fmt.parseInt(usize, value, 10)) else if (std.mem.eql(u8, word[0..eq], "memo")) memo = value else return 2;
    }
    server.log.init(io, false);
    var problem: []const u8 = "";
    const hf = server.hf_text.HfText.load(gpa, io, argv[1], null, a, &problem) catch {
        std.debug.print("{s}: {s}\n", .{ argv[1], problem });
        return 1;
    };
    defer hf.deinit();
    const corpus = try std.Io.Dir.cwd().readFileAlloc(io, "tools/reference/client_bench_corpus.txt", a, .limited(1 << 24));
    if (!std.mem.eql(u8, memo, "on") and !std.mem.eql(u8, memo, "off")) return 2;
    var segments: segments_file.Segments = try .init(gpa, io, hf.text(), &hf.tok, if (std.mem.eql(u8, memo, "on")) segments_file.default_bytes else 0);
    defer segments.deinit();
    var timed: Timed = .{ .io = io, .inner = if (segments.keeps) segments.text() else hf.text() };
    var stub: Stub = .{ .io = io };
    var bench: Bench = .{ .gpa = gpa, .io = io, .timed = &timed, .stub = &stub, .text = hf.text(), .corpus = corpus, .tokens = tokens, .messages = messages, .turns = turns, .memo = memo, .segments = &segments };
    var usage: server.cli.Usage = .{};
    const args = server.cli.parse(a, &.{ argv[1], "--host", "127.0.0.1", "--port", "0", "--name", "bench" }, &usage) catch {
        std.debug.print("{s}\n", .{usage.message});
        return 2;
    };
    const client = try std.Thread.spawn(.{}, Bench.thread, .{&bench});
    client.detach();
    return server.serve.run(gpa, io, args, .{ .engine = stub.engine(), .text = timed.text(), .served = "bench", .environ = init.environ_map, .on_listen = Bench.listening });
}
