//! `zspark serve DIR`: Unlimited-OCR-2 kept loaded behind TensorFold's HTTP server (spec 147): the host renders
//! zspark's template (the checkpoint has none: `../requests.zig`), tokenizes, takes the pages' bytes and speaks
//! the three APIs; `../served.zig`'s engine runs the tower and the decoder, one reply at a time. A config without
//! `sparse_attention_config` is the retired v1's and is refused before a weight is read (`../config.zig`).
//!
//! Its defaults are the checkpoint's: `--context` 131,072 (its `max_position_embeddings`, the HF code's
//! `max_length`), a reply's tokens where neither the request nor `--max-tokens` says the room the context leaves
//! (its `infer`'s `max_new_tokens`), greedy, and no prefix shared between requests. `--name` is the host's, the
//! directory's name (`Unlimited-OCR-2`). Thinking is off whatever `--thinking` says: the model does not reason,
//! and its whole reply is the message's content. zspark's own flag: `--rows N`, a prompt window's rows (1,024;
//! the same bits whatever the window). `--parallel` above 1 is refused, and the host's flags no model of zspark
//! reads (`command.zig` `unserved`), `--chat-template` among them.
//!
//! Its host takes a prompt and reply up to the state's context and 128 of the largest pages more
//! (`documentCapacity`): a long document's parts run in the state (spec 147 WS-08).

const std = @import("std");
const server = @import("server");
const template = @import("template");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const command = @import("../../../server/command.zig");
const segments_file = @import("../../../server/segments.zig");
const sampling = @import("../../../decode/sampling.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const forward_file = @import("../forward.zig");
const model_file = @import("../model.zig");
const config_file = @import("../config.zig");
const requests = @import("../requests.zig");
const served_file = @import("../served.zig");
const vision = @import("../vision.zig");

/// The tokens a prompt and its reply share unless `--context` says: the config's `max_position_embeddings`, the
/// HF code's `max_length`.
const default_context = 131072;
/// The tokens a prompt and its reply share at the host: the state's, and a document of `v2_images`
/// pages of the most rows past it, their replies read in parts (SGLang splits a document before its context
/// length is checked).
fn documentCapacity(context: usize) usize {
    return context + requests.v2_images * (requests.page_tokens + 1);
}
/// A prompt window's rows unless `--rows` says: a page's prompt (903 to 3,303 rows on the CER set) in one to
/// four windows.
const default_rows = 1024;

const Own = struct {
    rows: ?usize = null,

    const flags = [_]cli.Flag{
        .{ .name = "--rows", .field = "rows", .min = 1 },
    };

    fn kindOf(word: []const u8) command.Kind {
        for (flags) |each| if (std.mem.eql(u8, each.name, word)) return .own_valued;
        return .host;
    }
};

pub fn run(init: std.process.Init, _: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const io = init.io;
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    const path = words[0];
    const parted = try command.part(arena, words[1..], Own.kindOf);
    const own = cli.parse(Own, &Own.flags, .{ .quoted = false, .prefix = "serve: " }, parted.own) orelse return 2;
    const bounded = try command.bounded(arena, parted.rest) orelse return 2;
    var args = try command.hostArgs(arena, path, bounded.rest) orelse return 2;
    _ = command.parallelWithin(args, 1, 1, 1) orelse return 2;
    // the config first: the retired v1's is refused here, before the device or a weight is read
    var config_problem: config_file.Diagnostic = .{};
    {
        var config_dir = std.Io.Dir.cwd().openDir(io, path, .{}) catch |err| {
            std.log.err("{s}: {t}", .{ path, err });
            return 1;
        };
        defer config_dir.close(io);
        _ = config_file.Config.read(arena, io, config_dir, &config_problem) catch |err| {
            std.log.err("{s}: config.json: {s} ({t})", .{ path, config_problem.message, err });
            return 1;
        };
    }
    const chat_template = requests.template;
    const context = (command.askedContext(args, 64) catch return 2) orelse default_context;
    // a reply's tokens where neither the request nor `--max-tokens` says: the room the context leaves (0, the
    // host's; `infer`'s max_new_tokens)
    // the long documents (spec 147 WS-08): a prompt and reply the host takes past the state, which `../parts.zig`
    // runs in parts of pages; `ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW` a part's most pages where a request names none
    const capacity = documentCapacity(context);
    const page_window: ?u32 = if (init.minimal.environ.getPosix("ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW")) |text| window: {
        const n = std.fmt.parseInt(u32, text, 10) catch {
            std.log.err("ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW: {s} is not a count of pages", .{text});
            return 2;
        };
        break :window if (n == 0) null else n;
    } else null;
    // the model does not reason: a reply is content, never `reasoning_content` (the host's default is a reply
    // that thinks until `</think>`)
    args.thinking = false;
    const rows = own.rows orelse default_rows;
    command.warnUnread(args, "is not read by zspark's Unlimited-OCR", &.{
        .{ .name = "--prompt-cache-gib", .given = args.prompt_cache_gib != null },
        .{ .name = "--learn", .given = args.learn },
    });
    command.logVersion();
    server.log.init(io, false);
    var startup = try command.Startup.open(arena, init.environ_map) orelse return 1;
    defer startup.deinit();

    // the text before the load: zspark's template in the checkpoint's place
    const text = command.loadText(gpa, io, path, arena) orelse return 1;
    defer text.deinit();
    text.source = chat_template;
    var diag: template.Diag = .{};
    defer if (diag.msg.len > 0) gpa.free(diag.msg);
    text.compiled = template.compile(gpa, chat_template, &diag) catch |err| {
        std.log.err("the chat template does not compile: {s} ({t})", .{ diag.msg, err });
        return 1;
    };
    var segments: segments_file.Segments = try .init(gpa, io, text.text(), &text.tok, segments_file.default_bytes);
    defer segments.deinit();

    var m = model_file.open(gpa, io, path, rows, 1) catch |err| {
        std.log.err("serve: {t}", .{err});
        return 1;
    };
    defer m.deinit();
    command.warnEnds(text.eos, &.{m.config.eos});
    var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
    defer checkpoint.close();
    var diagnostic: vision.Diagnostic = .{};
    var tower = vision.Tower.load(gpa, arena, &checkpoint, .{}, &diagnostic) catch |err| {
        std.log.err("{s}: the vision tower: {s}", .{ path, if (diagnostic.message.len > 0) diagnostic.message else @errorName(err) });
        return 1;
    };
    defer tower.deinit();
    var state: forward_file.State = try .init(gpa, m.config, context);
    defer state.deinit();
    var buffers: forward_file.Buffers = try .init(m.config, rows, 1);
    defer buffers.deinit();
    var sampler: sampling.Sampler = try .init(gpa, m.config.vocab);
    defer sampler.deinit(gpa);
    const logits = try gpa.alloc(u16, m.config.vocab);
    defer gpa.free(logits);
    try cuda.synchronize();
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    std.log.info("Unlimited-OCR-2: {d:.2} GiB on the device; {d} tokens of context, windows of {d} rows, {d} image(s) a request", .{ gib, context, rows, requests.Offer.vision().max_images });

    // the grammar engine a structured request's constraint is compiled by, over the host's tokenizer
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var front: command.Front = undefined;
    if (!try front.openWith(gpa, io, dir, path, &text.tok, chat_template, m.config.vocab, &.{m.config.eos})) return 1;
    defer front.deinit(gpa);

    var served: served_file.Served = .{
        .gpa = gpa,
        .io = io,
        .behind = .{ .model = &m, .tower = &tower, .state = &state, .buffers = &buffers, .sampler = &sampler, .logits = logits, .page_window = page_window },
        .capacity = capacity,
        .vocab = @intCast(m.config.vocab),
        .lanes = 1,
        .max_queued = bounded.max_queued,
        .frontend = &front.frontend,
        .compiler = front.grammar.compiler(),
        .think = null,
        .device_bytes = &cuda.account.bytes,
        .vision = requests.Offer.vision(),
    };
    const status = startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = served.engine(),
        .text = segments.text(),
        .served = server.hub.servedName(args.name, args.model, path),
        .model_dir = path,
        .model_type = config_file.model_types[0],
        .sampling = try command.samplingDefaults(arena, io, path, args),
        .environ = init.environ_map,
        .started = started,
    });
    served.close();
    std.log.info("stopped", .{});
    return status;
}
