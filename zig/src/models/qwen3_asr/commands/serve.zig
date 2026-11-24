//! `zspark serve DIR`: Qwen3-ASR kept loaded behind TensorFold's HTTP server (spec 138 WS-05), serving
//! `POST /v1/audio/transcriptions` in vLLM's shape, `/v1/models`, and chat (audio parts) and completions as vLLM
//! serves this model, greedy, thinking off whatever `--thinking` says (spec 150). The served name is `--name`, else
//! the directory's basename (`Qwen3-ASR-1.7B`, `Qwen3-ASR-0.6B`), as the host gives every model's. The engine is
//! `../served.zig`'s.
//!
//! zspark's own flags: `--rows N`, a prompt window's rows (2,048); `--max-audio-clip-s S`, off unless given: a
//! clip longer than S seconds is cut into pieces of at most S seconds as vLLM cuts it (vLLM's
//! `SpeechToTextConfig.max_audio_clip_s`, which it takes from the checkpoint's `chunk_length`, 30), and
//! `--parallel N` of them decode at once (16; spec 142 WS-06). TensorFold's flags as every `serve` reads
//! them (`server/command.zig`); `--context` the tokens a clip's audio rows, its prompt and its reply share
//! (`served.defaultContext`, the longest clip's); `--max-queued N` the requests that may wait behind the running
//! one (64). `--parallel` is not read without `--max-audio-clip-s`: a chat runs alone, one at a time.

const std = @import("std");
const server = @import("server");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const audio = @import("../../../audio/audio.zig");
const command = @import("../../../server/command.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const forward_file = @import("../forward.zig");
const pieces_file = @import("../pieces.zig");
const model_file = @import("../model.zig");
const served_file = @import("../served.zig");
const tower_file = @import("../tower.zig");
const transcript = @import("../transcript.zig");

pub const usage = "serve DIR [--rows N] [--max-audio-clip-s S] [TensorFold's serve flags]";

/// A prompt window's rows unless `--rows` says.
const default_rows = 2048;
/// The most pieces `--parallel` decodes at once: a 1,200 s clip's in 30 s pieces.
const max_lanes = 64;

const Own = struct {
    rows: ?usize = null,
    max_audio_clip_s: ?usize = null,

    const flags = [_]cli.Flag{
        .{ .name = "--rows", .field = "rows", .min = 1 },
        .{ .name = "--max-audio-clip-s", .field = "max_audio_clip_s", .min = 2, .max = served_file.max_seconds },
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
    // the model does not reason: a chat's reply is content, never `reasoning_content` (the host's default is a
    // reply that thinks until `</think>`; vLLM runs this card with no reasoning parser), as Unlimited-OCR's serve
    args.thinking = false;
    const smallest = tower_file.tokens(100) + 64;
    const context = (command.askedContext(args, smallest) catch return 2) orelse served_file.defaultContext();
    const rows = own.rows orelse default_rows;
    const lanes: usize = if (own.max_audio_clip_s != null) command.parallelWithin(args, served_file.default_lanes, 1, @min(max_lanes, rows)) orelse return 2 else 0;
    command.warnUnread(args, "is not read by zspark's Qwen3-ASR", &.{
        .{ .name = "--prompt-cache-gib", .given = args.prompt_cache_gib != null },
        .{ .name = "--learn", .given = args.learn },
    });
    command.logVersion();
    server.log.init(io, false);
    var startup = try command.Startup.open(arena, init.environ_map) orelse return 1;
    defer startup.deinit();

    // the text first: a checkpoint the server cannot tokenize fails before the load (vocab.json and merges.txt,
    // read as transformers' Qwen2 conversion makes its tokenizer.json, where the checkpoint ships none)
    const text = command.loadText(gpa, io, path, arena) orelse return 1;
    defer text.deinit();
    var tok = command.loadTokenizer(gpa, &text.tok, path) orelse return 1;
    defer tok.deinit(gpa);
    const pad = tok.addedId(transcript.audio_pad) orelse {
        std.log.err("{s}: the tokenizer has no {s}", .{ path, transcript.audio_pad });
        return 1;
    };

    const start = cuda.monotonic();
    var model = try model_file.open(gpa, io, path, rows, 1);
    defer model.deinit();
    var tower = tower: {
        var checkpoint: safetensors.Checkpoint = try .open(gpa, io, .cwd(), path);
        defer checkpoint.close();
        var said: tower_file.Diagnostic = .{};
        const config = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}/config.json", .{path}), arena, .limited(1 << 20));
        const shape = tower_file.checkConfig(arena, config, &said) catch |err| {
            std.log.err("{s}: audio tower: {s}", .{ path, if (said.message.len > 0) said.message else @errorName(err) });
            return 1;
        };
        if (shape.out_dim != model.config.hidden) {
            std.log.err("{s}: audio tower: output_dim {d} is not the language model's hidden_size {d}", .{ path, shape.out_dim, model.config.hidden });
            return 1;
        }
        break :tower tower_file.Tower.load(gpa, arena, &checkpoint, shape, tower_file.max_chunks_default, tower_file.conv_chunks_default, &said) catch |err| {
            std.log.err("{s}: audio tower: {s}", .{ path, if (said.message.len > 0) said.message else @errorName(err) });
            return 1;
        };
    };
    defer tower.deinit();
    var state: forward_file.State = try .init(gpa, model.config, context);
    defer state.deinit();
    const lane_positions = if (own.max_audio_clip_s) |s| served_file.laneCapacity(s) else 0;
    var buffers: forward_file.Buffers = try .init(model.config, rows, @max(lanes, 1), @max(context, lane_positions));
    defer buffers.deinit();
    const extra = if (own.max_audio_clip_s) |s| served_file.pieceRows(s) else 0;
    var features: cuda.Buffer = try .init(.bf16, &.{ tower.max_chunks * tower_file.chunk_tokens + extra, tower.shape.out_dim });
    defer features.deinit();
    var lane_states: ?pieces_file.Lanes = if (own.max_audio_clip_s != null) try .init(gpa, model.config, lanes, lane_positions) else null;
    defer if (lane_states) |*l| l.deinit();
    const tables = try gpa.create(audio.mel.Tables);
    defer gpa.destroy(tables);
    tables.init();
    try cuda.synchronize();
    const gib = @as(f64, @floatFromInt(cuda.account.bytes)) / (1024.0 * 1024.0 * 1024.0);
    std.log.info("Qwen3-ASR: {d:.2} GiB in {d:.1} s; {d} tokens of context, windows of {d} rows", .{ gib, @as(f64, @floatFromInt(cuda.monotonic() - start)) / 1e9, context, rows });
    if (lane_states) |l| std.log.info("Qwen3-ASR: clips past {d} s in pieces, {d} at once of {d} positions each", .{ own.max_audio_clip_s.?, l.states.len, l.capacity() });

    var asr: served_file.Asr = .{
        .gpa = gpa,
        .io = io,
        .model = &model,
        .tower = &tower,
        .tables = tables,
        .tok = &tok,
        .state = &state,
        .buffers = &buffers,
        .features = features,
        .pad = pad,
        .context = context,
        .most_waiting = bounded.max_queued orelse served_file.most_waiting,
        .pieces = if (lane_states) |*l| .{ .seconds = own.max_audio_clip_s.?, .lanes = l } else null,
    };
    try asr.start();
    const status = startup.run(gpa, io, args, .{
        .identity = command.identity(),
        .engine = asr.engine(),
        .text = text.text(),
        .served = server.hub.servedName(args.name, args.model, path),
        .model_dir = path,
        .model_type = "qwen3_asr",
        .sampling = null,
        .environ = init.environ_map,
        .started = started,
    });
    asr.close();
    std.log.info("stopped", .{});
    return status;
}
