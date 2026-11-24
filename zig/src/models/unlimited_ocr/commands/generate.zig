//! `zspark generate DIR [max=N] [rows=N] [context=N] [ngram=no] (fixture=FILE | ID..)`: a prompt through
//! Unlimited-OCR-2's language model and its greedy reply (`draft.greedy`), timed. The prompt is token ids, or a
//! reference fixture's (`tools/reference/unlimited_ocr/`, `ZSPARK_REFERENCE_UNLIMITED_OCR_2`): its `input_ids`,
//! with its `prefill.embeds` laid over the whole prompt, its page coordinates from `State.paged`. The n-gram rule
//! (`decode/ngram.zig`, over the reply alone, a window of 1,024 for several pages) and the PageCountGuard after
//! it (`requests.Rules`) are on unless `ngram=no` drops the rule. Images themselves are WS-03's and WS-06's.

const std = @import("std");
const cli = @import("../../../cli.zig");
const cuda = @import("../../../cuda/cuda.zig");
const draft = @import("../draft.zig");
const forward_file = @import("../forward.zig");
const model_file = @import("../model.zig");
const api = @import("engine_api");
const ngram = @import("../../../decode/ngram.zig");
const requests = @import("../requests.zig");
const safetensors = @import("../../../weights/safetensors.zig");

pub const usage = "generate DIR [max=N] [rows=N] [context=N] [ngram=no] (fixture=FILE | ID..)";

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    var max: usize = 8192;
    var rows: usize = 512;
    var context: usize = 8192;
    var rule: ?ngram.Rule = ngram.Rule.image;
    var fixture: ?[]const u8 = null;
    var ids: std.ArrayList(i32) = .empty;
    for (words[1..]) |word| {
        if (cli.value(word, "max=")) |text| {
            max = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "rows=")) |text| {
            rows = try std.fmt.parseInt(usize, text, 10);
        } else if (cli.value(word, "context=")) |text| {
            context = try std.fmt.parseInt(usize, text, 10);
        } else if (std.mem.eql(u8, word, "ngram=no")) {
            rule = null;
        } else if (cli.value(word, "fixture=")) |text| {
            fixture = text;
        } else {
            try ids.append(arena, std.fmt.parseInt(i32, word, 10) catch {
                try stdout.print("usage: zspark {s}\n", .{usage});
                return 2;
            });
        }
    }
    var features: ?cuda.Buffer = null;
    defer if (features) |*f| f.deinit();
    if (fixture) |path| {
        var file: safetensors.File = try .open(gpa, init.io, .cwd(), path);
        defer file.close();
        const input = file.get("input_ids") orelse return refuse(path, "input_ids");
        const lm = file.get("prefill.embeds") orelse return refuse(path, "prefill.embeds");
        const prompt = std.mem.bytesAsSlice(i32, input.data);
        for (prompt) |id| try ids.append(arena, id);
        features = try .init(.bf16, &.{ @intCast(lm.shape[0]), @intCast(lm.shape[1]) });
        try features.?.upload(lm.data);
    }
    if (ids.items.len == 0) {
        try stdout.print("usage: zspark {s}\n", .{usage});
        return 2;
    }
    const start = cuda.monotonic();
    var m = try model_file.open(gpa, init.io, words[0], rows, 1);
    defer m.deinit();
    var state: forward_file.State = try .init(gpa, m.config, context);
    defer state.deinit();
    var b: forward_file.Buffers = try .init(m.config, rows, 1);
    defer b.deinit();
    try cuda.synchronize();
    const loaded = cuda.monotonic();
    state.paged(ids.items) catch |err| {
        std.log.err("{s}: {t}: Unlimited-OCR-2 reads ids with an image and a <PAGE> marker up to the last image token", .{ words[0], err });
        return 1;
    };
    if (features) |f| state.image = .{ .features = f };
    const logits = try gpa.alloc(u16, m.config.vocab);
    defer gpa.free(logits);
    var reply: std.ArrayList(u32) = .empty;
    defer reply.deinit(gpa);
    var timing: draft.Timing = .{};
    // a request with neither field asked, or `ngram=no`'s size 0
    const asked: api.Request = .{ .prompt = &.{}, .max_tokens = 0, .no_repeat = if (rule == null) .{ .size = 0 } else .{}, .eos = &.{m.config.eos} };
    const rules: requests.Rules = .of(asked, ids.items, m.config.eos);
    try draft.greedy(gpa, m, &state, &b, ids.items, max, rules, logits, &reply, &timing);
    const end = cuda.monotonic();
    for (reply.items) |token| try stdout.print("{d} ", .{token});
    const ms = struct {
        fn of(ns: u64) f64 {
            return @as(f64, @floatFromInt(ns)) / 1e6;
        }
    }.of;
    const steps = reply.items.len -| 1;
    try stdout.print("\nload {d:.0} ms, prompt {d} tokens {d:.1} ms, {d} tokens, {d:.1} tokens a second after the first\n", .{
        ms(loaded - start),                                                                    ids.items.len, ms(timing.first - loaded), reply.items.len,
        if (steps > 0) @as(f64, @floatFromInt(steps)) / (ms(end - timing.first) / 1e3) else 0,
    });
    return 0;
}

fn refuse(path: []const u8, what: []const u8) u8 {
    std.log.err("{s}: no {s}", .{ path, what });
    return 1;
}
