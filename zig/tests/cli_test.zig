//! `cli.zig`'s tests: a table of flags read without a log (`read`), and each refusal in the words `parse` logs.
//! The two tables stand for the two ways the commands read their arguments: `serve`'s (every argument a flag,
//! a refused value bare) and `chat`'s (one argument that is not a flag, a refused value between quotes).

const std = @import("std");
const testing = std.testing;
const cli = @import("../src/cli.zig");

const fx = struct {
    const Served = struct {
        host: []const u8 = "127.0.0.1",
        port: u16 = 8000,
        parallel: usize = 1,
        serial: bool = false,
        graphs: bool = true,

        const flags = [_]cli.Flag{
            .{ .name = "--host", .field = "host" },
            .{ .name = "--port", .field = "port" },
            .{ .name = "--parallel", .field = "parallel", .min = 1, .max = 64 },
            .{ .name = "--serial", .field = "serial" },
            .{ .name = "--no-graphs", .field = "graphs", .on = false },
        };
        const words: cli.Words = .{ .quoted = false };
    };

    const Chatted = struct {
        message: ?[]const u8 = null,
        temperature: ?f64 = null,
        top_k: ?u32 = null,
        think: bool = false,

        const flags = [_]cli.Flag{
            .{ .name = "--temperature", .field = "temperature" },
            .{ .name = "--top-k", .field = "top_k" },
            .{ .name = "--think", .field = "think" },
        };
        const words: cli.Words = .{ .positional = .{ .field = "message", .again = "more than one MESSAGE" }, .quoted = true };
    };

    /// What `parse` would log for `args`, which must be refused.
    fn refused(comptime Options: type, args: []const []const u8, buffer: []u8) ![]const u8 {
        var refusal: cli.Refusal = undefined;
        if (cli.read(Options, &Options.flags, Options.words, args, &refusal) != null) return error.TestExpectedRefusal;
        return std.fmt.bufPrint(buffer, "{f}", .{refusal});
    }

    fn expectRefused(comptime Options: type, args: []const []const u8, want: []const u8) !void {
        var buffer: [128]u8 = undefined;
        try testing.expectEqualStrings(want, try refused(Options, args, &buffer));
    }
};

test "flags: switches, values and the argument that is not a flag are read into the options" {
    var refusal: cli.Refusal = undefined;
    const served = cli.read(fx.Served, &fx.Served.flags, fx.Served.words, &.{ "--port", "8011", "--serial", "--host", "0.0.0.0", "--no-graphs", "--parallel", "64" }, &refusal).?;
    try testing.expectEqual(8011, served.port);
    try testing.expectEqualStrings("0.0.0.0", served.host);
    try testing.expectEqual(64, served.parallel);
    try testing.expect(served.serial and !served.graphs);
    try testing.expectEqual(fx.Served{}, cli.read(fx.Served, &fx.Served.flags, fx.Served.words, &.{}, &refusal).?);

    // a value is taken whatever it looks like: a negative number is the flag's, not a flag nor the message
    const chatted = cli.read(fx.Chatted, &fx.Chatted.flags, fx.Chatted.words, &.{ "--temperature", "-1", "hi", "--think", "--top-k", "0" }, &refusal).?;
    try testing.expectEqual(-1.0, chatted.temperature.?);
    try testing.expectEqualStrings("hi", chatted.message.?);
    try testing.expectEqual(0, chatted.top_k.?);
    try testing.expect(chatted.think);
}

test "flags: an unknown flag is said to be unknown wherever it stands" {
    // In the last place it was said to need a value, its name looked up only after its value was taken (spec
    // 056); a known flag there still needs one.
    try fx.expectRefused(fx.Served, &.{"--nosuch"}, "unknown option --nosuch");
    try fx.expectRefused(fx.Served, &.{ "--port", "8011", "--nosuch" }, "unknown option --nosuch");
    try fx.expectRefused(fx.Served, &.{ "--nosuch", "8011" }, "unknown option --nosuch");
    try fx.expectRefused(fx.Served, &.{ "--nosuch", "--port" }, "unknown option --nosuch");
    try fx.expectRefused(fx.Chatted, &.{ "hi", "--nosuch" }, "unknown option --nosuch");
    try fx.expectRefused(fx.Chatted, &.{ "--nosuch", "hi" }, "unknown option --nosuch");
    // without an argument that is not a flag, a bare word is an unknown flag too
    try fx.expectRefused(fx.Served, &.{"word"}, "unknown option word");

    try fx.expectRefused(fx.Served, &.{"--port"}, "--port needs a value");
    try fx.expectRefused(fx.Served, &.{ "--serial", "--host" }, "--host needs a value");
    try fx.expectRefused(fx.Chatted, &.{ "hi", "--temperature" }, "--temperature needs a value");
}

test "flags: a value a flag does not take, and a second message, in each command's words" {
    try fx.expectRefused(fx.Served, &.{ "--port", "http" }, "--port: http is not a value it takes");
    try fx.expectRefused(fx.Served, &.{ "--port", "65536" }, "--port: 65536 is not a value it takes");
    try fx.expectRefused(fx.Served, &.{ "--parallel", "0" }, "--parallel: 0 is not a value it takes");
    try fx.expectRefused(fx.Served, &.{ "--parallel", "65" }, "--parallel: 65 is not a value it takes");
    // the flag after a refused one is not read: the first refusal is the one said
    try fx.expectRefused(fx.Served, &.{ "--port", "--nosuch" }, "--port: --nosuch is not a value it takes");
    try fx.expectRefused(fx.Chatted, &.{ "--temperature", "warm", "hi" }, "--temperature: 'warm' is not a value it takes");
    try fx.expectRefused(fx.Chatted, &.{ "--top-k", "-1", "hi" }, "--top-k: '-1' is not a value it takes");
    try fx.expectRefused(fx.Chatted, &.{ "hi", "there" }, "more than one MESSAGE");
}

test "flags: bounds hold for an optional field and a float, and a NaN is outside them (spec 136 WS-04)" {
    const Bounded = struct {
        rows: ?usize = null,
        confidence: ?f32 = null,
        const flags = [_]cli.Flag{
            .{ .name = "--rows", .field = "rows", .min = 1, .max = 4096 },
            .{ .name = "--confidence", .field = "confidence", .min = 0 },
        };
        const words: cli.Words = .{ .quoted = false };
    };
    var refusal: cli.Refusal = undefined;
    const read = cli.read(Bounded, &Bounded.flags, Bounded.words, &.{ "--rows", "4096", "--confidence", "0" }, &refusal).?;
    try testing.expectEqual(4096, read.rows.?);
    try testing.expectEqual(0, read.confidence.?);
    try fx.expectRefused(Bounded, &.{ "--rows", "0" }, "--rows: 0 is not a value it takes");
    try fx.expectRefused(Bounded, &.{ "--rows", "4097" }, "--rows: 4097 is not a value it takes");
    try fx.expectRefused(Bounded, &.{ "--confidence", "-1" }, "--confidence: -1 is not a value it takes");
    try fx.expectRefused(Bounded, &.{ "--confidence", "nan" }, "--confidence: nan is not a value it takes");
}
