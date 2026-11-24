//! `dispatch.zig`'s tests: a table of two models, one of which lacks a command, and directories whose
//! `config.json` does and does not say a `model_type`.

const std = @import("std");
const testing = std.testing;
const dispatch = @import("../../src/decode/dispatch.zig");

const fx = struct {
    fn one(_: std.process.Init, _: *std.Io.Writer, _: []const []const u8) anyerror!u8 {
        return 1;
    }
    fn two(_: std.process.Init, _: *std.Io.Writer, _: []const []const u8) anyerror!u8 {
        return 2;
    }

    const models = [_]dispatch.Model{
        .{ .name = "First", .model_types = &.{ "first_a", "first_b" }, .commands = &.{
            .{ .name = "info", .least = 1, .run = one },
            .{ .name = "chat", .least = 2, .usage = "chat's own\n", .run = one },
        } },
        .{ .name = "Second", .model_types = &.{"second"}, .commands = &.{
            .{ .name = "info", .least = 1, .run = two },
            .{ .name = "draw", .least = 3, .run = two },
            .{ .name = "serve", .least = 1, .run = two, .dirless = "--from-elsewhere" },
        } },
    };
};

test "a command goes to the model its directory's model_type names, and is refused where none or that one lacks it" {
    const models = &fx.models;
    try testing.expect(dispatch.pick(models, "nosuch", null) == .none);
    try testing.expect(dispatch.pick(models, "nosuch", "second") == .none);

    // each of a model's names, and each model's own function
    try testing.expectEqual(&fx.one, dispatch.pick(models, "info", "first_a").command.run);
    try testing.expectEqual(&fx.one, dispatch.pick(models, "info", "first_b").command.run);
    try testing.expectEqual(&fx.two, dispatch.pick(models, "info", "second").command.run);
    try testing.expectEqualStrings("chat's own\n", dispatch.pick(models, "chat", "first_a").command.usage.?);
    try testing.expectEqual(@as(usize, 3), dispatch.pick(models, "draw", "second").command.least);

    // without a model_type: the first model that has the command
    try testing.expectEqual(&fx.one, dispatch.pick(models, "info", null).command.run);
    try testing.expectEqual(&fx.two, dispatch.pick(models, "draw", null).command.run);

    // a model that lacks it, and a model no table has; a name is matched whole
    try testing.expectEqualStrings("Second", dispatch.pick(models, "chat", "second").lacking.name);
    try testing.expectEqualStrings("First", dispatch.pick(models, "draw", "first_b").lacking.name);
    try testing.expectEqualStrings("third", dispatch.pick(models, "info", "third").unknown);
    try testing.expectEqualStrings("first", dispatch.pick(models, "info", "first").unknown);
    try testing.expectEqualStrings("", dispatch.pick(models, "info", "").unknown);

    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try testing.expectEqualStrings("First (first_a or first_b), Second (second)", try dispatch.known(arena, models));
    try testing.expectEqualStrings(
        "config.json describes model_type \"third\", not First (first_a or first_b), Second (second)",
        try dispatch.unknownText(arena, models, "third"),
    );
    try testing.expectEqualStrings(
        "Second (model_type \"second\") has no command `chat`; its commands: info draw serve",
        try dispatch.lackingText(arena, models[1], "second", "chat"),
    );
}

test "a directory's model_type is read from its config.json, and is none where it cannot be" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();

    const Case = struct { name: []const u8, config: ?[]const u8, expected: ?[]const u8 };
    for ([_]Case{
        .{ .name = "plain", .config = "{\"architectures\": [\"X\"], \"model_type\": \"second\", \"nested\": {\"model_type\": \"inner\"}}", .expected = "second" },
        .{ .name = "empty-name", .config = "{\"model_type\": \"\"}", .expected = "" },
        .{ .name = "absent", .config = "{\"nested\": {\"model_type\": \"inner\"}}", .expected = null },
        .{ .name = "number", .config = "{\"model_type\": 3}", .expected = null },
        .{ .name = "broken", .config = "{\"model_type\": \"second\"", .expected = null },
        .{ .name = "list", .config = "[\"model_type\"]", .expected = null },
        .{ .name = "no-file", .config = null, .expected = null },
    }) |case| {
        try tmp.dir.createDir(io, case.name, .default_dir);
        if (case.config) |text| {
            var sub = try tmp.dir.openDir(io, case.name, .{});
            defer sub.close(io);
            try sub.writeFile(io, .{ .sub_path = "config.json", .data = text });
        }
        const found = dispatch.modelType(arena, io, tmp.dir, case.name);
        if (case.expected) |expected| try testing.expectEqualStrings(expected, found.?) else try testing.expect(found == null);
    }
    try testing.expect(dispatch.modelType(arena, io, tmp.dir, "no-directory") == null);
}

test "a command's dirless flag names its model, the directory unread" {
    const models = &fx.models;
    try testing.expectEqualStrings("second", dispatch.dirless(models, "serve", &.{ "/nowhere", "--from-elsewhere" }).?);
    try testing.expectEqual(@as(?[]const u8, null), dispatch.dirless(models, "serve", &.{"/nowhere"}));
    try testing.expectEqual(@as(?[]const u8, null), dispatch.dirless(models, "info", &.{ "/nowhere", "--from-elsewhere" }));
    try testing.expectEqual(&fx.two, dispatch.pick(models, "serve", dispatch.dirless(models, "serve", &.{ "/nowhere", "--from-elsewhere" })).command.run);
}
