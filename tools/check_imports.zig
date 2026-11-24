//! Which part of `zig/src/` may import which (`docs/architecture.md`), read from every `@import("….zig")` of a
//! path: the layers below a model, a model beside another, and the directories inside a model. `check_specs.zig`
//! runs it with the spec tree and the size budgets. The tests under `zig/tests/` are not held to it: a test may
//! stand anywhere above what it tests.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;

const source_root = "zig/src";

/// A part of `zig/src/` (a directory, or a file at its top) and the parts its files may import beside itself.
const Part = struct { name: []const u8, may: []const []const u8 };

/// The layers, lowest first. A model stands on all of them, and on no other model; `main.zig` lists the models
/// and may import anything. A new directory under `zig/src/` takes a row here.
pub const parts = [_]Part{
    .{ .name = "cuda", .may = &.{} },
    .{ .name = "text", .may = &.{} },
    // a picture's bytes to RGB (spec 019): host only, below the server and the models that see images
    .{ .name = "image", .may = &.{} },
    // an audio file's bytes to 16 kHz samples and log-mel features (spec 138): host only, below the server and the
    // models that hear
    .{ .name = "audio", .may = &.{} },
    .{ .name = "weights", .may = &.{"cuda"} },
    // the words of a command line and the prompt a `chat` builds (sampled, so `decode`): below the server's shared
    // commands (spec 136 WS-04)
    .{ .name = "cli.zig", .may = &.{ "text", "weights", "decode" } },
    .{ .name = "ranks", .may = &.{ "cuda", "cli.zig" } },
    .{ .name = "decode", .may = &.{ "cuda", "weights", "text" } },
    // host only: no device, no checkpoint
    .{ .name = "server", .may = &.{ "decode", "text", "image", "audio", "cli.zig" } },
    .{ .name = "router", .may = &.{} },
    // `zspark stack` (spec 139): `serve`s and the router started and kept together, host only; it runs the router
    // as a process and imports nothing of it
    .{ .name = "stack", .may = &.{} },
    .{ .name = "models", .may = &.{ "cuda", "weights", "decode", "text", "image", "audio", "server", "ranks", "cli.zig" } },
};

const everything = [_][]const u8{ "blocks", "layers", "mtp", "dflash2", "session", "pool", "commands", "checks" };

/// A model's directories, by `docs/naming.md`'s names, and the others of the same model each may import. The
/// files at the model's top (its config, its `draft.zig`) are open to all of them. A layer's parts know nothing
/// above them, a drafter knows the layers and no other drafter, and what decodes knows no command.
pub const inside = [_]Part{
    .{ .name = "blocks", .may = &.{} },
    .{ .name = "layers", .may = &.{} },
    .{ .name = "mtp", .may = &.{ "blocks", "layers" } },
    .{ .name = "dflash2", .may = &.{ "blocks", "layers" } },
    .{ .name = "session", .may = &.{ "blocks", "layers", "mtp", "dflash2", "pool" } },
    .{ .name = "pool", .may = &.{ "blocks", "layers", "mtp", "dflash2", "session" } },
    .{ .name = "commands", .may = &everything },
    .{ .name = "checks", .may = &everything },
};

fn find(table: []const Part, name: []const u8) ?Part {
    for (table) |part| {
        if (std.mem.eql(u8, part.name, name)) return part;
    }
    return null;
}

fn allows(part: Part, name: []const u8) bool {
    for (part.may) |other| {
        if (std.mem.eql(u8, other, name)) return true;
    }
    return false;
}

/// Where a file of `zig/src/` stands: its part, and under `models/` its model and the model's directory (null
/// for a file at the model's top).
const Place = struct {
    part: []const u8,
    model: ?[]const u8 = null,
    within: ?[]const u8 = null,

    /// `path` is from `zig/src/`, its components in `names`.
    fn of(names: []const []const u8) Place {
        if (!std.mem.eql(u8, names[0], "models") or names.len < 3) return .{ .part = names[0] };
        return .{ .part = "models", .model = names[1], .within = if (names.len > 3) names[2] else null };
    }
};

const Checker = struct {
    arena: Allocator,
    io: Io,
    problems: *std.ArrayList([]const u8),

    fn fail(self: *Checker, comptime format: []const u8, args: anytype) Allocator.Error!void {
        try self.problems.append(self.arena, try std.fmt.allocPrint(self.arena, format, args));
    }
};

/// Every import under `root`'s `zig/src/` that crosses a boundary, appended to `problems`. A tree without the
/// directory has nothing to hold.
pub fn check(arena: Allocator, io: Io, root: Io.Dir, problems: *std.ArrayList([]const u8)) !void {
    var checker: Checker = .{ .arena = arena, .io = io, .problems = problems };
    var dir = root.openDir(io, source_root, .{ .iterate = true }) catch return;
    defer dir.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    try walk(&checker, dir, &names);
}

fn walk(checker: *Checker, dir: Io.Dir, names: *std.ArrayList([]const u8)) !void {
    var entries = dir.iterate();
    while (try entries.next(checker.io)) |entry| {
        if (entry.name[0] == '.') continue;
        try names.append(checker.arena, try checker.arena.dupe(u8, entry.name));
        defer _ = names.pop();
        switch (entry.kind) {
            .directory => {
                var below = try dir.openDir(checker.io, entry.name, .{ .iterate = true });
                defer below.close(checker.io);
                try walk(checker, below, names);
            },
            .file => if (std.mem.endsWith(u8, entry.name, ".zig")) {
                const text = try dir.readFileAlloc(checker.io, entry.name, checker.arena, .limited(1 << 20));
                try checkFile(checker, names.items, text);
            },
            else => {},
        }
    }
}

/// `names` are the file's components from `zig/src/`.
fn checkFile(checker: *Checker, names: []const []const u8, text: []const u8) !void {
    const from: Place = .of(names);
    const path = try std.mem.join(checker.arena, "/", names);
    if (std.mem.eql(u8, from.part, "main.zig")) return;
    const part = find(&parts, from.part) orelse
        return checker.fail("{s}/{s}: {s} has no row in tools/check_imports.zig's parts: say what it may import", .{ source_root, path, from.part });
    const own: ?Part = if (from.within) |within|
        find(&inside, within) orelse
            return checker.fail("{s}/{s}: a model's directory {s}/ has no row in tools/check_imports.zig's inside: take a name of docs/naming.md's, or add its row", .{ source_root, path, within })
    else
        null;

    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " \t"), "//")) continue;
        var rest = line;
        const opener = "@import(\"";
        while (std.mem.indexOf(u8, rest, opener)) |at| {
            rest = rest[at + opener.len ..];
            const end = std.mem.indexOfScalar(u8, rest, '"') orelse break;
            const target = rest[0..end];
            rest = rest[end..];
            if (!std.mem.endsWith(u8, target, ".zig")) continue;
            const resolved = try resolve(checker.arena, names[0 .. names.len - 1], target) orelse {
                // above `zig/src/`: the tests' and the tools' own, which no source imports
                try checker.fail("{s}/{s}: imports {s}, outside {s}/", .{ source_root, path, target, source_root });
                continue;
            };
            const to: Place = .of(resolved);
            if (!std.mem.eql(u8, to.part, from.part)) {
                if (!allows(part, to.part)) try checker.fail("{s}/{s}: imports {s}, but {s} may not import {s} (tools/check_imports.zig)", .{ source_root, path, target, from.part, to.part });
                continue;
            }
            const model = from.model orelse continue;
            // `models/x.zig` beside the models' directories would be a model's own file here: there is none
            const other = to.model orelse continue;
            if (!std.mem.eql(u8, model, other)) {
                try checker.fail("{s}/{s}: imports {s}: a model imports no other model; what two share goes to cuda/, weights/ or decode/", .{ source_root, path, target });
                continue;
            }
            const held = own orelse continue;
            const reached = to.within orelse continue;
            if (!std.mem.eql(u8, held.name, reached) and !allows(held, reached))
                try checker.fail("{s}/{s}: imports {s}, but a model's {s}/ may not import its {s}/ (tools/check_imports.zig)", .{ source_root, path, target, held.name, reached });
        }
    }
}

/// `target` read from a file in the directory `under` (components from `zig/src/`), as components from
/// `zig/src/`; null when it leaves the tree.
fn resolve(arena: Allocator, under: []const []const u8, target: []const u8) Allocator.Error!?[]const []const u8 {
    var out: std.ArrayList([]const u8) = .empty;
    try out.appendSlice(arena, under);
    var steps = std.mem.splitScalar(u8, target, '/');
    while (steps.next()) |step| {
        if (step.len == 0 or std.mem.eql(u8, step, ".")) continue;
        if (std.mem.eql(u8, step, "..")) {
            if (out.items.len == 0) return null;
            _ = out.pop();
        } else try out.append(arena, step);
    }
    return out.items;
}

const testing = std.testing;

/// The problems of a tree of `files` (path from the root, then text).
fn problemsOf(arena: Allocator, files: []const [2][]const u8) ![]const []const u8 {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    for (files) |file| {
        const path, const text = file;
        if (std.fs.path.dirname(path)) |dir| try tmp.dir.createDirPath(testing.io, dir);
        try tmp.dir.writeFile(testing.io, .{ .sub_path = path, .data = text });
    }
    var problems: std.ArrayList([]const u8) = .empty;
    try check(arena, testing.io, tmp.dir, &problems);
    return problems.items;
}

fn expectProblem(problems: []const []const u8, part: []const u8) !void {
    for (problems) |problem| {
        if (std.mem.indexOf(u8, problem, part) != null) return;
    }
    for (problems) |problem| std.debug.print("- {s}\n", .{problem});
    std.debug.print("no problem holds: {s}\n", .{part});
    return error.TestExpectedProblem;
}

test "the layers import downward, a model stands on them, and main.zig lists the models" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const problems = try problemsOf(arena_state.allocator(), &.{
        .{ "zig/src/main.zig", "const a = @import(\"models/one/commands.zig\");\nconst b = @import(\"models/two/commands.zig\");\n" },
        .{ "zig/src/cuda/cuda.zig", "const std = @import(\"std\");\n" },
        .{ "zig/src/weights/bf16.zig", "const cuda = @import(\"../cuda/cuda.zig\");\n" },
        .{ "zig/src/server/served.zig", "const sampling = @import(\"../decode/sampling.zig\"); const host = @import(\"tensorfold\");\n" },
        .{ "zig/src/models/one/commands.zig", "const serve = @import(\"commands/serve.zig\");\n" },
        .{ "zig/src/models/one/commands/serve.zig", "const served = @import(\"../../../server/served.zig\");\nconst s = @import(\"../session/session.zig\");\n" },
        .{ "zig/src/models/one/session/session.zig", "const l = @import(\"../layers/model.zig\");\nconst m = @import(\"../mtp/mtp.zig\");\nconst d = @import(\"../draft.zig\");\n" },
        .{ "zig/src/models/one/mtp/mtp.zig", "const l = @import(\"../layers/model.zig\");\nconst w = @import(\"../../../weights/bf16.zig\");\n" },
        .{ "zig/src/models/one/layers/model.zig", "const c = @import(\"config.zig\");\n// not @import(\"../session/session.zig\")\n" },
        .{ "zig/tests/weights/bf16_test.zig", "const m = @import(\"../../src/models/one/layers/model.zig\");\n" },
    });
    try testing.expectEqual(@as(usize, 0), problems.len);
}

test "an import that crosses a boundary is named: upward, into another model, across a model's directories" {
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const problems = try problemsOf(arena_state.allocator(), &.{
        .{ "zig/src/decode/draft.zig", "const m = @import(\"../models/one/draft.zig\");\n" },
        .{ "zig/src/server/served.zig", "const cuda = @import(\"../cuda/cuda.zig\");\n" },
        .{ "zig/src/models/one/draft.zig", "const other = @import(\"../two/layers/model.zig\");\n" },
        .{ "zig/src/models/one/mtp/mtp.zig", "pub const Linear = @import(\"../dflash2/weights.zig\").Linear;\n" },
        .{ "zig/src/models/one/layers/model.zig", "const s = @import(\"../session/session.zig\");\n" },
        .{ "zig/src/models/one/pool/hall.zig", "const c = @import(\"../commands/common.zig\");\n" },
        .{ "zig/src/models/one/serving/x.zig", "const std = @import(\"std\");\n" },
        .{ "zig/src/engine/x.zig", "const std = @import(\"std\");\n" },
        .{ "zig/src/text/x.zig", "const t = @import(\"../../tests/text/testing.zig\");\n" },
    });
    try expectProblem(problems, "zig/src/decode/draft.zig: imports ../models/one/draft.zig, but decode may not import models");
    try expectProblem(problems, "zig/src/server/served.zig: imports ../cuda/cuda.zig, but server may not import cuda");
    try expectProblem(problems, "zig/src/models/one/draft.zig: imports ../two/layers/model.zig: a model imports no other model");
    try expectProblem(problems, "zig/src/models/one/mtp/mtp.zig: imports ../dflash2/weights.zig, but a model's mtp/ may not import its dflash2/");
    try expectProblem(problems, "zig/src/models/one/layers/model.zig: imports ../session/session.zig, but a model's layers/ may not import its session/");
    try expectProblem(problems, "zig/src/models/one/pool/hall.zig: imports ../commands/common.zig, but a model's pool/ may not import its commands/");
    try expectProblem(problems, "zig/src/models/one/serving/x.zig: a model's directory serving/ has no row");
    try expectProblem(problems, "zig/src/engine/x.zig: engine has no row");
    try expectProblem(problems, "zig/src/text/x.zig: imports ../../tests/text/testing.zig, outside zig/src/");
    try testing.expectEqual(@as(usize, 9), problems.len);
}
