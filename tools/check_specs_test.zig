//! `check_specs.zig`'s tests (spec 104 WS-06 moved them out of `check_specs.zig`, whose test block names this
//! file).

const std = @import("std");
const check_specs = @import("check_specs.zig");
const Exception = check_specs.Exception;
const budget = check_specs.budget;
const check = check_specs.check;
const checkWith = check_specs.checkWith;
const exceptions = check_specs.exceptions;
const recipes_path = check_specs.recipes_path;
const vendored = check_specs.vendored;

// Tests run the validator on spec trees written to a temporary directory.

const testing = std.testing;

const Tree = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,
    /// The files a test's tree may hold over the sources' budget.
    allowed: []const Exception = &.{},

    fn init() !Tree {
        var tree: Tree = .{ .tmp = testing.tmpDir(.{ .iterate = true }), .arena = .init(testing.allocator) };
        try tree.tmp.dir.createDirPath(testing.io, "specs/archive");
        try tree.write("specs/000-template.md", "template\n");
        return tree;
    }

    fn deinit(self: *Tree) void {
        self.arena.deinit();
        self.tmp.cleanup();
    }

    fn write(self: *Tree, path: []const u8, data: []const u8) !void {
        try self.tmp.dir.writeFile(testing.io, .{ .sub_path = path, .data = data });
    }

    const Options = struct {
        id: []const u8 = "001",
        status: []const u8 = "Active",
        created: []const u8 = "2026-10-04",
        updated: []const u8 = "2026-10-04",
        extra: []const u8 = "",
        body: []const u8 = default_body,
    };

    const default_body =
        \\## Context
        \\## Goal and success criteria
        \\## Scope and non-goals
        \\## Decisions
        \\## Workstreams
        \\## Verification
        \\## Documentation sync
        \\## Progress and handoff
        \\## Completion summary
        \\
    ;

    fn spec(self: *Tree, path: []const u8, options: Options) !void {
        const text = try std.fmt.allocPrint(self.arena.allocator(),
            \\---
            \\id: "{s}"
            \\title: "A spec"
            \\status: {s}
            \\created: {s}
            \\updated: {s}
            \\coordinator: "someone"
            \\{s}---
            \\
            \\{s}
        , .{ options.id, options.status, options.created, options.updated, options.extra, options.body });
        try self.write(path, text);
    }

    fn index(self: *Tree, active: []const u8, archived: []const u8) !void {
        const text = try std.fmt.allocPrint(self.arena.allocator(),
            \\# Project specs
            \\Copy [000-template.md](000-template.md) to start a spec.
            \\## Active specs
            \\{s}
            \\
        , .{active});
        try self.write("specs/README.md", text);
        const archive = try std.fmt.allocPrint(self.arena.allocator(),
            \\# Archived specs
            \\## Archived specs
            \\{s}
            \\
        , .{archived});
        try self.write("specs/archive/README.md", archive);
    }

    const Head = struct {
        revision: []const u8 = "69e33439ae950f17bcbe95c98f117d80f759ab6d",
        model: []const u8 = "A model",
        family: []const u8 = "qwen4",
        machines: []const u8 = "1",
        device_gib: []const u8 = "57.7",
        before_status: []const u8 = "",
        status: []const u8 = "supported",
        variables: []const u8 = "ZSPARK_ONE, ZSPARK_TWO_MORE",
        body: []const u8 = "",
    };

    /// A recipe page `name` under docs/recipes/ with this head.
    fn recipe(self: *Tree, name: []const u8, head: Head) !void {
        const allocator = self.arena.allocator();
        const text = try std.fmt.allocPrint(allocator,
            \\---
            \\checkpoint: someone/a-checkpoint
            \\revision: {s}
            \\model: {s}
            \\family: {s}
            \\storage: EXL3
            \\license: a licence
            \\disk_gb: 80
            \\machines: {s}
            \\device_gib: {s}
            \\{s}status: {s}
            \\drafters: MTP head
            \\variables: {s}
            \\---
            \\
            \\# A page
            \\
            \\{s}
        , .{ head.revision, head.model, head.family, head.machines, head.device_gib, head.before_status, head.status, head.variables, head.body });
        try self.write(try std.fmt.allocPrint(allocator, "{s}/{s}", .{ recipes_path, name }), text);
    }

    fn problems(self: *Tree) ![]const []const u8 {
        return checkWith(self.arena.allocator(), testing.io, self.tmp.dir, self.allowed);
    }

    /// A file of `size` bytes at `path`, its directories made.
    fn sized(self: *Tree, path: []const u8, size: usize) !void {
        if (std.fs.path.dirname(path)) |dir| try self.tmp.dir.createDirPath(testing.io, dir);
        const data = try self.arena.allocator().alloc(u8, size);
        @memset(data, 'x');
        try self.write(path, data);
    }

    fn expectPass(self: *Tree) !void {
        const found = try self.problems();
        if (found.len != 0) std.debug.print("unexpected: {s}\n", .{found[0]});
        try testing.expectEqual(@as(usize, 0), found.len);
    }

    fn expectProblem(self: *Tree, needle: []const u8) !void {
        const found = try self.problems();
        for (found) |problem| if (std.mem.indexOf(u8, problem, needle) != null) return;
        std.debug.print("no problem mentions \"{s}\" among {d}\n", .{ needle, found.len });
        for (found) |problem| std.debug.print("  {s}\n", .{problem});
        return error.TestExpectedProblem;
    }
};

test "an empty index beside the template passes" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.index("", "");
    try tree.write("specs/AGENTS.md", "guidance\n");
    try tree.write("specs/WORKER.md", "rules\n");
    try tree.expectPass();
}

test "indexed active and completed specs pass" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/archive/001-first.md", .{ .status = "Completed" });
    try tree.spec("specs/002-second.md", .{ .id = "002", .status = "Draft" });
    try tree.index("[002](002-second.md)", "[001](001-first.md)");
    try tree.expectPass();
}

test "filenames must be NNN-kebab-case.md and 000 is reserved" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/1-short.md", .{});
    try tree.spec("specs/001-Upper_case.md", .{});
    try tree.spec("specs/000-other.md", .{ .id = "000" });
    try tree.index("", "");
    try tree.expectProblem("1-short.md: filename must be NNN-kebab-case.md");
    try tree.expectProblem("001-Upper_case.md: filename must be NNN-kebab-case.md");
    try tree.expectProblem("000-other.md: ID 000 is reserved");
}

test "the frontmatter id must match the filename" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{ .id = "002" });
    try tree.index("[001](001-first.md)", "");
    try tree.expectProblem("does not match the filename");
}

test "status follows the directory" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{ .status = "Completed" });
    try tree.spec("specs/archive/002-second.md", .{ .id = "002", .status = "Active" });
    try tree.index("[001](001-first.md)", "[002](002-second.md)");
    try tree.expectProblem("specs/001-first.md: status \"Completed\" does not belong");
    try tree.expectProblem("specs/archive/002-second.md: status \"Active\" does not belong");
}

test "IDs are unique across active and archive, and continuous" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{});
    try tree.spec("specs/archive/001-again.md", .{ .status = "Completed" });
    try tree.spec("specs/003-third.md", .{ .id = "003" });
    try tree.index("[001](001-first.md) [003](003-third.md)", "[001](001-again.md)");
    try tree.expectProblem("ID 001 is used by both");
    try tree.expectProblem("IDs are not continuous: expected 002");
}

test "required headings appear exactly once, outside fences" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{ .body =
        \\## Context
        \\## Context
        \\## Goal and success criteria
        \\## Scope and non-goals
        \\## Decisions
        \\## Workstreams
        \\```
        \\## Verification
        \\```
        \\## Documentation sync
        \\## Progress and handoff
        \\## Completion summary
        \\
    });
    try tree.index("[001](001-first.md)", "");
    try tree.expectProblem("heading \"## Context\" appears 2 times");
    try tree.expectProblem("heading \"## Verification\" is missing");
}

test "dates are real and updated is not before created" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{ .created = "2026-02-30" });
    try tree.spec("specs/002-second.md", .{ .id = "002", .created = "2026-10-04", .updated = "2026-10-03" });
    try tree.index("[001](001-first.md) [002](002-second.md)", "");
    try tree.expectProblem("created \"2026-02-30\" is not a real");
    try tree.expectProblem("002-second.md: updated is earlier than created");
}

test "structured frontmatter is refused" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{ .extra = "owners: [a, b]\ntags:\n  - x\n" });
    try tree.index("[001](001-first.md)", "");
    try tree.expectProblem("\"owners: [a, b]\" is not a flat key: value");
    try tree.expectProblem("\"tags:\" is not a flat key: value");
}

test "the index links every spec once, in its section, in order, to a file" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.spec("specs/001-first.md", .{});
    try tree.spec("specs/002-second.md", .{ .id = "002" });
    try tree.spec("specs/003-third.md", .{ .id = "003" });
    try tree.spec("specs/archive/004-fourth.md", .{ .id = "004", .status = "Completed" });
    try tree.spec("specs/005-fifth.md", .{ .id = "005" });
    try tree.write("specs/README.md",
        \\Copy [t](000-template.md) or [t](000-template.md).
        \\## Active specs
        \\[003](003-third.md) [002](002-second.md) [002](002-second.md) [004](archive/004-fourth.md) [009](009-gone.md)
        \\## Archived specs
        \\[005](005-fifth.md)
        \\
    );
    try tree.write("specs/archive/README.md",
        \\## Archived specs
        \\[003](../003-third.md) [006](006-gone.md)
        \\
    );
    try tree.expectProblem("links 000-template.md 2 times");
    try tree.expectProblem("specs/archive/README.md: ../003-third.md is indexed in the wrong section");
    try tree.expectProblem("specs/archive/README.md: link 006-gone.md does not resolve");
    try tree.expectProblem("does not index 001-first.md");
    try tree.expectProblem("indexes 002-second.md 2 times");
    try tree.expectProblem("002-second.md is out of ascending ID order");
    try tree.expectProblem("archive/004-fourth.md is indexed in the wrong section");
    try tree.expectProblem("005-fifth.md is indexed in the wrong section");
    try tree.expectProblem("link 009-gone.md does not resolve");
}

test "guidance holds to its budgets, at the root and below" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.index("", "");
    try tree.sized("AGENTS.md", budget.root_guidance);
    try tree.sized("zig/src/AGENTS.md", budget.nested_guidance);
    try tree.sized("specs/WORKER.md", budget.worker_rules);
    try tree.expectPass();
    try tree.sized("AGENTS.md", budget.root_guidance + 1);
    try tree.sized("zig/src/models/one/AGENTS.md", budget.nested_guidance + 1);
    try tree.sized("specs/AGENTS.md", budget.nested_guidance + 1);
    try tree.sized("specs/WORKER.md", budget.worker_rules + 1);
    try tree.expectProblem("AGENTS.md: 10001 bytes, over its budget of 10000");
    try tree.expectProblem("zig/src/models/one/AGENTS.md: 14001 bytes, over its budget of 14000");
    try tree.expectProblem("specs/AGENTS.md: 14001 bytes");
    try tree.expectProblem("specs/WORKER.md: 7001 bytes, over its budget of 7000");
    try testing.expectEqual(@as(usize, 4), (try tree.problems()).len);
}

test "a spec over its budget is to be split" {
    var tree: Tree = try .init();
    defer tree.deinit();
    const long = try tree.arena.allocator().alloc(u8, budget.spec);
    @memset(long, 'x');
    try tree.spec("specs/001-first.md", .{ .extra = try std.fmt.allocPrint(tree.arena.allocator(), "note: {s}\n", .{long}) });
    try tree.index("[001](001-first.md)", "");
    try tree.expectProblem("specs/001-first.md: ");
    try tree.expectProblem("over a spec's 60000: split it");
}

test "a source file over its budget fails unless it is listed, and a listed one must still need it" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.index("", "");
    try tree.sized("zig/src/cuda/small.zig", budget.source);
    try tree.sized("build.zig", budget.source);
    try tree.sized("zig/src/.cache/large", budget.source + 1);
    try tree.sized("tools/reference/__pycache__/large.pyc", budget.source + 1);
    try tree.sized("elsewhere/large.bin", budget.source + 1);
    try tree.expectPass();

    try tree.sized("zig/src/cuda/deep/large.zig", budget.source + 1);
    try tree.sized("zig/kernels/table.cu", budget.source + 5);
    try tree.sized("build.zig", budget.source + 1);
    try tree.expectProblem("zig/src/cuda/deep/large.zig: 60001 bytes, over a file's 60000");
    try tree.expectProblem("zig/kernels/table.cu: 60005 bytes, over a file's 60000");
    try tree.expectProblem("build.zig: 60001 bytes, over its budget of 60000");
    try testing.expectEqual(@as(usize, 3), (try tree.problems()).len);

    // listed, both pass; one entry removed, its file fails again
    try tree.sized("build.zig", 10);
    tree.allowed = &.{
        .{ .path = "zig/src/cuda/deep/large.zig", .why = "to split", .until = "a workstream" },
        .{ .path = "zig/kernels/table.cu", .why = "generated" },
    };
    try tree.expectPass();
    tree.allowed = tree.allowed[1..];
    try tree.expectProblem("zig/src/cuda/deep/large.zig: 60001 bytes");
    try testing.expectEqual(@as(usize, 1), (try tree.problems()).len);

    // a listed file that was split, or removed, keeps no entry
    tree.allowed = &.{
        .{ .path = "zig/src/cuda/deep/large.zig", .why = "to split", .until = "a workstream" },
        .{ .path = "zig/kernels/table.cu", .why = "generated" },
        .{ .path = "tools/gone.zig", .why = "to split" },
    };
    try tree.sized("zig/src/cuda/deep/large.zig", budget.source);
    try tree.expectProblem("zig/src/cuda/deep/large.zig: listed as an exception to the size budget, but it is gone or within 60000");
    try tree.expectProblem("tools/gone.zig: listed as an exception");
    try testing.expectEqual(@as(usize, 2), (try tree.problems()).len);
}

test "a vendored file keeps upstream's size, and guidance beside it its budget" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.index("", "");
    try tree.sized(vendored ++ "/one/zig/src/table.zig", budget.source + 1);
    try tree.sized(vendored ++ "/one/AGENTS.md", budget.nested_guidance);
    try tree.write(vendored ++ "/one/UPSTREAM", "somewhere\n");
    try tree.write(vendored ++ "/one/PATCHES.md", "| File | What | Why | Proposed upstream |\n| --- | --- | --- | --- |\n| `zig/src/table.zig` | added | a test | no |\n");
    try tree.write(vendored ++ "/one/MANIFEST.sha256", "");
    try tree.expectPass();
    // the exemption is the directory's, not the name's
    try tree.sized("zig/vendored.zig", budget.source + 1);
    try tree.sized(vendored ++ "/one/AGENTS.md", budget.nested_guidance + 1);
    try tree.expectProblem("zig/vendored.zig: 60001 bytes, over a file's 60000");
    try tree.expectProblem(vendored ++ "/one/AGENTS.md: 14001 bytes, over its budget of 14000");
    try testing.expectEqual(@as(usize, 2), (try tree.problems()).len);
}

test "a recipe page's head is the twelve keys in order, with values of their kinds" {
    var tree: Tree = try .init();
    defer tree.deinit();
    try tree.index("", "");
    try tree.tmp.dir.createDirPath(testing.io, "zig/src/models/qwen4");
    try tree.tmp.dir.createDirPath(testing.io, "docs/recipes");
    try tree.write("docs/environment.md", "| `ZSPARK_ONE` | a path |\n| `ZSPARK_TWO_MORE` | another |\n");
    try tree.write("docs/recipes/README.md", "[a](one.md) [b](two.md) [c](keys.md) [d](open.md) [e](old.md)\n");
    try tree.recipe("one.md", .{});
    try tree.recipe("two.md", .{ .machines = "2", .device_gib = "unmeasured", .status = "unverified", .variables = "none" });
    try tree.recipe("old.md", .{ .status = "retired", .body = "**Retired 2026-10-17:** by the owner.\n" });
    try tree.expectPass();

    try tree.recipe("one.md", .{ .revision = "69e3343", .machines = "3", .device_gib = "about 58", .status = "done" });
    try tree.recipe("two.md", .{ .device_gib = "unmeasured", .family = "qwen5", .variables = "ZSPARK_ONE, ZSPARK_TWO" });
    try tree.recipe("keys.md", .{ .before_status = "note: extra\n" });
    try tree.write("docs/recipes/open.md", "# A page without a head\n");
    try tree.recipe("unlisted.md", .{ .model = "" });
    try tree.recipe("old.md", .{ .status = "retired" });
    try tree.expectProblem("one.md: revision \"69e3343\" is not a commit's 40 hex digits");
    try tree.expectProblem("one.md: machines \"3\" is neither 1 nor 2");
    try tree.expectProblem("one.md: device_gib \"about 58\" is neither a number nor unmeasured");
    try tree.expectProblem("one.md: status \"done\" is none of supported, unverified and retired");
    try tree.expectProblem("old.md: a retired page has no **Retired DATE:** line");
    try tree.expectProblem("two.md: device_gib is unmeasured, which only an unverified page may say");
    try tree.expectProblem("two.md: family \"qwen5\" is no directory under zig/src/models/");
    try tree.expectProblem("two.md: variable ZSPARK_TWO has no row in docs/environment.md");
    try tree.expectProblem("keys.md: frontmatter key \"note\" stands where \"status\" belongs");
    try tree.expectProblem("open.md: frontmatter must open and close with ---");
    try tree.expectProblem("unlisted.md: frontmatter line \"model:\" is not a flat key: value");
    try tree.expectProblem("docs/recipes/README.md does not link unlisted.md");
    try testing.expectEqual(@as(usize, 12), (try tree.problems()).len);

    // a key left out is named, and nothing is read of a head that is not whole
    try tree.write("docs/recipes/open.md", "---\ncheckpoint: a/b\nrevision: short\n---\n");
    try tree.expectProblem("open.md: frontmatter key \"model\" is missing");
}

test "this repository's exceptions each say why, and are listed once" {
    for (exceptions, 0..) |exception, i| {
        try testing.expect(exception.why.len > 0);
        for (exceptions[0..i]) |before| try testing.expect(!std.mem.eql(u8, before.path, exception.path));
    }
}

test "missing directories and files are reported" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const none = try checkWith(arena.allocator(), testing.io, tmp.dir, &.{});
    try testing.expectEqualStrings("specs/ is missing", none[0]);
    try tmp.dir.createDirPath(testing.io, "specs");
    const bare = try checkWith(arena.allocator(), testing.io, tmp.dir, &.{});
    try testing.expectEqual(@as(usize, 4), bare.len);
}

test "this repository's specs pass" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const found = try check(arena.allocator(), testing.io, .cwd());
    for (found) |problem| std.debug.print("{s}\n", .{problem});
    try testing.expectEqual(@as(usize, 0), found.len);
}
