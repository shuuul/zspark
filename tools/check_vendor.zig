//! Another project's files kept in this tree (`zig/vendor/NAME/`, spec 072): which of them still are upstream's
//! bytes, and that each one that is not says so. A directory there holds upstream's files at upstream's paths,
//! and four files of this repository's: `MANIFEST.sha256` (a line a file of the import, `sha256sum`'s own form,
//! written by an import and by nothing else), `UPSTREAM`, `PATCHES.md` and `AGENTS.md`. A file whose bytes are
//! not the manifest's (changed, added here, or removed) has a row in `PATCHES.md`, and a row stands for no file
//! that is upstream's as it was. No network and no package: the manifest is what upstream was.
//! `check_specs.zig` runs it with the spec tree and the size budgets.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// Where vendored directories stand, one a project.
pub const root_path = "zig/vendor";
const manifest_name = "MANIFEST.sha256";
const patches_name = "PATCHES.md";
/// This repository's files at a vendored directory's top; every other file there is held to the manifest.
const own = [_][]const u8{ manifest_name, "UPSTREAM", patches_name, "AGENTS.md" };
// an interrupted import's transaction records, gone once a publish holds — not the import's content
const transaction = [_][]const u8{ ".vendor-journal", ".vendor-journal.pre" };
/// A patch row's cells: the file, what was changed, why, and whether it was proposed upstream.
const patch_cells = 4;

const Digest = [Sha256.digest_length * 2]u8;

const Checker = struct {
    arena: Allocator,
    io: Io,
    problems: *std.ArrayList([]const u8),

    fn fail(self: *Checker, comptime format: []const u8, args: anytype) Allocator.Error!void {
        try self.problems.append(self.arena, try std.fmt.allocPrint(self.arena, format, args));
    }
};

/// Every vendored file under `root` that differs from its manifest without a patch row, and every row without
/// such a file, appended to `problems`. A tree without `zig/vendor/` has nothing to hold.
pub fn check(arena: Allocator, io: Io, root: Io.Dir, problems: *std.ArrayList([]const u8)) !void {
    var checker: Checker = .{ .arena = arena, .io = io, .problems = problems };
    var dir = root.openDir(io, root_path, .{ .iterate = true }) catch return;
    defer dir.close(io);
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| {
        if (entry.kind != .directory) continue;
        const name = try arena.dupe(u8, entry.name);
        var project = try dir.openDir(io, name, .{ .iterate = true });
        defer project.close(io);
        try checkProject(&checker, project, name);
    }
}

/// One project's directory, `zig/vendor/NAME/`.
fn checkProject(checker: *Checker, dir: Io.Dir, name: []const u8) !void {
    const arena = checker.arena;
    var whole = true;
    for (own) |file| dir.access(checker.io, file, .{}) catch {
        try checker.fail("{s}/{s}/{s} is missing", .{ root_path, name, file });
        whole = false;
    };
    if (!whole) return;

    // what upstream was
    var upstream: std.StringArrayHashMapUnmanaged(Digest) = .empty;
    const manifest = try dir.readFileAlloc(checker.io, manifest_name, arena, .limited(16 << 20));
    var lines = std.mem.splitScalar(u8, manifest, '\n');
    var number: usize = 0;
    while (lines.next()) |line| {
        number += 1;
        if (line.len == 0) continue;
        // sha256sum: the digest, two spaces, the path; a path it had to escape opens the line with a backslash
        const sound = line.len > 66 and std.mem.eql(u8, line[64..66], "  ") and for (line[0..64]) |c| {
            if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) break false;
        } else true;
        if (!sound) {
            try checker.fail("{s}/{s}/{s}: line {d} is not a sha256 and a path, as sha256sum writes them", .{ root_path, name, manifest_name, number });
            continue;
        }
        try upstream.put(arena, line[66..], line[0..64].*);
    }

    // what PATCHES.md accounts for; one row is one change, so a row that repeats another exactly is a
    // record a merge or a replay left twice (spec 168)
    var patched: std.StringArrayHashMapUnmanaged(void) = .empty;
    var rows_seen: std.StringArrayHashMapUnmanaged(void) = .empty;
    const patches = try dir.readFileAlloc(checker.io, patches_name, arena, .limited(1 << 20));
    lines = std.mem.splitScalar(u8, patches, '\n');
    var rows: usize = 0;
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, "\r \t");
        if (line.len < 2 or line[0] != '|' or line[line.len - 1] != '|') continue;
        rows += 1;
        // the head and the rule under it
        if (rows <= 2) continue;
        var cells = std.mem.splitScalar(u8, line[1 .. line.len - 1], '|');
        var count: usize = 0;
        var filled = true;
        var file: []const u8 = "";
        var record: std.ArrayList(u8) = .empty;
        while (cells.next()) |cell| : (count += 1) {
            const text = std.mem.trim(u8, cell, " \t`");
            if (text.len == 0) filled = false;
            if (count == 0) file = text;
            if (count > 0) try record.appendSlice(arena, "|");
            try record.appendSlice(arena, text);
        }
        if (count != patch_cells or !filled) {
            try checker.fail("{s}/{s}/{s}: the row of \"{s}\" must fill {d} cells: the file, what, why, proposed upstream", .{ root_path, name, patches_name, file, patch_cells });
            if (file.len == 0) continue;
        }
        const seen = try rows_seen.getOrPut(arena, record.items);
        if (seen.found_existing) try checker.fail("{s}/{s}/{s}: the row of \"{s}\" repeats another row exactly: one row is one change", .{ root_path, name, patches_name, file });
        try patched.put(arena, file, {});
    }

    // what is here
    var present: std.StringArrayHashMapUnmanaged(Digest) = .empty;
    var names: std.ArrayList([]const u8) = .empty;
    try walk(checker, dir, &names, &present);

    for (upstream.keys(), upstream.values()) |path, was| {
        const row = patched.contains(path);
        if (present.get(path)) |now| {
            const same = std.mem.eql(u8, &was, &now);
            if (!same and !row) try checker.fail("{s}/{s}/{s}: differs from upstream ({s}) and has no row in {s}: one commit and one row a change", .{ root_path, name, path, manifest_name, patches_name });
            if (same and row) try checker.fail("{s}/{s}/{s}: has a row in {s}, but its bytes are upstream's: remove the row", .{ root_path, name, path, patches_name });
        } else if (!row) {
            try checker.fail("{s}/{s}/{s}: upstream's file ({s}) is gone and has no row in {s}", .{ root_path, name, path, manifest_name, patches_name });
        }
    }
    for (present.keys()) |path| {
        if (!upstream.contains(path) and !patched.contains(path))
            try checker.fail("{s}/{s}/{s}: is not upstream's (no line in {s}) and has no row in {s}", .{ root_path, name, path, manifest_name, patches_name });
    }
    for (patched.keys()) |path| {
        if (!upstream.contains(path) and !present.contains(path))
            try checker.fail("{s}/{s}/{s}: the row of \"{s}\" names no file, here or upstream", .{ root_path, name, patches_name, path });
    }

    // an unresolved merge marker a patch commit carries is a conflict that was never settled (spec 168):
    // it reads every patched file still here — upstream's own bytes are its publisher's, not held to this
    for (patched.keys()) |path| {
        if (!present.contains(path)) continue;
        const bytes = try dir.readFileAlloc(checker.io, path, arena, .limited(64 << 20));
        var at = std.mem.splitScalar(u8, bytes, '\n');
        var line_no: usize = 0;
        while (at.next()) |line| : (line_no += 1) {
            if (std.mem.startsWith(u8, line, "<<<<<<<") or std.mem.startsWith(u8, line, "|||||||") or std.mem.startsWith(u8, line, ">>>>>>>"))
                try checker.fail("{s}/{s}/{s}: line {d} is an unresolved conflict marker: settle it against upstream's text, never the old file", .{ root_path, name, path, line_no + 1 });
        }
    }
}

/// Every file below `dir` with its digest, by its path from the project's directory (`names` so far). Nothing
/// is skipped but this repository's own files at the top: a hidden file would be a file too.
fn walk(checker: *Checker, dir: Io.Dir, names: *std.ArrayList([]const u8), present: *std.StringArrayHashMapUnmanaged(Digest)) !void {
    var entries = dir.iterate();
    while (try entries.next(checker.io)) |entry| {
        try names.append(checker.arena, try checker.arena.dupe(u8, entry.name));
        defer _ = names.pop();
        switch (entry.kind) {
            .directory => {
                var below = try dir.openDir(checker.io, entry.name, .{ .iterate = true });
                defer below.close(checker.io);
                try walk(checker, below, names, present);
            },
            .file => {
                const ours = names.items.len == 1 and ((for (own) |file| {
                    if (std.mem.eql(u8, file, entry.name)) break true;
                } else false) or (for (transaction) |file| {
                    if (std.mem.eql(u8, file, entry.name)) break true;
                } else false));
                if (ours) continue;
                const bytes = try dir.readFileAlloc(checker.io, entry.name, checker.arena, .limited(64 << 20));
                try present.put(checker.arena, try std.mem.join(checker.arena, "/", names.items), digestOf(bytes));
            },
            else => {},
        }
    }
}

fn digestOf(bytes: []const u8) Digest {
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

// Tests run the check on a vendored directory written to a temporary directory.

const testing = std.testing;

const Tree = struct {
    tmp: testing.TmpDir,
    arena: std.heap.ArenaAllocator,

    const at = root_path ++ "/one/";
    const head = "| File | What | Why | Proposed upstream |\n| --- | --- | --- | --- |\n";

    /// A directory `zig/vendor/one/` whose import was `files` (path, then bytes), with no patch.
    fn init(files: []const [2][]const u8) !Tree {
        var tree: Tree = .{ .tmp = testing.tmpDir(.{ .iterate = true }), .arena = .init(testing.allocator) };
        errdefer tree.deinit();
        var manifest: std.ArrayList(u8) = .empty;
        for (files) |file| {
            const path, const bytes = file;
            try tree.write(path, bytes);
            try manifest.print(tree.arena.allocator(), "{s}  {s}\n", .{ digestOf(bytes), path });
        }
        try tree.write(manifest_name, manifest.items);
        try tree.write("UPSTREAM", "repository: somewhere\n");
        try tree.write("AGENTS.md", "the rules\n");
        try tree.write(patches_name, head);
        return tree;
    }

    fn deinit(self: *Tree) void {
        self.arena.deinit();
        self.tmp.cleanup();
    }

    /// `bytes` at `path` below the vendored directory.
    fn write(self: *Tree, path: []const u8, bytes: []const u8) !void {
        const full = try std.fmt.allocPrint(self.arena.allocator(), at ++ "{s}", .{path});
        try self.tmp.dir.createDirPath(testing.io, std.fs.path.dirname(full).?);
        try self.tmp.dir.writeFile(testing.io, .{ .sub_path = full, .data = bytes });
    }

    /// The patch list with these rows under its head.
    fn rows(self: *Tree, text: []const u8) !void {
        try self.write(patches_name, try std.fmt.allocPrint(self.arena.allocator(), "# Patches\n\n" ++ head ++ "{s}", .{text}));
    }

    fn problems(self: *Tree) ![]const []const u8 {
        var found: std.ArrayList([]const u8) = .empty;
        try check(self.arena.allocator(), testing.io, self.tmp.dir, &found);
        return found.items;
    }

    /// Exactly these problems, each by a part of its message.
    fn expect(self: *Tree, parts: []const []const u8) !void {
        const found = try self.problems();
        const all = found.len == parts.len and for (parts) |part| {
            const held = for (found) |problem| {
                if (std.mem.indexOf(u8, problem, part) != null) break true;
            } else false;
            if (!held) break false;
        } else true;
        if (all) return;
        for (found) |problem| std.debug.print("- {s}\n", .{problem});
        for (parts) |part| std.debug.print("expected: {s}\n", .{part});
        return error.TestExpectedProblems;
    }

    /// Exactly `count` problems, each holding this part of the message: a refusal's shape without its
    /// wording pinned, so a reworded diagnostic does not need the test rewritten.
    fn expectCount(self: *Tree, part: []const u8, count: usize) !void {
        const found = try self.problems();
        const all = found.len == count and for (found) |problem| {
            if (std.mem.indexOf(u8, problem, part) == null) break false;
        } else true;
        if (all) return;
        for (found) |problem| std.debug.print("- {s}\n", .{problem});
        std.debug.print("expected {d} problems, each holding \"{s}\"\n", .{ count, part });
        return error.TestExpectedProblems;
    }
};

const imported = [_][2][]const u8{
    .{ "LICENSE", "a licence\n" },
    .{ "zig/src/server/chat.zig", "const std = @import(\"std\");\n" },
    .{ "zig/tests/golden/choice-{\"type\": \"function\"}.json", "{}\n" },
    .{ ".zig-version", "0.17.0\n" },
};

test "an import as upstream has it passes, and a tree without vendored files has nothing to hold" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    try tree.expect(&.{});

    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var found: std.ArrayList([]const u8) = .empty;
    try check(tree.arena.allocator(), testing.io, tmp.dir, &found);
    try testing.expectEqual(@as(usize, 0), found.items.len);
}

test "a changed file needs its row, and a row needs a changed file" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    // the mutation: one byte of a vendored file, and nothing said
    try tree.write("zig/src/server/chat.zig", "const std = @import(\"std\");\n\n");
    try tree.expect(&.{"zig/vendor/one/zig/src/server/chat.zig: differs from upstream (MANIFEST.sha256) and has no row in PATCHES.md"});
    // said: it passes, with or without backticks around the path
    try tree.rows("| `zig/src/server/chat.zig` | a blank line | a test | no |\n");
    try tree.expect(&.{});
    try tree.rows("| zig/src/server/chat.zig | a blank line | a test | no |\n| zig/src/server/chat.zig | a second change | another | #12 |\n");
    try tree.expect(&.{});
    // the file is upstream's again: the row is refused
    try tree.write("zig/src/server/chat.zig", imported[1][1]);
    try tree.expect(&.{"zig/vendor/one/zig/src/server/chat.zig: has a row in PATCHES.md, but its bytes are upstream's: remove the row"});
}

test "a file added or removed here is a change too" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    try tree.write("zig/src/server/refused_test.zig", "test {}\n");
    try tree.write(".hidden", "x");
    try tree.tmp.dir.deleteFile(testing.io, Tree.at ++ ".zig-version");
    try tree.expect(&.{
        "zig/vendor/one/zig/src/server/refused_test.zig: is not upstream's (no line in MANIFEST.sha256) and has no row in PATCHES.md",
        "zig/vendor/one/.hidden: is not upstream's",
        "zig/vendor/one/.zig-version: upstream's file (MANIFEST.sha256) is gone and has no row in PATCHES.md",
    });
    try tree.rows(
        \\| `zig/src/server/refused_test.zig` | added | the 400's test | no |
        \\| `.hidden` | added | a test | no |
        \\| `.zig-version` | removed | not ours | no |
        \\
    );
    try tree.expect(&.{});
    // this repository's own names are its own at the top only
    try tree.write("zig/src/AGENTS.md", "not upstream's\n");
    try tree.expect(&.{"zig/vendor/one/zig/src/AGENTS.md: is not upstream's"});
}

test "a row says what, why and whether it went upstream, and names a file" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    try tree.write("LICENSE", "another licence\n");
    try tree.rows("| `LICENSE` | changed | | no |\n| `zig/src/gone.zig` | changed | why | no |\n| `LICENSE` | changed | why |\n");
    try tree.expect(&.{
        "zig/vendor/one/PATCHES.md: the row of \"LICENSE\" must fill 4 cells",
        "zig/vendor/one/PATCHES.md: the row of \"LICENSE\" must fill 4 cells",
        "zig/vendor/one/PATCHES.md: the row of \"zig/src/gone.zig\" names no file, here or upstream",
    });
}

test "the directory's own files are required, and the manifest is sha256sum's" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    try tree.write(manifest_name, "not a digest  LICENSE\n");
    try tree.expect(&.{
        "zig/vendor/one/MANIFEST.sha256: line 1 is not a sha256 and a path",
        "zig/vendor/one/LICENSE: is not upstream's",
        "zig/vendor/one/zig/src/server/chat.zig: is not upstream's",
        "zig/vendor/one/zig/tests/golden/choice-{\"type\": \"function\"}.json: is not upstream's",
        "zig/vendor/one/.zig-version: is not upstream's",
    });
    for ([_][]const u8{ manifest_name, "UPSTREAM", patches_name, "AGENTS.md" }) |file| {
        var bare: Tree = try .init(&imported);
        defer bare.deinit();
        try bare.tmp.dir.deleteFile(testing.io, try std.fmt.allocPrint(bare.arena.allocator(), Tree.at ++ "{s}", .{file}));
        try bare.expect(&.{try std.fmt.allocPrint(bare.arena.allocator(), "zig/vendor/one/{s} is missing", .{file})});
    }
}

test "a patch ledger rejects a row repeated exactly, and a patched file's unresolved conflict marker" {
    var tree: Tree = try .init(&imported);
    defer tree.deinit();
    try tree.expect(&.{});
    // a file may have several rows; the same row twice is one diagnostic
    try tree.write("zig/src/server/chat.zig", "const std = @import(\"std\");\n\n");
    try tree.rows("| `zig/src/server/chat.zig` | a blank line | a test | no |\n| `zig/src/server/chat.zig` | a blank line | a test | no |\n");
    try tree.expectCount("repeats another row exactly", 1);
    try tree.rows("| `zig/src/server/chat.zig` | a blank line | a test | no |\n| `zig/src/server/chat.zig` | a second change | another | no |\n");
    try tree.expect(&.{});
    // a marker a patch left unmerged is a conflict: its start and end are each a diagnostic
    try tree.write("zig/src/server/chat.zig", "const std = @import(\"std\");\n<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> abc1234\n");
    try tree.rows("| `zig/src/server/chat.zig` | a change | a test | no |\n");
    try tree.expectCount("is an unresolved conflict marker", 2);
}
