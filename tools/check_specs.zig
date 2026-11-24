//! Validates the tracked specs under specs/: filenames, flat frontmatter, lifecycle by location, required headings,
//! continuous IDs, and the indexes in specs/README.md and specs/archive/README.md; and the size budgets of
//! guidance, specs and sources (spec 053), with their exceptions listed here and nowhere else; and which part
//! of `zig/src/` imports which (`check_imports.zig`); and which vendored file is no longer upstream's (`check_vendor.zig`); and the head of
//! each checkpoint's page under docs/recipes/. Every problem of a run is reported together.

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const check_imports = @import("check_imports.zig");
const check_vendor = @import("check_vendor.zig");

const template_name = "000-template.md";
const required_keys = [_][]const u8{ "id", "title", "status", "created", "updated", "coordinator" };
const required_headings = [_][]const u8{
    "Context",
    "Goal and success criteria",
    "Scope and non-goals",
    "Decisions",
    "Workstreams",
    "Verification",
    "Documentation sync",
    "Progress and handoff",
    "Completion summary",
};

/// The size budgets, in bytes (the owner, 2026-10-07; spec 053). What a worker reads before it edits is paid on
/// every task, so each is held here and not by habit.
pub const budget = struct {
    /// The root `AGENTS.md`.
    pub const root_guidance = 10_000;
    /// An `AGENTS.md` below the root.
    pub const nested_guidance = 14_000;
    /// `specs/WORKER.md`, the standing rules a brief points to.
    pub const worker_rules = 7_000;
    /// A spec, active or archived; a longer one is split.
    pub const spec = 60_000;
    /// A file under `source_roots`, unless `exceptions` lists it.
    pub const source = 60_000;
};

/// The directories whose files hold to `budget.source`, and `build.zig` beside them.
const source_roots = [_][]const u8{ "zig", "tools" };

/// Another project's files kept as it publishes them (spec 072): their sizes are upstream's, so `budget.source`
/// does not hold below it. An `AGENTS.md` there is this repository's and keeps its budget.
pub const vendored = check_vendor.root_path;

/// A file allowed over `budget.source`: for good (`until` null: generated, or a fixture) or until the workstream
/// that splits or removes it. An entry whose file is gone or back within the budget is a problem too, so the
/// list only shrinks.
pub const Exception = struct {
    path: []const u8,
    why: []const u8,
    until: ?[]const u8 = null,
};

/// Every file over `budget.source`, in one place. Remove a temporary entry in the commit that splits its file.
pub const exceptions = [_]Exception{
    // generated tables and recorded fixtures: for good
    .{ .path = "zig/src/models/qwen4/draft_vocab.txt", .why = "TensorFold's draft vocabulary, as shipped" },
    .{ .path = "zig/tests/fixtures/audio/cases.json", .why = "fixture from tools/reference/audio_cases.py" },
    .{ .path = "zig/tests/fixtures/image/cases.json", .why = "fixture from tools/reference/image_cases.py" },
    .{ .path = "zig/tests/fixtures/requantize_cases.bin", .why = "fixture from tools/reference/requantize_cases.py" },
    .{ .path = "zig/tests/fixtures/sampling_cases.bin", .why = "fixture from tools/reference/sampling_cases.py" },
    .{ .path = "zig/tests/fixtures/text/grammar_masks.txt", .why = "fixture from tools/reference/grammar_masks.sh" },
    .{ .path = "zig/tests/fixtures/text/grammar_masks_checkpoint.txt", .why = "fixture from tools/reference/grammar_masks.sh" },
    .{ .path = "zig/tests/fixtures/text/message_cases.json", .why = "fixture from tools/reference/message_cases.py" },
    .{ .path = "zig/tests/fixtures/text/template_cases.json", .why = "fixture from tools/reference/template_cases.py" },
    .{ .path = "zig/tests/fixtures/text/tiny_cases.json", .why = "fixture from tools/reference/tokenizer_cases.py" },
    .{ .path = "tools/reference/client_bench_corpus.txt", .why = "client_bench.json's documents, frozen: a prompt's text is what a recorded hash holds" },
    .{ .path = "zig/tests/fixtures/text/tokenizer_cases.json", .why = "fixture from tools/reference/tokenizer_cases.py" },
    .{ .path = "tools/reference/flash_next_export.py", .why = "a fixture's export script, kept whole (the owner, 2026-10-07)" },
};

/// A recipe page's frontmatter, in its order: the keys of `docs/recipes/README.md`, "A page's head".
const recipe_keys = [_][]const u8{
    "checkpoint", "revision", "model",      "family", "storage",  "license",
    "disk_gb",    "machines", "device_gib", "status", "drafters", "variables",
};
pub const recipes_path = "docs/recipes";

const Location = enum { active, archive };

const Spec = struct {
    name: []const u8,
    id: u16,
    location: Location,
};

const Checker = struct {
    arena: Allocator,
    io: Io,
    problems: std.ArrayList([]const u8) = .empty,
    specs: std.ArrayList(Spec) = .empty,

    fn fail(self: *Checker, comptime format: []const u8, args: anytype) Allocator.Error!void {
        try self.problems.append(self.arena, try std.fmt.allocPrint(self.arena, format, args));
    }
};

/// Every problem with the spec tree and the size budgets under `root`, allocated in `arena`; none means it passes.
pub fn check(arena: Allocator, io: Io, root: Io.Dir) ![]const []const u8 {
    return checkWith(arena, io, root, &exceptions);
}

/// `check` with another list of files allowed over `budget.source` (the tests' trees have their own).
pub fn checkWith(arena: Allocator, io: Io, root: Io.Dir, allowed: []const Exception) ![]const []const u8 {
    var checker: Checker = .{ .arena = arena, .io = io };
    try checkSizes(&checker, root, allowed);
    try check_imports.check(arena, io, root, &checker.problems);
    try check_vendor.check(arena, io, root, &checker.problems);
    try checkRecipes(&checker, root);
    var specs_dir = root.openDir(io, "specs", .{ .iterate = true }) catch {
        try checker.fail("specs/ is missing", .{});
        return checker.problems.items;
    };
    defer specs_dir.close(io);

    if (specs_dir.openDir(io, "archive", .{ .iterate = true })) |opened| {
        var archive = opened;
        defer archive.close(io);
        try scan(&checker, archive, .archive);
    } else |_| try checker.fail("specs/archive/ is missing", .{});
    try scan(&checker, specs_dir, .active);

    specs_dir.access(io, template_name, .{}) catch try checker.fail("specs/{s} is missing", .{template_name});
    try checkIds(&checker);
    var files: std.ArrayList(IndexFile) = .empty;
    for ([_]IndexFile{
        .{ .label = "specs/README.md", .text = "README.md", .location = .active },
        .{ .label = "specs/archive/README.md", .text = "archive/README.md", .location = .archive },
    }) |file| {
        if (specs_dir.readFileAlloc(io, file.text, arena, .limited(1 << 20))) |text| {
            try files.append(arena, .{ .label = file.label, .text = text, .location = file.location });
        } else |_| try checker.fail("{s} is missing", .{file.label});
    }
    // without the active index there is no index to hold to its rules
    if (files.items.len > 0 and files.items[0].location == .active) try checkIndex(&checker, specs_dir, files.items);
    return checker.problems.items;
}

fn scan(checker: *Checker, dir: Io.Dir, location: Location) !void {
    const where = if (location == .archive) "specs/archive/" else "specs/";
    var entries = dir.iterate();
    while (try entries.next(checker.io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".md")) continue;
        // the indexes, the directory's guidance, the workers' rules and the template are not specs
        if (location == .archive and std.mem.eql(u8, entry.name, "README.md")) continue;
        if (location == .active) {
            const beside = for ([_][]const u8{ "README.md", "AGENTS.md", "WORKER.md", template_name }) |other| {
                if (std.mem.eql(u8, entry.name, other)) break true;
            } else false;
            if (beside) continue;
        }
        const name = try checker.arena.dupe(u8, entry.name);
        const id = formalId(name) orelse {
            try checker.fail("{s}{s}: filename must be NNN-kebab-case.md", .{ where, name });
            continue;
        };
        if (id == 0) {
            try checker.fail("{s}{s}: ID 000 is reserved for the template", .{ where, name });
            continue;
        }
        try checker.specs.append(checker.arena, .{ .name = name, .id = id, .location = location });
        const text = try dir.readFileAlloc(checker.io, name, checker.arena, .limited(4 << 20));
        if (text.len > budget.spec) try checker.fail("{s}{s}: {d} bytes, over a spec's {d}: split it", .{ where, name, text.len, budget.spec });
        try checkSpec(checker, where, name, id, location, text);
    }
}

/// The size of `path` under `root`, or null when there is no such file.
fn sizeOf(checker: *Checker, root: Io.Dir, path: []const u8) ?u64 {
    const stat = root.statFile(checker.io, path, .{}) catch return null;
    return if (stat.kind == .file) stat.size else null;
}

/// The budgets of guidance and sources. A tree without a file or a directory has nothing to hold there: the
/// spec tree's own checks say what must exist.
fn checkSizes(checker: *Checker, root: Io.Dir, allowed: []const Exception) !void {
    const single = [_]struct { []const u8, u64 }{
        .{ "AGENTS.md", budget.root_guidance },
        .{ "specs/AGENTS.md", budget.nested_guidance },
        .{ "specs/WORKER.md", budget.worker_rules },
        .{ "build.zig", budget.source },
    };
    for (single) |entry| {
        const path, const most = entry;
        const size = sizeOf(checker, root, path) orelse continue;
        if (size > most) try checker.fail("{s}: {d} bytes, over its budget of {d}", .{ path, size, most });
    }
    const used = try checker.arena.alloc(bool, allowed.len);
    @memset(used, false);
    for (source_roots) |name| {
        var dir = root.openDir(checker.io, name, .{ .iterate = true }) catch continue;
        defer dir.close(checker.io);
        try checkTree(checker, dir, name, allowed, used);
    }
    for (allowed, used) |exception, was| {
        if (!was) try checker.fail("{s}: listed as an exception to the size budget, but it is gone or within {d} bytes: remove its entry in tools/check_specs.zig", .{ exception.path, budget.source });
    }
}

/// Every file below `dir`, whose path from the root is `prefix`: an `AGENTS.md` against the nested budget, and
/// each file against the sources' unless `allowed` names it or it is below `vendored`. Hidden entries and Python's
/// caches are not sources.
fn checkTree(checker: *Checker, dir: Io.Dir, prefix: []const u8, allowed: []const Exception, used: []bool) !void {
    var entries = dir.iterate();
    while (try entries.next(checker.io)) |entry| {
        if (entry.name[0] == '.' or std.mem.eql(u8, entry.name, "__pycache__")) continue;
        const path = try std.fmt.allocPrint(checker.arena, "{s}/{s}", .{ prefix, entry.name });
        switch (entry.kind) {
            .directory => {
                var below = try dir.openDir(checker.io, entry.name, .{ .iterate = true });
                defer below.close(checker.io);
                try checkTree(checker, below, path, allowed, used);
            },
            .file => {
                const size = (try dir.statFile(checker.io, entry.name, .{})).size;
                if (std.mem.eql(u8, entry.name, "AGENTS.md") and size > budget.nested_guidance)
                    try checker.fail("{s}: {d} bytes, over its budget of {d}", .{ path, size, budget.nested_guidance });
                if (size <= budget.source or std.mem.startsWith(u8, path, vendored ++ "/")) continue;
                const listed = for (allowed, used) |exception, *was| {
                    if (std.mem.eql(u8, exception.path, path)) {
                        was.* = true;
                        break true;
                    }
                } else false;
                if (!listed) try checker.fail("{s}: {d} bytes, over a file's {d}: split it, or list it in tools/check_specs.zig's exceptions with its reason", .{ path, size, budget.source });
            },
            else => {},
        }
    }
}

/// The pages under docs/recipes/, each but the index: its head and its link from the index. A tree without the
/// directory has no checkpoint to describe.
fn checkRecipes(checker: *Checker, root: Io.Dir) !void {
    var dir = root.openDir(checker.io, recipes_path, .{ .iterate = true }) catch return;
    defer dir.close(checker.io);
    const index = dir.readFileAlloc(checker.io, "README.md", checker.arena, .limited(1 << 20)) catch null;
    if (index == null) try checker.fail("{s}/README.md is missing", .{recipes_path});
    const variables = root.readFileAlloc(checker.io, "docs/environment.md", checker.arena, .limited(1 << 20)) catch "";
    var entries = dir.iterate();
    while (try entries.next(checker.io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".md") or std.mem.eql(u8, entry.name, "README.md")) continue;
        const name = try checker.arena.dupe(u8, entry.name);
        const text = try dir.readFileAlloc(checker.io, name, checker.arena, .limited(4 << 20));
        try checkRecipe(checker, root, name, text, variables);
        const link = try std.fmt.allocPrint(checker.arena, "]({s})", .{name});
        if (index) |listed| if (std.mem.indexOf(u8, listed, link) == null)
            try checker.fail("{s}/README.md does not link {s}", .{ recipes_path, name });
    }
}

/// One page's head: exactly `recipe_keys`, in order, and what each value may be. `variables` is the text of
/// docs/environment.md, where a page's variables have their rows.
fn checkRecipe(checker: *Checker, root: Io.Dir, name: []const u8, text: []const u8, variables: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var values: [recipe_keys.len][]const u8 = undefined;
    var count: usize = 0;
    var sound = if (lines.next()) |first| std.mem.eql(u8, trimEnd(first), "---") else false;
    var closed = false;
    if (sound) while (lines.next()) |raw| {
        const line = trimEnd(raw);
        if (std.mem.eql(u8, line, "---")) {
            closed = true;
            break;
        }
        const key, const value = flatPair(line) orelse {
            try checker.fail("{s}/{s}: frontmatter line \"{s}\" is not a flat key: value", .{ recipes_path, name, line });
            sound = false;
            continue;
        };
        if (count < recipe_keys.len and std.mem.eql(u8, key, recipe_keys[count])) {
            values[count] = value;
        } else if (sound) {
            // the first key out of step is the problem; those after it only follow
            try checker.fail("{s}/{s}: frontmatter key \"{s}\" stands where \"{s}\" belongs", .{
                recipes_path, name, key, if (count < recipe_keys.len) recipe_keys[count] else "---",
            });
            sound = false;
        }
        count += 1;
    };
    if (!closed) {
        try checker.fail("{s}/{s}: frontmatter must open and close with ---", .{ recipes_path, name });
        return;
    }
    if (sound and count < recipe_keys.len) try checker.fail("{s}/{s}: frontmatter key \"{s}\" is missing", .{ recipes_path, name, recipe_keys[count] });
    if (!sound or count != recipe_keys.len) return;

    const revision, const family, const machines, const device, const status, const named = .{ values[1], values[3], values[7], values[8], values[9], values[11] };
    const hex = for (revision) |c| {
        if (!std.ascii.isHex(c)) break false;
    } else revision.len == 40;
    if (!hex) try checker.fail("{s}/{s}: revision \"{s}\" is not a commit's 40 hex digits", .{ recipes_path, name, revision });
    if (!std.mem.eql(u8, machines, "1") and !std.mem.eql(u8, machines, "2"))
        try checker.fail("{s}/{s}: machines \"{s}\" is neither 1 nor 2", .{ recipes_path, name, machines });
    const unverified = std.mem.eql(u8, status, "unverified");
    const retired = std.mem.eql(u8, status, "retired");
    if (!unverified and !retired and !std.mem.eql(u8, status, "supported"))
        try checker.fail("{s}/{s}: status \"{s}\" is none of supported, unverified and retired", .{ recipes_path, name, status });
    // a retired page says when and why in its body (docs/recipes/README.md, "Retiring a checkpoint")
    if (retired and std.mem.indexOf(u8, text, "\n**Retired ") == null)
        try checker.fail("{s}/{s}: a retired page has no **Retired DATE:** line", .{ recipes_path, name });
    if (std.mem.eql(u8, device, "unmeasured")) {
        if (!unverified) try checker.fail("{s}/{s}: device_gib is unmeasured, which only an unverified page may say", .{ recipes_path, name });
    } else {
        // parseFloat alone would take "inf" and "nan"
        const number = std.ascii.isDigit(device[0]) and if (std.fmt.parseFloat(f64, device)) |_| true else |_| false;
        if (!number) try checker.fail("{s}/{s}: device_gib \"{s}\" is neither a number nor unmeasured", .{ recipes_path, name, device });
    }
    const models = try std.fmt.allocPrint(checker.arena, "zig/src/models/{s}", .{family});
    const known = std.mem.indexOfAny(u8, family, "/.") == null and if (root.statFile(checker.io, models, .{})) |stat| stat.kind == .directory else |_| false;
    if (!known) try checker.fail("{s}/{s}: family \"{s}\" is no directory under zig/src/models/", .{ recipes_path, name, family });
    if (std.mem.eql(u8, named, "none")) return;
    var names = std.mem.tokenizeAny(u8, named, ", ");
    while (names.next()) |variable| {
        // backticked, so that a name is not found inside a longer one
        const row = try std.fmt.allocPrint(checker.arena, "`{s}`", .{variable});
        if (std.mem.indexOf(u8, variables, row) == null)
            try checker.fail("{s}/{s}: variable {s} has no row in docs/environment.md", .{ recipes_path, name, variable });
    }
}

/// The ID of `NNN-kebab-case.md`, or null when the name has another form.
fn formalId(name: []const u8) ?u16 {
    if (name.len < "000-a.md".len or name[3] != '-') return null;
    const id = std.fmt.parseInt(u16, name[0..3], 10) catch return null;
    for (name[0..3]) |c| if (!std.ascii.isDigit(c)) return null;
    const slug = name[4 .. name.len - ".md".len];
    if (slug.len == 0 or slug[0] == '-' or slug[slug.len - 1] == '-') return null;
    for (slug, 0..) |c, i| {
        if (c == '-') {
            if (slug[i - 1] == '-') return null;
        } else if (!std.ascii.isLower(c) and !std.ascii.isDigit(c)) return null;
    }
    return id;
}

fn checkSpec(checker: *Checker, where: []const u8, name: []const u8, id: u16, location: Location, text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    var values: [required_keys.len]?[]const u8 = @splat(null);
    const opened = if (lines.next()) |first| std.mem.eql(u8, trimEnd(first), "---") else false;
    var closed = false;
    if (opened) while (lines.next()) |raw| {
        const line = trimEnd(raw);
        if (std.mem.eql(u8, line, "---")) {
            closed = true;
            break;
        }
        const key, const value = flatPair(line) orelse {
            try checker.fail("{s}{s}: frontmatter line \"{s}\" is not a flat key: value", .{ where, name, line });
            continue;
        };
        for (required_keys, &values) |required, *slot| if (std.mem.eql(u8, key, required)) {
            slot.* = value;
        };
    };
    if (!opened or !closed) {
        try checker.fail("{s}{s}: frontmatter must open and close with ---", .{ where, name });
        return;
    }
    for (required_keys, values) |required, value| {
        if (value == null) try checker.fail("{s}{s}: frontmatter key \"{s}\" is missing", .{ where, name, required });
    }

    if (values[0]) |text_id| {
        const matches = if (std.fmt.parseInt(u16, text_id, 10)) |parsed| parsed == id and text_id.len == 3 else |_| false;
        if (!matches) try checker.fail("{s}{s}: frontmatter id \"{s}\" does not match the filename", .{ where, name, text_id });
    }
    if (values[2]) |status| {
        const allowed = switch (location) {
            .active => std.mem.eql(u8, status, "Draft") or std.mem.eql(u8, status, "Active"),
            .archive => std.mem.eql(u8, status, "Completed"),
        };
        if (!allowed) try checker.fail("{s}{s}: status \"{s}\" does not belong in {s} ({s})", .{
            where,                                                       name, status, where,
            if (location == .active) "Draft or Active" else "Completed",
        });
    }
    const created = try checkDate(checker, where, name, "created", values[3]);
    const updated = try checkDate(checker, where, name, "updated", values[4]);
    if (created != null and updated != null and updated.? < created.?)
        try checker.fail("{s}{s}: updated is earlier than created", .{ where, name });

    var counts: [required_headings.len]usize = @splat(0);
    var fenced = false;
    while (lines.next()) |raw| {
        const line = trimEnd(raw);
        if (std.mem.startsWith(u8, line, "```")) {
            fenced = !fenced;
            continue;
        }
        if (fenced or !std.mem.startsWith(u8, line, "## ")) continue;
        const heading = std.mem.trim(u8, line[3..], " ");
        for (required_headings, &counts) |required, *count| if (std.mem.eql(u8, heading, required)) {
            count.* += 1;
        };
    }
    for (required_headings, counts) |required, count| {
        if (count == 0) try checker.fail("{s}{s}: heading \"## {s}\" is missing", .{ where, name, required });
        if (count > 1) try checker.fail("{s}{s}: heading \"## {s}\" appears {d} times", .{ where, name, required, count });
    }
}

fn trimEnd(line: []const u8) []const u8 {
    return std.mem.trimEnd(u8, line, "\r \t");
}

/// A `key: value` line with a scalar value, quotes removed; null for anything structured or empty.
fn flatPair(line: []const u8) ?struct { []const u8, []const u8 } {
    if (line.len == 0 or line[0] == ' ' or line[0] == '\t' or line[0] == '-' or line[0] == '#') return null;
    const colon = std.mem.indexOfScalar(u8, line, ':') orelse return null;
    const key = line[0..colon];
    if (key.len == 0) return null;
    for (key) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return null;
    var value = std.mem.trim(u8, line[colon + 1 ..], " \t");
    if (value.len == 0) return null;
    // arrays, maps, block scalars, anchors, aliases and tags are structure this system does not read
    if (std.mem.indexOfScalar(u8, "[{|>&*!", value[0]) != null) return null;
    if (value.len >= 2 and (value[0] == '"' or value[0] == '\'') and value[value.len - 1] == value[0]) value = value[1 .. value.len - 1];
    if (value.len == 0) return null;
    return .{ key, value };
}

/// Days since a fixed origin for a real `YYYY-MM-DD`, so two dates compare.
fn checkDate(checker: *Checker, where: []const u8, name: []const u8, key: []const u8, value: ?[]const u8) !?u32 {
    const text = value orelse return null;
    const day = parseDate(text) orelse {
        try checker.fail("{s}{s}: {s} \"{s}\" is not a real YYYY-MM-DD date", .{ where, name, key, text });
        return null;
    };
    return day;
}

fn parseDate(text: []const u8) ?u32 {
    if (text.len != 10 or text[4] != '-' or text[7] != '-') return null;
    for (text, 0..) |c, i| if (i != 4 and i != 7 and !std.ascii.isDigit(c)) return null;
    const year = std.fmt.parseInt(u32, text[0..4], 10) catch return null;
    const month = std.fmt.parseInt(u32, text[5..7], 10) catch return null;
    const day = std.fmt.parseInt(u32, text[8..10], 10) catch return null;
    if (month < 1 or month > 12 or day < 1) return null;
    const leap = (year % 4 == 0 and year % 100 != 0) or year % 400 == 0;
    const lengths = [12]u32{ 31, if (leap) 29 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
    if (day > lengths[month - 1]) return null;
    return (year * 12 + month) * 31 + day;
}

fn checkIds(checker: *Checker) !void {
    const specs = checker.specs.items;
    std.mem.sort(Spec, specs, {}, struct {
        fn before(_: void, a: Spec, b: Spec) bool {
            return a.id < b.id;
        }
    }.before);
    var expected: u16 = 1;
    for (specs, 0..) |spec, i| {
        if (i > 0 and specs[i - 1].id == spec.id) {
            try checker.fail("ID {d:0>3} is used by both {s} and {s}", .{ spec.id, specs[i - 1].name, spec.name });
            continue;
        }
        if (spec.id != expected) try checker.fail("IDs are not continuous: expected {d:0>3} before {s}", .{ expected, spec.name });
        expected = spec.id + 1;
    }
}

/// One of the two index files: `specs/README.md` indexes the active specs, `specs/archive/README.md` the
/// archived ones, each by links relative to itself.
const IndexFile = struct { label: []const u8, text: []const u8, location: Location };

fn checkIndex(checker: *Checker, specs_dir: Io.Dir, files: []const IndexFile) !void {
    var template_links: usize = 0;
    const seen = try checker.arena.alloc(usize, checker.specs.items.len);
    @memset(seen, 0);

    for (files) |file| {
        var section: ?Location = null;
        var last: u16 = 0;
        var lines = std.mem.splitScalar(u8, file.text, '\n');
        while (lines.next()) |raw| {
            const line = trimEnd(raw);
            if (std.mem.startsWith(u8, line, "## ")) {
                const heading = line[3..];
                section = if (std.mem.eql(u8, heading, "Active specs")) .active else if (std.mem.eql(u8, heading, "Archived specs")) .archive else null;
                continue;
            }
            var rest = line;
            while (std.mem.indexOf(u8, rest, "](")) |open| {
                const after = rest[open + 2 ..];
                const close = std.mem.indexOfScalar(u8, after, ')') orelse break;
                const target = after[0..close];
                rest = after[close + 1 ..];
                if (std.mem.eql(u8, target, template_name)) {
                    if (file.location == .active) template_links += 1;
                    continue;
                }
                // a link from one index to the other's directory names a spec in the other's place
                const across = if (file.location == .active) "archive/" else "../";
                const crossed = std.mem.startsWith(u8, target, across);
                const name = if (crossed) target[across.len..] else target;
                const id = formalId(name) orelse continue;
                const location: Location = if (crossed) (if (file.location == .active) .archive else .active) else file.location;
                const path = if (location == .archive) try std.fmt.allocPrint(checker.arena, "archive/{s}", .{name}) else name;
                specs_dir.access(checker.io, path, .{}) catch {
                    try checker.fail("{s}: link {s} does not resolve", .{ file.label, target });
                    continue;
                };
                if (location != file.location or section != location) {
                    try checker.fail("{s}: {s} is indexed in the wrong section", .{ file.label, target });
                } else {
                    if (id < last) try checker.fail("{s}: {s} is out of ascending ID order", .{ file.label, target });
                    last = id;
                }
                for (checker.specs.items, seen) |spec, *count| {
                    if (spec.location == location and std.mem.eql(u8, spec.name, name)) count.* += 1;
                }
            }
        }
    }
    if (template_links != 1) try checker.fail("specs/README.md links {s} {d} times, not once", .{ template_name, template_links });
    for (checker.specs.items, seen) |spec, count| {
        const index = if (spec.location == .archive) "specs/archive/README.md" else "specs/README.md";
        if (count == 0) try checker.fail("{s} does not index {s}", .{ index, spec.name });
        if (count > 1) try checker.fail("{s} indexes {s} {d} times", .{ index, spec.name, count });
    }
}

pub fn main(init: std.process.Init) !u8 {
    var buffer: [4096]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};

    const problems = try check(init.arena.allocator(), init.io, .cwd());
    if (problems.len == 0) {
        try stdout.writeAll("Specs check passed.\n");
        return 0;
    }
    try stdout.writeAll("Specs check failed:\n");
    for (problems) |problem| try stdout.print("- {s}\n", .{problem});
    return 1;
}

// Tests run the validator on spec trees written to a temporary directory: `check_specs_test.zig`.

test {
    _ = check_imports;
    _ = check_vendor;
    _ = @import("host_descriptor.zig");
    _ = @import("check_specs_test.zig");
}
