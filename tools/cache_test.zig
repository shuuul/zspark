//! cache.zig's contract tests (spec 168 WS-04). Every root is a temporary directory the test itself
//! fills and removes; no real cache is touched and no `--apply` runs against the machine. Each test
//! stands up a fake home (`tmp`): `home/.cache/zspark/NAME-cache` qualifies as a known generated
//! root, as does `home/repo/proj/.zig-cache` beside checkout markers. `Policy.driver_exempt` lets the
//! test binary's own `zig` driver pass the build-driver refusal the CLI keeps.

const std = @import("std");
const linux = std.os.linux;
const cache = @import("cache.zig");
const Io = std.Io;
const Dir = Io.Dir;
const Timestamp = Io.Timestamp;
const testing = std.testing;

const max_path = linux.PATH_MAX;
const ns_day = 24 * std.time.ns_per_hour;

fn writeAt(dir: Dir, io: Io, sub: []const u8, data: []const u8) !void {
    try dir.writeFile(io, .{ .sub_path = sub, .data = data });
}

/// Every path under `dir`'s `sub` aged `days`, leaves before parents so a parent's utimens does not
/// re-touch a child.
fn ageTree(io: Io, dir: Dir, sub: []const u8, days: u32) !void {
    var children: std.ArrayList([]const u8) = .empty;
    defer {
        for (children.items) |n| testing.allocator.free(n);
        children.deinit(testing.allocator);
    }
    if (dir.openDir(io, sub, .{ .iterate = true, .follow_symlinks = false })) |opened| {
        var child = opened;
        defer child.close(io);
        var it = child.iterate();
        while (try it.next(io)) |item| {
            try children.append(testing.allocator, try std.fmt.allocPrint(testing.allocator, "{s}/{s}", .{ sub, item.name }));
        }
        for (children.items) |n| try ageTree(io, dir, n, days);
    } else |err| switch (err) {
        error.NotDir, error.SymLinkLoop => {},
        else => |e| return e,
    }
    var buffer: [512]u8 = undefined;
    const sub_z = try std.fmt.bufPrintSentinel(&buffer, "{s}", .{sub}, 0);
    const past = Timestamp.now(io, .real).toSeconds() - @as(i64, days) * std.time.s_per_day;
    const times = [2]linux.timespec{ .{ .sec = past, .nsec = 0 }, .{ .sec = past, .nsec = 0 } };
    const rc = linux.utimensat(dir.handle, sub_z, &times, linux.AT.SYMLINK_NOFOLLOW);
    try testing.expectEqual(linux.E.SUCCESS, linux.errno(rc));
}

fn mkTree(io: Io, dir: Dir) !void {
    try dir.createDirPath(io, "repo/proj/zig");
    try dir.createDirPath(io, "repo/proj/tools");
    try writeAt(dir, io, "repo/proj/build.zig.zon", ".{}");
    try dir.createDirPath(io, "repo/proj/.zig-cache/o/aa/sub");
    try writeAt(dir, io, "repo/proj/.zig-cache/o/aa/f", "aaaa");
    try dir.createDirPath(io, "repo/proj/.zig-cache/o/bb");
    try writeAt(dir, io, "repo/proj/.zig-cache/o/bb/f", "bb");
    try dir.createDirPath(io, "repo/proj/.zig-cache/tmp");
    try dir.createDirPath(io, "repo/proj/.zig-cache/z");
    try writeAt(dir, io, "repo/proj/.zig-cache/z/z1", "state");
    try dir.createDirPath(io, "repo/proj/.zig-cache/p/pkg1");
    try writeAt(dir, io, "repo/proj/.zig-cache/p/pkg1/f", "package");
    try dir.createDirPath(io, "repo/proj/.zig-cache/h");
    try writeAt(dir, io, "repo/proj/.zig-cache/h/h1", "manifest");
    try dir.createDirPath(io, "repo/proj/.zig-cache/c");
    try dir.createDirPath(io, "repo/proj/zig-out/bin");
    try writeAt(dir, io, "repo/proj/zig-out/bin/zspark", "binary");
    try dir.createDirPath(io, "repo/proj/zig-out/fast");
    try writeAt(dir, io, "repo/proj/zig-out/fast/zspark-fast", "fast binary");
    try dir.createDirPath(io, "repo/proj/zig-out/logs");
    try writeAt(dir, io, "repo/proj/zig-out/logs/build.log", "kept");
    try dir.createDirPath(io, ".cache/zspark/store/scratch-168");
    try writeAt(dir, io, ".cache/zspark/store/scratch-168/f", "evidence");
    try dir.createDirPath(io, ".cache/zspark/store/deploy-6a886889");
    try writeAt(dir, io, ".cache/zspark/store/deploy-6a886889/bin", "deployed");
    try dir.createDirPath(io, ".cache/zspark/store/logs");
    try writeAt(dir, io, ".cache/zspark/store/logs/a.log", "log");
    try dir.createDirPath(io, ".cache/zspark/store/triton-cache");
    try writeAt(dir, io, ".cache/zspark/store/triton-cache/k", "kernel");
    try dir.createDirPath(io, ".cache/zspark/store/.zig-cache");
    try writeAt(dir, io, ".cache/zspark/store/machine.lock", "");
    try writeAt(dir, io, ".cache/zspark/store/env.sh", "X=1");
}

fn findRow(rows: []const cache.Planned, name: []const u8) ?cache.Planned {
    for (rows) |row| if (std.mem.eql(u8, row.entry.name, name)) return row;
    return null;
}

fn findEntry(root: cache.Root, name: []const u8) ?cache.Entry {
    for (root.entries) |e| if (std.mem.eql(u8, e.name, name)) return e;
    return null;
}

/// `dir`'s own resolved path, for the fake home.
fn homeOf(io: Io, dir: Dir, buffer: *[max_path]u8) ![]const u8 {
    const n = try dir.realPath(io, buffer);
    return buffer[0..n];
}

/// `scanRoot` of `sub` under tmp, resolved, with tmp as the fake home.
fn scan(arena: std.mem.Allocator, io: Io, dir: Dir, sub: []const u8) !cache.Root {
    var buffer: [max_path]u8 = undefined;
    const home = try homeOf(io, dir, &buffer);
    var root_buffer: [max_path]u8 = undefined;
    const n = try dir.realPathFile(io, sub, &root_buffer);
    return cache.scanRoot(arena, io, root_buffer[0..n], home);
}

test "a zig cache reports its stores: artifacts eligible, packages protected" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try mkTree(io, tmp.dir);

    const root = try scan(arena, io, tmp.dir, "repo/proj/.zig-cache");
    try testing.expectEqual(cache.RootKind.zig_cache, root.kind);
    try testing.expectEqual(cache.Class.zig_artifact, findEntry(root, "o").?.class);
    try testing.expectEqual(cache.Class.zig_artifact, findEntry(root, "z").?.class);
    try testing.expectEqual(cache.Class.package_store, findEntry(root, "p").?.class);
    try testing.expectEqual(cache.Class.package_store, findEntry(root, "h").?.class);
    try testing.expectEqual(cache.Class.package_store, findEntry(root, "c").?.class);
    try testing.expect(findEntry(root, "o").?.bytes > 0);
    const want = findEntry(root, "o").?.bytes + findEntry(root, "tmp").?.bytes + findEntry(root, "z").?.bytes;
    try testing.expectEqual(root.eligible_bytes, want);
    // Inside a verified checkout (build.zig.zon, zig/, tools/), removal is permitted.
    try testing.expect(root.allowed);
}

test "a generic root keeps evidence, deployments, logs and locks; nested caches are not entries" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try mkTree(io, tmp.dir);

    const root = try scan(arena, io, tmp.dir, ".cache/zspark/store");
    try testing.expectEqual(cache.RootKind.generic, root.kind);
    try testing.expectEqual(cache.Class.evidence, findEntry(root, "scratch-168").?.class);
    try testing.expectEqual(cache.Class.deployment, findEntry(root, "deploy-6a886889").?.class);
    try testing.expectEqual(cache.Class.logs, findEntry(root, "logs").?.class);
    try testing.expectEqual(cache.Class.locks, findEntry(root, "machine.lock").?.class);
    try testing.expectEqual(cache.Class.environment, findEntry(root, "env.sh").?.class);
    try testing.expectEqual(cache.Class.generated_cache, findEntry(root, "triton-cache").?.class);
    // A nested .zig-cache is its own root: select it directly, never remove it whole here.
    try testing.expectEqual(cache.Class.nested_root, findEntry(root, ".zig-cache").?.class);
    try testing.expectEqual(@as(usize, 1), root.eligible.len);
    // `store` is a child of ~/.cache/zspark but not a regenerable name: report-only.
    try testing.expect(!root.applyable());
}

test "the plan takes the oldest eligible first until the byte bound holds" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/triton-cache/old");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/triton-cache/old/f", "1234567890");
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/upstream/new");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/upstream/new/f", "1234567890");

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "triton-cache", 30);
    try ageTree(io, dir, "upstream", 10);
    root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");

    const now = Timestamp.now(io, .real).toNanoseconds();
    try testing.expect(root.allowed);
    try testing.expectEqual(@as(usize, 2), root.eligible.len);
    try testing.expectEqualStrings("triton-cache", root.eligible[0].name); // oldest first

    // The bound admits removing only the first entry.
    const p1 = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .max_bytes = root.eligible[1].bytes, .lock = "-" }, now);
    try testing.expectEqual(@as(usize, 2), p1.rows.len);
    try testing.expectEqual(cache.Action.remove, p1.rows[0].action);
    try testing.expectEqualStrings("triton-cache", p1.rows[0].entry.name);
    try testing.expectEqual(cache.Action.keep, p1.rows[1].action);
    try testing.expect(!p1.unmet);

    // Both are too young for a tighter age bound; the byte bound is then unmet and reported.
    const p2 = try cache.plan(arena, io, &root, .{ .older_than_ns = 40 * ns_day, .max_bytes = 1, .lock = "-" }, now);
    try testing.expectEqual(@as(u64, 0), p2.removing);
    try testing.expect(p2.unmet);

    // Age alone removes everything eligible past it.
    const p3 = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .lock = "-" }, now);
    try testing.expectEqual(@as(usize, 2), p3.rows.len);
    for (p3.rows) |row| try testing.expectEqual(cache.Action.remove, row.action);
}

test "a size bound alone still keeps an entry touched in the last minute" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/fresh-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/fresh-cache/f", "just written");
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/stale-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/stale-cache/f", "older");

    var dir = try tmp.dir.openDir(io, ".cache/zspark/pool-cache", .{});
    defer dir.close(io);
    try ageTree(io, dir, "stale-cache", 5);

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    const now = Timestamp.now(io, .real).toNanoseconds();
    // No --older-than: the one-minute floor alone protects the fresh entry.
    const p = try cache.plan(arena, io, &root, .{ .max_bytes = 0, .lock = "-" }, now);
    for (p.rows) |row| {
        if (std.mem.eql(u8, row.entry.name, "fresh-cache")) {
            try testing.expectEqual(cache.Action.keep, row.action);
            try testing.expectEqualStrings("younger than the age bound", row.reason);
        } else {
            try testing.expectEqual(cache.Action.remove, row.action);
        }
    }
}

test "a symlinked entry is unlinked never followed, even out of the root" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // The escaping link's target lives outside the root at tmp/outside: three levels up from
    // .cache/zspark/thing-cache. The other eligible link's target lives inside it.
    try tmp.dir.createDirPath(io, "outside");
    try writeAt(tmp.dir, io, "outside/weights.txt", "model weights");
    try tmp.dir.createDirPath(io, ".cache/zspark/thing-cache/data");
    try writeAt(tmp.dir, io, ".cache/zspark/thing-cache/data/payload.bin", "payload inside root");
    var dir = try tmp.dir.openDir(io, ".cache/zspark/thing-cache", .{});
    defer dir.close(io);
    try dir.symLink(io, "../../../outside", "escape-cache", .{});
    try dir.symLink(io, "data", "link-cache", .{});

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/thing-cache");
    try testing.expect(root.allowed);
    for ([_][]const u8{ "escape-cache", "link-cache" }) |name| {
        const e = findEntry(root, name).?;
        try testing.expect(!e.is_dir);
        try testing.expect(e.bytes < 4096); // the link itself, never its target
    }
    try testing.expectEqualStrings("../../../outside", findEntry(root, "escape-cache").?.link_target.?);
    try testing.expectEqualStrings("data", findEntry(root, "link-cache").?.link_target.?);
    try testing.expectEqual(@as(usize, 2), root.eligible.len); // both links are generated entries

    var root_dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer root_dir.close(io);
    try ageTree(io, root_dir, "escape-cache", 20); // ages the link itself (NOFOLLOW), never its target
    try ageTree(io, root_dir, "link-cache", 20);
    root = try scan(arena, io, tmp.dir, ".cache/zspark/thing-cache");

    const now = Timestamp.now(io, .real).toNanoseconds();
    const policy: cache.Policy = .{ .older_than_ns = 5 * ns_day, .lock = "-", .driver_exempt = true };
    const p = try cache.plan(arena, io, &root, policy, now);
    try testing.expectEqual(@as(usize, 2), p.rows.len);
    for (p.rows) |row| try testing.expectEqual(cache.Action.remove, row.action);
    const removed = try cache.apply(arena, io, &root, policy, p, null);
    try testing.expectEqual(@as(usize, 2), removed.len);
    for (removed) |row| try testing.expectEqual(cache.Outcome.removed, row.outcome);

    // Both links are gone; both target trees are intact — removal unlinked each link and never
    // followed it, across the root boundary and within it alike.
    try testing.expectError(error.FileNotFound, root_dir.access(io, "link-cache", .{}));
    try testing.expectError(error.FileNotFound, root_dir.access(io, "escape-cache", .{}));
    try root_dir.access(io, "data/payload.bin", .{});
    const weights = try tmp.dir.readFileAlloc(io, "outside/weights.txt", arena, .limited(1 << 20));
    try testing.expectEqualStrings("model weights", weights);
}

test "an install root prunes generated bin and fast, keeps protected names" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try mkTree(io, tmp.dir);

    var root = try scan(arena, io, tmp.dir, "repo/proj/zig-out");
    try testing.expectEqual(cache.RootKind.zig_out, root.kind);
    try testing.expect(root.allowed); // beside a verified checkout
    try testing.expectEqual(cache.Class.generated_install, findEntry(root, "bin").?.class);
    try testing.expectEqual(cache.Class.generated_install, findEntry(root, "fast").?.class);
    try testing.expectEqual(cache.Class.logs, findEntry(root, "logs").?.class); // protected stays on top
    try testing.expectEqual(@as(usize, 2), root.eligible.len);

    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "bin", 20);
    try ageTree(io, dir, "fast", 20);
    root = try scan(arena, io, tmp.dir, "repo/proj/zig-out");

    // A byte bound just tight enough admits only the oldest entry; the plan stops there. A cap
    // below the protected bytes cannot be met at all and is reported unmet.
    var total: u64 = 0;
    for (root.entries) |e| total += e.bytes;
    const now = Timestamp.now(io, .real).toNanoseconds();
    // driver_exempt: the fixture lives under the zig-test driver's own cache tmp, so the driver
    // counts as a real zig build overlapping this root — exempt it as the plan's own caller.
    const bounded = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .max_bytes = total - root.eligible[0].bytes, .lock = "-", .driver_exempt = true }, now);
    // Rows may cover every entry, protected ones kept first: select by name, never by index.
    try testing.expectEqual(cache.Action.remove, findRow(bounded.rows, root.eligible[0].name).?.action);
    try testing.expectEqual(cache.Action.keep, findRow(bounded.rows, root.eligible[1].name).?.action);
    try testing.expect(!bounded.unmet);
    const hopeless = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .max_bytes = 0, .lock = "-", .driver_exempt = true }, now);
    try testing.expect(hopeless.unmet);

    const policy: cache.Policy = .{ .older_than_ns = 5 * ns_day, .lock = "-", .driver_exempt = true };
    const p = try cache.plan(arena, io, &root, policy, now);
    try testing.expectEqual(cache.Action.remove, findRow(p.rows, "bin").?.action);
    try testing.expectEqual(cache.Action.remove, findRow(p.rows, "fast").?.action);
    const removed = try cache.apply(arena, io, &root, policy, p, null);
    var removed_count: usize = 0;
    for (removed) |row| if (row.outcome == .removed) {
        removed_count += 1;
        try testing.expect(!std.mem.eql(u8, row.entry.name, "logs"));
    };
    try testing.expectEqual(@as(usize, 2), removed_count);
    try testing.expectError(error.FileNotFound, dir.access(io, "bin", .{}));
    try testing.expectError(error.FileNotFound, dir.access(io, "fast", .{}));
    var kept = try dir.openDir(io, "logs", .{}); // the protected name was never a candidate
    kept.close(io);
}

test "a cache-named directory under a protected path, or anywhere unlisted, refuses removal" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const now = Timestamp.now(io, .real).toNanoseconds();

    // A generated-looking directory nested under a deployment.
    try tmp.dir.createDirPath(io, "deploy-9a62f53f/tool-cache");
    try writeAt(tmp.dir, io, "deploy-9a62f53f/tool-cache/f", "inside a deployment");
    var root = try scan(arena, io, tmp.dir, "deploy-9a62f53f/tool-cache");
    try testing.expect(root.protected);
    const p = try cache.plan(arena, io, &root, .{ .older_than_ns = 0, .lock = "-" }, now);
    try testing.expect(p.refused);
    try testing.expectEqual(@as(u64, 0), p.removing);
    try testing.expectError(error.RootProtected, cache.apply(arena, io, &root, .{ .lock = "-" }, p, null));
    var kept = try tmp.dir.openDir(io, "deploy-9a62f53f/tool-cache", .{});
    kept.close(io);

    // The reviewer's case: `model-cache` under a models directory is weights, whatever its name.
    try tmp.dir.createDirPath(io, "models/model-cache");
    try writeAt(tmp.dir, io, "models/model-cache/shard", "weights");
    var model = try scan(arena, io, tmp.dir, "models/model-cache");
    try testing.expect(model.protected);
    const p2 = try cache.plan(arena, io, &model, .{ .older_than_ns = 0, .lock = "-" }, now);
    try testing.expect(p2.refused);
    try testing.expectError(error.RootProtected, cache.apply(arena, io, &model, .{ .lock = "-" }, p2, null));

    // A `*-cache` name away from every known location is report-only, not prunable.
    try tmp.dir.createDirPath(io, "elsewhere/pool-cache");
    try writeAt(tmp.dir, io, "elsewhere/pool-cache/f", "bytes");
    var other = try scan(arena, io, tmp.dir, "elsewhere/pool-cache");
    try testing.expect(!other.protected);
    const p3 = try cache.plan(arena, io, &other, .{ .older_than_ns = 0, .lock = "-" }, now);
    try testing.expect(p3.refused);
    try testing.expectEqual(@as(u64, 0), p3.removing);
    try testing.expectError(error.NotACache, cache.apply(arena, io, &other, .{ .lock = "-" }, p3, null));

    // `.zig-cache` outside a verified checkout is report-only too.
    try tmp.dir.createDirPath(io, "scratch/zig-cache/o/x");
    var unverified = try scan(arena, io, tmp.dir, "scratch/zig-cache");
    try testing.expectEqual(cache.RootKind.zig_cache, unverified.kind);
    try testing.expect(!unverified.applyable());
}

test "a process's open file keeps its entry; an unheld one is removed" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/held-cache/inner");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/held-cache/inner/f", "stale bytes");
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/free-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/free-cache/f", "free bytes");

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "held-cache", 20);
    try ageTree(io, dir, "free-cache", 20);
    root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");

    // The test itself holds an open fd inside held-cache: a live-process hit like a server's.
    const held = try Dir.openFileAbsolute(io, try std.fmt.allocPrint(arena, "{s}/held-cache/inner/f", .{root.path}), .{});
    defer held.close(io);

    const now = Timestamp.now(io, .real).toNanoseconds();
    const policy: cache.Policy = .{ .older_than_ns = 5 * ns_day, .lock = "-", .driver_exempt = true };
    const p = try cache.plan(arena, io, &root, policy, now);
    for (p.rows) |row| {
        if (std.mem.eql(u8, row.entry.name, "held-cache"))
            try testing.expectEqual(cache.Action.keep, row.action)
        else
            try testing.expectEqual(cache.Action.remove, row.action);
    }
    const removed = try cache.apply(arena, io, &root, policy, p, null);
    try testing.expectEqual(@as(usize, 1), removed.len);
    try testing.expectEqualStrings("free-cache", removed[0].entry.name);
    try testing.expectEqual(cache.Outcome.removed, removed[0].outcome);
    var kept = try dir.openDir(io, "held-cache", .{});
    kept.close(io);
}

test "a held path ending in -cache survives the (deleted) suffix: the name is not trimmed" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // `served-cache` recreated after a file inside it was opened: /proc reports the held path with
    // a ` (deleted)` suffix. A char-set trim would corrupt the name to `served-cach` and the new
    // directory would be removed while still recognizably in use.
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/served-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/served-cache/f", "held bytes");
    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    const held_path = try std.fmt.allocPrint(arena, "{s}/served-cache/f", .{root.path});
    const held = try Dir.openFileAbsolute(io, held_path, .{});
    defer held.close(io);
    try tmp.dir.deleteTree(io, ".cache/zspark/pool-cache/served-cache");
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/served-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/served-cache/g", "new generation");

    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "served-cache", 20);
    root = try cache.scanRoot(arena, io, root.path, root.path[0..std.mem.indexOf(u8, root.path, "/.cache/zspark").?]);

    const now = Timestamp.now(io, .real).toNanoseconds();
    const p = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .lock = "-" }, now);
    try testing.expectEqual(@as(usize, 1), p.rows.len);
    try testing.expectEqual(cache.Action.keep, p.rows[0].action);
    try testing.expectEqualStrings("in use by a running process", p.rows[0].reason);
}

test "an entry newly held between the plan and its removal is skipped" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/late-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/late-cache/f", "old");

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "late-cache", 20);
    root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");

    const now = Timestamp.now(io, .real).toNanoseconds();
    const policy: cache.Policy = .{ .older_than_ns = 5 * ns_day, .lock = "-", .driver_exempt = true };
    const p = try cache.plan(arena, io, &root, policy, now);
    try testing.expectEqual(cache.Action.remove, p.rows[0].action);
    // The entry comes into use after the plan was made; the rescan before removal keeps it.
    const held = try Dir.openFileAbsolute(io, try std.fmt.allocPrint(arena, "{s}/late-cache/f", .{root.path}), .{});
    defer held.close(io);
    const removed = try cache.apply(arena, io, &root, policy, p, null);
    try testing.expectEqual(@as(usize, 1), removed.len);
    try testing.expectEqual(cache.Outcome.skipped, removed[0].outcome);
    try testing.expectEqualStrings("in use by a running process", removed[0].why);
    var kept = try dir.openDir(io, "late-cache", .{});
    kept.close(io);
}

test "a child process whose cwd lies in an entry keeps it" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/served-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/served-cache/f", "old");

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "served-cache", 20);

    // A standing process inside the entry: what a server looks like to the scan. A raw
    // fork+chdir with a pipe handshake, so the cwd is set before the plan runs; one defer both
    // kills the child and reaps it, never the reverse order.
    var fds: [2]std.posix.fd_t = undefined;
    try testing.expectEqual(linux.E.SUCCESS, linux.errno(linux.pipe(&fds)));
    const child_pid = linux.fork();
    try testing.expect(linux.errno(child_pid) == .SUCCESS);
    if (child_pid == 0) {
        _ = linux.close(fds[0]);
        var cwd_buffer: [512]u8 = undefined;
        const cwd_z = std.fmt.bufPrintSentinel(&cwd_buffer, "{s}/served-cache", .{root.path}, 0) catch unreachable;
        _ = linux.chdir(cwd_z);
        _ = linux.write(fds[1], "r", 1);
        while (true) _ = linux.pause();
    }
    _ = linux.close(fds[1]);
    defer {
        _ = linux.kill(@intCast(child_pid), .KILL);
        var status: i32 = undefined;
        _ = linux.waitpid(@intCast(child_pid), &status, 0);
        _ = linux.close(fds[0]);
    }
    var ready: [1]u8 = undefined;
    _ = linux.read(fds[0], &ready, 1); // the child's chdir has completed

    const now = Timestamp.now(io, .real).toNanoseconds();
    const p = try cache.plan(arena, io, &root, .{ .older_than_ns = 5 * ns_day, .lock = "-" }, now);
    try testing.expectEqual(@as(usize, 1), p.rows.len);
    try testing.expectEqual(cache.Action.keep, p.rows[0].action);
    try testing.expectEqual(@as(usize, 1), p.blocked);
}

test "apply skips an entry touched between the scan and the delete" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, ".cache/zspark/pool-cache/x-cache");
    try writeAt(tmp.dir, io, ".cache/zspark/pool-cache/x-cache/f", "old");

    var root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");
    var dir = try Dir.openDirAbsolute(io, root.path, .{});
    defer dir.close(io);
    try ageTree(io, dir, "x-cache", 20);
    root = try scan(arena, io, tmp.dir, ".cache/zspark/pool-cache");

    const now = Timestamp.now(io, .real).toNanoseconds();
    const policy: cache.Policy = .{ .older_than_ns = 5 * ns_day, .lock = "-", .driver_exempt = true };
    const p = try cache.plan(arena, io, &root, policy, now);
    // A competing writer updates the entry after the plan: revalidation skips it.
    try writeAt(dir, io, "x-cache/f", "new");
    const removed = try cache.apply(arena, io, &root, policy, p, null);
    try testing.expectEqual(@as(usize, 1), removed.len);
    try testing.expectEqual(cache.Outcome.skipped, removed[0].outcome);
    try testing.expectEqualStrings("changed after the scan", removed[0].why);
    var kept = try dir.openDir(io, "x-cache", .{});
    kept.close(io);
}

test "a held shared lock refuses the exclusive one a build takes" {
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const io = testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var buffer: [max_path]u8 = undefined;
    const n = try tmp.dir.realPath(io, &buffer);
    const lock_path = try std.fmt.allocPrint(arena, "{s}/cache.lock", .{buffer[0..n]});

    // The build's shared hold: exclusive_nb fails against it, shared_block would queue behind an
    // exclusive one but coexist with this hold.
    const build = try Dir.createFileAbsolute(io, lock_path, .{});
    defer build.close(io);
    try testing.expectEqual(linux.E.SUCCESS, linux.errno(linux.flock(build.handle, std.posix.LOCK.SH)));

    try testing.expectError(error.BuildActive, cache.takeLock(io, lock_path, null, .exclusive_nb));
}
