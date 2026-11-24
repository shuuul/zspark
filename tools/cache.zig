//! Cache inspection and bounded pruning for this machine's generated caches (spec 168 WS-04). What it
//! removes is built or downloaded bytes a tool produces again on demand; what it preserves — deployments,
//! model weights, package caches, reference fixtures and the measurement logs — is never eligible.
//!
//! `zspark-cache` is the installed binary. `--apply` runs under the exclusive `cache.lock` and refuses a
//! process anywhere below a `zig` build driver: do not prune through `zig build cache -- --apply`, and a
//! build run under `tools/check.sh` cannot be pruned against anyway — its wrapper holds the shared lock
//! the exclusive one fails on. Report (`zig build cache` or the binary) takes only machine.lock shared.
//!
//!   zspark-cache                                   report of this machine's known caches
//!   zspark-cache ROOT                              report of ROOT, each entry classified
//!   zspark-cache ROOT --older-than DAYS            report plus the pruning plan; nothing removed
//!   zspark-cache ROOT --max-bytes BYTES            as above, bounded to BYTES retained
//!   zspark-cache ROOT --older-than D --apply       the plan, then removal (both bounds may appear)
//!
//! An entry is a child of the selected root, classified by name. A zig cache (`.zig-cache`,
//! `zig-cache*`, or a directory holding zig's `o/` and `h/` stores) has its `o/`, `tmp/` and `z/`
//! children eligible and its `b/`, `p/`, `h/` and `c/` children protected (downloaded packages,
//! manifests, incremental state). A `zig-out` install root's children are eligible — every entry is
//! generated — except protected names and nested cache roots. Elsewhere:
//! `*-cache`, `*_cache`, `upstream*` and `tensorfold-*` checkouts are regenerable; `scratch-*` (spec
//! evidence), `deploy-*`, `reference-*`, `logs`, `zig-pkg*`, `hub`, `xet`, `.git`, lock and env files,
//! `full-passed`, venvs and serving state (`learned`, `spilled`, `recall`) are protected. A child that
//! is itself a cache or install root (`.zig-cache`, `zig-out`) is `nested_root` — prune it by selecting
//! it, never as an entry of its parent. What no rule recognizes is `unknown` and never removed.
//!
//! `--apply` accepts only a known generated root, anything else is report-only:
//!   - `~/.cache/zig`, zig's global cache;
//!   - `~/.cache/zspark`, and its direct children whose names are regenerable
//!     (`*-cache`, `*_cache`, `upstream*`, `tensorfold-*`);
//!   - `.zig-cache`, `zig-cache*`, `zig-out` or `zig-out-*` whose parent is a verified zspark
//!     checkout (it holds `build.zig.zon`, `zig/` and `tools/`).
//! A root under a protected component — `deploy-*`, `scratch-*`, `reference-*`, `hub`, `xet`,
//! `models`, `weights`, `logs`, `.git`, locks, env files, venvs, `zig-pkg*`, `full-passed`, serving
//! state — is refused whatever its name, so a `model-cache` under `~/models` is never touched.
//!
//! A plan removes the oldest eligible entries first until `--max-bytes` holds or the age bound is
//! spent; a bound no eligible entry can meet is reported unmet. Entries touched more recently than
//! `--older-than` (and nothing under a minute in any case), entries a running process has open,
//! mapped, as its executable or inside its working directory, and every entry where a `zig` process's
//! build overlaps the root are kept. Before each removal the entry is re-measured, the process table
//! re-scanned, and only then the unlink issued; a candidate changed, replaced or newly held since the
//! plan is skipped. Symlinks are never followed: the link itself is what a report counts and a plan
//! removes.
//!
//! The tool itself holds `~/.cache/zspark/machine.lock` shared for its whole run — the hold every
//! heavy command owes, which waits out an exclusive timing and is never blocked by a server's own
//! shared hold — and `--apply` takes `~/.cache/zspark/cache.lock` exclusive inside it (order ML,
//! then CL). Every `zig build` under `tools/check.sh` takes `cache.lock` shared for the build's
//! duration (after `machine.lock`, never across a `gpu.lock` wait); a direct `zig build` should take
//! it too (`flock -s`). A `zig` run outside the lock is still caught by the process scan above.
//! Tests live in `cache_test.zig`.

const std = @import("std");
const linux = std.os.linux;
const Allocator = std.mem.Allocator;
const Io = std.Io;
const Dir = Io.Dir;
const Timestamp = Io.Timestamp;
const path = std.fs.path;

const ns_day: i96 = 24 * std.time.ns_per_hour;
const ns_recent: i96 = std.time.ns_per_min; // touched in the last minute is never old enough
const max_path = linux.PATH_MAX;
const max_depth = 64;

pub const Kind = enum { eligible, protected, unknown };

pub const Class = enum {
    zig_artifact,
    generated_cache,
    generated_install,
    checkout,
    nested_root,
    package_store,
    deployment,
    fixtures,
    evidence,
    logs,
    serving_state,
    environment,
    locks,
    vcs,
    unknown,

    pub fn kind(c: Class) Kind {
        return switch (c) {
            .zig_artifact, .generated_cache, .generated_install, .checkout => .eligible,
            .nested_root, .unknown => .unknown,
            else => .protected,
        };
    }
};

/// The name rules applied at one level: `root` a selected root's child, `zig` a zig cache's,
/// `install` a `zig-out` install root's — every entry generated, protected names still on top.
const Level = enum { root, zig, install };

/// A cache or install root in its own right: select it directly rather than removing it as an entry.
fn isNestedRoot(name: []const u8) bool {
    return std.mem.eql(u8, name, ".zig-cache") or std.mem.eql(u8, name, "zig-cache") or
        std.mem.startsWith(u8, name, "zig-cache-") or std.mem.eql(u8, name, "zig-out") or
        std.mem.startsWith(u8, name, "zig-out-");
}

/// A name whose directory holds what never goes: deployments, fixtures, evidence, weights, packages.
/// As an entry it classes the entry; as a root's path component it refuses the whole root.
fn isProtectedName(name: []const u8) bool {
    if (std.mem.eql(u8, name, ".git")) return true;
    if (std.mem.startsWith(u8, name, "deploy-")) return true;
    if (std.mem.startsWith(u8, name, "scratch-")) return true;
    if (std.mem.startsWith(u8, name, "reference-")) return true;
    if (std.mem.eql(u8, name, "models") or std.mem.eql(u8, name, "weights")) return true;
    if (std.mem.eql(u8, name, "logs")) return true;
    if (std.mem.startsWith(u8, name, "zig-pkg")) return true;
    if (std.mem.eql(u8, name, "hub") or std.mem.eql(u8, name, "xet") or std.mem.eql(u8, name, "modules"))
        return true; // model weights and their download state, not a cache to prune
    if (std.mem.eql(u8, name, "full-passed")) return true;
    if (std.mem.startsWith(u8, name, "env.sh") or std.mem.eql(u8, name, "two_ranks_replies.json"))
        return true;
    if (std.mem.endsWith(u8, name, ".lock")) return true;
    if (std.mem.startsWith(u8, name, "venv") or std.mem.endsWith(u8, name, "-env")) return true;
    if (std.mem.eql(u8, name, "learned") or std.mem.eql(u8, name, "spilled") or std.mem.eql(u8, name, "recall"))
        return true;
    return false;
}

/// `name` regenerates on demand: a tool cache or an upstream checkout.
fn isPrunableName(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "-cache") or std.mem.endsWith(u8, name, "_cache") or
        std.mem.startsWith(u8, name, "upstream") or std.mem.startsWith(u8, name, "tensorfold-");
}

fn classifyEntry(name: []const u8, level: Level) Class {
    if (level == .install) {
        if (isNestedRoot(name)) return .nested_root;
        if (isProtectedName(name)) return classifyEntry(name, .root);
        return .generated_install;
    }
    if (level == .zig) {
        if (std.mem.eql(u8, name, "o")) return .zig_artifact;
        if (std.mem.eql(u8, name, "tmp")) return .zig_artifact;
        if (std.mem.eql(u8, name, "z")) return .zig_artifact; // manifest state, rebuilt on demand
        if (std.mem.eql(u8, name, "p")) return .package_store;
        if (std.mem.eql(u8, name, "b")) return .package_store;
        if (std.mem.eql(u8, name, "h")) return .package_store;
        if (std.mem.eql(u8, name, "c")) return .package_store;
        return .unknown;
    }
    if (isNestedRoot(name)) return .nested_root;
    if (std.mem.eql(u8, name, ".git")) return .vcs;
    if (std.mem.startsWith(u8, name, "deploy-")) return .deployment;
    if (std.mem.startsWith(u8, name, "scratch-")) return .evidence;
    if (std.mem.startsWith(u8, name, "reference-")) return .fixtures;
    if (std.mem.startsWith(u8, name, "upstream")) return .checkout;
    if (std.mem.startsWith(u8, name, "tensorfold-")) return .checkout;
    if (std.mem.eql(u8, name, "models") or std.mem.eql(u8, name, "weights")) return .fixtures;
    if (std.mem.eql(u8, name, "logs")) return .logs;
    if (std.mem.startsWith(u8, name, "zig-pkg")) return .package_store;
    if (std.mem.eql(u8, name, "hub") or std.mem.eql(u8, name, "xet") or std.mem.eql(u8, name, "modules"))
        return .fixtures;
    if (std.mem.eql(u8, name, "full-passed")) return .logs;
    if (std.mem.startsWith(u8, name, "env.sh") or std.mem.eql(u8, name, "two_ranks_replies.json"))
        return .environment;
    if (std.mem.endsWith(u8, name, ".lock")) return .locks;
    if (std.mem.startsWith(u8, name, "venv") or std.mem.endsWith(u8, name, "-env")) return .environment;
    if (std.mem.eql(u8, name, "learned") or std.mem.eql(u8, name, "spilled") or std.mem.eql(u8, name, "recall"))
        return .serving_state;
    if (std.mem.endsWith(u8, name, "-cache") or std.mem.endsWith(u8, name, "_cache"))
        return .generated_cache;
    return .unknown;
}

pub const RootKind = enum { generic, zig_cache, zig_out };

fn dirHas(io: Io, dir: Dir, name: []const u8) bool {
    var child = dir.openDir(io, name, .{ .follow_symlinks = false }) catch return false;
    child.close(io);
    return true;
}

/// A `zig-cache` root: the name, or a directory already holding zig's `o/` and `h/` stores.
fn classifyRoot(name: []const u8, dir: Dir, io: Io) RootKind {
    if (std.mem.eql(u8, name, "zig-out") or std.mem.startsWith(u8, name, "zig-out-")) return .zig_out;
    if (std.mem.eql(u8, name, ".zig-cache") or std.mem.eql(u8, name, "zig-cache") or
        std.mem.startsWith(u8, name, "zig-cache-")) return .zig_cache;
    if (dirHas(io, dir, "o") and dirHas(io, dir, "h")) return .zig_cache;
    return .generic;
}

/// `path`'s components, resolved: any protected name makes the whole root off-limits to removal —
/// pruning `deploy-*/bin`, `models/NAME` or `reference-*/sub` is refused, never merely filtered.
pub fn protectedAncestry(resolved_path: []const u8) bool {
    var it = std.mem.tokenizeScalar(u8, resolved_path, '/');
    while (it.next()) |component| if (isProtectedName(component)) return true;
    return false;
}

/// `dir` is a zspark checkout or worktree: the files a generated cache sits beside.
fn isCheckout(io: Io, dir: Dir) bool {
    var zon = dir.openFile(io, "build.zig.zon", .{}) catch return false;
    zon.close(io);
    return dirHas(io, dir, "zig") and dirHas(io, dir, "tools");
}

/// Whether `--apply` may act on `resolved` at all: a canonical generated cache/install location.
/// `home` supplies the `~/.cache/zig` and `~/.cache/zspark` roots (the fake home a test builds);
/// the checkout rule needs no home — the parent itself proves it.
fn allowedRoot(io: Io, arena: Allocator, resolved: []const u8, home: ?[]const u8) bool {
    const base = path.basename(resolved);
    // A zig cache or install dir inside a verified zspark checkout.
    if (isNestedRoot(base)) {
        const parent = path.dirname(resolved) orelse return false;
        var parent_dir = Dir.openDirAbsolute(io, parent, .{ .follow_symlinks = false }) catch return false;
        defer parent_dir.close(io);
        if (isCheckout(io, parent_dir)) return true;
    }
    const h = home orelse return false;
    const zig_cache_home = std.fmt.allocPrint(arena, "{s}/.cache/zig", .{h}) catch return false;
    if (std.mem.eql(u8, resolved, zig_cache_home)) return true;
    const zspark_home = std.fmt.allocPrint(arena, "{s}/.cache/zspark", .{h}) catch return false;
    if (std.mem.eql(u8, resolved, zspark_home)) return true;
    // A regenerable child of ~/.cache/zspark, selected directly.
    const parent = path.dirname(resolved) orelse return false;
    if (std.mem.eql(u8, parent, zspark_home) and isPrunableName(base)) return true;
    return false;
}

pub const Entry = struct {
    name: []const u8,
    class: Class,
    /// Allocated blocks (st_blocks * 512) over the entry's subtree; a link's own blocks.
    bytes: u64 = 0,
    /// Newest mtime in the subtree, nanoseconds since the epoch; the age bound reads this.
    touched: i96 = 0,
    /// The entry's own inode and shape, for re-measurement before removal.
    ino: u64 = 0,
    is_dir: bool = false,
    link_target: ?[]const u8 = null,

    /// Old enough to remove under `min_age_ns`; `ns_recent` always applies, so a just-touched
    /// entry is kept even when no age bound was asked for.
    fn old(e: Entry, now: i96, min_age_ns: i96) bool {
        return now - e.touched >= @max(min_age_ns, ns_recent);
    }
};

pub const Root = struct {
    path: []const u8,
    kind: RootKind,
    /// The resolved path contains a protected component: removal refuses it entirely.
    protected: bool,
    /// The root is one of the known generated locations `--apply` may act on.
    allowed: bool,
    entries: []Entry,
    /// The eligible entries, oldest first.
    eligible: []const Entry,
    protected_bytes: u64 = 0,
    unknown_bytes: u64 = 0,
    eligible_bytes: u64 = 0,

    pub fn applyable(root: Root) bool {
        return root.allowed and !root.protected;
    }
};

fn statxAt(dir: Dir, name: []const u8, sx: *linux.Statx) linux.E {
    var name_buffer: [512]u8 = undefined;
    const name_z = std.fmt.bufPrintSentinel(&name_buffer, "{s}", .{name}, 0) catch return .NAMETOOLONG;
    const rc = linux.statx(dir.handle, name_z, linux.AT.SYMLINK_NOFOLLOW, .{
        .TYPE = true,
        .MTIME = true,
        .SIZE = true,
        .BLOCKS = true,
        .INO = true,
    }, sx);
    return linux.errno(rc);
}

fn mtimeNs(sx: linux.Statx) i96 {
    return @as(i96, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec;
}

/// The allocated blocks and newest mtime over `dir`'s subtree. Symlinks count their own blocks and
/// are never opened. `depth` bounds descent so a pathological tree stops instead of looping.
const Measured = struct { bytes: u64, touched: i96 };

fn treeBytes(arena: Allocator, io: Io, dir: Dir, depth: usize) !Measured {
    if (depth > max_depth) return error.DepthLimit;
    var iterable = try dir.openDir(io, ".", .{ .iterate = true, .follow_symlinks = false });
    defer iterable.close(io);
    var out: Measured = .{ .bytes = 0, .touched = 0 };
    var it = iterable.iterate();
    while (try it.next(io)) |item| {
        var sx: linux.Statx = undefined;
        if (statxAt(dir, item.name, &sx) != .SUCCESS) continue; // a vanished child counts nothing
        out.bytes += sx.blocks * 512;
        out.touched = @max(out.touched, mtimeNs(sx));
        if (item.kind == .directory) {
            var child = dir.openDir(io, item.name, .{ .follow_symlinks = false }) catch |err| switch (err) {
                error.NotDir, error.FileNotFound => continue,
                error.AccessDenied, error.PermissionDenied => return error.AccessDenied,
                else => return err,
            };
            defer child.close(io);
            const sub = try treeBytes(arena, io, child, depth + 1);
            out.bytes += sub.bytes;
            out.touched = @max(out.touched, sub.touched);
        }
    }
    return out;
}

/// One entry measured as it stands now (statx plus a subtree walk). Used by the scan and by the
/// re-measurement each removal takes.
fn measureEntry(arena: Allocator, io: Io, dir: Dir, name: []const u8) !Entry {
    var entry: Entry = .{ .name = name, .class = .unknown };
    var sx: linux.Statx = undefined;
    if (statxAt(dir, name, &sx) != .SUCCESS) return error.FileNotFound;
    entry.ino = sx.ino;
    entry.touched = mtimeNs(sx);
    switch (sx.mode & linux.S.IFMT) {
        linux.S.IFLNK => {
            var buffer: [max_path]u8 = undefined;
            const n = dir.readLink(io, name, &buffer) catch 0;
            entry.link_target = try arena.dupe(u8, buffer[0..n]);
            entry.bytes = sx.blocks * 512;
        },
        linux.S.IFDIR => {
            entry.is_dir = true;
            var child = try dir.openDir(io, name, .{ .follow_symlinks = false });
            defer child.close(io);
            const sub = try treeBytes(arena, io, child, 0);
            entry.bytes = sx.blocks * 512 + sub.bytes;
            entry.touched = @max(entry.touched, sub.touched);
        },
        else => entry.bytes = sx.blocks * 512,
    }
    return entry;
}

/// Every entry of `path` classified, with allocated bytes and newest touch, and whether the root is
/// one removal may act on (`home` names `~/.cache/zig` and `~/.cache/zspark`; a test passes its fake
/// home). The path is resolved once; an entry that is itself a link records its target and is
/// measured by lstat only.
pub fn scanRoot(arena: Allocator, io: Io, root_arg: []const u8, home: ?[]const u8) !Root {
    var resolved_buffer: [max_path]u8 = undefined;
    const resolved = try Dir.cwd().realPathFile(io, root_arg, &resolved_buffer);
    const root_path = try arena.dupe(u8, resolved_buffer[0..resolved]);
    var dir = try Dir.openDirAbsolute(io, root_path, .{ .iterate = true, .follow_symlinks = false });
    defer dir.close(io);
    const root_kind = classifyRoot(path.basename(root_path), dir, io);
    const level: Level = switch (root_kind) {
        .zig_cache => .zig,
        .zig_out => .install,
        .generic => .root,
    };

    var names: std.ArrayList([]const u8) = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |item| try names.append(arena, try arena.dupe(u8, item.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);

    const entries = try arena.alloc(Entry, names.items.len);
    for (names.items, entries) |name, *entry| {
        entry.* = measureEntry(arena, io, dir, name) catch |err| switch (err) {
            // A vanished or unreadable name is never removable.
            error.FileNotFound, error.AccessDenied, error.PermissionDenied, error.NotDir => .{ .name = name, .class = .unknown },
            else => |e| return e,
        };
        entry.name = name;
        if (entry.class == .unknown and entry.ino != 0) entry.class = classifyEntry(name, level);
    }

    var eligible_len: usize = 0;
    for (entries) |e| if (e.class.kind() == .eligible) {
        eligible_len += 1;
    };
    const eligible = try arena.alloc(Entry, eligible_len);
    var i: usize = 0;
    for (entries) |e| if (e.class.kind() == .eligible) {
        eligible[i] = e;
        i += 1;
    };
    std.mem.sort(Entry, eligible, {}, struct {
        fn older(_: void, a: Entry, b: Entry) bool {
            if (a.touched != b.touched) return a.touched < b.touched;
            return std.mem.lessThan(u8, a.name, b.name);
        }
    }.older);
    var root: Root = .{
        .path = root_path,
        .kind = root_kind,
        .protected = protectedAncestry(root_path),
        .allowed = allowedRoot(io, arena, root_path, home),
        .entries = entries,
        .eligible = eligible,
    };
    for (entries) |e| switch (e.class.kind()) {
        .eligible => root.eligible_bytes += e.bytes,
        .protected => root.protected_bytes += e.bytes,
        .unknown => root.unknown_bytes += e.bytes,
    };
    return root;
}

// -- In-use detection ----------------------------------------------------------------------------------

/// What live processes hold under the selected root, found through /proc: `used`, the entry names an
/// open fd, a mapping, an executable or a cwd touches, and `build_dirs`, the directories a `zig`
/// process works from (its cwd and any absolute path argument that is a directory).
const Live = struct {
    used: std.StringHashMap(void),
    build_dirs: std.ArrayList([]const u8),
    /// The process table or an iteration of it could not be read: removal refuses.
    blind: bool = false,
};

fn readLinkAlloc(arena: Allocator, io: Io, link_path: []const u8) ?[]const u8 {
    const buffer = arena.alloc(u8, max_path) catch return null;
    const n = Dir.readLinkAbsolute(io, link_path, buffer) catch return null;
    return arena.dupe(u8, buffer[0..n]) catch null;
}

fn readFileAlloc(arena: Allocator, io: Io, file_path: []const u8, limit: usize) ?[]const u8 {
    const file = Dir.openFileAbsolute(io, file_path, .{}) catch return null;
    defer file.close(io);
    const buffer = arena.alloc(u8, 4096) catch return null;
    var reader = file.reader(io, buffer);
    return reader.interface.readAllocAll(arena, limit) catch null;
}

/// True when `p` is `under` or lies below it.
fn inside(p: []const u8, under: []const u8) bool {
    if (p.len < under.len) return false;
    if (!std.mem.startsWith(u8, p, under)) return false;
    return p.len == under.len or p[under.len] == '/';
}

const deleted_suffix = " (deleted)";

/// `/proc` reports removed-but-held paths with a ` (deleted)` suffix; exactly that suffix, nothing
/// more — a name ending in `d` or `e` is not corrupted into a different one.
fn stripDeleted(p: []const u8) []const u8 {
    if (std.mem.endsWith(u8, p, deleted_suffix)) return p[0 .. p.len - deleted_suffix.len];
    return p;
}

fn liveScan(arena: Allocator, io: Io, root_path: []const u8, names: []const []const u8) !Live {
    var live: Live = .{ .used = .init(arena), .build_dirs = .empty };
    var proc = Dir.openDirAbsolute(io, "/proc", .{ .iterate = true }) catch {
        live.blind = true;
        return live;
    };
    defer proc.close(io);
    var own_buffer: [64]u8 = undefined;
    const own_pid = std.fmt.bufPrint(&own_buffer, "{d}", .{linux.getpid()}) catch "";
    var it = proc.iterate();
    while (it.next(io) catch {
        live.blind = true;
        return live;
    }) |item| {
        if (item.kind != .directory) continue;
        if (item.name.len == 0 or item.name[0] < '0' or item.name[0] > '9') continue;
        const pid = item.name;
        // This process's exe and maps would name the cache it was built from; its cwd and fds are
        // real holds and stay checked.
        const own = std.mem.eql(u8, pid, own_pid);

        var link: [max_path]u8 = undefined;
        var n = std.fmt.bufPrint(&link, "/proc/{s}/cwd", .{pid}) catch continue;
        const cwd = readLinkAlloc(arena, io, n);
        n = std.fmt.bufPrint(&link, "/proc/{s}/exe", .{pid}) catch continue;
        const exe = readLinkAlloc(arena, io, n);

        // A zig process's cwd or argv inside a checkout marks that checkout's caches busy; the
        // executable's name identifies it (argv[0] alone would take a renamed caller).
        const is_zig = if (exe) |e| std.mem.eql(u8, path.basename(stripDeleted(e)), "zig") else false;
        if (is_zig) {
            if (cwd) |c| if (!inside(c, root_path)) try live.build_dirs.append(arena, c);
            n = std.fmt.bufPrint(&link, "/proc/{s}/cmdline", .{pid}) catch continue;
            if (readFileAlloc(arena, io, n, 1 << 20)) |cmdline| {
                var args = std.mem.tokenizeScalar(u8, cmdline, 0);
                while (args.next()) |arg| {
                    if (arg.len == 0 or arg[0] != '/') continue;
                    var build = Dir.openDirAbsolute(io, stripDeleted(arg), .{}) catch continue;
                    build.close(io);
                    if (!inside(arg, root_path)) try live.build_dirs.append(arena, stripDeleted(arg));
                }
            }
        }

        // Any path under the root a process holds — exe, cwd, open fd or mapping — protects the
        // entry it lies in.
        var paths: std.ArrayList([]const u8) = .empty;
        if (cwd) |c| try paths.append(arena, c);
        if (exe) |e| if (!own) try paths.append(arena, e);
        n = std.fmt.bufPrint(&link, "/proc/{s}/fd", .{pid}) catch continue;
        if (Dir.openDirAbsolute(io, n, .{ .iterate = true })) |fd_dir_open| {
            var fd_dir = fd_dir_open;
            defer fd_dir.close(io);
            var fd_it = fd_dir.iterate();
            while (fd_it.next(io) catch {
                live.blind = true;
                return live;
            }) |fd_item| {
                var fd_link: [max_path]u8 = undefined;
                const f = std.fmt.bufPrint(&fd_link, "/proc/{s}/fd/{s}", .{ pid, fd_item.name }) catch continue;
                if (readLinkAlloc(arena, io, f)) |target| try paths.append(arena, target);
            }
        } else |_| {}
        if (!own) {
            n = std.fmt.bufPrint(&link, "/proc/{s}/maps", .{pid}) catch continue;
            if (readFileAlloc(arena, io, n, 8 << 20)) |maps| {
                var lines = std.mem.tokenizeScalar(u8, maps, '\n');
                while (lines.next()) |line| {
                    const sep = std.mem.lastIndexOfScalar(u8, line, ' ') orelse continue;
                    const p = std.mem.trim(u8, line[sep + 1 ..], " \t");
                    if (p.len > 0 and p[0] == '/') try paths.append(arena, p);
                }
            }
        }
        for (paths.items) |p| {
            const clean = stripDeleted(p);
            if (!inside(clean, root_path) or clean.len == root_path.len) continue;
            const rest = clean[root_path.len + 1 ..];
            const first = if (std.mem.indexOfScalar(u8, rest, '/')) |s| rest[0..s] else rest;
            if (first.len == 0) continue;
            for (names) |name| {
                if (std.mem.eql(u8, first, name)) {
                    try live.used.put(name, {});
                    break;
                }
            }
        }
    }
    return live;
}

/// A `zig` build's directory touches the root when the build works inside it, or when the root is
/// the checkout's `.zig-cache`/`zig-out` and the build works inside that checkout. A generic root
/// gains no protection from a `zig` running above it; a directory's name alone does not mark a build.
fn buildTouches(build_dir: []const u8, root: Root) bool {
    if (inside(build_dir, root.path)) return true;
    if (root.kind == .generic) return false;
    return inside(root.path, build_dir);
}

/// `/proc/PID/stat`'s parent pid: the field after the comm's last `)`.
fn ppidOf(stat_text: []const u8) ?i32 {
    const close = std.mem.lastIndexOfScalar(u8, stat_text, ')') orelse return null;
    var fields = std.mem.tokenizeScalar(u8, stat_text[close + 1 ..], ' ');
    _ = fields.next(); // state
    const ppid = fields.next() orelse return null;
    return std.fmt.parseInt(i32, ppid, 10) catch null;
}

/// This process runs anywhere below a `zig` executable — a `zig build` driver, wrapped or not.
fn underBuildDriver(arena: Allocator, io: Io) bool {
    var pid: i32 = @intCast(linux.getpid());
    var depth: usize = 0;
    while (depth < 8) : (depth += 1) {
        var stat_path: [64]u8 = undefined;
        const stat_s = std.fmt.bufPrint(&stat_path, "/proc/{d}/stat", .{pid}) catch return false;
        const stat_text = readFileAlloc(arena, io, stat_s, 4096) orelse return false;
        var exe_path: [64]u8 = undefined;
        const exe_s = std.fmt.bufPrint(&exe_path, "/proc/{d}/exe", .{pid}) catch return false;
        if (readLinkAlloc(arena, io, exe_s)) |exe| {
            if (std.mem.eql(u8, path.basename(stripDeleted(exe)), "zig")) return true;
        }
        pid = ppidOf(stat_text) orelse return false;
        if (pid <= 1) return false;
    }
    return false;
}

// -- The plan ------------------------------------------------------------------------------------------

pub const Policy = struct {
    /// The age bound in nanoseconds; `null` takes none (the one-minute floor still holds).
    older_than_ns: ?i96 = null,
    /// Allocated bytes the root may retain after pruning; `null` is unbounded.
    max_bytes: ?u64 = null,
    /// The build coordination lock's path; the CLI takes no option for it — tests may pass one
    /// (`"-"` skips the lock entirely).
    lock: ?[]const u8 = null,
    /// Skip the zig-build-driver refusal. Test-only: the CLI never sets it, so `zig build cache
    /// -- --apply` still refuses outright.
    driver_exempt: bool = false,
};

pub const Action = enum { remove, keep };

pub const Planned = struct {
    entry: Entry,
    action: Action,
    reason: []const u8,
};

pub const Plan = struct {
    rows: []const Planned,
    /// Eligible bytes the plan removes.
    removing: u64 = 0,
    /// Bytes the root holds after the plan.
    retained: u64 = 0,
    /// `max_bytes` cannot be met even with every eligible entry gone.
    unmet: bool = false,
    /// Entries kept only because a process or a build touches them.
    blocked: usize = 0,
    /// Nothing may be removed here: the root is not a known generated cache.
    refused: bool = false,
};

/// The removal order for `root` under `policy`: oldest eligible first, the age bound and
/// live-process checks applied, the byte ceiling the stop. `now` is nanoseconds since the epoch.
/// A root outside the known generated locations yields keep-rows only.
pub fn plan(arena: Allocator, io: Io, root: *Root, policy: Policy, now: i96) !Plan {
    var rows: std.ArrayList(Planned) = .empty;
    var retained: u64 = 0;
    for (root.entries) |e| retained += e.bytes;
    if (!root.applyable()) {
        for (root.eligible) |e| {
            try rows.append(arena, .{
                .entry = e,
                .action = .keep,
                .reason = if (root.protected) "root lies under a protected path" else "root is not a known generated cache",
            });
        }
        return .{ .rows = rows.items, .retained = retained, .refused = true };
    }

    const names = try arena.alloc([]const u8, root.entries.len);
    for (root.entries, names) |e, *n| n.* = e.name;
    const live = try liveScan(arena, io, root.path, names);
    if (live.blind) return error.ProcBlind;

    var removing: u64 = 0;
    var blocked: usize = 0;
    for (root.eligible) |e| {
        var action: Action = .remove;
        var reason: []const u8 = "oldest eligible first";
        if (live.used.contains(e.name)) {
            action = .keep;
            reason = "in use by a running process";
            blocked += 1;
        }
        if (action == .remove and !policy.driver_exempt) for (live.build_dirs.items) |b| {
            if (buildTouches(b, root.*)) {
                action = .keep;
                reason = "a running zig build overlaps this root";
                blocked += 1;
                break;
            }
        };
        if (action == .remove) {
            if (!e.old(now, policy.older_than_ns orelse 0)) {
                action = .keep;
                reason = "younger than the age bound";
            }
        }
        if (action == .remove) {
            if (policy.max_bytes) |cap| {
                if (retained <= cap) {
                    action = .keep;
                    reason = "retained bytes already within the bound";
                }
            }
        }
        if (action == .remove) {
            retained -= e.bytes;
            removing += e.bytes;
        }
        try rows.append(arena, .{ .entry = e, .action = action, .reason = reason });
    }
    return .{
        .rows = rows.items,
        .removing = removing,
        .retained = retained,
        .unmet = if (policy.max_bytes) |cap| retained > cap else false,
        .blocked = blocked,
    };
}

// -- Removal -------------------------------------------------------------------------------------------

pub const Outcome = enum { removed, skipped, failed };

pub const Removed = struct {
    entry: Entry,
    outcome: Outcome,
    why: []const u8,
};

pub const LockMode = enum {
    /// Shared, blocking: the mode every heavy command takes `machine.lock` in. A timing's
    /// exclusive hold makes this wait, which is the point — the CLI never scans during one.
    shared_block,
    /// Exclusive, nonblocking: `cache.lock` for a prune. A build's shared hold — check.sh's or a
    /// direct `flock -s` — turns this into error.BuildActive.
    exclusive_nb,
};

/// A hold of `lock_path` in `mode`. `lock_path` `null` takes none (tests); `~/` resolves against
/// `home`. The shared machine lock never fails on contention — it waits, and a long-lived server's
/// own shared hold does not exclude it.
pub fn takeLock(io: Io, lock_path: ?[]const u8, home: ?[]const u8, mode: LockMode) !?Io.File {
    const asked = lock_path orelse return null;
    var resolved: [max_path]u8 = undefined;
    var lock = asked;
    if (std.mem.startsWith(u8, lock, "~/")) {
        const h = home orelse return error.NoHome;
        lock = std.fmt.bufPrint(&resolved, "{s}/{s}", .{ h, lock[2..] }) catch return error.NameTooLong;
    }
    // The lock file may not exist yet; its directory is the deployment's and always does.
    const file = Dir.createFileAbsolute(io, lock, .{}) catch |err| switch (err) {
        error.FileNotFound, error.NotDir, error.BadPathName => return error.LockMissingDir,
        else => |e| return e,
    };
    errdefer file.close(io);
    const op: i32 = switch (mode) {
        .shared_block => std.posix.LOCK.SH,
        .exclusive_nb => std.posix.LOCK.EX | std.posix.LOCK.NB,
    };
    const rc = linux.flock(file.handle, op);
    switch (linux.errno(rc)) {
        .SUCCESS => return file,
        .AGAIN => return error.BuildActive,
        else => return error.LockFailed,
    }
}

pub const lock_name = "cache.lock";
pub const machine_lock_name = "machine.lock";

/// Apply `p`'s removals to `root`. The canonical path takes `~/.cache/zspark/machine.lock` shared
/// first — the hold every heavy command owes, which waits out an exclusive timing but is never
/// blocked by a long-lived server's own shared hold — then `cache.lock` exclusive nonblocking.
/// `policy.lock` is the test hook (`"-"` takes neither, another path replaces cache.lock and skips
/// machine.lock). Every candidate is re-measured, the process table re-scanned, and only then
/// unlinked; a difference of any kind keeps the entry.
pub fn apply(arena: Allocator, io: Io, root: *Root, policy: Policy, p: Plan, home: ?[]const u8) ![]const Removed {
    if (root.protected) return error.RootProtected;
    if (!root.applyable()) return error.NotACache;
    var machine_lock: ?Io.File = null;
    defer if (machine_lock) |f| f.close(io);
    var cache_lock: ?Io.File = null;
    defer if (cache_lock) |f| f.close(io);
    if (policy.lock) |named| {
        if (!std.mem.eql(u8, named, "-")) cache_lock = try takeLock(io, named, home, .exclusive_nb);
    } else {
        const h = home orelse return error.NoHome;
        var ml_buffer: [max_path]u8 = undefined;
        const ml_path = std.fmt.bufPrint(&ml_buffer, "{s}/.cache/zspark/{s}", .{ h, machine_lock_name }) catch return error.NameTooLong;
        machine_lock = try takeLock(io, ml_path, home, .shared_block);
        var cl_buffer: [max_path]u8 = undefined;
        const cl_path = std.fmt.bufPrint(&cl_buffer, "{s}/.cache/zspark/{s}", .{ h, lock_name }) catch return error.NameTooLong;
        cache_lock = try takeLock(io, cl_path, home, .exclusive_nb);
    }
    if (!policy.driver_exempt and underBuildDriver(arena, io)) return error.BuildDriver;

    var dir = try Dir.openDirAbsolute(io, root.path, .{ .iterate = true, .follow_symlinks = false });
    defer dir.close(io);
    var out: std.ArrayList(Removed) = .empty;
    const names = try arena.alloc([]const u8, root.entries.len);
    for (root.entries, names) |e, *n| n.* = e.name;

    for (p.rows) |row| {
        if (row.action != .remove) continue;
        var outcome: Outcome = .removed;
        var why: []const u8 = "removed";
        // First the candidate stands as it is now; then a fresh survey of who holds it, taken
        // immediately before its unlink — a hold taken since the plan keeps it.
        const fresh = measureEntry(arena, io, dir, row.entry.name) catch |err| switch (err) {
            error.FileNotFound, error.NotDir => {
                try out.append(arena, .{ .entry = row.entry, .outcome = .skipped, .why = "already gone" });
                continue;
            },
            else => |e| return e,
        };
        const live = try liveScan(arena, io, root.path, names);
        if (live.blind) return error.ProcBlind;
        const changed = fresh.ino != row.entry.ino or fresh.bytes != row.entry.bytes or
            fresh.touched != row.entry.touched or fresh.is_dir != row.entry.is_dir or
            (fresh.link_target == null) != (row.entry.link_target == null) or
            (fresh.link_target != null and row.entry.link_target != null and
                !std.mem.eql(u8, fresh.link_target.?, row.entry.link_target.?));
        if (changed) {
            outcome = .skipped;
            why = "changed after the scan";
        }
        if (outcome == .removed and live.used.contains(row.entry.name)) {
            outcome = .skipped;
            why = "in use by a running process";
        }
        if (outcome == .removed and !policy.driver_exempt) for (live.build_dirs.items) |b| {
            if (buildTouches(b, root.*)) {
                outcome = .skipped;
                why = "a running zig build overlaps this root";
                break;
            }
        };
        if (outcome == .removed) {
            dir.deleteTree(io, row.entry.name) catch |err| {
                outcome = .failed;
                why = @errorName(err);
            };
        }
        try out.append(arena, .{ .entry = row.entry, .outcome = outcome, .why = why });
    }
    return out.items;
}

// -- Output and entry point -----------------------------------------------------------------------------

fn gibi(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

fn printEntry(out: *Io.Writer, e: Entry, now: i96, extra: []const u8) !void {
    const days = @as(f64, @floatFromInt(@max(now - e.touched, 0))) / @as(f64, @floatFromInt(ns_day));
    const link = if (e.link_target) |t| t else "";
    try out.print("    {d:9.2} GiB  {d:7.1} d  {s:<16} {s} {s} {s}\n", .{
        gibi(e.bytes), days, @tagName(e.class), e.name, link, extra,
    });
}

fn printRoot(out: *Io.Writer, root: Root, p: ?Plan, now: i96) !void {
    try out.print("{s} ({s}{s}{s})\n", .{
        root.path,
        @tagName(root.kind),
        if (root.protected) ", protected ancestry" else "",
        if (!root.allowed) ", report only" else "",
    });
    try out.print("  eligible {d:.2} GiB, protected {d:.2} GiB, unknown {d:.2} GiB over {d} entries\n", .{
        gibi(root.eligible_bytes), gibi(root.protected_bytes), gibi(root.unknown_bytes), root.entries.len,
    });
    if (p) |pl| {
        for (pl.rows) |row| {
            const verb: []const u8 = if (row.action == .remove) "remove" else "keep  ";
            try out.print("  {s}", .{verb});
            try printEntry(out, row.entry, now, row.reason);
        }
        try out.print("  plan removes {d:.2} GiB and retains {d:.2} GiB", .{ gibi(pl.removing), gibi(pl.retained) });
        if (pl.blocked > 0) try out.print("; {d} kept in active use", .{pl.blocked});
        if (pl.unmet) try out.writeAll("; bound unmet: no eligible entry left to remove");
        try out.writeAll("\n");
    }
}

fn printReport(out: *Io.Writer, arena: Allocator, io: Io, paths: []const []const u8, home: ?[]const u8, now: i96) !void {
    for (paths) |root_path| {
        const root = scanRoot(arena, io, root_path, home) catch |err| {
            try out.print("{s}: {t}\n", .{ root_path, err });
            continue;
        };
        try printRoot(out, root, null, now);
        for (root.entries) |e| try printEntry(out, e, now, "");
    }
}

/// The machine's known roots for the default report: each `~/repo/zspark*` checkout's `.zig-cache`
/// and `zig-out`, the zig global cache, `~/.cache/zspark` and `~/.cache/huggingface`. Missing ones
/// are skipped.
fn defaultRoots(arena: Allocator, io: Io, home: []const u8) ![]const []const u8 {
    var roots: std.ArrayList([]const u8) = .empty;
    const repos = try std.fmt.allocPrint(arena, "{s}/repo", .{home});
    if (Dir.openDirAbsolute(io, repos, .{ .iterate = true })) |d| {
        var dir = d;
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |item| {
            if (item.kind != .directory or !std.mem.startsWith(u8, item.name, "zspark")) continue;
            for ([_][]const u8{ ".zig-cache", "zig-out" }) |leaf| {
                const p = try std.fmt.allocPrint(arena, "{s}/{s}/{s}", .{ repos, item.name, leaf });
                var check = Dir.openDirAbsolute(io, p, .{}) catch continue;
                check.close(io);
                try roots.append(arena, p);
            }
        }
    } else |_| {}
    for ([_][]const u8{ ".cache/zig", ".cache/zspark", ".cache/huggingface" }) |leaf| {
        const p = try std.fmt.allocPrint(arena, "{s}/{s}", .{ home, leaf });
        var check = Dir.openDirAbsolute(io, p, .{}) catch continue;
        check.close(io);
        try roots.append(arena, p);
    }
    return roots.items;
}

fn parseBytes(text: []const u8) !u64 {
    const units = [_]struct { suffix: []const u8, mult: u64 }{
        .{ .suffix = "GiB", .mult = 1 << 30 },  .{ .suffix = "MiB", .mult = 1 << 20 },
        .{ .suffix = "KiB", .mult = 1 << 10 },  .{ .suffix = "GB", .mult = 1_000_000_000 },
        .{ .suffix = "MB", .mult = 1_000_000 }, .{ .suffix = "KB", .mult = 1_000 },
        .{ .suffix = "B", .mult = 1 },
    };
    for (units) |u| {
        if (std.mem.endsWith(u8, text, u.suffix)) {
            const n = try std.fmt.parseFloat(f64, text[0 .. text.len - u.suffix.len]);
            if (!(n >= 0)) return error.BadBytes;
            return @intFromFloat(n * @as(f64, @floatFromInt(u.mult)));
        }
    }
    return std.fmt.parseInt(u64, text, 10);
}

const usage_text =
    \\usage: zspark-cache [ROOT] [--older-than DAYS] [--max-bytes BYTES] [--apply]
    \\  no ROOT            report this machine's caches (~/repo/zspark* zig caches, ~/.cache/zig,
    \\                     ~/.cache/zspark, ~/.cache/huggingface)
    \\  ROOT               report ROOT's entries: bytes allocated, days untouched, class
    \\  --older-than DAYS  plan only entries untouched this long (decimals allowed)
    \\  --max-bytes BYTES  keep at most BYTES allocated over ROOT (units: B, KB..GiB)
    \\  --apply            actually remove the plan, holding machine.lock shared and
    \\                     ~/.cache/zspark/cache.lock exclusive
    \\--apply accepts only ~/.cache/zig, ~/.cache/zspark and its regenerable children, and
    \\.zig-cache/zig-out of a verified zspark checkout; every other root is report-only.
    \\Run the installed binary: `zig build cache` runs below the zig build driver, which --apply
    \\refuses outright.
;

pub fn main(init: std.process.Init) !u8 {
    var buffer: [4096]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const out = &stdout_file.interface;
    defer out.flush() catch {};
    const arena = init.arena.allocator();
    const io = init.io;
    const home = init.minimal.environ.getPosix("HOME");

    var policy: Policy = .{};
    var do_apply = false;
    var root_arg: ?[]const u8 = null;
    const argv = try init.minimal.args.toSlice(arena);
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--older-than")) {
            i += 1;
            if (i >= argv.len) return complain(out, "--older-than needs a day count");
            const days = std.fmt.parseFloat(f64, argv[i]) catch return complain(out, "bad --older-than value");
            if (!(days > 0)) return complain(out, "the age bound must be positive");
            policy.older_than_ns = @intFromFloat(days * @as(f64, @floatFromInt(ns_day)));
        } else if (std.mem.eql(u8, arg, "--max-bytes")) {
            i += 1;
            if (i >= argv.len) return complain(out, "--max-bytes needs a byte count");
            policy.max_bytes = parseBytes(argv[i]) catch return complain(out, "bad --max-bytes value");
        } else if (std.mem.eql(u8, arg, "--apply")) {
            do_apply = true;
        } else if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try out.writeAll(usage_text);
            return 0;
        } else if (root_arg == null and arg.len > 0 and arg[0] != '-') {
            root_arg = arg;
        } else {
            try out.print("{s}: unknown argument\n{s}", .{ arg, usage_text });
            return 2;
        }
    }

    // The shared machine lock covers the whole run: scanning is heavy, and a timing's exclusive
    // hold must never race a prune. It waits on the timing; a server's shared hold does not block.
    var machine_lock: ?Io.File = null;
    defer if (machine_lock) |f| f.close(io);
    if (home) |h| {
        var ml_buffer: [max_path]u8 = undefined;
        if (std.fmt.bufPrint(&ml_buffer, "{s}/.cache/zspark/{s}", .{ h, machine_lock_name }) catch null) |ml_path| {
            machine_lock = takeLock(io, ml_path, home, .shared_block) catch null;
        }
    }

    const now = Timestamp.now(io, .real).toNanoseconds();
    if (root_arg == null) {
        const h = home orelse return complain(out, "HOME is unset");
        try printReport(out, arena, io, try defaultRoots(arena, io, h), h, now);
        try out.writeAll("\nReport only. To prune: zspark-cache ROOT --older-than DAYS [--max-bytes B] --apply\n");
        return 0;
    }
    var root = scanRoot(arena, io, root_arg.?, home) catch |err| {
        try out.print("{s}: {t}\n", .{ root_arg.?, err });
        return 1;
    };
    if (policy.older_than_ns == null and policy.max_bytes == null) {
        try printRoot(out, root, null, now);
        for (root.entries) |e| try printEntry(out, e, now, "");
        try out.writeAll("\nReport only: no bound given, nothing selected for removal.\n");
        return 0;
    }
    const p = plan(arena, io, &root, policy, now) catch |err| {
        try out.print("{s}: cannot survey live processes safely ({t}); nothing removed\n", .{ root.path, err });
        return 1;
    };
    try printRoot(out, root, p, now);
    if (!do_apply) {
        try out.writeAll("dry run: --apply removes what is listed as remove\n");
        return 0;
    }
    const removed = apply(arena, io, &root, policy, p, home) catch |err| switch (err) {
        error.BuildActive => {
            try out.writeAll("a build holds cache.lock (shared) or another prune is running; nothing removed\n");
            return 2;
        },
        error.BuildDriver => {
            try out.writeAll("refused: this process runs under a zig build driver; prune the installed binary\n");
            return 2;
        },
        error.ProcBlind => {
            try out.writeAll("live processes cannot be surveyed; nothing removed\n");
            return 1;
        },
        error.RootProtected => {
            try out.print("{s}: under a protected path; nothing removed\n", .{root.path});
            return 2;
        },
        error.NotACache => {
            try out.print("{s}: not a known generated cache or install root; nothing removed\n", .{root.path});
            return 2;
        },
        error.LockMissingDir, error.NoHome => {
            try out.writeAll("~/.cache/zspark/cache.lock cannot be opened; nothing removed\n");
            return 1;
        },
        else => |e| return e,
    };
    var done: usize = 0;
    var kept: usize = 0;
    var failed: usize = 0;
    for (removed) |r| {
        try out.print("  {s} {s} ({s})\n", .{ @tagName(r.outcome), r.entry.name, r.why });
        switch (r.outcome) {
            .removed => done += 1,
            .skipped => kept += 1,
            .failed => failed += 1,
        }
    }
    try out.print("removed {d}, skipped {d}, failed {d}\n", .{ done, kept, failed });
    return if (failed == 0) 0 else 1;
}

fn complain(out: *Io.Writer, text: []const u8) u8 {
    out.print("{s}\n", .{text}) catch {};
    return 2;
}
