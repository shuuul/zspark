//! States kept on disk: the state a prompt's shared prefix leaves (`serve --learn`, spec 064), so the first
//! conversation after a restart starts where the prefix ends, and the states that left a pool or a session
//! (`serve --spill-gib`, spec 074; `spill.zig` says which), each kind (`Kind`) a store under its own root.
//! Host only: a state is a header and parts of bytes, and what the parts are is the model's
//! (`models/qwen4/session/learnt.zig`).
//!
//! `ROOT/IDENTITY/`, as TensorFold lays its own out (`core/rows_store.zig`). IDENTITY is 16 hex digits the
//! caller hashed from everything a state's bytes depend on, `magic` among them, so a file is only ever read by
//! a load that would write the same bytes, and one of an older form is another identity's. Since spec 101 a
//! state is files by block (`learned_blocks.zig`): its rows in 256-token blocks, `STEM.blk`, that every state
//! over the same tokens names and so shares, and its own file, its header, its tokens, the table of its parts,
//! its own parts and its rows past its last whole block: `KEY.st` for a learned state, `KEY.sst` for a spilt
//! one. KEY is the hash of the state's tokens, which the file holds whole and a read compares, a hash being
//! no proof. Every file is written as `.part`, synced and renamed, the blocks before the state, so one that a
//! crash cut is never read and a state never stands without its blocks; a block no state names goes.
//!
//! The blocks are both kinds' (`Blocks`, spec 101): one directory an identity, which the learned and the
//! spilt states both name, so a harness's rows that a spilt conversation shares are on disk once. It is the
//! learned root's where that kind is kept (TensorFold opens one store a model, under its learn root:
//! `native/flashnext_disk.zig`), else the spilt root's; each kind's own files stay under its own root.
//!
//! What a reader asks for is the state's image, a part whole, as before the blocks: `Image` reads it from the
//! files. A file's parts are not summed: a read would pay for it before every first token, and a state's bytes
//! are checked by their sizes against the model that reads them, as vLLM's file tier and SGLang's check a byte
//! count and no checksum (spec 074); a block by its full name and its size. A state whose file or a block is
//! missing, or not what its name says, is a miss: its file is deleted and the prompt runs.
//!
//! Past `cap` bytes a kind's files go, other identities' under its root first (no load of this one reads
//! them), then its least recently read or written state, each block with the last state of either kind
//! naming it (vLLM's and SGLang's LRU; TensorFold weighs a state's tokens by the bytes only it holds). A kind
//! counts its own files, the blocks only its states name, and half of each block both kinds name: TensorFold
//! has one cap over both, zspark a flag a kind. A read sets a state's time.
//!
//! A store knows the length of every state of its identity (`held`, read from the files at `open`), so the
//! longest a prompt begins with is found by hashing the prompt's prefixes of those lengths (`longest`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const Io = std.Io;
const blocks = @import("learned_blocks.zig");
const spill_writer = @import("spill_writer.zig");
pub const Part = blocks.Part;
pub const Layout = blocks.Layout;
const Stem = blocks.Stem;

/// `\x03` since the blocks are both kinds' (spec 101): a directory of the form before is another identity.
pub const magic: [8]u8 = "zslearn\x03".*;

pub const Header = extern struct { magic: [8]u8, identity: u64, tokens: u32, parts: u32 };

pub fn keyOf(tokens: []const i32) u64 {
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(tokens));
}

/// An image's bytes before its parts: the header, the tokens and the parts' table.
pub fn prefixBytes(tokens: usize, parts: usize) u64 {
    return @sizeOf(Header) + tokens * 4 + parts * @sizeOf(Part);
}

const Name = [19]u8;

pub fn nameOf(key: u64) Name {
    var name: Name = undefined;
    _ = std.fmt.bufPrint(&name, "{x:0>16}.st", .{key}) catch unreachable;
    return name;
}

pub fn partOf(key: u64) [24]u8 {
    var name: [24]u8 = undefined;
    _ = std.fmt.bufPrint(&name, "{x:0>16}.st.part", .{key}) catch unreachable;
    return name;
}

/// What a state is kept as: `--learn`'s, where a shared prefix ends, or `--spill-gib`'s, a conversation that
/// left. Each kind's files are its own, and so is its cap; the blocks are both kinds' (`Blocks`).
pub const Kind = enum(u1) {
    learned,
    spilled,

    pub fn suffix(kind: Kind) []const u8 {
        return switch (kind) {
            .learned => ".st",
            .spilled => ".sst",
        };
    }

    fn other(kind: Kind) Kind {
        return @fromBackingInt(@intCast(1 - @backingInt(kind)));
    }
};

/// A state file's name of either kind, or its `.part`.
pub const FileName = struct {
    bytes: [32]u8 = undefined,
    len: u8 = 0,

    pub fn get(self: *const FileName) []const u8 {
        return self.bytes[0..self.len];
    }
};

/// `KEY.st` or `KEY.sst`, `.part` after it where `part`.
pub fn fileOf(kind: Kind, key: u64, part: bool) FileName {
    var out: FileName = .{};
    out.len = @intCast((std.fmt.bufPrint(&out.bytes, "{x:0>16}{s}{s}", .{ key, kind.suffix(), if (part) ".part" else "" }) catch unreachable).len);
    return out;
}

/// The key a state file's name of `kind` gives, null where it is no such state's name.
fn keyIn(kind: Kind, name: []const u8) ?u64 {
    const suffix = kind.suffix();
    if (name.len != 16 + suffix.len or !std.mem.endsWith(u8, name, suffix)) return null;
    return std.fmt.parseInt(u64, name[0..16], 16) catch null;
}

/// In a block directory: the spilt kind's roots, one absolute path a line, whose states name its blocks.
pub const roots_file = "spilled.roots";

fn hex(value: u64) [16]u8 {
    var name: [16]u8 = undefined;
    _ = std.fmt.bufPrint(&name, "{x:0>16}", .{value}) catch unreachable;
    return name;
}

/// An image's bytes for its files: `view` gives those from `offset`, `dest.len` at most and one at least, in
/// `dest` or in memory of its own.
pub const Source = struct {
    context: *const anyopaque,
    view: *const fn (context: *const anyopaque, offset: u64, dest: []u8) anyerror![]const u8,
};

/// An image from its prefix in memory and its parts from `parts`, which counts from the first part.
pub const Joined = struct {
    prefix: []const u8,
    parts: Source,

    pub fn source(self: *const Joined) Source {
        return .{ .context = self, .view = view };
    }

    fn view(context: *const anyopaque, offset: u64, dest: []u8) anyerror![]const u8 {
        const self: *const Joined = @ptrCast(@alignCast(context));
        if (offset < self.prefix.len) return self.prefix[@intCast(offset)..][0..@min(dest.len, self.prefix.len - offset)];
        return self.parts.view(self.parts.context, offset - self.prefix.len, dest);
    }
};

/// Parts in host memory, back to back.
pub const Slices = struct {
    parts: []const []const u8,

    pub fn source(self: *const Slices) Source {
        return .{ .context = self, .view = view };
    }

    fn view(context: *const anyopaque, offset: u64, dest: []u8) anyerror![]const u8 {
        const self: *const Slices = @ptrCast(@alignCast(context));
        var at = offset;
        for (self.parts) |part| {
            if (at < part.len) return part[@intCast(at)..][0..@min(dest.len, part.len - at)];
            at -= part.len;
        }
        return &.{};
    }
};

/// An image's prefix: its header, `tokens` and the table of `parts`.
pub fn prefixOf(gpa: Allocator, identity: u64, tokens: []const i32, parts: []const Part) ![]u8 {
    const out = try gpa.alloc(u8, @intCast(prefixBytes(tokens.len, parts.len)));
    const header: Header = .{ .magic = magic, .identity = identity, .tokens = @intCast(tokens.len), .parts = @intCast(parts.len) };
    @memcpy(out[0..@sizeOf(Header)], std.mem.asBytes(&header));
    @memcpy(out[@sizeOf(Header)..][0 .. tokens.len * 4], std.mem.sliceAsBytes(tokens));
    @memcpy(out[@sizeOf(Header) + tokens.len * 4 ..], std.mem.sliceAsBytes(parts));
    return out;
}

/// What stops a write between its files: a store's writer closing.
pub const Stop = struct { context: *anyopaque, call: *const fn (*anyopaque) bool };

/// `layout`'s files for the image `source` gives, whose prefix is `prefix` bytes: each block not on disk
/// whole, in `shared`, then the state's own file of `kind`, in `dir`, each `.part` synced and renamed. The
/// bytes written.
pub fn writeFiles(io: Io, shared: Io.Dir, dir: Io.Dir, kind: Kind, identity: u64, key: u64, layout: *const Layout, prefix: u64, source: Source, stage: []u8, stop: ?Stop) !u64 {
    var written: u64 = 0;
    for (layout.names, 0..) |name, k| {
        if (stop) |held| if (held.call(held.context)) return error.Closing;
        const file_name = blocks.blockFile(name);
        if (shared.statFile(io, &file_name, .{})) |stat| {
            if (stat.size == layout.blockBytes()) continue;
        } else |_| {}
        const part_name = blocks.blockPart(name);
        const file = try shared.createFile(io, &part_name, .{ .permissions = .fromMode(0o600) });
        errdefer shared.deleteFile(io, &part_name) catch {};
        {
            errdefer file.close(io);
            const header: blocks.BlockHeader = .{ .magic = blocks.block_magic, .identity = identity, .name = name, .bytes = layout.block_bytes };
            try file.writePositionalAll(io, std.mem.asBytes(&header), 0);
            for (0..layout.rows.len) |j| {
                const one = layout.slice(j, k);
                try copy(io, file, @sizeOf(blocks.BlockHeader) + one.offset, source, prefix + one.at, one.len, stage);
            }
            try file.sync(io);
        }
        file.close(io);
        try shared.rename(&part_name, shared, &file_name, io);
        written += layout.blockBytes();
    }
    if (stop) |held| if (held.call(held.context)) return error.Closing;
    const part = fileOf(kind, key, true);
    const part_name = part.get();
    const file = try dir.createFile(io, part_name, .{ .permissions = .fromMode(0o600) });
    errdefer dir.deleteFile(io, part_name) catch {};
    {
        errdefer file.close(io);
        try copy(io, file, 0, source, 0, prefix, stage);
        for (layout.segments) |one| if (one.block == null) try copy(io, file, prefix + one.offset, source, prefix + one.at, one.len, stage);
        try file.sync(io);
    }
    file.close(io);
    const whole = fileOf(kind, key, false);
    try dir.rename(part_name, dir, whole.get(), io);
    return written + prefix + layout.own_bytes;
}

fn copy(io: Io, file: Io.File, at: u64, source: Source, from: u64, len: u64, stage: []u8) !void {
    var done: u64 = 0;
    while (done < len) {
        const got = try source.view(source.context, from + done, stage[0..@intCast(@min(len - done, stage.len))]);
        if (got.len == 0) return error.Mismatch;
        try file.writePositionalAll(io, got, at + done);
        done += got.len;
    }
}

/// What a state's files are before they are written: its layout, its image's bytes and its prefix's, and the
/// bytes on disk it adds (its own file and the blocks no state held names).
pub const Plan = struct {
    layout: Layout,
    prefix: u64,
    image: u64,
    disk: u64,

    pub fn deinit(self: *Plan, gpa: Allocator) void {
        self.layout.deinit(gpa);
        self.* = undefined;
    }
};

/// The blocks of an identity's states (spec 101): one directory, which the states of both kinds name. A block
/// goes when no state of either kind names it: what a kind's store open on it holds (its index and its
/// writer's jobs), and what states of a kind no store holds in this load name, read at `open` and kept
/// (`foreign`): the block directory's own state files of that kind, and the spilt states under each root
/// `roots_file` lists, a root a load with both kinds under two roots kept them in. A load that opens one kind
/// alone so never deletes a block the other's states name; one that opens the spilt kind alone under its own
/// root reads none of those states either, since their blocks are not where it looks: they are a miss.
pub const Blocks = struct {
    gpa: Allocator,
    io: Io,
    identity: u64,
    dir: Io.Dir,
    /// Kinds `open` was told will join and have not: nothing is collected before they have.
    waiting: [2]bool,
    /// For each kind, the states of its store that name each block; kept once its store closes (its files
    /// stay on disk).
    refs: [2]std.AutoHashMapUnmanaged(Stem, u32) = .{ .empty, .empty },
    /// Each kind's writer, while its store has one: a block a listed job names is not collected.
    writers: [2]?*spill_writer.Writer = .{ null, null },
    foreign: std.AutoHashMapUnmanaged(Stem, void) = .empty,

    pub const Joining = struct {
        learned: bool = false,
        spilled: bool = false,
        /// The spilt kind's root, where it joins from a root of its own: listed in `roots_file`.
        spilled_root: ?Io.Dir = null,
    };

    /// `root`'s directory for `identity` (made 0700), what a write of a block did not finish deleted, and what
    /// the states of a kind that does not join name read.
    pub fn open(gpa: Allocator, io: Io, root: Io.Dir, identity: u64, joining: Joining) !*Blocks {
        const dir = try root.createDirPathOpen(io, &hex(identity), .{ .open_options = .{ .iterate = true }, .permissions = .fromMode(0o700) });
        const self = gpa.create(Blocks) catch |err| {
            dir.close(io);
            return err;
        };
        self.* = .{ .gpa = gpa, .io = io, .identity = identity, .dir = dir, .waiting = .{ joining.learned, joining.spilled } };
        errdefer self.close();
        var files = dir.iterate();
        while (try files.next(io)) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".blk.part")) dir.deleteFile(io, entry.name) catch {};
        }
        for ([_]Kind{ .learned, .spilled }) |kind| {
            if (!self.waiting[@backingInt(kind)]) _ = try self.readForeign(dir, kind);
        }
        try self.listRoots(root, joining.spilled_root);
        return self;
    }

    pub fn close(self: *Blocks) void {
        for (&self.refs) |*refs| refs.deinit(self.gpa);
        self.foreign.deinit(self.gpa);
        self.dir.close(self.io);
        self.gpa.destroy(self);
    }

    /// What every state of `kind` in `dir` names joins `foreign`; how many blocks they name.
    fn readForeign(self: *Blocks, dir: Io.Dir, kind: Kind) !usize {
        const io = self.io;
        var count: usize = 0;
        var files = dir.iterate();
        while (try files.next(io)) |entry| {
            if (entry.kind != .file or keyIn(kind, entry.name) == null) continue;
            const stems = stemsIn(self.gpa, io, dir, entry.name, self.identity) catch |err| switch (err) {
                error.OutOfMemory => return err,
                // not what its name says: its own store deletes it
                else => continue,
            };
            defer self.gpa.free(stems);
            for (stems) |one| try self.foreign.put(self.gpa, one, {});
            count += stems.len;
        }
        return count;
    }

    /// `roots_file`: each root it lists that the spilt kind does not join from, its states' blocks read into
    /// `foreign`, one whose states name none dropped; `spilled_root` added where it is not `root`.
    fn listRoots(self: *Blocks, root: Io.Dir, spilled_root: ?Io.Dir) !void {
        const io = self.io;
        const gpa = self.gpa;
        var own: [Io.Dir.max_path_bytes]u8 = undefined;
        var theirs: [Io.Dir.max_path_bytes]u8 = undefined;
        const joined: ?[]const u8 = if (spilled_root) |dir| path: {
            const at = try dir.realPath(io, &theirs);
            const here = try root.realPath(io, &own);
            break :path if (std.mem.eql(u8, theirs[0..at], own[0..here])) null else theirs[0..at];
        } else null;
        const text = self.dir.readFileAlloc(io, roots_file, gpa, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => try gpa.dupe(u8, ""),
            else => return err,
        };
        defer gpa.free(text);
        var kept: std.ArrayList(u8) = .empty;
        defer kept.deinit(gpa);
        var listed = false;
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            if (joined != null and std.mem.eql(u8, line, joined.?)) {
                if (listed) continue;
                listed = true;
            } else if (try self.readRoot(line) == 0) continue;
            try kept.appendSlice(gpa, line);
            try kept.append(gpa, '\n');
        }
        if (joined) |path| if (!listed) {
            try kept.appendSlice(gpa, path);
            try kept.append(gpa, '\n');
        };
        if (std.mem.eql(u8, kept.items, text)) return;
        if (kept.items.len == 0) return self.dir.deleteFile(io, roots_file) catch {};
        try self.dir.writeFile(io, .{ .sub_path = roots_file ++ ".part", .data = kept.items });
        try self.dir.rename(roots_file ++ ".part", self.dir, roots_file, io);
    }

    /// The blocks the spilt states under `path` (a root) name, read into `foreign`; 0 where there is none.
    fn readRoot(self: *Blocks, path: []const u8) !usize {
        if (!std.fs.path.isAbsolute(path)) return 0;
        var root = Io.Dir.openDirAbsolute(self.io, path, .{}) catch return 0;
        defer root.close(self.io);
        var dir = root.openDir(self.io, &hex(self.identity), .{ .iterate = true }) catch return 0;
        defer dir.close(self.io);
        return self.readForeign(dir, .spilled);
    }

    /// A store of `kind` has joined: what it holds is in `refs` from now on.
    fn join(self: *Blocks, kind: Kind) void {
        self.waiting[@backingInt(kind)] = false;
    }

    pub fn ready(self: *const Blocks) bool {
        return !self.waiting[0] and !self.waiting[1];
    }

    fn ref(self: *Blocks, kind: Kind, stems: []const Stem) !void {
        const refs = &self.refs[@backingInt(kind)];
        try refs.ensureUnusedCapacity(self.gpa, @intCast(stems.len));
        for (stems) |one| {
            const slot = refs.getOrPutAssumeCapacity(one);
            slot.value_ptr.* = if (slot.found_existing) slot.value_ptr.* + 1 else 1;
        }
    }

    fn unref(self: *Blocks, kind: Kind, stems: []const Stem) void {
        const refs = &self.refs[@backingInt(kind)];
        for (stems) |one| if (refs.getPtr(one)) |count| {
            count.* -|= 1;
            if (count.* == 0) _ = refs.remove(one);
        };
    }

    /// A state of `kind`, held or still to write, names the block.
    pub fn namedBy(self: *const Blocks, kind: Kind, block: Stem) bool {
        if (self.refs[@backingInt(kind)].contains(block)) return true;
        return if (self.writers[@backingInt(kind)]) |writer| writer.wants(block) else false;
    }

    /// Any state names the block: of either kind, in this load or on disk.
    pub fn named(self: *const Blocks, block: Stem) bool {
        return self.foreign.contains(block) or self.namedBy(.learned, block) or self.namedBy(.spilled, block);
    }

    /// A state of a kind other than `kind` names the block: `kind` counts half of it.
    pub fn shared(self: *const Blocks, kind: Kind, block: Stem) bool {
        return self.foreign.contains(block) or self.namedBy(kind.other(), block);
    }

    /// Each block file's bytes, by its stem.
    fn sizes(self: *const Blocks, out: *std.AutoHashMapUnmanaged(Stem, u64)) !void {
        var files = self.dir.iterate();
        while (try files.next(self.io)) |entry| {
            if (entry.kind != .file) continue;
            const one = blocks.stemIn(entry.name) orelse continue;
            const stat = self.dir.statFile(self.io, entry.name, .{}) catch continue;
            try out.put(self.gpa, one, stat.size);
        }
    }

    /// Deletes every block no state names; the bytes freed. None while a kind is still to join.
    pub fn collect(self: *Blocks) !u64 {
        if (!self.ready()) return 0;
        var held: std.AutoHashMapUnmanaged(Stem, u64) = .empty;
        defer held.deinit(self.gpa);
        try self.sizes(&held);
        var freed: u64 = 0;
        var it = held.iterator();
        while (it.next()) |kv| {
            if (self.named(kv.key_ptr.*)) continue;
            const name = blocks.stemFile(kv.key_ptr.*);
            self.dir.deleteFile(self.io, &name) catch continue;
            freed += kv.value_ptr.*;
        }
        return freed;
    }
};

/// The stems of the blocks the state file `name` in `dir` names, by its header, tokens and table.
fn stemsIn(gpa: Allocator, io: Io, dir: Io.Dir, name: []const u8, identity: u64) ![]Stem {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    var head = try headOf(gpa, io, .{ .file = file }, identity, null);
    defer head.deinit(gpa);
    var layout: Layout = try .init(gpa, identity, head.parts, head.tokens);
    defer layout.deinit(gpa);
    return stemsOf(gpa, &layout);
}

pub const Store = struct {
    gpa: Allocator,
    io: Io,
    /// The caller's, opened to iterate: every identity's directory lies in it.
    root: Io.Dir,
    /// This kind's states' files.
    dir: Io.Dir,
    identity: u64,
    kind: Kind,
    /// Where the blocks lie: `dir` itself, or another kind's root's (`Blocks`).
    blocks: *Blocks,
    /// `open`'s: the blocks are this store's alone, and close with it.
    owns_blocks: bool = false,
    /// Bytes of this kind's files at most, other identities' under `root` among them.
    cap: u64,
    /// The last time given to a file: the next is later, whatever the clock's step.
    last: i96 = 0,
    /// States read and written since `open`; one handed to `writer` counts as written.
    read: u64 = 0,
    written: u64 = 0,
    /// The thread that writes the states that leave, where there is one (`spill_writer.zig`): what it has
    /// still to write is held as the files are, and read from its memory.
    writer: ?*spill_writer.Writer = null,
    /// Every state of this identity and kind: its tokens' count and their hash, which is its name, its own
    /// file's bytes, and its blocks.
    held: std.ArrayList(Entry) = .empty,

    pub const Entry = struct { len: u32, key: u64, own: u64, blocks: []Stem };

    /// A store of its own blocks under `root`, of the learned kind (a test's, and one kind's alone).
    pub fn open(gpa: Allocator, io: Io, root: Io.Dir, identity: u64, cap: u64) !Store {
        const own = try Blocks.open(gpa, io, root, identity, .{ .learned = true });
        var self = openKind(gpa, io, root, identity, cap, .learned, own) catch |err| {
            own.close();
            return err;
        };
        self.owns_blocks = true;
        return self;
    }

    /// Opens `root`'s directory for `identity`, made with mode 0700 where there is none: a state is a
    /// client's prompt. What a write that did not finish left, a state missing a block, blocks where they are
    /// not `shared`'s, and what is no kind's file are deleted. The cap is applied once every kind `shared`
    /// waits for has joined (`sweep`).
    pub fn openKind(gpa: Allocator, io: Io, root: Io.Dir, identity: u64, cap: u64, kind: Kind, shared: *Blocks) !Store {
        const dir = try root.createDirPathOpen(io, &hex(identity), .{ .open_options = .{ .iterate = true }, .permissions = .fromMode(0o700) });
        var self: Store = .{ .gpa = gpa, .io = io, .root = root, .dir = dir, .identity = identity, .kind = kind, .blocks = shared, .cap = cap };
        errdefer self.close();
        const apart = !try sameDir(io, dir, shared.dir);
        var files = dir.iterate();
        while (try files.next(io)) |entry| {
            if (entry.kind != .file) continue;
            const name = entry.name;
            const state = keyIn(kind, name) != null or keyIn(kind.other(), name) != null;
            // the other kind's write that did not finish is its own store's to clear, a block's the blocks'
            const others = std.mem.endsWith(u8, name, ".part") and keyIn(kind.other(), name[0..name.len -| 5]) != null;
            const kept = blocks.stemIn(name) != null or std.mem.eql(u8, name, roots_file) or (!apart and std.mem.endsWith(u8, name, ".blk.part"));
            if (!state and !others and !kept) dir.deleteFile(io, name) catch {};
        }
        files = dir.iterate();
        while (try files.next(io)) |entry| {
            const key = keyIn(kind, entry.name) orelse continue;
            self.admit(key) catch |err| switch (err) {
                error.OutOfMemory => return err,
                else => dir.deleteFile(io, entry.name) catch {},
            };
        }
        if (apart) try self.clearApart();
        shared.join(kind);
        if (shared.ready()) try self.sweep();
        return self;
    }

    /// In a kind's directory that is not the blocks': the blocks a load under other flags left there, which
    /// no state of this load reads, deleted, but those the other kind's states there name (a load whose
    /// blocks lay here: their states are its).
    fn clearApart(self: *Store) !void {
        const io = self.io;
        var local: Blocks = .{ .gpa = self.gpa, .io = io, .identity = self.identity, .dir = self.dir, .waiting = .{ false, false } };
        defer local.foreign.deinit(self.gpa);
        _ = try local.readForeign(self.dir, self.kind.other());
        _ = try local.collect();
    }

    /// The state file `key` joins `held` where it is what its name says and its blocks stand whole.
    fn admit(self: *Store, key: u64) !void {
        const io = self.io;
        const name = fileOf(self.kind, key, false);
        const file = try self.dir.openFile(io, name.get(), .{});
        defer file.close(io);
        var head = try headOf(self.gpa, io, .{ .file = file }, self.identity, null);
        defer head.deinit(self.gpa);
        if (keyOf(head.tokens) != key) return error.Mismatch;
        var layout: Layout = try .init(self.gpa, self.identity, head.parts, head.tokens);
        defer layout.deinit(self.gpa);
        const own = try file.length(io);
        if (own != head.prefix + layout.own_bytes) return error.Mismatch;
        for (layout.names) |one| {
            const stat = try self.blocks.dir.statFile(io, &blocks.blockFile(one), .{});
            if (stat.size != layout.blockBytes()) return error.Mismatch;
        }
        try self.index(key, @intCast(head.tokens.len), own, try stemsOf(self.gpa, &layout));
    }

    /// A state of `len` tokens stands under `key`'s name, its own file `own` bytes, naming `stems`, which
    /// the store now owns.
    pub fn index(self: *Store, key: u64, len: u32, own: u64, stems: []Stem) !void {
        self.forget(key);
        errdefer self.gpa.free(stems);
        try self.held.ensureUnusedCapacity(self.gpa, 1);
        try self.blocks.ref(self.kind, stems);
        self.held.appendAssumeCapacity(.{ .len = len, .key = key, .own = own, .blocks = stems });
    }

    fn forget(self: *Store, key: u64) void {
        for (self.held.items, 0..) |entry, i| if (entry.key == key) {
            self.blocks.unref(self.kind, entry.blocks);
            self.gpa.free(entry.blocks);
            _ = self.held.swapRemove(i);
            return;
        };
    }

    /// How many tokens the longest state holds that `prompt` begins with, of those shorter than `under`;
    /// null with none. By the names alone: `find` then compares the tokens.
    pub fn longest(self: *const Store, prompt: []const i32, under: usize) ?usize {
        var best: ?usize = null;
        for (self.held.items) |entry| {
            if (entry.len >= under or entry.len > prompt.len or (best != null and entry.len <= best.?)) continue;
            if (keyOf(prompt[0..entry.len]) == entry.key) best = entry.len;
        }
        return if (self.writer) |writer| writer.longest(prompt, under, best) else best;
    }

    /// The store's files stay; what its states name stays named in the blocks (`Blocks.refs`).
    pub fn close(self: *Store) void {
        if (self.writer) |writer| writer.close();
        for (self.held.items) |entry| self.gpa.free(entry.blocks);
        self.held.deinit(self.gpa);
        self.dir.close(self.io);
        if (self.owns_blocks) self.blocks.close();
        self.* = undefined;
    }

    pub fn has(self: *const Store, tokens: []const i32) bool {
        const key = keyOf(tokens);
        if (self.writer) |writer| if (writer.has(key)) return true;
        return self.onDisk(key);
    }

    pub fn onDisk(self: *const Store, key: u64) bool {
        const name = fileOf(self.kind, key, false);
        _ = self.dir.statFile(self.io, name.get(), .{}) catch return false;
        return true;
    }

    fn touch(self: *Store, name: []const u8) void {
        const now = Io.Timestamp.now(self.io, .real).nanoseconds;
        self.last = @max(now, self.last + 1);
        self.dir.setTimestamps(self.io, name, .{ .modify_timestamp = .{ .new = .{ .nanoseconds = self.last } } }) catch {};
    }

    /// A state's files open for a read: its own and each block's, and where the image's bytes lie in them.
    const Disk = struct {
        own: Io.File,
        blocks: []Io.File,
        layout: Layout,
        prefix: u64,

        fn read(self: *const Disk, io: Io, dest: []u8, offset: u64) !usize {
            var done: usize = 0;
            while (done < dest.len) {
                const at = offset + done;
                const rest = dest[done..];
                const got = if (at < self.prefix) got: {
                    const n: usize = @intCast(@min(rest.len, self.prefix - at));
                    break :got try self.own.readPositionalAll(io, rest[0..n], at);
                } else got: {
                    const into = at - self.prefix;
                    if (into >= self.layout.total) break;
                    const one = self.layout.locate(into);
                    const within = into - one.at;
                    const n: usize = @intCast(@min(rest.len, one.len - within));
                    break :got if (one.block) |k|
                        try self.blocks[k].readPositionalAll(io, rest[0..n], @sizeOf(blocks.BlockHeader) + one.offset + within)
                    else
                        try self.own.readPositionalAll(io, rest[0..n], self.prefix + one.offset + within);
                };
                done += got;
                if (got == 0) break;
            }
            return done;
        }

        fn close(self: *Disk, io: Io, gpa: Allocator) void {
            self.own.close(io);
            for (self.blocks) |file| file.close(io);
            gpa.free(self.blocks);
            self.layout.deinit(gpa);
            gpa.destroy(self);
        }
    };

    /// Where a state's bytes are read from: its files, or the writer's job that has still to write them (its
    /// image whole); a state file alone while its head is read.
    const From = union(enum) {
        file: Io.File,
        disk: *Disk,
        job: *spill_writer.Job,

        fn read(self: From, io: Io, dest: []u8, offset: u64) !usize {
            return switch (self) {
                .file => |file| file.readPositionalAll(io, dest, offset),
                .disk => |disk| disk.read(io, dest, offset),
                .job => |job| job.read(dest, offset),
            };
        }

        fn close(self: From, io: Io, gpa: Allocator) void {
            switch (self) {
                .file => |file| file.close(io),
                .disk => |disk| disk.close(io, gpa),
                .job => |job| job.release(),
            }
        }
    };

    /// A state being read, part by part in the order written.
    pub const Image = struct {
        io: Io,
        from: From,
        /// Each part's bytes and rows.
        parts: []Part,
        next: usize = 0,
        /// Bytes of part `next` read so far.
        done: u64 = 0,
        offset: u64,

        /// The next part, which must be `dest.len` bytes.
        pub fn part(self: *Image, dest: []u8) !void {
            if (self.done != 0 or self.next == self.parts.len or self.parts[self.next].bytes != dest.len) return error.Mismatch;
            return self.some(dest);
        }

        /// The next `dest.len` bytes of the part being read, which holds as many at least: a part read in
        /// pieces (a pool's rows lie in several extents).
        pub fn some(self: *Image, dest: []u8) !void {
            if (self.next == self.parts.len or self.parts[self.next].bytes - self.done < dest.len) return error.Mismatch;
            if (try self.from.read(self.io, dest, self.offset) != dest.len) return error.Mismatch;
            self.offset += dest.len;
            self.done += dest.len;
            if (self.done == self.parts[self.next].bytes) {
                self.next += 1;
                self.done = 0;
            }
        }

        pub fn close(self: *Image, gpa: Allocator) void {
            self.from.close(self.io, gpa);
            gpa.free(self.parts);
            self.* = undefined;
        }
    };

    /// The state for `tokens`, where one is held and is what its name says: the writer's job where it is
    /// still listed, else its files. `gpa` holds the image's table and files.
    pub fn find(self: *Store, gpa: Allocator, tokens: []const i32) !?Image {
        const io = self.io;
        const key = keyOf(tokens);
        if (self.writer) |writer| if (writer.find(key)) |job| {
            const image = self.fromJob(gpa, job, tokens) catch |err| {
                job.release();
                return if (err == error.Mismatch) null else err;
            };
            self.read += 1;
            return image;
        };
        const name = fileOf(self.kind, key, false);
        const file = self.dir.openFile(io, name.get(), .{}) catch |err| return if (err == error.FileNotFound) null else err;
        const image = self.fromDisk(gpa, file, tokens) catch |err| {
            if (err != error.Mismatch and err != error.FileNotFound) return err;
            // a block missing or torn, or a state not what its name says: a miss, and not met again
            self.discard(tokens);
            return null;
        };
        self.touch(name.get());
        self.read += 1;
        return image;
    }

    fn fromJob(self: *Store, gpa: Allocator, job: *spill_writer.Job, tokens: []const i32) !Image {
        const head = try headOf(self.gpa, self.io, .{ .job = job }, self.identity, tokens);
        errdefer gpa.free(head.parts);
        defer gpa.free(head.tokens);
        var end = head.prefix;
        for (head.parts) |one| end = std.math.add(u64, end, one.bytes) catch return error.Mismatch;
        if (job.bytes != end) return error.Mismatch;
        return .{ .io = self.io, .from = .{ .job = job }, .parts = head.parts, .offset = head.prefix };
    }

    /// `file`, `tokens`'s state file, and its blocks, each checked; the file is the image's, or closed.
    fn fromDisk(self: *Store, gpa: Allocator, file: Io.File, tokens: []const i32) !Image {
        const io = self.io;
        var owned = true;
        defer if (owned) file.close(io);
        const head = try headOf(self.gpa, io, .{ .file = file }, self.identity, tokens);
        errdefer gpa.free(head.parts);
        defer gpa.free(head.tokens);
        var layout: Layout = try .init(gpa, self.identity, head.parts, tokens);
        errdefer layout.deinit(gpa);
        if (try file.length(io) != head.prefix + layout.own_bytes) return error.Mismatch;
        const files = try gpa.alloc(Io.File, layout.names.len);
        var opened: usize = 0;
        errdefer {
            for (files[0..opened]) |one| one.close(io);
            gpa.free(files);
        }
        for (layout.names, files) |name, *slot| {
            slot.* = try self.blocks.dir.openFile(io, &blocks.blockFile(name), .{});
            opened += 1;
            var header: blocks.BlockHeader = undefined;
            if (try slot.readPositionalAll(io, std.mem.asBytes(&header), 0) != @sizeOf(blocks.BlockHeader)) return error.Mismatch;
            if (!std.mem.eql(u8, &header.magic, &blocks.block_magic) or header.identity != self.identity or !std.mem.eql(u8, &header.name, &name) or header.bytes != layout.block_bytes) return error.Mismatch;
            if (try slot.length(io) != layout.blockBytes()) return error.Mismatch;
        }
        const disk = try gpa.create(Disk);
        disk.* = .{ .own = file, .blocks = files, .layout = layout, .prefix = head.prefix };
        owned = false;
        return .{ .io = io, .from = .{ .disk = disk }, .parts = head.parts, .offset = head.prefix };
    }

    /// The state for `tokens` goes: one whose parts were not what the model reads, or that a block is
    /// missing from; the blocks no other state of either kind names with it.
    pub fn discard(self: *Store, tokens: []const i32) void {
        const key = keyOf(tokens);
        if (self.writer) |writer| writer.discard(key);
        const name = fileOf(self.kind, key, false);
        self.dir.deleteFile(self.io, name.get()) catch {};
        self.forget(key);
        _ = self.blocks.collect() catch |err| std.log.warn("the kept states' blocks are not collected: {t}", .{err});
    }

    /// The files of a state of `parts` over `tokens`, before they are written.
    pub fn plan(self: *const Store, gpa: Allocator, tokens: []const i32, parts: []const Part) !Plan {
        var layout: Layout = try .init(gpa, self.identity, parts, tokens);
        errdefer layout.deinit(gpa);
        const prefix = prefixBytes(tokens.len, parts.len);
        var disk = prefix + layout.own_bytes;
        for (layout.names) |name| {
            if (!self.blocks.named(blocks.stem(name))) disk += layout.blockBytes();
        }
        return .{ .layout = layout, .prefix = prefix, .image = prefix + layout.total, .disk = disk };
    }

    /// Whether files of `bytes` fit the cap at all.
    pub fn fits(self: *const Store, bytes: u64) bool {
        return bytes <= self.cap;
    }

    /// Writes the state for `tokens` of `plan`'s layout, its image from `source`, on this thread; `stage`
    /// carries the bytes. The store is then within its cap again.
    pub fn put(self: *Store, tokens: []const i32, planned: *const Plan, source: Source, stage: []u8) !void {
        const key = keyOf(tokens);
        _ = try writeFiles(self.io, self.blocks.dir, self.dir, self.kind, self.identity, key, &planned.layout, planned.prefix, source, stage, null);
        const name = fileOf(self.kind, key, false);
        self.touch(name.get());
        self.written += 1;
        try self.index(key, @intCast(tokens.len), planned.prefix + planned.layout.own_bytes, try stemsOf(self.gpa, &planned.layout));
        self.sweep() catch |err| std.log.warn("the kept states are not held to their cap: {t}", .{err});
    }

    /// `put` of parts in host memory, rows by `per` where it is given (empty: none are): a test's.
    pub fn putHost(self: *Store, tokens: []const i32, bytes: []const []const u8, per: []const u32) !void {
        const gpa = self.gpa;
        const table = try gpa.alloc(Part, bytes.len);
        defer gpa.free(table);
        for (table, bytes, 0..) |*one, part, i| one.* = .{ .bytes = part.len, .per = if (per.len > 0) per[i] else 0 };
        var planned = try self.plan(gpa, tokens, table);
        defer planned.deinit(gpa);
        const prefix = try prefixOf(gpa, self.identity, tokens, table);
        defer gpa.free(prefix);
        const slices: Slices = .{ .parts = bytes };
        const joined: Joined = .{ .prefix = prefix, .parts = slices.source() };
        var stage: [4096]u8 = undefined;
        try self.put(tokens, &planned, joined.source(), &stage);
    }

    /// A file the cap may delete: another identity's under the root (`dir`, `name`), or a state of this
    /// store's (`key`).
    const Held = struct { key: ?u64, dir: [16]u8, name: [40]u8, len: u8, size: u64, time: i96 };

    fn sooner(_: void, a: Held, b: Held) bool {
        if ((a.key == null) != (b.key == null)) return a.key == null;
        return a.time < b.time;
    }

    /// Deletes files until this kind's are within `cap`, and every block no state names.
    pub fn sweep(self: *Store) !void {
        return self.clear(0);
    }

    /// The bytes this kind holds: its states' own files, the blocks only its states name, and half of each
    /// block another kind's states name too (`sizes`: each block's).
    pub fn holds(self: *const Store, sizes: *const std.AutoHashMapUnmanaged(Stem, u64)) !u64 {
        var seen: std.AutoHashMapUnmanaged(Stem, void) = .empty;
        defer seen.deinit(self.gpa);
        var total: u64 = 0;
        for (self.held.items) |entry| {
            total += entry.own;
            for (entry.blocks) |one| {
                if ((try seen.getOrPut(self.gpa, one)).found_existing) continue;
                const bytes = sizes.get(one) orelse 0;
                total += if (self.blocks.shared(self.kind, one)) bytes / 2 else bytes;
            }
        }
        return total;
    }

    /// `holds` over the blocks on disk now.
    pub fn holding(self: *const Store) !u64 {
        var sizes: std.AutoHashMapUnmanaged(Stem, u64) = .empty;
        defer sizes.deinit(self.gpa);
        try self.blocks.sizes(&sizes);
        return self.holds(&sizes);
    }

    /// `sweep` with `reserve` bytes of the cap kept for files still to write: other identities' files under
    /// the root first, the oldest first, then this kind's states, the least recently used first, each block
    /// going with the last state of either kind that names it. Nothing before every kind has joined the
    /// blocks.
    pub fn clear(self: *Store, reserve: u64) !void {
        if (!self.blocks.ready()) return;
        const io = self.io;
        const gpa = self.gpa;
        const cap = self.cap -| reserve;
        var list: std.ArrayList(Held) = .empty;
        defer list.deinit(gpa);
        var others: u64 = 0;
        const own = hex(self.identity);
        var any_other = false;
        var identities = self.root.iterate();
        while (try identities.next(io)) |entry| {
            if (entry.kind != .directory or entry.name.len != own.len or std.mem.eql(u8, entry.name, &own)) continue;
            any_other = true;
            var dir = self.root.openDir(io, entry.name, .{ .iterate = true }) catch continue;
            defer dir.close(io);
            var files = dir.iterate();
            while (try files.next(io)) |file| {
                if (file.kind != .file or file.name.len > 40) continue;
                const stat = dir.statFile(io, file.name, .{}) catch continue;
                others += stat.size;
                var held: Held = .{ .key = null, .dir = entry.name[0..16].*, .name = undefined, .len = @intCast(file.name.len), .size = stat.size, .time = stat.mtime.nanoseconds };
                @memcpy(held.name[0..held.len], file.name);
                try list.append(gpa, held);
            }
        }
        var sizes: std.AutoHashMapUnmanaged(Stem, u64) = .empty;
        defer sizes.deinit(gpa);
        try self.blocks.sizes(&sizes);
        var total = others + try self.holds(&sizes);
        if (total > cap) {
            for (self.held.items) |entry| {
                const name = fileOf(self.kind, entry.key, false);
                const stat = self.dir.statFile(io, name.get(), .{}) catch continue;
                try list.append(gpa, .{ .key = entry.key, .dir = undefined, .name = undefined, .len = 0, .size = stat.size, .time = stat.mtime.nanoseconds });
            }
            std.mem.sort(Held, list.items, {}, sooner);
            for (list.items) |held| {
                if (total <= cap) break;
                if (held.key) |key| {
                    const name = fileOf(self.kind, key, false);
                    self.dir.deleteFile(io, name.get()) catch continue;
                    self.forget(key);
                } else {
                    var dir = self.root.openDir(io, &held.dir, .{}) catch continue;
                    defer dir.close(io);
                    dir.deleteFile(io, held.name[0..held.len]) catch continue;
                    others -|= held.size;
                }
                total = others + try self.holds(&sizes);
            }
        }
        _ = try self.blocks.collect();
        // an identity no file is left of: its directory goes, which fails while it holds anything
        if (any_other) {
            identities = self.root.iterate();
            while (try identities.next(io)) |entry| {
                if (entry.kind == .directory and entry.name.len == own.len and !std.mem.eql(u8, entry.name, &own)) self.root.deleteDir(io, entry.name) catch {};
            }
        }
    }
};

const Head = struct {
    tokens: []i32,
    parts: []Part,
    prefix: u64,

    fn deinit(self: *Head, gpa: Allocator) void {
        gpa.free(self.tokens);
        gpa.free(self.parts);
    }
};

/// An image's header, tokens and table, the tokens compared with `tokens` where it is given.
fn headOf(gpa: Allocator, io: Io, from: Store.From, identity: u64, tokens: ?[]const i32) !Head {
    var header: Header = undefined;
    if (try from.read(io, std.mem.asBytes(&header), 0) != @sizeOf(Header)) return error.Mismatch;
    if (!std.mem.eql(u8, &header.magic, &magic) or header.identity != identity or header.tokens == 0 or header.tokens > 1 << 26 or header.parts > 1 << 16) return error.Mismatch;
    if (tokens) |want| if (header.tokens != want.len) return error.Mismatch;
    const held = try gpa.alloc(i32, header.tokens);
    errdefer gpa.free(held);
    if (try from.read(io, std.mem.sliceAsBytes(held), @sizeOf(Header)) != held.len * 4) return error.Mismatch;
    if (tokens) |want| if (!std.mem.eql(i32, held, want)) return error.Mismatch;
    const parts = try gpa.alloc(Part, header.parts);
    errdefer gpa.free(parts);
    if (try from.read(io, std.mem.sliceAsBytes(parts), @sizeOf(Header) + held.len * 4) != parts.len * @sizeOf(Part)) return error.Mismatch;
    return .{ .tokens = held, .parts = parts, .prefix = prefixBytes(held.len, parts.len) };
}

/// Whether two directories are one, by their paths.
pub fn sameDir(io: Io, a: Io.Dir, b: Io.Dir) !bool {
    var first: [Io.Dir.max_path_bytes]u8 = undefined;
    var second: [Io.Dir.max_path_bytes]u8 = undefined;
    const x = try a.realPath(io, &first);
    const y = try b.realPath(io, &second);
    return std.mem.eql(u8, first[0..x], second[0..y]);
}

/// The stems of a layout's blocks, in `gpa`'s memory.
pub fn stemsOf(gpa: Allocator, layout: *const Layout) ![]Stem {
    const out = try gpa.alloc(Stem, layout.names.len);
    for (layout.names, out) |name, *one| one.* = blocks.stem(name);
    return out;
}

/// A checkpoint directory as an identity reads it: `config.json`'s bytes and each shard's name and size, in
/// the names' order. Weights of the same sizes under the same names are told apart by what they compute
/// (the caller's probe), not here: hashing 60 GB would be a load's time again.
pub fn checkpointHash(gpa: Allocator, io: Io, dir: Io.Dir) !u64 {
    var names: std.ArrayList([]u8) = .empty;
    defer {
        for (names.items) |name| gpa.free(name);
        names.deinit(gpa);
    }
    var entries = dir.iterate();
    while (try entries.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const name = try gpa.dupe(u8, entry.name);
        errdefer gpa.free(name);
        try names.append(gpa, name);
    }
    std.mem.sort([]u8, names.items, {}, struct {
        fn before(_: void, a: []u8, b: []u8) bool {
            return std.mem.order(u8, a, b) == .lt;
        }
    }.before);
    var hasher: std.hash.Wyhash = .init(0);
    for (names.items) |name| {
        const stat = try dir.statFile(io, name, .{});
        hasher.update(name);
        hasher.update(std.mem.asBytes(&@as(u64, stat.size)));
    }
    const config = try dir.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
    defer gpa.free(config);
    hasher.update(config);
    return hasher.final();
}
