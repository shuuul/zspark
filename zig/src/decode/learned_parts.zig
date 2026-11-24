//! A state's parts to and from the files of `learned.zig` (`serve --learn`, spec 064; `serve --spill-gib`,
//! spec 074), for any model: a part is runs of bytes in device buffers or host memory (`Piece`), and what each
//! part is the model says (Flash Next's `models/qwen4/session/learnt.zig`, GLM's `models/glm5/session/learnt.zig`).
//! Here are the table a file's parts make, the copy to the store's writer or the files, the read back into the
//! model's buffers, a stop's wait for the writer, and the stores both kinds open under their roots once a model
//! has hashed its identity (`open`). Moved from Flash Next's `learnt.zig` when GLM took the same files (spec 108
//! WS-04; the root `AGENTS.md`: a rule two models use lies in `decode/`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const units = @import("../cuda/units.zig");
const learned = @import("learned.zig");
const spill = @import("spill.zig");
const spill_writer = @import("spill_writer.zig");
pub const Floor = spill.Floor;
pub const Store = learned.Store;

/// A run of bytes of a file's part.
pub const Piece = struct {
    source: union(enum) {
        /// `len` bytes of a device buffer from byte `from`.
        device: struct { buffer: cuda.Buffer, from: usize = 0, len: usize },
        host: []u8,
    },
    /// It goes on the part the piece before it began: a part's rows lie in several buffers (a sequence left
    /// holds its upper rows aside, a pool's chain lies in extents), and the file is the same.
    joined: bool = false,
    /// On a part's first piece: the positions a row of the part stands for, 0 where it is not rows (the
    /// state's own buffers). A cache's rows are 1, its pooled keys `ratio`.
    per: u32 = 0,

    pub fn size(self: Piece) usize {
        return switch (self.source) {
            .device => |part| part.len,
            .host => |bytes| bytes.len,
        };
    }
};

/// One array's first `count` rows, a row every `per` positions: `low`'s rows below `from`, and `high`'s
/// first rows from there on.
pub fn split(gpa: Allocator, list: *std.ArrayList(Piece), low: cuda.Buffer, low_rows: usize, high: ?cuda.Buffer, from: usize, count: usize, per: usize) !void {
    const row = low.bytes() / low_rows;
    const below = @min(count, from);
    try list.append(gpa, .{ .source = .{ .device = .{ .buffer = low, .len = below * row } }, .per = @intCast(per) });
    if (count > below) try list.append(gpa, .{ .source = .{ .device = .{ .buffer = high.?, .len = (count - below) * row } }, .joined = true });
}

pub fn largest(parts: []const Piece) usize {
    var most: usize = 0;
    for (parts) |piece| most = @max(most, piece.size());
    return most;
}

/// Each part's bytes and rows, the pieces joined.
pub fn tableOf(gpa: Allocator, parts: []const Piece) ![]learned.Part {
    var count: usize = 0;
    for (parts) |piece| count += @intFromBool(!piece.joined);
    const table = try gpa.alloc(learned.Part, count);
    @memset(table, .{ .bytes = 0 });
    var at: usize = 0;
    for (parts, 0..) |piece, i| {
        if (i > 0 and !piece.joined) at += 1;
        if (!piece.joined) table[at].per = piece.per;
        table[at].bytes += piece.size();
    }
    return table;
}

/// The pieces' bytes as a store reads them, back to back: a device piece's downloaded, a host piece's as
/// they stand.
const Pieces = struct {
    parts: []const Piece,
    /// Where each piece begins.
    starts: []u64,

    fn init(gpa: Allocator, parts: []const Piece) !Pieces {
        const starts = try gpa.alloc(u64, parts.len);
        var at: u64 = 0;
        for (parts, starts) |piece, *start| {
            start.* = at;
            at += piece.size();
        }
        return .{ .parts = parts, .starts = starts };
    }

    fn source(self: *const Pieces) learned.Source {
        return .{ .context = self, .view = view };
    }

    fn view(context: *const anyopaque, offset: u64, dest: []u8) anyerror![]const u8 {
        const self: *const Pieces = @ptrCast(@alignCast(context));
        // the last piece that begins at `offset` or below, which an empty one never is but at the end
        var lo: usize = 0;
        var hi: usize = self.starts.len;
        while (hi - lo > 1) {
            const mid = (lo + hi) / 2;
            if (self.starts[mid] <= offset) lo = mid else hi = mid;
        }
        const piece = self.parts[lo];
        const within: usize = @intCast(offset - self.starts[lo]);
        if (within >= piece.size()) return &.{};
        const n = @min(dest.len, piece.size() - within);
        return switch (piece.source) {
            .host => |bytes| bytes[within..][0..n],
            .device => |part| {
                try part.buffer.downloadAt(part.from + within, dest[0..n]);
                return dest[0..n];
            },
        };
    }
};

/// What `put` did: the bytes the state adds on disk (its file and the blocks no state held names), 0 where
/// nothing was written, and whether they went to the store's writer and not to the files.
pub const Put = struct { bytes: u64 = 0, handed: bool = false };

/// A state's bytes when a caller logs them.
pub fn told(what: Put) []const u8 {
    return if (what.handed) "copied for the writer" else "written";
}

/// Writes the state for `tokens`, of `parts`, unless the store holds it or its cap is under it. Where the
/// store has a writer (`spill_writer.zig`) the bytes are copied to host memory and the writer's thread
/// writes the files; with no room there the state is not written, unless `until`, a stop's, lets it wait.
/// Either way the files keep `floor`'s bytes free on their disk, where there is a floor (`spill.Floor`), and
/// a block on disk already is not written again.
pub fn put(gpa: Allocator, store: *Store, floor: ?*Floor, tokens: []const i32, parts: []const Piece, until: ?std.Io.Clock.Timestamp) !Put {
    const table = try tableOf(gpa, parts);
    defer gpa.free(table);
    if (store.writer) |writer| {
        var taking = try writer.begin(tokens, table, until) orelse return .{};
        errdefer taking.cancel();
        try cuda.synchronize();
        for (parts) |piece| switch (piece.source) {
            .host => |bytes| taking.copy(bytes),
            .device => |part| {
                var done: usize = 0;
                while (done < part.len) {
                    const dest = taking.span(part.len - done);
                    try part.buffer.downloadAt(part.from + done, dest);
                    done += dest.len;
                }
            },
        };
        const bytes = taking.job.disk;
        taking.commit();
        return .{ .bytes = bytes, .handed = true };
    }
    if (store.has(tokens)) return .{};
    var planned = try store.plan(gpa, tokens, table);
    defer planned.deinit(gpa);
    if (!store.fits(planned.disk) or !spill.admitted(floor, store.blocks.dir.handle, planned.disk, tokens.len)) return .{};
    var written = false;
    defer if (floor) |held| held.finish(store.blocks.dir.handle, planned.disk, written);
    const prefix = try learned.prefixOf(gpa, store.identity, tokens, table);
    defer gpa.free(prefix);
    const stage = try gpa.alloc(u8, @max(1, @min(largest(parts), 8 << 20)));
    defer gpa.free(stage);
    var source: Pieces = try .init(gpa, parts);
    defer gpa.free(source.starts);
    const joined: learned.Joined = .{ .prefix = prefix, .parts = source.source() };
    try cuda.synchronize();
    try store.put(tokens, &planned, joined.source(), stage);
    written = true;
    return .{ .bytes = planned.disk };
}

/// When a stop that began at `began` writes no more.
pub fn stopBy(began: std.Io.Timestamp) std.Io.Clock.Timestamp {
    return began.addDuration(.fromSeconds(spill.flush_seconds)).withClock(.awake);
}

/// The fixed prompt the identity's probe runs, in tokens.
pub const probe_tokens = 512;

/// The identity's probe's prompt, both models': `probe_tokens`, or what fits `capacity`. The caller frees it.
pub fn probePrompt(gpa: Allocator, capacity: usize, vocab: usize) ![]i32 {
    const count = @min(probe_tokens, capacity - 2);
    const prompt = try gpa.alloc(i32, count);
    // any tokens do: fixed ones, spread over the vocabulary
    for (prompt, 0..) |*token, i| token.* = @intCast((i * 2654435761 + 12345) % vocab);
    return prompt;
}

/// The end of a stop's writing: what the store's writer has still to write is waited for, till `stopBy`.
pub fn settle(store: *Store, began: std.Io.Timestamp) void {
    const writer = store.writer orelse return;
    if (!writer.drain(stopBy(began))) return std.log.warn("the states that left are not all written after {d} s", .{spill.flush_seconds});
    std.log.info("the states that left are written, {d:.3} s after the stop began, of {d} s at most", .{ units.secondsSince(writer.io, began), spill.flush_seconds });
}

/// Before a row is written: the file is this model's state, part for part.
pub fn matches(gpa: Allocator, image: *const Store.Image, parts: []const Piece) !void {
    const table = try tableOf(gpa, parts);
    defer gpa.free(table);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(table), std.mem.sliceAsBytes(image.parts))) return error.Mismatch;
}

/// Reads the file's parts into `parts`, which `matches` held to it.
pub fn fill(gpa: Allocator, image: *Store.Image, parts: []const Piece) !void {
    const staging = try gpa.alloc(u8, largest(parts));
    defer gpa.free(staging);
    try cuda.synchronize();
    for (parts) |piece| switch (piece.source) {
        .host => |bytes| try image.some(bytes),
        .device => |part| if (part.len > 0) {
            try image.some(staging[0..part.len]);
            try part.buffer.uploadAt(part.from, staging[0..part.len]);
        },
    };
}

/// Hashes `parts` as they stand into `hasher`: the state a model's identity probe leaves.
pub fn hash(gpa: Allocator, hasher: *std.hash.Wyhash, parts: []const Piece) !void {
    const staging = try gpa.alloc(u8, largest(parts));
    defer gpa.free(staging);
    try cuda.synchronize();
    for (parts) |piece| switch (piece.source) {
        .host => |bytes| hasher.update(bytes),
        .device => |part| {
            try part.buffer.downloadAt(part.from, staging[0..part.len]);
            hasher.update(staging[0..part.len]);
        },
    };
}

/// Where one kind of state is kept: the directory asked for, or the home directory's `.cache/zspark/NAME`,
/// and the bytes its files hold at most.
pub const Root = struct { dir: ?[]const u8, gib: f64 };

/// What `serve` was asked: `--learn` (`--learn-dir`, `--learn-gib`), `--spill-gib` (`--spill-dir`), and what
/// the identity is hashed from beside the model's probe.
pub const Asked = struct {
    learn: ?Root = null,
    spill: ?Root = null,
    home: ?[]const u8,
    /// The checkpoint's directory.
    checkpoint: []const u8,
    commit: []const u8,
    /// The build's tree differed from `commit`: its kernels are no commit's, and it reads and writes nothing.
    dirty: bool,
    /// Bytes a disk keeps free under both kinds' files (`--learn-min-free-gib`, `spill.Floor`).
    min_free: u64 = (Floor{}).floor,
};

/// What the identity's hash starts from, beside the model's probe: the files' form, the build, `own` (what
/// the model's caches are stored as) and the checkpoint's `config.json` and shard sizes.
pub fn base(gpa: Allocator, io: std.Io, asked: Asked, own: []const u8) !u64 {
    var hasher: std.hash.Wyhash = .init(0);
    hasher.update(&learned.magic);
    hasher.update(asked.commit);
    hasher.update(own);
    var dir = try std.Io.Dir.cwd().openDir(io, asked.checkpoint, .{ .iterate = true });
    defer dir.close(io);
    hasher.update(std.mem.asBytes(&try learned.checkpointHash(gpa, io, dir)));
    return hasher.final();
}

pub const Learning = struct {
    gpa: Allocator,
    io: std.Io,
    learn: ?Kept = null,
    spill: ?Kept = null,
    /// Both kinds' blocks (`learned.Blocks`, spec 101): under the learned root where it is kept, else the
    /// spilt one's.
    blocks: ?*learned.Blocks = null,
    /// A kind's root while its store is not open.
    roots: [2]?std.Io.Dir = .{ null, null },
    /// The one floor both kinds' writes keep (TensorFold's single store has one).
    floor: Floor = .{},

    pub const Kept = struct { root: std.Io.Dir, store: Store };

    pub fn close(self: *Learning) void {
        for ([_]*?Kept{ &self.learn, &self.spill }) |kept| if (kept.*) |*held| {
            held.store.close();
            held.root.close(self.io);
        };
        for (self.roots) |root| if (root) |dir| dir.close(self.io);
        if (self.blocks) |shared| shared.close();
        self.gpa.destroy(self);
    }
};

/// Opens the stores `asked` names under the identity `id` a model hashed: each kind's root, the blocks both
/// share (under the learned root, as TensorFold's one store does, else the spilt one), each store's cap, and
/// the spilt store's writer, which keeps the floor. A root that cannot be made is said and left out.
pub fn open(gpa: Allocator, io: std.Io, asked: Asked, id: u64) !*Learning {
    const held = try gpa.create(Learning);
    held.* = .{ .gpa = gpa, .io = io, .floor = .{ .floor = asked.min_free } };
    errdefer held.close();
    const asks = [_]?Root{ asked.learn, asked.spill };
    const flags = [_][]const u8{ "--learn", "--spill-gib" };
    var paths: [2]?[]const u8 = .{ null, null };
    var made: [2]bool = .{ false, false };
    defer for (paths, made) |path, own| if (own) gpa.free(path.?);
    for (asks, flags, [_][]const u8{ "learned", "spilled" }, 0..) |wanted, flag, name, k| {
        const root_asked = wanted orelse continue;
        const path = root_asked.dir orelse try std.fs.path.join(gpa, &.{ asked.home orelse {
            std.log.warn("{s}: no HOME and no directory given, so no state is kept", .{flag});
            continue;
        }, ".cache", "zspark", name });
        paths[k] = path;
        made[k] = root_asked.dir == null;
        held.roots[k] = std.Io.Dir.cwd().createDirPathOpen(io, path, .{ .open_options = .{ .iterate = true }, .permissions = .fromMode(0o700) }) catch |err| {
            std.log.warn("{s}: {s}: {t}, so no state is kept", .{ flag, path, err });
            continue;
        };
    }
    const first: ?usize = if (held.roots[0] != null) 0 else if (held.roots[1] != null) 1 else null;
    if (first) |at| {
        held.blocks = try .open(gpa, io, held.roots[at].?, id, .{ .learned = held.roots[0] != null, .spilled = held.roots[1] != null, .spilled_root = if (at == 0) held.roots[1] else null });
        for ([_]*?Learning.Kept{ &held.learn, &held.spill }, [_]learned.Kind{ .learned, .spilled }, 0..) |kept, kind, k| {
            const root = held.roots[k] orelse continue;
            const cap: u64 = @intFromFloat(@min(@max(asks[k].?.gib, 0), 1 << 20) * (1 << 30));
            const store: Store = try .openKind(gpa, io, root, id, cap, kind, held.blocks.?);
            held.roots[k] = null;
            kept.* = .{ .root = root, .store = store };
            std.log.info("{s}: states are kept in {s}/{x:0>16}, {d:.1} GiB at most, {d} held; their blocks in {s}/{x:0>16}", .{ flags[k], paths[k].?, id, asks[k].?.gib, store.held.items.len, paths[at].?, id });
        }
        // the first kind's cap is applied once the other's states are known
        for ([_]*?Learning.Kept{ &held.learn, &held.spill }) |kept| if (kept.*) |*one| try one.store.sweep();
    }
    // the writer starts here, where the store stays, so that it holds the floor whoever takes the store
    if (held.spill) |*kept| {
        spill_writer.Writer.start(&kept.store);
        if (kept.store.writer) |writer| writer.floor = &held.floor;
    }
    return held;
}
