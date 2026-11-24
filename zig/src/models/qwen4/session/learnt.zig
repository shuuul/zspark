//! A session's state to and from disk (`serve --learn`, spec 064; the files are `decode/learned.zig`'s): the
//! state where the prefix a prompt shares with other conversations ends (`ReplyOptions.kept`) is written once
//! its reply is done, and the first prompt over that prefix after a restart starts from it.
//!
//! A file holds what a state at position p is on this model: a `forward.Snapshot`'s buffers (the DeltaNet
//! states, the conv inputs, the PLE's tails and histories), then each attention cache's rows below p and its
//! pooled keys' complete blocks, then the head's cache's, whose entry p - 1 was absorbed with token p: the
//! file's tokens are the sequence through token p, and it serves a prompt that begins with them. A part of
//! rows says so (`Piece.per`), and the store lays those in blocks that states over the same tokens share
//! (`decode/learned_blocks.zig`, spec 101).
//!
//! A row's bits never depend on the rows beside it, so the rows read are the rows the prompt would have
//! written, whatever windows wrote them. That holds for one build on one checkpoint with one cache storage:
//! `identity` hashes the state a fixed prompt leaves into the directory's name, and `serve` the rest.

const std = @import("std");
const Allocator = std.mem.Allocator;
const attention = @import("../blocks/attention.zig");
const cuda = @import("../../../cuda/cuda.zig");
const forward = @import("../forward.zig");
const learned = @import("../../../decode/learned.zig");
const parked_file = @import("parked.zig");
const spill = @import("../../../decode/spill.zig");
const spill_writer = @import("../../../decode/spill_writer.zig");
const parts_file = @import("../../../decode/learned_parts.zig");
const units = @import("../../../cuda/units.zig");
const session_file = @import("session.zig");
const Session = session_file.Session;
pub const Floor = spill.Floor;
pub const Store = learned.Store;
pub const keyOf = learned.keyOf;
// a file's parts, the copy and the read, the stores: `decode/learned_parts.zig`, shared with GLM (spec 108)
pub const Piece = parts_file.Piece;
pub const split = parts_file.split;
pub const Put = parts_file.Put;
pub const told = parts_file.told;
pub const put = parts_file.put;
pub const stopBy = parts_file.stopBy;
pub const settle = parts_file.settle;
pub const matches = parts_file.matches;
pub const fill = parts_file.fill;
pub const Root = parts_file.Root;
pub const Asked = parts_file.Asked;
pub const Learning = parts_file.Learning;

/// A state is read where it stands this many tokens or more above where the prompt would start without it:
/// nearer, running them is as soon as reading 110 MiB and the rows. A test lowers it.
pub var worth: usize = 256;

/// A cache's rows below `at` and its pooled keys' complete blocks; `span`: what a sequence left holds of them
/// aside, from its `from` up.
fn rowsOf(gpa: Allocator, list: *std.ArrayList(Piece), cache: attention.Cache, ratio: usize, at: usize, span: ?parked_file.Span) !void {
    std.debug.assert(cache.table == null);
    for (parked_file.Span.arrays(cache), 0..) |array, i| if (array) |buffer| {
        const rows = if (span) |held| held.rows[i] else null;
        try split(gpa, list, buffer, cache.capacity, if (rows) |r| r.held else null, if (rows) |r| r.from else at, at, 1);
    };
    const blocks: usize = @intCast(cache.pooled.view.size[0]);
    try split(gpa, list, cache.pooled, blocks, if (span) |held| held.pooled.held else null, if (span) |held| held.pooled.from else at / ratio, at / ratio, ratio);
}

/// A state's own parts, the first of a file: the DeltaNet states, the conv inputs, the PLE's tails and
/// histories.
pub fn stateOf(gpa: Allocator, list: *std.ArrayList(Piece), snapshot: forward.Snapshot) !void {
    for ([_][]const cuda.Buffer{ snapshot.recurrent, snapshot.conv, snapshot.tails }) |buffers| for (buffers) |buffer| {
        try list.append(gpa, .{ .source = .{ .device = .{ .buffer = buffer, .len = buffer.bytes() } } });
    };
    for (snapshot.history) |history| try list.append(gpa, .{ .source = .{ .host = std.mem.sliceAsBytes(history) } });
}

/// The parts of the state `snapshot` over `caches` and the head's, in a file's order; `aside`: the sequence
/// left whose state it is.
fn pieces(gpa: Allocator, snapshot: forward.Snapshot, caches: []const attention.Cache, head: ?attention.Cache, ratio: usize, aside: ?*const parked_file.Sequence) ![]Piece {
    var list: std.ArrayList(Piece) = .empty;
    errdefer list.deinit(gpa);
    try stateOf(gpa, &list, snapshot);
    for (caches, 0..) |cache, i| try rowsOf(gpa, &list, cache, ratio, snapshot.pos, if (aside) |one| one.spans[i] else null);
    if (head) |cache| try rowsOf(gpa, &list, cache, ratio, snapshot.pos, if (aside) |one| one.head else null);
    return list.toOwnedSlice(gpa);
}

fn headCache(self: *const Session) ?attention.Cache {
    return if (self.head) |h| h.state.cache else null;
}

fn ratioOf(self: *const Session) usize {
    return if (self.m.attention) |held| held.shape.ratio else 1;
}

/// After a reply: the state at `self.kept` goes to disk where the session holds it and no file does. A
/// failure is logged and the reply stands.
pub fn learn(self: *Session, store: *Store) void {
    const at = self.kept orelse return;
    write(self, store, at, "a shared prefix's state", null) catch |err| std.log.warn("a shared prefix's state is not written: {t}", .{err});
}

/// The state the session holds at `at` of the sequence it stands on goes to disk, where no file holds it.
fn write(self: *Session, store: *Store, at: usize, what: []const u8, until: ?std.Io.Clock.Timestamp) !void {
    if (!self.trusted or at >= self.state.pos) return;
    const snapshot = for (self.snapshots.items) |one| {
        if (one.pos == at) break one;
    } else return;
    const start: std.Io.Timestamp = .now(self.io, .awake);
    const gpa = self.gpa;
    const parts = try pieces(gpa, snapshot, self.state.caches, headCache(self), ratioOf(self), null);
    defer gpa.free(parts);
    const done = try put(gpa, store, self.floor, self.known.items[0 .. at + 1], parts, until);
    if (done.bytes > 0) std.log.info("{s} at {d} is {s}: {d} MiB in {d:.3} s", .{ what, at, told(done), done.bytes >> 20, units.secondsSince(self.io, start) });
}

/// A sequence that leaves the lot for room (`parked.Lot.leaving`), its rows still held: the highest state of
/// its prompt goes to disk (`spill.pick`). The caches hold its rows below where it parted, the sequence the
/// rest.
fn leaving(context: *anyopaque, one: *const parked_file.Sequence) void {
    const self: *Session = @ptrCast(@alignCast(context));
    const store = self.spilled orelse return;
    spillLeft(self, store, one, null) catch |err| std.log.warn("a state that leaves is not written: {t}", .{err});
}

fn spillLeft(self: *Session, store: *Store, one: *const parked_file.Sequence, until: ?std.Io.Clock.Timestamp) !void {
    var held: [parked_file.states]spill.Candidate = undefined;
    const states = one.states.items;
    for (held[0..states.len], states) |*candidate, state| candidate.* = .{ .pos = state.pos, .prompt = state.pos < one.asked and state.pos < one.to };
    const snapshot = states[spill.pick(held[0..states.len]) orelse return];
    const start: std.Io.Timestamp = .now(self.io, .awake);
    const gpa = self.gpa;
    const parts = try pieces(gpa, snapshot, self.state.caches, headCache(self), ratioOf(self), one);
    defer gpa.free(parts);
    const done = try put(gpa, store, self.floor, one.tokens[0 .. snapshot.pos + 1], parts, until);
    if (done.bytes > 0) std.log.info("a state that leaves at {d} is {s}: {d} MiB in {d:.3} s", .{ snapshot.pos, told(done), done.bytes >> 20, units.secondsSince(self.io, start) });
}

/// What the session still holds goes to disk as what left did: the sequences left, the newest first, and
/// the one it stands on. When the server stops, and before the session is freed for the pool, which takes
/// the sequence it stands on (`live` false). Within `spill.flush_seconds`: a stop also waits that long for
/// the store's writer, and before the pool the states are the writer's to write as the pool serves.
pub fn flush(self: *Session, live: bool) void {
    const store = self.spilled orelse return;
    const began: std.Io.Timestamp = .now(self.io, .awake);
    defer if (live) settle(store, began);
    const until = stopBy(began);
    if (live and self.trusted) {
        var best: ?usize = null;
        for (self.snapshots.items) |one| {
            if (one.pos < self.asked and one.pos < self.state.pos and one.pos >= spill.floor) best = one.pos;
        }
        if (best) |at| write(self, store, at, "a state held", until) catch |err| std.log.warn("a state held is not written: {t}", .{err});
    }
    if (!self.trusted) return;
    var i = self.lot.list.items.len;
    while (i > 0 and spill.flushing(self.io, began)) {
        i -= 1;
        spillLeft(self, store, &self.lot.list.items[i], until) catch |err| std.log.warn("a state that leaves is not written: {t}", .{err});
    }
}

/// Gives the session the store of the states that leave it (`serve --spill-gib`), before any request, and
/// the store its writer.
pub fn spilling(self: *Session, store: *Store) void {
    spill_writer.Writer.start(store);
    self.spilled = store;
    self.lot.leaving = .{ .context = self, .call = leaving };
}

/// Inside `Session.resume`, once the session's own states and the sequences it left are settled for
/// `prompt`, which shares `shared` tokens with them: a shared prefix's state (`--learn`, at `kept`), then
/// the longest state that left (`--spill-gib`), each where it is worth its read. Returns what the prompt
/// then shares with the state read, null where nothing was, and 0 where a read failed with rows written.
pub fn recalled(self: *Session, prompt: []const i32, kept: ?usize, shared: usize) ?usize {
    var got: ?usize = null;
    if (self.learned) |store| if (kept) |at| {
        got = recall(self, store, prompt, at, shared, worth, "a shared prefix's state");
    };
    if (got == 0) return got;
    if (self.spilled) |store| {
        var under = prompt.len;
        while (store.longest(prompt, under)) |len| : (under = len) {
            if (len < 2) break;
            if (recall(self, store, prompt, len - 1, got orelse shared, spill.worth, "a state that left")) |theirs| return theirs;
        }
    }
    return got;
}

/// Where the file for the prompt's first `at + 1` tokens stands `above` over where the prompt would start,
/// its rows are written into the caches and its state joins the snapshots, the highest of them, for `resume`
/// to restore. Returns what the prompt then shares with the state, null where nothing was read, and 0 where
/// a read failed with rows written: the session then holds nothing.
pub fn recall(self: *Session, store: *Store, prompt: []const i32, at: usize, shared: usize, above: usize, what: []const u8) ?usize {
    if (at + 1 >= prompt.len or at + 1 >= self.options.capacity) return null;
    var start: usize = if (self.trusted and self.state.pos < shared) self.state.pos else 0;
    for ([_][]const forward.Snapshot{ self.snapshots.items, self.strides.items }) |held| {
        if (held.len > 0) start = @max(start, held[held.len - 1].pos);
    }
    if (at < start + above) return null;
    const began: std.Io.Timestamp = .now(self.io, .awake);
    var image = (store.find(self.gpa, prompt[0 .. at + 1]) catch |err| {
        std.log.warn("{s} is not read: {t}", .{ what, err });
        return null;
    }) orelse return null;
    defer image.close(self.gpa);
    var written = false;
    read(self, &image, prompt.len, at, &written) catch |err| {
        std.log.warn("{s} is not read: {t}", .{ what, err });
        // a file that is not this model's state, or is cut, is not met again
        if (err == error.Mismatch) store.discard(prompt[0 .. at + 1]);
        if (!written) return null;
        self.trusted = false;
        self.lot.clear();
        self.retireFrom(0);
        return 0;
    };
    std.log.info("{s} at {d} is read in {d:.3} s", .{ what, at, units.secondsSince(self.io, began) });
    return at + 1;
}

fn read(self: *Session, image: *Store.Image, len: usize, at: usize, written: *bool) !void {
    const gpa = self.gpa;
    // a state of this model's shapes, each buffer then written over
    var snapshot: forward.Snapshot = try .take(gpa, self.state);
    snapshot.pos = at;
    var states: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, 1);
    defer {
        for (states.items) |*one| one.deinit();
        states.deinit(gpa);
    }
    states.appendAssumeCapacity(snapshot);
    const parts = try pieces(gpa, snapshot, self.state.caches, headCache(self), ratioOf(self), null);
    defer gpa.free(parts);
    try matches(gpa, image, parts);
    // The rows written are the prompt's own: what a sequence left counts on in the caches lies below where
    // it parts from the prompt (`Lot.lower`, before this), and is the same rows.
    written.* = true;
    try fill(gpa, image, parts);
    // as a sequence taken up from the lot: the head stands where the state does
    self.trusted = false;
    if (self.head) |*h| {
        try h.state.trim();
        try h.state.position.set(at);
        h.full = false;
    }
    self.kept = at;
    var one: parked_file.Sequence = .{ .tokens = &.{}, .from = at, .to = at, .states = states, .spans = &.{}, .head = null, .left = 0 };
    self.adopt(&one, at + 1, len);
    states = one.states;
}

/// What a state's bytes are on this load, hashed with `base`: the state a fixed prompt (`learned_parts.zig`'s
/// `probePrompt`) leaves, the head's entries with it. Two loads that give it the same bits give every prefix the same, as
/// far as one prompt can say; the build and the checkpoint are in `base`. Run once, before any request: the
/// session is left holding nothing.
pub fn identity(self: *Session, base: u64) !u64 {
    const gpa = self.gpa;
    const prompt = try parts_file.probePrompt(gpa, self.options.capacity, self.m.shape.vocab);
    defer gpa.free(prompt);
    const Quiet = struct {
        fn write(_: *anyopaque, _: @import("../../../server/request.zig").Kind, _: []const u8) @import("../../../server/request.zig").Sink.Signal {
            return .more;
        }
    };
    var nothing: u8 = 0;
    _ = try self.reply(prompt, .{ .seed = 0, .temperature = 0 }, .{ .max_tokens = 1 }, .{ .context = &nothing, .writeFn = Quiet.write });
    defer {
        self.trusted = false;
        self.lot.clear();
        self.retireFrom(0);
    }
    var snapshot: forward.Snapshot = try .take(gpa, self.state);
    defer snapshot.deinit();
    const parts = try pieces(gpa, snapshot, self.state.caches, headCache(self), ratioOf(self), null);
    defer gpa.free(parts);
    var hasher: std.hash.Wyhash = .init(base);
    hasher.update(std.mem.asBytes(&@as(u64, prompt.len)));
    try parts_file.hash(gpa, &hasher, parts);
    return hasher.final();
}

/// Opens the states on disk for the session, after the probe (`identity`, once for both kinds): it is given
/// `--learn`'s, and the caller gives `spill`'s store to the session or the hall (`spilling`). Null, with a
/// warning, where none is kept: the session then serves as without the flags. Before any request.
pub fn open(self: *Session, asked: Asked) !?*Learning {
    const gpa = self.gpa;
    const io = self.io;
    if (asked.dirty) {
        std.log.warn("--learn, --spill-gib: this build's tree differs from its commit, so it reads and writes no state", .{});
        return null;
    }
    const start: std.Io.Timestamp = .now(io, .awake);
    var storage: [2]u8 = .{ 0, 0 };
    const stored: usize = if (self.m.attention) |held| two: {
        storage = .{ @backingInt(held.shape.kv), @backingInt(held.head) };
        break :two 2;
    } else 0;
    const id = try identity(self, try parts_file.base(gpa, io, asked, storage[0..stored]));
    const held = try parts_file.open(gpa, io, asked, id);
    errdefer held.close();
    if (held.learn) |*kept| self.learned = &kept.store;
    self.floor = &held.floor;
    std.log.info("the states' identity: {d:.3} s the probe", .{units.secondsSince(io, start)});
    return held;
}
