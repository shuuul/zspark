//! A session's states to and from disk (`serve --spill-gib`, spec 108 WS-04; the rule is `decode/spill.zig`'s,
//! the files `decode/learned.zig`'s, a state's parts as a file `decode/learned_parts.zig`'s, all three Flash
//! Next's of spec 074): the state a conversation leaves goes to disk, and a later prompt that begins with a
//! file's tokens starts from it, in the same load or the next.
//!
//! GLM's session holds one sequence and no conversations left (Flash Next's `parked.zig`): a conversation
//! leaves when a prompt parts from it below its last prompt's states, and the stop leaves the one it holds.
//! Of its marks one is written, `spill.pick`'s, the highest below its last prompt's end, as Flash Next writes
//! one state a sequence. A prompt that the session holds nothing for, or less, starts from the longest file it
//! begins with that stands `spill.worth` above where it would start: the file becomes a mark, and the prompt
//! goes on from it as from any other.
//!
//! A file holds what a mark at position p is on the ranks this process holds, rank by rank: the KDA layers'
//! conv windows and the side of their states a window reads (`kda.Mark`); each DSA layer's latents, index keys
//! and gates below p and its pooled keys' complete pools (`dsa.Cache`); the MTP layer's cache below p, where
//! the MTP layer drafts (it is not written under `--serial` or DFlash2, which leave it unabsorbed). DFlash2's
//! context is not: it is made of the drafter's weights, which nothing writes (spec 067), and a mark without
//! it starts the drafter's context empty at the mark (`restoreLocal`, `PairOf.startAt`). The MTP cache's
//! entry p - 1 was absorbed with token p, so a file's tokens are the sequence through token p.
//!
//! Two ranks (vLLM's and SGLang's disk tiers, and TensorFold's GLM `--learn`, write a rank's shard on that
//! rank): each rank keeps its own half under its own identity, in its own directory. Rank 0 decides and tells
//! rank 1 (`Op.spill_*`): the tokens, then the mark to write or the file to read. Rank 1's files lie under the
//! root's `rank1/`. A write is each rank's own;
//! a read is taken only where both ranks hold their half and it is this model's (an all-gather of one word,
//! where TensorFold's rank 1 replies through its control channel, which zspark's rank 1 never writes to), and
//! a fill that fails on either rank starts the sequence again on both.

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const learned = @import("../../../decode/learned.zig");
const parts_file = @import("../../../decode/learned_parts.zig");
const units = @import("../../../cuda/units.zig");
const spill = @import("../../../decode/spill.zig");
const dsa = @import("../layers/dsa.zig");
const held_file = @import("../layers/ranks_forward.zig");
const marking = @import("marks.zig");
const snapshot = @import("snapshot.zig");
const Piece = parts_file.Piece;
const Op = held_file.Op;
pub const Store = learned.Store;
pub const Floor = parts_file.Floor;
pub const Learning = parts_file.Learning;
pub const Asked = parts_file.Asked;

/// A sequence's tokens a frame of `Op.spill_tokens` carries at most.
const frame_tokens = 4096;

/// `count` rows of each of `layers` layers of `buffer`, a layer `span` rows, a row every `per` positions: a part
/// a layer, since a part's rows are its positions in order (`learned_blocks.zig`).
fn layerRows(gpa: Allocator, list: *std.ArrayList(Piece), buffer: cuda.Buffer, layers: usize, span: usize, count: usize, per: usize) !void {
    const row = buffer.bytes() / (layers * span);
    for (0..layers) |layer| try list.append(gpa, .{ .source = .{ .device = .{ .buffer = buffer, .from = layer * span * row, .len = count * row } }, .per = @intCast(per) });
}

/// A cache's rows below `pos`: the latents, the index keys and gates a row a position, and the pooled keys' codes
/// and scales of the pools complete below it (`dsa.Cache.cut` makes the unfinished one zero).
fn rowsOf(gpa: Allocator, list: *std.ArrayList(Piece), cache: dsa.Cache, pos: usize) !void {
    try layerRows(gpa, list, cache.rows, cache.layers, cache.span, pos, 1);
    const indexer = cache.indexer orelse return;
    try layerRows(gpa, list, indexer.keys, cache.layers, cache.span, pos, 1);
    try layerRows(gpa, list, indexer.gates, cache.layers, cache.span, pos, 1);
    const ratio = cache.span / cache.pooled_rows;
    try layerRows(gpa, list, indexer.codes, cache.layers, cache.pooled_rows, pos / ratio, ratio);
    try layerRows(gpa, list, indexer.scales, cache.layers, cache.pooled_rows, pos / ratio, ratio);
}

/// Whether a file holds the MTP layer's cache: where the MTP layer drafts, which absorbs every position.
fn keepsMtp(self: anytype) bool {
    return !self.options.serial and self.options.drafter == null;
}

/// The parts of the state `shots` at `pos` over the caches this process holds, in a file's order.
fn pieces(self: anytype, shots: []const snapshot.Shot, pos: usize) ![]Piece {
    const gpa = self.gpa;
    var list: std.ArrayList(Piece) = .empty;
    errdefer list.deinit(gpa);
    for (shots, self.state.ranks, 0..) |shot, rank, r| {
        if (shot.kda) |kept| for ([_]cuda.Buffer{ kept.conv, kept.rec }) |buffer| try list.append(gpa, .{ .source = .{ .device = .{ .buffer = buffer, .len = buffer.bytes() } } });
        if (rank.dsa) |cache| try rowsOf(gpa, &list, cache, pos);
        if (keepsMtp(self)) try rowsOf(gpa, &list, self.pair.caches[r], pos);
    }
    return list.toOwnedSlice(gpa);
}

/// Rank 0: `tokens` told to rank 1 in frames, before the operation that names them.
fn tellTokens(self: anytype, tokens: []const i32) !void {
    if (!self.leads()) return;
    var from: usize = 0;
    while (from < tokens.len) : (from += frame_tokens) {
        const chunk = tokens[from..@min(tokens.len, from + frame_tokens)];
        try self.lead.tell(@backingInt(Op.spill_tokens), &.{@intCast(from)}, chunk);
    }
}

/// The mark a conversation leaves where the next prompt starts at `reach`: the highest below its last prompt's
/// end at `spill.floor` and above (`spill.pick`), where it stands above `reach`. Null: none, or the drafter's
/// cache that a file holds is not in step with it.
fn leaving(self: anytype, reach: usize) !?usize {
    const end = self.prompt_end orelse return null;
    if (keepsMtp(self) and !self.in_step) return null;
    const held = try self.gpa.alloc(spill.Candidate, self.marks.held.items.len);
    defer self.gpa.free(held);
    for (held, self.marks.held.items) |*candidate, mark| candidate.* = .{ .pos = mark.pos, .prompt = mark.pos < end };
    const at = held[spill.pick(held) orelse return null].pos;
    if (at <= reach or at + 1 > self.known.items.len) return null;
    return at;
}

/// Before a prompt that starts at `reach` from what the session holds: the conversation's state that would
/// be lost goes to disk on both ranks. A failure is said and the reply goes on.
pub fn leave(self: anytype, reach: usize) void {
    const at = (leaving(self, reach) catch return) orelse return;
    write(self, at, null) catch |err| std.log.warn("a state that leaves is not written: {t}", .{err});
}

/// Rank 0: the mark at `at` goes to disk here and on rank 1, a stop's within `until`.
fn write(self: anytype, at: usize, until: ?std.Io.Clock.Timestamp) !void {
    const tokens = self.known.items[0 .. at + 1];
    try tellTokens(self, tokens);
    if (self.leads()) try self.lead.tell(@backingInt(Op.spill_write), &.{ @intCast(at), @intFromBool(until != null) }, &.{});
    writeLocal(self, tokens, at, until) catch |err| std.log.warn("rank {d}: a state that leaves at {d} is not written: {t}", .{ self.first, at, err });
}

/// This rank's half of the mark at `at`, the sequence through token `at` being `tokens`, to its store.
fn writeLocal(self: anytype, tokens: []const i32, at: usize, until: ?std.Io.Clock.Timestamp) !void {
    const store = self.spilled orelse return;
    const mark = for (self.marks.held.items) |one| {
        if (one.pos == at) break one;
    } else return error.NoSnapshot;
    const start: std.Io.Timestamp = .now(self.io, .awake);
    const parts = try pieces(self, &mark.ranks, at);
    defer self.gpa.free(parts);
    const done = try parts_file.put(self.gpa, store, self.floor, tokens, parts, until);
    if (done.bytes > 0) std.log.info("rank {d}: a state that leaves at {d} is {s}: {d} MiB in {d:.3} s", .{ self.first, at, parts_file.told(done), done.bytes >> 20, units.secondsSince(self.io, start) });
}

/// When the server stops, rank 0: the state the session holds goes to disk as one that left, within
/// `spill.flush_seconds`, and this rank's writer is waited for.
pub fn flush(self: anytype) void {
    const store = self.spilled orelse return;
    const began: std.Io.Timestamp = .now(self.io, .awake);
    defer parts_file.settle(store, began);
    const at = (leaving(self, 0) catch return) orelse return;
    write(self, at, parts_file.stopBy(began)) catch |err| std.log.warn("a state held is not written: {t}", .{err});
}

/// Rank 1 after the end it was told: what its writer has still to write, within `spill.flush_seconds`.
pub fn settle(self: anytype) void {
    const store = self.spilled orelse return;
    parts_file.settle(store, .now(self.io, .awake));
}

/// Rank 0, once `prompt` starts at `cached` from what the session holds: the longest file the prompt begins
/// with that stands `spill.worth` above it becomes a mark, where both ranks hold their half. Returns where the
/// prompt then starts: the file's position, `cached` where none was read, 0 where a fill failed with rows
/// written (the caller starts the sequence again).
pub fn recall(self: anytype, prompt: []const i32, cached: usize) !usize {
    const store = self.spilled orelse return cached;
    var under = prompt.len;
    while (spill.serving(store, prompt, cached, under)) |pos| : (under = pos + 1) {
        if (pos >= self.options.capacity) continue;
        const tokens = prompt[0 .. pos + 1];
        try tellTokens(self, tokens);
        if (self.leads()) try self.lead.tell(@backingInt(Op.spill_read), &.{ @intCast(pos), @intCast(cached) }, &.{});
        switch (try readLocal(self, tokens, pos, cached)) {
            .read => return pos,
            .spoilt => return 0,
            .missed => {},
        }
    }
    return cached;
}

const Read = enum { read, missed, spoilt };

/// Both ranks agree whether each holds its half of the file of `tokens`, a state at `pos`, and it is this
/// model's; then each fills its caches and a new mark from it and they agree again. The marks above `cached`,
/// where the prompt stood without it, go before a row is written. Calls `agree` the same times on both ranks
/// whatever either finds.
fn readLocal(self: anytype, tokens: []const i32, pos: usize, cached: usize) !Read {
    const gpa = self.gpa;
    const began: std.Io.Timestamp = .now(self.io, .awake);
    var image: ?Store.Image = null;
    defer if (image) |*one| one.close(gpa);
    if (self.spilled) |store| image = store.find(gpa, tokens) catch |err| none: {
        std.log.warn("rank {d}: a state that left at {d} is not read: {t}", .{ self.first, pos, err });
        break :none null;
    };
    var mark = marking.blankMark(self, pos) catch |err| {
        std.log.warn("rank {d}: no mark for a state that left at {d}: {t}", .{ self.first, pos, err });
        _ = try self.agree(false);
        return .missed;
    };
    var kept = false;
    defer if (!kept) mark.release(gpa, &self.marks.spare);
    const parts = pieces(self, &mark.ranks, pos) catch |err| {
        std.log.warn("rank {d}: a state that left at {d} is not read: {t}", .{ self.first, pos, err });
        _ = try self.agree(false);
        return .missed;
    };
    defer gpa.free(parts);
    var ok = image != null;
    if (image) |*one| parts_file.matches(gpa, one, parts) catch |err| {
        std.log.warn("rank {d}: a state that left at {d} is not read: {t}", .{ self.first, pos, err });
        // a file that is not this model's state is not met again
        if (err == error.Mismatch) self.spilled.?.discard(tokens);
        ok = false;
    };
    if (!try self.agree(ok)) return .missed;
    // the rows below `cached` are the prompt's on both sides; the marks above stand on rows written over
    if (cached == 0) marking.clearMarks(self) else marking.dropAbove(self, cached);
    var filled = true;
    parts_file.fill(gpa, &image.?, parts) catch |err| {
        std.log.warn("rank {d}: a state that left at {d} is not read: {t}", .{ self.first, pos, err });
        filled = false;
    };
    if (!try self.agree(filled)) return .spoilt;
    try self.marks.held.append(gpa, mark);
    kept = true;
    try marking.restoreLocal(self, pos);
    // the MTP layer's cache is the file's through `pos`, or DFlash2's context starts empty there (`restoreLocal`)
    self.in_step = true;
    std.log.info("rank {d}: a state that left at {d} is read in {d:.3} s", .{ self.first, pos, units.secondsSince(self.io, began) });
    return .read;
}

/// Rank 1: what rank 0 told of a state on disk (`Op.spill_*`). A file that cannot be written or read is said
/// and costs that state; only the ranks' agreement and what breaks the device end the pair.
pub fn told(self: anytype, op: Op, arguments: []align(1) const u32, tokens: []const i32) !void {
    switch (op) {
        .spill_tokens => {
            if (arguments[0] == 0) self.told.clearRetainingCapacity();
            if (arguments[0] != self.told.items.len) return error.OutOfStep;
            try self.told.appendSlice(self.gpa, tokens);
        },
        .spill_open => {
            const asked = self.asked orelse return error.OutOfStep;
            self.learning = openLocal(self, asked) catch |err| failed: {
                std.log.warn("rank {d}: --spill-gib: {t}, so this rank keeps no state", .{ self.first, err });
                break :failed null;
            };
        },
        .spill_write => {
            if (arguments[0] + 1 != self.told.items.len) return error.OutOfStep;
            const until: ?std.Io.Clock.Timestamp = if (arguments[1] != 0) parts_file.stopBy(.now(self.io, .awake)) else null;
            writeLocal(self, self.told.items, arguments[0], until) catch |err| std.log.warn("rank {d}: a state that leaves at {d} is not written: {t}", .{ self.first, arguments[0], err });
        },
        .spill_read => {
            if (arguments[0] + 1 != self.told.items.len or arguments[1] > self.state.pos) return error.OutOfStep;
            _ = try readLocal(self, self.told.items, arguments[0], arguments[1]);
        },
        else => return error.OutOfStep,
    }
}

/// Rank 0, before any request: the identity's probe, a fixed prompt's state, which each rank hashes into its
/// own (rank 1 told, `Op.spill_open`), then the stores of `asked` opened under it on each. Null, with a
/// warning, where none is kept here: the session then serves as without the flag. The session is left
/// holding nothing.
pub fn open(self: anytype, asked: Asked) !?*Learning {
    if (asked.dirty) {
        std.log.warn("--spill-gib: this build's tree differs from its commit, so it reads and writes no state", .{});
        return null;
    }
    const gpa = self.gpa;
    const started: std.Io.Timestamp = .now(self.io, .awake);
    const prompt = try parts_file.probePrompt(gpa, self.options.capacity, self.logits.len);
    defer gpa.free(prompt);
    var out: [1]i32 = undefined;
    _ = try self.reply(prompt, .{ .seed = 0, .temperature = 0 }, .{ .eos = &.{}, .fresh = true }, &out);
    if (self.leads()) try self.lead.tell(@backingInt(Op.spill_open), &.{}, &.{});
    const held = openLocal(self, asked) catch |err| failed: {
        std.log.warn("rank {d}: --spill-gib: {t}, so no state is kept", .{ self.first, err });
        break :failed null;
    };
    // the probe's sequence is no client's
    try self.state.reset();
    marking.clearMarks(self);
    self.known.clearRetainingCapacity();
    std.log.info("the states' identity: {d:.3} s the probe", .{units.secondsSince(self.io, started)});
    return held;
}

/// This rank's identity, the state it holds hashed with what `asked` names, and its stores opened under it.
/// The one word the ranks agree by is made first, so that a rank that keeps nothing still answers.
fn openLocal(self: anytype, asked: Asked) !*Learning {
    const gpa = self.gpa;
    if (@TypeOf(self.*).paired and self.agreed == null) self.agreed = try .init(.i32, &.{3});
    var shots: [@TypeOf(self.*).ranks_held]snapshot.Shot = undefined;
    var taken: usize = 0;
    defer for (shots[0..taken]) |*shot| shot.deinit();
    for (&shots, self.state.ranks) |*shot, rank| {
        shot.* = try snapshot.takeRank(rank, null);
        taken += 1;
    }
    const parts = try pieces(self, &shots, self.state.pos);
    defer gpa.free(parts);
    var hasher: std.hash.Wyhash = .init(try parts_file.base(gpa, self.io, asked, @tagName(self.options.kv)));
    hasher.update(std.mem.asBytes(&[_]u64{ self.first, self.state.pos, @intFromBool(keepsMtp(self)) }));
    try parts_file.hash(gpa, &hasher, parts);
    // rank 1's half under the root's `rank1/`, as TensorFold names its halves `.r1`: two ranks over one disk
    // would each take the other's states for another identity's and sweep them
    var mine = asked;
    var own: ?[]u8 = null;
    defer if (own) |path| gpa.free(path);
    if (self.first != 0) if (asked.spill) |root| {
        var name: [16]u8 = undefined;
        const rank = std.fmt.bufPrint(&name, "rank{d}", .{self.first}) catch unreachable;
        own = if (root.dir) |dir| try std.fs.path.join(gpa, &.{ dir, rank }) else if (asked.home) |home| try std.fs.path.join(gpa, &.{ home, ".cache", "zspark", "spilled", rank }) else null;
        if (own) |path| mine.spill.?.dir = path;
    };
    const held = try parts_file.open(gpa, self.io, mine, hasher.final());
    if (held.spill) |*kept| self.spilled = &kept.store;
    self.floor = &held.floor;
    return held;
}
