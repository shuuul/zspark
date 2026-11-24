//! The pool's kept states to and from disk (`serve --spill-gib`, spec 074; the rule is `decode/spill.zig`'s,
//! the files `decode/learned.zig`'s). A state that goes for room (`prefix.Store.leaving`: the store's bound or
//! the states' budget lets it go, or the extent under it is taken back) is written while its rows are still
//! held, where it is the highest its chain keeps of its prompt; what is held when the server stops is written
//! then (`flush`). A prompt that the pool holds no state for, or a lower one, starts from the longest file it
//! begins with (`recall`): the file becomes an idle chain with one state, and the prompt forks from it as
//! from any other.
//!
//! A file is the one a session writes (`session/learnt.zig`'s header): the state's own buffers, then each
//! cache's rows below it and its pooled keys, then the head's. Here the rows lie in the chain's extents, so a
//! part is read and written in runs of extents; the bytes are the same, as `hall_carry.zig` moves a
//! session's rows into the pool unchanged. So a load's lone session reads what a pool wrote before a restart,
//! and the pool what a session wrote, and the two share their rows' blocks (`decode/learned_blocks.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const prefix = @import("../../../decode/prefix.zig");
const spill = @import("../../../decode/spill.zig");
const spill_writer = @import("../../../decode/spill_writer.zig");
const attention = @import("../blocks/attention.zig");
const hall_engine = @import("hall_engine.zig");
const keep_file = @import("hall_keep.zig");
pub const learnt = @import("../session/learnt.zig");
const Engine = hall_engine.Engine;
const Keep = keep_file.Keep;
const Piece = learnt.Piece;
pub const Store = learnt.Store;

/// Gives the engine the store, once it stands where it stays, and the store its writer.
pub fn attach(engine: *Engine, store: *Store) void {
    spill_writer.Writer.start(store);
    engine.keeps.spill = store;
    engine.store.leaving = .{ .context = engine, .call = going };
}

/// One pool array's rows of `chain` below `pos`, a piece each run of extents that follow one another.
fn runs(gpa: Allocator, list: *std.ArrayList(Piece), chain: *const extents.Chain, array: cuda.Buffer, pos: usize) !void {
    const row = array.bytes() / @as(usize, @intCast(array.view.size[0]));
    const table = chain.table.items;
    var first: usize = 0;
    var joined = false;
    while (first * extents.positions < pos) {
        var next = first + 1;
        while (next * extents.positions < pos and table[next] == table[next - 1] + 1) next += 1;
        const from = first * extents.positions;
        const count = @min(pos, next * extents.positions) - from;
        const at = @as(usize, @intCast(table[first])) * extents.positions;
        try list.append(gpa, .{ .source = .{ .device = .{ .buffer = array, .from = at * row, .len = count * row } }, .joined = joined, .per = 1 });
        joined = true;
        first = next;
    }
}

/// The parts of `keep` in a file's order: its state, then each layer's rows in the extents and its pooled
/// keys, then the head's.
fn pieces(engine: *const Engine, keep: Keep) ![]Piece {
    const gpa = engine.gpa;
    var list: std.ArrayList(Piece) = .empty;
    errdefer list.deinit(gpa);
    try learnt.stateOf(gpa, &list, keep.state);
    const ratio: u32 = if (engine.m.attention) |shared| @intCast(shared.shape.ratio) else 1;
    for (0..engine.trunk.len + @intFromBool(engine.head_pool != null)) |i| {
        const pool: attention.Pool = if (i < engine.trunk.len) engine.trunk[i] else engine.head_pool.?;
        for ([_]?cuda.Buffer{ pool.k, pool.v, pool.k_scale, pool.v_scale, pool.index_keys }) |array| {
            if (array) |buffer| try runs(gpa, &list, keep.chain, buffer, keep.pos);
        }
        const pooled = keep.pooled.list[i];
        try list.append(gpa, .{ .source = .{ .device = .{ .buffer = pooled, .len = if (pooled.view.ptr != null) pooled.bytes() else 0 } }, .per = ratio });
    }
    return list.toOwnedSlice(gpa);
}

/// The state `chain` keeps at `pos` goes to disk, where it can be a file: the head stands where it does.
fn write(engine: *Engine, store: *Store, chain: *const extents.Chain, pos: usize, until: ?std.Io.Clock.Timestamp) !void {
    const keep = (engine.keeps.at(chain, pos) orelse return).*;
    if (chain.len() < pos or keep.pooled.list.len != engine.trunk.len + @intFromBool(engine.head_pool != null)) return;
    if (engine.head_pool != null and keep.head_len != pos) return;
    const gpa = engine.gpa;
    const tokens = try gpa.alloc(i32, pos + 1);
    defer gpa.free(tokens);
    @memcpy(tokens[0..pos], chain.tokens.items[0..pos]);
    tokens[pos] = keep.next;
    const start: ?std.Io.Timestamp = if (engine.io) |io| .now(io, .awake) else null;
    const parts = try pieces(engine, keep);
    defer gpa.free(parts);
    const done = try learnt.put(gpa, store, if (store.writer) |writer| writer.floor else null, tokens, parts, until);
    // a chain shortened extent by extent would else write each state below this one as it goes
    for (engine.store.marks.items) |*mark| {
        if (mark.chain == chain and mark.pos <= pos) mark.spilt = true;
    }
    if (done.bytes > 0) std.log.info("a state that leaves the pool at {d} is {s}: {d} MiB in {d:.3} s", .{ pos, learnt.told(done), done.bytes >> 20, if (start) |began| @as(f64, @floatFromInt(began.untilNow(engine.io.?, .awake).toNanoseconds())) / 1e9 else 0 });
}

/// The highest state of its prompt that `chain` keeps above `above` with none of the chain written at or
/// above it, where `spill.pick` writes one.
fn chosen(engine: *const Engine, chain: *const extents.Chain, above: usize) ?usize {
    var held: [prefix.states_max + 1]spill.Candidate = undefined;
    var n: usize = 0;
    for (engine.store.marks.items) |mark| {
        if (mark.chain != chain or mark.pos <= above or n == held.len) continue;
        held[n] = .{ .pos = mark.pos, .prompt = mark.kind.thins() and !mark.spilt };
        n += 1;
    }
    return held[spill.pick(held[0..n]) orelse return null].pos;
}

fn going(context: *anyopaque, what: prefix.Store.Going) void {
    const engine: *Engine = @ptrCast(@alignCast(context));
    const store = engine.keeps.spill orelse return;
    const chain, const pos = switch (what) {
        // the last of its prompt's that its chain keeps: a lower one goes with a higher still held
        .one => |mark| .{ mark.chain, if (chosen(engine, mark.chain, 0) == mark.pos) mark.pos else return },
        .past => |cut| .{ cut.chain, chosen(engine, cut.chain, cut.len) orelse return },
    };
    // a stream's own chain is not a conversation left
    if (chain.live) return;
    write(engine, store, chain, pos, null) catch |err| std.log.warn("a state that leaves the pool is not written: {t}", .{err});
}

/// What the pool still holds goes to disk as what left did, the chain used last first, and what the store's
/// writer has still to write is waited for, all within `spill.flush_seconds`: when the server stops.
pub fn flush(engine: *Engine) void {
    const store = engine.keeps.spill orelse return;
    const io = engine.io orelse return;
    const began: std.Io.Timestamp = .now(io, .awake);
    defer learnt.settle(store, began);
    var last: u64 = std.math.maxInt(u64);
    while (spill.flushing(io, began)) {
        var next: ?*extents.Chain = null;
        for (engine.pool.chains.items) |chain| {
            if (chain.used < last and (next == null or chain.used > next.?.used)) next = chain;
        }
        const chain = next orelse return;
        last = chain.used;
        const pos = chosen(engine, chain, 0) orelse continue;
        write(engine, store, chain, pos, learnt.stopBy(began)) catch |err| std.log.warn("a state the pool holds is not written: {t}", .{err});
    }
}

/// Before `prompt` is begun, which starts at `start` from what the pool holds: the longest file it begins
/// with that is worth its read becomes an idle chain and its one state. Returns whether one did.
pub fn recall(engine: *Engine, prompt: []const i32, start: usize) bool {
    const store = engine.keeps.spill orelse return false;
    var under = prompt.len;
    while (spill.serving(store, prompt, start, under)) |pos| : (under = pos + 1) {
        if (pos > engine.pool.capacity) continue;
        const began: ?std.Io.Timestamp = if (engine.io) |io| .now(io, .awake) else null;
        const read = bring(engine, store, prompt[0 .. pos + 1]) catch |err| failed: {
            std.log.warn("a state that left the pool is not read: {t}", .{err});
            if (err == error.Mismatch) store.discard(prompt[0 .. pos + 1]);
            break :failed false;
        };
        if (!read) continue;
        std.log.info("a state that left the pool at {d} is read in {d:.3} s", .{ pos, if (began) |since| @as(f64, @floatFromInt(since.untilNow(engine.io.?, .awake).toNanoseconds())) / 1e9 else 0 });
        return true;
    }
    return false;
}

/// `serve --learn`: before `prompt` is begun, which starts at `start` from what the pool holds, the file of
/// the shared prefix the request says ends at `kept` becomes an idle chain and its one state, where it stands
/// `learnt.worth` over `start`: the session's rule (`learnt.recall`). TensorFold 1.0.12's engine reads its
/// learned store for every lane (`native/cuda_disk.zig`); the pool here reads and does not write, which the
/// lone session alone does. Returns whether a state was brought in.
pub fn learned(engine: *Engine, prompt: []const i32, kept: ?usize, start: usize) bool {
    const store = engine.keeps.learned orelse return false;
    const at = kept orelse return false;
    if (at + 1 >= prompt.len or at > engine.pool.capacity or at < start + learnt.worth) return false;
    const began: ?std.Io.Timestamp = if (engine.io) |io| .now(io, .awake) else null;
    const read = bring(engine, store, prompt[0 .. at + 1]) catch |err| {
        std.log.warn("a shared prefix's state is not read: {t}", .{err});
        if (err == error.Mismatch) store.discard(prompt[0 .. at + 1]);
        return false;
    };
    if (read) std.log.info("a shared prefix's state at {d} is read in {d:.3} s", .{ at, if (began) |since| @as(f64, @floatFromInt(since.untilNow(engine.io.?, .awake).toNanoseconds())) / 1e9 else 0 });
    return read;
}

/// The file for `tokens` into a new idle chain of the tokens below its state, and the state kept on it.
/// False: no file, or no room for it.
fn bring(engine: *Engine, store: *Store, tokens: []const i32) !bool {
    const gpa = engine.gpa;
    const pos = tokens.len - 1;
    var image = try store.find(gpa, tokens) orelse return false;
    defer image.close(gpa);
    const chain = try engine.pool.open();
    var held = false;
    // its extents are free again, and the state's mark gone with it
    errdefer {
        engine.store.release(chain) catch engine.pool.release(chain);
        keep_file.sweep(engine);
    }
    while (true) {
        engine.pool.grow(chain, pos) catch |err| switch (err) {
            error.Short => {
                if (try engine.store.evict(null) == null) return error.Short;
                keep_file.sweep(engine);
                continue;
            },
            else => return err,
        };
        break;
    }
    engine.pool.append(chain, tokens[0..pos]);
    engine.pool.retire(chain);
    const slot = engine.slots[0];
    var state = try keep_file.blank(engine, pos);
    errdefer if (!held) state.deinit();
    try keep_file.histories(&state, slot.ple);
    const blocks = if (engine.m.attention) |shared| pos / shared.shape.ratio else 0;
    // buffers of the pooled keys' shape, written over
    const pooled = try keep_file.pooledOf(gpa, slot.caches, slot.head, blocks);
    var keep: Keep = .{ .chain = chain, .pos = pos, .head_len = pos, .next = tokens[pos], .kind = .window, .target = pos, .state = state, .pooled = pooled };
    errdefer if (!held) {
        // the state is freed by the scope above
        var none = keep.strip();
        _ = &none;
    };
    const parts = try pieces(engine, keep);
    defer gpa.free(parts);
    try learnt.matches(gpa, &image, parts);
    try learnt.fill(gpa, &image, parts);
    try cuda.synchronize();
    try engine.keeps.list.append(gpa, keep);
    held = true;
    try engine.store.note(chain, pos, .window);
    keep_file.sweep(engine);
    keep_file.room(engine);
    return engine.keeps.at(chain, pos) != null;
}
