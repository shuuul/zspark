//! A row of logits over two ranks, each holding half of the vocabulary (bf16 [rows, vocab / world] a rank), and
//! a window's greedy rows settled on the device (spec 095 for Flash Next, spec 114 WS-02 for GLM): what both
//! models' `ranks_forward.zig` do alike, moved here by spec 136 WS-05. What a model tells its follower first and
//! when it settles at all stay its own. The NCCL comm is `comm: anytype` (`ranks/nccl.zig`'s `Comm`): `decode/`
//! does not import `ranks/`.
//!
//! A round of several streams settles every seat's rows the same way at once (spec 144 WS-03, Flash Next's pool,
//! one rank or two): each row cut at the rank its seat's sampling needs (`launchRound`), one gather and one
//! download a round, and each row's token chosen from its start at its own position (`readRound`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../cuda/cuda.zig");
const picking = @import("pick.zig");
const sampling = @import("sampling.zig");

/// Row `row` of a rank's logits, bf16 [rows, half], as a view of that row: [1, half].
pub fn rowOf(logits: cuda.Buffer, row: usize, half: usize) cuda.View {
    const base: [*]u8 = @ptrCast(logits.view.ptr.?);
    return .init(@ptrCast(base + row * half * 2), .bf16, &.{ 1, half });
}

/// A row of logits over the vocabulary from this rank's half of it, `own` (bf16 [1, vocab / world]): the halves
/// gathered through `comm` into `whole` (bf16 [world, vocab / world]), rank 0's first, and read back. Every
/// rank asks for the same rows in the same order, as it decodes the same reply.
pub fn gatherRow(comm: anytype, own: cuda.View, whole: cuda.Buffer, out: []u16) !void {
    std.debug.assert(whole.view.count() == out.len);
    try comm.allGather(own, whole.view);
    try cuda.synchronize();
    try whole.download(std.mem.sliceAsBytes(out));
}

/// Row `row`'s logits as bf16 bits [vocab], rank 0's half and then rank 1's. `logits` are the halves of the
/// ranks held, rising from rank `first`; where fewer than `world` are held the other's half comes through
/// `gather` (a `ranks.Gather` over NCCL) into `whole` (`gatherRow`).
pub fn logitsRow(comptime world: usize, comptime held: usize, logits: [held]cuda.Buffer, first: usize, gather: anytype, whole: ?cuda.Buffer, row: usize, out: []u16) !void {
    const half = out.len / world;
    for (logits) |one| std.debug.assert(@as(usize, @intCast(one.view.size[1])) == half);
    if (held < world) return gatherRow(gather.nccl, rowOf(logits[0], row, half), whole.?, out);
    try cuda.synchronize();
    for (logits, first..) |one, rank| {
        try one.downloadAt(row * half * 2, std.mem.sliceAsBytes(out[rank * half ..][0..half]));
    }
}

/// `settle`'s launches on every rank: this rank's half of rows `row` .. `row + rows - 1` of `logits` picked in
/// one set of launches (`picking.pickRows`), each cut at its byte of `asks` where given, else at the rank its
/// words ask; and the picks gathered through `comm` where `world` is over 1, rank 0's rows first, as u32 [world,
/// rows, a pick's words] from `picks`' start. One rank alone (`world` 1) passes `{}` for `comm`.
pub fn gatherPicks(comptime world: usize, comm: anytype, logits: cuda.Buffer, picks: cuda.Buffer, asks: ?cuda.View, row: usize, rows: usize) !void {
    std.debug.assert(rows <= picks.view.size[1] and world <= picks.view.size[0]);
    const slot: usize = @intCast(picks.view.size[2]);
    const rank: usize = if (world == 1) 0 else comm.rank;
    const words_at: [*]u8 = @ptrCast(picks.view.ptr.?);
    const own: cuda.View = .init(@ptrCast(words_at + rank * rows * slot * 4), .u32, &.{ rows, slot });
    try picking.pickRows(logits.span(row, rows), own, asks);
    if (world > 1) try comm.allGather(own, .init(picks.view.ptr.?, .u32, &.{ world, rows, slot }));
}

/// Row `i`'s start from the picks read back (`host`, u32 [world, rows, slot] as `gatherPicks` leaves them), each
/// rank's part of `half` columns, joined in rank order (`picking.joined`); null where a part's pick gave none.
pub fn startOf(comptime world: usize, host: []const u32, rows: usize, slot: usize, half: usize, i: usize, joined: *picking.Joined) ?sampling.Leading {
    var parts: [world]picking.Picked = undefined;
    for (&parts, 0..) |*one, rank| one.* = .read(host[(rank * rows + i) * slot ..][0..picking.Picked.words]);
    return picking.joined(&parts, half, joined);
}

/// `draft.Chooser`'s rows settled on the device, for a greedy choice: each rank picks the start of its half of
/// each of the window's rows (`gatherPicks`), one gather brings them all, read back into `picked` (the host's
/// copy, every slot asked to rank 1), and `out[i]` is the token `sampler.choose` gives for the whole row, the
/// largest, the lower id among equals: `chooseLeading` of the joined start. It stays null where the start does
/// not settle that (a zero on top, a NaN or an infinity, more equals than a pick holds): the row is read whole.
/// Every rank settles the same rows from the same picks, so the rows read are the same on each.
pub fn settle(comptime world: usize, comm: anytype, logits: cuda.Buffer, picks: cuda.Buffer, picked: []u32, row: usize, sampler: *sampling.Sampler, settings: sampling.Sampling, out: []?i32) !void {
    const rows = out.len;
    try gatherPicks(world, comm, logits, picks, null, row, rows);
    try cuda.synchronize();
    const slot: usize = @intCast(picks.view.size[2]);
    const half: usize = @intCast(logits.view.size[1]);
    const host = picked[0 .. world * rows * slot];
    try picks.downloadAt(0, std.mem.sliceAsBytes(host));
    var joined: picking.Joined = .{};
    for (out, 0..) |*token, i| {
        const start = startOf(world, host, rows, slot, half, i, &joined) orelse continue;
        token.* = sampler.chooseLeading(start, 0, settings);
    }
}

/// Where a round's rows are settled (`launchRound`, `readRound`): on the device the picks, u32 [most ranks, rows,
/// a pick's words over a rank's columns], and the rank each row is asked to, u8 [rows, rounded up to a word];
/// their host copies (`asked` is what `launchRound` sends), and a round's rows and tokens settled on the host.
/// Made for two ranks, of which one rank alone uses the first.
pub const Picks = struct {
    words: cuda.Buffer,
    asks: cuda.Buffer,
    host: []u32,
    asked: []u8,
    rows: []?Row,
    out: []?i32,
    gpa: Allocator,

    pub const most_ranks = 2;

    /// For rounds of up to `rows` rows, a rank scoring `columns` of the vocabulary.
    pub fn init(gpa: Allocator, rows: usize, columns: usize) !Picks {
        const slot = picking.Picked.wordsFor(columns);
        var words: cuda.Buffer = try .init(.u32, &.{ most_ranks, rows, slot });
        errdefer words.deinit();
        var asks: cuda.Buffer = try .init(.u8, &.{padded(rows)});
        errdefer asks.deinit();
        const host = try gpa.alloc(u32, most_ranks * rows * slot);
        errdefer gpa.free(host);
        const asked = try gpa.alloc(u8, padded(rows));
        errdefer gpa.free(asked);
        const held = try gpa.alloc(?Row, rows);
        errdefer gpa.free(held);
        const out = try gpa.alloc(?i32, rows);
        return .{ .words = words, .asks = asks, .host = host, .asked = asked, .rows = held, .out = out, .gpa = gpa };
    }

    /// The device bytes `init` makes.
    pub fn bytesFor(rows: usize, columns: usize) usize {
        return cuda.Buffer.bytesOf(.u32, &.{ most_ranks, rows, picking.Picked.wordsFor(columns) }) + cuda.Buffer.bytesOf(.u8, &.{padded(rows)});
    }

    pub fn deinit(self: *Picks) void {
        self.words.deinit();
        self.asks.deinit();
        self.gpa.free(self.host);
        self.gpa.free(self.asked);
        self.gpa.free(self.rows);
        self.gpa.free(self.out);
        self.* = undefined;
    }

    /// The rows a round may settle.
    pub fn most(self: Picks) usize {
        return self.rows.len;
    }

    fn padded(rows: usize) usize {
        return (rows + 3) / 4 * 4;
    }
};

/// A row of a round settled on the device: what its seat samples by and the position of the token chosen
/// after it, as `draft.Chooser.accept` keys it.
pub const Row = struct { settings: sampling.Sampling, position: u64 };

/// The rank each of a round's rows is asked to, `Picked.rankFor` its seat's sampling on a rank's `columns` of
/// `vocab`, into `out`; 0 where the row is not settled here (null in `rows`) or its start cannot settle it.
/// Whether any row is asked.
pub fn asksOf(rows: []const ?Row, columns: usize, vocab: usize, out: []u8) bool {
    var any = false;
    for (rows, out[0..rows.len]) |maybe, *ask| {
        ask.* = if (maybe) |one| @intCast(picking.Picked.rankFor(one.settings, columns, vocab)) else 0;
        any = any or ask.* != 0;
    }
    return any;
}

/// A round's launches on every rank, with the same asks in `picks.asked` (`asksOf`): rows 0 to `n` of this
/// rank's `logits` each cut at its ask (a row asked 0 at 1, its words unread), in one set of launches, and
/// gathered through `comm` where `world` is over 1 (`gatherPicks`; `{}` alone).
pub fn launchRound(comptime world: usize, comm: anytype, logits: cuda.Buffer, picks: *Picks, n: usize) !void {
    std.debug.assert(n > 0 and n <= picks.most());
    const padded = Picks.padded(n);
    @memset(picks.asked[n..padded], 0);
    try picks.asks.uploadLaunched(0, picks.asked[0..padded]);
    const asked: cuda.View = .init(picks.asks.view.ptr.?, .u8, &.{n});
    try gatherPicks(world, comm, logits, picks.words, asked, 0, n);
}

/// Rank 0's, after `launchRound` of `n` rows: the picks read back in one download, and `picks.out[i]` the token
/// `sampler.choose` gives row i's whole row of the vocabulary under `picks.rows[i]`'s sampling at its position
/// (`chooseLeading` of the joined start); null where the row was not asked or its start does not settle it,
/// and the row is read whole. `columns` are a rank's. Returns how many rows it settled.
pub fn readRound(comptime world: usize, picks: *Picks, n: usize, columns: usize, sampler: *sampling.Sampler) !usize {
    const rows = picks.rows[0..n];
    const asks = picks.asked[0..n];
    const out = picks.out[0..n];
    const slot: usize = @intCast(picks.words.view.size[2]);
    const host = picks.host[0 .. world * n * slot];
    try cuda.synchronize();
    try picks.words.downloadAt(0, std.mem.sliceAsBytes(host));
    var joined: picking.Joined = .{};
    var settled: usize = 0;
    for (rows, asks, out, 0..) |maybe, ask, *token, i| {
        token.* = null;
        const one = maybe orelse continue;
        if (ask == 0) continue;
        const start = startOf(world, host, n, slot, columns, i, &joined) orelse continue;
        token.* = sampler.chooseLeading(start, one.position, one.settings);
        settled += @intFromBool(token.* != null);
    }
    return settled;
}
