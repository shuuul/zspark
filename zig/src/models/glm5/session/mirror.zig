//! Rank 1 follows rank 0 (spec 045 WS-04; `zig/src/ranks/lead.zig`): the operations `ranks_forward.zig` and `mtp.zig` tell
//! before they run, read back and run on this rank's share, in the order told. The follower chooses nothing:
//! it has no sampler, no constraint and no drafter's rule, and a reply ends for it when rank 0 tells no more of
//! it. Its part of a gathered row of logits is sent as rank 0 reads it; what arrives here is read by no one.

const std = @import("std");
const channel_file = @import("../../../ranks/channel.zig");
const lead_file = @import("../../../ranks/lead.zig");
const picking = @import("../../../decode/pick.zig");
const phases = @import("../layers/forward.zig");
const flash = @import("../dflash2/drafter.zig");
const flash_pass = @import("../dflash2/pass.zig");
const flash_follow = @import("../../../decode/dflash2/follow.zig");
const held_file = @import("../layers/ranks_forward.zig");
const kda = @import("../layers/kda.zig");
const mtp = @import("../mtp/mtp.zig");
const seats_file = @import("../mtp/mtp_seats.zig");
const slots_file = @import("../pool/slots.zig");

const H = held_file.Held(1);
const Op = held_file.Op;

/// What the operations run on: this rank's share, as rank 0 holds its own.
pub const Followed = struct {
    m: H.Model,
    state: *H.State,
    b: *H.Buffers,
    /// As rank 0's: null where it launches every window.
    graphs: ?*H.Graphs,
    pair: *mtp.PairOf(1),
    /// The DFlash2 drafter where rank 0 drafts by it: its passes run here, its chain nowhere.
    flash: ?*flash.DrafterOf(1) = null,
    /// The sequences that share a window where rank 0 decodes several replies at once (`slots.zig`).
    slots: ?*slots_file.SlotsOf(1) = null,
    /// The MTP layer's entries of each of them.
    seats: ?*mtp.SeatsOf(1) = null,
    flash_seats: ?*flash_pass.SeatsOf(1) = null,
    /// A row of logits over the vocabulary: where a gathered row lands.
    logits: []u16,
    /// A snapshot of the committed state, and the return to the one at a position (`Op.keep`, `Op.back`).
    keep: ?*const fn (context: *anyopaque) anyerror!void = null,
    back: ?*const fn (context: *anyopaque, pos: usize) anyerror!void = null,
    /// A snapshot after the rows told of the window that has run, and the one at a position going
    /// (`Op.keep_within`, `Op.drop`): rank 0 plans the snapshots, this rank thins none of its own.
    within: ?*const fn (context: *anyopaque, rows: usize) anyerror!void = null,
    /// The snapshots inside the next window, asked before it runs (`Op.keep_ahead`): the rows into it of each.
    ahead: ?*const fn (context: *anyopaque, rows: []align(1) const u32) anyerror!void = null,
    drop: ?*const fn (context: *anyopaque, pos: usize) anyerror!void = null,
    /// A new sequence: the snapshots held are the one just left (`Op.reset`).
    fresh: ?*const fn (context: *anyopaque) void = null,
    /// A state to and from disk (`Op.spill_*`): the operation, its arguments and the tokens it was told.
    spill: ?*const fn (context: *anyopaque, op: Op, arguments: []align(1) const u32, tokens: []const i32) anyerror!void = null,
    snap: *anyopaque = undefined,
};

pub const Error = error{
    /// A frame that is no operation, or one this rank cannot run where it stands: the ranks have parted.
    OutOfStep,
    /// The expert picks of the last window are not rank 0's: the ranks' caches have parted.
    PicksDiffer,
} || channel_file.Error || held_file.Error || flash_follow.Error(flash_pass.Target) || std.mem.Allocator.Error;

/// Runs what rank 0 tells until it tells the end; returns the operations run, the end among them.
pub fn follow(channel: *channel_file.Channel, f: Followed) Error!usize {
    // a prompt window's tokens: `--rows` past `Lead.most` (1,024) takes the widest frame
    var window: [lead_file.Lead.widest]i32 = undefined;
    var ran: usize = 0;
    while (true) {
        const told = switch (try channel.next()) {
            .request => |bytes| lead_file.Told.read(bytes) orelse return error.OutOfStep,
            .barrier => return error.OutOfStep,
        };
        const op = std.enums.fromInt(Op, told.kind) orelse return error.OutOfStep;
        ran += 1;
        const arguments: usize = switch (op) {
            .reset, .mtp_row, .end, .keep, .spill_open => 0,
            .slot_reset, .mtp_row_of, .spill_tokens => 1,
            .spill_write, .spill_read => 2,
            .flash_taps, .flash_block, .flash_round => flash_follow.arguments(flash_pass.Target, op, told.rest) orelse return error.OutOfStep,
            .mtp_steps => if (told.rest.len >= 1 and told.rest[0] <= lead_file.Lead.most) seats_file.told(told.rest[0]) else return error.OutOfStep,
            .slots_forward => if (told.rest.len >= 1 and told.rest[0] <= lead_file.Lead.most) slots_file.told(told.rest[0]) else return error.OutOfStep,
            .slots_commit => if (told.rest.len >= 1 and told.rest[0] <= lead_file.Lead.most) 1 + told.rest[0] else return error.OutOfStep,
            .keep_ahead => if (told.rest.len >= 1 and told.rest[0] <= kda.most_asked) 1 + told.rest[0] else return error.OutOfStep,
            .commit, .row, .mtp_len, .back, .keep_within, .drop => 1,
            .graph, .settle => 2,
            .forward => 3,
            .mtp_step => 5,
        };
        const tokens = told.tokens(arguments, &window) orelse return error.OutOfStep;
        const a = told.rest[0..arguments];
        const rows = tokens.len;
        const windowed = op == .forward or op == .graph or op == .commit or op == .mtp_step or op == .slots_forward or op == .mtp_steps;
        // a sequence's tokens to name a state on disk come in frames of their own, of any width
        if (op != .spill_tokens and (windowed != (rows > 0) or rows > f.m.max_rows)) return error.OutOfStep;
        if (op == .forward or op == .graph or op == .mtp_step or op == .slots_forward or op == .mtp_steps) {
            const theirs = @as(u64, a[arguments - 2]) | @as(u64, a[arguments - 1]) << 32;
            const mine = try H.picks(f.b.*);
            if (mine != theirs) {
                std.log.err("operation {d}, {t}: the expert picks of the window before the last are {x} here and {x} on rank 0", .{ ran, op, mine, theirs });
                return error.PicksDiffer;
            }
        }
        switch (op) {
            .reset => {
                try f.state.reset();
                if (f.fresh) |fresh| fresh(f.snap);
            },
            .forward => {
                const score = std.enums.fromInt(phases.Score, a[0]) orelse return error.OutOfStep;
                if (f.state.pos + rows > f.b.capacity) return error.OutOfStep;
                try H.forwardScoring(f.m, f.state, f.b, tokens, score);
            },
            .graph => {
                const graphs = f.graphs orelse return error.OutOfStep;
                if (f.state.pos + rows > f.b.capacity) return error.OutOfStep;
                try graphs.forward(f.m, f.state, f.b, tokens);
            },
            .commit => {
                if (a[0] > rows) return error.OutOfStep;
                try H.commit(f.m, f.state, f.b.*, tokens, a[0]);
            },
            .row => {
                if (a[0] >= f.b.rows) return error.OutOfStep;
                try H.logitsRow(f.b.*, a[0], f.logits);
            },
            .settle => {
                if (a[1] == 0 or a[1] > H.settled_rows or @as(usize, a[0]) + a[1] > f.b.rows) return error.OutOfStep;
                try H.gatherPicks(f.b.*, a[0], a[1]);
            },
            .mtp_step => {
                if (a[0] > 1 or a[2] > picking.Picked.most or f.pair.len() + rows > f.pair.caches[0].capacity) return error.OutOfStep;
                if (a[0] == 0 and a[1] + rows > f.b.rows) return error.OutOfStep;
                // a step that scores cuts this rank's half of the row at the rank rank 0 asks its own to
                if (a[2] != 0) f.pair.ask = a[2];
                try f.pair.step(f.m, f.b, if (a[0] == 0) .{ .main = a[1] } else .own, tokens, a[2] != 0);
            },
            .mtp_len => {
                if (a[0] > f.pair.caches[0].capacity) return error.OutOfStep;
                try f.pair.setLen(a[0]);
            },
            .mtp_row => try f.pair.wholeRow(f.b.*, f.logits),
            .flash_taps, .flash_block, .flash_round => try flash_follow.follow(flash_pass.Target, op, a, tokens, f.flash, f.flash_seats, f.b.rows, f.b.ranks[0].taps),
            .mtp_steps => {
                const seats = f.seats orelse return error.OutOfStep;
                if (a[0] == 0 or a[0] > seats.count()) return error.OutOfStep;
                const items = seats.items[0..a[0]];
                var total: usize = 0;
                var scored: usize = 0;
                for (items, 0..) |*item, at| {
                    const words = a[1 + 5 * at ..][0..5];
                    if (words[0] >= seats.count() or words[2] == 0 or words[2] > rows or words[4] > picking.Picked.most) return error.OutOfStep;
                    if (@as(usize, words[1]) + words[2] > seats.capacity) return error.OutOfStep;
                    for (items[0..at]) |before| if (before.slot == words[0]) return error.OutOfStep;
                    // the scored first: their rows are packed
                    if (words[4] != 0 and scored != at) return error.OutOfStep;
                    scored += @intFromBool(words[4] != 0);
                    const row = words[3] >> 1;
                    if (words[3] & 1 == 0) {
                        if (@as(usize, row) + words[2] > f.b.rows) return error.OutOfStep;
                    } else if (row >= seats.ran or words[2] != 1) return error.OutOfStep;
                    item.* = .{ .slot = words[0], .at = words[1], .rows = words[2], .hidden = if (words[3] & 1 == 0) .{ .main = row } else .{ .own = row }, .ask = words[4] };
                    total += item.rows;
                }
                if (total != rows) return error.OutOfStep;
                try seats.step(f.pair, f.m, f.b, items, tokens);
            },
            .mtp_row_of => {
                const seats = f.seats orelse return error.OutOfStep;
                if (a[0] >= seats.scored) return error.OutOfStep;
                try seats.wholeRow(f.pair, f.b.*, a[0], f.logits);
            },
            .slot_reset => {
                const slots = f.slots orelse return error.OutOfStep;
                if (a[0] >= slots.count()) return error.OutOfStep;
                try slots.reset(f.m, a[0]);
            },
            .slots_forward => {
                const slots = f.slots orelse return error.OutOfStep;
                const parts = slots.parts;
                if (a[0] == 0 or a[0] > parts.len) return error.OutOfStep;
                var total: usize = 0;
                var scored: usize = 0;
                for (parts[0..a[0]], 0..) |*part, at| {
                    const words = a[1 + 3 * at ..][0..3];
                    const score = std.enums.fromInt(phases.Score, words[2]) orelse return error.OutOfStep;
                    if (words[0] >= slots.count() or words[1] == 0 or words[1] > rows) return error.OutOfStep;
                    for (parts[0..at]) |before| if (before.slot == words[0]) return error.OutOfStep;
                    if (slots.pos(words[0]) + words[1] > slots.capacity) return error.OutOfStep;
                    part.* = .{ .slot = words[0], .rows = words[1], .score = score };
                    total += part.rows;
                    scored += part.scored();
                }
                if (total != rows or scored > slots.most_scored) return error.OutOfStep;
                try slots.forward(f.m, f.b, parts[0..a[0]], tokens);
            },
            .slots_commit => {
                const slots = f.slots orelse return error.OutOfStep;
                if (a[0] != slots.ran) return error.OutOfStep;
                var keeps: [lead_file.Lead.most]usize = undefined;
                for (keeps[0..a[0]], a[1..], slots.spans[0..a[0]]) |*keep, word, span| {
                    if (word > span.rows or slots.pos(span.slot) + word > slots.capacity) return error.OutOfStep;
                    keep.* = word;
                }
                try slots.commit(f.m, keeps[0..a[0]]);
            },
            .keep => {
                const keep = f.keep orelse return error.OutOfStep;
                keep(f.snap) catch return error.OutOfStep;
            },
            .back => {
                const back = f.back orelse return error.OutOfStep;
                back(f.snap, a[0]) catch return error.OutOfStep;
            },
            .keep_within => {
                const within = f.within orelse return error.OutOfStep;
                within(f.snap, a[0]) catch return error.OutOfStep;
            },
            .keep_ahead => {
                const ahead = f.ahead orelse return error.OutOfStep;
                ahead(f.snap, a[1..]) catch return error.OutOfStep;
            },
            .drop => {
                const drop = f.drop orelse return error.OutOfStep;
                drop(f.snap, a[0]) catch return error.OutOfStep;
            },
            .spill_tokens, .spill_open, .spill_write, .spill_read => {
                const spill = f.spill orelse return error.OutOfStep;
                spill(f.snap, op, a, tokens) catch return error.OutOfStep;
            },
            .end => return ran,
        }
    }
}
