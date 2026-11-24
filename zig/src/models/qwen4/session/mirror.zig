//! Rank 1 runs the operations rank 0 told (`ranks_forward.zig`'s `Op`), in that order, and chooses nothing.
//! A reply ends for it when rank 0 tells no more of it. The tokens of the reply arrive as `pick`, after the
//! window that chose them; what arrives here is the list rank 0 emitted. Which states are kept and dropped is
//! rank 0's to say too (`decode/book.zig`): this rank refuses a state that would not stand above those it holds,
//! one inside a window it did not just run, and a position it does not hold. DFlash2's passes join the stream
//! (`flash_taps`, `flash_block`, `flash_round`): this rank consumes the leader's taps and runs the block; it
//! proposes nothing itself.

const std = @import("std");
const draft_file = @import("../../../decode/draft.zig");
const channel_file = @import("../../../ranks/channel.zig");
const lead_file = @import("../../../ranks/lead.zig");
const held_file = @import("../ranks_forward.zig");
const mtp_ranks = @import("../mtp_ranks.zig");
const rounds_ranks = @import("../pool/rounds_ranks.zig");
const flash = @import("../dflash2/drafter.zig");
const flash_pass = @import("../dflash2/pass.zig");
const flash_follow = @import("../../../decode/dflash2/follow.zig");

const H = held_file.Held(1);
const Op = held_file.Op;
const Desk = rounds_ranks.Desk;

pub const Followed = struct {
    m: H.Model,
    state: *H.State,
    b: *H.Buffers,
    graphs: ?*H.Graphs,
    logits: []u16,
    /// The positions the state was made for.
    capacity: usize,
    gpa: std.mem.Allocator,
    /// The reply's tokens, in the order rank 0 emitted them.
    tokens: *std.ArrayList(i32),
    /// This rank's states, at the positions rank 0 kept.
    shelf: *H.Shelf,
    /// The states asked of the next window (`Op.keep_ahead`), until it has run and each is taken.
    asked: *H.Asked,
    /// Set when rank 0 may tell a round of several streams. The one-reply path leaves it null.
    desk: ?*Desk = null,
    /// This rank's half of the MTP head where it is split (`mtp_ranks.zig`); null, rank 0 tells no pass of it.
    head: ?*mtp_ranks.Held(1).Split = null,
    /// What runs a reply's rounds as rank 0 does once it tells `decode` (`pair.zig`'s `Rounds`); null, it
    /// never does. `prompt` runs a prompt told once (`pair.zig`'s `Prompt`) the same way, a frame at a time.
    rounds: ?Rounds = null,
    /// The DFlash2 drafter where rank 0 drafts by it: its passes run here, its chain nowhere.
    flash: ?*flash.DrafterOf(1) = null,
    /// Several sequences' DFlash2 seats where rank 0's hall tells a round.
    flash_seats: ?*flash_pass.SeatsOf(1) = null,

    pub const Rounds = struct {
        context: *anyopaque,
        /// The operation's arguments and tokens; returns once the reply's last round has run.
        run: *const fn (context: *anyopaque, arguments: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void,
        /// A frame of a prompt told once (`pair.zig`'s `promptFollowed`); the last frame's arrival runs the
        /// prompt. Null, the prompt is told per pass.
        prompt: ?*const fn (context: *anyopaque, arguments: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void = null,
    };
};

pub const Error = error{OutOfStep} || channel_file.Error || H.Error || rounds_ranks.Error || draft_file.Error(H) || flash_follow.Error(flash_pass.Target) || std.mem.Allocator.Error;

/// Runs what rank 0 tells until `end`. Returns how many operations ran, the end among them.
pub fn follow(channel: *channel_file.Channel, f: Followed) Error!usize {
    // as wide as a pair's frames (`Lead.widest`): a window told a pass is a prompt's, 2,048 rows and more
    var window: [lead_file.Lead.widest]i32 = undefined;
    // the window run last and not committed, which a state may be kept inside
    var last: [lead_file.Lead.widest]i32 = undefined;
    var open: usize = 0;
    var ran: usize = 0;
    while (true) {
        const told = switch (try channel.next()) {
            .request => |bytes| lead_file.Told.read(bytes) orelse return error.OutOfStep,
            .barrier => return error.OutOfStep,
        };
        const op = std.enums.fromInt(Op, told.kind) orelse return error.OutOfStep;
        ran += 1;
        const arguments: usize = switch (op) {
            .reset, .end, .pick, .graph, .keep => 0,
            .commit, .row, .forward, .back, .keep_within, .drop => 1,
            .head, .table => 3,
            .decode => 19,
            .prompt => 20,
            .keep_ahead => if (told.rest.len >= 2 and told.rest[1] <= H.most_asked) 2 + @as(usize, told.rest[1]) else return error.OutOfStep,
            .flash_taps, .flash_block, .flash_round => flash_follow.arguments(flash_pass.Target, op, told.rest) orelse return error.OutOfStep,
            // the pool's engine is followed by `pool/hall_ranks.zig`, never here
            .hall_round, .hall_logits, .hall_commit, .hall_reset, .hall_head, .hall_keep, .hall_wanted, .hall_within, .hall_hold, .hall_drop, .hall_fork, .hall_capture, .hall_settle => return error.OutOfStep,
            .streams, .streams_commit => blk: {
                if (told.rest.len < 1 or told.rest[0] == 0 or told.rest[0] > Desk.most) return error.OutOfStep;
                break :blk 1 + @as(usize, if (op == .streams) 2 else 1) * told.rest[0];
            },
        };
        const tokens = told.tokens(arguments, &window) orelse return error.OutOfStep;
        const a = told.rest[0..arguments];
        switch (op) {
            .reset => {
                if (tokens.len != 0) return error.OutOfStep;
                H.dropAsked(f.shelf, f.asked);
                for (&f.state.ranks) |*one| try one.reset();
                f.state.pos = 0;
                open = 0;
            },
            .forward => {
                const score = std.enums.fromInt(held_file.Score, a[0]) orelse return error.OutOfStep;
                if (tokens.len == 0 or f.state.pos + tokens.len > f.capacity) return error.OutOfStep;
                if (f.asked.count() > 0 and f.asked.rows != tokens.len) return error.OutOfStep;
                try H.forwardScoring(f.m, f.state, f.b, tokens, score);
                @memcpy(last[0..tokens.len], tokens);
                open = tokens.len;
            },
            .graph => {
                const graphs = f.graphs orelse return error.OutOfStep;
                if (tokens.len == 0 or f.state.pos + tokens.len > f.capacity or f.asked.count() > 0) return error.OutOfStep;
                try graphs.forward(f.m, f.state, f.b, tokens);
                @memcpy(last[0..tokens.len], tokens);
                open = tokens.len;
            },
            .commit => {
                if (a[0] == 0 or a[0] > tokens.len) return error.OutOfStep;
                try H.commit(f.m, f.state, f.b.*, tokens, a[0]);
                H.dropAsked(f.shelf, f.asked);
                open = 0;
            },
            .row => {
                if (a[0] >= f.b.rows) return error.OutOfStep;
                try H.logitsRow(f.b.*, a[0], f.logits);
            },
            .pick => {
                if (tokens.len == 0) return error.OutOfStep;
                try f.tokens.appendSlice(f.gpa, tokens);
            },
            .keep => {
                if (tokens.len != 0) return error.OutOfStep;
                try H.keepLocal(f.shelf, f.state);
            },
            .keep_within => {
                if (open == 0 or a[0] == 0 or a[0] > open or !std.mem.eql(i32, last[0..open], tokens)) return error.OutOfStep;
                try H.keepWithinLocal(f.shelf, f.asked, f.m, f.state, f.b, tokens, a[0]);
            },
            .keep_ahead => {
                if (tokens.len != 0 or open != 0) return error.OutOfStep;
                try H.askLocal(f.shelf, f.asked, f.state, f.b, a[0], a[2..]);
            },
            .drop => {
                if (tokens.len != 0) return error.OutOfStep;
                try H.dropLocal(f.shelf, a[0]);
            },
            .back => {
                if (tokens.len != 0) return error.OutOfStep;
                H.dropAsked(f.shelf, f.asked);
                try H.backLocal(f.shelf, f.state, a[0]);
                open = 0;
            },
            .table => {
                const desk = f.desk orelse return error.OutOfStep;
                // rank 0 tells a table's entries from the first that changed to its end
                if (a[2] < a[1] or a[2] - a[1] != tokens.len) return error.OutOfStep;
                try desk.place(a[0], a[1], tokens);
            },
            .streams => {
                const desk = f.desk orelse return error.OutOfStep;
                var parts: [Desk.most]rounds_ranks.Part = undefined;
                for (parts[0..a[0]], 0..) |*part, at| part.* = .{ .slot = a[1 + 2 * at], .rows = a[2 + 2 * at] };
                try rounds_ranks.shared(f.m, desk, f.b, parts[0..a[0]], tokens);
                open = 0;
            },
            .streams_commit => {
                const desk = f.desk orelse return error.OutOfStep;
                if (tokens.len != 0) return error.OutOfStep;
                var keeps: [Desk.most]usize = undefined;
                for (keeps[0..a[0]], a[1..][0..a[0]]) |*keep, word| keep.* = word;
                try rounds_ranks.commit(f.m, desk, f.b, keeps[0..a[0]]);
            },
            .head => {
                const head = f.head orelse return error.OutOfStep;
                const own: ?usize = if (a[1] == 0) null else a[1] - 1;
                if (tokens.len == 0 or tokens.len > head.rows or a[0] + tokens.len > head.capacity or a[2] > 3) return error.OutOfStep;
                if (own) |row| if (tokens.len != 1 or row >= head.rows) return error.OutOfStep;
                try head.follow(a[0], own, tokens, a[2] & mtp_ranks.scored != 0, a[2] & mtp_ranks.launched != 0);
            },
            .decode => {
                const rounds = f.rounds orelse return error.OutOfStep;
                try rounds.run(rounds.context, a, tokens);
                open = 0;
            },
            .prompt => {
                const rounds = f.rounds orelse return error.OutOfStep;
                const prompt = rounds.prompt orelse return error.OutOfStep;
                try prompt(rounds.context, a, tokens);
                open = 0;
            },
            .end => return ran,
            .flash_taps, .flash_block, .flash_round => try flash_follow.follow(flash_pass.Target, op, a, tokens, f.flash, f.flash_seats, f.b.rows, f.b.ranks[0].taps),
            .hall_round, .hall_logits, .hall_commit, .hall_reset, .hall_head, .hall_keep, .hall_wanted, .hall_within, .hall_hold, .hall_drop, .hall_fork, .hall_capture, .hall_settle => unreachable,
        }
    }
}
