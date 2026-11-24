//! The head chase of the pool (`hall_engine.zig`'s `Engine`): a round's rows taken in by the head, the seats'
//! chains followed as far as the host takes them and the head run for the drafts it must, DFlash2's drafts in
//! the head's place, the price of a draft in the round and what the seats keep of it. `Engine` names each
//! function, so a call reads `self.chase(live)`.

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const masks_file = @import("../../../decode/masks.zig");
const draft = @import("../draft.zig");
const hall_engine = @import("hall_engine.zig");
const hall_file = @import("hall.zig");
const hall_seats = @import("hall_seats.zig");
const keep_file = @import("hall_keep.zig");
const head_file = @import("hall_head.zig");
const depth_file = @import("hall_depth.zig");
const mtp = @import("../mtp.zig");
const rounds = @import("rounds.zig");
const Engine = hall_engine.Engine;
const Live = hall_seats.Live;
const Seat = hall_seats.Seat;
const Span = hall_seats.Span;
const window_max = masks_file.max_rows;

pub fn absorbAndChase(self: *Engine, segments: []const rounds.Segment, nreply: usize, prompt_slot: ?usize, prompt_tokens: []const i32, prompt_done: bool, reasoned: u64) !void {
    // Each decoding seat's chain is begun before the head takes in the round's rows: the head scores a
    // seat's last row only where its first draft is the head's to choose, as `mtp.Head.drafts` does.
    var live = try self.gpa.alloc(Live, self.slots.len);
    defer self.gpa.free(live);
    for (live) |*one| one.* = .{};
    var decoding: usize = 0;
    for (self.seats) |maybe| if (maybe) |seat| if (seat.phase == .decode) {
        decoding += 1;
    };
    for (self.seats, 0..) |maybe, i| {
        if (maybe == null) continue;
        const seat = &self.seats[i].?;
        if (seat.phase != .decode) continue;
        const most = @min(if (seat.first) seat.opening else seat.depth, window_max - 1);
        seat.first = false;
        const room = depth_file.room(self.depth_cap, self.ranks != null, decoding, most);
        if (room < most) self.capped += 1;
        const capped = @min(room, if (seat.max_tokens > seat.made) seat.max_tokens - seat.made - 1 else 0);
        // a seat the round grants no verify row to: its kept rows are still absorbed, and no chain is begun
        // for it (TensorFold #1779: the priced-out stream skips the drafter's proposal work)
        if (capped == 0) continue;
        live[i].chain = try .begin(live[i].buf[0..capped], self.guide(i));
        live[i].scored = live[i].chain.scores();
    }

    var spans: [64]Span = undefined;
    var nspan: usize = 0;
    var row: usize = 0;
    for (segments[0..nreply]) |segment| {
        const seat = &self.seats[segment.slot].?;
        // A reply that ended in this round is taken in too, and scores nothing: the head then stands
        // where the trunk does, for the state `retire` keeps of it (as `Session.reply` absorbs its last
        // window for the next prompt).
        spans[nspan] = .{
            .slot = segment.slot,
            .source = self.buffers.h,
            .source_row = row,
            .n = segment.keep,
            .tokens = seat.follow[0..segment.keep],
            .pos = 0,
            // The last kept row's draft logits are the next token. Sampling the verify logits again
            // would draft the token just emitted, and the next window would reject it.
            .score = live[segment.slot].scored,
        };
        nspan += 1;
        row += segment.tokens.len;
    }
    if (nspan > 0) try self.absorb(spans[0..nspan]);
    if (prompt_slot) |index| {
        const seat = &self.seats[index].?;
        if (seat.phase != .done or prompt_done) {
            const n = prompt_tokens.len;
            if (!prompt_done) {
                const from = seat.filled - n;
                @memcpy(seat.follow[0..n], seat.prompt[from + 1 .. from + 1 + n]);
            } else {
                if (n > 1) @memcpy(seat.follow[0 .. n - 1], seat.prompt[seat.prompt.len - n + 1 ..]);
                seat.follow[n - 1] = seat.last;
            }
            spans[0] = .{
                .slot = index,
                .source = self.buffers.h,
                .source_row = row,
                .n = n,
                .tokens = seat.follow[0..n],
                .pos = 0,
                .score = live[index].scored,
            };
            try self.absorb(spans[0..1]);
        }
    }
    // before the head chains its drafts: its cache holds the round's rows and no more
    try self.keepRound(segments[0..nreply], prompt_slot, reasoned);

    self.priced = self.price(segments);
    const passes = self.chosen.passes;
    try self.chase(live);
    for (self.seats, live, 0..) |*maybe, one, index| {
        const seat = &(maybe.* orelse continue);
        if (seat.phase != .decode) continue;
        if (seat.paying) |*paying| paying.counts.levels += self.chosen.passes - passes;
        seat.draft_n = one.chain.count;
        if (one.chain.out.len > 0) seat.last_chain = mtp.Paying.told(one.chain);
        @memcpy(seat.drafts[0..one.chain.count], one.buf[0..one.chain.count]);
        // as a session's `Copied`: the copy is the window where the head's first draft is its first
        if (seat.copies) |*copies| if (seat.draft_n > 0) {
            seat.draft_n = try copies.offer(seat.draft_n, seat.drafts[0..one.chain.out.len]);
        };
        seat.digest.update(std.mem.sliceAsBytes(seat.drafts[0..seat.draft_n]));
        seat.digest.update(&.{@intCast(seat.draft_n)});
        if (seat.masks) |*masks| {
            const sure = masks.certain(seat.drafts[0..seat.draft_n]);
            // Forced drafts are the reply's tokens; the window that verifies them still runs.
            seat.ahead = if (seat.logprobs != null) 0 else sure;
        }
        // the window the next round runs, as far as the round's cuts leave it: its rows past those the passes
        // asked (copied drafts, a chain's forced tail) gathered while the host finishes the round (spec 144 WS-10)
        if (self.ahead) |*ahead| {
            var window: [window_max]i32 = undefined;
            const drafts = @min(seat.draft_n, window_max - 1);
            window[0] = seat.last;
            @memcpy(window[1..][0..drafts], seat.drafts[0..drafts]);
            ahead.ask(self.m, self.slots, index, window[0 .. 1 + drafts]);
        }
    }
}

/// DFlash2's drafts after the round (`SeatsOf.draft`): the rows each stream kept of it into its seat's
/// context, then one pass of the blocks of the streams that decode, as a lone stream's drafter takes a
/// window's kept rows and drafts after them (`DrafterOf.drafts`). A prompt chunk's rows are taken in as a
/// session's prompt windows are, the last chunk's with the reply's first token, after which it drafts.
pub fn flashRound(self: *Engine, replies: []const rounds.Segment, prompt_slot: ?usize, prompt_tokens: []const i32, prompt_done: bool) !void {
    var jobs: [hall_file.most_parallel]draft.Job = undefined;
    var n: usize = 0;
    var row: usize = 0;
    for (replies) |segment| {
        defer row += segment.tokens.len;
        // a reply that ended leaves its seat: what goes on from its state begins it again
        if (self.seats[segment.slot].?.phase != .decode) continue;
        jobs[n] = self.flashJob(segment.slot, row, self.seats[segment.slot].?.follow[0..segment.keep], true);
        n += 1;
    }
    if (prompt_slot) |index| {
        const seat = &self.seats[index].?;
        if (seat.phase != .done) {
            const count = prompt_tokens.len;
            // the tokens after each of the chunk's rows: the prompt's own, then the reply's first
            if (!prompt_done) {
                const from = seat.filled - count;
                @memcpy(seat.follow[0..count], seat.prompt[from + 1 .. from + 1 + count]);
            } else {
                if (count > 1) @memcpy(seat.follow[0 .. count - 1], seat.prompt[seat.prompt.len - count + 1 ..]);
                seat.follow[count - 1] = seat.last;
            }
            jobs[n] = self.flashJob(index, row, seat.follow[0..count], prompt_done);
            n += 1;
        }
    }
    if (n == 0) return;
    if (self.rank_flash) |flash| try flash.seats.draft(&flash.flash.d, jobs[0..n]) else try self.flash_seats.?.draft(&self.flash.?.d, jobs[0..n]);
    for (jobs[0..n]) |job| {
        const seat = &self.seats[job.slot].?;
        if (seat.phase != .decode) continue;
        seat.first = false;
        seat.draft_n = job.count;
        seat.digest.update(std.mem.sliceAsBytes(seat.drafts[0..seat.draft_n]));
        seat.digest.update(&.{@intCast(seat.draft_n)});
        if (seat.masks) |*masks| {
            const sure = masks.certain(seat.drafts[0..seat.draft_n]);
            seat.ahead = if (seat.logprobs != null) 0 else sure;
        }
    }
}

/// Seat `index`'s part of `flashRound`: `kept`, its rows from `from` of the round's, then at most as many
/// drafts as the head would be asked for (none where `drafts` is false), into the seat's `drafts`.
pub fn flashJob(self: *Engine, index: usize, from: usize, kept: []const i32, drafts: bool) draft.Job {
    const seat = &self.seats[index].?;
    const most = if (seat.first) @min(seat.depth, seat.opening) else seat.depth;
    const left = if (seat.max_tokens > seat.made) seat.max_tokens - seat.made - 1 else 0;
    const room = if (drafts) @min(most, window_max - 1, left) else 0;
    return .{ .slot = index, .from = from, .kept = kept, .position = self.slots[index].pos, .settings = seat.settings, .ends = seat.eos, .guide = self.guide(index), .out = seat.drafts[0..room], .full = &seat.flash_full };
}

/// The price of a draft in the round the decoding seats share (`mtp.Paying.Round`): their rates summed
/// as they stand after this round, a row at this round's rows, the prompt chunk's among them.
pub fn price(self: *const Engine, segments: []const rounds.Segment) ?mtp.Paying.Round.Price {
    const costs = (self.m.mtp orelse return null).costs orelse return null;
    var sums: mtp.Paying.Round = .{};
    for (self.seats) |maybe| {
        const seat = maybe orelse continue;
        if (seat.phase == .decode) if (seat.paying) |paying| sums.add(paying);
    }
    var rows: usize = 0;
    for (segments) |segment| rows += segment.tokens.len;
    return sums.price(costs, rows);
}

/// What the round's seats keep of it (`hall_keep.after`), the head having taken its rows in.
pub fn keepRound(self: *Engine, replies: []const rounds.Segment, prompt_slot: ?usize, reasoned: u64) !void {
    for (replies) |segment| try keep_file.after(self, segment.slot, false, reasoned >> @intCast(segment.slot) & 1 != 0);
    if (prompt_slot) |index| try keep_file.after(self, index, true, false);
}

pub fn trim(self: *Engine, index: usize) void {
    const seat = &self.seats[index].?;
    self.slots[index].head_len -= seat.drafted_rows;
    seat.drafted_rows = 0;
}

pub fn absorb(self: *Engine, spans: []Span) !void {
    for (spans) |span| self.trim(span.slot);
    for (spans) |*span| span.pos = self.slots[span.slot].head_len;
    try head_file.headPass(self, spans, false);
}

pub fn chase(self: *Engine, live: []Live) !void {
    var spins: usize = 0;
    while (spins < 64) : (spins += 1) {
        var run_of: [64]usize = undefined;
        var nrun: usize = 0;
        var halted: usize = 0;
        for (live, 0..) |*one, i| {
            if (try self.host(one, i) == .halt) {
                halted += 1;
                continue;
            }
            run_of[nrun] = i;
            nrun += 1;
        }
        if (halted == live.len or nrun == 0) return;
        var spans: [64]Span = undefined;
        for (run_of[0..nrun], 0..) |index, s| {
            const one = &live[index];
            spans[s] = .{
                .slot = index,
                .source = self.held.?,
                .source_row = index,
                .n = 1,
                .tokens = one.buf[one.chain.ran..][0..1],
                .pos = self.slots[index].head_len,
                .score = true,
            };
        }
        try head_file.headPass(self, spans[0..nrun], true);
        self.chosen.rows += nrun;
        self.chosen.passes += 1;
        for (run_of[0..nrun]) |index| {
            live[index].chain.ran += 1;
            live[index].scored = true;
        }
    }
    return error.DraftStuck;
}

/// Seat `index`'s chain as far as the host takes it: until it ends, or the head must run a draft for a
/// later one to be its own (`draft.Chain.step`, as a session's head: a run of forced tokens costs no
/// head row).
pub fn host(self: *Engine, live: *Live, index: usize) !draft.Chain.Need {
    while (true) {
        // a halted chain may be an empty seat's: nothing of the seat is read for it
        if (live.chain.halt) return .halt;
        switch (try live.chain.step(self.guide(index))) {
            .halt => return .halt,
            .run => return .run,
            .choose => {
                std.debug.assert(live.scored);
                try self.choose(live, index);
            },
        }
    }
}

/// The constraint of seat `index`'s reply as a chain asks it, where it has one.
pub fn guide(self: *Engine, index: usize) ?masks_file.Guide {
    return if (self.seats[index].?.masks) |*masks| masks.guide() else null;
}

/// The head's choice of seat `index`'s next draft, by `rule`: from what the device found in the seat's
/// scored row where that settles it, else from the row read back.
pub fn choose(self: *Engine, live: *Live, index: usize) !void {
    const seat = &self.seats[index].?;
    const chain = &live.chain;
    var rule = self.rule;
    rule.paying = if (seat.paying) |paying| paying.within(self.priced) else null;
    const position: u64 = @intCast(self.slots[index].pos + 1 + chain.count);
    var choice: ?mtp.Rule.Choice = null;
    // a mask is laid over the row itself; a free draft needs the row's start alone
    if (chain.next == .free and self.rule.on_device and self.asked[index].rank != 0) {
        try cuda.synchronize();
        var raw: [mtp.Picked.words]u32 = undefined;
        try self.picked[index].downloadAt(0, std.mem.sliceAsBytes(&raw));
        choice = rule.onDevice(&raw, self.m.mtp.?.scored(), self.m.shape.vocab, &self.sampler, seat.settings, position, chain.product, chain.trail[0..chain.own]);
    }
    self.chosen.device += @intFromBool(choice != null);
    if (choice == null) {
        const row = try head_file.fetch(self, index);
        const among: ?[]const u32 = switch (chain.next) {
            .among => |mask| mask,
            else => null,
        };
        choice = try rule.onHost(row, self.head_rows[index], among, &self.whole.?, &self.sampler, seat.settings, position, chain.product, chain.trail[0..chain.own]) orelse {
            chain.halt = true;
            return;
        };
        self.chosen.host += 1;
    }
    try chain.take(self.guide(index), choice.?.token, choice.?.p, choice.?.low, choice.?.closing or std.mem.indexOfScalar(u32, seat.eos, @intCast(choice.?.token)) != null);
}
