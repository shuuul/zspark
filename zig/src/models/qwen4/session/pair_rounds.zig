//! A reply's rounds run on both ranks of a pair (spec 095), moved out of `pair.zig` (spec 104 WS-06): what
//! rank 0 says inside the stream (`Words`) and the rounds themselves (`Rounds`), each rank drafting and choosing
//! from the same gathered rows. `pair.zig`'s `Of(held)` names them.
const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const copied = @import("../../../decode/copied.zig");
const draft_file = @import("../../../decode/draft.zig");
const lead_file = @import("../../../ranks/lead.zig");
const sampling = @import("../../../decode/sampling.zig");
const nccl = @import("../../../ranks/nccl.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const request_file = @import("../../../server/request.zig");
const mtp = @import("../mtp.zig");
const held_file = @import("../ranks_forward.zig");
const pair_file = @import("pair.zig");
const Watch = pair_file.Watch;

pub fn Of(comptime held: usize) type {
    const H = held_file.Held(held);
    const Self = pair_file.Of(held);
    return struct {
        const Keeping = Self.Keeping;
        const Take = Self.Take;
        const halted = Self.halted;
        const outOfStep = Self.outOfStep;

        /// What rank 0 says inside the stream while both ranks run a reply's rounds, `width` words over NCCL
        /// (`ranks.Said`, which the pool's operations share): a few microseconds on the link where a told
        /// operation wakes a blocked read (spec 095 WS-00: some 0.4 ms).
        pub const Words = struct {
            said: ranks_file.Said,

            /// Rank 0's after a window's choices (`Rounds.keptLed`): a kind, the end, the rows kept and emitted,
            /// the next window's drafts asked, the states taken (two at most), the window run, what the copies'
            /// lookup found for the next window.
            pub const width = 48;
            /// The first word: which words, so that two ranks out of step stop at once.
            pub const state_kind = 2;
            pub const found_at = 27;

            pub fn init(comm: nccl.Comm) cuda.Error!Words {
                return .{ .said = try .init(comm, width) };
            }

            /// Rank 0's.
            fn say(self: Words, words: *const [width]u32) draft_file.Error(H)!void {
                try self.said.say(words);
            }

            /// Rank 1's: rank 0's words, once they have come.
            fn hear(self: Words, out: *[width]u32) draft_file.Error(H)!void {
                try self.said.hear(out);
            }
        };

        /// A reply's rounds run on both ranks (spec 095): rank 0 tells rank 1 once (`Op.decode`, `told`), and
        /// each then drafts with its own half of the head, runs the window and chooses its rows, all from the
        /// rows the gathers leave alike on both (`H.settle`, the head's scored row). After each window's choices
        /// and before its commit rank 0 says the rest (`Words`): the reply's end, the tokens emitted, the next
        /// window's drafts asked, the states kept inside the window and, where copies are offered, what their
        /// lookup found for the next window, asked ahead (`copied.Copied.ahead`), which rank 1 offers over its
        /// own drafts as rank 0 does (`copyOf`). Rank 1 checks the window and its kept rows against them. The
        /// last window's absorbing pass is each rank's, with nothing said.
        pub const Rounds = struct {
            pair: *Self,
            /// `Pair.chosen`: after a prompt, its last window's rows; after a round, its kept rows.
            chosen: []i32,
            /// Copies may be offered: rank 0 looks them up ahead and says what it found.
            copying: bool,
            /// The most drafts a window carries (`Decoder.depth`).
            depth: usize = 0,
            /// Rank 0's: what keeps the states inside the windows, and what decides the reply's end.
            keeping: ?*Keeping = null,
            lead: ?Led = null,
            /// What the last window's state words said: the reply ends, and the next window's drafts asked.
            ended: ?request_file.Finish = null,
            want: usize = 0,
            /// Rank 1's: what the copies' lookup found for the next window, as rank 0 said it.
            found: [draft_file.forward_max_rows]u32 = undefined,
            found_len: usize = 0,

            pub const most_ends = 64;
            /// The most rows a prompt's last window keeps: the first `Op.decode` frame carries them beside its
            /// arguments, the end tokens and a lookup's finds, in one frame of the channel (`Lead.widest`):
            /// 4,132, a window of 4,096 rows whole (spec 111).
            pub const kept_most = lead_file.Lead.widest - 1 - arguments - most_ends - draft_file.forward_max_rows;

            const Led = struct {
                before: usize,
                max_tokens: usize,
                depth: usize,
                eos: []const u32,
                hang_at: ?usize,
                watch: ?Watch,
                stopped: *?i32,
            };

            /// `Op.decode`'s arguments: the sampling (`seed`, `temperature`, `top_k`, `top_p`, `min_p`, nine
            /// words), the head's position, chained drafts and fullness, the first window's drafts asked, the
            /// tokens chosen after the last window's rows, the end tokens, the head's `confidence` and `chain`,
            /// whether copies are offered, what their lookup found for the first window. The tokens: those chosen,
            /// the end tokens, the lookup's.
            const arguments = 19;

            /// Rank 0: tells rank 1, then runs the rounds from `kept` rows chosen into `chosen`, the first asked
            /// `want` drafts. Returns how the reply ended; `kept` is the last window's.
            pub fn led(self: *Rounds, rounds: *draft_file.Decoder(H), drafter: draft_file.Drafter, kept: *usize, want: usize) !request_file.Finish {
                const pair = self.pair;
                const lead = self.lead.?;
                const h = &pair.head.?;
                std.debug.assert(lead.eos.len <= most_ends and !pair.lead.muted);
                if (self.keeping) |keeping| keeping.said = true;
                self.depth = rounds.depth();
                const found = try self.lookAhead(self.chosen[0..kept.*], pair.state.pos, want);
                const s = rounds.settings;
                const temperature: u64 = @bitCast(s.temperature);
                const top_p: u64 = @bitCast(s.top_p);
                const min_p: u64 = @bitCast(s.min_p);
                const told: [arguments]u32 = .{
                    @truncate(s.seed),         @truncate(s.seed >> 32),
                    @truncate(temperature),    @truncate(temperature >> 32),
                    s.top_k,                   @truncate(top_p),
                    @truncate(top_p >> 32),    @truncate(min_p),
                    @truncate(min_p >> 32),    @intCast(h.state.position.host),
                    @intCast(h.state.drafted), @intFromBool(h.full),
                    @intCast(want),            @intCast(kept.*),
                    @intCast(lead.eos.len),    @bitCast(h.confidence),
                    @bitCast(h.chain),         @intFromBool(self.copying),
                    @intCast(found.len),
                };
                var tokens: [kept_most + most_ends + draft_file.forward_max_rows]i32 = undefined;
                @memcpy(tokens[0..kept.*], self.chosen[0..kept.*]);
                for (tokens[kept.*..][0..lead.eos.len], lead.eos) |*token, end| token.* = @bitCast(end);
                for (tokens[kept.* + lead.eos.len ..][0..found.len], found) |*token, word| token.* = @bitCast(word);
                try pair.lead.tell(@backingInt(held_file.Op.decode), &told, tokens[0 .. kept.* + lead.eos.len + found.len]);
                pair.lead.muted = true;
                pair.b.settling = true;
                const before_kept = rounds.kept;
                rounds.kept = .{ .context = self, .window = keptLed };
                defer rounds.kept = before_kept;
                var asked = want;
                while (true) {
                    var drafts: [draft_file.forward_max_rows]i32 = undefined;
                    const n = try rounds.draft(drafter, self.chosen[0..kept.*], asked, &drafts);
                    kept.* = try rounds.round(self.chosen[kept.* - 1], drafts[0..n], self.chosen);
                    pair.seen.appendSliceAssumeCapacity(self.chosen[0..kept.*]);
                    if (self.ended) |one| return one;
                    asked = self.want;
                }
            }

            /// Rank 0's: what the copies' lookup finds for the window after `tokens`, the last of them at
            /// `position`, asked `want` drafts: asked now where the head will draft one at least (`Copied.ahead`),
            /// else not asked, as the draft would not ask it. Nothing where no copy is offered.
            fn lookAhead(self: *Rounds, tokens: []const i32, position: usize, want: usize) draft_file.Error(H)![]const u32 {
                if (!self.copying) return &.{};
                const h = &self.pair.head.?;
                const len = @min(want, self.depth, draft_file.forward_max_rows);
                // `mtp.Head.drafts` drafts none for a head full or that fills here, one at least else: a row all
                // minus infinity is a mask's, and a reply's rounds here hold none
                if (len == 0 or h.full or position > h.state.cache.capacity) return &.{};
                const found = self.pair.copies.?.ahead(tokens, position, len) catch |err| return @errorCast(err);
                return found[0..@min(found.len, len)];
            }

            /// `Decoder.Kept` on rank 0: the states inside the window kept, then the chosen tokens emitted as the
            /// loop in `reply` emits them, and all of it said before the commit.
            fn keptLed(context: *anyopaque, tokens: []const i32, rows_kept: usize) draft_file.Error(H)!void {
                const self: *Rounds = @ptrCast(@alignCast(context));
                const pair = self.pair;
                const lead = self.lead.?;
                var words: [Words.width]u32 = @splat(0);
                if (self.keeping) |keeping| {
                    keeping.recorded = 0;
                    try Keeping.kept(keeping, tokens, rows_kept);
                    for (keeping.takes[0..keeping.recorded], 0..) |one, i| {
                        words[6 + 2 * i] = if (one.drop) |pos| @intCast(pos + 1) else 0;
                        words[7 + 2 * i] = @intCast(one.rows);
                    }
                    words[5] = @intCast(keeping.recorded);
                }
                var emitted: usize = 0;
                var ended: ?request_file.Finish = null;
                if (halted(lead.watch)) ended = .cut else for (self.chosen[0..rows_kept]) |next| {
                    emitted += 1;
                    ended = pair.handed(next, lead.eos, lead.before, lead.max_tokens, lead.hang_at, lead.watch, lead.stopped) catch |err| return @errorCast(err);
                    if (ended != null) break;
                }
                if (ended == null and halted(lead.watch)) ended = .cut;
                self.ended = ended;
                self.want = if (ended == null) @min(lead.depth, lead.max_tokens - (pair.tokens.items.len - lead.before) - 1) else 0;
                words[0] = Words.state_kind;
                words[1] = @intFromBool(ended != null);
                words[2] = @intCast(rows_kept);
                words[3] = @intCast(emitted);
                words[4] = @intCast(self.want);
                words[10] = @intCast(tokens.len);
                for (words[11..][0..tokens.len], tokens) |*word, token| word.* = @bitCast(token);
                if (ended == null) {
                    const found = try self.lookAhead(self.chosen[0..rows_kept], pair.state.pos + rows_kept, self.want);
                    words[Words.found_at] = @intCast(found.len);
                    @memcpy(words[Words.found_at + 1 ..][0..found.len], found);
                }
                try pair.words.?.say(&words);
            }

            /// Rank 1, told `Op.decode`: the reply's rounds as rank 0 runs them, until its words say it ended;
            /// the tokens it emitted go to `tokens`, as `pick` would bring them.
            fn followed(pair: *Self, a: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void {
                if (a.len != arguments) return error.DraftFailed;
                const kept_first = a[13];
                const ends = a[14];
                const found = a[18];
                const h = &(pair.head orelse return outOfStep());
                if (kept_first == 0 or kept_first > pair.chosen.len or ends > most_ends or found > draft_file.forward_max_rows or kept_first + ends + found != tokens.len) return outOfStep();
                for (pair.ends[0..ends], tokens[kept_first..][0..ends]) |*end, token| end.* = @bitCast(token);
                const settings: sampling.Sampling = .{
                    .seed = @as(u64, a[0]) | @as(u64, a[1]) << 32,
                    .temperature = @bitCast(@as(u64, a[2]) | @as(u64, a[3]) << 32),
                    .top_k = a[4],
                    .top_p = @bitCast(@as(u64, a[5]) | @as(u64, a[6]) << 32),
                    .min_p = @bitCast(@as(u64, a[7]) | @as(u64, a[8]) << 32),
                };
                // the head as rank 0's stands: its pass at this position told last, with nothing chosen since
                h.settings = settings;
                h.ends = pair.ends[0..ends];
                h.fresh();
                h.confidence = @bitCast(a[15]);
                h.chain = @bitCast(a[16]);
                if (h.state.position.host != a[9]) {
                    try cuda.synchronize();
                    try h.state.position.set(a[9]);
                }
                h.state.drafted = a[10];
                h.full = a[11] != 0;
                const chosen = pair.chosen;
                @memcpy(chosen[0..kept_first], tokens[0..kept_first]);
                var rounds: draft_file.Decoder(H) = .{
                    .m = pair.m,
                    .state = &pair.state,
                    .b = &pair.b,
                    .graphs = if (pair.graphs) |*g| g else null,
                    .sampler = &pair.sampler,
                    .settings = settings,
                    .eos = pair.ends[0..ends],
                    .logits = pair.logits,
                };
                var self: Rounds = .{ .pair = pair, .chosen = chosen, .copying = a[17] != 0, .found_len = found };
                for (self.found[0..found], tokens[kept_first + ends ..]) |*word, token| word.* = @bitCast(token);
                rounds.kept = .{ .context = &self, .window = keptFollowed };
                pair.b.settling = true;
                defer pair.b.settling = false;
                const drafter = h.drafter();
                var kept: usize = kept_first;
                var asked: usize = a[12];
                var drafts: [draft_file.forward_max_rows]i32 = undefined;
                while (true) {
                    var n = try rounds.draft(drafter, chosen[0..kept], asked, &drafts);
                    // the copy rank 0 offers over the same drafts, from what its lookup found
                    if (self.found_len > 0) {
                        // rank 0 looks ahead only where its head drafts one at least, as this one does
                        if (n == 0 or self.found_len > @min(asked, rounds.depth())) return outOfStep();
                        if (copied.Copied.copyOf(self.found[0..self.found_len], pair.ends[0..ends], drafts[0..@min(asked, rounds.depth())])) |rows| n = rows;
                    }
                    kept = try rounds.round(chosen[kept - 1], drafts[0..n], chosen);
                    pair.own_rounds += 1;
                    if (self.ended != null) break;
                    asked = self.want;
                }
                _ = try rounds.draft(drafter, chosen[0..kept], 0, &drafts);
            }

            /// `Decoder.Kept` on rank 1: rank 0's words for the window, checked against it, and the states it
            /// says taken.
            fn keptFollowed(context: *anyopaque, tokens: []const i32, rows_kept: usize) draft_file.Error(H)!void {
                const self: *Rounds = @ptrCast(@alignCast(context));
                const pair = self.pair;
                var words: [Words.width]u32 = undefined;
                try pair.words.?.hear(&words);
                const rows = words[10];
                if (words[0] != Words.state_kind or words[2] != rows_kept or rows != tokens.len or words[3] > rows_kept or words[5] > 2) return outOfStep();
                const window: []align(1) const i32 = @ptrCast(words[11..][0..rows]);
                for (window, tokens) |word, token| if (word != token) return outOfStep();
                for (0..words[5]) |i| {
                    const drop = words[6 + 2 * i];
                    try Take.run(.{ .drop = if (drop == 0) null else drop - 1, .rows = words[7 + 2 * i] }, pair, tokens);
                }
                try pair.tokens.appendSlice(pair.gpa, self.chosen[0..words[3]]);
                self.ended = if (words[1] != 0) .stop else null;
                self.want = words[4];
                self.found_len = words[Words.found_at];
                if (self.found_len > @min(self.want, draft_file.forward_max_rows)) return outOfStep();
                for (self.found[0..self.found_len], words[Words.found_at + 1 ..][0..self.found_len]) |*word, one| word.* = one;
            }
        };

        /// Rank 1, told `Op.decode` (`mirror.zig`): `Rounds.followed`.
        pub fn decodeFollowed(context: *anyopaque, a: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void {
            return Rounds.followed(@ptrCast(@alignCast(context)), a, tokens);
        }
    };
}
