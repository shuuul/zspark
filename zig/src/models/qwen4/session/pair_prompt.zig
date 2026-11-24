//! A prompt told once to the rank that follows (spec 097), moved out of `pair.zig` (spec 104 WS-06): rank 0
//! builds the prompt's frames (`promptFrame`, `toldPrompt`) and rank 1 runs the prompt from them
//! (`promptFollowed`). `pair.zig`'s `Of(held)` names them.
const cuda = @import("../../../cuda/cuda.zig");
const draft_file = @import("../../../decode/draft.zig");
const sampling = @import("../../../decode/sampling.zig");
const lead_file = @import("../../../ranks/lead.zig");
const held_file = @import("../ranks_forward.zig");
const book_file = @import("../../../decode/book.zig");
const pair_file = @import("pair.zig");
const pair_rounds = @import("pair_rounds.zig");

pub fn Of(comptime held: usize) type {
    const H = held_file.Held(held);
    const Self = pair_file.Of(held);
    return struct {
        const Rounds = pair_rounds.Of(held).Rounds;
        const Take = Self.Take;
        const outOfStep = Self.outOfStep;

        /// The frame of a prompt told once that speaks from its token `at` on: the arguments, its words, and
        /// the token the next frame starts at, the chunk as wide as `chunk_most` lets it be (`PromptFrame.most` as
        /// told, less in a check, which splits a short prompt to run its frames apart).
        pub fn promptFrame(self: *Self, start: book_file.Resumed, drops: []const usize, planned: []const usize, due: []const usize, prompt: []const i32, at: usize, chunk_most: usize, settings: sampling.Sampling, eos: []const u32, absorbs: bool, frame: *PromptFrame) !usize {
            const words: []i32 = &frame.words;
            var said: usize = 0;
            for (drops) |pos| {
                words[said] = @intCast(pos);
                said += 1;
            }
            for (planned) |pos| {
                words[said] = @intCast(pos);
                said += 1;
            }
            const gone = said;
            for (due) |pos| {
                words[said] = @intCast(pos);
                said += 1;
            }
            const taken = said - gone;
            const chunk = @min(prompt.len - at, chunk_most - said - eos.len);
            for (prompt[at..][0..chunk]) |token| {
                words[said] = token;
                said += 1;
            }
            for (eos) |end| {
                words[said] = @bitCast(end);
                said += 1;
            }
            const temperature: u64 = @bitCast(settings.temperature);
            const top_p: u64 = @bitCast(settings.top_p);
            const min_p: u64 = @bitCast(settings.min_p);
            frame.arguments = .{
                @truncate(settings.seed),                                                               @truncate(settings.seed >> 32),
                @truncate(temperature),                                                                 @truncate(temperature >> 32),
                settings.top_k,                                                                         @truncate(top_p),
                @truncate(top_p >> 32),                                                                 @truncate(min_p),
                @truncate(min_p >> 32),
                @intCast(switch (start.to) {
                    .stay => self.state.pos,
                    .back => |pos| pos,
                    .reset => 0,
                }),
                // a rewind or a reset trims the head's chained drafts; DFlash2 in its place says none of it
                0,                                                                                      if (self.head) |h| @intFromBool(start.to != .reset and h.full) else 0,
                if (self.head) |h| @bitCast(h.confidence) else 0,                                       if (self.head) |h| @bitCast(h.chain) else 0,
                if (!absorbs) Absorbs.none else if (self.head != null) Absorbs.head else Absorbs.flash, @intCast(prompt.len),
                switch (start.to) {
                    .stay => @as(u32, 0),
                    .reset => 1,
                    .back => |pos| @intCast(pos + 1),
                },
                @intCast(gone),                                                                         @intCast(taken),
                @intCast(eos.len),
            };
            frame.said = said;
            return at + chunk;
        }

        /// Tells the prompt's frames (`Op.prompt`), as many as its tokens need. The lead must be clear: a
        /// muted `tell` sends nothing, and the frames are what rank 1 runs the prompt from.
        pub fn toldPrompt(self: *Self, start: book_file.Resumed, drops: []const usize, planned: []const usize, due: []const usize, prompt: []const i32, settings: sampling.Sampling, eos: []const u32, absorbs: bool) !void {
            var frame: PromptFrame = undefined;
            var at: usize = 0;
            while (at < prompt.len) {
                at = try self.promptFrame(start, drops, planned, due, prompt, at, PromptFrame.most, settings, eos, absorbs, &frame);
                try held_file.told(self.m.lead, .prompt, &frame.arguments, frame.words[0..frame.said]);
            }
        }

        /// A prompt told once (spec 097), as a reply's rounds are (`Rounds`): rank 0 tells rank 1 the prompt's
        /// frames (`Op.prompt`) before it runs, and runs the prompt's windows with the lead muted; rank 1 runs
        /// them from the frames (`promptFollowed`), its own head drafting between the windows as rank 0's does.
        /// The frames carry what the windows would be told per pass — the sampling, the head as the prompt
        /// finds it, the tokens that end the reply, where the prompt goes on from, the states that go for it
        /// and the states its windows take within themselves, all known once the prompt is planned — and the
        /// prompt's tokens, in frames as wide as the channel lets them be (`Lead.widest`). The words of a round
        /// (`Words`) stay the reply's rounds': nothing per window is left to say of a prompt.
        pub const Prompt = struct {
            /// Every frame's arguments: the sampling (`seed`, `temperature`, `top_k`, `top_p`, `min_p`, nine
            /// words), the head's position as the prompt finds it (a rewind or a reset has said it), its
            /// chained drafts (0: a rewind or a reset trims them) and fullness, its `confidence` and `chain`,
            /// whether the head drafts between the windows, the prompt's whole length, where the prompt goes
            /// on from (0 where the state stands, 1 emptied, else the state's position + 1), the states that
            /// go before it, the states its windows take within themselves, and the tokens that end the
            /// reply.
            const arguments = 20;
        };

        /// Who absorbs a told prompt's windows between them on rank 1 (a frame's argument 14).
        const Absorbs = struct {
            const none: u32 = 0;
            /// The MTP head, split over the ranks.
            const head: u32 = 1;
            /// The pair's DFlash2 drafter (`Of.flash`): its updates cross nothing.
            const flash: u32 = 2;
        };

        /// One frame of a prompt told once (`promptFrame` builds them, `promptFollowed` runs them).
        pub const PromptFrame = struct {
            arguments: [Prompt.arguments]u32,
            /// The frame's words: the states that go, the states the prompt's windows take within themselves,
            /// the prompt's next tokens and the tokens that end the reply.
            words: [most]i32,
            said: usize = 0,

            /// A frame's words beside its kind and arguments, in the channel's widest frame.
            pub const most = lead_file.Lead.widest - 1 - Prompt.arguments;
        };

        /// Rank 1, told `Op.prompt` (`mirror.zig`): a frame of the prompt told once, as `promptFrame` built
        /// it. The frames are appended until the prompt is whole, and the last frame's arrival runs it: the
        /// states that go, where the state stands for the prompt, the head as rank 0's stands, then the
        /// prompt's windows — the states its frames say are taken within them taken at the same points
        /// (`Prompted`), the head drafting between the windows as rank 0's does. What rank 1 chooses after the
        /// prompt's last row is its own: rank 0 says the token its reply goes on from, with its rounds
        /// (`Rounds`) or as `pick`.
        pub fn promptFollowed(pair: *Self, a: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void {
            const total = a[15];
            const drops = a[17];
            const taken = a[18];
            const ends = a[19];
            if (total == 0 or total > pair.capacity or a[16] > pair.capacity + 1 or drops + taken + ends > tokens.len or ends > Rounds.most_ends) return outOfStep();
            try pair.prompt_told.appendSlice(pair.gpa, tokens[drops + taken ..][0 .. tokens.len - drops - taken - ends]);
            if (pair.prompt_told.items.len < total) return;
            if (pair.prompt_told.items.len > total) return outOfStep();
            const prompt = pair.prompt_told.items;
            defer pair.prompt_told.clearRetainingCapacity();
            const absorbs = a[14];
            if (absorbs > Absorbs.flash) return outOfStep();
            for (pair.ends[0..ends], tokens[tokens.len - ends ..]) |*end, token| end.* = @bitCast(token);
            const settings: sampling.Sampling = .{
                .seed = @as(u64, a[0]) | @as(u64, a[1]) << 32,
                .temperature = @bitCast(@as(u64, a[2]) | @as(u64, a[3]) << 32),
                .top_k = a[4],
                .top_p = @bitCast(@as(u64, a[5]) | @as(u64, a[6]) << 32),
                .min_p = @bitCast(@as(u64, a[7]) | @as(u64, a[8]) << 32),
            };
            var drafter: ?draft_file.Drafter = null;
            if (pair.head) |*h| {
                // the head as rank 0's stands for the prompt: a rewind or a reset has said its position, and a
                // pass carries it after
                h.settings = settings;
                h.ends = pair.ends[0..ends];
                h.fresh();
                h.confidence = @bitCast(a[12]);
                h.chain = @bitCast(a[13]);
                if (h.state.position.host != a[9]) {
                    try cuda.synchronize();
                    try h.state.position.set(a[9]);
                }
                h.state.drafted = a[10];
                h.full = a[11] != 0;
                if (absorbs == Absorbs.head) drafter = h.drafter();
            } else if (absorbs == Absorbs.head) return outOfStep();
            if (absorbs == Absorbs.flash) {
                const f = pair.flash orelse return outOfStep();
                f.settings = settings;
                f.ends = pair.ends[0..ends];
                f.full = false;
                drafter = f.drafter();
            }
            // Where the state stands for the prompt, before the states that go: rank 0 drops the plan's after
            // it went back (`pair.zig` `reply`), and the plan may drop the very state it went back to, as a
            // long prompt's does with the budget full. The start's drops all stand above where it goes back.
            switch (a[16]) {
                0 => {},
                1 => {
                    for (&pair.state.ranks) |*one| try one.reset();
                    pair.state.pos = 0;
                },
                else => H.back(pair.m, &pair.state, &pair.shelf, a[16] - 1) catch |err| return Take.parted(err),
            }
            for (tokens[0..drops]) |word| {
                H.drop(pair.m, &pair.shelf, @intCast(@as(u32, @bitCast(word)))) catch |err| return Take.parted(err);
            }
            // DFlash2's context ends where the state now stands, as rank 0's was set before its windows
            if (absorbs == Absorbs.flash) {
                const f = pair.flash.?;
                if (f.pair.context_end != pair.state.pos) try f.pair.setEnd(pair.state.pos);
            }
            const due = try pair.gpa.alloc(usize, taken);
            defer pair.gpa.free(due);
            for (due, tokens[drops..][0..taken]) |*at, word| {
                at.* = @intCast(@as(u32, @bitCast(word)));
                if (at.* > prompt.len) return outOfStep();
            }
            const chosen = pair.chosen;
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
            var self: Prompted = .{ .pair = pair, .due = due, .ahead = .{ .prompt = prompt, .rounds = &rounds, .room = chosen.len } };
            rounds.within = .{ .context = &self, .window = Prompted.takenFollowed, .before = Prompted.before };
            defer rounds.within = null;
            _ = try rounds.prompt(drafter, prompt, chosen);
        }

        /// The states a told prompt takes within its windows (`promptFollowed`): the positions rank 0
        /// planned, each taken inside the window that runs over it, as `Keeping.window` takes them where the
        /// prompt is told per pass.
        const Prompted = struct {
            pair: *Self,
            due: []const usize,
            ahead: Self.Ahead,

            fn before(context: *anyopaque, run: []const i32) draft_file.Error(H)!void {
                const self: *Prompted = @ptrCast(@alignCast(context));
                self.ahead.before(self.pair, run);
                try self.pair.askWithin(self.due, run);
            }

            fn takenFollowed(context: *anyopaque, run: []const i32, last: bool) draft_file.Error(H)!void {
                _ = last;
                const self: *Prompted = @ptrCast(@alignCast(context));
                const from = self.pair.state.pos;
                for (self.due) |at| {
                    if (at <= from or at > from + run.len) continue;
                    try Take.run(.{ .drop = null, .rows = at - from }, self.pair, run);
                }
                H.dropAsked(&self.pair.shelf, &self.pair.asked);
            }
        };

        /// Rank 1, told `Op.prompt` (`mirror.zig`): `promptFollowed`.
        pub fn promptTold(context: *anyopaque, a: []align(1) const u32, tokens: []const i32) draft_file.Error(H)!void {
            return promptFollowed(@ptrCast(@alignCast(context)), a, tokens);
        }
    };
}
