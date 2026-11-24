//! Several replies of GLM-5.3-Flash at once (spec 068 WS-04): a slot a stream, and a round that runs every
//! live stream's rows as one window (`slots.zig`), so the ranks exchange once a round whatever the streams.
//! TensorFold's host serves GLM's streams the same way; where it gives no rule (which slot a prompt takes,
//! how a round's rows are shared) the choice is zspark's and changes no token.
//!
//! A stream's reply is the one `session.zig` gives it alone, token for token, greedy and sampled, drafted or
//! not: its rows' bits are its sequence's alone, each token is chosen by the key of its own place
//! (`draft.Chooser`, the one place a draft is accepted), and its constraint is its own. What differs beside
//! others is only how its prompt is cut into windows and how deep a window is drafted, which change no row.
//!
//! A round: each replying stream's newest token and its drafts, scored; each prompt's next rows from what is
//! left of the window, the last of them scored; one forward; a choice a stream on rank 0's host; one commit;
//! then the drafter, for every stream together. The MTP layer takes the streams' rows kept in one step and
//! every running chain's draft in one more, each on its stream's own entries (`mtp_seats.zig`); DFlash2
//! runs the streams' blocks as one pass, each over its own context (`decode/dflash2/seats.zig`). So a round's
//! exchanges for its drafts are a chain's depth or a block's eleven gathers, whatever the streams.
//!
//! The decoding thread's alone: nothing here locks.

const std = @import("std");
const Allocator = std.mem.Allocator;
const grammar = @import("../../../text/grammar.zig");
const request_file = @import("../../../server/request.zig");
const masks_file = @import("../../../decode/masks.zig");
const prefix_wait = @import("../../../decode/prefix_wait.zig");
const sampling = @import("../../../decode/sampling.zig");
const shared = @import("../../../decode/draft.zig");
const cuda = @import("../../../cuda/cuda.zig");
const glm5_draft = @import("../draft.zig");
const held_file = @import("../layers/ranks_forward.zig");
const lead_file = @import("../../../ranks/lead.zig");
const kda = @import("../layers/kda.zig");
const model = @import("../layers/model.zig");
const flash_file = @import("../dflash2/drafter.zig");
const phases = @import("../layers/forward.zig");
const flash_pass = @import("../dflash2/pass.zig");
const mtp = @import("../mtp/mtp.zig");
const seats_file = @import("../mtp/mtp_seats.zig");
const slots_file = @import("slots.zig");
const scheduler_file = @import("../../../server/scheduler.zig");

/// A reply asked of the hall. `prompt` and `eos` are copied; a constraint is the caller's until the stream
/// has left.
pub const Request = struct {
    prompt: []const i32,
    settings: sampling.Sampling,
    /// The tokens that end the reply; none, and it runs to its length.
    eos: []const u32,
    /// The reply's tokens at most, the one that ends it among them; with the prompt within the capacity.
    max_tokens: usize,
    /// Whether the hall's drafter drafts for it.
    drafts: bool = true,
    /// From an empty sequence whatever the slots hold.
    fresh: bool = false,
    constraint: ?grammar.Constraint = null,
    gate: ?i32 = null,
    /// Set where the request waited for another prompt's fill (spec 152 WS-06): the decoding thread's.
    wait: prefix_wait.Waiting = .{},
};

pub const Finish = request_file.Finish;

pub const Ended = struct {
    finish: Finish,
    stats: shared.Stats,
};

pub const Admitted = struct {
    slot: usize,
    /// The prompt's tokens the slot's caches held.
    cached: usize,
};

pub fn HallOf(comptime held: usize) type {
    return struct {
        const Self = @This();
        pub const H = held_file.Held(held);
        pub const Slots = slots_file.SlotsOf(held);

        /// A round's scored rows as `draft.Chooser`'s model: row `row` is the round's scored row.
        const Rows = struct {
            pub const Buffers = struct { slots: *const Slots, b: *const H.Buffers };
            pub const Error = held_file.Error;

            pub fn logitsRow(rows: Buffers, row: usize, out: []u16) Error!void {
                return rows.slots.logitsRow(rows.b.*, row, out);
            }
        };
        const Chooser = shared.Chooser(Rows);

        const Stream = struct {
            phase: enum { idle, prompt, reply } = .idle,
            /// Live: the prompt and the reply so far. Idle: the tokens the slot's caches hold.
            seq: std.ArrayList(i32) = .empty,
            /// The prompt's length, and its tokens run.
            prompt: usize = 0,
            fed: usize = 0,
            settings: sampling.Sampling = .{ .seed = 0, .temperature = 0 },
            eos: std.ArrayList(u32) = .empty,
            max_tokens: usize = 0,
            made: usize = 0,
            /// The drafter drafts for it, and has absorbed every position the caches hold.
            drafted: bool = false,
            in_step: bool = true,
            /// The MTP layer's `full`.
            full: bool = false,
            /// The drafts its next window carries.
            drafts: [shared.forward_max_rows]i32 = undefined,
            pending: usize = 0,
            /// Of `seq`, where its fill is: `fed` while the prompt runs, `prompt` once its state serves a
            /// waiter's cut (`leaders`).
            reached: usize = 0,
            /// The one cut its fill serves a waiter from, `prompt - 1`: a slice of it stays valid while the
            /// stream lives (`leaders`).
            cut: [1]usize = .{0},
            /// A constraint's mask of one row, and the constraint where the reply has one.
            mask: []u32,
            masks: ?masks_file.Masks = null,
            stats: shared.Stats = .{},
        };

        gpa: Allocator,
        m: H.Model,
        b: *H.Buffers,
        slots: Slots,
        streams: []Stream,
        sampler: *sampling.Sampler,
        /// One row's logits on the host.
        logits: []u16,
        /// The most drafts a window carries: what leaves half of a round's rows to the prompts.
        depth: usize,
        /// The rows of a round at most.
        most: usize,
        /// The MTP layer where it drafts, the session's, and each slot's entries of it.
        head: ?*mtp.HeadOf(held) = null,
        seats: ?mtp.SeatsOf(held) = null,
        /// The DFlash2 drafter where it drafts in the layer's place, and each slot's context of it.
        flash: ?*flash_file.DrafterOf(held) = null,
        flash_seats: ?flash_pass.SeatsOf(held) = null,
        /// The rounds as captured graphs, where the caller set them: the loopback's serve does, the
        /// two-rank's does not (a capture over NCCL must be told to the rank that follows, as the serial
        /// round's is) and neither does an eager one.
        graphs: ?Slots.Graphs = null,
        /// How long a request waits for a fill that shares a cut of its prompt (`decode/prefix_wait.zig`;
        /// ZSPARK_PREFIX_WAIT_MS, 0 off), and the fills the wait sees.
        prefix_wait: prefix_wait.Wait = .{},
        pending: std.ArrayList(Pending) = .empty,
        fillers: std.ArrayList(prefix_wait.Filler) = .empty,
        /// A round's parts, whose slot each is, its tokens, the rows kept, and a window's choices.
        parts: []slots_file.Part,
        tokens: []i32,
        keeps: []usize,
        chosen: []i32,
        /// The MTP layer's drafts of a round, asked together (`mtp_seats.zig`).
        jobs: []glm5_draft.Job,
        rounds: u64 = 0,

        /// The most drafts a window of `count` streams carries over windows of `rows` rows.
        pub fn depthOf(rows: usize, count: usize) usize {
            return @min(shared.forward_max_rows, mostOf(rows, count) / (2 * count)) -| 1;
        }

        /// The rows of a round at most: the window's, and what a frame told to the other rank holds.
        fn mostOf(rows: usize, count: usize) usize {
            return @min(rows, lead_file.Lead.most - 1 - seats_file.told(count));
        }

        /// What drafts a hall's replies: nothing, the MTP layer, or the DFlash2 drafter, whose pair launches its
        /// passes from the host and whose contexts were made with `gpa`.
        pub const Drafting = union(enum) { none, mtp: *mtp.HeadOf(held), flash: *flash_file.DrafterOf(held) };

        /// The device bytes a slot takes on a rank for `capacity` positions: the sequence's states and latents,
        /// and the drafter's entries of it.
        pub fn slotBytes(rank: model.Model, capacity: usize, drafting: Drafting) usize {
            return model.Seq.bytes(rank, capacity) + switch (drafting) {
                .none => 0,
                .mtp => mtp.SeatsOf(held).bytes(rank, capacity),
                .flash => |d| flash_pass.SeatsOf(held).bytes(d.pair),
            };
        }

        /// Whether windows of `rows` rows hold `count` streams' rounds: two rows a stream at least.
        pub fn holds(rows: usize, count: usize) bool {
            return count >= 1 and seats_file.told(count) + 1 < lead_file.Lead.most and mostOf(rows, count) >= 2 * count;
        }

        /// `count` sequences as a hall of them holds them: rank 1's, which follows rank 0's hall.
        pub fn slotsFor(gpa: Allocator, m: H.Model, count: usize, capacity: usize) !Slots {
            std.debug.assert(holds(m.max_rows, count));
            return .init(gpa, m, count, capacity, @min(m.max_rows, count * (depthOf(m.max_rows, count) + 1)));
        }

        /// `count` slots of `capacity` positions over `m` and its buffers `b`, drafted by `drafting`.
        /// `sampler` and `logits` are the caller's, used by nothing else while the hall decodes.
        pub fn init(gpa: Allocator, m: H.Model, b: *H.Buffers, count: usize, capacity: usize, vocab: usize, sampler: *sampling.Sampler, logits: []u16, drafting: Drafting) !Self {
            std.debug.assert(holds(m.max_rows, count) and logits.len == vocab);
            const depth = depthOf(m.max_rows, count);
            var slots: Slots = try slotsFor(gpa, m, count, capacity);
            errdefer slots.deinit();
            const head: ?*mtp.HeadOf(held) = if (drafting == .mtp) drafting.mtp else null;
            const flash: ?*flash_file.DrafterOf(held) = if (drafting == .flash) drafting.flash else null;
            var seats: ?mtp.SeatsOf(held) = if (head != null) try .init(gpa, m, count, capacity) else null;
            errdefer if (seats) |*one| one.deinit();
            var flash_seats: ?flash_pass.SeatsOf(held) = if (flash) |d| try .init(gpa, d.pair, count) else null;
            errdefer if (flash_seats) |*one| one.deinit();
            const streams = try gpa.alloc(Stream, count);
            errdefer gpa.free(streams);
            var done: usize = 0;
            errdefer for (streams[0..done]) |one| gpa.free(one.mask);
            for (streams) |*one| {
                one.* = .{ .mask = try gpa.alloc(u32, (vocab + 31) / 32) };
                done += 1;
            }
            const parts = try gpa.alloc(slots_file.Part, count);
            errdefer gpa.free(parts);
            const tokens = try gpa.alloc(i32, m.max_rows);
            errdefer gpa.free(tokens);
            const keeps = try gpa.alloc(usize, count);
            errdefer gpa.free(keeps);
            const chosen = try gpa.alloc(i32, m.max_rows);
            errdefer gpa.free(chosen);
            const jobs = try gpa.alloc(glm5_draft.Job, count);
            return .{ .gpa = gpa, .m = m, .b = b, .prefix_wait = .fromEnv(), .slots = slots, .streams = streams, .sampler = sampler, .logits = logits, .depth = depth, .most = mostOf(m.max_rows, count), .head = head, .seats = seats, .flash = flash, .flash_seats = flash_seats, .parts = parts, .tokens = tokens, .keeps = keeps, .chosen = chosen, .jobs = jobs };
        }

        pub fn deinit(self: *Self) void {
            const gpa = self.gpa;
            if (self.graphs) |*graphs| graphs.deinit();
            for (self.streams) |*one| {
                one.seq.deinit(gpa);
                one.eos.deinit(gpa);
                gpa.free(one.mask);
            }
            gpa.free(self.streams);
            gpa.free(self.parts);
            gpa.free(self.tokens);
            gpa.free(self.keeps);
            gpa.free(self.chosen);
            self.pending.deinit(self.gpa);
            self.fillers.deinit(self.gpa);
            gpa.free(self.jobs);
            if (self.seats) |*one| one.deinit();
            if (self.flash_seats) |*one| one.deinit();
            self.slots.deinit();
            self.* = undefined;
        }

        /// A queued request's wait for another prompt's fill: its prompt names it (the caller's slice's
        /// address is the queue's to it; `Request.wait` marks one the caller tracked itself).
        const Pending = struct {
            prompt: [*]const i32,
            prompt_len: usize,
            waiting: prefix_wait.Waiting,
        };

        /// Streams live, and how many waits the prefix wait counted (, ; a test reads
        /// them).
        pub fn live(self: Self) usize {
            var count: usize = 0;
            for (self.streams) |one| count += @intFromBool(one.phase != .idle);
            return count;
        }

        /// Slot `to` takes slot `from`'s committed state at `n` positions: every rank's KDA chains and conv
        /// windows whole (a fill's standing state), and the DSA latents, keys, gates and pooled keys under
        /// the waiter's own extents (`dsa.Cache.copyPrefix`). `n` is a position the fill reached, so what
        /// the copies read is committed.
        fn copyPrefix(self: *Self, from: usize, to: usize, n: usize) !void {
            for (&self.slots.seqs) |seqs| {
                const src = &seqs[from];
                const dst = &seqs[to];
                if (src.kda) |*kept| {
                    try dst.kda.?.conv.copyLaunched(kept.conv);
                    try dst.kda.?.rec.copyLaunched(kept.rec);
                    dst.kda.?.cur = kept.cur;
                }
                if (dst.dsa) |*cache| try cache.copyPrefix(&(src.dsa orelse unreachable), n);
                dst.pos = n;
            }
        }

        fn cachedOf(one: Stream, prompt: []const i32, drafted: bool) usize {
            const known = one.seq.items;
            if (known.len == 0 or known.len >= prompt.len or (drafted and !one.in_step)) return 0;
            return if (std.mem.eql(i32, prompt[0..known.len], known)) known.len else 0;
        }

        /// A free slot for the request, or null where none is: the one whose caches hold the most of the
        /// prompt, else the one that holds the least of anything. Its prompt starts with the next round.
        pub fn admit(self: *Self, r: Request) !?Admitted {
            std.debug.assert(r.prompt.len >= 1 and r.max_tokens >= 1 and r.prompt.len + r.max_tokens <= self.slots.capacity);
            const drafted = (self.head != null or self.flash != null) and r.drafts;
            var best: ?usize = null;
            var cached: usize = 0;
            for (self.streams, 0..) |one, slot| {
                if (one.phase != .idle) continue;
                const has = if (r.fresh) 0 else cachedOf(one, r.prompt, drafted);
                if (best == null or has > cached or (has == cached and cached == 0 and one.seq.items.len < self.streams[best.?].seq.items.len)) {
                    best = slot;
                    cached = has;
                }
            }
            const slot = best orelse return null;
            const one = &self.streams[slot];
            // A request that waited for a fill holds it still only while the fill lives short of the cut;
            // `pending` keeps the record, the `Request` here a copy's (`prefix_wait`, spec 152 WS-06). Its
            // `have` is what a slot's own caches hold (`cached`); the wait adds nothing below it.
            const now = self.nowNs();
            const found = for (self.pending.items, 0..) |*w, i| {
                if (w.prompt == r.prompt.ptr and w.prompt_len == r.prompt.len) break i;
            } else null;
            if (found) |i| {
                const waiting = self.pending.items[i].waiting;
                if (prefix_wait.holds(&self.prefix_wait, &waiting, try self.leaders(), now)) return null;
                // the wait is over: the fill stands at the cut or is gone. Where its slot still holds the
                // prompt's rows the waiter takes them; a fill that left takes the slot's own `cached`.
                _ = self.pending.swapRemove(i);
                if (waiting.leader) |context| {
                    const leader: *Stream = @ptrCast(@alignCast(@constCast(context)));
                    const from = (@intFromPtr(leader) - @intFromPtr(self.streams.ptr)) / @sizeOf(Stream);
                    if (from < self.streams.len and self.streams[from].phase != .idle and waiting.at <= self.slots.pos(from) and waiting.at > cached and waiting.at <= r.prompt.len) {
                        try self.copyPrefix(from, slot, waiting.at);
                        cached = waiting.at;
                    }
                }
            } else if (!r.wait.waited) {
                if (prefix_wait.pick(&self.prefix_wait, try self.leaders(), r.prompt, cached, now)) |waiting| {
                    try self.pending.append(self.gpa, .{ .prompt = r.prompt.ptr, .prompt_len = r.prompt.len, .waiting = waiting });
                    self.prefix_wait.waits += 1;
                    return null;
                }
            }
            // a constraint that cannot start refuses the request with the slot as it was
            const masks: ?masks_file.Masks = if (r.constraint) |constraint| try masks_file.Masks.init(constraint, one.mask, r.gate) else null;
            // until the reply is whole the caches are no prompt's prefix
            one.seq.clearRetainingCapacity();
            try one.seq.appendSlice(self.gpa, r.prompt);
            one.eos.clearRetainingCapacity();
            try one.eos.appendSlice(self.gpa, r.eos);
            if (cached == 0) {
                try self.slots.reset(self.m, slot);
                one.in_step = true;
                // the drafter's seat starts again with the sequence's first row
                one.full = false;
            }
            if (!drafted) one.in_step = false;
            std.debug.assert(self.slots.pos(slot) == cached);
            one.masks = masks;
            one.prompt = r.prompt.len;
            one.fed = cached;
            one.settings = r.settings;
            one.max_tokens = r.max_tokens;
            one.made = 0;
            one.drafted = drafted;
            one.pending = 0;
            one.stats = .{};
            one.reached = cached;
            one.cut = .{r.prompt.len - 1};
            one.phase = .prompt;
            return .{ .slot = slot, .cached = cached };
        }

        /// The prompt fills now running, as `prefix_wait`'s fillers: a prompt stream's `seq` is its
        /// prompt, `fed` where it stands, `marks` where it keeps a state.
        fn leaders(self: *Self) ![]prefix_wait.Filler {
            self.fillers.clearRetainingCapacity();
            for (self.streams) |*one| {
                if (one.phase != .prompt) continue;
                // the fill's end is the one cut it serves a waiter from: it is the state's position on the
                // sequence, the slot's caches holding the tokens before it
                try self.fillers.append(self.gpa, .{ .context = one, .prompt = one.seq.items[0..one.prompt], .reached = one.reached, .marks = &one.cut, .id = @intFromPtr(one.seq.items.ptr) });
            }
            return self.fillers.items;
        }

        /// The wait's clock: the host's monotonic nanoseconds, as `pool.zig`'s `nowNs` reads them.
        fn nowNs(self: *Self) i96 {
            _ = self;
            var t: std.c.timespec = undefined;
            if (std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &t) != 0) return 0;
            return @as(i96, @intCast(t.sec)) * std.time.ns_per_s + @as(i96, @intCast(t.nsec));
        }

        /// Idle slots retain one reusable sequence each, in their allocated KDA and DSA storage.
        pub fn cacheSnapshot(self: *const Self) scheduler_file.Cache.Snapshot {
            var states: usize = 0;
            for (self.streams) |one| if (one.phase == .idle and one.seq.items.len > 0) {
                states += 1;
            };
            var each: usize = 0;
            for (self.m.ranks) |rank| each += model.Seq.bytes(rank, self.slots.capacity);
            return .{ .states = states, .bytes = states * each };
        }

        /// The slot's stream leaves, wherever it stands: the slot keeps what its caches hold.
        pub fn retire(self: *Self, slot: usize) void {
            const one = &self.streams[slot];
            if (one.phase == .idle) return;
            one.phase = .idle;
            one.masks = null;
            one.seq.shrinkRetainingCapacity(self.slots.pos(slot));
        }

        fn chooser(self: *Self, one: *Stream) Chooser {
            return .{ .sampler = self.sampler, .settings = one.settings, .eos = one.eos.items, .logits = self.logits, .masks = if (one.masks) |*masks| masks else null };
        }

        /// One round over the live streams. `sink.tokens(slot, tokens)` is given a stream's new tokens, the
        /// one that ends it among them, and `sink.ended(slot, Ended)` that it ended: the caller then retires
        /// it. Returns whether anything ran.
        pub fn round(self: *Self, sink: anytype) !bool {
            var count: usize = 0;
            var rows: usize = 0;
            var prompts: usize = 0;
            // a reply's window: its newest token, not yet committed, and the drafts after it
            for (self.streams, 0..) |*one, slot| switch (one.phase) {
                .idle => {},
                .prompt => prompts += 1,
                .reply => {
                    const given = one.drafts[0..one.pending];
                    const drafts = if (one.masks) |*masks| given[0..try masks.window(given)] else given;
                    self.tokens[rows] = one.seq.items[one.seq.items.len - 1];
                    @memcpy(self.tokens[rows + 1 ..][0..drafts.len], drafts);
                    self.parts[count] = .{ .slot = slot, .rows = 1 + drafts.len, .score = .all };
                    rows += 1 + drafts.len;
                    count += 1;
                },
            };
            if (count == 0 and prompts == 0) return false;
            // the prompts share the rows left, a prompt's last rows scored at their end; the parts of bf16 rows
            // before the quantized ones (spec 115, `slots.formFrom`), so that each part takes its own rows' form
            if (prompts > 0) {
                const each = (self.most - rows) / prompts;
                std.debug.assert(each >= 1);
                for ([_]bool{ false, true }) |quantized| for (self.streams, 0..) |*one, slot| {
                    if (one.phase != .prompt) continue;
                    // a part before a prompt's last ends at a KDA chunk's start (spec 116 WS-06), as `promptRows`,
                    // or where none is within its share, runs the recurrent chain. A waiter's cut caps it:
                    // its state at the cut is copied to the waiter's slot before the fill runs past it.
                    var left = one.prompt - one.fed;
                    for (self.pending.items) |*w| {
                        if (w.waiting.leader == @as(?*const anyopaque, @ptrCast(one)) and w.waiting.at > one.fed) left = @min(left, w.waiting.at - one.fed);
                    }
                    const cut = shared.promptCut(one.fed, each, left, kda.chunk);
                    const take = if (cut < left and kda.chunks(cut) and (one.fed + cut) % kda.chunk != 0) kda.recurrent_rows else cut;
                    if ((take > phases.bf16_rows) != quantized) continue;
                    @memcpy(self.tokens[rows..][0..take], one.seq.items[one.fed..][0..take]);
                    self.parts[count] = .{ .slot = slot, .rows = take, .score = if (take == one.prompt - one.fed) .last else .none };
                    rows += take;
                    count += 1;
                };
            }
            const parts = self.parts[0..count];
            if (self.graphs) |*graphs|
                try graphs.forward(&self.slots, self.m, self.b, parts, self.tokens[0..rows])
            else
                try self.slots.forward(self.m, self.b, parts, self.tokens[0..rows]);
            self.rounds += 1;

            // each stream's choices, by the key of each one's own place
            var at: usize = 0;
            for (parts, self.keeps[0..count], 0..) |part, *keep, index| {
                const one = &self.streams[part.slot];
                const window = self.tokens[at..][0..part.rows];
                const chosen = self.chosen[at..][0..part.rows];
                at += part.rows;
                keep.* = part.rows;
                switch (part.score) {
                    .none => {},
                    .last => {
                        try self.slots.logitsRow(self.b.*, self.slots.scoredAt(index), self.logits);
                        chosen[part.rows - 1] = try self.chooser(one).first(one.prompt);
                    },
                    .all => {
                        keep.* = (try self.chooser(one).accept(.{ .slots = &self.slots, .b = self.b }, self.slots.scoredAt(index), self.slots.pos(part.slot), window, chosen)).keep;
                        if (one.masks) |*masks| try masks.commit(keep.*, chosen[keep.* - 1]);
                        one.stats.rounds += 1;
                        one.stats.drafted += part.rows - 1;
                        one.stats.accepted += keep.* - 1;
                    },
                }
            }
            try self.slots.commit(self.m, self.keeps[0..count]);

            at = 0;
            var asked: usize = 0;
            for (parts, self.keeps[0..count], 0..) |part, keep, index| {
                const slot = part.slot;
                const one = &self.streams[slot];
                // the tokens after the rows kept: a prompt's own next ones, or the choices
                const chosen = self.chosen[at..][0..keep];
                at += part.rows;
                var given: []const i32 = chosen;
                if (one.phase == .prompt) {
                    @memcpy(chosen[0 .. keep - 1], one.seq.items[one.fed + 1 ..][0 .. keep - 1]);
                    if (part.score == .none) chosen[keep - 1] = one.seq.items[one.fed + keep];
                    one.fed += keep;
                    one.reached = one.fed;
                    given = if (part.score == .last) chosen[keep - 1 ..] else &.{};
                    if (part.score == .last) one.phase = .reply;
                }
                var finish: ?Finish = null;
                if (given.len > 0) {
                    try one.seq.appendSlice(self.gpa, given);
                    one.made += given.len;
                    sink.tokens(slot, given);
                    const whole = if (one.masks) |*masks| masks.ended() else false;
                    if (whole or std.mem.indexOfScalar(u32, one.eos.items, @intCast(given[given.len - 1])) != null) {
                        finish = .stop;
                    } else if (one.made == one.max_tokens) finish = .length;
                }
                // the rows kept into the drafter, and the next window's drafts out of it; a reply that has
                // ended is absorbed too, so that a prompt that goes on from it finds every position there
                one.pending = 0;
                if (one.drafted) {
                    const want = if (one.phase == .reply and finish == null) @min(self.depth, one.max_tokens - one.made - 1) else 0;
                    // every stream's rows go to the drafter together, once the last of them is known: its rows
                    // kept start at the part's first row of the window
                    self.jobs[asked] = .{ .slot = slot, .from = self.slots.first(index), .kept = chosen, .position = self.slots.pos(slot), .settings = one.settings, .ends = one.eos.items, .guide = if (one.masks) |*masks| masks.guide() else null, .out = one.drafts[0..want], .full = &one.full };
                    asked += 1;
                }
                if (finish) |how| sink.ended(slot, .{ .finish = how, .stats = one.stats });
            }
            if (asked > 0) {
                const jobs = self.jobs[0..asked];
                (if (self.head) |head| self.seats.?.draft(head, jobs) else self.flash_seats.?.draft(self.flash.?, jobs)) catch |err| {
                    std.log.err("round {d}, drafting for {d} streams: {t}", .{ self.rounds, asked, err });
                    return error.DraftFailed;
                };
                for (jobs) |job| self.streams[job.slot].pending = job.count;
            }
            return true;
        }
    };
}
