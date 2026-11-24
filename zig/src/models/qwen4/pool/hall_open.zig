//! How the pool's `Engine` is opened (`hall_engine.zig` names `openAfter` as its own): the extents, the
//! slots, a round's windows and graphs, the seats' host rows, the head's buffers, and what each cost; and,
//! before any of it is made, the extents' bytes, the rest's, and whether they fit (`extentsBytes`, `Rest`,
//! `Fit`: `serve`'s load).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const extents = @import("../../../decode/extents.zig");
const masks_file = @import("../../../decode/masks.zig");
const prefix = @import("../../../decode/prefix.zig");
const sampling = @import("../../../decode/sampling.zig");
const settle_file = @import("../../../decode/settle.zig");
const attention = @import("../blocks/attention.zig");
const forward = @import("../forward.zig");
const gdn = @import("../blocks/gdn.zig");
const engine_file = @import("hall_engine.zig");
const hall_file = @import("hall.zig");
const hall_seats = @import("hall_seats.zig");
const model_file = @import("../model.zig");
const mtp = @import("../mtp.zig");
const rounds = @import("rounds.zig");
const Between = engine_file.Between;
const Engine = engine_file.Engine;
const Asked = Engine.Asked;
const EngineOptions = hall_file.EngineOptions;
const HeadRound = hall_seats.HeadRound;
const Seat = hall_seats.Seat;
const window_max = masks_file.max_rows;

/// The device bytes of a pool's `places` extents for `m`, with nothing made: what `openAfter` asks before
/// `between`, an `attention.Pool` each attention layer and the head's in its own storage. Nothing else of
/// the pool grows with its positions: the slots' states and tables, a round's windows and the head's
/// buffers are the streams' and the context's, and are `Rest`'s.
pub fn extentsBytes(m: model_file.Model, places: usize) usize {
    const shared = m.attention orelse return 0;
    var bytes: usize = 0;
    for (m.layers) |layer| {
        if (layer.block == .attention) bytes += attention.Pool.bytesFor(shared.shape, places);
    }
    if (m.mtp != null) bytes += attention.Pool.bytesFor(mtp.over(m).attention.?.headShape(), places);
    return bytes;
}

/// The rows a pool's forward buffers score for `slots` streams and a model loaded for `rows`, which are the
/// rows each DeltaNet layer's short projection holds (`forward.keptRows`). Two rounds bound it. Every slot's
/// reply window and the row a prompt's last logits take beside them: `slots * window_max + 1`, all this was
/// until spec 086. And the widest round none of whose segments reaches `gdn.wide_rows`, which writes every
/// layer's projection there (`rounds.compute`): a prompt chunk one row short of the wide kernels beside
/// every other slot's window. At four streams that is three windows and 31 rows, 79, and the 14 rows past
/// the 65 were written and read past each layer's buffer.
pub fn shortRows(rows: usize, slots: usize) usize {
    std.debug.assert(slots >= 1 and gdn.wide_rows >= 1);
    const narrow = gdn.wide_rows - 1;
    return @min(rows, @max(slots * window_max + 1, (slots - 1) * @min(window_max, narrow) + narrow));
}

/// The device bytes of everything `openAfter` makes after `between` for `slots` streams of `capacity`
/// positions over `m`, with nothing made: each part by the `bytesFor` beside what allocates it, in the
/// three parts the opening's line says. It is the buffers' account (`cuda.account`) to the byte
/// (`hall_open_test.zig` opens pools and holds it); the allocator's rounding of each buffer and a captured
/// graph's memory are the runtime's and in no figure here. The rows are the model's (`max_rows`, what
/// `serve` loaded it for: `hall.rowsFor` of its streams, or what `EngineOptions.rows` leaves of them, which
/// `serve` says in the `m` it hands), so a figure for other streams than the model's is for rounds as wide as
/// the model's.
pub const Rest = struct {
    /// The slots' states, tables and pooled keys.
    slots: usize,
    /// A round's tables, its scratch and replays, the forward's buffers, the graphs' GEMM regions, the picks
    /// its reply rows are settled by.
    windows: usize,
    /// The head's buffers, its round's tables, the seats' scored rows and `picked`; 0 without a head.
    head: usize,
    /// DFlash2's taps of a round's rows, where the pool drafts by it (`session.Options.drafter`, `drafting`);
    /// its seats are made when the pool takes the session's drafter (`Engine.seatDrafter`), in no figure here.
    flash: usize = 0,

    pub fn of(m: model_file.Model, slots: usize, capacity: usize) Rest {
        const of_buffer = cuda.Buffer.bytesOf;
        const rows = m.max_rows;
        const scored = shortRows(rows, slots);
        var deltas: usize = 0;
        for (m.layers) |layer| deltas += @intFromBool(layer.block == .delta_net);
        var windows = rounds.Round.bytesFor(m, slots, rows) + (1 + deltas) * gdn.Scratch.bytesFor(rows, m.shape.delta_net);
        windows += forward.Buffers.bytesFor(m, m.layers, rows, scored, capacity, false) + rounds.Graphs.bytesFor(m, slots, rows, capacity);
        windows += settle_file.Picks.bytesFor(scored, m.head.n);
        var head: usize = 0;
        if (m.mtp) |held| {
            const columns = if (held.scored()) |ids| ids.len else m.shape.vocab;
            // the head's layer attends in the trunk's scratch, of the same rows and capacity
            head = mtp.Buffers.bytesFor(m, rows, capacity, true) + HeadRound.bytesFor(slots, rows);
            head += of_buffer(.bf16, &.{ rows, m.shape.wide() }) + of_buffer(.bf16, &.{ slots, m.shape.wide() });
            head += 2 * of_buffer(.bf16, &.{ slots, columns }) + of_buffer(.bf16, &.{ slots, m.shape.hidden });
            head += slots * of_buffer(.u32, &.{mtp.Picked.wordsFor(columns)});
        }
        return .{ .slots = slots * rounds.Slot.bytesFor(m, capacity, m.mtp != null), .windows = windows, .head = head };
    }

    /// `of`, with DFlash2's `taps` layers' means of a round's rows (`forward.Buffers.setTaps`); 0 taps: none.
    pub fn drafting(m: model_file.Model, slots: usize, capacity: usize, taps: usize) Rest {
        var rest = of(m, slots, capacity);
        if (taps > 0) rest.flash = forward.Buffers.tapBytes(m.max_rows, m.shape.hidden, taps);
        return rest;
    }

    pub fn bytes(self: Rest) usize {
        return self.slots + self.windows + self.head + self.flash;
    }
};

/// Whether the pool `serve` was asked for fits what the device can still give, said at the load and not
/// when a second stream comes (specs 082 and 084).
///
/// The comparison is the hand-over's (`hall.Core.migrate`), in its two steps. `left` is read once the lone
/// stream's session is made, so its buffers are already out of it. The extents are made whole beside that
/// session, whose rows are copied into them before it is freed: they are asked of `left` as it stands. A
/// pool that would fit only with the session gone could still open, with nothing carried (`migrate`'s
/// second try), and is refused here all the same. The rest (`Rest`) is made after the session is freed, in
/// its place: it is asked of `left` less the extents, with `session` given back. `cuda.reserve` is kept
/// free at both steps, as GLM's slots keep it.
///
/// `session` is the device bytes of the session's buffers as the load made it, by the buffers' account:
/// all of them are freed at the hand-over, a session then holding no state.
///
/// `states` is the states' budget (`--prompt-cache-gib`, `hall.default_states_bytes` unless given), held out
/// of `left` at both steps, as vLLM reserves its cache at the load (the owner, 2026-10-16; spec 101): the
/// session's parked sequences hold up to it before the hand-over, its snapshots are carried into the pool's
/// store, and the pool keeps and makes ready states up to it after (`EngineOptions.states_bytes`,
/// `hall_keep.ready` right after the opening). The budget is counted once a step, since what is carried is
/// moved, not copied. Not counted: a session's few snapshots beside its parked sequences (eight of about
/// 0.11 GiB), and what is the runtime's: the graphs the session captured and frees, those the pool captures,
/// a buffer's rounding, all within the reserve.
pub const Fit = struct {
    /// The pool's positions (`hall.poolFor`) and its extents' device bytes.
    positions: usize,
    bytes: usize,
    /// The rest of the pool's device bytes, by part.
    rest: Rest,
    /// What the device could still give (`cuda.available`), the session's buffers on it, and the states'
    /// budget held out of it.
    left: usize,
    session: usize,
    states: usize,
    fits: bool,
    /// The extents fit beside the session; false: the rest is not what refuses.
    extents_fit: bool,
    /// The most streams of this context, the asked ones at most, whose pool fits; 1: none, the session
    /// alone. No more than asked: the model's rows are for the rounds of the streams it was loaded for.
    parallel: usize,
    /// The longest context whose pool of these streams fits, a whole number of extents; 0: none.
    context: usize,

    pub fn of(m: model_file.Model, context: usize, parallel: usize, left: usize, session: usize, states: usize) Fit {
        const places = extents.wanted(context, parallel);
        const each = extentsBytes(m, 1);
        const free = left -| states;
        // a model without attention has no extent to refuse
        const room = if (each == 0) places else (free -| cuda.reserve) / each;
        // The rest grows with the streams and with the context, so the most of each is found from the
        // extents' own bound downward; what is said to fit is what `holds` says does.
        var streams = @max(@min(extents.slotsWithin(room, context, hall_file.most_parallel), parallel), 1);
        while (streams > 1 and !holds(m, context, streams, free, session)) streams -= 1;
        var longest = extents.capacityWithin(room, parallel) / extents.positions;
        while (longest > 0 and !holds(m, longest * extents.positions, parallel, free, session)) longest -= 1;
        return .{
            .positions = places * extents.positions,
            .bytes = extentsBytes(m, places),
            .rest = .of(m, parallel, context),
            .left = left,
            .session = session,
            .states = states,
            .fits = holds(m, context, parallel, free, session),
            .extents_fit = places <= room,
            .parallel = streams,
            .context = longest * extents.positions,
        };
    }

    /// The widest states' budget beside this pool that `holds`: what both steps leave once the extents and
    /// the rest are counted, the reserve kept; 0 where the pool alone does not fit. `serve` holds a budget
    /// it was not given to it, as vLLM sizes its cache by what the load leaves (`--gpu-memory-utilization`,
    /// refusing only where one request's cache does not fit, which the extents are here).
    pub fn statesWithin(m: model_file.Model, context: usize, parallel: usize, left: usize, session: usize) usize {
        if (!holds(m, context, parallel, left, session)) return 0;
        const bytes = extentsBytes(m, extents.wanted(context, parallel));
        const first = (left -| cuda.reserve) - bytes;
        const second = ((left - bytes + session) -| cuda.reserve) - Rest.of(m, parallel, context).bytes();
        return @min(first, second);
    }

    /// The rule: the extents within `left` (the states' budget out of it) less the reserve, then the rest
    /// within what the extents leave of it with the session's bytes given back, less the reserve.
    fn holds(m: model_file.Model, context: usize, parallel: usize, left: usize, session: usize) bool {
        const bytes = extentsBytes(m, extents.wanted(context, parallel));
        if (bytes > left -| cuda.reserve) return false;
        return Rest.of(m, parallel, context).bytes() <= (left - bytes + session) -| cuda.reserve;
    }
};

/// `open`, with `between` called once the extents are made and before anything else is: the session's
/// rows are copied into them there and the session freed (`hall_carry.zig`), so the two are both on the
/// device for the copy alone.
pub fn openAfter(gpa: Allocator, loaded: model_file.Model, eos: []const u32, options: EngineOptions, hook: ?Between) !Engine {
    // The pool's storage is the model's, which the lone stream's caches had too (`storageFor`); its rounds
    // the rows `options` say, where fewer
    var m = loaded;
    if (options.rows) |rows| m.max_rows = @min(m.max_rows, rows);
    std.debug.assert(options.slots >= 1 and options.slots <= hall_file.most_parallel);
    const places = options.pool / extents.positions;
    std.debug.assert(places >= extents.lowerBound(options.session.capacity, options.slots));
    // What the pool adds to the device, by part, from the buffers' own account (spec 051).
    const bytes_start = cuda.account.bytes;
    var n_attn: usize = 0;
    for (m.layers) |layer| n_attn += @intFromBool(layer.block == .attention);
    const trunk = try gpa.alloc(attention.Pool, n_attn);
    errdefer gpa.free(trunk);
    var made: usize = 0;
    errdefer for (trunk[0..made]) |*pool| pool.deinit();
    if (n_attn > 0) {
        const shape = m.attention.?.shape;
        for (trunk) |*pool| {
            pool.* = try attention.Pool.init(shape, places);
            made += 1;
        }
    }
    var head_pool: ?attention.Pool = null;
    errdefer if (head_pool) |*pool| pool.deinit();
    // the head's rows lie where the trunk's do (one table a sequence) in arrays of its own storage
    // beside a share, the head's own sizes (`mtp.over`): whole on rank 0, or this rank's half of a split head
    if (m.mtp != null) head_pool = try attention.Pool.init(mtp.over(m).attention.?.headShape(), places);
    const bytes_extents = cuda.account.bytes;
    var pool = try extents.Pool.init(gpa, .{ .extents = places, .capacity = options.session.capacity, .slots = options.slots });
    errdefer pool.deinit();
    if (hook) |called| called.call(called.context, .{ .trunk = trunk, .head_pool = head_pool, .pool = &pool });
    const bytes_between = cuda.account.bytes;
    const slots = try gpa.alloc(rounds.Slot, options.slots);
    errdefer gpa.free(slots);
    var n_slots: usize = 0;
    errdefer for (slots[0..n_slots]) |*slot| slot.deinit();
    if (options.sight) |sight| std.debug.assert(sight.ropes.len >= options.slots);
    for (slots, 0..) |*slot, i| {
        slot.* = try rounds.Slot.open(gpa, m, trunk, if (head_pool) |*heads| heads else null, options.session.capacity);
        n_slots += 1;
        // an image prompt's rotary table, `serve`'s beside the tower (spec 019 WS-05): none of the pool's bytes
        if (options.sight) |sight| slot.rope = &sight.ropes[i];
    }
    var store = try prefix.Store.init(gpa, &pool, options.slots, prefix.round_positions);
    errdefer store.deinit();
    const bytes_slots = cuda.account.bytes;
    const rows = m.max_rows;
    var staged: rounds.Round = try .init(m, options.slots, rows);
    errdefer staged.deinit();
    var shared: rounds.Shared = try .init(rows, m.shape.delta_net);
    errdefer shared.deinit();
    const scored = shortRows(rows, options.slots);
    var buffers: forward.Buffers = try .initFor(gpa, m, m.layers, rows, scored, options.session.capacity);
    errdefer buffers.deinit();
    // DFlash2's taps, before any graph is captured over the buffers: every round writes them
    // (`rounds.computeOver`), and the seats read them (`Engine.seatDrafter`)
    const taps: []const u32 = if (options.session.drafter) |opened| opened.config.taps else &.{};
    if (taps.len > 0) try buffers.setTaps(taps, .f32_acc);
    var deltas: usize = 0;
    for (m.layers) |layer| deltas += @intFromBool(layer.block == .delta_net);
    const saves = try gpa.alloc(gdn.Scratch, deltas);
    errdefer gpa.free(saves);
    var saved: usize = 0;
    errdefer for (saves[0..saved]) |*scratch| scratch.deinit();
    for (saves) |*scratch| {
        scratch.* = try .init(rows, m.shape.delta_net);
        saved += 1;
    }
    var graphs: rounds.Graphs = try .init(gpa, m, options.slots, rows, options.session.capacity);
    errdefer graphs.deinit();
    // a round's reply rows settled on the device (`Engine.settleRound`), a pick a rank's scored row
    var settling: settle_file.Picks = try .init(gpa, scored, m.head.n);
    errdefer settling.deinit();
    const bytes_windows = cuda.account.bytes;
    const seats = try gpa.alloc(?Seat, options.slots);
    errdefer gpa.free(seats);
    @memset(seats, null);
    const logits = try gpa.alloc([]u16, options.slots);
    errdefer gpa.free(logits);
    const head_rows = try gpa.alloc([]u16, options.slots);
    errdefer gpa.free(head_rows);
    var logit_at: usize = 0;
    errdefer for (logits[0..logit_at], head_rows[0..logit_at]) |row, other| {
        gpa.free(row);
        gpa.free(other);
    };
    for (logits, head_rows) |*row, *other| {
        row.* = try gpa.alloc(u16, m.shape.vocab);
        other.* = gpa.alloc(u16, m.shape.vocab) catch |err| {
            gpa.free(row.*);
            return err;
        };
        @memset(other.*, masks_file.minus_infinity);
        logit_at += 1;
    }
    const ids = try gpa.alloc(i32, rows);
    errdefer gpa.free(ids);
    const sid_host = try gpa.alloc(i32, rows);
    errdefer gpa.free(sid_host);
    const pos_host = try gpa.alloc(i32, rows);
    errdefer gpa.free(pos_host);
    const seg_host = try gpa.alloc(i32, options.slots * 4);
    errdefer gpa.free(seg_host);
    var self: Engine = .{
        .gpa = gpa,
        .m = m,
        .eos = eos,
        .context = options.session.capacity,
        .prompt_rows = @min(options.session.prompt_rows, rows),
        .chunk = @min(options.chunk, rows),
        .depth = options.depth,
        .copying = options.session.copies,
        .sight = options.sight,
        .pool = pool,
        .store = store,
        .trunk = trunk,
        .head_pool = head_pool,
        .slots = slots,
        .seats = seats,
        .logits = logits,
        .head_rows = head_rows,
        .staged = staged,
        .shared = shared,
        .buffers = buffers,
        .saves = saves,
        .graphs = graphs,
        .settling = settling,
        .eager = options.lazy_graphs,
        .ids = ids,
        .sid_host = sid_host,
        .pos_host = pos_host,
        .seg_host = seg_host,
        .keeps = .{ .budget = options.states_bytes, .reply_stride = options.session.reply_stride },
        .marking = options.session.marking(),
        .sampler = try sampling.Sampler.init(gpa, m.shape.vocab),
    };
    errdefer self.sampler.deinit(gpa);
    if (m.mtp != null) {
        self.hb = try mtp.Buffers.initBeside(gpa, m, rows, options.session.capacity, &self.buffers);
        errdefer self.hb.?.deinit();
        self.head = try HeadRound.init(options.slots, rows);
        errdefer self.head.?.deinit();
        self.gathered = try cuda.Buffer.init(.bf16, &.{ rows, m.shape.wide() });
        errdefer self.gathered.?.deinit();
        self.held = try cuda.Buffer.init(.bf16, &.{ options.slots, m.shape.wide() });
        errdefer self.held.?.deinit();
        self.columns = try gpa.alloc(u16, if (m.mtp.?.scored()) |scored_ids| scored_ids.len else 0);
        errdefer self.gpa.free(self.columns);
        // a split head's: every rank's ids, each rank's part laid beside the other's (`hall_ranks.Link.scoreHead`)
        const scored_columns = if (m.mtp.?.scored()) |scored_ids| scored_ids.len else m.shape.vocab;
        self.scored = try cuda.Buffer.init(.bf16, &.{ options.slots, scored_columns });
        errdefer self.scored.?.deinit();
        self.score_in = try cuda.Buffer.init(.bf16, &.{ options.slots, m.shape.hidden });
        errdefer self.score_in.?.deinit();
        self.score_out = try cuda.Buffer.init(.bf16, &.{ options.slots, scored_columns });
        errdefer self.score_out.?.deinit();
        self.picked = try gpa.alloc(cuda.Buffer, options.slots);
        errdefer gpa.free(self.picked);
        var picked_at: usize = 0;
        errdefer for (self.picked[0..picked_at]) |*buffer| buffer.deinit();
        for (self.picked) |*buffer| {
            buffer.* = try cuda.Buffer.init(.u32, &.{mtp.Picked.wordsFor(scored_columns)});
            picked_at += 1;
        }
        self.asked = try gpa.alloc(Asked, options.slots);
        errdefer gpa.free(self.asked);
        @memset(self.asked, .{});
        self.whole = try mtp.Normalizer.init(gpa);
        errdefer self.whole.?.deinit(gpa);
        try self.bindHead();
    }
    const bytes_head = cuda.account.bytes;
    // what the load was told of (`Fit`): a difference is a `bytesFor` that left its `init`
    const reckoned = Rest.drafting(m, options.slots, options.session.capacity, taps.len).bytes();
    if (reckoned != bytes_head - bytes_between) std.log.warn("the pool's slots, windows and head were reckoned at {d} bytes and took {d}", .{ reckoned, bytes_head - bytes_between });
    if (options.capture_at_open and !options.lazy_graphs) try self.captureFirst();
    std.log.info("the pool's device buffers: {d} MiB of extents ({d} of {d} positions), {d} MiB the {d} slots' states and tables, {d} MiB a round's windows of {d} rows, {d} MiB the head's, {d} MiB at the graphs' capture; {d} MiB with the model", .{
        (bytes_extents - bytes_start) >> 20, places,                             extents.positions,
        (bytes_slots - bytes_between) >> 20, options.slots,                      (bytes_windows - bytes_slots) >> 20,
        rows,                                (bytes_head - bytes_windows) >> 20, (cuda.account.bytes - bytes_head) >> 20,
        cuda.account.bytes >> 20,
    });
    return self;
}
