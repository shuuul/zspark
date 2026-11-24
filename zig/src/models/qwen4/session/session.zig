//! A loaded model that decodes one reply after another. `Session` keeps the state, buffers, graphs, sampler and
//! MTP head a reply needs and decodes a prompt's ids into text. A prompt that begins as the sequence before it
//! did is run from where they part: from the state as it stands, or from a snapshot taken on the way. `zspark chat`
//! decodes one reply through it and `zspark serve` every request's (`served.zig`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const request_file = @import("../../../server/request.zig");
const draft = @import("../draft.zig");
const copied = @import("../../../decode/copied.zig");
const units = @import("../../../cuda/units.zig");
const qwen4_config = @import("../config.zig");
const forward = @import("../forward.zig");
const grammar = @import("../../../text/grammar.zig");
const masks_file = @import("../../../decode/masks.zig");
const book_file = @import("../../../decode/book.zig");
const parked_file = @import("parked.zig");
const learnt = @import("learnt.zig");
const session_marks_file = @import("session_marks.zig");
const session_resume = @import("session_resume.zig");
const session_handover = @import("session_handover.zig");
pub const Handover = session_handover.Handover;
const model = @import("../model.zig");
const mtp = @import("../mtp.zig");
const rig = @import("../dflash2/rig.zig");
const pass = @import("../dflash2/pass.zig");
const dflash2_pass = @import("../../../decode/dflash2/pass.zig");
const logprob = @import("../../../decode/logprob.zig");
const sampling = @import("../../../decode/sampling.zig");
const tokenizer = @import("../../../text/tokenizer.zig");
const chat_file = @import("chat.zig");
const Sending = chat_file.Sending;

const options_file = @import("session_options.zig");
pub const Options = options_file.Options;
pub const prompt_rows = options_file.prompt_rows;
pub const promptRows = options_file.promptRows;
pub const lone_prompt_rows = options_file.lone_prompt_rows;
pub const lonePromptRows = options_file.lonePromptRows;
pub const most_prompt_rows = options_file.most_prompt_rows;
pub const refusedRows = options_file.refusedRows;
pub const prepared_buckets = options_file.prepared_buckets;
pub const Reply = options_file.Reply;
pub const default_depth = options_file.default_depth;
pub const default_opening_depth = options_file.default_opening_depth;
pub const openingFor = options_file.openingFor;

/// Where a reply's tokens go in place of its text, for a caller that makes the text elsewhere: `hall.zig`'s
/// lone stream, whose connection detokenizes on its own task (`server/scheduler.zig`). The sink is then asked
/// nothing. Both calls are on the thread that decodes.
pub const Tap = struct {
    context: *anyopaque,
    /// One token of the reply, in order, each once; never an end token.
    takeFn: *const fn (context: *anyopaque, token: i32) void,
    /// Asked before the reply's first token is given and after each window's are: true ends the reply there as
    /// `.cut` (the client has gone, or another request waits for the decoder). The state then holds what was
    /// given, as after any reply, and a prompt of this one and those tokens goes on from it.
    haltFn: *const fn (context: *anyopaque) bool,
    /// Told once, before the prompt runs: its tokens that will not be run, the state holding them.
    heldFn: ?*const fn (context: *anyopaque, cached: usize) void = null,
    /// For a reply that asks for log probabilities (`ReplyOptions.logprobs`): the row of the token `takeFn` is
    /// given next, or of the end token that ends the reply, before it.
    rowFn: ?*const fn (context: *anyopaque, row: *const request_file.LogprobRow) void = null,

    pub fn take(self: Tap, token: i32) void {
        self.takeFn(self.context, token);
    }

    fn halt(self: Tap) bool {
        return self.haltFn(self.context);
    }
};

pub const ReplyOptions = struct {
    /// The reply's tokens at most.
    max_tokens: usize,
    /// Drafts a window at most; the head's and the window's own limits apply too.
    depth: usize = default_depth,
    /// Drafts the reply's first window at most, where less than `depth` (`openingFor`).
    opening: usize = default_depth,
    /// Special tokens stay in the text and all of it is `content`, as the model wrote it.
    specials: bool = false,
    /// Where the reply's tokens are written, `[max_tokens]` at least.
    record: ?[]i32 = null,
    /// What the reply must be (`text/grammar.zig`): compiled by an engine made for this session's tokenizer,
    /// logits width (`Session.m.shape.vocab`) and end tokens (`Session.eos`). Every token is then chosen among
    /// the ones it allows, drafted or not, and the reply ends where its document does. Where the prompt
    /// leaves the reply thinking (its last `<think>` is not closed) the constraint begins after `</think>`
    /// and the reasoning is free. The caller keeps and frees it; it may serve the next reply too.
    constraint: ?grammar.Constraint = null,
    /// Where the prompt's messages start, ascending positions in it: where a later prompt most likely parts
    /// from this one's sequence (a message edited, the conversation cut back to it), so snapshots go there
    /// first. Empty: the session knows of none, and keeps its states by position alone.
    starts: []const usize = &.{},
    /// `marks.Request.kept`: where the prefix other conversations share ends, of `starts`.
    kept: ?usize = null,
    /// The reply's tokens go here and no text to the sink.
    tap: ?Tap = null,
    /// Of the prompt's last tokens, how many are this reply's own, given before it was cut (`Tap.haltFn`): the
    /// constraint begins before them and is advanced over them, and `max_tokens` counts what is left.
    sent: usize = 0,
    /// The tokens that end this reply; null: the session's. Empty, none does and the reply runs to its length
    /// (`ignore_eos`). The head still ends a chain at the session's (`mtp.Head.ends`), which costs a draft.
    eos: ?[]const u32 = null,
    /// Log probabilities, with this many best tokens a row (0 to 20): each token's row goes to `tap.rowFn`
    /// (`decode/logprob.zig`). The drafts a constraint forces are then not sent before their window.
    logprobs: ?u8 = null,
    /// An image prompt (spec 019 WS-04): its placeholders' features and every row's three positions. It is run
    /// from an empty state, as TensorFold 0.6.5 refuses an image prompt that resumes (decode.py:354), and leaves
    /// no state a later prompt goes on from: no snapshot, no lot, nothing learnt; the next prompt starts empty.
    image: ?ImagePrompt = null,
};

/// An image prompt's rows as `forward.State.attach` takes them; lent for the reply.
pub const ImagePrompt = struct {
    image: forward.Image,
    /// [prompt rows, 3]
    positions: []const [3]i32,
    delta: i32,
};

/// What a session lends a reply that asks for log probabilities: the device's words for a window's rows, and
/// each of `Session.chosen`'s tokens' row.
const Logprobs = struct {
    scratch: logprob.Scratch,
    rows: []logprob.Row,

    fn init(gpa: Allocator, rows: usize) !Logprobs {
        var scratch: logprob.Scratch = try .init(gpa, draft.forward_max_rows);
        errdefer scratch.deinit();
        return .{ .scratch = scratch, .rows = try gpa.alloc(logprob.Row, rows) };
    }

    fn deinit(self: *Logprobs, gpa: Allocator) void {
        self.scratch.deinit();
        gpa.free(self.rows);
    }
};

pub const Session = struct {
    gpa: Allocator,
    io: std.Io,
    m: model.Model,
    tokens: *const tokenizer.Tokenizer,
    /// Tokens that end a reply.
    eos: []const u32,
    options: Options,
    state: forward.State,
    buffers: forward.Buffers,
    /// Dropped before the state and buffers its graphs point into.
    graphs: ?forward.Graphs,
    sampler: sampling.Sampler,
    /// The MTP head where the model was loaded with one; it chooses drafts with `sampler`.
    head: ?mtp.Head,
    /// DFlash2 in the MTP head's place, loaded when `Options.drafter` is set.
    flash: ?*rig.Rig(2) = null,
    /// A test's drafter, asked in the head's place: it asks the head itself, which must absorb each window.
    drafting: ?draft.Drafter = null,
    /// The drafter around the head's that offers copies of the context, made by the first reply that drafts.
    copies: ?copied.Copied = null,
    /// One row's logits on the host.
    logits: []u16,
    /// One row's mask under a constraint: `[(vocab + 31) / 32]`.
    mask: []u32,
    /// The think block's tags, where the tokenizer has both.
    think: ?chat_file.Think,
    /// The sequence the state holds, `state.pos` tokens, then the token chosen after them, which the head's
    /// last entry was absorbed with.
    known: std.ArrayList(i32),
    /// `known`, the state and the head agree: the last reply ended without an error and its windows were all
    /// absorbed. Otherwise the next prompt starts from an empty state.
    trusted: bool = false,
    /// States of `known`'s sequence at earlier positions, lowest first; the head has absorbed each one's.
    snapshots: std.ArrayList(forward.Snapshot),
    /// Of `known`, the last prompt: a prompt that holds it but for its tail is the conversation's next turn or
    /// its reply cut, and the sequence is not left for it.
    asked: usize = 0,
    /// `ReplyOptions.kept` of the prompt running, and until `resume` has left the sequence, of the last.
    kept: ?usize = null,
    /// The sequences left before `known`, each to be taken up by a prompt that holds the whole of it.
    lot: parked_file.Lot,
    /// `serve --learn`'s states on disk (`learnt.zig`), the caller's; set before any request.
    learned: ?*learnt.Store = null,
    /// `serve --spill-gib`'s (`learnt.spilling`): the states that left.
    spilled: ?*learnt.Store = null,
    /// What the disk keeps free under either's files (`learnt.open`), the caller's; null: no floor.
    floor: ?*learnt.Floor = null,
    /// States of `known`'s sequence taken on a reply's stride (`Options.reply_stride`), lowest first, at most
    /// `Options.reply_marks`: beside `snapshots`, whose budget they do not share. The oldest's buffers are
    /// the next one's. They stay after their reply, until a prompt parts below them or a reply takes others.
    strides: std.ArrayList(forward.Snapshot),
    /// The states of `strides` a prompt parted below, kept for their device buffers: the next state on a
    /// reply's stride is written over one (`forward.Snapshot.again`). Freeing them was 3 to 5 ms of that
    /// prompt's time to its first token, and the buffers of two new ones 6 to 9 ms each of the reply's.
    /// What the two lists lack of `Options.reply_marks` is made while the prompt's last window computes
    /// (`readyStrides`, spec 089), so no round of a reply waits for an allocation. `strides` and these are
    /// `Options.reply_marks` together at most.
    spare: std.ArrayList(forward.Snapshot),
    /// The states of `snapshots` a prompt parted below or `thinned` named, kept for their device buffers as
    /// `spare`'s are (spec 041): the states the prompt takes are written into them. An edited last message
    /// parts below the two marks of the turn before and takes two; freed and allocated they were 3 to 5 ms
    /// each of its first token. `snapshots` and these are `Options.snapshots` together at most, so a session
    /// holds what its eight held at their most and no more.
    unused: std.ArrayList(forward.Snapshot),
    /// How many of the newest snapshots wait for their DeltaNet states (`forward.settle`), and the position
    /// their window ran from: a prompt's last window leaves them until the reply's first token is sent
    /// (`settle`), where the device has nothing to do but the next window.
    settling: usize = 0,
    settle_from: usize = 0,
    /// The snapshots the window that is running was asked for before it ran (`forward.Snapshot.ahead`): a
    /// prompt's wide window, between `Marks.before` and `Marks.window`. Empty otherwise.
    ahead: std.ArrayList(forward.Snapshot),
    /// The tokens after each row of a window, a prompt's widest included: `[rows]`.
    chosen: []i32,
    /// The rows of `chosen`'s tokens and the device's words for them, made by the first reply that asks for log
    /// probabilities (`ReplyOptions.logprobs`).
    logprobs: ?Logprobs = null,
    /// The positions of `snapshots` and `strides`, loaded before each question of where a prompt goes on
    /// from and which states it keeps (`booked`), the rules `decode/book.zig`'s; its `due` are those the
    /// prompt running takes, and its `from_stride` whether the state the last `resume` restored was the
    /// newest of `strides`, which `plan` may then hold among the snapshots.
    book: book_file.Book,

    /// On the heap: the head and the decoder point into the session. `m`, `tokens` and `eos` outlive it.
    pub fn create(gpa: Allocator, io: std.Io, m: model.Model, tokens: *const tokenizer.Tokenizer, eos: []const u32, options: Options) !*Session {
        const self = try gpa.create(Session);
        errdefer gpa.destroy(self);
        const rows = options.promptWindow(m.max_rows);
        var state: forward.State = try .initKeeping(gpa, m, rows, @min(rows, options.window), options.capacity);
        errdefer state.deinit();
        var buffers: forward.Buffers = try .initFor(gpa, m, m.layers, rows, @min(rows, options.window), options.capacity);
        errdefer buffers.deinit();
        var graphs: ?forward.Graphs = if (options.eager) null else try .init(gpa);
        errdefer if (graphs) |*g| g.deinit();
        const logits = try gpa.alloc(u16, m.shape.vocab);
        errdefer gpa.free(logits);
        const mask = try gpa.alloc(u32, (m.shape.vocab + 31) / 32);
        errdefer gpa.free(mask);
        var known: std.ArrayList(i32) = try .initCapacity(gpa, options.capacity + 1);
        errdefer known.deinit(gpa);
        var snapshots: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, options.snapshots + 1);
        errdefer snapshots.deinit(gpa);
        var strides: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, options.reply_marks);
        errdefer strides.deinit(gpa);
        var spare: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, options.reply_marks);
        errdefer spare.deinit(gpa);
        var unused: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, options.snapshots + 1);
        errdefer unused.deinit(gpa);
        var ahead: std.ArrayList(forward.Snapshot) = try .initCapacity(gpa, options.snapshots + 1);
        errdefer ahead.deinit(gpa);
        const chosen = try gpa.alloc(i32, @max(rows, draft.forward_max_rows));
        errdefer gpa.free(chosen);
        var book: book_file.Book = try .init(gpa, .{ .marks = options.marking(), .reply_stride = options.reply_stride, .reply_marks = options.reply_marks });
        errdefer book.deinit(gpa);
        const open = tokens.addedId("<think>");
        const close = tokens.addedId("</think>");
        self.* = .{
            .gpa = gpa,
            .io = io,
            .m = m,
            .tokens = tokens,
            .eos = eos,
            .options = options,
            .state = state,
            .buffers = buffers,
            .graphs = graphs,
            .sampler = try .init(gpa, logits.len),
            .head = null,
            .logits = logits,
            .mask = mask,
            .think = if (open != null and close != null) .{ .open = @intCast(open.?), .close = @intCast(close.?) } else null,
            .known = known,
            .snapshots = snapshots,
            .lot = .{ .gpa = gpa, .most = if (options.snapshots >= 2) options.parked else 0, .cap = options.parked_bytes },
            .strides = strides,
            .spare = spare,
            .unused = unused,
            .ahead = ahead,
            .chosen = chosen,
            .book = book,
        };
        errdefer self.sampler.deinit(gpa);
        if (m.mtp != null) {
            self.head = try .init(gpa, m, &self.sampler, .{ .seed = 0 }, .{ .rows = rows, .capacity = options.capacity, .graphs = !options.eager, .beside = &self.buffers });
            self.head.?.ends = eos;
        }
        if (options.drafter) |opened| {
            var arena_state: std.heap.ArenaAllocator = .init(gpa);
            defer arena_state.deinit();
            const arena = arena_state.allocator();
            // on the heap: the rig points into itself (its drafter at its pair, the pair at its passes), and the
            // pool takes it when it opens (`takeFlash`)
            const flash = try gpa.create(rig.Rig(2));
            errdefer gpa.destroy(flash);
            var source: rig.Source = .{ .checkpoint = &opened.checkpoint.? };
            try flash.load(arena, gpa, opened, &source, .{ 0, 1 }, m.scratch, options.drafter_stored, null);
            errdefer flash.unload();
            const heads = try rig.draftHeads(2, arena, gpa, m, options.drafter_stored, .{ .pool = m.scratch, .count = dflash2_pass.batch });
            try flash.init(gpa, opened, heads, null, m, &self.buffers, options.capacity, .{ .seed = 0 }, options.eager);
            self.flash = flash;
        }
        return self;
    }

    pub fn destroy(self: *Session) void {
        const gpa = self.gpa;
        if (self.copies) |*copies| copies.deinit();
        for (self.snapshots.items) |*snapshot| snapshot.deinit();
        self.snapshots.deinit(gpa);
        self.lot.deinit();
        for (self.strides.items) |*snapshot| snapshot.deinit();
        self.strides.deinit(gpa);
        for (self.spare.items) |*snapshot| snapshot.deinit();
        self.spare.deinit(gpa);
        for (self.unused.items) |*snapshot| snapshot.deinit();
        self.unused.deinit(gpa);
        self.dropAhead();
        self.ahead.deinit(gpa);
        gpa.free(self.chosen);
        if (self.logprobs) |*held| held.deinit(gpa);
        self.book.deinit(gpa);
        self.known.deinit(gpa);
        if (self.flash) |f| {
            f.deinit();
            gpa.destroy(f);
        }
        if (self.head) |*h| h.deinit();
        self.sampler.deinit(gpa);
        gpa.free(self.logits);
        gpa.free(self.mask);
        if (self.graphs) |*g| g.deinit();
        self.buffers.deinit();
        self.state.deinit();
        gpa.destroy(self);
    }

    /// Device bytes the snapshots hold, those on a reply's stride and the buffers kept for the next included.
    pub fn snapshotBytes(self: *const Session) usize {
        var total: usize = 0;
        for (self.snapshots.items) |one| total += one.bytes();
        for (self.strides.items) |one| total += one.bytes();
        for (self.spare.items) |one| total += one.bytes();
        for (self.unused.items) |one| total += one.bytes();
        return total + self.lot.spareBytes();
    }

    /// The Book with the positions of the states held, and `kept`.
    pub fn booked(self: *Session) *book_file.Book {
        const book = &self.book;
        book.clear();
        for (self.snapshots.items) |one| book.put(one.pos, false);
        for (self.strides.items) |one| book.put(one.pos, true);
        book.kept = self.kept;
        return book;
    }

    /// The position of the newest state held, of either kind; 0 with none.
    pub fn newest(self: *const Session) usize {
        const a = if (self.snapshots.getLastOrNull()) |one| one.pos else 0;
        const b = if (self.strides.getLastOrNull()) |one| one.pos else 0;
        return @max(a, b);
    }

    /// Graphs captured so far, the head's included, and the nanoseconds capturing them took.
    pub fn captured(self: *const Session) struct { count: usize, nanoseconds: u64 } {
        const g = self.graphs orelse return .{ .count = 0, .nanoseconds = 0 };
        const head_graphs = if (self.head) |h| h.graphs else null;
        return .{
            .count = g.cache.count() + if (head_graphs) |hg| hg.cache.count() else 0,
            .nanoseconds = g.cache.capturing() + if (head_graphs) |hg| hg.cache.capturing() else 0,
        };
    }

    /// Captures the graphs replies of up to `depth` drafts a round ask for, so that none is captured while a
    /// reply decodes: the step's for windows of 1 to `depth + 1` rows on either DeltaNet state side, and the
    /// head's (`mtp.Head.prepare`), over each of the first `buckets` buckets of context. A sequence that
    /// ends past them captures as it goes. Nothing is launched, so the state, the head and `known` stand as
    /// they did, whenever this is called; a session that launches each kernel (`eager`) has nothing to do,
    /// and one without a head only its one-row step.
    pub fn prepare(self: *Session, depth: usize, buckets: usize) !void {
        const graphs = if (self.graphs) |*g| g else return;
        // DFlash2's windows are a block's at most (`DrafterOf.depthOf`)
        const drafts = if (self.flash) |f| f.d.depthOf(depth) else if (self.head != null) depth else 0;
        const rows = @min(drafts + 1, self.buffers.scoredRows(), self.m.max_rows, draft.forward_max_rows);
        var context: ?usize = null;
        for (0..buckets) |_| {
            context = forward.Graphs.bucketAfter(self.buffers, context) orelse break;
            try graphs.prepareWindows(self.m, &self.state, &self.buffers, rows, context.?);
        }
        const head = if (self.head) |*h| h else return;
        context = null;
        for (0..buckets) |_| {
            context = forward.Graphs.bucketAfter(head.b.layer, context) orelse break;
            try head.prepare(self.buffers.h, context.?);
        }
    }

    /// The DFlash2 drafter, the caller's from here on, and none left here: the pool's when it opens
    /// (`../pool/hall.zig`'s `Handing`), before the session is freed.
    pub fn takeFlash(self: *Session) ?*rig.Rig(2) {
        const flash = self.flash;
        self.flash = null;
        return flash;
    }

    // Where a prompt goes on from and the states kept on the way (`session_resume.zig`).
    const @"resume" = session_resume.@"resume";
    pub const leave = session_resume.leave;
    pub const adopt = session_resume.adopt;
    pub const retireFrom = session_resume.retireFrom;
    const plan = session_resume.plan;
    pub const retire = session_resume.retire;
    pub const readyStrides = session_resume.readyStrides;
    const settle = session_resume.settle;
    const unsettled = session_resume.unsettled;
    pub const dropAhead = session_resume.dropAhead;

    const Marks = session_marks_file.Marks;

    const Ends = session_marks_file.Ends;

    pub const keepStride = session_resume.keepStride;
    pub const keepWithin = session_resume.keepWithin;

    /// What the session hands the pool that takes it over (`session_handover.zig`).
    pub const handover = session_handover.handover;

    /// The line a request leaves in the log, into `buffer`: the prompt and what of it was held, the reply's
    /// rate, the drafts kept, and where the states stand. `serve` writes it for every reply, under any
    /// `--parallel` (`served.zig`, and `hall.zig` for a lone stream).
    pub fn replyLine(self: *const Session, buffer: []u8, prompt_tokens: usize, done: Reply) []const u8 {
        var marks: [32]usize = undefined;
        var strides: [8]usize = undefined;
        const n = @min(marks.len, self.snapshots.items.len);
        for (marks[0..n], self.snapshots.items[0..n]) |*pos, one| pos.* = one.pos;
        const k = @min(strides.len, self.strides.items.len);
        for (strides[0..k], self.strides.items[0..k]) |*pos, one| pos.* = one.pos;
        var copies_text: [96]u8 = undefined;
        var levels_text: [160]u8 = undefined;
        var w: std.Io.Writer = .fixed(buffer);
        w.print("prompt {d} tokens, {d} held, in {d:.2} s; reply {d} tokens in {d:.2} s, {d:.1} tok/s; {d} of {d} drafts kept{s}{s}; {d} snapshots, {d} MiB, at {any}; on a reply's stride {any}", .{
            prompt_tokens,                                     done.cached,
            done.prompt_seconds,                               done.tokens,
            done.seconds,                                      @as(f64, @floatFromInt(done.tokens)) / @max(done.seconds, 1e-9),
            done.stats.accepted,                               done.stats.drafted,
            done.copies.line(&copies_text),                    done.levels.line(&levels_text),
            self.snapshots.items.len + self.strides.items.len, self.snapshotBytes() >> 20,
            marks[0..n],                                       strides[0..k],
        }) catch {};
        return w.buffered();
    }

    /// Whether `prompt` leaves the reply thinking: its last `<think>` is not closed. The reply then opens as
    /// reasoning, and a constraint begins after its `</think>`.
    pub fn thinks(self: *const Session, prompt: []const i32) bool {
        return chat_file.thinks(self.think, prompt);
    }

    /// The next round's window as far as it is known, said to `forward.ask`: the kept token when the round
    /// before is accepted, then the drafts the head goes on from.
    const Asking = struct {
        session: *Session,
        window: [draft.forward_max_rows]i32 = undefined,

        fn last(self: *Asking, token: i32) void {
            self.window[0] = token;
            forward.ask(self.session.m, &self.session.state, &self.session.buffers, self.window[0..1]);
        }

        fn drafts(context: *anyopaque, picked: []const i32) void {
            const self: *Asking = @ptrCast(@alignCast(context));
            if (picked.len >= self.window.len) return;
            @memcpy(self.window[1..][0..picked.len], picked);
            forward.ask(self.session.m, &self.session.state, &self.session.buffers, self.window[0 .. 1 + picked.len]);
        }
    };

    /// An image prompt's start (`ReplyOptions.image`): every state this session held let go, the state and the
    /// head empty, then the image's rows and positions attached.
    fn fromImage(self: *Session, image: ImagePrompt, settings: sampling.Sampling) !usize {
        self.trusted = false;
        self.lot.clear();
        self.retireFrom(0);
        self.kept = null;
        self.book.from_stride = false;
        try self.state.reset();
        try self.state.attach(image.image, image.positions, image.delta);
        if (self.head) |*h| {
            try h.reset(settings);
            try h.attach(self.state.rope);
        }
        return 0;
    }

    /// Decodes the reply to `prompt`: its tokens are those an empty state gives, whatever this session decoded
    /// before, though the prompt is run only from where it parts from the sequence the state holds. The text
    /// goes to `sink` as it is decoded; the reply opens as reasoning when the prompt's last `<think>` is not
    /// closed, and the blank lines the template puts after `</think>` are dropped.
    pub fn reply(self: *Session, prompt: []const i32, settings: sampling.Sampling, options: ReplyOptions, sink: request_file.Sink) !Reply {
        std.debug.assert(prompt.len >= 1 and prompt.len + options.max_tokens <= self.options.capacity);
        if (self.head) |*h| h.fresh();
        if (self.flash) |f| {
            f.d.settings = settings;
            f.d.full = false;
        }
        var drafter: ?draft.Drafter = if (options.depth > 0)
            self.drafting orelse if (self.flash) |f| f.d.drafter() else if (self.head) |*h| h.drafter() else null
        else
            null;
        if (drafter != null and self.options.copies) {
            if (self.copies == null) self.copies = try .init(self.gpa, drafter.?, &.{});
            const copies = &self.copies.?;
            copies.inner = drafter.?;
            copies.stops = options.eos orelse self.eos;
            copies.store = self.options.recall;
            try copies.begin(prompt);
            drafter = copies.drafter();
        }
        // a head that is not asked absorbs nothing, and its entries would lag the state
        const absorbing = (self.head == null and self.flash == null) or drafter != null;
        const prompt_start: std.Io.Timestamp = .now(self.io, .awake);
        // a text prompt after an image's: what it goes on from is a text state (no image's is kept), at its
        // own positions, which a snapshot restored does not set (no copy where they are so already)
        if (options.image == null) {
            try self.state.rope.clear();
            if (self.head) |*h| try h.state.rope.clear();
        }
        const shared = if (options.image) |image| try self.fromImage(image, settings) else try self.@"resume"(prompt, settings, options.kept);
        if (self.flash) |f| {
            if (f.pair.context_end != self.state.pos) {
                if (self.state.pos == 0) f.pair.reset() else try f.pair.startAt(self.state.pos);
            }
        }
        self.trusted = false;
        const cached = self.state.pos;
        if (options.tap) |tap| if (tap.heldFn) |held| held(tap.context, cached);
        var rounds: draft.Decoder = .{ .m = self.m, .state = &self.state, .b = &self.buffers, .graphs = if (self.graphs) |*g| g else null, .sampler = &self.sampler, .settings = settings, .eos = options.eos orelse self.eos, .logits = self.logits };
        if (options.logprobs) |k| {
            if (self.logprobs == null) self.logprobs = try .init(self.gpa, self.chosen.len);
            rounds.logprobs = .{ .k = k, .scratch = &self.logprobs.?.scratch, .rows = self.logprobs.?.rows };
        }
        // each token's row, beside `chosen`, for a reply that asks for log probabilities
        const rows: ?[]const logprob.Row = if (rounds.logprobs) |lp| lp.rows else null;
        var decoder: tokenizer.StreamDecoder = try .init(self.gpa, self.tokens, .{ .skip_special = !options.specials });
        defer decoder.deinit(self.gpa);

        const thinking = self.thinks(prompt);
        var masks: masks_file.Masks = undefined;
        if (options.constraint) |constraint| {
            const before = prompt[0 .. prompt.len - options.sent];
            masks = try .init(constraint, self.mask, if (self.thinks(before)) self.think.?.close else null);
            try masks.advance(prompt[before.len..]);
            masks.io = self.io;
            rounds.masks = &masks;
        }

        // Snapshots on the way (`plan`): at the messages' starts first, then where this prompt parted from the
        // sequence before it, a little before its end, and along a long message. Each is taken from inside the
        // window that runs over it (`forward.Snapshot.within`), so the prompt is cut into no more windows than
        // its rows need: a short new turn is one window.
        const inside = absorbing and self.options.snapshots > 0 and options.image == null;
        var marks: Marks = .{ .session = self, .end = prompt.len, .prompt = prompt, .rounds = &rounds, .room = self.chosen.len, .reply = if (inside and self.options.reply_marks > 0 and self.options.reply_stride > 0) options.max_tokens else 0 };
        if (inside) {
            self.plan(rounds, prompt.len, shared, options.starts);
            marks.at = self.book.due;
        }
        rounds.within = .{ .context = &marks, .window = Marks.window, .before = Marks.before, .launched = Marks.launched };
        defer self.dropAhead();
        defer self.unsettled();
        // a window said and never staged (a prompt that failed) is not expected of the next prompt
        defer forward.expect(&self.buffers, &.{});
        // A round's table rows are asked for as its tokens are known, and gathered while the head drafts.
        var asking: Asking = .{ .session = self };
        if (self.head) |*h| h.picked = .{ .context = &asking, .tell = Asking.drafts };
        defer if (self.head) |*h| {
            h.picked = null;
        };

        // The tokens after the rows the last window kept: after the prompt, the last of them is the reply's first.
        const chosen = self.chosen;
        var kept = try rounds.prompt(drafter, prompt, chosen);
        rounds.within = null;
        // And inside the reply: where its reasoning ends, and on its stride (`Ends`).
        var ends: Ends = .{
            .session = self,
            .close = if (thinking and inside) self.think.?.close else null,
            .stride = if (inside and self.options.reply_marks > 0) self.options.reply_stride else 0,
        };
        rounds.kept = .{ .context = &ends, .window = Ends.window };
        const prompt_seconds = units.secondsSince(self.io, prompt_start);
        self.known.clearRetainingCapacity();
        self.known.appendSliceAssumeCapacity(prompt);
        self.asked = prompt.len;
        self.known.appendAssumeCapacity(chosen[kept - 1]);

        var sending: Sending = .{ .session = self, .options = options, .sink = sink, .speaker = .{ .decoder = &decoder, .close = if (self.think) |tags| tags.close else null, .reasoning = thinking and !options.specials } };

        const start: std.Io.Timestamp = .now(self.io, .awake);
        var drafts: [draft.forward_max_rows]i32 = undefined;
        var drafted: std.hash.Wyhash = .init(0);
        // of the last window's tokens, the ones not sent yet: after the prompt, the reply's first alone
        var fresh: usize = kept - 1;
        var ahead: usize = 0;
        // the reply's first window is the one that may take fewer drafts (`ReplyOptions.opening`)
        var first = true;
        var opened: usize = 0;
        const finish: request_file.Finish = reply: while (true) {
            if (first) if (options.tap) |tap| if (tap.halt()) break :reply .cut;
            for (chosen[fresh..kept], fresh..) |next, i| if (sending.take(next, if (rows) |r| &r[i] else null)) |end| break :reply end;
            // the reply's first token is sent: what the prompt's last window left of its snapshots
            try self.settle();
            if (sending.speaker.count == options.max_tokens) break .length;
            if (options.tap) |tap| if (tap.halt()) break :reply .cut;
            // a draft past the last token asked for would be verified for nothing
            const most = if (first) @min(options.depth, options.opening) else options.depth;
            asking.last(chosen[kept - 1]);
            const n = if (drafter) |d| try rounds.draft(d, chosen[0..kept], @min(most, options.max_tokens - sending.speaker.count - 1), &drafts) else 0;
            if (first) opened = n;
            first = false;
            drafted.update(std.mem.sliceAsBytes(drafts[0..n]));
            drafted.update(&.{@intCast(n)});
            // The drafts a constraint forces are the reply's tokens whatever their rows compute (a row whose
            // mask leaves one token chooses it), so their text goes before the window that verifies them: a
            // required tool call's opening reaches its client a window sooner (spec 037). The window runs all
            // the same, where the sink says stop too, for the state to hold every token that was sent.
            const sure = if (rows != null) 0 else if (rounds.masks) |m| m.certain(drafts[0..n]) else 0;
            var ended: ?request_file.Finish = null;
            for (drafts[0..sure]) |next| {
                ended = sending.take(next, null);
                if (ended != null) break;
            }
            ahead += sure;
            kept = try rounds.round(chosen[kept - 1], drafts[0..n], chosen);
            self.known.appendSliceAssumeCapacity(chosen[0..kept]);
            if (kept <= sure or !std.mem.eql(i32, chosen[0..sure], drafts[0..sure])) {
                std.log.err("a window after position {d} did not keep the {d} tokens its constraint forces", .{ self.state.pos, sure });
                return error.DraftFailed;
            }
            if (ended) |end| break end;
            fresh = sure;
        };
        try self.settle();
        // The head absorbs the last window, as it would before the next: the state and it then stand at the
        // same position, for the next prompt to go on from.
        if (drafter) |d| _ = try rounds.draft(d, chosen[0..kept], 0, &drafts);
        if (drafter != null and self.options.copies) self.copies.?.end();
        self.trusted = absorbing and (self.head == null or !self.head.?.full) and options.image == null;
        std.debug.assert(self.known.items.len == self.state.pos + 1);
        sending.speaker.end(finish == .cut, sink);
        self.lot.sweep();
        // the store the next prompt that leaves this sequence holds its rows in (spec 101)
        if (self.trusted) if (self.m.attention) |att| self.lot.prepare(self.state, if (self.head) |h| h.state.cache else null, att.shape.ratio);
        if (self.learned) |store| if (options.image == null) learnt.learn(self, store);
        return .{ .finish = finish, .end = sending.end, .tokens = sending.speaker.count, .opening = opened, .ahead = ahead, .reasoning_tokens = sending.speaker.reasoning_tokens, .stats = rounds.stats, .copies = if (self.options.copies and drafter != null) self.copies.?.stats else .{}, .levels = if (self.head) |head| if (head.paying) |paying| paying.counts else .{} else .{}, .cached = cached, .windows = marks.windows, .graphed = rounds.graphed > 0, .settled = marks.settled, .taken = marks.taken + ends.taken, .drafts = drafted.final(), .prompt_seconds = prompt_seconds, .seconds = units.secondsSince(self.io, start), .masks = if (rounds.masks) |m| m.stats else .{} };
    }
};
