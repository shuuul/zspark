//! The seats' and the round's types of `hall.zig`'s `Engine`. A `Seat` is one stream's place and state in
//! the pool, from its fill to its reply's end (`Phase`). `Span` is a seat's rows the head pass scores and
//! `Live` its drafts in a round (`hall_chase.zig`), `HeadRound` the device buffers that pass is given at
//! the open (`hall_open.zig`).

const std = @import("std");
const cuda = @import("../../../cuda/cuda.zig");
const copied = @import("../../../decode/copied.zig");
const draft = @import("../draft.zig");
const extents = @import("../../../decode/extents.zig");
const marks = @import("../../../decode/marks.zig");
const masks_file = @import("../../../decode/masks.zig");
const mtp = @import("../mtp.zig");
const prefix = @import("../../../decode/prefix.zig");
const request_file = @import("../../../server/request.zig");
const sampling = @import("../../../decode/sampling.zig");

const hall_file = @import("hall.zig");

const window_max = masks_file.max_rows;

pub const Phase = enum { fill, decode, done };

pub const Seat = struct {
    chain: *extents.Chain,
    prompt: []i32,
    filled: usize = 0,
    seq: u64,
    phase: Phase = .fill,
    last: i32 = 0,
    made: usize = 0,
    max_tokens: usize,
    settings: sampling.Sampling,
    depth: usize,
    opening: usize,
    first: bool = true,
    masks: ?masks_file.Masks = null,
    /// Drafts copied from the seat's context where the head begins the same (`decode/copied.zig`); null
    /// under a constraint, without a head, and where the load asked for none.
    copies: ?copied.Copied = null,
    eos: []const u32,
    window: [window_max]i32 = undefined,
    drafts: [window_max]i32 = undefined,
    draft_n: usize = 0,
    /// DFlash2's seat is past its context's capacity: nothing more is absorbed, and nothing drafted
    /// (`DrafterOf.full`, a seat's).
    flash_full: bool = false,
    /// As `mtp.Head`'s: the stream's rate, and the head's last chain until its round is kept.
    paying: ?mtp.Paying = null,
    last_chain: ?mtp.Paying.Told = null,
    /// Of `drafts`, how many the constraint already gave the client.
    ahead: usize = 0,
    follow: []i32,
    drafted_rows: usize = 0,
    stop: bool = false,
    /// The reply's windows, for `sched.Tail`.
    rounds: usize = 0,
    /// Of `prompt`'s last tokens, the reply's own that its client had before this admission.
    sent: usize = 0,
    /// For the request's line in the log: the prompt's tokens found in a kept state, the reply's tokens given,
    /// the drafts its windows carried and kept, and when it began, gave its first token and ended.
    cached: usize = 0,
    given: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    /// A digest of every draft the head gave the seat, in order, as `session.Reply.drafts`.
    digest: std.hash.Wyhash = .init(0),
    began: ?std.Io.Timestamp = null,
    first_at: ?std.Io.Timestamp = null,
    ended_at: ?std.Io.Timestamp = null,
    /// Where the prompt keeps its state on the way: `marks.Planner`, a window each chunk, as a session's
    /// prompt a window each of its own. `list` and `dropped` are its, `starts` the request's, copied.
    planner: marks.Planner = undefined,
    list: [2 + prefix.prompt_marks + 1]usize = undefined,
    dropped: [prefix.prompt_marks + 1]usize = undefined,
    starts: []usize = &.{},
    /// `Ask.kept`: the state there is a shared prefix's end to the store (`prefix.Mark.shared`).
    kept: ?usize = null,
    /// What the chunk now running keeps where it ends, where the plan names that position.
    ending: ?prefix.Kind = null,
    /// The newest state the stream's sequence holds: where it forked, then each state it keeps.
    newest: usize = 0,
    /// The reply's next stride: the state is kept at the last round's end at or below it (`hall_keep.zig`'s `strideDue`).
    stride_at: usize = 0,
    /// `</think>` while the reply reasons and the state after it is wanted (`session.Ends`).
    close: ?i32 = null,
    /// `Ask.warm`: it steps aside for any other stream (`hall_warm.zig`) and leaves no `.end` behind.
    warm: bool = false,
    /// `Ask.logprobs`, and the rows of `follow`'s tokens (the prompt's last row's in the first), `[forward_max_rows]`.
    logprobs: ?u8 = null,
    rows: []request_file.LogprobRow = &.{},
    /// `Ask.image`: its features are made when a chunk first takes its rows, and no state above its first
    /// row is kept.
    image: ?hall_file.Picture = null,
};

pub const Span = struct {
    slot: usize,
    source: cuda.Buffer,
    source_row: usize,
    n: usize,
    tokens: []const i32,
    pos: usize,
    score: bool,
};

pub const Live = struct {
    /// The seat's chain of drafts, into `buf`; a seat that drafts nothing this round is halted.
    chain: draft.Chain = .{ .out = &.{}, .halt = true },
    buf: [window_max]i32 = undefined,
    /// The head's logits for the draft after the chain's are in the seat's row of `Engine.scored`.
    scored: bool = false,
};

pub const HeadRound = struct {
    sid: cuda.Buffer,
    posr: cuda.Buffer,
    seg: cuda.Buffer,
    ptrs: cuda.Buffer,
    tables: cuda.Buffer,
    bases: cuda.Buffer,

    pub fn init(slots: usize, rows: usize) !HeadRound {
        var sid: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer sid.deinit();
        var posr: cuda.Buffer = try .init(.i32, &.{rows});
        errdefer posr.deinit();
        var seg: cuda.Buffer = try .init(.i32, &.{ slots, 4 });
        errdefer seg.deinit();
        var ptrs: cuda.Buffer = try .init(.i64, &.{ slots, 2 });
        errdefer ptrs.deinit();
        var tables: cuda.Buffer = try .init(.i64, &.{slots});
        errdefer tables.deinit();
        var bases: cuda.Buffer = try .init(.i64, &.{ slots, 3 });
        errdefer bases.deinit();
        return .{ .sid = sid, .posr = posr, .seg = seg, .ptrs = ptrs, .tables = tables, .bases = bases };
    }

    /// The device bytes `init` asks, with nothing made.
    pub fn bytesFor(slots: usize, rows: usize) usize {
        const of = cuda.Buffer.bytesOf;
        return 2 * of(.i32, &.{rows}) + of(.i32, &.{ slots, 4 }) + of(.i64, &.{ slots, 2 }) + of(.i64, &.{slots}) + of(.i64, &.{ slots, 3 });
    }

    pub fn deinit(self: *HeadRound) void {
        self.sid.deinit();
        self.posr.deinit();
        self.seg.deinit();
        self.ptrs.deinit();
        self.tables.deinit();
        self.bases.deinit();
    }
};
