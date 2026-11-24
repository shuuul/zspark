//! Nemotron-3.5-Transcribe's rank 1 of a two-rank serve (spec 151 WS-06/WS-08): rank 0 alone serves the
//! HTTP routes and decodes each reply — the choices, the sampler, the stop strings and the early exits
//! live there only. What crosses the control channel is the operations rank 0 runs, told before they run
//! (`ranks/lead.zig`'s frames of `ranks_forward.Op`); this rank replays each on its own share — the same
//! window's ids, the same kept rows, the same draft asks — so both ranks meet the same collectives at the
//! same points and the gathered rows' bits are the same. There is no job queue here: a cancel or a stop
//! on rank 0 is simply no more ops until the next `reset`.
//!
//! The prompt's audio reaches this rank as the file's bytes in an `audio` op: its own perception encodes
//! them into `features` and `state.audio`, as rank 0's did — the embeddings over the placeholder rows are
//! identical, so the forward over them meets.

const std = @import("std");
const channel_file = @import("../../ranks/channel.zig");
const cuda = @import("../../cuda/cuda.zig");
const draft_file = @import("draft.zig");
const mtp_file = @import("mtp.zig");
const perception_file = @import("perception.zig");
const ranks_forward = @import("ranks_forward.zig");
const held_ns = @import("held_ns.zig");
const state_file = @import("state.zig");
const flash_file = @import("dflash2/drafter.zig");
const flash_follow = @import("../../decode/dflash2/follow.zig");
const flash_pass = @import("dflash2/pass.zig");
const forward_file = @import("forward.zig");
const lead_file = @import("../../ranks/lead.zig");
const Allocator = std.mem.Allocator;

const Op = ranks_forward.Op;

const model_file = @import("model.zig");

pub const Error = error{ OutOfStep, DraftFailed } || channel_file.Error || perception_file.Error || held_ns.Error || flash_follow.Error(flash_pass.Target);

/// What the mirror holds: this rank's share of the model and the decode's working set — no engine, no
/// requests: the ops carry everything a reply needs.
pub const Followed = struct {
    gpa: Allocator,
    /// The rank's model share (`model_file.Model` of `loadShared`), its state and window buffers, and the
    /// pair's gather (`ranks_forward.Ranks`, this rank's `rank` 1).
    share: model_file.Model,
    state: *state_file.State,
    b: *state_file.Buffers,
    ranks: *ranks_forward.Ranks,
    capacity: usize,
    /// The clips' encoded rows, as `served`'s `features`: `audio` ops write them here.
    features: cuda.Buffer,
    perception: *perception_file.Perception,
    /// A drafter of this rank's own where rank 0 has one: the MTP head (`draft` ops) or a DFlash2 rig's
    /// `d` (`flash_*` ops).
    mtp: ?*mtp_file.Drafter = null,
    flash: ?*flash_file.DrafterOf(1) = null,
    /// Scratch for a told op's ids, grown to `b.rows` (a prompt window's width), and the open window's
    /// ids beside it (`commit`'s told ids must equal them — the parse lands apart from the store).
    window: []i32 = &.{},
    last: []i32 = &.{},
    /// The open window's rows, for `commit`'s told-ids check.
    open: usize = 0,
    /// The reply's audio rows so far (`audio` ops append; `reset` clears), and `state.audio.rows` — one
    /// allocation (the features' row count) so the slice `state.audio` holds never moves.
    audio_rows: []u32 = &.{},
    audio_off: usize = 0,
    /// An `audio` op's row chunk parsed (the lead sends a thousand a frame).
    chunk: [1024]i32 = undefined,
    /// Audio row ids accumulate before the clip's bounded PCM frames.
    pending: std.ArrayList(u32) = .empty,
    pending_rows: usize = 0,
    pending_samples: usize = 0,
    samples: std.ArrayList(f32) = .empty,
    /// A draft's out rows.
    drafts: [draft_file.forward_max_rows]i32 = undefined,
    pub fn deinit(f: *Followed) void {
        if (f.audio_rows.len > 0) f.gpa.free(f.audio_rows);
        f.pending.deinit(f.gpa);
        f.samples.deinit(f.gpa);
        if (f.window.len > 0) f.gpa.free(f.window);
        if (f.last.len > 0) f.gpa.free(f.last);
    }
};

/// The next frame that is an op's; `barrier` means out of step.
fn told(channel: *channel_file.Channel) Error!lead_file.Told {
    return switch (try channel.next()) {
        .request => |bytes| lead_file.Told.read(bytes) orelse error.OutOfStep,
        .barrier => error.OutOfStep,
    };
}

const Parts = struct { a: []align(1) const u32, tokens: []const i32 };

/// A `told`'s `n` argument words and its i32 tokens after them.
fn parts(t: lead_file.Told, n: usize, out: []i32) Error!Parts {
    if (t.rest.len < n) return error.OutOfStep;
    const tokens = t.tokens(n, out) orelse return error.OutOfStep;
    return .{ .a = t.rest[0..n], .tokens = tokens };
}

/// Runs what rank 0 tells until `end` — the whole session's ops replayed on this rank's share. Returns
/// the operations ran.
pub fn follow(channel: *channel_file.Channel, f: *Followed) Error!usize {
    if (f.window.len < f.b.rows) {
        f.window = try f.gpa.alloc(i32, f.b.rows);
        f.last = try f.gpa.alloc(i32, f.b.rows);
    }
    if (f.audio_rows.len < f.features.view.size[0]) f.audio_rows = try f.gpa.alloc(u32, @intCast(f.features.view.size[0]));
    var ran: usize = 0;
    while (true) {
        const t = try told(channel);
        const op = std.enums.fromInt(Op, t.kind) orelse return error.OutOfStep;
        ran += 1;
        switch (op) {
            .end => {
                try channel.pass(ran);
                return ran;
            },
            .reset => {
                if (t.rest.len != 0) return error.OutOfStep;
                try f.state.reset();
                if (f.flash) |flash| flash.reset(.{ .seed = 0, .temperature = 0.0 });
                f.audio_off = 0;
                f.pending_rows = 0;
                f.open = 0;
            },
            .audio => {
                // {n_rows, n_samples} and row-id chunks, then bounded frames of 16-kHz f32 PCM.
                // The admitted request owns the source; no prepared-file cache survives refusals.
                const a = try parts(t, 2, &f.chunk);
                if (f.pending_rows == 0) {
                    f.pending_rows = a.a[0];
                    f.pending_samples = a.a[1];
                    if (f.pending_rows == 0 or f.pending_samples == 0) return error.OutOfStep;
                    f.pending.clearRetainingCapacity();
                } else if (a.a[0] != 0 or a.a[1] != 0) return error.OutOfStep;
                for (a.tokens) |id| {
                    if (id < 0) return error.OutOfStep;
                    f.pending.append(f.gpa, @intCast(id)) catch return error.OutOfMemory;
                }
                if (f.pending.items.len > f.pending_rows) return error.OutOfStep;
                if (f.pending.items.len < f.pending_rows) continue;
                const rows = f.pending.items;
                const n_rows = f.pending_rows;
                try f.samples.resize(f.gpa, f.pending_samples);
                const samples = f.samples.items;
                const pcm = std.mem.sliceAsBytes(samples);
                var at: usize = 0;
                while (at < pcm.len) {
                    const n = @min(ranks_forward.audio_frame_bytes, pcm.len - at);
                    const bytes = switch (try channel.next()) {
                        .request => |blob| blob,
                        .barrier => return error.OutOfStep,
                    };
                    if (bytes.len != n) return error.OutOfStep;
                    @memcpy(pcm[at..][0..n], bytes);
                    at += n;
                }
                const made = try f.perception.rows(f.gpa, samples.len);
                if (made != n_rows or f.audio_off + n_rows > f.features.view.size[0]) return error.OutOfStep;
                try f.perception.encode(f.gpa, samples, f.features.span(f.audio_off, n_rows));
                @memcpy(f.audio_rows[f.audio_off..][0..rows.len], rows);
                f.audio_off += n_rows;
                f.state.audio = .{ .features = f.features.rows(f.audio_off), .rows = f.audio_rows[0..f.audio_off] };
                f.pending_rows = 0;
            },
            .window => {
                const a = try parts(t, 3, f.window);
                const score = std.enums.fromInt(forward_file.Score, a.a[1]) orelse return error.OutOfStep;
                if (a.a[0] != a.tokens.len or a.tokens.len == 0 or a.tokens.len > f.last.len) return error.OutOfStep;
                if (f.state.pos + a.tokens.len > f.capacity) return error.OutOfStep;
                try ranks_forward.forward(f.share, f.state, f.b, f.ranks, a.tokens, score, false);
                // the window's gather is inside the op — the scored rows' halves join here as they did
                // on the lead (no `wide`: the joined halves stay in `Ranks.row` and are dropped)
                const n: usize = switch (score) {
                    .all => a.tokens.len,
                    .last => 1,
                    .none => 0,
                };
                if (n != 0) try held_ns.gather(f.ranks, null, n);
                @memcpy(f.last[0..a.tokens.len], a.tokens);
                f.open = a.tokens.len;
            },
            .commit => {
                const a = try parts(t, 1, f.window);
                if (f.open == 0 or a.a[0] == 0 or a.a[0] > f.open) return error.OutOfStep;
                if (a.tokens.len != f.open or !std.mem.eql(i32, a.tokens, f.last[0..f.open])) return error.OutOfStep;
                try forward_file.commit(f.share, f.state, f.b.*, a.tokens, a.a[0]);
                f.open = 0;
            },
            .row => {
                // the lead's `logitsRow` download — no collective left inside; the args are checked so
                // a step's ops stay counted
                const a = try parts(t, 2, f.window);
                if (a.a[1] == 0 or a.a[0] >= a.a[1] or a.tokens.len != 0) return error.OutOfStep;
            },
            .draft => {
                const a = try parts(t, 2, f.window);
                const d = f.mtp orelse return error.OutOfStep;
                if (a.tokens.len == 0 or a.tokens.len > f.b.rows or a.a[1] != f.state.mtp_pos) return error.OutOfStep;
                // the kept window's streams and ids are this rank's own — the same rows the lead's draft
                // saw, its asks told (`want`); the drafts it makes are discarded here
                const streams = f.b.hn.rows(a.tokens.len);
                const drafter = d.drafter();
                _ = drafter.draft(drafter.context, streams, a.tokens, a.a[1], null, f.drafts[0..@min(a.a[0], f.drafts.len)]) catch |err| {
                    std.log.err("rank 1: MTP draft: {t}", .{err});
                    return error.DraftFailed;
                };
            },
            .flash_taps, .flash_block, .flash_round => {
                // the flash ops' args are the shared `flash_follow`'s
                const n = flash_follow.arguments(flash_pass.Target, op, t.rest) orelse return error.OutOfStep;
                const a = try parts(t, n, f.window);
                try flash_follow.follow(flash_pass.Target, op, a.a, a.tokens, f.flash, null, f.b.rows, f.b.taps);
            },
            else => return error.OutOfStep,
        }
    }
}
