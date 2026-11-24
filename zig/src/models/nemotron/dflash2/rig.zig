//! Nemotron-3.5-Transcribe's DFlash2 drafter beside the MTP head (`commands/generate.zig`'s `drafter=DIR`;
//! spec 151 WS-07): `decode/dflash2/rig.zig` with what is Nemotron's. The directory's opening and `fit`
//! live in `../dflash2.zig` (the null window's `parseFull`); the target's taps are `state.Buffers.setTaps`
//! with the config's, the stream itself bf16 — one residual stream, no mean. `heads` (`drafter.draftHeads`)
//! are the rig's once `finish` takes them (`Target.owns_heads`).

const std = @import("std");
const Allocator = std.mem.Allocator;
const cuda = @import("../../../cuda/cuda.zig");
const sampling = @import("../../../decode/sampling.zig");
const shared = @import("../../../decode/dflash2/rig.zig");
const weights = @import("../../../decode/dflash2/weights.zig");
const opened_file = @import("../dflash2.zig");
const model_file = @import("../model.zig");
const state_file = @import("../state.zig");
const drafter_file = @import("drafter.zig");
const pass = @import("pass.zig");

pub const Opened = opened_file.Opened;
pub const Source = shared.Source;
pub const Early = shared.Early;
pub const name = shared.name;
pub const published = opened_file.published;
pub const fit = opened_file.fit;
pub const enter = opened_file.enter;
pub const given = opened_file.given;

/// How `generate drafter=DIR` holds the drafter's linears and proposing head: `.four` copies made at load
/// (the default, TensorFold's `draft_head`), or BF16 as stored (`drafter_stored`). The reply's tokens are
/// the same; the measurement of drafts kept and tokens a second is the spec's gate.
pub var drafter_stored: bool = false;

/// The drafter's share, context and passes over `held` ranks.
pub fn Rig(comptime held: usize) type {
    return shared.Rig(held, pass.Target);
}

/// `Rig.init`: the one-rank target `m` and its buffers `b`, not yet tapped and no graph captured over them;
/// `capacity` the positions of a sequence. `heads` (`drafter.draftHeads`) are the rig's once the pair is
/// made: freed with it, or where the drafter fails (`Target.owns_heads`). Nothing leads or gathers: one
/// rank, the loopback gather.
pub fn initRig(self: anytype, gpa: Allocator, opened: *const Opened, heads: anytype, widened: anytype, m: anytype, b: anytype, capacity: usize, settings: sampling.Sampling, eager: bool) !void {
    const c = &opened.config;
    try b.setTaps(c.taps);
    try self.begin(gpa, c, capacity, 0, .loopback, null, eager);
    errdefer self.abandon();
    try self.finish(gpa, c, drafter_file.Of(heads.len).whole(m, b), heads, widened, settings);
}

/// `Rig.initHeld`: the held ranks of a two-rank split (WS-06): `shares` an array of
/// `{ .rank, .m, .b }` (the rank's own `Model`, `Buffers` and its place of the two), `first` the lowest
/// rank number held, `gather` the pair's (`ranks_forward.Ranks.gather`), `lead` where this rank tells
/// the drafter's passes. Each held rank's buffers are tapped with the config's; the target reads each
/// rank's own embedding, its half of `lm_head` and its taps (`Of(held).heldBy`, `draftHeads` on the
/// rank's `m.head`, which the split already holds halved).
pub fn initRigHeld(self: anytype, gpa: Allocator, opened: *const Opened, heads: anytype, widened: anytype, shares: anytype, first: usize, gather: anytype, lead: anytype, capacity: usize, settings: sampling.Sampling, eager: bool) !void {
    const c = &opened.config;
    const Of = @import("drafter.zig").Of(heads.len);
    var target_shares: [heads.len]Of.Share = undefined;
    for (&target_shares, shares) |*one, share| {
        try share.b.setTaps(c.taps);
        one.* = .{ .rank = share.rank, .m = share.m, .b = share.b };
    }
    try self.begin(gpa, c, capacity, first, gather, lead, eager);
    errdefer self.abandon();
    try self.finish(gpa, c, Of.heldBy(target_shares), heads, widened, settings);
}

// `initRig` builds the same `Of` whole: `Target.finish`'s `target` is `Of(held).whole(m, b)`.
