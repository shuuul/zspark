//! Nemotron-3.5-Transcribe's side of the shared DFlash2 drafter (`decode/dflash2/`, spec 151 WS-07): the
//! `Target` its generics take (`decode/dflash2/pass.zig`'s header) and the names bound to it. One rank: no
//! peer leads the passes and no job table batches them, so `Op` has no caller, `wide` is false (a prompt's
//! taps are added `batch` rows a call, the target's own buffers holding them), and the target is the
//! one-rank model, its whole vocabulary the proposing head's.

const shared = @import("../../../decode/dflash2/pass.zig");
const seats_file = @import("../../../decode/dflash2/seats.zig");

pub const Target = struct {
    pub const ranks = @import("../../../ranks/ranks.zig");
    pub const lead = @import("../../../ranks/lead.zig");
    /// A leading rank's tell; nothing calls it here (one rank), kept for `seats.zig`'s compiled call.
    pub const Op = @import("../ranks_forward.zig").Op;
    /// The target's buffers are `state.Buffers` sized by the window's rows: `batch` rows an update, read back
    /// into the context from the device (no `wide` scratch).
    pub const wide = false;
    pub const Job = @import("../draft.zig").Job;
    /// One rank only runs `initRig`; the two-rank split's held pair (WS-06) runs `rig.initRigHeld` over
    /// each held rank's `Share` (`Of(held).heldBy`).
    pub const initRigHeld = @import("rig.zig").initRigHeld;
    pub const Of = @import("drafter.zig").Of;
    /// The four-bit heads are the rig's once `initRig` hands them over.
    pub const owns_heads = true;
    pub const initRig = @import("rig.zig").initRig;
};

pub fn PairOf(comptime held: usize) type {
    return shared.PairOf(held, Target);
}

pub fn KeptOf(comptime held: usize) type {
    return shared.KeptOf(held, Target);
}

/// The drafter for several sequences at once, where a pool of Nemotron replies lands (none yet: the
/// command line runs one sequence).
pub fn SeatsOf(comptime held: usize) type {
    return seats_file.SeatsOf(held, Target);
}

/// Both drafter ranks in one process, the drafter's own weights split — the target stays one rank.
pub const Pair = PairOf(shared.world);
/// Both ranks' context in one process.
pub const Kept = KeptOf(shared.world);
pub const GatherError = shared.GatherError(Target);
