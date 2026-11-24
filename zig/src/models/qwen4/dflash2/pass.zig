//! Qwen3.8 Flash Next's side of the shared DFlash2 drafter (`decode/dflash2/`, spec 136 WS-03): the `Target`
//! its generics take (`decode/dflash2/pass.zig`'s header) and the names bound to it. A leading rank tells the
//! drafter's passes by Qwen's `ranks_forward.Op`; a prompt's update is read where the taps stand, `wide` rows a
//! pass (spec 135 WS-02); the drafter's target is the one-rank model or the held pair (`drafter.zig`), and the
//! rig owns the heads it is given.

const shared = @import("../../../decode/dflash2/pass.zig");
const seats_file = @import("../../../decode/dflash2/seats.zig");
const rig = @import("rig.zig");

pub const Target = struct {
    pub const ranks = @import("../../../ranks/ranks.zig");
    pub const lead = @import("../../../ranks/lead.zig");
    pub const Op = @import("../ranks_forward.zig").Op;
    pub const wide = true;
    pub const Job = @import("../draft.zig").Job;
    pub const Of = @import("drafter.zig").Of;
    pub const owns_heads = true;
    pub const initRig = rig.initRig;
    pub const initRigHeld = rig.initRigHeld;
};

pub fn PairOf(comptime held: usize) type {
    return shared.PairOf(held, Target);
}

pub fn KeptOf(comptime held: usize) type {
    return shared.KeptOf(held, Target);
}

/// The drafter for several sequences at once (`../pool/hall.zig`): a context a slot, and their blocks as one pass.
pub fn SeatsOf(comptime held: usize) type {
    return seats_file.SeatsOf(held, Target);
}

/// Both ranks in one process, as `loopback.zig` runs the main model's.
pub const Pair = PairOf(shared.world);
pub const GatherError = shared.GatherError(Target);
