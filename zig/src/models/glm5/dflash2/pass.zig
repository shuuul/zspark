//! GLM-5.3-Flash's side of the shared DFlash2 drafter (`decode/dflash2/`, spec 136 WS-03): the `Target` its
//! generics take (`decode/dflash2/pass.zig`'s header) and the names bound to it. A leading rank tells the
//! drafter's passes by GLM's `ranks_forward.Op`; a prompt's update is `batch` rows copied in, captured where
//! the pair has graphs; the drafter's target is the held pair (`drafter.zig`), whose heads are the session's.

const shared = @import("../../../decode/dflash2/pass.zig");
const seats_file = @import("../../../decode/dflash2/seats.zig");

pub const Target = struct {
    pub const ranks = @import("../../../ranks/ranks.zig");
    pub const lead = @import("../../../ranks/lead.zig");
    pub const Op = @import("../layers/ranks_forward.zig").Op;
    pub const wide = false;
    pub const Job = @import("../draft.zig").Job;
    pub const Of = @import("drafter.zig").Of;
    /// The four-bit heads are the session's (`session/session.zig`), not the rig's.
    pub const owns_heads = false;
    pub const initRig = @import("rig.zig").initRig;
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
/// Both ranks' context in one process.
pub const Kept = KeptOf(shared.world);
pub const GatherError = shared.GatherError(Target);
