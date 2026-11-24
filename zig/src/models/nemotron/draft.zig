//! Drafted decoding of Nemotron-3.5-Transcribe's backbone: `zig/src/decode/draft.zig`'s decoder over this
//! model's forward (`namespace`). Its drafter is the MTP head (`mtp.zig`); a reply without one is serial
//! steps, a window of one row each.

const cuda = @import("../../cuda/cuda.zig");
const forward_file = @import("forward.zig");
const shared = @import("../../decode/draft.zig");

/// Nemotron as `shared.Decoder`'s `M`.
pub const namespace = struct {
    pub const Model = forward_file.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Graphs = forward_file.Graphs;
    pub const Error = forward_file.Error;
    pub const forward = forward_file.forward;
    pub const forwardScoring = forward_file.forwardScoring;
    pub const commit = forward_file.commit;
    pub const logitsRow = forward_file.logitsRow;
    pub const logitRows = forward_file.logitRows;
    pub const streams = forward_file.streams;
};

pub const forward_max_rows = shared.forward_max_rows;
pub const Job = shared.Job;
pub const Error = shared.Error(namespace);
pub const Chooser = shared.Chooser(namespace);
pub const Decoder = shared.Decoder(namespace);
pub const Drafter = shared.Drafter;
