//! Drafted decoding of Flash Next: `zig/src/decode/draft.zig`'s decoder over this model's forward (`namespace`), with the
//! MTP head as its drafter (`mtp.zig`) or DFlash2 (`dflash2/`) in that place. What a caller here names (`Decoder`,
//! `Chooser`, `Drafter`, `Chain`) is that file's, instantiated once; this file adds nothing to a round. Below them,
//! the serial replies the tests hold the decoder to, written on the forward alone.

const cuda = @import("../../cuda/cuda.zig");
const forward_file = @import("forward.zig");
const forward = forward_file;
const model = @import("model.zig");
const sampling = @import("../../decode/sampling.zig");
const shared = @import("../../decode/draft.zig");

/// Flash Next as `shared.Decoder`'s `M`: each name is the model's own, so a round's calls are direct.
pub const namespace = struct {
    pub const Model = model.Model;
    pub const State = forward_file.State;
    pub const Buffers = forward_file.Buffers;
    pub const Graphs = forward_file.Graphs;
    pub const Error = forward_file.Error;
    pub const forward = forward_file.forward;
    pub const forwardScoring = forward_file.forwardScoring;
    pub const commit = forward_file.commit;
    pub const logitsRow = forward_file.logitsRow;

    pub fn streams(b: forward_file.Buffers, rows: usize) cuda.View {
        return b.h.rows(rows);
    }

    /// Rows `row` to `row + n` of the window's logits, bf16 [n, vocab], for a reply's log probabilities.
    pub fn logitRows(b: forward_file.Buffers, row: usize, n: usize) cuda.View {
        return b.logits.span(row, n);
    }
};

pub const Job = shared.Job;

pub const Drafter = shared.Drafter;
pub const Stats = shared.Stats;
pub const Follow = shared.Follow;
pub const Chain = shared.Chain;
pub const forward_max_rows = shared.forward_max_rows;
pub const Error = shared.Error(namespace);
pub const Chooser = shared.Chooser(namespace);
pub const Logprobs = shared.Logprobs;
pub const Decoder = shared.Decoder(namespace);
