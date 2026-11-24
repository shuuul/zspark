//! Drafted decoding of GLM-5.3-Flash: `zig/src/decode/draft.zig`'s decoder over both ranks in one process
//! (`namespace`, `loopback.zig`'s forward), with the MTP layer as its drafter (`mtp.zig`). What a caller here
//! names is that file's, instantiated once. Below, the serial reply the tests hold the decoder to.

const std = @import("std");
const cuda = @import("../../cuda/cuda.zig");
const held_file = @import("layers/ranks_forward.zig");
const kda = @import("layers/kda.zig");
const loopback = @import("layers/loopback.zig");
const sampling = @import("../../decode/sampling.zig");
const shared = @import("../../decode/draft.zig");

/// GLM over the ranks a process holds as `shared.Decoder`'s `M`: each name is the model's own, so a round's
/// calls are direct.
pub fn Namespace(comptime held: usize) type {
    return struct {
        const forward_file = held_file.Held(held);
        pub const Model = forward_file.Model;
        pub const State = forward_file.State;
        pub const Buffers = forward_file.Buffers;
        pub const Graphs = forward_file.Graphs;
        pub const Error = held_file.Error;
        pub const forward = forward_file.forward;
        pub const forwardScoring = forward_file.forwardScoring;
        pub const commit = forward_file.commit;
        pub const logitsRow = forward_file.logitsRow;
        pub const settle = forward_file.settle;
        /// A prompt's windows end at the KDA chunks' starts (spec 116 WS-06).
        pub const prompt_align = kda.chunk;

        /// What the MTP layer reads of a row: its final normed row, the same bits on every rank; the first's.
        pub fn streams(b: Buffers, rows: usize) cuda.View {
            return b.ranks[0].final.rows(rows);
        }
    };
}

/// Both ranks in one process (`loopback.zig`): the tests' decoder, and the reference for a rank a process.
pub const namespace = Namespace(2);

pub const Job = shared.Job;

pub const Drafter = shared.Drafter;
pub const Stats = shared.Stats;
pub const Chain = shared.Chain;
pub const forward_max_rows = shared.forward_max_rows;
pub const Error = shared.Error(namespace);
pub const Chooser = shared.Chooser(namespace);
pub const Decoder = shared.Decoder(namespace);
