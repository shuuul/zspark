//! `rounds.zig`'s round where the rank holds a share (spec 094): where a block's out projection and the
//! experts' slots go. Each writes its fp32 partial, the ranks' partials are gathered and added rank 0 first,
//! and the sum is rounded once, as `ranks_forward.zig`'s `launch` runs a lone stream's layer; the rows are the
//! round's, every stream's together, so a row's sum is its own. One rank passes none of this and its round is
//! `compute`'s as it was. `rounds_ranks.zig` is the round that passes it.
//!
//! A round of `ranks_forward.split_rows` rows or more, a prompt's chunk, splits its glue by rows over the ranks
//! from the last PLE on (spec 149 WS-02), as a pair's window does (spec 106 WS-02, `ranks_forward.Glued`): each
//! rank writes back and reads out its own rows, shares them before each block and the MoE, and receives only
//! its own rows of the other's partials. Every glue kernel works a row at a time and the partials are summed
//! rank 0 first as before, so a row's bits are the whole glue's, which are its stream's alone.

const cuda = @import("../../../cuda/cuda.zig");
const ranks_file = @import("../../../ranks/ranks.zig");
const forward = @import("../forward.zig");
const hc = @import("../blocks/hc.zig");
const model = @import("../model.zig");
const moe = @import("../blocks/moe.zig");
const partials = @import("../partials.zig");
const held_file = @import("../ranks_forward.zig");
const share_file = @import("../share.zig");

pub const Glued = held_file.Glued;

pub const Error = cuda.Error || ranks_file.Error;

/// A rank's place among the partials of a round. The buffers are `ranks_forward.zig`'s `Buffers`', made for
/// the round's most rows and used tight (`ranks/ranks.zig`).
pub const Parts = struct {
    /// fp32 [world, rows, hidden]: each rank's branch partial.
    branch: cuda.View,
    /// fp32 [world, rows, hidden]: each rank's folded MoE partial.
    moe: cuda.View,
    /// bf16 [rows, hidden]: the branch sum, rounded once.
    branch_bf16: cuda.Buffer,
    /// fp32 [rows, hidden]: the MoE sum before its write-back rounds it.
    moe_sum: cuda.Buffer,
    /// What brings the other rank's partial. `.loopback` leaves it at zero: this rank alone, the add still run.
    gather: ranks_file.Gather,
    /// This rank, 0 or 1.
    rank: usize,

    /// Where this rank's block writes its branch for a round of `rows` rows: fp32 [rows, hidden].
    pub fn part(self: Parts, rows: usize) cuda.View {
        return ranks_file.partial(self.branch, self.rank, rows);
    }

    /// The other rank's partial of `buffer` beside this one's, both ranks' as the reader takes them.
    fn gathered(self: Parts, buffer: cuda.View, rows: usize) Error!cuda.View {
        try cuda.zeroLaunched(ranks_file.partial(buffer, 1 - self.rank, rows));
        const all = ranks_file.gathered(buffer, rows);
        if (self.gather != .loopback) try self.gather.run(all);
        return all;
    }

    /// After the block wrote `part`: the ranks' sum as the bf16 branch the next write-back takes.
    pub fn landed(self: Parts, rows: usize, inject: cuda.View) Error!hc.Pending {
        try partials.roundSum(try self.gathered(self.branch, rows), self.branch_bf16.rows(rows));
        return .{ .branch = .{ .branch = self.branch_bf16.rows(rows), .inject = inject } };
    }

    /// After `moe.forward`: this rank's slots folded by weight into its partial, and the ranks' partials as
    /// what the next write-back adds and rounds.
    pub fn folded(self: Parts, buffers: moe.Buffers, rows: usize, slots: usize, inject: cuda.View) Error!hc.Pending {
        try partials.fold(buffers.y.rows(rows * slots), buffers.shared_y.rows(rows), buffers.weights.rows(rows), ranks_file.partial(self.moe, self.rank, rows));
        return .{ .ranks_sum = .{ .gathered = try self.gathered(self.moe, rows), .inject = inject, .into = self.moe_sum.rows(rows) } };
    }

    /// This rank's rows of a round of `rows` rows whose glue splits (`ranks_forward.splits`' rule: over NCCL,
    /// from `split_rows`, every rank given rows; not where a drafter's taps read every row's streams, nor in a
    /// captured round, whose replies are fewer rows), else null. Both ranks decide alike from what they share.
    pub fn glued(self: Parts, b: forward.Buffers, rows: usize, captured: bool) ?Glued {
        if (self.gather != .nccl or captured or rows < held_file.split_rows or b.tap_layers.len > 0) return null;
        for (0..share_file.world) |rank| {
            const own = ranks_file.share(rows, share_file.world, rank);
            if (own[1] == own[0]) return null;
        }
        return .of(b, rows, self.rank);
    }

    /// Before a split layer's block: `pending` (this rank's rows') written back and the read-out of `connection`,
    /// every rank's rows of `mixed` then on every rank, which the block reads whole.
    pub fn glueIn(self: Parts, own: Glued, b: forward.Buffers, rows: usize, connection: model.HyperConnection, shape: model.Shape, pending: hc.Pending) Error!void {
        try own.before(connection, shape, pending);
        try self.gather.shareRows(b.mixed.rows(rows));
    }

    /// After a split layer's block wrote `part`: this rank's rows of the other's partial brought, the sum
    /// written back, the MLP's read-out (`connection`) of them, and `mixed` shared for the experts.
    pub fn landedRows(self: Parts, own: Glued, b: forward.Buffers, rows: usize, connection: model.HyperConnection, shape: model.Shape) Error!void {
        const all = ranks_file.gathered(self.branch, rows);
        try self.gather.scatter(all);
        try own.landed(all, self.branch_bf16, connection, shape);
        try self.gather.shareRows(b.mixed.rows(rows));
    }

    /// After a split layer's `moe.forward`: this rank's slots folded into its partial, this rank's rows of the
    /// other's brought, and their sum, the landing the next split layer writes back.
    pub fn foldedRows(self: Parts, own: Glued, buffers: moe.Buffers, rows: usize, slots: usize) Error!hc.Pending {
        try partials.fold(buffers.y.rows(rows * slots), buffers.shared_y.rows(rows), buffers.weights.rows(rows), ranks_file.partial(self.moe, self.rank, rows));
        const all = ranks_file.gathered(self.moe, rows);
        try self.gather.scatter(all);
        return own.folded(all, self.moe_sum);
    }

    /// After the last split layer: `pending` written back, the final read-out, and every rank's rows of `mixed`
    /// and `h` on every rank: what the head, the MTP head and a state kept read whole (`launch`'s split end).
    pub fn glueEnd(self: Parts, own: Glued, b: forward.Buffers, rows: usize, mixer: model.HyperConnection, shape: model.Shape, pending: hc.Pending) Error!void {
        try own.end(mixer, shape, pending);
        try self.gather.shareRows(b.mixed.rows(rows));
        try self.gather.shareRows(b.h.rows(rows));
    }
};
