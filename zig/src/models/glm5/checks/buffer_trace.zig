//! What GLM's `zspark ranks-check` reads, for `zig/src/ranks/trace.zig`'s trace: shared by the main model's
//! check (`ranks_check.zig`) and the drafter's (`ranks_check_drafter.zig`).

const trace = @import("../../../ranks/trace.zig");

pub const What = enum {
    attention,
    mlp,
    streams,
    taps,
    hidden,
    final,
    logits,
    kda,
    latents,
    mtp_attention,
    mtp_mlp,
    mtp_stream,
    mtp_final,
    mtp_logits,
    drafter_context,
    drafter_kv_rows,
    drafter_attention,
    drafter_mlp,
    drafter_rows,
    drafter_final,
    drafter_logits,
    drafter_selected,
    drafter_tops,
    drafter_k,
    drafter_v,
    drafter_candidates,
    drafter_values,
    drafter_drafts,

    pub fn text(self: What) []const u8 {
        return switch (self) {
            .attention => "the attention's gathered partials",
            .mlp => "the MLP's gathered partials",
            .streams => "the residual rows",
            .taps => "the taps",
            .hidden => "the streams' mean after the last layer",
            .final => "the final rows",
            .logits => "the rank's half of the logits",
            .kda => "the committed KDA state",
            .latents => "the latents",
            .mtp_attention => "the MTP layer's gathered attention partials",
            .mtp_mlp => "the MTP layer's gathered MoE partials",
            .mtp_stream => "the MTP layer's rows",
            .mtp_final => "the MTP layer's hidden row",
            .mtp_logits => "the MTP layer's half of the logits",
            .drafter_context => "the drafter's context rows of an update",
            .drafter_kv_rows => "the drafter's K and V rows of an update",
            .drafter_attention => "the drafter's gathered attention partials",
            .drafter_mlp => "the drafter's gathered MLP partials",
            .drafter_rows => "the drafter's block rows",
            .drafter_final => "the drafter's normed draft rows",
            .drafter_logits => "the drafter's half of the drafts' logits",
            .drafter_selected => "the drafter's projected rows",
            .drafter_tops => "the drafter's gathered candidates (a rank's values, then its tokens)",
            .drafter_k => "the drafter's context K",
            .drafter_v => "the drafter's context V",
            .drafter_candidates => "the merged candidates' tokens",
            .drafter_values => "the merged candidates' logits",
            .drafter_drafts => "the drafts chosen",
        };
    }

    pub fn gathered(self: What) bool {
        return switch (self) {
            .attention, .mlp, .mtp_attention, .mtp_mlp, .drafter_attention, .drafter_mlp, .drafter_tops => true,
            else => false,
        };
    }
};

pub const Trace = trace.Trace(What);
pub const atEnd = Trace.atEnd;
pub const differs = Trace.differs;
