//! A row's precision form by its window's kind, the rule both FP4-Local checkpoints take from their cards' vLLM
//! (GLM spec 115, Flash Next spec 143 WS-06; moved here from GLM's `layers/forward.zig` when Flash Next took it).
//! A window of more than a model's bound of rows is a prompt's: its rows take the quantized forms (MXFP8 linears
//! on FP8 rows, routed experts W4A4 where a model has them, each as its card's prompt path quantizes). A step, a
//! verify window (at most `draft.forward_max_rows`) and a prompt window of up to the bound take bf16 rows, as the
//! cards run the batches their CUDA graphs hold (MoE A16 there: b12x drops its A4 prompt parts under a capture) and
//! the MXFP8 plans they may tune A16 (their capture sizes). Every row of a window takes its window's form, and
//! within a form a row's bits are its own: a verify window and a serial step are one form, so drafting stays exact.
//! The bound is the model's (its card's capture sizes, a test's `oneForm`); what a model quantizes in every phase
//! (GLM's block-FP8 linears) is the model's, not this rule's.

const std = @import("std");

/// The widest window a model's bound may be under: a reply's verify window and a serial step are one form.
pub fn checkBound(comptime card_rows: usize) void {
    std.debug.assert(@import("draft.zig").forward_max_rows <= card_rows);
}

/// A call outside every window's form: none of its rows quantized (a drafter's head, a load's checks).
pub const none: usize = std.math.maxInt(usize);

/// The first row of a window of `rows` rows that takes the quantized forms under a bound of `bf16_rows`: 0, or
/// `rows` where none does.
pub fn quantizedFrom(rows: usize, bf16_rows: usize) usize {
    return if (rows > bf16_rows) 0 else rows;
}

/// The first row of a round of several parts (`Part` has `rows`, or `tokens`) that takes the quantized forms: the
/// rows of its parts before the first part of more than `bf16_rows`. A part of bf16 rows after it would take the
/// quantized forms, so a round orders its bf16 parts first (GLM's `hall`, Flash Next's `rounds.promptBeside`): a
/// reply's window never comes there, and its rows stay a serial step's form.
pub fn formFrom(comptime Part: type, parts: []const Part, bf16_rows: usize) usize {
    var from: usize = 0;
    for (parts, 0..) |part, at| {
        if (rowsOf(part) > bf16_rows) {
            for (parts[at..]) |after| std.debug.assert(rowsOf(after) > bf16_rows);
            return from;
        }
        from += rowsOf(part);
    }
    return from;
}

fn rowsOf(part: anytype) usize {
    return if (@hasField(@TypeOf(part), "rows")) part.rows else part.tokens.len;
}
