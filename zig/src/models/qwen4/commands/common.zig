//! What the commands of Flash Next share.

const std = @import("std");
const cli = @import("../../../cli.zig");
const qwen4_config = @import("../config.zig");
const qwen4_draft = @import("../draft.zig");
const qwen4_layout = @import("../layout.zig");
const qwen4_model = @import("../model.zig");
const qwen4_session = @import("../session/session.zig");

/// TensorFold's `DEPTH`: the most drafts a step verifies unless asked.
pub const draft_depth = qwen4_session.default_depth;

/// A checkpoint directory opened as Flash Next's: its config, its tensor files and its layout.
pub const Opened = cli.Opened(qwen4_config.Config, qwen4_layout.Layout, qwen4_config.Diagnostic);

/// Whether a rank of two loads the MTP head (spec 090): where the command drafts, each rank its half
/// (`model.ranks_head`); or, where `ZSPARK_RANKS_HEAD=whole`, rank 0 whole beside its share and rank 1 never,
/// whatever its command line says of drafting. Null, said, where `ZSPARK_DRAFTS_STORED` asks for the head
/// as stored: beside a share its scorer is the draft rows of the whole `lm_head`, the share's own head being
/// half a vocabulary.
pub fn headOnRank(rank: usize, drafts: bool) ?bool {
    if (!drafts or (rank != 0 and qwen4_model.ranks_head != .split)) return false;
    if (qwen4_model.drafts_stored) {
        std.log.err("ZSPARK_DRAFTS_STORED is not read on two ranks: unset it, or decode without drafts (--serial)", .{});
        return null;
    }
    return true;
}

/// What a handshake's `alike` says of the head: nothing where it is whole, rank 1 running nothing for it; split,
/// that it is and whether the command drafts, since both ranks compute a level and one that did not load its
/// half would leave the other waiting in a gather.
pub fn headAlike(drafts: bool) []const u8 {
    if (qwen4_model.ranks_head != .split) return "";
    return if (drafts) " head=split" else " head=split, no drafts";
}

/// What a handshake's `alike` says of the attention caches: the trunk's storage (`--kv-dtype`, or what
/// `layout.storageFor` gives without it). Each rank holds its half of the KV heads in it, and ranks asked
/// for two storages refuse each other at the join.
pub fn cachesAlike(caches: qwen4_layout.Caches) []const u8 {
    return switch (caches.trunk) {
        .bf16 => " kv=bf16",
        .fp8 => " kv=fp8",
        .fp8_nvfp4 => " kv=fp8-nvfp4",
    };
}

/// The flags of decoding that `chat` and `serve` both take; each has the fields they name.
pub const decode_flags = [_]cli.Flag{
    .{ .name = "--serial", .field = "serial" },
    .{ .name = "--no-copies", .field = "no_copies" },
    .{ .name = "--lazy-graphs", .field = "lazy_graphs" },
    .{ .name = "--depth", .field = "depth", .max = qwen4_draft.forward_max_rows - 1 },
    .{ .name = "--reply-stride", .field = "reply_stride" },
    .{ .name = "--kv-dtype", .field = "kv" },
    .{ .name = "--max-num-batched-tokens", .field = "max_num_batched_tokens" },
};

/// The rows a lone stream's prompt window runs on one rank and the model is loaded for: `--max-num-batched-tokens
/// N` where it was given (spec 105 WS-07), else the checkpoint's (`session.lonePromptRows`, 4,096 since spec 148
/// WS-02). vLLM's name for the tokens a step takes, which for a lone prompt is its chunk and sizes the runner's
/// buffers, as these rows do here; SGLang's is `--chunked-prefill-size`. Null, said, where it is refused: outside
/// 2,048 to 4,096 rows.
pub fn promptRows(config: qwen4_config.Config, asked: ?usize) ?usize {
    return rowsAsked(config, asked orelse return qwen4_session.lonePromptRows(config));
}

/// A pair's prompt windows on two ranks (spec 111): as `promptRows`, unset `session.lonePromptRows` (4,096; spec
/// 149 WS-04, the owner 2026-10-28: a cold prompt 9 to 12% sooner for 3.2 GiB more a rank).
pub fn pairRows(config: qwen4_config.Config, asked: ?usize) ?usize {
    return rowsAsked(config, asked orelse return qwen4_session.lonePromptRows(config));
}

fn rowsAsked(config: qwen4_config.Config, rows: usize) ?usize {
    if (qwen4_session.refusedRows(config, rows)) |why| {
        std.log.err("--max-num-batched-tokens {d}: {s}", .{ rows, why });
        return null;
    }
    return rows;
}

/// What one rank loads the model with, whole, for `chat`, `generate` and `serve`: `max_rows` and `mtp` are each
/// command's, the draft rows, `ZSPARK_DRAFTS_STORED` and the caches' storage the same for all three.
pub fn loadOptions(arena: std.mem.Allocator, config: qwen4_config.Config, max_rows: usize, mtp: bool, caches: qwen4_layout.Caches) !qwen4_model.Model.Options {
    return .{ .layers = config.layers, .max_rows = max_rows, .mtp = mtp, .draft_vocab = try qwen4_model.DraftHead.defaultIds(arena), .drafts_stored = qwen4_model.drafts_stored, .kv = caches.trunk, .head_kv = caches.head };
}

/// The rows of a round of the pool on two ranks (spec 149 WS-06): `--max-num-batched-tokens N` where it was
/// given, as a pair's, else `session.promptRows` (2,048; the default is the owner's to change). A round's frame
/// over the link holds its rows and as many of the next chunk's tokens (`Lead.widest`).
pub fn poolRows(config: qwen4_config.Config, asked: ?usize) ?usize {
    return rowsAsked(config, asked orelse return qwen4_session.promptRows(config));
}
