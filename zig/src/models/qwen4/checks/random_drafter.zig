//! What the Qwen DFlash2 drafter's tests share: a two-layer drafter's `config.json`, and a random drafter
//! beside the forward's random model (`testing.random_model`'s `fx`), as a source of tensors with the published
//! file's names and shapes. Host only: the host tier's tests read it too. The reading and the tensors are
//! `decode/dflash2/random_drafter.zig`'s (spec 154 WS-02).

const std = @import("std");
const shared = @import("../../../decode/dflash2/random_drafter.zig");
const config_file = @import("../../../decode/dflash2/config.zig");

pub const Source = shared.Source;
pub const put = shared.put;

/// A two-layer drafter beside a main model of three layers, hidden 256: every field the parser reads. The
/// window is stored (Qwen), not exclusive.
pub const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","hidden_size":256,"vocab_size":320,
    \\"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":128,
    \\"intermediate_size":256,"rms_norm_eps":1e-05,"rope_parameters":{"rope_theta":10000.0},
    \\"sliding_window":64,"is_causal":false,"num_target_layers":3,
    \\"dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":317,
    \\"selector_rank":64,"selector_top_k":8,"target_layer_ids":[0,2]}}
;

/// The fixture's config for a main model of `hidden`, `vocab` and `layers`, tapped after `taps`. Window stored,
/// one key/value head a query head.
pub fn config(arena: std.mem.Allocator, hidden: usize, vocab: usize, layers: usize, taps: []const u32) !config_file.Config {
    return shared.config(arena, .stored, fixture, .{ .kv_divisor = 1 }, hidden, vocab, layers, taps);
}
