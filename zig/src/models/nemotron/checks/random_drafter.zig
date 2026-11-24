//! What Nemotron's DFlash2 drafter tests share (spec 151 WS-07): a two-layer drafter's `config.json` of the
//! published drafter's schema — `sliding_window` null and a GQA whose q_inner is not hidden — and a random
//! drafter as a source of tensors with the published file's names and shapes. Host only: the host tier's
//! tests read it too. The reading and the tensors are `decode/dflash2/random_drafter.zig`'s (spec 154 WS-02).

const std = @import("std");
const shared = @import("../../../decode/dflash2/random_drafter.zig");
const config_file = @import("../../../decode/dflash2/config.zig");

pub const Source = shared.Source;
pub const put = shared.put;

/// A two-layer drafter beside a main model of four layers, hidden 512, its queries 1,024 wide (GQA, as the
/// published drafter's 4,096 over hidden 2,688): every field the parser reads, the window null.
pub const fixture =
    \\{"architectures":["DFlash2DraftModel"],"model_type":"qwen3","hidden_size":512,"vocab_size":128,
    \\"num_hidden_layers":2,"num_attention_heads":8,"num_key_value_heads":2,"head_dim":128,
    \\"intermediate_size":256,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000.0},
    \\"sliding_window":null,"is_causal":false,"num_target_layers":4,
    \\"dflash_config":{"block_size":4,"conv_group_size":16,"conv_kernel_size":2,"mask_token_id":120,
    \\"selector_rank":64,"selector_top_k":8,"target_layer_ids":[1,3]}}
;

/// The fixture's config for a main model of `hidden`, `vocab` and `layers`, tapped after `taps`; the drafter
/// keeps Nemotron's GQA (q_inner is twice hidden) and its null window.
pub fn config(arena: std.mem.Allocator, hidden: usize, vocab: usize, layers: usize, taps: []const u32) !config_file.Config {
    return shared.config(arena, .full, fixture, .{ .heads_scale = 2 }, hidden, vocab, layers, taps);
}
