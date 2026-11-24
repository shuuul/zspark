//! `zspark info` on a GLM-5.3-Flash ModelOpt checkpoint (FP4-Local): its config, and each kind of tensor with what a
//! rank of two holds.

const std = @import("std");
const cli = @import("../../../cli.zig");
const glm5_index = @import("../layers/index.zig");
const glm5_layout = @import("../layers/layout.zig");
const common = @import("common.zig");

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

/// Parses `path`'s config.json, finds the model's tensors in its shards and reports both; `list_all` prints
/// every kind of tensor, else the totals by split.
pub fn checkpointInfo(init: std.process.Init, stdout: *std.Io.Writer, path: []const u8, list_all: bool, context: ?u64) !u8 {
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, path, .raise) or !try opened.configure(init)) return 1;
    const c = opened.config;
    try stdout.print("{s}: ModelOpt MIXED_PRECISION, {d} entries in quantized_layers, a linear the map does not name BF16\n", .{ c.model_type, c.quant.layers.len });
    try stdout.print("  model      hidden {d}, vocab {d}, {d} layers ({d} KDA, {d} DSA; {d} dense MLPs, {d} MoE), norm eps {e}\n", .{
        c.hidden,           c.vocab,             c.layers, c.count(.linear_attention), c.count(.deepseek_sparse_attention),
        c.countMlp(.dense), c.countMlp(.sparse), c.eps,
    });
    try stdout.writeAll("  layers    ");
    for (c.layer_types, c.mlp_types) |attention, mlp| {
        try stdout.print(" {c}{c}", .{ @as(u8, if (attention == .linear_attention) 'K' else 'D'), @as(u8, if (mlp == .dense) 'd' else 'm') });
    }
    try stdout.writeAll("\n");
    try stdout.print("  streams    {d} residual streams, {d} mix dots a site, Sinkhorn {d} iterations (eps {e})\n", .{ c.streams, c.mix(), c.sinkhorn, c.hc_eps });
    try stdout.print("  KDA        {d} heads of {d}, conv {d} taps, gate floor {d}\n", .{ c.kda_heads, c.kda_head_dim, c.conv_kernel, c.gate_lower_bound });
    try stdout.print("  DSA        {d} heads, query rank {d}, latent rank {d}, key {d} and value {d} a head, no positions\n", .{ c.heads, c.q_rank, c.kv_rank, c.key_dim, c.value_dim });
    try stdout.print("  indexer    {d} heads of {d}, top {d} tokens as pools of {d}{s}\n", .{ c.index_heads, c.index_dim, c.index_top, c.index_pool, if (c.index_tail) ", the tail kept" else "" });
    if (context) |tokens| try glm5_index.writeCache(stdout, tokens);
    try stdout.print("  MLP        dense width {d}; {d} experts, {d} a token, width {d}, one shared, weights scaled by {d}, swiglu limit {d}\n", .{ c.intermediate, c.experts, c.top_k, c.moe_width, c.routed_scale, c.swiglu_limit });
    try stdout.print("  MTP        {d} layer (DSA and MoE, a plain residual)\n", .{c.mtp_layers});
    try stdout.print("  eos        {any}\n", .{c.eos});

    if (!try opened.resolve(init)) return 1;
    const checkpoint = opened.checkpoint.?;
    const layout = opened.layout;
    try stdout.print("{d} tensors in {d} shards: {d} of the language model, {d:.2} GiB; {d} of the vision tower, {d:.2} GiB, not read\n", .{
        checkpoint.tensors.count(), checkpoint.shards.len, layout.tensors, gib(layout.bytes), layout.dropped, gib(layout.dropped_bytes),
    });
    const schemes = layout.schemes;
    try stdout.print("  linears    {d} BF16, {d} MXFP8, {d} FP8_PB_WO, {d} NVFP4 ({d} of them csf_32x2_v1); {d} map entries of the language model met\n", .{
        schemes.get(.bf16), schemes.get(.mxfp8), schemes.get(.fp8_block), schemes.get(.nvfp4), layout.csf32, layout.mapped,
    });
    var by: std.enums.EnumArray(glm5_layout.Split, glm5_layout.Found) = .initFill(.{});
    for (glm5_layout.kinds, layout.found) |kind, found| {
        const sum = by.getPtr(kind.split);
        sum.count += found.count;
        sum.bytes += found.bytes;
        sum.rank += found.rank;
        if (!list_all) continue;
        var dims: [3]u64 = undefined;
        var text: [40]u8 = undefined;
        var w: std.Io.Writer = .fixed(&text);
        for (kind.shape(c, &dims), 0..) |dim, at| w.print("{s}{d}", .{ if (at > 0) " x " else "", dim }) catch {};
        const top = kind.where == .top and std.mem.startsWith(u8, kind.leaf, glm5_layout.prefix);
        // a linear is stored as the map says, its parts counted with it (the `linears` line)
        const stored: []const u8 = if (kind.linear) "map" else @tagName(kind.dtype);
        try stdout.print("  {s:<44} {s:<5} {s:<16} {t:<6} {d:>6}  {d:>8.3} GiB  a rank {d:>8.3}\n", .{
            kind.leaf[if (top) glm5_layout.prefix.len else 0..], stored,      w.buffered(),
            kind.split,                                          found.count, gib(found.bytes),
            gib(found.rank),
        });
    }
    try stdout.print("a rank of {d} holds:\n", .{glm5_layout.world});
    for (std.enums.values(glm5_layout.Split)) |split| {
        if (split == .drop) continue;
        const sum = by.get(split);
        try stdout.print("  {t:<6} {d:>6} tensors  {d:>8.3} GiB of {d:>8.3}\n", .{ split, sum.count, gib(sum.rank), gib(sum.bytes) });
    }
    try stdout.print("  in all {d:.2} GiB of {d:.2}\n", .{ gib(layout.rank), gib(layout.bytes) });
    return 0;
}
