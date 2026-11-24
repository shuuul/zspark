//! `zspark info`: a checkpoint's config and where its weights are.

const std = @import("std");
const qwen4_config = @import("../config.zig");
const qwen4_layout = @import("../layout.zig");
const qwen4_share = @import("../share.zig");
const common = @import("common.zig");

/// Parses `path`'s config.json, resolves the model's tensors in its shards and reports both.
/// `words` may hold `all` (every missing tensor) and `rank=R/2` (what that rank holds).
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const words = argv[1..];
    const gib = 1024.0 * 1024.0 * 1024.0;
    var list_all = false;
    var rank: ?qwen4_share.Share = null;
    for (words) |word| {
        if (std.mem.eql(u8, word, "all")) {
            list_all = true;
        } else if (std.mem.startsWith(u8, word, "rank=")) {
            rank = shareOf(word) orelse {
                std.log.err("{s}: a rank is 0/2 or 1/2", .{word});
                return 1;
            };
        } else {
            std.log.err("info takes 'all' and rank=R/2, not '{s}'", .{word});
            return 1;
        }
    }
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, path, .raise) or !try opened.configure(init)) return 1;
    const c = opened.config;
    var kinds: std.enums.EnumArray(qwen4_config.LayerType, usize) = .initFill(0);
    for (c.layer_types) |kind| kinds.getPtr(kind).* += 1;
    var algos: std.enums.EnumArray(qwen4_config.Algo, usize) = .initFill(0);
    for (c.quant.layers) |layer| algos.getPtr(layer.algo).* += 1;
    try stdout.print("{s}: ModelOpt {s}; quantized modules: {d} NVFP4 (blocks of {d}), {d} block FP8, {d} FP8, {d} MXFP8\n", .{
        c.model_type, c.quant.algo, algos.get(.nvfp4), c.quant.nvfp4_group, algos.get(.fp8_block), algos.get(.fp8), algos.get(.mxfp8),
    });
    try stdout.print("  model      hidden {d}, vocab {d}, {d} layers ({d} DeltaNet, {d} attention), norm eps {e}\n", .{
        c.hidden, c.vocab, c.layers, kinds.get(.linear_attention), kinds.get(.full_attention), c.eps,
    });
    try stdout.print("  streams    {d} residual streams, hyper-connection rank {d}\n", .{ c.streams, c.low });
    try stdout.print("  attention  {d} heads, {d} kv heads, head dim {d}, rotary {d} of it (theta {d}, sections {any})\n", .{
        c.heads, c.kv_heads, c.head_dim, c.rotary_dim, c.rope_theta, c.mrope_section,
    });
    try stdout.print("  indexer    {d} query + {d} key heads of {d}, budget {d}, ratio {d} ({d} blocks)\n", .{
        c.index_heads, c.index_kv_heads, c.index_dim, c.index_budget, c.index_ratio, c.topBlocks(),
    });
    try stdout.print("  DeltaNet   {d} key heads of {d}, {d} value heads of {d}, conv {d} channels x {d} taps\n", .{
        c.key_heads, c.key_head_dim, c.value_heads, c.value_head_dim, c.convDim(), c.conv_kernel,
    });
    try stdout.print("  MoE        {d} experts, {d} a token, width {d}, shared expert width {d}\n", .{ c.experts, c.top_k, c.moe_width, c.shared_width });
    if (c.ple_layers.len > 0) {
        try stdout.print("  PLE        layers {any}: {d}-grams, {d} heads of {d}, {d} shards, sizes from {d}, conv {d} taps\n", .{
            c.ple_layers, c.ngram_size, c.ngramHeads(), c.ngramDims(), c.ngram_shards, c.ngram_base, c.ple_kernel,
        });
    }
    if (c.mtp) |mtp| {
        try stdout.print("  MTP        {d} layer ({t}), theta {d}\n", .{ mtp.layers, mtp.layer_types[0], mtp.rope_theta });
    } else try stdout.writeAll("  MTP        none\n");
    try stdout.print("  eos        {any}\n", .{c.eos});

    if (!try opened.resolve(init)) return 1;
    const checkpoint = opened.checkpoint.?;
    const layout = opened.layout;
    try stdout.print("{d} tensors in {d} shards, the language model under {s}\n", .{ checkpoint.tensors.count(), checkpoint.shards.len, layout.prefix });
    for (std.enums.values(qwen4_layout.Scheme)) |scheme| {
        try stdout.print("  {t:<10} {d:>6} linears  {d:>8.3} GiB\n", .{ scheme, layout.stats.linears.get(scheme), @as(f64, @floatFromInt(layout.stats.bytes.get(scheme))) / gib });
    }
    if (layout.stats.csf != 0) try stdout.print("  {d} of the nvfp4 linears with csf-encoded block scales\n", .{layout.stats.csf});
    try stdout.print("  {s:<10} {d:>6} tensors  {d:>8.3} GiB\n", .{ "vectors", layout.stats.small, @as(f64, @floatFromInt(layout.stats.small_bytes)) / gib });
    if (layout.stats.table_scheme) |scheme| {
        try stdout.print("  {s:<10} {d:>6} shards   {d:>8.3} GiB ({t})\n", .{ "n-grams", layout.stats.table_shards, @as(f64, @floatFromInt(layout.stats.table_bytes)) / gib, scheme });
    }
    try stdout.print("norm weights: {s}\n", .{if (layout.norms_centred) |centred|
        (if (centred) "centred (gamma - 1)" else "scales (gamma)")
    else
        "unknown (no hc_norm yet)"});
    const state = [2][]const u8{ "missing", "found" };
    try stdout.print("found: {d} of {d} layers; embedding {s}, mixer {s}, head {s}, MTP {s}\n", .{
        layout.layersFound(),
        c.layers,
        state[@intFromBool(layout.embed != null)],
        state[@intFromBool(layout.mixer != null)],
        state[@intFromBool(layout.head != null)],
        if (c.mtp == null) "not in the config" else state[@intFromBool(layout.mtp != null)],
    });
    var under_mtp: usize = 0;
    for (layout.missing) |name| under_mtp += @intFromBool(std.mem.startsWith(u8, name, "mtp."));
    try stdout.print("missing: {d} tensors ({d} of the MTP head)\n", .{ layout.missing.len, under_mtp });
    const shown = if (list_all) layout.missing.len else @min(layout.missing.len, 8);
    for (layout.missing[0..shown]) |name| try stdout.print("  {s}\n", .{name});
    if (shown < layout.missing.len) try stdout.print("  ... {d} more\n", .{layout.missing.len - shown});
    if (rank) |share| {
        const sums = qwen4_share.sumsOf(&checkpoint);
        try stdout.print("rank {d} of {d}\n", .{ share.rank, qwen4_share.world });
        try qwen4_share.report(stdout, sums);
    }
    return 0;
}

fn shareOf(word: []const u8) ?qwen4_share.Share {
    const spec = word["rank=".len..];
    const slash = std.mem.indexOfScalar(u8, spec, '/') orelse return null;
    const rank = std.fmt.parseInt(u32, spec[0..slash], 10) catch return null;
    const of = std.fmt.parseInt(u32, spec[slash + 1 ..], 10) catch return null;
    if (of != qwen4_share.world or rank >= of) return null;
    return .of(rank);
}
