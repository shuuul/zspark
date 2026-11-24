//! `zspark info` on a DFlash2 drafter's directory: its config, each tensor with what a rank of two holds and
//! how it is held once loaded, and the context's ring. Host only: it reads the file's header.

const std = @import("std");
const cli = @import("../../../cli.zig");
const drafter_config = @import("../../../decode/dflash2/config.zig");
const drafter_layout = @import("../../../decode/dflash2/layout.zig");
const rig = @import("../dflash2/rig.zig");

fn mib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) anyerror!u8 {
    var opened: rig.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, words[0], .raise) or !try opened.configure(init)) return 1;
    const c = opened.config;
    try stdout.print("{s}: GLM-5.3-Flash's drafter, BF16 as stored\n", .{drafter_config.architecture});
    try stdout.print("  model      hidden {d}, vocab {d}, {d} layers, norm eps {e}\n", .{ c.hidden, c.vocab, c.layers, c.eps });
    try stdout.print("  attention  {d} heads of {d} over {d} KV heads (a rank: {d} and {d}), rotary theta {d}, a window of {?d} context rows, {s}\n", .{
        c.heads, c.head_dim, c.kv_heads, c.rankHeads(), c.rankKvHeads(), c.theta, c.window, if (c.causal) "causal" else "a block's rows see each other",
    });
    try stdout.print("  conv       two taps over groups of {d} channels, a kernel a row of {d}\n", .{ c.group, c.kernelWidth() });
    try stdout.print("  MLP        width {d} (a rank: {d})\n", .{ c.intermediate, c.rankWidth() });
    try stdout.print("  block      {d} rows: the pending token and {d} masks ({d}), a draft each\n", .{ c.block, c.block - 1, c.mask });
    try stdout.print("  taps       after layers {any} of {d}, [rows, {d}]\n", .{ c.taps, c.target_layers, c.tapWidth() });
    try stdout.print("  selector   the top {d} of a rank's half head, rows projected to {d}\n", .{ c.top_k, c.selector_rank });

    if (!try opened.resolve(init)) return 1;
    const layout = opened.layout;
    try stdout.print("{d} tensors, {d:.1} MiB; a rank of {d} holds, in MiB (TensorFold's and zspark's storage after load):\n", .{ layout.tensors, mib(layout.bytes), drafter_config.world });
    for (drafter_layout.kinds, layout.found) |kind, found| {
        var dims: [3]u64 = undefined;
        var text: [40]u8 = undefined;
        var w: std.Io.Writer = .fixed(&text);
        for (kind.shape(c, &dims), 0..) |dim, at| w.print("{s}{d}", .{ if (at > 0) " x " else "", dim }) catch {};
        try stdout.print("  {s:<44} {s:<16} {t:<6} {d:>2}  stored {d:>8.2}  {t:<4} {d:>8.2}  {t:<4} {d:>8.2}\n", .{
            kind.leaf,       w.buffered(),                           kind.split,   found.count,                         mib(found.rank),
            kind.tensorfold, mib(kind.tensorfold.bytes(found.rank)), kind.storage, mib(kind.storage.bytes(found.rank)),
        });
    }
    try stdout.print("a rank's device bytes: {d:.1} MiB as stored, {d:.1} as TensorFold holds it, {d:.1} as zspark does; the codebooks, {d:.1} MiB of fp32 on the host\n", .{
        mib(layout.device(false, true)), mib(layout.device(true, false)), mib(layout.device(false, false)), mib(layout.host()),
    });
    const ring = c.ringRows();
    try stdout.print("the context: a ring of {d} rows (the window, a block and a tile of {d}, in whole tiles), K and V bf16 over {d} layers and {d} KV heads: {d:.1} MiB a rank\n", .{
        ring, drafter_config.tile, c.layers, c.rankKvHeads(), mib(c.contextBytes(ring)),
    });
    return 0;
}
