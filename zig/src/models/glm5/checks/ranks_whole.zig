//! `zspark ranks-check DIR reference=REF [layers=L] [drafter=DRAFTER] [kv=fp8] [rank=R at=ADDR:PORT]`: the
//! checkpoint on the ranks this process holds against the vLLM container's reference (`ranks_vllm.zig`). With a
//! rank and a place this is one rank of two; without, both ranks are here, which a Spark holds for the first
//! layers alone. The comparison against TensorFold's two-rank export of the EXL3 pack (spec 045 WS-05) was retired
//! with the pack (tag `retired-checkpoints-2026-10-21`).

const std = @import("std");
const cli = @import("../../../cli.zig");
const join_file = @import("../../../ranks/join.zig");
const nccl = @import("../../../ranks/nccl.zig");
const dsa = @import("../layers/dsa.zig");
const rig = @import("../dflash2/rig.zig");
const common = @import("../commands/common.zig");
const ranks_vllm = @import("ranks_vllm.zig");

const Options = struct {
    place: join_file.Place = .{},
    reference: ?[]const u8 = null,
    layers: ?usize = null,
    drafter: ?[]const u8 = null,
    /// The DSA caches' storage (`kv=bf16|fp8`).
    kv: dsa.Storage = .bf16,
};

pub fn run(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) !u8 {
    var options: Options = .{};
    for (words[1..]) |word| {
        if (options.place.take(word)) continue;
        if (cli.value(word, "reference=")) |text| {
            options.reference = text;
        } else if (cli.value(word, "drafter=")) |text| {
            options.drafter = text;
        } else if (cli.value(word, "layers=")) |text| {
            options.layers = std.fmt.parseInt(usize, text, 10) catch return 2;
        } else if (cli.value(word, "kv=")) |text| {
            options.kv = dsa.Storage.parse(text) orelse return 2;
        } else return 2;
    }
    const paired = options.place.rank != null or options.place.at != null;
    if (options.reference == null or (paired and !options.place.complete())) return 2;
    if (paired and !nccl.linked) {
        std.log.err("ranks-check: this build has no NCCL; build with -Dnccl=PREFIX (/usr for the apt packages)", .{});
        return 1;
    }
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.enter(init.io, words[0], .raise) or !try opened.configure(init) or !try opened.resolve(init)) return 1;
    const layers = options.layers orelse opened.config.layers;
    if (layers == 0 or layers > opened.config.layers or (!paired and layers == opened.config.layers)) {
        std.log.err("ranks-check: {d} layers of the model's {d}; the whole model is a rank a Spark (rank=R at=ADDR:PORT)", .{ layers, opened.config.layers });
        return 1;
    }
    var drafting: rig.Opened = .{};
    defer drafting.close(init.io);
    if (options.drafter) |path| if (!try rig.enter(init, &drafting, path, &opened.config, layers)) return 1;
    const drafter: ?*const rig.Opened = if (options.drafter != null) &drafting else null;
    // vLLM's replies and score (FP4-Local, spec 107 WS-02), drafted by DFlash2 too (WS-04)
    if (ranks_vllm.holds(init.io, options.reference.?)) return ranks_vllm.run(init, stdout, &opened, drafter, options.place, options.reference.?, layers, options.kv);
    std.log.err("ranks-check: {s} holds no vLLM reference (NAME.json, score.json; tools/reference/vllm_reference_ranks.sh)", .{options.reference.?});
    return 1;
}
