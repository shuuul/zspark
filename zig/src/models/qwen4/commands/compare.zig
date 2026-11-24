//! `zspark compare`: the forward against TensorFold's reference tensors.

const std = @import("std");
const bf16 = @import("../../../weights/bf16.zig");
const experts = @import("../../../weights/experts.zig");
const safetensors = @import("../../../weights/safetensors.zig");
const qwen4_compare = @import("../compare.zig");
const qwen4_forward = @import("../forward.zig");
const qwen4_model = @import("../model.zig");
const cli = @import("../../../cli.zig");
const common = @import("common.zig");

/// Runs a reference file's ids as one window and reports, for each kind of point, how far the forward is from
/// TensorFold's tensors. Each block starts from the reference's input unless `free`. `rows` adds one point's
/// distance by row and layer, with each row's first token here and there; `perturb=SIZE[@LAYER[:ROW]]` then runs
/// free `seeds=N` times more with noise added to a layer's input, each against the run without it: what a
/// difference of that size becomes by the logits.
pub fn run(init: std.process.Init, stdout: *std.Io.Writer, argv: []const []const u8) !u8 {
    const path = argv[0];
    const reference_path = argv[1];
    const options = argv[2..];
    const arena = init.arena.allocator();
    var free = false;
    var by_point: ?qwen4_forward.Point = null;
    var noise: ?qwen4_compare.Perturb = null;
    var seeds: usize = 1;
    var first_layers: ?usize = null;
    for (options) |option| {
        if (std.mem.eql(u8, option, "free")) {
            free = true;
        } else if (cli.value(option, "layers=")) |count| {
            first_layers = try std.fmt.parseInt(usize, count, 10);
        } else if (std.mem.eql(u8, option, "rows")) {
            by_point = .h_in;
        } else if (cli.value(option, "rows=")) |point| {
            by_point = std.meta.stringToEnum(qwen4_forward.Point, point) orelse {
                std.log.err("{s}: no such point; one of embed_h, h_in, attn_mixed, projection, block_out, branch, h_mid, mlp_mixed, moe_logits, moe_weights, moe_y", .{option});
                return 2;
            };
        } else if (cli.value(option, "perturb=")) |given| {
            noise = try qwen4_compare.Perturb.parse(given);
        } else if (cli.value(option, "seeds=")) |count| {
            seeds = try std.fmt.parseInt(usize, count, 10);
        } else {
            std.log.err("compare: {s} is not free, layers=N, rows[=POINT], perturb=SIZE[@LAYER[:ROW]] or seeds=N", .{option});
            return 2;
        }
    }
    if (noise != null and !free) {
        std.log.err("compare: perturb runs free; add free", .{});
        return 2;
    }
    var opened: common.Opened = .{};
    defer opened.close(init.io);
    if (!try opened.open(init, path, .raise)) return 1;
    const config = opened.config;
    const found = opened.layout;
    var reference: safetensors.File = safetensors.File.open(init.gpa, init.io, .cwd(), reference_path) catch |err| {
        switch (err) {
            error.IsDir => std.log.err("{s} is a directory: REF is one reference file in it, NAME.serial.f32.safetensors", .{reference_path}),
            error.FileNotFound => std.log.err("{s}: no such reference file", .{reference_path}),
            error.InvalidSafetensors => std.log.err("{s} is not a safetensors file: REF is NAME.serial.f32.safetensors", .{reference_path}),
            else => return err,
        }
        return 1;
    };
    defer reference.close();
    const ids_tensor = reference.get("ids") orelse {
        std.log.err("{s} holds no ids: REF is a file the exporter wrote, NAME.serial.f32.safetensors", .{reference_path});
        return 1;
    };
    const tokens = try arena.alloc(i32, @intCast(ids_tensor.count()));
    @memcpy(std.mem.sliceAsBytes(tokens), ids_tensor.data);

    const layer_count = first_layers orelse config.layers;
    if (layer_count == 0 or layer_count > config.layers) {
        std.log.err("compare: layers={d}, and the model has {d}", .{ layer_count, config.layers });
        return 2;
    }
    var model: qwen4_model.Model = try .load(init.gpa, config, found, layer_count, tokens.len);
    defer model.deinit();
    var state: qwen4_forward.State = try .init(init.gpa, model, tokens.len, 4096);
    defer state.deinit();
    var buffers: qwen4_forward.Buffers = try .init(init.gpa, model, tokens.len, 4096);
    defer buffers.deinit();
    // an image prompt's reference (spec 019): TensorFold's features over its placeholders, its rotary positions
    var image = try qwen4_compare.ReferenceImage.read(init.gpa, &reference);
    defer if (image) |*one| one.deinit(init.gpa);
    if (image) |*one| try one.attach(&state);
    // the reference holds the input of a layer the model does not: the model is its first layers (a reference
    // of the first layers alone, `--layers`, ends where the model does)
    var after_buffer: [96]u8 = undefined;
    const partial = reference.get(try qwen4_compare.Comparison.plain(&after_buffer, .h_in, layer_count)) != null;
    var comparison: qwen4_compare.Comparison = .{ .reference = &reference, .slots = model.shape.top + 1, .force = !free, .partial = partial, .gpa = init.gpa };
    defer comparison.deinit();
    var by_row: ?qwen4_compare.ByRow = if (by_point != null or noise != null) try .init(init.gpa, tokens.len, model.layers.len) else null;
    defer if (by_row) |*by| by.deinit();
    if (by_row) |*by| comparison.by_row = by;
    var own: ?qwen4_compare.Own = if (noise != null) try .init(init.gpa, tokens.len, model.shape.wide(), model.shape.vocab) else null;
    defer if (own) |*kept| kept.deinit();
    if (own) |*kept| comparison.own = kept;
    try qwen4_forward.forwardTapped(model, &state, &buffers, tokens, comparison.tap());

    try stdout.print("{d} rows, {s}\n", .{ tokens.len, if (free) "running free: errors accumulate" else "each block fed the reference's input" });
    try stdout.print("{s:<14} {s:>7} {s:>12} {s:>6} {s:>12} {s:>14}\n", .{ "point", "layers", "rel rms", "at", "max abs", "differing" });
    for (std.enums.values(qwen4_forward.Point)) |point| {
        const stats = comparison.stats.get(point);
        if (stats.layers == 0) continue;
        try stdout.print("{t:<14} {d:>7} {e:>12.3} {d:>6} {e:>12.3} {d:>6}/{d}\n", .{ point, stats.layers, stats.relative, stats.worst_layer, stats.max_abs, stats.mismatched, stats.elements });
    }
    try stdout.print("picks differing: {d}; widest gap between the two experts' reference logits: {e:.3}\n", .{ comparison.picks_differing, comparison.widest_pick_gap });

    if (by_point) |point| {
        const by = by_row.?;
        // a row a line, a layer a column: rms(mine - reference) / rms(reference) of the row, in thousandths
        try stdout.print("\n{t} by row (down) and layer (across), relative rms in 1e-3; then final_h, final_mixed, logits and the row's first token here and in the reference\n", .{point});
        try stdout.writeAll("row ");
        for (0..model.layers.len) |layer| try stdout.print(" {d:>6}", .{layer});
        try stdout.print(" {s:>7} {s:>7} {s:>7} {s:>7} {s:>7}\n", .{ "final_h", "mixed", "logits", "mine", "theirs" });
        for (0..tokens.len) |row| {
            try stdout.print("{d:>3} ", .{row});
            for (0..model.layers.len) |layer| try stdout.print(" {d:>6.2}", .{by.get(point, layer)[row] * 1e3});
            try stdout.print(" {d:>7.2} {d:>7.2} {d:>7.2} {d:>7} {d:>7}{s}\n", .{ by.get(.final_h, by.layers)[row] * 1e3, by.get(.final_mixed, by.layers)[row] * 1e3, by.get(.logits, by.layers)[row] * 1e3, by.mine_top[row], by.reference_top[row], if (by.mine_top[row] != by.reference_top[row]) " *" else "" });
        }
    }

    if (noise) |settings| {
        const kept = &own.?;
        kept.record = false;
        const by = &by_row.?;
        const last = tokens.len - 1;
        const unperturbed_top = by.mine_top[last];
        const unperturbed = try arena.dupe(f64, by.get(.logits, by.layers));
        const unperturbed_tops = try arena.dupe(u32, by.mine_top);
        try stdout.print("\nnoise of {e:.2} of a row's rms added to layer {d}'s h_in, ", .{ settings.relative, settings.layer });
        if (settings.row) |row| try stdout.print("row {d}", .{row}) else try stdout.writeAll("every row");
        try stdout.print(": each row's logits against the run without it, relative rms in 1e-3 (against the reference: without the noise, then with it)\n", .{});
        for (0..seeds) |seed| {
            var perturb = settings;
            perturb.seed = seed + 1;
            try state.reset();
            if (image) |*one| try one.attach(&state);
            var noisy: qwen4_compare.Comparison = .{ .reference = &reference, .slots = model.shape.top + 1, .force = false, .by_row = by, .own = kept, .perturb = &perturb, .gpa = init.gpa };
            defer noisy.deinit();
            try qwen4_forward.forwardTapped(model, &state, &buffers, tokens, noisy.tap());
            try stdout.print("seed {d}: {e:.3} after the bf16 rounding; row {d}'s first token {d} (without the noise {d}, the reference's {d}); {d} picks differ from the reference's\n", .{ perturb.seed, perturb.achieved, last, by.mine_top[last], unperturbed_top, by.reference_top[last], noisy.picks_differing });
            try stdout.writeAll("  own:      ");
            for (kept.logits_relative) |relative| try stdout.print(" {d:>6.1}", .{relative * 1e3});
            try stdout.writeAll("\n  reference:");
            for (unperturbed) |relative| try stdout.print(" {d:>6.1}", .{relative * 1e3});
            try stdout.writeAll("\n  with it:  ");
            for (by.get(.logits, by.layers)) |relative| try stdout.print(" {d:>6.1}", .{relative * 1e3});
            try stdout.writeAll("\n  final_h:  ");
            for (kept.final_relative) |relative| try stdout.print(" {d:>6.1}", .{relative * 1e3});
            try stdout.writeAll("\n  rows whose first token is another than without the noise:");
            for (by.mine_top, unperturbed_tops, 0..) |top, before, row| {
                if (top != before) try stdout.print(" {d} ({d}, without it {d})", .{ row, top, before });
            }
            try stdout.writeAll("\n");
        }
    }
    return 0;
}
