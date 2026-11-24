//! `share_layout.zig`: a rank's cut of NVFP4 experts whose block scales are csf-encoded. The streams stay whole
//! and the rank decodes its window (`layout.Csf.window`): its scales are the whole decoded matrix's rows (gate,
//! up) or columns (down), byte for byte, beside its half of the codes.

const std = @import("std");
const testing = std.testing;
const config_file = @import("../../../src/models/qwen4/config.zig");
const layout = @import("../../../src/models/qwen4/layout.zig");
const share = @import("../../../src/models/qwen4/share.zig");
const share_layout = @import("../../../src/models/qwen4/share_layout.zig");
const csf = @import("../../../src/weights/csf.zig");
const safetensors = @import("../../../src/weights/safetensors.zig");

test "a mixed-precision checkpoint's csf experts: each rank's window decodes to its half of the whole matrix" {
    const path = testing.environ.getPosix("ZSPARK_FLASH_NEXT_FP4_LOCAL") orelse return error.SkipZigTest;
    var checkpoint: safetensors.Checkpoint = try .open(testing.allocator, testing.io, .cwd(), path);
    defer checkpoint.close();
    var arena_state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var dir = try std.Io.Dir.cwd().openDir(testing.io, path, .{});
    defer dir.close(testing.io);
    var diagnostic: config_file.Diagnostic = .{};
    const heads = share.Heads.of(try config_file.Config.read(arena, testing.io, dir, &diagnostic));
    var windows: usize = 0;
    for ([_]u32{ 0, 47 }) |layer| for ([_]u32{ 0, 511 }) |expert| for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }) |proj| {
        const base = try std.fmt.allocPrint(arena, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ layer, expert, proj });
        const weight = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight", .{base})).?;
        const fixed = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale{s}", .{ base, csf.fixed_suffix })).?;
        const exceptions = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale{s}", .{ base, csf.exceptions_suffix })).?;
        const scale_2 = checkpoint.get(try std.fmt.allocPrint(arena, "{s}.weight_scale_2", .{base})).?;
        const rows: u32 = @intCast(weight.shape[0]);
        const cols: u32 = @intCast(weight.shape[1] * 2);
        const whole: layout.Linear = .{ .name = base, .scheme = .nvfp4, .rows = rows, .cols = cols, .weight = weight, .csf = .{ .fixed = fixed, .exceptions = exceptions }, .scale_2 = scale_2 };
        const all = try whole.blockScales(arena);
        const width: usize = cols / 16;
        const by_cols = share.groupOf(base).cut() == .col;
        for (0..share.world) |rank| {
            const part = try share_layout.sliceLinear(arena, whole, .of(@intCast(rank)), heads);
            try testing.expectEqual(if (by_cols) rows else rows / share.world, part.rows);
            try testing.expectEqual(if (by_cols) cols / share.world else cols, part.cols);
            try testing.expectEqual(@as(usize, part.rows) * part.cols / 2, part.weight.data.len);
            try testing.expect(part.scale_2.?.data.ptr == scale_2.data.ptr);
            const mine = try part.blockScales(arena);
            const w: usize = part.cols / 16;
            try testing.expectEqual(@as(usize, part.rows) * w, mine.len);
            for (0..part.rows) |r| {
                const from = if (by_cols) r * width + rank * w else (rank * part.rows + r) * width;
                try testing.expectEqualSlices(u8, all[from..][0..w], mine[r * w ..][0..w]);
            }
            windows += 1;
        }
    };
    try testing.expectEqual(@as(usize, 24), windows);
}
