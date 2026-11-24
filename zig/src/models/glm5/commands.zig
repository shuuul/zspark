//! What GLM-5.3-Flash registers with `zspark`'s dispatch (`zig/src/decode/dispatch.zig`): its `model_type` and its
//! commands, a file each under `commands/`. What decodes needs both ranks, a rank a Spark (`serve --rank`).

const std = @import("std");
const dispatch = @import("../../decode/dispatch.zig");
const glm5_config = @import("layers/config.zig");
const chat = @import("commands/chat.zig");
const drafter_config = @import("../../decode/dflash2/config.zig");
const drafter_info = @import("commands/drafter_info.zig");
const generate = @import("commands/generate.zig");
const info_file = @import("commands/info.zig");
const ranks_check = @import("checks/ranks_check.zig");
const serve_file = @import("commands/serve.zig");

pub const model: dispatch.Model = .{
    .name = "GLM-5.3-Flash",
    .model_types = &glm5_config.model_types,
    .commands = &.{
        .{ .name = "info", .least = 1, .run = info },
        .{ .name = "generate", .least = 2, .usage = generate.usage, .run = generate.run },
        .{ .name = "chat", .least = 2, .usage = chat.usage, .run = chat.run },
        .{ .name = "serve", .least = 1, .run = serve_file.run, .dirless = "--weights-from-peer" },
        .{ .name = "ranks-check", .least = 1, .usage = ranks_check.usage, .run = dispatch.usageOnTwo(ranks_check.run, ranks_check.usage) },
    },
};

/// The DFlash2 drafter's directory, which a GLM command is given beside the checkpoint (`--drafter`) and `info` reads
/// on its own.
pub const drafter: dispatch.Model = .{
    .name = "GLM-5.3-Flash's DFlash2 drafter",
    .model_types = &drafter_config.model_types,
    .commands = &.{
        .{ .name = "info", .least = 1, .run = drafter_info.run },
    },
};

fn info(init: std.process.Init, stdout: *std.Io.Writer, words: []const []const u8) anyerror!u8 {
    var list_all = false;
    var context: ?u64 = null;
    for (words[1..]) |word| {
        if (std.mem.eql(u8, word, "all")) {
            list_all = true;
        } else if (std.mem.startsWith(u8, word, "context=")) {
            context = std.fmt.parseInt(u64, word["context=".len..], 10) catch {
                std.log.err("context={s}: not a token count", .{word["context=".len..]});
                return 2;
            };
        } else {
            std.log.err("info: {s} is not all or context=N", .{word});
            return 2;
        }
    }
    return info_file.checkpointInfo(init, stdout, words[0], list_all, context);
}
