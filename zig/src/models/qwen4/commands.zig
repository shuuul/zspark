//! What Flash Next registers with `zspark`'s dispatch (`zig/src/decode/dispatch.zig`): its `model_type`s and its commands,
//! a file each under `commands/`, whose `run` reads the command's words.

const dispatch = @import("../../decode/dispatch.zig");
const qwen4_config = @import("config.zig");
const chat_file = @import("commands/chat.zig");
const compare_file = @import("commands/compare.zig");
const generate_file = @import("commands/generate.zig");
const info_file = @import("commands/info.zig");
const ranks_check = @import("checks/ranks_check.zig");
const score_file = @import("commands/score.zig");
const serve_file = @import("commands/serve.zig");

pub const model: dispatch.Model = .{
    .name = "Qwen3.8 Flash Next",
    .model_types = &qwen4_config.model_types,
    .commands = &.{
        .{ .name = "info", .least = 1, .run = info_file.run },
        .{ .name = "compare", .least = 2, .run = compare_file.run },
        .{ .name = "chat", .least = 2, .usage = chat_file.chat_usage, .run = chat_file.run },
        .{ .name = "serve", .least = 1, .run = serve_file.run },
        .{ .name = "generate", .least = 3, .run = generate_file.run },
        .{ .name = "score", .least = 3, .run = score_file.run },
        .{ .name = "ranks-check", .least = 1, .usage = ranks_check.usage, .run = dispatch.usageOnTwo(ranks_check.run, ranks_check.usage) },
    },
};
