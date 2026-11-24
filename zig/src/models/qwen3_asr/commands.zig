//! What Qwen3-ASR registers with `zspark`'s dispatch (`zig/src/decode/dispatch.zig`): its `model_type` and its
//! commands, a file each under `commands/`.

const dispatch = @import("../../decode/dispatch.zig");
const config_file = @import("config.zig");
const generate = @import("commands/generate.zig");
const serve = @import("commands/serve.zig");

pub const model: dispatch.Model = .{
    .name = "Qwen3-ASR",
    .model_types = &config_file.model_types,
    .commands = &.{
        .{ .name = "generate", .least = 2, .usage = generate.usage, .run = generate.run },
        .{ .name = "serve", .least = 1, .usage = serve.usage, .run = serve.run },
    },
};
