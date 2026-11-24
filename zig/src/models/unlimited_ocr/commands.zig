//! What Unlimited-OCR-2 registers with `zspark`'s dispatch (`zig/src/decode/dispatch.zig`): its `model_type` and its
//! commands, a file each under `commands/`.

const dispatch = @import("../../decode/dispatch.zig");
const config_file = @import("config.zig");
const generate = @import("commands/generate.zig");
const serve = @import("commands/serve.zig");

pub const model: dispatch.Model = .{
    .name = "Unlimited-OCR-2",
    .model_types = &config_file.model_types,
    .commands = &.{
        .{ .name = "generate", .least = 2, .usage = generate.usage, .run = generate.run },
        .{ .name = "serve", .least = 1, .run = serve.run },
    },
};
