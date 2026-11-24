//! What Nemotron-3.5-Transcribe registers with `zspark`'s dispatch (`zig/src/decode/dispatch.zig`): its
//! `model_type` and its commands, a file each under `commands/`.

const dispatch = @import("../../decode/dispatch.zig");
const config_file = @import("config.zig");
const generate = @import("commands/generate.zig");
const ranks = @import("commands/ranks.zig");
const compare_file = @import("commands/compare.zig");
const serve = @import("commands/serve.zig");

pub const model: dispatch.Model = .{
    .name = "Nemotron-3.5-Transcribe",
    .model_types = &config_file.model_types,
    .commands = &.{
        .{ .name = "generate", .least = 2, .usage = generate.usage, .run = generate.run },
        .{ .name = "ranks", .least = 2, .usage = ranks.usage, .run = ranks.run },
        .{ .name = "compare", .least = 2, .usage = compare_file.usage, .run = compare_file.run },
        .{ .name = "serve", .least = 1, .usage = serve.usage, .run = serve.run },
    },
};
