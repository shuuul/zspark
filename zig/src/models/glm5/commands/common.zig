//! What GLM-5.3-Flash's commands share (Flash Next's is `../../qwen4/commands/common.zig`).

const std = @import("std");
const cli = @import("../../../cli.zig");
const glm5_config = @import("../layers/config.zig");
const glm5_layout = @import("../layers/layout.zig");

/// A checkpoint's directory opened in order: its config, then its layout (`cli.Opened`). The drafter's is
/// `../dflash2/rig.zig`'s `Opened`.
pub const Opened = cli.Opened(glm5_config.Config, glm5_layout.Layout, glm5_config.Diagnostic);

/// The other rank failed or is gone, or this one failed following: said, and this rank ends.
pub fn gone(session: anytype, err: anyerror) u8 {
    const said = session.joined.channel.said();
    if (err == error.PeerFailed) {
        std.log.err("rank {d}: the other rank failed: {s}", .{ session.first, said });
    } else std.log.err("rank {d}: {t}{s}{s}", .{ session.first, err, if (said.len > 0) ": " else "", said });
    return 1;
}
