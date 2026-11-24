//! `zig/src/models/qwen3_asr/`'s tests alone (`zig build test-qwen3_asr`, spec 138; needs the GPU): the full tier's
//! tests of Qwen3-ASR's files, with `tier.whole` false. `tests.zig` imports it for the tiers. `comptime`, not
//! `test`: see `text_tests.zig`.

comptime {
    _ = @import("src/models/qwen3_asr/tower.zig");
    _ = @import("src/models/qwen3_asr/config.zig");
    _ = @import("src/models/qwen3_asr/layout.zig");
    _ = @import("src/models/qwen3_asr/model.zig");
    _ = @import("src/models/qwen3_asr/forward.zig");
    _ = @import("src/models/qwen3_asr/draft.zig");
    _ = @import("src/models/qwen3_asr/commands.zig");
    _ = @import("src/models/qwen3_asr/transcript.zig");
    _ = @import("src/models/qwen3_asr/served.zig");
    _ = @import("tests/models/qwen3_asr/tower_test.zig");
    _ = @import("tests/models/qwen3_asr/forward_test.zig");
    _ = @import("tests/models/qwen3_asr/transcript_test.zig");
    _ = @import("tests/models/qwen3_asr/served_test.zig");
}
