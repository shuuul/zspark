//! `zig/src/audio/`'s tests alone, on the host: `zig build test-audio` (spec 138 WS-02). `host_tests.zig` and
//! `tests.zig` import it for the tiers; the imports stand in a `comptime` block, as `image_tests.zig`'s do.

comptime {
    _ = @import("src/audio/audio.zig");
    _ = @import("tests/audio/audio_test.zig");
    _ = @import("tests/audio/decode_test.zig");
    _ = @import("tests/audio/split_test.zig");
    _ = @import("src/audio/text.zig");
    _ = @import("tests/audio/text_test.zig");
}
