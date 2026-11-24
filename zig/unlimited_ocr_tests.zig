//! `zig/src/models/unlimited_ocr/`'s device tests alone (`zig build test-unlimited_ocr`; needs the GPU): every
//! source with a device test, and its tests under `tests/models/unlimited_ocr/`. `tests.zig` imports it for the
//! tiers; the files that need no device are in `host_tests.zig`. `comptime`, not `test`: see `text_tests.zig`.

comptime {
    _ = @import("src/models/unlimited_ocr/vision.zig");
    _ = @import("tests/models/unlimited_ocr/vision_test.zig");
    _ = @import("src/models/unlimited_ocr/forward.zig");
    _ = @import("src/models/unlimited_ocr/draft.zig");
    _ = @import("src/models/unlimited_ocr/commands.zig");
    _ = @import("tests/models/unlimited_ocr/forward_kernels_test.zig");
    _ = @import("tests/models/unlimited_ocr/forward_dash_test.zig");
    _ = @import("src/models/unlimited_ocr/dash_attention.zig");
    _ = @import("tests/models/unlimited_ocr/dash_attention_test.zig");
    _ = @import("src/models/unlimited_ocr/router.zig");
    _ = @import("tests/models/unlimited_ocr/router_test.zig");
}
