//! `zig/src/image/`'s tests alone, on the host: `zig build test-image` (spec 019 WS-07). `host_tests.zig` and
//! `tests.zig` import it for the tiers; the imports stand in a `comptime` block, as `text_tests.zig`'s do.

comptime {
    _ = @import("src/image/image.zig");
    _ = @import("src/image/resize.zig");
    _ = @import("tests/image/image_test.zig");
    _ = @import("tests/image/resize_test.zig");
    _ = @import("tests/image/damage_test.zig");
}
