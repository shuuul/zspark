//! The two whole-model tests alone (`zig build test-whole`, spec 133; needs the GPU), built with
//! `@import("tier").whole` true and filtered in `build.zig`, so that the host tier, the areas and this step
//! together are the full tier's tests. `comptime`: see `text_tests.zig`.

comptime {
    _ = @import("qwen4_tests.zig");
}
