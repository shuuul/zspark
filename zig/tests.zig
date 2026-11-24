//! The full tier's root (`zig build test`): the program, so that every file of it is compiled, and
//! every test, by the tiers' roots beside this file. A root stands in `zig/` because a module reaches only
//! what is under its root's directory, and a test under `tests/` imports its source under `src/`.

comptime {
    _ = @import("src/main.zig");
}

// The tiers' tests: the host tier's root (the shared files listed once, there) and the areas' roots, which
// the area steps run alone (spec 053). A file reached twice is compiled once and its tests run once.
test {
    _ = @import("host_tests.zig");
    _ = @import("text_tests.zig");
    _ = @import("server_tests.zig");
    _ = @import("image_tests.zig");
    _ = @import("audio_tests.zig");
    _ = @import("weights_tests.zig");
    _ = @import("qwen4_tests.zig");
    _ = @import("glm5_tests.zig");
    _ = @import("unlimited_ocr_tests.zig");
    _ = @import("qwen3_asr_tests.zig");
    _ = @import("nemotron_tests.zig");
}
