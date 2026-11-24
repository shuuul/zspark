//! `zig/src/server/`'s tests alone, on the host: `zig build test-server` (spec 053). The files import `text/`
//! beside them; `build.zig` gives the step the filter `tests.server.`, so the tests of
//! `text/` this root reaches are left to `test-text` and the code of `text/` no server test calls is not
//! compiled. `host_tests.zig` and `tests.zig` import it for the tiers. `comptime`, not `test`: see `text_tests.zig`.
//! `served_test.zig` imports TensorFold's host by name (`engine_api`, `server`), which every root's module has.

comptime {
    _ = @import("src/server/command.zig");
    _ = @import("tests/server/command_test.zig");
    _ = @import("src/server/request.zig");
    _ = @import("src/server/segments.zig");
    _ = @import("tests/server/segments_test.zig");
    _ = @import("src/server/scheduler.zig");
    _ = @import("tests/server/scheduler_test.zig");
    _ = @import("src/server/images.zig");
    _ = @import("tests/server/images_test.zig");
    _ = @import("tests/server/transcriptions_test.zig");
    _ = @import("src/server/served.zig");
    _ = @import("src/server/served_live.zig");
    _ = @import("tests/server/served_test.zig");
    _ = @import("tests/server/served_queue_test.zig");
    _ = @import("tests/server/served_logprobs_test.zig");
    _ = @import("tests/server/served_vision_test.zig");
    _ = @import("src/server/structured.zig");
    _ = @import("tests/server/structured_test.zig");
    _ = @import("src/server/xgrammar.zig");
    _ = @import("tests/server/xgrammar_test.zig");
}
