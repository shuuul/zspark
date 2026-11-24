//! `zig/src/models/nemotron/`'s tests alone (`zig build test-nemotron`, spec 151): the checkpoint manifest's,
//! the glue kernels' and the Mamba mixer's tests, plus the perception module's mel and encoder. `tests.zig`
//! imports it for the tiers; the files that reach no `cuda.zig` (config, layout, glue_ref, mel) are in
//! `host_tests.zig` too. `comptime`, not `test`: see `text_tests.zig`.

comptime {
    _ = @import("src/models/nemotron/config.zig");
    _ = @import("tests/models/nemotron/config_test.zig");
    _ = @import("src/models/nemotron/layout.zig");
    _ = @import("tests/models/nemotron/layout_test.zig");
    _ = @import("src/models/nemotron/glue.zig");
    _ = @import("src/models/nemotron/glue_ref.zig");
    _ = @import("tests/models/nemotron/glue_test.zig");
    _ = @import("tests/models/nemotron/glue_ref_test.zig");
    _ = @import("src/models/nemotron/mamba.zig");
    _ = @import("tests/models/nemotron/mamba_test.zig");
    _ = @import("src/models/nemotron/attention.zig");
    _ = @import("tests/models/nemotron/attention_test.zig");
    _ = @import("src/models/nemotron/prompt.zig");
    _ = @import("tests/models/nemotron/prompt_test.zig");
    _ = @import("src/models/nemotron/mel.zig");
    _ = @import("tests/models/nemotron/mel_test.zig");
    _ = @import("src/models/nemotron/perception.zig");
    _ = @import("tests/models/nemotron/perception_test.zig");
    _ = @import("tests/models/nemotron/checks/reference_test.zig");
    _ = @import("tests/models/nemotron/accuracy_test.zig");
    _ = @import("src/models/nemotron/speaker.zig");
    _ = @import("tests/models/nemotron/speaker_test.zig");
    _ = @import("src/models/nemotron/moe.zig");
    _ = @import("tests/models/nemotron/moe_test.zig");
    _ = @import("src/models/nemotron/model.zig");
    _ = @import("src/models/nemotron/state.zig");
    _ = @import("src/models/nemotron/forward.zig");
    _ = @import("src/models/nemotron/mtp.zig");
    _ = @import("src/models/nemotron/draft.zig");
    _ = @import("tests/models/nemotron/forward_test.zig");
    _ = @import("src/models/nemotron/dflash2.zig");
    _ = @import("src/models/nemotron/checks/random_drafter.zig");
    _ = @import("tests/models/nemotron/dflash2_test.zig");
    _ = @import("tests/models/nemotron/profile_test.zig");
    _ = @import("src/models/nemotron/served.zig");
    _ = @import("tests/models/nemotron/served_test.zig");
}
