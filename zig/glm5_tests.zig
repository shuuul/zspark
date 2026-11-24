//! `zig/src/models/glm5/`'s tests alone (`zig build test-glm5`; needs the GPU): every source with a test, and its
//! tests under `tests/models/glm5/`. `tests.zig` imports it for the tiers; the files that need no device are in
//! `host_tests.zig` too. `comptime`, not `test`: see `text_tests.zig`.

comptime {
    _ = @import("src/models/glm5/layers/config.zig");
    _ = @import("tests/models/glm5/layers/config_test.zig");
    _ = @import("src/models/glm5/layers/layout.zig");
    _ = @import("tests/models/glm5/layers/layout_test.zig");
    _ = @import("src/models/glm5/layers/index.zig");
    _ = @import("tests/models/glm5/layers/index_test.zig");
    _ = @import("src/models/glm5/layers/index_launch.zig");
    _ = @import("tests/models/glm5/layers/index_device_test.zig");
    _ = @import("src/models/glm5/layers/share.zig");
    _ = @import("tests/models/glm5/layers/share_test.zig");
    _ = @import("src/models/glm5/layers/glue_ref.zig");
    _ = @import("tests/models/glm5/layers/glue_ref_test.zig");
    _ = @import("src/models/glm5/layers/hc.zig");
    _ = @import("tests/models/glm5/layers/hc_test.zig");
    _ = @import("tests/models/glm5/profile_test.zig");
    _ = @import("src/models/glm5/layers/moe.zig");
    _ = @import("tests/models/glm5/layers/moe_test.zig");
    _ = @import("src/models/glm5/layers/mlp.zig");
    _ = @import("tests/models/glm5/layers/mlp_test.zig");
    _ = @import("src/models/glm5/layers/kda.zig");
    _ = @import("tests/models/glm5/layers/kda_test.zig");
    _ = @import("src/models/glm5/layers/dsa.zig");
    _ = @import("tests/models/glm5/layers/dsa_test.zig");
    _ = @import("tests/models/glm5/layers/extents_test.zig");
    _ = @import("tests/models/glm5/layers/mla_list_test.zig");
    _ = @import("src/models/glm5/layers/model.zig");
    _ = @import("src/models/glm5/layers/forward.zig");
    _ = @import("src/models/glm5/layers/loopback.zig");
    _ = @import("tests/models/glm5/layers/forward_test.zig");
    _ = @import("src/models/glm5/pool/slots.zig");
    _ = @import("tests/models/glm5/pool/slots_test.zig");
    _ = @import("src/models/glm5/mtp/mtp.zig");
    _ = @import("tests/models/glm5/mtp/mtp_test.zig");
    _ = @import("src/models/glm5/session/pack.zig");
    _ = @import("tests/models/glm5/session/pack_test.zig");
    _ = @import("tests/models/glm5/session/learnt_test.zig");
    _ = @import("tests/models/glm5/session/session_test.zig");
    _ = @import("tests/models/glm5/session/snapshot_test.zig");
    _ = @import("tests/models/glm5/session/marks_test.zig");
    _ = @import("src/models/glm5/draft.zig");
    _ = @import("tests/models/glm5/draft_test.zig");
    _ = @import("src/models/glm5/pool/hall.zig");
    _ = @import("tests/models/glm5/pool/hall_test.zig");
    _ = @import("tests/models/glm5/pool/hall_wait_test.zig");
    _ = @import("tests/models/glm5/pool/hall_pack_test.zig");
    _ = @import("src/models/glm5/pool/pool.zig");
    _ = @import("tests/models/glm5/pool/pool_test.zig");
    _ = @import("tests/models/glm5/layers/halves_test.zig");
    _ = @import("tests/models/glm5/dflash2/rig_test.zig");
    _ = @import("src/models/glm5/dflash2/pass.zig");
    _ = @import("tests/models/glm5/dflash2/pass_test.zig");
    _ = @import("tests/models/glm5/dflash2/pass_checkpoint_test.zig");
    _ = @import("src/models/glm5/dflash2/drafter.zig");
    _ = @import("tests/models/glm5/dflash2/drafter_test.zig");
}
