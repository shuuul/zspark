//! `zig/src/text/`'s tests alone, on the host: `zig build test-text` (spec 053). An edit under `text/` is answered
//! by this root in a part of the host tier's time, since neither the server nor the model is compiled.
//! `host_tests.zig` and `tests.zig` import it for the tiers. The imports stand in a `comptime` block and not in
//! a `test`: a `test` block counts as a test, and a tier's count would grow by one for each root it imports.

comptime {
    _ = @import("src/text/call_gate.zig");
    _ = @import("src/text/call_gate_strings.zig");
    _ = @import("src/text/call_gate_plan.zig");
    _ = @import("tests/text/call_gate_test.zig");
    _ = @import("tests/text/call_gate_walk_test.zig");
    _ = @import("tests/text/call_gate_checkpoint_test.zig");
    _ = @import("src/text/frontend.zig");
    _ = @import("tests/text/frontend_test.zig");
    _ = @import("src/text/grammar.zig");
    _ = @import("tests/text/grammar_test.zig");
    _ = @import("src/text/json_schema.zig");
    _ = @import("tests/text/json_schema_test.zig");
    _ = @import("tests/text/json_schema_prop_test.zig");
    _ = @import("tests/text/json_schema_oracle_test.zig");
    _ = @import("src/text/messages.zig");
    _ = @import("tests/text/messages_test.zig");
    _ = @import("src/text/pyjson.zig");
    _ = @import("tests/text/pyjson_test.zig");
    _ = @import("src/text/template.zig");
    _ = @import("tests/text/template_test.zig");
    _ = @import("src/text/tokenizer.zig");
    _ = @import("tests/text/tokenizer_test.zig");
    _ = @import("src/text/vision.zig");
    _ = @import("tests/text/vision_test.zig");
}
