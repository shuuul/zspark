//! The tests of the device code every model shares, alone: the bindings, the linears and the experts, each
//! storage on its kernels, a row's log probabilities and the ranks' communicator (`zig build test-weights`, spec 053;
//! needs the GPU). No model is compiled.
//! `tests.zig` imports it for the tiers. `comptime`, not `test`: see `text_tests.zig`.

comptime {
    _ = @import("tests/weights/numeric_test.zig");
    _ = @import("tests/weights/safetensors_test.zig");
    _ = @import("tests/weights/checkpoint_reader_test.zig");
    _ = @import("tests/weights/csf_test.zig");
    _ = @import("tests/weights/csf_blocks_test.zig");
    _ = @import("tests/weights/modelopt_test.zig");
    _ = @import("tests/weights/drive_test.zig");
    _ = @import("tests/weights/ahead_test.zig");
    _ = @import("src/cuda/cuda.zig");
    _ = @import("tests/cuda/cuda_test.zig");
    _ = @import("tests/cuda/contract_test.zig");
    _ = @import("tests/decode/graphs_test.zig");
    _ = @import("src/decode/dflash2/kernels.zig");
    _ = @import("tests/decode/dflash2/kernels_test.zig");
    _ = @import("src/decode/dflash2/weights.zig");
    _ = @import("tests/decode/dflash2/weights_test.zig");
    _ = @import("tests/ranks/ranks_test.zig"); // the full tier's alone until spec 133
    _ = @import("src/weights/bf16.zig");
    _ = @import("tests/weights/bf16_test.zig");
    _ = @import("tests/weights/bf16_experts_test.zig");
    _ = @import("tests/weights/bf16_profile_test.zig");
    _ = @import("src/weights/nvfp4.zig");
    _ = @import("tests/weights/nvfp4_test.zig");
    _ = @import("tests/weights/nvfp4_a8_test.zig");
    _ = @import("src/weights/experts.zig");
    _ = @import("src/weights/experts_pack.zig");
    _ = @import("tests/weights/experts_test.zig");
    _ = @import("src/weights/face.zig");
    _ = @import("tests/weights/face_test.zig");
    _ = @import("tests/weights/nvfp4_profile_test.zig");
    _ = @import("src/weights/requantize.zig");
    _ = @import("tests/weights/requantize_test.zig");
    _ = @import("src/weights/linear.zig");
    _ = @import("tests/weights/linear_test.zig");
    _ = @import("src/decode/logprob.zig");
    _ = @import("tests/decode/logprob_test.zig");
    _ = @import("tests/weights/testing.zig");
    _ = @import("src/weights/share.zig");
    _ = @import("tests/weights/share_test.zig");
}
