# CUTLASS on the GB10 (4.8.0 read)

What applies to `sm_121a` and what does not. Part of the `cuda-gb10` skill.

- "Support for Blackwell SM121 kernels for DGX Spark GPUs. Share the major codes with Blackwell SM120
  kernels" (CHANGELOG). The arch tag is `arch::Sm120`; there is no `Sm121` struct.
- SM120 collectives, all TMA and warp specialized (cooperative or pingpong):
  `gemm/collective/sm120_mma_tma.hpp`, `sm120_mma_array_tma.hpp` (pointer-array grouped GEMM, FP8),
  `sm120_blockscaled_mma_tma.hpp`, the blockwise-scaling and sparse variants. MMA atoms in
  `cute/arch/mma_sm120.hpp` (`SM120_16x8x32_TN`, F8F6F4 kinds only).
- Their epilogue is Hopper's visitor tree: `epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp` builds
  on `Sm90EVT` (per-row and per-column vectors, elementwise activations, a second output, reductions). The
  classic 2.x `GemmUniversal` on `arch::Sm80` has an elementwise epilogue only.
- The builder (`builders/sm120_mma_builder.inl`) asserts: both elements F8F6F4 ("only supports F8F6F4
  MMA"), TN layout only, cluster of 1. So `CollectiveBuilder` gives bf16 x bf16 nothing on SM120, and such a
  GEMM is built on `arch::Sm80` (cp.async, no TMA, no EVT).
- The collective under the builder carries no such assert, and works with bf16 assembled by hand (spec 070):
  `CollectiveMma<MainloopSm120TmaWarpSpecialized<2, 2, cluster 1,
  KernelTmaWarpSpecializedCooperativeSm120<2>>>` with `make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN,
  Layout<Shape<_4,_2,_1>>, Tile<_128,_32,_16>)`, `SM90_TMA_LOAD`, `UMMA::Layout_K_SW128_Atom<bfloat16_t>`,
  `Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>`; the epilogue from the Sm120 `CollectiveBuilder` with bf16 out.
  Tile 128x128x64 at 2 stages is 79,872 B of shared memory. Right at 1 to 512 rows. Measured against the Sm80
  path at window rows (spec 159, 2026-11-05, `ZSPARK_BF16_SM120`, medians of 9 passes, weights past L2): slower
  at every shape and row, 2.9% (16512 x 2560, 1,024 rows) to 15.9% (2560 x 6144 at 2,048 and 10240 x 320 at
  512). And not bit-exact: its fp32 accumulation differs from the GEMM's element for element, 2,645 of
  4,194,304 at n 4,096, K 2,048 (1,024 rows) off by one bf16 ulp, 5 of 32,768 at n 256, K 512, none at K 256
  and below. Two stages against three cost more than warp specialization and TMA save. Not adopted; nothing of
  it is in the tree.
- Blockscaled tiles 128x32xK and 128x64xK: "up to 30%" on SM121 (CHANGELOG); tile N of 8 and 16 exist.
- Examples that apply: 79 (`blackwell_geforce_gemm`), 80, 87. Examples 70 to 78, 81 to 86, 92, 93 are
  `sm_100` (tcgen05); 89 and 90 `sm_103`.
- CuTe DSL (Python) kernels published for Hopper (arch 90, `wgmma`) or datacenter Blackwell do not run here
  as they are: only their epilogue's design carries over.

- A bias and an activation in the Sm80 path's elementwise epilogue (spec 019 WS-02,
  `zig/kernels/bf16/gemm_bias.cu`): the bias as C with a row stride of 0 (example 12's way) and
  `LinearCombinationGeneric<Act, bf16, 8, float, float, ScaleType::NoBetaScaling>`. `Act` is instantiated on
  `float` and on `Array<float, 8>`: a custom functor needs both specializations, else it does not compile.
  128x128x32 tile, 64x64x32 warps, 3 stages: 226 registers, 49,152 B, two blocks an SM. At 4,096 rows: qkv
  (N 3,456, K 1,152), proj (1,152 x 1,152) and fc1 (4,304 x 1,152) at 80 to 84 TFLOPS, fc2 (1,152 x 4,304, K not
  a multiple of 32) at 62. `GemmIdentityThreadblockSwizzle<8>` instead of `<1>` took 16,384 rows of qkv from
  159 ms to 45 for 27 launches, fc1 202 to 59, fc2 247 to 71 (L2 reuse of the weight's column tiles; the same
  bits), and 4,096 rows' fc2 23.2 to 17.8.

- A bf16 or fp16 GEMM of a decoding round (CUTLASS `Sm80`, 1 to 16 rows) is the memory's, not the tensor
  cores': from L2 (`hot`, one copy) it takes 23% of its time past L2 (6.5 MB: 7.3 us for 31.5), ncu counts the
  weight's bytes filled from memory once (205,584 sectors of 32 B for 6.55 MB), and a launch is its bytes at
  225 to 232 GB/s plus 2 to 7 us. The tile is then a question of blocks an SM: 128 columns by 4 stages is
  73,728 B, one block an SM; 24,576 B (8 columns by 8 stages for a weight of one K slice, 32 by 4 for one
  split over K) is four and 1 to 5% shorter, the same bits; neither wins on the other's shapes (spec
  073 WS-04, `bf16/gemm.cu`, `ZSPARK_BF16_PROFILE`).
- A window's GEMM (128 rows a tile) is the other way: 128 x 128 x 64 by 3 stages is 98,304 B, one block an SM,
  8% occupancy, the SMs' compute at 64% with the warps waiting on the tensor pipe (ncu, 512 rows). Stages of 32
  values of K (49,152 B, two blocks an SM) are 5 to 10% shorter from 2048 rows and for a weight L2 holds, and
  5 to 16% longer under 1024 rows on a weight larger than L2: more blocks at once spread over more column
  tiles than L2 keeps. Dispatch by rows and bytes; tiles change no bit (same spec, `bf16.Tile.of`).
- A window's split-K summed inside the GEMM by CUTLASS's serial mode (kGemm with K slices, a semaphore a tile,
  slice z adding `fma(1, acc_z, D)` after z - 1: the parallel split plus an in-order reduce, bit for bit)
  halves a thin wide-K GEMM whose slices L2 cannot hold: 320 x 10,240 in 32 slices at 2,048 rows on the half
  tile 674 to 353 us, where the parallel split wrote 84 MB of fp32 slices; its reader then reads one slice.
  On a wide N in few slices the chained epilogues cost: 2,560 x 6,144 in 4 slices 1,026 to 1,215 us, the
  reduce after it 439 to 142 (now a bf16 rounding of one slice), 7% off the pair. The chain is SK epilogues
  long whatever the rows, so on few rows it loses: the 32-slice read-out 52 to 136 us at 64 rows, even at 512,
  354 to 250 at 1,024; 513 x 2,560 in 8 slices 64 to 90 at 512, 144 to 124 at 1,024. Gate it by rows (GB10's
  ncu gives `dram__bytes` as n/a; spec 095 WS-03).
