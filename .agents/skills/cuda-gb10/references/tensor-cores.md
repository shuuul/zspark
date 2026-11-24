# Tensor cores on the GB10: tiles, registers, measured peaks

Traps around the `mma` chains: an epilogue's global reads, what a window's GEMM reaches in TFLOPS, and
split-K by shape. Part of the `cuda-gb10` skill.

- An epilogue's own global reads after an `mma` chain wait unless asked for early, and asked for into L2, not
  registers (spec 095 WS-03, `hc_window.cu`, 2,048 x 10,240 x 320 bf16 with `mix` reading four bf16 a
  (row, d) in the epilogue, 8 warps, 96 registers, two blocks an SM, ncu): read in the epilogue, 500 us, 9.4 of
  19.1 cycles between issues waiting on L1TEX; the same 32 values a thread loaded into registers before the
  main loop, 606 (3 stages 580); a `prefetch.global.L2` of the block's 512 lines of 64 B when it starts, the
  reads left in the epilogue, 465. 3 stages (61,440 B, one block an SM) lost to 2 (40,960 B, two): 499 against
  465.
- A wide window's bf16 GEMM is the tensor cores', and they give more than 33 TFLOPS (spec 087 WS-00, a served
  prompt's node trace): CUTLASS `Sm80` 128x128x32, 2,048 rows by 2,560 by 16,512 columns (84.5 MB of weight) in
  1.91 to 1.97 ms, 2 M N K = 1.73e11, so 88 to 91 TFLOPS counting a multiply-add as two; 1,008 rows on the
  128x128x64 tile 82. Hold a window's kernel against 90, a decoding round's against its bytes.
- Split-K is a decoding round's form and costs a window: the 384-column read-out split 32 over K writes 100 MB
  of fp32 partials at 2,048 rows and takes 0.70 ms plus its reduce where its products are 0.18 ms at 90 TFLOPS;
  the reduces are 4% of a window (same trace). A window wants the slices summed inside the block, in slice order.
- An A fragment read as two LDS.64 of a row-major tile comes in the wrong register order for `m16n8k16`: the
  SASS moves four registers an HMMA (`experts_tile.cu`, 32 HMMA and some 140 other instructions an inner loop).
  Stage it in the fragment's own order (a lane's four A registers one 16-byte chunk, cp.async 4 bytes at a
  time): one LDS.128 feeds an mma and the MOVs go (spec 107 WS-03, SASS by `cuobjdump -sass`).
- Folding a 16-input block's e4m3 scale into the bf16 weights (`fma.rn.bf16x2`, exact for e2m1 times e4m3) and
  letting the mma add into the sums, dropping P = mma from zero plus four fmaf an mma, costs the NVFP4 window
  experts 4 to 5% at 2,048 rows, 11 to 12% at 4,096, a step the same; the tensor pipe then 53% active on gate
  and up, the first stalls math-pipe throttle and the barrier (spec 107 WS-03, ncu). vLLM's b12x does the same.
- `mul.rn.f32x2` and `add.rn.f32x2` compile for `sm_121a` but come out as scalar FFMA/FADD pairs: no packed fp32
  pipe to gain from (spec 107 WS-03, SASS).
- An MXFP8 lane matmul of 256 rows or more is shorter on a 128x64 tile of 8 warps (WN 8, 2 stages, 41 KB, two
  blocks an SM) than on the 64x64 one (spec 105 WS-03, `ZSPARK_NVFP4_PROFILE=lanewide`, 2,048 rows): 16512x2560
  2723 us against 3109, 2560x6144 1255 against 1540, the same bits; a third stage was slower (3.34 ms against
  2.77), and `__launch_bounds__` minBlocks set on every form changed the small tiles' registers.
