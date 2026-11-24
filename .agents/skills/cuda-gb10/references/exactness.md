# Exactness on the GB10: reduction order and the bits

Where a result's bits are the constraint: a serial recurrence parallelized only so the sums keep their order.
Part of the `cuda-gb10` skill.

- A serial recurrence on one thread costs about 1 us for some 50 fp32 operations (16 divisions among them)
  and holds its block at a barrier while it runs: GLM's 4x4 Sinkhorn of 20 rounds on thread 0 was 18 of
  `pre_kernel`'s 27 us, at 1 row as at 8, from L2 as from DRAM. Find one by passing the iteration count as an
  argument before reaching for ncu (1 round: 8.7 us found it). Spread it over the first warp's lanes, an
  element a lane, the row's and column's sums by two `__shfl_xor_sync` each: 7.9 us and the same bits, since
  (a + c) + (b + d) is the lane two away and then the lane one away, either way round (spec 073 WS-05,
  `glm5/hc.cu` `pre_lanes_kernel`).
- A host pin that replays a kernel's reduction tree (a thread's chain, the xor butterfly 16..1 lane by lane,
  the warps' order) in Zig, where floats are never contracted, meets `--fmad=false` kernels to the bit on
  random rows: f64 RMS trees at K = 128 and 1,280 (`rsqrt.approx.f64` against libm `1/sqrt`: 0 of 38,000
  outputs off after the bf16 round) and a f64 sigmoid (`__frsqrt_rn`, `__fsqrt_rn` against host f64: 0 of
  491,520 off). Allow one bf16 ulp for the approximate rsqrt anyway, and hold rows alone to their window bits
  device against device (spec 119 WS-02).
- Several warp sums at once in one butterfly keep each sum's bits (`spread_sum`, TensorFold 995f98af): at each
  of warp_sum's xor levels a lane adds its partner's half of the values it keeps, so every sum is warp_sum's
  pairs in its order (a + b and b + a are equal to the bit). Four sums take 6 shuffles and 4 broadcasts
  against 20. Where the shuffles' issue, not their latency, bounds a row loop, two butterflies a row pay:
  Flash Next's decode chain went 101.8 to 89.5 to 90.0 ms over 2,664 launches of a traced decode (-12%; 2.1 to
  1.6 us a row), `replay_kernel` (four sums a row) -3%, the same tokens (spec 143 WS-05); the pool's
  `chain_slots_kernel` 170.8 / 168.1 to 139.8 / 139.8 us a launch's mean at 4 streams (-17%), the same replies
  (spec 143 WS-10).
