# The NVFP4 and W4A4 routed experts on the GB10

What binds the NVFP4 experts' kernels (Flash Next's and GLM's, A16 and W4A4): grids of serial chains, readings of
an expert's weights, L2, the windows' tile kernels. Window sizes and their effect on bytes per row are described in
[`chunking.md`](chunking.md).

- A grid-stride loop of units in whole rounds (`nvfp4/experts.cu` EVEN) pays for 2 or 3 rounds only. At some 35
  rounds (`experts_wide.cu`, a 2048-row window) there is no tail to win: no faster at 768 to 1,536 rows, 1 to 3%
  slower at 2,048, the cause not resolved (spec 073 WS-09).
- A wide tile kernel is not bound by its tensor cores (spec 105 WS-03, ncu): `experts_wide.cu` at 2,048 rows of
  512 experts runs 28 instructions an mma, the tensor pipe 30% active, ALU 26, three blocks an SM by registers,
  stalled on long_scoreboard and wait; an expert's 40 pairs in items of 32 read its weights twice. Staging rows
  and weights through shared memory by cp.async for 8 warps (`experts_tile.cu`, 64 pairs a reading, 128
  registers, two blocks an SM) took plan, gate+up and down from 10.04 to 7.73 ms, about 80% of DRAM for the
  1.4 GB; the weights in registers under the same tiles 9.57 (7.96 at 128 registers), half the barriers or the
  stages run on across units no faster. ncu reads no DRAM throughput on GB10 ("n/a"): judge a streaming kernel
  by bytes over its loop time.
- A grid-stride loop over units that are each a long serial chain (a warp sums one expert's 32-column block over
  80 stages of K, 64 us alone even from L2) must end in whole rounds: with 800 units on 768 warps the last 32
  run alone for a unit's own pace after the grid has ended, 380 us where 328 is the memory's rate.
  `ceil(units / rounds)` warps, spread over the blocks, each `rounds` units: the same bits, 4 to 14% off those
  launches, none slower (spec 073 WS-04, `nvfp4/experts.cu` EVEN). Look for it wherever a launcher caps a grid
  by occupancy and the work count is known only on the device.
- That chain is not memory latency: a ring of 4 stages a warp for 2 (165 registers for 128, 3 blocks an SM for
  4) was slower from L2 (70.9 us for 65.1) and no faster past it; 576 warps read 50 to 130 MB as fast as 768. A
  one-row launch (200 units on 200 warps, 18.4 MB) is 97 us where the memory's rate gives 86; closing it needs a
  unit split over more warps (its four column tiles), not resolved (spec 073 WS-04).
- A kernel that reads an expert's weights again for every 32 of its (row, slot) pairs pays for the pairs, but
  not in DRAM (spec 087 WS-04, ncu at 2,048 rows on 512 experts, two readings each): `experts_wide.cu` takes
  6.6, 10.1, 13.6, 17.3 ms a layer at 1,008, 2,032, 3,056, 4,080 rows; gate and up run at an L2 hit rate of 57%
  (31% at one reading each, where its 943 MB go by in 4.23 ms, 223 GB/s: the DRAM's bound) and an L1 one of 41
  to 59%. A second reading is met in L2 behind the first, which a neighbouring warp is making, and costs about
  44% of a first (5.7 us for 12.9, gate, up and down), in decode and the chain's latency. Per stage a warp reads
  more input than weight (2,048 B of X for 1,152), and a block's four warps share those rows in L1.
- So "read it once" has little to take there and every form lost standalone (spec 087 WS-04, ms a layer at
  2,048 rows on 512 experts, base 10.0 to 10.3): a second pair of 16-row tiles under each stage with its sums in
  shared memory 12.9, and 14.1 with one decode for the four (210 and 160 registers for 168 and 128, so 2 and 3
  blocks an SM for 3 and 4, and 32 float4 read and written a lane a stage); a block's warps on one column block
  of four items 14.5 (the shared rows lost; not profiled); an item of 64 pairs, one warp reading twice, 13.4.
  Sums that leave the registers inside the K loop, or a block that gives up its shared rows, cost more than a
  reading met in L2. The trellis window kernel won by the other rule: sums stay in registers through K.
- A wider prompt window buys nothing on NVFP4 Flash Next (spec 106 WS-03, six served loads, `--parallel 1`;
  TensorFold's #720): 4,096-row windows against 2,048 put a cold prompt's first token at 4.226 against 4.245 s
  at 7,879 tokens and 17.43 against 17.35 at 31,954 (medians of six, ranges 4.20 to 4.27 and 17.29 to 17.56),
  the same bits, for 1.8 GiB more of session buffers. The experts' layer time is near linear in rows, and what a
  window holds fixed is little.
- A decoding round of NVFP4 Flash Next at one stream is its products' (spec 106 WS-04, one `serve --parallel 1`
  load under nsys, node trace, replies of 256 tokens): the CUTLASS bf16 GEMMs 60 to 63% of the kernel time, the
  NVFP4 experts 28 to 30, the draft head 3, the DeltaNet chain 1.9, and all the glue TensorFold's
  loads-in-flight commits touch (`site`, `topk`, `act`, `mix`, the attention's `chunk` and `merge`, `select`)
  3.0% at 4k and 3.4 to 3.5 at 53k prompt tokens, of which the attention's chunks 1.0 and 1.3 (43 to 58 us a
  launch, 40 to 50 blocks). Latency-bound but small: every one at its launch floor (1.1 us), and the chunks at
  nothing would be 2.8% of a token — a ceiling a token shows outside the loads' 1.5% spread only if over half of
  it is won. The sparse attention keeps decode at 53k near 4k's.
- Flash Next's routing is skewed: of 512 experts a 512-row call of prose touches 309 and a 1,024-row one 353
  (the 3.05-bit pack, `ZSPARK_EXPERTS_TALLY`), so a window's expert bytes are some 70% of the table.
- The host's rows from a 51 GB mapped table cost by the page cache: 2,048 rows' n-gram lookups hold the device
  idle 90 ms between windows on text not seen and 29 on text read a minute before (spec 106 WS-04).
- A prompt's routed experts by window (FP4-Local, `experts_tile.cu`, 64 pairs a reading; spec 105 WS-05): in
  serve at 2,048 rows a layer-window is gate/up 4.78 ms and down 3.10 (nsys, a 16k prompt); all 512 experts'
  weights would be 943 and 472 MB, 4.0 to 4.4 and 3.0 to 3.3 ms at 215 to 235 GB/s (down also writes 210 MB of
  fp32 y slots). Prose routes about 400 of the 512 there, in 535 readings of 64 pairs for 20,480 pairs (the
  hottest 1,556 pairs): 64% of the slots read hold a pair. A layer's time by the window's width alone:
  [`chunking.md`](chunking.md). Not at the bytes, though: ncu at 2,048 rows of that skew (same spec) has gate/up
  4.78 ms with DRAM at 20% of its peak, L2 hit 46% (1.5 GB of L2 reads), issue slots 51% busy, the tensor pipe
  32%, about 15 instructions a mma, top stalls `wait` then `barrier`, 16 warps an SM; down 3.01 ms, L2 hit 51%,
  long scoreboard and wait. The bound is the decode chain a reading at two blocks an SM. Readings of 128 pairs
  would be 446 against 535 (-17%), at the cost of one block an SM; 4,096-row windows read 533 of 128 for twice
  the rows.
- GLM FP4-Local's tile kernel at 2,048 rows is at the bus, not at the tensor pipe (spec 107 WS-03, ncu,
  `nvfp4_profile_test.zig` `experts:2048:0` with `ZSPARK_NVFP4_SHAPE=glm`). Gate and up fill 1.64 GB from DRAM
  (`lts__d_sectors_fill_sysmem`) in 8.16 ms, about 200 GB/s, for 1.36 GB of weights; down reads 0.83 GB for 0.68
  and writes 268 MB of fp32 slots in 5.24 ms, about 210 GB/s. The tensor pipe's 51% and the stall after
  `BAR.SYNC` (17%) are the stage's copies arriving. Every row of an item read by the item's first column block
  misses L2: rows read by eight experts' items, eight times. Those first touches, about 6M of the 52M sectors,
  are the only bytes to win (all rows on row 0: 46.2M) — and nothing won them: the mma of an item's empty m16
  tiles skipped, 13.81 ms for 13.70 (plan, gate and up, and down); an expert's units taken column group first
  14.05 (the fills the same); the rows prefetched `L2::evict_last` 13.88; the weights' copies `L2::evict_first`
  14.44; an access policy window over the rows, persisting, 18 MB, 20.8 — and the calls that do not use it
  slower too.
- GLM FP4-Local's W4A4 tile kernel at a 4,096-row window is its units' deal, not the bus (spec 114 WS-09, ncu,
  `nvfp4_profile_test` `experts:4096:0:a4`, layer 3's real picks). Units of (item, column group) dealt by a
  stride of 48 fixed each block's column group (48 is a multiple of the 8 groups), and the blocks on long items
  ended up to 25% after the rest (SM active 14.7M to 18.4M cycles of 18.7M). Taken from an atomic counter as
  blocks free, the blocks end together and an expert's items on one column group run near each other: gate and
  up's DRAM fills 53M to 46M sectors (the weights read once and the rows), 8.7 to 9.9 ms to 7.4; the real call
  15.90 to 14.55 ms. Down was balanced and at the bus with its fp32 slots (0.70 GB read, 0.54 written, 5.8 to
  6.3 ms). A unit's last stages asking for the next unit's first (no drain between units) gained nothing on
  either (4,096 rows 14.54 against 14.33 ms standalone).
