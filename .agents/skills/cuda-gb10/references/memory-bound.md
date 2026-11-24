# Memory-bound kernels on the GB10

What a kernel at the memory's 215 to 235 GB/s pays: weight streaming, `cp.async` stages and their alignment,
shared-memory banks and registers at an SM's edge, L2 across calls, and the GB/s figures. Part of the
`cuda-gb10` skill.

- A memory-bound kernel's lessons (spec 070): a weight read lane by lane with `__ldg` arrives at 110 to 120
  GB/s, staged by `cp.async` into shared memory at 190 and more; what counts is the weight bytes on their way
  an SM (the 100 KB is the SM's, shared by its blocks), so stage nothing else that can be smaller; a stage of
  32 or 64 values of K, not 48 and not the whole K; time a kernel by its own duration (a trace), not a loop of
  launches, and after a write if it follows one (dirty lines going back cost the next read 5 to 10 us).
- A warp's `cp.async` stage in shared memory starts on a line of 128 bytes: the same stages 16 bytes off (a
  16-byte header before them) made a 34 us weight-streaming launch 43 (spec 070 WS-04,
  `zig/tests/kernels/qwen4/hc_read_one.cu`).
- A weight-streaming kernel wants a few stages on their way, not the weight (spec 070 WS-07, `hc_read_fp8.cu`,
  3.3 MB of FP8, standalone): a ring of 4 to 6 groups of 64 inputs a warp 18.1 us, 10 groups 18.5, the whole
  20 (90 KB a block, one an SM) 20.0, 2 groups 21.9. A block that takes most of an SM's shared memory is
  slower even alone on it.
- Shared-memory banks (a word's address modulo 32 words) cost little in a memory-bound kernel: eight lanes
  storing a float2 to one bank pair, 16 stores a half pass, were 1 to 2 us of a 50 to 80 us launch at 16 rows
  and 3% at 64; rows of sums 136 floats apart instead of 128 (8 more modulo 32, float4 reads still aligned)
  take it back (spec 071 WS-11, `linear_padded.cuh`). Measure before padding: the same padding in a kernel at
  its SM's edge loses a block (next line).
- A kernel at an SM's edge by registers and by shared memory at once (`grouped_kernel`: 167 registers x 128
  threads and 32,832 B, three blocks an SM by either) loses a block to anything: 2 KB of padding and 5
  registers made it 2 an SM and 3 to 5% slower. A `switch` on a width inside its k loop made 253 registers
  (`-maxrregcount` ignored under `__launch_bounds__`; a second argument of 3 kept 168 by spilling to the
  stack, slower still); a loop written out a width kept 168. Read `registersPerThread` in the trace.
- A kernel as slow from L2 as from DRAM whose time steps with every 48 blocks is bound by what an SM issues,
  not by launch waves and not by memory: `mix_kernel`'s threads each made 256 two-byte loads for 128 products,
  18, 33, 47 us at 3, 6, 8 rows of 16 blocks. Loading 16 bytes at a time (`uint4`, 8 bf16) and keeping the
  sum's order (lanes by index mod 8, folded after): 7.5, 10.0, 13.2 us, and 523 for 2,420 at 512 rows. Count a
  thread's load instructions before its arithmetic (spec 073 WS-05, `mix_octs_kernel`).
- A kernel that reads many rows of a weight a few bytes of each a step waits for each step: GLM's `expand`
  (128 rows a block, 32 bytes a step) read 8.4 MB at 133 GB/s where `absorb`, which reads rows whole, had 227.
  `prefetch.global.L1` of the step one or two ahead: 45 us for 64 (4 ahead 46, 8 ahead 51, a block's every row
  before the first step 55). From L2 the asking form is slower, 19 us for 10: use it only where a round cannot
  hold the weight in L2, and say which (spec 073 WS-05, `glm5/mla_proj.cu` `expand_asked_kernel`).
- The scalar-store trap again, in GLM's KDA chain: 2.1 MB of fp32 state stored a float a thread at a time was
  252,751 sectors written (8.09 MB), 67,465 as `float4`. From L2 the launch went 16.1 to 7.5 us at one row;
  from DRAM, as a rank's 34 layers of states (71 MB a row) are found, only 26.3 to 25: the read of the state
  hides it. Report both, a round lies between (spec 073 WS-05, `glm5/kda_quads.cu`).
- GLM-5.3-Flash's round is its bf16 weights' bytes: a rank's KDA projection is 103 MB a launch, 446 us at 1 to
  8 rows (231 GB/s), the dense gate and up 437, and the GEMMs are 44% of a round's kernel time; the tile by
  shape is 0 to 4% better than the narrow one and nothing else in the kernel is left (spec 073 WS-05,
  `ZSPARK_BF16_PROFILE=glm`). A model whose large linears are stored unquantized decodes at the DRAM rate
  whatever its own kernels do: there they were 11% of the round.
- A transposing store's banks (spec 105 WS-05, ncu): `attention.cu`'s FP8 block writes a tile's value codes
  transposed, rows 80 bytes apart, lane l a 16-byte chunk l % 16 of a four of keys: its 16 lanes' rows are
  16 x 80 = 1,280 bytes apart, the same bank, so 260 M of the shared stores' 382 M wavefronts conflicted and
  MIO throttle was the top stall. Lanes over sixteen fours of one chunk (consecutive words), the other
  half-warp's words in the other order (4 rows of 80 bytes later is 16 banks over), took a 2,048-row launch
  over 16k positions from 5.62 to 3.97 ms, the same bits. Read
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st` before blaming latency.
- The same conflict at decoding rows does not bind (spec 105 WS-06, ncu and nsys): `attn_chunk_kernel` at 1 to
  16 rows over 4k and 16k fp8 positions (16 to 1,024 blocks, 8 tiles each) is stalled on long scoreboard (59%
  of the cycles at 1 row, 44% at 16), MIO throttle 4 to 6%. WS-05's lane order there removes every conflict
  (122,880 and 2.03 M to 0 and 8 k) but its value loads lose their 256 contiguous bytes a half-warp: 1 row
  33.3 to 30.3 us, 4 rows 50.9 to 52.8, 16 rows 86.8 to 102.9 at 16k. Conflicts cost where a kernel waits on
  MIO, not where it waits on its loads.
- A row read by 16 lanes at a stride of 16 words met one bank 8 times: GLM's hc mix at 2,048 rows held at 80%
  of L1 and 1,330 us; swizzling a 16-word group by its index (`(c & ~15) | ((c & 15) ^ ((c >> 4) & 7))`) gave
  642, then a persistent grid with a cp.async ring of 2 gave 470 (from 2,173 for the per-row kernel); 2 blocks
  an SM by `__launch_bounds__` spilled, 1,416 (spec 107 WS-03, ncu).
