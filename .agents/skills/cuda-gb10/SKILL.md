---
name: cuda-gb10
description: "What the NVIDIA GB10 (DGX Spark, compute capability 12.1, sm_121a) gives a CUDA kernel and what it does not: which tensor-core instructions exist per dtype (bf16, FP8, FP4, block-scaled), TMA, shared memory, which CUTLASS collectives and epilogues apply, measured bounds (TFLOPS, memory GB/s), and how a kernel is optimized here (the chip's features, FlashAttention-style fusion, CODA-style packing of memory-bound and compute-bound work), and zspark's measured lessons by topic: chunking and windows, precision forms by window, two ranks and the link, drafting, states and marks, loading, measuring. Use before writing, fusing or tuning a kernel for a DGX Spark, before choosing a CUTLASS path, or when a claim about 'Blackwell' or 'Hopper' features has to be checked against this chip. Kept in this repository under .agents/skills; extends the `cuda` skill."
license: private
---

# CUDA on the GB10 (DGX Spark)

Facts about one chip, each with where it was read or measured. "Blackwell" in papers and in CUTLASS usually
means `sm_100` (datacenter): most of it does not apply here. Check a feature against this file, then against
the `cuda` skill's `references/ptx-isa/index.md` ("Target ISA Notes"), before building on it.

Every claim carries its evidence as one pointer, `(spec NNN WS-NN)`; a date appears only where it is the
provenance (a probe with no spec row) or the fact is time-sensitive (driver, toolkit, a checkpoint's
availability). A new measurement or source **rewrites** the entry it touches to the rule that now holds;
nothing is appended as a log line. A claim's history — what was tried and withdrawn, raw outputs, the runs'
dates — lives in `specs/archive/`, not here. Mark what was not measured as not measured.

## Contents, by topic

This file holds the chip's facts and the rules that bind every kernel; each topic below has its own file, with
its figures and their sources.

| Topic | What it holds | File |
| --- | --- | --- |
| The chip | cudaDeviceProp, shared memory, the bus; the pages the CPU wrote last and an allocation's fast tail | this file, [`references/specs.md`](references/specs.md) |
| Optimizing a kernel | the three levers in order (the owner), what zspark has of each and what does not pay here | [`references/optimizing.md`](references/optimizing.md) |
| Chunking and windows | a prompt's window rows by checkpoint and ranks and what each width bought; KDA chunked against recurrent; attention's key chunks and their merge; kernel forms by width; split rows on two ranks | [`references/chunking.md`](references/chunking.md) |
| Precision forms by window | the owner's rules, GLM's forms by window kind (T = 48), what narrower forms bought, errors against vLLM's own spread, cache storages, what was refused | [`references/precision-forms.md`](references/precision-forms.md) |
| Two ranks and the link | the link's rate (about 23 GB/s) and why, what crosses it a round and a window, the told follower, the overlaps that pay, what a second Spark buys | [`references/two-ranks.md`](references/two-ranks.md) |
| Drafting | a drafted row's cost against a round's, stopping rules (depth, caps, confidence, floors), copied and nucleus drafts, what a drafter may store, GLM's MTP and DFlash2 | [`references/drafting.md`](references/drafting.md) |
| States, snapshots and marks | what a state weighs, allocation as the cost, where marks are taken (against TensorFold, vLLM, SGLang), KDA marks under chunking, spill and park, replay | [`references/states-and-marks.md`](references/states-and-marks.md) |
| Loading | **the standard load pipeline** (the owner, 2026-10-20): O_DIRECT into kept stages, unpacking on the GPU, a rank's own bytes; the disk's, copy's, `cudaMalloc`'s and link's rates; no GDS | [`references/loading.md`](references/loading.md) |
| Measuring | off, on, on, off; the loads' spread; the first load; traced is not timed; safe against fast builds; what a whole-load tok/s cannot resolve; ncu and nsys use | [`references/measuring.md`](references/measuring.md) |
| Memory-bound kernels | weight streaming and `cp.async` stages, banks and registers at an SM's edge, L2 across calls, the GB/s reached | [`references/memory-bound.md`](references/memory-bound.md) |
| Tensor cores | an `mma` epilogue's reads, a window GEMM's real TFLOPS, split-K by shape, the MXFP8 lane tile | [`references/tensor-cores.md`](references/tensor-cores.md) |
| CUTLASS | the bf16 GEMM's tiles, serial split-K, the SM120 collectives, measured | [`references/cutlass.md`](references/cutlass.md) |
| FP8 linears | a cluster's K-slice sums by shape | [`references/fp8-linears.md`](references/fp8-linears.md) |
| NVFP4 and W4A4 experts | what binds the routed experts at decode and in windows, A16 against A4, `experts_wide` | [`references/experts-nvfp4.md`](references/experts-nvfp4.md) |
| Launches and graphs | hand-offs inside a launch, clusters, launch floors and waves, captured graphs, the two-Spark host message | [`references/launches-and-graphs.md`](references/launches-and-graphs.md) |
| A round's switches | each switch of a decoding round and what it costs (spec 073 and after) | [`references/round-switches.md`](references/round-switches.md) |
| Exactness | reduction order and the bits | [`references/exactness.md`](references/exactness.md) |
| Several processes | time slicing (half each), compute-policy (root), CUDA MPS as the user: thread percentage and client priority honoured, measured on the stack | [`references/sharing.md`](references/sharing.md) |

## The chip

| | |
| --- | --- |
| Device | NVIDIA GB10, compute capability 12.1 (`nvidia-smi --query-gpu=compute_cap`) |
| nvcc target | `-gencode=arch=compute_121a,code=sm_121a`; CUDA 13.0 toolkit, driver 580 |
| Family | `sm_120` (GeForce RTX 50) and `sm_121` share kernels; PTX's "`sm_120f` or higher in the same family" includes it |
| Memory | 128 GB unified with the CPU: no separate VRAM, `nvidia-smi` reports memory.total as N/A |
| SMs | 48; 1,536 threads, 65,536 registers, 24 blocks at most each |
| Shared memory | 102,400 B an SM, for all the blocks on it; a block 49,152 B by default, 101,376 opted in (`sm_90a` and `sm_100a` have 228 KB) |
| L2 | 24 MB |

Every property the runtime reports, the host side (CPU, pages, memory) and what each means for a kernel:
[`references/specs.md`](references/specs.md).

## Optimizing a kernel here: three levers, in this order

The owner, 2026-10-10. Detail, what zspark has of each and what does not pay here:
[`references/optimizing.md`](references/optimizing.md).

First find which side the kernel is bound on: its bytes a launch over its own duration against 215 to 235
GB/s, its products against the tensor cores' peak ([`references/cutlass.md`](references/cutlass.md)). Then:

1. **Use what CUDA gives on this chip.** Most products in zspark are `m16n8k16` on bf16 or f16, weights widened
   in registers; GLM's MXFP8 and block-FP8 linears take `mxf8f6f4` with both operands FP8 (`qmm_a8.cu`, spec 114
   WS-04). Not used: FP6 and FP4 `f8f6f4` products, the block-scaled
   `mxf4nvf4` (NVFP4 weights with their scales straight into the instruction), TMA with warp specialization,
   L2 persistence, unified memory mapped instead of copied. Used and tuned: `cp.async` rings, `ldmatrix`,
   graphs, a hand-off inside a launch. Check the tables below before reaching for one; measure both forms.
2. **Fuse a chain as FlashAttention does**: an intermediate larger than its inputs is never written, a small
   running state is carried from tile to tile. It pays where the intermediate leaves L2 (a prompt's window,
   many rows, a long cache), not at one decoding row. Softmax attention here is always this form. The tile
   order comes from the shape alone, so a row's bits stay its own.
3. **Pack memory-bound and compute-bound work in one kernel, as CODA does.** A kernel at the memory's rate has
   idle SMs: give them, in the order that has paid, a denser storage to decode (FP8, FP4, a trellis: fewer
   bytes for more arithmetic, up to the point where decoding saturates the SMs), a second reader of the same
   bytes (more rows a pass, the next weight asked for meanwhile), then the next steps' arithmetic in the
   epilogue. A compute-bound kernel takes memory work that is waiting anyway.

Whatever is packed in obeys the next section: it must not take the bytes the weight in flight needs.

## Shared memory is the budget: account for it every time

The 100 KB is the SM's. A block that takes 51 KB or more is alone on its SM, so the grid should then be 48
blocks or a multiple; smaller blocks share it (floor(102,400 / (bytes + 1,024)) a SM, 24 at most, 1,536 threads).
Before a kernel is written or tuned, and again in its review, write down a table of what its shared memory
holds: each region, its bytes a warp and a block, what it is for, and whether it could be smaller or elsewhere
(registers, global memory, a region reused once its first use is over). Then ask of each region:

- Is it weight on its way, or something else? In a memory-bound kernel only the bytes of weight in flight an
  SM buy speed (the base bf16 GEMM keeps 64 KB); activations, partial sums and outputs staged beside them
  take that away. zspark's `down_act` staged 16 rows of x whatever the window, two thirds of its stage, and
  was 2.5 us behind the GEMM until it staged the window's own rows (spec 070).
- Is it sized for the worst case where the usual case is smaller (rows of a window, columns of a last tile)?
- Can a region be reused (a stage that becomes the place the warps' sums meet), or dropped (a value a lane
  already holds in registers)?
- How many blocks does that leave an SM, and are all 48 SMs busy? 41 blocks of 96 KB leave 7 idle.

The table goes in the kernel's header comment; a change that moves a region updates it.

## Tensor-core instructions by dtype

| Instruction | On the GB10 | Note |
| --- | --- | --- |
| `wgmma.mma_async` (Hopper) | no | `sm_90a` only |
| `tcgen05.*` / UMMA, TMEM (datacenter Blackwell) | no | `sm_100a`, `sm_103a`, `sm_110a` |
| `mma.sync` bf16, f16 `m16n8k16` | yes | the same instruction as `sm_80`; nothing newer exists for bf16 |
| `mma.sync` FP8 (`.e4m3`, `.e5m2`) | yes | `sm_89` or higher |
| `mma.sync.kind::f8f6f4` `m16n8k32` | yes | FP8, FP6 (`.e3m2`, `.e2m3`), FP4 (`.e2m1`), mixed A and B; needs `sm_120a` |
| `mma.sync.kind::mxf8f6f4`, `mxf4`, `mxf4nvf4` with `.block_scale` | yes | MX and NVFP4 block scales (`.ue8m0`, `.ue4m3`), `m16n8k32` and `m16n8k64`; `m16n8k32 kind::mxf8f6f4.block_scale.scale_vec::1X.e4m3.e4m3.ue8m0` runs and is in zspark (`nvfp4/qmm_a8.cu`, spec 114 WS-04) |
| sparse `mma.sp` for the kinds above | yes | |

FP4 and FP6 elements sit in 8-bit containers in shared memory for `f8f6f4` (FP4 in the central 4 bits, hence
CUTLASS's "left shift for FP4"). So a narrow-precision product is on tensor cores only when BOTH operands are
FP8/FP6/FP4. bf16 rows times FP4 weights is not an instruction: dequantize one side, or quantize the other.

`kind::mxf4nvf4` takes its e2m1 packed two a byte instead (A 16 x 64 in four registers, B 64 x 8 in two, a
register of four ue4m3 scales a row or column a lane quad picks). It runs on `sm_121a`, bit-equal against an
f64 host sum in a one-warp probe (spec 114 WS-03);
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3` is the
form: lane (q, c) = (l / 4, l % 4); A register r holds row q + 8 (r % 2), inputs 32 (r / 2) + 8 c .. + 7, the
lower input in the low nibble (so a row of the checkpoint's packed bytes is read as is); B register j column
q, inputs 32 j + 8 c .. + 7; the SFA word of lane c in {0, 1} is row q + 8 c's four scales, byte i the
16-input group i (lanes c in {2, 3} are not read); the SFB word column q's four; the sums as `m16n8k16`'s.
The selector operands are `{0, 0}` (vLLM's b12x `nvfp4_mma_m16n8k64_f32_e2m1`, which pins the same map).

## Data movement

- L2 cache hints: `createpolicy.fractional.L2::evict_first` on `cp.async` runs; `evict_last` runs in a one-copy
  test (`ld.global` and `cp.async.cg`) but stopped `experts_tile.cu` with "illegal instruction" (the policy's
  `desc[UR1]`, the SASS otherwise evict_first's) — do not reach for it there; `prefetch.global.L2::evict_last`
  runs (spec 107 WS-03).
- TMA (`cp.async.bulk.tensor`, tensor maps): yes (`sm_90` or higher). Restrictions on `sm_120a`: towards
  global (`.shared::cluster.global`) no sub-byte types and no `.swizzle_atomicity`.
- Clusters exist, but `.multicast::cluster` is advised for `sm_90a`/`sm_100*`/`sm_103*`/`sm_110*` only and
  "may have substantially reduced performance" elsewhere. CUTLASS's SM120 builder asserts a cluster of 1.
- `cp.async`, `ldmatrix`: as on `sm_80`.
- A `cp.async.cg` 16-byte copy whose shared destination sits 16 B off the 32-byte grain of its global source moves
  twice the sectors: a static `__shared__ int` beside `extern __shared__` stages shifts the dynamic base by 16 B
  (ncu `sm__sass_l1tex_t_sectors_pipe_lsu_mem_global_op_ldgsts_cache_bypass` 61M to 123M for the same instructions,
  the W4A4 tile call 1.5 ms slower at 4,096 rows). Keep a block's scalars after its dynamic stages (spec 114 WS-09).

## CUTLASS

[`references/cutlass.md`](references/cutlass.md): the arch tag is `arch::Sm120` (TMA, warp specialized, `Sm90EVT`
epilogues); its builder takes F8F6F4 operands only, so a bf16 GEMM is `arch::Sm80` (cp.async, elementwise
epilogue) or the SM120 collective assembled by hand, which works.

## Measured bounds

One figure a bound; the topic file holds the shapes around it and the runner.

- A hand `m16n8k16` chain reaches about 33 TFLOPS — a hand chain's, not the chip's (spec 011,
  [`references/cutlass.md`](references/cutlass.md)).
- A CUTLASS Sm80 bf16 GEMM at thousands of rows reaches 80 to 84 TFLOPS (128x128x32, 3 stages, swizzle 8), a
  FlashAttention-style tile of 128 queries about 75 (spec 019 WS-02).
- bf16 rows times NVFP4 or MXFP8 weights decoded to bf16 in registers, then `m16n8k16`: about 55 TFLOPS on the
  vision tower's linears at 4,096 rows; the smaller tile (4 warps, 2 stages) was 10% faster at 1,024 and 4,096
  rows. Decoding is paid a weight a tile, so the quantized tower is slower than CUTLASS on its BF16 weights
  (73) for a third of the bytes (spec 019 WS-08).
- Memory: a kernel that only reads gets 215 to 235 GB/s from 80 blocks of 128 threads up, at 6 MB as at 4 GB;
  20 blocks of 128 get 85 GB/s, 40 get 154 (spec 070,
  [`references/memory-bound.md`](references/memory-bound.md)). A weight-streaming GEMM at 200 to 230 GB/s is
  at the ceiling: TMA or a fused epilogue cannot shorten it, fewer bytes can.
- K split at decoding rows: one block a tile that adds its slices in slice order beats a cluster of 8 and
  gives the same bits (MXFP8 2560 x 6144 at 1 to 32 rows: 69.4 to 70.0 us against 82 to 85; NVFP4 45 to 49
  against 55 to 57.5). From about 40 tiles the tiles alone fill the SMs, and the cluster's meeting only costs
  (spec 105 WS-03).
- A bf16 GEMM at 1 to 16 rows: a weight of 2 MB or more is its bytes at the bus plus some 2 us; one of 1 MB or
  less is floor-bound, 6 to 7 us from L2 as from memory; plus b16_reduce's 1 to 2 us. CUTLASS Sm80 Slim with 2
  stages was no faster; a hand `m16n8k16` kernel was 1.5 to 3x slower on every shape — take Slim, do not
  hand-roll (spec 114 WS-02).
- Routed experts W4A4 on `mxf4nvf4` (GLM FP4-Local, a rank's 288 experts, 2 GB a layer): at a 2,048-row window
  gate and up are at the bus (214 to 230 GB/s) in A16 and A4 alike, so A4 gains only where an expert's weights
  are read fewer times: 128 pairs a reading 11.88 ms against A16's 14.25 on real picks; a call's NVFP4 rows
  cost 0.56 ms of it, the 7-way branch of the e2m1 code 20% of that (spec 114 WS-03).
- An epilogue that quantizes an m16n8 product's rows to NVFP4 in registers instead of writing bf16 and
  launching a quantizer: 0.27 to 0.34 ms less a 2,048-row W4A4 experts call (2 to 3%), 0.1 at 512 rows;
  decoding's 1 to 6 rows within the spread (spec 114 WS-07).
- An LLM decoding round at one row is bound by the weight bytes read, the SMs at 4 to 23% of their peak:
  fusing launches saves little there; reading fewer bytes, or more rows a pass, does (spec 038).
- Unified memory: page cache and the CPU's allocations compete with the GPU's; check `free` as well as
  `nvidia-smi` before a large load.
- `cudaMalloc` of 142 MB on a serving path: 5 ms with 100 GB of the device held, 50 to 100 ms with 108 GB, and
  `cudaFree` 7 to 25 ms. Keep the buffers instead of allocating them per request (GLM's snapshots, spec 107 WS-03).

## Traps

- A paper's "Blackwell" speed-up through `tcgen05`, TMEM or 228 KB of shared memory does not transfer.
- cuDNN and cuBLAS pick an algorithm by shape: where bits must not depend on the batch, do not call them.
- nvcc contracts to FMA by default; `--fmad=false` where a replay must reproduce serial steps.
- Padding is not free in a memory-bound launch: 60 padded columns of 384, read and not stored, were 1.5 us.
- 36 launches of 1.7 us as one launch of 36 blocks: 22 us cold, 10 to 12 warm. A launch's floor is paid a
  launch, whatever it does (spec 073 WS-07).
- A GEMM after a memory-bound writer is slower than alone: `out_proj` after `chain_kernel` 163 us for 144
  (+13%), after a small kernel 146 (spec 073 WS-04). Time a candidate in the order the round launches it.
- Two loads of one binary and one setting differ by up to 1.5% of decode tok/s (0.3% on the 3.05 bpw pack,
  0.9% of a cold prefill): a round's claim under that is not one. The standard is off, on, on, off
  ([`references/measuring.md`](references/measuring.md), spec 073 WS-04).
- `m16n8k32 kind::mxf8f6f4.block_scale.scale_vec::1X`, e4m3 x e4m3, ue8m0 scales (probed on the GPU,
  2026-10-19): A is `m16n8k16` bf16's A fragment with each b16 a pair of bytes (row g / g+8, k 4t..4t+3 and
  16+4t..), so `ldmatrix.x4` on b16 loads it; B: b0 k 4t..4t+3, b1 16+4t.., column g. Scale A with thread-id 0:
  lanes t%4 == 0 give row g, t%4 == 1 row g+8; scale B: t%4 == 0 give column g; byte-id an immediate picks the
  register's byte. Sums at fp32's error (9e-8 relative).
- An FP8 GEMM on it (`qmm_a8.cu` bm 64: 64 x 64 tile, 4 warps, 3 stages of 128 K, 49,920 B, two blocks an SM, rows
  quantized by a kernel before): 67 to 88 TFLOPS at 512 to 4,096 rows of GLM's shapes, the quantize kernel
  included, against 45 to 64 for bf16 rows times the same weight decoded (`qmmf`); a 128-row tile on 8 warps
  (2 stages, 159 to 194 registers, or spills at two blocks an SM) was 9 to 50% slower (spec 114 WS-04). The
  instruction's own peak is not measured.
- IEEE division (`__fdiv_rn`) is dear in a memory-bound kernel that computes per element: 16 a thread a stage, in
  every block, made a block-FP8 decode GEMM 1.3 to 2.3 times slower than the bus; one `__frcp_rn` a group and
  q0 = v r, rn(q0 + fma(-q0, s, v) r) (Markstein; the e4m3 bytes held to division on 4M host pairs) are the bus's
  time (spec 114 WS-04).
- Plain `mma.sync m16n8k32.row.col.f32.e4m3.e4m3.f32` runs (one-warp probe against an f64 host sum, 3e-6
  relative): fragments as the block-scaled one's. An FP8 B whose K runs down a shared array's rows (V in P.V) is
  `ldmatrix.x4.trans` on b16 (a lane: keys 2t, 2t + 1 of values 2g, 2g + 1, a matrix 8 keys) and two `prmt`s a pair of
  matrices (`0x6420` the even values, `0x7531` the odd), each the B of its own mma, with A's k slots permuted alike
  (spec 114 WS-05, `glm5/mla_q8.cu`). That attend (records raw at a 528-byte stride, conflict-free for `ldmatrix`;
  38.5 KB, two blocks an SM) took 60 / 73 us at 1 / 8 rows and 3.12 ms for a 512-row window where the bf16 mma
  over records widened to bf16 took 80 / 93 us and 4.74 ms.
- A synchronous `cudaMemcpy` from the device into pageable host memory (the host allocator's, mapped and freed
  over and over) sometimes misses it, with no CUDA error (driver 580.178.04, ATS addressing): pages of the
  destination keep what the CPU wrote (Zig's 0xAA), from a 64-byte line on, and other pages of the process
  change. Reproduced in 10 of 60 runs of the rank rounds' downloads, never (0 of 60) into `cudaMallocHost`
  memory; a bare loop of mapped buffers and downloads did not show it in 40,000 copies. Download through
  page-locked staging; the staging costs 22.8 us against 12.5 direct at 296 KiB, 2.5 to 3 times from 4 MiB up
  (spec 132).
- Not known, to be measured when needed: the FP8 and FP4 `mma` peaks in TFLOPS, whether TMA beats `cp.async`
  on a memory-bound GEMM, why a stage of 48 values of K is slower than 32 or 64.

## Sources

- PTX ISA: the `cuda` skill, `references/ptx-isa/index.md` (shared state space table; `mma` Target ISA
  Notes; `cp.async.bulk.tensor` restrictions).
- This skill's `references/specs.md` (cudaDeviceProp) and `references/cutlass.md`.
- CUTLASS: `include/cutlass/gemm/collective/`, `include/cutlass/epilogue/fusion/`, `include/cute/arch/`,
  `CHANGELOG.md`; in zspark under `zig-pkg/`.
- zspark: `zig/kernels/AGENTS.md` ("CUTLASS", "Mega kernels").
- CODA: arXiv 2605.19269 (Guo, Thakkar, Dao and others); FlashAttention: arXiv 2205.14135.
