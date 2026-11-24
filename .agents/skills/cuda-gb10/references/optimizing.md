# Optimizing a kernel on the GB10: three levers

The owner, 2026-10-10, in this order: use what CUDA gives on this chip; fuse a chain as FlashAttention does;
pack memory-bound and compute-bound work into one kernel as CODA does. Each lever below says what it is, when
it pays here, what zspark has and what does not pay here. A new measurement rewrites the entry it touches to
the rule that now holds; its history goes to the spec archive.

First, for any kernel: its bytes read a launch over its own duration against the memory's 215 to 235 GB/s,
and its products against the tensor cores' (bf16: 80 to 91 TFLOPS reached by CUTLASS on a window, about 106
the pipe's peak by ncu; the 33 of spec 011 was one hand chain's, `SKILL.md`). That says which side it is bound
on, and which lever applies. A decoding round at one row is 85 to 90% weights at the memory's rate, the SMs at
4 to 23% of their peak (specs 038, 070): memory-bound almost everywhere, with compute to spare.

## 1. CUDA's features on this chip

What the chip has (the tables in `SKILL.md`) against what zspark's kernels use, in `zig/kernels/`. An unused
feature is a candidate, not a promise: measure both forms.

| Feature | Used | Where, or what it would take |
| --- | --- | --- |
| `mma.sync` bf16 / f16 `m16n8k16` | everywhere | every product: the bf16 GEMM, FP8 and NVFP4 linears and experts (weights widened to bf16 in registers), EXL3 (trellis decoded to fp16 fragments), attention |
| `mma.sync` FP8 (`.e4m3`), `kind::f8f6f4` `m16n8k32` | GLM, since spec 114 | plain e4m3 in DSA's P.V (`glm5/mla_q8.cu`) and the indexer's scores (`glm5/index.cu`, spec 116 WS-05, about 60 TFLOPS at a 4,096-row window). The instruction's own peak is not measured |
| `kind::mxf4nvf4` / `mxf8f6f4` with `.block_scale` | GLM's prompt windows, since spec 114 | `mxf8f6f4` m16n8k32 for FP8 rows on MXFP8 and block-FP8 weights (`nvfp4/qmm_a8.cu`, 67 to 90 TFLOPS) and DSA's scores; `mxf4nvf4` m16n8k64 for the W4A4 experts (`experts_a4.cu`, `experts_tile_a4.cu`). Where both operands are narrow only by the card's forms ([`precision-forms.md`](precision-forms.md)) |
| `cp.async` staging into shared memory, a ring of stages | yes | 190 GB/s and more against `__ldg`'s 110 to 120; 4 to 6 stages of 64 K values, on 128-byte lines |
| `ldmatrix` | yes | fragments from the stage |
| TMA (`cp.async.bulk.tensor`) | no | one thread issues a tile's load, no address arithmetic a lane; CUTLASS's SM120 collective assembled by hand ran for bf16. Whether it beats `cp.async` on a memory-bound GEMM is not measured |
| Warp specialization (producer warps load, consumer warps multiply) | no | CUTLASS SM120's form; goes with TMA |
| Clusters, DSMEM | yes, where blocks are few | the meeting costs 6 to 7 us; from 40 tiles one block sums its slices |
| CUDA graphs | yes | a round is replayed: no launch cost between kernels, which is why joining two launches saves little |
| Hand-off inside a launch (`atomicAdd` counters) | withdrawn | 2.8 us for an empty 48-block launch; the test kernel and `ZSPARK_FUSED_READOUT` were deleted (spec 121) |
| L2 persistence (`cudaStreamAttrValue`'s access policy window) | lost | 18 MB persisting; a window over a prompt's rows made the NVFP4 experts 1.5x slower (spec 107 WS-03); a small hot table (norm weights, scales) not tried |
| Prefetch (`prefetch.global.L2`) | three kernels | `experts_wide.cu`, `gdn_wide.cu`, `gdn_slots.cu`; refused for `linear_kernel` (spec 039) |
| Unified memory (host and device share pages) | at load only | the load's kernels read the pageable O_DIRECT stage through ATS and unpack into device buffers ([`loading.md`](loading.md)); weights are still copied into device allocations, pinned host memory as device memory cost about 6% of decoding's bandwidth |

Rules that came out of using them (details in `SKILL.md`, "Traps"): shared memory is the SM's and holds weight
on its way first; all 48 SMs busy; time a kernel by its own duration.

## 2. Fusion as FlashAttention does it

The form: a chain whose intermediate is larger than its inputs and outputs (scores, rows by keys) is never
written. One kernel walks it tile by tile, the tile's intermediate living in registers and shared memory, and
carries a small running state across tiles (the online softmax's maximum and sum). What is saved is the
intermediate's trip through memory, twice.

- When it pays: the intermediate does not fit L2 (24 MB), or is written by one launch and read by the next at
  a size where that trip is a share of the kernel. That is rows times something: a prompt's window, many
  streams, attention over a long cache.
- When it does not: a decoding round at one row. Its intermediates are 0.65 GB a round and stay in L2;
  joining two launches there saved under 2% (spec 038), and the read-out's fused launches gained nothing on
  bf16 (spec 070).
- The row rule: the tile order and the running state's reduction order come from the shape alone, never from
  how many rows share the launch. A row's result must be the same alone and among others; test it so.
- What zspark has: `qwen4/attention.cu`, `glm5/mla.cu` and `common/dflash2_attend.cu` are online softmaxes on
  the tensor cores (`attention.cu`: tiles of 64 keys; bf16 probabilities). The DFlash2 drafter's first
  attention, a key a warp with fp32 `exp`, was 5.3 times slower than this form (0.129 against 0.024 ms).
- Candidates not measured: a prompt's 512-row window by kernel (spec 070's backlog), where intermediates are
  rows times width and the chain norm, GEMM, activation, GEMM, residual writes each; the delta-rule chains
  (`chain_kernel`, `replay_kernel`).

## 3. Packing memory-bound and compute-bound work, as CODA does

CODA (arXiv 2605.19269) leaves a GEMM's mainloop alone and runs what follows it in the epilogue, on the output
tile still in registers: residual, the norm's weight, SwiGLU, rotary pairs, a row's sum of squares as partials.
The general idea: a kernel waiting on memory has idle tensor cores and ALUs, and a kernel saturating them has
idle memory bandwidth; put one of each in the same kernel so both are busy.

On this chip the two sides are far apart at decode (memory at its ceiling, SMs under a quarter), so the
question for a memory-bound kernel is what compute to give its idle SMs. In order of what it has paid:

1. Decoding a denser storage of the same weight. Fewer bytes arrive, more arithmetic unpacks them: block FP8
   for bf16 halved the bytes and gave the hybrid's prose +14%, +19% with the draft head (40.5 to 46.1 to 48.2
   tokens a second, spec 070); withdrawn as a default (spec 071): the model runs as stored, so the pack lives
   on for drafts only, a draft 1.97 to 1.55 ms. NVFP4 experts
   are at the ceiling. The balance point is where both are near full: a format too dense is
   compute-bound and wants fewer operations a byte (a wider program, a table), one too wide is memory-bound.
2. A second reader of the same bytes: more rows a pass (a drafted row costs 3 to 4 ms where a round costs 50),
   two projections of one input read once, a weight asked for while the previous one is multiplied (the
   `hc_read_one` form, withdrawn in spec 121).
3. The next steps' arithmetic in the epilogue (CODA proper): what follows the GEMM on its output tile, and a
   row's norm factor moved past the next GEMM into that one's epilogue (allowed once its error is measured;
   not built). At one row its bound was 4 to 10% of a round and bit-equal forms gained 0 to 2%; at many rows
   the intermediates are what moves and the bound is another, not measured.

And for a compute-bound kernel (the trellis, attention over a long cache): give it memory work that is
waiting anyway, such as the next layer's first stages asked for (`cp.async`, a prefetch) while it computes.
Not tried.

Every packed kernel keeps the account of its shared memory (`SKILL.md`): what is packed in must not take the
bytes the weight in flight needs.
