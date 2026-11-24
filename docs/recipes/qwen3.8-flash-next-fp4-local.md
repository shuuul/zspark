---
checkpoint: nvidia/Qwen3.8-Flash-Next-FP4-Local
revision: b9ce18a6887e18c089fdea8509a33e63429a3300
model: Qwen3.8 Flash Next
family: qwen4
storage: NVFP4 routed experts with csf scales, NVFP4 n-gram table, BF16 head and MTP, MXFP8 side layers
license: NVIDIA Open Model Agreement
disk_gb: 102
machines: 1
device_gib: 72.5
status: supported
drafters: MTP head, the default; DFlash2 ([`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) at `a22a1744e85c17364dcc54a6a420af7135d4c7f6`, Apache-2.0) with `--drafter`, one rank and two (spec 126; at two ranks since spec 162 every stream of the shared pool drafts by it, each rank its own share and a context a stream; the two-rank `serve --parallel 1` line: stored 0.886 of MTP's decode, mixed 0.950, cold 4k first token 1.69 s against 1.49, spec 135 WS-03)
variables: ZSPARK_FLASH_NEXT_FP4_LOCAL
---

# Qwen3.8 Flash Next, FP4-Local (`nvidia/Qwen3.8-Flash-Next-FP4-Local`)

**Machines: one DGX Spark**, which holds it: 72.5 GiB of device memory under a `generate` load, bf16 caches
and no pool (spark-a, 2026-10-18, `000c579`); a `serve` load at the default context is not measured yet (spec
105 WS-03). Two ranks are optional (`zig build -Dnccl=/usr`, `tools/two_ranks.sh
serve DIR`): 44,276 MiB a rank after a `serve` load at context 262144, and a stream passes one Spark
([on two ranks](#on-two-ranks)). Every tier passes on it: the full tier against the vLLM
container's reference and `check.sh ranks` (d0586321, 2026-10-21).

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `nvidia/Qwen3.8-Flash-Next-FP4-Local` at `b9ce18a6887e18c089fdea8509a33e63429a3300` |
| Base model | `Qwen/Qwen3.8-Flash-Next`, trained on by quantization-aware distillation (the card) |
| Licence | NVIDIA Open Model Agreement (`license_name: nvidia-open-model-agreement`); the base model's Qwen Community License 1.0 |
| Producer | ModelOpt, `quant_algo: MIXED_PRECISION` in `config.json`'s `quantization_config` (no `hf_quant_config.json`) |
| On disk | 102 GB (95.2 GiB): 21 `tensors/model-000NN-of-00021.safetensors`, 375,487 tensors |
| TensorFold family | `qwen4_exp`; in zspark `zig/src/models/qwen4/`. No TensorFold reads `csf` |
| Variables | `ZSPARK_FLASH_NEXT_FP4_LOCAL` (`docs/environment.md`) |

The model, from `config.json` (`zspark info DIR`), has hidden size 2560, vocabulary
248,320, 48 layers (36 DeltaNet, 12 attention), 4 residual streams with hyper-connections of rank 320, 512
experts a layer of width 640 with 10 a token and a shared expert, a 3-gram PLE table on layer 1 (16 heads of
160, 128 shards), one MTP layer.

## What is stored, and how it is multiplied

`zspark info DIR` counts them:

| Storage | Tensors | Size | Which | Multiplied by |
| --- | --- | --- | --- | --- |
| NVFP4, block scales `csf`-encoded | 73,728 linears | 61.3 GiB with the next row | the routed experts of the 48 layers: 512 x (gate, up, down) | `zig/kernels/nvfp4/`'s experts kernel, bf16 activations, on the stored codes and the scales decoded at load |
| NVFP4, plain e4m3 scales (`W4A16_NVFP4`) | 1,536 linears | | the MTP layer's routed experts | the same kernel, as stored |
| MXFP8 (e4m3, a UE8M0 exponent a 32) | 384 linears | 2.8 GiB | DeltaNet's five projections, attention's q, k, v, o and `index_qk_proj`, each layer's shared expert | `qmmf.cu`'s lane matmul, mode 2, bf16 activations (W8A16): decode rows and prompt windows alike |
| BF16 | 410 linears | 3.9 GiB | `ignore`'s: the embedding, the head, routers, hyper-connections, the MTP's projections, attention and shared expert | `zig/kernels/bf16/`, as stored |
| NVFP4 n-gram table (e4m3 a 16, one F32 `weight_scale_2`) | 128 shards | 26.8 GiB | the PLE's table, 2,500,012 x 80 a shard | read through the file's mapping and decoded on the host a row at a time, as TensorFold's `NVFP4Table`; not uploaded |
| The vision tower: MXFP8 | 54 linears | 141 MiB | each block's attention qkv and proj | `nvfp4/prompt_bias.cu`, mode 2, bf16 activations (W8A16), the bias and GELU in its epilogue |
| The vision tower: NVFP4 (`W4A16_NVFP4`, plain e4m3 scales) | 56 linears | 161 MiB | each block's MLP fc1 and fc2, the merger's fc1 and fc2 | the same kernel, mode 0; fc2's K of 4,304 read as 4,352, zero codes and scales past it, fc1 writing those columns as zeros |
| The vision tower: BF16 | 333 tensors with the linears' biases | 9.2 MiB | the Conv3d, the position table, the norms, the biases | the BF16 path in `zig/kernels/bf16/` and `qwen4/vision.cu` |

## What changes while it loads

- **The routed experts' `csf` scales are decoded** to the e4m3 bytes the experts kernel takes, a layer at a
  time on every host core (`weights/csf.zig`, from vLLM PR 60646, lossless): 7.03 GiB on the device against
  3.69 compressed. One core decodes 2.2 GiB/s, sixteen 7.0 (a bench of the loader's tasks into fresh memory):
  1.0 s of a load against 3.2. The card's container decodes them at load the same way by default.
- The MXFP8 weights are laid in the lane matmul's fragment order and their exponents beside them; the bytes
  are the stored ones.
- **Requantized for drafting only**: the draft head is the head's draft-vocabulary
  rows as NVFP4 (`ZSPARK_DRAFTS_STORED=1` scores drafts on the stored bf16 head). The MTP experts are stored
  NVFP4 and run as stored.
- No weight is quantized. Activations: a prompt's windows of more than 80 rows run as the card's vLLM does a
  prompt's chunks (spec 143 WS-06 and WS-08, `ZSPARK_PROMPT_FORM`, on by default): the MXFP8 linears on FP8 rows,
  each as b12x's time race chose it at its 4,096-row capacity (DeltaNet's b and a, 96 outputs, alone A16), and
  the routed experts W4A4 (`B12X_W4A16_A4_PREFILL`, each route stored bf16 before the top-k sum); steps, verify
  windows and shorter windows keep bf16 rows, one form. Figures of the MXFP8 linears alone (WS-06's form):
  spark-a, `ef5d610f`, three loads (card bf16 card), three prompts each, the machine alone: a 4k prompt's first token 1.92 s against
  2.065 (bf16 to bf16 0.006 apart in four loads), 16k's 7.62 to 7.67 against 8.16 to 8.17; the load 8.2 s either way.

### How it loads

The GB10's load pipeline (`.agents/skills/cuda-gb10/references/loading.md`), 2026-10-20, branch `fn-load`:
the embedding, each layer, then the mixer and the head are read by O_DIRECT into two kept stage buffers, a
part ahead of the loader (`weights/loading.zig`, `model_load.Reading`); the routed experts are placed by the
device from the stage into the A16 kernels' blocks (`zs_nvfp4_experts_pack_a16`), their csf scales decoded on
the host (Flash Next's csf is `weights/csf.zig`'s, not GLM's `csf_32x2_v1`). A rank reads its rows of gate and
up, down whole, and takes down's columns where they lie; serve reads the first two parts while the ranks join.
`ZSPARK_LOAD_PHASES=1`, `serve --context 262144`, the
page cache dropped by O_DIRECT, the machines held alone, the same three greedy replies' shas before and after:

| | Before (`b2155950`) | After |
| --- | --- | --- |
| One Spark: model loaded, s | 15.9 | 8.2 |
| One Spark: start to `/v1/models`, s | 17.3 | 9.6 |
| Two ranks: a share loaded, s (rank 0, rank 1) | 14.4, 14.7 | 5.2, 5.6 |
| Two ranks: start to `/v1/models` (`loaded in`), s | 16.5 | 7.3 |
| Read a rank, GiB | 68.0 | 49.4 |

Both are now bound by the reads (68 GiB at 9.2 GiB/s on one Spark; 49 GiB at 9.1 to 10.0 a rank), the loader
waiting 2 to 3 s for them; under them `cudaMalloc` (1,149 calls, 3.0 s on one Spark, 1.4 to 1.6 a rank).

2026-10-21, branch `load3`, one load each against one of `3a1653ab`, the replies' shas unchanged: each rank reads
its half of down's outputs and trades the inputs the peer keeps of them over NCCL (`experts.Swap`, as GLM's two
ranks), 39.8 GiB a rank; the MTP head is a staged part after the head, placed by the device as the layers'
experts (rank 0's whole, or each rank its half under a split head); one Spark reads its first parts while the
tokenizer and the template load. The halves are spans of some 450 KiB, which 8 reader threads read at 8 GiB/s
(no gain: ready 8.6 s); the stage reads spans under 640 KiB on 16 (`safetensors.Stage`):

| | Before (`3a1653ab`) | After |
| --- | --- | --- |
| One Spark: model loaded, s | 8.67 | 8.50 |
| One Spark: launch to `/v1/models`, s | 9.42 | 9.12 |
| Two ranks: a share loaded, s (rank 0, rank 1) | 5.20, 5.22 | 4.32, 4.31 |
| Two ranks: launch to `/v1/models`, s | 8.51 | 7.77 |
| Read a rank, GiB | 49.4 | 39.8 |
- **`serve` loads the vision tower** beside the model at any `--parallel`, and a request's image parts are
  read (spec 019 WS-04 and WS-05; [`docs/serving.md`](../serving.md), "Images"): the tower's 302 MiB of linears
  as stored and one features buffer of 4,096 rows (20 MiB, bf16), shared by the session and the pool's seats;
  above `--parallel 1` each slot holds an image's positions too (12 bytes a context row). This checkpoint alone serves images (`Config.servesImages`:
  mixed precision with MXFP8 layers).

## Defaults that follow this checkpoint

| | Here | Why |
| --- | --- | --- |
| Caches (`--kv-dtype`) | bf16 for `chat`, `generate` and `--parallel 1`; FP8 above `--parallel 1` on one Spark, from the load; bf16 on two ranks at any `--parallel` | With `--parallel` above 1 a load has one storage, for one live stream and for the pool, so a reply does not change when a second stream opens the pool. The pool is `--parallel` contexts of `--context` tokens, 1,048,576 positions at the defaults, and a bf16 pool of that size does not fit beside this checkpoint; the storage does not follow a smaller `--context`. Two ranks hold half the KV heads each and a bf16 pool fits (65.8 GiB of buffers a rank at `--parallel 4`); FP8 kept 3 to 17% fewer drafts on every text measured, so two ranks take bf16: 149 tok/s summed at 4 streams against 134, one stream 72.4 against 63.8 (spec 144 WS-01, the owner 2026-10-26). An explicit `--kv-dtype` wins for both; `--parallel 1` or `--kv-dtype bf16` with fewer positions gives bf16's rate |
| Drafts a stream chains in the pool (two ranks) | at most 4 while 3 or more streams decode in the round (`pool/hall_depth.zig`); below 3, and on one Spark, the head's own depth (15) and rule | spec 144 WS-04: on two ranks the head's levels are eager and the round waits for its deepest stream; WS-04's model over WS-01's rounds put a cap of 4 at +4.8% summed at 3 streams and +7.0% at 4, -3.0% at 1, +3.8% at 4 on one Spark (its head graphed). A departure from TensorFold (6 at any width), vLLM (opt-in, no default) and SGLang (cuts by width from 8 streams). Drafts choose nothing: the replies are the same. Bench owed |
| The drafting head's cache | bf16, beside the trunk's FP8, in the session and in the pool | spec 052: the trunk checks every draft, so the head's storage changes how many drafts are kept and never a token |
| A prompt's window | 4,096 rows in one rank's lone stream (`chat`, `serve`'s session, before a second stream opens the pool) and in a pair on two ranks (`--parallel 1`); 2,048 in a round of the pool (on two ranks the flag's where given, spec 149 WS-06); `--max-num-batched-tokens` 2,048 to 4,096 | spec 031: a wide window's DeltaNet chain runs as three kernels, every row with the bits a decoding step gives it. Spec 148 WS-02 (the owner, 2026-10-28; vLLM's card runs `--max-num-batched-tokens 4096`): a 16k cold prompt 6.86 to 6.08 s for 2.61 GiB more after the load. Spec 149 WS-04 (the owner, 2026-10-28), the pair on two ranks, four loads: cold first token 9 to 12% sooner on corpus prose (16k 5.43 to 4.95 s), 3 to 5% on sparkDash's text, 3.2 GiB more a rank |
| `--parallel` under `zspark stack` | 2 beside OCR and ASR on one Spark, 3 beside one of them; serve's own alone (4); 4 on two Sparks, not two ranks' serve's 1 | spec 139 WS-03: 4 at the full context does not fit beside them (beside ASR alone it leaves the states 0.15 GiB); `--parallel` lowered before `--context`, by the spec's decision. Two Sparks, 2026-10-26: each rank's bf16 pool of 4 leaves 37.6 and 39.4 GiB free (FP8's 43.5 and 45.7); prose, 512 tokens, 70.3 tok/s at 1 stream, 100.3, 114.1, 127.5 summed at 2 to 4 (at `--parallel 1`, 77.4 alone and about 71 to 77 summed at any concurrency). `-- --parallel N` chooses; "Beside OCR and ASR" below |
| The pool's extents | `--parallel` x `--context` positions: at the defaults 17.4 GiB for 1,048,576, FP8 with the head's in bf16 | said at the load, and refused there where the device cannot give them or the rest of the pool after them (spec 084: 13.45 GiB at the defaults in the session's 7.01, 18.69 for eight streams of 196608 in its 5.74, by the load's own line at `7ec1f4e`); the account is under "Several streams at once" |
| A prompt's chunk beside replies (`--prompt-chunk`) | what the replies' windows leave of the round's 2048 rows: 2032 beside one reply, 1984 beside four | spec 101 (the owner: vLLM's nearest): vLLM's and SGLang's chunked prefill is one budget of tokens a step that the decodes take first, 8,192 on a device of this memory (vLLM 3313825 `get_batch_defaults`, SGLang 1291f31 `memory_hook.py`); a round's buffers grow with its rows (11.2 GiB of windows at 2048) and four times that does not fit beside the pool, so the budget is the model's rows. 512 until spec 101. `--prompt-chunk N`: at most N, 1 to the round's rows |
| The states' budget (`--prompt-cache-gib`) | 6.66 GiB or the memory available after allocating the pool | spec 101: counted at the load's two steps as vLLM reserves its cache; a budget given is held whole or the load refused. Held out whole, the defaults were refused with 31.93 GiB left (13.47 GiB of the rest against 12.84 of room) |
| `--spill-gib`, `--max-queued` | off; no bound | vLLM's and SGLang's defaults: `--kv-offloading-size` None and the hierarchical cache off; `max_num_queued_reqs` and `max_queued_requests` None |
| Device buffers | the model and every session have their own allocations | |
| The DFlash2 drafter's storage (`--drafter`) | `mixed`, the default since 2026-10-25 (the owner): its large linears NVFP4 made at load and a four-bit copy of the head proposing, its picks scored again against the stored rows; `ZSPARK_DRAFTER_STORAGE=stored`: BF16 as the file has it, every linear and the proposing head | spec 126 WS-03 kept it stored; spec 135 WS-03 at two ranks: `mixed` decodes 1.072 times stored (31 of 36 cases), 0.86 GiB less a rank, load +0.3 s, 0.74% fewer drafts kept; the owner made it the default (TensorFold's drafters propose in four bits, `--drafter-bits 4`). Drafts only propose: a reply's tokens are the same either way |
| n-gram table | mapped from disk, out of the direct reads; its drive kept awake while rows are gathered (`weights/drive.zig`, spec 089) | as on every checkpoint (`zig/src/models/qwen4/AGENTS.md`) |

`--max-num-batched-tokens N` on `serve` and `chat` (vLLM's flag, 2,048 to 4,096; spec 105 WS-07) sets the rows
the model is loaded for and a lone stream's prompt window. Unset it is 4,096 on one rank (spec 148 WS-02, the
owner: an option until then), so the routed experts are read half as often as at 2,048; `--max-num-batched-tokens
2048` gives the earlier load and its 2.61 GiB back. The pool's rounds keep 2,048 rows (at 4,096 they take 25.1 GiB
of slots and windows against 13.5, and the defaults' pool of four streams of 262,144 would not fit). On two ranks
unset is 4,096 too and the flag one reply's (a pair's windows, spec 111) and, since spec 149 WS-06,
the pool's rounds (`common.poolRows`; unset 2,048, the owner's to change): at 4,096 and `--parallel 4` a rank
holds 79,819 MiB against 68,403 (round windows 20.3 GiB against 10.2) and the host keeps 35-36 GiB available
against 46-48; one Spark's pool keeps 2,048 beside its lone session. The figures under "As a client sees it".

Two-rank DFlash2 serves four streams through one shared model pool. Both machines need the
same checkpoint and drafter paths, and the same NCCL-enabled binary:

```bash
tools/two_ranks.sh serve ~/models/Qwen3.8-Flash-Next-FP4-Local \
  --drafter ~/models/Qwen3.8-Flash-Next-DFlash2 --parallel 4
```

Each rank holds its drafter share once and a separate context per stream. Without `--drafter`, MTP stays
the default. Spec 162 validates exact replies and lifecycle behavior; it makes no throughput claim.
Our LAN stack explicitly selects this DFlash2 configuration by default since 2026-11-11, beside Nemotron
ASR on both Sparks; its launch command is in `docs/serving.md`, under the stack examples.

## Measured

### DFlash2 versus MTP in the LAN stack

2026-11-11, clean `6a886889`, GB10 TP2, `--parallel 4`, bf16 KV, mixed DFlash2, with Nemotron ASR
loaded on both Sparks and MPS off. Both machines held alone for three measured loads:
DFlash2, MTP, DFlash2. sparkDash decode used 400 tokens, temperature 0.7, thinking on, top-k 20,
top-p 1 and empty start context; each cell is the median of three jobs. Its Structured prompt asks
for counting numbers; it does not request a JSON schema or a grammar. Four-stream rates are aggregate
client first-to-last-token throughput, not four times the median stream's rate. Seeds were not fixed.

| Workload | DFlash2, first load | MTP | DFlash2, second load |
| --- | --- | --- | --- |
| Structured, 1 stream, tok/s | 132.92 | 155.38 | 126.72 |
| Prose, 1 stream, tok/s | 75.11 | 67.85 | 76.79 |
| Structured, 4 streams, tok/s summed | 263.64 | 250.28 | 292.87 |
| Prose, 4 streams, tok/s summed | 166.80 | 161.38 | 191.11 |
| Cold 4k prompt, first token, s | 1.608 | 1.519 | not repeated |
| Cold 16k prompt, first token, s | 6.669 | 6.943 | not repeated |

The prefill cells are sparkDash's medians of three near-cold samples, with 7–9 cached tokens; their
prompts used fresh salts. The short decode prompts were warmed and reused, so their TTFT is not cold
prefill. Sampled throughput ranges overlap; the second DFlash2 load's four-stream medians are higher
than the first's. This is a workload-dependent tradeoff, not a universal speed ordering.

The host completion logs account for all 130 measured streams and give the mechanism: sampled
Structured alone gets 7.74 tokens per verification round with MTP against DFlash2's 5.31–5.69, despite
DFlash2 keeping 87–88% of its offered drafts against MTP's 83%. MTP can chain fifteen drafts alone;
DFlash2's block is eight rows. At three or more active TP2 streams MTP chains at most four drafts,
whereas DFlash2 batches its streams' blocks in one pass. In the greedy, thinking-off diagnostic,
Structured alone is 218.78 tok/s with MTP against 189.58 with DFlash2, but four streams are 313.57
against 449.88. This changes two request settings together; it does not isolate sampling alone.
The greedy comparisons have matching logged token hashes across drafters for each prompt and stream
variant; the speed difference there is not from generating different answers.

The LAN default stays DFlash2: both sampled four-stream medians favor it, Prose alone favors it,
and switching to MTP would exchange those gains for Structured single-stream speed. Choose MTP by
omitting `--drafter` when that workload matters more. No model precision, draft depth or kernel changed.
The logs measure request wall time and rounds, not separate drafter, NCCL and target kernel durations.
The [drafter card](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2/blob/main/README.md)
benchmarks GB300 TP1 with an FP8 target and block four; those numbers are not a baseline for this stack.

### Host and rank-control attribution

Spec 166, 2026-11-11, clean diagnostic source `5a1d55f0`, fast build, CUDA 13.0.88 and driver
580.178.04 on two GB10s. The isolated server uses TP2, DFlash2, `--parallel 4`, context 262144 and
bf16 KV. Both machines are held exclusively; one diagnostic load covers sampled/greedy,
Structured/Prose and one/four streams, plus cold 4k/16k prompts. Each decode case has a prime,
an untraced wave, a node-traced wave and an untraced wave. These are attribution recordings within
one load, not a speedup comparison or a measurement of load-to-load spread.

The host-only segment study encodes already-rendered two-message conversations, at target sizes
512 and 16300 tokens, with rendered masking or no masking, identical/differing requests and one/four
workers. Each worker handles 24 requests; three clean repeats and one counter run per size/masking
give 16 matrices, all token IDs equal to uncached text. Clean request times are 0.01–0.02 ms at 512,
0.02–0.06 ms at 16300 with masking and 0.04–0.09 ms without masking. Counter runs show about
0.004–0.006 ms/request for hit copies. Other-worker duplicate inserts account for 22–47 cold misses
per four-worker batch; their wasted encoding is below 0.5 ms per 96 requests. Summed lock waits
overlap across workers and counters perturb the path, so they are not an additive speedup estimate.
The measured encoding path is too small to justify miss coalescing or new cache ownership machinery.

Control framing is small: no independent `flash_taps` or `flash_block` frames occur in any of the
87 requests. There is one `flash_round` per pooled round (32–83 payload bytes, about 9–15 µs of
host send time). Counting the other control sends without counting nested channel writes twice
gives 0.05–0.22% of wave wall time. Deriving those already-batched messages on rank 1 has little
host time to recover. Follower waits include the leader's GPU work and admission gaps.

In the sampled single-stream Structured trace, rank 0 executes 67 graph replays. Its 7,537 NCCL
AllGather kernels occupy 247.6 ms of a 2767 ms kernel span, with no overlap with compute kernels;
routed experts, bf16 GEMMs and block-FP8 products occupy most of the remaining device time. Both
ranks confirm serialized gathers during decode. This is communication dependency time; a trace
does not establish how much can be overlapped safely. Decode has no `columns_place` launches;
cold prompts do use placement, at less than 1 ms per trace on rank 1.

Reproduction: the corrected diagnostic source is in `5a1d55f0`; `scratch-166/diagnostic-final.patch`,
`host_runs.sh`, `launch.sh`, `run_cases.mjs`, `case_client.mjs`, `environment.json` and raw logs/traces
are under `~/.cache/zspark/`. The host command is `host_runs.sh FAST_HELD_TURN_BINARY` under the
exclusive machine lock. GPU commands are `launch.sh flash` and `node run_cases.mjs flash`;
the runner accepts comma-separated case names for staged captures. The final production source
removes all temporary probes.

### One `generate` load

spark-a, 2026-10-18, `000c579` (`zig-out/fast`), the machine held alone; cases of
`~/.cache/zspark/scratch-105/ws02/cases.txt`, bf16 caches.

| | zspark |
| --- | --- |
| Load (`generate`), the page cache warm | 15.9 s (the whole process 1:14 with the score and six replies) |
| Device memory after the load (`nvidia-smi`, the process) | 74,235 MiB (72.5 GiB) |
| Decoding, serial | 26.4 tok/s (37.9 ms a step) |
| Decoding, drafted | 54.5 tok/s on the sky reply (64 tokens), 60.1 on primes (32), 55.3 on 256 tokens of prose |
| Score, 5,336 tokens | nll 2.13927, ranked first 0.5547 |

Where a serial step goes (`nsys --cuda-graph-trace=node`, the same commit, a second load): the MXFP8
linears 14.3 ms (39%, their 2.9 GiB at about 210 GB/s; 6.7 of it the 36 DeltaNet layers' stacked input face,
188 us each), the BF16 linears 12.6 ms (35%: the head 5.0 ms, and about 240 small GEMMs of the routers and
hyper-connections 7.4), the routed experts 7.2 ms (20%), the rest 1.8 ms.

### As a client sees it

`tools/reference/client_bench.sh zspark` on spark-a, 2026-10-18, `52c8f610` (`zig-out/fast`), `CONTEXT=262144`,
seed 1, one load, the machine held alone (spec 105 WS-05; a row's two repeats where they differ); the card's
vLLM container (`client_bench.sh vllm`, MTP of 4) on spark-b the same day, `VLLM=card` as the card writes it
and `VLLM=a16` with its head and prompt activations at 16 bits.

| | zspark | vLLM card | vLLM a16 |
| --- | --- | --- | --- |
| Start to `/v1/models`, s | 17.0 | 999.9 | 378.7 |
| First token, 50-token prompt cold, s | 0.147 / 0.139 | 0.273 | 0.286 |
| First token, 4,000 held, 30 new, s | 0.109 / 0.107 | 0.261 | 0.254 |
| Prompt tok/s, 500 / 4,000 / 16,000 cold | 1146-1203 / 2031-2056 / 2079-2088 | 887 / 1857 / 2089 | 776 / 1711 / 1964 |
| prose-long / code-long / tool-call, tok/s | 47.4 / 83.6-85.7 / 124.0-126.8 | 33.4 / 55.9 / 77.9 | 31.6 / 54.7 / 82.7 |
| Every streamed reply but `warm`, tok/s | 60.6 | 46.6 | 44.3 |

The pool's drafting head pass, from its embedding to the head layer's last launch, runs as a captured graph
keyed by its rows and context bucket as a round's are (fn-tails, 2026-10-20, `d572555e`; spark-b held alone,
`serve --parallel 4` traced with `nsys`, `client_bench.py streams --counts 4`, prose then code, 128 tokens,
off, on, on, off). A pass from its embedding to the projection's start took 1,263 / 1,249 us off and 1,184 / 1,158 on (medians of 514 to 539
passes a load): 80 us a pass, about 0.5% of decoding at four streams, under what a load's tok/s resolves. The
replies, the drafts and the accepts (603 of 1,096) were the same in all four loads. A split head on two ranks
runs its passes kernel by kernel still.

A 16,000-token cold prompt (16,282 ids) takes 7.80 and 7.83 s to its first token: at par with the card's vLLM,
not past it, and past a16. A prompt window's attention runs a row a block (`attention_window.cu`, from 64
rows, `ZSPARK_ATTN_WINDOW`): `generate` on that prompt, four loads off, on, on, off, 8.69 / 8.35 / 8.35 / 8.70 s,
the same greedy tokens. What is left is the routed experts at 2048-row windows, bound by their decode chain a
reading, not by bytes. The decoding rows do not run the new kernel; their figure is the
load's spread around WS-03's (62.2).

`--max-num-batched-tokens 4096` (spec 105 WS-07, `347aa824`, the same harness with six cold passes a load,
`prompt_repeats` 6, one load each way, the machine held alone; `~/.cache/zspark/scratch-105/ws07/`):

| 16,000-token cold prompt, first token | unset (2,048 rows) | 4,096 rows |
| --- | --- | --- |
| Six passes, s | 7.92, 7.82, 7.82, 7.83, 8.40, 7.83 | 7.27, 7.45, 7.42, 7.24, 8.25, 7.40 |
| Median, s (prompt tok/s) | 7.83 (2080) | 7.41 (2199) |
| Device memory after the load (`nvidia-smi`), MiB | 81,683 | 84,302 (+2.56 GiB) |
| The session's windows and head (the load's line), GiB | 7.02 | 9.19 |

The same greedy tokens both ways. 0.42 s off, not the 0.7 s the experts' figure alone gave (WS-05): the
other kernels gain little from the width.

4,096 the default (spec 148 WS-02, 2026-10-28, `6b9a41ad` with spec 148 WS-03's fused read-out, the card's prompt
form; `client_bench.sh zspark`, `CONTEXT=262144`, seed 1, fp8 KV and `--parallel` 4, three cold passes a load, four
loads off/on/on/off, the machine held alone; `~/.cache/zspark/scratch-148/ws02/`):

| Cold prompt, first token | `--max-num-batched-tokens 2048` | unset (4,096 rows) |
| --- | --- | --- |
| 16,000 (16,282 ids), passes, s | 6.940, 6.851, 6.874 / 6.943, 6.851, 6.841 | 6.176, 6.064, 6.051 / 6.195, 6.090, 6.079 |
| 16,000, median, s (prompt tok/s) | 6.86 (2,376) | 6.08 (2,678) |
| 4,000 (4,062 ids), once, s (tok/s) | - | 1.678 (2,419) |
| 500 (513 ids), once, s (tok/s) | - | 0.427 (1,201) |
| Device memory after the load (`nvidia-smi`), MiB | 81,740 | 84,416 (+2.61 GiB) |
| The session's windows and head (the load's line), GiB | 7.06 | 9.26 |

-0.78 s (-11.4%) on 16k, against 0.03 s from load to load either way; Mia's AI Lab's recipe is 6.74 s.
Every pass of one salt gave the same reply hash both ways. At `--parallel 2` (the one-Spark stack's, beside OCR
and ASR) the load takes 84,410 MiB and the pool's 8.67 GiB of slots sit inside the session's 9.26, where at 2,048
they were 1.57 GiB past its 7.06: with the pool open it holds about 1.0 GiB more than before, not 2.6.

### On two ranks

Spec 105 WS-04, 2026-10-18, one binary on both Sparks (`6bbb6c0a`, `-Dnccl=/usr -Doptimize=fast`; the prompt
kernels of WS-03's last two commits not in it), `client_bench.json` seed 1, `CONTEXT=262144`, both machines
held alone: one Spark by `client_bench.sh zspark`, two ranks by `MACHINE=-x tools/two_ranks.sh serve` on
127.0.0.1:18888 and `client_bench.py run`, the MTP head half a rank (the default).

| | One Spark | Two ranks |
| --- | --- | --- |
| Start to `/v1/models`, s | 16.9 | 17.7 |
| Device memory after the load, MiB | 81,046 | 44,276 a rank |
| First token, 50-token prompt cold, s | 0.148 (0.151) | 0.140 (0.102) |
| First token, a turn's 30 new tokens on 836 to 4,366 held, s | 0.118 (0.116 to 0.122) | 0.141 to 0.151 (0.082 to 0.083) |
| First token, 23 to 46 new on 61 to 16,297 held (`prompt-*`, edited turns), s | (0.091 to 0.133) | (0.065 to 0.091) |
| Prompt tok/s, 500 / 4,000 / 16,000 cold | 1147 / 1739 / 1751 | 1487 / 1728 / 1755 |
| prose-long / code-long / tool-call, tok/s | 47.3 / 85.6 / 118.1 | 50.9 / 101.3 / 226.5 |
| Every streamed reply but `warm`, tok/s | 62.0 | 70.4 (1.14x) |

A rank cuts the checkpoint's tensors as stored: MXFP8's codes and exponents alike, NVFP4's codes and block
scales, and the csf streams are decoded for the rank's own window of the scale matrix (gate and up their rows,
down its columns, as the card's vLLM selects them). A two-rank reply is not one Spark's bit for bit (each rank's
fp32 partial added rank 0 first): on the vLLM cases it leaves vLLM's primes reply at token 19, where vLLM's own
first two choices are 0.31 apart, and holds the other two as one Spark does; the score is nll 2.14379 against
one Spark's 2.13927 and vLLM's 2.14806 and 2.14350, a token 0.197 from vLLM where vLLM's runs are 0.219 apart.
Its drafted, sampled, pool and held-state replies are its own serial ones in bits.

In parentheses, spec 110 WS-02 (`fe77bae2` on both and one Spark; the turn rows' base `33d3e4af` on two ranks
0.141 to 0.144), `client_bench.py --only warm,prompt,turns`, seed 1, `CONTEXT=262144`, both machines held
alone: a turn's new tokens are one window on both ranks, as on one Spark, where they were a window and then
the last 16. The 55 replies are the base's in bits; cold prompts gain a window less (500: 1394 to 1608 tok/s;
4,000 and 16,000: 1768 and 1799 to 1800 and 1812).

Spec 111 WS-01 (`16501d9f` on both, base `a366ed95`; `client_bench.py --only warm,prompt`, seed 1,
`CONTEXT=262144`, both machines held alone, two loads after): a pair's prompt windows are one Spark's 2,048
rows, where they were 512, so a 16,000-token prompt reads the routed experts 8 times, not 32. Device memory
44,276 to 48,454 MiB a rank. The 25 replies are the base's in bits.

| Prompt, cold | One Spark | Two ranks, 512 rows | Two ranks, 2,048 | Two ranks, `--max-num-batched-tokens 4096` |
| --- | --- | --- | --- | --- |
| 500, tok/s | 1166 | 1606 | 1589 / 1646 | 1602 |
| 4,000, tok/s (first token, s) | 2037 (1.99) | 1798 (2.26) | 2489 / 2556 (1.61) | 2728 (1.49) |
| 16,000, tok/s (first token, s) | 2068 (7.87) | 1810 (9.00) | 2553 / 2598 (6.32) | 2764 (5.89) |
| Device memory a rank, MiB | 81,046 (the Spark) | 44,276 | 48,454 | 54,028 |

Spec 111 WS-02 (`cbc666d5` on both, the same requests, two loads of two repeats): a pair scores 16 rows of a
window, not every row (bf16 logits of half the vocabulary: 48,454 to 47,955 MiB a rank at 2,048 rows, about
0.95 GiB less at 4,096, computed), and gathers its prompt's next window's PLE table rows while the device
runs the one before (`forward.expect`), where the host gathered them with the device idle. The replies are
the base's in bits.

| Prompt, cold, tok/s (first token, s) | 16 rows scored (`19a0eee3`, one load) | And the rows gathered ahead (`cbc666d5`) |
| --- | --- | --- |
| 500 | 1602 / 1744 | 1580 to 1760 |
| 4,000 | 2475 (1.65) / 2608 (1.56) | 2650 to 2678 (1.52) |
| 16,000 | 2539 (6.42) / 2616 (6.22) | 2775 to 2783 (5.85 to 5.87) |

A pair keeps the rows each layer keeps, as one Spark's session (fn-tails, 2026-10-20, `d089317d`): the states
inside a prompt's window are asked of both ranks before it runs, no longer taken from the window's every row
kept after it. A rank's device memory (`nvidia-smi`, both ranks, one load each way, 2,048-row windows) 47,955
to 46,128 MiB after the load and 48,622 to 46,795 after a 23,368-token cold prompt: 1,827 MiB (1.78 GiB) a
rank less. That prompt's reply, its 8 states (one inside a window) and its drafts were the same, and its first
token 8.36 s against 8.33.

Where a 16,000-token cold prompt's time goes on two ranks (nsys on both, at `19a0eee3`, before the rows were
gathered ahead; first token 6.24 and 6.46 s traced, 6.22 and 6.42 untraced). Its 8 windows of 2,048 rows
span 5,683 ms on the device; each rank's kernels other than NCCL's 4,885 and 4,814 ms; NCCL's 681 and 745 ms,
of which the link (each collective's shorter duration of the two ranks') 460 ms, the rest one rank waiting
for the other's compute (221 and 285 ms); and 50 to 78 ms of host time before each window (the PLE rows),
about 0.5 s, which WS-02 removed. A layer's link: the read-out's rows shared before the block and before the
MoE, 5.2 MB each way, 143 to 153 us (35 GB/s); the block's and the MoE's partials scattered, 10.5 MB, 350 to
364 us (30 GB/s); 47 layers split by rows and the first, the PLE's, whole: two all-gathers of
1.07 ms, 18 ms a prompt. The link is not hidden: each layer's collectives sit between kernels that need
them whole (a share before the block, a scatter before the next read-out), so hiding them takes a window cut
into parts whose experts are read once a part. Halving the experts' rows costs more than it could hide:
4,096-row windows took 0.37 s off a 16,000-token prompt against 2,048 (WS-01), and the link a part could
hide is at most 0.46 s, its first share and last scatter overlapping nothing. The pool's round on two ranks
splits its glue by rows the same way from 256 rows (a prompt's chunk, replies beside it or not; spec 149
WS-02), where it had gathered both ranks' whole partials each layer; a round of replies, captured or fewer
rows, keeps the whole glue.

### The vision tower

`tests/models/qwen4/vision_test.zig`, spark-b, 2026-10-18. The features, from TensorFold's
`pixel_values` of the reference images, against a BF16 reference tower (worst row, median row,
whole, relative): `shapes` 1.86, 0.31, 0.35; `bands` 0.64, 0.28, 0.25; `tiny` (one row) 0.62. Against
TensorFold's BF16 features they are the same (`shapes` 1.40, 0.31, 0.32), where the BF16 tower is 0.51, 2.8e-2,
7.2e-2: the spread is the checkpoint's quantization. A stored weight decoded on the host is 2.7% from the BF16
one as MXFP8 and 9.5% as NVFP4, and a block from TensorFold's BF16 input is 6 to 9% from its output, the merger
6 to 7% (the BF16 tower's blocks: 2 to 5e-3). How far the features' spread moves a reply was not measured.

A pass's wall time, random pixels, one image, the machine held alone (passes 1 to 3 of four, the first a
warm-up; the quantized tower at its tile 4 but where said; `ZSPARK_VISION_PROFILE`, kernels from `nsys`):

| Patches | BF16 reference tower | Quantized tower |
| --- | --- | --- |
| 256 | 8.8 to 9.5 ms (8.0 of kernels) | 9.0 to 9.7 ms (8.5 of kernels at tile 0) |
| 1,024 | not timed | 21.8 to 22.3 ms |
| 4,096 | 95.0 to 96.1 ms (88 of kernels) | 99.6 to 101.3 ms (tile 0: 110 to 115, 103 to 107 of kernels) |
| 16,384 | 711 to 722 ms (690 of kernels) | tile 0: 760 to 769 ms (737 of kernels) |

The quantized linears are bound by the tensor cores, not by their bytes: `prompt_bias.cu` decodes each weight
to bf16 for `m16n8k16`, about 55 TFLOPS at tile 0 against CUTLASS's 73 on the BF16 weights, so the tower is
slower, not faster, for its 302 MiB of linears against 860. Attention, the norms and the rope are the BF16 tower's.

### Against the card's vLLM

The reference is the card's container (`eugr/spark-vllm-b12x:latest`) on spark-b, as the owner chose (spec
105): `tools/reference/vllm_reference.sh`, one sequence, no MTP, the card's quantized activations kept bf16
(`VLLM_MXFP8_LM_HEAD=0`, `B12X_W4A16_A4_PREFILL=0`, `VLLM_B12X_MOE_FP4_FORCE_A16=1`). 2026-10-18, image
`eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595` (ID `a7f76ae9f18c`),
one load of 854 s with its compiles, each case and the score twice with the prefix cache reset between; held to
the zspark load above (`000c579`). Kept in `~/.cache/zspark/reference-fp4-local/` on both Sparks.

| | zspark against vLLM | vLLM against itself |
| --- | --- | --- |
| Score, 5,336 tokens: nll | 2.13927 against 2.14806 (-0.0088) | |
| A token's loss, mean \|difference\| (99th percentile) | 0.195 (1.33) | 0.219, the largest 3.83; 5,332 tokens of 5,336 differ |
| Greedy, sky 64 and primes 32 | every token the same | its two runs the same |
| Greedy, 256 tokens of prose | the same to token 100, then a tie: vLLM's first two at 100 are equal (-0.8556 both, 1056 and 28750); zspark took 28750 | its two runs the same |

The tolerance is vLLM's own: its two runs of the same prompt differ by 0.219 a token on the score text (its
kernels' order is not fixed), so zspark's 0.195 from it is inside that, and a reply that parts where vLLM's
first two choices are equal is a tie, not a difference.

`VLLM=a16` (the reference above, `score.json`) holds the head and the experts at 16 bits only: its MXFP8 linears
ran b12x's "auto", quantized in prompt chunks by its own heuristic. `score-a4.json` beside it is
the card's A4 prefill (`DOCKER_EXTRA="-e B12X_W4A16_A4_PREFILL=1"`, 2026-10-26, two runs the same bits), 0.2754
a token from it. Scored in windows of 2,048 (`compare_vllm_test`, spec 143 WS-06): the card's prompt form 0.2161
from the A16 reference and 0.2730 from the A4 one, nll 2.13911; every row bf16 0.1993 and 0.2614, nll 2.14344.
The FP8 rows move the score a little away from both, inside vLLM's own spread. The shipped form (both parts,
spec 143 WS-08, on the W4A4 pack): 0.2798 and 0.2787 from the A16 reference's two runs, 0.3086 from the A4 one,
nll 2.14697; bf16 on that pack 0.1980, 0.1960, 0.2636, nll 2.13844. vLLM's A4 lies 0.2754 and 0.2780 from those
two runs. vLLM's A16 prefill differs run to run (0.212 to 0.220 a token), its A4 not at all, so the gate (the
owner, 2026-10-26) takes a token's mean over the A16 runs `score.json` holds (four since WS-08): from it the card
is 0.2428 and the A4 0.2431, and the card's nll is within 0.02 of both references'. First tokens with both parts,
cold, the machine alone: 4k 1.866 s against bf16's 2.057 (-9.3%), 16k 7.425 against 8.171 (-9.1%), in windows of
2,048; 6.08 s in windows of 4,096 with the fused read-out (spec 148 WS-02, above).

## Tests

With `ZSPARK_FLASH_NEXT_FP4_LOCAL` set (each skips without):

- `tests/models/qwen4/layout_test.zig`: the whole checkpoint's headers against the quantization map.
- `tests/weights/csf_test.zig`: a checkpoint's streams decoded.
- `tests/models/qwen4/model_test.zig`, "a mixed-precision checkpoint's faces": the first layers' MXFP8 faces
  stacked and alone against the host's product.
- `tests/models/qwen4/forward_checkpoint_test.zig`, "a mixed-precision checkpoint's first layers": graphs,
  wide windows and a snapshot give the serial step's bits.
- the same file, "... in prompt windows of 4096 rows": loaded for `--max-num-batched-tokens 4096`, 4,200
  rows in windows of 4,096 give each row the bits of 2,048-row windows and of its serial step.
- `tests/models/qwen4/draft_test.zig`, the same layers: drafted decoding gives the serial reply.
- `tests/models/qwen4/checkpoint_prefix_test.zig`, "a checkpoint's first layers, one load": a load a cache
  storage; on it a pool stream's kept states and a session's hand-over a serial slot's, a head pass's seats
  each scored alone, a reply the serial one through the session and the pool and drafted, learnt states read back.
- `tests/models/qwen4/vision_test.zig`, "a quantized tower": read by its tensors' formats; its features near
  the BF16 tower's and TensorFold's (the bounds the measured spread and a quarter); its blocks from
  TensorFold's inputs; an image's features the same bits alone, beside another and in another order.
- `tests/models/qwen4/served_test.zig`, "an image request on FP4-Local" (the whole tier, with
  `ZSPARK_REFERENCE_VISION` for `shapes`' prompt): the host's prepared picture through `served.Eye` and a session
  is the forward's reply from the same features, drafted and serial, twice. Not run yet.

On two ranks, `tools/check.sh ranks` (this checkpoint by default, `ZSPARK_REFERENCE_FP4_LOCAL` its
reference): the pool on the first layers, every bit over NCCL against both ranks in one process (`hall`); every
layer against the vLLM reference (`checks/ranks_vllm.zig`): drafted greedy and sampled replies their serial ones,
the replies along vLLM's within its ties, the score within its own spread; and `two_ranks.sh replies`, bf16 and
FP8 caches. `tests/models/qwen4/share_layout_test.zig` holds the csf windows on the checkpoint's experts.

Without a checkpoint: `tests/weights/nvfp4_test.zig` (MXFP8 against the host, a row's bits across row counts,
alone and stacked), `tests/models/qwen4/blocks/ple_gather_test.zig` (the NVFP4 table's rows) and
`tests/models/qwen4/vision_test.zig`'s "an MXFP8 and an NVFP4 linear with bias and GELU" (the tower's kernel
against the host, a padded K and width, a row's bits across row counts and tiles).
