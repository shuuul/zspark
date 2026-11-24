---
checkpoint: nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4
revision: 66c6adc4020179567074d0399cda2a4eeb2bb347
model: NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
family: nemotron
storage: NVFP4 group-16 attention and experts; per-tensor FP8 Mamba, perception and MTP; BF16 embeddings, norms, gates, convolutions and head
license: OpenMDW-1.1
disk_gb: 21.57
machines: 2
device_gib: 14.76
status: supported
drafters: DFlash2 default ([`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2) at `4172ee5b7bfe55f59e784922878c1bd9bdfeb6ba`, OpenMDW-1.1); MTP selectable
variables: ZSPARK_NEMOTRON_NVFP4, ZSPARK_REFERENCE_NEMOTRON, ZSPARK_NEMOTRON_HF
---

# Nemotron-3.5-Transcribe-30B-A4B, official NVFP4

**Machines: two DGX Sparks** for the intended stack placement. Spec 151's accuracy, two-rank and serving
acceptance gates pass (below); the current LAN stack serves Nemotron ASR alongside Flash Next.

## Checkpoint and storage

The [official checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4)
revision above was downloaded by the owner. Its five safetensors were copied from spark-a to spark-b over
ConnectX on 2026-11-01 with `tools/spark_copy.sh` and `SPARK_VERIFY=1`; all five SHA256 sums matched.
The [DFlash2 checkpoint](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2)
was copied separately and its safetensors SHA256 matched too. No weights are committed.

| File | Bytes | Role |
| --- | ---: | --- |
| `model-00001-of-00002.safetensors` | 9,988,250,728 | Backbone, first shard |
| `model-00002-of-00002.safetensors` | 9,192,549,192 | Backbone, second shard |
| `model-mtp-fp8.safetensors` | 1,335,778,712 | MTP drafter |
| `perception.safetensors` | 838,256,800 | ASR encoder, projection and Sortformer speaker infusion |
| `model-ctc-bf16.safetensors` | 201,897,998 | Timestamp heads, outside transcription scope |

`hf_quant_config.json` declares ModelOpt `MIXED_PRECISION`. Attention projections and routed/shared experts
are NVFP4 with group size 16. Mamba projections, the perception encoder's linears and the MTP block are
per-tensor FP8. Embeddings, norms, routers, convolutions and `lm_head` remain BF16. The target weights are
never requantized. MTP-only FP8 experts currently follow the drafter exception and requantize to NVFP4 at
load; draft acceptance and speed are measured below.

The stored backbone configuration derives its layer count from `layers_block_type`; this revision omits
`num_hidden_layers`. An explicit count must agree with the layer list.

The MTP attention export retains distinct q/k/v `weight_scale` values. Native loading keeps their codes
and each tensor's scalar through the shared block-scaled FP8 linear (`linear.Linear.fp8_block`); equal
scales keep the per-tensor fast path. A part's scale spans its full input and output dimensions. The
block-scaled kernel accumulates scaled 64-input groups, so it may round differently from applying a
single tensor scale after the complete sum; each row still has the same bits at every window width.

## Reference launch and precision

The card's reference is vLLM 0.32.0 plus NeMo Speech's `nemo_speechlm` plugin. Its launch selects
`--dtype bfloat16 --kv-cache-dtype fp8`,
`VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel`,
`--use-replayssm --mamba-backend flashinfer --mamba-cache-mode none`, chunked prefill, no prefix caching,
65,536 context tokens and at most one 57,600,000-sample audio recording per request (3,600 seconds at 16 kHz).
`kv_cache_quant_algo` is null in the quant map, so the FP8 KV choice must be explicit.

The card selects MTP with `--speculative-config '{"method":"mtp","num_speculative_tokens":8}'`.
Its alternative DFlash2 launch selects
`--speculative-config '{"method":"dflash","num_speculative_tokens":15,"model":"nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2"}'`
and disables CTC timestamps with `--hf-overrides '{"ctc_timestamps":{"enabled":false}}'`.
The two drafters are alternatives. The transcription prompt uses `nemotron3p5`, thinking off by default.

Native `serve` and `generate` select DFlash2 unless `--mtp` / `mtp` or `--serial` / `serial` is given.
The default directory is `~/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`; `--drafter DIR` /
`drafter=DIR` overrides it. Missing drafter weights are an error. Native serving uses the card's
65,536-token total context, with an unspecified transcription reply limit taking the room after the
prompt. The generation config supplies EOS 11 and padding 0; it supplies no 16,384-token output cap.

The native attention implementation holds BF16 KV rows where the card's reference runs FP8 caches, and
the native perception FP8 linears take BF16 activations where the reference encoder applies its static
FP8 activation scales. These differences are accepted, not erased: spec 151's accuracy gate measured
them and passed — encoder cosine against the reference and per-clip WER/CER against its vLLM launch
(below), with transcripts equal across the two ranks. Native replies are not claimed bit-identical to
the FP8 reference; the gate's recorded figures are the bound.

The native backbone consumes the checkpoint's activation scales in every window: static FP8 for
Mamba's projections, NVFP4 for routed/shared relu2 experts and attention's projections. Fused q/k/v
uses the largest input scale. Attention's normalized FP4 values and e4m3 block scales widen exactly
to BF16 for the existing tensor-core GEMM; its global input factor is applied in the GEMM alpha.
independence; the gate above was passed on the measured accuracy evidence, not on claimed activation
precision parity with the reference.

## Local paths and verification

`ZSPARK_NEMOTRON_NVFP4` points to the complete Hugging Face snapshot, including the five weight files.
The old `~/models/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` directory on spark-a contains only metadata;
use the snapshot until that directory is completed. `ZSPARK_NEMOTRON_HF` names the earlier public metadata
mirror for weightless tests. `ZSPARK_REFERENCE_NEMOTRON` names the clip set and exported fixtures.

Reference setup, fixture capture and provenance are documented in
[the exporter instructions](../../tools/reference/nemotron_transcribe/README.md).
`tools/check.sh nemotron` exercises synthetic row-rule, Mamba rollback, perception and speaker-path checks.
The first real one-rank backbone load and eight serial steps passed on 2026-11-01 (debug build based on
`9178cacc` with spec 151's config/FP8 fixes): 19,782 MiB allocated with `rows=64`, an 11-token context
and the MTP weights loaded. This excludes the perception module and is not the serving memory budget.
The smoke held a shared machine lock; its printed timing fields are not benchmark figures.
A one-rank DFlash2 HTTP smoke (`94ee70d3`, dirty integration tree) used 21.36 GiB with context 4,096
and `--rows 128`, including perception and the drafter. English short audio answered through multipart
transcription, `audio_url` and `input_audio`; repeated requests, a first-token stop and recovery after
disconnect were exercised. This is not the default-context stack memory account or a speed benchmark.

Full fixed-set native decoding, one-rank service checks, two-rank transcript equality and the stack's
memory account pass below. The initialization-range dt clamp must not be used for inference.

On 2026-11-02, the corrected speaker path, checkpoint activation scales and Mamba inference
`time_step_limit=[0,infinity)` gave these short-clip results in the dirty `spec-151-resume` tree:

| Clip | Encoder cosine against vLLM | Free reply error against reference text |
| --- | ---: | --- |
| English 5 s | 0.999699 | WER 0 |
| Chinese 5 s | 0.997447 | CER 0 |

`zspark compare CHECKPOINT fixtures/model/{en_5s,zh_5s}.safetensors` produced these figures;
`compare-inference-dt-limit-native-short-20261102.log` is under `scratch-151/`. Every comparable
reference top-1 pick agrees (23 English tokens, 21 Chinese). Mamba's initialization bounds
`time_step_min/max` do not constrain inference; ModelOpt encodes an infinite limit as
`{"__float__":"Infinity"}`. The parser accepts that encoding and checks ordered nonnegative bounds.

With the same decoder, default context 65,536 and rows 2,048, a real DFlash2 service holds 26.76 GiB.
Seeded serial/drafted repeats and Chinese serial/drafted replies have equal token hashes; both audio
chat formats, multipart transcription, a one-token bound, cancellation after a streamed token and
the next request's recovery pass. These shared-lock runs establish correctness, not benchmark speed.

Uncapped native decoding covers all nine clips, every available reference prefill chunk and the first
32 diagnostic decode steps. Scores against ground truth use the fixture scorer: English is lowercase,
punctuation-free WER; Chinese is CER with whitespace removed and punctuation retained. These differ
from the short clips' zero distance against reference transcripts above.

| Clip | Metric | Native | vLLM reference |
| --- | --- | ---: | ---: |
| English 5 s | WER | 0 | 0 |
| English 39 s | WER | 0 | 0 |
| English 143 s | WER | 0.006944 | 0.004630 |
| English 604 s | WER | 0.025143 | 0.023429 |
| Chinese 5 s | CER | 0.100000 | 0.100000 |
| Chinese 32 s | CER | 0.075758 | 0.075758 |
| Chinese 146 s | CER | 0.122677 | 0.126394 |
| Chinese 603 s | CER | 0.165217 | 0.164760 |
| English 3,494 s | WER | 1.215067 | 1.301339 |

All clips through 604 s finish at EOS. On the joined-chapter 3,494 s clip, both implementations reach
the 65,536-token context limit after 21,840 reply tokens (`finish_reason=length`, no EOS). Hour-context
support is not an accuracy guarantee: this clip's WER exceeds 1. Native transcripts, the score inputs
and results are retained under `scratch-151/` as `NAME.native.txt`, `native-accuracy-20261102.json` and
`compare-inference-dt-limit-native-{short,long}-20261102.log`.

One-rank timings on spark-b, 2026-11-02, use the fast NCCL integration build based on `b233ac46`
(dirty), context 65,536 and rows 2,048. The fixed 39.04 s English FLAC produces the same 138-token
greedy reply in all modes. Each mode has one warmup and four measured requests; serial/DFlash2
alternate off/on/on/off twice in one load, then MTP uses a separate load. The machine lock is exclusive
with the GPU lock held; uptime load for the corrected MTP run is 1.29/1.33/1.30. Draft depth is 15.

| Mode | Median decode tokens/s | Median first token, s | Median request wall, s | Wall/audio real-time factor | Drafts kept |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serial | 74.7286 | 0.304620 | 2.146704 | 0.054987 | — |
| DFlash2 (default) | 241.0061 | 0.311334 | 0.889138 | 0.022775 | 117/121 |
| MTP | 60.0644 | 0.301340 | 2.591088 | 0.066370 | 109/420 |

All response token hashes agree. `bench-summary-20261102.json`,
`bench-serial-dflash2-39s-20261102.json` and `bench-mtp-39s-20261102.json` retain the responses;
`bench-{dflash2,mtp}-service-20261102.log` retain the service logs. Timings describe an isolated
single Spark, not simultaneous stack or two-rank performance.


Two-rank parity and the stack memory account were measured the same evening on both Sparks under the
runner's machine and GPU locks. A TP2 pair (`zspark serve` with `rank=0/1`, the barrier fix `2309076b`)
holds 14.76/14.75 GiB a rank and answers the same cases: the 39 s FLAC's serial and DFlash2 replies are
the same 138 tokens as the one-rank replies (sha256 equal, `dcenmicu`); the English 5 s pair decodes
23 tokens (16/18 drafts kept), the Chinese 5 s 21 tokens (11/15 kept); seeded sampled repeats are equal;
multipart, a one-token bound and cancel-recovery pass. Replies are retained in
`tp2-{39s,short,sampled,routes}-view-fixed-20261102.json`; the rank logs
`~/.cache/zspark/logs/20261102-210058-serve.rank{0,1}.log` show 221.7 tok/s on the 39 s clip.

`zspark stack FLASH_DIR --asr $ZSPARK_NEMOTRON_NVFP4 --peer ADDR` loads the ASR pair first
(14.76/14.75 GiB a rank) then Flash Next's pair (65.84 GiB a rank); the 2026-11-02 run left
spark-a/spark-b with 24.8/25.0 GiB free. Routed `/v1/audio/transcriptions`, `/v1/chat/completions`
and `/v1/models` answer 200 (`stack-peer-accept-20261102.json`; rank shares in
`~/.cache/zspark/logs/20261102-210058-serve.rank{0,1}.log`). The planner's `--parallel 1` warning on
Nemotron is expected: the decoder serves one reply at a time anyway.

## Decode-step profile

A `ZSPARK_NEMOTRON_PROFILE` harness (`tests/models/nemotron/profile_test.zig`, the
`docs/environment.md` row's `nsys`/`ncu` command lines) launches each of Nemotron's kernels
alone at a rank's shapes. The 2026-11-03 budget (`~/.cache/zspark/scratch-153/budget-dspark{1,2}.{md,csv}`)
— per-launch medians × launches-per-step on random weights at the checkpoint's shapes, not a
traced decode — puts the 1-row serial step's kernel sum near the measured 13.4 ms: **MoE W4A4
~40%** (`moeup`/`moedown`), mamba-2 ~26% (the FP8 `in_proj`, not the scan), lm_head ~19%,
attention ~13% (overread at a fixed 32k position; real audio decode is smaller). At a 16-row
DFlash2 window MoE is ~80% and the draft block 0.12 ms — drafting does not bound, so no
depth sweep was run. Both Sparks agree the MoE bound.

Two kernels were tried bit-exact. The spec-152 two-warps-a-unit form was ported to the W4A4
expert kernel (`experts_half_a4.cu`); it is bit-exact (`weights experts` 10/10, `nemotron`
80/80) but gains ~1% on `moeup:1`, inside the loads' spread, and only the 1-row case fits the
resident bound — the branch was dropped, the kernel already being memory/compute bound. The
FP8 `in_proj`/`out_proj` static-scale path now quantizes x once per call instead of per
weight-tile block (`qmm_a8.cu`, bit-exact): `out_proj` −5.1 µs warm-L2, `in_proj` unchanged
(already ~273 GB/s streaming). The decode step is near its hardware bound on GB10; the 241
tok/s DFlash2 rate above is the serving figure.