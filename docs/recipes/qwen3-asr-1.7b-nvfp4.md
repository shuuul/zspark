---
checkpoint: shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16
revision: 5bbf02c1e8324975b39e637f7ce3357d40895047
model: Qwen3-ASR-1.7B
family: qwen3_asr
storage: the language model's linears of layers 2 to 25 NVFP4 W4A16 as stored (ModelOpt, e4m3 scale a 16, fp32 scale a tensor); the audio tower, layers 0, 1, 26, 27, norms and the tied head BF16 as stored
license: apache-2.0
disk_gb: 2.34
machines: 1
device_gib: 5.57
status: supported
drafters: none (greedy, as the bf16 base; spec 138)
variables: ZSPARK_QWEN3_ASR_NVFP4, ZSPARK_REFERENCE_QWEN3_ASR
---

# Qwen3-ASR-1.7B, NVFP4 W4A16 (`shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16`)

The owner's own offline NVFP4 build of [`Qwen/Qwen3-ASR-1.7B`](qwen3-asr-1.7b.md) (spec 140, the exception of
2026-10-25 to running the published checkpoint as stored): what it serves, its route, its defaults and its tower
are the bf16 base's page; this page holds what the build changes. **Machines: one DGX Spark.** It remains an
optional stack ASR (spec 140 WS-05: the gate passed, 1.8x the base's decode, 1.6 GiB less), selected with
`--asr DIR --asr-name Qwen3-ASR-1.7B` so clients keep the base's id. The current LAN stack uses Nemotron
([serving.md](../serving.md#different-models-behind-one-address)).

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16` at `5bbf02c1e8324975b39e637f7ce3357d40895047` (fetched 2026-10-28) |
| Base | `Qwen/Qwen3-ASR-1.7B` at `7278e1e70fe206f11671096ffdd38061171dd6e5` |
| Built by | NVIDIA Model-Optimizer 0.47.0, `gptq` calibrated, `export_hf_checkpoint`; the [ModelQuant](https://github.com/shuuul/ModelQuant) recipe `recipes/qwen3-asr-1.7b` at `783a9a5` (variant `default`); `manifest.json` lists every module and file hash, `gate.json` its builder's own gate |
| Licence | Apache-2.0 |
| On disk | 2.34 GB: one `model.safetensors`, the base's `config.json` with a `quantization_config`, `hf_quant_config.json`, the base's tokenizer and processor files |
| Variables | `ZSPARK_QWEN3_ASR_NVFP4` (its test), `ZSPARK_REFERENCE_QWEN3_ASR` (`docs/environment.md`) |

## What is stored, and how it is multiplied

168 linears (q, k, v, o, gate, up, down of the language model's layers 2 to 25; 1,208 M parameters) are NVFP4:
`weight` U8 `[n, k/2]` (two e2m1 codes a byte, the low nibble first), `weight_scale` F8_E4M3 `[n, k/16]`,
`weight_scale_2` F32 `[]`; `input_scale` is not stored (W4A16: activations stay bf16). They run as stored through
`nvfp4.Fp4Linear` (`qmmf`'s lane matmul): every e2m1 weight is exact in bf16, the block scales are applied in
fp32 and the tensor scale on the sum. A fused stack (q, k, v; gate, up) shares one `weight_scale_2` in this
export; a stack whose parts differ is refused, where vLLM runs the largest and warns.

Everything else is BF16 and runs as on the base: the audio tower, layers 0, 1, 26 and 27, the norms, and the head,
which this export stores once, as `thinker.model.embed_tokens.weight` (no `thinker.lm_head.weight`; the loader
takes the embedding as the head, the tie the base declares).

## What changes while it loads

- `config.json`'s `quantization_config` is read the way vLLM's ModelOpt loader reads it
  (`zig/src/weights/modelopt.zig`, `uniform`): `quant_method` `modelopt`, one `quant_algo` (`W4A16_NVFP4`, group
  16 from `config_groups`), and the modules it leaves out from `ignore` (or `exclude_modules`), matched as vLLM's
  `is_layer_excluded` does, by substring or wildcard. Any other algorithm, a mixed-precision map, or activations
  in other than 16 bits is refused in words.
- Each linear is read by `modelopt.read` against what the map says it is, so a tensor missing or of another dtype
  stops the load. Nothing is requantized or dequantized.

## Measured

All on spark-b at 190cdab4, a fast build, the machine held alone (`machine.lock` exclusive), one `serve` load
each against the bf16 base on the base page's 64-clip set (`tools/reference/qwen3_asr/serve_wer.py`), then
`zspark generate` on `en_140s.free` three times each (spec 140 WS-03; raw lines
`~/.cache/zspark/scratch-140/ws03/`). The bf16 pass repeats the base page's figures.

| | bf16 base | this build |
| --- | --- | --- |
| Load (`serve`) | 7.19 GiB in 1.0 s | 5.57 GiB in 1.1 s |
| Decode, en_140s, 525 tokens | 58.8 to 59.1 tok/s | 101.7 to 102.1 tok/s (1.73x) |
| Prompt, 1,869 rows (en_140s) | 190 to 194 ms | 156 to 157 ms |
| `generate`'s device memory | 4,015 MB | 2,359 MB |
| Real-time factor over HTTP, 4,202 s of audio | 0.0487 (204.6 s) | 0.0297 (124.7 s) |
| RTF, clips to 30 s / longer | 0.0531 / 0.0484 | 0.0306 / 0.0296 |
| A 600 s clip (en_u00, zh_u00) | 34.7 s, 27.5 s | 22.9 s, 18.2 s |
| English WER / Chinese CER (ground truth) | 1.618% / 9.728% | 1.656% / 9.668% (+0.038 / -0.060 points) |
| Ended at EOS (`finish=stop`) | 64 of 64 | 64 of 64 |
| Texts equal to the bf16 base's | | 33 of 64 |
| Texts equal to `qwen-asr`'s (HF bf16) | 56 of 64 | 34 of 64 |

Where the texts leave the base's they mostly differ in punctuation, a spelling (Neoplatonists, Mainhall, saviour,
Saint-Aignan), numerals as digits or words, GHz against 赫兹, or a near-homophone (萨赫勒/撒哈勒, 乌波卢/乌波鲁).
The first divergence falls at characters 8 to 1,158 (335 of 7,370 on en_u00). One short clip,
zh_s07, repeats its last two characters ("相似。相似。") before it stops, in vLLM too (below), so it is the build's.
Not measured here: SGLang on this build, the `keep-o-down` variant, and `--max-audio-clip-s 30`.

### Speed against vLLM

Spec 140 WS-04 (spark-b alone at 5bd4e712, the base page's harness: loads zspark, vLLM, zspark, vLLM, two passes of
the 64 clips each, the second kept; raw lines `~/.cache/zspark/scratch-140/ws04/`), the bf16 columns the base
page's. vLLM is the base page's image (the fork `local-inference-lab/vllm` @8244d1c44, `v0.29.1rc0-603`; no 0.27
image exists here), the card's `vllm serve`. It loads this build as stored with no flag: `modelopt_fp4`,
`W4A16_NVFP4`, "Using MarlinNvFp4LinearKernel" (weight-only FP4, bf16 activations).

| | zspark bf16 | zspark NVFP4 | vLLM bf16 | vLLM NVFP4 |
| --- | --- | --- | --- | --- |
| RTF, all 64 / 30 s or less (28) / longer (36) | 0.0486 / 0.0519 / 0.0484 | 0.0296 / 0.0306 / 0.0296 | 0.0198 / 0.0692 / 0.0167 | 0.0108-0.0116 / 0.0358-0.0366 / 0.0092-0.0100 |
| First token, median / 30 s or less / the 600 s clip | 70 / 33 / 1,117 ms | 62-63 / 32-34 / 1,085-1,099 ms | 43 / 36 / 540 ms | 31-34 / 23-24 / 590-894 ms |
| Tokens a second, one stream (30 s or less) | 62.2 | 111.9 | 46.1 | 90.6-92.2 |
| English WER / Chinese CER | 1.618% / 9.728% | 1.656% / 9.668% | 1.887% / 9.4% | 1.849% / 9.3% |
| Texts equal to the engine's bf16 | | 33 of 64 | its two loads: 56 of 64 | 37 to 39 of 64 |
| Ended at EOS | 64 of 64 | 64 of 64 | not recorded | 64 of 64 (no clip above 1.5x its bf16 tokens) |
| Load to ready; device memory | 2 s; 7.19 GiB | 2 s; 5.57 GiB | 180 s; weights 4.48 GiB | 168 to 269 s; weights 2.29 GiB |

On one stream zspark decodes the build 22% faster than vLLM and its RTF on clips to 30 s is 15% lower; vLLM's first
token on short clips is about 10 ms shorter, and it is 2.6x faster on the whole set because it cuts a clip over 30 s
into 30 s pieces and runs them at once (the base page's "Long clips in pieces"). Each engine's build stays within
0.04 WER and 0.2 CER points of its own bf16. vLLM reserves its KV pool beside the weights (105 GiB at its default
memory share); zspark's 5.57 GiB is all it holds. `shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16` loads and transcribes in the
same vLLM (weights 1.04 GiB; six clips checked, untimed).

## Tests

`ZSPARK_QWEN3_ASR_NVFP4=DIR ZSPARK_REFERENCE_QWEN3_ASR=~/.cache/zspark/reference-qwen3-asr tools/check.sh qwen3_asr NVFP4`
(`tests/models/qwen3_asr/forward_test.zig`): the build is loaded as stored and, beside it, with its NVFP4 linears
replaced by their bf16 values (code times block scale times tensor scale, built in the test), which is the
weights it stores; on the 30 s English and Chinese and the 5 s fixtures each layer alone agrees at cosine 0.99994
or more, the logits of the prompt's last row and 32 greedy steps at 0.9977 or more (floor 0.995), with no argmax
missed; both row-rule tests pass on it. `tests/weights/modelopt_test.zig` holds the map's reading (host tier).
