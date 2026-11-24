---
checkpoint: shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16
revision: 8e868ac7d0e1fa8da3e981c56a830638c88dc465
model: Qwen3-ASR-0.6B
family: qwen3_asr
storage: the language model's linears of layers 2 to 25 NVFP4 W4A16 as stored (ModelOpt, e4m3 scale a 16, fp32 scale a tensor); the audio tower, layers 0, 1, 26, 27, norms and the tied head BF16 as stored
license: apache-2.0
disk_gb: 1.03
machines: 1
device_gib: 4.23
status: supported
drafters: none (greedy, as the bf16 base; spec 138)
variables: ZSPARK_QWEN3_ASR_0_6B_NVFP4, ZSPARK_REFERENCE_QWEN3_ASR_0_6B
---

# Qwen3-ASR-0.6B, NVFP4 W4A16 (`shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16`)

The owner's own offline NVFP4 build of [`Qwen/Qwen3-ASR-0.6B`](qwen3-asr-0.6b.md) (spec 140, the exception of
2026-10-25 to running the published checkpoint as stored), made and read as
[1.7B's build](qwen3-asr-1.7b-nvfp4.md): that page's storage, multiplication and load rules hold here; this page
holds what differs. It loads through spec 140 WS-03's path with no code of its own. **Machines: one DGX Spark.**

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16` at `8e868ac7d0e1fa8da3e981c56a830638c88dc465` (fetched 2026-10-28) |
| Base | `Qwen/Qwen3-ASR-0.6B` at `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| Built by | NVIDIA Model-Optimizer 0.47.0, `gptq` calibrated, `export_hf_checkpoint`; the [ModelQuant](https://github.com/shuuul/ModelQuant) recipe `recipes/qwen3-asr-0.6b` at `783a9a5`; `manifest.json` lists every module and file hash, `gate.json` its builder's own gate (120 clips, passed) |
| Licence | Apache-2.0 |
| On disk | 1.03 GB: one `model.safetensors`, the base's `config.json` with a `quantization_config`, `hf_quant_config.json`, the base's tokenizer and processor files |
| Variables | `ZSPARK_QWEN3_ASR_0_6B_NVFP4` (its test), `ZSPARK_REFERENCE_QWEN3_ASR_0_6B` (`docs/environment.md`) |

168 linears (q, k, v, o, gate, up, down of the language model's layers 2 to 25; 377 M parameters) are NVFP4 and
run as stored through `nvfp4.Fp4Linear`; the audio tower (width 896), layers 0, 1, 26, 27, the norms and the head,
stored once as `embed_tokens`, are BF16.

## Measured

All on spark-a at a60b3cee, a fast build, the machine held alone (`machine.lock` exclusive), one `serve` load each
against the bf16 base on the 64-clip set (`tools/reference/qwen3_asr/serve_wer.py`, 0.6B's own references), then
`zspark generate` on `en_140s.free` three times each (spec 140 WS-09; raw lines `~/.cache/zspark/scratch-140/ws09/`).

| | bf16 base | this build |
| --- | --- | --- |
| Load (`serve`) | 4.74 GiB in 0.7 s | 4.23 GiB in 0.7 s |
| Decode, en_140s, 524 / 525 tokens | 139.2 to 140.0 tok/s | 205.3 to 206.4 tok/s (1.47x) |
| Prompt, 1,869 rows (en_140s) | 151 to 154 ms | 84 to 85 ms |
| `generate`'s device memory | 1,853 MB | 1,335 MB |
| Real-time factor over HTTP, 4,202 s of audio | 0.0224 (94.2 s) | 0.0159 (66.8 s) |
| RTF, clips to 30 s / longer | 0.0212 / 0.0225 | 0.0140 / 0.0160 |
| A 600 s clip (en_u00, zh_u00) | 18.5 s, 14.3 s | 14.3 s, 11.1 s |
| English WER / Chinese CER (ground truth) | 2.118% / 10.404% | 2.253% / 10.450% (+0.135 / +0.046 points) |
| Ended at EOS (`finish=stop`) | 64 of 64 | 64 of 64 |
| Texts equal to the bf16 base's | | 31 of 64 |
| Texts equal to `qwen-asr`'s (HF bf16) | 51 of 64 | 31 of 64 |

The criterion (WER/CER within 0.5 points of bf16) holds. Where the texts leave the base's they differ in
punctuation, numerals as digits or words (一九九零年/1990年, 30%/百分之三十), a spelling (Burkeson/Burgeson,
比博/比伯) or a near-homophone; the first divergence falls at characters 0 to 1,470. Not measured here: vLLM or
SGLang on this build, and `--max-audio-clip-s 30`.

## Tests

`ZSPARK_QWEN3_ASR_0_6B=... ZSPARK_REFERENCE_QWEN3_ASR_0_6B=... ZSPARK_QWEN3_ASR_0_6B_NVFP4=DIR tools/check.sh qwen3_asr NVFP4`
(`tests/models/qwen3_asr/forward_test.zig`): as 1.7B's build, against itself dequantized to bf16; each layer alone
at cosine 0.99992 or more, the carried logits 0.9978 (en_30s), 0.9954 (zh_30s), 0.9941 (en_5s), floor 0.99 (1.7B's
0.995), no argmax missed; both row-rule tests pass on it.
