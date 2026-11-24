---
checkpoint: Qwen/Qwen3-ASR-0.6B
revision: 5eb144179a02acc5e5ba31e748d22b0cf3e303b0
model: Qwen3-ASR-0.6B
family: qwen3_asr
storage: BF16 as stored (the audio tower, the Qwen3 0.6B language model, one copy of the tied head and embedding)
license: apache-2.0
disk_gb: 1.88
machines: 1
device_gib: 4.74
status: supported
drafters: none (greedy, as 1.7B; spec 138)
variables: ZSPARK_QWEN3_ASR_0_6B, ZSPARK_REFERENCE_QWEN3_ASR_0_6B
---

# Qwen3-ASR-0.6B (`Qwen/Qwen3-ASR-0.6B`)

The smaller sibling of [`Qwen/Qwen3-ASR-1.7B`](qwen3-asr-1.7b.md), run by the same code: its
route, defaults, formats, long-clip pieces and departures from the package are 1.7B's page; this page holds what
the checkpoint changes. **Machines: one DGX Spark**, 4.74 GiB of device memory after a `serve` load at the defaults.

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `Qwen/Qwen3-ASR-0.6B` at `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` (fetched 2026-10-28) |
| Licence | Apache-2.0 |
| On disk | 1.88 GB: one `model.safetensors`, `config.json`, `preprocessor_config.json`, `chat_template.json`, and `vocab.json`, `merges.txt` and `tokenizer_config.json` (no `tokenizer.json`) |
| Variables | `ZSPARK_QWEN3_ASR_0_6B`, `ZSPARK_REFERENCE_QWEN3_ASR_0_6B` (`docs/environment.md`) |
| Behaviour reference | the `qwen-asr` 0.0.6 package's transformers backend, as 1.7B's |

The model, from `config.json` (612 tensors, about 0.94 B parameters):

- The audio tower (186.4 M): 1.7B's mel, chunks and convolutions, then 18 pre-LN layers of width 896 (14 heads of
  64, GELU FFN 3,584) in 104-token blocks, `ln_post` and the projector to 1,024.
- The language model: Qwen3-0.6B's shape, 28 layers, hidden 1,024, 16 query and 8 kv heads of 128, FFN 3,072, q/k
  RMSNorm, theta 1e6.

## What is stored, and how it is multiplied

Everything is BF16 and runs as stored, by 1.7B's kernels: the tower takes its shapes from `audio_config`
(`tower.Shape`), and its attention kernel is instantiated for width 896 beside 1,024 (`tower_attention.cu`). The
language model is config only. `embed_tokens` and `lm_head` are both stored and bitwise identical (read
2026-10-28), so `lm_head` is the head and the embedding table, as on 1.7B. The owner's NVFP4 build is
[its own page](qwen3-asr-0.6b-nvfp4.md).

## What changes while it loads

As 1.7B's, with this tower's shapes: q, k and v fused into one `[2688, 896]` linear; the 301 tower tensors are
checked by name, dtype and shape; `serve` refuses a tower whose `output_dim` is not the language model's
`hidden_size`.

## Defaults that follow this checkpoint

1.7B's, but `--name`, which is the directory's name (`Qwen3-ASR-0.6B`).

## Measured

All on spark-a at a60b3cee, a fast build, the machine held alone (`machine.lock` exclusive), one `serve` load on
1.7B's 64-clip set against 0.6B's own reference transcripts (`serve_wer.py`), then `zspark generate` on
`en_140s.free` three times (spec 140 WS-09; raw lines `~/.cache/zspark/scratch-140/ws09/`).

| Figure | Value |
| --- | --- |
| Load (`serve`) | 4.74 GiB in 0.7 s |
| Decode, en_140s, 524 tokens | 139.2 to 140.0 tok/s (1.7B 58.8 to 59.1) |
| Prompt, 1,869 rows (en_140s) | 151 to 154 ms |
| `generate`'s device memory | 1,853 MB |
| Real-time factor over HTTP, 4,202 s of audio | 0.0224 (94.2 s); to 30 s 0.0212, longer 0.0225 |
| A 600 s clip (en_u00, zh_u00) | 18.5 s, 14.3 s |
| English WER / Chinese CER (ground truth) | 2.118% / 10.404% (`qwen-asr` 0.0.6, HF bf16: 2.176% / 10.495%) |
| Ended at EOS (`finish=stop`) | 64 of 64 |
| Texts equal to `qwen-asr`'s | 51 of 64 |

## Tests

`ZSPARK_QWEN3_ASR_0_6B=DIR ZSPARK_REFERENCE_QWEN3_ASR_0_6B=~/.cache/zspark/reference-qwen3-asr-0.6b tools/check.sh qwen3_asr`
runs 1.7B's tests on this checkpoint (each test on every checkpoint set): the tower against the package's
per-stage exports at cosine 0.998 (0.999 on 1.7B: one zh_30s row turns unstable in bf16 after the last layer; the
package's own bf16 against its fp32 is 0.99912 at `ln_post`, that row 0.685, zspark's 0.99884), the language
model's layers, logits and greedy tokens, both row-rule tests, and the pieces. The references come from
`tools/reference/qwen3_asr/export.py` with `ZSPARK_QWEN3_ASR_MODEL` set to this checkpoint.
