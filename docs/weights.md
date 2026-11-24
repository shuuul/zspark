# Weight sources and local storage

zspark loads local checkpoint directories. It does not download or update model weights at runtime. Supported checkpoints are published on Hugging Face; the [recipe index](recipes/README.md) links the pinned main-checkpoint revisions, storage formats, licenses and verification evidence. A directory under `~/models` is not by itself evidence that its files match a particular Hub revision.

## Supported checkpoints

| Role | Hugging Face repository | Revision and provenance |
| --- | --- | --- |
| Flash Next LLM | [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | `b9ce18a`; full revision in its [recipe](recipes/qwen3.8-flash-next-fp4-local.md) |
| GLM LLM | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) | `76348dd`; full revision in its [recipe](recipes/glm-5.3-flash-fp4-local.md) |
| Nemotron ASR | [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4) | `66c6adc`; full revision and required files in its [recipe](recipes/nemotron-3.5-transcribe-30b-a4b-nvfp4.md) |
| Qwen ASR, BF16 | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B), [`Qwen/Qwen3-ASR-0.6B`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `7278e1e`, `5eb1441`; [1.7B](recipes/qwen3-asr-1.7b.md), [0.6B](recipes/qwen3-asr-0.6b.md) recipes |
| Qwen ASR, NVFP4 | [`shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16`](https://huggingface.co/shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16), [`shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16`](https://huggingface.co/shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16) | `5bbf02c`, `8e868ac`; owner-built ModelQuant exports published on HF, with builder provenance in the [1.7B](recipes/qwen3-asr-1.7b-nvfp4.md) and [0.6B](recipes/qwen3-asr-0.6b-nvfp4.md) recipes |
| OCR2 | [`baidu/Unlimited-OCR-2`](https://huggingface.co/baidu/Unlimited-OCR-2) | `9cf077c`; full revision in its [recipe](recipes/unlimited-ocr-2.md); rejected experimental OCR NVFP4 packs are not supported |

The current personal LAN deployment selects Flash Next and Nemotron ASR, both across two Sparks, with DFlash2 drafting and no OCR. The root launcher accepts an explicit local configuration; deployment choices are not an implicit model download list.

## Drafters and licenses

DFlash2 weights are separate checkpoints:

- Flash Next: [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) at `a22a1744e85c17364dcc54a6a420af7135d4c7f6` (Apache-2.0), selected explicitly by `--drafter DIR`.
- Nemotron: [`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2) at `4172ee5b7bfe55f59e784922878c1bd9bdfeb6ba` (OpenMDW-1.1), required by Nemotron's default drafting mode. `--mtp` selects its bundled MTP shard instead; see the recipe for command support.
- GLM: [`incoai/GLM-5.3-Flash-DFlash2`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at `bf582e4eacc1810f76656d1811693ff6c6737d2a`, optional and licensed CC BY-NC-ND 4.0. The [GLM recipe](recipes/glm-5.3-flash-fp4-local.md#drafting) records the repository's stricter handling policy: no committed, vendored, uploaded or redistributed weights, and no derived weight copies written to disk. Users supply their own authorized local source. Do not include this drafter in a weight synchronization or redistribution bundle.

Model licenses remain separate from zspark's Apache-2.0 license. Check the pinned model card and its license before downloading or using any checkpoint. HF-hosted does not mean that a checkpoint has Apache-2.0 terms.

## Download and paths

Install the Hugging Face `hf` CLI separately. Use the full 40-character commit from the selected recipe, not `main`, to reproduce its checkpoint. For example, the Nemotron target snapshot can be downloaded into the HF cache:

```bash
hf download nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4 \
  --revision 66c6adc4020179567074d0399cda2a4eeb2bb347
```

The command prints the snapshot directory. Set the launch configuration to that path, or use `--local-dir "$HOME/models/NAME"` to materialize a directory outside the Hub cache. These are two supported operator storage choices; zspark does not maintain their equivalence. Download the entire checkpoint, including configuration, tokenizer, processor and auxiliary weight files, rather than filtering only `*.safetensors`.

For a separate drafter, record its full revision alongside the target revision in your deployment configuration. A target recipe's `revision` identifies the target checkpoint, not its independent drafter. The Flash Next and Nemotron drafter revisions above were checked against the public HF model API on 2026-11-13. Do not fetch `main` when reproducing a pinned deployment.

On 2026-11-13 a full SHA256 check established the deployed files' identity on both Sparks: live process inspection confirmed the serving paths, every file in all four checkpoints was hashed, both nodes' manifests are byte-identical, and every file the upstream snapshot has equals the pinned revision's — 33 LFS files by the repository's published LFS `oid` SHA256, 37 non-LFS files by refetching the pinned bytes. That includes the Nemotron DFlash2 `model.safetensors` at `~/models`, which matches the `4172ee5` LFS oid even though the directory is not an HF cache snapshot; its only divergences are an extra local `configuration.json` and an absent license text file. Per-file sums, deployed paths and the reproduction commands are in the [deployed weight identity manifest](weights-manifest.md).

For two Sparks, follow the [stack guide](serving.md#flash-next-ocr-and-asr-from-one-command) for which paths must exist on each node and the peer binary path. Use `tools/spark_copy.sh` for eligible weight files over the direct link, with `SPARK_VERIFY=1` when verifying copies. Preserve or recreate snapshot symlink targets; copying links alone does not copy the HF cache blobs. The GLM drafter exception above still applies.

Runtime unpacking and allowed drafter quantization produce process/device buffers, not new HF repositories or persisted target checkpoints. Reference activations under `~/.cache/zspark/reference-*` are test fixtures, not served model weights.
