# Recipes

One page a checkpoint zspark supports. A page holds everything the checkpoint decides: where it is published and
at which revision, what each tensor is stored as and which kernel multiplies it, what is changed while it
loads, the defaults that follow its storage, what it holds and how fast it decodes, its TensorFold reference
and its tests. What does not depend on the checkpoint (the server, drafting's rules, the caches of a
conversation) is `README.md`'s.

| Checkpoint | Revision | Storage | Machines | Page | State |
| --- | --- | --- | --- | --- | --- |
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `b9ce18a` | NVFP4 experts (csf scales), MXFP8 side layers, NVFP4 n-gram table | 1 | [qwen3.8-flash-next-fp4-local.md](qwen3.8-flash-next-fp4-local.md) | every tier; two ranks by `check.sh ranks`; timed against the card's vLLM on one Spark and two |
| `nvidia/GLM-5.3-Flash-FP4-Local` | `76348dd` | NVFP4 experts (csf_32x2_v1 scales), MXFP8 and FP8_PB_WO side layers | **2, no fewer** | [glm-5.3-flash-fp4-local.md](glm-5.3-flash-fp4-local.md) | the tiers hold its first four layers; the whole model on two ranks against the card's vLLM by `check.sh ranks glm-fp4-local`; `serve` timed in spec 107 WS-03 |
| `Qwen/Qwen3-ASR-1.7B` | `7278e1e` | BF16 as stored (speech to text) | 1 | [qwen3-asr-1.7b.md](qwen3-asr-1.7b.md) | its area and `audio` pass; transcripts against the `qwen-asr` package on 64 clips; timed alone |
| `shuuul/Qwen3-ASR-1.7B-NVFP4-W4A16` | `5bbf02c` | the language model's layers 2 to 25 NVFP4 W4A16 as stored (ModelOpt), the rest BF16 | 1 | [qwen3-asr-1.7b-nvfp4.md](qwen3-asr-1.7b-nvfp4.md) | its NVFP4 test against its dequantized weights; the 64-clip set against the bf16 base, timed alone |
| `Qwen/Qwen3-ASR-0.6B` | `5eb1441` | BF16 as stored (speech to text) | 1 | [qwen3-asr-0.6b.md](qwen3-asr-0.6b.md) | 1.7B's area tests on it against its own `qwen-asr` references; transcripts on the 64 clips; timed alone |
| `shuuul/Qwen3-ASR-0.6B-NVFP4-W4A16` | `8e868ac` | the language model's layers 2 to 25 NVFP4 W4A16 as stored (ModelOpt), the rest BF16 | 1 | [qwen3-asr-0.6b-nvfp4.md](qwen3-asr-0.6b-nvfp4.md) | its NVFP4 test against its dequantized weights; the 64-clip set against the bf16 base, timed alone |
| `baidu/Unlimited-OCR-2` | `9cf077c` | BF16 as stored (pages to tagged text, DASH attention) | 1 | [unlimited-ocr-2.md](unlimited-ocr-2.md) | supported; DASH and host tests pass; page outputs checked against HF and SGLang (spec 147, closed) |
| `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4` | `66c6adc` | NVFP4 attention/MoE, FP8 Mamba/perception/MTP, BF16 remaining tensors | 2 | [nemotron-3.5-transcribe-30b-a4b-nvfp4.md](nemotron-3.5-transcribe-30b-a4b-nvfp4.md) | supported; accuracy, two-rank and serving gates passed; current LAN stack ASR |

## A page's head

Every page opens with the same frontmatter, one `key: value` a line, so that `head -16 docs/recipes/*.md`
answers what a checkpoint needs:

| Key | Value |
| --- | --- |
| `checkpoint` | the repository, as published |
| `revision` | the full commit of the repository the page describes |
| `model` | the model's name |
| `family` | zspark's directory under `zig/src/models/` (TensorFold's family name) |
| `storage` | the formats its weights are stored in, the largest first |
| `license` | the weights' licence |
| `disk_gb` | the directory's size, GB |
| `machines` | DGX Sparks it runs on: `1`, or `2` where the weights do not fit one and it runs on two ranks and no fewer |
| `device_gib` | device memory after a load, GiB, a rank where `machines` is 2, at the default context or the one the page's first line names; `unmeasured` until a load |
| `status` | `supported` (every tier passes on it), `unverified` (not loaded yet) or `retired` (no longer supported) |
| `drafters` | what drafts for it |
| `variables` | its `ZSPARK_*` variables (`docs/environment.md`), or `none` |

`zig build check-specs` holds every page to this head (`tools/check_specs.zig`): the twelve keys in this order,
a 40-digit revision, a family that is a directory, variables that have their rows, and a link from the table
above.

Under the title, the first line of the page says the same in words: **Machines: one DGX Spark**, or **two DGX
Sparks, no fewer**, with the memory it rests on.

## Measuring a checkpoint

Record the checkpoint revision, server configuration, workload, number of loads, and source for each comparison.

## The rules of a page

- **A checkpoint is run as it is stored.** zspark quantizes nothing of its own; where TensorFold requantizes a
  tensor that only drafts, the page says so and names the switch that runs it as stored.
- **A revision is a checkpoint.** Two revisions of one repository that differ in a tensor are two pages, two
  variables in `docs/environment.md` and two references.
- **A number has its source**: the checkpoint's own files (`zspark info DIR`, its `config.json` and
  quantization config), or a measurement with its date, machine and commit. A number that was not measured on
  this checkpoint is not on its page.
- A checkpoint that is added, or a change that moves one of a page's facts (a default, a kernel, a requantized
  tensor), changes the page in the same commit.
