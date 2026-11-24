# `zig/src/models/` instructions

This file extends `zig/src/AGENTS.md` and the root `AGENTS.md`. A directory a model, under TensorFold's family name
(spec 040): `qwen4/` is Qwen3.8 Flash Next, `glm5/` GLM-5.3-Flash (spec 044, building). Each has an
`AGENTS.md` of its own for the model's structure, sources and conventions. `qwen3_asr/` is Qwen3-ASR-1.7B and 0.6B, speech to
text (spec 138): TensorFold has no family for it, so the directory and `zs_qwen3_asr_*` take the checkpoint's
`model_type` (`docs/naming.md`); it serves `/v1/audio/transcriptions` and chat with audio parts (spec 150).
`unlimited_ocr/` is Unlimited-OCR-2, page image to tagged text, named the same way (`zs_unlimited_ocr_*`, its `model_type` `unlimited-ocr`): DeepEncoder's SAM and CLIP
tower and a DeepSeek-V2 decoder with DASH attention; it serves `/v1/chat/completions` with up to 128 pages and
has its own `AGENTS.md`. `nemotron/` is Nemotron-3.5-Transcribe-30B-A4B on its
official NVFP4 checkpoint (spec 151, building): a NeMo perception module before a NemotronH backbone of Mamba-2, MoE and
attention layers; it has its own `AGENTS.md`.

## Names

- Its C symbols and import aliases carry its directory's name (`zs_qwen4_*`, `qwen4_config`). What a user sees
  kept its name: the CLI, `ZSPARK_FLASH_NEXT*`, the served model's name, `tools/reference/flash_next_export.py`.

## What a model gives the shared code

A second model copies neither of these from the first; it declares its own and the shared file does the rest.
`docs/architecture.md` has the layers, the list for a new model and who owns what is passed.

Each contract in full (its commands, its drafted decoding, its serving) and what still waits for an interface:
`REFERENCE.md`; a contract's meaning is the doc comment on its generic.

DFlash2 is shared (`decode/dflash2/`, spec 136 WS-03): a model's `dflash2/pass.zig` declares the `Target` its
generics take (`decode/dflash2/pass.zig`'s header), `drafter.zig` the target's taps and heads, `rig.zig` what its
checkpoint decides. Callers import the shared files for what names no model. Rank 1's drafter operations are `decode/dflash2/follow.zig`'s; the
rows of logits over two ranks, their settling and the forward's graphs are `decode/settle.zig` and `decode/graphs.zig` (WS-05).

## When editing here

- Do: add to the contract in `draft.Decoder`'s doc comment whatever a model must newly declare, to `owed`
  beside it (`decode/contract.zig` holds a model to it when it compiles), and to the model in
  `zig/tests/decode/draft_test.zig` with it. `served.Served` has its `owed` the same way.
- Do: keep a model's refusals in its own words, and the dispatch's in terms of what was found and what is known.
- Don't: make a file of `zig/src/` generic over a model before a second model uses it; an interface drawn before
  its second user is drawn twice (spec 055's Decisions).
- Don't: import one model's directory from another's (`tools/check_imports.zig` refuses it, and says which
  of a model's directories may import which).
- Do: declare a kernel entry a second file launches once — in `kernels/zs/abi.h` and `cuda.kernels` — and call
  it `cuda.kernels.zs_*`; a file-local `extern fn` is for an entry that file alone wraps
  (`kernels/AGENTS.md`, "The C ABI").
