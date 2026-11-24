# Names

How a file, a directory and a word are named in `zig/`. Read it before a file is added or renamed; the nearest
`AGENTS.md` says what each file holds. Sources read for it (2026-10-10, spec 069): TensorFold `1b31cc74`
(`zig/src`), vLLM `3313825`, SGLang `1291f31`.

## The three rules

1. **TensorFold's name for TensorFold's thing.** Where a file does what a file of TensorFold's Zig tree does,
   it takes that file's name (`slots.zig`, `state.zig`, `forward.zig`, `mirror.zig`, `mtp.zig`, `snapshot.zig`,
   `config.zig`, `weights.zig`, `marks.zig`, `pack.zig`, `fixtures.zig`). vLLM and SGLang are read where
   TensorFold has nothing of the kind. A model's directory and symbols keep TensorFold's family name
   (`qwen4`, `glm5`). Where TensorFold has no family (its ASR and OCR models, specs 137 and 138), the directory
   and C prefix take the checkpoint's `model_type`: `qwen3_asr`, as `dispatch.modelType` reads it. Unlimited-OCR is the second case: `unlimited_ocr`
   (`models/unlimited_ocr/`, `kernels/unlimited_ocr/`, `zs_unlimited_ocr_*`), from `model_type` `unlimited-ocr`.
   Spec 137's Decisions (2026-10-25) record it as a departure from this rule's first sentence, since there is no
   TensorFold name to take.
2. **A word means one thing in the whole tree.** The table below is the vocabulary; a new file does not use one
   of its words for something else, and does not name the same thing by a new word.
3. **The same role has the same name in both models.** A directory or file that plays one part in `qwen4/` and
   in `glm5/` is called the same in both, so that a reader of one finds his way in the other.

A name says what the file holds, not when or why it was made: no spec number, no `new`, `old`, `v2`, `util`,
`misc`, `helpers`, `common` outside `commands/common.zig`.

## Vocabulary

| Word | Means here | TensorFold | vLLM, SGLang |
| --- | --- | --- | --- |
| `session` | one reply at a time over one sequence's state | `session.zig` (qwen27) | (none) |
| `pool` | several replies at once, above the session | `core/lanes/`, `*_lanes.zig` | the scheduler's running batch |
| `hall` | the pool's engine: who holds which slot, and its rounds | `core/lanes/engine.zig` | `Scheduler` |
| `slot`, `slots` | one sequence's place and state in the pool | `glm/slots.zig`, `Stream` | `Request`; `Req` |
| `round` | one forward for every live slot | `round.zig`, `tree_round.zig` | a step; `ScheduleBatch` |
| `window` | the rows one sequence runs in one round | `core/lanes/windows.zig` | (none) |
| `seat` | a drafter's per-slot state (`seats.zig`) | (none) | (none) |
| `draft`, `drafter` | speculative tokens, and what proposes them | `draft_ops`, `*_draft.zig` | spec decode; draft |
| `served` | a model behind `engine_api.Engine` | `native/*_host.zig` | `AsyncLLM`; `TokenizerManager` |
| `request` | what a reply is asked to be, and how it ends | `core/engine_api.zig` | `EngineCoreRequest`; `io_struct.py` |
| `scheduler` | the queue and the decoding thread over a pool | `core/lane_host.zig` | `sched/scheduler.py`; `scheduler.py` |
| `drive` | the disk under a checkpoint's mapping (`weights/drive.zig`: kept awake while rows are read) | (none) | (none) |
| `ranks` | the two halves of a model over NCCL | `ep.zig`, `tp.zig` | (none) |
| `mirror` | rank 0's commands replayed on rank 1 | `glm/mirror.zig` | (none) |
| `host` | the CPU side: no device, no kernels linked | (its `host` is the server's engine) | (none) |
| `_ref` | a kernel's arithmetic on the host, to the bit | `glue_ref.zig` | (none) |
| `fixtures` | small made-up configs and tensors for tests | `fixtures.zig`, `*_fixtures.zig` | `fixtures/`, `Dummy*`, `Mock*` |
| `random` | a small model with random weights | `toy.zig` | `DummyModelLoader` |
| `fake` | a stand-in for a part that is not the one tested | `fake.zig` | `Mock*`, `Fake*` |
| `router` | one address over whole copies of `serve`, a conversation kept on its copy (`zig/src/router/`) | (none) | SGLang's `sgl-router` |
| `copy` | one whole `zspark serve` behind the router | (none) | a worker; vLLM's "replica" |
| `checks` | what a command runs to hold a model against itself | `checks.zig` | (none) |
| `image` | a picture's bytes to RGB rows, before a vision processor (`zig/src/image/`) | `core/picture/` | `MediaConnector`, `ImageMediaIO`; `load_image` |
| `audio` | an audio file's bytes to 16 kHz samples and log-mel features, before a speech model (`zig/src/audio/`) | (none) | `AudioMediaIO`, `load_audio`; `WhisperFeatureExtractor` |

Two words differ from TensorFold on purpose. `host` there is the engine a server stands on; here the root
`AGENTS.md` has always meant "no GPU" by it (the host tier, `host_tests.zig`), so the engine side is `served`
and a file is never called `*_host.zig`. `backend` there is what the lane core asks a device to compute; zspark
has no such layer, and the word is left unused so that it cannot be mistaken for it.

## Files

- `snake_case.zig`, a noun for what it holds. A file that is one type is that type's name in lower case
  (`session.zig` holds `Session`); the import alias of a file whose name a local would shadow ends in `_file`
  (`session_file`, `request_file`).
- A part split from a file keeps the file's name as its prefix and says what it took: `hall_engine.zig`,
  `hall_open.zig`, `rounds_graphs.zig`, `session_options.zig`. A split is of whole declarations.
- Host arithmetic that a kernel is held against: `NAME_ref.zig` beside the kernels' wrapper (`glue_ref.zig`,
  `kernels_ref.zig`). Shared drafting two models launch lives under `decode/dflash2/`, not under `models/`.
- A forward: `forward.zig` is one process's phases in order; what runs it over the ranks is `ranks_forward.zig`,
  and both ranks in one process is `loopback.zig`.
- Tests: `NAME_test.zig` under `zig/tests/` at the source's path; several for one source are named for what
  they hold (`mtp_pick_test.zig`). What two test files share: `NAME_fixture.zig`, or the area's `testing.zig`.
  Only what a command of the program reads stays in `zig/src/` (a model's `checks/`).
- Guidance: `AGENTS.md` binds every task under its directory; `REFERENCE.md` beside it is what a task reads
  only when it touches what a section names (a file map, shapes, sources, figures).
- A kernel source is named for its step (`attention.cu`, `kda.cu`), and a variant by what differs after it:
  `_rows`, `_table`, `_slots`, `_wide`, `_once`.

## A model's directories

| Directory | Holds | `qwen4/` | `glm5/` |
| --- | --- | --- | --- |
| top | what `decode/draft.zig` and `dispatch.zig` ask | `config`, `layout`, `model`, `forward`, `draft`, `mtp`, `served` | `draft`, `commands` |
| `blocks/`, `layers/` | a layer's parts, each over its kernels | `blocks/` | `layers/` (with config, layout, model, forward) |
| `mtp/`, `dflash2/` | a drafter of its own | (in `mtp.zig`) | both |
| `session/` | one reply at a time, and what it keeps between replies | yes | yes (with `served.zig`) |
| `pool/` | several replies at once | yes | yes |
| `commands/` | one file a command, `common.zig` what they share | yes | yes |
| `checks/` | commands' self-checks and the random model | (none) | yes |

Known and left: `qwen4/` keeps its forward at the top and `glm5/` under `layers/`; `served.zig` is at the top
in `qwen4/` (it stands over the session and the pool) and under `session/` in `glm5/`. Moving either is a
rename with no reader's gain yet.

## Choosing a name

1. Does TensorFold's `zig/src` have a file for this? Take its name.
2. Does the other model have this role? Take its name there.
3. Is there a word in the vocabulary? Use it, with a suffix from "Files".
4. Otherwise a noun for what the file holds, and a row in the vocabulary if the word is new.

A rename is one commit of `git mv` and the references, checked by `tools/check.sh compile`; the paths in
`AGENTS.md`, `README.md` and comments move in the same commit. Specs are records and keep the old names.
