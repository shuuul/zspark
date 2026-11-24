# States, snapshots and marks on the GB10

What a held state costs on this chip (bytes, allocations, writes), where zspark takes marks and what each
choice bought. Part of the `cuda-gb10` skill. A state is a sequence's recurrent layers (DeltaNet, KDA) and
conv windows at a position; a mark is a state kept at a position so a later prompt resumes there instead of
running cold; a snapshot is the copy a verify window or a pair takes to roll back. The cache's extents are
attention's and are not states.

## What a state weighs

| Model | A state | Source |
| --- | --- | --- |
| Flash Next NVFP4 | 73 device buffers, 110 MiB fixed plus 30.8 kB a token in bf16; 4,809 tokens 251 MiB, written in 0.19 to 0.20 s, read back warm in 31 to 37 ms | 064, 089 |
| Flash Next, two ranks | a share's state 57 MiB (reckoned); 22 kept states 1,391 MiB a rank | 094 |
| GLM FP4-Local, a mark a rank | 138.4 MiB (`kda.Kept`, both sides of `rec`) to 70.4 MiB (`kda.Mark`, the live `cur` side alone, as Flash Next keeps): 8 + 2 marks on two ranks 2.70 to 1.37 GiB, bits equal | 112 WS-05 |
| GLM, DFlash2's part | about 20 MiB a mark a rank (the drafter's context); 21 MiB of a snapshot | 114, 067 |
| GLM, KDA kept states | 0.135 GiB a slot (reckoned) | 082 |

## Allocation is the cost, not the copy

- `cudaMalloc` on the reply path costs milliseconds here, and more with the device full: Flash Next's state
  was allocated twice a 512-token reply, 6 to 7 ms idle each; now made behind the prompt's last window
  (`Marks.launched`): no allocation in a reply, 220 MiB held (089). New states 6 ms each, freeing 4k to 16k
  tokens 13 to 28 ms: buffers are handed to the next state and freed after the reply (066).
- GLM at 4,096-row windows: each snapshot's `cudaMalloc` of 142 MB took 50 to 100 ms with 109 GB of the device
  held, against 5 ms at 2,048 rows' 104 GB: short prompts slower (483 tokens 0.86 to 1.04 s) by no kernel. A
  spare shelf of kept buffers fixed it (107 WS-03); one generic shelf in `decode/` serves both models (112
  WS-02).
- Writing over a kept state's buffers: 1.45 to 1.96 ms, against 6.1 to 9.0 into new buffers (036).

## Where marks are taken (spec 112 WS-03)

| | zspark Flash Next | zspark GLM (before WS-06) | TensorFold 1.1.1 | vLLM (card image) | SGLang 0.5.22 |
| --- | --- | --- | --- | --- | --- |
| Marked | message starts, part-1, end-2, reasoning end, reply strides | the last window's start, the prompt's end, reply strides, the reply's end | one `Store`: the history's end, the latest user message, shared cuts; GLM's floored to chunk starts | `request_boundaries`: the instruction prefix, the prefill tail, the prompt's and reply's ends | the prefill's last state on the grid, the branching point |
| Stride | 512 in a prompt, 256 in a reply | 256 in a reply | none (gap 256) | none | 256 in decoding |
| Kept | 8 + 2, 4 conversations parked | 8 + 2, one conversation | by bytes | 4 a request | the pool |

- One policy, the owner, 2026-10-20: Flash Next's `Planner` moved to `decode/` with options a model, GLM on
  it. A mark inside a GLM window is replayed to its row, never made by cutting the window: a cut of 48 rows
  or fewer would switch the rows to bf16 ([`precision-forms.md`](precision-forms.md)) and change replies.
- What it bought (112 WS-06): an edit of an earlier message's first token 0.40-0.66 s to 0.12-0.24 s; the
  client set's first tokens 0.953 and 0.974x; cold short prompts 10 to 15% faster. The warm request 0.45
  against 0.36-0.38 s: the first marks allocate. 44 of 71 replies kept their text; the rest change bits by
  design.

## KDA marks under chunking (GLM, spec 116 WS-06)

A mark inside a chunked window ([`chunking.md`](chunking.md)) first ran the window's chunks again to its row,
state only (`Mark.chainWithin`). Measured (a rank's 32 heads, one layer, nine passes):

| Rows run again | Median | Best |
| --- | --- | --- |
| 64 | 95.6 us | 95.1 us |
| 512 | 454.9 us | 451.8 us |
| 2,048 | 1,808.8 us | 1,796.6 us |
| 4,096 | 3,514.9 us | 3,465.1 us |

- About 0.86 us a row a layer, 29 us a row over 34 layers: a mark 3,600 rows into a window some 105 ms a
  rank, always before the first token. A 200-message agent history (6,208 tokens) ran 10,496 rows again,
  about 0.30 s.
- Ask the window instead: `state_kernel` writes its register state at the top of the asked chunk (an `Asked`,
  `__grid_constant__`, at most 64), the state a window cut there starts from, so the same bits. A 64 KB store
  a head, mark and layer; 1.107 GB less a rank at 4,096 rows (the per-layer decay gone; by the sizes, not on
  a load). On two ranks rank 0 tells `Op.keep_ahead` before the forward.
- Marks snap to chunk starts: a resume runs up to 63 rows again in the same chunk; a mark in a cold prompt's
  first chunk (GLM's header, a short system prompt) is dropped, so a conversation sharing only that starts
  cold. Holding the 63 rows' saves instead would cost up to 88 MB more a mark beside its 71 (reckoned): not
  taken.

## Spill, park, pool

- `--spill-gib` (074): a 16,400-token state resumes in 0.23 s instead of 19.7. The host copy is 0.012 to 0.03
  s for 590 MiB; the file write 0.28 s on its own thread (on the decoding thread it would stall every stream).
  Files by 256-token block 907 against 1,008 MiB; a 4 GiB free-disk floor (101 WS-03).
- Parked sequences: 4 in a session make turns 122 to 165 ms instead of 1.0 to 1.2 s (064); lowering one 16.9
  to 0.7 ms (101 WS-04). Parking one's own next turn does not work: the template rewrites the prompt's last
  token.
- The pool store (054): 6,816 MiB, at most 64 states, a stride state every 256 rows; carrying the session
  into the pool cut the live stream's wait 9.16 to 1.59 s, of which the extents' allocation is 0.6 to 1.1 s.
  The pool is never left: leaving would cost the next arrival 1.7 to 2.0 s.
- Two ranks keep their sequence: a 16k prompt sent again 10.5 to 0.128 s (094).
- A harness prefix past 2,048 tokens was once left unprotected; fixed, 1.46 to 0.10-0.12 s (108 WS-01).

## Replay and rollback

- Replay left to the next chain: 0.58 to 0.05 replays a chain, +1.7% prose, +1.0% code on the pack (089
  WS-03).
- Staged replay and conditional state stores: a behind row 1.9 to 0.85 us, NVFP4 prose +0.8%, pack +1.5% (097
  WS-01).
- A coded cache row never spans positions (FP8 scale a position and head, an NVFP4 block inside one head's
  head_dim), so a rollback needs no cache row (094 WS-05).
- GLM KDA cannot step back: before marks, an edited prompt reran whole, 1.13 s against TensorFold's 0.22
  (068); marks answer it.
- A GLM mark after the first token was refused: inside the spread (113 WS-09).

## Sources

Specs 036, 054, 064, 066, 067, 068, 074, 082, 089, 094, 097, 101, 107, 108, 112, 113, 114, 116;
`docs/serving.md` ("A conversation's states").
