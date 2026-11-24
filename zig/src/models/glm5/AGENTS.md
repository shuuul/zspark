# `zig/src/models/glm5/` instructions

Extends `../AGENTS.md`, `zig/src/AGENTS.md` and the root `AGENTS.md`. GLM-5.3-Flash (`model_type`
`glm5_next`) on NVIDIA's FP4-Local (spec 107), on two ranks as stored. Each linear a `face.Part` of
its format (`share.partOf`), the routed experts NVFP4 (`mlp.Moe.routed`), the head
MXFP8, F32 norms fp32; no weight is requantized. Its forms are the card's vLLM's by a window's rows (spec 115,
`forward.bf16_rows`, 48): a step, a verify window and a prompt window of up to 48 rows run bf16 rows (routed
experts A16 on the W4A4 layers, MXFP8 linears on bf16 rows); a longer prompt window quantizes them (experts A4 as
the card's `prefill_a4`, MXFP8 rows to FP8, spec 114 WS-04). Block-FP8 linears quantize every row in every
phase, as the card. `ranks_forward.compute` sets the form a window, `pool/slots.zig` a round's parts (`decode/forms.zig`'s `formFrom`, Flash Next's too:
bf16 parts first, the hall orders them), the scored rows of a round and the last row of a window head on bf16
rows; the MTP layer and DFlash2 keep the quantized forms (they only draft). A prompt cut so that its chunks lie
on both sides of 48 is not bit for bit the same prompt cut otherwise.

## Purpose

Each file's `//!` header says what it does; its tests are `zig/tests/models/glm5/DIR/NAME_test.zig`, registered
in `zig/glm5_tests.zig`, and in `zig/host_tests.zig` too when it reaches no `cuda.zig`.
A directory imports those before it alone (but `mtp.zig`, whose head is a `decode/dflash2/weights.zig` linear):
`layers/` (the files below to `forward.zig`, `ranks_forward.zig`, `loopback.zig`), `mtp/`, `dflash2/`, `pool/`
(`slots`, `hall`, `pool`), `session/` (`pack`, `mirror`, `learnt`, `session`, `marks`, `snapshot`, `served`), `commands/`, `checks/`
(`random*`, `fixtures`, `ranks_*`, `buffer_trace`); `commands.zig`, `draft.zig` above.

The file map, what each split kind halves, TensorFold's GLM tree and how to read it, the fixtures' config and the
MTP layer's tensors: `REFERENCE.md`, beside this file.

- `index.zig`: the indexer's host model (no device). The cache's bytes a token, by item; a key's LayerNorm,
  a pool of four under the gates' softmax, the rotation and FP8 coding of the pooled keys and queries and their
  score as the card's vLLM (spec 116 WS-05; `scoreBf16` is TensorFold's, for its fixtures), and the selection
  (the kept pools, ties to the lower, the unfinished tail). `zspark info DIR context=N` prints the bytes. The launches are `index_launch.zig`
  (`zig/kernels/glm5/index.cu`): keys and gates through the extent table, the pooled keys contiguous. A launch's
  grid is what the window can touch (its rows, the pools it completes, the context's pools), never a buffer's
  size: a block that starts and returns is paid for (spec 093).
- `slots.zig`: sequences sharing one window; its `Graphs` runs a round through a graph keyed on the round's
  shape (the parts, each slot's side of the KDA states and its context bucket), captured at its first sight.
  `hall.zig`: replies decoded at once (`serve --parallel N`), a slot its sequence and drafter seat, counted
  before made (`slotBytes`); its rounds run through the graphs where the session set them (`seat`: the
  loopback's serve, not `--eager`, not two ranks — over NCCL a capture must be told to the rank that follows,
  as the serial round's is). `pool.zig` puts it under `server/scheduler.zig`. A reply among others is the
  reply alone: tokens, rounds, drafts kept and its rows' bits. `mtp_seats.zig`, `decode/dflash2/seats.zig`: a round's
  drafts, all streams in one step.

## The two ranks

- A name the rule has no place for is refused (`UnknownTensor`), as TensorFold raises; so is a tensor under a
  layer whose kind does not hold it.
- A new tensor kind is one row of `kinds` and one case in `layout_test.zig`'s list against `split.py`'s.

## Tests

- A block over a rank's share is held three ways (`mlp_test.zig`): against whole matrices on the host in fp64;
  both ranks added against the whole block on the device (2e-6: the cut changes fp32's order alone); a row
  alone, bit for bit. `fixtures.zig`'s
  `putLinear` and `Source` make a checkpoint of random tensors under real names.
- A kernel's test holds it against `glue_ref.zig` bit for bit, and a row alone against the row among others. Where
  `exp` stands between host and device (a gate, the Sinkhorn) the bound is a rounding, and a case with no dots
  (gates of a half exactly) holds the rounding after it. `hc_test.zig`'s `tellingRow` builds a row whose norm
  says how its squares were summed; a new sum over shared memory gets one.

- A block with state (`kda_test.zig`, `dsa_test.zig`) is held by `play`: one input through plans of
  `(rows, keep)` windows, each against serial steps bit for bit (outputs, states, latents), with a chunk's
  boundary among them; KDA's chunked windows (more than 48 rows) against the recurrent chain within a bound,
  and bit for bit against the same rows cut at chunks' starts. That holds no wiring: an fp64 model of the whole layer does
  (`hostLayer`, `hostRow`), and the ranks' states and attention are the whole layer's heads to the bit.
- The whole forward (`forward_test.zig`) runs a random GLM of five layers (`testing.zig`'s `Fixture`, `random.zig`'s `fx`: hidden 512, both
  attentions, both MLPs, NVFP4 experts as FP4-Local stores them, F32 norms) in loopback: windows of 1 to 80 rows, windows cut short and graphs
  against serial steps, and a first window phase by phase, where each site is `glue_ref.zig`'s from the layer's
  own tensors and each branch its layer's weights loaded alone. A new phase or layer kind joins that test.
  A test holding a window of more than 48 rows to its serial steps takes one form (`testing.oneForm`).
  `draft_test.zig` runs the shared decoder on it with an oracle's drafts, `mtp_test.zig` the MTP layer the
  same way, then as the drafter.

- `dflash2/` is GLM's DFlash2 drafter (spec 067, 122): a checkpoint of its own (`model_type` `qwen3`, told by
  `architectures`), `ZSPARK_GLM_DRAFTER` its directory. The drafter is `decode/dflash2/` and
  `zig/kernels/common/dflash2*.cu` (spec 136 WS-03); this directory keeps what is GLM's: `pass.zig` its `Target`
  (narrow updates, GLM's `Op`, the session's heads), `drafter.zig` the target (GLM's ranks' taps and heads),
  `rig.zig` the checkpoint (its exclusive window, `cli.Opened`, the heads). Its weights are CC BY-NC-ND 4.0: no test, fixture or tool writes them or anything made of them
  outside `~/.cache/zspark/`. `checks/random_drafter.zig` makes a random drafter beside `fx`.
- The drafter's taps (`model.Buffers.setTaps`, `forward.tap`) are the streams' mean after its layers, side by
  side; set before a graph is captured, and the forward's bits do not change with them.
- `ZSPARK_GLM_FP4_LOCAL` names FP4-Local: its headers, a few linears' halves and layer 3's csf planes on the
  host; `pack_test.zig`, `session_test.zig`, `learnt_test.zig` and `hall_pack_test.zig` its first four layers
  and MTP layer in loopback, unset they skip. No TensorFold reads it: the whole
  model is held to the card's vLLM at TP 2 (`ZSPARK_REFERENCE_GLM_FP4_LOCAL`, `checks/ranks_vllm.zig`,
  `check.sh ranks glm-fp4-local`). `tests/models/glm5/layers/modelopt_fixture.zig` is the four-layer config under a ModelOpt map and its
  synthetic checkpoint.
- TensorFold weighs a draft over 56 candidates; zspark over the row, from its start picked on device (`fromPicks`, spec 068).
