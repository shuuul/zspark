# `zig/src/models/glm5/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## The map

- `config.zig`: `config.json` into sizes, the layers' kinds (`linear_attention` is KDA, `deepseek_sparse_attention`
  DSA; `dense` or `sparse` MLP) and what is refused in the model's words. It reads `quantization_config`
  itself (`ModelOpt`): a `MIXED_PRECISION` map (FP4-Local, spec 107), each entry a `weights/modelopt.zig`
  scheme, NVFP4 only on the routed experts.
- `layout.zig`: `kinds`, one row a tensor kind with its dtype, the shape the config makes and its `Split`;
  `rule` gives a name's split; `Layout.init` checks every tensor the config asks for and refuses one it has no
  place for. A `linear` kind is read by `modelopt.read` with its parts (`linear_parts`, `input_scale` dropped), every language entry of the map met.
  `kindOf` is a name's kind by its name; `kindIn` by the config too, where one leaf is two kinds (KDA's and MLA's
  `o_proj`, the model's F32 `post_attention_layernorm` and the MTP layer's BF16).
- `commands.zig`, `commands/info.zig`: `zspark info DIR [all]`, the config, the layer map, what a rank holds.
- `pack.zig`: a rank's share of a checkpoint onto the device a part at a time (`Model.loadEach`'s hook): a layer's
  tensors held by direct reads, packed, uploaded and given back with the host copies its share made, so the
  host never holds more than a layer.

- `glue_ref.zig`: one row of every step in `kernels/glm5/`, on the host and in the kernels' order (`fold`, `@mulAdd`
  where a kernel has `fmaf`): what the device's bits are tested against.
- `hc.zig`: the hyper-connections. `mix` leaves a row's 24 dots and its squares in 16 parts (`partShape`),
  `pre` makes the gates, the Sinkhorn's mix and the collapsed, normalized row; `post` adds the ranks' fp32
  partials rank 0 first, rounds the branch to bf16 and writes the four streams; `residualAdd`, `streamMean`,
  `rmsnorm`, `embed`.
- `moe.zig`: `select` (the top experts by sigmoid plus bias, the lower id among equals, from a BF16 router's K
  slices or from logits), `swiglu`, `combine`. The router is a BF16 linear, not `glue.py`'s `_router_part`.
- `share.zig`: a rank's share of a tensor by its split (`take` calls `weights/share.zig`: half of the rows is a
  slice of the checkpoint's bytes, half of the columns a copy). `modeloptLinear` is a ModelOpt
  linear's share (`modelopt.cut`): scales by whole blocks, csf_32x2_v1 streams whole with the rank's window.
  `partOf` reads a linear as a `face.Part` of its stored format (BF16, MXFP8, FP8_PB_WO), `faceOf` makes the
  `face.Face` a layer runs, `bf16Weight` widens an MXFP8 weight to bf16 where it is exact (the MTP layer's
  `kv_b`, which `dsa.zig` reads as bf16) and refuses it where not.
- `mlp.zig`: `Dense` (a gated MLP, each linear a `face.Face` as stored: `[gate | up]` stacked, SwiGLU, down into
  fp32) and `Moe` (the router whole; the routed experts an NVFP4
  `experts.Experts` from csf_32x2_v1 or plain scales, of the rank's half width with GLM's bf16 activation and
  clip; the shared expert a `Dense`), each ending in the rank's fp32 partial. `load` reads a layer from
  anything with `get(name)`: a checkpoint, or a test's map of tensors.
- `kda.zig`: KDA over a rank's heads. `Weights` (`[q | k | v | f_a | g_a | b]` stacked, padded to 8 outputs
  so the `f_a` and `g_a` columns feed a matmul where they lie); `State`, what every KDA layer keeps of a
  sequence: the committed states and the last window's side by side (`cur`), the saved rows a replay reads,
  the conv's three rows. `forward` chains a window from the committed state, row by row up to
  `recurrent_rows` (48), chunked above (`kda_chunk.cu`, spec 116 WS-06; `chunked_windows` the switch);
  `commit(rows, keep)` takes the state beside it or replays `keep` rows (a recurrent window's), then moves the
  conv's window. A mark inside a chunked window stands at a chunk's start (`session/marks.zig` `snapped`) and
  is asked before the window runs (`State.ask`, `marks.zig` `askAhead`, `Op.keep_ahead`): the chain writes the
  state there into the mark's buffers as it passes, and `Mark.convWithin` completes it.
- `dsa.zig`: DSA over a rank's heads. `Weights` (`[q_a | kv_a]` stacked, the two norms on its columns,
  `kv_b` as stored: a head's key rows, then its value rows); `Cache`, a latent a position and layer with the
  position on the device; `Store`, one rank's latents, keys and gates for a pool's extents, a stream's cache
  a view of it. Dense write and attend read through the extent table (a null table is the identity). A row
  past the dense limit then attends its selected tokens. `Buffers`. `Cache.commit(keep)` moves the position.
- `model.zig`: a rank's share on the device (`Model`: the embedding, each layer's two `Site`s, its attention
  and MLP as unions, `slot` its place among its kind's layers, the norm (bf16 or fp32 as stored, `uploadNorm`),
  the rank's rows of the head, a `linear.Linear`: BF16, or MXFP8 as FP4-Local stores it), what a
  sequence keeps (`State`) and a forward's `Buffers`.
- `forward.zig`: the phases between which the ranks exchange partials (`embed`, `attention`, `mlp`, `finish`,
  `head`). A phase takes the rank's `partial` and the `gathered` view.
- `ranks_forward.zig`: the ranks a process holds, `Held(2)` or `Held(1)` with a `ranks.Gather`
  (`zig/src/ranks/`: a gathered buffer is tight, a rank's rows at `rank * rows * dims`; NCCL gathers in place).
  `Model`, `State`, `Buffers`, `Graphs` are `draft.Decoder`'s contract; `logitsRow` joins the vocabulary's
  halves. `loopback.zig` is `Held(2)`, the tests' reference. From `split_rows` rows (`splits`) a window's glue
  is split by rows (`glue`, `glueEnd`, `forward.readOutRows`, `finishRows`): a rank's own rows of the other's
  partial (`Gather.scatter`), the read-outs then shared; the same bits (spec 107 WS-03). The out-projections and
  the routed combine before a scatter make the others' rows first, send them on `Beside`'s stream and make the
  rank's own meanwhile (`scattered`, `forward.emitRows`/`fillRows`; spec 116 WS-03).
- `checks/ranks_check.zig`: `zspark ranks-check random`, a rank a process over NCCL
  held to the loopback's bits (`ranks_check_drafter.zig`: DFlash2's). `ranks_whole.zig`: `ranks-check DIR
  reference=REF`'s words, then `ranks_vllm.zig` on vLLM's files (FP4-Local), the first layers' digest a
  loopback's and a pair's alike.
- `session.zig`: the ranks loaded for replies in turn; rank 0 leads, rank 1 follows (`mirror.zig`: an
  operation rank 0 gains is told there), drafted by MTP or `dflash2/rig.zig`. `snapshot.zig`: a position's
  KDA state and conv window, the cache cut there (its tail pool unfinished), and the drafter's context at it
  when one is loaded (DFlash2's brought to the mark first, `marks.zig`'s `flashTo`); `serve`, `chat` and `generate` hold eight, a reply two more while it runs, before the
  prompt's last window, at the prompt's end and each 256 tokens of the reply. The same prompt again resumes
  from the mark before that window. `learnt.zig`: `serve --spill-gib`, a mark that leaves to disk and back,
  each rank its half (rank 1 told by `Op.spill_*`, the ranks agreeing on a read over NCCL), on
  `decode/learned_parts.zig` as Flash Next's; DFlash2's context never. `served.zig`: TensorFold's engine. `generate`, `chat`, `serve` stand on it.
- `draft.zig`: the loopback as `draft.Decoder`'s namespace, `Decoder` and `Chooser` instantiated.
- `mtp.zig`: the MTP layer (`layers.N`): `x = eh [norm_e(embed(next)) | norm_h(final row)]`, the first
  embedding of a sequence zero, then a DSA over its own latents and a MoE with plain residuals, its norm and
  a 4-bit head. Three phases as the forward's; `PairOf(held)` runs in the main model's scratch, an entry
  a position absorbed and then one a chained draft (`trim` drops those); `Head` is the `draft.Drafter`
  (TensorFold's rule: the first draft kept, a stop before the product falls below `confidence`), its `cap`
  cutting the room the chain is begun with (`--cap N`).
- Under `zig/tests/models/glm5/`: `testing.zig` (random rows, a host slice onto the device and back, the random
  model, `Fixture`) and `NAME_fixture.zig`, what several test files share. `checks/fixtures.zig`: the host's.

## The two ranks

The model does not fit one Spark (FP4-Local: 164.87 GiB of language model, a rank holds 88.11), so every tensor has a split,
TensorFold's `split.py:rule` as a table (`layout.kinds`), and `world` is 2.

- `row`, `col`: half of a matrix's outputs or inputs; a routed expert's `gate_proj` and `up_proj` halve their
  outputs, its `down_proj` its inputs.
- `whole`: replicated. `vocab`: `lm_head`, which the rule replicates and the loader halves by vocabulary rows;
  `drop`: the vision tower (`model.visual.`), counted and not read.

## Sources

TensorFold's CUDA GLM is 0.6.6's (`cb2ebf0`); the tree beside this one is 1.0.2 (Zig and Metal).
Read it with `git -C ../TensorFold show cb2ebf0:src/tensorfold/families/glm5_next/FILE`.

## Fixtures

- `fixtures.zig`'s `config` is a four-layer config (three KDA with dense MLPs, one DSA with a MoE, one MTP
  layer); `layout_test.zig`'s `synthetic` makes its checkpoint of headers alone.

## The MTP layer's tensors

- The layer after the last (`layers.N`) is the MTP's: DSA and MoE, no `hc_*` tensors, `eh_proj`, `enorm`,
  `hnorm` and `shared_head.norm` of its own. `Layout.holds` says which layer holds which kind.
