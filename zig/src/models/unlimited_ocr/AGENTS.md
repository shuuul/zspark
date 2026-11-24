# `src/models/unlimited_ocr/` instructions

Extends `src/AGENTS.md` and the root `AGENTS.md`. Unlimited-OCR-2 (`model_type` `unlimited-ocr`): DeepSeek OCR's DeepEncoder (`vision*.zig`,
WS-03) before a DeepSeek-V2-style language model whose every attention is DASH. No TensorFold source: the
checkpoint's own HF code (`modeling_deepseekv2.py`, `modeling_unlimitedocr.py`, Apache-2.0) is the behaviour
reference, vLLM's precision the ceiling. Kernels: `kernels/unlimited_ocr/`, C symbols `zs_unlimited_ocr_*`; the
routed experts `kernels/bf16/experts.cu` (`bf16.Experts`). Tests: `zig/tests/models/unlimited_ocr/`, the area
`tools/check.sh unlimited_ocr`.

`config.zig` requires a top-level `sparse_attention_config`
(`Config.dash` is always set); it also refuses what else the object may say (another stage 0 or router, a 1d
rope, a layer without DASH, no `sparse_rope_tok_theta`).

## The language model

- 12 layers of hidden 1,280; MHA of 10 heads of 128 (kv heads = heads), RMSNorm eps 1e-6. Layer 0 a dense
  SwiGLU MLP (6,848); layers 1 to 11 MoE: 64 routed experts of 896, top 6 by the gate's fp32 softmax over its
  bf16 logits, unnormalized, scale 1; two shared experts fused as one MLP of 1,792. Vocab 129,280, `lm_head`
  untied. `config.zig` refuses what else a DeepSeek-V2 config may say (MLA, grouped top-k, normalized weights,
  sigmoid scoring).
- Everything bf16 as stored; nothing quantized. Roundings follow HF under bf16 autocast (`forward.zig`'s
  header): DASH keeps fp32 scores and weights, rounded to bf16 only as P.V's operand. Tests hold a layer from
  the reference's own input (`forward_dash_test.zig`).
- RoPE is DASH's page2d (`config.Page2d`: 63 token pairs at 10,000 "resample", one page pair at 256, the page
  mod 2); `rope_theta` (500,000) and the yarn entries are dead and are not read. q and k turn at a row's
  coordinates (`zs_attn_prep_page2d`, `Model.page2d`, which the router shares).
- Image rows: `State.image` lays the tower's features over the image rows (`Image.runs`, a run a page);
  `forwardEmbeds` takes a window's embeddings whole.
- Requests (host, `requests.zig`, spec 147 WS-06): the page template (`chat_template_v2.jinja`, the
  checkpoint's `prompt.txt`), `Offer` (up to 128 pages, gundam alone, base among several, or the request's
  mode: `modeOf`, `prepare_asked`) and `Rules` (the n-gram rule over the reply, then `decode/page_guard.zig`).
  `served.zig` and `draft.greedy` both lay `Rules`; a rule of a reply goes there, not into one path.
- Decoding: `draft.zig` instantiates `decode/draft.zig`'s `Decoder` with no chain; `draft.greedy` is the
  serial reply under `Rules`, its prompt `decode/draft.zig`'s `promptWindow`s as Qwen3-ASR's. `generate` runs
  a fixture's prompt or ids.
- Steps run as captured graphs (`forward.Graphs`, WS-07): a window with no image row is captured once a (rows,
  scoring, forms) and replayed, so positions are device words and every kernel reads them there; a caller
  holds a `Graphs` (`init`, `deinit`), its capture `decode/graphs.zig`'s. A layer chained from the last
  writes the next norm in its residual add (`layerForms`' `chained`, common/glue.cu's `zs_residual_rmsnorm`).

## The attend and its router

- The attend (WS-04): `dash_attention.zig` over `kernels/unlimited_ocr/dash_attention.cu`, a tile kernel and a
  step kernel that give a row the same bits (a 32-key block a unit in both, partials folded in ascending block
  order), so the caller picks by cost (the step kernel to about 8 rows). The router's ids are -1 where a slot
  is not valid.
- The router (WS-03): `router.zig` wraps `kernels/unlimited_ocr/router.cu`; the forward calls `project`,
  `prep`, `kbar`, `route` in that order over the index cache (bf16 [capacity, 128] by position), k_bar (fp32
  [blocks, 128]), a block's image pages and a row's coordinates (`Coords`, the query page `marker count - 1 -
  prompt pages` in a reply).
- Weights: each layer's `self_attn.dash_router.{head_cls, q_proj, k_proj, q_norm, k_norm}` (`layout.Router`;
  `model.Router` stacks q and k); `self_attn.indexer.*` and `vision_model.embeddings.position_ids` are stored
  and read by nothing (`layout.unused`).
- Sources (the checkpoint's own, the behaviour reference with Triton off): `modeling_deepseekv2.py`
  (`_patched_init`, `configure_dash_inference`, the prompt and step attends), `dash_attention.py` (the
  router), `_sparse_base_impl.py` (page ids, page2d RoPE), `modeling_unlimitedocr.py` (`infer`, the decode
  allow mask).
- The state (`State.Dash`) keeps the index keys, k_bar (whole capacity, so a graph's route is fixed) and the
  blocks' image pages by position: a rejected draft leaves nothing to roll back. Coordinates (`pages.zig`):
  the prompt's from the prompt whole (`State.paged`, called before its first window, after `reset`;
  `generate` and `served` call it), a reply row's from `pages.Track` over the tokens kept (`commit`), a
  window's staged with its words. A prompt with no image or no `<PAGE>` up to its last one is refused
  (`error.Unpaged`; HF would fall back to 1d RoPE). The step kernel to `forward.dash_step_rows` (8) rows, the
  tile kernel above. `Model.load` keeps its own copy of the config's per-layer lists (`open`'s arena is freed
  after the load).
- Tests: `forward_dash_test.zig` (one load: layers from HF's input, the reply fed HF's tokens, the row rule
  with verify windows and rejected drafts, both greedy replies with HF's n-gram rule over the reply and its
  PageCountGuard), `pages_test.zig` (host, HF's coordinates).

## Long documents

(WS-08, after SGLang's v2 wheel: `doc_split.py`, `serving_chat.py`'s windows, `uocr_doc_reuse.py`):
`documents.zig` is the host plan (parts, lookahead, `TailCopy`, text tail, page base), `parts.zig` runs the
parts in the one state: a part that takes over copies the previous part's keys, values and index keys aside,
resets, lays them at the new prompt's head and after it, remakes their blocks' k_bar (`forward.summarize`) and
goes on from a forced `<PAGE>` (`State.carried`); RoPE pages are the document's (`State.pagedFrom`,
`Dash.base`, the page add where page > 0; query and block pages stay the part's). One part is the request as
it stands (`parts_test.zig` holds it to the bit); the copied rows are SGLang's approximation, a part's own
rows keep the row rule.
