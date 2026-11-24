---
checkpoint: baidu/Unlimited-OCR-2
revision: 9cf077c4b206e0e2a31879df937b0b0bcba3661c
model: Unlimited-OCR-2
family: unlimited_ocr
storage: BF16 as stored (the vision tower, the language model with its DASH routers, two files)
license: mit
disk_gb: 6.70
machines: 1
device_gib: 14.71
status: supported
drafters: none (greedy `Decoder(M)` with an empty chain; spec 147)
variables: ZSPARK_UNLIMITED_OCR_2, ZSPARK_REFERENCE_UNLIMITED_OCR_2
---

# Unlimited-OCR-2 (`baidu/Unlimited-OCR-2`)

**Machines: one DGX Spark**, which it fills by 14.71 GiB of device memory after a `serve` load at the defaults
below (spec 147 WS-06, spark-a). It reads one page image or several through `POST /v1/chat/completions`
([`docs/serving-media.md`](../serving-media.md#ocr)) and writes each page as tagged text after a `<PAGE>` marker.
Spec 147 is closed (2026-10-28); the figures against SGLang's v2 wheel are [below](#against-sglangs-v2-wheel).
It runs its bf16 checkpoint as stored: the NVFP4 builds failed spec 140's gate and were dropped (the owner, 2026-10-28). `zspark stack` serves v2 in place of v1
(the owner, 2026-10-28), with v2's instruction "Multi page parsing."; v1's instruction sent to v2 loops on nearly every page.

## The checkpoint

| | |
| --- | --- |
| Repository, revision | `baidu/Unlimited-OCR-2` at `9cf077c4b206e0e2a31879df937b0b0bcba3661c` (fetched 2026-10-27) |
| Licence | MIT |
| On disk | 6.70 GB for the directory: two weights files (6.69 GB), `config.json`, `tokenizer.json`, the HF modelling code, its README and paper |
| TensorFold family | none; zspark's `zig/src/models/unlimited_ocr/` follows `model_type: unlimited-ocr` and requires DASH `sparse_attention_config` |
| Variables | `ZSPARK_UNLIMITED_OCR_2`, `ZSPARK_REFERENCE_UNLIMITED_OCR_2` (`docs/environment.md`) |
| Behaviour reference | the checkpoint's own HF code (`modeling_unlimitedocr.py` `infer`, `_uocr_build_inputs`, `_uocr_generate`, `_uocr_page_count_guard`), held against the SGLang wheel its README serves it with |

The decoder has 12 layers, hidden size 1280, 10 heads of width 128, and MoE from layer 1.
OCR2 uses DASH attention: a causal window of 128 keys and up to 12
remote 32-token blocks a row chosen by a learned router a layer, with page2d RoPE over a row's page and its place
in the page (spec 147 Context).

## What is stored, and how it is multiplied

Everything is BF16 and runs as stored: BF16 linears and vision products, stacked BF16 routed experts,
and BF16 activations. The router's projection is a bf16 linear and its prep, block keys and route are
`unlimited_ocr/` kernels (`router.zig`); the attend is
`unlimited_ocr/dash_attention.cu`'s tile and step kernels, of the same bits (spec 147 WS-03, WS-04).

## Defaults that follow this checkpoint

| Setting | Default | Why |
| --- | --- | --- |
| `--name` | `Unlimited-OCR-2` (the directory's name) | |
| `--context` | 131,072 tokens | `max_position_embeddings`, `infer`'s `max_length` |
| reply tokens | the room the context leaves | `infer`'s `max_new_tokens` (`max_length` less the prompt); `--max-tokens` changes it |
| images | up to 128 pages a request | SGLang's wheel serves a document a request (the decision of 2026-10-27); 128 base pages fill the image budget (34,944 rows) |
| page prompt | `<PAGE><image>` a page, then the text or "Multi page parsing." | `infer`'s: a text with no `<image>` gets the pages before it; one `<image>` among several pages is replaced by them; as many as the pages stay; any other count is a 400 |
| crops | gundam (at most 16 tiles) for one picture; base (the 1,024 global view alone) for several, where the request names no mode | the HF processor's auto policy (`_uocr_build_inputs`, `crop_mode` unset). A request's `images_config` (or `mm_processor_kwargs`) `image_mode` (`auto`/`gundam`, `base`) or `crop_mode` picks as SGLang's v2 wheel reads them (`_resolve_mode_v2`); `tiny`, `small`, `large` are refused |
| tail rule | off | SGLang's `custom_params` `rep_min_size`, `rep_max_size`, `rep_min_count` (`_tail_has_repeat`) end a reply whose tail repeats; `custom_params.max_page_tokens` may lower the page cap |
| long documents | one window while the pages and their replies fit the state; else parts of pages with the previous part's tail taken over | SGLang's `doc_split_options` and `uocr_doc_reuse`; a part's pages: `max_batch_pages`, else `ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW`, else the state's (SGLang's page estimate, 274 base or 1,514 gundam, and 1,312 reply rows a page, measured: 82 base or 46 gundam pages a part at the default context); `overlap_pages` above 0 refused |
| `context_window` | 391,424 (the state's 131,072 and 128 gundam pages) | the host takes a document past the state so its parts can run; a request past the state that is not a document of pages ends `length` or is a 400 |
| n-gram rule | 35 in 128 for one page, 35 in 1,024 for several, over the reply alone | the README's request; `ngram_size` and `window_size` at top level or under `vllm_xargs`, a size alone taking window 128, a window alone disabling the rule. HF's v2 and SGLang (`SGLANG_UOCR_NGRAM_OUTPUT_ONLY`) read only the reply |
| PageCountGuard | on; a page at most `max_page_tokens` (20,000) | `infer`'s: the end token is banned while pages are owed, a marker too many forces the end, a page at the cap forces the next `<PAGE>` (or the end on the last); `max_page_tokens` in the body (top level or `vllm_xargs`) sets the cap, 0 none |
| decoding | greedy, one reply at a time | `--parallel` above 1 refused; no prefix shared between requests |

## Measured

spark-a, a fast build (spec 147 WS-06; raw lines `~/.cache/zspark/scratch-147/ws06/`).

| Figure | Value |
| --- | --- |
| Load | 2.9 s (`serve`) |
| On the device | 14.71 GiB at the default context |
| `serve`, `paper_p01` alone (909 prompt, 520 reply) | 160.6 tok/s server side |
| `serve`, `paper_p01` and `paper_p03` (553 prompt, 1,381 reply) | 163.5 tok/s server side |

Against HF's greedy reply (`ZSPARK_REFERENCE_UNLIMITED_OCR_2`): through `serve`, the two-page request equals HF's
tokens to step 114 and the one-page request to step 85, both steps where HF's own top two logits tie exactly; with
HF's image features instead of zspark's tower (`forward_dash_test.zig`) one page is equal to the end and two pages
part at the same tie at 114.

### Against SGLang's v2 wheel

Spec 147 WS-07, spark-a, zspark at `3815564b` (fast build) against `sglang-0.5.20.dev10+g8d62f5fb9c`, the wheel in the
checkpoint repository's `wheel/` at `9cf077c4` (torch 2.13.0+cu130, triton 3.7.1, Python 3.10 venv; it runs on GB10
once the venv's `ninja` and `/usr/local/cuda/bin` are on `PATH` for its JIT kernels), launched with the README's
command and environment on 127.0.0.1. Each side two loads in the order zspark, SGLang, zspark, SGLang, the machine
held alone (`machine.lock` exclusive, `gpu.lock`); a pass is one load: a warm-up, spec 137's 36 pages one request
each, then the two-page fixture (paper_p01 + paper_p03). Every request is the README's: the pages, then
`<PAGE><image>` a page + "Multi page parsing.", `max_tokens` 8,000 a page, greedy, `skip_special_tokens` false,
the n-gram rule 35 in 128 (1,024 for several pages), `max_page_tokens` 20,000, streamed; SGLang gets them as
`custom_params` with its `DeepseekOCRNoRepeatNGramLogitProcessor` and `images_config.crop_mode` true for one page,
false for several (gundam and base, as zspark and HF). Client-side figures (`bench.py`); decode is
(tokens - 1) / (end - first token). Raw lines and the scripts: `~/.cache/zspark/scratch-147/ws07/` (`W` below).

```bash
W=~/.cache/zspark/scratch-147/ws07
$W/bin/pass.sh zspark 1; $W/bin/pass.sh sglang 1; $W/bin/pass.sh zspark 2; $W/bin/pass.sh sglang 2  # timings
$W/bin/pass.sh sglang noov --disable-overlap-schedule       # SGLang's texts, the CER reference
python tools/reference/unlimited_ocr/export_v2.py cer --max-new 8000  # HF's eager replies (reference env)
$W/score-venv/bin/python $W/bin/score.py zspark-1 zspark-2 sglang-noov sglang-1 sglang-2
```

`score.py` takes `cer.py`'s strip and rapidfuzz's Levenshtein (the same distance; `cer.py`'s pure-Python one takes
minutes on 8,000-token replies).

| 31 pages without the GIF frames (21,470 / 21,479 tokens) | zspark, load 1 / 2 | SGLang, load 1 / 2 |
| --- | --- | --- |
| Seconds a page | 4.47 / 4.47 | 6.10 / 5.95 |
| First token, median (range) | 0.203 / 0.193 s (0.08 to 0.38) | 0.547 / 0.351 s (0.18 to 1.79) |
| Decode | 162.8 / 162.8 tok/s | 126.6 / 127.0 tok/s |
| Two-page fixture (553 prompt, 1,381 reply) | 8.64 / 8.59 s, first 0.154 s | 11.59 / 11.16 s, first 0.52 / 0.32 s |
| All 36 pages | 201.8 / 201.6 s, 31,520 tokens | 416.8 / 410.3 s, 49,615 tokens |
| Load to ready | 3.0 s | 67 to 92 s (graph capture) |
| Device memory, `nvidia-smi` | 14.9 GiB (context 131,072) | 78.5 GiB (`--mem-fraction-static 0.7`: 13.5 GB of weights, 70 GB of KV) |

zspark's server-side figures agree: first token median 0.20 / 0.19 s, 162.8 / 162.7 tok/s (160.5 to 163.6). Both
sides write the same text on every load. Load to load, zspark's spread is under 0.5%; SGLang's first tokens
spread by a third.

CER (`cer.py`: v1's det spans, v2's `<PAGE>` and `<BLOCK>..<FORK>` spans stripped). The reference is SGLang's
text with `--disable-overlap-schedule` (one more load, no speed taken from it): with overlap scheduling, SGLang's
default and the README's launch, its n-gram processor reads the reply one token stale and the rule does not fire
(spec 147's loops round). In this streamed request the no-overlap load still runs gif_f070, f140 and f210 to
the cap, where the loops round's non-streamed no-overlap request ends gif_f070 at 1,110 tokens, zspark's count;
the streaming interaction is open. The GIF frames are dense UI captures where the replies loop; the cap pages are apart.

| CER | zspark against SGLang (no overlap) | SGLang default against no overlap | zspark against HF's eager path |
| --- | --- | --- | --- |
| 30 pages, no GIF, no baidu | 0.0004 mean, 0.0078 max (onepager_p01) | 0 | 0.0003 mean, 0.0078 max |
| baidu (6 characters, RGBA) | 0.3333 ("Bai 百度" against "Bai du 百度") | 0 | 0.3333 |
| two-page fixture | 0 | 0 | 0 |
| gif_f000, gif_f140, gif_f270 | 0.0005, 0.8945, 0.6373 | 0, 0.9446, 0 | 0.0005, 0.5412, 0.5358 |
| cap pages gif_f070, gif_f210 | 0.8831, 0.5531 | 0.6516, 0.4706 | 0.8489, 0.7928 |

Reply lengths on the GIF frames (f070, f140, f210, f270): zspark 1,110, 2,142, 4,274, 1,815, all ending at EOS;
SGLang 8,000 (the cap), 8,000, 8,000, 3,428 with or without overlap scheduling in this streamed request; HF's eager
path 8,000, 4,771, 8,000, 1,545. zspark scores 0 against SGLang on 26 of the 31 pages without GIF frames.
HF's eager path (spec 147's first reference, `export_v2.py cer`, 22 minutes) decodes at 33 to 45 tok/s.

### Long documents and repetition loops

Spec 147 WS-08, spark-a, 36 pages (Mamba, arXiv 2312.00752v2, CC BY 4.0, 144 dpi, base mode, the README's request
with SGLang's fields), one load each: zspark in one window 47,182 tokens, 317.1 s (8.81 s a page, first token
2.58 s, 150 tok/s); zspark in windows of 10 (3 taken over) 299.0 s (8.31 s a page); SGLang in windows of 10 368.8 s
(10.24 s a page, load 69 s). CER: windows of 10 against one window 1.45%; SGLang against zspark one window 2.27%,
against zspark windows of 10 0.98%. Peak: 15.3 GiB on the device in both forms. The state costs 64,704 bytes a row.

The loops: on four GIF frames of a demo UI both v1 and v2 degenerate to the cap on HF, SGLang, vLLM and
zspark alike; the exact n-gram rule (35 in 128) cannot stop a period over 94 tokens or a counter, and zspark keeps it
as HF and vLLM do. SGLang's default launch (overlap scheduling) feeds its n-gram processor a stale `output_ids`, so
its rule never fires; with `--disable-overlap-schedule` it ends gif_f070 at zspark's 1,110 tokens. No normal page
reaches the cap on any side. The opt-in tail stop (`custom_params` `rep_*`) fired on no normal page.

## Tests

`ZSPARK_UNLIMITED_OCR_2=DIR ZSPARK_REFERENCE_UNLIMITED_OCR_2=~/.cache/zspark/reference-unlimited-ocr-2
tools/check.sh unlimited_ocr [FILTER]`: the DASH layers, the row rule and both greedy replies
(`forward_dash_test.zig`), the attend's kernels, the page coordinates (`pages_test.zig`); host tests: the
crop grids and page token budgets (`preprocess_test.zig`), shared Pillow resize/pad checks
(`tests/image/resize_test.zig`), and the template, host image expansion to HF's `input_ids`, and reply rules
(`requests_test.zig`), plus the guard (`tests/decode/page_guard_test.zig`). The OCR2 fixtures are
`tools/reference/unlimited_ocr/export_v2.py`'s; the tower's embedding and row checks use those fixtures too.
