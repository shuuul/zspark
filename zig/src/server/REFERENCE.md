# `zig/src/server/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## TensorFold's interface left at its defaults

Of TensorFold 1.0.4's and 1.0.5's interface and on, each left at its default by `served.zig`, so that the host does
what it did under 1.0.2 (`zig/tests/server/served_test.zig` asks the routes):

- `Request.decode_spans` (which rows of the prompt an earlier reply decoded, found by the host from the
  tokens; `Engine.submit` validates them, so `SubmitError.InvalidSpans` is the host's own) is not read:
  TensorFold keys a kept state by them where decoded and prefilled rows differ in bits. Here a row's bits
  are its own whoever made it (the row rule), so reading them would only matter to a cache that keeps
  both kinds apart.
- `Info.warm_turns` is `Served.warm_turns`, false but under `serve --warm-turns` (spec 106 WS-07): the
  host then submits a prompt-only request (`max_tokens` 0, `background`) behind each chat reply, the
  conversation with the reply as its client got it and the next user turn's header (`server/warm.zig`).
  Such a request reaches the model as an `Ask` of one token with `warm` set, which the model runs only
  when nothing else is live and drops otherwise; its token is given to no one (`Live.give`), so the host
  sees its end alone. A request of `max_tokens` 0 that is not one is the `length` it always was.
- `Info.vision` is `Served.vision` (spec 019 WS-04): `images.zig`'s `Offer` for Flash Next FP4-Local on
  one session, null otherwise, where an image part is the host's 400, "this server accepts text parts
  only; image, audio and video inputs are unsupported". The host decodes and patchifies each image
  through it and expands its placeholder; `Request.images` reaches the model as `Ask.images`.
- `Info.plain_only` false (it only words the start line's `drafts: off`; `--serial` and `--no-drafts`
  already do). `Info.prompt_cache_plan` is the command's immutable budget and whether
  `--prompt-cache-gib` was supplied; Flash Next gives it to `/memory`.
- `VTable.learn` null (Sliding Weights): `POST /v1/slide/learn` is a 501. It would take a learner that
  steps the read-out's weights between rounds and writes shards; TensorFold has it for one family on
  Metal. `Open.drafter`, `drafter_bits`, `slide` and `learn_min_free_gib` are fields of TensorFold's own
  `open`, which zspark's commands do not call: `command.zig`'s `unserved` refuses the flags that fill them.
- `Info.logprobs` is `Served.logprobs` (spec 101 WS-02): true for Flash Next on one rank, false on two
  and for GLM, where a request with `logprobs: true` is the host's 400, "logprobs are not supported by
  this model or backend". The host asks for them of a chat completion, streamed or whole, and of a legacy
  completion, with no tools, stop strings, structure or thinking budget, and needs the text's
  `token_bytes` (`Segments` hands the host's on). The model is then asked `Ask.logprobs` (how many best
  tokens a row, 0 to 20) and gives each token's row to `Live.row`, which sends it as an `Event.logprobs`
  just before the token's `tokens` event, the token that ends the reply included: the token, its log
  probability under the model's raw distribution (the read-out's logits, before the temperature, the
  sampling's filters and any mask, as TensorFold's and vLLM's `raw_logprobs` by default; SGLang's sampled
  path takes them after the temperature), and the best ids with theirs, logits descending, ties to the
  lower id (`decode/logprob.zig`). A row's are the same bits from a serial step, a verify window, a
  prompt's last window and a pool's round, as its token is. No rank is given (vLLM's, which only
  `prompt_logprobs` would read).
- 1.0.5, `Info.prompt_cache` true: `prefilled` counts the tokens restored, and `/metrics` publishes
  `prompt_tokens_cached_total`, `prefix_cache_queries_total` and `prefix_cache_hits_total`. The flag also
  enables the host's extra probe render for `Request.rewind_len` (where the last user message's text
  starts); its result is unread here: `decode/prefix.zig` keeps states by its own marks.
- 1.0.5, `Info.greedy_only` false (the engine samples); `Memory.pinned` 0 (`/metrics` prints
  `pinned_memory_bytes 0`: no pinned host buffer is counted).
- 1.0.12: `Info.prompt_logprobs` false (`/v1/completions`' `prompt_logprobs` is the host's 400; the
  `prompt_logprobs` event is never sent); `Info.admit` null (TensorFold's own lane host reads it, to hold a
  request until its caches fit beside the running streams; the server does not, and here `submit` and the
  pool's queue decide, as before); `Info.prefill_step` 0 (the engine cuts a prompt itself); `Info.decides`,
  `pictures` false, and `VTable.decide`, `recall`, `imprint`, `drain` and `keepalive`
  null; `Request.cache_key`, `born` and `decide` not read; `SubmitError.Stale` never returned
  (Living Weights); `Memory.limit` null. `Info.thinking_budget` is its default, true: the budget is
  enforced in `served_live.zig` (`served.zig`'s header). `VTable.score`, with 1.0.12's `mass`, stays null.
- 1.1.1 (spec 113 WS-03): `Info.text_generation` true, `speech` null and `VTable.submit_speech` null
  (`POST /v1/audio/speech` is the host's refusal), `slide_conversations` false; `Stats.queued_seconds`
  measures engine submission through the first admission: the session's turn, or the scheduler's
  first successful offer timestamp, including its queue and excluding prompt execution. `Event.audio`
  is never sent; `Request`'s and `Open`'s new fields (`Open.kv8`, `--kv-cache`) not read; `Opened.halt` and
  `follow` are TensorFold's own
  `open`'s, which zspark's commands do not call. A stop signal is now the host's drain: replies in
  progress have `TF_DRAIN_S` (20 s) to end while new requests are 503s; the commands hand `serve.run` no
  `EngineStop`, so nothing is halted at its deadline and their `close` ends what is left after it returns.
- 1.1.4 (spec 141 WS-02): `Info.embeddings` null (a request's `embedding` is the host's `InvalidRequest`,
  `SubmitError.InvalidRequest` never returned here) and `Info.answer_limit` false (`Request.answer_max` not
  read; `--thinking-reserve` is the host's "not applied" line); `Reason.silence` never given (speech's alone;
  the transcription route answers it with a 500); `Open.max_frames`, `silence` and `prompt_int8` are
  their flags', which `command.zig` `unserved` refuses. The lane engine's `ofLanes`, `images` and
  `prefill_yield` are TensorFold's own lane host's.
- 1.1.5 (spec 145 WS-01): `Request.user_end` (where the last user message ends, for TensorFold's kept interior
  prompt state, #1262) is the host's and not read; `Info.prompt_cache_capacity` gives Flash Next's
  retained-state budget and the keep planner's smallest state (one recurrent side, conv inputs and PLE
  tails, no pooled rows). The session's budget bounds its parked sequences; its own snapshots are beside
  it. GLM has fixed slots and snapshot counts, with no byte budget, so capacity and plan stay null.
  `Open.loops` is `--loops`', which `unserved` refuses, and `Open.cfg_enabled`,
  `cfg_batched` the `--cfg-*` flags', which the host refuses off macOS. `RenderOptions.clear_thinking` reaches
  the template through `segments.zig`'s `render`, which hands every option on.
- 1.2.1 (spec 152 WS-03): `Info.draft_sources` is filled by each `serve` command (`Served.draft_sources`), so
  the serving line's `drafts:` names the proposals a model offers (`MTP`, `DFlash2`, `copies`); left at the
  default it would print `none` where 1.1.5 printed `on`. `Info.penalties` false (a request's
  `presence_penalty`/`frequency_penalty` are ignored, and the host says so once); `Info.decode_share` null
  (`--decode-share` is host-native and zspark does not read it: `command.zig` `unserved`); `Request.penalties`
  and `Request.loop_stop` not read (`--loop-guard-stop` unserved); `VTable.load` null (`GET /v1/load` then has
  no engine group, only the ledger's own counts); `Structure.compiled`, `budget` and `free` are the host's
  alone (it compiles under xgrammar and frees after `finished`; zspark's `refusal` path hands the request
  uncompiled as before); `SubmitError.MemoryPressure` never returned; `Stats` moved to `engine_stats.zig`, its
  new `grammar`/`gpu_seconds` left. Of `Open`, `drafter_name`, `rope_yarn`, `ssd_experts_gib`, `ple_on_ssd`,
  `fsr_trace`, `spill` and `decode_share` are fields of TensorFold's own `open`, which zspark's commands do not
  call: the flags that fill them are refused in `command.zig` `unserved` (`--rope-yarn`, `--ssd-experts`,
  `--ple-on-ssd`, `--fsr-route-trace`, `--decode-share`; `--spill-gib` is zspark's own, parted before the
  host's words). `--name-priority`, `--loop-guard-stop` and the FSR flags are also `unserved`; the
  `--max-inflight*`/`--requests-per-second`/`--request-burst` admission flags are the host's own 429s, off by
  default, and served.

- 1.2.3: `Info.readouts`, `rerank`, `benchmark_ranks`, `benchmark_group` and `ledger` null (no model of
  zspark is a readout or a reranker: `POST /v1/rerank` is the host's "this model is not a reranker", and a
  `Request.readout` never arrives, `submit` refusing it first), `Info.reasoning` true and
  `reported_context_window` null (`/models` and the status line name `context_window` itself). `cli.Args`
  gains `--profile`, whose one accepted value is the default `stable`, and `--vision-pdf`, which the parser
  refuses without `--vision` (itself `unserved`): neither reaches `unserved`. `Request.readout` is the
  engine's own; a non-readout engine never sees it.

- 1.2.5: `Info.image` null and `VTable.submit_image` unset (no model of zspark serves a generated picture:
  `POST /v1/images` is the host's own refusal), `Info.thinking_switch` and `thinking_applicable` the
  server's from the template and tokenizer as upstream sets them, `Info.prefill_min_chunk` its default,
  `Memory.standing` left (the streams' ledger is the host's own account), `Family` not handed (the
  models-name paths read `Info.name` as before), `platform/environment.zig`'s settings table upstream's
  (a malformed `TF_*` is refused by name at startup, zspark's own `ZSPARK_*` untouched). `Engine.refusal`
  is upstream's own now (#2082): zspark's entry answers in its caller buffer, no allocation, an
  out-of-memory answering none (the reply failing as before). `Info.penalties` stays false: where the
  host ignored a nonzero presence/frequency penalty and logged once, it now refuses it with a 400
  ("send 0 or leave them out"); a client-visible change of the host's, reported, not zspark's choice.

- 1.2.8 (spec 174): `Info.ocr` null — upstream's `POST /v1/ocr` reads page images through its own
  `Info.ocr` engine interface (Mistral's contract, `server/ocr/`); no zspark engine is an OCR reader, so
  the route answers the host's own `bad_input` refusal rather than being `unserved`. `chat.Input` gains
  `cancel` (the `POST /v1/requests/{id}/cancel` route's ask, wired by the host into `Generation`), 
  `image_bytes` and `fill_room` (the host's OCR route sets them; chat keeps its defaults), and
  `Prepared.frames`/`Rendered.frames`/`PreparedImage.frame`/`Size` pass through unread. `Admit`, `Ledger`,
  `Memory` and `MemorySource` moved to `engine_admission`/`engine_status` as re-exports, shapes unchanged.
  The connection flags (`--max-connections`, `--max-connections-per-address`, `--trusted-proxy`) are the
  host's own accept-layer admission (`Config.max_connections`/`peering` built by `host.serve.run` from
  `cli.Args.connections`), off by default, and served — not `unserved`. `--compact-at` now has the later
  flag win (upstream #2883) and `--profile` names its refusals; no `Args` field zspark reads changed.
  The tokenizer is hardened, not changed in behaviour: ids are capped at `models.max_id` and a split
  pattern has a work budget (`error.RegexBudget`, which `hf_text` maps to `error.Template` with its own
  words) — a hostile or enormous prompt fails to tokenize instead of exhausting the server.

## Cache and prefill status

`Status.prompt_cache` reads atomic publications from the decoding thread, never the model's store from
an HTTP thread. The existing scheduler counts are read before `Served.mutex`; no cache hook takes the
scheduler lock. A session publishes after its turn; pools publish after admission, rounds, retirement and
preemption. The gauges are the latest completed publication, so a long running prompt can lag a sample.

Flash Next's pool reports `store.marks.items.len` and `keeps.bytes()`, including pending states and spare
buffers. Before the shared pool opens, and in a single session, it counts prompt/reply snapshots and
parked snapshots, with `snapshotBytes() + lot.bytes()` for their storage and reusable buffers. A pair
reports its retained shelf positions and snapshot bytes on this process's rank, including spare buffers.
GLM's pool retains one sequence per idle nonempty slot; bytes are its allocated KDA and DSA storage on
the ranks this process holds. Its session reports the current retained sequence, marks and spare KDA
buffers, including a mark's DFlash2 context. The active decoding workspace is excluded. OCR and
Nemotron restart each request from empty and retain no prefix cache, so their occupancy is null.

`Info.prompt_cache_capacity` describes the load's capacity plan. Its budget follows the applied states
budget: `--prompt-cache-gib` is held whole; on one rank an unset budget takes what the pool's fit leaves
of the default, including zero. The answer reads the pool's options directly (spec 161). The smallest
state is the keep planner's fixed figure taken at load; a session's budget is its `parked_bytes`.

`Status.fill_unknown` is true while an admitted foreground request has not finished prefill. A session
sets it when it takes its turn; a pool sets it on successful admission. The prefilled event clears it,
as do retirement and preemption. Queued requests and background warm passes do not set it. Prompts run
inside a decoder's admission call are visible after that call returns successfully; no progress total
is invented for them.

