# `zig/src/server/` instructions

This file extends `zig/src/AGENTS.md` and the root `AGENTS.md`. What is left of zspark's own server since spec 059:
`zspark serve` is TensorFold's host (its routes, parsers, template and tokenizer, kept in
`zig/vendor/tensorfold/` since spec 072) over `served.zig`'s engine, and this directory holds it and what it calls. Nothing here
imports `cuda/cuda.zig` or a model. Records: specs 017 (structured output), 018 B6 and 051 (scheduler), 059.

## Purpose

Each file's `//!` header says what it holds, its TensorFold source and every departure from it. Read it before
an edit and keep it true.

- `request.zig`: the names a model and what stands before it share: what a reply must be (`Structure`, with
  its tools: `Request`), where its text goes (`Sink`: the `chat` command's), `Finish`, `Diagnostic`, `Error`.
- `served.zig`: `Served(M)`, a model behind `engine_api.Engine`: the requests' table and their admission,
  `status`, `memory`, a request read, its constraint and its refusal; the model's `reply` decodes
  (`../models/AGENTS.md`).
- `served_live.zig`: `served.zig`'s `Live` (spec 171), one request's own thread: the request read, the
  session's turn (`enters`, `leaves`), the constraint lent to the model (`constrained`), and the reply's
  events (`give`, `closing`, `row`, `taker`).
  `zig/tests/server/served_test.zig` (and `served_queue_test.zig`) runs them on the host behind TensorFold's
  server, over a socket, with a model that decodes nothing; `models/qwen4/served_test.zig` runs them on a
  model. What of TensorFold's interface they leave at its defaults, and why: `REFERENCE.md`.
- `command.zig`: what the models' `serve` commands share: the words parted from the host's (`part`), the
  host's flags and those of them zspark refuses (`unserved`), text loading and its diagnostics (`loadText`,
  `loadTokenizer`), the grammar engine's setup (`Grammar`) and `samplingDefaults`. Callers own the loaded
  text and free their tokenizer wrapper before the host tokenizer it borrows.
- `structured.zig`: a `request.zig` `Structure` compiled into the reply's `grammar.Constraint` (`Compiler`):
  `served.zig` makes one from TensorFold's `CallGate` or `Structure`. The tests' compiler on the tiny tokenizer
  is `zig/tests/server/structured_fixture.zig`'s `Bench`.
- `xgrammar.zig`: a structure the host compiled through xgrammar (`libtfgrammar`, since TensorFold 1.0.12) as
  a `grammar.Constraint`, positions a tree over the one matcher it rolls back (spec 106).
- `segments.zig`: `Segments`, the host's `model_text.Text` as the `serve` commands hand it over: `encode`
  answers a prompt's segments (an added token and the text up to the next) from the ids kept of earlier
  requests, 32 MiB a server at most; a held turn's three encodes of the whole conversation were 10 to 20 ms
  at 17,000 tokens. Its header has why a segment's ids are its own and what turns it off; a change to the
  tokenizer's added-token split (`zig/vendor/tensorfold/`, `Tokenizer.encode`) is a change here.
- `images.zig`: `Offer`, a checkpoint's image settings as the host's `engine_api.Vision` (spec 019 WS-04): an
  image part's bytes decoded (`../image/`), resized and patchified (`../text/vision.zig`), and the refusals in
  vLLM's words (`explain`). The vendored host's `server/images.zig`, a module of the same name, takes the bytes
  from the URL and calls it; a format a client may send or a refusal's words change here and in
  `docs/serving.md`'s "Images".
- `scheduler.zig`: several replies over one decoding thread: `Scheduler` and a model's `Decoder`, called on
  the decoding thread alone. `Scheduler.decode` takes a job its caller made and gives its tokens to a `Taker`
  on the caller's thread (`served.zig`); nothing here makes text. A decoder says a reply's end token and counts before its end (`Round.tail`,
  `Done.tail`); the end token is never among the emitted tokens. `zig/tests/server/scheduler_fake.zig`: `Fake`, the
  tests' model, and its text side around `decode` (`Fake.Model.generate`).
- `NAME_test.zig`: a file's tests. `../server_tests.zig`: this directory's test root.

## Rules

- What a client sees is TensorFold's host's, and is changed there, in `zig/vendor/tensorfold/`, by that
  directory's rules (`AGENTS.md` there): the import is upstream's bytes, a local change is one commit with
  its row in `PATCHES.md` and its test, and `zig build test-golden` runs after each. What changes a reply
  goes to the owner first. What the engine owes the host is in `served.zig`'s header (`engine_api.Engine`).
- The engine has no lock on the grammar engine; which thread does what stands for one. A constraint is
  compiled on the connection's thread (`Served.refusal`), or on the request's where the host did not ask: a
  compile writes only its own constraint. Every `start`, `fill` and `accept` writes the caches constraints
  share: a session asks them in its turn (`Live.enters`), the scheduler on its decoding thread alone. The
  constraint is the request's: `Live.constrained` takes or compiles it and lends it, and it is freed once the
  model's `reply` has returned and its slot is left. A mask asked anywhere else races.
- What cannot be enforced is `error.BadRequest` with the keyword and its path, before anything is decoded,
  and a 400 before anything is sent: the host asks `Served.refusal` on the connection's thread (`Engine.refusal`,
  a patch of the vendored host), which compiles the request's constraint under no lock, asking no mask, and
  keeps it for the reply (`Served.ready`: `Live.constrained` takes it, so a request compiles once). A refusal
  `constraintFor` gains is therefore a 400 with no further change; one made later in a model's `reply` is a
  failed reply, a 500.
- The loop guard and the thinking budget (`../decode/guard.zig`, spec 074) are a constraint `Live.constrained`
  puts around the request's own, so a model that decodes under `decode/masks.zig` has both and no model file
  names them. They are TensorFold's rules (`core/lanes/stream.zig`; its `detectLoop` is called, not copied): a
  change there is taken with the release, and `guard_test.zig` holds its engine's cases. A forced token is a
  mask of one bit, a function of the reply's tokens before it: nothing of another stream, and no mask at all
  where nothing is forced (`Guard.free`), so a guarded reply's bits are an unguarded one's until it fires.
- A required call is written in the template's form where the tool's schema can be held in it and as the JSON
  payload where not (spec 023). A change to how the host reads a call (its closers, how it types
  a parameter) is a change to `text/call_gate.zig` too.
- A required call or a document after thinking is decoded with what the template writes after `</think>`
  (`text/frontend.zig` `call_lead`, `content_lead`; specs 034, 035).
- `--parallel 1` is `models/qwen4/session/session.zig` alone: one reply, the others wait. Above it the scheduler
  stands over `models/qwen4/pool/hall.zig`; one live stream is still that session (spec 051), and a second stream
  opens the pool. An image prompt runs on either (spec 019 WS-05). GLM's is `models/glm5/pool/pool.zig` over its hall, through `Scheduler.decode` alone; its
  `--parallel 1` is its session and nothing of the scheduler.
- Under the scheduler the tokens are the decoding thread's and what is made of them the request's thread's: a
  token at a time, in order. A slow taker fills its own box and holds no round.
- A stop is heard late and changes nothing a client sees: when the taker says `.stop` its stream's flag is
  set and no further token is taken.
- A request waiting or being filled is asked every `Options.probe` whether its reply is still wanted, and at
  once when `Served.cancel` wakes it (`Scheduler.wake`): one still in the queue is taken out by its own
  thread and offered to no decoder, whatever round is being run.
- Admission (spec 074): `submit` takes every request unless `--max-queued` is given (`command.zig` `bounded`,
  `Served.max_queued`), and then refuses with `error.Busy` what is beyond `lanes` and that many waiting; a
  default that refuses is the owner's. What a client reads for `Busy` is the host's, a 503 (`served_queue_test.zig`): a patch of `openai.zig`'s
  `Run.fail`, to carry over a release. A stream is refused so before its headers: since TensorFold 1.1.1
  (#912) the host submits before it opens a stream, as vLLM's `_preflight` (`Engine.full` retired, spec 113). A session's requests take its turn in arrival order through `Live.enters` and
  `Live.leaves`, never a mutex of their own: `cancel` must be able to end one that waits.
- A request the context cannot hold is the host's to refuse (`chat.zig` `prepare`, before the engine is asked):
  a 400 with `context_length_exceeded`, streamed or not. The stream's is a patch of `openai.zig`'s
  `Run.stream`, to carry over a release (spec 083); upstream sends it as an event of a 200.
- The scheduler has one mutex and both conditions; a stream owns none, so its request may free it once it
  reads `ended`. Neither a `Taker` nor the `Decoder` is called under the mutex.
- The scheduling rules are in `scheduler.zig`'s header, each with a test and a mutation; a new one has both.
- Expected values come from TensorFold's own functions through `tools/reference/`. Don't hand-edit a fixture.

## Local commands

```bash
tools/check.sh server [FILTER]    # host only (zig build test-server)
zig build test-golden             # TensorFold's recorded replies against its server, over its scripted engine
zig build bench-held-turn -Doptimize=fast -- "$ZSPARK_FLASH_NEXT_FP4_LOCAL" memo=on   # the host's ms before `submit`; memo=off: without `segments.zig`
```

- The scheduler's tests run `Fake`. A test that needs an order holds the rounds (`Fake.manual`, `allow`,
  `free`), never sleeps.
- `tests/server/served_test.zig` and the tests beside it (`served_queue_test.zig`: Busy, `--max-queued`, the
  turn and the pool's queue) run TensorFold's server: a status, a header or an error body a client sees is
  checked there, on a port of the loopback the kernel gives
  (`served_fixture.zig`'s `Stand`).
- The server on a loaded model is checked through `models/qwen4/served_test.zig` (the random model) and
  `tools/check.sh accept CKPT`, which starts `zspark serve` on 127.0.0.1:18888.
