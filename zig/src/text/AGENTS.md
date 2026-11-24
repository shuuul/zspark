# `zig/src/text/` instructions

This file extends `zig/src/AGENTS.md` and the root `AGENTS.md`. Text to token ids and back, a request's messages
to the prompt's text, and a reply held to a schema or a tool call, on the host: nothing here needs the GPU or
imports `cuda.zig`. Records: specs 004 (text), 017 (constraints), 023, 034 and 035 (the gate, the leads).

## Purpose

Each file's `//!` header says what it reads, what it refuses and where it departs from its reference. Read it
before an edit and keep it true.

- `tokenizer.zig`: `Tokenizer` over TensorFold's tokenizer (the build's `tokenizer` module, spec 059): text to
  ids is the dependency's; this file keeps what zspark's decoding reads, each token's bytes, which ids are
  special, and `StreamDecoder` (ids one at a time, whole characters out). `over` borrows the host's instance.
- `template.zig`: `Template` over TensorFold's Jinja engine (the `template` module), and the text helpers
  `pyjson.zig` and `server/structured.zig` share. The engine names no line; a failure is its message.
- `messages.zig`: what TensorFold does to a request's messages before the template, and its variables.
- `pyjson.zig`: Python's `json.loads` into a `std.json.Value`, for tool-call arguments.
- `vision.zig`: an image prompt from decoded RGB (spec 019 WS-01), TensorFold 0.6.5's processor: `Settings`,
  `smartResize`, torchvision's resize (`picture.resample` `.torch`) and normalization, the patches, `content_hash`,
  `prepare` (markers expanded, re-tokenized, refused in 0.6.5's words) and `Layout` (M-RoPE positions, `shift`).
- `frontend.zig`: a checkpoint's tokenizer, template and generation config; messages to the prompt's ids. At
  load it probes the template for its `openers` (one for every message, else each role's as TensorFold's
  `markers`; `starts`: where a prompt's messages begin, for the sessions' marks), its `call_lead` and its `content_lead` (the text after `</think>`, kept when newlines alone).
- `grammar.zig`: the decoder's interface to a constraint (`Engine`, `Spec`, `Constraint`, `Position`, a value);
  `Led` forces tokens before any constraint.
- `json_schema.zig`: the one engine, a JSON schema as a mask over the vocabulary (`Vocabulary`, `Grammar`,
  `Matcher`, `Engine`); `Validator`, a second reading of a schema written apart, for the tests, is
  `zig/tests/text/json_schema_validator.zig`.
  `json_builder.zig`: a schema's text into its tree (`Schema`, `Builder`). `json_machine.zig`: the frames and
  `Machine`, a byte from a position.
- `call_gate.zig`: a required tool call as a `grammar.Constraint`, in the template's form or as JSON.
  `call_gate_strings.zig`: the text a value may not hold (`automaton`, `Strings`); `call_gate_plan.zig`: a tool's
  schema read for the template's form (`planOf`).
- `zig/tests/text/NAME_test.zig`: the tests of the files above that were split (`json_schema`'s are three, with the
  properties and the oracle apart; `call_gate`'s three, the template's form in `call_gate_walk_test.zig` and the
  checkpoint's template and tokenizer in `call_gate_checkpoint_test.zig`).
  `testing.zig` beside them: the fixtures two of them read, `schemas` and `gate`. `zig/text_tests.zig`: the root.
- `zig/tests/fixtures/text/`: what the `tokenizers` library, jinja2, TensorFold and xgrammar give, for the tests to equal.

## The reference is the `tokenizers` library, through TensorFold's tokenizer

Ids must equal the library's (0.23.1 in the TensorFold container). The pipeline (normalizer, pre-tokenizer, BPE,
added tokens) is the dependency's and is not changed here; the fixtures check it on the tiny tokenizer and on
the checkpoint's, so a new TensorFold version is taken with `check.sh text` green or not at all. What is ours:

- Decoding: each token's bytes as the `ByteLevel` decoder writes them (the token as it is when a character is
  no byte's), joined, then `String::from_utf8_lossy`. Another decoder is refused at load.
- Text that is not UTF-8 is refused before the dependency sees it.
- An added token with `single_word` is refused by the dependency; the tiny tokenizer has none for that reason.

## The template renders as jinja2 does, or not at all

- What the engine does not read fails at parse; a value it cannot print fails at render. A render writes
  nothing unless the whole render succeeds.
- One difference from jinja2 is known and skipped by name in `template_test.zig`: a kept `d|items` prints as a
  list, where jinja2 prints a generator. No checkpoint's template writes one.
- A caller renders through `messages.prepare` and `messages.context`; the template alone does not normalize.
- Checkpoint templates are not in the repository (their licences are not Apache-2.0). Tests read
  `chat_template.jinja` from `ZSPARK_FLASH_NEXT_FP4_LOCAL` and `ZSPARK_GLM_FP4_LOCAL`, check its SHA-256, and skip without the
  variable.

## The schema matcher is zspark's own, so its tests are what stands behind it

A wrong mask is a reply that is not the schema's, or one that cannot end. What holds the matcher:

- xgrammar is the oracle. Over the corpus in `tools/reference/grammar_masks.zig` (`$ref` and recursion,
  `anyOf`/`oneOf`, a Pydantic model, the server tests' tools, nesting, empty objects and arrays, enums that go
  on from one another, escapes and characters past ASCII, numbers, whitespace) every mask before every token of
  documents cut three ways and of random walks is xgrammar's, bit for bit on the tiny tokenizer and by hash on
  the checkpoint's. The differences are meant, named in the test and counted:
  1. Special tokens are never allowed (xgrammar takes one where it is handed the token's text).
  2. A string is well-formed UTF-8: no surrogate encoded as ED A0..BF.
  3. An escaped surrogate has its partner: `\uD83D` alone, or `\uDE00` alone, is refused, as `std.json` and
     strict parsers refuse it; xgrammar takes any four hex digits.
  4. A schema with `minItems` or `maxItems` and neither `type` nor `items` (`{"maxItems":1}`) is an array
     within its bounds; xgrammar takes any value there, of any length. With `items`, or with object keywords,
     both read the kind the keywords describe.
  xgrammar is also where two readings came from that JSON Schema itself would not give: an object's keys in
  the schema's order, and `"properties": {}` taking `{}` alone.
- Properties, seeded: what the masks take is a valid document by `Validator`; a valid document is taken however
  it is cut into tokens; no position is a dead end and the stop is allowed exactly when the document is
  complete; a mask depends only on the tokens before it. And a schema changed at random compiles or is refused,
  never crashes, and what compiles still takes only valid documents.
- Refusals: every keyword that is not read is refused by its name wherever it stands; a malformed schema is an
  error; nesting past `max_depth` (64), a schema past `Options.max_schema_bytes`, a literal past `max_literal`
  are refused with the bound. `accept` adds at most two frames a byte and fails with `ConstraintFailed` past
  `Options.max_frames`, for that reply only.
- A change to the machine is tried as a mutation: break the rule on purpose and see a test fail. Spec 017's
  handoff lists the ones that were.

Known limits, each refused or stated, none silent: a schema whose every document is infinite never reaches a
stop (the reply runs to its token limit); a literal with a byte the tokenizer has no token for leaves an empty
mask, which the decoder reports as `ConstraintFailed`; integers and strings have no length bound of their own.

## Local commands

The script runs in the TensorFold container, without the GPU; its docstring has the full command.

- `tools/reference/tokenizer_cases.py CKPT --repo . --out-dir zig/tests/fixtures/text` writes the fixtures;
  `--more N --more-dir DIR` writes N random strings more outside the repository.

```bash
zig build test-host                                            # the tiny tokenizer's cases
ZSPARK_FLASH_NEXT_FP4_LOCAL=<checkpoint dir> zig build test-host         # and the checkpoint's tokenizer and template
ZSPARK_TOKENIZER_MORE=<DIR> zig build test-text -Dtest-filter=random   # and the random strings
tools/check.sh text json_schema                                # the schema matcher alone; the checkpoint's
                                                               # vocabulary too where env.sh names it
```

## When editing here

- Do: regenerate the tables and the fixtures when the container's `tokenizers` moves, and read the diff.
- Don't: write a tokenizer or a template engine here again; a fault in either is reported to TensorFold with
  the fixture's case, and the dependency's files are not edited.
- Do: add a case to `tokenizer_cases.py` for an input class that is not covered, rather than a hand-written id
  list in Zig.
- Don't: allocate in `StreamDecoder.next`; its buffer is sized at `init` for the longest token.
- Do: add a jinja2-produced case in `tools/reference/template_cases.py` before supporting a new construct.
- Don't: hand-edit a fixture; regenerate it in the TensorFold container.
- Do: probe the template at load (`frontend.zig`) for a new form of reply that must render as it was
  decoded; its text is not written into the code.
- Do: put a narrowing of the masks such as `keyTokens` in at the interface, in `fill` and `accept` together,
  never one, and run the property tests both ways (`Options.key_tokens` false is xgrammar's masks).
- Do: when the host's reading of a call changes, change `call_gate.zig`'s `automaton` and `planOf` with it;
  the gate's tests parse its replies by those rules (`testing.zig`'s `gate.read`), written apart from any parser.
- Do: refuse what the matcher does not read, by the keyword's name. A schema is never approximated: a mask
  that is too wide breaks the client's parser, one that is too narrow hides answers.
- Do: add a schema class to the corpus in `tools/reference/grammar_masks.zig` and regenerate, rather than
  write an expected mask in Zig. A difference from xgrammar is either fixed or given a name in the test.
- Don't: allocate in `fill` beyond the caches (a frame's contents are worked out once), or change a frame
  after it is added: positions are indexes into them.
