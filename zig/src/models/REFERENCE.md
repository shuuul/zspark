# `zig/src/models/` reference

Read when a task touches what a section names; the `AGENTS.md` beside this file binds every task here.

## The contracts in full

- **Its commands**, to `zig/src/decode/dispatch.zig`. `NAME/commands.zig` declares one `dispatch.Model`: the name messages
  use, every `model_type` its `config.json` may carry, and a `dispatch.Command` for each command it has (the
  name, the fewest words after it with the directory first, a usage of its own if it has one, `run` over those
  words). `main.zig` lists the model in `models`, one line. `zspark COMMAND DIR ..` reads `DIR/config.json`'s
  `model_type` and runs that model's command. A model leaves out a command it lacks, and `zspark` then refuses
  it naming the commands the model has; a `model_type` no model lists is refused naming those known. A
  directory whose `config.json` cannot be read as far as `model_type` goes to the first model listed that has
  the command, which reports the fault in its own words. A command is a file under `NAME/commands/`; it opens
  its directory with `cli.Opened` over the model's `Config`, `Layout` and `Diagnostic`, aliased once a model
  (`commands/common.zig`'s `Opened`; GLM's drafter's is `dflash2/rig.zig`'s), and reads `--flag`s with `cli.parse`.
- **Its drafted decoding**, to `zig/src/decode/draft.zig`. `NAME/draft.zig` declares the namespace that `Decoder(M)` asks
  for and instantiates `Decoder` and `Chooser` once; the model's other files name those. The contract is the
  doc comment on `draft.Decoder`: `Model`, `State`, `Buffers`, `Graphs`, `Error`, `forward`, `forwardScoring`,
  `commit`, `logitsRow`, `streams`, and the two rules that make a draft's verification
  exact (a row's bits never depend on the rows beside it; a window equals its serial steps). `M` is a
  compile-time parameter: a round calls the model's functions directly, and nothing may put a pointer to one
  between the decoder and the forward. A draft source (an MTP head; GLM's DFlash2 beside it) is a
  `draft.Drafter`, chosen when a reply starts, and steps its drafts with `draft.Chain`. A session wraps it
  in `decode/copied.zig`'s `Copied` (drafts copied from the context) and tells it each reply's whole prompt;
  a pool that drafts its streams together gives each one a `Copied` and calls its `after` and `offer`.
  `zig/src/decode/draft_test.zig` is the smallest model that satisfies the contract, on the host: start from it.
- **Its serving**, to `zig/src/server/served.zig`, the engine TensorFold's host serves. `NAME/.../served.zig`
  declares the namespace `Served(M)` asks for and instantiates it once. The contract is the doc comment on
  `served.Served`: `Behind` (its session, its pool, what it reads of a request beside them), `reply` (one
  request's reply on the request's thread, through `Live`: `give`, `taker`, `prefilled`, `closing`,
  `reasonOf`), `counts` (its pool's for `status`, or null) and `closed`. The table of requests, `status`,
  `memory`, the request's reading and its constraint are the shared file's. Where the two models' replies
  differ in what a client sees, each keeps its own in its file, named in its header: do not make one of them
  without the owner. Its `serve` command takes what both share from `server/command.zig`.

## What still waits for an interface

The session and the pool (`qwen4/session/`, `pool/`; `glm5/session/`, `pool/`) are each model's and
are not shared: `served.zig` reaches them through the model's `reply` alone. What both sessions do alike is
`decode/`'s (spec 112): where a prompt keeps its states and which held state goes (`marks.zig`'s `Planner`,
Flash Next's since spec 054 and GLM's since WS-06, each with its options, and `thinned`; `keptOf`, the state
kept for a prefix other conversations share), and the states held with the buffers of those dropped lent to
the next taken (`shelf.zig`: Flash Next's two ranks' `Shelf` and its lot's spares, GLM's marks), and the
account of the positions held, where a prompt goes on from and a reply's states (`book.zig`'s `Book`: Flash Next's
two ranks' own, read from the states held by Flash Next's one rank and by GLM, whose rules are its options); what a state
is and how it is copied stay the model's. GLM takes a planned mark inside the window that holds it by replay
(`kda.Mark.takeWithin`), the window not cut; Flash Next's snapshot is taken inside its window as its own. GLM's snapshot is the KDA
state and conv window at one position, the latent cache cut there (spec 046). Spec 040 still draws the
shared boundary as a session generic over the same namespace with a `Snapshot` and a `Head`. `Decoder`
never looks inside a state; `within` and `kept` are where a session keeps one, so that boundary needs no
change to it.
