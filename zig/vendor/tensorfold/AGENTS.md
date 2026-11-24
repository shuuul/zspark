# `zig/vendor/tensorfold/` instructions

This file extends the root `AGENTS.md`. TensorFold's host, kept in this tree since spec 072: the files
`zspark serve` is built from (`build.zig`'s `Host`: the modules `engine_api`, `json`, `lanes`, `living`,
`picture`, `server`, `template`, `tokenizer`, and `sys`, `save_guard`, `checkpoint_config`, `profile`,
`bf16_round`, `nemotron_limits` handed to the ones that name them),
`test-golden` (upstream's checker, its scripted engine, 337 recorded replies and 69 security cases) and
`zig build grammar`'s `libtfgrammar` (`zig/src/grammar/`'s `tf_grammar.cc` and `tf_budget.cc` over an
xgrammar checkout), at upstream's
paths. `UPSTREAM` names the commit and how the list of files was derived; `MANIFEST.sha256` is that import,
a line a file; `PATCHES.md` is every difference from it. Everything else here is upstream's, Apache-2.0, with
its `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md` and `LICENSES/`.

The one definition of what is taken and how the build wires it is `tools/host_descriptor.zig` (spec 168):
each module's name, root, imports and stub, the five shared names, the files beside the closure, the
directories whole and the notices. `build.zig`'s `Host` builds from it and `tools/vendor_import.sh` takes
from it; a root that moves is changed there and nowhere else.

## Rules

- **An import is upstream's bytes and nothing else**, in a commit of its own: no fix, no formatting, no file
  that upstream does not have. `MANIFEST.sha256` is written by `tools/vendor_import.sh` in that commit and
  never by hand or by anything else: it is what the check compares with.
- **A local change is one commit with one row in `PATCHES.md`** (the file, what, why, whether it was proposed
  upstream), and its test. A file added here or an upstream file removed is a change and has its row.
  `zig build check-specs` hashes every file against the manifest: a file that differs without a row fails, and
  so does a row whose file is upstream's again.
- Mark a changed place with a comment that opens `// zspark:` and says why, so that it is found again when
  upstream's file has moved under it. Keep a patch small and where upstream would take it; propose it
  upstream when it is not zspark's alone, and say so in the row.
- **This tree is out of a search's default** (`.ignore`, spec 154 WS-03): it is upstream's bytes, and a
  behaviour is changed through `PATCHES.md`. Read a line with `git grep -n TERM -- 'zig/vendor/tensorfold/**'`
  or `rg --no-ignore`, and start from the patch row rather than the file.
- **`zig fmt` is never run here** (five of upstream's files are not as it would write them):
  `zig fmt --exclude zig/vendor zig build.zig tools`. The 60,000 bytes of a source do not hold here either.
- A recorded reply (`zig/tests/server/golden/`) is never edited. A patch that changes one on purpose names
  the case in `golden_check.zig`'s `known_diffs` (a patched file, with its row) and in its commit.
- Only what the build compiles is here. A file a patch newly imports comes from the same upstream commit, as
  it is, through the import script (its closure follows the imports), not by a copy of one file.
- What a client sees is still decided here, in upstream's host; `zig/src/server/` stays zspark's engine
  behind `engine_api.Engine`. A change of what a client sees goes to the owner first (`specs/WORKER.md`, 10).

## Local commands

```bash
zig build test-golden                           # HTTP and security cases (no GPU)
zig build check-specs                           # the manifest against the files, the rows of PATCHES.md (a row repeated exactly and a patched file's unresolved conflict marker are refused too)
tools/vendor_import.sh --compare UPSTREAM_DIR   # the files here against a checkout of upstream, byte for byte (a dirty or patched tree is fine)
tools/vendor_import.sh --stage UPSTREAM_DIR     # the candidate made and checked outside this tree, kept under $XDG_CACHE_HOME/zspark/vendor-stage-*
tools/vendor_import.sh --publish STAGE_DIR      # a staged candidate published (a dirty worktree is refused first)
tools/vendor_import.sh --replay COMMIT...       # local patch commits replayed onto HEAD in a worktree beside this repository
tools/vendor_import.sh --publish-replay DIR     # a finished replay fast-forwarded onto this branch
(cd UPSTREAM_DIR && sha256sum -c --quiet "$OLDPWD/zig/vendor/tensorfold/MANIFEST.sha256")   # the manifest against it
```

The result is the line `N/total equal to server goldens, M unexpected differences`, a plan a run
(`grep 'equal to server goldens'`; up to 1.0.12 `equal to the frozen Python answers`, and up to 1.0.4 the two
programs printed their debug allocators' leaks around it), and the known ones are the table's last column summed. `known_diffs` in `golden_check.zig` is
upstream's own list; a build that fails prints no such line, so an empty `grep` is a failure.

## Taking in a release

Imports so far (spec 072 WS-05: 1.0.2 to 1.0.4, then 1.0.5; spec 106: 1.0.12; spec 113: 1.1.1;
spec 141: 1.1.4; spec 145: 1.1.5; spec 152: 1.2.1; spec 167: 1.2.7; spec 174: 1.2.8; spec 175: 1.3.2; spec 176: 1.3.3; spec 177: 1.3.4).

1. Get TensorFold at the tag to take outside this tree (a clone under `~/.cache/zspark/upstream/`; no lock).
   Read what changed in the paths taken and why, before anything is copied:
   `git -C CLONE log OLD..NEW -- zig/src/core zig/src/server zig/tests/server`, its `CHANGELOG.md`, and
   `tools/vendor_import.sh --compare CLONE` on the tree as it is, which lists the files that will change and
   come. Time what the brief asks before the import (`bench-held-turn`): afterwards the old host is gone.
   Read upstream's engine too (`zig/src/families/` of the models zspark serves, `zig/kernels`, on CUDA): its
   gains are not imported, so list what zspark lacks, by expected gain, for ports of their own (2026-10-25).
2. List the patches to carry, oldest first: the commits since the manifest was last written,
   `git log --reverse --oneline $(git log -1 --format=%H -- zig/vendor/tensorfold/MANIFEST.sha256)..HEAD -- zig/vendor/tensorfold`,
   against the rows of `PATCHES.md`; keep the rows' text aside. For each, read upstream's new file: a patch
   upstream has taken is dropped, and its row with it.
3. `tools/vendor_import.sh UPSTREAM_DIR`: the candidate is staged and checked outside this tree, then
   published — every file here but this directory's own four is replaced by upstream's, and the manifest is
   written. It refuses a dirty worktree first; stage alone with `--stage` and publish with `--publish` if a
   step is interrupted. An interrupted publish's journal and preimage records stand at the top until a
   resume of the same candidate holds or they are removed by hand; a file changed since the interruption
   refuses the resume. Files come and gone: the old manifest's paths against the new one's
   (`cut -c67- MANIFEST.sha256 | sort`, `comm`). If a root or a notice moved, the script says so:
   `tools/host_descriptor.zig` changes, and `build.zig`'s `Host` moves with it (the old split kept the two by
   hand: 1.0.12 `json`, a sixth module; 1.1.1 `sys`, `save_guard`, `checkpoint_config` for every module and
   `living` for the server; 1.1.4 `dsv41_chat` a root and `drama_prompt` a stub; 1.2.3 `profile` and
   `bf16_round`; 1.2.7 `nemotron_limits`, which the 167 import missed until `Host` named it, and
   `startup_receipt`, kept one file by making the `core` stub re-export `profile`'s rather than a second
   module's root). A by-name import that nothing analysed calls stays unresolved (1.1.4: `native_engines`,
   `build_options`) and `UPSTREAM` names it.
   A file the build compiles that no Zig file imports is a `beside` of the descriptor (1.0.12:
   `tf_grammar.cc`, 1.1.8's `tf_budget.h` and `tf_budget.cc` — their closure is C++, which the walk does not
   follow — built by `zig build grammar`).
4. Edit `UPSTREAM` (commit, version, date, the counts the script printed, what came, went and was left) and
   empty `PATCHES.md`'s table: the patched files are upstream's again, and `check-specs` refuses a row whose
   file is. `zig build check-specs`, `zig build test-golden` (upstream's own: it builds although zspark may
   not compile here), then commit the import, alone, with the counts. The hook fails silently: `git log`.
5. `tools/vendor_import.sh --replay COMMIT...` replays each patch, in order and a commit a patch, in a
   worktree made beside this repository: the tree here is not touched, a pick's hooks run, and a pick that
   conflicts stops on it with the worktree kept. Settle the conflict there against upstream's new text and
   never by taking the old file (`git -C REPLAY cherry-pick --continue`), then
   `tools/vendor_import.sh --publish-replay REPLAY_DIR` fast-forwards this branch. `check-specs` refuses a
   patched file that still holds a conflict marker and a row that repeats another exactly. A patch's first
   commit holds zspark's side too and may hold two patches (1.0.4): split it in the replay worktree — pick
   with `--no-commit`, unstage and commit file by file. The commit names the commit it was picked from
   (`-x` does). `zig build test-golden` after the patches, and `tools/vendor_import.sh --compare`: exactly
   the patched files differ. A patch's own test runs after step 6 where it needs zspark compiled. An oracle
   before the replay (1.1.5): `diff3 -m` of each patched file, the old upstream's and the new one's, outside
   the tree; after it each patched file is byte for byte its merge with the conflicts settled, and
   `PATCHES.md` the last pass's.
6. What zspark owes the new host: `tools/check.sh compile` says what no longer compiles, and what compiles
   and means something else is found by reading `engine_api.zig`'s diff (a new field of `Info`, `Request` or
   the vtable is left at its default unless the host fails without it, and listed in `served.zig`'s header)
   and `cli.zig`'s (a flag newly the host's that zspark does not serve: `command.zig` `unserved`), and
   `model_text.zig`'s (`segments.zig` and the bench hand every entry of the text on; 1.1.4: `encode_masked`, which
   a chat fails closed without, and `added`). Then the
   version in root `NOTICE`, `README.md`, `AGENTS.md`, `build.zig`, here; `tools/check.sh host` and `server`
   (1.1.1: a test's raw request names a loopback `Host`, as the guard asks). A reply that
   a client sees differently is named to the owner before the merge.
