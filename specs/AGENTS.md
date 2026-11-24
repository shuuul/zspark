# `specs/` instructions

This file extends the root `AGENTS.md`. `README.md` here has the index and the full rules; this is the short form
for whoever is about to edit a spec. `WORKER.md` holds a worker's standing rules (locks, servers, tiers, the
measurement budget, commits, the report); a brief points to it.

## Local commands

- `zig build check-specs` before and after any edit here; `zig build test-specs` after touching the validator.

## When editing here

- Do: append your own dated entry under "Progress and handoff" (changed, evidence, remaining, blockers, next
  action) and set your workstream row's status. An entry is about 40 lines plus its tables: final figures
  with their command and commit, what was tried and refused and why. Raw output goes under
  `~/.cache/zspark/scratch-NNN/` and is named, not pasted.
- Do: read a spec by its Decisions, your row and the latest entries; use the archive index to find completed work.
- Do: split a spec before it passes 60,000 bytes (`check-specs` refuses it): the rest goes to a new ID. A subagent's entry is written by the coordinator from its report
  and says so.
- Do: add a decision to the Decisions table on the day it is taken, with its rationale.
- Do: keep frontmatter flat `key: value`, bump `updated`, and keep `README.md`'s index row in step.
- Don't: rewrite another worker's entry, renumber or delete a spec, or add a fourth status.
- Don't: paste a log, keep a diary of the order of work, or restate `WORKER.md` in a spec or a brief.
- Don't: leave lasting rules only here. Move them to the nearest `AGENTS.md` or the code and note it under
  "Documentation sync".

## Briefs and sessions

- Do, a brief: name the spec's row, the files and line ranges the task needs and the commands that check it; a
  report of at most 25 lines, raw output in `~/.cache/zspark/scratch-NNN/`. A worker opens what the brief names,
  not a directory.
- Do, reading: `grep -n` to find, then `sed -n A,Bp` of the range; never `cat` a source, a script or a log whole;
  a long output to a file, then its tail or a `grep` of it. Independent reads and checks go in one call.
- Do, a session: a coordinator past about 150 k tokens of context writes the state into the spec's entry and
  hands off to a fresh session; work that needs only a conclusion goes to a subagent, whose context is its own.
