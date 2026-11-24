# Project specs

Specs are tracked execution records: what a stretch of work set out to do, the decisions taken along the way, who
did which part, and the evidence it was done. A worker's standing rules are in `WORKER.md`. Copy [000-template.md](000-template.md) to start a spec.

## Active specs

| Spec | Status | Outcome |
| --- | --- | --- |

## The archive

Completed specs are listed in [archive/README.md](archive/README.md).

## Rules

**Lifecycle.** A spec has one of three statuses. `Draft` and `Active` specs live in `specs/`; a `Completed` spec
lives in `specs/archive/`. A `Draft` is not decision-complete; an `Active` spec is an execution contract its
workstreams may proceed on. Blocking is recorded on the workstream (`Blocked`, with the evidence, the decision
needed, the owner and the next action), never as a fourth spec status.

**Numbering.** IDs are three digits, continuous from `001` across `specs/` and `specs/archive/`. Take the next ID
from both directories. Never reuse, renumber or delete an ID. `000` is the template.

**Coordination.** The coordinator owns the frontmatter, the scope, cross-workstream decisions, the index row and
the closeout. Reserve the ID and add the Active index row before parallel work starts. Every workstream has a
stable ID; a worker claims its row before editing and appends handoff entries instead of rewriting another
worker's record.

**What does not belong here.** Specs record execution. Lasting behaviour, interfaces and maintenance decisions go
into code, tests, public docs and the nearest `AGENTS.md`; a spec's "Documentation sync" says where they went.

**Size.** A spec is at most 60,000 bytes; one that would pass it is split, the rest going to a new ID. A
workstream's entry under "Progress and handoff" is about 40 lines plus its tables: what changed, the evidence
(final figures with their command and commit), what was tried and refused and why, what remains. Raw output,
listings and intermediate figures go under `~/.cache/zspark/scratch-NNN/` and are named in the entry, not
pasted. An entry is not a diary: what a later entry supersedes is not kept in the earlier one's detail.

**Closeout.** Before archiving: cut "Progress and handoff" to what a later reader must not rediscover, at most
about 1,500 bytes, the summary holding the final figures; satisfy every success criterion or record the decision that removed it; move
lasting knowledge into its owning docs; record the final commands and evidence; complete the summary and set
`status: Completed`; move the file to `archive/` and its index row to `archive/README.md` in the same change. Run
`zig build check-specs` before and after.

**Validation.** `zig build check-specs` checks filenames, flat frontmatter, status by directory, the required
headings, ID continuity and both indexes, and the size budgets: a spec 60,000 bytes, root `AGENTS.md` 10,000, a
nested `AGENTS.md` 14,000, `WORKER.md` 7,000, and a tracked file under `src/`, `kernels/`, `tools/` or
`build.zig` 60,000 unless `tools/check_specs.zig` lists it as an exception, with its reason. Frontmatter is flat `key: value` only. `zig build test-specs` runs the
validator's own tests; CI runs both.
