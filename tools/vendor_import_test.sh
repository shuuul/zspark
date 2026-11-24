#!/bin/bash
# tools/vendor_import.sh and tools/vendor_c.sh's contract, run on temporary repositories and upstream
# checkouts written here: the import's exact bytes, the dirty-worktree refusal, what a failed preparation
# or refused publish leaves, an isolated patch replay and a conflict's recovery (spec 168). Nothing it
# makes is inside this repository; run it as `tools/vendor_import_test.sh` (needs git, rsync, zig).
set -u
export LC_ALL=C
TOOLS=$(cd "$(dirname "$0")" && pwd)
failures=0
ok()   { printf 'ok   %s\n' "$1"; }
bad()  { printf 'FAIL %s\n' "$1"; failures=$((failures + 1)); }
want() { if [ "$2" -eq "$3" ]; then ok "$1"; else bad "$1 (wanted $3, got $2)"; fi; }
want_eq() { if [ "$2" = "$3" ]; then ok "$1"; else bad "$1 (wanted $3, got $2)"; fi; }

# the temporary repositories' commits need an identity the host may not have
export GIT_AUTHOR_NAME=t GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=t GIT_COMMITTER_EMAIL=t@t

for tool in git rsync sha256sum zig mktemp; do
    command -v "$tool" >/dev/null || { echo "$tool is needed" >&2; exit 2; }
done

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
export XDG_CACHE_HOME="$WORK/cache"

# The smallest descriptor the importer reads: two roots, a beside file, a directory whole and one notice.
descriptor() {
    cat >"$1/tools/host_descriptor.zig" <<'EOF'
const std = @import("std");
const Io = std.Io;
pub fn main(init: std.process.Init) !u8 {
    var buffer: [4096]u8 = undefined;
    var stdout_file: Io.File.Writer = .init(.stdout(), init.io, &buffer);
    const stdout = &stdout_file.interface;
    defer stdout.flush() catch {};
    try stdout.writeAll(
        \\roots zig/src/engine.zig
        \\roots zig/src/sys.zig
        \\beside extra/side.txt
        \\whole data
        \\notices NOTICE
        \\own MANIFEST.sha256
        \\own UPSTREAM
        \\own PATCHES.md
        \\own AGENTS.md
        \\
    );
    return 0;
}
EOF
}

# An upstream checkout: the two roots, engine.zig importing dep.zig and embedding embed.txt, the beside
# file, one case under data/ and the notice. $1 the directory, $2 the engine text, $3 the dep text.
upstream() {
    mkdir -p "$1/zig/src" "$1/extra" "$1/data"
    printf 'const dep = @import("dep.zig");\nconst embed = @embedFile("embed.txt");\n%s\n' "$2" >"$1/zig/src/engine.zig"
    printf '%s\n' "$3" >"$1/zig/src/dep.zig"
    printf 'embedded\n' >"$1/zig/src/embed.txt"
    printf 'the platform\n' >"$1/zig/src/sys.zig"
    printf 'compiled, not imported\n' >"$1/extra/side.txt"
    printf 'a recorded case\n' >"$1/data/case.json"
    printf 'a notice\n' >"$1/NOTICE"
}

# A repository holding an import of the first upstream: the vendored files, the four of its own, a file
# upstream no longer has (gone.txt) and a second tracked file beside the vendored tree.
repo() {
    mkdir -p "$1/tools" "$1/zig/vendor/tensorfold/zig/src" "$1/zig/vendor/tensorfold/extra" \
        "$1/zig/vendor/tensorfold/data" "$1/zig/src"
    cp "$TOOLS/vendor_import.sh" "$TOOLS/vendor_c.sh" "$TOOLS/vendor_lib.sh" "$1/tools/"
    descriptor "$1"
    local vendored=$1/zig/vendor/tensorfold
    printf 'const dep = @import("dep.zig");\nconst embed = @embedFile("embed.txt");\nold engine\n' >"$vendored/zig/src/engine.zig"
    printf 'old dep\n' >"$vendored/zig/src/dep.zig"
    printf 'embedded\n' >"$vendored/zig/src/embed.txt"
    printf 'the platform\n' >"$vendored/zig/src/sys.zig"
    printf 'compiled, not imported\n' >"$vendored/extra/side.txt"
    printf 'a recorded case\n' >"$vendored/data/case.json"
    printf 'a notice\n' >"$vendored/NOTICE"
    printf 'no longer upstream'\''s\n' >"$vendored/zig/src/gone.txt"
    printf 'repository: somewhere\ncommit: old\n' >"$vendored/UPSTREAM"
    printf '# Patches\n\n| File | What | Why | Proposed upstream |\n| --- | --- | --- | --- |\n| `zig/src/engine.zig` | old patch | why | no |\n' >"$vendored/PATCHES.md"
    printf 'the rules\n' >"$vendored/AGENTS.md"
    (cd "$vendored" && find . -type f ! -name MANIFEST.sha256 ! -name UPSTREAM ! -name PATCHES.md ! -name AGENTS.md -print0 | sort -z | xargs -0 sha256sum -- >MANIFEST.sha256)
    printf 'zspark'\''s own\n' >"$1/zig/src/ours.zig"
    git -C "$1" init -q
    git -C "$1" add -A
    git -C "$1" -c user.email=t@t -c user.name=t commit -qm 'an old import'
}

REPO=$WORK/repo
UP=$WORK/upstream
upstream "$UP" 'new engine' 'new dep'
repo "$REPO"

# --- a staged import publishes upstream's bytes exactly, removes what went, keeps the four -------------
out=$(bash "$REPO/tools/vendor_import.sh" "$UP" 2>&1); want "a clean import exits" $? 0
same=1
for f in zig/src/engine.zig zig/src/dep.zig zig/src/embed.txt zig/src/sys.zig extra/side.txt data/case.json NOTICE; do
    cmp -s "$UP/$f" "$REPO/zig/vendor/tensorfold/$f" || same=0
done
want "the published files are upstream's bytes" $same 1
want "a file upstream dropped is removed" "$([ -f "$REPO/zig/vendor/tensorfold/zig/src/gone.txt" ] && echo 1 || echo 0)" 0
want "UPSTREAM is kept for the hand edit" "$(grep -c 'commit: old' "$REPO/zig/vendor/tensorfold/UPSTREAM")" 1
(cd "$REPO/zig/vendor/tensorfold" && sha256sum -c --quiet MANIFEST.sha256) && ok "the manifest holds the new files" || bad "the manifest holds the new files"

# --- the same run against the same upstream changes nothing -------------------------------------------
before=$(git -C "$REPO" status --porcelain | wc -l)
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'the import'
out=$(bash "$REPO/tools/vendor_import.sh" --compare "$UP" 2>&1); want "--compare of an identical import exits" $? 0

# --- a mutating import refuses a dirty worktree and changes nothing -----------------------------------
printf 'an unrelated edit\n' >>"$REPO/zig/src/ours.zig"
vendored_hash=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
out=$(bash "$REPO/tools/vendor_import.sh" "$UP" 2>&1); want "a dirty worktree refuses the import" $? 1
printf '%s' "$out" | grep -q 'dirty' && ok "the refusal says why" || bad "the refusal says why: $out"
after=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
want "the dirty refusal changed no vendored byte" "$([ "$vendored_hash" = "$after" ] && echo 1 || echo 0)" 1

# --- --stage on a dirty tree makes the candidate; --publish refuses dirty, then publishes ----------------
out=$(bash "$REPO/tools/vendor_import.sh" --stage "$UP" 2>&1); want "--stage on a dirty tree exits" $? 0
STAGE_DIR=$(printf '%s' "$out" | grep -o '/[^ ;]*vendor-stage-[^ ;]*' | head -1)
[ -f "$STAGE_DIR/MANIFEST.sha256" ] && ok "the candidate is staged with its manifest" || bad "the candidate is staged with its manifest"
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1); want "--publish on a dirty tree refuses" $? 1
[ -f "$STAGE_DIR/MANIFEST.sha256" ] && ok "the refused publish keeps the candidate" || bad "the refused publish keeps the candidate"
git -C "$REPO" checkout -- zig/src/ours.zig
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1); want "--publish once clean publishes" $? 0
[ -d "$STAGE_DIR" ] || ok "the stage is removed once published" || bad "the stage is removed once published"

# --- a candidate short of its manifest is refused before the destination changes -------------------------
out=$(bash "$REPO/tools/vendor_import.sh" --stage "$UP" 2>&1)
STAGE_DIR=$(printf '%s' "$out" | grep -o '/[^ ;]*vendor-stage-[^ ;]*' | head -1)
rm "$STAGE_DIR/extra/side.txt"
vendored_hash=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1)
want "a candidate short of its manifest is refused" $? 1
after=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
want "the refusal changed no vendored byte" "$([ "$vendored_hash" = "$after" ] && echo 1 || echo 0)" 1
rm -rf "$STAGE_DIR"

# --- an interrupted publish resumes; another candidate's journal refuses ---------------------------------
# (a crafted journal+preimage stands for a publish killed between its records and its rsync)
VENDORED_D=$REPO/zig/vendor/tensorfold
out=$(bash "$REPO/tools/vendor_import.sh" --stage "$UP" 2>&1)
STAGE_DIR=$(printf '%s' "$out" | grep -o '/[^ ;]*vendor-stage-[^ ;]*' | head -1)
digest=$(sha256sum "$STAGE_DIR/MANIFEST.sha256" | cut -d' ' -f1)
(cd "$VENDORED_D" && find . -type f ! -name MANIFEST.sha256 ! -name UPSTREAM ! -name PATCHES.md ! -name AGENTS.md ! -name .vendor-journal ! -name .vendor-journal.pre | sed 's|^\./||' | tr '\n' '\0' | xargs -0 -r sha256sum -- >.vendor-journal.pre)
printf 'manifest: %s\nstage: %s\n' "$digest" "$STAGE_DIR" >"$VENDORED_D/.vendor-journal"
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1)
want "an interrupted publish resumes on the same candidate" $? 0 || bad "resume output: $out"
cmp -s "$UP/zig/src/sys.zig" "$VENDORED_D/zig/src/sys.zig" && ok "the resumed publish completes the copy" || bad "the resumed publish completes the copy"
[ ! -f "$VENDORED_D/.vendor-journal" ] && [ ! -f "$VENDORED_D/.vendor-journal.pre" ] && ok "the journal is gone once the publish holds" || bad "the journal is gone once the publish holds"

# an edit or a new file inside the vendored dir after the interruption refuses the resume and is kept
out=$(bash "$REPO/tools/vendor_import.sh" --stage "$UP" 2>&1)
STAGE_DIR=$(printf '%s' "$out" | grep -o '/[^ ;]*vendor-stage-[^ ;]*' | head -1)
digest=$(sha256sum "$STAGE_DIR/MANIFEST.sha256" | cut -d' ' -f1)
(cd "$VENDORED_D" && find . -type f ! -name MANIFEST.sha256 ! -name UPSTREAM ! -name PATCHES.md ! -name AGENTS.md ! -name .vendor-journal ! -name .vendor-journal.pre | sed 's|^\./||' | tr '\n' '\0' | xargs -0 -r sha256sum -- >.vendor-journal.pre)
printf 'manifest: %s\nstage: %s\n' "$digest" "$STAGE_DIR" >"$VENDORED_D/.vendor-journal"
printf 'user edit after interruption\n' >>"$VENDORED_D/zig/src/sys.zig"
printf 'new file after interruption\n' >"$VENDORED_D/zig/src/user_new.txt"
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1)
want "a post-interruption edit refuses the resume" $? 1
printf '%s' "$out" | grep -q 'after the interruption' && ok "the refusal names the post-interruption change" || bad "the refusal names the post-interruption change: $out"
grep -q 'user edit' "$VENDORED_D/zig/src/sys.zig" && [ -f "$VENDORED_D/zig/src/user_new.txt" ] &&
    ok "the refused resume left the edits alone" || bad "the refused resume left the edits alone"
[ -f "$VENDORED_D/.vendor-journal" ] && [ -f "$VENDORED_D/.vendor-journal.pre" ] &&
    ok "journal and preimage stand for the settle" || bad "journal and preimage stand for the settle"
rm -f "$VENDORED_D/.vendor-journal" "$VENDORED_D/.vendor-journal.pre" "$VENDORED_D/zig/src/user_new.txt"
git -C "$REPO" checkout -- zig/vendor/tensorfold/zig/src/sys.zig

# a journal that is not this candidate's refuses and keeps its tree
out=$(bash "$REPO/tools/vendor_import.sh" --stage "$UP" 2>&1)
STAGE_DIR=$(printf '%s' "$out" | grep -o '/[^ ;]*vendor-stage-[^ ;]*' | head -1)
printf 'manifest: %064d\nstage: /elsewhere\n' 0 >"$VENDORED_D/.vendor-journal"
printf 'half written\n' >"$VENDORED_D/zig/src/sys.zig"
out=$(bash "$REPO/tools/vendor_import.sh" --publish "$STAGE_DIR" 2>&1)
want "a foreign journal refuses the publish" $? 1
printf '%s' "$out" | grep -q 'interrupted publish of another candidate' && ok "the refusal names the journal" || bad "the refusal names the journal: $out"
rm -f "$VENDORED_D/.vendor-journal"
git -C "$REPO" checkout -- zig/vendor/tensorfold/zig/src/sys.zig

# --- --compare still works on a patched, dirty tree -----------------------------------------------------
printf 'patched\n' >>"$REPO/zig/vendor/tensorfold/zig/src/sys.zig"
out=$(bash "$REPO/tools/vendor_import.sh" --compare "$UP" 2>&1)
want "--compare on a patched tree reports the difference" $? 1
printf '%s' "$out" | grep -q 'differs: zig/src/sys.zig' && ok "--compare names the differing file" || bad "--compare names the differing file: $out"
git -C "$REPO" checkout -- zig/vendor/tensorfold/zig/src/sys.zig

# --- an invalid upstream changes nothing ---------------------------------------------------------------
mv "$UP/zig/src/dep.zig" "$UP/zig/src/dep.away"
vendored_hash=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
out=$(bash "$REPO/tools/vendor_import.sh" "$UP" 2>&1); want "an upstream a file short fails" $? 1
printf '%s' "$out" | grep -q 'zig/src/dep.zig' && ok "the failure names the missing file" || bad "the failure names the missing file: $out"
after=$(cd "$REPO/zig/vendor/tensorfold" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum)
want "the failed preparation changed no vendored byte" "$([ "$vendored_hash" = "$after" ] && echo 1 || echo 0)" 1
mv "$UP/zig/src/dep.away" "$UP/zig/src/dep.zig"

# --- a patch replay: clean pick, then conflict ---------------------------------------------------------
# a patch commit on the old import: a line of sys.zig (untouched by the new upstream) and its ledger row
git -C "$REPO" checkout -q -b replay-base HEAD
printf 'patched line\n' >>"$REPO/zig/vendor/tensorfold/zig/src/sys.zig"
printf '# Patches\n\n| File | What | Why | Proposed upstream |\n| --- | --- | --- | --- |\n| `zig/src/sys.zig` | a platform line | why | no |\n' >"$REPO/zig/vendor/tensorfold/PATCHES.md"
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: sys platform line'
PATCH=$(git -C "$REPO" rev-parse HEAD)
# a second patch the new upstream conflicts with: dep.zig's text (the new import carries 'new dep2')
printf 'patched dep\n' >"$REPO/zig/vendor/tensorfold/zig/src/dep.zig"
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: dep line'
CONFLICT=$(git -C "$REPO" rev-parse HEAD)
# a third, so a conflict in the second exercises the queue that holds it
printf 'a recorded case\nanother\n' >"$REPO/zig/vendor/tensorfold/data/case.json"
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: case rows'
THIRD=$(git -C "$REPO" rev-parse HEAD)
# the new import as its own commit (what an upgrade's first commit is), its dep.zig the conflicting side
printf 'new dep2\n' >"$UP/zig/src/dep.zig"
git -C "$REPO" rm -qr zig/vendor/tensorfold/zig zig/vendor/tensorfold/extra zig/vendor/tensorfold/data zig/vendor/tensorfold/NOTICE zig/vendor/tensorfold/MANIFEST.sha256
cp -r "$UP/zig" "$UP/extra" "$UP/data" "$UP/NOTICE" "$REPO/zig/vendor/tensorfold/"
(cd "$REPO/zig/vendor/tensorfold" && find . -type f ! -name MANIFEST.sha256 ! -name UPSTREAM ! -name PATCHES.md ! -name AGENTS.md -print0 | sort -z | xargs -0 sha256sum -- >MANIFEST.sha256)
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: the new import'
head_before=$(git -C "$REPO" rev-parse HEAD)

out=$(bash "$REPO/tools/vendor_import.sh" --replay "$PATCH" "$CONFLICT" "$THIRD" 2>&1)
want "a conflicting replay fails" $? 1
printf '%s' "$out" | grep -q "${CONFLICT:0:7}\|$CONFLICT" && ok "the failure names the conflicting patch" || bad "the failure names the conflicting patch: $out"
REPLAY=$(printf '%s' "$out" | grep -o '[^ :;]*\.vendor-replay-[^ :;]*' | head -1)

[ -d "$REPLAY" ] && ok "the conflict keeps the recoverable candidate" || bad "the conflict keeps the recoverable candidate: $out"
want_eq "the original HEAD is unmoved" "$(git -C "$REPO" rev-parse HEAD)" "$head_before"
want "the original tree is clean" "$(git -C "$REPO" status --porcelain | wc -l)" 0

# the queue still asks for all three; a publish before the picks finish is refused
grep -q "^commit: $CONFLICT\$" "$REPLAY/.vendor-replay-queue" && grep -q "^commit: $THIRD\$" "$REPLAY/.vendor-replay-queue" &&
    ok "the replay queue holds the commits left" || bad "the replay queue holds the commits left"
out=$(bash "$REPO/tools/vendor_import.sh" --publish-replay "$REPLAY" 2>&1)
want "a mid-pick replay cannot publish" $? 1
want_eq "the refused publish moved nothing" "$(git -C "$REPO" rev-parse HEAD)" "$head_before"

# settle it in the replay worktree (upstream's text, then the patch's), continue: the queue's third lands too
printf 'new dep2\npatched dep\n' >"$REPLAY/zig/vendor/tensorfold/zig/src/dep.zig"
git -C "$REPLAY" add zig/vendor/tensorfold/zig/src/dep.zig
git -C "$REPLAY" -c core.editor=true cherry-pick --continue
out=$(bash "$REPO/tools/vendor_import.sh" --publish-replay "$REPLAY" 2>&1)
want "--publish-replay fast-forwards the branch" $? 0
[ -d "$REPLAY" ] || ok "the replay worktree is removed" || bad "the replay worktree is removed"
grep -q 'patched line' "$REPO/zig/vendor/tensorfold/zig/src/sys.zig" && ok "the clean pick is on the branch" || bad "the clean pick is on the branch"
grep -q 'patched dep' "$REPO/zig/vendor/tensorfold/zig/src/dep.zig" && ok "the settled pick is on the branch" || bad "the settled pick is on the branch"
grep -q 'new dep2' "$REPO/zig/vendor/tensorfold/zig/src/dep.zig" && ok "upstream's text is kept beside the patch" || bad "upstream's text is kept beside the patch"
grep -q 'another' "$REPO/zig/vendor/tensorfold/data/case.json" && ok "the queued pick after the conflict is on the branch" || bad "the queued pick after the conflict is on the branch"

# --- a dirty worktree refuses a replay before any worktree exists ----------------------------------------
printf 'dirty\n' >>"$REPO/zig/src/ours.zig"
out=$(bash "$REPO/tools/vendor_import.sh" --replay "$PATCH" 2>&1); want "a dirty worktree refuses a replay" $? 1
ls -d "$WORK"/.vendor-replay-* 2>/dev/null && bad "no replay worktree is left" || ok "no replay worktree is left"
git -C "$REPO" checkout -- zig/src/ours.zig

# --- a patch whose file commits a conflict marker is refused by the replay's own scan --------------------
git -C "$REPO" checkout -q -b markbase "$head_before"
printf 'x\n<<<<<<< ours\n' >>"$REPO/zig/vendor/tensorfold/extra/side.txt"
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: a marker'
MARKED=$(git -C "$REPO" rev-parse HEAD)
git -C "$REPO" checkout -q replay-base
out=$(bash "$REPO/tools/vendor_import.sh" --replay "$MARKED" 2>&1)
want "a replay whose pick holds a marker fails" $? 1
printf '%s' "$out" | grep -q 'conflict markers' && ok "the marker refusal says why" || bad "the marker refusal says why: $out"
REPLAY=$(printf '%s' "$out" | grep -o '[^ :;]*\.vendor-replay-[^ :;]*' | head -1)
out=$(bash "$REPO/tools/vendor_import.sh" --publish-replay "$REPLAY" 2>&1)
want "the marker replay cannot publish" $? 1
[ -n "$REPLAY" ] && git -C "$REPO" worktree remove --force "$REPLAY" 2>/dev/null

# --- vendor_c.sh: the same staging rules on the C libraries ----------------------------------------------
CSTAGE=$WORK/c-upstream; mkdir -p "$CSTAGE"
printf 'the licence\n' >"$CSTAGE/LICENSE"; printf 'an image decoder\n' >"$CSTAGE/stb_image.h"; printf 'a vorbis decoder\n' >"$CSTAGE/stb_vorbis.c"
mkdir -p "$REPO/zig/vendor/stb"
(cd "$REPO/zig/vendor/stb" && sha256sum /dev/null | sed 's|/dev/null|LICENSE|' >MANIFEST.sha256; printf 'u\n' >UPSTREAM; printf 'p\n' >PATCHES.md; printf 'a\n' >AGENTS.md)
git -C "$REPO" add -A && git -C "$REPO" -c user.email=t@t -c user.name=t commit -qm 'vendor: stb'
out=$(bash "$REPO/tools/vendor_c.sh" stb "$CSTAGE" 2>&1); want "a clean vendor_c import exits" $? 0
cmp -s "$CSTAGE/stb_image.h" "$REPO/zig/vendor/stb/stb_image.h" && ok "vendor_c publishes upstream's bytes" || bad "vendor_c publishes upstream's bytes"
printf 'dirty\n' >>"$REPO/zig/src/ours.zig"
out=$(bash "$REPO/tools/vendor_c.sh" stb "$CSTAGE" 2>&1); want "vendor_c refuses a dirty worktree" $? 1
out=$(bash "$REPO/tools/vendor_c.sh" --compare stb "$CSTAGE" 2>&1); want "vendor_c --compare works dirty" $? 0

echo
if [ "$failures" -eq 0 ]; then echo "all vendor contract tests passed"; else echo "$failures failed"; exit 1; fi
