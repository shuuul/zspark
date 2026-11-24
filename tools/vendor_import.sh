#!/bin/bash
# TensorFold's host taken from a checkout of TensorFold into zig/vendor/tensorfold/, as upstream has it
# (spec 072; staged since spec 168).
#
#   tools/vendor_import.sh UPSTREAM_DIR            stages the candidate, checks it, publishes it
#   tools/vendor_import.sh --stage UPSTREAM_DIR    only stages: the candidate stands under
#                                                  $XDG_CACHE_HOME/zspark/vendor-stage-* for a later publish
#   tools/vendor_import.sh --publish STAGE_DIR     publishes a staged candidate (after an interrupted one)
#   tools/vendor_import.sh --compare UPSTREAM_DIR  changes nothing: compares the files here with upstream's
#   tools/vendor_import.sh --replay COMMIT...      replays local patch commits onto HEAD in a worktree
#                                                  beside this repository; --publish-replay REPLAY_DIR
#                                                  fast-forwards this branch to a finished replay
#
# UPSTREAM_DIR is TensorFold at the commit to take, outside this tree (a clone, or a fetched package). The
# files taken are the build's: the roots, beside files, whole directories and notices of
# tools/host_descriptor.zig, the one list build.zig's Host and this script share — every file the module
# roots reach through @import and @embedFile of a relative path (an import by name is a module the
# descriptor names), the files the build compiles beside them, the recorded replies and their fixtures
# whole, and the notices.
#
# A mutating run refuses a dirty worktree before it changes a file, and the candidate is copied and checked
# against its new manifest outside the destination before it is published: a failed copy, an invalid
# upstream or an interrupted preparation leaves the tracked bytes and unrelated edits as they were, the
# staged candidate kept for a --publish where the failure was a publish's. A publish that fails or is
# interrupted midway leaves its journal and the destination's preimage record at the top: --publish of
# the same candidate resumes it, of another refuses, and a file under the destination that changed since
# the interruption — edit or addition — refuses the resume and is left alone. --compare works on a
# patched, dirty tree. The vendored directory's own four (MANIFEST.sha256, UPSTREAM, PATCHES.md,
# AGENTS.md) stand; a local patch is gone and is applied again as a
# commit --replay makes (zig/vendor/tensorfold/AGENTS.md has the steps). UPSTREAM and PATCHES.md are edited
# by hand afterwards. No lock, no network, no GPU; `--compare` exits 1 on a difference.
set -euo pipefail
export LC_ALL=C
NAME=tensorfold
. "$(dirname "$0")/vendor_lib.sh"
vendor_repo || exit 2
usage() { sed -n 4,16p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

COMPARE= STAGE= PUBLISH= REPLAY= PUBLISH_REPLAY=
while [ $# -gt 0 ]; do case $1 in
    --compare) COMPARE=1; shift ;;
    --stage) STAGE=1; shift ;;
    --publish) PUBLISH=1; shift ;;
    --replay) REPLAY=1; shift ;;
    --publish-replay) PUBLISH_REPLAY=1; shift ;;
    *) break ;;
esac; done

if [ -n "$PUBLISH" ]; then
    [ $# -eq 1 ] && [ -d "$1" ] || usage
    vendor_publish "$1" || exit 1
    echo "candidate $1 published; now UPSTREAM and PATCHES.md by hand"
    exit
fi

if [ -n "$REPLAY" ]; then
    [ $# -ge 1 ] || usage
    vendor_refuse_dirty || exit 1
    vendor_replay "$(git -C "$REPO" rev-parse HEAD)" "$@" || exit 1
    echo "replay finished; publish it with: $(basename "$0") --publish-replay REPLAY_DIR (named above)"
    exit
fi

if [ -n "$PUBLISH_REPLAY" ]; then
    [ $# -eq 1 ] && [ -d "$1" ] || usage
    vendor_publish_replay "$1" || exit 1
    echo "replayed patches are on $(git -C "$REPO" branch --show-current || echo HEAD)"
    exit
fi

[ $# -eq 1 ] && [ -d "$1" ] || usage
UPSTREAM=$(cd "$1" && pwd)
case $UPSTREAM/ in "$REPO"/zig/vendor/*) echo "UPSTREAM_DIR is the vendored directory itself" >&2; exit 2 ;; esac

# The lists are the descriptor's (`SECTION PATH` a line), run once, only where the upstream set is needed:
# --publish and the replays above work without zig.
command -v zig >/dev/null || { echo "the module list is tools/host_descriptor.zig's: zig is needed on PATH" >&2; exit 2; }
# the local cache outside the repository: a .zig-cache in it would make every worktree dirty
DESCRIPTOR=$(zig run --cache-dir "${XDG_CACHE_HOME:-$HOME/.cache}/zspark/zig-cache" "$REPO/tools/host_descriptor.zig" -- all)
section() { printf '%s\n' "$DESCRIPTOR" | sed -n "s/^$1 //p"; }
ROOTS=$(section roots)
BESIDE=$(section beside)
WHOLE=$(section whole)
NOTICES=$(section notices)
OWN=$(section own)

cd "$UPSTREAM"
for each in $ROOTS $NOTICES $BESIDE; do [ -f "$each" ] || { echo "$UPSTREAM has no $each: a root or a notice moved; see tools/host_descriptor.zig" >&2; exit 1; }; done
for each in $WHOLE; do [ -d "$each" ] || { echo "$UPSTREAM has no $each: a taken directory moved; see tools/host_descriptor.zig" >&2; exit 1; }; done

# The closure: a file's imports are read until a pass adds none. A target with no '.' is a module's name.
list=$(printf '%s\n' $ROOTS | sort -u)
while :; do
    found=$(printf '%s\n' "$list" | grep '\.zig$' | while IFS= read -r file; do
        { grep -o '@\(import\|embedFile\)([[:space:]]*"[^"]*"' "$file" || true; } | sed 's/^[^"]*"//; s/"$//' | { grep -F . || true; } |
            while IFS= read -r target; do realpath -m --relative-to=. "$(dirname "$file")/$target"; done
    done)
    next=$(printf '%s\n%s\n' "$list" "$found" | sed '/^$/d' | sort -u)
    [ "$next" = "$list" ] && break
    list=$next
done
list=$(printf '%s\n' "$list" $NOTICES $BESIDE "$(find $WHOLE -type f)" | sort -u)
while IFS= read -r file; do
    [ -f "$file" ] || { echo "$file is imported and is not in $UPSTREAM" >&2; exit 1; }
    case $file in ../* | /*) echo "$file is outside $UPSTREAM" >&2; exit 1 ;; *'\'* | *$'\r'*) echo "$file: a name sha256sum would escape" >&2; exit 1 ;; esac
done <<<"$list"

if [ -n "$COMPARE" ]; then
    differing=0
    while IFS= read -r file; do cmp -s "$file" "$VENDORED/$file" || { differing=$((differing + 1)); echo "differs: $file"; }; done <<<"$list"
    here=$(cd "$VENDORED" && find . -type f | sed 's|^\./||' | grep -vxF "$(printf '%s\n' $OWN)" | sort)
    extra=$(comm -13 <(printf '%s\n' "$list") <(printf '%s\n' "$here"))
    [ -z "$extra" ] || { differing=$((differing + $(printf '%s\n' "$extra" | wc -l))); printf '%s\n' "$extra" | sed 's/^/not upstream'"'"'s list: /'; }
    echo "$(printf '%s\n' "$list" | wc -l) files of upstream's list compared, $differing differing or beside it"
    [ "$differing" -eq 0 ]
    exit
fi

# --stage only makes the candidate, which a dirty tree allows; the mutating default refuses a dirty
# worktree before anything is staged (--publish and --replay refuse it above too).
if [ -n "$STAGE" ]; then
    stage=$(printf '%s\n' "$list" | vendor_stage) || exit 1
    echo "$(printf '%s\n' "$list" | wc -l) files staged at $stage; publish with: $(basename "$0") --publish $stage"
    exit
fi
vendor_refuse_dirty || exit 1
stage=$(printf '%s\n' "$list" | vendor_stage) || exit 1
vendor_publish "$stage" || exit 1
bytes=$(printf '%s\n' "$list" | tr '\n' '\0' | xargs -0 cat -- | wc -c)
echo "$(printf '%s\n' "$list" | wc -l) files, $bytes bytes from $UPSTREAM; now UPSTREAM and PATCHES.md by hand"
