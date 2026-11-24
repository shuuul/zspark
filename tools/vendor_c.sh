#!/bin/bash
# A C library's files taken from a checkout into zig/vendor/NAME/, as upstream has them (spec 019 WS-07):
# the image decoders' stb_image and libwebp's decoder, the audio decoders' dr_libs and stb_vorbis
# (spec 138 WS-02). TensorFold's host has its own script (tools/vendor_import.sh). Staged since spec 168.
#
#   tools/vendor_c.sh NAME UPSTREAM_DIR            stages the candidate, checks it, publishes it
#   tools/vendor_c.sh --stage NAME UPSTREAM_DIR    only stages: the candidate stands under
#                                                  $XDG_CACHE_HOME/zspark/vendor-stage-* for a later publish
#   tools/vendor_c.sh --publish NAME STAGE_DIR     publishes a staged candidate (after an interrupted one)
#   tools/vendor_c.sh --compare NAME UPSTREAM_DIR  changes nothing: compares the files here with upstream's
#   tools/vendor_c.sh --replay NAME COMMIT...      replays local patch commits onto HEAD in a worktree
#                                                  beside this repository; --publish-replay NAME REPLAY_DIR
#                                                  fast-forwards this branch to a finished replay
#
# NAME is stb, libwebp or dr_libs; UPSTREAM_DIR its checkout at the commit to take, outside this tree. The
# files taken are what the build compiles and nothing else: stb's stb_image.h and stb_vorbis.c and its
# licence; dr_libs' dr_wav.h, dr_flac.h and dr_mp3.h and its licence; libwebp's decoder (the makefile's
# DEC_OBJS, DSP_DEC_OBJS and UTILS_DEC_OBJS) for aarch64, without the MIPS, MSA, SSE and AVX files, which
# compile to nothing here, then every header those sources include as `zig cc -MM` lists them, and the
# notices.
#
# A mutating run refuses a dirty worktree before it changes a file, and the candidate is copied and checked
# against its new manifest outside the destination before it is published: a failed copy, an invalid
# upstream or an interrupted preparation leaves the tracked bytes and unrelated edits as they were, the
# staged candidate kept for a --publish where the failure was a publish's; an interrupted publish's
# journal and preimage records resume on the same candidate and refuse a file changed since the
# interruption. --compare works on a patched, dirty tree. The vendored directory's own four
# (MANIFEST.sha256, UPSTREAM, PATCHES.md, AGENTS.md) stand.
# UPSTREAM and PATCHES.md are edited by hand afterwards. No lock, no network.
set -euo pipefail
export LC_ALL=C
. "$(dirname "$0")/vendor_lib.sh"
OWN="MANIFEST.sha256 UPSTREAM PATCHES.md AGENTS.md"
usage() { sed -n 6,15p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

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
    [ $# -eq 2 ] && [ -d "$2" ] || usage
    NAME=$1
    vendor_repo || exit 2
    vendor_publish "$2" || exit 1
    echo "candidate $2 published; now UPSTREAM and PATCHES.md by hand"
    exit
fi

if [ -n "$REPLAY" ]; then
    [ $# -ge 2 ] || usage
    NAME=$1; shift
    vendor_repo || exit 2
    vendor_refuse_dirty || exit 1
    vendor_replay "$(git -C "$REPO" rev-parse HEAD)" "$@" || exit 1
    echo "replay finished; publish it with: $(basename "$0") --publish-replay $NAME REPLAY_DIR (named above)"
    exit
fi

if [ -n "$PUBLISH_REPLAY" ]; then
    [ $# -eq 2 ] && [ -d "$2" ] || usage
    NAME=$1
    vendor_repo || exit 2
    vendor_publish_replay "$2" || exit 1
    echo "replayed patches are on $(git -C "$REPO" branch --show-current || echo HEAD)"
    exit
fi

[ $# -eq 2 ] && [ -d "$2" ] || usage
NAME=$1
vendor_repo || exit 2
UPSTREAM=$(cd "$2" && pwd)
case $UPSTREAM/ in "$REPO"/zig/vendor/*) echo "UPSTREAM_DIR is the vendored directory itself" >&2; exit 2 ;; esac
cd "$UPSTREAM"

case $NAME in
stb) list=$(printf '%s\n' LICENSE stb_image.h stb_vorbis.c) ;;
dr_libs) list=$(printf '%s\n' LICENSE dr_flac.h dr_mp3.h dr_wav.h) ;;
libwebp)
    sources=$(sed -n '/^DEC_OBJS =/,/^$/p; /^DSP_DEC_OBJS =/,/^$/p; /^UTILS_DEC_OBJS =/,/^$/p' makefile.unix |
        grep -o 'src/[a-z0-9_/]*\.o' | sed 's/\.o$/.c/' | grep -vE '_(mips32|mips_dsp_r2|msa|sse2|sse41|avx2)\.c$')
    [ -n "$sources" ] || { echo "makefile.unix lists no decoder sources: the makefile moved" >&2; exit 1; }
    headers=$(for source in $sources; do zig cc -target aarch64-linux-gnu -I. -MM "$source"; done |
        tr ' \\' '\n\n' | grep -E '\.h$' | xargs -r realpath -m --relative-to=.)
    list=$(printf '%s\n' $sources $headers COPYING PATENTS AUTHORS | sort -u)
    ;;
*) echo "NAME is stb, libwebp or dr_libs" >&2; exit 2 ;;
esac
while IFS= read -r file; do
    [ -f "$file" ] || { echo "$file is not in $UPSTREAM" >&2; exit 1; }
    case $file in ../* | /*) echo "$file is outside $UPSTREAM" >&2; exit 1 ;; esac
done <<<"$list"

if [ -n "$COMPARE" ]; then
    differing=0
    while IFS= read -r file; do cmp -s "$file" "$VENDORED/$file" || { differing=$((differing + 1)); echo "differs: $file"; }; done <<<"$list"
    here=$(cd "$VENDORED" && find . -type f | sed 's|^\./||' | { grep -vxF "$(printf '%s\n' $OWN)" || true; } | sort)
    extra=$(comm -13 <(printf '%s\n' "$list") <(printf '%s\n' "$here"))
    [ -z "$extra" ] || { differing=$((differing + $(printf '%s\n' "$extra" | wc -l))); printf '%s\n' "$extra" | sed 's/^/not upstream'"'"'s list: /'; }
    echo "$(printf '%s\n' "$list" | wc -l) files of upstream's list compared, $differing differing or beside it"
    [ "$differing" -eq 0 ]
    exit
fi

# --stage only makes the candidate, which a dirty tree allows; the mutating default refuses a dirty
# worktree before anything is staged.
if [ -n "$STAGE" ]; then
    stage=$(printf '%s\n' "$list" | vendor_stage) || exit 1
    echo "$(printf '%s\n' "$list" | wc -l) files staged at $stage; publish with: $(basename "$0") --publish $NAME $stage"
    exit
fi
vendor_refuse_dirty || exit 1
stage=$(printf '%s\n' "$list" | vendor_stage) || exit 1
vendor_publish "$stage" || exit 1
bytes=$(printf '%s\n' "$list" | tr '\n' '\0' | xargs -0 cat -- | wc -c)
echo "$(printf '%s\n' "$list" | wc -l) files, $bytes bytes from $UPSTREAM; now UPSTREAM and PATCHES.md by hand"
