# What the vendored imports share (spec 168): a candidate is built and checked outside the destination
# before it is published, a mutating import or replay refuses a dirty target worktree first, and a local
# patch is replayed in a throwaway worktree of the repository, left for recovery where it fails.
# Sourced by tools/vendor_import.sh and tools/vendor_c.sh; expects NAME set and calls vendor_repo() first
# (REPO and VENDORED its results). Every function is written to run under a caller's `if`/`||`, where
# errexit is off: each fallible command is checked on its own, and a doubt fails closed — nothing mutates
# the destination or the original ref on an unverified state.

# The repository this tool serves, canonicalized before any caller cd: the script's own tree (the library
# sits beside it), never a directory named by an environment variable, and inside a worktree its own root
# — a subdirectory would let the dirty check's porcelain see only part of it. Fails closed where the tree
# cannot be verified.
vendor_repo() {
    local repo
    repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd -P) ||
        { echo "cannot resolve the repository this tool stands in" >&2; return 1; }
    REPO=$repo
    VENDORED=$REPO/zig/vendor/$NAME
    git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        echo "$REPO is not inside a worktree: cleanliness unverifiable, mutating anyway" >&2
        return 0
    }
    [ "$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null)" = "$REPO" ] ||
        { echo "$REPO is not a worktree's root: the dirty check would see only part of it" >&2; return 1; }
}

# Refuse a mutating step before it changes a file: the destination's worktree must be clean. A target that
# is not a worktree at all cannot be checked — vendor_repo already said so, go on; a target that is one
# but whose status cannot be read is refused. Untracked or ignored files under the vendored directory
# count as dirty there, since publishing removes every file not upstream's or the directory's own.
vendor_refuse_dirty() {
    git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 || {
        if [ -e "$REPO/.git" ]; then
            echo "cannot verify $REPO's cleanliness: git failed" >&2
            return 1
        fi
        return 0
    }
    local state
    state=$(git -C "$REPO" status --porcelain) || { echo "cannot read $REPO's status" >&2; return 1; }
    [ -z "$state" ] || { echo "refusing to change files: the worktree is dirty (an import is a commit of its own):" >&2; printf '%s\n' "$state" >&2; return 1; }
    state=$(git -C "$REPO" status --porcelain --ignored -- "$VENDORED") || { echo "cannot read $REPO's status" >&2; return 1; }
    [ -z "$state" ] || { echo "refusing to change files: files under $VENDORED are untracked or ignored:" >&2; printf '%s\n' "$state" >&2; return 1; }
}

# The paths a staged candidate holds: its manifest's. Fails where the manifest is missing, unreadable or
# empty — an empty set must never authorize a deletion.
vendor_stage_files() {
    [ -s "$1/MANIFEST.sha256" ] || return 1
    cut -c67- "$1/MANIFEST.sha256"
}

# A candidate's manifest and complete file set, checked before any destination change and again on a
# retry: the manifest is nonempty and reads as sha256sum's, the staged bytes verify against it, and
# nothing sits in the candidate beside its files and its two records — a stray, a missing or a truncated
# file fails here, not at the destination.
vendor_stage_check() {
    local stage=$1 listed found
    [ -f "$stage/.vendor-meta" ] && [ -s "$stage/MANIFEST.sha256" ] || { echo "$stage is not a staged candidate" >&2; return 1; }
    grep -q "^name: $NAME\$" "$stage/.vendor-meta" || { echo "$stage is not a staged $NAME candidate" >&2; return 1; }
    (cd "$stage" && sha256sum -c --quiet MANIFEST.sha256 >/dev/null) || { echo "$stage does not match its manifest" >&2; return 1; }
    listed=$(vendor_stage_files "$stage" | sort) || { echo "$stage: cannot read its manifest's file list" >&2; return 1; }
    [ -n "$listed" ] || { echo "$stage: the manifest holds no file" >&2; return 1; }
    found=$(cd "$stage" && find . -type f ! -name MANIFEST.sha256 ! -name .vendor-meta | sed 's|^\./||' | sort) ||
        { echo "$stage: cannot list its files" >&2; return 1; }
    [ "$listed" = "$found" ] || {
        echo "$stage: the candidate's files and its manifest disagree" >&2
        comm -23 <(printf '%s\n' "$listed") <(printf '%s\n' "$found") | sed 's/^/  not in the manifest: /' >&2
        comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$found") | sed 's/^/  missing from the candidate: /' >&2
        return 1
    }
}

# The files present under $1 beside the transaction records and the directory's own files — $OWN, which
# the host importer sets from tools/host_descriptor.zig's `own` and a staged candidate's .vendor-meta
# carries so --publish needs no descriptor; vendor_c.sh sets its own contract. Missing: refuse — there
# is no second list here to fall back on.
vendor_present() {
    local excludes=() each
    for each in ${OWN:-}; do excludes+=( ! -name "$each" ); done
    [ ${#excludes[@]} -gt 0 ] || { echo "vendor_present: OWN is empty: the importer or the candidate's record must name the directory's own files" >&2; return 1; }
    (cd "$1" && find . -type f "${excludes[@]}" \
        ! -name .vendor-journal ! -name .vendor-journal.pre \
        | sed 's|^\./||' | sort)
}

# Validate an interrupted publish's destination before resuming: every file now under $VENDORED must be
# byte-for-byte what the transaction's preimage recorded (never reached) or what the candidate carries
# (already copied); a preimage path the transaction would not have deleted must still be there or hold
# candidate bytes. Anything else is a user's edit made after the interruption and refuses the resume —
# the journal and preimage stand so the state can be settled by hand. Reads $stage's manifest itself.
vendor_journal_resume() {
    local stage=$1 pre=$2 line p h bad=0
    [ -f "$pre" ] || { echo "the journal's preimage record is missing: cannot verify what the transaction left; refusing" >&2; return 1; }
    declare -A preh candh
    while IFS= read -r line; do h=${line:0:64}; p=${line:66}; preh[$p]=$h; done < "$pre"
    while IFS= read -r line; do h=${line:0:64}; p=${line:66}; candh[$p]=$h; done < "$stage/MANIFEST.sha256"
    while IFS= read -r p; do
        [ -n "$p" ] || continue
        h=$(cd "$VENDORED" && sha256sum -- "$p" | cut -d' ' -f1)
        if [ -z "$h" ] || { [ "${preh[$p]:-}" != "$h" ] && [ "${candh[$p]:-}" != "$h" ]; }; then
            echo "  changed or added after the interruption: $p" >&2; bad=1
        fi
    done < <(vendor_present "$VENDORED")
    for p in "${!preh[@]}"; do
        [ -e "$VENDORED/$p" ] && continue
        if [ -n "${candh[$p]:-}" ]; then
            echo "  deleted after the interruption (the candidate expects it): $p" >&2; bad=1
        else
            # a file the transaction itself removes is the one absence it could have left; there is no
            # telling it from a user's deletion, so the resume cannot prove the state — fail closed
            echo "  gone after the interruption (cannot tell the transaction's removal from a user's): $p" >&2; bad=1
        fi
    done
    if [ "$bad" -ne 0 ]; then
        echo "files under $VENDORED changed since the transaction began; nothing of them is touched." >&2
        echo "restore them, or after checking the tree remove the journal and $pre and publish cleanly" >&2
        return 1
    fi
}

# Copy the files on stdin (one path a line, from UPSTREAM's root) into a fresh staging directory, write its
# MANIFEST.sha256 and .vendor-meta (NAME and the directory the files came from), and check it. Prints the
# staging directory's path. Nothing outside the staging directory is changed; a failure removes it, so a
# partial candidate is never left.
vendor_stage() {
    local list stage parent
    list=$(cat)
    parent=${XDG_CACHE_HOME:-$HOME/.cache}/zspark
    mkdir -p "$parent" || { echo "cannot make the staging root $parent" >&2; return 1; }
    stage=$(mktemp -d "$parent/vendor-stage-$NAME.XXXXXX") || { echo "mktemp failed under $parent" >&2; return 1; }
    if ! printf '%s\n' "$list" | tr '\n' '\0' | rsync -a --from0 --files-from=- "$UPSTREAM/" "$stage/"; then
        rm -rf "$stage"; echo "copying the candidate into $stage failed; $VENDORED is unchanged" >&2; return 1
    fi
    if ! (cd "$UPSTREAM" && printf '%s\n' "$list" | tr '\n' '\0' | xargs -0 sha256sum -- >"$stage/MANIFEST.sha256"); then
        rm -rf "$stage"; echo "writing the manifest failed; $VENDORED is unchanged" >&2; return 1
    fi
    if ! printf 'name: %s\nfrom: %s\nown: %s\n' "$NAME" "$UPSTREAM" "${OWN:-}" >"$stage/.vendor-meta"; then
        rm -rf "$stage"; echo "writing the candidate's record failed; $VENDORED is unchanged" >&2; return 1
    fi
    if ! vendor_stage_check "$stage"; then
        rm -rf "$stage"; echo "the staged candidate failed its check; $VENDORED is unchanged" >&2; return 1
    fi
    printf '%s\n' "$stage"
}

# Publish the staged candidate at $1 into $VENDORED, as one transaction the journal at the destination's
# top records: the destination's preimage (every file's path and hash), the journal, then upstream's
# files (the manifest among them), then the removal of every destination file the candidate does not
# name, then both records gone. The destination's bytes and file set are checked against the new
# manifest afterwards. A publish that fails or is interrupted leaves journal and preimage: a later
# --publish of the same candidate resumes it — but only when every file under the destination is still
# exactly what the preimage or the candidate says; a file edited or added after the interruption refuses
# the resume and is left alone.
vendor_publish() {
    local stage=$1 journal preimage digest listed found gone rel dirty
    vendor_stage_check "$stage" || return 1
    # the candidate carries the own-file list it was staged with, so publishing needs no descriptor
    [ -n "${OWN:-}" ] || OWN=$(sed -n 's/^own: //p' "$stage/.vendor-meta")
    digest=$(sha256sum "$stage/MANIFEST.sha256" | cut -d' ' -f1) || { echo "cannot hash the candidate's manifest" >&2; return 1; }
    journal=$VENDORED/.vendor-journal
    preimage=$VENDORED/.vendor-journal.pre
    rel=${VENDORED#$REPO/}
    if [ -f "$journal" ]; then
        if grep -q "^manifest: $digest\$" "$journal"; then
            echo "resuming the interrupted publish of $stage (its journal stands at $journal)" >&2
            git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1 && {
                dirty=$(git -C "$REPO" status --porcelain) || { echo "cannot read $REPO's status" >&2; return 1; }
                dirty=$(printf '%s\n' "$dirty" | grep -v " $rel/" || true)
                [ -z "$dirty" ] || { echo "files outside $VENDORED changed since the transaction began; settle them or undo it" >&2; printf '%s\n' "$dirty" >&2; return 1; }
            }
            vendor_journal_resume "$stage" "$preimage" || return 1
        else
            echo "$VENDORED has an interrupted publish of another candidate (its journal at $journal);" >&2
            echo "publish that stage again or remove $journal after checking the tree" >&2
            return 1
        fi
    else
        vendor_refuse_dirty || return 1
    fi
    listed=$(vendor_stage_files "$stage" | sort) || { echo "cannot read the candidate's file list" >&2; return 1; }
    [ -n "$listed" ] || { echo "the candidate's manifest holds no file: refusing" >&2; return 1; }
    mkdir -p "$VENDORED" || { echo "cannot make $VENDORED" >&2; return 1; }
    if [ ! -f "$journal" ]; then
        # the preimage is taken before the first mutation so a resume can prove what the transaction
        # left; it is never rewritten — a resume of a further interruption keeps the original's
        (cd "$VENDORED" 2>/dev/null && vendor_present "$VENDORED" | tr '\n' '\0' | xargs -0 -r sha256sum -- >"$preimage") ||
            { echo "cannot record $VENDORED's preimage; nothing changed" >&2; return 1; }
        printf 'manifest: %s\nstage: %s\n' "$digest" "$stage" >"$journal" ||
            { rm -f "$preimage"; echo "cannot write the publish journal; nothing changed" >&2; return 1; }
    fi
    # --checksum: content, not size+mtime — a same-size edit made within rsync's second of resolution
    # would otherwise publish stale bytes and then fail the manifest check
    printf '%s\n' "$listed" | tr '\n' '\0' | rsync -a --checksum --from0 --files-from=- "$stage/" "$VENDORED/" || {
        echo "copying $stage into $VENDORED failed; the journal stands, --publish resumes it" >&2; return 1
    }
    cp "$stage/MANIFEST.sha256" "$VENDORED/MANIFEST.sha256" ||
        { echo "installing the new manifest failed; the journal stands, --publish resumes it" >&2; return 1; }
    found=$(vendor_present "$VENDORED") || { echo "cannot list $VENDORED; the journal stands, --publish resumes it" >&2; return 1; }
    gone=$(comm -13 <(printf '%s\n' "$listed") <(printf '%s\n' "$found")) || { echo "cannot diff the file sets; the journal stands" >&2; return 1; }
    if [ -n "$gone" ]; then
        printf '%s\n' "$gone" | tr '\n' '\0' | (cd "$VENDORED" && xargs -0 -r rm -f --) || {
            echo "removing the files the candidate does not hold failed; the journal stands, --publish resumes it" >&2; return 1
        }
    fi
    find "$VENDORED" -mindepth 1 -type d -empty -delete ||
        { echo "tidying empty directories failed; the journal stands, --publish resumes it" >&2; return 1; }
    (cd "$VENDORED" && sha256sum -c --quiet MANIFEST.sha256 >/dev/null) || {
        echo "$VENDORED does not match the new manifest after publish; the journal stands, --publish resumes it" >&2; return 1
    }
    found=$(vendor_present "$VENDORED") || { echo "cannot list $VENDORED; the journal stands" >&2; return 1; }
    [ "$listed" = "$found" ] || { echo "$VENDORED's files and the new manifest disagree; the journal stands" >&2; return 1; }
    rm -f "$journal" "$preimage" || { echo "cannot drop the publish journal" >&2; return 1; }
    rm -rf "$stage" || { echo "$stage published but not removed" >&2; return 1; }
}
# Replay the local patch commits $2... onto the base commit $1 in a fresh worktree of $REPO, as one
# sequenced cherry-pick: the queue (base, then each commit a line) is written into the worktree first so a
# conflict's recovery resumes every commit, not only the one that failed. Nothing of the caller's tree is
# changed; a pick that conflicts or a hook that refuses leaves the worktree with the failing commit named.
# Prints the worktree's path on success.
vendor_replay() {
    local base=$1 replay failed
    shift
    [ -d "$REPO/.git" ] || [ -f "$REPO/.git" ] || { echo "$REPO is not a git tree: replay needs one" >&2; return 1; }
    [ $# -ge 1 ] || { echo "replay takes at least one commit" >&2; return 1; }
    replay=$(mktemp -d --dry-run "$(dirname "$REPO")/.vendor-replay-XXXXXX") ||
        { echo "cannot name a replay worktree" >&2; return 1; }
    if ! git -C "$REPO" worktree add --detach "$replay" "$base" >&2; then
        echo "making the replay worktree failed; nothing changed" >&2; return 1
    fi
    if ! { printf 'base: %s\n' "$base"; printf 'commit: %s\n' "$@"; } >"$replay/.vendor-replay-queue"; then
        rm -f "$replay/.vendor-replay-queue"
        git -C "$REPO" worktree remove "$replay" 2>/dev/null
        echo "writing the replay queue failed; nothing changed" >&2; return 1
    fi
    if git -C "$replay" cherry-pick -x "$@" >&2; then
        :
    else
        failed=$(git -C "$replay" rev-parse --short CHERRY_PICK_HEAD 2>/dev/null || true)
        if git -C "$replay" status --porcelain | grep -qE '^[ADU][ADU]'; then
            echo "the pick of ${failed:-a commit} conflicts. The replay worktree is kept at $replay:" >&2
            git -C "$replay" status --porcelain | grep -E '^[ADU][ADU]' >&2
            echo "  settle it there against upstream's new text (never the old file), then" >&2
            echo "  git -C $replay cherry-pick --continue   (the sequencer holds the commits left)" >&2
            echo "  $(basename "$0") --publish-replay $replay" >&2
        else
            echo "the pick of ${failed:-a commit} failed without a conflict (a hook's refusal, or an empty pick)." >&2
            echo "the replay worktree is kept at $replay: finish or undo it there, then --publish-replay it" >&2
            echo "  or remove it: git -C $REPO worktree remove --force $replay" >&2
        fi
        return 1
    fi
    vendor_replay_markers "$replay" || { echo "the replay worktree is kept at $replay" >&2; return 1; }
    printf '%s\n' "$replay"
}

# Conflict markers inside a replay worktree, searched at the repository-relative path (an absolute
# $VENDORED would be outside the worktree and read as an error, not an empty result). Fails where the
# search cannot run, so an unreadable tree is not a clean one.
vendor_replay_markers() {
    local replay=$1 rel=${VENDORED#$REPO/} found rc=0
    # git grep: 0 = markers found, 1 = clean tree, >1 = the search itself failed
    found=$(git -C "$replay" grep -lE '^(<<<<<<<|>>>>>>>|\|\|\|\|\|\|\|)' -- "$rel" 2>&1) || rc=$?
    if [ "$rc" -gt 1 ]; then
        echo "the marker search could not run in $replay: $found" >&2
        return 1
    fi
    [ "$rc" -eq 0 ] || return 0
    echo "unresolved conflict markers were picked into:" >&2; printf '%s\n' "$found" >&2; return 1
}

# Publish a finished replay worktree $1: fast-forward the caller's branch to the replay's HEAD. Refuses
# before touching the original ref: a missing queue, a replay worktree that is dirty or mid-pick, markers
# still in it, the current HEAD not the replay's base, or a replay whose new commits do not record every
# queued pick.
vendor_publish_replay() {
    local replay=$1 head dir queue_head commit dirty
    git -C "$replay" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$replay is not a worktree" >&2; return 1; }
    vendor_refuse_dirty || return 1
    dir=$(git -C "$replay" rev-parse --absolute-git-dir) || { echo "cannot locate $replay's git directory" >&2; return 1; }
    for state in CHERRY_PICK_HEAD MERGE_HEAD REBASE_HEAD REVERT_HEAD; do
        [ -e "$dir/$state" ] || continue
        echo "$replay has an unfinished $state: finish or undo it there first" >&2
        return 1
    done
    [ ! -s "$dir/sequencer/todo" ] || { echo "$replay's sequencer still holds picks" >&2; return 1; }
    # the queue file is the replay's own record, not its dirt
    dirty=$(git -C "$replay" status --porcelain) || { echo "cannot read $replay's status" >&2; return 1; }
    dirty=$(printf '%s\n' "$dirty" | grep -v '?? .vendor-replay-queue' | grep . || true)
    [ -z "$dirty" ] || { echo "$replay is dirty: commit or undo what it holds first" >&2; printf '%s\n' "$dirty" >&2; return 1; }
    vendor_replay_markers "$replay" || return 1
    [ -f "$replay/.vendor-replay-queue" ] || { echo "$replay holds no .vendor-replay-queue: not a replay this tool made" >&2; return 1; }
    grep -q '^commit: ' "$replay/.vendor-replay-queue" || { echo "$replay's queue holds no commit" >&2; return 1; }
    head=$(git -C "$REPO" rev-parse HEAD) || { echo "cannot read this HEAD" >&2; return 1; }
    grep -q "^base: $head\$" "$replay/.vendor-replay-queue" ||
        { echo "this HEAD has moved since the replay was made: replay again on it" >&2; return 1; }
    queue_head=$(git -C "$replay" rev-parse HEAD) || { echo "cannot read $replay's HEAD" >&2; return 1; }
    [ "$queue_head" != "$head" ] || { echo "$replay holds no new commit" >&2; return 1; }
    # every queued commit's pick records its name in the new commit's message (`-x` writes it)
    while IFS= read -r commit; do
        commit=${commit#commit: }
        git -C "$replay" log --format=%B "$head..$queue_head" | grep -q "cherry picked from commit $commit" ||
            { echo "$replay is missing the pick of $commit: the queue's commits must all land" >&2; return 1; }
    done < <(grep '^commit: ' "$replay/.vendor-replay-queue")
    rm -f "$replay/.vendor-replay-queue" ||
        { echo "cannot drop the queue record; nothing merged" >&2; return 1; }
    git -C "$REPO" merge --ff-only "$queue_head" &&
        git -C "$REPO" worktree remove "$replay" ||
        { echo "the merge or the worktree's removal failed; $replay is left" >&2; return 1; }
}