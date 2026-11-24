# The test areas' inventory, sourced by check.sh (spec 171): tools/test_areas.tsv is the one list of the
# areas' step names, Zig roots, host/GPU kind, tier and split membership, fallback costs, model membership
# and area filters. `build.zig` embeds it; this file reads and validates it for every consumer (`areas_load`),
# so a malformed row or a contradiction between the columns fails the verb before a step is built. The
# verbs read: AREA_STEPS every row in file order, SPLIT_STEPS the split=yes ones, SCOPE_MODELS the model
# names, SPLIT_FIXED the fallback seconds, and per-step AREA_KIND/AREA_TIER/AREA_ROOT/AREA_FILTERS.
#
# REPO names the tree read; run under the caller's `set -u`.

AREAS_FILE=${AREAS_FILE:-tools/test_areas.tsv}
AREAS_LOADED=
SPLIT_STEPS=()
AREA_STEPS=()
SPLIT_ROOTS=()
SCOPE_MODELS=
AREA_WHOLE=
declare -A AREA_KIND=() AREA_TIER=() AREA_ROOT=() AREA_FILTERS=() SPLIT_FIXED=()

# STEP's kind a run takes the locks for: gpu for everything but a host row.
areas_kind() { case ${AREA_KIND[${1:-}]:-host} in host) echo host ;; *) echo gpu ;; esac; }
# STEP is a row of the inventory.
areas_step() { [ -n "${AREA_KIND[${1:-}]:-}" ]; }

# Read and check tools/test_areas.tsv once. A problem goes to stderr with its row; the load fails and the
# caller says which verb could not run.
areas_load() {
  [ -z "$AREAS_LOADED" ] || return 0
  local file=$REPO/$AREAS_FILE n=0 line step kind tier split seconds model root filters about bad=0 wholes=0
  problem() { echo "$AREAS_FILE:$n: $*" >&2; bad=1; }
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*|'') continue ;; esac
    n=$((n + 1))
    IFS=$'\t' read -r step kind tier split seconds model root filters about <<<"$line"
    case $about in *$'\t'*) problem "more than nine columns"; continue ;; esac
    [ -n "$about" ] || { problem "fewer than nine columns"; continue; }
    [[ $step =~ ^[a-z0-9_]+$ ]] || { problem "step \"$step\" is not [a-z0-9_]+"; continue; }
    [ -z "${AREA_KIND[$step]:-}" ] || { problem "step $step is given twice"; continue; }
    case $kind in
      host) ;;
      gpu) ;;
      whole) wholes=$((wholes + 1)) ;;
      *) problem "kind \"$kind\" is not host, gpu or whole"; continue ;;
    esac
    case $tier in
      yes) [ "$kind" = host ] || problem "tier=yes on a $kind step" ;;
      no) ;;
      *) problem "tier \"$tier\" is not yes or no"; continue ;;
    esac
    case $split in
      yes)
        [[ $seconds =~ ^[0-9]+$ ]] || problem "split=yes needs its fallback seconds, has \"$seconds\"" ;;
      no)
        # a GPU or whole row left out of the split would drop its tests from `full` with no refusal:
        # only the host's subareas, whose tests the host tier's root already holds, may stay out
        [ "$kind" = host ] || problem "split=no on a $kind step: full would never run it"
        [ "$seconds" = - ] || problem "split=no needs - for seconds, has \"$seconds\"" ;;
      *) problem "split \"$split\" is not yes or no"; continue ;;
    esac
    case $model in
      -) ;;
      *[!a-z0-9_]*|'') problem "model \"$model\" is not a name or -" ;;
      *)
        [ "$kind" = gpu ] || problem "model $model on a $kind step"
        [ -d "$REPO/zig/src/models/$model" ] || problem "model $model: no zig/src/models/$model" ;;
    esac
    [ "$root" = "zig/${step}_tests.zig" ] || problem "root $root is not zig/${step}_tests.zig"
    [ -f "$REPO/$root" ] || problem "root $root is not a file"
    case $filters in
      -) [ "$kind" != whole ] || problem "a whole step lists its gated tests as filters" ;;
      *'||'*|'|'*|*'|') problem "filters \"$filters\" holds an empty test-filter" ;;
    esac
    AREA_STEPS+=("$step")
    AREA_KIND[$step]=$kind
    AREA_TIER[$step]=$tier
    AREA_ROOT[$step]=$root
    AREA_FILTERS[$step]=$filters
    if [ "$kind" = whole ]; then AREA_WHOLE=$step; fi
    if [ "$split" = yes ]; then
      SPLIT_STEPS+=("$step")
      SPLIT_ROOTS+=("$root")
      [[ $seconds =~ ^[0-9]+$ ]] && SPLIT_FIXED[$step]=$seconds
    fi
    [ "$model" = - ] || SCOPE_MODELS+="$model "
  done < "$file"
  [ $wholes = 1 ] || { echo "$AREAS_FILE: $wholes whole rows, exactly one needed" >&2; bad=1; }
  [ -n "${AREA_KIND[host]:-}" ] || { echo "$AREAS_FILE: no host step" >&2; bad=1; }
  [ $bad = 0 ] || return 1
  SCOPE_MODELS=${SCOPE_MODELS% }
  AREAS_LOADED=1
}

# The static checks shared by inventory and sets.
areas_validate() {
  local command=$1
  areas_load || { echo "$command FAILED: the test-area inventory is inconsistent (above); $AREAS_FILE" >&2; return 1; }
  areas_check_qwen4 || { echo "$command FAILED: qwen4's curated file coverage is inconsistent (above); tools/check_qwen4_files.txt" >&2; return 1; }
}

# The `tier.whole`-gated test names and stray gates below a tests directory, by `sets`' rules: an alias
# `const X = @import("tier").whole` read inside a test's body counts as the gate, and a gate outside a
# test's body (a helper a test calls) is a STRAY line. Output: `FILE\tNAME` a gated test,
# `STRAY\tFILE:LINE` a stray, FILE the path below zig/ with dots for slashes.
gated_find() { # DIR
  find "$1" -name '*.zig' -print0 | xargs -0 awk '
    FNR == 1 { file = FILENAME; sub(/^zig\//, "", file); sub(/\.zig$/, "", file); gsub(/\//, ".", file); alias = ""; name = "" }
    match($0, /^const [A-Za-z_0-9]+ = @import\("tier"\)\.whole;/) { alias = $2; next }
    match($0, /^test "/) { name = substr($0, 7); sub(/" \{$/, "", name); body = 1; next }
    /^}/ { name = ""; body = 0 }
    ($0 ~ /@import\("tier"\)\.whole/ && $0 !~ /^const /) || (alias != "" && $0 ~ ("[^A-Za-z_0-9]" alias "[^A-Za-z_0-9]")) {
      if (!body) print "STRAY\t" FILENAME ":" FNR
      else if (name != "") { print file "\t" name; name = "" }
    }'
}

# `qwen4-files`' coverage (spec 103, checked since spec 171): every `*_test.zig` under
# zig/tests/models/qwen4/ either has its one entry in tools/check_qwen4_files.txt — a file the area runs,
areas_check_qwen4() { # sets QWEN4_FILES
  local file=$REPO/tools/check_qwen4_files.txt n=0 line entry dotted zpath name
  local -A covered=()
  QWEN4_FILES=()
  local -A entries=()
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '#'*|'') continue ;; esac
    n=$((n + 1))
    entry=$line
    [ -z "${entries[$entry]:-}" ] || { echo "check_qwen4_files.txt:$n: the entry is given twice" >&2; return 1; }
    entries[$entry]=1
    dotted=${entry%%.test.*}
    name=${entry#*.test.}
    case $dotted in qwen4.*) ;; *) echo "check_qwen4_files.txt:$n: an entry names a file outside qwen4 ($dotted)" >&2; return 1 ;; esac
    zpath=zig/tests/models/${dotted//./\/}.zig
    { [ -n "$dotted" ] && [ "$dotted" != "$entry" ]; } || { echo "check_qwen4_files.txt:$n: not a path.test.name entry" >&2; return 1; }
    [ -f "$REPO/$zpath" ] || { echo "check_qwen4_files.txt:$n: no test file $zpath" >&2; return 1; }
    awk -v want="test \"$name" 'index($0, want) == 1 { found = 1 } END { exit !found }' "$REPO/$zpath" ||
      { echo "check_qwen4_files.txt:$n: no test of $zpath starts \"$name\"" >&2; return 1; }
    [ -z "${covered[$dotted]:-}" ] || { echo "check_qwen4_files.txt:$n: $dotted has an entry already" >&2; return 1; }
    covered[$dotted]=$entry
    QWEN4_FILES+=("$entry")
  done < "$file"
  [ ${#QWEN4_FILES[@]} -gt 0 ] || { echo "check_qwen4_files.txt: no entries" >&2; return 1; }
  # the gated tests of qwen4's files, sets' rules; a stray gate fails as it does there
  local gate_out stray
  gate_out=$(cd "$REPO" && gated_find zig/tests/models/qwen4)
  stray=$(grep -c '^STRAY' <<<"$gate_out" || true)
  [ "$stray" = 0 ] || { grep '^STRAY' <<<"$gate_out" | cut -f2 | sed "s/^/check_qwen4_files.txt: tier.whole outside a test's body: /" >&2; return 1; }
  local -A gated_count=()
  local -a gated_names=()
  local g gfile gname
  while IFS=$'\t' read -r gfile gname; do
    [ -n "${gfile:-}" ] || continue
    # gated_find names the file tests.models.qwen4...; the entries and the walk below drop tests.models.
    gfile=${gfile#tests.models.}
    gated_count[$gfile]=$(( ${gated_count[$gfile]:-0} + 1 ))
    gated_names+=("$gfile"$'\t'"$gname")
  done < <(grep -v '^STRAY' <<<"$gate_out" || true)
  # each file: all gated, an entry, or a problem
  local rel f total ng ename gfile2 gname2
  while IFS= read -r f; do
    rel=${f#"$REPO"/zig/tests/models/}
    dotted=${rel%.zig}
    dotted=${dotted//\//.}
    total=$(grep -c '^test "' "$f" || true)
    [ "$total" -gt 0 ] || continue
    ng=${gated_count[$dotted]:-0}
    if [ "$ng" = "$total" ]; then
      [ -z "${covered[$dotted]:-}" ] || { echo "check_qwen4_files.txt: $dotted's tests are all gated on tier.whole; its entry is stale" >&2; return 1; }
      continue
    fi
    ename=${covered[$dotted]:-}
    [ -n "$ename" ] || { echo "check_qwen4_files.txt: $rel's $((total - ng)) runnable test(s) have no entry" >&2; return 1; }
    ename=${ename#*.test.}
    for g in ${gated_names[@]+"${gated_names[@]}"}; do
      gfile2=${g%%$'\t'*}
      gname2=${g#*$'\t'}
      [ "$gfile2" = "$dotted" ] || continue
      case $gname2 in "$ename"*) echo "check_qwen4_files.txt: $dotted's entry names a tier.whole test: \"$ename\"" >&2; return 1 ;; esac
    done
  done < <(find "$REPO/zig/tests/models/qwen4" -name '*_test.zig' | sort)
}
