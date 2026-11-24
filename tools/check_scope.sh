# Test selection and pass evidence shared by split and full. Load the area inventory first.

# Include the run mode and delimit environment entries with NUL so multiline values are hashed whole.
pass_key() {
  {
    printf 'full-scope-v2\0optimize=%s\0bare=%s\0' "${DEBUG:+debug}" "$BARE"
    while IFS= read -r -d '' setting; do
      case $setting in ZSPARK_*=*) printf '%s\0' "$setting" ;; esac
    done < <(env -0) | LC_ALL=C sort -z
  } | sha256sum | cut -c1-12
}
pass_mark_path() { printf '%s/full-passed/%s-%s\n' "$CACHE" "$1" "$(pass_key)"; }

pass_current_mark() {
  local status head
  status=$(git status --porcelain) && [ -z "$status" ] || return 1
  head=$(git rev-parse HEAD) || return 1
  pass_mark_path "$head"
}

pass_record() {
  [ -n "$1" ] || return 0
  local current
  current=$(pass_current_mark) && [ "$current" = "$1" ] || return 0
  mkdir -p "$CACHE/full-passed" && printf '%s\n' "$2" > "$current"
}

scope_base() {
  local key head file sha n least= best=
  key=$(pass_key); head=$(git rev-parse HEAD)
  for file in "$CACHE"/full-passed/*-"$key"; do
    [ -f "$file" ] || continue
    sha=${file##*/}; sha=${sha%-*}
    [ "$sha" = "$head" ] && continue
    git merge-base --is-ancestor "$sha" HEAD 2> /dev/null || continue
    n=$(git rev-list --count "$sha..HEAD")
    if [ -z "$least" ] || ((n < least)); then least=$n; best=$sha; fi
  done
  echo "$best"
}
# scope BASE [HEAD]: SCOPE_STEPS the steps to run, SCOPE_LINE what was chosen and why. HEAD absent: the working tree.
scope() {
  local base head=${2:-} short paths edges untracked
  base=$(git rev-parse --verify --end-of-options "$1^{commit}" 2>/dev/null) ||
    { printf 'invalid base revision: %s\n' "$1" >&2; return 2; }
  short=$(git rev-parse --short "$base") || return 2
  local -a grep_options=() grep_revisions=()
  if [ -n "$head" ]; then
    head=$(git rev-parse --verify --end-of-options "${2}^{commit}" 2>/dev/null) ||
      { printf 'invalid head revision: %s\n' "$2" >&2; return 2; }
    paths=$(git diff --name-only --no-renames "$base" "$head" --) || return 2
    grep_revisions=("$head")
  else
    paths=$(git diff --name-only --no-renames "$base" --) || return 2
    untracked=$(git ls-files --others --exclude-standard) || return 2
    paths=$(printf '%s\n' "$paths" "$untracked" | sort -u)
    grep_options=(--untracked)
  fi
  edges=$(git grep -n -o -E "${grep_options[@]}" -e '@(import|embedFile)\("[^"]+"\)' "${grep_revisions[@]}" -- 'zig/*.zig' ':!zig/vendor') ||
    [ "$?" = 1 ] || return 2
  if [ -n "$head" ]; then edges=$(sed "s/^$head://" <<<"$edges"); fi
  local out
  # shellcheck disable=SC2016
  out=$(awk -v steps="${SPLIT_STEPS[*]}" -v roots="${SPLIT_ROOTS[*]}" -v models="$SCOPE_MODELS" -v areasfile="$AREAS_FILE" -v base="$short" -v paths="$paths" '
    function norm(p,   n, a, s, i, k, out) {
      n = split(p, a, "/"); k = 0
      for (i = 1; i <= n; i++) { if (a[i] == "." || a[i] == "") continue; if (a[i] == "..") { if (k) k--; continue } s[++k] = a[i] }
      out = s[1]; for (i = 2; i <= k; i++) out = out "/" s[i]; return out
    }
    function reach(st, why) { if (!(st in first)) { first[st] = p " (" why ")"; ran++ } else more[st]++ }
    { file = $0; sub(/:.*/, "", file); t = $0; sub(/^[^:]*:[0-9]+:@(import|embedFile)\("/, "", t); sub(/"\)$/, "", t)
      if (t !~ /\// && t !~ /\.[a-z]+$/) next   # a module by its name: std, server, tier
      if (t ~ /^fixtures\//) q = "zig/tests/" t; else { d = file; sub(/\/[^\/]*$/, "", d); q = norm(d "/" t) }
      edge[file] = edge[file] "\n" q }
    END {
      ns = split(steps, step, " "); split(roots, rootl, " "); split(models, mlist, " "); for (i in mlist) model[mlist[i]] = 1
      for (i = 1; i <= ns; i++) {   # each step its tests import, from its root
        st = step[i]; root = rootl[i]; head = 1; tail = 0; queue[++tail] = root; seen[st, root] = 1
        while (head <= tail) {
          f = queue[head++]; holds[st, f] = 1
          d = f; sub(/\/[^\/]*$/, "", d); dirs[st, d] = 1
          if (f ~ /^zig\/src\/models\//) { m = f; sub(/^zig\/src\/models\//, "", m); sub(/\/.*/, "", m); usesmodel[st, m] = 1 }
          ne = split(edge[f], e, "\n"); for (j = 2; j <= ne; j++) if (!((st, e[j]) in seen)) { seen[st, e[j]] = 1; queue[++tail] = e[j] }
        }
      }
      np = split(paths, list, "\n"); count = 0; ran = 0
      for (k = 1; k <= np; k++) {
        p = list[k]; if (p == "") continue; count++
        if (p ~ /^zig\// && p ~ /\.md$/) continue
        if (p ~ /^tools\/(check.*\.sh|link_lib\.sh|local_env\.sh|check_qwen4_files\.txt|check_(specs(_test)?|imports|vendor)\.zig|cache(_test)?\.zig)$/) {
          if (!toolp) toolp = p
          continue
        }
        if (p !~ /^zig\// && p != "build.zig" && p != "build.zig.zon" && p != "tools/test_runner.zig" && p != "tools/host_descriptor.zig" && p != areasfile) continue
        if (p ~ /^(zig\/vendor|zig\/src\/(cuda|decode|weights|ranks))\// || p == "zig/src/main.zig" || p !~ /^zig\//) {
          if (!nshared++) sharedp = p
          for (i = 1; i <= ns; i++) reach(step[i], "shared"); continue
        }
        if (p ~ /^zig\/kernels\//) {
          m = p; sub(/^zig\/kernels\//, "", m); sub(/\/.*/, "", m)
          if (m in model) { for (i = 1; i <= ns; i++) if (step[i] != "host" && (step[i], m) in usesmodel) reach(step[i], "kernels/" m) }
          else for (i = 1; i <= ns; i++) if (step[i] != "host") reach(step[i], "shared kernel")
          continue
        }
        hit = 0; for (i = 1; i <= ns; i++) if ((step[i], p) in holds) { reach(step[i], "imported"); hit = 1 }
        if (hit || p ~ /\.zig$/) continue
        if (p ~ /^zig\/tests\/fixtures\//) {
          m = p; sub(/^zig\/tests\/fixtures\//, "", m); sub(/\/.*/, "", m)
          if (m in model) { for (i = 1; i <= ns; i++) if (step[i] == m) reach(m, "fixtures/" m); continue }
        }
        if (p ~ /^zig\/(src|tests)\//) {
          d = p; sub(/\/[^\/]*$/, "", d)
          for (i = 1; i <= ns; i++) if ((step[i], d) in dirs) { reach(step[i], "beside its .zig"); hit = 1 }
          if (hit) continue
        }
        if (!nshared++) sharedp = p " (unmapped)"
        for (i = 1; i <= ns; i++) reach(step[i], "unmapped, shared")
      }
      if (ran && !("whole" in first)) { first["whole"] = "always with runtime checks"; ran++ }
      if (toolp) { p = toolp; reach("host", "host tools") }
      chosen = ""; line = ""; skipped = ""
      for (i = 1; i <= ns; i++) if (step[i] in first) {
        chosen = chosen " " step[i]; line = line ", " step[i] ": " first[step[i]] (more[step[i]] ? " and " more[step[i]] " more" : "")
      } else skipped = skipped ", " step[i]
      print substr(chosen, 2)
      if (nshared) line = ", every step: " sharedp " shared" (nshared > 1 ? ", and " nshared - 1 " more shared paths" : "")
      if (!ran) line = ", nothing: no path a test builds"
      printf "since %s (%d paths changed): %s%s\n", base, count, ran ? "run" substr(line, 2) : substr(line, 3), skipped == "" || !ran ? "" : "; skip " substr(skipped, 3) ": no changed path their tests import, none under their kernels/ or fixtures/, none shared"
    }' <<<"$edges") || return 2
  read -r -a SCOPE_STEPS <<<"$(head -1 <<<"$out")"
  SCOPE_LINE=$(tail -1 <<<"$out")
}
