#!/bin/bash
# Both ranks of a two-rank command, rank 0 here and rank 1 on the other Spark over ssh (spec 045).
#
#   tools/two_ranks.sh ranks-check random [model=flash-next]  a rank a process over NCCL against both in one,
#                                           bit for bit: GLM's random model, or Flash Next's (`session`:
#                                           a pair's replies; `kv=bf16|fp8|fp8-nvfp4`: an attention
#                                           layer too, its caches so stored, spec 094)
#   tools/two_ranks.sh ranks-check DIR [layers=L]  Flash Next's first L layers the same way (spec 050)
#   tools/two_ranks.sh ranks-check DIR reference=REF [drafter=DIR]  the checkpoint against TensorFold's export,
#                                           or the vLLM container's replies and scores (spec 105), and its own serial
#                                           steps and replies; drafter= DFlash2 beside either model (107, 126; LIMIT 300)
#   tools/two_ranks.sh generate DIR ID..    GLM's reply to a prompt's ids, a rank a Spark (DIR on both); Flash
#                                           Next's is `generate DIR N ID..`, greedy, drafted by rank 0's
#                                           MTP head (`serial`: one token a step)
#   tools/two_ranks.sh chat DIR MESSAGE     its reply to a message, as text
#   tools/two_ranks.sh serve DIR FLAG..     GLM or Flash Next (DIR the FP4-Local checkpoint, spec 126) served by
#                                           rank 0 here, both ranks given the same flags (LIMIT seconds, a day;
#                                           SIGINT, or SIGTERM to rank 0's zspark, ends both); client_bench.sh
#                                           RANKS=2 measures it
#   tools/two_ranks.sh replies DIR [kv=bf16|fp8|fp8-nvfp4]
#                                           Flash Next's `generate` (sampled) and `chat` (greedy, copies offered)
#                                           with the MTP head split, each drafted and `--serial`: the same tokens
#                                           (spec 095); four runs of the commands above; `kv=` codes both's
#                                           caches (`--kv-dtype`)
#
# Rank 1 is started first and listens on the link's address; rank 0 connects to it. Rank 0's time limit ends
# it in order (SIGTERM), and rank 1 with it; rank 1's own is a minute later. Both run
# zig-out/bin/zspark of this repository's path on their machine, built with -Dnccl=/usr, and the script
# refuses two binaries whose `zspark version` differ.
#
# Locks: this machine's `machine.lock` then `gpu.lock`, then the other's over ssh, each held for the run. A timing
# (MACHINE=-x before generate, chat or serve) holds both machines alone (`machine.lock` whole); a check and
# `--local` take it shared. Rank 0 starts once rank 1 holds the other's locks, however long it queues there.
# A GPU that another process holds stops the script before anything runs.
#
# The link: PEER (198.51.100.2) is the other Spark's ConnectX address, PORT (29551) rank 1's. NCCL is told the
# link's netdev, its RoCE device, the index of that device's RoCE v2 GID of the netdev's IPv4 address, every
# other RoCE device with an IPv4 RoCE v2 GID at that index (the second rail), each read from `ip` and sysfs on
# its own machine, and 4 channels: the kit's settings. RAILS=1 keeps the link's device alone, CHANNELS= leaves
# the channels to NCCL. PEER_REPO is the repository on the other Spark when its path differs.
# WRAP is a command both ranks run zspark under, on each machine (e.g. `nsys launch --session-new=NAME`, whose
# capture `nsys start --session=NAME` and `nsys stop` begin and end on each; spec 111's trace).
#
# ZSPARK_RANKS_HEAD (`docs/environment.md`: Flash Next's MTP head `split`, half a rank, or `whole` on rank 0)
# and ZSPARK_DRAFTER_STORAGE (the DFlash2 drafter `stored` or `mixed`, spec 135) are handed to rank 1 as this
# shell has them, and so are ZSPARK_QUARANTINE (freed device buffers held back and read again, a debugging
# switch) and ZSPARK_LOAD_PHASES (where a load's time goes); no other ZSPARK_* variable is, and two ranks that
# were told different heads or drafters refuse each other when they join.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
. "$HERE/link_lib.sh"
. "$HERE/local_env.sh"
load_local_env "$REPO" || exit $?
CACHE=$HOME/.cache/zspark
PORT=${PORT:-29551}
PEER_REPO=${PEER_REPO:-$REPO}
RAILS=${RAILS:-2}
CHANNELS=${CHANNELS-4}
usage() { sed -n 2,25p "$0" | sed 's/^# \{0,1\}//'; exit 2; }

LOCAL=
[ "${1:-}" = --local ] && { LOCAL=1; shift; }
[ $# -ge 1 ] || usage
COMMAND=$1; shift
if [ -z "$LOCAL" ]; then : "${PEER:?Set PEER in .env.local}"; fi

# Each run's reply, from rank 0's lines: what `generate` prints of its tokens, or what `chat` prints of its text.
replies() {
  [ $# -ge 1 ] || usage
  local dir=$1 kv=
  shift
  if [ $# -ge 1 ]; then
    case $1 in kv=*) kv=${1#kv=} ;;
    *) usage ;;
    esac
    shift
  fi
  [ $# -eq 0 ] || usage
  local out messages failed=0 coded=
  out=$(mktemp -d "$CACHE/two_ranks_replies.XXXXXX")
  messages=$CACHE/two_ranks_replies.json
  # a reply that repeats what came before it, so that copies are offered; at the same path on both machines
  printf '%s\n' '[{"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number. Then write the same function again, unchanged, but with a docstring."}]' > "$messages"
  ssh -o BatchMode=yes "$PEER" "cat > $messages" < "$messages" || exit 2
  export ZSPARK_RANKS_HEAD=split
  local ids kvword kvflag=
  kvword=$kv
  [ -n "$kv" ] && kvflag="--kv-dtype $kv"
  ids=$("$REPO/zig-out/bin/zspark" chat "$dir" --render --no-thinking $kvflag "Tell me a short story about the sea." 2>&1 | sed -n 's/^[0-9]* ids: //p')
  [ -n "$ids" ] || { echo "two_ranks.sh: the prompt was not rendered"; exit 2; }
  for how in drafted serial; do
    local flag=; [ $how = serial ] && flag=serial
    "$0" generate "$dir" 96 $kvword temperature=0.7 seed=7 $flag $ids > "$out/generate-$how.log" 2>&1 || { echo "generate $how failed: $out/generate-$how.log"; exit 1; }
    flag=; [ $how = serial ] && flag=--serial
    "$0" chat "$dir" --max-tokens 256 --temperature 0 --no-thinking $flag $kvflag --messages "$messages" > "$out/chat-$how.log" 2>&1 || { echo "chat $how failed: $out/chat-$how.log"; exit 1; }
  done
  for command in generate chat; do
    if [ $command = generate ]; then
      grep '^sampled:' "$out/$command-drafted.log" > "$out/$command-drafted.reply"
      grep '^sampled:' "$out/$command-serial.log" > "$out/$command-serial.reply"
    else
      # rank 0's lines before rank 1's, less the link's, the switches' and the logged ones
      for how in drafted serial; do
        sed '/^rank 1:$/,$d' "$out/$command-$how.log" | grep -v -e '^rank 0 ' -e '^rank 1 ' -e '^zspark: ' -e '^info: ' -e '^logs: ' > "$out/$command-$how.reply"
      done
    fi
    if [ -s "$out/$command-drafted.reply" ] && cmp -s "$out/$command-drafted.reply" "$out/$command-serial.reply"; then
      echo "$command: drafted equals serial, $(wc -c < "$out/$command-drafted.reply") bytes; $(grep -h -o -e '[0-9]* rounds, [0-9]* of [0-9]* drafts kept' -e '[0-9]* of [0-9]* drafts kept' "$out/$command-drafted.log" | head -1)"
    else
      echo "$command: the drafted reply is not the serial one: $out/$command-*.reply"; failed=1
    fi
  done
  echo "logs: $out"
  exit $failed
}
[ "$COMMAND" = replies ] && replies "$@"
case $COMMAND in
  ranks-check) MACHINE=-s; LIMIT=${LIMIT:-300} ;;
  generate|chat) MACHINE=${MACHINE:--s}; LIMIT=${LIMIT:-1800} ;;
  serve) MACHINE=${MACHINE:--s}; LIMIT=${LIMIT:-86400} ;;
  *) usage ;;
esac

free_gpu=$(gpu_guard two_ranks.sh)

mkdir -p "$CACHE/logs"
LOG=$CACHE/logs/$(date +%Y%m%d-%H%M%S)-$COMMAND
Z=$REPO/zig-out/bin/zspark
[ -x "$Z" ] || { echo "two_ranks.sh: no $Z: zig build -Dnccl=/usr" >&2; exit 2; }

if [ -n "$LOCAL" ]; then
  # shellcheck disable=SC2016
  flock -s "$CACHE/machine.lock" flock "$CACHE/gpu.lock" bash -c "$free_gpu"'
    export NCCL_MULTI_RANK_GPU_ENABLE=1
    timeout "$(($4 + 60))" "$1" "$2" "${@:6}" rank=1 "at=127.0.0.1:$3" > "$5.rank1.log" 2>&1 &
    timeout "$4" "$1" "$2" "${@:6}" rank=0 "at=127.0.0.1:$3" 2>&1 | tee "$5.rank0.log"; status=${PIPESTATUS[0]}
    wait $! || status=$?
    echo "rank 1:"; cat "$5.rank1.log"; exit "$status"' local "$Z" "$COMMAND" "$PORT" "$LIMIT" "$LOG" "$@"
  exit $?
fi

SSH=(ssh -o BatchMode=yes "$PEER")
here=$("$Z" version) || exit 1
there=$("${SSH[@]}" "$PEER_REPO/zig-out/bin/zspark version") || { echo "two_ranks.sh: no zspark at $PEER:$PEER_REPO" >&2; exit 2; }
[ "$here" = "$there" ] || { echo "two_ranks.sh: '$here' here, '$there' on $PEER: build the same commit on both" >&2; exit 2; }
mine=$(link_env "$PEER" "$RAILS" "$CHANNELS") || exit 1
ME=$(ip -o route get "$PEER" | sed -n 's/.* src \([^ ]*\).*/\1/p')
theirs=$("${SSH[@]}" bash -s <<<"$(declare -f link_env); link_env $ME '$RAILS' '$CHANNELS'") || exit 1
# the one variable both ranks must read alike: rank 1's shell is not this one
[ -n "${ZSPARK_RANKS_HEAD:-}" ] && theirs="$theirs ZSPARK_RANKS_HEAD=$ZSPARK_RANKS_HEAD"
[ -n "${ZSPARK_DRAFTER_STORAGE:-}" ] && theirs="$theirs ZSPARK_DRAFTER_STORAGE=$ZSPARK_DRAFTER_STORAGE"
[ -n "${ZSPARK_QUARANTINE:-}" ] && theirs="$theirs ZSPARK_QUARANTINE=$ZSPARK_QUARANTINE"
[ -n "${ZSPARK_LOAD_PHASES:-}" ] && theirs="$theirs ZSPARK_LOAD_PHASES=$ZSPARK_LOAD_PHASES"
echo "rank 0 $(hostname): $mine"
echo "rank 1 $PEER: $theirs"

# shellcheck disable=SC2016,SC2029
flock $MACHINE "$CACHE/machine.lock" flock "$CACHE/gpu.lock" bash -c "$free_gpu"'
  z=$1 command=$2 peer=$3 port=$4 limit=$5 log=$6 mine=$7 theirs=$8 machine=$9 peer_repo=${10} free_gpu=${11}; shift 11
  ssh -o BatchMode=yes "$peer" "flock $machine ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock bash -c '\''$free_gpu
    echo \"two_ranks.sh: rank 1 holds its locks\"
    exec env $theirs timeout $((limit + 60)) ${WRAP:-} $peer_repo/zig-out/bin/zspark $command $* rank=1 at=$peer:$port'\''" > "$log.rank1.log" 2>&1 &
  remote=$!
  # rank 0 starts once rank 1 holds its machine'\''s locks: a peer queued behind another run listens late, and a
  # rank 0 started before would find no one (ConnectionRefused). Rank 1 refused or gone: rank 0 never starts
  until grep -q "^two_ranks.sh: rank 1 holds its locks" "$log.rank1.log" 2>/dev/null; do
    if ! kill -0 $remote 2>/dev/null; then
      wait $remote; status=$?
      echo "rank 1 did not start:"; cat "$log.rank1.log"; exit $((status == 0 ? 1 : status))
    fi
    sleep 1
  done
  env $mine timeout "$limit" ${WRAP:-} "$z" "$command" "$@" rank=0 "at=$peer:$port" 2>&1 | tee "$log.rank0.log"; status=${PIPESTATUS[0]}
  wait $remote || status=$?
  echo "rank 1:"; cat "$log.rank1.log"; exit "$status"' pair "$Z" "$COMMAND" "$PEER" "$PORT" "$LIMIT" "$LOG" "$mine" "$theirs" "$MACHINE" "$PEER_REPO" "$free_gpu" "$@"
status=$?
echo "logs: $LOG.rank0.log $LOG.rank1.log"
exit $status
