#!/bin/bash
# What a client sees of shared rounds on one load (spec 051): four short prompts alone, the four at once and
# again after, whose texts must be equal; a conversation of about 9000 tokens before a second stream has
# opened the pool (first turn, second turn: the session's prefix reuse), and the same two turns once the pool
# is open (the first filled again in wide chunks, its text the session's; the second from the reply's end the
# pool kept). It prints a line a check and the device memory after the load and after four streams, and
# leaves the bodies and server.log in OUT_DIR. It does not judge: equal hashes and `cached_tokens` are read.
#
#   tools/reference/shared_acceptance.sh CHECKPOINT_DIR OUT_DIR [serve args...]
#
# The server is ZSPARK (a `-Doptimize=fast` build) on 127.0.0.1:18888 with its defaults. The machine is held
# alone (`machine.lock` whole, `gpu.lock` inside), as tools/reference/client_bench.sh holds it.
set -uo pipefail
if [ $# -lt 2 ]; then sed -n 2,12p "$0"; exit 2; fi
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=tools/reference/serve_lib.sh
. "$HERE/serve_lib.sh"
serve_locked SHARED_ACCEPTANCE_LOCKED "$@"
CK=$(realpath "$1"); O=$2; shift 2
Z=${ZSPARK:-$HERE/../../zig-out/bin/zspark}
BASE=http://127.0.0.1:18888; U=$BASE/v1/chat/completions
serve_gpu_free
mkdir -p "$O"; rm -f "$O"/*.json "$O"/times "$O"/mem_* "$O"/server.log
{ date -Is; uptime; } > "$O/before.txt"
START=$(date +%s.%N)
serve_start 127.0.0.1 18888 "$O/server.log" "$Z" "$CK" --name m "$@"
trap 'serve_stop INT' EXIT
# a server that never answers is not waited for further: the requests below say so
serve_wait $BASE/health 600 1
if [ $? = 1 ]; then echo "the server ended: $O/server.log" >&2; exit 1; fi
echo "load $(echo "$(date +%s.%N)-$START" | bc) s" > "$O/load"
mem() { nvidia-smi --query-compute-apps=used_memory --format=csv,noheader > "$O/mem_$1"; }
mem load
P=("Explain how a hash map resolves collisions, in one paragraph."
   "Write a Python function that merges two sorted lists."
   "List five facts about the planet Mars."
   "Describe the water cycle to a ten year old.")
post() { # name body: the reply's body in name.json, its seconds in times
  curl -s -w "$1 %{time_total}\n" -m 600 $U -H 'content-type: application/json' -d "$2" -o "$O/$1.json" >> "$O/times"
}
req() { # name prompt
  post "$1" "{\"model\":\"m\",\"max_tokens\":160,\"temperature\":0,\"seed\":1,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"$2\"}]}"
}
LONG=$(python3 -c "print(' '.join('Item %d is the number %d squared, which is %d.'%(i,i,i*i) for i in range(400)))")
T1="{\"role\":\"user\",\"content\":\"$LONG Which item is 49?\"}"
HEAD="\"model\":\"m\",\"max_tokens\":24,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}"
turns() { # tag: the first turn, then the conversation with its reply and a question more
  post "turn1$1" "{$HEAD,\"messages\":[$T1]}"
  local a; a=$(jq -c '.choices[0].message.content' "$O/turn1$1.json")
  post "turn2$1" "{$HEAD,\"messages\":[$T1,{\"role\":\"assistant\",\"content\":$a},{\"role\":\"user\",\"content\":\"And which is 64?\"}]}"
}
turns ""
mem turns
for i in 0 1 2 3; do req alone$i "${P[$i]}"; done
pids=(); for i in 0 1 2 3; do req both$i "${P[$i]}" & pids+=($!); done; wait "${pids[@]}"
mem four
for i in 0 1 2 3; do req after$i "${P[$i]}"; done
turns pool
mem end
trap - EXIT
serve_stop INT
h() { jq -r '.choices[0].message.content' "$O/$1.json" | md5sum | cut -c1-8; }
for i in 0 1 2 3; do
  echo "prompt $i: alone $(h alone$i) four $(h both$i) after $(h after$i), tokens $(jq -c '.usage.completion_tokens' "$O/alone$i.json" "$O/both$i.json" "$O/after$i.json" | tr '\n' ' ')"
done
for t in turn1 turn2; do
  echo "$t: session $(h $t) pool $(h ${t}pool), usage $(jq -c .usage "$O/$t.json") pool $(jq -c .usage "$O/${t}pool.json")"
done
echo "memory: after load $(cat "$O/mem_load"), after the turns $(cat "$O/mem_turns"), after four $(cat "$O/mem_four"), at the end $(cat "$O/mem_end"); $(cat "$O/load")"
tr '\n' ' ' < "$O/times"; echo
