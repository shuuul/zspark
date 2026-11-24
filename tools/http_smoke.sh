#!/usr/bin/env bash
# tools/http_smoke.sh — a client's minimal check that a running `zspark serve` (or the stack's
# zspark-router) answers as documented in docs/serving.md. curl, jq and coreutils only; no
# Python, no build, no GPU. It sends requests and verifies status codes, JSON fields, SSE
# events and error bodies — it does not start or stop anything and never modifies the server.
#
#   tools/http_smoke.sh [--url BASE] [--model ID] [--timeout SECONDS]
#                       [--api-key KEY | --api-key-file PATH]
#                       [--asr FILE [--asr-model ID]] [--ocr FILE [--ocr-model ID]]
#                       [-v|--verbose]
#
#   --url         Base address (default $ZSPARK_SMOKE_URL or http://127.0.0.1:8888, the
#                 stack router's default; a lone `zspark serve` listens on 127.0.0.1:8888
#                 too unless --port says otherwise).
#   --model       Model id sent on chat requests (default $ZSPARK_SMOKE_MODEL, else the
#                 first id /v1/models lists).
#   --timeout     Per-request deadline in seconds (default 120; streamed replies get it too).
#   --api-key     Bearer token, as `serve --api-key`/TENSORFOLD_API_KEY expects. Sent only as
#                 an Authorization header through a private curl config file: never printed,
#                 never on the command line. Prefer $ZSPARK_API_KEY or --api-key-file so the
#                 key stays off the process list.
#   --asr FILE    Opt-in: POST FILE to /v1/audio/transcriptions (multipart, as served when an
#                 ASR engine is configured; WAV, FLAC, MP3 or Ogg Vorbis, <= 25 MiB, <= the
#                 engine's clip limit). --asr-model names the served ASR id (default
#                 $ZSPARK_SMOKE_ASR_MODEL; when unset the field is left out and the router's
#                 --audio backend answers).
#   --ocr FILE    Opt-in: POST a chat completion holding FILE as an image_url data URL to the
#                 OCR model (PNG/JPEG/etc., <= 10 MiB). --ocr-model names it (default
#                 $ZSPARK_SMOKE_OCR_MODEL or Unlimited-OCR-2).
#   -v            Print each request's status and the fields verified.
#
# Every check reports PASS/FAIL and the script exits 0 only if all ran checks pass. Skipped
# media checks are reported SKIP, not counted. A non-2xx or unparseable body is a FAIL with
# the server's own error text. Requires: curl >= 7.76 (-K), jq, awk, base64, mktemp.

set -u
PROG=${0##*/}
fail() { echo "$PROG: $*" >&2; exit 2; }

URL=${ZSPARK_SMOKE_URL:-http://127.0.0.1:8888}
MODEL=${ZSPARK_SMOKE_MODEL:-}
TIMEOUT=${ZSPARK_SMOKE_TIMEOUT:-120}
API_KEY=${ZSPARK_API_KEY:-}
API_KEY_FILE=
ASR_FILE= ; ASR_MODEL=${ZSPARK_SMOKE_ASR_MODEL:-}
OCR_FILE= ; OCR_MODEL=${ZSPARK_SMOKE_OCR_MODEL:-Unlimited-OCR-2}
VERBOSE=0

usage() { sed -n '3,30p' "$0"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case $1 in
    --url) URL=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --timeout) TIMEOUT=$2; shift 2 ;;
    --api-key) API_KEY=$2; shift 2 ;;
    --api-key-file) API_KEY_FILE=$2; shift 2 ;;
    --asr) ASR_FILE=$2; shift 2 ;;
    --asr-model) ASR_MODEL=$2; shift 2 ;;
    --ocr) OCR_FILE=$2; shift 2 ;;
    --ocr-model) OCR_MODEL=$2; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    -h|--help|help) usage 0 ;;
    *) fail "unknown argument: $1 (try $PROG --help)" ;;
  esac
done

command -v curl >/dev/null || fail "curl not found"
command -v jq   >/dev/null || fail "jq not found"
[ -n "$ASR_FILE" ] && [ ! -f "$ASR_FILE" ] && fail "--asr: no such file: $ASR_FILE"
[ -n "$OCR_FILE" ] && [ ! -f "$OCR_FILE" ] && fail "--ocr: no such file: $OCR_FILE"
[ -n "$API_KEY_FILE" ] && [ ! -f "$API_KEY_FILE" ] && fail "--api-key-file: no such file: $API_KEY_FILE"
[ -n "$API_KEY_FILE" ] && API_KEY=$(tr -d '[:space:]' < "$API_KEY_FILE")
URL=${URL%/}

TMP=$(mktemp -d) || exit 2
trap 'rm -rf "$TMP"' EXIT
CURL_AUTH=()
if [ -n "$API_KEY" ]; then
  # A curl -K file keeps the key off argv and out of any -v output; the temp dir is 0700.
  chmod 700 "$TMP"
  printf 'header = "Authorization: Bearer %s"\n' "$API_KEY" > "$TMP/auth.curl"
  chmod 600 "$TMP/auth.curl"
  CURL_AUTH=(-K "$TMP/auth.curl")
fi

PASS=0 ; FAILED=0 ; SKIPPED=0
note() { [ "$VERBOSE" = 1 ] && echo "  $*"; }
ok()   { PASS=$((PASS+1));   echo "PASS $1"; }
bad()  { FAILED=$((FAILED+1)); echo "FAIL $1: $2"; }
skip() { SKIPPED=$((SKIPPED+1)); echo "SKIP $1: $2"; }

# REQ METHOD PATH [curl args...] < BODY   ->  HTTP status in $STATUS, body in $TMP/body
req() {
  local method=$1 path=$2; shift 2
  : > "$TMP/body"
  STATUS=$(curl -sS -o "$TMP/body" -w '%{http_code}' --max-time "$TIMEOUT" \
             -X "$method" "${CURL_AUTH[@]}" "$@" "$URL$path") || { STATUS=000; }
  note "$method $path -> $STATUS"
}
# BODY as a JSON string, compact
body_json() { jq -c . "$TMP/body" 2>/dev/null || cat "$TMP/body"; }

echo "$PROG: $URL"

# ---- /health ------------------------------------------------------------------------------
# A lone serve: {"status":"ok", "model":..., "version":..., "backend":..., "device":...}.
# The router: 200 with "ok":true only while every configured model has a backend; a keyed
# server answers the bare {"status":"ok"}. Any shape needs the status and one ok marker.
req GET /health
if [ "$STATUS" != 200 ]; then
  bad "GET /health" "HTTP $STATUS (wanted 200): $(head -c 300 "$TMP/body")"
else
  H=$(jq -r 'if (.status=="ok" or .ok==true) then "ok" else "bad" end' "$TMP/body" 2>/dev/null)
  if [ "$H" = ok ]; then
    ok "GET /health"
    jq -r 'to_entries|map("\(.key)=\(.value|tostring)")|.[]' "$TMP/body" 2>/dev/null | head -8 | while read -r l; do note "health: $l"; done
  else
    bad "GET /health" "no ok marker in $(body_json | head -c 300)"
  fi
fi

# ---- /v1/models ---------------------------------------------------------------------------
req GET /v1/models
MODELS_JSON=
if [ "$STATUS" != 200 ]; then
  bad "GET /v1/models" "HTTP $STATUS: $(head -c 300 "$TMP/body")"
elif ! jq -e '.data | type == "array" and length > 0 and all(.[]; .id | type == "string")' \
      "$TMP/body" >/dev/null 2>&1; then
  bad "GET /v1/models" "not an OpenAI list with string ids: $(body_json | head -c 300)"
else
  ok "GET /v1/models"
  MODELS_JSON=$(cat "$TMP/body")
  note "models: $(jq -r '[.data[].id] | join(", ")' "$TMP/body")"
  if [ -z "$MODEL" ]; then
    MODEL=$(jq -r '.data[0].id' "$TMP/body")
    echo "model: $MODEL (first of /v1/models; pass --model to choose)"
  fi
fi

CHAT=
if [ -n "$MODELS_JSON" ]; then
  if jq -e --arg m "$MODEL" '.data[] | select(.id == $m)' <<<"$MODELS_JSON" >/dev/null; then
    CHAT=1
  else
    bad "model selection" "'$MODEL' is not in /v1/models; pass --model ID from that list"
  fi
fi


# ---- streamed chat ------------------------------------------------------------------------
if [ -n "$CHAT" ]; then
  jq -nc --arg m "$MODEL" \
    '{model:$m, thinking:{type:"disabled"},
      messages:[{role:"user",content:"Reply with exactly the word: ready"}],
      stream:true, stream_options:{include_usage:true}, max_tokens:64}' > "$TMP/stream.json"
  STATUS=$(curl -sS -o "$TMP/sse" -w '%{http_code}' --max-time "$TIMEOUT" \
           -H 'content-type: application/json' -H 'accept: text/event-stream' \
           "${CURL_AUTH[@]}" --data-binary @"$TMP/stream.json" \
           "$URL/v1/chat/completions") || STATUS=000
  note "POST /v1/chat/completions (stream) -> $STATUS"
  if [ "$STATUS" != 200 ]; then
    bad "streamed chat" "HTTP $STATUS: $(head -c 300 "$TMP/sse")"
  else
    # SSE: 'data: {...}' events, then 'data: [DONE]'. Parse each event with jq.
    grep -q '^data: \[DONE\]' "$TMP/sse" && HAS_DONE=1 || HAS_DONE=0
    sed -n 's/^data: //p' "$TMP/sse" | grep -vE '^\[DONE\]$|^[[:space:]]*$' > "$TMP/events" || true
    NEVENTS=$(wc -l < "$TMP/events" | tr -d ' ')
    if ! jq -R -c 'fromjson? // error("event is not JSON")' "$TMP/events" > "$TMP/events.json" 2>/dev/null; then
      bad "streamed chat" "an SSE data event is not JSON: $(head -5 "$TMP/events")"
    else
      NPJSON=$(wc -l < "$TMP/events.json" | tr -d ' ')
      FINISH=$(jq -rs 'map(.choices[0].finish_reason // empty) | .[-1] // ""' "$TMP/events.json")
      CONTENT=$(jq -rs '[.[] | .choices[0].delta.content // empty] | join("")' "$TMP/events.json" | tr -d '"')
      MODEL_SEEN=$(jq -rs '.[0].model // empty' "$TMP/events.json")
      USAGE=$(jq -rs 'map(select(.usage != null) | .usage) | .[-1] // empty' "$TMP/events.json")
      PROBLEMS=""
      [ "$NEVENTS" -lt 2 ] && PROBLEMS="$PROBLEMS only $NEVENTS data event(s);"
      [ "$NEVENTS" != "$NPJSON" ] && PROBLEMS="$PROBLEMS $((NEVENTS-NPJSON)) event(s) dropped in parse;"
      [ "$HAS_DONE" != 1 ] && PROBLEMS="$PROBLEMS no [DONE];"
      [ -z "$CONTENT" ] && PROBLEMS="$PROBLEMS empty joined delta.content;"
      [ -z "$FINISH" ] && PROBLEMS="$PROBLEMS no finish_reason;"
      [ -z "$USAGE" ] && PROBLEMS="$PROBLEMS no usage event (stream_options.include_usage);"
      if [ -z "$PROBLEMS" ]; then
        ok "streamed chat"
        note "events=$NEVENTS finish=$FINISH model=$MODEL_SEEN content=${CONTENT:0:80} usage=$(jq -c . <<<"$USAGE")"
      else
        bad "streamed chat" "$PROBLEMS"
      fi
    fi
  fi
fi

# ---- required tool call (whole reply) ------------------------------------------------------
if [ -n "$CHAT" ]; then
  jq -nc --arg m "$MODEL" \
    '{model:$m, thinking:{type:"disabled"},
      messages:[{role:"user",content:"What is the weather in Paris, in celsius?"}],
      tools:[{type:"function",function:{name:"get_weather",
        description:"Current weather for a city",
        parameters:{type:"object",
          properties:{city:{type:"string"},days:{type:"integer"},
                      units:{type:"string",enum:["c","f"]}},
          required:["city"]}}}],
      tool_choice:"required", parallel_tool_calls:false, max_tokens:256}' > "$TMP/tools.json"
  req POST /v1/chat/completions -H 'content-type: application/json' --data-binary @"$TMP/tools.json"
  if [ "$STATUS" != 200 ]; then
    bad "required tool call" "HTTP $STATUS: $(head -c 300 "$TMP/body")"
  else
    TC_OK=$(jq -r '
      .choices[0] as $c
      | if $c.finish_reason == "tool_calls"
           and ($c.message.tool_calls | type == "array" and length >= 1)
           and ($c.message.tool_calls[0]."function".name == "get_weather")
           and (($c.message.tool_calls[0]."function".arguments | fromjson? // {}).city
                | type == "string" and length > 0)
        then "ok" else "bad" end' "$TMP/body" 2>/dev/null)
    if [ "$TC_OK" = ok ]; then
      ok "required tool call"
      note "args=$(jq -c '.choices[0].message.tool_calls[0]."function".arguments' "$TMP/body")"
    else
      bad "required tool call" "finish/tool_calls/city not as required: $(jq -c '.choices[0] | {finish_reason, tool_calls: .message.tool_calls}' "$TMP/body" 2>/dev/null | head -c 400)"
    fi
  fi
fi

# ---- an error is an error body, in the API's shape -------------------------------------------
if [ -n "$CHAT" ]; then
  req POST /v1/chat/completions -H 'content-type: application/json' \
    -d '{"model":"definitely-not-a-served-model-id","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
  if [ "$STATUS" = 404 ] || [ "$STATUS" = 400 ]; then
    if grep -qi 'error\|does not exist\|not.*served\|model' "$TMP/body"; then
      ok "unknown model refused ($STATUS)"
      note "$(head -c 300 "$TMP/body")"
    else
      bad "unknown model refused" "HTTP $STATUS but body has no error text: $(head -c 200 "$TMP/body")"
    fi
  else
    bad "unknown model refused" "HTTP $STATUS (wanted 400 or 404): $(head -c 300 "$TMP/body")"
  fi
fi

# ---- opt-in ASR -----------------------------------------------------------------------------
if [ -n "$ASR_FILE" ]; then
  ASR_FORM=(-F "file=@$ASR_FILE")
  [ -n "$ASR_MODEL" ] && ASR_FORM+=(-F "model=$ASR_MODEL")
  req POST /v1/audio/transcriptions "${ASR_FORM[@]}"
  if [ "$STATUS" != 200 ]; then
    bad "ASR transcription" "HTTP $STATUS: $(head -c 300 "$TMP/body")"
  elif ! jq -e '.text | type == "string" and length > 0' "$TMP/body" >/dev/null 2>&1; then
    bad "ASR transcription" "no non-empty .text: $(body_json | head -c 300)"
  else
    ok "ASR transcription"
    note "text=$(jq -r '.text[0:120]' "$TMP/body") usage=$(jq -c '.usage // empty' "$TMP/body")"
  fi
else
  skip "ASR transcription" "no --asr FILE (pass a WAV/FLAC/MP3/Ogg clip to enable)"
fi

# ---- opt-in OCR -----------------------------------------------------------------------------
if [ -n "$OCR_FILE" ]; then
  MIME=$(file --mime-type -b "$OCR_FILE" 2>/dev/null || echo application/octet-stream)
  case $MIME in image/*) ;; *) MIME=image/png ;; esac
  B64=$(base64 < "$OCR_FILE" | tr -d '\n')
  jq -nc --arg m "$OCR_MODEL" --arg url "data:$MIME;base64,$B64" \
    '{model:$m, skip_special_tokens:false,
      messages:[{role:"user",content:[
        {type:"text",text:"<PAGE><image>\nMulti page parsing."},
        {type:"image_url",image_url:{url:$url}}]}],
      max_tokens:512}' > "$TMP/ocr.json"
  req POST /v1/chat/completions -H 'content-type: application/json' --data-binary @"$TMP/ocr.json"
  if [ "$STATUS" != 200 ]; then
    bad "OCR chat completion" "HTTP $STATUS: $(head -c 300 "$TMP/body")"
  elif ! jq -e '.choices[0].message.content | type == "string" and length > 0' "$TMP/body" >/dev/null 2>&1; then
    bad "OCR chat completion" "no non-empty content: $(body_json | head -c 300)"
  else
    ok "OCR chat completion"
    note "content=$(jq -r '.choices[0].message.content[0:120]' "$TMP/body")"
  fi
else
  skip "OCR chat completion" "no --ocr FILE (pass a page image to enable)"
fi

echo "$PROG: $PASS passed, $FAILED failed, $SKIPPED skipped"
[ "$FAILED" = 0 ]
