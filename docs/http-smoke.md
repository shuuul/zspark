# HTTP smoke

`tools/http_smoke.sh` checks that a running `zspark serve`, or the `zspark stack` router,
answers as [serving.md](serving.md) documents. It is a client: it starts nothing, stops
nothing and changes no server state. Dependencies are `curl`, `jq` and coreutils — no
Python, no build, no GPU.

```bash
# stack router or a lone `zspark serve`, default address
tools/http_smoke.sh

# another address / model / deadline
tools/http_smoke.sh --url http://127.0.0.1:8000 --model qwen3.8-flash-next --timeout 120
```

`--url` defaults to `http://127.0.0.1:8888`, the router's and `serve`'s default listener.
`--model` defaults to the first id `GET /v1/models` lists; name one explicitly when the
address serves several models so the chat checks go to the engine you intend.

## What it verifies

Each check prints `PASS`, `FAIL` (with the server's own error text) or `SKIP`, and the
script exits non-zero on any failure:

- `GET /health`: HTTP 200 and an ok marker — a lone `serve`'s `{"status": "ok"}` body or
  the router's `{"ok": true}`.
- `GET /v1/models`: a 200 OpenAI `{"object": "list", "data": [...]}` with non-empty string
  ids, and that `--model` is among them.
- `POST /v1/chat/completions` with `stream: true`: HTTP 200, more than one
  `data:` event each parsing as a chat-completion chunk, non-empty joined `delta.content`,
  a `finish_reason`, a `usage` event (requested with `stream_options.include_usage`), and a
  terminating `data: [DONE]`.
- `POST /v1/chat/completions` with `tool_choice: "required"` (a whole reply): a
  `finish_reason` of `tool_calls`, a `get_weather` call, and `arguments` parsing as JSON
  with a non-empty string `city`.
- An error check: a chat request naming a model id that is not served must come back 400 or
  404 with an error body — an unknown name is refused, not silently routed.

## Optional media checks

ASR and OCR are exercised only when you pass a file — the smoke never guesses a media
engine and reports SKIP otherwise.

```bash
# ASR: a supported audio clip such as WAV or M4A (<= 25 MiB, within the engine's clip limit)
tools/http_smoke.sh --asr clip.flac --asr-model Qwen3-ASR-1.7B

# OCR: one page image (PNG/JPEG/etc., <= 10 MiB)
tools/http_smoke.sh --ocr page.png --ocr-model Unlimited-OCR-2
```

- `--asr FILE` posts `FILE` as `multipart/form-data` to `POST /v1/audio/transcriptions`
  (see [serving-media.md](serving-media.md#audio)) and requires a 200 with non-empty
  `.text`. `--asr-model` sends the `model` form field; leave it unset to let the router's
  `--audio` backend answer without an id. A `model` the ASR engine does not serve is that
  engine's 404 — pass its `/v1/models` id.
- `--ocr FILE` embeds the image as a `data:` `image_url` part in a chat completion against
  `--ocr-model` (default `Unlimited-OCR-2`; see [serving-media.md](serving-media.md#ocr))
  and requires a 200 with non-empty `choices[0].message.content`. On the stack router this
  also proves `model`-based routing reaches the OCR backend.

## Authentication and timeouts

When the server asks for a key (`serve --api-key`, `--api-key-file` or
`TENSORFOLD_API_KEY`), give it through `--api-key`, `--api-key-file PATH`, or the
`ZSPARK_API_KEY` environment variable. The key is sent as an `Authorization: Bearer`
header through a private curl config file — it is never printed and never appears on the
process list; prefer the file or environment so the key stays off your shell's argv.

`--timeout SECONDS` (default 120) is each request's deadline. Raise it for a first request
against a cold engine. The environment forms `ZSPARK_SMOKE_URL`, `ZSPARK_SMOKE_MODEL`,
`ZSPARK_SMOKE_TIMEOUT`, `ZSPARK_SMOKE_ASR_MODEL` and `ZSPARK_SMOKE_OCR_MODEL` hold the same
values as the flags.
