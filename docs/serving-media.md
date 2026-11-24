# Serving images, OCR and audio

The models that read more than text, and the routes they answer: Flash Next's image parts, Unlimited-OCR-2's
`/v1/chat/completions` with page images, and Qwen3-ASR's `/v1/audio/transcriptions`. Split from
[serving.md](serving.md) in spec 142 WS-02, which holds the server's other routes and flags; what a checkpoint
decides is on its page under [`recipes/`](recipes/README.md).

## Images

Spec 019 WS-04 and WS-05. A message's image parts are read on `nvidia/Qwen3.8-Flash-Next-FP4-Local` alone, at
any `--parallel`: chat completions' `image_url` parts, Responses' `input_image` and Messages' `image` blocks. The
log says at load whether they are on (`images: on, 4 images and 4096 image tokens a request at most`), and why
not. GLM and two-rank serving reject image parts with HTTP 400, `this server accepts text parts only; image, audio and video inputs are
unsupported`. The vision tower loads beside the model at start (its weights as stored, spec 019 WS-08).

- Where the bytes come from: a `data:` URL always; a public `https` URL only under `--vision-urls`
  (TensorFold's bounds: the bytes, a deadline, redirects; no `http`, no `file`). Without the flag an `https`
  URL is a 400, `image URLs are off on this server; send the image as a data URL, or start the server with
  --vision-urls`; any other scheme `images require data URLs or public HTTPS URLs`.
- What is decoded follows vLLM (`vllm/multimodal/media/image.py` `load_bytes`), not TensorFold: the format is
  sniffed from the bytes, whatever the URL's or the media type's name says. JPEG, PNG, GIF, WebP, BMP, TGA, PSD
  (RGB colour) and PPM/PGM (binary, P5 and P6) are decoded; an animated GIF, PNG or WebP gives its first frame;
  the EXIF orientation is applied; alpha goes over white. The bounds are TensorFold's: 10 MB of bytes, 8192
  pixels a side and 16,777,216 pixels a picture (vLLM's own pixel cap is 178,956,970).
- A request holds 4 images and 4,096 image tokens at most (TensorFold 0.6.5's defaults; the four other
  `--vision-*` flags stay refused); a `low` detail is 256 tokens at most, as the host gives it. The picture is resized
  and cut into patches as 0.6.5's processor does, and its tokens take the template's place for
  it.
- A refusal is a 400 in each API's own error shape, in vLLM's words where vLLM has them: `Failed to load
  image: cannot identify image file` (bytes no decoder knows), `Failed to load image: image file is truncated or
  damaged`, `Failed to load image: image is empty or exceeds the encoded byte limit of 10 MB`, `Image dimensions
  exceed the maximum of 8192 pixels a side or 16777216 pixels`, `absolute aspect ratio must be smaller than 200`,
  and `a request may hold at most 4 images (--vision-max-images); it has N` (the host's).
- Gaps, formats vLLM's Pillow opens and no vendored decoder reads, each a 400 that names it, `Failed to load
  image: X images are not decoded by this server; send JPEG, PNG, GIF, WebP, BMP, TGA, PSD or PPM`: TIFF, AVIF,
  HEIC, ICO, JPEG 2000, QOI, DDS, PNM in text or 1-bit form (P1 to P4), a PSD of another colour mode, and a CMYK
  JPEG (as TensorFold 1.0.12).
- An image prompt keeps no conversation state: under `--parallel 1`, or as the one live stream, the turn runs
  from an empty state, and the next one too. Beside other streams (WS-05) it may go on from a kept state of
  text only, at or before its first image token, as vLLM's prefix cache keys its blocks before the image; no
  state past that token is kept, so the next turn runs the image's rows again. Its reply is the one it gets
  alone, to the bit.

## OCR

`zspark serve ~/models/Unlimited-OCR-2` serves `baidu/Unlimited-OCR-2` (spec 147;
[`recipes/unlimited-ocr-2.md`](recipes/unlimited-ocr-2.md)) on `POST /v1/chat/completions` through
TensorFold 1.3.4's host. `/v1/models` names the checkpoint directory (`Unlimited-OCR-2`), or `--name`.
`zspark stack --ocr DIR` starts it beside Flash Next and ASR
([serving.md](serving.md#flash-next-ocr-and-asr-from-one-command)).

The engine (`zig/src/models/unlimited_ocr/served.zig`) follows the checkpoint's `infer` and the SGLang
wheel its README uses:

- Images are decoded as above, with EXIF orientation applied and alpha over white. A `detail: low` part
  is refused because the smallest supported view has 273 tokens. A literal `<image>` in a text-only
  request stays text under the host's control-token masking.
- Replies contain tagged text (`<|ref|>type<|/ref|><|det|>[x0, y0, x1, y1]<|/det|>text`, coordinates
  0 to 1000). `skip_special_tokens` defaults to true; false keeps the tags. Thinking is off.
- Decode is greedy, one reply at a time (`--parallel` above 1 refused), with no shared prefix between
  requests. `--rows` sets the prompt window (1,024); the row's bits are independent of its window.

- Up to 128 images a request, each a page. The prompt is `<PAGE><image>` a page, then the message's text, or
  "Multi page parsing." where it has none: a text with no `<image>` gets the pages before it, one `<image>` among
  several pages is replaced by all of them, as many `<image>` as pages stay where they are (the README's own
  `<PAGE><image>` a page), and another count is a 400 (`infer`'s error). The template is
  `models/unlimited_ocr/chat_template_v2.jinja`.
- One picture is cut in gundam mode, at most 16 tiles; several pages are base mode, each its 1,024 global view
  alone (273 tokens), as the HF processor's auto policy, where the request names no mode. SGLang's request fields
  name one as its v2 wheel reads them: `images_config.image_mode` (`auto` or `gundam`, `base`;
  stripped, any case; `tiny`, `small`, `large` are refused as not served, another value is a 400 "Unknown
  image_mode"), else `images_config.crop_mode` (true gundam, false base); `mm_processor_kwargs` is read as
  `images_config` beneath it. Neither object given: the auto policy above.
- `custom_params` (SGLang's `custom_logit_processor` parameters, read with or without a processor named, as
  SGLang injects its own): `max_page_tokens` may lower the page cap (never raise it), and `rep_min_size`,
  `rep_max_size`, `rep_min_count` the periodic tail rule: where `rep_max_size` > 0 and `rep_min_count` >= 2, a
  reply whose tail repeats a span of those sizes that many times is ended (SGLang's `_tail_has_repeat`). A value
  that is not a whole number turns all three off, as SGLang's `int()` failing does; a `custom_params` that is not
  an object is a 400.
- The reply writes a `<PAGE>` before each page. The PageCountGuard holds it to the prompt's count: the end token
  is banned while pages are owed, a marker too many forces the end, and a page of `max_page_tokens` tokens (20,000
  by default; top level or under `vllm_xargs`, a negative value 0, 0 no cap) forces the next `<PAGE>` or, on the
  last page, the end. A request with `ignore_eos` has no guard.
- The n-gram rule reads the reply alone; neither field given is 35 in 128 for one
  page, 35 in 1,024 for several (the README's request). `ngram_size` and `window_size` are read from
  the body's top level or `vllm_xargs`: a size alone takes window 128, a window alone disables the rule.
- The context is 131,072 tokens and a reply that names no `max_tokens` takes the room the context leaves (`infer`'s).
- A long document (spec 147 WS-08, `models/unlimited_ocr/parts.zig`, after SGLang's `doc_split.py` and
  `uocr_doc_reuse.py`): while its pages and their replies fit the state, one window, the request as it stands.
  Else parts of consecutive pages (`doc_split_options.max_batch_pages`, else `ZSPARK_UNLIMITED_OCR_2_PAGE_WINDOW`,
  else the pages the state holds), each reading the next part's first page too and ending on the `<PAGE>` past its
  own; a part after one that ended so takes over its last page's last 128 image rows, its lookahead page and its
  last text rows (keys copied, not read again) and goes on from a forced `<PAGE>`, its RoPE pages the document's.
  The client sees one reply. `overlap_pages` above 0 is a 400. The host takes a prompt up to the state's tokens and
  128 of the largest pages more (`context_window` 391,424 at the default), so a document past the state reaches
  the parts.

```bash
curl localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "Unlimited-OCR-2", "skip_special_tokens": false, "ngram_size": 35, "window_size": 128,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "<PAGE><image>\nMulti page parsing."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}]}]}'
```

## Audio

Spec 138. `zspark serve ~/models/Qwen3-ASR-1.7B` serves `POST /v1/audio/transcriptions` in vLLM's shape (its
`speech_to_text` protocol), and `/v1/chat/completions` with audio parts as vLLM serves this model (spec 150,
"Chat" below). `/v1/models` lists
`Qwen3-ASR-1.7B` (the directory's name, or `--name`). `zspark stack --asr DIR` starts it beside Flash Next and
OCR (the stack's is the NVFP4 build [`qwen3-asr-1.7b-nvfp4.md`](recipes/qwen3-asr-1.7b-nvfp4.md) under `--asr-name Qwen3-ASR-1.7B`), and the router sends every `/v1/audio/` path to it ([serving.md](serving.md#different-models-behind-one-address)). The model's page is
[`qwen3-asr-1.7b.md`](recipes/qwen3-asr-1.7b.md).

- Request: `multipart/form-data` with `file` (required) and optional `model`, `language`, `prompt`,
  `response_format`, `temperature`, `stream` and `max_completion_tokens`; other fields are ignored, as FastAPI's
  form model ignores them. A `model` that is sent and not served is vLLM's 404 (`The model X does not exist.
  Valid aliases: [...]`).
- Response: `json` (the default) and `text` both answer `{"text": ..., "usage": {"type": "duration", "seconds":
  N}}`, N the clip's seconds rounded up, compact UTF-8 as FastAPI writes it. `text` is the transcript alone, with
  the model's `language X<asr_text>` head removed. No `language` field is returned.
- `verbose_json` and `diarized_json` are 400s as vLLM refuses them for this model; so are `stream` (`Streaming is
  not supported for this model`) and a `temperature` above 0 (only greedy decoding is served).
- Audio formats are detected from the bytes and converted to 16 kHz mono. Supported inputs include WAV
  (RF64, RIFX and Wave64), FLAC, MP3, Ogg Vorbis, M4A/MP4 with AAC-LC or ALAC audio, ADTS AAC-LC,
  Ogg/WebM Opus, and PCM AIFF. Unsupported containers or codecs such as HE-AAC return
  `Invalid or unsupported audio file.` A damaged file returns `Invalid or corrupted audio data...`.
  AAC in MP4 and Opus in WebM may include codec priming or padding in their decoded duration.
  By default a clip up to 1,200 s runs whole, as the package runs it. Longer audio returns
  `Audio exceeds maximum allowed duration.`. `serve --max-audio-clip-s N` (vLLM's `max_audio_clip_s`, which vLLM
  takes from the checkpoint's `chunk_length`, 30, with no switch) cuts a clip longer than N seconds as vLLM does:
  pieces of at most N s, each cut at the quietest 100 ms of its last second, run at once (up to `--parallel`, 16)
  as requests of their own with no shared context, their texts joined with nothing when `language` is Chinese or
  Japanese and with a space otherwise (a free request too). One request still counts once in `/metrics`. The file is at most 25 MiB (`Maximum file size exceeded (parameter=audio_filesize_mb, value=N)`);
  the host's body cap is 32 MiB. Refusals are vLLM's `{"error": {"message", "type", "param", "code"}}` with 400
  (`BadRequestError`), 404, 500 or 503 (the queue, 64 waiting, is full).
- Prompt: the `qwen-asr` package's, not vLLM's: the system turn is always written, holding `prompt` (the page
  context aimd sends; `<|...|>` tokens and `<asr_text>` are stripped from it until none is left), the user turn
  holds the audio, and a `language` appends `language X<asr_text>` to the assistant's turn so the model need not
  guess. The reply is read as the package reads it (`parse_asr_output`): stripped, runs of repeats folded, the text
  after `<asr_text>` kept.
- `language`: an ISO 639-1 code (what OpenAI's clients and aimd send; vLLM's table) or the model's name for it,
  in any case (`en`, `English`, `chinese`), and `yue` and `fil`; blank means detect. Anything else is `Unsupported
  language: 'X'.  Must be one of [...]`.
- Decoding is greedy up to 4,096 tokens (`max_completion_tokens` lowers it) and the context is 20,208 tokens
  (`--context` changes it; a clip past it is a 400). One transcription runs at a time, the rest wait; a client that
  leaves cancels its own. `--rows N` sets the prompt window (2,048).
- aimd sends `model`, `file` (a segment of up to 600 s, about 19 MB), `language` and `prompt`, reads only `text`
  and keeps what follows `<asr_text>` if it is there. aimd can send M4A and AAC segments directly.

```bash
curl localhost:8000/v1/audio/transcriptions -F model=Qwen3-ASR-1.7B -F file=@clip.m4a -F language=en
```

### Chat

Spec 150. `/v1/chat/completions` takes the clip as a content part, as vLLM's chat does for this model:
`{"type": "input_audio", "input_audio": {"data": BASE64, "format": "wav"}}` or `{"type": "audio_url", "audio_url":
{"url": "data:audio/wav;base64,..."}}` (an `https://` URL is fetched only under `serve --vision-urls`, as an
image's is; other schemes are refused). The checkpoint's `chat_template.json` makes the prompt: every system
message's text is the context (as the route's `prompt`), each audio part becomes `<|audio_start|><|audio_pad|>
<|audio_end|>` in the user turn, its pad the clip's rows, and the user's text is dropped. With no system text the
prompt is the route's, token for token.

- The reply is the model's text as it is, `language English<asr_text>...`, not parsed (a client keeps what follows
  `<asr_text>`); streamed or whole; `usage.prompt_tokens` counts the audio rows, as vLLM's does. It is the
  message's `content`: `serve` turns thinking off whatever `--thinking` says, as vLLM runs the card with no
  reasoning parser (a request's `chat_template_kwargs` `enable_thinking: true` still moves it to
  `reasoning_content`, as on Unlimited-OCR-2).
- Greedy only: a `temperature` above 0 is a 400 (vLLM samples; the route and the checkpoint's
  `generation_config` are greedy). Stop strings, `max_tokens` and `ignore_eos` are read as on any chat.
- Audio as on the route (formats, 25 MiB a part as `Maximum file size exceeded`), any number of parts up to 999
  (vLLM's `At most 999 audio(s) may be provided in one prompt.`), 1,200 s in all; no `--max-audio-clip-s` cut,
  as vLLM cuts on the route alone. An audio part in a system message, and image or video parts, are 400s.
- A text-only chat and `/v1/completions` are answered, as vLLM answers them for this model.
- Behind the router a chat naming `Qwen3-ASR-1.7B` goes to the ASR copy by its `model`.

```bash
# the body in a file: a long clip's base64 is past a command line's limit
printf '{"model":"Qwen3-ASR-1.7B","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"%s","format":"wav"}}]}]}' \
  "$(base64 -w0 clip.wav)" > body.json
curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' --data-binary @body.json
```

## Nemotron-3.5-Transcribe

Spec 151 WS-08. `zspark serve "$ZSPARK_NEMOTRON_NVFP4" --name nemotron-transcribe` serves
`POST /v1/audio/transcriptions` in vLLM's shape (the same `speech_to_text` protocol as above), and
`/v1/chat/completions` and `/v1/completions` with `audio_url`/`input_audio` parts, as vLLM 0.32's NeMo
SpeechLM plugin serves this checkpoint. `/v1/models` lists the directory's name, or `--name`. Own flags:
`--rows N` (prompt window rows, 2,048), `--drafter DIR` (the `NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-DFlash2`
checkpoint drafting in the MTP head's place — the default drafter, looked up under `~/models` when the
flag is absent and refused with the tried path when the directory is missing), `--drafter-stored` (the
drafter's linears BF16 as the checkpoint stores them), `--mtp` (the MTP head drafting instead),
`--serial` (no drafting at all), `--cap N` (the most drafts of a round). Two ranks serve the same engine —
`--rank 0|1 --at ADDR:PORT`, rank 1 listening, Flash/GLM's convention: each process holds half the
weights, rank 0 runs the routes and decoding and tells each operation over the join's channel.
Rank 1 encodes the same decoded PCM and replays operations on its share. `--parallel` is
not read: the decoder's state and the perception are one, one reply at a time, the rest queued
(`--max-queued`, 64).

- Transcription: the same form and the same answers as the Qwen3-ASR route above (`json`/`text`,
  `verbose_json`/`diarized_json` and `stream` 400, temperature above 0 400), but `prompt` becomes the user
  turn's instruction and `language` writes `Language: X.` into it, and the reply's text is what the
  `nemotron_v3` reasoning parser's content is (thinking off by default, below). `language` is a name the
  checkpoint's `prompt` table knows; anything else is a 400 as vLLM's refusal.
- Audio: the formats and 25 MiB above, but a clip runs whole to 3,600 s (the plugin's bound) and is never
  cut — the ParallelExpertEncoder's own online windows segment it; past 3,600 s is
  `Audio exceeds maximum allowed duration.`. Each clip's perception rows replace the prompt's one
  `<|audio|>` placeholder row (the checkpoint's `audio_locator_tag`, token 131,072); one audio part per
  request, as in the card's launch configuration.
- Prompt: the checkpoint's `nemotron3p5` template render — an empty system turn, the user turn holding the
  instruction and `<|audio|>` before its end, and the assistant opener closed by `<think></think>`:
  thinking is off by default (`serve` makes the host's `--thinking` default false; an explicit
  `--thinking`/`--no-thinking` is honoured, and a request's `chat_template_kwargs` `enable_thinking` moves
  the reasoning to `reasoning_content` as on the other models). The reply's text is the reasoning split's
  content (`prompt.split`/`transcript`). The default context is 65,536 tokens (`--context` changes it);
  the prompt and reply share that window. Transcription decodes greedily until EOS or the remaining
  context is exhausted; `max_completion_tokens` can lower the reply limit.
- Chat: the host renders the checkpoint's template; each audio part's bytes are decoded and resampled as
  the route's file (`prepare` says the rows), its placeholder expanded to that many `<|audio|>` ids, and
  the reply streams token by token. An audio part in a system message, and image or video parts, are 400s,
  as on the ASR route. The reply is not parsed for `<asr_text>` — this checkpoint's grammar ends turns at
  `<|im_end|>` and the reasoning split is the answer's shape.

```bash
curl http://127.0.0.1:8888/v1/audio/transcriptions -F model=nemotron-transcribe -F file=@clip.m4a -F language=en
```
