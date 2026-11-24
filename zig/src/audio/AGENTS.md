# `src/audio/` instructions

Extends `src/AGENTS.md` and the root `AGENTS.md`. An audio file's bytes to 16 kHz mono samples and log-mel
features, on the CPU (spec 138 WS-02). Host only: it imports no other layer of zspark's (`tools/check_imports.zig`);
`server/` and `models/` import it. The reference is the `qwen-asr` 0.0.6 package's `normalize_audio_input` and
transformers 4.57.6's `WhisperFeatureExtractor`; each file's header says what it follows and how near.

## Files

- `audio.zig`: `clip(bytes, max_seconds)`, the whole path (decode, channels averaged, resample, peak-normalize and
  clip to [-1, 1], features); `message` maps an error to the route's vLLM-shaped 400 text.
- `decode.zig`: the format is detected from the bytes. dr_wav reads WAV (RIFF, RF64, RIFX and Wave64) and
  PCM AIFF. dr_flac reads FLAC, dr_mp3 reads MP3 with LAME delay and padding trimmed, and stb_vorbis reads
  Ogg Vorbis. `build.zig`'s `audioLibrary` compiles the vendored decoders through `dr_libs.c` and `vorbis.c`.
- `codecs.rs`: the static Symphonia adapter reads AAC-LC and ALAC from MP4/M4A, ADTS AAC-LC, and Opus from
  Ogg or WebM. The adapter lends decoded frames to `decode.zig` for channel averaging and limits. Its dependencies
  are pinned by `Cargo.lock` and built by `build.zig`'s `audioCodecs`.
- `resample.zig`: librosa's default `soxr_hq` approximated by one Kaiser-windowed rational polyphase sinc of its
  band edges (libsoxr is LGPL: neither vendored nor ported); soxr's output length and time origin.
- `mel.zig`: 128 Slaney-scale bins, `n_fft` 400, hop 160, reflect padding, periodic Hann, a mixed-radix f32 FFT,
  `log10` floored, the clip's max less 8, `(x + 4) / 4`, the last frame dropped. The maximum is the clip's, so the
  features are made once a clip, never streamed. The model casts them to bf16 (`models/qwen3_asr/tower.zig`).

- `split.zig`: vLLM's `split_audio` (`vllm/multimodal/audio.py`): a clip's pieces of at most N seconds, each cut
  at the start of the quietest 1,600-sample stride of its last second (the region's last stride unread, the first
  of equal ones kept), the RMS summed pairwise as numpy does on float32.
- `text.zig`: the text helpers the ASR routes' prompts share (spec 151): Python's `str.strip()` and
  `stripTags`, the fixpoint removal of ChatML-like tags from a client's fields (vLLM's transcription-route
  sanitizing). `tests/audio/text_test.zig`.

## Tests and fixtures

`zig/tests/audio/audio_test.zig`, the area `tools/check.sh audio` (no GPU). Its cases come from
`tools/reference/audio_cases.py` (the package's own decode, resample and features on generated and recorded
clips); tolerances are in the test, against soxr and librosa, not bits. `split_test.zig` holds the cuts to vLLM's
code (`tools/reference/audio_split_cases.py`): a generated signal and the WER set's clips past 30 s, exact.

## When editing here

- Do: keep a decoder's provenance in its vendor directory (an import is upstream's bytes; a change is a row).
- Do: say a departure from the package or vLLM in the file's header and the spec entry.
- Do not: decode by file extension, add a GPL/LGPL dependency, or make features depend on anything but the clip.
