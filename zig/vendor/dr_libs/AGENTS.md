# `zig/vendor/dr_libs/` instructions

This file extends the root `AGENTS.md`. dr_wav, dr_flac and dr_mp3 (spec 138 WS-02): the WAV, FLAC and MP3
decoders of `zig/src/audio/`, compiled from `zig/src/audio/dr_libs.c` with the options there into the audio
library (`build.zig`, `audioLibrary`). `UPSTREAM` names the commit, `MANIFEST.sha256` the import, `PATCHES.md`
every difference from it (none).

## Rules

- An import is upstream's bytes and nothing else, in a commit of its own, written by
  `tools/vendor_c.sh dr_libs DIR` (the manifest too, never by hand); `tools/vendor_c.sh --compare dr_libs DIR`
  checks the files against a checkout.
- A local change is one commit with its row in `PATCHES.md` and its test; prefer an option in
  `zig/src/audio/dr_libs.c` to a patch here.
- `zig fmt` and the source budget do not hold here.
- A new commit is read for what changes decoding (the samples, an MP3's encoder delay and padding, a refusal)
  before it is taken: `zig/tests/audio/` holds the samples against libsndfile's; run `tools/check.sh host audio`
  after it.
