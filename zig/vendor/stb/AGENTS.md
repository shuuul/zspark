# `zig/vendor/stb/` instructions

This file extends the root `AGENTS.md`. stb_image (spec 019 WS-07): the JPEG decoder of `zig/src/image/`, and
its BMP, TGA, PSD and PNM ones (WS-04, `stb.zig`), compiled from `zig/src/image/stb_image.c` with the options
there. stb_vorbis (spec 138 WS-02): the Ogg Vorbis decoder of `zig/src/audio/`, compiled from
`zig/src/audio/vorbis.c`. `UPSTREAM` names the commit, `MANIFEST.sha256` the import, `PATCHES.md` every
difference from it.

## Rules

- An import is upstream's bytes and nothing else, in a commit of its own, written by
  `tools/vendor_c.sh stb DIR` (the manifest too, never by hand); `tools/vendor_c.sh --compare stb DIR` checks
  the files against a checkout.
- A local change is one commit with its row in `PATCHES.md` and its test; prefer an option in
  `zig/src/image/stb_image.c` to a patch here.
- `zig fmt` and the source budget do not hold here.
- A new release is read for what changes the JPEG path before it is taken: `zig/tests/image/` holds the
  pixels against Pillow's and the damaged files; run `tools/check.sh image` after it. stb_vorbis's samples
  are held against libsndfile's in `zig/tests/audio/` (`tools/check.sh host audio`).
