# `zig/vendor/libwebp/` instructions

This file extends the root `AGENTS.md`. libwebp's decoder (spec 019 WS-07): the WebP decoder of
`zig/src/image/`, compiled whole into the image library by `build.zig` (`image_c`). `UPSTREAM` names the commit,
`MANIFEST.sha256` the import, `PATCHES.md` every difference from it (none).

## Rules

- An import is upstream's bytes and nothing else, in a commit of its own, written by
  `tools/vendor_c.sh libwebp DIR` (the manifest too, never by hand); `tools/vendor_c.sh --compare libwebp
  DIR` checks the files against a checkout. The list is the compiler's: the decoder's sources for aarch64 and
  the headers they include, so a release that adds a source or a header changes it.
- A local change is one commit with its row in `PATCHES.md` and its test; prefer a define in
  `build.zig` to a patch here.
- `zig fmt` and the source budget do not hold here.
- A new release is read for what changes the decoder before it is taken: `zig/tests/image/` holds the
  pixels against Pillow's and the damaged files; run `tools/check.sh image` after it.
