# Patches

What differs from upstream (`UPSTREAM`) in this directory: one row a change, in the commit that makes it.
`zig build check-specs` refuses a file whose bytes are not `MANIFEST.sha256`'s without a row here, and a row
whose file is upstream's as it was. The path is from this directory.

| File | What | Why | Proposed upstream |
| --- | --- | --- | --- |
| `stb_image.h` | `stbi__resample_row_h_2`: the last column's left output is `(3 * in[w-1] + in[w-2] + 2) / 4` | Upstream wrote `3 * in[w-2] + in[w-1]`, weighting the wrong sample: a 4:2:2 JPEG's last column was up to 107 levels from Pillow's (libjpeg-turbo's `h2v1_fancy_upsample`) on `jpeg422`, 37 pixels wide; 3 after, as the other cases `zig/tests/image/image_test.zig` (spec 019 WS-07) | Yes |
| `stb_image.h` | `stbi__parse_entropy_coded_data`: continue a non-interleaved progressive scan while buffered bits or an EOB run remain after `nomore` | In v2.31 `nomore` also means a marker was prefetched with entropy bits still buffered. Skipping those MCUs loses coefficients and can make a later refinement scan fail: `jpeg_gray_progressive` (16 x 9, one component) was refused instead of decoded within the existing Pillow bounds (`zig/tests/image/image_test.zig`, spec 167) | Yes |
| `stb_vorbis.c` | `start_page_no_capturepattern` exempts the all-ones granule position from the sample-position bound | Ogg uses this sentinel on pages that complete no packets. The memory decoder reads these pages while assembling packets that span pages (spec 175 WS-02). | No |
