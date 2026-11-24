"""Pillow's `Image.resize(BICUBIC)` and `ImageOps.pad` on made pictures, for `zig/src/image/resize.zig` (spec 137 WS-02).

Each case is a random RGB picture of a small size (rows of three bytes, base64) and Pillow's result (its size and
the SHA-256 of its bytes): resizes that shrink, grow, keep a side and meet a kernel wider than the picture, and pads
whose margin is odd (the centering rounds a half to even). Needs only Pillow (12.3 recorded in `versions`):

    source tools/reference/hf_env.sh && python -I tools/reference/unlimited_ocr/resize_cases.py zig/tests/fixtures/image/resize_cases.json

The output is `{"versions": {...}, "cases": [{op, name, width, height, rgb, to: [w, h], fill?, out_size: [w, h], sha256}]}`.
"""
import base64
import hashlib
import json
import random
import sys

import PIL
from PIL import Image, ImageOps

OUT = sys.argv[1]
rng = random.Random(137)

RESIZES = [(7, 5, 3, 2), (7, 5, 20, 13), (40, 30, 13, 11), (33, 9, 33, 4), (9, 33, 5, 33), (1, 1, 8, 8), (64, 48, 64, 48),
           (3, 50, 31, 2), (100, 3, 7, 1), (53, 41, 90, 55), (71, 13, 13, 71), (129, 4, 31, 9)]
PADS = [(10, 6, 16, 16), (6, 10, 16, 16), (7, 4, 16, 16), (4, 7, 16, 16), (5, 5, 16, 16), (16, 16, 16, 16), (3, 2, 17, 17),
        (30, 11, 16, 16), (11, 30, 16, 16), (9, 3, 20, 10), (50, 21, 16, 16)]


def picture(w, h):
    # smooth ramps with noise: clipping at both ends and mid-gray rounding both occur
    px = bytearray()
    for y in range(h):
        for x in range(w):
            for c in range(3):
                v = (x * 255 // max(w - 1, 1)) if c == 0 else (y * 255 // max(h - 1, 1)) if c == 1 else rng.randrange(256)
                px.append(min(255, max(0, v + rng.randrange(-40, 41) if rng.random() < 0.3 else v)))
    return bytes(px)


cases = []
for op, table in (("resize", RESIZES), ("pad", PADS)):
    for i, (w, h, dw, dh) in enumerate(table):
        raw = picture(w, h)
        im = Image.frombytes("RGB", (w, h), raw)
        if op == "resize":
            out = im.resize((dw, dh), Image.BICUBIC)
        else:
            out = ImageOps.pad(im, (dw, dh), color=(127, 127, 127))
        cases.append({"op": op, "name": f"{op}_{i}_{w}x{h}_{dw}x{dh}", "width": w, "height": h,
                      "rgb": base64.b64encode(raw).decode(), "to": [dw, dh], "fill": 127,
                      "out_size": list(out.size), "sha256": hashlib.sha256(out.tobytes()).hexdigest()})
json.dump({"versions": {"pillow": PIL.__version__}, "cases": cases}, open(OUT, "w"), separators=(",", ":"))
print(len(cases), "cases", PIL.__version__)
