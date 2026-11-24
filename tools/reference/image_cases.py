"""Pictures and the RGB TensorFold decoded them to, for `zig/src/image/` (spec 019 WS-07).

Each case is a small file made here (PNG and GIF written byte by byte where Pillow's writer cannot reach a
layout, JPEG and WebP by Pillow's), and TensorFold 0.6.5's own `vision.images._decode` run on it: Pillow's
open, the size limits, `verify`, `load`, `ImageOps.exif_transpose`, then alpha over white or `convert("RGB")`.
A GIF goes through the same steps with GIF among the formats and without the frame-count refusal: its first
frame, as vLLM and SGLang take it (the owner, spec 019's Decisions of 2026-10-13); 0.6.5 itself refused GIF.
BMP, TGA, PSD and PPM, which 0.6.5 refused and vLLM opens (spec 019 WS-04), go through vLLM's `load_bytes`
steps (`vllm_decode`) under 0.6.5's size limits. A case records the pixels or 0.6.5's refusal. In TensorFold's container (Pillow 12.3.0, libjpeg-turbo
3.1.4.1, libwebp 1.6.0, zlib 1.3), without the GPU:

    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/repo -v ~/.cache/zspark/tensorfold-609ca41:/tf:ro -w /repo \
        tensorfold-glm53:v0.6.0 python3 -I tools/reference/image_cases.py /tf/src zig/tests/fixtures/image/cases.json

The output is `{"versions": {...}, "cases": [{name, format, bytes, width, height, rgb} or {name, format,
bytes, problem}]}`, `bytes` and `rgb` in base64; `note` says where zspark departs on purpose.
"""
from __future__ import annotations

import base64
import io
import json
import random
import struct
import sys
import zlib

TF = sys.argv[1]
OUT = sys.argv[2]
sys.path.insert(0, TF)

import PIL  # noqa: E402
import PIL.features  # noqa: E402
from PIL import Image  # noqa: E402

from tensorfold.vision import images as tf_images  # noqa: E402

CASES: list[dict] = []


def add(name: str, fmt: str, data: bytes, note: str | None = None, ref: str | None = None) -> None:
    """`ref` "vllm": vLLM's steps (`vllm_decode`) where 0.6.5 refused, as an animation's first frame."""
    CASES.append({"name": name, "format": fmt, "data": data, "note": note, "ref": ref})


# --- PNG, byte by byte ------------------------------------------------------------------------------------

def chunk(kind: bytes, body: bytes, crc: int | None = None) -> bytes:
    c = zlib.crc32(kind + body) if crc is None else crc
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", c & 0xFFFFFFFF)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def filtered(raw: bytes, prev: bytes, kind: int, bpp: int) -> bytes:
    out = bytearray([kind])
    for i, v in enumerate(raw):
        left = raw[i - bpp] if i >= bpp else 0
        up = prev[i]
        corner = prev[i - bpp] if i >= bpp else 0
        pred = (0, left, up, (left + up) // 2, paeth(left, up, corner))[kind]
        out.append((v - pred) & 0xFF)
    return bytes(out)


CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))


def pack_row(samples: list[int], depth: int) -> bytes:
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    if depth == 8:
        return bytes(samples)
    out, acc, bits = bytearray(), 0, 0
    for s in samples:
        acc, bits = (acc << depth) | s, bits + depth
        if bits == 8:
            out.append(acc)
            acc, bits = 0, 0
    if bits:
        out.append(acc << (8 - bits))
    return bytes(out)


def png(w: int, h: int, depth: int, color: int, *, seed: int = 1, interlace: bool = False, plte: bytes | None = None,
        trns: bytes | None = None, before: list[bytes] = (), after: list[bytes] = (), samples=None,
        idat_parts: int = 1, tail: bytes = b"") -> bytes:
    rng = random.Random(seed)
    top = (1 << depth) - 1
    if color == 3:
        top = min(top, len(plte) // 3 - 1)
    n = CHANNELS[color]
    px = samples or [[[rng.randint(0, top) for _ in range(n)] for _ in range(w)] for _ in range(h)]
    bpp = max(1, n * depth // 8)
    raw = bytearray()
    passes = ADAM7 if interlace else ((0, 0, 1, 1),)
    k = 0
    for x0, y0, dx, dy in passes:
        cols = range(x0, w, dx)
        rows = range(y0, h, dy)
        if not cols or not rows:
            continue
        prev = None
        for y in rows:
            row = pack_row([s for x in cols for s in px[y][x]], depth)
            prev = prev or bytes(len(row))
            raw += filtered(row, prev, k % 5, bpp)
            k += 1
            prev = row
    z = zlib.compress(bytes(raw), 9) + tail
    step = max(1, -(-len(z) // idat_parts))
    data = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, color, 0, 0, int(interlace)))
    data += b"".join(before)
    if plte is not None:
        data += chunk(b"PLTE", plte)
    if trns is not None:
        data += chunk(b"tRNS", trns)
    for i in range(0, len(z), step):
        data += chunk(b"IDAT", z[i:i + step])
    return data + b"".join(after) + chunk(b"IEND", b"")


def palette(n: int, seed: int) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randint(0, 255) for _ in range(3 * n))


def exif(orientation: int, big: bool = False) -> bytes:
    if big:
        return b"MM\x00\x2a\x00\x00\x00\x08\x00\x01" + struct.pack(">HHIHH", 0x0112, 3, 1, orientation, 0) + b"\0\0\0\0"
    return b"II\x2a\x00\x08\x00\x00\x00\x01\x00" + struct.pack("<HHIHH", 0x0112, 3, 1, orientation, 0) + b"\0\0\0\0"


def png_cases() -> None:
    for depth in (1, 2, 4, 8, 16):
        add(f"png_gray{depth}", "png", png(13, 7, depth, 0, seed=depth))
        add(f"png_gray{depth}_interlaced", "png", png(13, 7, depth, 0, seed=depth + 50, interlace=True))
        key = (1 << depth) - 1 if depth < 16 else 300
        add(f"png_gray{depth}_key", "png", png(13, 7, depth, 0, seed=depth + 10, trns=struct.pack(">H", key)))
    for depth in (8, 16):
        for color, kind in ((2, "rgb"), (4, "graya"), (6, "rgba")):
            add(f"png_{kind}{depth}", "png", png(13, 7, depth, color, seed=depth + color))
            add(f"png_{kind}{depth}_interlaced", "png", png(13, 7, depth, color, seed=depth + color + 7, interlace=True))
        rgb_key = [[[1, 2, 3] if (x + y) % 3 == 0 else [200, 100, 50] for x in range(13)] for y in range(7)]
        if depth == 16:
            rgb_key = [[[v * 257 for v in p] for p in row] for row in rgb_key]
        key = struct.pack(">HHH", *((1, 2, 3) if depth == 8 else (257, 514, 771)))
        add(f"png_rgb{depth}_key", "png", png(13, 7, depth, 2, samples=rgb_key, trns=key))
    for depth in (1, 2, 4, 8):
        n = min(1 << depth, 200)
        plte = palette(n, depth)
        add(f"png_palette{depth}", "png", png(13, 7, depth, 3, seed=depth + 20, plte=plte))
        add(f"png_palette{depth}_interlaced", "png", png(13, 7, depth, 3, seed=depth + 30, plte=plte, interlace=True))
        alphas = bytes((i * 37) & 0xFF for i in range(max(1, n // 2)))
        add(f"png_palette{depth}_alpha", "png", png(13, 7, depth, 3, seed=depth + 40, plte=plte, trns=alphas))
    # Pillow compares a key's low byte with the 8-bit values of its mode (Convert.c rgbT2a): 300 keys 44
    gray = [[[v] for v in (44, 300, 255, 1000, 0, 44, 7)] for _ in range(3)]
    add("png_gray16_key_low_byte", "png", png(7, 3, 16, 0, samples=gray, trns=struct.pack(">H", 300)))
    add("png_gray8_key_past_255", "png", png(7, 3, 8, 0, samples=[[[min(v, 255)] for v in r] for r in [[44, 1, 2, 44, 255, 0, 3]] * 3], trns=struct.pack(">H", 300)))
    add("png_rgb8_key_past_255", "png", png(3, 2, 8, 2, samples=[[[44, 2, 3], [1, 2, 3], [9, 9, 9]]] * 2, trns=struct.pack(">HHH", 300, 2, 3)))
    add("png_gray1_key_256", "png", png(13, 7, 1, 0, seed=61, trns=struct.pack(">H", 256)))
    add("png_one_pixel", "png", png(1, 1, 8, 2))
    add("png_tall_interlaced", "png", png(1, 9, 8, 6, interlace=True))
    add("png_split_idat", "png", png(13, 7, 8, 6, idat_parts=4))
    add("png_after_stream", "png", png(13, 7, 8, 2, tail=b"junk after the stream"))
    for o in (2, 3, 4, 5, 6, 7, 8):
        add(f"png_orient{o}", "png", png(5, 3, 8, 2, seed=o, before=[chunk(b"eXIf", exif(o, big=o % 2 == 0))]))
    add("png_orient_xmp", "png", png(5, 3, 8, 2, before=[chunk(b"iTXt", b"XML:com.adobe.xmp\0\0\0\0\0<x tiff:Orientation=\"6\"/>")]))
    # damage and refusals
    good = png(13, 7, 8, 2)
    ihdr_end = 8 + 25
    add("png_bad_header_crc", "png", good[:ihdr_end - 1] + bytes([good[ihdr_end - 1] ^ 1]) + good[ihdr_end:])
    add("png_bad_ancillary_crc", "png", png(13, 7, 8, 2, before=[chunk(b"tEXt", b"k\0v", crc=0)]))
    add("png_bad_idat_crc", "png", good[:-12 - 1] + bytes([good[-13] ^ 1]) + good[-12:])
    add("png_truncated_idat", "png", good[:len(good) // 2])
    add("png_no_iend", "png", good[:-12])
    add("png_bad_filter", "png", b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 1, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x07\x01\x02\x03")) + chunk(b"IEND", b""))
    add("png_huge", "png", b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 9000, 10, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * 10)) + chunk(b"IEND", b""))
    add("png_many_pixels", "png", b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 5000, 5000, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\0" * 10)) + chunk(b"IEND", b""))
    add("png_zero_width", "png", b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 0, 7, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"")) + chunk(b"IEND", b""))
    actl = chunk(b"acTL", struct.pack(">II", 2, 0))
    add("png_animated", "png", png(13, 7, 8, 2, before=[actl]), ref="vllm")


# --- GIF, byte by byte and by Pillow ----------------------------------------------------------------------

def lzw_literal(indices: list[int], min_size: int) -> bytes:
    """Every index a literal code, a clear code before the table would widen: valid LZW with no matches."""
    clear, end = 1 << min_size, (1 << min_size) + 1
    width = min_size + 1
    run = (1 << min_size) - 2
    codes = []
    for i in range(0, len(indices), run):
        codes.append(clear)
        codes += indices[i:i + run]
    codes.append(end)
    out, acc, bits = bytearray(), 0, 0
    for c in codes:
        acc |= c << bits
        bits += width
        while bits >= 8:
            out.append(acc & 0xFF)
            acc >>= 8
            bits -= 8
    if bits:
        out.append(acc & 0xFF)
    blocks = b"".join(bytes([len(out[i:i + 255])]) + bytes(out[i:i + 255]) for i in range(0, len(out), 255))
    return bytes([min_size]) + blocks + b"\0"


def gif(screen: tuple[int, int], frames: list[dict], *, global_table: bytes | None, background: int = 0,
        version: bytes = b"GIF89a", trailer: bool = True) -> bytes:
    flags = 0
    if global_table is not None:
        bits = (len(global_table) // 3).bit_length() - 1
        flags = 0x80 | (bits - 1)
    data = version + struct.pack("<HHBBB", screen[0], screen[1], flags, background, 0) + (global_table or b"")
    for f in frames:
        if f.get("transparent") is not None:
            data += b"!\xf9\x04" + bytes([1]) + struct.pack("<H", 0) + bytes([f["transparent"]]) + b"\0"
        x, y, w, h = f["box"]
        fl = 0x40 if f.get("interlace") else 0
        table = f.get("table")
        if table is not None:
            fl |= 0x80 | ((len(table) // 3).bit_length() - 2)
        data += b"," + struct.pack("<HHHHB", x, y, w, h, fl) + (table or b"")
        idx = f["indices"]
        if f.get("interlace"):
            rows = [idx[r * w:(r + 1) * w] for r in range(h)]
            order = list(range(0, h, 8)) + list(range(4, h, 8)) + list(range(2, h, 4)) + list(range(1, h, 2))
            idx = [v for r in order for v in rows[r]]
        data += lzw_literal(idx, f.get("min_size", 2))
    return data + (b";" if trailer else b"")


def pillow_gif(im: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, "GIF", **kw)
    return buf.getvalue()


def gif_cases() -> None:
    rng = random.Random(5)
    for n in (2, 4, 16, 256):
        im = Image.new("P", (13, 7))
        im.putpalette(list(palette(n, n)))
        im.putdata([rng.randrange(n) for _ in range(13 * 7)])
        add(f"gif_palette{n}", "gif", pillow_gif(im))
    im = Image.new("P", (13, 7))
    im.putpalette(list(palette(16, 3)))
    im.putdata([rng.randrange(16) for _ in range(13 * 7)])
    add("gif_transparent", "gif", pillow_gif(im, transparency=3))
    add("gif_interlaced_pillow", "gif", pillow_gif(im, interlace=True))
    frames = [im, im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    add("gif_animated", "gif", buf.getvalue(), note="0.6.5 refused animated pictures; zspark takes the first frame")
    table = palette(4, 9)
    idx = [rng.randrange(4) for _ in range(13 * 7)]
    add("gif_hand", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}], global_table=table))
    add("gif87a", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}], global_table=table, version=b"GIF87a"))
    add("gif_interlaced", "gif", gif((5, 11), [{"box": (0, 0, 5, 11), "interlace": True, "indices": [rng.randrange(4) for _ in range(55)]}], global_table=table))
    small = [rng.randrange(4) for _ in range(4 * 3)]
    add("gif_small_frame", "gif", gif((13, 7), [{"box": (3, 2, 4, 3), "indices": small}], global_table=table, background=2))
    add("gif_small_frame_transparent", "gif", gif((13, 7), [{"box": (3, 2, 4, 3), "indices": small, "transparent": 1}], global_table=table))
    add("gif_frame_past_screen", "gif", gif((6, 4), [{"box": (2, 1, 8, 5), "indices": [rng.randrange(4) for _ in range(40)]}], global_table=table))
    add("gif_local_table", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx, "table": palette(4, 77)}], global_table=table))
    add("gif_no_table", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}], global_table=None))
    gray = b"".join(bytes([i, i, i]) for i in range(4))
    add("gif_gray_table", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx, "transparent": 2}], global_table=gray))
    add("gif_index_past_table", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": [rng.randrange(8) for _ in range(91)], "min_size": 3}], global_table=table))
    add("gif_no_trailer", "gif", gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}], global_table=table, trailer=False))
    two = gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}, {"box": (1, 1, 2, 2), "indices": [0, 1, 2, 3]}], global_table=table)
    add("gif_two_frames_hand", "gif", two, note="0.6.5 refused animated pictures; zspark takes the first frame")
    good = gif((13, 7), [{"box": (0, 0, 13, 7), "indices": idx}], global_table=table)
    add("gif_truncated", "gif", good[:len(good) // 2])
    add("gif_no_frame", "gif", good[:13 + 12] + b";")
    add("gif_huge", "gif", gif((9000, 1), [{"box": (0, 0, 1, 1), "indices": [0]}], global_table=table))
    add("gif_huge_frame", "gif", gif((2, 2), [{"box": (0, 0, 9000, 1), "indices": [0]}], global_table=table))


# --- JPEG and WebP, by Pillow -----------------------------------------------------------------------------

def scene(w: int, h: int, mode: str = "RGB", seed: int = 2) -> Image.Image:
    rng = random.Random(seed)
    im = Image.new("RGB", (w, h))
    im.putdata([((x * 255) // max(1, w - 1), (y * 255) // max(1, h - 1), (x * y * 7 + rng.randrange(40)) & 0xFF)
                for y in range(h) for x in range(w)])
    return im.convert(mode)


def saved(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def exif_block(o: int) -> bytes:
    return b"Exif\0\0" + exif(o)


def jpeg_cases() -> None:
    im = scene(37, 23)
    for sub, name in ((0, "444"), (1, "422"), (2, "420")):
        add(f"jpeg{name}", "jpeg", saved(im, "JPEG", quality=90, subsampling=sub))
    add("jpeg420_progressive", "jpeg", saved(im, "JPEG", quality=85, subsampling=2, progressive=True))
    add("jpeg444_progressive", "jpeg", saved(im, "JPEG", quality=95, subsampling=0, progressive=True))
    add("jpeg_optimized", "jpeg", saved(im, "JPEG", quality=75, optimize=True))
    add("jpeg_restarts", "jpeg", saved(im, "JPEG", quality=80, restart_marker_blocks=2))
    add("jpeg_gray", "jpeg", saved(scene(37, 23, "L"), "JPEG", quality=90))
    add("jpeg_gray_progressive", "jpeg", saved(scene(16, 9, "L"), "JPEG", progressive=True))
    add("jpeg_small", "jpeg", saved(scene(1, 1), "JPEG"))
    add("jpeg_photo", "jpeg", saved(scene(160, 120, seed=7), "JPEG", quality=92))
    for o in (3, 6, 8):
        add(f"jpeg_orient{o}", "jpeg", saved(scene(9, 5, seed=o), "JPEG", exif=exif_block(o)))
    add("jpeg_cmyk", "jpeg", saved(scene(13, 7, "CMYK"), "JPEG"), note="zspark refuses a CMYK JPEG by name, as TensorFold 1.0.12 does")
    good = saved(im, "JPEG", quality=90)
    add("jpeg_truncated", "jpeg", good[:len(good) // 2])
    add("jpeg_no_end", "jpeg", good[:-2])
    sof = good.index(b"\xff\xc0")
    big = good[:sof + 5] + struct.pack(">HH", 1, 9000) + good[sof + 9:]
    add("jpeg_huge", "jpeg", big)


def webp_cases() -> None:
    im = scene(21, 13)
    rgba = im.convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize((21, 13)))
    add("webp_lossy", "webp", saved(im, "WEBP", quality=80))
    add("webp_lossy_alpha", "webp", saved(rgba, "WEBP", quality=80))
    add("webp_lossless", "webp", saved(im, "WEBP", lossless=True))
    add("webp_lossless_alpha", "webp", saved(rgba, "WEBP", lossless=True))
    add("webp_lossy_odd", "webp", saved(scene(7, 3, seed=4), "WEBP", quality=50))
    for o in (3, 6):
        add(f"webp_orient{o}", "webp", saved(scene(9, 5, seed=o), "WEBP", lossless=True, exif=exif(o)))
    add("webp_xmp", "webp", saved(scene(9, 5), "WEBP", lossless=True, xmp=b"<x tiff:Orientation=\"8\"/>"))
    buf = io.BytesIO()
    im.save(buf, "WEBP", save_all=True, append_images=[im.transpose(Image.Transpose.FLIP_TOP_BOTTOM)], duration=50)
    add("webp_animated", "webp", buf.getvalue(), ref="vllm")
    buf = io.BytesIO()
    rgba.save(buf, "WEBP", save_all=True, append_images=[rgba.transpose(Image.Transpose.FLIP_TOP_BOTTOM)], duration=50, lossless=True)
    add("webp_animated_alpha", "webp", buf.getvalue(), ref="vllm")
    buf = io.BytesIO()
    im.save(buf, "WEBP", save_all=True, append_images=[im.rotate(180)], duration=50, quality=80)
    add("webp_animated_lossy", "webp", buf.getvalue(), ref="vllm")
    buf = io.BytesIO()
    rgba.save(buf, "PNG", save_all=True, append_images=[rgba.transpose(Image.Transpose.FLIP_TOP_BOTTOM)], duration=50)
    add("png_animated_alpha", "png", buf.getvalue(), ref="vllm")
    good = saved(im, "WEBP", quality=80)
    add("webp_truncated", "webp", good[:len(good) // 2])
    add("webp_huge", "webp", saved(Image.new("RGB", (9000, 1)), "WEBP", lossless=True))


def psd(im: Image.Image, rle: bool = False) -> bytes:
    """A PSD of one merged picture and no layers, byte by byte (Pillow writes none): raw or PackBits planes."""
    bands = im.getbands()
    mode = {"L": 1, "RGB": 3}[im.mode]
    w, h = im.size
    out = b"8BPS" + struct.pack(">H6xHIIHH", 1, len(bands), h, w, 8, mode)
    out += struct.pack(">I", 0) + struct.pack(">I", 0) + struct.pack(">I", 0)
    planes = [im.getchannel(b).tobytes() for b in bands]
    if not rle:
        return out + struct.pack(">H", 0) + b"".join(planes)
    rows = []
    for plane in planes:
        for y in range(h):
            row = plane[y * w:(y + 1) * w]
            packed = bytearray()
            i = 0
            while i < len(row):  # a run of one value, else literals: PackBits
                run = 1
                while i + run < len(row) and run < 128 and row[i + run] == row[i]:
                    run += 1
                if run > 1:
                    packed += bytes([257 - run, row[i]])
                    i += run
                else:
                    j = i
                    while j < len(row) and j - i < 128 and (j + 1 >= len(row) or row[j + 1] != row[j]):
                        j += 1
                    j = max(j, i + 1)
                    packed += bytes([j - i - 1]) + row[i:j]
                    i = j
            rows.append(bytes(packed))
    return out + struct.pack(">H", 1) + b"".join(struct.pack(">H", len(r)) for r in rows) + b"".join(rows)


def vllm_cases() -> None:
    """The formats vLLM's Pillow opens beside 0.6.5's (spec 019 WS-04): stb_image's in zspark."""
    im = scene(11, 7, seed=5)
    rgba = im.convert("RGBA")
    rgba.putalpha(Image.linear_gradient("L").resize((11, 7)))
    add("bmp", "bmp", saved(scene(4, 3), "BMP"))
    add("bmp_rgb", "bmp", saved(im, "BMP"))
    add("bmp_gray", "bmp", saved(im.convert("L"), "BMP"))
    add("bmp_palette", "bmp", saved(im.convert("P", palette=Image.Palette.ADAPTIVE, colors=16), "BMP"))
    add("bmp_1bit", "bmp", saved(im.convert("1"), "BMP"))
    add("tga_rgb", "tga", saved(im, "TGA"))
    add("tga_rgb_top", "tga", saved(im, "TGA", orientation=1))
    add("tga_rle", "tga", saved(im, "TGA", compression="tga_rle"))
    add("tga_rgba", "tga", saved(rgba, "TGA"))
    add("tga_gray", "tga", saved(im.convert("L"), "TGA"))
    add("ppm_p6", "ppm", saved(im, "PPM"))
    add("ppm_p5", "ppm", saved(im.convert("L"), "PPM"))
    add("ppm_p3", "ppm", b"P3\n2 1\n255\n255 0 0  0 128 255\n", note="zspark refuses PNM's text forms (P1 to P4), which stb_image does not read")
    add("psd_rgb", "psd", psd(im))
    add("psd_rgb_rle", "psd", psd(im, rle=True))
    add("psd_gray", "psd", psd(im.convert("L")))
    good = saved(im, "BMP")
    add("bmp_truncated", "bmp", good[:len(good) // 2])
    add("bmp_huge", "bmp", good[:18] + struct.pack("<ii", 9000, 1) + good[26:])


def others() -> None:
    add("garbage", "other", b"not a picture at all")
    add("empty", "other", b"")


# --- the reference ----------------------------------------------------------------------------------------

def reference(case: dict) -> dict:
    data = case["data"]
    limits = tf_images.DEFAULT_LIMITS
    out = {"name": case["name"], "format": case["format"], "bytes": base64.b64encode(data).decode()}
    if case["note"]:
        out["note"] = case["note"]
    try:
        if case["format"] == "gif":
            decoded = gif_decode(data, limits)
        elif case["format"] in VLLM_FORMATS or case["ref"] == "vllm":
            decoded = vllm_decode(data, limits)
        else:
            decoded = tf_images._decode(data, "auto", limits, limits.max_total_pixels)
        out.update(width=decoded.width, height=decoded.height, rgb=base64.b64encode(decoded.pixels).decode())
    except tf_images.ImageInputError as e:
        out["problem"] = str(e)
    except Exception as e:  # what 0.6.5 let escape: a refusal all the same, named by its type
        out["problem"] = f"uncaught {type(e).__name__}"
    return out


def gif_decode(data: bytes, limits):
    """0.6.5's `_decode` with GIF the format and no frame count: the first frame."""
    import warnings
    from PIL import ImageOps
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=("GIF",)) as source:
                width, height = source.size
                if (min(width, height) <= 0 or max(width, height) > limits.max_dimension
                        or width * height > limits.max_pixels):
                    raise tf_images.ImageInputError("image dimensions exceed the decoded pixel limit")
                source.verify()
            with Image.open(io.BytesIO(data), formats=("GIF",)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    rgb = Image.new("RGB", oriented.size, "white")
                    rgb.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    rgb = oriented.convert("RGB")
                return tf_images.ImageInput(rgb.width, rgb.height, rgb.tobytes(), "auto")
    except tf_images.ImageInputError:
        raise
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise tf_images.ImageInputError("image bytes are invalid or unsupported; use JPEG, PNG or WebP") from None


# vLLM 3313825's `ALLOWED_IMAGE_FORMATS` beyond 0.6.5's that zspark decodes, by the names Pillow opens them as
VLLM_FORMATS = {"bmp": "BMP", "tga": "TGA", "psd": "PSD", "ppm": "PPM"}


def vllm_decode(data: bytes, limits):
    """vLLM 3313825's `ImageMediaIO.load_bytes` (`vllm/multimodal/media/image.py`): `open_image`, `normalize_image`
    (`exif_transpose`), `load`, then `convert_image_mode` to RGB, alpha over white; 0.6.5's size limits first, as
    zspark keeps them, and 0.6.5's words for a refusal, as `zig/src/image/` words it."""
    from PIL import ImageOps
    try:
        with Image.open(io.BytesIO(data), formats=tuple(VLLM_FORMATS.values()) + ("PNG", "WEBP")) as source:
            width, height = source.size
            if (min(width, height) <= 0 or max(width, height) > limits.max_dimension
                    or width * height > limits.max_pixels):
                raise tf_images.ImageInputError("image dimensions exceed the decoded pixel limit")
            try:
                image = ImageOps.exif_transpose(source)
            except Exception:
                image = source
            image.load()
            if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                rgb = Image.new("RGB", image.size, (255, 255, 255))
                rgb.paste(rgba, mask=rgba.split()[3])
            else:
                rgb = image.convert("RGB")
            return tf_images.ImageInput(rgb.width, rgb.height, rgb.tobytes(), "auto")
    except tf_images.ImageInputError:
        raise
    except (OSError, ValueError, SyntaxError, EOFError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise tf_images.ImageInputError("image bytes are invalid or unsupported; use JPEG, PNG or WebP") from None


def main() -> None:
    png_cases()
    gif_cases()
    jpeg_cases()
    webp_cases()
    vllm_cases()
    others()
    names = [c["name"] for c in CASES]
    assert len(names) == len(set(names)), "case names repeat"
    versions = {"pillow": PIL.__version__, "libjpeg_turbo": PIL.features.version("libjpeg_turbo"),
                "webp": PIL.features.version("webp"), "zlib": PIL.features.version("zlib"),
                "tensorfold": "0.6.5 (609ca41), vision/images.py _decode"}
    result = {"versions": versions, "cases": [reference(c) for c in CASES]}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=0, sort_keys=False)
        f.write("\n")
    taken = sum("rgb" in c for c in result["cases"])
    print(f"{len(CASES)} cases, {taken} decoded, {len(CASES) - taken} refused; {versions}")


if __name__ == "__main__":
    main()
