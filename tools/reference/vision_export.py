#!/usr/bin/env python3
"""Reference images, their processing, the vision tower and an image prompt's rows, exported from TensorFold itself.

A test fixture (spec 019 WS-06): it runs inside the TensorFold container (PyTorch, Triton, transformers, Pillow),
never at zspark's runtime. It makes its own images, sends them through TensorFold 0.6.5's image path as
``tensorfold serve --vision`` does, and writes what each step gave: the decoded RGB, the processor's outputs, the
tower's features and its blocks' rows, and an image prompt's rows through the language model, block by block, in
the form of ``flash_next_export.py``'s files (whose ``Tap``, ``save`` and ``compare`` it imports).

Pinned to TensorFold 0.6.5 (commit 609ca41). Besides ``flash_next_export.py``'s list, it relies on these, read
or wrapped from here and none edited:

- ``tensorfold.vision.images``: ``ImageSource``, ``load_images`` (a ``data:`` URL's bytes through ``_decode``:
  Pillow, EXIF orientation, alpha over white, ``ImageInput``'s RGB bytes and ``content_hash``);
- ``tensorfold.vision.qwen_processing.QwenImageProcessor``: ``from_directory``, ``prepare`` (transformers'
  ``Qwen2VLImageProcessor``: the resize, rescale, normalize and patching; the expanded prompt re-tokenized;
  ``media_positions``: the three rotary axes and the delta), with ``max_image_tokens=TOKENS_PER_IMAGE`` as
  ``QwenCudaVision.prepare`` passes it;
- ``tensorfold.vision.qwen_cuda``: ``vision_config``, ``QwenCudaVision(dir, device)`` (the tower alone,
  transformers' ``Qwen3_5VisionModel``, bf16) and its ``encode(prepared, prompt)`` -> ``EncodedVision`` (rows,
  features, positions (3, n), rope_delta); forward hooks on ``tower.patch_embed``, ``tower.blocks[i]``,
  ``tower.merger.norm``, ``tower.merger.linear_fc1`` and ``tower.merger`` (they read outputs, change no bits);
- ``families.qwen4_exp.cuda.image_rows.attach`` (the positions and delta on the state; ``state.State.reset``
  clears them, so it runs after the reset) and ``image_rows.finish`` (features end with the prompt, positions
  stay), and ``forward.forward(w, st, b, tokens, features=(target, source))``, the decode path's way to put the
  features over a chunk's placeholder rows (forward.py:499-501; ``image_rows.embed`` runs on prompt buffers only);
- with ``--stage whole``: ``families.qwen4_exp.cuda_engine(dir, no_drafts=True, vision=True, parallel=2,
  context=...)`` (0.6.5 serves images only from ``--parallel 2``, engine.py:75) and its ``generate(ids, n, None,
  on_tokens, draft=..., stop_eos=False, vision=prepared)``: TensorFold's served reply, through its scheduler,
  ``image_rows.begin`` and the prompt kernels.

The checkpoint's ``vision_config.model_type``. The published NVFP4 directory says ``qwen4_exp_vision``, which
0.6.5's ``vision_config`` refuses ("CUDA vision requires a Qwen3.5-compatible vision checkpoint",
qwen_cuda.py:47). The exporter builds a scratch directory of symbolic links to every file of the checkpoint and
one rewritten ``config.json`` whose ``vision_config.model_type`` is ``qwen4_exp`` (nothing else changes), and
gives that directory to the processor, the tower and the engine; the language model's weights load from the
checkpoint itself. The checkpoint is never written. Provenance names both (``config_patched``).

The images (made here, nothing taken from elsewhere; ``IMAGES``):

    shapes   256 x 256  a red disc, a blue square, a green bar on white: a multiple of 32 on both sides, which
                        the resize leaves as it is (grid 1 x 16 x 16, 64 visual tokens)
    bands    300 x 200  a sky gradient, a yellow sun, green ground: resized to 288 x 192 by the processor's
                        bicubic resize (grid 1 x 12 x 18, 54 tokens)
    tiny      48 x 40   a black and white checkerboard with a red centre: resized to 64 x 32 (grid 1 x 2 x 4,
                        2 tokens)

The least pixels are 32 x 32, not the checkpoint's ``size.shortest_edge`` (65536): ``prepare`` passes
``min_pixels=min(getattr(processor, "min_pixels", 32**2), cap)`` and transformers 5.18's processor has no such
attribute, so the default is what the call gets, and the call's ``min_pixels`` overrides ``size``. The most is
``TOKENS_PER_IMAGE`` (4096) tokens' pixels the same way. Where the resize keeps the size, ``pixel_values`` is
``float32((rgb in float64 / 255 - 0.5) / 0.5)`` patched (checked on ``shapes``).

Each is a PNG; ``shapes`` also as a JPEG (quality 90), a WebP (lossy, quality 80) and a GIF (one frame, an adaptive
palette). Every file but the GIF goes through
TensorFold's ``load_images`` as a ``data:`` URL, whose RGB is the fixture's ("the pixels TensorFold's decoder
gave", spec 019's Decisions); TensorFold takes no GIF (its ``_FORMATS``: JPEG, PNG, WebP), so the GIF's RGB is
Pillow's first frame converted to RGB, as vLLM and SGLang read it. Only the PNGs' RGB feed the processor.

The cases (``CASES``): ``shapes`` (one image) and ``pair`` (``bands`` then ``tiny`` in one message), each a user
message of image parts and a question, rendered by the checkpoint's chat template through transformers'
``apply_chat_template(add_generation_prompt=True, enable_thinking=False)``.

Stages (``--stage``):

- ``processor`` (no GPU, no lock): the images and the processor; writes the small fixtures and the RGB and
  processor tensors.
- ``layers`` (``--layers N``, default 4; ``gpu.lock``): the tower, and each case's serial steps and windows
  through the first N layers (``weights.Config.read`` shortened, as ``flash_next_export.py --layers``).
- ``whole`` (``gpu.lock``, one whole load): the engine's replies, the tower again from the engine's own, and
  each case's serial steps and windows through the 48 layers and its greedy reply on the decode path.

Each GPU stage runs the processor again and refuses to go on if it does not give the processor stage's bits.

Files under ``--out-dir`` (``~/.cache/zspark/reference-vision``):

    images.safetensors              IMAGE.rgb U8 [H, W, 3] for each PNG; shapes.jpeg.rgb, shapes.webp.rgb,
                                    shapes.gif.rgb (the other files' RGB)
    CASE.processor.safetensors      ids I32 [n]  the expanded prompt
                                    pixel_values F32 [patches, 1536]   as the processor gave them; the tower
                                                 takes them as bf16 (``torch.tensor(..., dtype=bfloat16)``)
                                    image_grid_thw I64 [images, 3]
                                    position_ids I32 [3, n]  t, h, w (``prepared.position_ids[:, 0, :]``)
                                    rope_delta I32 [1];  image_spans I32 [images, 2] (begin, end) of the pads
    CASE.tower.safetensors          features BF16 [rows, 2560]  ``encode``'s, all the case's images in one call
                                    rows I32 [rows]  the placeholders' prompt rows
                                    IMAGE.alone BF16 [tokens, 2560]  the image's features from a call of its own
    IMAGE.tower.safetensors         the tower's points on the image alone, rows in the processor's patch order
                                    (2 x 2 merge windows contiguous):
                                    pixel_values BF16 [P, 1536]  what the tower read
                                    patch_embed BF16 [P, 1152]   the patch convolution's output
                                    blocks.0.in BF16 [P, 1152]   after the interpolated position embedding
                                    blocks.{i}.out BF16 [P, 1152]  i < 27
                                    merger.norm BF16 [P, 1152];  merger.fc1 BF16 [P / 4, 4608]
                                    features BF16 [P / 4, 2560]
    CASE[.layersN].serial[.f32].safetensors, and with ``layers`` .window and .window16 (``whole`` compares its
                                    windows in the JSON and writes no file of them): ``flash_next_export.py``'s
                                    names and form
                                    (``zspark compare DIR REF layers=4`` reads them), and beside them:
                                    vision.features BF16 [rows, 2560], vision.rows I32 [rows],
                                    vision.positions I32 [3, n], vision.rope_delta I32 [1]
    CASE[.layersN][.f32].json       the comparisons, as ``flash_next_export.py`` writes them; with ``whole`` the
                                    greedy reply on the decode path and the engine's (``greedy_engine``)

``.f32`` files use fp32 expert slots (what zspark ports), the others bf16 slots; ``.layersN`` files are the
``layers`` stage's, whose ``final.*`` and ``logits`` follow layer N-1.

Under ``--fixtures`` (``zig/tests/fixtures/vision``, committed): the encoded images and ``cases.json``: each
file's size, decoder and RGB's sha256 and TensorFold's ``content_hash``; each case's rendered prompt, expanded
ids, grids, spans, positions and delta, the processor's settings and the pixel_values' sha256; with ``whole``
the replies.

The commands (``tools/reference/AGENTS.md``'s container, ``/tf`` the 0.6.5 archive; the checkpoint's parent
directory mounted at ``/models``, since in this image a mount at the host's ``/home/example/...`` path does not
show). The processor stage takes no GPU and no lock; the other two take ``--gpus all`` and run under
``flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock``. ``layers``: 16 s to load, 31 GiB on
the device; ``whole``: one full load:

    mkdir -p ~/.cache/zspark/reference-vision zig/tests/fixtures/vision
    docker run --rm [--gpus all] --user "$(id -u):$(id -g)" -e HOME=/cache -e PYTHONPATH=/tf/src \\
      -v ~/.cache/zspark/tensorfold-609ca41:/tf:ro -v ~/models:/models:ro \\
      -v ~/.cache/zspark/tensorfold-kernels:/cache -v "$PWD/tools/reference":/ref:ro \\
      -v "$PWD/zig/tests/fixtures/vision":/fix -v ~/.cache/zspark/reference-vision:/out \\
      --entrypoint python tensorfold-glm53:v0.6.0 /ref/vision_export.py /models/Qwen3.8-Flash-Next-NVFP4 \\
      --out-dir /out --fixtures /fix --tensorfold-commit 609ca41 --stage processor|layers|whole
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flash_next_export as fne  # noqa: E402  the text exporter: Tap, save, compare, greedy, provenance

PINNED = "0.6.5"
CASES = {
    "shapes": (("shapes",), "What shapes and colours are in this picture? Answer in one sentence."),
    "pair": (("bands", "tiny"), "Describe each of the two pictures in a few words."),
}


# -- the images ----------------------------------------------------------------------------------------
def make_images() -> dict[str, np.ndarray]:
    """RGB uint8 [H, W, 3], drawn with numpy alone (no font, nothing read)."""

    out = {}
    h, w = 256, 256
    y, x = np.mgrid[0:h, 0:w]
    img = np.full((h, w, 3), 255, np.uint8)
    img[(y - 92) ** 2 + (x - 80) ** 2 <= 56 ** 2] = (220, 30, 30)
    img[140:216, 132:216] = (30, 60, 210)
    img[226:242, 24:232] = (20, 150, 60)
    out["shapes"] = img
    h, w = 200, 300
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    t = y / (h - 1)
    img[..., 0] = (70 + 150 * t).astype(np.uint8)
    img[..., 1] = (130 + 100 * t).astype(np.uint8)
    img[..., 2] = (235 - 20 * t).astype(np.uint8)
    img[(y - 60) ** 2 + (x - 220) ** 2 <= 30 ** 2] = (250, 210, 40)
    ground = y >= 140 + (8 * np.sin(x / 23.0)).astype(int)
    img[ground] = (40, 140, 50)
    out["bands"] = img
    h, w = 40, 48
    y, x = np.mgrid[0:h, 0:w]
    img = np.where((((y // 8) + (x // 8)) % 2 == 0)[..., None], 255, 0).astype(np.uint8).repeat(3, axis=2)
    img[16:24, 20:28] = (230, 20, 20)
    out["tiny"] = img
    return out


def encode_files(rgb: dict[str, np.ndarray]) -> dict[str, bytes]:
    """Each image as a PNG; ``shapes`` also as JPEG, WebP, GIF and an RGBA PNG. Pillow's encoders, fixed options."""

    from PIL import Image

    files = {}
    for name, px in rgb.items():
        buf = io.BytesIO()
        Image.fromarray(px, "RGB").save(buf, format="PNG", optimize=True)
        files[f"{name}.png"] = buf.getvalue()
    shapes = Image.fromarray(rgb["shapes"], "RGB")
    for ext, fmt, opts in (("jpeg", "JPEG", {"quality": 90}), ("webp", "WEBP", {"quality": 80, "method": 4}),
                           ("gif", "GIF", {})):
        buf = io.BytesIO()
        (shapes.quantize(colors=64, method=Image.Quantize.MEDIANCUT) if fmt == "GIF" else shapes).save(
            buf, format=fmt, **opts)
        files[f"shapes.{ext}"] = buf.getvalue()
    return files


def decode_files(files: dict[str, bytes]) -> tuple[dict, dict]:
    """Every file through TensorFold's ``load_images`` (a ``data:`` URL), the GIF through Pillow. Returns the
    ``ImageInput`` of each PNG by image and each file's record (RGB, decoder, hashes)."""

    from PIL import Image
    from tensorfold.vision.images import ImageSource, load_images

    mime = {"png": "image/png", "jpeg": "image/jpeg", "webp": "image/webp"}
    inputs, records = {}, {}
    for fname, data in files.items():
        ext = fname.rsplit(".", 1)[1]
        if ext == "gif":
            with Image.open(io.BytesIO(data)) as im:
                im.seek(0)
                px = np.asarray(im.convert("RGB"))
            decoder, content_hash = "Pillow Image.open(...).convert('RGB'), first frame (TensorFold takes no GIF)", None
        else:
            url = f"data:{mime[ext]};base64," + base64.b64encode(data).decode()
            (got,) = load_images([ImageSource(url)])
            px = np.frombuffer(got.pixels, np.uint8).reshape(got.height, got.width, 3)
            decoder, content_hash = "tensorfold.vision.images.load_images (Pillow)", got.content_hash
            if fname.count(".") == 1 and ext == "png":
                inputs[fname[:-4]] = got
        records[fname] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "decoder": decoder,
                          "width": int(px.shape[1]), "height": int(px.shape[0]),
                          "rgb_sha256": hashlib.sha256(np.ascontiguousarray(px).tobytes()).hexdigest(),
                          "content_hash": content_hash, "rgb": np.ascontiguousarray(px)}
    return inputs, records


def rgb_key(fname: str) -> str:
    """images.safetensors' name: shapes.png -> shapes.rgb, shapes.jpeg -> shapes.jpeg.rgb."""

    stem, ext = fname.rsplit(".", 1)
    return f"{stem}.rgb" if ext == "png" else f"{stem}.{ext}.rgb"


# -- the patched directory --------------------------------------------------------------------------------
def patched_dir(model_dir: Path) -> tuple[Path, dict]:
    """A scratch directory of symbolic links to the checkpoint's files, ``config.json`` rewritten so 0.6.5's
    ``vision_config`` takes its ``vision_config.model_type``."""

    raw = json.loads((model_dir / "config.json").read_text())
    was = raw.get("vision_config", {}).get("model_type")
    scratch = Path(tempfile.mkdtemp(prefix="vision-config-"))
    for entry in model_dir.iterdir():
        if entry.name != "config.json":
            (scratch / entry.name).symlink_to(entry.resolve())
    raw["vision_config"]["model_type"] = "qwen4_exp"
    (scratch / "config.json").write_text(json.dumps(raw, indent=2))
    note = {"directory": str(scratch), "vision_config.model_type": {"checkpoint": was, "given": "qwen4_exp"},
            "why": "0.6.5's vision_config (qwen_cuda.py:47) takes qwen3_5 or qwen4_exp only; symbolic links to "
                   "the checkpoint's files and one rewritten config.json, the checkpoint untouched"}
    return scratch, note


# -- the processor --------------------------------------------------------------------------------------
def prepare_cases(vdir: Path, inputs: dict) -> tuple[dict, dict]:
    from tensorfold.vision.qwen_cuda import TOKENS_PER_IMAGE
    from tensorfold.vision.qwen_processing import QwenImageProcessor

    frontend = QwenImageProcessor.from_directory(vdir)
    out = {}
    for case, (names, question) in CASES.items():
        content = [{"type": "image_url", "image_url": {"url": f"{n}.png"}} for n in names]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        rendered = frontend.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                          enable_thinking=False)
        prepared = frontend.prepare(rendered, [inputs[n] for n in names], max_image_tokens=TOKENS_PER_IMAGE)
        out[case] = {"names": names, "question": question, "rendered": rendered, "prepared": prepared}
    p = frontend.processor
    settings = {"processor": type(p).__name__, "resample": int(getattr(p, "resample", -1)),
                "size": {k: v for k, v in dict(getattr(p, "size", {}) or {}).items() if v is not None},
                "min_pixels_attribute": getattr(p, "min_pixels", None),
                "max_pixels_attribute": getattr(p, "max_pixels", None),
                "tokens_per_image": TOKENS_PER_IMAGE,
                "note": "prepare passes max_pixels=min(getattr(processor, 'max_pixels', limit*32^2), limit*32^2) "
                        "and min_pixels=min(getattr(processor, 'min_pixels', 32^2), cap); the grids say what "
                        "transformers made of them"}
    return out, settings


def processor_tensors(c: dict) -> dict[str, torch.Tensor]:
    p = c["prepared"]
    return {"ids": torch.tensor(p.token_ids, dtype=torch.int32),
            "pixel_values": torch.from_numpy(np.ascontiguousarray(p.pixel_values, dtype=np.float32)),
            "image_grid_thw": torch.from_numpy(np.asarray(p.image_grid_thw, dtype=np.int64)),
            "position_ids": torch.from_numpy(np.ascontiguousarray(p.position_ids[:, 0, :], dtype=np.int32)),
            "rope_delta": torch.tensor([p.rope_delta], dtype=torch.int32),
            "image_spans": torch.tensor(p.image_spans, dtype=torch.int32).reshape(-1, 2)}


def same_bits(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[str]:
    return [k for k in a if k not in b or a[k].shape != b[k].shape or not torch.equal(a[k], b[k])]


def save_plain(path: Path, tensors: dict[str, torch.Tensor], provenance: dict) -> dict:
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path),
              metadata={"provenance": json.dumps(provenance)})
    return {"path": path.name, "bytes": path.stat().st_size, "tensors": len(tensors)}


def load_plain(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path))


# -- the tower ------------------------------------------------------------------------------------------
def tower_points(vision, pixels: np.ndarray, grid) -> dict[str, torch.Tensor]:
    """One image alone through the tower, as ``_encode`` calls it, with its blocks' outputs."""

    from torch.nn.attention import SDPBackend, sdpa_kernel

    tower, rec, hooks = vision.tower, {}, []

    def keep(name):
        return lambda _m, _i, out: rec.__setitem__(name, out.detach().clone().cpu())

    hooks.append(tower.patch_embed.register_forward_hook(keep("patch_embed")))
    hooks.append(tower.blocks[0].register_forward_pre_hook(
        lambda _m, args, kwargs=None: rec.__setitem__("blocks.0.in", args[0].detach().clone().cpu())))
    for i, blk in enumerate(tower.blocks):
        hooks.append(blk.register_forward_hook(keep(f"blocks.{i}.out")))
    hooks.append(tower.merger.norm.register_forward_hook(keep("merger.norm")))
    hooks.append(tower.merger.linear_fc1.register_forward_hook(keep("merger.fc1")))
    try:
        with torch.inference_mode(), sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            px = torch.tensor(pixels, dtype=torch.bfloat16, device=vision.device)
            grids = torch.tensor(np.asarray(grid).reshape(1, 3), dtype=torch.int64, device=vision.device)
            feats = vision.tower(px, grid_thw=grids, return_dict=True).pooler_output.to(dtype=torch.bfloat16)
            torch.cuda.synchronize()
    finally:
        for h in hooks:
            h.remove()
    rec["pixel_values"] = px.cpu()
    rec["features"] = feats.cpu()
    return rec


def run_tower(vision, cases: dict, out_dir: Path, provenance: dict) -> tuple[dict, dict]:
    """Each case's ``encode`` (what the engine embeds) and each image alone with its points."""

    encoded, report = {}, {}
    for case, c in cases.items():
        p = c["prepared"]
        enc = vision.encode(p, p.token_ids)
        torch.cuda.synchronize()
        encoded[case] = enc
        tensors = {"features": enc.features.detach().cpu(), "rows": torch.tensor(enc.rows, dtype=torch.int32)}
        done = 0
        alone = {}
        for name, grid, (begin, end) in zip(c["names"], p.image_grid_thw, p.image_spans):
            patches = int(np.prod(grid))
            pts = tower_points(vision, p.pixel_values[done:done + patches], grid)
            done += patches
            tensors[f"{name}.alone"] = pts["features"]
            row0 = enc.rows.index(begin)
            among = tensors["features"][row0:row0 + end - begin]
            alone[name] = {"tokens": end - begin, "alone_equals_among": bool(torch.equal(among, pts["features"])),
                           "max_abs": float((among.float() - pts["features"].float()).abs().max())}
            if case == next(k for k, v in cases.items() if name in v["names"]):
                save_plain(out_dir / f"{name}.tower.safetensors", pts, dict(provenance, mode="tower", image=name,
                                                                            grid=[int(x) for x in grid]))
        report[case] = {"files": save_plain(out_dir / f"{case}.tower.safetensors", tensors,
                                            dict(provenance, mode="tower", case=case)),
                        "rope_delta": enc.rope_delta, "rows": len(enc.rows), "alone": alone,
                        "features_rms": float(enc.features.float().pow(2).mean().sqrt())}
        print(f"[vision] {case}: tower {len(enc.rows)} rows; alone vs among {json.dumps(alone)}", flush=True)
    return encoded, report


# -- the language model's rows ----------------------------------------------------------------------------
def run_image(fwd, image_rows, tap, w, st, b, ids: list[int], rows: int, enc) -> dict[str, torch.Tensor]:
    """``flash_next_export.run_windows`` with the image: positions attached after the reset, each chunk's
    placeholder rows given their features through ``forward(..., features=)``. ``embed.h`` is captured by the
    ``_embed`` hook before ``compute`` writes the features, so the same rows are written into the capture."""

    st.reset(w)
    image_rows.attach(st, enc, len(ids))
    streams = w.cfg.streams
    parts = []
    for at in range(0, len(ids), rows):
        chunk = ids[at:at + rows]
        inside = [(i, p - at) for i, p in enumerate(enc.rows) if at <= p < at + len(chunk)]
        feats = None
        if inside:
            source, target = zip(*inside)
            src = enc.features.index_select(0, torch.tensor(source, dtype=torch.int64, device=enc.features.device))
            feats = (torch.tensor(target, dtype=torch.int64, device=enc.features.device), src)
        fwd.forward(w, st, b, chunk, features=feats)
        torch.cuda.synchronize()
        rec = tap.take()
        if feats is not None:
            rec["embed.h"][list(target)] = src.cpu().to(rec["embed.h"].dtype).repeat(1, streams)
        parts.append(rec)
        fwd.commit(w, st, b, len(chunk), len(chunk))
    return {name: torch.cat([p[name] for p in parts]) for name in parts[0]}


def export_image_case(fwd, image_rows, tap, w, st, b, case: str, ids: list[int], enc, widths: list[int],
                      out_dir: Path, tag: str, provenance: dict, greedy_tokens: int, engine_greedy, text,
                      window_files: bool = True) -> dict:
    out: dict = {"ids": ids, "text": text(ids)}
    windows = {}
    for n in widths:
        win = len(ids) if n <= 0 else n
        windows["window" if n <= 0 else f"window{n}"] = (win, run_image(fwd, image_rows, tap, w, st, b, ids, win, enc))
    serial = run_image(fwd, image_rows, tap, w, st, b, ids, 1, enc)
    cont = None
    if greedy_tokens:
        image_rows.finish(st)                       # as decode.prefill: features end, positions and delta stay
        first = int(serial["logits"][-1].float().argmax())
        cont = fne.greedy(fwd, tap, w, st, b, first, greedy_tokens)
        out["greedy_decode_path"] = cont
        out["greedy_decode_path_text"] = text(cont)
        if engine_greedy:
            out["greedy_engine"] = engine_greedy
    again = run_image(fwd, image_rows, tap, w, st, b, ids, 1, enc)
    rep = [p for p in fne.compare(serial, again) if not p["equal"]]
    out["serial_vs_serial_again"] = {"differing": len(rep), "first": rep[0] if rep else None}
    out["nonfinite"] = {n: int((~torch.isfinite(t.float())).sum()) for n, t in serial.items()
                        if t.is_floating_point() and not bool(torch.isfinite(t.float()).all())}
    out["argmax"] = {"serial": serial["logits"].float().argmax(dim=-1).tolist()}
    extra = {"vision.features": enc.features.detach().cpu(), "vision.rows": torch.tensor(enc.rows, dtype=torch.int32),
             "vision.positions": enc.positions.detach().cpu().to(torch.int32),
             "vision.rope_delta": torch.tensor([enc.rope_delta], dtype=torch.int32)}
    files = {}

    def write(mode: str, rec: dict, per: int) -> None:
        prov = dict(provenance, mode=mode, ids=ids, rows_per_forward=per, rope_delta=enc.rope_delta)
        if cont is not None:
            prov["greedy_decode_path"] = cont
        name = f"{case}{tag}.{mode}{provenance['suffix']}.safetensors"
        files[mode] = fne.save(out_dir / name, ids, {**rec, **extra}, prov)

    write("serial", serial, 1)
    for mode, (win, rec) in windows.items():
        points = fne.compare(serial, rec)
        differing = [p for p in points if not p["equal"]]
        out["serial_vs_" + mode] = {"rows_per_forward": win, "points": len(points), "differing": len(differing),
                                    "first": differing[0] if differing else None, "all": differing}
        out["argmax"][mode] = rec["logits"].float().argmax(dim=-1).tolist()
        if window_files:
            write(mode, rec, win)
        print(f"[vision] {case}{tag}{provenance['suffix']}: serial vs {mode}: {len(differing)} of {len(points)} "
              f"points differ" + (f", first {json.dumps(differing[0])}" if differing else ""), flush=True)
    out["files"] = files
    out["shapes"] = {n: [str(t.dtype).removeprefix("torch."), list(t.shape)] for n, t in serial.items()}
    print(f"[vision] {case}{tag}{provenance['suffix']}: a repeated serial run differs at {len(rep)} points; "
          f"non-finite {out['nonfinite']}", flush=True)
    if cont is not None:
        print(f"[vision] {case}: greedy on the decode path -> {cont} {out['greedy_decode_path_text']!r}", flush=True)
    return out


# -- main -----------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True, help="the committed fixtures' directory")
    ap.add_argument("--stage", choices=("processor", "layers", "whole"), required=True)
    ap.add_argument("--layers", type=int, default=4, help="--stage layers: the first N layers")
    ap.add_argument("--window-rows", type=int, action="append", default=[], metavar="N",
                    help="windows beside the serial steps (0: the prompt as one); default 0 and 16")
    ap.add_argument("--moe-slots", choices=("bf16", "f32", "both"), default="both")
    ap.add_argument("--greedy-tokens", type=int, default=32, help="--stage whole: the reply's length")
    ap.add_argument("--capacity", type=int, default=4096, help="context rows of the exporter's own state")
    ap.add_argument("--context", type=int, default=8192, help="--stage whole: the engine's context window")
    ap.add_argument("--tensorfold-commit", default=None, metavar="HASH")
    args = ap.parse_args()
    widths = args.window_rows or [0, 16]

    import tensorfold

    if tensorfold.__version__ != PINNED:
        print(f"[vision] TensorFold {tensorfold.__version__}, written against {PINNED}", file=sys.stderr)
    model_dir = args.model_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    commit = fne.tensorfold_commit(Path(tensorfold.__file__).resolve())
    vdir, note = patched_dir(model_dir)
    import PIL
    import transformers

    provenance = {
        "tensorfold_version": tensorfold.__version__,
        "tensorfold_commit": args.tensorfold_commit if commit == "unknown" and args.tensorfold_commit else commit,
        "checkpoint": str(model_dir), "checkpoint_revision": fne.revision(model_dir), "config_patched": note,
        "torch": torch.__version__, "transformers": transformers.__version__, "pillow": PIL.__version__,
        "numpy": np.__version__, "stage": args.stage, "suffix": "",
        "template": "apply_chat_template(add_generation_prompt=True, enable_thinking=False)",
    }
    try:
        from tensorfold.vision.qwen_cuda import vision_config

        vision_config(model_dir)
        note["checkpoint_refused"] = None
    except ValueError as err:
        note["checkpoint_refused"] = str(err)

    rgb = make_images()
    files = encode_files(rgb)
    inputs, records = decode_files(files)
    for name, px in rgb.items():
        if not np.array_equal(records[f"{name}.png"]["rgb"], px):
            raise SystemExit(f"{name}.png: TensorFold's decoder did not give the drawn pixels back")
    cases, settings = prepare_cases(vdir, inputs)
    proc = {case: processor_tensors(c) for case, c in cases.items()}
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def text(ids):
        return tok.decode(list(ids), skip_special_tokens=False)

    fixture = args.fixtures / "cases.json"
    if args.stage == "processor":
        for fname, data in files.items():
            (args.fixtures / fname).write_bytes(data)
        save_plain(args.out_dir / "images.safetensors",
                   {rgb_key(f): torch.from_numpy(r["rgb"].copy()) for f, r in records.items()},
                   dict(provenance, mode="images"))
        summary = {"provenance": {k: v for k, v in provenance.items() if k != "suffix"}, "processor": settings,
                   "images": {f: {k: v for k, v in r.items() if k != "rgb"} | {"rgb_name": rgb_key(f)}
                              for f, r in records.items()},
                   "cases": {}}
        for case, c in cases.items():
            p, t = c["prepared"], proc[case]
            save_plain(args.out_dir / f"{case}.processor.safetensors", t,
                       dict(provenance, mode="processor", case=case, rendered=c["rendered"],
                            image_hashes=list(p.image_hashes)))
            summary["cases"][case] = {
                "images": list(c["names"]), "question": c["question"], "rendered": c["rendered"],
                "ids": list(p.token_ids), "image_grid_thw": np.asarray(p.image_grid_thw).tolist(),
                "image_spans": [list(s) for s in p.image_spans], "rope_delta": int(p.rope_delta),
                "position_ids": np.asarray(p.position_ids[:, 0, :]).tolist(), "image_hashes": list(p.image_hashes),
                "pixel_values": {"shape": list(p.pixel_values.shape), "dtype": str(p.pixel_values.dtype),
                                 "sha256": hashlib.sha256(np.ascontiguousarray(p.pixel_values).tobytes()).hexdigest()},
                "visual_tokens": p.visual_tokens}
            print(f"[vision] {case}: {len(p.token_ids)} ids, grids {np.asarray(p.image_grid_thw).tolist()}, spans "
                  f"{p.image_spans}, delta {p.rope_delta}", flush=True)
        fixture.write_text(json.dumps(summary, indent=1) + "\n")
        print(f"[vision] processor: {json.dumps(settings)}; refused as stored: {note['checkpoint_refused']!r}",
              flush=True)
        return 0

    for case in cases:
        diff = same_bits(proc[case], load_plain(args.out_dir / f"{case}.processor.safetensors"))
        if diff:
            raise SystemExit(f"{case}: the processor gave other bits than the processor stage's: {diff}")

    from tensorfold.cuda import build
    from tensorfold.families.qwen4_exp.cuda import forward as fwd
    from tensorfold.families.qwen4_exp.cuda import image_rows, weight_types, weights
    from tensorfold.families.qwen4_exp.cuda.engine import build_kernels
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    engine, engine_greedy, report = None, {}, {"provenance": provenance}
    if args.stage == "whole":
        from tensorfold.families.qwen4_exp import cuda_engine

        engine = cuda_engine(vdir, no_drafts=True, vision=True, parallel=2, context=args.context)
        w, vision = engine.w, engine.vision
        for case, c in cases.items():
            ids = list(c["prepared"].token_ids)
            for draft in (True, False):
                got: list[int] = []
                try:
                    stats = engine.generate(ids, args.greedy_tokens, None, lambda new: got.extend(new) and False,
                                            draft=draft, stop_eos=False, vision=c["prepared"])
                    engine_greedy.setdefault(case, {})["own_state" if draft else "twin"] = {
                        "tokens": got, "text": text(got),
                        "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float, str, bool))}
                        if isinstance(stats, dict) else repr(stats)}
                except Exception as err:     # the decode path's reply below still comes from this load
                    engine_greedy.setdefault(case, {})["own_state" if draft else "twin"] = {
                        "error": repr(err), "tokens_before": got}
                print(f"[vision] {case}: engine generate(draft={draft}) -> "
                      f"{json.dumps(engine_greedy[case]['own_state' if draft else 'twin'])[:400]}", flush=True)
        tag = ""
    else:
        from tensorfold.vision.qwen_cuda import QwenCudaVision

        build.refuse_small_gpu()
        read = weight_types.Config.read.__func__

        def shortened(cls, directory):
            full = read(cls, directory)
            n = args.layers
            if not 1 <= n <= full.layers:
                raise ValueError(f"--layers {n}: the model has {full.layers}")
            return replace(full, layers=n, layer_types=full.layer_types[:n],
                           ple_layers=[i for i in full.ple_layers if i < n])

        weight_types.Config.read = classmethod(shortened)
        build_kernels(exl3=False, nvfp4=True, solo=True)
        w = weights.load(model_dir, "cuda", mtp=False, draft_vocab=None)
        vision = QwenCudaVision(vdir, torch.device("cuda", 0))
        tag = f".layers{args.layers}"
    torch.cuda.synchronize()
    report["load"] = {"seconds": round(time.time() - t0, 1), "layers": len(w.layers), "memory": fne.memory()}
    print(f"[vision] loaded: {json.dumps(report['load'])}", flush=True)

    encoded, tower = run_tower(vision, cases, args.out_dir, dict(provenance, suffix=None))
    report["tower"] = tower
    earlier = args.out_dir / "tower_bits.json"
    sums = {case: hashlib.sha256(e.features.detach().cpu().view(torch.int16).numpy().tobytes()).hexdigest()
            for case, e in encoded.items()}
    if earlier.exists():
        before = json.loads(earlier.read_text())
        report["tower_vs_earlier_stage"] = {case: sums[case] == h for case, h in before.items()}
        print(f"[vision] tower features equal the earlier stage's: {report['tower_vs_earlier_stage']}", flush=True)
    earlier.write_text(json.dumps(sums))

    longest = max(len(c["prepared"].token_ids) for c in cases.values())
    rows = max(8, longest)
    tap = fne.Tap(fwd)
    tap.install()
    summary = json.loads(fixture.read_text())
    try:
        for slots in ("bf16", "f32") if args.moe_slots == "both" else (args.moe_slots,):
            b = Buffers(w, rows, args.capacity, moe_prefill=slots == "bf16")
            st = State(w, args.capacity, rows, "bf16")
            suffix = "" if slots == "bf16" else ".f32"
            for case, c in cases.items():
                ids = list(c["prepared"].token_ids)
                prov = dict(provenance, moe_slots=slots, suffix=suffix, layers=len(w.layers),
                            path="forward.forward(features=) + forward.commit on state.Buffers (decode path), "
                                 "image_rows.attach's positions, no drafts, bf16 KV")
                got = export_image_case(fwd, image_rows, tap, w, st, b, case, ids, encoded[case], widths,
                                        args.out_dir, tag, prov, args.greedy_tokens if engine is not None else 0,
                                        engine_greedy.get(case), text, window_files=engine is None)
                (args.out_dir / f"{case}{tag}{suffix}.json").write_text(json.dumps(
                    {"provenance": prov, "load": report["load"], "tower": tower[case],
                     "tower_vs_earlier_stage": report.get("tower_vs_earlier_stage"), **got,
                     "memory": fne.memory()}, indent=1))
                if engine is not None:
                    summary["cases"][case].setdefault("replies", {})[f"decode_path{suffix}"] = {
                        "tokens": got["greedy_decode_path"], "text": got["greedy_decode_path_text"]}
                    if slots == "bf16":
                        summary["cases"][case]["replies"]["engine"] = engine_greedy.get(case)
            del b, st
            torch.cuda.empty_cache()
    finally:
        tap.remove()
    if engine is not None:
        summary["cases_replies_note"] = (f"{args.greedy_tokens} greedy tokens, stop_eos off: decode_path is the "
                                         "exporter's serial steps (bf16 or .f32 slots), engine TensorFold's "
                                         "served reply (cuda_engine(vision=True, parallel=2), prompt kernels)")
        fixture.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"[vision] memory at the end: {json.dumps(fne.memory())}", flush=True)
    if engine is not None:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
