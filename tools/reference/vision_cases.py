#!/usr/bin/env python3
"""TensorFold 0.6.5's image processor on chosen sizes and budgets: its grids and refusals, and its resize's bytes.

A test fixture (spec 019 WS-01), beside ``vision_export.py``: no GPU, no lock, no model load. It runs inside the
TensorFold container, never at zspark's runtime, and calls ``QwenImageProcessor.from_directory(CKPT).prepare``
as ``QwenCudaVision.prepare`` does (``max_image_tokens=TOKENS_PER_IMAGE``) on images it makes itself:

- ``probes``: each a list of image sizes (random pixels), an optional ``detail`` and ``prepare``'s keyword
  arguments (``max_visual_tokens``, ``max_prompt_tokens``); what came back: the grids, the expanded prompt's
  length and the delta, or the exception's text. The rendered prompt is ``RENDERED`` with one image marker an
  image.
- ``random``: images of noise and of stepped gradients at sizes and budgets chosen to take every path of the
  resize (both axes up, both down, one kept, a strong shrink, a side 200 times the other); for each its grid,
  delta, ``content_hash`` and the sha256 of ``pixel_values`` (f32 bytes), and in
  ``--out-dir``/``processor_more.safetensors`` its RGB (``rI.rgb`` U8 [h, w, 3]) and the processor's resized
  bytes (``rI.resized`` U8 [H, W, 3]), read back from ``pixel_values``: every patch row's two frames are checked
  equal and every value checked to be one of torchvision's 256, ``(f32(x) - 127.5) / 127.5``, before its byte is
  taken.

Writes ``--fixtures``/``processor_cases.json`` (committed) and ``processor_more.safetensors``. The command (the
checkpoint's parent mounted at ``/models``, as ``vision_export.py`` says):

    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e PYTHONPATH=/tf/src \\
      -v ~/.cache/zspark/tensorfold-609ca41:/tf:ro -v ~/models:/models:ro -v ~/.cache/huggingface:$HOME/.cache/huggingface:ro \\
      -v "$PWD/tools/reference":/ref:ro -v "$PWD/zig/tests/fixtures/vision":/fix \\
      -v ~/.cache/zspark/reference-vision:/out --entrypoint python tensorfold-glm53:v0.6.0 \\
      /ref/vision_cases.py /models/Qwen3.8-Flash-Next-NVFP4 --out-dir /out --fixtures /fix
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

PINNED = "0.6.5"
SEED = 19
RENDERED = "<|im_start|>user\n{images}Hi.<|im_end|>\n<|im_start|>assistant\n"
MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

# (name, [(width, height), ...], detail, prepare's keyword arguments)
PROBES = [
    ("one_pixel", [(1, 1)], "auto", {}),
    ("ten", [(10, 10)], "auto", {}),
    ("under_floor", [(31, 33)], "auto", {}),
    ("thin", [(300, 5)], "auto", {}),
    ("aspect_201", [(1608, 8)], "auto", {}),
    ("aspect_200", [(1600, 8)], "auto", {}),
    ("aspect_400", [(2000, 5)], "auto", {}),
    ("big", [(5000, 3000)], "auto", {}),
    ("four_big", [(3000, 3000)] * 4, "auto", {}),
    ("low", [(3000, 3000)], "low", {}),
    ("one_token", [(64, 64)], "auto", {"max_visual_tokens": 1}),
    ("floor_past_budget", [(300, 5)], "auto", {"max_visual_tokens": 1}),
    ("count_past_budget", [(64, 64)] * 3, "auto", {"max_visual_tokens": 2}),
    ("two_share", [(1000, 1000), (50, 50)], "auto", {"max_visual_tokens": 100}),
    ("past_context", [(512, 512)], "auto", {"max_prompt_tokens": 100}),
    ("within_context", [(512, 512)], "auto", {"max_prompt_tokens": 300}),
    ("no_image", [], "auto", {}),
    ("wide", [(8192, 41)], "auto", {}),
]

# (width, height, max_visual_tokens)
RANDOM = [
    (300, 200, 4096), (48, 40, 4096), (1000, 750, 4096), (1000, 750, 64), (33, 97, 4096), (640, 480, 4096),
    (17, 1500, 256), (1920, 1080, 1024), (129, 65, 4096), (2000, 10, 4096), (511, 513, 4096), (800, 600, 16),
]


def run(fe, images, kw):
    from tensorfold.vision.qwen_cuda import TOKENS_PER_IMAGE

    rendered = RENDERED.format(images=MARKER * len(images))
    return fe.prepare(rendered, images, max_image_tokens=TOKENS_PER_IMAGE, **kw)


def resized_bytes(pv: np.ndarray, grid, p: int = 16, m: int = 2, t: int = 2) -> np.ndarray:
    """The processor's resized RGB from its patches (merge order, each row [channel][frame][y][x])."""

    lut = ((torch.arange(256, dtype=torch.float32) - 127.5) / 127.5).numpy()
    _, gh, gw = (int(x) for x in grid)
    out = np.zeros((gh * p, gw * p, 3), np.uint8)
    n = 0
    for by in range(gh // m):
        for bx in range(gw // m):
            for my in range(m):
                for mx in range(m):
                    row = pv[n].reshape(3, t, p, p)
                    assert all((row[:, 0] == row[:, k]).all() for k in range(t)), "frames differ"
                    idx = np.searchsorted(lut, row[:, 0])
                    assert (lut[np.minimum(idx, 255)] == row[:, 0]).all(), "a value off torchvision's 256"
                    y0, x0 = (by * m + my) * p, (bx * m + mx) * p
                    out[y0:y0 + p, x0:x0 + p] = idx.transpose(1, 2, 0)
                    n += 1
    return out


def image(rng, w: int, h: int, stepped: bool):
    rgb = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    if stepped:   # flat steps and an edge every 7 columns, where the kernel's lobes show
        rgb = np.clip(rgb.astype(int) // 64 * 64 + np.indices((h, w))[1][..., None] % 7 * 30, 0, 255).astype(np.uint8)
    return rgb


def main() -> int:
    from tensorfold.vision.images import ImageInput
    from tensorfold.vision.qwen_processing import QwenImageProcessor
    import tensorfold

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    args = ap.parse_args()
    if getattr(tensorfold, "__version__", PINNED) != PINNED:
        raise SystemExit(f"pinned to TensorFold {PINNED}")
    import PIL
    import torchvision
    import transformers

    fe = QwenImageProcessor.from_directory(args.model_dir)
    rng = np.random.default_rng(SEED)
    doc = {"provenance": {"tensorfold_version": PINNED, "checkpoint": args.model_dir.name,
                          "transformers": transformers.__version__, "torch": torch.__version__,
                          "torchvision": torchvision.__version__, "pillow": PIL.__version__, "seed": SEED,
                          "rendered": RENDERED, "marker": MARKER},
           "probes": [], "random": []}
    for name, sizes, detail, kw in PROBES:
        images = [ImageInput(w, h, image(rng, w, h, False).tobytes(), detail) for w, h in sizes]
        row = {"name": name, "sizes": [list(s) for s in sizes], "detail": detail, "kwargs": kw}
        try:
            p = run(fe, images, kw)
            row.update(grids=np.asarray(p.image_grid_thw).tolist(), ids=len(p.token_ids), delta=int(p.rope_delta))
        except ValueError as exc:
            row["error"] = str(exc)
        doc["probes"].append(row)
        print(row, flush=True)
    tensors = {}
    for i, (w, h, budget) in enumerate(RANDOM):
        rgb = image(rng, w, h, bool(i % 2))
        img = ImageInput(w, h, rgb.tobytes())
        p = run(fe, [img], {"max_visual_tokens": budget})
        pv = np.ascontiguousarray(p.pixel_values, dtype=np.float32)
        grid = np.asarray(p.image_grid_thw)[0].tolist()
        tensors[f"r{i}.rgb"] = torch.from_numpy(rgb)
        tensors[f"r{i}.resized"] = torch.from_numpy(resized_bytes(pv, grid))
        doc["random"].append({"name": f"r{i}", "width": w, "height": h, "max_visual_tokens": budget, "grid": grid,
                              "ids": len(p.token_ids), "delta": int(p.rope_delta), "content_hash": img.content_hash,
                              "pixel_values_sha256": hashlib.sha256(pv.tobytes()).hexdigest()})
        print(doc["random"][-1], flush=True)
    from safetensors.torch import save_file

    save_file(tensors, str(args.out_dir / "processor_more.safetensors"),
              metadata={"provenance": json.dumps(doc["provenance"])})
    (args.fixtures / "processor_cases.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
