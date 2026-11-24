"""Export what TensorFold's keyed sampler picks, as the fixture `zig/src/decode/sampling.zig` is tested against.

It calls `tensorfold.cuda.sampling.sample_rows` on bf16 logits on the device, one row and one position a call:
the function `families/qwen4_exp/cuda/decode.py`'s `Engine.sample` reaches on one rank for the main head. That
is `argmax` at temperature 0, `torch.topk` of `top_k + MARGIN` candidates into `exact_sampling.choose_rows` with
top-k on, and `nucleus_rows` (the fixed-point mass rule) with top-k off. The seeds come from
`exact_sampling.seed_for` and the raw draws from `exact_sampling.uniform`. Because the picks go through
`torch.topk` on the device, they carry its order among equal logits (by bits, then id), which the Zig side
reproduces; a PyTorch build that orders ties differently would change the cases on the rows built to tie.

The rows are synthetic and small (their bf16 bits are stored in the fixture), plus rows of real logits read
from a reference file `flash_next_export.py` wrote; of those only the tensor's row index is stored, so the
Zig test needs the same file and skips without it.

The fixture, little-endian:

    "ZSSAMPL1"
    u32 rows;      each: u32 vocab, u16 bits[vocab]
    u32 settings;  each: f64 temperature, f64 top_p, f64 min_p, u32 top_k, u32 0
    u32 cases;     each: u32 row (bit 31 set: a row of the reference file's `logits`), u32 setting, u64 seed,
                         u64 position, u32 token
    u32 draws;     each: u64 seed, u64 position, u64 token, u64 bits of the uniform double
    u32 seeds;     each: i64 salt, u32 count, u32 tokens[count], u64 seed

Run in the TensorFold container (0.6.5, commit 609ca41); it holds a few hundred MiB on the device:

    docker run --rm --gpus all --user "$(id -u):$(id -g)" --entrypoint python -e PYTHONPATH=/tf/src \
      -v ~/repo/TensorFold:/tf:ro -v "$PWD/tools/reference":/ref:ro -v ~/.cache/zspark/reference:/data:ro \
      -v "$PWD/tests/fixtures":/out tensorfold-glm53:v0.6.0 /ref/sampling_cases.py \
      --reference /data/primes.serial.f32.safetensors --out /out/sampling_cases.bin
"""

from __future__ import annotations

import argparse
import itertools
import json
import struct

import numpy as np
import torch

from tensorfold.cuda.sampling import sample_rows
from tensorfold.engine.exact_sampling import Sampling, seed_for, uniform

TEMPERATURES = (0.0, 0.3, 1.0, 1.7)
TOP_KS = (0, 1, 20, 100)
TOP_PS = (1.0, 0.95, 0.5)
MIN_PS = (0.0, 0.05)
SEEDS = (0, 1, 42, (1 << 63) - 1, (1 << 64) - 1, -5, 0x9E3779B97F4A7C15, seed_for([248045, 846, 198]))
POSITIONS = (0, 1, 23, 4095, 100_000, (1 << 31) + 7)
REAL_ROWS = (0, 11, 22)
MASK = (1 << 64) - 1


def bf16(values: np.ndarray) -> np.ndarray:
    """fp32 values rounded to bf16 as torch rounds them, as their 16 bits."""

    return torch.from_numpy(values.astype(np.float32)).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def synthetic() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20261004)
    rows: list[tuple[str, np.ndarray]] = []

    def add(name: str, values: np.ndarray) -> None:
        rows.append((name, bf16(values)))

    add("normal, scale 1", rng.normal(0.0, 1.0, 2048))
    add("normal, scale 4", rng.normal(0.0, 4.0, 4096))
    add("normal, scale 12", rng.normal(0.0, 12.0, 1024))
    peaked = rng.normal(0.0, 2.0, 512)
    peaked[317] = 30.0
    add("peaked", peaked)
    add("flat", np.full(512, 3.25))
    zeros = np.zeros(512)
    zeros[::2] = -0.0                       # one value to argmax and lexsort, two to torch.topk
    add("flat, zeros of both signs", zeros)
    top = rng.normal(0.0, 2.0, 1024)
    top[[900, 17, 512, 3, 640]] = 11.0
    add("five tied at the top", top)
    # ranks 18..24 share a value (top-k 20 cuts among them) and so do ranks 97..103 (top-k 100)
    cut = rng.normal(0.0, 1.0, 2048)
    order = rng.permutation(2048)
    cut[order[:17]] = np.linspace(12.0, 8.0, 17)
    cut[order[17:24]] = 7.5
    cut[order[24:96]] = np.linspace(7.0, 5.0, 72)
    cut[order[96:103]] = 4.5
    add("ties at the top-k cut", cut)
    # groups of equal logits, so 0.5 and 0.95 of the mass are reached inside a group
    groups = np.full(2048, -20.0)
    order = rng.permutation(2048)
    groups[order[0]] = 10.0
    groups[order[1:5]] = 9.0
    groups[order[5:15]] = 8.0
    groups[order[15:45]] = 7.0
    groups[order[45:200]] = 5.0
    add("ties at the top-p cut", groups)
    # 40 tokens tied across ranks 10..49: more than top-k's margin of candidates, so torch.topk chooses among them
    many = rng.normal(0.0, 1.0, 2048)
    order = rng.permutation(2048)
    many[order[:10]] = np.linspace(9.0, 7.0, 10)
    many[order[10:50]] = 6.0
    add("forty tied across the top-k cut", many)
    add("integers", rng.integers(-6, 7, 2048).astype(np.float64))
    masked = rng.normal(0.0, 3.0, 1024)
    masked[rng.permutation(1024)[:512]] = -np.inf
    add("half masked", masked)
    add("nearly flat", rng.normal(0.0, 0.05, 4096))     # a nucleus wider than nucleus_rows's 1,024 candidates
    add("narrower than the candidates", rng.normal(0.0, 3.0, 64))
    # torch.topk takes +0 ahead of -0, so with 22 of them its top_k + MARGIN candidates are those and the first
    # -0s by id; the host then orders all of them by id alone
    few = np.full(512, -0.0)
    few[rng.permutation(512)[:22]] = 0.0
    add("zeros, 22 of them positive", few)
    return rows


def reference_rows(path: str) -> np.ndarray:
    """`logits` (BF16 [rows, vocab]) of a safetensors file, as its 16-bit words."""

    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        entry = header["logits"]
        if entry["dtype"] != "BF16":
            raise SystemExit(f"{path}: logits are {entry['dtype']}, expected BF16")
        begin, end = entry["data_offsets"]
        f.seek(8 + size + begin)
        return np.frombuffer(f.read(end - begin), dtype="<u2").reshape(entry["shape"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="a safetensors file with BF16 `logits`; omit to leave real rows out")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    settings = [(0.0, 1.0, 0.0, 0)]
    settings += [(t, p, m, k) for t, k, p, m in itertools.product(TEMPERATURES[1:], TOP_KS, TOP_PS, MIN_PS)]
    rows = synthetic()
    real = reference_rows(args.reference) if args.reference else None

    def pick(bits: np.ndarray, setting, seed: int, position: int) -> int:
        t, p, m, k = setting
        logits = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)[None].cuda()
        return sample_rows(logits, [position], Sampling(seed, t, k, p, m))[0]

    cases = []
    turn = 0
    for r, (_, bits) in enumerate(rows):
        for s, setting in enumerate(settings):
            for _ in range(2):                           # two keys a setting, walking both tables
                seed, position = SEEDS[turn % len(SEEDS)], POSITIONS[(turn // 3) % len(POSITIONS)]
                turn += 1
                cases.append((r, s, seed & MASK, position, pick(bits, setting, seed, position)))
    if real is not None:
        for r in REAL_ROWS:
            for s, setting in enumerate(settings):
                seed, position = SEEDS[turn % len(SEEDS)], POSITIONS[(turn // 3) % len(POSITIONS)]
                turn += 1
                cases.append((r | 1 << 31, s, seed & MASK, position, pick(real[r], setting, seed, position)))

    draws = []
    for seed, position in itertools.product(SEEDS, POSITIONS):
        ids = np.array([0, 1, 20, 248319, (1 << 31) - 1], dtype=np.int64)
        for token, value in zip(ids, uniform(seed, position, ids)):
            draws.append((seed & MASK, position, int(token), int(np.float64(value).view(np.uint64))))

    prompts = [[], [0], [248045, 846, 198], list(range(300)),
               [248045, 846, 198, 826, 279, 1118, 4097, 9944, 4947, 11, 18101, 539, 73982, 13, 248046, 198, 248045,
                74455, 198, 248068, 271, 248069, 271]]
    seeds = [(salt, prompt, seed_for(prompt, salt)) for prompt in prompts for salt in (0, 1, -7, 1 << 40)]

    out = bytearray(b"ZSSAMPL1")
    out += struct.pack("<I", len(rows))
    for _, bits in rows:
        out += struct.pack("<I", len(bits)) + bits.astype("<u2").tobytes()
    out += struct.pack("<I", len(settings))
    for t, p, m, k in settings:
        out += struct.pack("<dddII", t, p, m, k, 0)
    out += struct.pack("<I", len(cases))
    for case in cases:
        out += struct.pack("<IIQQI", *case)
    out += struct.pack("<I", len(draws))
    for draw in draws:
        out += struct.pack("<QQQQ", *draw)
    out += struct.pack("<I", len(seeds))
    for salt, prompt, seed in seeds:
        out += struct.pack("<qI", salt, len(prompt)) + struct.pack(f"<{len(prompt)}I", *prompt) + struct.pack("<Q", seed)
    with open(args.out, "wb") as f:
        f.write(out)

    for r, (name, bits) in enumerate(rows):
        print(f"row {r}: {name}, vocab {len(bits)}")
    print(f"{len(settings)} settings, {len(cases)} cases ({sum(1 for c in cases if c[0] >> 31)} on real rows), "
          f"{len(draws)} draws, {len(seeds)} seeds, {len(out)} bytes -> {args.out}")


if __name__ == "__main__":
    main()
