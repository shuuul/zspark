"""Export what flashinfer's b12x loader gives for `csf_32x2_v1` scale planes, as the fixture
`zig/src/weights/csf_blocks.zig`'s host decoder is tested against (spec 107 WS-01).

Everything comes from `flashinfer/experimental/b12x/loader/_csf_blocks.py` as the card's vLLM container
(`eugr/spark-vllm-b12x`) ships it, loaded by its path so that no torch or CUDA is imported:
`_from_logical_components` encodes, `CsfBlockPlane.validate` checks and `logical_csf_components` decodes. No GPU,
no server: numpy on the CPU.

The cases:

- planes: random scale planes of shapes that cover a block of one column, clipped tiles at the bottom and the
  right, and several pages of 32 tile groups; each plane's bytes (a row's base, a 4-bit offset, about one in
  twenty an exception of any byte) and its five streams as the encoder makes them;
- with `--checkpoint DIR`: the SHA-256 of a few of the checkpoint's stored planes as `logical_csf_components`
  decodes them, by tensor name (`NAME.weight_scale.csf32_*`), and their shapes. Hashes only: no byte of the
  checkpoint is written.

The fixture, little-endian:

    "ZSCSF321"
    u32 planes; each:   u32 rows, u32 cols, u8 logical[rows, cols]
                        u32 counts[5] (bases, codes, page_bases, offsets, exceptions; elements)
                        u8 bases[], u8 codes[], u32 page_bases[], u16 offsets[], u16 exceptions[]
    u32 stored; each:   u32 name length, the name, u32 rows, u32 cols, u8 sha256[32]

Run in the card's container, without the GPU or the network:

    docker run --rm --network none --user "$(id -u):$(id -g)" --entrypoint python3 \
      -v "$PWD/tools/reference":/ref:ro -v "$PWD/zig/tests/fixtures":/out \
      -v ~/models/GLM-5.3-Flash-FP4-Local/:/ckpt:ro \
      eugr/spark-vllm-b12x:latest -I /ref/csf_blocks_cases.py --out /out/csf_blocks_cases.bin --checkpoint /ckpt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import sys

import numpy as np

DEFAULT_MODULE = "/usr/local/lib/python3.12/dist-packages/flashinfer/experimental/b12x/loader/_csf_blocks.py"

# (rows, cols): a single column; two and three; clipped tiles both ways; 34 tiles (two pages); 51 (two, clipped)
SHAPES = [(32, 1), (32, 2), (32, 3), (64, 5), (96, 7), (160, 9), (128, 4), (256, 68), (288, 41)]

STORED = [
    "model.language_model.layers.3.mlp.experts.0.gate_proj",
    "model.language_model.layers.3.mlp.experts.0.up_proj",
    "model.language_model.layers.3.mlp.experts.0.down_proj",
    "model.language_model.layers.44.mlp.experts.287.gate_proj",
    "model.language_model.layers.44.mlp.experts.287.up_proj",
    "model.language_model.layers.44.mlp.experts.287.down_proj",
]

PARTS = ("bases", "codes", "page_bases", "offsets", "exceptions")


def load(path):
    spec = importlib.util.spec_from_file_location("_csf_blocks", path)
    module = importlib.util.module_from_spec(spec)
    # a dataclass looks its module up by name
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def logical_of(blocks, plane):
    bases, fields, positions, values = blocks.logical_csf_components(plane)
    logical = (bases[:, None].astype(np.uint16) + fields).astype(np.uint8)
    logical.ravel()[positions] = values
    return logical


def synthetic(blocks, rng, rows, cols):
    bases = rng.integers(0, 241, size=rows, dtype=np.uint8)
    fields = rng.integers(0, 16, size=(rows, cols), dtype=np.uint8)
    outlier = rng.random((rows, cols)) < 0.05
    positions = np.flatnonzero(outlier)
    values = rng.integers(0, 256, size=positions.size, dtype=np.uint8)
    fields[outlier] = 0
    plane = blocks._from_logical_components(bases, fields, positions, values, "nvfp4")
    plane.validate()
    logical = (bases[:, None].astype(np.uint16) + fields).astype(np.uint8)
    logical.ravel()[positions] = values
    assert np.array_equal(logical_of(blocks, plane), logical)
    return logical, plane


def tensors(directory, names):
    """The named tensors from the checkpoint's shards, read by their headers (numpy alone)."""
    with open(os.path.join(directory, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    out = {}
    for name in names:
        with open(os.path.join(directory, weight_map[name]), "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(size))
            entry = header[name]
            begin, end = entry["data_offsets"]
            f.seek(8 + size + begin)
            dtype = {"U8": "u1", "U16": "<u2", "U32": "<u4"}[entry["dtype"]]
            out[name] = np.frombuffer(f.read(end - begin), dtype=dtype).reshape(entry["shape"])
    return out


def stored(blocks, directory, module):
    names = [f"{module}.weight_scale.csf32_{part}" for part in PARTS]
    found = tensors(directory, names)
    bases = found[names[0]]
    rows = bases.size
    cols = found[names[1]].size * 2 // rows
    plane = blocks.CsfBlockPlane(rows, cols, "nvfp4", *(found[name] for name in names))
    plane.validate()
    return rows, cols, hashlib.sha256(logical_of(blocks, plane).tobytes()).digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--module", default=DEFAULT_MODULE)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    blocks = load(args.module)
    rng = np.random.default_rng(107)

    out = bytearray(b"ZSCSF321")
    out += struct.pack("<I", len(SHAPES))
    for rows, cols in SHAPES:
        logical, plane = synthetic(blocks, rng, rows, cols)
        out += struct.pack("<II", rows, cols) + logical.tobytes()
        parts = [getattr(plane, part) for part in PARTS]
        out += struct.pack("<5I", *(part.size for part in parts))
        for part in parts:
            out += part.astype(part.dtype.newbyteorder("<")).tobytes()

    found = [(module, *stored(blocks, args.checkpoint, module)) for module in STORED] if args.checkpoint else []
    out += struct.pack("<I", len(found))
    for module, rows, cols, digest in found:
        out += struct.pack("<I", len(module)) + module.encode() + struct.pack("<II", rows, cols) + digest
    with open(args.out, "wb") as f:
        f.write(out)
    print(f"{args.out}: {len(SHAPES)} planes, {len(found)} stored, {len(out)} bytes")


if __name__ == "__main__":
    main()
