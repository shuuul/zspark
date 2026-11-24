"""Export what TensorFold's draft-only NVFP4 quantizer gives, as the fixture `zig/src/weights/requantize.zig` is tested against.

TensorFold reads the MTP layer's block-FP8 experts as bf16 (`families/qwen4_exp/cuda/weights.py` `weight_bf16`,
a closure of the loader, so its expression is repeated here over `Fp8BlockLinear.column_scales`) and quantizes
them with `tensorfold.cuda.nvfp4.experts.quantize`. Both run on the device, as the loader's tensors do: there
PyTorch divides by a Python scalar by multiplying with its fp32 reciprocal, which the Zig side reproduces, so a
fixture made on the CPU would differ in a last bit of some scales.

The cases are small and synthetic: block-FP8 experts from Gaussian values and from random bytes, with zeroed
blocks of 16 and a zeroed expert, and bf16 experts built to sit on the quantizer's ties.

The fixture, little-endian:

    "ZSREQNT1"
    u32 cases;  each: u32 kind (0: block FP8, 1: bf16), u32 experts, u32 n, u32 k
                kind 0: u8 codes[experts, n, k], u16 scale_inv[experts, ceil(n / 128), k / 128] (bf16 bits)
                kind 1: u16 weight[experts, n, k] (bf16 bits)
                then    u8 words[experts, n, k / 2], u8 scales[experts, n, k / 16] (e4m3), f32 expert[experts]

Run in the TensorFold container (0.6.5, commit 609ca41); it holds a few MiB on the device for a moment:

    docker run --rm --gpus all --user "$(id -u):$(id -g)" --entrypoint python -e PYTHONPATH=/tf/src \
      -v ~/repo/TensorFold:/tf:ro -v "$PWD/tools/reference":/ref:ro -v "$PWD/tests/fixtures":/out \
      tensorfold-glm53:v0.6.0 /ref/requantize_cases.py --out /out/requantize_cases.bin
"""

from __future__ import annotations

import argparse
import struct

import torch

from tensorfold.cuda.nvfp4 import experts as nvx
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear


def block_fp8(generator: torch.Generator, experts: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 codes [experts, n, k] and bf16 block scales, as the published checkpoint stores an MTP expert."""

    codes = torch.empty((experts, n, k), dtype=torch.uint8)
    # Gaussian values over the format's range, as a quantized weight looks
    values = torch.randn((n, k), generator=generator) * 96.0
    codes[0] = values.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    # every byte but the two NaNs
    for e in range(1, experts):
        raw = torch.randint(0, 256, (n, k), generator=generator, dtype=torch.int64)
        raw = torch.where((raw & 0x7F) == 0x7F, raw & 0xFE, raw)
        codes[e] = raw.to(torch.uint8)
    codes[0, 3, 32:48] = 0                    # a block of 16 that is all zero
    codes[0, 130:, 128:] = 0                  # and a stretch of them
    if experts > 2:
        codes[2] = 0                          # an expert that is all zero
    rows, cols = -(-n // 128), k // 128
    scale_inv = (torch.rand((experts, rows, cols), generator=generator) * 4e-4 + 1e-5).to(torch.bfloat16)
    return codes, scale_inv


def dequantize(codes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """One expert as `weight_bf16` reads it: e4m3 times its block's scale in fp32, rounded to bf16."""

    n, k = codes.shape
    cols = Fp8BlockLinear.column_scales(scale_inv, n, k)
    w = codes.view(torch.float8_e4m3fn)
    return (w.float().view(-1, k // 64, 64) * cols[:, :, None]).view(-1, k).to(torch.bfloat16)


def ties() -> torch.Tensor:
    """bf16 [2, 4, 32]: values on the code midpoints, both signs, blocks too small for a normal scale or for any."""

    w = torch.zeros((2, 4, 32), dtype=torch.float32)
    # The expert's largest is 2.625 / 8 = 6 x 448 x 2^-13, so its scale is 2^-13 exactly (the fp32 reciprocal of
    # 2688 times 2.625 rounds to 2^-10) and a block's step is its e4m3 scale times 2^-13. Each value is a count
    # of steps exact in bf16, so the quotients land on the midpoints and take the smaller code.
    steps = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0,
                          -6.0])
    w[0, 0, :16] = steps * 448.0 * 2.0 ** -13
    w[0, 0, 16:] = steps.flip(0) * 28.0 * 2.0 ** -13  # a block 16 times smaller: scale 28
    w[0, 1, :16] = torch.linspace(-0.3, 0.3, 16)
    w[0, 2, 16:] = torch.linspace(-2e-6, 8e-6, 16)    # its scale is below 2^-6: a subnormal e4m3
    w[0, 3, :16] = torch.linspace(-5e-7, 5e-7, 16)    # below half the smallest scale's reach: scale 0, codes 0
    w[1, 2, 5] = -0.4375
    return w.to(torch.bfloat16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(609)

    out = bytearray(b"ZSREQNT1")
    cases = []
    for experts, n, k in ((3, 144, 256),):
        codes, scale_inv = block_fp8(generator, experts, n, k)
        w = torch.stack([dequantize(codes[e].to(device), scale_inv[e].to(device)) for e in range(experts)])
        source = codes.numpy().tobytes() + scale_inv.view(torch.int16).numpy().tobytes()
        cases.append((0, w, source))
    w = ties().to(device)
    cases.append((1, w, w.cpu().view(torch.int16).numpy().tobytes()))

    out += struct.pack("<I", len(cases))
    for kind, w, source in cases:
        words, scales, expert = nvx.quantize(w)
        e, n, k = w.shape
        assert words.shape == (e, n, k // 2) and scales.shape == (e, n, k // 16) and expert.shape == (e,)
        out += struct.pack("<IIII", kind, e, n, k) + source
        out += words.cpu().numpy().tobytes() + scales.cpu().numpy().tobytes()
        out += expert.to(torch.float32).cpu().numpy().tobytes()
        print(f"kind {kind}: [{e}, {n}, {k}], expert scales {expert.tolist()}")
    with open(args.out, "wb") as file:
        file.write(out)
    print(f"{args.out}: {len(out)} bytes")


if __name__ == "__main__":
    main()
