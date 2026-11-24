#!/usr/bin/env python3
"""Flash Next DFlash2 completed-layer taps: synthetic means, or the target's captured in the card's vLLM.

A test fixture (spec 122 WS-02). Python is allowed only here; it never runs at zspark's runtime.

The pinned contract (``~/.cache/zspark/scratch-122/tap_contract.txt``): after layer i returns, AFTER
``normalized_input = norms.output``, capture ``hidden_states`` (the completed 4-stream residual), NOT the
RMS-normed input. ``hidden_states.unflatten(-1, (4, 2560)).mean(dim=-2)`` -> bf16 [rows, 2560] for
i in {5, 15, 25, 35, 45}; concat -> [rows, 12800]. The drafter's ``fc.weight`` is BF16 [2560, 12800].

This script cannot produce those tensors from the FP4-Local target without a full model load. TensorFold no
longer reads that checkpoint (spec 121: ``csf`` scales). The SGLang patch ``qwen38-dflash2-capture.patch`` on
``qwen4_exp.py`` (sglang 3c560089) is the capture the contract pins; running it needs the target loaded.
``--from-sglang`` prints that command and exits 2. The coordinator prices the load.

Default ``--synthetic`` writes made 4-stream residuals and both rounding modes (GLM f32-acc and sequential
bf16), no checkpoint, no GPU. zspark's host tests (`taps.zig`) are the bit oracle of those means; this file is
the same arithmetic in Python so a later load can compare the real capture against both.

    python3 tools/reference/flash_next_tap_export.py --synthetic \
      --out ~/.cache/zspark/scratch-122/taps.synthetic.safetensors

    python3 tools/reference/flash_next_tap_export.py --from-sglang \
      --target ~/models/Qwen3.8-Flash-Next-FP4-Local

``--from-vllm fetch|assemble`` (spec 125 WS-02) is the real capture, in the card's vLLM image:
``flash_next_taps_vllm.sh`` boots it with ``flash_next_taps_vllm.patch`` on a scratch copy of the image's
``vllm/models/qwen4_exp/nvidia/model.py`` and ``--enforce-eager`` (no graph, so the Python hook runs), and
calls ``fetch`` as its client. The layers combine late there: layer i's completed residual (after its MoE's
writeback, before any norm) is the ``combined`` that layer i+1's ``attn_hyper_connection.combine_and_mix``
returns, cloned to the host as the hook's dump (bf16 [rows, 4 x 2560], stream-major), with the card's patch's
op on it (``unflatten(-1, (4, 2560)).mean(dim=-2)``), the positions and the ids, a forward a file set, under
the label ``fetch`` writes before a request. ``fetch`` asks, each after a prefix-cache reset, greedy,
``ignore_eos``: sky-long (256 tokens), primes (32) and prose (score.json's first 2,048 ids, 1 token), each twice
(r1, r2), then sky-long and primes again as one prompt of the prompt and r1's reply but its last token (pf, 1
token): the rows r1 decoded, prefilled. ``assemble`` (in the image, on the CPU) joins a request's forwards by
position and writes RAW/fixture/NAME.safetensors from r1 and RAW/fixture/MANIFEST.json (provenance, the
replies, vLLM's spreads, each file's sha256); NAME's tensors are those below plus ``ids`` (I32 [R]) and
``mean.{i}`` (vLLM's op), ``concat`` (of ``mean.{i}``), and the metadata's ``bound.rel`` and ``bound.max``:
twice vLLM's own spread (the larger of r1 against r2 and of the decoded rows against them prefilled), 2^-7 only
where that spread is zero (bit-equal forms; the coordinator, 2026-10-25), over the case's tap layers (MANIFEST.json's spreads hold it a layer at a time, which Z1 reads) (spec 125's plan; the runs' spread counts where r1 and r2 differ).

    ZSPARK_TAPS_LAYERS (default 5,15,25,35,45) is read by the hook in the container.

Tensor names, rows on the first axis, BF16 little-endian:

    h.{i}                 BF16 [R, 4, 2560]   made residual of completed layer i
    mean.f32_acc.{i}      BF16 [R, 2560]      GLM: f32 acc / 4, rn-even
    mean.bf16.{i}         BF16 [R, 2560]      round to bf16 after each add, then / 4
    concat.f32_acc        BF16 [R, 12800]     slots 5/15/25/35/45 side by side
    concat.bf16           BF16 [R, 12800]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

LAYERS = (5, 15, 25, 35, 45)
HIDDEN = 2560
STREAMS = 4
CONCAT = len(LAYERS) * HIDDEN
TARGET_REV = "b9ce18a6887e18c089fdea8509a33e63429a3300"
SGLANG_REV = "3c560089"
PATCH = "qwen38-dflash2-capture.patch"


def f32_from_bf16(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


def bf16_from_f32(value: float) -> int:
    packed = struct.pack("<f", value)
    bits = struct.unpack("<I", packed)[0]
    if math.isnan(value):
        return (bits >> 16) | 0x40
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF


def pack_bf16(values: list[int]) -> bytes:
    return struct.pack("<" + "H" * len(values), *values)


def mean_f32_acc(streams: list[int]) -> int:
    acc = f32_from_bf16(streams[0])
    for bits in streams[1:]:
        acc += f32_from_bf16(bits)
    return bf16_from_f32(acc / len(streams))


def mean_bf16(streams: list[int]) -> int:
    acc = streams[0]
    for bits in streams[1:]:
        acc = bf16_from_f32(f32_from_bf16(acc) + f32_from_bf16(bits))
    return bf16_from_f32(f32_from_bf16(acc) / len(streams))


def residual_value(layer: int, row: int, stream: int, dim: int) -> float:
    return (layer + 1) * 1.25 + stream * 0.375 + row * 0.0625 + dim * 0.00390625


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], bytes]], metadata: dict[str, str]) -> None:
    header: dict[str, object] = {"__metadata__": metadata}
    offset = 0
    parts: list[bytes] = []
    for name, (shape, data) in tensors.items():
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + len(data)]}
        parts.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - (len(raw) % 8)) % 8
    raw = raw + b" " * pad
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(raw)))
        fh.write(raw)
        for part in parts:
            fh.write(part)


def synthetic(rows: int, out: Path) -> None:
    tensors: dict[str, tuple[list[int], bytes]] = {}
    concat_acc = [0] * (rows * CONCAT)
    concat_bf = [0] * (rows * CONCAT)
    for slot, layer in enumerate(LAYERS):
        h_bits: list[int] = []
        acc_bits: list[int] = []
        bf_bits: list[int] = []
        for row in range(rows):
            for dim in range(HIDDEN):
                streams = [
                    bf16_from_f32(residual_value(layer, row, stream, dim)) for stream in range(STREAMS)
                ]
                h_bits.extend(streams)
                a = mean_f32_acc(streams)
                b = mean_bf16(streams)
                acc_bits.append(a)
                bf_bits.append(b)
                concat_acc[row * CONCAT + slot * HIDDEN + dim] = a
                concat_bf[row * CONCAT + slot * HIDDEN + dim] = b
        tensors[f"h.{layer}"] = ([rows, STREAMS, HIDDEN], pack_bf16(h_bits))
        tensors[f"mean.f32_acc.{layer}"] = ([rows, HIDDEN], pack_bf16(acc_bits))
        tensors[f"mean.bf16.{layer}"] = ([rows, HIDDEN], pack_bf16(bf_bits))
    tensors["concat.f32_acc"] = ([rows, CONCAT], pack_bf16(concat_acc))
    tensors["concat.bf16"] = ([rows, CONCAT], pack_bf16(concat_bf))
    metadata = {
        "spec": "122",
        "kind": "synthetic",
        "layers": ",".join(str(i) for i in LAYERS),
        "hidden": str(HIDDEN),
        "streams": str(STREAMS),
        "rows": str(rows),
        "mean_default": "f32_acc",
        "note": "made residuals; not a capture of FP4-Local. Host tests in taps.zig are the bit oracle.",
    }
    write_safetensors(out, tensors, metadata)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps({
        "out": str(out),
        "bytes": out.stat().st_size,
        "tensors": {name: {"shape": shape} for name, (shape, _) in tensors.items()},
        "metadata": metadata,
        "real_capture": "needs the FP4-Local target loaded; see --from-sglang",
    }, indent=1) + "\n")
    print(f"wrote {out} ({out.stat().st_size} bytes) and {sidecar}", flush=True)


def from_sglang(target: Path) -> int:
    print(
        "Real DFlash2 taps need the FP4-Local target loaded. TensorFold does not read it (spec 121).\n"
        "The capture the contract pins is the SGLang patch, not this repository's forward.\n\n"
        f"  target:  {target}\n"
        f"  revision:{TARGET_REV}\n"
        f"  patch:   {PATCH} on qwen4_exp.py (sglang {SGLANG_REV})\n"
        "  site:    after layer(i) returns, after normalized_input = norms.output,\n"
        "           hidden_states unflatten(-1, (4, 2560)).mean(dim=-2)\n"
        "           i in {5,15,25,35,45} -> concat bf16 [rows, 12800]\n\n"
        "Command (coordinator prices the load; do not run it from this workstream):\n\n"
        "  python3 tools/reference/flash_next_tap_export.py --from-sglang "
        f"--target {target}\n\n"
        "That flag only prints this text. A capture implementation that loads the checkpoint is WS-03+'s,\n"
        "once the owner schedules the load. Default mean in zspark is GLM f32-acc until that fixture lands.",
        file=sys.stderr,
    )
    return 2


VLLM_CASES = (("sky-long", "sky-long-drafted.json", 256), ("primes", "primes-drafted.json", 32), ("prose", "score.json", 1))
PROSE_ROWS = 2048
BOUND_FLOOR = 2.0**-7


def vllm_fetch(reference: Path, raw: Path, url: str) -> int:
    """The client of the booted server: every request of the plan, a label each, into RAW/taps and RAW/replies.json."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vllm_reference as vr

    model = vr.model_of(url)
    label = raw / "taps" / "label"

    def ask(name: str, ids: list[int], count: int) -> list[dict]:
        if not vr.reset(url):
            raise SystemExit("the server resets no prefix cache (VLLM_SERVER_DEV_MODE=1)")
        label.write_text(name)
        try:
            body = {"model": model, "prompt": ids, "max_tokens": count, "temperature": 0.0, "logprobs": vr.TOP,
                    "ignore_eos": True, "skip_special_tokens": False, "return_tokens_as_token_ids": True}
            lp = vr.post(url + "/v1/completions", body)["choices"][0]["logprobs"]
        finally:
            label.write_text("")
        steps = [{"token": vr.token_id(t), "logprob": l, "top": {str(vr.token_id(k)): v for k, v in top.items()}}
                 for t, l, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])]
        print(f"{name}: {len(ids)} prompt ids, {len(steps)} tokens", flush=True)
        return steps

    prompts = {}
    for name, file, count in VLLM_CASES:
        prompt = json.loads((reference / file).read_text())["prompt"]
        prompts[name] = (prompt[:PROSE_ROWS] if name == "prose" else prompt, count)
    # one smoke request first: the hook writes, or the boot is not the one the plan needs
    smoke = prompts["sky-long"][0]
    ask("smoke", smoke, 1)
    where = raw / "taps" / "smoke"
    if not (where / "00001.L5.h").exists():
        print("the hook wrote nothing: the forward is not the patched Python one", file=sys.stderr)
        return 1
    # one forward of the prompt's rows, unpadded, its ids the prompt's
    pos = struct.unpack(f"<{len(smoke)}q", (where / "00001.pos").read_bytes()) if (where / "00001.pos").stat().st_size == 8 * len(smoke) else None
    ids = struct.unpack(f"<{len(smoke)}q", (where / "00001.ids").read_bytes()) if (where / "00001.ids").exists() and (where / "00001.ids").stat().st_size == 8 * len(smoke) else None
    if pos != tuple(range(len(smoke))) or ids != tuple(smoke) or (where / "00001.L5.h").stat().st_size != len(smoke) * STREAMS * HIDDEN * 2:
        print(f"the smoke forward's dump is not the prompt's {len(smoke)} rows: positions {pos}, ids {ids}", file=sys.stderr)
        return 1
    replies: dict[str, dict] = {}
    for name, (prompt, count) in prompts.items():
        one = {"prompt": prompt, "count": count}
        for run in ("r1", "r2"):
            steps = ask(f"{name}.{run}", prompt, count)
            one[run] = [step["token"] for step in steps]
            one[f"{run}.steps"] = steps
        replies[name] = one
    for name in ("sky-long", "primes"):
        one = replies[name]
        steps = ask(f"{name}.pf", one["prompt"] + one["r1"][:-1], 1)
        one["pf"] = [step["token"] for step in steps]
    for name, file, _ in VLLM_CASES[:2]:
        theirs = json.loads((reference / file).read_text())["runs"][0]["tokens"]
        mine = replies[name]["r1"]
        same = next((i for i, (a, b) in enumerate(zip(mine, theirs)) if a != b), min(len(mine), len(theirs)))
        replies[name]["compiled_agree"] = same
        print(f"{name}: r1 and the compiled boot's reply ({file}) agree on {same} of {len(mine)}; "
              f"r1 == r2 {replies[name]['r1'] == replies[name]['r2']}; pf's token == r1's last "
              f"{replies[name]['pf'] == replies[name]['r1'][-1:]}", flush=True)
    (raw / "replies.json").write_text(json.dumps(replies))
    return 0


def vllm_assemble(raw: Path, owner: str | None) -> int:
    """In the image, on the CPU: the dumps joined into the fixture, vLLM's spreads, the manifest."""
    import hashlib
    import importlib.metadata
    import os

    import numpy as np
    import torch
    from safetensors.torch import save_file

    layers = tuple(int(w) for w in os.environ.get("ZSPARK_TAPS_LAYERS", ",".join(map(str, LAYERS))).split(","))
    width = STREAMS * HIDDEN

    def load(label: str) -> dict:
        where = raw / "taps" / label
        seqs = sorted({p.name.split(".")[0] for p in where.iterdir()})
        pos = np.concatenate([np.fromfile(where / f"{q}.pos", dtype=np.int64) for q in seqs])
        ids = np.concatenate([np.fromfile(where / f"{q}.ids", dtype=np.int64) for q in seqs])
        out = {"pos": pos, "ids": ids, "forwards": len(seqs)}
        for i in layers:
            h = np.concatenate([np.fromfile(where / f"{q}.L{i}.h", dtype=np.int16) for q in seqs]).reshape(-1, width)
            m = np.concatenate([np.fromfile(where / f"{q}.L{i}.mean", dtype=np.int16) for q in seqs]).reshape(-1, HIDDEN)
            assert h.shape[0] == pos.shape[0] == m.shape[0], (label, i, h.shape, pos.shape, m.shape)
            out[f"h.{i}"] = torch.from_numpy(h.copy()).view(torch.bfloat16)
            out[f"mean.{i}"] = torch.from_numpy(m.copy()).view(torch.bfloat16)
        if not np.array_equal(pos, np.arange(pos.shape[0])):
            raise SystemExit(f"{label}: positions are not 0..{pos.shape[0] - 1}")
        return out

    def f32_acc(h: torch.Tensor) -> torch.Tensor:
        s = h.view(-1, STREAMS, HIDDEN).float()
        acc = s[:, 0]
        for k in range(1, STREAMS):
            acc = acc + s[:, k]
        return (acc * (1.0 / STREAMS)).bfloat16()

    def bf16_seq(h: torch.Tensor) -> torch.Tensor:
        s = h.view(-1, STREAMS, HIDDEN)
        acc = s[:, 0]
        for k in range(1, STREAMS):
            acc = (acc.float() + s[:, k].float()).bfloat16()
        return (acc.float() * (1.0 / STREAMS)).bfloat16()

    def spread(a: torch.Tensor, b: torch.Tensor) -> dict:
        a64, b64 = a.double(), b.double()
        d = a64 - b64
        rows = d.norm(dim=-1) / b64.norm(dim=-1).clamp_min(1e-30)
        return {"equal": bool(torch.equal(a.view(torch.int16), b.view(torch.int16))),
                "rel": float(d.norm() / b64.norm()), "worst_row_rel": float(rows.max()),
                "max": float(d.abs().max() / b64.abs().max())}

    replies = json.loads((raw / "replies.json").read_text())
    fixture = raw / "fixture"
    fixture.mkdir(exist_ok=True)
    spreads: dict[str, dict] = {}
    files: dict[str, str] = {}
    for name, _, _ in VLLM_CASES:
        one = replies[name]
        r1 = load(f"{name}.r1")
        r2 = load(f"{name}.r2")
        expected = one["prompt"] + one["r1"][:-1]
        if r1["ids"].tolist() != expected or r2["ids"].tolist() != one["prompt"] + one["r2"][:-1]:
            raise SystemExit(f"{name}: the dumped ids are not the prompt and the reply")
        rows = len(expected)
        pf = load(f"{name}.pf") if "pf" in one else None
        if pf is not None and pf["ids"].tolist() != expected:
            raise SystemExit(f"{name}.pf: the dumped ids are not r1's rows")
        tensors = {"ids": torch.tensor(expected, dtype=torch.int32)}
        case: dict[str, dict] = {}
        floor = {"rel": BOUND_FLOOR, "max": BOUND_FLOOR}
        bound = {"rel": 0.0, "max": 0.0}
        for i in layers:
            h, mean = r1[f"h.{i}"], r1[f"mean.{i}"]
            acc, seq = f32_acc(h), bf16_seq(h)
            entry = {"runs": spread(r2[f"mean.{i}"], mean) if r2["ids"].tolist() == expected else None,
                     "runs_h": spread(r2[f"h.{i}"], h) if r2["ids"].tolist() == expected else None,
                     "op_vs_f32_acc": int((mean.view(torch.int16) != acc.view(torch.int16)).sum()),
                     "op_vs_bf16": int((mean.view(torch.int16) != seq.view(torch.int16)).sum())}
            if pf is not None:
                decoded = slice(len(one["prompt"]), rows)
                entry["prefill"] = spread(pf[f"mean.{i}"][decoded], mean[decoded])
                entry["prefill_h"] = spread(pf[f"h.{i}"][decoded], h[decoded])
            # twice vLLM's own spread, the larger of its two runs' and its decoded rows against them prefilled
            # 2^-7 only where vLLM's spread is zero (its forms bit-equal); everywhere V1 measured, the spread decides
            own = {"rel": 0.0, "max": 0.0}
            for kind in ("runs", "prefill"):
                if entry.get(kind) is not None:
                    for key in ("rel", "max"):
                        own[key] = max(own[key], 2 * entry[kind][key])
            for key in ("rel", "max"):
                if own[key] == 0.0:
                    own[key] = floor[key]
            entry["bound"] = own
            for key in ("rel", "max"):
                bound[key] = max(bound[key], own[key])
            case[str(i)] = entry
            tensors[f"h.{i}"] = h.reshape(rows, STREAMS, HIDDEN).contiguous()
            tensors[f"mean.{i}"] = mean.contiguous()
            tensors[f"mean.f32_acc.{i}"] = acc.contiguous()
            tensors[f"mean.bf16.{i}"] = seq.contiguous()
        tensors["concat"] = torch.cat([tensors[f"mean.{i}"] for i in layers], dim=-1).contiguous()
        case["bound"] = bound
        case["forwards"] = {"r1": r1["forwards"], "r2": r2["forwards"], "pf": pf["forwards"] if pf else None}
        spreads[name] = case
        metadata = {"spec": "125", "kind": "vllm", "case": name, "layers": ",".join(map(str, layers)),
                    "hidden": str(HIDDEN), "streams": str(STREAMS), "rows": str(rows), "prompt": str(len(one["prompt"])),
                    "target_revision": TARGET_REV, "bound.rel": repr(bound["rel"]), "bound.max": repr(bound["max"]),
                    "note": "r1 of the card's vLLM, eager; mean.{i} is vLLM's op on h.{i}; MANIFEST.json beside; a bound is twice "
                            "vLLM's own spread, 2^-7 only where that spread is zero (the coordinator, 2026-10-25)"}
        path = fixture / f"{name}.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{name}: {rows} rows, bound rel {bound['rel']:.3e} max {bound['max']:.3e}", flush=True)
    provenance = json.loads((raw / "provenance.json").read_text(), strict=False) if (raw / "provenance.json").exists() else {}
    manifest = {"spec": "125 WS-02 V1", "provenance": provenance, "target_revision": TARGET_REV,
                "versions": {"vllm": importlib.metadata.version("vllm"), "torch": torch.__version__},
                "patch_sha256": (raw / "patch.sha256").read_text().split()[0] if (raw / "patch.sha256").exists() else None,
                "eager": True, "layers": list(layers),
                "bound_note": "per tap layer and figure, twice the larger of vLLM's r1-r2 and decode/prefill spreads; "
                              "2^-7 only where that spread is zero (bit-equal r1/r2 and forms). Set after V1's data, "
                              "accepted by the coordinator 2026-10-25",
                "requests": {name: {k: v for k, v in one.items() if not k.endswith(".steps")} for name, one in replies.items()},
                "spreads": spreads, "files": files}
    (fixture / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if owner:
        for path in [raw / "taps", raw / "fixture"]:
            os.system(f"chown -R {owner} {path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true", default=False,
                    help="write made 4-stream residuals and both means; no checkpoint (default if neither flag)")
    ap.add_argument("--from-sglang", action="store_true", default=False,
                    help="print the real-capture command and exit 2; needs --target")
    ap.add_argument("--from-vllm", choices=("fetch", "assemble"), default=None,
                    help="the vLLM capture: fetch (the client, URL last) or assemble (in the image)")
    ap.add_argument("--reference", type=Path, default=None, help="fetch: vllm_reference.sh's output for FP4-Local")
    ap.add_argument("--raw", type=Path, default=None, help="--from-vllm: the run's directory (taps/, replies.json)")
    ap.add_argument("--owner", default=None, help="assemble: uid:gid the dumps and the fixture are given to")
    ap.add_argument("url", nargs="?", default=None, help="fetch: the server's URL")
    ap.add_argument("--target", type=Path, default=None,
                    help="FP4-Local directory; required with --from-sglang")
    ap.add_argument("--rows", type=int, default=3, help="synthetic rows (default 3)")
    ap.add_argument("--out", type=Path,
                    default=Path.home() / ".cache/zspark/scratch-122/taps.synthetic.safetensors")
    args = ap.parse_args()
    if args.from_vllm == "fetch":
        if args.reference is None or args.raw is None or args.url is None:
            ap.error("--from-vllm fetch needs --reference, --raw and the URL")
        return vllm_fetch(args.reference, args.raw, args.url.rstrip("/"))
    if args.from_vllm == "assemble":
        if args.raw is None:
            ap.error("--from-vllm assemble needs --raw")
        return vllm_assemble(args.raw, args.owner)
    if args.from_sglang:
        if args.target is None:
            ap.error("--from-sglang needs --target DIR")
        return from_sglang(args.target)
    if args.rows < 1:
        ap.error("--rows must be >= 1")
    synthetic(args.rows, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
