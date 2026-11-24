#!/usr/bin/env python3
"""GLM-5.3-Flash's DFlash2 drafter alone on made taps, both ranks in one process, exported from TensorFold itself.

A test fixture (spec 067 WS-13): it runs inside the TensorFold container, never at zspark's runtime, and beside
``glm5_export.py``, whose two rank threads and stand-in ``comm`` it reuses (``Hub``, ``ThreadComm``,
``run_ranks``, ``merge``, ``versus``). It runs ``families/glm5_next/cuda/dflash2.py:Drafter`` of TensorFold 0.6.6
(commit cb2ebf0, an archive) and writes what its two passes leave: the context rows ``add_taps`` makes of the
taps, and one block pass after them, point by point. ``zig/tests/models/glm5/dflash2/chain_reference_test.zig``
reads ``NAME.chain.json``; the test of the rest ran on the EXL3 pack, retired, and was deleted (spec 121 WS-00,
tag ``retired-checkpoints-2026-10-21``). TensorFold's GLM reads the pack alone: MODEL_DIR is that pack's directory.

No main model runs. The taps are made: ``torch.randn`` from the case's seed, rounded to bf16, [positions, 5 x
4096]; so every row here is the drafter's arithmetic on noise, a reference for its kernels and wiring and never
for a draft. The drafter is given the smallest ``weights.Weights`` it reads, made by hand and not by
``weights.load``: the device, ``rank``, ``world``, ``comm``, ``cfg`` (for ``vocab_offset``), the pack's
``embed_tokens`` and the rank's half of ``lm_head`` as stored (a ``qmm.B16``; ``draft_head`` None), read through
``split.RankReader``. No main layer is loaded.

BF16 faces. TensorFold drafts from affine 4-bit copies (``dflash2._quantize4``). Here ``dflash2._quantize4`` is
replaced, at run time and from outside, by ``qmm.make_b16``: ``qmm.matmul`` takes a ``B16`` as it takes a ``Q4``
and ignores the group sums, so ``_mm`` and ``_row`` run unchanged and every linear is the stored BF16 weight.
The head is ``lm_head`` as stored for the same reason. So the architecture is compared apart from a
quantization; TensorFold's own drafts (4-bit) are another export's.

The drafter's weights are CC BY-NC-ND 4.0. This writes activations only, under the ``--out-dir`` given
(``~/.cache/zspark/reference-glm-drafter/``, never the repository); no tensor of the drafter is written.

It relies on these internals of ``dflash2.py``, all wrapped from outside and none edited: ``_quantize4``, ``_mm``
and ``_dconv`` (module functions ``Drafter`` calls by name); ``Drafter._norm``, ``_prep``, ``_row``, ``_layer``;
``Drafter.add_taps``, ``candidates``, ``reset``; its ``layers`` (``a_kp``, ``m_kp``, ``qkv``, ``gu``), ``kc``,
``vc``, ``ring``, ``cap``, ``ids``, ``packed``, ``proj``, ``top_k``, ``block``, ``D``.

A case is ``NAME=POSITIONS,PENDING,SEED``: POSITIONS rows of taps added from an empty context in updates of 64
(``Drafter.tap_in``), then one block pass with PENDING first. ``NAME.safetensors`` holds, rows on the first axis
(P positions, B = 8 block rows, D = 4096; ``.r0``/``.r1`` a rank's own, the others the same bits on both ranks,
which is checked and stored once):

    taps                       BF16 [P, 5*D]      the made taps
    ctx.fc                     BF16 [P, D]        fc on the taps (_norm's input in _taps_compute)
    ctx.normed                 BF16 [P, D]        hidden_norm: a context row
    ctx.{i}.k.r{k}, .v.r{k}    BF16 [P, 4, 128]   layer i's K and V of the rank's KV heads, by position (_prep)
    block.ids                  I32  [B]           pending, then the mask token
    block.embed                BF16 [B, D]        layer 0's input
    block.{i}.attn.normed      BF16 [B, D]        input_layernorm
    block.{i}.attn.dyn         BF16 [B, 1024]     attention_conv.kernel_projection
    block.{i}.attn.mixed       BF16 [B, D]        the convolution's branch 0
    block.{i}.attn.qkv.r{k}    BF16 [B, 3072]     [q of 16 heads | k of 4 | v of 4]
    block.{i}.attn.q.r{k}      BF16 [B, 16, 128]  normed and rotated (_prep); .k.r{k}, .v.r{k} [B, 4, 128]
    block.{i}.attn.out.r{k}    BF16 [B, 2048]     the attention's rows, o_proj's input
    block.{i}.attn.gath        F32  [B, 2, D]     the gathered partials of o_proj, axis 1 the rank
    block.{i}.x_mid            BF16 [B, D]        branch 1 with the residual
    block.{i}.mlp.normed, .dyn, .mixed            the MLP's, as the attention's
    block.{i}.mlp.gu.r{k}      BF16 [B, 12288]    [gate | up] of the rank's 6144
    block.{i}.mlp.act.r{k}     BF16 [B, 6144]     SwiGLU, down_proj's input
    block.{i}.mlp.gath         F32  [B, 2, D]
    block.{i}.x_out            BF16 [B, D]        the layer's output
    block.final                BF16 [B-1, D]      norm on rows 1 ..
    block.logits.r{k}          BF16 [B-1, 77440]  the rank's rows of lm_head, as stored
    block.top.values, .ids     F32, I32 [B-1, 2, 16]  each rank's torch.topk as gathered, axis 1 the rank
    block.projected            F32  [B-1, 256]    hidden_projection
    candidates.tokens, .values I32, F32 [B-1, 16] Drafter.candidates' merge

``NAME.chain.json`` has the merged candidates, the projected rows and ``Drafter.chain``'s picks on them
(``chains``; no codebook row: ``chain_reference_test.zig`` reads those from the drafter's directory).
``NAME.json`` has the shapes, the provenance (in the file's metadata too) and three checks of TensorFold against
itself: ``ranks_differ`` (replicated points whose bits differ between the ranks), ``ring_holds`` (the ring's
slots are the rows captured, by position) and ``flat_in_pieces`` (a second drafter with a flat context, the same
taps added in updates of 1 to 64 rows: every point against the first run's).

    . ~/.cache/zspark/env.sh; mkdir -p ~/.cache/zspark/reference-glm-drafter
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -e HOME=/cache -e PYTHONPATH=/tf/src \
      -v ~/.cache/zspark/tensorfold-cb2ebf0:/tf:ro -v ~/.cache/huggingface:/hf:ro \
      -v ~/.cache/zspark/tensorfold-kernels:/cache -v "$PWD/tools/reference":/ref:ro \
      -v ~/.cache/zspark/reference-glm-drafter:/out --entrypoint python tensorfold-glm53:v0.6.0 \
      /ref/glm5_drafter_export.py "/hf/${PACK#$HOME/.cache/huggingface/}" \
      --drafter "/hf/${ZSPARK_GLM_DRAFTER#$HOME/.cache/huggingface/}" --out-dir /out \
      --tensorfold-commit cb2ebf0540f42604e2759b2ddef497861e928248 --case short=37,1234,1 --case long=2376,1234,2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import glm5_export as base

WORLD = base.WORLD
STEPS = ("attn.mixed", "x_mid", "mlp.mixed", "x_out")        # a layer's four convolutions, in _layer's order


def tap():
    return getattr(base.LOCAL, "tap", None)


class Tap:
    """One rank's captures of a pass: clones, since the drafter's buffers are reused by the next layer."""

    def __init__(self, rank: int) -> None:
        self.rank = rank
        self.d = None
        self.rec: dict[str, torch.Tensor] = {}
        self.rows: dict[str, list[torch.Tensor]] = {}    # context points: an entry an update
        self.split: set[str] = set()
        self.names: dict[int, tuple[str, bool]] = {}     # a weight's id -> the point its product is
        self.phase = None                                # "ctx" or "block"
        self.layer = None
        self.convs = 0                                   # convolutions the layer has run
        self.gath = None                                 # the next gather's name

    def begin(self, d, head, phase: str) -> None:
        self.d, self.phase, self.layer = d, phase, None
        self.names = {id(head): ("logits", True)}
        for L in d.layers:
            self.names.update({id(L.a_kp): ("attn.dyn", False), id(L.qkv): ("attn.qkv", True),
                               id(L.m_kp): ("mlp.dyn", False), id(L.gu): ("mlp.gu", True)})

    def side(self) -> str:
        return "attn" if self.convs < 2 else "mlp"

    def where(self, name: str) -> str:
        return f"block.{name}" if self.layer is None else f"block.{self.layer}.{name}"

    def put(self, name: str, t: torch.Tensor, split: bool = False) -> None:
        self.rec[name] = t.detach().clone()
        if split:
            self.split.add(name)

    def add(self, name: str, t: torch.Tensor, split: bool = False) -> None:
        self.rows.setdefault(name, []).append(t.detach().clone())
        if split:
            self.split.add(name)

    def take(self) -> dict[str, torch.Tensor]:
        got = {name: t.cpu().contiguous() for name, t in self.rec.items()}
        got.update({name: torch.cat(parts).cpu().contiguous() for name, parts in self.rows.items()})
        self.rec, self.rows, self.phase = {}, {}, None
        return got


class TapComm(base.ThreadComm):
    """``ThreadComm`` that leaves a named gather with the rank's Tap: [world, rows, D] as [rows, world, D]."""

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        super().all_gather(send, recv)
        t = tap()
        if t is not None and t.gath is not None:
            t.put(t.gath, recv.view(WORLD, -1, t.d.D).permute(1, 0, 2))
            t.gath = None


def install(dfl) -> None:
    """Wrap what ``Drafter``'s passes call; a thread without a Tap passes through."""

    o_mm, o_dconv = dfl._mm, dfl._dconv
    D = dfl.Drafter
    o_norm, o_prep, o_row, o_layer = D._norm, D._prep, D._row, D._layer

    def mm(x, w, xs=None, *, f32=False):
        out = o_mm(x, w, xs, f32=f32)
        t = tap()
        if t is not None and t.phase == "block" and id(w) in t.names:
            name, split = t.names[id(w)]
            t.put(t.where(name), out, split)
        return out

    def dconv(x, dyn, base_kernel, branch, group_size, residual=None):
        out = o_dconv(x, dyn, base_kernel, branch, group_size, residual)
        t = tap()
        if t is not None and t.phase == "block":
            t.put(t.where(STEPS[t.convs]), out)
            t.convs += 1
        return out

    def norm(self, x, weight):
        out, xs = o_norm(self, x, weight)
        t = tap()
        if t is not None and t.phase == "ctx":
            t.add("ctx.fc", x)
            t.add("ctx.normed", out)
        elif t is not None and t.phase == "block":
            t.put(t.where("final" if t.layer is None else t.side() + ".normed"), out)
        return out, xs

    def prep(self, qkv, L, cos, sin, heads):
        q, k, v = o_prep(self, qkv, L, cos, sin, heads)
        t = tap()
        if t is not None and t.phase == "ctx":
            i = next(i for i, one in enumerate(self.layers) if one is L)
            t.add(f"ctx.{i}.k", k.permute(1, 0, 2), True)
            t.add(f"ctx.{i}.v", v.permute(1, 0, 2), True)
        elif t is not None and t.phase == "block":
            for name, rows in (("q", q), ("k", k), ("v", v)):
                t.put(t.where("attn." + name), rows.permute(1, 0, 2), True)
        return q, k, v

    def row(self, x, q, xs=None):
        t = tap()
        if t is not None and t.phase == "block":
            t.put(t.where("attn.out" if t.side() == "attn" else "mlp.act"), x, True)
            t.gath = t.where(t.side() + ".gath")
        return o_row(self, x, q, xs)

    def layer(self, i, x, cos, sin, idx):
        t = tap()
        if t is not None:
            if i == 0:
                t.put("block.embed", x)
            t.layer, t.convs = i, 0
        out = o_layer(self, i, x, cos, sin, idx)
        if t is not None:
            t.layer = None
        return out

    dfl._mm, dfl._dconv = mm, dconv
    D._norm, D._prep, D._row, D._layer = norm, prep, row, layer


class Run:
    """Both ranks' made ``Weights``, two drafters a rank (the context a ring, and flat) and their taps."""

    def __init__(self, dfl, ws: list, drafter_dir: Path, capacity: int) -> None:
        self.ws = ws
        self.hub = base.Hub()
        for rank, w in enumerate(ws):
            w.comm = TapComm(self.hub, rank)
        self.ring = [dfl.Drafter(drafter_dir, w, capacity=capacity, ring=True) for w in ws]
        self.flat = [dfl.Drafter(drafter_dir, w, capacity=capacity, ring=False) for w in ws]
        self.taps = [Tap(rank) for rank in range(WORLD)]


def pieces(positions: int) -> list[int]:
    """Updates as rounds and prompt chunks make them: nine rows alone, then a cycle of 1 to 64."""

    out: list[int] = []
    cycle = (7, 1, 4, 64, 23, 2, 8)
    while sum(out) < positions:
        n = 1 if len(out) < 9 else cycle[len(out) % len(cycle)]
        out.append(min(n, positions - sum(out)))
    return out


def ring_holds(d, rec: dict[str, torch.Tensor], positions: int) -> list[str]:
    """The context's rows the drafter holds (the ring's slots, or the flat buffer's) against those captured."""

    keep = min(positions, d.cap)
    pos = torch.arange(positions - keep, positions, device=d.dev)
    slot = pos % d.ring if d.ring else pos
    bad = []
    for i in range(len(d.layers)):
        for name, held in ((f"ctx.{i}.k", d.kc[i]), (f"ctx.{i}.v", d.vc[i])):
            got = held[:, slot].permute(1, 0, 2).cpu()
            if bool((base.bits(got) != base.bits(rec[name][positions - keep:])).any()):
                bad.append(f"{name}.r{d.rank}")
    return bad


def play(run: Run, ds: list, taps: torch.Tensor, sizes: list[int], pending: int):
    """The taps from an empty context in updates of ``sizes`` rows, then one block pass, both ranks."""

    positions = taps.shape[0]
    ctx: list[dict[str, torch.Tensor]] = []
    bad: list[str] = []
    for rank, d in enumerate(ds):                    # a context update gathers nothing: a rank after the other
        t = run.taps[rank]
        t.begin(d, run.ws[rank].head, "ctx")
        base.LOCAL.tap = t
        d.reset()
        at = 0
        with torch.no_grad():
            for n in sizes:
                d.add_taps(taps[at:at + n])
                at += n
        base.LOCAL.tap = None
        ctx.append(t.take())
        bad += ring_holds(d, ctx[rank], positions)

    def block(rank: int):
        d, t = ds[rank], run.taps[rank]
        t.begin(d, run.ws[rank].head, "block")
        base.LOCAL.tap = t
        tokens, values, proj = d.candidates(pending, d.block - 1)
        k = d.top_k
        t.put("block.ids", d.ids)
        t.put("block.top.values", d.packed[:, :, :k].permute(1, 0, 2))
        t.put("block.top.ids", d.packed[:, :, k:].contiguous().view(torch.int32).permute(1, 0, 2))
        t.put("block.projected", d.proj)
        base.LOCAL.tap = None
        return t.take(), tokens, values, proj

    got = base.run_ranks(run.hub, block)
    if not ((got[0][1] == got[1][1]).all() and (got[0][2] == got[1][2]).all()):
        raise RuntimeError("the ranks merged other candidates")
    split = set().union(*(t.split for t in run.taps))
    rec, differ = base.merge([dict(ctx[rank], **got[rank][0]) for rank in range(WORLD)], split)
    rec["candidates.tokens"] = torch.from_numpy(got[0][1]).to(torch.int32)
    rec["candidates.values"] = torch.from_numpy(got[0][2]).to(torch.float32)
    torch.cuda.synchronize()
    return rec, differ, bad, got[0][3]


def chains(d, rec: dict[str, torch.Tensor], proj, pending: int) -> dict:
    """``Drafter.chain`` on the block's candidates: greedy and sampled, under confidences, from two anchors.

    What ``NAME.chain.json`` holds: the merged candidates, the projected rows as ``candidates`` gave them, and
    each chain's picks. The codebooks' rows are not written: a reader takes them from the drafter's directory.
    """

    from tensorfold.engine.exact_sampling import Sampling

    tokens = rec["candidates.tokens"].numpy().astype("int64")
    values = rec["candidates.values"].numpy().astype("float64")
    noises = [None] + [(seed, t) for seed in range(1, 17) for t in (0.6, 1.0)]
    out = []
    for anchor in (pending, int(tokens[0, 5])):
        for first in (d.context_end + 1, 5):
            for noise in noises:
                for confidence in (0.0, 0.2, 0.5):
                    sampling = None if noise is None else Sampling(seed=noise[0], temperature=noise[1])
                    picks = d.chain(tokens, values, proj, anchor, first, sampling, confidence)
                    out.append({"anchor": anchor, "first": first, "seed": noise[0] if noise else None,
                                "temperature": noise[1] if noise else None, "confidence": confidence,
                                "picks": [int(x) for x in picks]})
    return {"top": int(tokens.shape[1]), "rank": int(proj.shape[1]), "tokens": tokens.tolist(),
            "values": values.tolist(), "projected": proj.tolist(), "chains": out}


def export_case(run: Run, name: str, positions: int, pending: int, seed: int, out_dir: Path, provenance: dict) -> dict:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    taps = torch.randn((positions, run.ring[0].tap_in.shape[1]), generator=g, dtype=torch.float32).to(torch.bfloat16)
    dev = run.ws[0].device
    whole = [min(64, positions - at) for at in range(0, positions, 64)]
    rec, differ, bad, proj = play(run, run.ring, taps.to(dev), whole, pending)
    again, a_differ, a_bad, _ = play(run, run.flat, taps.to(dev), pieces(positions), pending)
    d = run.ring[0]
    prov = dict(provenance, case=name, positions=positions, pending=pending, seed=seed, updates=whole,
                taps="torch.randn(positions, 5 * hidden; torch.Generator().manual_seed(seed)) rounded to bf16",
                ring_rows=d.ring, context_rows=d.cap, window=d.window, block=d.block, mask=d.mask_id)
    path = out_dir / f"{name}.safetensors"
    tensors = {"taps": taps, **{n: t.contiguous() for n, t in rec.items()}}
    save_file(tensors, str(path), metadata={"provenance": json.dumps(prov)})
    picked = chains(d, rec, proj, pending)
    (out_dir / f"{name}.chain.json").write_text(json.dumps(dict(picked, provenance=prov)) + "\n")
    print(f"[drafter] {name}: {len(picked['chains'])} chains, "
          f"{len({tuple(c['picks']) for c in picked['chains']})} of them distinct", flush=True)
    case = {"provenance": prov, "ranks_differ": differ, "ring_holds": {"differing": bad},
            "flat_in_pieces": dict(base.versus(rec, again, 0), updates=pieces(positions), ranks_differ=a_differ,
                                   holds=a_bad, ring_rows=run.flat[0].ring),
            "nonfinite": base.nonfinite(rec),
            "candidates": {"tokens": rec["candidates.tokens"].tolist(), "values": rec["candidates.values"].tolist()},
            "shapes": base.shapes(tensors),
            "file": {"path": str(path), "bytes": path.stat().st_size, "tensors": len(tensors)}}
    v = case["flat_in_pieces"]
    print(f"[drafter] {name}: {positions} positions (ring of {d.ring}), pending {pending}; between the ranks "
          f"{len(differ)} replicated points differ; the ring's slots differ from the rows captured at {len(bad)}; "
          f"flat and in pieces {v['differing']} of {v['points']} points differ"
          + (f", first {json.dumps(v['first'])}" if v["first"] else "")
          + f"; non-finite {case['nonfinite']}; first candidates {case['candidates']['tokens'][0][:4]}; "
          f"file {json.dumps(case['file'])}", flush=True)
    return case


def parse_case(text: str) -> tuple[str, int, int, int]:
    name, _, rest = text.partition("=")
    got = [int(x) for x in rest.split(",")] if rest else []
    if not name or len(got) != 3 or got[0] < 1:
        raise argparse.ArgumentTypeError(f"{text!r}: expected NAME=POSITIONS,PENDING,SEED")
    return name, got[0], got[1], got[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path, help="GLM's pack: its embed_tokens and lm_head are read, no layer")
    ap.add_argument("--drafter", type=Path, required=True, help="the DFlash2 drafter's directory")
    ap.add_argument("--case", type=parse_case, action="append", required=True, metavar="NAME=POSITIONS,PENDING,SEED")
    ap.add_argument("--out-dir", type=Path, required=True, help="where NAME.safetensors and NAME.json go")
    ap.add_argument("--capacity", type=int, default=2560, help="the context's positions (Drafter's default)")
    ap.add_argument("--tensorfold-commit", default=None, metavar="HASH",
                    help="the commit of the TensorFold archive the package is imported from: recorded as stated")
    args = ap.parse_args()

    import tensorfold
    import triton
    from tensorfold.cuda import build
    from tensorfold.families.glm5_next.cuda import dflash2 as dfl
    from tensorfold.families.glm5_next.cuda import qmm, split, weights

    if tensorfold.__version__ != base.PINNED:
        print(f"[drafter] TensorFold {tensorfold.__version__}, written against {base.PINNED}: the hooks may not fit",
              file=sys.stderr)
    build.refuse_old_gpu()
    for _, positions, _, _ in args.case:
        if positions > args.capacity:
            raise SystemExit(f"a case of {positions} positions: past --capacity {args.capacity}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    dev = torch.device("cuda")
    cfg = weights.Config.read(args.model_dir)
    half = cfg.vocab // WORLD
    rd = split.RankReader(args.model_dir, 0)
    try:
        embed = rd.get(weights.PREFIX + "embed_tokens.weight").to(torch.bfloat16).contiguous().to(dev)
        lm_head = rd.get("lm_head.weight")
        if tuple(embed.shape) != (cfg.vocab, cfg.hidden) or tuple(lm_head.shape) != (cfg.vocab, cfg.hidden):
            raise SystemExit(f"embed_tokens {tuple(embed.shape)}, lm_head {tuple(lm_head.shape)}: not whole")
        heads = [qmm.make_b16(lm_head[rank * half:(rank + 1) * half].to(dev)) for rank in range(WORLD)]
        del lm_head
    finally:
        rd.close()
    ws = [weights.Weights(cfg, embed, [], None, heads[rank], None, rank, WORLD, dev) for rank in range(WORLD)]
    dfl._quantize4 = qmm.make_b16                # BF16 faces: every linear of the drafter as stored
    run = Run(dfl, ws, args.drafter, args.capacity)
    install(dfl)
    torch.cuda.synchronize()
    faces = sorted({type(q).__name__ for d in run.ring + run.flat for L in d.layers
                    for q in (L.a_kp, L.m_kp, L.qkv, L.kv, L.o, L.gu, L.down)} | {type(run.ring[0].fc).__name__})
    provenance = {
        "tensorfold_version": tensorfold.__version__,
        "tensorfold_commit": args.tensorfold_commit or "unknown",
        "tensorfold_commit_source": f"stated with --tensorfold-commit: {Path(tensorfold.__file__).resolve()} is an "
                                    "archive without .git",
        "drafter": str(args.drafter), "drafter_revision": base.revision(args.drafter),
        "checkpoint": str(args.model_dir), "checkpoint_revision": base.revision(args.model_dir),
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "faces": f"{faces}: dflash2._quantize4 replaced by qmm.make_b16, the head lm_head's rows as stored "
                 "(Weights.draft_head None); TensorFold's own drafts read 4-bit copies",
        "main": "no main layer: weights.Weights made by hand with embed_tokens and each rank's half of lm_head",
        "ranks": "both ranks in this process, a thread each, w.comm a barrier (ThreadComm) in NCCL's place",
        "path": "Drafter.add_taps and Drafter.candidates, eager (no graphs), the context a ring",
    }
    report = {"load": {"seconds": round(time.time() - t0, 1), "faces": faces,
                       "drafter_gib": [round(d.nbytes() / 2**30, 2) for d in run.ring], "memory": base.memory()}}
    print(f"[drafter] loaded: {json.dumps(report['load'])}", flush=True)
    for name, positions, pending, seed in args.case:
        t1 = time.time()
        case = export_case(run, name, positions, pending, seed, args.out_dir, provenance)
        out = dict(report, **case, seconds=round(time.time() - t1, 1), gathers=run.hub.gathers, memory=base.memory())
        (args.out_dir / f"{name}.json").write_text(json.dumps(out, indent=1))
    print(f"[drafter] done: {run.hub.gathers} exchanges; memory {json.dumps(base.memory())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
