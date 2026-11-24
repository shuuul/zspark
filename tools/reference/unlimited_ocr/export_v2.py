#!/usr/bin/env python3
"""Unlimited-OCR-2 reference fixtures from the checkpoint's own HF code, Triton off (spec 147 WS-01).

Run in the reference environment (`source tools/reference/hf_env.sh`):

    python tools/reference/unlimited_ocr/export_v2.py dry                 # CPU, no weights: imports, builds inputs
    flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock \
        python tools/reference/unlimited_ocr/export_v2.py fixtures [--out DIR] [--pages DIR] [--max-new 8192] [--only CASE ...]
    flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock \
        python tools/reference/unlimited_ocr/export_v2.py cer [--out DIR] [--cer DIR] [--max-new 8000] [--only NAME ...]
        # a CER set (DIR/manifest.json's pages, as spec 137's `cer/` laid them out: NAME.png beside a
        # {"pages": [{"name", "image"}]} list) one page a request, as SGLang's README sends it (gundam,
        # 35 in 128, max_page_tokens 20000, 8000 tokens a page): OUT/cer/NAME.txt, NAME.json (tokens,
        # eos, seconds), the page's image and manifest.json, for `cer.py`; no hooks, so
        # `generate_seconds` is HF's speed
    flock -s ~/.cache/zspark/machine.lock python tools/reference/unlimited_ocr/export_v2.py fixtures --smoke
        # CPU only (CUDA_VISIBLE_DEVICES="" and `Tensor.cuda` shimmed to identity, for the hard-coded `.cuda()` in
        # UnlimitedOCRModel.forward): a made page as both cases, 3 new tokens, into scratch-147. It proves
        # the hooks and the bitwise checks, not the bits: the fixtures are the GPU run's. `--cpu` runs the real
        # cases the same way into scratch-147/export-v2-cpu (a preview of the reply; CPU kernels' bits).

`fixtures` writes OUT/CASE.safetensors (tensors; provenance in the metadata), OUT/CASE.json (prompt, ids, the
reply, finish reason, the router's parameters as run, near-tie counts, timings), OUT/pages/ (the input pages)
and OUT/manifest.json (every tensor's name, shape, dtype and bytes). The cases, from the page images of
`--pages DIR` (OUT/pages unset): `one_page` (paper_p01, `infer`'s single-page policy: gundam, tiles) and
`two_pages` (paper_p01 + paper_p03, the multi-page policy: base, one 1024 view a page). The model runs
through its own `infer(..., eval_mode=True, return_dict=True)`: `_uocr_build_inputs` (`<PAGE><image>` a
page + "Multi page parsing."), `_uocr_generate` (prefill, then one token a call on a DynamicCache),
PageCountGuard (max_page_tokens 20000) and the n-gram rule at the README's evaluation setting (35; window
128 for one page, 1024 for several; `infer`'s default is off).

Triton off, and shown off: every layer's `dash_router.use_triton` is set False after the load, so
`_infer_prefill_attend` takes `DashAttention._forward_dense` (SDPA with the router's additive mask) and a step
`_decode_attend` (fp32); `DashAttention._forward_triton`, the base attention forward that `_patched_forward`
falls back to (`modeling_deepseekv2._orig_forward`) and `triton.runtime.jit.JITFunction.run` (any Triton launch)
are replaced by functions that raise, and INDEXER_SCORE_TRITON=0. The router's parameters are asserted against
config.json after `configure_dash_inference` (stage0 head_cls, K 12, top-M 16, scale 128/sqrt(128), floor 1e-4,
sigma 1): a lost checkpoint snapshot would route silently with stage0 "indexer".

Hooks, from outside (no checkpoint file is edited): the module globals `_infer_prefill_attend`,
`_dash_decode_forward` (the layer), `_router_qk`, `_block_scores_infer`, `_infer_route_select`, `_decode_attend`,
`_dash_page_ids` of modeling_deepseekv2, `_uocr_build_inputs` and `_uocr_dash_decode_allow` of
modeling_unlimitedocr are wrapped; forward hooks on the inner model (calls), each layer, its self_attn, the
two page2d rotaries (main and router: their (tok, page) arguments) and lm_head. k_bar, the top-16 and the
entmax weights are not returned by HF's functions: they are recomputed here with HF's own ops on the same
device and asserted bitwise against what HF returned (block scores, ids, bias, valid), every call.

Tensor names (L = 12 layers, S prompt rows, NB = ceil(S/32) blocks, V vocab; bf16 unless said):
  input_ids [S] i32, seq_mask [S] u8, images_spatial_crop [P,2] i32 (per page: width, height tiles),
  views.global_u8 [P,1024,1024,3] u8, views.global [P,3,1024,1024], views.crops_u8 [T,640,640,3] u8,
  views.crops [T,3,640,640] (absent without tiles; normalize(u8) == what `infer` fed, asserted),
  prefill.tok / prefill.page [S] i32: the page2d coordinates of each row (page raw, the rotary folds it mod 2;
    main and router coordinates asserted equal in every layer), prefill.image_mask [S] u8,
    prefill.image_page_ids / prefill.query_page_ids [S] i32 (`_build_page_aware_ids`, -1 none),
    prefill.cand [S,NB] u8 (the page-aware candidate domain), prefill.cand_eff [S,NB] u8 (remote & domain, with
    the per-row fallback to every remote block: what entmax sees; both the same in every layer, asserted),
  prefill.embeds [S,1280], prefill.attn_in [L,S,1280] (self_attn's input, after input_layernorm),
  prefill.idx_q / prefill.idx_k [L,S,128] f32 (the router's q and k after Gemma norm and page2d rope),
  prefill.k_bar [L,NB,128] f32, prefill.scores [L,S,NB] f32 (idx_q . k_bar / sqrt(128), clamped; -1e9 not
    remote), prefill.x17 [L,S,17] f32 (the entmax input `scores * 128/sqrt(128)` on candidates, else -1e4,
    sorted descending, -inf past NB), prefill.top16_ids [L,S,16] i32 / .top16_x f32 / .entmax f32 (weights),
  prefill.block_ids [L,S,12] i32, prefill.bias [L,S,12] f32, prefill.valid [L,S,12] u8: the 12 slots (an
    invalid slot's id is tie-break noise among -1e4 entries: compare ids as a set over valid; fewer than 12
    blocks pad with id -1, bias 0, invalid),
  prefill.ncand [L,S] i32, prefill.gap12 / prefill.gap16 [L,S] f32: x17[11]-x17[12] and x17[15]-x17[16] in
    entmax-input units (+inf when fewer than 12 / 16 candidates: no tie can change the set),
  prefill.q [L,S,10,128] (rotated), prefill.attend [L,S,10,128] (the attend's output before o_proj),
  prefill.attn_out [L,S,1280] (after o_proj), prefill.layers [L,S,1280] (each layer's output),
  prefill.logits_last [V] f32,
  state.bmin / state.bmax [NB] i32, state.pp / state.mcum [1] i32 (`_uocr_dash_decode_allow`'s per-block image
    pages after the prompt), state.pf_cnt [n_pg] i32 (prompt rows a page bucket, seeds the decode coordinates),
  step.* for every decode call (row i is call i+1, which feeds tokens[i] at position S+i; C rows):
    step.token / step.tok / step.page / step.nblk [C] i32, step.allow [C,NBmax] u8 (the decode candidate blocks,
    padded 0), step.idx_q [C,L,128] f32, step.block_ids / .bias / .valid [C,L,12], step.top16_ids / .top16_x /
    .entmax [C,L,16], step.x17 [C,L,17], step.ncand / step.gap12 / step.gap16 [C,L],
  detail.* for the detailed calls detail.calls [D] i32 (the first --steps, --page-n from each fed <PAGE>,
    --late-n from --late0): detail.scores [D,L,NBmax] f32 (NaN padded), detail.k_bar [D,L,NBmax,128] f32,
    detail.attn_in [D,L,1280], detail.q [D,L,10,128], detail.attend [D,L,10,128] f32 (`_decode_attend`, before
    the bf16 cast), detail.attn_out / detail.layers [D,L,1280], detail.logits [D,V] f32 (raw, NaN for a call
    whose pick never came: the cap),
  pick.argmax [K] i32, pick.gap [K] f32 (top-1 minus top-2 raw logit, before the n-gram rule and the guard),
    tokens [K'] i32 (the reply, EOS dropped as `infer` returns it; pick k chose tokens[k], or EOS at the end),
  cache.k / cache.v [L,R,10,128] (the KV cache at the end, rotated keys, R rows), cache.idx_k [L,R,128] f32.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if "--smoke" in sys.argv or "--cpu" in sys.argv or "dry" in sys.argv[1:2]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # nothing of these touches the GPU
os.environ["INDEXER_SCORE_TRITON"] = "0"
os.environ.pop("DASH_STAGE0", None)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL = os.path.expanduser("~/models/Unlimited-OCR-2")
OUT = os.path.expanduser("~/.cache/zspark/reference-unlimited-ocr-2")
SMOKE_OUT = os.path.expanduser("~/.cache/zspark/scratch-147/export-v2-smoke")
CPU_OUT = os.path.expanduser("~/.cache/zspark/scratch-147/export-v2-cpu")
CASES = [("one_page", ["paper_p01.png"], 128), ("two_pages", ["paper_p01.png", "paper_p03.png"], 1024)]
NGRAM = 35
PAGE_ID, EOS_ID = 128827, 1


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize(u8):
    """ToTensor + Normalize(0.5, 0.5) in fp32, then bf16, as `BasicImageTransform` + infer."""
    x = u8.permute(0, 3, 1, 2).float() / 255.0
    return ((x - 0.5) / 0.5).to(torch.bfloat16)


def git_commit():
    import subprocess

    root = Path(__file__).resolve().parents[3]
    try:
        c = subprocess.check_output(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
            c += " dirty"
        return c
    except Exception:
        return "unknown"


def shards(model_dir):
    return sorted(Path(model_dir).glob("model-*.safetensors"))


def checkpoint_digests(model_dir, out):
    """sha256 a shard, cached in OUT/.checkpoint-sha256 (JSON) while the shards are older than it."""
    cache = Path(out) / ".checkpoint-sha256"
    files = shards(model_dir)
    if cache.exists() and all(cache.stat().st_mtime > f.stat().st_mtime for f in files):
        d = json.loads(cache.read_text())
        if sorted(d) == [f.name for f in files]:
            return d
    d = {f.name: sha256_file(f) for f in files}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d, indent=1))
    return d


def provenance(args, extra):
    import transformers

    meta = {
        "source": "checkpoint HF code (trust_remote_code), infer(eval_mode=True), Triton off (_forward_dense, _decode_attend)",
        "checkpoint": str(args.model),
        "checkpoint_sha256": json.dumps(checkpoint_digests(args.model, args.out)),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "zspark": git_commit(),
        "exporter": "tools/reference/unlimited_ocr/export_v2.py",
    }
    meta.update({k: str(v) for k, v in extra.items()})
    return meta


# ---------------------------------------------------------------------------------------------------------------
# loading, the remote modules, Triton off


def remote_modules(model_obj_or_cls):
    um = sys.modules[model_obj_or_cls.__module__ if isinstance(model_obj_or_cls, type) else type(model_obj_or_cls).__module__]
    pkg = um.__name__.rsplit(".", 1)[0]
    return um, sys.modules[pkg + ".modeling_deepseekv2"], sys.modules[pkg + ".dash_attention"]


def trip(name):
    def f(*a, **k):
        raise RuntimeError(f"export_v2: {name} reached; only the torch reference path may run")

    return f


def triton_off(model, dsv2, da):
    """Route every attend to the torch reference and make any Triton path raise."""
    for layer in model.model.layers:
        r = layer.self_attn.dash_router
        assert r is not None, f"layer {layer.self_attn.layer_idx} has no dash_router"
        r.use_triton = False
    da.DashAttention._forward_triton = trip("DashAttention._forward_triton")
    dsv2._orig_forward = trip("the base attention forward (non-DASH path)")
    try:
        import triton.runtime.jit as tj

        tj.JITFunction.run = trip("a Triton kernel launch")
    except ImportError:
        pass


def expected_params(model_dir, layer):
    sp = json.loads((Path(model_dir) / "config.json").read_text())["sparse_attention_config"]
    return {"idx_scale": float(sp["dash_scale_per_layer"][layer]) / math.sqrt(int(sp["sparse_index_dim"])),
            "floor": float(sp["dash_floor_per_layer"][layer]), "sigma": float(sp["dash_sigma"]),
            "K": int(sp["dash_topk_blocks"]), "topm": int(sp["sparse_topk_blocks"]), "stage0": sp["dash_stage0"]}


def load(args):
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cpu":
        torch.Tensor.cuda = lambda self, *a, **k: self  # UnlimitedOCRModel.forward's `.cuda()`, CPU run only
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16)
    model = model.eval().to(args.device)
    return tok, model


# ---------------------------------------------------------------------------------------------------------------
# HF's router math, recomputed with HF's ops (k_bar, top-16 and entmax are internal to its functions)


def k_bar_of(router, idx_k, nblk, act_clamp):
    B, S, d = idx_k.shape
    bs = router.block_size
    inv_sqrt_d = 1.0 / math.sqrt(d)
    Sp = nblk * bs
    if Sp > S:
        kk = F.pad(idx_k, (0, 0, 0, Sp - S))
        real = F.pad(idx_k.new_ones(B, S, dtype=torch.bool), (0, Sp - S), value=False)
    else:
        kk = idx_k[:, :Sp]
        real = idx_k.new_ones(B, Sp, dtype=torch.bool)
    kk = kk.view(B, nblk, bs, d)
    vmask = real.view(B, nblk, bs)
    hc = router.head_cls.to(kk.dtype).view(1, 1, 1, d)
    aw = ((kk * hc).sum(-1) * inv_sqrt_d).clamp(-act_clamp, act_clamp)
    neg = torch.finfo(aw.dtype).min
    row_max = torch.where(vmask, aw, aw.new_full((), neg)).amax(dim=-1, keepdim=True)
    row_max = torch.where(vmask.any(dim=-1, keepdim=True), row_max, torch.zeros_like(row_max))
    e = torch.exp(aw - row_max) * vmask.to(aw.dtype)
    w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    k_bar = torch.nan_to_num((w.unsqueeze(-1) * kk).sum(dim=2), nan=0.0, posinf=0.0, neginf=0.0)
    return k_bar, vmask.any(dim=-1)


def scores_of(router, idx_q, k_bar, chunk_valid, q_pos, nblk, act_clamp, score_neg):
    d = k_bar.shape[-1]
    inv_sqrt_d = 1.0 / math.sqrt(d)
    blk_ids = torch.arange(nblk, device=k_bar.device)
    out = []
    for c0 in range(0, idx_q.shape[1], router.q_chunk):
        c1 = min(c0 + router.q_chunk, idx_q.shape[1])
        z = torch.matmul(idx_q[:, c0:c1], k_bar.transpose(-1, -2)) * inv_sqrt_d
        z = z.clamp(-act_clamp, act_clamp)
        qblk = q_pos[:, c0:c1] // router.block_size
        keep = (blk_ids[None, None, :] <= (qblk[:, :, None] - router.local_blocks)) & chunk_valid[:, None, :]
        out.append(z.masked_fill(~keep, score_neg))
    return torch.cat(out, dim=1)


def route_of(sc, block_cand, p, dsv2):
    """`_infer_route_select`'s ops, keeping what it drops: cand, x, the top-M window and its entmax weights."""
    neg = dsv2._ENTMAX_NEG
    remote = sc > (dsv2._SCORE_NEG * 0.5)
    cand = remote if block_cand is None else (remote & block_cand)
    cand = torch.where(~cand.any(dim=-1, keepdim=True), remote, cand)
    x = torch.where(cand, sc * p["idx_scale"], sc.new_full((), neg))
    x = torch.nan_to_num(x, nan=neg, posinf=neg, neginf=neg)
    nblk = x.shape[-1]
    M = max(1, min(int(p["topm"]), nblk)) if int(p["topm"]) > 0 else nblk
    xv, xi = x.topk(M, dim=-1)
    prior = dsv2._entmax15(xv, dim=-1)
    is_cand = cand.gather(-1, xi)
    K = max(1, min(int(p["K"]), M))
    topi, pk, valid = xi[..., :K], prior[..., :K], is_cand[..., :K]
    logp = torch.log(pk.clamp_min(p["floor"]))
    cnt = valid.sum(dim=-1, keepdim=True).clamp(min=1)
    mu = torch.where(valid, logp, torch.zeros_like(logp)).sum(dim=-1, keepdim=True) / cnt
    bias = torch.where(valid, (logp - mu) * (1.0 / p["sigma"]), torch.zeros_like(logp)).float()
    # margins: the sorted entmax input, 17 wide (-inf past nblk)
    xs = x.sort(dim=-1, descending=True).values
    if nblk < 17:
        xs = F.pad(xs, (0, 17 - nblk), value=-math.inf)
    x17 = xs[..., :17]
    ncand = cand.sum(-1)
    inf = torch.full_like(x17[..., 0], math.inf)
    gap12 = torch.where(ncand >= 12, x17[..., 11] - x17[..., 12], inf)
    gap16 = torch.where(ncand >= 16, x17[..., 15] - x17[..., 16], inf)
    pad = lambda t, w, v: F.pad(t, (0, w - t.shape[-1]), value=v) if t.shape[-1] < w else t
    k = int(p["K"])  # fewer blocks than K: the slots past them are padded (id -1, bias 0, invalid)
    return {"cand": cand, "topi": topi, "bias": bias, "valid": valid, "block_ids": pad(topi, k, -1),
            "bias12": pad(bias, k, 0.0), "valid12": pad(valid, k, False), "top16_ids": pad(xi, 16, -1),
            "top16_x": pad(xv, 16, -math.inf), "entmax": pad(prior, 16, 0.0), "x17": x17, "ncand": ncand,
            "gap12": gap12, "gap16": gap16}


# ---------------------------------------------------------------------------------------------------------------
# the recorder


class Recorder:
    def __init__(self, model, um, dsv2, da, args):
        self.m, self.um, self.dsv2, self.da, self.args = model, um, dsv2, da, args
        self.inner = model.model
        self.nl = len(self.inner.layers)
        self.call, self.layer, self.detail_now, self.page_until = -1, None, False, -1
        self.cur = defaultdict(dict)  # name -> {layer: tensor} for this call
        self.pre, self.steps, self.details, self.picks = {}, [], [], []
        self.state, self.cache, self.inputs, self.params = {}, None, None, {}
        self.counts = defaultdict(int)
        self.handles, self.saved = [], []
        self.install()

    # -- plumbing
    def wrap(self, mod, name, make):
        orig = getattr(mod, name)
        self.saved.append((mod, name, orig))
        setattr(mod, name, make(orig))

    def hook(self, mod, fn, pre=False, kw=False):
        self.handles.append(mod.register_forward_pre_hook(fn, with_kwargs=kw) if pre else mod.register_forward_hook(fn))

    def remove(self):
        for h in self.handles:
            h.remove()
        for mod, name, orig in reversed(self.saved):
            setattr(mod, name, orig)

    def put(self, name, t, layer=None):
        self.cur[name][self.layer if layer is None else layer] = t.detach().clone().cpu() if torch.is_tensor(t) else t

    @property
    def prefill(self):
        return self.call == 0

    def keep(self):
        return self.prefill or self.detail_now

    def install(self):
        R, dsv2, um = self, self.dsv2, self.um
        act, sneg = self.da._ACT_CLAMP, dsv2._SCORE_NEG

        def w_build(orig):
            def f(*a, **k):
                out = orig(*a, **k)
                R.inputs = out
                return out
            return f

        def w_prefill_attend(orig):
            def f(self_attn, router, hidden_states, q, k, v, idx_pe, pos_ids, block_cand, attn_mask, past_kv):
                R.layer = self_attn.layer_idx
                R.check_params(router)
                R.counts["prefill_attend"] += 1
                if attn_mask is not None:
                    R.counts["prefill_attn_mask"] += 1
                n0 = R.counts["forward_dense"]
                out = orig(self_attn, router, hidden_states, q, k, v, idx_pe, pos_ids, block_cand, attn_mask, past_kv)
                assert R.counts["forward_dense"] == n0 + 1, "the prefill attend did not take _forward_dense"
                R.put("q", q[0].transpose(0, 1))
                R.put("attend", out[0].transpose(0, 1))
                return out
            return f

        def w_dense(orig):
            def f(*a, **k):
                R.counts["forward_dense"] += 1
                return orig(*a, **k)
            return f

        def w_decode_forward(orig):
            def f(self_attn, *a, **k):
                R.layer = self_attn.layer_idx
                R.check_params(self_attn.dash_router)
                R.counts["decode_forward"] += 1
                return orig(self_attn, *a, **k)
            return f

        def w_router_qk(orig):
            def f(router, hidden_states, pe):
                iq, ik = orig(router, hidden_states, pe)
                R.put("idx_q", iq[0])
                if R.prefill:
                    R.put("idx_k", ik[0])
                return iq, ik
            return f

        def w_scores(orig):
            def f(router, idx_q, idx_k, q_pos, nblk):
                sc = orig(router, idx_q, idx_k, q_pos, nblk)
                kb, cv = k_bar_of(router, idx_k, nblk, act)
                mine = scores_of(router, idx_q, kb, cv, q_pos, nblk, act, sneg)
                assert torch.equal(mine, sc), f"layer {R.layer} call {R.call}: recomputed block scores differ from HF's"
                if R.keep():
                    R.put("scores", sc[0])
                    R.put("k_bar", kb[0])
                return sc
            return f

        def w_route(orig):
            def f(block_scores, block_cand, p):
                topi, bias, valid = orig(block_scores, block_cand, p)
                r = route_of(block_scores, block_cand, p, dsv2)
                for a, b, n in ((r["topi"], topi, "ids"), (r["bias"], bias, "bias"), (r["valid"], valid, "valid")):
                    assert torch.equal(a, b), f"layer {R.layer} call {R.call}: recomputed {n} differ from HF's"
                for n, m in (("topi", "block_ids"), ("bias", "bias12"), ("valid", "valid12")):
                    R.put(n, r[m][0])
                for n in ("top16_ids", "top16_x", "entmax", "x17", "ncand", "gap12", "gap16"):
                    R.put(n, r[n][0])
                if R.prefill:
                    R.put("cand", block_cand[0] if block_cand is not None else torch.ones_like(r["cand"][0]))
                    R.put("cand_eff", r["cand"][0])
                return topi, bias, valid
            return f

        def w_decode_attend(orig):
            def f(q, *a, **k):
                out = orig(q, *a, **k)
                R.counts["decode_attend"] += 1
                if R.keep():
                    R.put("q", q[0, :, 0])
                    R.put("attend", out[0, :, 0])
                return out
            return f

        def w_page_ids(orig):
            def f(self_attn, idx, src_ids, key_length):
                itm, ipi, qpi = orig(self_attn, idx, src_ids, key_length)
                if self_attn.layer_idx == 0:
                    R.put("image_mask", itm[0], 0)
                    R.put("image_page_ids", ipi[0], 0)
                    R.put("query_page_ids", qpi[0], 0)
                return itm, ipi, qpi
            return f

        def w_allow(orig):
            def f(indexer, input_ids, cache):
                out = orig(indexer, input_ids, cache)
                if out is not None:
                    R.put("allow", out[0], 0)
                return out
            return f

        self.wrap(um, "_uocr_build_inputs", w_build)
        self.wrap(um, "_uocr_dash_decode_allow", w_allow)
        self.wrap(dsv2, "_infer_prefill_attend", w_prefill_attend)
        self.wrap(dsv2, "_dash_decode_forward", w_decode_forward)
        self.wrap(dsv2, "_router_qk", w_router_qk)
        self.wrap(dsv2, "_block_scores_infer", w_scores)
        self.wrap(dsv2, "_infer_route_select", w_route)
        self.wrap(dsv2, "_decode_attend", w_decode_attend)
        self.wrap(dsv2, "_dash_page_ids", w_page_ids)
        self.wrap(self.da.DashAttention, "_forward_dense", w_dense)

        self.hook(self.inner, self.on_call, pre=True, kw=True)
        self.hook(self.inner, self.on_call_end)
        self.hook(self.m.lm_head, self.on_logits)
        for i, layer in enumerate(self.inner.layers):
            at = layer.self_attn
            assert at.rotary_emb_2d_main is not None and at.indexer.rotary_emb_2d is not None, f"layer {i}: no page2d rope"
            self.hook(at.rotary_emb_2d_main, lambda m, a, i=i: self.on_coords("main", i, a), pre=True)
            self.hook(at.indexer.rotary_emb_2d, lambda m, a, i=i: self.on_coords("router", i, a), pre=True)
            self.hook(at, lambda m, a, k, i=i: self.on_attn_in(i, a, k), pre=True, kw=True)
            self.hook(at, lambda m, a, o, i=i: self.put("attn_out", o[0][0], i) if self.keep() else None)
            self.hook(layer, lambda m, a, o, i=i: self.put("layers", (o[0] if isinstance(o, tuple) else o)[0], i) if self.keep() else None)
            if i == 0:
                self.hook(layer, lambda m, a, k: self.put("embeds", (k["hidden_states"] if "hidden_states" in k else a[0])[0], 0) if self.prefill else None, pre=True, kw=True)

    def check_params(self, router):
        li = self.layer
        if li in self.params:
            return
        exp = expected_params(self.args.model, li)
        p = dict(router._infer_params)
        assert router.stage0 == exp["stage0"], f"layer {li}: stage0 {router.stage0!r}, the checkpoint says {exp['stage0']!r}"
        for k in ("idx_scale", "floor", "sigma", "K", "topm"):
            assert abs(float(p[k]) - float(exp[k])) <= 1e-12 * max(1.0, abs(float(exp[k]))), f"layer {li}: {k} {p[k]} != {exp[k]}"
        assert self.dsv2._base._PAGE_SCOPE_PREFILL, "page-scoped prefill is off"
        self.params[li] = {**{k: (float(v) if isinstance(v, float) else v) for k, v in p.items()}, "stage0": router.stage0}

    # -- hooks
    def on_call(self, mod, args, kwargs):
        self.call += 1
        self.cur = defaultdict(dict)
        ids = kwargs.get("input_ids")
        if self.call == 0:
            self.cache = kwargs.get("past_key_values")
            self.detail_now = False
            return
        a = self.args
        fed = int(ids[0, -1])
        if fed == PAGE_ID:
            self.page_until = self.call + a.page_n - 1
        self.detail_now = self.call <= a.steps or self.call <= self.page_until or a.late0 <= self.call < a.late0 + a.late_n
        self.cur["token"][0] = fed

    def on_coords(self, kind, i, a):
        tok, page = a[1][0].detach().cpu().to(torch.int32), a[2][0].detach().cpu().to(torch.int32)
        self.cur[f"coords.{kind}"][i] = (tok, page)

    def on_attn_in(self, i, a, k):
        if self.keep():
            self.put("attn_in", (k["hidden_states"] if "hidden_states" in k else a[0])[0], i)

    def on_logits(self, mod, a, o):
        lg = o[0, -1].detach().float()
        top2 = lg.topk(2)
        self.picks.append((int(top2.indices[0]), float(top2.values[0] - top2.values[1])))
        k = len(self.picks) - 1
        if k == 0:
            self.pre["prefill.logits_last"] = lg.cpu()
        elif self.details and self.details[-1]["call"] == k:
            self.details[-1]["logits"] = lg.cpu()

    def layers_of(self, name):
        d = self.cur[name]
        assert sorted(d) == list(range(self.nl)), f"call {self.call}: {name} seen for layers {sorted(d)}"
        return [d[i] for i in range(self.nl)]

    def coords(self):
        main = self.layers_of("coords.main")
        rout = self.layers_of("coords.router")
        t0, p0 = main[0]
        for t, p in main + rout:
            assert torch.equal(t, t0) and torch.equal(p, p0), f"call {self.call}: page2d coordinates differ across layers"
        return t0, p0

    def on_call_end(self, mod, a, o):
        st = torch.stack
        tok, page = self.coords()
        route = {n: st(self.layers_of(n)) for n in ("topi", "bias", "valid", "top16_ids", "top16_x", "entmax", "x17", "ncand", "gap12", "gap16")}
        if self.call == 0:
            P = self.pre
            P["prefill.tok"], P["prefill.page"] = tok, page
            for n in ("image_mask", "image_page_ids", "query_page_ids"):
                P[f"prefill.{n}"] = self.cur[n][0]
            for n in ("cand", "cand_eff"):
                ls = self.layers_of(n)
                assert all(torch.equal(x, ls[0]) for x in ls), f"prefill {n} differs across layers"
                P[f"prefill.{n}"] = ls[0]
            P["prefill.embeds"] = self.cur["embeds"][0]
            for n in ("attn_in", "idx_q", "idx_k", "k_bar", "scores", "q", "attend", "attn_out", "layers"):
                P[f"prefill.{n}"] = st(self.layers_of(n))
            P["prefill.block_ids"] = route.pop("topi")
            for n, x in route.items():
                P[f"prefill.{n}"] = x
            c = self.cache
            pg = getattr(c, "_uocr_pg", None)
            assert pg is not None, "the prompt left no decode allow state (_uocr_pg)"
            self.state = {f"state.{k}": (v[0] if v.dim() > 1 else v.reshape(-1)).cpu() for k, v in pg.items()}
            rs = getattr(c, "_page_rope_state", None)
            assert rs is not None and 0 in rs, "the prompt left no page-rope state"
            self.state["state.pf_cnt"] = rs[0]["pf_cnt"][0].cpu()
            return
        s = {"call": self.call, "token": self.cur["token"][0], "tok": int(tok[0]), "page": int(page[0])}
        s["allow"] = self.cur["allow"].get(0)
        s["idx_q"] = st([x[0] for x in self.layers_of("idx_q")])
        for n, x in route.items():
            s[n] = x[:, 0]
        s["nblk"] = int(s["allow"].numel()) if s["allow"] is not None else -1
        self.steps.append(s)
        if self.detail_now:
            d = {"call": self.call}
            for n in ("scores", "attn_in", "attn_out", "layers"):  # one row: [1, ...]
                d[n] = st([x[0] for x in self.layers_of(n)])
            for n in ("k_bar", "q", "attend"):
                d[n] = st(self.layers_of(n))
            self.details.append(d)

    # -- the end of a run
    def final_cache(self):
        c = self.cache
        ks, vs = [], []
        for i in range(self.nl):
            k = c.key_cache[i] if hasattr(c, "key_cache") else c.layers[i].keys
            v = c.value_cache[i] if hasattr(c, "value_cache") else c.layers[i].values
            ks.append(k[0].transpose(0, 1).cpu())
            vs.append(v[0].transpose(0, 1).cpu())
        ik = [c._dash_idx_k[i][0].cpu() for i in range(self.nl)]
        return torch.stack(ks), torch.stack(vs), torch.stack(ik)


def pad_stack(xs, width, value):
    return torch.stack([F.pad(x, (0, width - x.shape[-1]), value=value) if x.shape[-1] < width else x for x in xs])


def collect(rec):
    t = dict(rec.pre)
    t.update(rec.state)
    if rec.steps:
        S = rec.steps
        nbmax = max(s["nblk"] for s in S)
        for n in ("token", "tok", "page", "nblk"):
            t[f"step.{n}"] = torch.tensor([s[n] for s in S], dtype=torch.int32)
        t["step.allow"] = pad_stack([s["allow"].to(torch.uint8) for s in S], nbmax, 0)
        t["step.idx_q"] = torch.stack([s["idx_q"] for s in S])
        t["step.block_ids"] = torch.stack([s["topi"] for s in S])
        for n in ("bias", "valid", "top16_ids", "top16_x", "entmax", "x17", "ncand", "gap12", "gap16"):
            t[f"step.{n}"] = torch.stack([s[n] for s in S])
    if rec.details:
        D = rec.details
        nbmax = max(d["scores"].shape[-1] for d in D)
        t["detail.calls"] = torch.tensor([d["call"] for d in D], dtype=torch.int32)
        t["detail.scores"] = pad_stack([d["scores"] for d in D], nbmax, math.nan)
        t["detail.k_bar"] = torch.stack([F.pad(d["k_bar"], (0, 0, 0, nbmax - d["k_bar"].shape[-2]), value=math.nan) for d in D])
        for n in ("attn_in", "q", "attend", "attn_out", "layers"):
            t[f"detail.{n}"] = torch.stack([d[n] for d in D])
        nan = torch.full((rec.m.lm_head.out_features,), math.nan)
        t["detail.logits"] = torch.stack([d.get("logits", nan) for d in D])  # NaN: a call past the last pick
    t["pick.argmax"] = torch.tensor([p[0] for p in rec.picks], dtype=torch.int32)
    t["pick.gap"] = torch.tensor([p[1] for p in rec.picks], dtype=torch.float32)
    t["cache.k"], t["cache.v"], t["cache.idx_k"] = rec.final_cache()
    out = {}
    for k, x in t.items():
        if x.dtype in (torch.int64, torch.int16, torch.int8):
            x = x.to(torch.int32)
        elif x.dtype == torch.bool:
            x = x.to(torch.uint8)
        out[k] = x.contiguous()
    return out


def views(um, images, crop, spatial, imgs):
    """The uint8 views `_uocr_build_inputs` normalizes, from its own helpers; asserted against what it fed."""
    import numpy as np
    from PIL import ImageOps

    t = {}
    g = [torch.from_numpy(np.asarray(ImageOps.pad(im, (1024, 1024), color=(127, 127, 127))).copy()) for im in images]
    t["views.global_u8"] = torch.stack(g)
    assert torch.equal(normalize(t["views.global_u8"]), imgs[1].cpu()), "global views: uint8 path differs from infer's"
    t["views.global"] = imgs[1].cpu()
    tiles = []
    for im, (w, h) in zip(images, spatial.tolist()):
        if crop and (w > 1 or h > 1):
            tl, ratio = um.dynamic_preprocess(im, min_num=2, max_num=16, image_size=640)
            assert list(ratio) == [w, h]
            tiles += [torch.from_numpy(np.asarray(x).copy()) for x in tl]
    if tiles:
        t["views.crops_u8"] = torch.stack(tiles)
        assert torch.equal(normalize(t["views.crops_u8"]), imgs[0].cpu()), "crops: uint8 path differs from infer's"
        t["views.crops"] = imgs[0].cpu()
    return t


def export_case(model, tok, mods, name, pages, window, args, out):
    from PIL import Image

    um, dsv2, da = mods
    rec = Recorder(model, um, dsv2, da, args)
    t0 = time.time()
    try:
        res = model.infer(tok, prompt="", image_file=[str(p) for p in pages], eval_mode=True, return_dict=True,
                          max_new_tokens=args.max_new, no_repeat_ngram_size=NGRAM, ngram_window=window,
                          page_count_guard=True, max_page_tokens=20000, dash_topm=16)
    finally:
        rec.remove()
    seconds = time.time() - t0
    input_ids, imgs, seq_mask, spatial = rec.inputs
    images = [Image.open(p).convert("RGB") for p in pages]
    t = views(um, images, len(pages) == 1, spatial, imgs)
    t["input_ids"], t["seq_mask"] = input_ids.to(torch.int32), seq_mask.to(torch.uint8)
    t["images_spatial_crop"] = spatial.to(torch.int32)
    t["tokens"] = torch.tensor(res["token_ids"], dtype=torch.int32)
    t.update(collect(rec))
    S = int(input_ids.numel())
    assert t["prefill.tok"].numel() == S and len(rec.picks) == len(rec.steps) + (res["finish_reason"] == "stop")
    eps = (1e-3, 1e-2, 1e-1)
    ties = {}
    for part in ("prefill", "step"):
        if f"{part}.gap12" in t:
            for g in ("gap12", "gap16"):
                x = t[f"{part}.{g}"]
                ties[f"{part}.{g}"] = {"min": float(x.min()), **{f"<{e}": int((x < e).sum()) for e in eps}, "rows": int(x.numel())}
    raw = t["pick.argmax"][: t["tokens"].numel()]
    changed = int((raw != t["tokens"][: raw.numel()]).sum())
    meta = provenance(args, {"case": name, "pages": [p.name for p in pages], "prompt": res["prompt"], "ngram": NGRAM,
                             "window": window, "max_new": args.max_new, "steps": args.steps, "page_n": args.page_n,
                             "late0": args.late0, "late_n": args.late_n})
    from safetensors.torch import save_file

    save_file(t, out / f"{name}.safetensors", metadata=meta)
    info = {"case": name, "pages": [p.name for p in pages], "prompt": res["prompt"], "n_ids": S,
            "n_image_tokens": int(seq_mask.sum()), "spatial": spatial.tolist(), "n_tokens": len(res["token_ids"]),
            "finish_reason": res["finish_reason"], "num_pages_out": res["num_pages_out"],
            "expected_pages": res["expected_pages"], "decode_calls": len(rec.steps),
            "detail_calls": [d["call"] for d in rec.details], "raw_argmax_changed_by_ngram_or_guard": changed,
            "router_params": {str(k): v for k, v in sorted(rec.params.items())}, "counts": dict(rec.counts),
            "triton": "off: use_triton False; _forward_triton, the base attention forward and JITFunction.run raise",
            "near_ties": ties, "generate_seconds": res["time"], "prefill_seconds": res["prefill_time"],
            "total_seconds": seconds, "input_ids": input_ids.tolist(), "tokens": res["token_ids"], "text": res["text"],
            "provenance": meta}
    (out / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
    print(f"{name}: ids={S} img={int(seq_mask.sum())} new={len(res['token_ids'])} finish={res['finish_reason']} "
          f"pages_out={res['num_pages_out']}/{res['expected_pages']} gen={res['time']:.1f}s ngram/guard_changed={changed} "
          f"min gap12 prefill={ties.get('prefill.gap12', {}).get('min')} step={ties.get('step.gap12', {}).get('min')}", flush=True)
    return {k: {"shape": list(x.shape), "dtype": str(x.dtype).replace("torch.", ""), "bytes": x.numel() * x.element_size()} for k, x in t.items()}


def cmd_fixtures(args):
    out = Path(args.out)
    (out / "pages").mkdir(parents=True, exist_ok=True)
    if args.smoke:
        # a made page stands in for a fixture page: the hooks and the bitwise checks, not the bits
        from PIL import Image

        src = out / "pages" / "smoke.png"
        if not src.exists():
            Image.linear_gradient("L").convert("RGB").resize((440, 133), Image.BICUBIC).save(src)
        cases = [("one_page", [src], 128), ("two_pages", [src, src], 1024)]
    else:
        pages_dir = Path(args.pages) if args.pages else out / "pages"
        cases = [(n, [pages_dir / p for p in ps], w) for n, ps, w in CASES]
    if args.only:
        cases = [(n, ps, w) for n, ps, w in cases if n in args.only]
    import shutil

    mf = out / "manifest.json"
    manifest = json.loads(mf.read_text()) if mf.exists() else {"cases": {}}
    keep = set(manifest["cases"]) - {n for n, _, _ in cases}
    for _, ps, _ in cases:
        for p in ps:
            if not p.exists():
                raise SystemExit(f"{p}: no such page image (point --pages at the reference's pages)")
            if p.resolve() != (out / "pages" / p.name).resolve():
                # new bytes would replace OUT's copy: a case the manifest keeps, unregenerated this run,
                # must not reference them under their old hash
                h = sha256_file(p)
                stale = sorted(n for n in keep if manifest["cases"][n].get("pages", {}).get(p.name) not in (None, h))
                if stale:
                    raise SystemExit(f"{p.name}: new bytes would stale {stale}; regenerate all cases (--only off)")
                shutil.copy(p, out / "pages" / p.name)
    tok, model = load(args)
    mods = remote_modules(model)
    triton_off(model, mods[1], mods[2])
    for name, pages, window in cases:
        manifest["cases"][name] = {"file": f"{name}.safetensors",
                                   "pages": {p.name: sha256_file(out / "pages" / p.name) for p in pages},
                                   "tensors": export_case(model, tok, mods, name, [out / "pages" / p.name for p in pages], window, args, out)}
        manifest["cases"][name]["bytes"] = (out / f"{name}.safetensors").stat().st_size
        manifest["provenance"] = provenance(args, {})
        mf.write_text(json.dumps(manifest, indent=1))


def cmd_cer(args):
    """A CER page set through HF's v2, one page a request, no hooks (the replies and HF's own timings).

    The set is DIR/manifest.json's pages (spec 137's `cer/` layout): `{"pages": [{"name", "image"}]}`,
    each image a file beside the manifest or a path it names. The page's image is copied next to its reply
    and the output manifest names it, so OUT/cer is itself a valid `--cer` input.
    """
    import shutil

    src = Path(args.cer) if args.cer else Path(args.out) / "cer"
    mf = src / "manifest.json"
    if not mf.exists():
        raise SystemExit(f"{mf}: no CER manifest (point --cer at a page set)")
    out = Path(args.out) / "cer"
    out.mkdir(parents=True, exist_ok=True)
    pages = json.loads(mf.read_text())["pages"]
    tok, model = load(args)
    mods = remote_modules(model)
    triton_off(model, mods[1], mods[2])
    # an in-place or --only run must not lose the pages it did not generate: every entry of the output
    # manifest is kept, input order first, the unselected ones verbatim (their images copied beside, so the
    # output set stays self-contained)
    prior_mf = out / "manifest.json"
    prior = json.loads(prior_mf.read_text())["pages"] if prior_mf.exists() and prior_mf != mf else []
    result = []
    for page in pages:
        name = page["name"]
        image = Path(page["image"])
        if not image.is_absolute():
            image = src / image
        copied = f"{name}{image.suffix}"
        if image.resolve() != (out / copied).resolve():
            if not image.exists():
                raise SystemExit(f"{image}: page {name}'s image is missing")
            shutil.copy(image, out / copied)
        if args.only and name not in args.only:
            result.append({**{k: v for k, v in page.items() if k != "tokens"}, "image": copied})
            continue
        t0 = time.time()
        res = model.infer(tok, prompt="", image_file=str(image), eval_mode=True, return_dict=True,
                          max_new_tokens=args.max_new, no_repeat_ngram_size=NGRAM, ngram_window=128,
                          page_count_guard=True, max_page_tokens=20000, dash_topm=16)
        info = {"name": name, "image": copied, "prompt": res["prompt"], "mode": "gundam",
                "ngram": NGRAM, "window": 128, "n_tokens": len(res["token_ids"]), "eos": res["finish_reason"] == "stop",
                "finish_reason": res["finish_reason"], "max_new": args.max_new, "generate_seconds": res["time"],
                "prefill_seconds": res["prefill_time"], "total_seconds": time.time() - t0, "tokens": res["token_ids"]}
        (out / f"{name}.txt").write_text(res["text"])
        (out / f"{name}.json").write_text(json.dumps(info, ensure_ascii=False))
        result.append(info)
        print(f"{name}: tokens={info['n_tokens']} {res['finish_reason']} gen={res['time']:.1f}s prefill={res['prefill_time']:.2f}s", flush=True)
    names = {p["name"] for p in pages}
    result += [x for x in prior if x["name"] not in names]  # output entries the input no longer names
    meta = provenance(args, {"set": str(src), "ngram": NGRAM, "window": 128, "max_new": args.max_new})
    slim = [{k: v for k, v in x.items() if k != "tokens"} for x in result]
    (out / "manifest.json").write_text(json.dumps({"provenance": meta, "pages": slim}, indent=1))


# ---------------------------------------------------------------------------------------------------------------
# dry: imports, the model's structure on the meta device and the inputs, on the CPU without weights


def empty_model(cls, cfg):
    """The model's modules with its parameters on the meta device (buffers stay real, as accelerate's
    init_empty_weights): its structure without 6.7 GB of weights."""
    from transformers.modeling_utils import no_init_weights

    reg = torch.nn.Module.register_parameter

    def on_meta(self, name, param):
        reg(self, name, param)
        if param is not None:
            self._parameters[name] = torch.nn.Parameter(param.to("meta"), requires_grad=param.requires_grad)

    torch.nn.Module.register_parameter = on_meta
    try:
        with no_init_weights():
            return cls(cfg)
    finally:
        torch.nn.Module.register_parameter = reg


def cmd_dry(args):
    from PIL import Image
    from transformers import AutoConfig, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_unlimitedocr.UnlimitedOCRForCausalLM", args.model)
    um, dsv2, da = remote_modules(cls)
    for mod, names in ((um, ("_uocr_build_inputs", "_uocr_dash_decode_allow", "dynamic_preprocess")),
                       (dsv2, ("_infer_prefill_attend", "_dash_decode_forward", "_router_qk", "_block_scores_infer",
                               "_infer_route_select", "_decode_attend", "_dash_page_ids", "_orig_forward", "_entmax15",
                               "_ENTMAX_NEG", "_SCORE_NEG", "_base", "configure_dash_inference")),
                       (da, ("DashAttention", "_ACT_CLAMP"))):
        missing = [n for n in names if not hasattr(mod, n)]
        assert not missing, f"{mod.__name__} lacks {missing}"
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    m = empty_model(cls, cfg)
    snap = dsv2._CKPT_SP.get(id(cfg.sparse_attention_config))
    assert snap is not None and snap["dash_stage0"] == "head_cls", "the checkpoint's router snapshot is missing"
    for i, layer in enumerate(m.model.layers):
        at = layer.self_attn
        assert at.dash_router is not None and at.rotary_emb_2d_main is not None and at.indexer.rotary_emb_2d is not None, i
        assert at.dash_router.use_triton, "use_triton read False from the config: this script's switch would be moot"
    triton_off(m, dsv2, da)
    print(f"model: {len(m.model.layers)} layers, dash_router on each, page2d main and router rope; Triton switched off")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    for name, ps, w in CASES:
        pages_dir = Path(args.pages) if args.pages else Path(args.out) / "pages"
        images = [Image.open(pages_dir / p).convert("RGB") for p in ps]
        prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
        ids, imgs, mask, spatial = um._uocr_build_inputs(tok, prompt, images, crop_mode=None, max_crops=16)
        np_ = int((ids == PAGE_ID).sum())
        print(f"{name}: {[im.size for im in images]} ids={ids.numel()} image_tokens={int(mask.sum())} pages={np_} "
              f"spatial={spatial.tolist()} crops={tuple(imgs[0].shape)} globals={tuple(imgs[1].shape)} window={w}")
        views(um, images, len(images) == 1, spatial, imgs)
    print("dry: ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["fixtures", "dry", "cer"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out")
    ap.add_argument("--pages", help="the fixture cases' page images (one_page, two_pages; OUT/pages unset)")
    ap.add_argument("--cer", help="a CER page set: DIR/manifest.json's pages (OUT/cer unset)")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--steps", type=int, default=8, help="detailed decode calls from the first")
    ap.add_argument("--page-n", type=int, default=4, help="detailed decode calls from each one that feeds <PAGE>")
    ap.add_argument("--late0", type=int, default=300, help="first decode call of the late detailed window")
    ap.add_argument("--late-n", type=int, default=4)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--smoke", action="store_true", help="CPU, a made page, 3 new tokens, into scratch-147")
    ap.add_argument("--cpu", action="store_true", help="the real cases on the CPU, into scratch-147 (not fixtures)")
    args = ap.parse_args()
    args.device = "cpu" if args.smoke or args.cpu else "cuda"
    if args.smoke:
        args.out = args.out or SMOKE_OUT
        args.max_new = min(args.max_new, 3)
    elif args.cpu:
        args.out = args.out or CPU_OUT
    args.out = args.out or OUT
    {"fixtures": cmd_fixtures, "dry": cmd_dry, "cer": cmd_cer}[args.cmd](args)


if __name__ == "__main__":
    main()
