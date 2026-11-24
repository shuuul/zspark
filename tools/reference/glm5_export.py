#!/usr/bin/env python3
"""Reference tensors of GLM-5.3-Flash's first layers, both ranks in one process, exported from TensorFold itself.

Kept for ``glm5_drafter_export.py``, which imports it (spec 121 WS-00): the tests that read this export's own
files ran on the EXL3 pack, retired, and were deleted (tag ``retired-checkpoints-2026-10-21`` holds them).

A test fixture: it runs inside the TensorFold container (PyTorch, Triton), never at zspark's runtime. This is the
small mode of spec 044: the first N layers of the EXL3 pack (``--layers 4``: three KDA layers with dense MLPs
and layer 3, DSA with an MoE) and, with ``--mtp``, the MTP layer, on one Spark. Without ``--layers`` it is the
whole model, which one Spark cannot hold at once (both ranks are 164 GiB), in pieces.

Pieces (``--pieces P``, spec 067 WS-13). Layers a .. b-1 of both ranks are loaded, run and freed, P layers at a
time. ``forward.compute`` is ``glue.embed`` and then a loop in which a layer reads the streams ``b.x`` and its own
state (a KDA layer's ``rec``, ``conv``, ``proj`` and scratch by ``st.kda_index``, a DSA layer's latents by
``st.dsa_index``, the position); so a piece run from an empty state on the streams the piece before left
(``final.h``, written over ``glue.embed``'s output by the hook) is those layers of the whole forward, provided
the rows start at row 0 of the sequence, which a case's do. ``--layers 4 --pieces 2 --against DIR`` holds that
against the files a single load of four layers wrote, bit for bit. A piece's ``LayerW.index`` is set to the
model's, so names, ``State``'s indices and the taps are the whole model's; every piece but the last also runs
``stream_mean``, the norm and its copy of the head on its last streams, which nothing keeps.

Taps (``--taps 5,14,24,33,42``, DFlash2's ``target_layer_ids``): ``Buffers.set_taps`` with the piece's own, so
``compute`` writes them; ``taps.{L}`` BF16 [R, D] is ``b.taps[slot]`` after layer L.

Both ranks in one process. TensorFold's CUDA GLM exists for two ranks only (``weights.load`` sets ``world = 2``),
and a block ends in ``forward.gather``, which asks ``w.comm.all_gather`` for every rank's fp32 partial before
``glue.hc_post`` adds them rank 0 first. Here each rank is a thread with its own ``Weights``, ``State`` and
``Buffers`` on the one GPU, and ``w.comm`` is ``ThreadComm``, a stand-in for ``tensorfold.cuda.comm.NCCL`` in
the form of TensorFold's own test double (``tests/cuda/test_flashnext_tp.py:_ThreadComm``): a rank leaves its
``send``, waits at a barrier for the other, copies both into ``recv`` in rank order and waits again. So
``forward.compute``, ``mtp.mtp_compute`` and ``decode.sample_rows`` run unchanged, in their own order, and
every exchange is the one two machines make. One thread runs at a time (``Hub.turn``, given up only at the
barrier): Triton's compile cache and ``kda._wide_scratch`` are shared by the threads and not written for two.

A truncation. ``--layers N`` shortens ``weights.Config.read`` (``layers``, ``kinds``, ``mlp_kinds``), so
``weights.load`` builds layers 0 .. N-1, the norm and each rank's half of ``lm_head``; ``forward.compute`` then
applies ``stream_mean``, the final norm and the head to layer N-1's streams. ``final.*``, ``logits.*`` and the
MTP layer's ``mtp.hidden`` are therefore that arithmetic on a 4-layer model's rows, not GLM's final row: a
reference for the kernels and the wiring, not for a token. The MTP layer is the pack's real one (``layers.45``):
the loader asks for ``layers.N.*`` and a ``split.RankReader`` subclass reads ``layers.45.*`` in its place.

Pinned to TensorFold 0.6.6 (commit cb2ebf0), run from an archive of that commit (TensorFold's tree has moved to
1.0.2, where this family is Zig). It relies on these internals of ``tensorfold.families.glm5_next.cuda``, all
read or wrapped from here and none edited:

- ``weights.load(dir, rank=r, mtp=...)``, ``weights.Config.read`` and ``split.RankReader`` (``get``,
  ``prefetch``), wrapped as above; rank 1's ``Weights.embed`` is replaced by rank 0's once they are equal bits;
- ``forward.Buffers(w, rows, capacity)`` and ``forward.State(w, capacity, rows)``, built as
  ``decode.Engine.__init__`` builds ``buf``, ``mbuf`` and ``st`` (no prompt buffers: prompts run through the
  decode path, as zspark runs them); ``forward.forward`` and ``forward.commit``; ``State.reset``,
  ``set_mtp_len``, ``mtp_drafted``, ``rec``, ``cur``, ``conv``, ``proj``, ``scratch``, ``kc``, ``mtp_kc``, ``pos``;
- the functions ``forward.layer_forward`` calls by name, wrapped to clone the buffers they leave: ``kda_block``,
  ``dsa_block``, ``mlp_block``, ``moe_block``, and ``glue.embed``, ``glue.hc_pre``, ``glue.hc_post``;
- the buffers: ``b.x``, ``b.normed``, ``b.post``, ``b.comb``, ``b.hcpart``, ``b.gath`` (through ``hc_post``'s
  argument), ``b.ka``, ``b.kg``, ``b.dp``, ``b.qr``, ``b.lat``, ``b.q``, ``b.lat_s.qa``, ``b.lat_s.ol``, ``b.vn``,
  ``b.gu``, ``b.act``, ``b.mlog``, ``b.pick``, ``b.wts``, ``b.ey``, ``b.hidden``, ``b.fnormed``, ``b.fxs``,
  ``b.logits``, ``b.ids``, ``b.hin``, ``b.me``, ``b.mx``;
- with ``--mtp``: ``mtp.mtp_forward(w, st, mb, next_tokens, hidden, last_only=...)``, the names ``mtp.py``
  imported from ``forward.py`` and calls as its own globals (``mm``, ``dsa_block``, ``moe_block``),
  ``glue.residual_add``, ``Weights.mtp`` (``eh``), ``Weights.head`` and ``Weights.draft_head``,
  ``decode.sample_rows(w, logits, positions, None, None, probs)``, ``decode.draft`` on a ``decode.Engine`` made
  without ``__init__`` (``w``, ``st``, ``buf``, ``mbuf``, no graphs), and ``forward.mm``;
- ``TF_GLM_LATENT`` left at its default: DSA on the 512-wide latent cache (``latent.ENABLED``); no long context,
  so ``State.index`` is None and the indexer does not run. ``--index`` is the exception: after the cases it
  builds a second state with the indexer on and runs one prompt past the dense limit, writing ``index.json``
  and ``index.safetensors`` (the pool's rows and one sparse row's list).

Tensor names. Rows on the first axis; R rows, D = hidden (4096), S = streams (4). A name ending ``.r0`` or
``.r1`` is that rank's (its half of the heads, of an inner width or of the vocabulary); a name without is the
same bits on both ranks, which the exporter checks (``ranks_differ`` in the JSON) and stores once.

    ids                              I32  [R]
    embed.h                          BF16 [R, S*D]   the streams after glue.embed (b.x)
    layers.{i}.h_in                  BF16 [R, S*D]   the streams the layer read
    layers.{i}.attn_hc.parts         F32  [R, 16, 25] glue._hc_partial's blocks: 24 dots, then the sum of squares
    layers.{i}.attn_hc.normed        BF16 [R, D]     the site's collapsed, normed row (b.normed)
    layers.{i}.attn_hc.post          F32  [R, S]
    layers.{i}.attn_hc.comb          F32  [R, S, S]  [r, i, s]: stream i's weight in output stream s
    layers.{i}.kda.proj.r{k}         BF16 [R, 12576] KDA: [q | k | v | f_a | g_a | b] of the rank (st.proj)
    layers.{i}.kda.f.r{k}            BF16 [R, 4096]  f_b on the f_a columns (b.ka)
    layers.{i}.kda.g.r{k}            BF16 [R, 4096]  g_b on the g_a columns (b.kg)
    layers.{i}.kda.out.r{k}          BF16 [R, 4096]  the chain's rows, o_proj's input (st.scratch[li].out)
    layers.{i}.dsa.proj              BF16 [R, 2048]  DSA: [q_a | kv_a] (b.dp)
    layers.{i}.dsa.q_normed          BF16 [R, 1536]  (b.qr)
    layers.{i}.dsa.latent            BF16 [R, 512]   the latent written to the cache (b.lat)
    layers.{i}.dsa.q.r{k}            BF16 [R, 32, 256]   q_b (b.q)
    layers.{i}.dsa.qa.r{k}           BF16 [R, 32, 512]   the query with kv_b's key rows absorbed (b.lat_s.qa)
    layers.{i}.dsa.ol.r{k}           BF16 [R, 32, 512]   attention over the latents (b.lat_s.ol)
    layers.{i}.dsa.out.r{k}          BF16 [R, 32, 256]   expanded by kv_b's value rows, o_proj's input (b.vn)
    layers.{i}.attn.gath             F32  [R, 2, D]  the gathered partials, axis 1 the rank: what hc_post adds
    layers.{i}.h_mid                 BF16 [R, S*D]   the streams after the attention's write-back
    layers.{i}.mlp_hc.parts, .normed, .post, .comb   the MLP's site, as the attention's
    layers.{i}.mlp.gu.r{k}           BF16 [R, 12288] dense: [gate | up] of the rank's 6144 (b.gu)
    layers.{i}.mlp.act.r{k}          BF16 [R, 6144]  (b.act)
    layers.{i}.moe.logits            F32  [R, 288]   MoE: the router (b.mlog)
    layers.{i}.moe.picks             I32  [R, 9]     the last slot is the shared expert (id 288)
    layers.{i}.moe.weights           F32  [R, 9]
    layers.{i}.moe.y.r{k}            F32  [R, 9, D]  the slots' rows over the rank's half width (b.ey)
    layers.{i}.mlp.gath              F32  [R, 2, D]
    taps.{L}                         BF16 [R, D]     glue.stream_mean after layer L (b.taps), with --taps
    final.h                          BF16 [R, S*D]   after the last exported layer's write-back
    final.mean                       BF16 [R, D]     glue.stream_mean (b.hidden)
    final.normed                     BF16 [R, D]     the final norm (b.fnormed): the MTP layer's input rows
    logits.r{k}                      BF16 [R, 77440] the rank's rows of lm_head, ids k * 77440 ..
    state.kda.rec.r{k}               F32  [KDA layers, 32, 128, 128]   after the prompt was committed
    state.kda.conv.r{k}              BF16 [KDA layers, 3, 12288]       the conv's three rows
    state.dsa.latents                BF16 [R, DSA layers, 512]         the caches' first R slots

``NAME.serial.f32.safetensors`` is R one-row steps, ``NAME.window.f32.safetensors`` one window of the R rows,
each from an empty state with every row committed; ``NAME.f32.json`` names every tensor with its dtype and
shape and compares the two point by point (``.f32`` as Flash Next's files with fp32 expert slots: an EXL3
pack's slots are fp32).

The MTP layer (``--mtp``). Row i's ``final.normed`` of the serial run and the token after row i are its
inputs; the last row pairs with the token TensorFold's sampler takes from the last row's logits
(``pending``). It runs on buffers of its own from an empty cache, as R one-row steps (``NAME.mtp.serial``)
and as one window (``NAME.mtp.window``), every row headed (``last_only=False``):

    mtp.next                         I32  [R]
    mtp.hidden                       BF16 [R, D]     (b.hin)
    mtp.me                           BF16 [R, D]     the next tokens' embedding, row 0 of a sequence zeroed (b.me)
    mtp.e_normed, mtp.h_normed       BF16 [R, D]     enorm and hnorm: the halves of eh_proj's input (b.mcat)
    mtp.x                            BF16 [R, D]     eh_proj (b.mx)
    mtp.attn.normed                  BF16 [R, D]     input_layernorm (b.normed)
    mtp.dsa.*                        as layers.{i}.dsa.*
    mtp.attn.gath                    F32  [R, 2, D]
    mtp.x_mid                        BF16 [R, D]     after glue.residual_add
    mtp.mlp.normed                   BF16 [R, D]     post_attention_layernorm
    mtp.moe.*                        as layers.{i}.moe.*
    mtp.mlp.gath                     F32  [R, 2, D]
    mtp.x_out                        BF16 [R, D]
    mtp.final                        BF16 [R, D]     shared_head.norm: a chained draft's hidden row (b.fnormed)
    mtp.logits.r{k}                  BF16 [R, 77440] TensorFold's draft head: quantize4's 4-bit copy of lm_head
    mtp.logits_bf16.r{k}             BF16 [R, 77440] the exporter's: lm_head as stored on the same mtp.final
    mtp.cache.latents                BF16 [R, 512]   the layer's own cache (st.mtp_kc)

``NAME.mtp.chain`` has those names with a chain step on the first axis (``--mtp-depth`` rows), after rows
0 .. R-2 were absorbed: row 0 is the last prompt row with ``pending`` again, row j > 0 is
``mtp_forward([draft j-1], row j-1's mtp.final)``. ``mtp.draft`` I32 and ``mtp.prob`` F32 are what
``decode.sample_rows`` chose from each row's ``mtp.logits`` and its probability (over the 28 best of each
rank, ``decode._probability``). The exporter chains every row; ``NAME.mtp.json`` holds what TensorFold's rule
keeps of them (``chain.rule_keeps``) and what ``decode.draft`` itself returned from the same state
(``chain.decode_draft``; depth 3 and confidence 0.35, the engine's greedy policy).

Each file's metadata holds ``provenance``, a JSON string: TensorFold's version and commit (stated with
``--tensorfold-commit``, the archive has no ``.git``), the pack's revision, the layers exported, the
truncation, torch and triton versions, the mode and the ids. The command is in ``README.md``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import torch

PINNED = "0.6.6"
WORLD = 2
MTP_DEPTH, MTP_CONFIDENCE = 3, 0.35      # engine.py:_run_once, a greedy reply's MTP policy
LOCAL = threading.local()                # .tap: the rank's Tap while its forward runs in this thread


# -- the ranks -----------------------------------------------------------------------------------------
class Hub:
    """What the two rank threads share: the sends of an exchange, the barrier they meet at, and the turn."""

    def __init__(self) -> None:
        self.slots: list = [None] * WORLD
        self.barrier = threading.Barrier(WORLD, timeout=600)
        self.turn = threading.Lock()         # held by the one rank that runs
        self.gathers = 0


class ThreadComm:
    """``comm.NCCL``'s ``all_gather`` between ranks that are threads of one process on one GPU."""

    def __init__(self, hub: Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, WORLD

    def _meet(self) -> None:
        self.hub.turn.release()
        try:
            self.hub.barrier.wait()
        finally:
            self.hub.turn.acquire()

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """recv [world * n] <- every rank's send [n], in rank order."""

        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        self.hub.slots[self.rank] = send
        self._meet()                         # both sends stand; neither rank writes its own before the next meet
        n, flat = send.numel(), recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        self.hub.gathers += self.rank == 0
        self._meet()

    def barrier(self) -> None:
        self._meet()


def run_ranks(hub: Hub, fn) -> list:
    """fn(rank) on both ranks at once, a thread each; the results in rank order."""

    results: list = [None] * WORLD
    errors: list = []

    def body(rank: int) -> None:
        hub.turn.acquire()
        try:
            with torch.no_grad():
                results[rank] = fn(rank)
        except BaseException as exc:         # noqa: BLE001  (raised below; the other rank is let go)
            errors.append(exc)
            hub.barrier.abort()
        finally:
            LOCAL.tap = None
            hub.turn.release()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(WORLD)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        hub.barrier.reset()
        own = [e for e in errors if not isinstance(e, threading.BrokenBarrierError)]
        raise (own or errors)[0]
    return results


# -- capture -------------------------------------------------------------------------------------------
class Tap:
    """One rank's captures of a forward: name -> a clone on the device, in the order they were produced.

    TensorFold's buffers are reused by the next layer and the next step, so every capture is a clone taken as
    the function that wrote it returns. ``split`` holds the names that are the rank's own.
    """

    def __init__(self, rank: int, w) -> None:
        self.rank, self.w = rank, w
        self.rec: dict[str, torch.Tensor] = {}
        self.split: set[str] = set()
        self.layer = None                    # the LayerW ``layer_forward`` is inside
        self.site = 0                        # 0: the attention's hyper-connection, 1: the MLP's
        self.adds = 0                        # the MTP layer's residual adds so far
        self.feed = None                     # a later piece: the streams its first layer reads, [R, S*D]
        self.at = 0                          # the forward's first row in ``feed``

    def put(self, name: str, t: torch.Tensor, split: bool = False) -> None:
        self.rec[name] = t.detach().clone()
        if split:
            self.split.add(name)

    def take(self) -> dict[str, torch.Tensor]:
        got, self.rec = {name: t.cpu() for name, t in self.rec.items()}, {}
        return got


def tap() -> Tap | None:
    return getattr(LOCAL, "tap", None)


def dsa_points(t: Tap, p: str, b, R: int) -> None:
    t.put(f"{p}.dsa.proj", b.dp[:R])
    t.put(f"{p}.dsa.q_normed", b.qr[:R])
    t.put(f"{p}.dsa.latent", b.lat[:R])
    t.put(f"{p}.dsa.q", b.q[:R], True)
    t.put(f"{p}.dsa.qa", b.lat_s.qa[:R], True)
    t.put(f"{p}.dsa.ol", b.lat_s.ol[:R], True)
    t.put(f"{p}.dsa.out", b.vn[:R], True)


def moe_points(t: Tap, p: str, b, R: int) -> None:
    t.put(f"{p}.moe.logits", b.mlog[:R])
    t.put(f"{p}.moe.picks", b.pick[:R])
    t.put(f"{p}.moe.weights", b.wts[:R])
    t.put(f"{p}.moe.y", b.ey[:R], True)


def install(fwd, glue, mtp_mod) -> None:
    """Wrap the functions ``forward.compute`` and ``mtp.mtp_compute`` call; a thread without a Tap passes through."""

    o_embed, o_pre, o_post, o_add = glue.embed, glue.hc_pre, glue.hc_post, glue.residual_add
    o_layer, o_kda, o_dsa, o_mlp, o_moe = fwd.layer_forward, fwd.kda_block, fwd.dsa_block, fwd.mlp_block, fwd.moe_block
    m_mm, m_dsa, m_moe = mtp_mod.mm, mtp_mod.dsa_block, mtp_mod.moe_block

    def embed(ids, table, dims, copies, out):
        got = o_embed(ids, table, dims, copies, out)
        t = tap()
        if t is not None and copies > 1:     # the MTP layer's one copy is mtp.me, taken once row 0 is zeroed
            if t.feed is None:
                t.put("embed.h", out)
            else:                            # a later piece: the streams the piece before it left
                out.copy_(t.feed[t.at:t.at + out.shape[0]])
        return got

    def layer_forward(layer, w, st, b, R, *a, **k):
        t = tap()
        if t is not None:
            t.layer, t.site = layer, 0
            t.put(f"layers.{layer.index}.h_in", b.x[:R])
        got = o_layer(layer, w, st, b, R, *a, **k)
        if t is not None:
            t.layer = None
        return got

    def hc_pre(x, fn, base, scale, norm_w, out, xs, post, comb, part, *a):
        o_pre(x, fn, base, scale, norm_w, out, xs, post, comb, part, *a)
        t = tap()
        if t is not None:
            p = f"layers.{t.layer.index}." + ("attn_hc" if t.site == 0 else "mlp_hc")
            R, S = post.shape
            t.put(p + ".parts", part[:, :, :25])          # columns 25 .. 31 of a block are never written
            t.put(p + ".normed", out)
            t.put(p + ".post", post)
            t.put(p + ".comb", comb.view(R, S, S))

    def hc_post(x, xout, gathered, post, comb):
        t = tap()
        if t is not None:
            t.put(f"layers.{t.layer.index}." + ("attn" if t.site == 0 else "mlp") + ".gath", gathered.permute(1, 0, 2))
        o_post(x, xout, gathered, post, comb)
        if t is not None:
            if t.site == 0:
                t.put(f"layers.{t.layer.index}.h_mid", xout)
            t.site = 1

    def kda_block(layer, w, st, b, R, *a, **k):
        g = o_kda(layer, w, st, b, R, *a, **k)
        t = tap()
        if t is not None:
            li, p = st.kda_index[layer.index], f"layers.{layer.index}.kda"
            t.put(p + ".proj", st.proj[li, :R], True)
            t.put(p + ".f", b.ka[:R], True)
            t.put(p + ".g", b.kg[:R], True)
            t.put(p + ".out", st.scratch[li].out[:R], True)
        return g

    def dsa_block(layer, w, kc, vc, pos_dev, b, R, *a, **k):
        g = o_dsa(layer, w, kc, vc, pos_dev, b, R, *a, **k)
        t = tap()
        if t is not None:
            dsa_points(t, f"layers.{layer.index}", b, R)
        return g

    def mlp_block(layer, w, b, R):
        g = o_mlp(layer, w, b, R)
        t = tap()
        if t is not None:
            t.put(f"layers.{layer.index}.mlp.gu", b.gu[:R], True)
            t.put(f"layers.{layer.index}.mlp.act", b.act[:R], True)
        return g

    def moe_block(layer, w, b, R):
        g = o_moe(layer, w, b, R)
        t = tap()
        if t is not None:
            moe_points(t, f"layers.{layer.index}", b, R)
        return g

    def mtp_mm(b, x, q, xs, out, f32=False):
        t = tap()
        if t is None:
            return m_mm(b, x, q, xs, out, f32)
        n, D = x.shape[0], t.w.cfg.hidden
        if q is t.w.mtp.eh:
            t.adds = 0
            t.put("mtp.next", b.ids[:n])
            t.put("mtp.hidden", b.hin[:n])
            t.put("mtp.me", b.me[:n])
            t.put("mtp.e_normed", x[:, :D])
            t.put("mtp.h_normed", x[:, D:])
            got = m_mm(b, x, q, xs, out, f32)
            t.put("mtp.x", out)
            return got
        t.put("mtp.final", x)                # the head's call: the rows shared_head.norm left
        got = m_mm(b, x, q, xs, out, f32)
        t.put("mtp.logits", got, True)
        return got

    def mtp_dsa(layer, w, kc, vc, pos_dev, b, R, *a, **k):
        t = tap()
        if t is not None:
            t.put("mtp.attn.normed", b.normed[:R])
        g = m_dsa(layer, w, kc, vc, pos_dev, b, R, *a, **k)
        if t is not None:
            dsa_points(t, "mtp", b, R)
            t.put("mtp.attn.gath", g.permute(1, 0, 2))
        return g

    def mtp_moe(layer, w, b, R):
        t = tap()
        if t is not None:
            t.put("mtp.mlp.normed", b.normed[:R])
        g = m_moe(layer, w, b, R)
        if t is not None:
            moe_points(t, "mtp", b, R)
            t.put("mtp.mlp.gath", g.permute(1, 0, 2))
        return g

    def residual_add(x, xout, gathered):
        o_add(x, xout, gathered)
        t = tap()
        if t is not None:
            t.put("mtp.x_mid" if t.adds == 0 else "mtp.x_out", xout)
            t.adds += 1

    glue.embed, glue.hc_pre, glue.hc_post, glue.residual_add = embed, hc_pre, hc_post, residual_add
    fwd.layer_forward, fwd.kda_block, fwd.dsa_block = layer_forward, kda_block, dsa_block
    fwd.mlp_block, fwd.moe_block = mlp_block, moe_block
    mtp_mod.mm, mtp_mod.dsa_block, mtp_mod.moe_block = mtp_mm, mtp_dsa, mtp_moe


# -- runs ----------------------------------------------------------------------------------------------
class Run:
    """Both ranks' weights, states and buffers, and TensorFold's modules."""

    def __init__(self, fwd, mtp_mod, decode, ws: list, rows: int, capacity: int, taps: tuple = ()) -> None:
        self.fwd, self.mtp_mod, self.decode, self.ws = fwd, mtp_mod, decode, ws
        self.hub = Hub()
        for rank, w in enumerate(ws):
            w.comm = ThreadComm(self.hub, rank)
        self.bufs = [fwd.Buffers(w, rows, capacity) for w in ws]
        for b in self.bufs:
            b.set_taps(tuple(taps), ws[0].cfg.hidden)
        self.mbufs = [fwd.Buffers(w, rows, capacity) if w.mtp is not None else None for w in ws]
        self.sts = [fwd.State(w, capacity, rows) for w in ws]
        self.taps = [Tap(rank, w) for rank, w in enumerate(ws)]

    def both(self, fn) -> list:
        return run_ranks(self.hub, fn)

    def split(self) -> set[str]:
        return set().union(*(t.split for t in self.taps))


def bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def merge(per_rank: list[dict[str, torch.Tensor]], split: set[str]) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The ranks' captures as one record: a rank's own tensors under ``.r0`` and ``.r1``, the others once, with
    the names whose bits differ between the ranks (there should be none)."""

    out: dict[str, torch.Tensor] = {}
    differ: list[str] = []
    for name, first in per_rank[0].items():
        if name in split:
            for rank, rec in enumerate(per_rank):
                out[f"{name}.r{rank}"] = rec[name]
            continue
        for rec in per_rank[1:]:
            if rec[name].shape != first.shape or bool((bits(rec[name]) != bits(first)).any()):
                differ.append(name)
        out[name] = first
    return out, differ


def joined(parts: list[list[dict[str, torch.Tensor]]]) -> list[dict[str, torch.Tensor]]:
    """parts[rank][forward] -> a record a rank, the forwards' rows one after another."""

    return [{name: torch.cat([p[name] for p in rank]) for name in rank[0]} for rank in parts]


def run_windows(run: Run, ids: list[int], rows: int, feed=None) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The ids from an empty state in windows of ``rows`` rows (1: serial steps), every row committed; ``feed``
    is the streams a later piece's first layer reads in place of the embedding."""

    for st in run.sts:
        st.reset()
    parts: list[list] = [[] for _ in range(WORLD)]
    for at in range(0, len(ids), rows):
        chunk = ids[at:at + rows]
        n = len(chunk)

        def step(rank: int) -> dict[str, torch.Tensor]:
            w, st, b, t = run.ws[rank], run.sts[rank], run.bufs[rank], run.taps[rank]
            LOCAL.tap = t
            t.feed, t.at = feed, at
            logits = run.fwd.forward(w, st, b, chunk)
            for layer, slots in b.tap_at.items():
                t.put(f"taps.{layer}", b.taps[slots[0]][:n])
            t.put("final.h", b.x[:n])
            t.put("final.mean", b.hidden[:n])
            t.put("final.normed", b.fnormed[:n])
            t.put("logits", logits[:n], True)
            LOCAL.tap = None
            run.fwd.commit(w, st, b, n, n)
            return t.take()

        for rank, rec in enumerate(run.both(step)):
            parts[rank].append(rec)
    per_rank = joined(parts)
    split = run.split() | {"state.kda.rec", "state.kda.conv"}
    for rank, st in enumerate(run.sts):
        if st.cur:
            per_rank[rank]["state.kda.rec"] = st.rec[st.cur[0]].cpu()
            per_rank[rank]["state.kda.conv"] = st.conv.cpu()
        if st.kc:
            per_rank[rank]["state.dsa.latents"] = torch.stack([kc[:st.pos] for kc in st.kc], dim=1).cpu()
    torch.cuda.synchronize()
    return merge(per_rank, split)


def run_mtp(run: Run, hidden: torch.Tensor, nxt: list[int], rows: int) -> tuple[dict[str, torch.Tensor], list[str]]:
    """The MTP layer from an empty cache over (hidden row, next token) in windows of ``rows`` rows, every row
    headed, the way ``decode.absorb`` advances the cache."""

    for st in run.sts:
        st.set_mtp_len(0)
        st.mtp_drafted = 0
    parts: list[list] = [[] for _ in range(WORLD)]
    for at in range(0, len(nxt), rows):
        part = nxt[at:at + rows]
        n = len(part)

        def step(rank: int) -> dict[str, torch.Tensor]:
            w, st, mb, t = run.ws[rank], run.sts[rank], run.mbufs[rank], run.taps[rank]
            LOCAL.tap = t
            run.mtp_mod.mtp_forward(w, st, mb, part, hidden[at:at + n], last_only=False)
            bf16_head(run, rank, n)
            LOCAL.tap = None
            st.set_mtp_len(st.mtp_len + n)
            return t.take()

        for rank, rec in enumerate(run.both(step)):
            parts[rank].append(rec)
    per_rank = joined(parts)
    for rank, st in enumerate(run.sts):
        per_rank[rank]["mtp.cache.latents"] = st.mtp_kc[:st.mtp_len].cpu()
    return merge(per_rank, run.split())


def bf16_head(run: Run, rank: int, n: int) -> None:
    """``lm_head`` as stored on the rows the MTP layer just normed: TensorFold drafts from a 4-bit copy instead."""

    w, mb = run.ws[rank], run.mbufs[rank]
    full = torch.empty((n, w.head.n), dtype=torch.bfloat16, device=w.device)
    run.fwd.mm(mb, mb.fnormed[:n], w.head, mb.fxs[:n], full)
    run.taps[rank].put("mtp.logits_bf16", full, True)


def run_chain(run: Run, last_hidden: torch.Tensor, pending: int, absorbed: int, position: int, depth: int):
    """``decode.draft``'s chain after ``absorbed`` rows, every step captured: row 0 absorbs (last hidden row,
    pending), row j > 0 runs the previous draft on the layer's own normed row; no confidence ends it."""

    for st in run.sts:
        st.set_mtp_len(absorbed)
        st.mtp_drafted = 0
    parts: list[list] = [[] for _ in range(WORLD)]
    drafts: list[int] = []
    probs: list[float] = []
    for j in range(depth):
        token = pending if j == 0 else drafts[-1]

        def step(rank: int):
            w, st, mb, t = run.ws[rank], run.sts[rank], run.mbufs[rank], run.taps[rank]
            LOCAL.tap = t
            prev = last_hidden if j == 0 else mb.fnormed[0:1]        # Engine.draft_hidden
            logits = run.mtp_mod.mtp_forward(w, st, mb, [token], prev, last_only=True)
            bf16_head(run, rank, 1)
            LOCAL.tap = None
            p: list[float] = []
            d = run.decode.sample_rows(w, logits[:1], [position + j], None, None, p)[0]
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += j > 0
            return t.take(), d, p[0]

        got = run.both(step)
        if got[0][1:] != got[1][1:]:
            raise RuntimeError(f"chain step {j}: the ranks chose {got[0][1:]} and {got[1][1:]}")
        for rank, (rec, _, _) in enumerate(got):
            parts[rank].append(rec)
        drafts.append(got[0][1])
        probs.append(got[0][2])
    rec, differ = merge(joined(parts), run.split())
    rec["mtp.draft"] = torch.tensor(drafts, dtype=torch.int32)
    rec["mtp.prob"] = torch.tensor(probs, dtype=torch.float32)
    return rec, differ, drafts, probs


def rule_keeps(drafts: list[int], probs: list[float], count: int, confidence: float) -> list[int]:
    """What ``decode.draft`` keeps of a chain: the first draft always, then drafts while the product holds."""

    out: list[int] = []
    chain = 1.0
    for j in range(min(count, len(drafts))):
        if confidence > 0 and j > 0 and chain * probs[j] < confidence:
            break
        out.append(drafts[j])
        if confidence > 0:
            chain *= probs[j]
            if chain < confidence:
                break
    return out


def decode_draft(run: Run, last_hidden: torch.Tensor, pending: int, absorbed: int, position: int) -> list[int]:
    """TensorFold's own ``decode.draft`` from the same state, on an Engine holding the exporter's buffers."""

    def step(rank: int) -> list[int]:
        e = object.__new__(run.decode.Engine)
        e.w, e.st, e.buf, e.mbuf = run.ws[rank], run.sts[rank], run.bufs[rank], run.mbufs[rank]
        e.graphs = e.constraint = e.window = None
        e.draft_n = e.w.head.n
        e.replays = {"main": 0, "sparse": 0, "mtp": 0, "sparse_mtp": 0, "eager": 0}
        e.st.set_mtp_len(absorbed)
        e.st.mtp_drafted = 0
        return run.decode.draft(e, last_hidden, [pending], position, MTP_DEPTH, None, MTP_CONFIDENCE)

    got = run.both(step)
    if got[0] != got[1]:
        raise RuntimeError(f"decode.draft: the ranks drafted {got[0]} and {got[1]}")
    return got[0]


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[dict]:
    """Per captured point, in forward order: whether two runs hold the same bits, and where they first part."""

    out = []
    for name, x in a.items():
        y = b.get(name)
        if y is None:
            out.append({"name": name, "equal": False, "why": "missing"})
            continue
        if x.shape != y.shape or x.dtype != y.dtype:
            out.append({"name": name, "equal": False, "why": f"{tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}"})
            continue
        diff = bits(x) != bits(y)
        if not bool(diff.any()):
            out.append({"name": name, "equal": True})
            continue
        rows = diff.reshape(diff.shape[0], -1).any(dim=1).nonzero().flatten().tolist()
        delta = (x.double() - y.double()).abs()
        out.append({"name": name, "equal": False, "rows": rows, "values": int(diff.sum()),
                    "max_abs": float(delta.max()), "max_ref": float(x.double().abs().max())})
    return out


def versus(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], per: int) -> dict:
    points = compare(a, b)
    differing = [p for p in points if not p["equal"]]
    return {"rows_per_forward": per, "points": len(points), "differing": len(differing),
            "first": differing[0] if differing else None, "all": differing}


def shapes(rec: dict[str, torch.Tensor]) -> dict:
    return {n: [str(t.dtype).removeprefix("torch."), list(t.shape)] for n, t in rec.items()}


def nonfinite(rec: dict[str, torch.Tensor]) -> dict:
    return {n: int((~torch.isfinite(t.float())).sum()) for n, t in rec.items()
            if t.is_floating_point() and not bool(torch.isfinite(t.float()).all())}


def save(path: Path, ids: list[int], rec: dict[str, torch.Tensor], provenance: dict) -> dict:
    from safetensors.torch import save_file

    tensors = {"ids": torch.tensor(ids, dtype=torch.int32)}
    tensors.update({name: t.contiguous() for name, t in rec.items()})
    save_file(tensors, str(path), metadata={"provenance": json.dumps(provenance)})
    return {"path": str(path), "bytes": path.stat().st_size, "tensors": len(tensors)}


def argmax(rec: dict[str, torch.Tensor], name: str = "logits") -> list[int]:
    """Each row's first largest logit over both ranks' halves of the vocabulary."""

    return torch.cat([rec[f"{name}.r{r}"] for r in range(WORLD)], dim=1).float().argmax(dim=-1).tolist()


STATE = {"state.kda.rec": 0, "state.kda.conv": 0, "state.dsa.latents": 1}   # the axis a piece's layers lie on


def extend(whole: dict[str, torch.Tensor], part: dict[str, torch.Tensor]) -> None:
    """A piece's record after those before it: its layers' points, the state by layer, the model's end again."""

    for name, t in part.items():
        axis = STATE.get(name.removesuffix(".r0").removesuffix(".r1"))
        before = whole.pop(name, None)
        whole[name] = t if axis is None or before is None else torch.cat([before, t], dim=axis)


class Case:
    """One prompt over the pieces: one window, serial steps and the serial steps again, each piece fed the
    streams the piece before it left in that same run."""

    def __init__(self, name: str, ids: list[int]) -> None:
        self.name, self.ids = name, ids
        self.recs: dict[str, dict] = {"window": {}, "serial": {}, "again": {}}
        self.differ: dict[str, list] = {"window": [], "serial": []}
        self.pending: list[int] = []

    def piece(self, run: Run, last: bool) -> None:
        """The serial steps go second: their state is what ``pending`` and the MTP layer follow."""

        dev = run.ws[0].device
        for tag in ("window", "serial", "again"):
            before = self.recs[tag]
            feed = before["final.h"].to(dev) if before else None
            rec, differ = run_windows(run, self.ids, len(self.ids) if tag == "window" else 1, feed)
            extend(before, rec)
            if tag in self.differ:
                self.differ[tag] += differ
            if tag == "serial" and last:
                R = len(self.ids)
                self.pending = run.both(
                    lambda rank: run.decode.sample_rows(run.ws[rank], run.bufs[rank].logits[:1], [R], None)[0])

    def finish(self, out_dir: Path, provenance: dict, text) -> dict:
        name, ids, R = self.name, self.ids, len(self.ids)
        window, serial, again = (self.recs[tag] for tag in ("window", "serial", "again"))
        pending, s_differ, w_differ = self.pending, self.differ["serial"], self.differ["window"]
        case: dict = {"ids": ids, "text": text(ids), "pending": pending[0], "pending_text": text([pending[0]]),
                      "ranks_chose_alike": pending[0] == pending[1],
                      "ranks_differ": {"serial": s_differ, "window": w_differ},
                      "serial_vs_window": versus(serial, window, R),
                      "serial_vs_serial_again": versus(serial, again, 1),
                      "nonfinite": nonfinite(serial),
                      "argmax": {"serial": argmax(serial), "window": argmax(window)},
                      "shapes": shapes(serial), "files": {}}
        for tag, rec, per in (("serial", serial, 1), ("window", window, R)):
            prov = dict(provenance, mode=tag, ids=ids, rows_per_forward=per)
            case["files"][tag] = save(out_dir / f"{name}.{tag}.f32.safetensors", ids, rec, prov)
        v = case["serial_vs_window"]
        print(f"[export] {name}: {R} ids, serial vs one window: {v['differing']} of {v['points']} points differ"
              + (f", first {json.dumps(v['first'])}" if v["first"] else "")
              + f"; a repeated serial run differs at {case['serial_vs_serial_again']['differing']}; between the "
              f"ranks {len(s_differ) + len(w_differ)} replicated points differ; non-finite {case['nonfinite']}; "
              f"pending {pending[0]} {case['pending_text']!r}; files {json.dumps(case['files'])}", flush=True)
        return case


def against(out_dir: Path, other: Path, stem: str) -> dict:
    """Every tensor file of a case against the file of that name in ``other``, bit for bit."""

    from safetensors.torch import load_file

    out: dict = {}
    for path in sorted(out_dir.glob(stem + ".*.safetensors")):
        if not (other / path.name).exists():
            out[path.name] = "no such file there"
            continue
        mine, theirs = load_file(str(path)), load_file(str(other / path.name))
        v = versus(theirs, mine, 0)
        out[path.name] = {"points": v["points"], "differing": v["differing"], "first": v["first"],
                          "missing": sorted(set(theirs) - set(mine)), "new": sorted(set(mine) - set(theirs))}
        print(f"[export] {path.name} against {other}: {json.dumps(out[path.name])}", flush=True)
    return out


def export_mtp_case(run: Run, name: str, ids: list[int], serial: dict, pending: int, depth: int, out_dir: Path,
                    provenance: dict, text) -> dict:
    """The MTP layer over the prompt's rows, as one window and as serial steps, then a chain of drafts."""

    R = len(ids)
    dev = run.ws[0].device
    hidden = serial["final.normed"].to(dev)
    nxt = ids[1:] + [pending]
    window, w_differ = run_mtp(run, hidden, nxt, R)
    rows, s_differ = run_mtp(run, hidden, nxt, 1)
    kept = decode_draft(run, hidden[R - 1:R], pending, R - 1, R)
    chain, c_differ, drafts, probs = run_chain(run, hidden[R - 1:R], pending, R - 1, R, depth)
    keeps = rule_keeps(drafts, probs, MTP_DEPTH, MTP_CONFIDENCE)
    case: dict = {"ids": ids, "next": nxt, "pending": pending,
                  "ranks_differ": {"serial": s_differ, "window": w_differ, "chain": c_differ},
                  "serial_vs_window": versus(rows, window, R),
                  "chain_row0_vs_serial_last_row": versus({n: t[:1] for n, t in chain.items() if n in rows
                                                           and n != "mtp.cache.latents"},
                                                          {n: t[R - 1:R] for n, t in rows.items()}, 1),
                  "nonfinite": nonfinite(rows),
                  "argmax": {"draft_head": argmax(rows, "mtp.logits"), "bf16_head": argmax(rows, "mtp.logits_bf16")},
                  "chain": {"depth": depth, "drafts": drafts, "drafts_text": [text([d]) for d in drafts],
                            "probs": probs, "argmax_bf16_head": argmax(chain, "mtp.logits_bf16"),
                            "policy": {"depth": MTP_DEPTH, "confidence": MTP_CONFIDENCE},
                            "rule_keeps": keeps, "decode_draft": kept, "decode_draft_is_rule_keeps": kept == keeps},
                  "shapes": {"serial": shapes(rows), "chain": shapes(chain)}, "files": {}}
    for tag, rec, per in (("serial", rows, 1), ("window", window, R), ("chain", chain, 1)):
        prov = dict(provenance, mode="mtp." + tag, ids=ids, next=nxt, rows_per_forward=per)
        if tag == "chain":
            prov.update(absorbed=R - 1, drafts=drafts)
        case["files"][tag] = save(out_dir / f"{name}.mtp.{tag}.safetensors", ids, rec, prov)
    v = case["serial_vs_window"]
    print(f"[export] {name}.mtp: {R} rows, serial vs one window: {v['differing']} of {v['points']} points differ"
          + (f", first {json.dumps(v['first'])}" if v["first"] else "")
          + f"; chain {drafts} {case['chain']['drafts_text']} probs {[round(p, 4) for p in probs]}; the rule keeps "
          f"{keeps}, decode.draft returned {kept}; files {json.dumps(case['files'])}", flush=True)
    return case


# -- load ----------------------------------------------------------------------------------------------
def load_ranks(weights, split, model_dir: Path, first: int, end: int, mtp: bool) -> tuple[list, int]:
    """Both ranks' shares of layers ``first`` .. ``end - 1`` (and the MTP layer), through ``weights.load`` itself,
    which is shown a model of those layers alone; each ``LayerW.index`` is then the model's again."""

    read = weights.Config.__dict__["read"]
    full = weights.Config.read(model_dir)
    count = end - first

    def shortened(cls, directory):
        got = read.__func__(cls, directory)
        return replace(got, layers=count, kinds=got.kinds[first:end], mlp_kinds=got.mlp_kinds[first:end])

    stem = f"{weights.PREFIX}layers."

    def actual(name: str) -> str:
        """The loader's layer i is the pack's ``first + i``; the one after its last is the MTP layer."""

        if not name.startswith(stem):
            return name
        index, _, rest = name[len(stem):].partition(".")
        return f"{stem}{full.layers if int(index) == count else first + int(index)}.{rest}"

    class Renamed(split.RankReader):
        """The pack's layers under the names the shortened config gives them."""

        def get(self, name: str):
            return super().get(actual(name))

        def prefetch(self, names, device=None) -> None:
            super().prefetch([actual(n) for n in names], device)

    reader = split.RankReader
    weights.Config.read, split.RankReader = classmethod(shortened), Renamed
    try:
        ws = [weights.load(model_dir, rank=rank, mtp=mtp) for rank in range(WORLD)]
    finally:
        weights.Config.read, split.RankReader = read, reader
    for w in ws:
        for i, layer in enumerate(w.layers):
            layer.index = first + i
        if w.mtp is not None:
            w.mtp.layer.index = full.layers
    if bool((ws[0].embed.view(torch.int16) == ws[1].embed.view(torch.int16)).all()):
        ws[1].embed = ws[0].embed            # a replicated tensor: one copy serves both threads
    torch.cuda.empty_cache()
    return ws, full.layers


def revision(model_dir: Path) -> str:
    """A Hugging Face cache path's snapshot hash, else the directory's name."""

    parts = model_dir.resolve().parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts[:-1] else model_dir.name


def memory() -> dict:
    """Device and host memory now, in GiB (on a Spark they are the same pool)."""

    free, total = torch.cuda.mem_get_info()
    rss = 0
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    g = 2**30
    return {"torch_allocated_gib": round(torch.cuda.memory_allocated() / g, 2),
            "torch_reserved_gib": round(torch.cuda.memory_reserved() / g, 2),
            "torch_peak_gib": round(torch.cuda.max_memory_allocated() / g, 2),
            "device_used_gib": round((total - free) / g, 2), "device_total_gib": round(total / g, 2),
            "process_rss_gib": round(rss / g, 2)}


# -- main ----------------------------------------------------------------------------------------------
def parse_case(text: str) -> tuple[str, list[int]]:
    name, _, ids = text.partition("=")
    got = [int(x) for x in ids.replace(",", " ").split()]
    if not name or not got:
        raise argparse.ArgumentTypeError(f"{text!r}: expected NAME=ID,ID,...")
    return name, got


def export_index(run: Run, fwd, out_dir: Path, tokens: int) -> None:
    """One DSA layer's pools and one row's selected list, past the dense limit, for the host model.

    The cases above stay on the dense path. This builds its own state (``long_context``) on the weights already
    loaded and walks ``tokens`` in the case's window, both ranks, then keeps rank 0's indexer rows.
    """

    from safetensors.torch import save_file
    from tensorfold.families.glm5_next.cuda import sparse

    ws = run.ws
    for w in ws:
        w.meta["long_context"] = True
    c = ws[0].cfg
    layer = next((one for one in ws[0].layers if one.kind == "dsa"), None)
    if layer is None or layer.dsa.index is None:
        raise SystemExit("--index: the loaded layers have no DSA indexer")
    rows = run.bufs[0].rows
    if tokens <= 0:
        tokens = ((c.dense_limit // rows) + 1) * rows
    if tokens > run.sts[0].capacity or tokens <= c.dense_limit:
        raise SystemExit(f"--index-tokens {tokens}: past the dense limit {c.dense_limit} and within the "
                         f"capacity {run.sts[0].capacity}")
    di = None
    sts = [fwd.State(w, run.sts[0].capacity, rows) for w in ws]
    if sts[0].index is None:
        raise SystemExit("--index: the state has no indexer caches")
    di = sts[0].dsa_index[layer.index]
    held: dict[str, torch.Tensor] = {}
    pos = 0
    while pos < tokens:
        n = min(rows, tokens - pos)
        window = [(pos + i) % 251 + 1 for i in range(n)]

        def step(rank: int, n: int = n, pos: int = pos, window: list[int] = window) -> None:
            w, st, b = ws[rank], sts[rank], run.bufs[rank]
            fwd.forward(w, st, b, window, logits=False)
            if rank == 0 and pos == 0:
                held["raw"] = b.ikr[:1, :c.index_dim].detach().cpu()
                held["key"] = st.index[di][0][:1].detach().cpu()
            if rank == 0 and pos + n == tokens:
                held["qi"] = b.qi[n - 1].detach().cpu()
                held["weights"] = b.ikr[n - 1, c.index_dim:].detach().cpu()
                held["qi_dev"] = b.qi[n - 1:n].clone()
                held["w_dev"] = b.ikr[n - 1:n, c.index_dim:].clone()
            fwd.commit(w, st, b, n, n)

        run.both(step)
        pos += n
    position = tokens - 1
    visible = (position + 1) // c.kpool
    ik, ig, pk = sts[0].index[di]
    other = sts[1].index[di][0][:tokens]
    ranks_equal = torch.equal(ik[:tokens], other)
    kept = c.index_topk // c.kpool
    dev = held["qi_dev"].device
    pos_dev = torch.tensor([position], dtype=torch.int32, device=dev)
    selected, counts = sparse.select_tokens(held["qi_dev"], held["w_dev"], pk, position, 1, pk.shape[0] - 2, pos_dev)
    flat = selected[0, :kept * c.kpool].tolist()
    pool_ids = [flat[i * c.kpool] // c.kpool for i in range(kept)]
    ix = layer.dsa.index
    tensors = {
        "pool.keys": ik[:8].contiguous(),
        "pool.gates": ig[:8].contiguous(),
        "pool.ape": ix.ape.contiguous(),
        "pool.pooled": pk[:2].contiguous(),
        "norm.raw": held["raw"].contiguous(),
        "norm.weight": ix.ln_w.contiguous().cpu(),
        "norm.bias": ix.ln_b.contiguous().cpu(),
        "norm.key": held["key"].contiguous(),
        "row.qi": held["qi"].view(c.index_heads, c.index_dim).contiguous(),
        "row.weights": held["weights"].contiguous(),
        "row.pooled": pk[:visible].contiguous().cpu(),
    }
    tensors = {name: value.cpu().contiguous() for name, value in tensors.items()}
    save_file(tensors, str(out_dir / "index.safetensors"))
    record = {
        "layer": layer.index,
        "dense_limit": c.dense_limit,
        "tokens": tokens,
        "ranks_equal": bool(ranks_equal),
        "rows": [{
            "position": position,
            "kept": kept,
            "pool": c.kpool,
            "pools": pool_ids,
            "count": int(counts[0]),
            "qi": "row.qi",
            "weights": "row.weights",
            "pooled": "row.pooled",
        }],
    }
    (out_dir / "index.json").write_text(json.dumps(record))
    print(f"[export] index: layer {layer.index}, position {position}, {kept} pools, count {int(counts[0])}, "
          f"ranks equal {bool(ranks_equal)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--case", type=parse_case, action="append", required=True, metavar="NAME=ID,ID,...",
                    help="a prompt's token ids; repeat for more cases")
    ap.add_argument("--out-dir", type=Path, required=True, help="where NAME.*.safetensors and NAME.*.json go")
    ap.add_argument("--layers", type=int, default=None,
                    help="the first N layers only (4: three KDA layers and the first DSA and MoE layer); "
                         "without it the whole model, which needs --pieces")
    ap.add_argument("--pieces", type=int, default=None, metavar="P",
                    help="load, run and free P layers at a time, each piece fed the streams of the one before")
    ap.add_argument("--taps", default="", metavar="L,L,...",
                    help="the layers after which the streams' mean is exported as taps.L (DFlash2: 5,14,24,33,42)")
    ap.add_argument("--against", type=Path, default=None, metavar="DIR",
                    help="compare every file written with the file of its name in DIR, bit for bit")
    ap.add_argument("--mtp", action="store_true",
                    help="load the MTP layer too and write NAME.mtp.serial, .window, .chain and NAME.mtp.json")
    ap.add_argument("--mtp-depth", type=int, default=6, help="chain steps exported (TensorFold drafts at most 7)")
    ap.add_argument("--capacity", type=int, default=2560, help="cache slots (the engine's while DSA stays dense)")
    ap.add_argument("--index", action="store_true",
                    help="after the cases, one prompt past the dense limit: index.json and index.safetensors")
    ap.add_argument("--index-tokens", type=int, default=0,
                    help="tokens of that prompt (0: the dense limit rounded up to a window)")
    ap.add_argument("--tensorfold-commit", default=None, metavar="HASH",
                    help="the commit of the TensorFold archive the package is imported from: recorded as stated")
    args = ap.parse_args()

    import tensorfold
    import triton
    from tensorfold.cuda import build
    from tensorfold.families.glm5_next.cuda import decode, glue, latent, split, weights
    from tensorfold.families.glm5_next.cuda import forward as fwd
    from tensorfold.families.glm5_next.cuda import mtp as mtp_mod        # bound to forward.py before any hook

    if tensorfold.__version__ != PINNED:
        print(f"[export] TensorFold {tensorfold.__version__}, written against {PINNED}: the hooks may not fit",
              file=sys.stderr)
    if not latent.ENABLED:
        raise SystemExit("TF_GLM_LATENT=0: the exporter reads the latent cache's buffers")
    build.refuse_old_gpu()
    model_dir = args.model_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = max(8, *(len(ids) for _, ids in args.case))
    if rows > 63:
        raise SystemExit(f"a case of {rows} ids: the decode buffers this exports on take windows under 64 rows")

    def text(ids: list[int]) -> str | None:
        """The ids as text, for a reader of the JSON (nothing depends on it)."""

        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(str(model_dir / "tokenizer.json")).decode(ids, skip_special_tokens=False)
        except Exception:                    # no tokenizers package, or no tokenizer beside the weights
            return None

    full = weights.Config.read(model_dir).layers
    layers = full if args.layers is None else args.layers
    per = layers if args.pieces is None else args.pieces
    if not 1 <= layers <= full or per < 1 or (layers == full and args.pieces is None):
        raise SystemExit(f"--layers {layers} of {full}, --pieces {args.pieces}: the whole model does not fit one "
                         "Spark at once and is exported in pieces")
    pieces = [(a, min(a + per, layers)) for a in range(0, layers, per)]
    taps = tuple(int(x) for x in args.taps.split(",") if x)
    if any(not 0 <= layer < layers for layer in taps):
        raise SystemExit(f"--taps {args.taps!r}: layers of the {layers} exported")

    torch.cuda.reset_peak_memory_stats()
    provenance = {
        "tensorfold_version": tensorfold.__version__,
        "tensorfold_commit": args.tensorfold_commit or "unknown",
        "tensorfold_commit_source": f"stated with --tensorfold-commit: {Path(tensorfold.__file__).resolve()} is an "
                                    "archive without .git",
        "checkpoint": str(model_dir), "checkpoint_revision": revision(model_dir),
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "layers": layers, "model_layers": full, "pieces": pieces, "taps": list(taps),
        "mtp_layer": f"layers.{full}" if args.mtp else None,
        "truncation": None if layers == full else
                      f"the first {layers} of {full} layers: final.*, logits.* and the MTP layer's "
                      f"mtp.hidden are stream_mean, the final norm and lm_head on layer {layers - 1}'s "
                      "streams, not GLM's final row",
        "ranks": "both ranks in this process, a thread each, w.comm a barrier (ThreadComm) in NCCL's place",
        "path": "forward.forward + forward.commit on forward.Buffers (decode path), DSA dense on the latent "
                "cache, no indexer, no drafts, text only"
                + ("" if len(pieces) == 1 else "; in pieces, each from an empty state on the streams of the one "
                   "before (final.h over glue.embed's output)"),
    }
    report: dict = {"provenance": provenance, "loads": []}
    cases = [Case(name, list(ids)) for name, ids in args.case]
    install(fwd, glue, mtp_mod)
    kinds: list = []
    for first, end in pieces:
        last = end == layers
        before = memory()
        t0 = time.time()
        ws, _ = load_ranks(weights, split, model_dir, first, end, args.mtp and last)
        torch.cuda.synchronize()
        if args.mtp and last and ws[0].mtp is None:
            raise SystemExit("--mtp: this checkpoint has no MTP layer")
        cfg = ws[0].cfg
        kinds += list(zip(cfg.kinds, cfg.mlp_kinds))
        load = {"layers": [first, end], "seconds": round(time.time() - t0, 1),
                "weights_gib": [round(w.nbytes() / 2**30, 2) for w in ws],
                "embedding_shared": ws[1].embed is ws[0].embed, "quant": cfg.quant,
                "head": [type(w.head).__name__ for w in ws],
                "draft_head": [None if w.draft_head is None else type(w.draft_head).__name__ for w in ws],
                "before": before, "after": memory()}
        run = Run(fwd, mtp_mod, decode, ws, rows, args.capacity, tuple(x for x in taps if first <= x < end))
        t1 = time.time()
        for case in cases:
            case.piece(run, last)
        load.update(run_seconds=round(time.time() - t1, 1), gathers=run.hub.gathers, memory=memory())
        report["loads"].append(load)
        print(f"[export] layers {first} .. {end - 1}: {json.dumps(load)}", flush=True)
        if not last:
            del run, ws
            gc.collect()
            torch.cuda.empty_cache()
    provenance["layer_kinds"] = kinds
    report["load"], report["buffers"] = report["loads"][-1], {"rows": rows, "capacity": args.capacity}
    for one in cases:
        name, ids = one.name, one.ids
        t1 = time.time()
        case = one.finish(args.out_dir, provenance, text)
        serial = one.recs["serial"]
        if args.mtp:
            t2 = time.time()
            mtp_case = export_mtp_case(run, name, ids, serial, case["pending"], args.mtp_depth, args.out_dir,
                                       provenance, text)
            out = dict(report, **mtp_case, seconds=round(time.time() - t2, 1), gathers=run.hub.gathers,
                       memory=memory())
            (args.out_dir / f"{name}.mtp.json").write_text(json.dumps(out, indent=1))
        if args.against is not None:
            case["against"] = against(args.out_dir, args.against, name)
        out = dict(report, **case, seconds=round(time.time() - t1, 1), gathers=run.hub.gathers, memory=memory())
        (args.out_dir / f"{name}.f32.json").write_text(json.dumps(out, indent=1))
    if args.index:
        export_index(run, fwd, args.out_dir, args.index_tokens)
    print(f"[export] done: {len(pieces)} pieces; memory {json.dumps(memory())}", flush=True)
    return 0


if __name__ == "__main__":
    import glm5_export                       # one module for the drafter's exporter too (LOCAL, the hooks)

    sys.exit(glm5_export.main())
