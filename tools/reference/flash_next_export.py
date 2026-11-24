#!/usr/bin/env python3
"""Reference tensors of Flash Next's decode-path forward, exported from TensorFold itself.

A test fixture: it runs inside the TensorFold container (PyTorch, Triton), never at zspark's runtime. It loads a
checkpoint through TensorFold's own loader, runs token ids through TensorFold's decode-path forward as serial
one-row steps and again as windows, and writes what each block produced as named tensors in safetensors files
that zspark's tests read.

Pinned to TensorFold 0.6.5 (commit 609ca41). It relies on these internals of
``tensorfold.families.qwen4_exp.cuda``, all read or wrapped from here and none edited:

- ``weights.load(model_dir, "cuda", mtp=False, draft_vocab=None)`` and ``weight_types.Config.read`` (wrapped by
  ``--layers`` to shorten ``layers``, ``layer_types`` and ``ple_layers``; the loader takes ``range(cfg.layers)``);
- ``state.Buffers(w, rows, capacity, moe_prefill=...)`` and ``state.State(w, capacity, max_rows, "bf16")``, built
  the way ``decode.Engine.__init__`` builds its decode buffers and state;
- ``forward.forward(w, st, b, tokens)`` and ``forward.commit(w, st, b, R, keep)``;
- the module-level functions ``forward.compute`` calls by name, which are wrapped to copy the buffers they leave:
  ``_embed``, ``layer_forward``, ``_writeback``, ``ple_block``, ``hc_block``, ``gdn_block``, ``attn_block``,
  ``moe_block``, ``finish``;
- the buffers those functions write: ``b.h``, ``b.mixed``, ``b.inj_a``, ``b.inj_m``, ``b.proj[li]``, ``b.gout``,
  ``b.pa``, ``b.q``, ``b.iq``, ``b.gated``, ``b.branch``, ``b.moe.logits``, ``b.moe.pick``, ``b.moe.wts``,
  ``b.moe.y``, ``b.streams``, ``b.logits``; and ``State.lin_index``;
- with ``--engine``: ``tensorfold.families.qwen4_exp.cuda_engine(dir, no_drafts=True)`` and its ``generate``;
- with ``--mtp`` (the multi-token-prediction head, below): ``weights.load(..., mtp=True, draft_vocab="default")``,
  ``mtp.mtp_forward(w, st, b, next_tokens, streams, last_only=...)``, the names ``mtp.py`` imported from
  ``forward.py`` and calls as its own globals (``_embed``, ``_mm``, ``layer_forward``, ``finish``),
  ``glue.add_streams``, ``State.set_mtp_len``, ``State.mtp_drafted``, ``State.mtp_kc`` and ``State.mtp_ikc``,
  ``Weights.mtp`` (``fc_e``, ``fc_h``, ``layer`` whose index is -1), ``Weights.draft_head`` and
  ``Weights.draft_ids``; and with ``--engine`` too: ``cuda_engine(dir)`` with drafts, ``decode.mtp_decode``,
  ``decode.draft``, ``decode.mtp_forward``, ``decode.Engine.mtp_forward`` and ``decode.Engine.sample_draft``,
  wrapped from outside, and ``FlashNextEngine.cache`` (emptied so every run prefills its whole prompt);
- with ``--sampled``: ``tensorfold.engine.exact_sampling.Sampling``, ``cuda_engine(dir)`` with drafts and its
  ``generate(prompt, count, sampling, on_tokens, draft=..., stop_eos=False)``, ``decode.Engine.sample`` (wrapped
  to record the positions it is called with), and ``tensorfold.cuda.sampling.sample_rows`` on the exporter's own
  decode-path steps;
- on an EXL3 pack (``tensorfold.cuda.exl3.format.is_exl3``): ``engine.build_kernels(exl3=True)``, and nothing
  else of its own: ``weights.load`` hands the pack to ``exl3.load``, which fills the same dataclasses, and the
  functions above pick the pack's matrices by type (``exl3_mm.X3``, ``F16``, ``Stack``; ``forward._exl3_moe``,
  whose slots are the expert kernel's fp32 scratch on the decode path, whatever ``--moe-slots`` says);
- with ``--prefill``: ``state.Buffers(w, decode.PREFILL_ROWS, capacity, prefill=True)``, built as
  ``decode.Engine.pbuf``, and ``forward.forward`` and ``forward.commit`` on it, the prompt as one chunk; on those
  buffers ``gdn_block`` writes ``b.proj[0]`` and ``finish`` heads the last row only;
- with ``--score``: ``forward.stage`` and ``forward.compute(..., ends=every row)`` on prompt buffers of
  ``state.ENDS`` rows, the way a prompt pass heads several prompts' last rows.

Tensor names (rows on the first axis; R rows, D = hidden, S = streams):

    ids                              I32  [R]
    embed.h                          BF16 [R, S*D]   the streams after embedding
    layers.{i}.h_in                  BF16 [R, S*D]   the streams after the previous layer's write-back
    layers.{i}.h_after_ple           BF16 [R, S*D]   PLE layer only: after the n-gram branch was added
    layers.{i}.attn_hc.mixed         BF16 [R, D]
    layers.{i}.attn_hc.inject        BF16 [R, S]
    layers.{i}.block.projection      BF16 [R, *]     DeltaNet: b.proj[li]; attention: b.pa
    layers.{i}.block.q               BF16 [R, heads, head_dim]        attention only: queries after norm and RoPE
    layers.{i}.block.index_q         BF16 [R, index_heads, index_dim] attention only: the indexer's queries
    layers.{i}.block.out             BF16 [R, *]     DeltaNet: b.gout; attention: b.gated
    layers.{i}.block.branch          BF16 [R, D]     the output projection
    layers.{i}.h_mid                 BF16 [R, S*D]   the streams after the block's branch was written back
    layers.{i}.mlp_hc.mixed          BF16 [R, D]
    layers.{i}.mlp_hc.inject         BF16 [R, S]
    layers.{i}.moe.logits            F32  [R, experts + 1]   the last column is the shared expert's gate logit
    layers.{i}.moe.picks             I32  [R, top_k + 1]     the last slot is the shared expert (id = experts)
    layers.{i}.moe.weights           F32  [R, top_k + 1]
    layers.{i}.moe.y                 F32 or BF16 [R, top_k + 1, D]   see --moe-slots
    final.h                          BF16 [R, S*D]   after the last write-back
    final.mixed                      BF16 [R, D]
    logits                           BF16 [R, vocab]

TensorFold's prompt path (``--prefill``). Not a reference: a measurement of how far TensorFold's own prompt path
is from its own decode path, for a checkpoint whose prompt kernels give other bits (an EXL3 pack: a prompt
chunk decodes each trellis projection once and multiplies with another GEMM, ``exl3_mm.X3.prefill``, which is
where the two first part, at layer 0's ``block.projection``; it also rounds the experts' slots to bf16 and runs
attention and DeltaNet on the prompt kernels). zspark runs prompts through its decode kernels, so it is compared
with ``NAME.serial``; ``NAME.prefill`` says how much of a difference from TensorFold's served replies is
TensorFold's own. The prompt runs as one chunk from an empty
state on prompt buffers, and ``NAME.prefill.safetensors`` holds the same names with two differences:
``final.mixed`` and ``logits`` have one row, the prompt's last (a prompt pass heads only that row), and
``layers.{i}.moe.y`` is BF16. ``NAME.json`` gets ``prefill_vs_serial``: every point's distance from the serial
steps in forward order (``all``: equal bits, else the largest difference, the largest reference magnitude and
the difference's RMS over the reference's; for ``moe.picks`` the rows whose picks differ and those whose set of
experts does; ``moe.y`` is compared slot by slot, so a row whose picks come in another order is far there), the
first point that differs, and the worst of each kind of point over the layers (``by_kind``). The distances
accumulate: a point's input already differs by what the points before it did.

A score (``--score FILE``). The ids in FILE (whitespace-separated) run from an empty state through the decode
path in windows of 16 rows, and each token after the first is scored against the logits of the row before it:
the mean negative log-likelihood and the share ranked first, over all tokens and over the second half, the way
``zspark score`` computes them (bf16 logits read as fp32, the sum of exponentials in fp64, first = the target's
logit equals the row's maximum). ``score.json`` holds them (``decode_path``) and, beside them, the same ids
through TensorFold's prompt kernels in chunks of 16 rows with every row headed (``prompt_path``), with how far
each row's ``final.h`` and logits are from the decode path's (``prompt_vs_decode``). It runs before any hook is
installed. ``--score-each`` adds ``by_token``: each token's loss on both paths (``zspark score DIR each`` prints
its own) and each row's two distances, to pair two scores token by token (spec 079).

A row's close calls (``--candidates NAME=ID,ID,...``). Where a greedy reply parts from TensorFold's, the case to
export is the prompt followed by TensorFold's reply up to the token before the parting: its last serial row is
the row that chose. ``NAME.json`` then gets ``last_row``: the row's dtype as ``decode.Engine.sample`` argmaxes
it (``b.logits``, bf16), torch's argmax (the first maximum), how many ids hold the maximum, the top five, and
each candidate's logit, bits and rank (the count of logits above it), with ``candidates_equal``. On an EXL3
pack ``unrounded`` holds the same from the head run once more on the row's ``final.mixed`` into an fp32 output
(``exl3.linear.Exl3Linear.__call__`` with ``out_dtype``; the kernel's own sums before its bf16 store), and
whether rounding that gives the row's bits.

The MTP head (``--mtp``). ``--mtp`` writes only ``NAME.mtp.*`` files and leaves the files above alone. The main
model runs the prompt as serial steps on the engine's decode buffers (bf16 slots); row i's ``final.h`` and the
token after row i are the head's inputs, the way TensorFold absorbs a prompt: rows 0 .. R-2 pair with ids[1:]
(``decode.prefill_chunk``) and the last row with the first sampled token (``decode.mtp_decode``'s first
``draft``), so the head's cache holds R entries once the first draft is scored. The head runs on buffers built as
``decode.Engine.mbuf`` (``state.Buffers(w, rows, capacity)``: the experts' decode tile, fp32 slots), from an empty
head cache, as R one-row absorbs (``NAME.mtp.serial``) and as one window (``NAME.mtp.window``). TensorFold itself
absorbs rows 0 .. R-2 on its prompt buffers (``Engine.pbuf``) from the prompt kernels' streams; ``NAME.mtp.json``
records how far that is from these files (``engine_vs_export``).

    ids                              I32  [R]        the prompt
    final.h                          BF16 [R, S*D]   the main model's streams, as in NAME.serial (b.streams)
    mtp.next                         I32  [R]        the token after each row (b.ids of the head's buffers)
    mtp.in                           BF16 [R, S*D]   the head's input streams (b.mtp_in), equal to final.h
    mtp.e                            BF16 [R, D]     the next tokens' embedding (b.mtp_e)
    mtp.e_normed                     BF16 [R, D]     after pre_fc_norm_embedding (b.mixed, fc_embedding's input)
    mtp.eo                           BF16 [R, D]     after fc_embedding (b.mtp_eo)
    mtp.hn                           BF16 [R, S*D]   the streams after pre_fc_norm_hidden (b.mtp_hn)
    mtp.hs                           BF16 [R, S*D]   after fc_hidden, a stream at a time (b.mtp_hs [R*S, D])
    mtp.h                            BF16 [R, S*D]   mtp.eo added to every stream of mtp.hs (b.h, glue.add_streams)
    mtp.layer.h_in                   BF16 [R, S*D]   b.h as the layer's first hyper-connection read it (= mtp.h)
    mtp.layer.attn_hc.mixed          BF16 [R, D]
    mtp.layer.attn_hc.inject         BF16 [R, S]
    mtp.layer.block.projection       BF16 [R, *]     b.pa
    mtp.layer.block.q                BF16 [R, heads, head_dim]
    mtp.layer.block.index_q          BF16 [R, index_heads, index_dim]
    mtp.layer.block.out              BF16 [R, *]     b.gated
    mtp.layer.block.branch           BF16 [R, D]
    mtp.layer.h_mid                  BF16 [R, S*D]
    mtp.layer.mlp_hc.mixed           BF16 [R, D]
    mtp.layer.mlp_hc.inject          BF16 [R, S]
    mtp.layer.moe.logits             F32  [R, experts + 1]
    mtp.layer.moe.picks              I32  [R, top_k + 1]
    mtp.layer.moe.weights            F32  [R, top_k + 1]
    mtp.layer.moe.y                  F32  [R, top_k + 1, D]   fp32 slots: the head's buffers use the decode tile
    mtp.streams                      BF16 [R, S*D]   the streams the head leaves (b.streams; a chained draft's input)
    mtp.mixed                        BF16 [R, D]     the head's mixer read-out (b.mixed)
    mtp.logits                       BF16 [R, vocab] the full head on every row (mtp_forward, last_only=False)
    mtp.cache.k, mtp.cache.v         BF16 [R, kv_heads, head_dim]   the head's attention cache (State.mtp_kc)
    mtp.cache.index_k                BF16 [R, index_dim]            its indexer keys (State.mtp_ikc)
    mtp.draft_ids                    I32  [V]        window and chain files: the draft vocabulary (Weights.draft_ids)
    mtp.draft_logits                 BF16 [1, V]     window file: the draft head on the last row (last_only=True);
                                                     column j is token mtp.draft_ids[j]

``NAME.mtp.chain`` holds the chain ``decode.draft`` runs after that absorb, greedy, with the same names and a
chain step on the first axis (DEPTH rows). Row 0 is the absorb's last row again (the main model's last streams and
the first sampled token); row j > 0 is ``mtp_forward([draft j-1], row j-1's mtp.streams)`` with the head's cache
one entry longer. ``mtp.draft_logits`` is [DEPTH, V] there, and two more tensors say what each row proposed:

    mtp.draft                        I32  [DEPTH]    the draft head's greedy token of each row
    mtp.prob                         F32  [DEPTH]    its probability at temperature 1 (Engine.sample_draft's rule)

The exporter chains all DEPTH rows; TensorFold stops before a draft under CONFIDENCE, which ``NAME.mtp.json``
marks (``chain.engine_keeps``). With ``--engine`` the JSON also holds ``engine_drafts``: ``decode.mtp_decode``'s
``DecodeResult`` for each ``--greedy`` case from TensorFold's serving engine (rounds, drafted, accepted, keeps,
widths, tokens, tok/s), the serial twin beside it, and each round's absorbed tokens, candidates and drafts.

A sampled reply (``--sampled NAME[:N]`` with ``--engine``, and ``--seed``, ``--temperature``, ``--top-k``,
``--top-p``, ``--min-p``). It writes only ``NAME.sampled.json``: the prompt's ids, the settings, and N tokens that
TensorFold's serving engine chose for them, drafted (``tokens``, what ``tensorfold serve`` decodes) and by its
serial twin, which are the same reply. Beside them, for telling an arithmetic's drift from a sampler's, the
exporter's own decode-path steps sampled with ``sample_rows`` and the same key, on bf16 expert slots and on fp32
ones (``decode_path``; the fp32 slots are what zspark ports, the engine's prompt kernels are what neither has).

The key. A token is drawn from ``exact_sampling.uniform(seed, position, token id)``, where ``position`` is the
index in the sequence of the token being chosen: ``decode.prefill`` samples the first reply token at
``len(prompt)``, ``serial_decode`` the next at ``st.pos + 1`` (``st.pos`` rows are committed and the row being run
is the token at ``st.pos``), and ``mtp_decode`` row r of a window at ``st.pos + 1 + r``. ``generate`` uses
``Sampling.seed`` as given; only the servers derive one (``seed_for(prompt)``) for a request without a seed. The
file records the positions ``Engine.sample`` was called with (``key.engine_positions``). zspark's
``Sampler.choose(logits, position, settings)`` takes the same position and seed (``draft.Decoder.prompt`` and
``round``, ``draft.serialReply``).

Each file's metadata holds ``provenance``, a JSON string: TensorFold's version and commit, the checkpoint's
revision, torch and triton versions, the run's mode, the ids and, with ``--greedy``, the continuation. Where
the TensorFold tree has no ``.git`` (a copy), ``--tensorfold-commit`` states the commit and
``tensorfold_commit_source`` says it was stated. For a
directory ``hybrid_prepare.py`` wrote, ``checkpoint_prepared`` holds its source, the source's revision and the
options; the tensor names and the hooks are the same, since TensorFold picks a linear's kernel from its tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PINNED = "0.6.5"


# -- capture -------------------------------------------------------------------------------------------
class Tap:
    """Wraps the functions ``forward.compute`` calls so each one's result is cloned as it is produced.

    TensorFold's buffers are reused by the next layer and the next step, so every capture is a clone taken right
    after the function that wrote it returns. ``rec`` is refilled by every forward: name -> tensor [R, ...] on
    the host, in the order the forward produced them.
    """

    NAMES = ("_embed", "layer_forward", "_writeback", "ple_block", "hc_block", "gdn_block", "attn_block",
             "moe_block", "finish")
    MTP_NAMES = ("_embed", "_mm", "layer_forward", "finish")     # mtp.py's own bindings of forward.py's functions

    def __init__(self, fwd) -> None:
        self.fwd = fwd
        self.rec: dict[str, torch.Tensor] = {}
        self.layer = None                    # the LayerW ``layer_forward`` is inside, None in ``finish``
        self.orig = {name: getattr(fwd, name) for name in self.NAMES}
        self.mtp_orig = None                 # (mtp module, its bindings, glue module, glue.add_streams)

    @staticmethod
    def base(layer) -> str:
        """A layer's name prefix: the MTP head's one layer is loaded with index -1."""

        return "mtp.layer" if layer.index < 0 else f"layers.{layer.index}"

    def put(self, name: str, t: torch.Tensor) -> None:
        self.rec[name] = t.detach().clone().cpu()

    def install(self) -> None:
        o, put, tap = self.orig, self.put, self

        def _embed(w, ids, copies, out):
            got = o["_embed"](w, ids, copies, out)
            put("embed.h", out)
            return got

        def layer_forward(layer, w, segs, b, R, pending, **kw):
            tap.layer = layer
            try:
                return o["layer_forward"](layer, w, segs, b, R, pending, **kw)
            finally:
                tap.layer = None

        def _writeback(h, b, R, c, pending):
            # called by the PLE layer (the pending MoE goes into the streams before the n-gram branch reads them)
            # and by ``finish`` (layer None; its streams are captured there)
            o["_writeback"](h, b, R, c, pending)
            if tap.layer is not None:
                put(f"{tap.base(tap.layer)}.h_in", h[:R])

        def ple_block(layer, w, segs, b, R):
            key = f"layers.{layer.index}.h_in"
            if key not in tap.rec:           # a PLE on the first layer: nothing was pending
                put(key, b.h[:R])
            o["ple_block"](layer, w, segs, b, R)
            put(f"layers.{layer.index}.h_after_ple", b.h[:R])

        def hc_block(hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=None, y=None, wts=None):
            # the write-back into h (in place), then the read-out, which only reads h
            o["hc_block"](hc, b, R, eps, streams, low, mode, inject_prev, inject_out, h, branch=branch, y=y, wts=wts)
            layer = tap.layer
            base = tap.base(layer)
            if hc is layer.attn_hc:
                side = "attn_hc"
                if base + ".h_in" not in tap.rec:
                    put(base + ".h_in", h[:R])
            else:
                side = "mlp_hc"
                put(base + ".h_mid", h[:R])
            put(f"{base}.{side}.mixed", b.mixed[:R])
            put(f"{base}.{side}.inject", inject_out[:R])

        def gdn_block(layer, w, segs, b, R, cuts=()):
            mode, branch = o["gdn_block"](layer, w, segs, b, R, cuts)
            if mode != 1:
                raise RuntimeError(f"layer {layer.index}: DeltaNet's output projection came back in mode {mode}")
            base = tap.base(layer) + ".block"
            li = 0 if b.prefill else segs[0][0].lin_index[layer.index]   # a prompt chunk commits layer by layer
            put(base + ".projection", b.proj[li, :R])
            put(base + ".out", b.gout[:R])
            put(base + ".branch", branch[:R])
            return mode, branch

        def attn_block(layer, w, segs, b, R, mtp, context=None):
            mode, branch = o["attn_block"](layer, w, segs, b, R, mtp, context)
            if mode != 1:
                raise RuntimeError(f"layer {layer.index}: attention's output projection came back in mode {mode}")
            base = tap.base(layer) + ".block"
            put(base + ".projection", b.pa[:R])
            put(base + ".q", b.q[:R])
            put(base + ".index_q", b.iq[:R])
            put(base + ".out", b.gated[:R])
            put(base + ".branch", branch[:R])
            return mode, branch

        def moe_block(layer, w, b, R):
            mode, y, wts = o["moe_block"](layer, w, b, R)
            if mode != 2:
                raise RuntimeError(f"layer {layer.index}: the MoE came back in mode {mode}")
            base = tap.base(layer) + ".moe"
            put(base + ".logits", b.moe.logits[:R])
            put(base + ".picks", b.moe.pick[:R])
            put(base + ".weights", wts[:R])
            put(base + ".y", y[:R])
            return mode, y, wts

        def finish(w, mixer, b, R, pending, logits=True, ends=()):
            out = o["finish"](w, mixer, b, R, pending, logits=logits, ends=ends)
            put("final.h", b.streams[:R])
            headed = max(1, len(ends)) if b.prefill else R       # a prompt pass mixes and heads its ends only
            put("final.mixed", b.mixed[:headed])
            if out is not None:
                put("logits", out[:headed])
            return out

        for name, fn in (("_embed", _embed), ("layer_forward", layer_forward), ("_writeback", _writeback),
                         ("ple_block", ple_block), ("hc_block", hc_block), ("gdn_block", gdn_block),
                         ("attn_block", attn_block), ("moe_block", moe_block), ("finish", finish)):
            setattr(self.fwd, name, fn)

    def install_mtp(self, mtp_mod, glue, w) -> None:
        """The head's points. ``mtp.py`` calls its own bindings of ``_embed``, ``_mm``, ``layer_forward`` and
        ``finish``; what those call inside ``forward.py`` (the layer's blocks) goes through ``install``'s hooks."""

        o = {name: getattr(mtp_mod, name) for name in self.MTP_NAMES}
        add = glue.add_streams
        self.mtp_orig = (mtp_mod, o, glue, add)
        put, tap, head, streams = self.put, self, w.mtp, w.cfg.streams

        def _embed(w, ids, copies, out):
            got = o["_embed"](w, ids, copies, out)
            put("mtp.e", out)
            return got

        def _mm(x, q, xs, out, b, **kw):
            got = o["_mm"](x, q, xs, out, b, **kw)
            if q is head.fc_e:
                put("mtp.e_normed", x)
                put("mtp.eo", out)
            elif q is head.fc_h:             # rows of one stream each: [R*S, D], kept as [R, S*D]
                rows = x.shape[0] // streams
                put("mtp.hn", x.reshape(rows, -1))
                put("mtp.hs", out.reshape(rows, -1))
            return got

        def add_streams(e, hs, out, n):
            add(e, hs, out, n)
            put("mtp.h", out)

        def layer_forward(layer, w, segs, b, R, pending, **kw):
            tap.layer = layer
            try:
                return o["layer_forward"](layer, w, segs, b, R, pending, **kw)
            finally:
                tap.layer = None

        def finish(w, mixer, b, R, pending, logits=True, ends=()):
            out = o["finish"](w, mixer, b, R, pending, logits=logits, ends=ends)
            put("mtp.streams", b.streams[:R])
            put("mtp.mixed", b.mixed[:1 if b.prefill else R])
            return out

        for name, fn in (("_embed", _embed), ("_mm", _mm), ("layer_forward", layer_forward), ("finish", finish)):
            setattr(mtp_mod, name, fn)
        glue.add_streams = add_streams

    def remove(self) -> None:
        for name, fn in self.orig.items():
            setattr(self.fwd, name, fn)
        if self.mtp_orig is not None:
            mtp_mod, o, glue, add = self.mtp_orig
            for name, fn in o.items():
                setattr(mtp_mod, name, fn)
            glue.add_streams = add
            self.mtp_orig = None

    def take(self) -> dict[str, torch.Tensor]:
        got, self.rec = self.rec, {}
        return got


# -- runs ----------------------------------------------------------------------------------------------
def run_windows(fwd, tap: Tap, w, st, b, ids: list[int], rows: int) -> dict[str, torch.Tensor]:
    """The ids from an empty state in windows of ``rows`` rows (1: serial steps), every row committed."""

    st.reset(w)
    parts: list[dict[str, torch.Tensor]] = []
    for at in range(0, len(ids), rows):
        chunk = ids[at:at + rows]
        fwd.forward(w, st, b, chunk)
        torch.cuda.synchronize()
        parts.append(tap.take())
        fwd.commit(w, st, b, len(chunk), len(chunk))
    return {name: torch.cat([p[name] for p in parts]) for name in parts[0]}


def run_prefill(fwd, tap: Tap, w, st, pb, ids: list[int]) -> dict[str, torch.Tensor]:
    """The ids from an empty state as one prompt chunk on prompt buffers (``decode.prefill_chunk``'s forward and
    commit, no MTP absorb): TensorFold's prompt path, whose last row alone is mixed and headed."""

    if len(ids) > pb.rows:
        raise ValueError(f"a prompt of {len(ids)} rows, prompt buffers of {pb.rows}")
    st.reset(w)
    fwd.forward(w, st, pb, ids)
    torch.cuda.synchronize()
    rec = tap.take()
    fwd.commit(w, st, pb, len(ids), len(ids))
    return rec


def distance(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """How far ``got`` is from ``ref`` by value, whatever their float types: ``closeness`` for tensors that may
    differ in dtype (the experts' slots are bf16 on the prompt path, fp32 on the decode path)."""

    if ref.shape != got.shape:
        return {"equal": False, "why": f"{tuple(ref.shape)} vs {tuple(got.shape)}"}
    if not ref.is_floating_point():                      # the experts picked: rows whose picks differ at all
        rows = (ref != got).reshape(ref.shape[0], -1).any(dim=1)
        same_set = (ref.sort(dim=-1).values == got.sort(dim=-1).values).reshape(ref.shape[0], -1).all(dim=1)
        return {"equal": not bool(rows.any()), "rows_differing": int(rows.sum()),
                "rows_with_other_experts": int((~same_set).sum()), "rows": int(ref.shape[0])}
    a, b = ref.double(), got.double()
    out = {"equal": ref.dtype == got.dtype and not bool((bits(ref) != bits(got)).any())}
    if ref.dtype != got.dtype:
        out["dtypes"] = [str(ref.dtype).removeprefix("torch."), str(got.dtype).removeprefix("torch.")]
    if not out["equal"]:
        out.update(max_abs=float((a - b).abs().max()), max_ref=float(a.abs().max()),
                   rms_diff_over_rms_ref=float((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt().clamp_min(1e-30)))
    return out


def prefill_distances(serial: dict[str, torch.Tensor], prefill: dict[str, torch.Tensor]) -> dict:
    """TensorFold's prompt path against its serial steps on the same rows, point by point in forward order."""

    points, by_kind = [], {}
    for name, ref in serial.items():
        if name not in prefill:
            points.append({"name": name, "missing": True})
            continue
        got = prefill[name]
        if name in ("final.mixed", "logits"):            # the prompt pass headed its last row only
            ref = ref[-got.shape[0]:]
        d = {"name": name, **distance(ref, got)}
        points.append(d)
        parts = name.split(".")
        layer = int(parts[1]) if parts[0] == "layers" else None
        kind = ".".join(parts[2:]) if layer is not None else name
        k = by_kind.setdefault(kind, {"points": 0, "equal": 0})
        k["points"] += 1
        k["equal"] += int(d["equal"])
        if "rows_differing" in d:
            k["rows_differing"] = k.get("rows_differing", 0) + d["rows_differing"]
            k["rows_with_other_experts"] = k.get("rows_with_other_experts", 0) + d["rows_with_other_experts"]
            k["rows"] = k.get("rows", 0) + d["rows"]
        elif not d["equal"] and "max_abs" in d:
            if d["rms_diff_over_rms_ref"] >= k.get("worst_rms_diff_over_rms_ref", -1.0):
                k["worst_rms_diff_over_rms_ref"], k["worst_rms_layer"] = d["rms_diff_over_rms_ref"], layer
            if d["max_abs"] >= k.get("worst_max_abs", -1.0):
                k["worst_max_abs"], k["worst_max_abs_layer"], k["max_ref_there"] = d["max_abs"], layer, d["max_ref"]
    differing = [p for p in points if not p.get("equal", False)]
    return {"note": "TensorFold's prompt path (one chunk on prompt buffers) against its serial steps (decode "
                    "buffers), the same rows from an empty state. Distances accumulate along the forward. "
                    "final.mixed and logits: the last row only. zspark runs prompts on the decode kernels, so its "
                    "reference is NAME.serial; this is how far TensorFold's own prompts are from that",
            "points": len(points), "differing": len(differing), "first": differing[0] if differing else None,
            "by_kind": by_kind, "all": points}


def greedy(fwd, tap: Tap, w, st, b, first: int, count: int) -> list[int]:
    """``count`` greedy tokens on the decode path from the committed state, as ``decode.serial_decode`` steps."""

    out = [first]
    while len(out) < count:
        logits = fwd.forward(w, st, b, [out[-1]])
        out.append(int(logits[:1].argmax(dim=-1)))      # sampling.sample_rows' rule at temperature 0
        fwd.commit(w, st, b, 1, 1)
        tap.take()
    return out


def standing(row: torch.Tensor, candidates: list[int], text) -> dict:
    """One row of logits as an argmax reads it: the first maximum, the ids that hold it, the top five, and where
    each candidate stands. Values are exact as fp64 whatever the row's dtype."""

    v = row.double()
    top = torch.topk(v, 5)
    best = float(v.max())

    def one(i: int) -> dict:
        return {"id": i, "logit": float(v[i]), "bits": int(bits(row)[i]), "rank": int((v > v[i]).sum()),
                "text": text([i])}

    return {"dtype": str(row.dtype).removeprefix("torch."), "argmax": int(row.argmax()),
            "at_maximum": (v == best).nonzero().flatten().tolist()[:16],
            "top": [dict(one(int(i)), logit=float(x)) for x, i in zip(top.values, top.indices)],
            "candidates": [one(i) for i in candidates],
            "candidates_equal": len({float(v[i]) for i in candidates}) == 1}


def last_row(w, serial: dict[str, torch.Tensor], ids: list[int], candidates: list[int], text) -> dict:
    """``--candidates``: the last serial row's logits as TensorFold argmaxes them, and on an EXL3 pack the head
    again on that row's ``final.mixed`` into an fp32 output."""

    row = serial["logits"][-1]
    out = {"row": len(ids) - 1, "position_chosen": len(ids),
           "argmax_rule": "decode.Engine.sample at temperature 0: logits.argmax(dim=-1) on b.logits as the head "
                          "wrote it, the first (lowest-id) maximum",
           **standing(row, candidates, text)}
    lin = getattr(w.head, "lin", None)                   # exl3_mm.X3: its Exl3Linear writes any float output
    if lin is None:
        out["unrounded"] = None
        return out
    x = serial["final.mixed"][-1:].to("cuda").contiguous()
    wide = lin(x, out_dtype=torch.float32)
    torch.cuda.synchronize()
    again = lin(x, out_dtype=torch.bfloat16)[0].cpu()    # the same call into bf16: is it the row the step wrote?
    wide = wide[0].cpu()
    out["unrounded"] = {
        "note": "w.head.lin on the row's final.mixed (bf16) with an fp32 output: the kernel's sums before the "
                "store that rounds them to bf16",
        "bf16_again_is_the_row": not bool((bits(again) != bits(row)).any()),
        "rounds_to_the_row": not bool((bits(wide.to(torch.bfloat16)) != bits(row)).any()),
        **standing(wide, candidates, text)}
    return out


def bits(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16)
    if t.dtype == torch.float32:
        return t.view(torch.int32)
    return t


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> list[dict]:
    """Per captured point, in forward order: whether two runs hold the same bits, and where they first part."""

    out = []
    for name, x in a.items():
        y = b[name]
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


def save(path: Path, ids: list[int], rec: dict[str, torch.Tensor], provenance: dict) -> dict:
    from safetensors.torch import save_file

    tensors = {"ids": torch.tensor(ids, dtype=torch.int32)}
    tensors.update({name: t.contiguous() for name, t in rec.items()})
    save_file(tensors, str(path), metadata={"provenance": json.dumps(provenance)})
    return {"path": str(path), "bytes": path.stat().st_size, "tensors": len(tensors)}


def export_case(fwd, tap: Tap, w, st, b, name: str, ids: list[int], widths: list[int], out_dir: Path, suffix: str,
                provenance: dict, greedy_tokens: int, engine_greedy: dict | None, text, pb=None,
                candidates: list[int] | None = None) -> dict:
    """One prompt: its serial steps, each window width, their comparison, the files and the continuation; with
    prompt buffers ``pb``, TensorFold's prompt path beside them."""

    case: dict = {"ids": ids, "text": text(ids)}
    windows = {}
    for n in widths:                                         # 0: the whole prompt as one window
        win = len(ids) if n <= 0 else n
        windows["window" if n <= 0 else f"window{n}"] = (win, run_windows(fwd, tap, w, st, b, ids, win))
    prompt_path = run_prefill(fwd, tap, w, st, pb, ids) if pb is not None else None
    serial = run_windows(fwd, tap, w, st, b, ids, 1)         # last: its state is what the continuation starts from
    cont = None
    if greedy_tokens:
        first = int(serial["logits"][-1].float().argmax())
        cont = greedy(fwd, tap, w, st, b, first, greedy_tokens)
        case["greedy_decode_path"] = cont
        case["greedy_decode_path_text"] = text(cont)
        if engine_greedy:
            case["greedy_engine"] = engine_greedy
    again = run_windows(fwd, tap, w, st, b, ids, 1)          # a second serial run: is the forward repeatable?
    rep = [p for p in compare(serial, again) if not p["equal"]]
    case["serial_vs_serial_again"] = {"differing": len(rep), "first": rep[0] if rep else None}
    case["nonfinite"] = {n: int((~torch.isfinite(t.float())).sum()) for n, t in serial.items()
                         if t.is_floating_point() and not bool(torch.isfinite(t.float()).all())}
    case["argmax"] = {"serial": serial["logits"].float().argmax(dim=-1).tolist()}
    if candidates:
        case["last_row"] = last_row(w, serial, ids, candidates, text)
        print(f"[export] {name}{suffix}: the last row's close calls: {json.dumps(case['last_row'])}", flush=True)
    files = {}

    def write(tag: str, rec: dict, per: int) -> None:
        prov = dict(provenance, mode=tag, ids=ids, rows_per_forward=per)
        if cont is not None:
            prov["greedy_decode_path"] = cont
            if engine_greedy:
                prov["greedy_engine"] = {k: v["tokens"] for k, v in engine_greedy.items()}
        files[tag] = save(out_dir / f"{name}.{tag}{suffix}.safetensors", ids, rec, prov)

    write("serial", serial, 1)
    for tag, (win, rec) in windows.items():
        points = compare(serial, rec)                        # point by point, in forward order
        differing = [p for p in points if not p["equal"]]
        case["serial_vs_" + tag] = {"rows_per_forward": win, "points": len(points), "differing": len(differing),
                                    "first": differing[0] if differing else None, "all": differing}
        case["argmax"][tag] = rec["logits"].float().argmax(dim=-1).tolist()
        write(tag, rec, win)
        print(f"[export] {name}{suffix}: {len(ids)} ids, serial vs {tag} ({win} rows a forward): "
              f"{len(differing)} of {len(points)} points differ"
              + (f", first {json.dumps(differing[0])}" if differing else ""), flush=True)
    if prompt_path is not None:
        got = prefill_distances(serial, prompt_path)
        case["prefill_vs_serial"] = got
        case["argmax"]["prefill_last_row"] = prompt_path["logits"].float().argmax(dim=-1).tolist()
        prov = dict(provenance, mode="prefill", ids=ids, rows_per_forward=len(ids),
                    path="forward.forward + forward.commit on state.Buffers(prefill=True): TensorFold's prompt "
                         "path, one chunk; a measurement, not zspark's reference")
        files["prefill"] = save(out_dir / f"{name}.prefill{suffix}.safetensors", ids, prompt_path, prov)
        by_name = {p["name"]: p for p in got["all"]}
        print(f"[export] {name}{suffix}: the prompt path against serial steps: {got['differing']} of "
              f"{got['points']} points differ, first {json.dumps(got['first'])}; final.h "
              f"{json.dumps(by_name.get('final.h'))}; logits (last row) {json.dumps(by_name.get('logits'))}",
              flush=True)
    case["files"] = files
    case["shapes"] = {n: [str(t.dtype).removeprefix("torch."), list(t.shape)] for n, t in serial.items()}
    print(f"[export] {name}{suffix}: a repeated serial run differs at {len(rep)} points; non-finite "
          f"{case['nonfinite']}; files {json.dumps(files)}", flush=True)
    if cont is not None:
        print(f"[export] {name}{suffix}: greedy on the decode path -> {cont} {case['greedy_decode_path_text']!r}",
              flush=True)
    return case


# -- the MTP head --------------------------------------------------------------------------------------
def closeness(ref: torch.Tensor, got: torch.Tensor) -> dict:
    """How far ``got`` is from ``ref``: equal bits, else the largest difference and the difference's RMS."""

    if ref.shape != got.shape or ref.dtype != got.dtype:
        return {"equal": False, "why": f"{tuple(ref.shape)} {ref.dtype} vs {tuple(got.shape)} {got.dtype}"}
    if not bool((bits(ref) != bits(got)).any()):
        return {"equal": True}
    a, b = ref.double(), got.double()
    return {"equal": False, "max_abs": float((a - b).abs().max()), "max_ref": float(a.abs().max()),
            "rms_diff_over_rms_ref": float((a - b).pow(2).mean().sqrt() / a.pow(2).mean().sqrt().clamp_min(1e-30))}


def run_mtp(mtp_mod, tap: Tap, w, st, mb, streams: torch.Tensor, nxt: list[int], rows: int) -> dict[str, torch.Tensor]:
    """The head over rows of main-model streams and their next tokens, from an empty head cache, ``rows`` rows an
    absorb (1: serial), the full head on every row; then the cache entries the absorbs left."""

    st.mtp_drafted = 0
    st.set_mtp_len(0)
    parts: list[dict[str, torch.Tensor]] = []
    for at in range(0, len(nxt), rows):
        toks = nxt[at:at + rows]
        n = len(toks)
        logits = mtp_mod.mtp_forward(w, st, mb, toks, streams[at:at + n], last_only=False)
        torch.cuda.synchronize()
        part = {"mtp.next": mb.ids[:n].clone().cpu(), "mtp.in": mb.mtp_in[:n].clone().cpu()}
        part.update(tap.take())
        part["mtp.logits"] = logits[:n].clone().cpu()
        parts.append(part)
        st.set_mtp_len(st.mtp_len + n)                   # as decode.absorb does after its forward
    rec = {name: torch.cat([p[name] for p in parts]) for name in parts[0]}
    n = len(nxt)
    rec["mtp.cache.k"] = st.mtp_kc.k[:n].clone().cpu()
    rec["mtp.cache.v"] = st.mtp_kc.v[:n].clone().cpu()
    rec["mtp.cache.index_k"] = st.mtp_ikc[:n].clone().cpu()
    return rec


def pick_draft(logits: torch.Tensor, draft_ids) -> tuple[int, float]:
    """``decode.Engine.sample_draft``'s greedy rule: the first maximum and its probability at temperature 1."""

    row = logits[:1].float()
    lse = torch.logsumexp(row, dim=-1, keepdim=True)
    top, col = row.max(dim=-1, keepdim=True)
    c = int(col)
    tok = int(draft_ids[c]) if draft_ids is not None else c
    return tok, float(np.exp(float(top) - float(lse)))


def run_chain(mtp_mod, tap: Tap, w, st, mb, last_streams: torch.Tensor, pending: int, rows_absorbed: int,
              depth: int) -> dict[str, torch.Tensor]:
    """``decode.draft``'s chain after ``rows_absorbed`` rows are in the head's cache (the last of them the row
    ``last_streams`` with ``pending``): row 0 scores that row again, row j > 0 runs the previous row's draft on the
    previous row's output streams. Every row is run twice on the same cache entry: the full head, the draft head."""

    st.mtp_drafted = 0
    st.set_mtp_len(rows_absorbed - 1)
    src, tok = last_streams.clone(), pending
    parts: list[dict[str, torch.Tensor]] = []
    for j in range(depth):
        full = mtp_mod.mtp_forward(w, st, mb, [tok], src, last_only=False)
        torch.cuda.synchronize()
        part = {"mtp.next": mb.ids[:1].clone().cpu(), "mtp.in": mb.mtp_in[:1].clone().cpu()}
        part.update(tap.take())
        part["mtp.logits"] = full[:1].clone().cpu()
        out = mb.streams[:1].clone()                     # ``draft`` passes e.mbuf.streams[:1] to the next step
        scored = mtp_mod.mtp_forward(w, st, mb, [tok], src, last_only=True)[:1].clone()
        torch.cuda.synchronize()
        again = tap.take()
        if bool((bits(again["mtp.streams"]) != bits(part["mtp.streams"])).any()):
            raise RuntimeError(f"chain row {j}: the head's second run left other streams")
        d, p = pick_draft(scored, w.draft_ids if w.draft_head is not None else None)
        part["mtp.draft_logits"] = scored.cpu()
        part["mtp.draft"] = torch.tensor([d], dtype=torch.int32)
        part["mtp.prob"] = torch.tensor([p], dtype=torch.float32)
        parts.append(part)
        st.set_mtp_len(st.mtp_len + 1)
        if j:
            st.mtp_drafted += 1
        src, tok = out, d
    return {name: torch.cat([p[name] for p in parts]) for name in parts[0]}


def engine_keeps(drafts: list[int], probs: list[float], confidence: float) -> list[int]:
    """What ``decode.draft`` returns for these chained drafts: the first always, later ones until one is under
    ``confidence``, and nothing after a first one under it."""

    kept: list[int] = []
    for j, (d, p) in enumerate(zip(drafts, probs)):
        low = confidence > 0 and p < confidence
        if low and j > 0:
            break
        kept.append(d)
        if low:
            break
    return kept


def engine_drafts(engine, decode_mod, ids: list[int], count: int, text) -> tuple[dict, dict]:
    """TensorFold's serving engine on one prompt, greedy: ``mtp_decode``'s result twice untouched (the timing),
    the serial twin, and once more with ``draft`` and the head's forwards wrapped to see each round. Returns the
    JSON summary and the traced run's tensors (the prompt's absorb and the first draft's)."""

    results: list = []
    o_decode = decode_mod.mtp_decode

    def mtp_decode(*a, **kw):
        res = o_decode(*a, **kw)
        results.append(res)
        return res

    def run(draft: bool) -> tuple[list[int], dict]:
        engine.cache = []                    # no kept prompt state: every run prefills from the first token
        got: list[int] = []
        stats = engine.generate(list(ids), count, None, lambda new: got.extend(new) and False, draft=draft,
                                stop_eos=False)
        return got, stats

    def summary(got: list[int], stats: dict, res) -> dict:
        return {"tokens": got, "text": text(got), "stats": stats, "rounds": res.rounds, "drafted": res.drafted,
                "accepted": res.accepted, "keeps": list(res.keeps), "widths": list(res.widths),
                "seconds": round(res.seconds, 4), "tokens_per_second": round(res.tokens_per_second, 2),
                "acceptance": round(res.accepted / res.drafted, 4) if res.drafted else None}

    out: dict = {"count": count, "sampling": "greedy (sampling=None)", "stop_eos": False}
    decode_mod.mtp_decode = mtp_decode
    rounds: list[dict] = []
    trace: dict = {}
    o_draft, o_fwd = decode_mod.draft, decode_mod.mtp_forward
    o_sample, o_emtp = decode_mod.Engine.sample_draft, decode_mod.Engine.mtp_forward
    try:
        out["timed"] = []
        for _ in range(2):
            got, stats = run(True)
            out["timed"].append(summary(got, stats, results[-1]))
        got, stats = run(False)
        out["serial_twin"] = {"tokens": got, "text": text(got), "stats": stats}

        def draft(e, streams, next_tokens, position, n, sampling, confidence=0.0):
            rounds.append({"absorbed": [int(t) for t in next_tokens], "position": int(position), "count": int(n),
                           "mtp_len_before": e.st.mtp_len, "mtp_drafted_before": e.st.mtp_drafted,
                           "candidates": []})
            got = o_draft(e, streams, next_tokens, position, n, sampling, confidence)
            rounds[-1]["drafts"] = [int(t) for t in got]
            return got

        def sample_draft(self, logits, position, sampling):
            tok, p = o_sample(self, logits, position, sampling)
            rounds[-1]["candidates"].append([int(tok), round(float(p), 6)])
            return tok, p

        def mtp_forward(w, st, b, next_tokens, streams, **kw):
            first = "prompt" not in trace and b is engine.e.pbuf      # prefill_chunk's absorb of the prompt
            if first:
                src, before = streams.detach().clone().cpu(), st.mtp_len
            got = o_fwd(w, st, b, next_tokens, streams, **kw)
            if first:
                torch.cuda.synchronize()
                trace["prompt"] = {"next": [int(t) for t in next_tokens], "in": src, "mtp_len_before": before,
                                   "streams": b.streams[:len(next_tokens)].clone().cpu()}
            return got

        def engine_mtp_forward(self, next_tokens, streams):
            first = "first" not in trace                             # mtp_decode's first draft: its absorb
            if first:
                src, before = streams.detach().clone().cpu(), self.st.mtp_len
            got = o_emtp(self, next_tokens, streams)
            if first:
                torch.cuda.synchronize()
                trace["first"] = {"next": [int(t) for t in next_tokens], "in": src, "mtp_len_before": before,
                                  "streams": self.mbuf.streams[:len(next_tokens)].clone().cpu(),
                                  "logits": got.clone().cpu()}
            return got

        decode_mod.draft, decode_mod.mtp_forward = draft, mtp_forward
        decode_mod.Engine.sample_draft, decode_mod.Engine.mtp_forward = sample_draft, engine_mtp_forward
        got, stats = run(True)
        res = results[-1]
        for r, keep, width in zip(rounds, res.keeps, res.widths):
            r["verified_rows"], r["kept"] = width, keep
        out["traced"] = {**summary(got, stats, res), "draft_calls": rounds,
                         "mtp_len_end": engine.e.st.mtp_len, "mtp_drafted_end": engine.e.st.mtp_drafted,
                         "pos_end": engine.e.st.pos,
                         "note": "timing here includes the wrappers' copies; read tok/s from 'timed'"}
    finally:
        decode_mod.mtp_decode, decode_mod.draft, decode_mod.mtp_forward = o_decode, o_draft, o_fwd
        decode_mod.Engine.sample_draft, decode_mod.Engine.mtp_forward = o_sample, o_emtp
    engine.cache = []
    tokens = [run_["tokens"] for run_ in out["timed"]] + [out["traced"]["tokens"]]
    out["tokens_equal_serial_twin"] = all(t == out["serial_twin"]["tokens"] for t in tokens)
    return out, trace


def export_mtp_case(fwd, mtp_mod, tap: Tap, w, st, b, mb, name: str, ids: list[int], out_dir: Path, provenance: dict,
                    depth: int, confidence: float, drafts: tuple[dict, dict] | None, text) -> dict:
    """One prompt through the head: serial absorbs, one window, the draft head's row, the chain, the files."""

    from safetensors.torch import load_file

    n = len(ids)
    main = run_windows(fwd, tap, w, st, b, ids, 1)           # the engine's decode arithmetic, as NAME.serial
    pending = int(main["logits"][-1].float().argmax())       # the first sampled token, greedy
    nxt = ids[1:] + [pending]
    streams = main["final.h"].to(w.device)
    case: dict = {"ids": ids, "text": text(ids), "pending": pending, "next_tokens": nxt,
                  "inputs": "row i: the main model's final.h[i] (b.streams of forward.forward on the decode "
                            "buffers) with next token ids[i + 1]; the last row with `pending`, the main model's "
                            "greedy token after the prompt",
                  "mtp_len": {"after_prompt_prefill": n - 1, "after_first_draft_absorb": n,
                              "note": "decode.prefill_chunk absorbs rows 0..R-2 with ids[1:]; mtp_decode's first "
                                      "draft absorbs the last row with the first sampled token; chained drafts "
                                      "add one entry each (State.mtp_drafted) and the next absorb trims them"}}
    existing = out_dir / f"{name}.serial.safetensors"
    if existing.is_file():                                    # the head's inputs are the file's the tests already read
        old = load_file(str(existing))
        case["main_vs_existing_serial_file"] = {k: closeness(old[k], main[k]) for k in ("final.h", "logits")
                                                if k in old}
    serial = run_mtp(mtp_mod, tap, w, st, mb, streams, nxt, 1)
    window = run_mtp(mtp_mod, tap, w, st, mb, streams, nxt, n)          # last: the chain starts from its cache
    points = compare(serial, window)
    differing = [p for p in points if not p["equal"]]
    case["serial_vs_window"] = {"rows_per_forward": n, "points": len(points), "differing": len(differing),
                                "first": differing[0] if differing else None, "all": differing}
    print(f"[export] {name}.mtp: {n} rows, serial vs window: {len(differing)} of {len(points)} points differ"
          + (f", first {json.dumps(differing[0])}" if differing else ""), flush=True)
    case["nonfinite"] = {k: int((~torch.isfinite(t.float())).sum()) for k, t in window.items()
                         if t.is_floating_point() and not bool(torch.isfinite(t.float()).all())}
    case["argmax"] = {"main": main["logits"].float().argmax(dim=-1).tolist(),
                      "mtp_serial": serial["mtp.logits"].float().argmax(dim=-1).tolist(),
                      "mtp_window": window["mtp.logits"].float().argmax(dim=-1).tolist()}

    chain = run_chain(mtp_mod, tap, w, st, mb, streams[n - 1:n], pending, n, depth)
    has_draft_head = w.draft_head is not None
    extra = {"mtp.draft_logits": chain["mtp.draft_logits"][:1].clone()}
    if has_draft_head:
        ids32 = w.draft_ids.to(torch.int32).cpu()
        extra["mtp.draft_ids"] = ids32
        chain["mtp.draft_ids"] = ids32
    drafts_chain = chain["mtp.draft"].tolist()
    probs = [float(p) for p in chain["mtp.prob"].tolist()]
    full = chain["mtp.logits"].float()
    full_top = full.argmax(dim=-1)
    full_p = torch.softmax(full, dim=-1).gather(1, full_top[:, None])[:, 0]
    row0 = [p for p in compare({k: v[-1:] for k, v in window.items() if k in chain and k != "mtp.draft_ids"
                                and not k.startswith("mtp.cache.")}, {k: v[:1] for k, v in chain.items()})
            if not p["equal"]]
    case["chain"] = {
        "depth": depth, "confidence": confidence, "input_tokens": chain["mtp.next"].tolist(),
        "drafts": drafts_chain, "drafts_text": [text([d]) for d in drafts_chain], "probs": [round(p, 6) for p in probs],
        "engine_keeps": engine_keeps(drafts_chain, probs, confidence),
        "full_head_argmax": full_top.tolist(), "full_head_probs": [round(float(p), 6) for p in full_p],
        "draft_head_agrees_with_full_head": drafts_chain == full_top.tolist(),
        "mtp_len_by_row": [n - 1 + j for j in range(depth)],
        "row0_vs_window_last_row_differing": [p["name"] for p in row0],
        "note": "row j ran with mtp_len_by_row[j] cache entries before it; rows after the engine's kept drafts "
                "are extra (TensorFold stops before a draft under confidence)"}
    print(f"[export] {name}.mtp: chain drafts {drafts_chain} probs {case['chain']['probs']} -> the engine keeps "
          f"{case['chain']['engine_keeps']}", flush=True)

    files = {}
    for tag, rec, per in (("mtp.serial", serial, 1), ("mtp.window", {**window, **extra}, n),
                          ("mtp.chain", chain, 1)):
        prov = dict(provenance, mode=tag, ids=ids, rows_per_forward=per, pending=pending, next_tokens=nxt)
        body = dict(rec) if tag == "mtp.chain" else {"final.h": main["final.h"], **rec}
        files[tag] = save(out_dir / f"{name}.{tag}.safetensors", ids, body, prov)
    case["files"] = files
    case["shapes"] = {"final.h": ["bfloat16", list(main["final.h"].shape)],
                      **{k: [str(t.dtype).removeprefix("torch."), list(t.shape)] for k, t in {**window, **extra}.items()}}
    case["chain_shapes"] = {k: [str(t.dtype).removeprefix("torch."), list(t.shape)] for k, t in chain.items()}

    if drafts is not None:
        summary, trace = drafts
        case["engine_drafts"] = summary
        against: dict = {"note": "TensorFold's own absorbs (prompt kernels and prompt buffers for rows 0..R-2, "
                                 "decode buffers and graphs for the last row) against this export's decode-buffer "
                                 "window"}
        if "prompt" in trace:
            t = trace["prompt"]
            k = len(t["next"])
            against["prompt_rows"] = {"rows": k, "mtp_len_before": t["mtp_len_before"],
                                      "next_tokens_equal": t["next"] == nxt[:k],
                                      "in_vs_final.h": closeness(main["final.h"][:k], t["in"]),
                                      "streams_vs_mtp.streams": closeness(window["mtp.streams"][:k], t["streams"])}
        if "first" in trace:
            t = trace["first"]
            against["last_row"] = {"mtp_len_before": t["mtp_len_before"], "next_tokens_equal": t["next"] == nxt[-1:],
                                   "in_vs_final.h": closeness(main["final.h"][-1:], t["in"]),
                                   "streams_vs_mtp.streams": closeness(window["mtp.streams"][-1:], t["streams"]),
                                   "draft_logits": closeness(extra["mtp.draft_logits"], t["logits"][:1]),
                                   "draft": pick_draft(t["logits"].to(w.device),
                                                       w.draft_ids if has_draft_head else None)[0]}
        case["engine_vs_export"] = against
    return case


# -- a sampled reply -----------------------------------------------------------------------------------
def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def engine_sampled(engine, decode_mod, ids: list[int], count: int, sampling) -> dict:
    """TensorFold's serving engine on one prompt with ``sampling``: the drafted reply and the serial twin's, and
    the positions ``Engine.sample`` keyed the main model's rows with in each."""

    o_sample = decode_mod.Engine.sample
    seen: list[int] = []

    def sample(self, logits, positions, sampling, **kw):
        if not kw.get("draft", False):       # the main model's rows; a draft is keyed by the place it guesses
            seen.extend(int(p) for p in positions)
        return o_sample(self, logits, positions, sampling, **kw)

    out: dict = {}
    decode_mod.Engine.sample = sample
    try:
        for tag, draft in (("drafted", True), ("serial", False)):
            engine.cache = []                # no kept prompt state: every run prefills from the first token
            seen.clear()
            got: list[int] = []
            stats = engine.generate(list(ids), count, sampling, lambda new: got.extend(new) and False, draft=draft,
                                    stop_eos=False)
            out[tag] = {"tokens": got, "stats": stats, "positions": sorted(set(seen)), "rows_sampled": len(seen)}
    finally:
        decode_mod.Engine.sample = o_sample
    engine.cache = []
    return out


def decode_path_sampled(fwd, sample_rows, w, st, b, ids: list[int], count: int, sampling) -> list[int]:
    """``count`` sampled tokens after ``ids`` by one-row decode-path steps from an empty state, each keyed by its
    index in the sequence: ``decode.serial_decode``'s loop on the exporter's buffers, the prompt included."""

    st.reset(w)
    out: list[int] = []
    for at in range(len(ids) + count - 1):
        logits = fwd.forward(w, st, b, [ids[at] if at < len(ids) else out[at - len(ids)]])
        if st.pos != at:
            raise RuntimeError(f"row {at}: the state holds {st.pos} rows")
        if at >= len(ids) - 1:
            out.append(int(sample_rows(logits[:1], [st.pos + 1], sampling)[0]))
        fwd.commit(w, st, b, 1, 1)
    return out


# -- a score -------------------------------------------------------------------------------------------
class Score:
    """``zspark score``'s sums: each token after the first against the row before it, all and the second half."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.nll, self.first, self.scored = [0.0, 0.0], [0, 0], [0, 0]

    def add(self, at: int, logits: torch.Tensor, targets: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows ``at`` .. of the sequence with their logits; ``targets`` the tokens after them (the last row of the
        sequence has none). Returns each scored row's loss and whether its target ranked first."""

        n = len(targets)
        lf = logits[:n].float()
        high = lf.max(dim=-1, keepdim=True).values
        total = torch.exp((lf - high).double()).sum(dim=-1)
        t = torch.tensor(targets, dtype=torch.long, device=lf.device)
        own = lf.gather(1, t[:, None])
        loss = (torch.log(total) - (own - high)[:, 0].double()).cpu()
        top = (own == high)[:, 0].cpu()
        for r in range(n):
            for part, counted in enumerate((True, at + r + 1 >= self.count // 2)):
                if counted:
                    self.nll[part] += float(loss[r])
                    self.first[part] += int(top[r])
                    self.scored[part] += 1
        return loss, top

    def result(self) -> dict:
        return {name: {"tokens": self.scored[part], "nll": round(self.nll[part] / max(1, self.scored[part]), 5),
                       "ranked_first": round(self.first[part] / max(1, self.scored[part]), 4)}
                for part, name in enumerate(("all", "second_half"))}


def quantiles(values: list[float]) -> dict:
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)),
            "p99": float(np.quantile(v, 0.99)), "max": float(v.max()), "row_of_max": int(v.argmax())}


def score(fwd, w, Buffers, State, ids: list[int], window: int, prompt_path: bool, each: bool = False) -> dict:
    """The ids through the decode path in windows of ``window`` rows, scored; and, with ``prompt_path``, through
    the prompt kernels in chunks of as many rows from a second state, every row headed, scored and compared.
    ``each``: every token's loss on both paths and every row's distances too (``by_token``)."""

    n = len(ids)
    b = Buffers(w, window, n, moe_prefill=True)
    st = State(w, n, window, "bf16")
    st.reset(w)
    own = Score(n)
    out: dict = {"tokens": n, "window": window}
    if prompt_path:
        pb = Buffers(w, window, n, prefill=True)
        pst = State(w, n, window, "bf16")
        pst.reset(w)
        theirs = Score(n)
        h_rel, l_rel, l_max, loss_gap = [], [], [], []
        same_top = flips = 0
    t0 = time.time()
    seconds = [0.0, 0.0]
    losses: list[list[float]] = [[], []]
    for at in range(0, n, window):
        chunk = ids[at:at + window]
        rows = len(chunk)
        targets = ids[at + 1:at + rows + 1]
        t1 = time.time()
        logits = fwd.forward(w, st, b, chunk)
        torch.cuda.synchronize()
        seconds[0] += time.time() - t1
        loss, top = own.add(at, logits, targets)
        losses[0] += loss.tolist()
        if prompt_path:
            t1 = time.time()
            with torch.no_grad():
                plogits = fwd.compute(w, fwd.stage(w, pb, [(pst, chunk)]), pb, logits=True, ends=list(range(rows)))
            torch.cuda.synchronize()
            seconds[1] += time.time() - t1
            ploss, ptop = theirs.add(at, plogits, targets)
            losses[1] += ploss.tolist()
            a, c = b.streams[:rows].double(), pb.streams[:rows].double()
            h_rel += ((a - c).pow(2).mean(dim=1).sqrt() / a.pow(2).mean(dim=1).sqrt().clamp_min(1e-30)).tolist()
            a, c = logits[:rows].double(), plogits[:rows].double()
            l_rel += ((a - c).pow(2).mean(dim=1).sqrt() / a.pow(2).mean(dim=1).sqrt().clamp_min(1e-30)).tolist()
            l_max += (a - c).abs().amax(dim=1).tolist()
            same_top += int((logits[:rows].float().argmax(dim=-1) == plogits[:rows].float().argmax(dim=-1)).sum())
            loss_gap += (ploss - loss).abs().tolist()
            flips += int((top != ptop).sum())
            fwd.commit(w, pst, pb, rows, rows)
        fwd.commit(w, st, b, rows, rows)
    out["decode_path"] = {**own.result(), "forward_seconds": round(seconds[0], 2),
                          "path": "forward.forward + forward.commit on state.Buffers(moe_prefill=True), "
                                  f"{window} rows a window: the arithmetic of TensorFold's decode steps"}
    if prompt_path:
        out["prompt_path"] = {**theirs.result(), "forward_seconds": round(seconds[1], 2),
                              "path": f"forward.compute on state.Buffers(prefill=True), {window} rows a chunk, "
                                      "every row an end: TensorFold's prompt kernels"}
        out["prompt_vs_decode"] = {
            "rows": n, "final.h_rms_diff_over_rms_ref_by_row": quantiles(h_rel),
            "logits_rms_diff_over_rms_ref_by_row": quantiles(l_rel), "logits_max_abs_by_row": quantiles(l_max),
            "rows_with_the_same_argmax": same_top, "abs_loss_difference_by_token": quantiles(loss_gap),
            "tokens_ranked_first_in_one_only": flips,
            "note": "per row: the prompt path's value against the decode path's (the reference); both ran the "
                    "whole sequence on their own state, so a row's difference includes its context's"}
    if each:
        # token i + 1's loss at index i; a row's distances at its own index
        out["by_token"] = {"decode_loss": losses[0]}
        if prompt_path:
            out["by_token"].update(prompt_loss=losses[1], final_h_relative_by_row=h_rel, logits_relative_by_row=l_rel)
    out["seconds"] = round(time.time() - t0, 1)
    return out


# -- provenance ----------------------------------------------------------------------------------------
def tensorfold_commit(package: Path) -> str:
    """HEAD of the checkout the package was imported from, read from .git without git (the container has none)."""

    for root in package.parents:
        git = root / ".git"
        if not git.is_dir():
            continue
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        packed = git / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0]
    return "unknown"


def revision(model_dir: Path) -> str:
    """A Hugging Face cache path's snapshot hash, else the directory's name."""

    parts = model_dir.resolve().parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts[:-1] else model_dir.name


def prepared(model_dir: Path) -> dict | None:
    """What ``hybrid_prepare.py`` recorded beside a checkpoint it wrote: the source, its revision and the options."""

    manifest = model_dir / "hybrid_manifest.json"
    if not manifest.is_file():
        return None
    got = json.loads(manifest.read_text())
    return {k: got.get(k) for k in ("layout", "source", "source_revision", "shared_experts", "tensors_rewritten")}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--case", type=parse_case, action="append", required=True, metavar="NAME=ID,ID,...",
                    help="a prompt's token ids; repeat for several prompts in one load")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="NAME.serial.safetensors, NAME.window.safetensors and NAME.json are written here")
    ap.add_argument("--layers", type=int, default=None,
                    help="load and run only the first N layers, then the mixer and head (default: all)")
    ap.add_argument("--window-rows", type=int, action="append", default=[], metavar="N",
                    help="run the prompt in windows of N rows, written as NAME.windowN (repeatable; default and 0: "
                         "the whole prompt as one window, NAME.window)")
    ap.add_argument("--moe-slots", choices=("bf16", "f32", "both"), default="bf16",
                    help="bf16: the experts' prompt arithmetic and bf16 slots, as decode.Engine builds its decode "
                         "buffers (moe_prefill=True), so what TensorFold serves; f32: state.Buffers' own default, "
                         "the decode tile with fp32 slots, whose files are named NAME.*.f32.safetensors")
    ap.add_argument("--greedy", action="append", default=[], metavar="NAME[:N]",
                    help="after NAME's serial run, continue greedily on the decode path (repeatable); :N overrides "
                         "--greedy-tokens for that case")
    ap.add_argument("--greedy-tokens", type=int, default=32)
    ap.add_argument("--candidates", type=parse_case, action="append", default=[], metavar="NAME=ID,ID,...",
                    help="record in NAME.json (last_row) how NAME's last serial row ranks these token ids: its top "
                         "five, the ids' logits as TensorFold argmaxes them, and on an EXL3 pack the head's fp32 "
                         "sums before the bf16 store (repeatable; not with --mtp or --sampled)")
    ap.add_argument("--engine", action="store_true",
                    help="load through TensorFold's serving engine with no drafts (admission, warm-up, graphs) and "
                         "take each --greedy continuation from its generate() too; not with --layers")
    ap.add_argument("--sampled", action="append", default=[], metavar="NAME[:N]",
                    help="write only NAME.sampled.json: N tokens (default --greedy-tokens) the serving engine "
                         "samples after NAME's prompt with --seed, --temperature, --top-k, --top-p and --min-p, "
                         "drafted and by its serial twin, and the exporter's own decode-path steps sampled with the "
                         "same key (repeatable; needs --engine, not with --mtp)")
    ap.add_argument("--seed", type=int, default=7, help="--sampled: the sampling key's seed")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--capacity", type=int, default=4096, help="context rows of the exporter's own state")
    ap.add_argument("--prefill", action="store_true",
                    help="also run each prompt through TensorFold's prompt path (one chunk on prompt buffers), write "
                         "NAME.prefill and record its distance from the serial steps in NAME.json "
                         "(prefill_vs_serial); not with --mtp or --sampled")
    ap.add_argument("--score", type=Path, default=None, metavar="FILE",
                    help="score FILE's whitespace-separated token ids as `zspark score` does, on the decode path in "
                         "windows of 16 rows and on the prompt kernels beside it, into score.json; not with --mtp, "
                         "--sampled or --layers")
    ap.add_argument("--score-each", action="store_true",
                    help="with --score: every token's loss on both paths and every row's distances, in score.json")
    ap.add_argument("--tensorfold-commit", default=None, metavar="HASH",
                    help="the commit of a TensorFold tree that has no .git (a copy): recorded as stated")
    ap.add_argument("--mtp", action="store_true",
                    help="capture the MTP head instead: NAME.mtp.serial, NAME.mtp.window, NAME.mtp.chain and "
                         "NAME.mtp.json, and nothing else. With --engine the engine drafts, and each --greedy case "
                         "gets TensorFold's draft statistics. --window-rows and --moe-slots do not apply")
    args = ap.parse_args()
    counts = {}
    for item in args.greedy:
        case_name, _, n = item.partition(":")
        counts[case_name] = int(n) if n else args.greedy_tokens
    args.greedy = list(counts)
    sampled = {}
    for item in args.sampled:
        case_name, _, n = item.partition(":")
        sampled[case_name] = int(n) if n else args.greedy_tokens
    if args.engine and args.layers is not None:
        ap.error("--engine loads the whole model: drop --layers")
    if sampled and (not args.engine or args.mtp):
        ap.error("--sampled takes the reply from the serving engine: add --engine, drop --mtp")
    if (args.prefill or args.score) and (args.mtp or sampled):
        ap.error("--prefill and --score go with the main model's export: drop --mtp and --sampled")
    if args.score and args.layers is not None:
        ap.error("--score scores the whole model: drop --layers")

    import tensorfold
    import triton
    from tensorfold.cuda import build
    from tensorfold.cuda.exl3.format import is_exl3
    from tensorfold.families.qwen4_exp.cuda import CONFIDENCE, DEPTH
    from tensorfold.families.qwen4_exp.cuda import decode as decode_mod
    from tensorfold.families.qwen4_exp.cuda import forward as fwd
    from tensorfold.families.qwen4_exp.cuda import glue as glue_mod
    from tensorfold.families.qwen4_exp.cuda import mtp as mtp_mod        # bound to forward.py before any hook
    from tensorfold.families.qwen4_exp.cuda import weight_types, weights
    from tensorfold.families.qwen4_exp.cuda.engine import build_kernels
    from tensorfold.families.qwen4_exp.cuda.state import Buffers, State

    if tensorfold.__version__ != PINNED:
        print(f"[export] TensorFold {tensorfold.__version__}, written against {PINNED}: the hooks may not fit",
              file=sys.stderr)
    model_dir = args.model_dir
    names = [name for name, _ in args.case]
    for name in args.greedy:
        if name not in names:
            ap.error(f"--greedy {name}: no such --case")
    for name in sampled:
        if name not in names:
            ap.error(f"--sampled {name}: no such --case")
    close = dict(args.candidates)
    for name in close:
        if name not in names:
            ap.error(f"--candidates {name}: no such --case")
    if close and (args.mtp or sampled):
        ap.error("--candidates reads the main model's serial rows: drop --mtp and --sampled")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    exl3 = is_exl3(model_dir)
    commit = tensorfold_commit(Path(tensorfold.__file__).resolve())
    provenance = {
        "tensorfold_version": tensorfold.__version__,
        "tensorfold_commit": args.tensorfold_commit if commit == "unknown" and args.tensorfold_commit else commit,
        "checkpoint": str(model_dir), "checkpoint_revision": revision(model_dir),
        "checkpoint_prepared": prepared(model_dir),
        "torch": torch.__version__, "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
        "moe_slots": args.moe_slots, "layers": args.layers,
        "path": "forward.forward + forward.commit on state.Buffers (decode path), no drafts, text only, bf16 KV",
    }
    if commit == "unknown" and args.tensorfold_commit:
        provenance["tensorfold_commit_source"] = ("stated with --tensorfold-commit: "
                                                  f"{Path(tensorfold.__file__).resolve()} is in a tree without .git")
    if exl3:
        provenance["format"] = ("exl3: the decode path's linears (exl3_mm.X3.__call__, F16.partials) and the "
                                "grouped expert kernel with fp32 slots on either --moe-slots; TensorFold's prompt "
                                "path (X3.prefill, bf16 slots) is in NAME.prefill only")
    if args.mtp:
        provenance.update(moe_slots="bf16", path="forward.forward on state.Buffers(moe_prefill=True), then "
                          "mtp.mtp_forward on state.Buffers (decode path, fp32 expert slots), text only, bf16 KV")
    report: dict = {"provenance": provenance}

    def text(ids: list[int]) -> str | None:
        """The ids as text, for a reader of the JSON (nothing depends on it)."""

        try:
            from tokenizers import Tokenizer

            return Tokenizer.from_file(str(model_dir / "tokenizer.json")).decode(ids, skip_special_tokens=False)
        except Exception:                    # no tokenizers package, or no tokenizer beside the weights
            return None

    # -- load ----------------------------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    before = memory()
    t0 = time.time()
    engine = None
    engine_greedy: dict[str, dict] = {}
    if args.engine:
        # what ``tensorfold serve --no-drafts`` builds: admission, kernels, weights, buffers, graphs, warm-up
        from tensorfold.families.qwen4_exp import cuda_engine

        engine = cuda_engine(model_dir, no_drafts=not (args.mtp or sampled))
        w = engine.w
        report["engine"] = {"max_len": engine.max_len, "context_window": engine.context_window,
                            "prefill_rows": engine.prefill_rows, "graphs": engine.e.graphs is not None,
                            "graphs_requested": engine.graphs_enabled, "depth": engine.depth,
                            "confidence": engine.confidence, "decode_rows": engine.e.rows,
                            "mtp_graphs": len(engine.e.graphs.mtp) if engine.e.graphs is not None else 0,
                            "capacity_plan": {k: v for k, v in engine.capacity_plan.items()
                                              if isinstance(v, (int, float, str, bool))}}
    else:
        build.refuse_small_gpu()
        if args.layers is not None:
            read = weight_types.Config.read.__func__

            def shortened(cls, directory):
                full = read(cls, directory)
                n = args.layers
                if not 1 <= n <= full.layers:
                    raise ValueError(f"--layers {n}: the model has {full.layers}")
                return replace(full, layers=n, layer_types=full.layer_types[:n],
                               ple_layers=[i for i in full.ple_layers if i < n])

            weight_types.Config.read = classmethod(shortened)
        build_kernels(exl3=exl3, nvfp4=not exl3, solo=True)
        w = weights.load(model_dir, "cuda", mtp=args.mtp, draft_vocab="default" if args.mtp else None)
    torch.cuda.synchronize()
    if args.mtp and w.mtp is None:
        raise SystemExit("--mtp: this checkpoint has no MTP head")
    report["load"] = {"seconds_total": round(time.time() - t0, 1),
                      "seconds_weights": round(float(w.meta.get("load_seconds", 0.0)), 1),
                      "weights_gib": round(w.nbytes() / 2**30, 2), "layers": len(w.layers),
                      "norms_around_one": bool(w.around_one), "quant": w.cfg.quant,
                      "before": before, "after": memory()}
    print(f"[export] loaded: {json.dumps(report['load'])}", flush=True)

    # -- the MTP head ----------------------------------------------------------------------------------
    if args.mtp:
        def kernel_of(q) -> str:              # an EXL3 pack's matrices carry no ``kernel``: their type says it
            return getattr(q, "kernel", type(q).__name__ if exl3 else "qmm")

        report["mtp"] = {
            "depth": engine.depth if engine is not None else DEPTH,
            "confidence": engine.confidence if engine is not None else CONFIDENCE,
            "draft_vocab": None if w.draft_ids is None else int(w.draft_ids.numel()),
            "draft_vocab_source": "families/qwen4_exp/cuda/draft_vocab.txt (weights.draft_token_ids('default'))",
            "draft_head": None if w.draft_head is None else
            f"{type(w.draft_head).__name__}: lm_head's draft rows requantized at load "
            f"(kernel {getattr(w.draft_head, 'kernel', 'qmm')}, {w.draft_head.n} rows)",
            "full_vocab": int(w.head.n), "experts_kernel": kernel_of(w.mtp.layer.moe.experts),
            "fc_kernel": kernel_of(w.mtp.fc_e),
            "graphs": engine.e.graphs is not None if engine is not None else False}
        print(f"[export] mtp: {json.dumps(report['mtp'])}", flush=True)
        traced: dict[str, tuple[dict, dict]] = {}
        if engine is not None:                # before any hook is installed: the engine's graphs replay untouched
            for name, ids in args.case:
                if name in counts:
                    traced[name] = engine_drafts(engine, decode_mod, list(ids), counts[name], text)
                    for run in traced[name][0]["timed"]:
                        print(f"[export] {name}: mtp_decode {run['tokens_per_second']} tok/s, {run['rounds']} rounds, "
                              f"drafted {run['drafted']}, accepted {run['accepted']}, keeps {run['keeps']}",
                              flush=True)
                    print(f"[export] {name}: serial twin {traced[name][0]['serial_twin']['stats']}; same tokens: "
                          f"{traced[name][0]['tokens_equal_serial_twin']}", flush=True)
        rows = max(8, max(len(ids) for _, ids in args.case))
        tap = Tap(fwd)
        tap.install()
        tap.install_mtp(mtp_mod, glue_mod, w)
        try:
            b = Buffers(w, rows, args.capacity, moe_prefill=True)      # decode.Engine.buf
            mb = Buffers(w, rows, args.capacity)                       # decode.Engine.mbuf
            st = State(w, args.capacity, rows, "bf16")
            for name, ids in args.case:
                case = export_mtp_case(fwd, mtp_mod, tap, w, st, b, mb, name, ids, args.out_dir, provenance,
                                       report["mtp"]["depth"] or DEPTH, report["mtp"]["confidence"],
                                       traced.get(name), text)
                (args.out_dir / f"{name}.mtp.json").write_text(
                    json.dumps({"provenance": provenance, "load": report["load"], "engine": report.get("engine"),
                                "mtp": report["mtp"], **case}, indent=1))
                print(f"[export] {name}.mtp: files {json.dumps(case['files'])}", flush=True)
        finally:
            tap.remove()
        print(f"[export] memory at the end: {json.dumps(memory())}", flush=True)
        if engine is not None:
            engine.close()
        return 0

    # -- a sampled reply ---------------------------------------------------------------------------------
    if sampled:
        from tensorfold.cuda.sampling import sample_rows
        from tensorfold.engine.exact_sampling import SEED_SALT, Sampling

        sampling = Sampling(seed=args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
                            min_p=args.min_p)
        settings = {"seed": int(sampling.seed), "temperature": float(sampling.temperature),
                    "top_k": int(sampling.top_k), "top_p": float(sampling.top_p), "min_p": float(sampling.min_p)}
        runs = {name: engine_sampled(engine, decode_mod, list(ids), sampled[name], sampling)
                for name, ids in args.case if name in sampled}      # before the exporter's own buffers exist
        for name, ids in args.case:
            if name not in sampled:
                continue
            count, run = sampled[name], runs[name]
            tokens = run["drafted"]["tokens"]
            want = list(range(len(ids), len(ids) + count))
            own = {}
            for slots in ("bf16", "f32"):
                b = Buffers(w, 8, args.capacity, moe_prefill=slots == "bf16")
                st = State(w, args.capacity, 8, "bf16")
                got = decode_path_sampled(fwd, sample_rows, w, st, b, list(ids), count, sampling)
                own[slots] = {"tokens": got, "text": text(got), "common_prefix_with_engine": common_prefix(got, tokens)}
                del b, st
                torch.cuda.empty_cache()
            case = {
                "provenance": dict(provenance, mode="sampled", moe_slots="bf16",
                                   path="FlashNextEngine.generate: decode.prefill on the prompt kernels, then "
                                        "decode.mtp_decode (drafted) or serial_decode (the twin), bf16 expert "
                                        "slots, graphs, text only, bf16 KV"),
                "load": report["load"], "engine_config": report.get("engine"),
                "ids": ids, "text": text(ids), "count": count, "stop_eos": False, "sampling": settings,
                "tokens": tokens, "tokens_text": text(tokens),
                "key": {
                    "draw": "exact_sampling.uniform(seed, position, token id): splitmix64 of the three",
                    "seed": "Sampling.seed as passed to generate(), masked to 64 bits; nothing is derived from the "
                            "prompt (the servers call seed_for(prompt) only for a request without a seed)",
                    "seed_salt": SEED_SALT,
                    "position": "the index in the sequence of the token being chosen: len(ids) for the first reply "
                                "token (decode.prefill), st.pos + 1 for a serial step, st.pos + 1 + r for row r of "
                                "a verify window",
                    "first_position": len(ids),
                    "engine_positions": {tag: run[tag]["positions"] for tag in ("drafted", "serial")},
                    # a window's rows past its kept ones are sampled too, so the drafted run may look further
                    "engine_positions_are_the_replys": {"serial": run["serial"]["positions"] == want,
                                                        "drafted": run["drafted"]["positions"][:count] == want}},
                "engine": {tag: {"tokens": run[tag]["tokens"], "text": text(run[tag]["tokens"]),
                                 "stats": run[tag]["stats"], "rows_sampled": run[tag]["rows_sampled"]}
                           for tag in ("drafted", "serial")},
                "engine_drafted_equals_serial": run["drafted"]["tokens"] == run["serial"]["tokens"],
                "decode_path": {**own, "note": "forward.forward + forward.commit one row at a time on state.Buffers, "
                                               "sampled with cuda.sampling.sample_rows and the same key: bf16 is "
                                               "the engine's decode arithmetic without its prompt kernels, f32 "
                                               "the decode tile with fp32 slots (what zig/src/models/qwen4/blocks/moe.zig "
                                               "ports)"},
            }
            (args.out_dir / f"{name}.sampled.json").write_text(json.dumps(case, indent=1))
            print(f"[export] {name}.sampled: {settings} -> {tokens} {text(tokens)!r}", flush=True)
            print(f"[export] {name}.sampled: drafted equals serial: {case['engine_drafted_equals_serial']}; "
                  f"positions {case['key']['engine_positions_are_the_replys']}; the decode path agrees with the "
                  f"engine for {own['bf16']['common_prefix_with_engine']} (bf16 slots) and "
                  f"{own['f32']['common_prefix_with_engine']} (fp32 slots) of {count} tokens; engine stats "
                  f"{run['drafted']['stats']}", flush=True)
        print(f"[export] memory at the end: {json.dumps(memory())}", flush=True)
        engine.close()
        return 0

    # -- TensorFold's own serving path, for the continuation --------------------------------------------
    if engine is not None:
        for name, ids in args.case:
            if name not in args.greedy:
                continue
            for draft in (True, False):       # True: the engine's own state (graphs); False: its serial twin
                got: list[int] = []
                stats = engine.generate(list(ids), counts[name], None, lambda new: got.extend(new) and False,
                                        draft=draft, stop_eos=False)
                engine_greedy.setdefault(name, {})["own_state" if draft else "twin"] = {
                    "tokens": got, "text": text(got), "stats": stats}
                print(f"[export] {name}: engine generate(draft={draft}) -> {got} {text(got)!r} {stats}", flush=True)

    # -- a score, before any hook ----------------------------------------------------------------------------
    if args.score is not None:
        from tensorfold.families.qwen4_exp.cuda.state import ENDS

        score_ids = [int(x) for x in args.score.read_text().split()]
        got = score(fwd, w, Buffers, State, score_ids, ENDS, True, args.score_each)
        torch.cuda.empty_cache()
        got = {"provenance": dict(provenance, mode="score"), "load": report["load"], "ids_file": str(args.score),
               **got, "memory": memory()}
        (args.out_dir / "score.json").write_text(json.dumps(got, indent=1))
        print(f"[export] score: {len(score_ids)} ids, decode path {json.dumps(got['decode_path'])}; prompt path "
              f"{json.dumps(got['prompt_path'])}; {json.dumps(got['prompt_vs_decode'])}", flush=True)

    # -- the exporter's own buffers and state, as decode.Engine builds them ----------------------------------
    longest = max(len(ids) for _, ids in args.case)
    widths = args.window_rows or [0]
    rows = max(8, max(longest if n <= 0 else min(longest, n) for n in widths))
    tap = Tap(fwd)
    tap.install()
    try:
        for slots in ("bf16", "f32") if args.moe_slots == "both" else (args.moe_slots,):
            b = Buffers(w, rows, args.capacity, moe_prefill=slots == "bf16")
            st = State(w, args.capacity, rows, "bf16")
            pb = Buffers(w, decode_mod.PREFILL_ROWS, args.capacity, prefill=True) if args.prefill else None
            suffix = "" if slots == "bf16" else ".f32"
            for name, ids in args.case:
                prov = dict(provenance, moe_slots=slots)
                case = export_case(fwd, tap, w, st, b, name, ids, widths, args.out_dir, suffix, prov,
                                   counts.get(name, 0), engine_greedy.get(name), text, pb, close.get(name))
                (args.out_dir / f"{name}{suffix}.json").write_text(
                    json.dumps({"provenance": prov, "load": report["load"], "engine": report.get("engine"),
                                **case, "memory": memory()}, indent=1))
            del b, st, pb
            torch.cuda.empty_cache()
    finally:
        tap.remove()
    report["memory_end"] = memory()
    print(f"[export] memory at the end: {json.dumps(report['memory_end'])}", flush=True)
    if engine is not None:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
