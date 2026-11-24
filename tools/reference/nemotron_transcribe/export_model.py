#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Nemotron-3.5-Transcribe's model fixtures (spec 151 WS-04), run inside
`zspark151-nemo-vllm:local` against the card's vLLM 0.32.0 + `nemo_speechlm` stack, offline
(vLLM's `LLM` entry point, no server). For each fixture clip it captures, under the same greedy
decode:

  * `enc.out`      — the perception encoder's output rows as the LM gets them (`embed_multimodal`'s
                     return, bf16 [audio_rows, 2688]);
  * `forward.*`    — per backbone forward: `forward.num_rows`/`forward.prefill` [F],
                     `forward.input_ids`, `forward.positions` (concatenated over kept forwards;
                     MTP verify forwards score several rows, so boundaries are explicit);
  * `hidden.L*` / `residual.L*` — per layer per kept forward: the block's mixer output and its
                     returned residual (NemotronH blocks return `(hidden_states, residual)`;
                     `fused_add_rms_norm` returns residual as fp32(in+residual)→bf16 — vLLM's
                     completed-layer convention is hidden+residual, interfaces.py:1609);
  * `logits.step*` — `compute_logits`'s output for the prefill and each kept decode step, plus
                     `logits.top5_ids`/`logits.top5_vals`;
  * `prompt.ids`, `audio.placeholder_rows`, `tokens.greedy`, `finish_reason`,
                     `NAME.transcript.txt`.

Capture bounds: EVERY prefill forward (all chunked-prefill chunks — positions cover
0..prompt_len-1 contiguously) plus the first `ZSPARK_CAPTURE_DECODE_FORWARDS` (default 32)
decode/verify forwards; generation still runs to its end (tokens/transcript complete);
`forward.total` records kept-vs-total. Clones are spooled to a per-clip scratch dir one
forward×layer chunk at a time and assembled straight into the safetensors file — peak RAM stays
bounded for the ~44k-row hour clip (an unbounded in-RAM capture OOM-killed the host at 32).

Faithfulness notes:

  * `VLLM_ENABLE_V1_MULTIPROCESSING=0` keeps the engine in-process so `llm.apply_model` can reach the
    live `NeMoSpeechLMForConditionalGeneration` (with multiprocessing the model lives in the
    EngineCore child and cannot be hooked). Engine behaviour is unchanged.
  * `enforce_eager=True`: with torch.compile/cudagraphs the backbone runs inside a captured CUDA
    graph and Python forward hooks never fire. Eager decode uses the same kernels.
  * One clip per `llm.chat` call (`max_num_seqs=1`): hooks see exactly that request's forwards.
    The prompt goes through `LLM.chat` with the card's message schema — the plugin's processor
    renders the template, expands the `<|audio|>` placeholder to `audio_placeholders` rows, and
    feeds `embed_multimodal` — identical to `vllm serve`.
  * MTP speculative decode stays on (the card's config); hooks attach to the TARGET model only
    (`model.language_model.model.layers`), never the draft.
  * max_tokens defaults to the remaining context (65536 − prompt_len) so a long clip reaches a
    real EOS/finish_reason rather than an artificial cap.

Weights: ZSPARK_NEMOTRON_NVFP4 (default the HF hub snapshot 66c6adc). Without it every command exits
naming what waits.

  docker run --rm --gpus all -e ZSPARK_HOST_HOME=$HOME -v $HOME:$HOME zspark151-nemo-vllm:local \
      python3 ~/repo/zspark-151-reference/tools/reference/nemotron_transcribe/export_model.py
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

SR = 16000


def _snapshot():
    """The HF hub snapshot 66c6adc under the invoking user's HOME (the container's $HOME is /root
    while the mount is the user's)."""
    import pwd
    home = os.path.expanduser("~")
    for h in (home, pwd.getpwuid(os.getuid()).pw_dir, os.environ.get("ZSPARK_HOST_HOME", "")):
        if h:
            p = os.path.join(h, ".cache/huggingface/hub",
                             "models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
                             "snapshots/66c6adc4020179567074d0399cda2a4eeb2bb347")
            if os.path.exists(os.path.join(p, "config.json")):
                return p
    return os.path.join(home, ".cache/huggingface/hub",
                        "models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
                        "snapshots/66c6adc4020179567074d0399cda2a4eeb2bb347")


CK = os.environ.get("ZSPARK_NEMOTRON_NVFP4", _snapshot())


def _fixtures():
    for h in (os.path.expanduser("~"), os.environ.get("ZSPARK_HOST_HOME", "")):
        if h:
            p = os.path.join(h, ".cache/zspark/scratch-151/fixtures")
            if os.path.exists(os.path.join(p, "manifest.json")):
                return p
    return os.path.expanduser("~/.cache/zspark/scratch-151/fixtures")


FIX = os.environ.get("ZSPARK_REFERENCE_NEMOTRON", _fixtures())
CARD_USER_TEXT = "Transcribe the following audio."
# <|audio|> token id (the tokenizer's added token; the plugin's `_AUDIO_PLACEHOLDER` is the string).
AUDIO_ID = 131072
DECODE_STEPS = 32  # decode/verify forwards whose hiddens/logits are kept (all prefill always kept)
KEEP_DECODE = int(os.environ.get("ZSPARK_CAPTURE_DECODE_FORWARDS", DECODE_STEPS))
CONTEXT = 65536  # the card's max_model_len / serve profile


def get_model(llm):
    """The live NeMoSpeechLMForConditionalGeneration inside the worker (in-process engine only)."""
    models = llm.apply_model(lambda m: m)
    assert len(models) == 1
    return models[0]


def backbone_layers(model):
    """(name, module) per backbone block, in order. NeMoSpeechLMForConditionalGeneration.language_model
    is the family backbone (NemotronHForCausalLM here); its `.model.layers` ModuleList holds the
    stacked blocks (mamba / attention / moe)."""
    import torch.nn as nn

    lm = model.language_model
    layers = getattr(getattr(lm, "model", None), "layers", None)
    if isinstance(layers, nn.ModuleList):
        return [(f"language_model.model.layers.{i}", m) for i, m in enumerate(layers)]
    for name, mod in lm.named_modules():
        if name.split(".")[-1] == "layers" and isinstance(mod, nn.ModuleList):
            return [(f"{name}.{i}", m) for i, m in enumerate(mod)]
    raise RuntimeError("backbone layer list not found under language_model")


class Capture:
    """Spooled per-forward capture. `forwards[i]` = {num_rows, prefill, ids, pos} (small metadata;
    tensors live as {layer:04d}.{part}.pt files under `dir`). A forward is prefill while the
    cumulative row count stays below prompt_len — grounded in vLLM's chunked-prefill scheduling,
    not a heuristic: the sum of scheduled rows before decode equals the prompt length exactly."""

    def __init__(self, spool_dir, keep_decode):
        self.dir = spool_dir
        self.keep_decode = keep_decode
        self.enc = []
        self.forwards = []   # metadata per backbone forward
        self.logits = []     # (kept) compute_logits outputs
        self.logits_top5 = []
        self.rows_seen = 0
        self.decode_seen = 0
        self.cur = None      # in-flight forward index or None if not kept

    def start_forward(self, num_rows):
        is_prefill = self.rows_seen < self.prompt_len
        keep = is_prefill or self.decode_seen < self.keep_decode
        idx = len(self.forwards)
        self.forwards.append({"num_rows": int(num_rows), "prefill": bool(is_prefill),
                              "captured": bool(keep)})
        self.cur = idx if keep else None
        if is_prefill:
            self.rows_seen += num_rows
        else:
            self.decode_seen += 1
        return idx

    def chunk_path(self, layer_i, part):
        return os.path.join(self.dir, f"{layer_i:04d}.{part}.pt")

    prompt_len = 0  # set per clip before generate


def install_hooks(model, cap):
    """Wrap embed_multimodal / compute_logits / the backbone's forward, hook every block, and spool
    kept outputs to cap.dir. Returns (detach, n_layers)."""
    import torch

    fwd_of_layer_call = [None]

    orig_embed = model.embed_multimodal

    def embed_wrap(**kw):
        embs = orig_embed(**kw)
        cap.enc.append([e.detach().to("cpu", torch.bfloat16).clone() for e in embs])
        return embs

    model.embed_multimodal = embed_wrap

    inner = (model.language_model.model
             if hasattr(model.language_model, "model") else model.language_model)
    orig_inner_forward = inner.forward

    def forward_wrap(input_ids=None, positions=None, intermediate_tensors=None,
                     inputs_embeds=None, **kw):
        n = positions.shape[-1] if torch.is_tensor(positions) else (
            input_ids.shape[-1] if torch.is_tensor(input_ids) else 0)
        idx = cap.start_forward(n)
        if cap.forwards[idx]["captured"]:
            cap.forwards[idx]["input_ids"] = (
                input_ids.detach().to("cpu", torch.int32).reshape(-1).clone()
                if torch.is_tensor(input_ids) else None)
            cap.forwards[idx]["positions"] = (
                positions.detach().to("cpu", torch.int32).reshape(-1).clone()
                if torch.is_tensor(positions) else None)
        return orig_inner_forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kw)

    inner.forward = forward_wrap

    def make_hook(i):
        def h(mod, inp, out):
            if cap.cur is None:
                return
            if isinstance(out, (tuple, list)) and len(out) == 2:
                hid, res = out
            else:
                hid, res = out, None
            f = cap.forwards[cap.cur]
            for part, t in (("hidden", hid), ("residual", res)):
                if torch.is_tensor(t):
                    torch.save(t.detach().to("cpu", torch.bfloat16).clone(),
                               cap.chunk_path(i, part) + f".f{cap.cur}")
        return h

    layers = backbone_layers(model)
    handles = [mod.register_forward_hook(make_hook(i)) for i, (_, mod) in enumerate(layers)]

    orig_logits = model.compute_logits

    def logits_wrap(hidden):
        lg = orig_logits(hidden)
        # Kept for prefill forwards' sampling call and the first keep_decode decode steps.
        if lg is not None and (cap.decode_seen == 0 or cap.decode_seen <= cap.keep_decode):
            cap.logits.append(lg.detach().to("cpu", torch.float32).clone())
            top = lg.float().topk(5, dim=-1)
            cap.logits_top5.append((top.indices.cpu().clone(), top.values.cpu().clone()))
        return lg

    model.compute_logits = logits_wrap

    def detach():
        model.embed_multimodal = orig_embed
        model.compute_logits = orig_logits
        inner.forward = orig_inner_forward
        for hd in handles:
            hd.remove()

    return detach, len(layers)

import torch as _torch_mod  # module-level dtype map only; heavy use stays lazy in functions

_ST_DTYPE = {_torch_mod.uint8: "U8", _torch_mod.int8: "I8", _torch_mod.int16: "I16",
             _torch_mod.int32: "I32", _torch_mod.int64: "I64", _torch_mod.float16: "F16",
             _torch_mod.bfloat16: "BF16", _torch_mod.float32: "F32", _torch_mod.float64: "F64",
             _torch_mod.bool: "BOOL"}
_ST_ITEMSIZE = {"U8": 1, "I8": 1, "I16": 2, "I32": 4, "I64": 8, "F16": 2, "BF16": 2,
                "F32": 4, "F64": 8, "BOOL": 1}


class _TensorSource:
    """One safetensors entry: either a single torch tensor (materialized) or a list of chunk
    files whose tensors concatenate along dim 0. Streams bytes without holding the full tensor."""

    def __init__(self, name, spec, stream_from_files=None):
        self.name = name
        self.spec = spec            # {dtype, shape}
        self.files = stream_from_files  # [(path, None)] means torch.load per file
        self.tensor = None if stream_from_files else spec.pop("tensor")

    def numel(self):
        n = 1
        for d in self.spec["shape"]:
            n *= d
        return n

    def nbytes(self):
        return self.numel() * _ST_ITEMSIZE[self.spec["dtype"]]

    def stream(self, out):
        """Append this tensor's dense little-endian bytes to `out`, one chunk (or one whole
        small tensor) at a time — peak extra RAM = one chunk."""
        import torch
        if self.tensor is not None:
            out.write(self._bytes(self.tensor))
            return
        for path, _ in self.files:
            t = torch.load(path, map_location="cpu")
            if t.dim() == 1:
                t = t.unsqueeze(0)
            out.write(self._bytes(t))
            del t

    @staticmethod
    def _bytes(t):
        import torch
        t = t.contiguous()
        if t.dtype == torch.bfloat16:
            t = t.view(torch.int16)
        elif t.dtype == torch.bool:
            t = t.view(torch.uint8)
        return t.numpy().tobytes()


def assemble(cap, out_path, meta):
    """Serialize all captured fixtures to safetensors, streaming per-tensor bytes.
    Layout: u64 header_len LE | JSON header | dense tensor bytes (data_offsets per entry).
    RAM stays bounded by one chunk (or one small metadata tensor) at a time."""
    import struct
    import torch

    kept = [i for i, f in enumerate(cap.forwards) if f["captured"]]
    sources = []  # ordered _TensorSource list

    def add_tensor(name, t):
        sources.append(_TensorSource(name, {"dtype": _ST_DTYPE[t.dtype],
                                            "shape": list(t.shape), "tensor": t}))

    for name, t in meta["base"].items():
        add_tensor(name, t)
    add_tensor("forward.num_rows", torch.tensor([cap.forwards[i]["num_rows"] for i in kept],
                                                dtype=torch.int32))
    add_tensor("forward.prefill", torch.tensor([1 if cap.forwards[i]["prefill"] else 0
                                                for i in kept], dtype=torch.int32))
    add_tensor("forward.total", torch.tensor([len(cap.forwards)], dtype=torch.int32))
    ids = [cap.forwards[i]["input_ids"] for i in kept
           if cap.forwards[i].get("input_ids") is not None]
    pos = [cap.forwards[i]["positions"] for i in kept
           if cap.forwards[i].get("positions") is not None]
    if ids:
        add_tensor("forward.input_ids", torch.cat(ids))
    if pos:
        add_tensor("forward.positions", torch.cat(pos))

    n_layers = meta["n_layers"]
    for i in range(n_layers):
        for part in ("hidden", "residual"):
            files = []
            rows = 0
            cols = None
            for fi in kept:
                p = cap.chunk_path(i, part) + f".f{fi}"
                if os.path.exists(p):
                    files.append((p, None))
            if not files:
                continue
            # Row count from forward metadata; width from the first chunk on disk.
            rows = sum(cap.forwards[fi]["num_rows"] for fi in kept
                       if os.path.exists(cap.chunk_path(i, part) + f".f{fi}"))
            t0 = torch.load(files[0][0], map_location="cpu")
            cols = int(t0.shape[-1])
            del t0
            sources.append(_TensorSource(
                f"{part}.L{i:02d}", {"dtype": "BF16", "shape": [rows, cols]},
                stream_from_files=files))
    for j, lg in enumerate(cap.logits):
        add_tensor(f"logits.step{j:02d}", lg)
    if cap.logits_top5:
        add_tensor("logits.top5_ids", torch.cat([t[0] for t in cap.logits_top5]))
        add_tensor("logits.top5_vals", torch.cat([t[1] for t in cap.logits_top5]))

    # Header pass: compute each entry's data_offsets.
    header = {}
    off = 0
    for s in sources:
        nbytes = s.nbytes()
        header[s.name] = {"dtype": s.spec["dtype"], "shape": s.spec["shape"],
                          "data_offsets": [off, off + nbytes]}
        off += nbytes
    if meta.get("file_meta"):
        header["__metadata__"] = {k: str(v) for k, v in meta["file_meta"].items()}
    hjson = json.dumps(header).encode("utf-8")
    pad = (8 - len(hjson) % 8) % 8  # keep data section 8-aligned like the reference writer
    hjson += b" " * pad
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(hjson)))
        out.write(hjson)
        for s in sources:
            s.stream(out)
    return [s.name for s in sources]




def cmd_export(args):
    import torch

    ck = args.ck
    if not os.path.exists(os.path.join(ck, "config.json")):
        raise SystemExit(f"weights wait for WS-00: no checkpoint at {ck}")

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_PLUGINS", "nemo_speechlm")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=ck,
        served_model_name="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
        trust_remote_code=True,
        dtype="bfloat16",
        kv_cache_dtype="fp8",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=CONTEXT,
        max_num_batched_tokens=4096,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        use_replayssm=True,
        mamba_backend="flashinfer",
        mamba_cache_mode="none",
        mm_processor_cache_gb=0,
        async_scheduling=False,
        reasoning_parser="nemotron_v3",
        limit_mm_per_prompt={"audio": {"count": 1, "length": 57600000}},
        speculative_config={"method": "mtp", "num_speculative_tokens": 8},
        enforce_eager=True,  # hooks do not fire inside CUDA graphs; numerics are the same kernels
    )
    model = get_model(llm)
    man = json.load(open(os.path.join(FIX, "manifest.json")))
    names = args.only or list(man["clips"])
    os.makedirs(args.out, exist_ok=True)
    spool_root = args.spool or os.path.join(args.out, ".spool")
    os.makedirs(spool_root, exist_ok=True)
    try:
        for name in names:
            info = man["clips"][name]
            cap = Capture(tempfile.mkdtemp(prefix=name + ".", dir=spool_root), KEEP_DECODE)
            detach, n_layers = install_hooks(model, cap)
            try:
                import base64
                audio_b64 = base64.b64encode(
                    open(os.path.join(FIX, "clips", name + ".flac"), "rb").read()).decode()
                # Prompt length = audio placeholders + the non-audio ids of the rendered
                # template fixture (prompt.chat.json's own tokenizer output — not a magic
                # constant). Asserted equal to len(prompt_token_ids) below.
                prompt_ids_fixture = json.load(
                    open(os.path.join(FIX, "prompt.chat.json")))["renders"][
                        "enable_thinking_false"]["ids"]
                text_ids = [t for t in prompt_ids_fixture if t != AUDIO_ID]
                assert len(text_ids) == len(prompt_ids_fixture) - 1, (
                    "fixture must contain exactly one audio placeholder")
                est_prompt = info["audio_placeholders"] + len(text_ids)
                cap.prompt_len = est_prompt
                max_tokens = max(1, CONTEXT - est_prompt)  # full remaining context, real EOS
                params = SamplingParams(temperature=0, max_tokens=max_tokens, seed=0,
                                        skip_special_tokens=False)
                outs = llm.chat(
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": CARD_USER_TEXT},
                        {"type": "audio_url",
                         "audio_url": {"url": f"data:audio/flac;base64,{audio_b64}"}},
                    ]}],
                    sampling_params=params,
                    chat_template_kwargs={"enable_thinking": False},
                    use_tqdm=False,
                )
            finally:
                detach()
            out0 = outs[0].outputs[0]
            pids = list(getattr(outs[0], "prompt_token_ids", None) or [])
            assert len(pids) == est_prompt or not pids, (
                f"{name}: prompt_len {len(pids)} vs estimate {est_prompt}")
            if pids:
                cap.prompt_len = len(pids)
            # Coverage assert: captured prefill positions must tile 0..prompt_len-1 exactly —
            # the guard that caught en_3500s's truncated fixture in run 4.
            captured_pos = sorted(
                int(p) for f in cap.forwards if f["captured"] and f.get("positions") is not None
                for p in f["positions"].tolist())
            prefill_pos = [p for p in captured_pos if p < cap.prompt_len]
            assert prefill_pos == list(range(cap.prompt_len)), (
                f"{name}: prefill positions cover {len(prefill_pos)}/{cap.prompt_len}")
            base = {"tokens.greedy": torch.tensor(list(out0.token_ids), dtype=torch.int32),
                    "prompt.ids": torch.tensor(pids, dtype=torch.int32)}
            enc_items = [e for call in cap.enc for e in call]
            if enc_items:
                assert len(enc_items) == 1, f"{name}: {len(enc_items)} audio items, want 1"
                base["enc.out"] = enc_items[0]
                ph = [i for i, t in enumerate(pids) if t == AUDIO_ID]
                if ph:
                    assert len(ph) == enc_items[0].shape[0], (
                        f"{name}: {len(ph)} placeholders vs enc rows {enc_items[0].shape[0]}")
                    base["audio.placeholder_rows"] = torch.tensor(ph, dtype=torch.int32)
            meta = {"base": base, "n_layers": n_layers,
                    "file_meta": {"clip": name, "ck": ck,
                                  "vllm": __import__("vllm").__version__,
                                  "seconds": str(info["seconds"]),
                                  "audio_placeholders": str(info["audio_placeholders"]),
                                  "num_forwards": str(len(cap.forwards)),
                                  "finish_reason": str(getattr(out0, "finish_reason", "")),
                                  "stop_reason": str(getattr(out0, "stop_reason", "")),
                                  "decode_capture": str(KEEP_DECODE)}}
            keys = assemble(cap, os.path.join(args.out, name + ".safetensors"), meta)
            shutil.rmtree(cap.dir, ignore_errors=True)
            with open(os.path.join(args.out, name + ".transcript.txt"), "w") as f:
                f.write(out0.text)
            npf = sum(1 for f in cap.forwards if f["prefill"])
            print(name, info["seconds"], "s:", len(out0.token_ids), "tokens,",
                  f"{npf} prefill +{len(cap.forwards) - npf} decode forwards "
                  f"({sum(1 for f in cap.forwards if f['captured'])} kept),",
                  len(cap.logits), "logit steps, finish:", getattr(out0, "finish_reason", "?"),
                  ";", repr(out0.text[:60]), flush=True)
    finally:
        if args.spool is None:  # only remove the default root we may have created/emptied
            try:
                os.rmdir(spool_root)
            except OSError:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ck", default=CK)
    ap.add_argument("--out", default=os.path.join(FIX, "model"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--spool", help="scratch dir for spooled captures (default OUT/.spool)")
    a = ap.parse_args()
    cmd_export(a)