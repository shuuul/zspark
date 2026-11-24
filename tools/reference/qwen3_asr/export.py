#!/usr/bin/env python3
"""Qwen3-ASR-1.7B and 0.6B reference fixtures for zspark (spec 138, WS-01; 0.6B spec 140 WS-09).

Runs the `qwen-asr` package's transformers backend (bf16, GPU, batch 1, greedy) on speech clips and writes what
zspark's audio layer, audio tower and language model are compared against. Python only here, never at runtime;
the environment is `env.sh` beside this file. Run it under the locks of specs/WORKER.md:

  PY=$(tools/reference/qwen3_asr/env.sh)
  flock -s ~/.cache/zspark/machine.lock flock ~/.cache/zspark/gpu.lock \\
      "$PY" -I tools/reference/qwen3_asr/export.py fixtures --src DL_DIR      # the 8 clips x scenarios
  ... export.py wer --src DL_DIR                                              # the >= 50-clip WER/CER set
  ... export.py tied                                                          # embed_tokens vs lm_head
  ... export.py attn                                                          # windowed vs the shipped full audio attention
  ... export.py short                                                         # clips under 1 s, cut from the 5 s ones

DL_DIR holds `ls/LibriSpeech/test-clean` (openslr.org/resources/12, CC BY 4.0) and `fl/test/*.wav` with
`fl_test.tsv` (google/fleurs cmn_hans_cn test, CC BY 4.0). Output: ~/.cache/zspark/reference-qwen3-asr/
(`ZSPARK_REFERENCE_QWEN3_ASR`). README.md beside this file says how every file reads.

Audio attention. The shipped transformers backend gives the audio encoder's attention no mask under sdpa or eager:
`Qwen3ASRAudioEncoder.forward` builds `cu_seqlens` and hands it to the attention only as FlashAttention-2 varlen
arguments, so without flash-attn (not installable here) every token attends to the whole clip. The model is
meant to attend inside blocks of 104 tokens (the FA2 path, vLLM and SGLang). `--audio-attn windowed` (default,
the reference) registers an attention function that runs sdpa per `cu_seqlens` block, which is what varlen FA2
computes; `full` is the backend as it ships. They agree for clips of at most 104 audio tokens (about 8 s).
"""
import argparse, glob, hashlib, json, os, random, sys, time
os.environ.setdefault("TRITON_CACHE_DIR", os.path.expanduser("~/.cache/zspark/qwen3-asr-env/triton"))  # ~/.triton may be root-owned
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # torch 2.14 routes some ops (bmm outer product, norm, topk) to triton kernels; the reference uses ATen/cuBLAS
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as _save_file


def save_file(tensors, path, metadata=None):
    """Atomic: tests/audio reads every *.safetensors here, so a file appears whole (temp name, then rename)."""
    _save_file(tensors, path + ".tmp", metadata=metadata)
    os.replace(path + ".tmp", path)

OUT = os.path.expanduser(os.environ.get("ZSPARK_REFERENCE_QWEN3_ASR", "~/.cache/zspark/reference-qwen3-asr"))
MODEL = os.path.expanduser(os.environ.get("ZSPARK_QWEN3_ASR_MODEL", "~/models/Qwen3-ASR-1.7B"))
SR = 16000
ENC_LAYERS = [0, 5, 11, 17, 23]  # encoder layers whose outputs are stored: those the tower has, and its last


def enc_layers(depth):
    """1.7B's 24 layers: 0, 5, 11, 17, 23; 0.6B's 18: 0, 5, 11, 17."""
    return sorted({L for L in ENC_LAYERS if L < depth} | {depth - 1})


def weights_digest():
    """sha256 of the shard index, or of a single-file checkpoint's safetensors header (0.6B ships one file)."""
    index = MODEL + "/model.safetensors.index.json"
    if os.path.exists(index):
        return hashlib.sha256(open(index, "rb").read()).hexdigest()
    with open(MODEL + "/model.safetensors", "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return hashlib.sha256(f.read(n)).hexdigest()
MAX_NEW = 4096  # the package's vLLM default; its transformers default (512) cuts a 600 s transcript


# ---------------------------------------------------------------- clips
def ls_chapters(src):
    root = os.path.join(src, "ls/LibriSpeech/test-clean")
    out = {}
    for tr in sorted(glob.glob(root + "/*/*/*.trans.txt")):
        d = os.path.dirname(tr)
        utts = []
        for line in open(tr):
            uid, text = line.strip().split(" ", 1)
            utts.append((os.path.join(d, uid + ".flac"), text))
        out[os.path.relpath(d, root)] = utts
    return out


def fleurs_utts(src):
    seen, out = set(), []
    for line in open(os.path.join(src, "fl_test.tsv"), encoding="utf-8"):
        c = line.rstrip("\n").split("\t")
        raw = c[2]
        if raw in seen:  # the same sentence by several speakers: keep the first
            continue
        seen.add(raw)
        out.append((os.path.join(src, "fl/test", c[1]), raw, int(c[5]) / SR))
    return out


def cat(items, sep=" "):
    """items: [(path, text)] -> (int16 samples at 16 kHz, text, [source ids])."""
    pcm, ids = [], []
    for p, t in items:
        x, sr = sf.read(p, dtype="float32")  # LibriSpeech: int16 FLAC; FLEURS: 32-bit float WAV
        assert sr == SR and x.ndim == 1, p
        pcm.append(np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16))  # exact for the int16 sources
    return np.concatenate(pcm), sep.join(t for _, t in items), [os.path.basename(p) for p, _ in items]


def take(utts, target, start=0):
    got, tot = [], 0.0
    for p, t in utts[start:]:
        got.append((p, t))
        tot += sf.info(p).duration
        if tot >= target:
            break
    return got


def fixture_clips(src):
    ch = ls_chapters(src)
    fl = [(p, t, d) for p, t, d in fleurs_utts(src)]
    rng = random.Random(1381)
    zh_order = fl[:]
    rng.shuffle(zh_order)
    zh_items = [(p, t) for p, t, _ in zh_order]
    clips = {}
    allutts = [(p, t) for u in ch.values() for p, t in u]
    best = min(allutts, key=lambda u: abs(sf.info(u[0]).duration - 5.0) + (0.5 if sf.info(u[0]).frames % 1600 == 0 else 0))
    clips["en_5s"] = ([best], "LibriSpeech test-clean " + os.path.basename(best[0]))
    clips["en_30s"] = (take(ch["2830/3980"], 30), "LibriSpeech test-clean 2830/3980 from its first utterance")
    clips["en_140s"] = (take(ch["1320/122617"], 140), "LibriSpeech test-clean 1320/122617 from its first utterance")
    big = ch["5639/40744"] + take(ch["1188/133604"], 120)
    clips["en_600s"] = (take(big, 600), "LibriSpeech test-clean 5639/40744 whole, then 1188/133604 (two speakers)")
    z5 = min(fl, key=lambda u: abs(u[2] - 5.0))
    clips["zh_5s"] = ([(z5[0], z5[1])], "FLEURS cmn_hans_cn test " + os.path.basename(z5[0]))
    for n, t in (("zh_30s", 30), ("zh_140s", 140), ("zh_600s", 600)):
        clips[n] = (take(zh_items, t), f"FLEURS cmn_hans_cn test, {len(take(zh_items, t))} sentences in seeded order (1381), different speakers joined")
    return clips


def wer_clips(src):
    """>= 50 clips, 5 s to 600 s, seeded; returns [(id, lang, items, source)]."""
    ch = ls_chapters(src)
    rng = random.Random(1382)
    allutts = [(p, t) for u in ch.values() for p, t in u]
    dur = {p: sf.info(p).duration for p, _ in allutts}
    short = [u for u in allutts if 5 <= dur[u[0]] <= 15]
    rng.shuffle(short)
    chaps = sorted(ch)
    out = []
    for i, u in enumerate(short[:14]):
        out.append((f"en_s{i:02d}", "en", [u], "LibriSpeech test-clean " + os.path.basename(u[0])))
    cdur = {c: sum(dur[p] for p, _ in u) for c, u in ch.items()}
    def run(lang, name, target, k):
        cands = [c for c in chaps if cdur[c] >= target * 1.4]
        c = rng.choice(cands)
        items = take(ch[c], target, start=rng.randrange(0, max(1, int(len(ch[c]) * 0.25))))
        return (name, lang, items, f"LibriSpeech test-clean {c}")
    for i in range(8): out.append(run("en", f"en_m{i:02d}", 30, i))
    for i in range(5): out.append(run("en", f"en_l{i:02d}", 60, i))
    for i in range(3): out.append(run("en", f"en_x{i:02d}", 140, i))
    out.append(run("en", "en_t00", 300, 0))
    big = ch["4507/16021"] + take(ch["8455/210777"], 130)
    out.append(("en_u00", "en", take(big, 600), "LibriSpeech test-clean 4507/16021 then 8455/210777"))
    fl = fleurs_utts(src)
    zh = [(p, t) for p, t, _ in fl]
    rng.shuffle(zh)
    pos = 0
    def zrun(name, target):
        nonlocal pos
        items = take(zh, target, start=pos)
        pos += len(items)
        return (name, "zh", items, f"FLEURS cmn_hans_cn test, {len(items)} sentences joined")
    zshort = [u for u in zh if 5 <= next(d for p, _, d in fl if p == u[0]) <= 15][:14]
    for i, u in enumerate(zshort): out.append((f"zh_s{i:02d}", "zh", [u], "FLEURS cmn_hans_cn test " + os.path.basename(u[0])))
    pos = 0
    zh = [u for u in zh if u not in zshort]
    for i in range(8): out.append(zrun(f"zh_m{i:02d}", 30))
    for i in range(5): out.append(zrun(f"zh_l{i:02d}", 60))
    for i in range(3): out.append(zrun(f"zh_x{i:02d}", 140))
    out.append(zrun("zh_t00", 300))
    out.append(zrun("zh_u00", 600))
    return out


# ---------------------------------------------------------------- model
SEEN_CU = []  # the cu_seqlens the tower itself built, last layer call


def load_asr(audio_attn):
    import qwen_asr  # registers qwen3_asr with the Auto classes
    from qwen_asr import Qwen3ASRModel
    from transformers.modeling_utils import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def windowed_sdpa(module, q, k, v, attention_mask, dropout=0.0, scaling=None, is_causal=False, cu_seq_lens_q=None, **kw):
        # q, k, v: [1, heads, S, 64]; one sdpa per cu_seqlens block = FlashAttention-2 varlen, non-causal.
        b = cu_seq_lens_q.tolist()
        SEEN_CU[:] = b
        outs = [F.scaled_dot_product_attention(q[:, :, a:z], k[:, :, a:z], v[:, :, a:z], scale=scaling)
                for a, z in zip(b[:-1], b[1:])]
        return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None

    AttentionInterface.register("windowed_sdpa", windowed_sdpa)
    asr = Qwen3ASRModel.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa", max_new_tokens=MAX_NEW)
    tower = asr.model.thinker.audio_tower
    if audio_attn == "windowed":
        tower.config._attn_implementation = "windowed_sdpa"
    asr.audio_attn = audio_attn
    return asr


def enc_geometry(n_mel):
    """chunk lengths, tokens per chunk, cu_seqlens (token offsets of the attention blocks): the tower's rules."""
    cl = [100] * (n_mel // 100) + ([n_mel % 100] if n_mel % 100 else [])
    def out(n):
        f = (n % 100 - 1) // 2 + 1
        return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13
    toks = [out(n) for n in cl]
    win = 13 * (800 // 100)  # the padded chunk width after the convs is always 13 here (>= 1 full chunk) times 8 chunks
    total = out(n_mel)
    cu, left = [0], total
    while left > 0:
        s = min(win, left)
        cu.append(cu[-1] + s)
        left -= s
    return cl, toks, cu, total


def run_clip(asr, wav_path, context, language, name, scenario, full):
    """One clip x one scenario. Returns (tensors, info). full=False: transcript only (the WER set)."""
    from qwen_asr.inference.utils import load_audio_any, normalize_audio_input, parse_asr_output, normalize_language_name
    dev = asr.model.device
    src, sr0 = load_audio_any(wav_path)
    pcm = normalize_audio_input(wav_path)
    lang = normalize_language_name(language) if language else None
    text = asr._build_text_prompt(context=context, force_language=lang)
    inputs = asr.processor(text=[text], audio=[pcm], return_tensors="pt", padding=True)
    mel32 = inputs["input_features"].clone()
    inputs = inputs.to(dev).to(asr.model.dtype)  # as the package: the mel is cast to bf16 before the tower
    ids = inputs["input_ids"][0]
    n_mel = int(inputs["feature_attention_mask"].sum())
    cl, toks, cu, total = enc_geometry(n_mel)
    pad_id = asr.model.config.thinker_config.audio_token_id
    n_pad = int((ids == pad_id).sum())
    assert n_pad == total == cu[-1], (n_pad, total, cu[-1])
    seq = ids.shape[0]
    t0 = time.time()
    with torch.no_grad():
        g = asr.model.generate(**inputs, max_new_tokens=MAX_NEW, output_logits=full)
    if asr.audio_attn == "windowed":
        assert SEEN_CU == cu, "cu_seqlens of the tower differ from enc_geometry"
    dt = time.time() - t0
    new = g.sequences[0, seq:]
    raw = asr.processor.batch_decode(new[None], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    plang, ptext = parse_asr_output(raw, user_language=lang)
    info = dict(name=name, scenario=scenario, context=context, forced_language=lang, prompt_text=text,
                src_sr=sr0, src_shape=list(src.shape), n_samples_16k=int(pcm.shape[0]), n_mel=n_mel, n_audio_tokens=n_pad,
                prompt_len=seq, n_new=int(new.shape[0]), raw_text=raw,
                raw_text_with_special=asr.processor.batch_decode(new[None], skip_special_tokens=False)[0],
                parsed_language=plang, parsed_text=ptext, generate_seconds=round(dt, 2),
                audio_attn=asr.audio_attn, eos_hit=bool(new[-1].item() in (151645, 151643)))
    if not full:
        return None, info, new
    T = {}
    T["pcm.src"] = torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32))
    T["pcm.16k"] = torch.from_numpy(pcm.astype(np.float32))
    T["mel.f32"] = mel32[0, :, :n_mel].contiguous()
    T["mel.bf16"] = inputs["input_features"][0, :, :n_mel].contiguous().cpu()
    T["enc.chunk_lengths"] = torch.tensor(cl, dtype=torch.int32)
    T["enc.tokens_per_chunk"] = torch.tensor(toks, dtype=torch.int32)
    T["enc.cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
    T["prompt.ids"] = ids.to(torch.int32).cpu()
    T["prompt.audio_pad_first"] = torch.tensor([int((ids == pad_id).nonzero()[0])], dtype=torch.int32)
    T["tokens.greedy"] = new.to(torch.int32).cpu()  # to EOS (or MAX_NEW), EOS included
    lg = torch.stack([l[0] for l in g.logits]).float()  # [n_new, V]
    top = lg.topk(5, dim=-1)
    T["logits.top5_ids"] = top.indices.to(torch.int32).cpu()
    T["logits.top5_vals"] = top.values.cpu()
    T["logits.first33"] = lg[:33].cpu()
    del g, lg
    return T, info, new


def export_clip(asr, wav_path, context, language, name, scenario, outdir, keep_enc, conv_stages):
    """The full fixture for one scenario: generate (logits) + a hooked 33-token generate (encoder, hidden)."""
    from qwen_asr.inference.utils import normalize_audio_input, normalize_language_name
    T, info, new = run_clip(asr, wav_path, context, language, name, scenario, full=True)
    dev = asr.model.device
    lang = normalize_language_name(language) if language else None
    pcm = normalize_audio_input(wav_path)
    text = asr._build_text_prompt(context=context, force_language=lang)
    inputs = asr.processor(text=[text], audio=[pcm], return_tensors="pt", padding=True).to(dev).to(asr.model.dtype)
    seq = inputs["input_ids"].shape[1]
    tower = asr.model.thinker.audio_tower
    cap, hooks = {}, []
    def hk(key, fn=lambda o: o):
        def f(m, a, o):
            cap[key] = fn(o).detach().cpu()
        return f
    gelu = lambda o: F.gelu(o)
    if conv_stages:
        for i, c in enumerate((tower.conv2d1, tower.conv2d2, tower.conv2d3), 1):
            hooks.append(c.register_forward_hook(hk(f"enc.conv{i}_gelu", gelu)))
        hooks.append(tower.conv_out.register_forward_hook(hk("enc.conv_out")))
    hooks.append(tower.layers[0].register_forward_pre_hook(lambda m, a: cap.__setitem__("enc.after_conv_pos", a[0].detach().cpu())))
    for L in enc_layers(len(tower.layers)):
        hooks.append(tower.layers[L].register_forward_hook(hk(f"enc.layer{L:02d}", lambda o: o[0] if isinstance(o, tuple) else o)))
    hooks.append(tower.ln_post.register_forward_hook(hk("enc.ln_post")))
    hooks.append(tower.proj1.register_forward_hook(hk("enc.proj1")))
    hooks.append(tower.proj2.register_forward_hook(hk("enc.proj2")))
    with torch.no_grad():
        g = asr.model.generate(**inputs, max_new_tokens=33, output_hidden_states=True, output_logits=True)
    for h in hooks: h.remove()
    n33 = g.sequences.shape[1] - seq
    assert torch.equal(g.sequences[0, seq:].cpu(), new[:n33].cpu()), "33-token run diverged from the full run"
    T.update({k: v.contiguous() for k, v in cap.items()})
    hs = g.hidden_states  # tuple over steps of tuple over 29 states [1, s, 2048]; the last state is after the final norm
    pre = hs[0]
    tail = seq <= 1024
    T["lm.prefill.rows_from"] = torch.tensor([0 if tail else seq - 32], dtype=torch.int32)
    for L, h in enumerate(pre):
        x = h[0].cpu()
        T[f"lm.prefill.hidden.{L:02d}"] = (x if (tail or L == 0) else x[-32:]).contiguous()
    if len(hs) > 1:
        T["lm.decode.hidden"] = torch.stack([torch.stack([h[0, -1].cpu() for h in step]) for step in hs[1:]]).contiguous()  # [steps, 29, 2048]
    lg33 = torch.stack([l[0] for l in g.logits]).float().cpu()
    ref = T["logits.first33"]
    info["logits33_max_abs_diff_between_runs"] = float((lg33[: ref.shape[0]] - ref[: lg33.shape[0]]).abs().max())
    info["n_decode_steps_with_hidden"] = len(hs) - 1
    info["prefill_hidden_rows"] = "all" if tail else "layer 00 all, others last 32"
    meta = {k: str(v) for k, v in provenance(asr).items()}
    meta["name"], meta["scenario"] = name, scenario
    save_file(T, f"{outdir}/{name}.{scenario}.safetensors", metadata=meta)
    json.dump(info, open(f"{outdir}/{name}.{scenario}.json", "w"), ensure_ascii=False, indent=1)
    return info


def provenance(asr):
    import transformers, librosa, qwen_asr
    return dict(model=MODEL, model_files_sha256_index=weights_digest(),
                qwen_asr="0.0.6", transformers=transformers.__version__, torch=torch.__version__, librosa=librosa.__version__,
                gpu=torch.cuda.get_device_name(0), dtype="bfloat16", backend="qwen-asr transformers", attn_lm="sdpa",
                attn_audio=asr.audio_attn, max_new_tokens=MAX_NEW, resampler="librosa.resample res_type=soxr_hq (default)")


def write_flac(path, pcm16, sr=SR, ch=None):
    sf.write(path, pcm16, sr, subtype="PCM_16", format="FLAC")


# ---------------------------------------------------------------- commands
def cmd_tied(a):
    from safetensors import safe_open
    index = MODEL + "/model.safetensors.index.json"
    idx = json.load(open(index))["weight_map"] if os.path.exists(index) else {}
    ks = ["thinker.model.embed_tokens.weight", "thinker.lm_head.weight"]
    t = {}
    for k in ks:
        with safe_open(f"{MODEL}/{idx.get(k, 'model.safetensors')}", "pt") as f:
            t[k] = f.get_tensor(k)
    eq = torch.equal(t[ks[0]].view(torch.int16), t[ks[1]].view(torch.int16))
    print(json.dumps({"bitwise_equal": eq, "shape": list(t[ks[0]].shape)}))


SCEN_FIXTURES = [  # (clip, scenario, language, context)
    ("en_5s", "free", None, ""), ("en_30s", "free", None, ""), ("en_140s", "free", None, ""), ("en_600s", "free", None, ""),
    ("zh_5s", "free", None, ""), ("zh_30s", "free", None, ""), ("zh_140s", "free", None, ""), ("zh_600s", "free", None, ""),
    ("en_30s", "forced", "English", ""), ("zh_30s", "forced", "Chinese", ""),
    ("en_30s", "context", None, "This is an audiobook recording read aloud by a volunteer."),
    ("zh_30s", "context", None, "这是一段普通话朗读录音。"),
]


def cmd_fixtures(a):
    os.makedirs(OUT + "/clips", exist_ok=True)
    clips = fixture_clips(a.src)
    manifest = {"clips": {}, "scenarios": []}
    for n, (items, source) in clips.items():
        pcm, text, ids = cat(items)
        write_flac(f"{OUT}/clips/{n}.flac", pcm)
        manifest["clips"][n] = dict(source=source, files=ids, seconds=round(len(pcm) / SR, 3), ground_truth=text)
    # resampler fixtures: derived, not recorded at those rates
    import librosa
    x, _ = sf.read(f"{OUT}/clips/en_5s.flac", dtype="float32")
    st = np.stack([librosa.resample(x, orig_sr=SR, target_sr=48000), 0.5 * librosa.resample(x, orig_sr=SR, target_sr=48000)], axis=1)
    sf.write(f"{OUT}/clips/rs_en_5s_48k_stereo.flac", st, 48000, subtype="PCM_16", format="FLAC")
    z, _ = sf.read(f"{OUT}/clips/zh_5s.flac", dtype="float32")
    sf.write(f"{OUT}/clips/rs_zh_5s_44k1_mono.flac", librosa.resample(z, orig_sr=SR, target_sr=44100), 44100, subtype="PCM_16", format="FLAC")
    for n, base in (("rs_en_5s_48k_stereo", "en_5s"), ("rs_zh_5s_44k1_mono", "zh_5s")):
        manifest["clips"][n] = dict(source=f"derived from {base}: librosa soxr_hq upsampled, stereo = [x, 0.5 x] for the 48k one, PCM_16 FLAC; tests decode, downmix and the 16 kHz resampler",
                                    seconds=manifest["clips"][base]["seconds"], ground_truth=manifest["clips"][base]["ground_truth"])
    asr = load_asr(a.audio_attn)
    pv = provenance(asr)
    manifest["provenance"] = {k: str(v) for k, v in pv.items()}
    from qwen_asr.inference.utils import load_audio_any, normalize_audio_input
    for n in ("rs_en_5s_48k_stereo", "rs_zh_5s_44k1_mono"):  # audio-only
        p = f"{OUT}/clips/{n}.flac"
        src, sr0 = load_audio_any(p)
        pcm = normalize_audio_input(p)
        mel = asr.processor.feature_extractor(pcm, sampling_rate=SR, padding=True, truncation=False, return_tensors="pt")["input_features"][0]
        T = {"pcm.src": torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32)), "pcm.16k": torch.from_numpy(pcm.astype(np.float32)),
             "mel.f32": mel[:, : len(pcm) // 160].contiguous()}
        save_file(T, f"{OUT}/{n}.audio.safetensors", metadata={**{k: str(v) for k, v in pv.items()}, "src_sr": str(sr0), "src_shape": str(list(src.shape))})
    for (n, sc, lang, ctx) in SCEN_FIXTURES:
        if a.only and n not in a.only: continue
        p = f"{OUT}/clips/{n}.flac"
        t0 = time.time()
        info = export_clip(asr, p, ctx, lang, n, sc, OUT, keep_enc=True, conv_stages=n.endswith("_5s"))
        info["gt_text"] = manifest["clips"][n]["ground_truth"]
        json.dump(info, open(f"{OUT}/{n}.{sc}.json", "w"), ensure_ascii=False, indent=1)
        manifest["scenarios"].append(dict(clip=n, scenario=sc, n_audio_tokens=info["n_audio_tokens"], n_new=info["n_new"], seconds=round(time.time() - t0, 1)))
        print(n, sc, info["n_audio_tokens"], info["n_new"], info["parsed_language"], repr(info["parsed_text"][:80]), flush=True)
        torch.cuda.empty_cache()
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)


# Clips under one chunk (100 mel frames): the package runs a lone chunk unpadded (`split_audio_into_chunks` pads
# only a split clip's short pieces), so its convolutions see their zero border right past the clip.
SHORT = (("en_short", "en_5s", 0.6), ("zh_short", "zh_5s", 0.35))


def cmd_short(a):
    """Fixtures of clips under 1 s, cut from the 5 s ones (`fixtures` first)."""
    asr = load_asr(a.audio_attn)
    for n, base, sec in SHORT:
        pcm, sr = sf.read(f"{OUT}/clips/{base}.flac", dtype="int16")
        assert sr == SR
        write_flac(f"{OUT}/clips/{n}.flac", pcm[: int(sec * SR)])
        info = export_clip(asr, f"{OUT}/clips/{n}.flac", "", None, n, "free", OUT, keep_enc=True, conv_stages=False)
        print(n, info["n_mel"], info["n_audio_tokens"], repr(info["parsed_text"]), flush=True)


def cmd_attn(a):
    """Cosine of the tower's proj2 output, windowed vs the shipped full attention, and the transcripts."""
    asr = load_asr("windowed")
    tower = asr.model.thinker.audio_tower
    res = []
    for n in ("en_5s", "en_30s", "en_140s", "zh_30s", "zh_600s"):
        p = f"{OUT}/clips/{n}.flac"
        outs = {}
        for mode, impl in (("windowed", "windowed_sdpa"), ("full", "sdpa")):
            tower.config._attn_implementation = impl; asr.audio_attn = mode
            cap = {}
            h = tower.proj2.register_forward_hook(lambda m, i, o: cap.__setitem__("x", o.detach().float().cpu()))
            _, info, _ = run_clip(asr, p, "", None, n, mode, full=False)
            h.remove()
            outs[mode] = (cap["x"], info["raw_text"])
        a_, b_ = outs["windowed"][0], outs["full"][0]
        cos = F.cosine_similarity(a_, b_, dim=-1)
        res.append(dict(clip=n, tokens=a_.shape[0], cos_mean=float(cos.mean()), cos_min=float(cos.min()), max_abs=float((a_ - b_).abs().max()),
                        same_transcript=outs["windowed"][1] == outs["full"][1], windowed=outs["windowed"][1][:120], full=outs["full"][1][:120]))
        print(json.dumps(res[-1], ensure_ascii=False), flush=True)
    json.dump(res, open(f"{OUT}/attn_windowed_vs_full.json", "w"), ensure_ascii=False, indent=1)


def cmd_wer(a):
    os.makedirs(OUT + "/wer/clips", exist_ok=True)
    clips = wer_clips(a.src)
    asr = load_asr("windowed")
    full = None
    rows = []
    rng = random.Random(1383)
    for i, (name, lg, items, source) in enumerate(clips):
        pcm, gt, ids = cat(items)
        p = f"{OUT}/wer/clips/{name}.flac"
        write_flac(p, pcm)
        mode = "free"
        language, context = None, ""
        if i % 4 == 3:
            mode, language = "forced", ("English" if lg == "en" else "Chinese")
        elif i % 8 == 1:
            mode, context = "context", ("This is an audiobook recording read aloud by a volunteer." if lg == "en" else "这是一段普通话朗读录音。")
        _, info, _ = run_clip(asr, p, context, language, name, mode, full=False)
        row = dict(id=name, lang=lg, mode=mode, forced_language=language, context=context, seconds=round(len(pcm) / SR, 3), source=source,
                   files=ids, ground_truth=gt, hf_raw=info["raw_text"], hf_language=info["parsed_language"], hf_text=info["parsed_text"],
                   n_audio_tokens=info["n_audio_tokens"], n_new=info["n_new"], eos_hit=info["eos_hit"])
        if a.also_full and row["n_audio_tokens"] > 104:
            asr.model.thinker.audio_tower.config._attn_implementation = "sdpa"; asr.audio_attn = "full"
            _, i2, _ = run_clip(asr, p, context, language, name, mode, full=False)
            asr.model.thinker.audio_tower.config._attn_implementation = "windowed_sdpa"; asr.audio_attn = "windowed"
            row["hf_full_attn_raw"] = i2["raw_text"]
        rows.append(row)
        print(i, name, row["seconds"], mode, row["n_new"], repr(row["hf_text"][:60]), flush=True)
        with open(f"{OUT}/wer/refs.jsonl", "w") as f:
            for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({k: str(v) for k, v in provenance(asr).items()} | {"clips": len(rows)}, open(f"{OUT}/wer/provenance.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fixtures", "short", "wer", "tied", "attn"])
    ap.add_argument("--src", default=os.path.expanduser("~/.cache/zspark/scratch-138/dl"))
    ap.add_argument("--audio-attn", choices=["windowed", "full"], default="windowed")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--also-full", action="store_true", help="wer: also decode clips above 104 audio tokens with the as-shipped full attention")
    a = ap.parse_args()
    {"fixtures": cmd_fixtures, "short": cmd_short, "wer": cmd_wer, "tied": cmd_tied, "attn": cmd_attn}[a.cmd](a)
