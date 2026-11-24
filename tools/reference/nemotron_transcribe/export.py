#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Nemotron-3.5-Transcribe-30B-A4B-NVFP4's reference fixtures (spec 151 WS-04).

The oracle is the model card's serving stack: vLLM 0.32.0 + NeMo Speech main's `nemo_speechlm` plugin,
inside the `zspark151-nemo-vllm:local` image (see Dockerfile; run this script inside it). Everything in the
`weightless` command runs without the checkpoint: the mel of the fixed clips through NeMo's
AudioToMelSpectrogramPreprocessor (the checkpoint's `config.json` `perception.preprocessor`), the rendered
`chat_template.jinja` prompt and its token ids for the card's chat example, and the plugin's own audio
placeholder count per clip (`NeMoSpeechLMProcessingInfo._estimate_audio_tokens`, the function vLLM calls
to size the `<|audio|>` expansion). `wer` and `model` wait for WS-00's download.

  python export.py weightless [--src DIR] [--out DIR]     # mel + prompts + placeholder counts + clips
  python export.py wer [--serve CMDFILE]                  # needs the NVFP4 checkpoint; per-clip WER
  python export.py model [--serve CMDFILE] [--only N...]  # needs it; encoder/hiddens/logits/transcripts

Clip set (--src, default ~/.cache/zspark/scratch-138/dl, spec 138's download): LibriSpeech test-clean
(openslr.org/resources/12, CC BY 4.0) and FLEURS cmn_hans_cn test (CC BY 4.0), the same utterance sets
spec 138 picked: en_5s/en_30s/en_140s/en_600s and zh_5s/zh_30s/zh_140s/zh_600s, plus en_3500s (LibriSpeech
test-clean chapters joined to just under an hour, near the checkpoint's 3,600 s context). Clips are
concatenated utterances at 16 kHz mono int16 FLAC, ground truth in manifest.json.

Output (~/.cache/zspark/scratch-151/fixtures/, ZSPARK_REFERENCE_NEMOTRON, mirrored on both Sparks):
  clips/NAME.flac           the 16 kHz mono int16 clip
  manifest.json             per clip: source, seconds, ground truth, audio placeholder count
  NAME.mel.safetensors      `mel.f32` [128, T] (NeMo's preprocessor, log-mel) and `pcm.16k` F32 [n]
  prompt.chat.json          rendered chat_template.jinja for the card's example, enable_thinking both
                            ways, the tokenizer's ids, and the placeholder position
  wer/refs.jsonl            (weights) per clip: the vLLM transcript, WER vs ground truth
  model/NAME.safetensors    (weights) encoder output rows, per-layer backbone hiddens, logits, tokens

WER scoring: English WER on lowercased text without punctuation (jiwer); zh CER without spaces.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time

SR = 16000
OUT = os.environ.get("ZSPARK_REFERENCE_NEMOTRON", os.path.expanduser("~/.cache/zspark/scratch-151/fixtures"))
MODEL = os.environ.get("ZSPARK_NEMOTRON_NVFP4", os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4/"
    "snapshots/66c6adc4020179567074d0399cda2a4eeb2bb347"))  # the weights; ~/models/… is metadata-only
HF = os.environ.get("ZSPARK_NEMOTRON_HF", os.path.expanduser("~/.cache/zspark/scratch-151/hf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4"))

CARD_USER_TEXT = "Transcribe the following audio."

# ---------------------------------------------------------------- clips (spec 138's code, unchanged)
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
        if raw in seen:
            continue
        seen.add(raw)
        out.append((os.path.join(src, "fl/test", c[1]), raw, int(c[5]) / SR))
    return out


def cat(items):
    """items: [(path, text)] -> (int16 samples at 16 kHz, text, [source basenames])."""
    import numpy as np
    import soundfile as sf

    pcm, ids = [], []
    for p, t in items:
        x, sr = sf.read(p, dtype="float32")  # LibriSpeech: int16 FLAC; FLEURS: 32-bit float WAV
        assert sr == SR and x.ndim == 1, p
        pcm.append(np.clip(np.round(x * 32768.0), -32768, 32767).astype(np.int16))
    return np.concatenate(pcm), " ".join(t for _, t in items), [os.path.basename(p) for p, _ in items]


def take(utts, target, start=0):
    import soundfile as sf

    got, tot = [], 0.0
    for p, t in utts[start:]:
        got.append((p, t))
        tot += sf.info(p).duration
        if tot >= target:
            break
    return got


def fixture_clips(src):
    import soundfile as sf

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
    # Long clip near the hour context: whole chapters in reading order to just under 3,500 s.
    long_items = []
    tot = 0.0
    for c in ("5639/40744", "672/122797", "8230/279154", "1188/133604", "2300/131720",
              "7729/102255", "2094/142345", "3575/170457"):
        for p, t in ch[c]:
            d = sf.info(p).duration
            if tot + d > 3500:
                break
            long_items.append((p, t))
            tot += d
        if tot >= 3500:
            break
    clips["en_3500s"] = (long_items, f"LibriSpeech test-clean chapters joined to {tot:.0f} s, near the hour")
    return clips


# ---------------------------------------------------------------- the plugin's placeholder count
def audio_token_count(samples):
    """`NeMoSpeechLMProcessingInfo._estimate_audio_tokens` over the checkpoint's exported estimator:
    (n + 2*256 - 512) // 160 frames, then ceil/8 feature stacking. This checkpoint mounts a
    ParallelExpertEncoder (`pe_encoder_config` in config.json), so `_get_encoder_chunk_size_seconds`
    returns None — the PEE owns its own windowing — and the estimator runs ONE single pass over the
    whole clip (`_get_audio_token_estimator_config` forces the exported `chunk_size_seconds` to None).
    The class method is used where the plugin is importable; the same arithmetic is inline so the
    fixture builds wherever it is not."""
    try:
        from nemo.collections.speechlm2.vllm.salm.audio import NeMoSpeechLMProcessingInfo

        est = {
            "preprocessor": {"n_fft": 512, "hop_length": 160, "stft_pad_amount": 256},
            "subsampling": {"type": "feature_stacking", "factor": 8},
            "chunk_size_seconds": None,
        }
        return NeMoSpeechLMProcessingInfo._estimate_audio_tokens(int(samples), None, est)
    except ImportError:
        length = (int(samples) + 2 * 256 - 512) // 160
        return max(1, -(-length // 8))

# ---------------------------------------------------------------- weightless fixtures


def cmd_weightless(args):
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file

    out = args.out
    os.makedirs(out + "/clips", exist_ok=True)
    clips = fixture_clips(args.src)

    # NeMo's preprocessor, configured by the checkpoint's config.json perception.preprocessor.
    cfg = json.load(open(os.path.join(HF, "config.json")))
    pre_cfg = dict(cfg["perception"]["preprocessor"])
    pre_cfg.pop("_target_", None)
    from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor

    pre = AudioToMelSpectrogramPreprocessor(**pre_cfg)
    pre.eval()
    manifest = {"clips": {}, "sources": {
        "librispeech": "LibriSpeech test-clean, openslr.org/resources/12, CC BY 4.0",
        "fleurs": "google/fleurs cmn_hans_cn test, CC BY 4.0",
        "checkpoint": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4, OpenMDW-1.1",
        "nemo": nemo_commit(),
    }, "preprocessor": pre_cfg}
    for name, items, source in ((n,) + c for n, c in clips.items()):
        pcm, truth, ids = cat(items)
        wav = os.path.join(out, "clips", name + ".flac")
        sf.write(wav, pcm, SR, subtype="PCM_16")
        x = torch.from_numpy(pcm.astype(np.float32) / 32768.0)[None]
        with torch.no_grad():
            mel, length = pre.get_features(x, torch.tensor([pcm.shape[0]]))
        mel = mel[0]  # [128, T]
        n_tok = audio_token_count(pcm.shape[0])
        save_file({"pcm.16k": x[0].clone(), "mel.f32": mel.clone()},
                  os.path.join(out, name + ".mel.safetensors"),
                  metadata={"clip": name, "nemo": nemo_commit()})
        manifest["clips"][name] = {
            "source": source, "utterances": ids, "lang": "zh" if name.startswith("zh") else "en",
            "seconds": round(pcm.shape[0] / SR, 3), "samples": int(pcm.shape[0]),
            "mel_frames": int(mel.shape[1]), "mel_length_returned": int(length[0]),
            "audio_placeholders": n_tok, "ground_truth": truth,
        }
        print(name, manifest["clips"][name]["seconds"], "mel", tuple(mel.shape), "placeholders", n_tok, flush=True)
    json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), ensure_ascii=False, indent=1)

    # The card's chat example through chat_template.jinja and the tokenizer.
    write_prompt_fixture(out)


def write_prompt_fixture(out):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF)
    template = open(os.path.join(HF, "chat_template.jinja")).read()
    messages = [{"role": "user", "content": [
        {"type": "text", "text": CARD_USER_TEXT},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
    ]}]
    doc = {"user_text": CARD_USER_TEXT, "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
           "messages": messages, "renders": {}}
    for flag in (False, True):
        text = tok.apply_chat_template(messages, chat_template=template, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=flag)
        ids = tok.apply_chat_template(messages, chat_template=template, tokenize=True,
                                      add_generation_prompt=True, enable_thinking=flag)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        ids = list(ids)
        doc["renders"][f"enable_thinking_{str(flag).lower()}"] = {
            "text": text, "ids": ids,
            "audio_placeholder_id": tok.convert_tokens_to_ids("<|audio|>"),
            "audio_placeholder_first": ids.index(tok.convert_tokens_to_ids("<|audio|>")) if tok.convert_tokens_to_ids("<|audio|>") in ids else None,
        }
    json.dump(doc, open(os.path.join(out, "prompt.chat.json"), "w"), ensure_ascii=False, indent=1)
    for k, r in doc["renders"].items():
        print(k, len(r["ids"]), "ids; placeholder at", r["audio_placeholder_first"], flush=True)


def nemo_commit():
    try:
        import nemo

        return getattr(nemo, "__git_commit__", None) or getattr(nemo, "version_info", None) or "nemo-toolkit main"
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------- weights-dependent
def cmd_wer(args):
    """Per-clip transcripts; WER/CER against ground truth. Transcripts come from (in order):
      1. `model/NAME.transcript.txt` already exported by export_model.py (the same greedy decode —
         reuses the offline capture, no second GPU load);
      2. the reference server on 127.0.0.1:8000 (`--serve FILE` starts it once for the whole run and
         stops it after; without --serve an already-running server is used)."""
    man = json.load(open(os.path.join(args.out, "manifest.json")))
    names = args.only or list(man["clips"])
    import subprocess
    os.makedirs(os.path.join(args.out, "wer"), exist_ok=True)
    proc = None
    try:
        refs = []
        need_server = []
        for name in names:
            txt = os.path.join(args.out, "model", name + ".transcript.txt")
            if os.path.exists(txt):
                hyp = open(txt).read().strip()
            else:
                need_server.append(name)
                continue
            info = man["clips"][name]
            refs.append(_score_row(name, info, hyp, source="model-fixture"))
            print(name, refs[-1].get("wer", refs[-1].get("cer")), repr(hyp[:80]), flush=True)
        if need_server:
            if args.serve:
                cmd = open(args.serve).read().strip().replace("\\\n", " ")
                log = open(os.path.join(args.out, "wer", "server.log"), "w")
                proc = subprocess.Popen(cmd, shell=True, env=dict(os.environ),
                                        stdout=log, stderr=subprocess.STDOUT)
            wait_ready("http://127.0.0.1:8000")
            for name in need_server:
                hyp = transcribe(os.path.join(args.out, "clips", name + ".flac"))
                refs.append(_score_row(name, man["clips"][name], hyp, source="server"))
                print(name, refs[-1].get("wer", refs[-1].get("cer")), repr(hyp[:80]), flush=True)
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=60)
    with open(os.path.join(args.out, "wer", "refs.jsonl"), "w") as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    en = [r for r in refs if r["lang"] == "en"]
    zh = [r for r in refs if r["lang"] == "zh"]
    summary = {"en_mean_wer": sum(r["wer"] for r in en) / max(1, len(en)),
               "zh_mean_cer": sum(r["cer"] for r in zh) / max(1, len(zh)),
               "clips": len(refs), "server": args.serve, "nemo": nemo_commit()}
    json.dump(summary, open(os.path.join(args.out, "wer", "summary.json"), "w"), indent=1)
    print(summary)


def _score_row(name, info, hyp, source):
    err = score(info["lang"], info["ground_truth"], hyp)
    return {"id": name, "lang": info["lang"], "seconds": info["seconds"],
            "ground_truth": info["ground_truth"], "vllm_text": hyp, "source": source,
            ("cer" if info["lang"] == "zh" else "wer"): err}


def score(lang, ref, hyp):
    if lang == "zh":
        import jiwer

        return jiwer.cer("".join(ref.split()), "".join(hyp.split()))
    import jiwer

    tr = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                        jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    return jiwer.wer(tr(ref), tr(hyp))


def cmd_model(args):
    """The model fixtures: encoder rows, per-layer hiddens, logits, transcripts — captured by
    hooking the vLLM engine in-process (export_model.py), not over HTTP. Delegates with the same
    --only/--out arguments."""
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "export_model.py"), "--out",
           os.path.join(args.out, "model")]
    if args.only:
        cmd += ["--only", *args.only]
    raise SystemExit(subprocess.call(cmd))


def transcribe(wav_path):
    """One clip through the already-running reference server on 127.0.0.1:8000."""
    import base64
    from openai import OpenAI

    client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
    audio = base64.b64encode(open(wav_path, "rb").read()).decode()
    r = client.chat.completions.create(
        model="nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": CARD_USER_TEXT},
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio}"}},
        ]}],
        temperature=0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}, "skip_special_tokens": False,
                    "spaces_between_special_tokens": False, "seed": 0},
    )
    return r.choices[0].message.content

def wait_ready(url, timeout=1800):
    import urllib.request

    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url + "/health", timeout=5):
                return
        except Exception:
            time.sleep(10)
    raise SystemExit("server did not come up")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["weightless", "wer", "model"])
    ap.add_argument("--src", default=os.path.expanduser("~/.cache/zspark/scratch-138/dl"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--serve", help="file holding the card's `vllm serve` command (started on 127.0.0.1)")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    {"weightless": cmd_weightless, "wer": cmd_wer, "model": cmd_model}[a.cmd](a)
