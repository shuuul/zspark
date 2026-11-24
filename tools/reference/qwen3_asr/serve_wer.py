#!/usr/bin/env python3
"""The WER/CER set (`export.py wer`) through a running `zspark serve` (spec 138 WS-05). Standard library only.

  python3 -I tools/reference/qwen3_asr/serve_wer.py [--url http://127.0.0.1:18888] [--out FILE.jsonl]

Each clip of `$ZSPARK_REFERENCE_QWEN3_ASR/wer/refs.jsonl` is posted to `/v1/audio/transcriptions` as aimd posts
it (multipart: `model`, `file`, and `language` and `prompt` where the clip's scenario sets them). Its text is
compared with the package's (`hf_text`): exact or not, and where it first parts. Then WER (English, words) and CER
(Chinese, characters) of both against the ground truth, normalized alike: case folded, punctuation and spaces
dropped for CER, punctuation dropped for WER. The criterion (spec 138): zspark's within 0.5 points of the
package's. Also the wall time a clip, its tokens a second and its real-time factor (wall / audio seconds).
"""
import argparse, json, os, sys, time, unicodedata, urllib.request, uuid


def post(url, fields, path):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    data = open(path, "rb").read()
    name = os.path.basename(path)
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\nContent-Type: audio/flac\r\n\r\n'.encode() + data + b"\r\n")
    parts.append(f"--{b}--\r\n".encode())
    req = urllib.request.Request(url + "/v1/audio/transcriptions", data=b"".join(parts), headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def norm(s, chars):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c if not unicodedata.category(c).startswith("P") else " " for c in s)
    return [c for c in s if not c.isspace()] if chars else s.split()


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18888")
    ap.add_argument("--ref", default=os.environ.get("ZSPARK_REFERENCE_QWEN3_ASR", os.path.expanduser("~/.cache/zspark/reference-qwen3-asr")))
    ap.add_argument("--model", default="Qwen3-ASR-1.7B")
    ap.add_argument("--out")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(f"{a.ref}/wer/refs.jsonl")]
    if a.only:
        rows = [r for r in rows if r["id"] in a.only]
    tot = {k: [0, 0, 0] for k in ("en", "zh")}  # zspark's edits, the package's edits, the reference's length
    exact = 0
    audio = wall = 0.0
    out = open(a.out, "w") if a.out else None
    for r in rows:
        fields = {"model": a.model}
        if r["forced_language"]:
            fields["language"] = r["forced_language"]
        if r["context"]:
            fields["prompt"] = r["context"]
        t0 = time.time()
        got = post(a.url, fields, f"{a.ref}/wer/clips/{r['id']}.flac")["text"]
        dt = time.time() - t0
        chars = r["lang"] == "zh"
        gt = norm(r["ground_truth"], chars)
        e_z, e_h = edits(norm(got, chars), gt), edits(norm(r["hf_text"], chars), gt)
        t = tot[r["lang"]]
        t[0] += e_z; t[1] += e_h; t[2] += len(gt)
        same = got == r["hf_text"]
        exact += same
        first = next((i for i, (x, y) in enumerate(zip(got, r["hf_text"])) if x != y), None if same else min(len(got), len(r["hf_text"])))
        audio += r["seconds"]; wall += dt
        line = dict(id=r["id"], seconds=r["seconds"], wall=round(dt, 3), rtf=round(dt / r["seconds"], 4), exact=same, first_divergence=first,
                    err_zspark=e_z, err_hf=e_h, ref_len=len(gt), text=got)
        if out:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
        print(f"{r['id']:8s} {r['seconds']:7.1f}s wall {dt:6.2f}s {'=' if same else 'x at ' + str(first):>10s} err {e_z}/{e_h} of {len(gt)}", flush=True)
    for k, (z, h, n) in tot.items():
        if n:
            print(f"{'WER' if k == 'en' else 'CER'} {k}: zspark {100 * z / n:.3f}%  package {100 * h / n:.3f}%  difference {100 * (z - h) / n:+.3f} points")
    print(f"exact {exact} of {len(rows)}; audio {audio:.1f} s in {wall:.1f} s, real-time factor {wall / audio:.4f}")


if __name__ == "__main__":
    sys.exit(main())
