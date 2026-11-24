#!/usr/bin/env python3
"""Where vLLM cuts a long clip (spec 142 WS-06): `split_audio` and `find_split_point` as
`vllm/multimodal/audio.py` has them (the fork local-inference-lab/vllm 8244d1c44 and upstream main a6c898e2,
the same code), copied below, on numpy float32. Host only, no model. Writes zig/tests/fixtures/audio/
split_cases.json: the pieces' ends for a generated signal (`synthetic`, which `zig/tests/audio/split_test.zig`
makes again from the same integers) and for every clip of Qwen3-ASR's WER set (spec 138) past 30 s (`clips`, read as vLLM reads
them: soundfile, float32; they are 16 kHz already).

  uv run --no-project --with numpy --with soundfile python -I tools/reference/audio_split_cases.py OUT.json [REF_DIR]
"""
import json, math, os, sys
import numpy as np
import soundfile


def find_split_point(wav, start_idx, end_idx, min_energy_window):
    segment = wav[start_idx:end_idx]
    min_energy = math.inf
    quietest_idx = start_idx
    for i in range(0, len(segment) - min_energy_window, min_energy_window):
        window = segment[i : i + min_energy_window]
        energy = (window**2).mean() ** 0.5
        if not math.isnan(energy) and energy < min_energy:
            quietest_idx = i + start_idx
            min_energy = energy
    return quietest_idx


def split_audio(audio_data, sample_rate, max_clip_duration_s, overlap_duration_s, min_energy_window_size):
    chunk_size = int(sample_rate * max_clip_duration_s)
    overlap_size = int(sample_rate * overlap_duration_s)
    chunks = []
    i = 0
    while i < audio_data.shape[-1]:
        if i + chunk_size >= audio_data.shape[-1]:
            chunks.append(audio_data[..., i:])
            break
        search_start = i + chunk_size - overlap_size
        search_end = min(i + chunk_size, audio_data.shape[-1])
        split_point = find_split_point(audio_data, search_start, search_end, min_energy_window_size)
        if split_point <= i:
            split_point = min(i + chunk_size, audio_data.shape[-1])
        chunks.append(audio_data[..., i:split_point])
        i = split_point
    return chunks


def ends(chunks):
    out, at = [], 0
    for c in chunks:
        at += c.shape[-1]
        out.append(at)
    return out


def synthetic():
    """95 s at 16 kHz: a 64-bit LCG's high bits as int16, shifted right (quieter) by an integer envelope; the
    same integers in split_test.zig."""
    n = 95 * 16000
    v = np.empty(n, dtype=np.int64)
    s = 0x2545F4914F6CDD1D
    for k in range(n):
        s = (s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        v[k] = ((s >> 48) & 0xFFFF) - 32768
    # (first, last + 1, right shift; 0 sets the samples to zero): a quiet stride in the first region, a zero
    # stretch over the second (its strides tie at 0), in the third a quieter last stride than the one taken
    for lo, hi, shift in [(468800, 470400, 8), (473600, 475200, 6), (935000, 950000, 0),
                          (1408000, 1409600, 2), (1416000, 1417600, 10)]:
        v[lo:hi] = 0 if shift == 0 else v[lo:hi] >> shift
    return (v.astype(np.float32) / np.float32(32768.0)).astype(np.float32)


def main():
    out = sys.argv[1]
    ref = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/.cache/zspark/reference-qwen3-asr")
    cases = {"max_seconds": 30, "overlap_seconds": 1, "window": 1600,
             "synthetic": ends(split_audio(synthetic(), 16000, 30, 1, 1600)), "clips": {}}
    for line in open(f"{ref}/wer/refs.jsonl"):
        r = json.loads(line)
        y, sr = soundfile.read(f"{ref}/wer/clips/{r['id']}.flac", dtype="float32", always_2d=False)
        assert sr == 16000 and y.ndim == 1, (r["id"], sr, y.shape)
        if y.shape[-1] / sr > 30:
            cases["clips"][r["id"]] = ends(split_audio(y, 16000, 30, 1, 1600))
    with open(out, "w") as f:
        json.dump(cases, f, indent=1)
        f.write("\n")
    print(f"{len(cases['clips'])} clips, synthetic {cases['synthetic']}")


if __name__ == "__main__":
    main()
