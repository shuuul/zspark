# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy==2.5.5", "soundfile==0.14.0", "soxr==1.1.0", "librosa==1.0.1", "transformers==4.57.6"]
# ///
"""Audio files and what the reference makes of them, for `zig/src/audio/` (spec 138 WS-02).

The reference is the `qwen-asr` 0.0.6 package (the transformers backend Qwen3-ASR-1.7B's card measures): its
`normalize_audio_input` on a file (`inference/utils.py`: `librosa.load(path, sr=None, mono=False)` over
libsndfile, `to_mono`'s mean, `librosa.resample` with its default `soxr_hq` where the rate is not 16 kHz, then
`float_range_normalize`), and transformers' `WhisperFeatureExtractor` with the checkpoint's
`preprocessor_config.json` as `Qwen3ASRProcessor` calls it (`padding=True`, `truncation=False`). The two small
functions of `utils.py` are restated here so the package's own dependencies (vLLM, gradio) are not needed. No
torch is installed, so the extractor takes its numpy path; transformers says it agrees with the torch path,
which the full reference uses, within 1e-5.

Each case is a short file written here by libsndfile (WAV, RF64, FLAC, MP3, Ogg Vorbis) from a made-up signal,
with libsndfile's decode of it (`decoded`, mono at the file's rate, every 4th sample), the package's 16 kHz clip
(`clip`, every 2nd) and the features (`mel`, `[128][frames]`, whole). Files the layer refuses carry `problem`. `synthetic` is a 3 s signal both sides
compute from one formula (`signal` below and the test's): its features every 7th frame. `tokens` is the tower's
token count for frame counts. Arrays are little-endian f32 in base64. Run with uv, no GPU:

    uv run tools/reference/audio_cases.py zig/tests/fixtures/audio/cases.json
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf
import soxr
import transformers
from transformers import WhisperFeatureExtractor

OUT = sys.argv[1]
SAMPLE_RATE = 16000

# the checkpoint's preprocessor_config.json, as Qwen3ASRProcessor builds its extractor
FE = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400,
                             padding_value=0.0, dither=0.0, return_attention_mask=True)


def to_mono(audio: np.ndarray) -> np.ndarray:  # qwen_asr/inference/utils.py
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")


def float_range_normalize(audio: np.ndarray) -> np.ndarray:  # qwen_asr/inference/utils.py
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)


def features(clip: np.ndarray) -> tuple[np.ndarray, int]:
    out = FE(clip, sampling_rate=16000, padding=True, truncation=False, return_attention_mask=True)
    return out["input_features"][0].astype(np.float32), int(out["attention_mask"].sum())


def f32(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode()


def speech(rate: int, seconds: float, channels: int, seed: int) -> np.ndarray:
    """A voiced, amplitude-modulated harmonic signal with a little noise, `[frames, channels]`."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(rate * seconds)) / rate
    f0 = 140 + 40 * np.sin(2 * np.pi * 2.0 * t)
    phase = 2 * np.pi * np.cumsum(f0) / rate
    nyquist = rate / 2
    out = []
    for c in range(channels):
        x = sum(np.sin(k * phase + c) / k for k in range(1, int(nyquist / 200)))
        x = x * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t + c)) + 0.03 * rng.standard_normal(len(t))
        out.append(0.4 * x / np.max(np.abs(x)))
    return np.stack(out, axis=1)


def signal(n: int) -> np.ndarray:
    """The synthetic clip at 16 kHz, from integer arithmetic and f64 sines alone, cast to f32 at the end."""
    out = np.empty(n, dtype=np.float64)
    state = 0x2545F4914F6CDD1D
    for i in range(n):
        t = i / 16000.0
        state = (state * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        noise = ((state >> 40) / float(1 << 24) - 0.5) * 0.02
        out[i] = (0.3 * math.sin(2 * math.pi * (220.0 * t + 600.0 * t * t)) + 0.2 * math.sin(2 * math.pi * 3150.0 * t)
                  * (0.5 + 0.5 * math.sin(2 * math.pi * 3.0 * t)) + noise)
    return out.astype(np.float32)


def reference(data: bytes, suffix: str) -> tuple[np.ndarray, int, np.ndarray]:
    """librosa.load on a file, as the package reads a path: (decoded mono at the file's rate, rate, 16 kHz clip)."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as f:
        f.write(data)
        f.flush()
        audio, sr = librosa.load(f.name, sr=None, mono=False)
    audio = np.asarray(audio, dtype=np.float32)
    mono = to_mono(audio)
    clip = mono
    if sr != SAMPLE_RATE:
        clip = librosa.resample(mono, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    return mono, int(sr), float_range_normalize(clip)


def written(x: np.ndarray, rate: int, fmt: str, subtype: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, rate, format=fmt, subtype=subtype)
    return buf.getvalue()


CASES: list[dict] = []
# what is kept of the samples, to keep the file small: every 4th decoded sample and every 2nd of the clip
DECODED_EVERY = 4
CLIP_EVERY = 2


def case(name: str, data: bytes, suffix: str, fmt: str) -> None:
    mono, sr, clip = reference(data, suffix)
    mel, frames = features(clip)
    assert mel.shape == (128, frames), (mel.shape, frames)
    CASES.append({"name": name, "format": fmt, "rate": sr, "bytes": base64.b64encode(data).decode(),
                  "samples": len(mono), "decoded": f32(mono[::DECODED_EVERY]), "clip_samples": len(clip),
                  "clip": f32(clip[::CLIP_EVERY]),
                  "frames": frames, "mel": f32(mel)})


def refused(name: str, data: bytes, problem: str) -> None:
    CASES.append({"name": name, "bytes": base64.b64encode(data).decode(), "problem": problem})


case("pcm16-16k-mono.wav", written(speech(16000, 0.15, 1, 1), 16000, "WAV", "PCM_16"), ".wav", "wav")
case("pcm24-48k-stereo.wav", written(speech(48000, 0.05, 2, 2), 48000, "WAV", "PCM_24"), ".wav", "wav")
case("float-8k-mono.rf64", written(speech(8000, 0.15, 1, 3), 8000, "RF64", "FLOAT"), ".wav", "wav")
case("44k-stereo.flac", written(speech(44100, 0.08, 2, 4), 44100, "FLAC", "PCM_16"), ".flac", "flac")
case("48k-mono.mp3", written(speech(48000, 0.15, 1, 5), 48000, "MP3", "MPEG_LAYER_III"), ".mp3", "mp3")
case("22k-stereo.ogg", written(speech(22050, 0.15, 2, 6), 22050, "OGG", "VORBIS"), ".ogg", "vorbis")
# a peak above 1 (a float WAV), which float_range_normalize divides down
case("loud-float-16k.wav", written(1.8 * speech(16000, 0.1, 1, 7), 16000, "WAV", "FLOAT"), ".wav", "wav")

refused("ftyp.m4a", b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00M4A isommp42" + bytes(64), "Invalid")
refused("adts.aac", bytes([0xFF, 0xF1, 0x50, 0x80, 0x02, 0x1F, 0xFC]) + bytes(64), "Invalid")
refused("empty", b"", "Unsupported")
refused("text.txt", b"hello, not audio\n", "Unsupported")
refused("header-only.wav", written(speech(16000, 0.1, 1, 9), 16000, "WAV", "PCM_16")[:44], "Invalid")

n = 3 * SAMPLE_RATE + 77
syn_mel, syn_frames = features(signal(n))
synthetic = {"samples": n, "frames": syn_frames, "every": 7, "mel": f32(syn_mel[:, ::7])}


def tokens(n: int) -> int:  # Qwen3ASR's _get_feat_extract_output_lengths
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13

token_cases = [[n, tokens(n)] for n in [0, 1, 2, 3, 7, 8, 9, 50, 99, 100, 101, 199, 200, 800, 801, 3000, 60000, 120000]]

versions = {"numpy": np.__version__, "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__,
            "soxr": soxr.__version__, "librosa": librosa.__version__, "transformers": transformers.__version__}
with open(OUT, "w") as f:
    json.dump({"versions": versions, "cases": CASES, "synthetic": synthetic, "tokens": token_cases}, f)
    f.write("\n")
print(OUT, os.path.getsize(OUT), "bytes,", len(CASES), "cases")
