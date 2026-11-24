"""Export the ban sets of Unlimited-OCR's sliding-window no-repeat n-gram rule, as the fixture
`zig/src/decode/ngram.zig` is tested against (spec 137 WS-05). Host only, no dependencies.

`banned` is a line-for-line port of `SlidingWindowNoRepeatNgramProcessor.__call__` in the checkpoint's
`modeling_unlimitedocr.py` (the loop over one batch row, `scores` replaced by the returned set). It departs from
vLLM's `NoRepeatNGramLogitsProcessor` (`vllm/model_executor/models/deepseek_ocr.py`) in the two places the HF
loop is explicit about: a sequence the caller gives whole (vLLM is handed the output ids alone) and `ngram_size`
1, where HF bans every token of the window and vLLM's `output_ids[-0:]` prefix (the whole list) matches nothing.

The sequences are made from a seeded generator; the cases are the sets of `CASES`, the 35/128 and 35/1024 of the
checkpoint's two uses with phrases repeated inside and outside each window. The fixture, little-endian:

    "ZSNGRAM1"
    u32 cases; each: u32 ngram, u32 window, u32 whitelist count, u32 whitelist[count],
                     u32 sequence length, u32 sequence[length], u32 banned count, u32 banned[count] (ascending)

    python3 tools/reference/ngram_cases.py --out zig/tests/fixtures/ngram_cases.bin
"""

from __future__ import annotations

import argparse
import random
import struct


def banned(sequence: list[int], ngram_size: int, window: int, whitelist: set[int]) -> set[int]:
    """One row of HF's `SlidingWindowNoRepeatNgramProcessor.__call__`: the ids whose logit it sets to -inf."""

    if len(sequence) < ngram_size:
        return set()
    search_start = max(0, len(sequence) - window)
    search_end = len(sequence) - ngram_size + 1
    if search_end <= search_start:
        return set()
    if ngram_size > 1:
        current_prefix = tuple(sequence[-(ngram_size - 1):])
    else:
        current_prefix = tuple()
    out = set()
    for idx in range(search_start, search_end):
        ngram = sequence[idx:idx + ngram_size]
        if ngram_size == 1 or tuple(ngram[:-1]) == current_prefix:
            out.add(ngram[-1])
    out.difference_update(whitelist)
    return out


def noise(rng: random.Random, count: int, low: int = 1000, high: int = 150_000) -> list[int]:
    return [rng.randrange(low, high) for _ in range(count)]


def phrase_cases(rng: random.Random) -> list[tuple[int, int, set[int], list[int]]]:
    """35-grams with the phrase `p` (80 tokens) repeated at distances inside and outside 128 and 1024."""

    out = []
    p = noise(rng, 80)
    # the earlier 35-gram starts gap + 114 tokens before the end: inside a window w iff gap + 114 <= w
    for window, gaps in ((128, (14, 15, 100, 300)), (1024, (14, 100, 910, 911, 1100))):
        for gap in gaps:
            # p, `gap` tokens of other text, then p's first 34 tokens again: the next token would repeat p[34]
            seq = noise(rng, 50) + p + noise(rng, gap) + p[:34]
            out.append((35, window, set(), seq))
            if gap == 14:
                # the same with p[34] whitelisted
                out.append((35, window, {p[34]}, seq))
        # two earlier uses of the prefix that continue differently
        q = p[:34] + [7, 8, 9, 10]
        r = p[:34] + [11, 12, 13]
        seq = noise(rng, 30) + q + noise(rng, 40) + r + noise(rng, 20) + p[:34]
        out.append((35, window, set(), seq))
    return out


def small_cases(rng: random.Random) -> list[tuple[int, int, set[int], list[int]]]:
    out = []
    for ngram in (1, 2, 3, 5):
        for window in (1, 2, 4, 9, 64):
            for length in (0, 1, 2, 3, 5, 8, 20, 70):
                # a vocabulary of 4 repeats a lot, so n-grams recur
                seq = [rng.randrange(0, 4) for _ in range(length)]
                out.append((ngram, window, set(), seq))
                out.append((ngram, window, {0, 3}, seq))
    # periodic text and a run of one token
    out.append((3, 100, set(), [1, 2, 3] * 20 + [1, 2]))
    out.append((3, 100, set(), [5] * 30))
    out.append((1, 10, set(), list(range(30))))
    # the window bounds: the 2-gram (3, 4) is last in the window at window = 6 and out of it at 5
    for window in (4, 5, 6, 7):
        out.append((2, window, set(), [3, 4, 0, 0, 0, 0, 3]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rng = random.Random(137)
    cases = phrase_cases(rng) + small_cases(rng)
    data = bytearray(b"ZSNGRAM1")
    data += struct.pack("<I", len(cases))
    for ngram, window, whitelist, seq in cases:
        ids = sorted(banned(seq, ngram, window, whitelist))
        white = sorted(whitelist)
        data += struct.pack("<III", ngram, window, len(white)) + struct.pack(f"<{len(white)}I", *white)
        data += struct.pack("<I", len(seq)) + struct.pack(f"<{len(seq)}I", *seq)
        data += struct.pack("<I", len(ids)) + struct.pack(f"<{len(ids)}I", *ids)
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"{len(cases)} cases, {len(data)} bytes, {sum(1 for c in cases if banned(c[3], c[0], c[1], c[2]))} with a ban")


if __name__ == "__main__":
    main()
