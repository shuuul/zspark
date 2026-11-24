#!/usr/bin/env python3
"""CER of replies against the reference transcripts (spec 137): tags stripped, Levenshtein over characters.

    cer.py REF_DIR HYP_DIR      # NAME.txt in both; REF_DIR is OUT/cer
    cer.py --self REF_DIR       # a reference against itself (0.0): checks the stripping on every page

Prints per page: CER, the first differing token index (REF_DIR/NAME.json's `tokens` against HYP_DIR/NAME.json's,
where given), whether the reply ended at EOS; then the mean and the max. The stripping is the README's
`remove_det` for OmniDocBench: `<|ref|>..<|/ref|>` and `<|det|>..<|/det|>` spans and the EOS text go; for v2's
replies `<PAGE>` and the `<BLOCK>..<FORK>` spans (label, level, box) go the same way.
"""
import json
import re
import sys
from pathlib import Path

# v2 (spec 147) writes `<PAGE>` a page and `<BLOCK>label<LEVELn>[box]<FORK>text`: the label and box go as v1's det spans
TAGS = re.compile(r"<\|ref\|>.*?<\|/ref\|>|<\|det\|>.*?<\|/det\|>|<BLOCK>.*?<FORK>|<PAGE>|<｜end▁of▁sentence｜>", re.S)


# a reply may open with a box whose `<|det|>` the model did not write (baidu: `title [0, 0, 999, 999]<|/det|>...`)
OPENING = re.compile(r"^(?:(?!<\|det\|>).)*?<\|/det\|>", re.S)


def strip(s):
    return TAGS.sub("", OPENING.sub("", s.lstrip(), count=1)).strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def first_div(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    self_check = sys.argv[1] == "--self"
    ref = Path(sys.argv[2 if self_check else 1])
    hyp = ref if self_check else Path(sys.argv[2])
    rows = []
    for t in sorted(ref.glob("*.txt")):
        r, h = strip(t.read_text()), strip((hyp / t.name).read_text())
        c = lev(r, h) / max(len(r), 1)
        rj, hj = json.loads(t.with_suffix(".json").read_text()), None
        hp = (hyp / t.name).with_suffix(".json")
        div = first_div(rj["tokens"], json.loads(hp.read_text())["tokens"]) if hp.exists() and "tokens" in json.loads(hp.read_text()) else None
        rows.append(c)
        print(f"{t.stem:32s} cer={c:.4f} first_div={div} eos={rj['eos']}")
    print(f"pages={len(rows)} mean={sum(rows) / len(rows):.4f} max={max(rows):.4f}")


if __name__ == "__main__":
    main()
