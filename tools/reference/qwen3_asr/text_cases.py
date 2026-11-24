#!/usr/bin/env python3
"""Qwen3-ASR's text side for zspark's tests (spec 138 WS-05): the tokenizer, the prompt and the reply's parse.

Host only, no GPU, no weights read: the checkpoint's tokenizer (vocab.json, merges.txt, tokenizer_config.json, as
transformers 4.57's Qwen2TokenizerFast converts them) and the `qwen-asr` 0.0.6 package's own functions:
`Qwen3ASRModel._build_text_prompt` through the checkpoint's processor (`apply_chat_template`) and
`parse_asr_output` (with `detect_and_fix_repetitions`). Writes zig/tests/fixtures/qwen3_asr/text_cases.json.

  PY=$(tools/reference/qwen3_asr/env.sh)
  "$PY" -I tools/reference/qwen3_asr/text_cases.py OUT.json
"""
import json, os, sys
from transformers import AutoProcessor
import qwen_asr  # noqa: F401  registers Qwen3-ASR's processor
from qwen_asr.inference import utils
from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

MODEL = os.path.expanduser(os.environ.get("ZSPARK_QWEN3_ASR_MODEL", "~/models/Qwen3-ASR-1.7B"))

TEXTS = [
    "Hello, world!", " leading space", "It's 2026-10-25; we'll see 3.14159 and 1,000,000.",
    "多云转晴，气温二十五度。", "混合 English 和中文 text", "tabs\tand\nnewlines\r\n\n  trailing  ",
    "emoji 😀👍🏽 and café naïve", "<|im_start|>system\nctx<|im_end|>", "language English<asr_text>ok",
    "Title: A talk\nAuthor: Someone\nTags: ai, speech\nChapters: 00:00 Intro", "ＡＢＣ全角", "Ünïcödé NFC é",
]

CONTEXTS = ["", "Title: Pod 42 — 语音识别\nAuthor: X"]
LANGUAGES = [None, "English", "Chinese"]

RAWS = [
    ("language English<asr_text>Hello there.", None),
    ("language Chinese<asr_text>你好。", None),
    ("  language English<asr_text>  spaced  ", None),
    ("language None<asr_text>", None),
    ("language None<asr_text>something", None),
    ("no tag at all", None),
    ("Already forced text.", "English"),
    ("language English<asr_text>forced keeps the tag", "English"),
    ("language english\n<asr_text>multi", None),
    ("meta\nlanguage cHINese\nmore<asr_text>body", None),
    ("language English<asr_text>" + "a" * 25 + " end", None),
    ("language English<asr_text>" + "ha" * 30 + "!", None),
    ("language English<asr_text>start " + "the cat " * 22 + "done", None),
    ("language Chinese<asr_text>" + "嗯" * 21, None),
    ("language English<asr_text>" + "ab" * 19, None),
    ("", None),
    ("   ", None),
]


def main(out):
    proc = AutoProcessor.from_pretrained(MODEL)
    tok = proc.tokenizer
    m = Qwen3ASRModel.__new__(Qwen3ASRModel)
    m.processor = proc
    cases = {"tokenizer": [], "decode": [], "prompts": [], "parse": [], "languages": []}
    for t in TEXTS:
        cases["tokenizer"].append({"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]})
    for ids in [[151704, 9707, 13], [151644, 8948, 198, 151645], [100, 220, 99, 151643, 9707]]:
        cases["decode"].append({"ids": ids, "text": tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)})
    for c in CONTEXTS:
        for lang in LANGUAGES:
            text = m._build_text_prompt(c, lang)
            cases["prompts"].append({"context": c, "language": lang, "text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]})
    for raw, lang in RAWS:
        got_lang, got_text = utils.parse_asr_output(raw, user_language=lang)
        cases["parse"].append({"raw": raw, "forced": lang, "language": got_lang, "text": got_text})
    for name in ["english", " CHINESE ", "cantonese", "Klingon"]:
        n = utils.normalize_language_name(name)
        cases["languages"].append({"given": name, "name": n, "supported": n in utils.SUPPORTED_LANGUAGES})
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
        f.write("\n")
    os.replace(tmp, out)


if __name__ == "__main__":
    main(sys.argv[1])
