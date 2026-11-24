"""Write the tokenizer's test fixtures: what the `tokenizers` library gives, for zig/src/text/tokenizer.zig to equal.

Three files go to the output directory (zig/tests/fixtures/text/ in the repository):

- `tokenizer_cases.json`: a corpus encoded with the checkpoint's own tokenizer.json (`add_special_tokens=False`),
  and id sequences decoded by it. The Zig test that reads it needs the 12.8 MB tokenizer.json, so it runs only
  with ZSPARK_FLASH_NEXT_FP4_LOCAL set.
- `tiny_tokenizer.json`: a synthetic tokenizer with the checkpoint's normalizer, pre-tokenizer and decoder, some
  290 vocabulary entries, a handful of merges built to tie and to reach one token by two routes, and added tokens
  carrying the flags TensorFold's tokenizer reads (`lstrip`, `rstrip`, `special`). Bytes 0x09, 0xF0, `z` and `Q` are left out
  of its vocabulary, so the library drops them. Merges are in the legacy "a b" form the checkpoint uses;
  `tiny_tokenizer_pairs.json` is the same tokenizer with its merges as pairs, the form the library saves.
- `tiny_cases.json`: the corpus and more through the tiny tokenizer, plus the library's NFC of some strings and
  the pieces its `Split` pre-tokenizer cuts them into. These tests always run.

Every file is written by the library and then reloaded from disk before the expected values are computed.

`--more N --more-dir DIR` also writes `tokenizer_cases_more.json` and `tiny_cases_more.json` to DIR (outside the
repository): N strings each of random code points from every plane mixed with the corpus alphabet, whitespace,
marks and added tokens. The Zig tests read them when ZSPARK_TOKENIZER_MORE names DIR.

    CK=~/.cache/huggingface/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/b9ce18a6887e18c089fdea8509a33e63429a3300
    docker run --rm --user "$(id -u):$(id -g)" --entrypoint python -v ~/.cache/huggingface:$HOME/.cache/huggingface:ro \\
      -v "$PWD":/repo:ro -v "$PWD/tests/fixtures/text":/out tensorfold-glm53:v0.6.0 \\
      /repo/tools/reference/tokenizer_cases.py "$CK" --repo /repo --out-dir /out
"""

import argparse
import json
import os
import random

import tokenizers
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CJK = (
    "分词器把文本切成词元：先做 NFC 归一化，再按正则表达式切分，最后用字节对编码合并。"
    "模型一共有二十四万八千零四十四个词元，其中包括常用汉字、标点符号（，。！？；：“”‘’【】《》）和数字１２３。"
    "東京都は日本の首都です。ひらがなとカタカナ、ｶﾀｶﾅ、そして漢字が混ざっています。"
    "한국어 문장도 있습니다. 토크나이저는 한글 음절을 바이트로 나눕니다.\n"
    "第二段：价格是￥1,234.50，时间是2026年10月4日 16:08。网址 https://example.com/路径?q=查询&x=1 也要正确切分。"
)

PROSE = [
    "",
    " ",
    "a",
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. THE QUICK BROWN FOX; the quick brown fox?",
    "It's what I'd say: they're here, we've won, I'm sure, you'll see, he didn't. Who'd know?",
    "IT'S WHAT I'D SAY: THEY'RE HERE, WE'VE WON, I'M SURE, YOU'LL SEE, HE DIDN'T.",
    "it'S i'D they'Re we'vE i'M you'lL didn'T 'S 'T 'RE 'VE 'M 'LL 'D 's't're've'm'll'd",
    "'ſ it'ſ 'K 'ſt 'r 'v 'l 'rx 'lx 'l'l ''s ' s 'se 'ree 'sé '",
    "rock'n'roll o'clock 'tis 'em don''t x'll'll y're've",
    # after a letter the contraction is cut off what follows it; U+017F folds to s, U+212A (Kelvin) is no letter here
    "it'ſt a'ſs b'sſ c'\u212at d'Ts e'Mm f'dd g'rex h'VEx i'lLl j'tſ",
    "0 1 12 123 1234 12345 1234567890 3.14159 1e-9 0x1F 1,000,000 99% $5.00 #42 2026-10-04T16:08:00Z",
    "٣٤٥ ४५६ ௧௨ ⅓ ½ Ⅻ ²³ ① 一二三 １２３ 𝟘𝟙 a1b2c3 x²",
    "a b  c   d    e     f      g",
    "tabs\there\t\tand\t\t\tthere \t mixed\t ",
    "line one\nline two\n\nline four\n\n\nline seven\n",
    "crlf one\r\ncrlf two\r\n\r\ncr only\rmixed \r\n \n\r end",
    "trailing spaces   ",
    "trailing newline\n",
    "trailing mix \n \t \n  ",
    "   leading spaces",
    "\n\nleading newlines",
    " \n \n a \n\n  b\n  \n",
    "a \nb  \nc   \nd\t\ne \r\n f",
    "x = 1;\n    y = 2;\n\tz = 3;\n        return x + y + z;\n",
    "!!! ??? ... --- === *** ((())) [[]] {{}} <<>> ->=> != <= >= && || :: ;; ,, `` ~~ ^^ \\\\ // @@ $$",
    "a. b, c; d: e! f? g- h_ i+ j= k/ l\\ m| n~ o` p@ q# r$ s% t^ u& v* w( x) y[ z]",
    "hello !\nhello!\n hello ! \nhello !\r\nhello !\n\nhello",
    ". \n. \r\n.\n\n .\n ..\n",
    "snake_case camelCase PascalCase SCREAMING_SNAKE kebab-case dotted.name path/to/file.zig C:\\Windows\\System32",
    "def f(x):\n    return [i ** 2 for i in range(x) if i % 2 == 0]\n\nprint(f(10))\n",
    'const std = @import("std");\n\npub fn main() !void {\n    std.debug.print("{s}\\n", .{"hi"});\n}\n',
    '{"key": [1, 2.5, true, null, "va\\"lue"], "nested": {"a": {"b": []}}}',
    "<html><body class=\"x\">a &amp; b</body></html><!-- c --><br/>",
    "non\u00a0breaking\u00a0space, thin\u2009space, ideographic\u3000space, line\u2028separator, para\u2029graph, nel\u0085x",
    "\u00a0\u00a0lead \u3000\u3000 tail\u2009\u2009",
    "zero\u200bwidth\u200cnon\u200djoiner\ufeffbom\u00adsoft",
    "control \x00 \x01 \x1b[0m \x7f chars \x1c\x1d\x1e\x1f",
    "naïve café résumé Ångström Zürich jalapeño smörgåsbord œuvre straße ǅ ĳ ŉ",
    "Ελληνικά κείμενο, Русский текст, Українська мова, עברית, ქართული, Հայերեն",
    "العربية لغة جميلة. مَرْحَبًا بِكُمْ ١٢٣",
    "हिन्दी भाषा में लिखा गया वाक्य। क्षत्रिय श्री ज़िंदगी",
    "ภาษาไทยไม่มีช่องว่างระหว่างคำ น้ำ ที่นี่ ๑๒๓",
    "தமிழ் বাংলা ગુજરાતી ਪੰਜਾਬੀ తెలుగు ಕನ್ನಡ മലയാളം සිංහල မြန်မာ ខ្មែរ ລາວ བོད་",
    "漢字かなカナ 한글 ㄱㄴㄷ ㈱ ㍿ 〇 々 〆 ヶ ー ・",
    "中文，没有空格。English mixed 中文 mixed123数字４５６。",
    "😀 😃😄 👍🏽 👩‍👩‍👧‍👦 👨🏿‍💻 🏳️‍🌈 🇯🇵🇺🇸 🇨 1️⃣ ❤️ ❤ ☺︎ 🧑‍🤝‍🧑 🫠",
    "text😀text 😀\n😀  😀 a👍🏽b 🇩🇪x",
    "math: ∑∏∫√∞≈≠≤≥ ∀x∃y x̄ ŷ ℝ ℵ₀ → ← ↔ ⇒ α+β=γ",
    "currency: $ € £ ¥ ₹ ₿ ¢ ₩; symbols: © ® ™ § ¶ † ‡ • … ‰ ′ ″ « » „ “ ” ‘ ’ – —",
    # decomposed input that NFC recomposes
    "e\u0301 a\u0300 o\u0308 n\u0303 c\u0327 A\u030a u\u0308\u0304 cafe\u0301 Zu\u0308rich",
    # input NFC must reorder: dot below (220) after acute (230), cedilla (202) after both
    "q\u0301\u0323 a\u0301\u0323\u0327 e\u0308\u0323 o\u031b\u0323\u0301 s\u0307\u0323 d\u0323\u0307 a\u0315\u0300\u05ae\u0062",
    # blocked composition, singletons, exclusions
    "a\u0308\u0301 a\u0301\u0308 \u212b \u2126 \u212a \u0340 \u0341 \u0343 \u0344 \u0374 \u037e \u0387 \u0958 \u09dc \ufb1f \u2adc \U0001d15e",
    "\u1e9b\u0323 \u1e9b \u01c4 \ufb01 \u2460 \uff21 \u00b5 \u0132 \u1fbe \u2000 \u2001 \u3008 \u2329 \uf900 \ufa0e \U0002f800",
    # Hangul: conjoining jamo that compose, syllables, partial sequences
    "\u1112\u1161\u11ab\u1100\u1173\u11af \u1100\u1161 \uac00\u11a8 \uac01\u11a8 \u1100 \u1161 \u11a8 \u1100\u1100\u1161 \u1161\u11a8 \u3131\u314f",
    # lone combining marks, at the start, after spaces, digits, punctuation and newlines
    "\u0301",
    "\u0301\u0301 \u0301a \u0323\u0301",
    "a \u0301 1\u0301 .\u0301 \n\u0301 \t\u0308x 9\u20e3 -\u0308- '\u0301s",
    "\u093f \u0947\u0902 \u0e31 \u05b0\u05b1 \u0651\u064e \u3099\u309a か\u3099 は\u309a \u1ab0 \u20d0",
    # marks newer than the normalizer's data, and letters newer than Unicode 15
    "a\u0c3c\u0301 \u07fd\u0301\u0323 \u0d3b\u0301 \U00011935\U00011930 \U00011938 \u1c89 \ua7cb \U00010d40\U00010d69",
    # unassigned, private use, noncharacters, astral
    "\u0378 \ue000 \uf8ff \ufdd0 \uffff \ufffd \U000e0001 \U0010ffff \U000f0000 \U0001f9ff \U00020000 \U0002ebf0 \U00031350",
    "<|im_sta",
    "<|im_start",
    "<|im_start |>",
    "<|im_end|",
    "|im_end|>",
    "< |im_end|>",
    "<|IM_START|>",
    "<think",
    "<think >",
    "</think",
    "<tool_call",
    "<tool_cal<tool_call>l>",
    "<|<|im_start|>|>",
    "<<think>>",
    "<|endoftext|",
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nList the primes below 20.<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n2, 3, 5, 7, 11, 13, 17, 19<|im_end|>\n",
    "<|im_start|>assistant\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"北京\"}}\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n{\"temp\": 21}\n</tool_response><|im_end|>\n<|endoftext|>",
    "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>What is this?",
    "<|fim_prefix|>fn a() {<|fim_suffix|>}<|fim_middle|>",
    "e<think>\u0301 >\u0338<think>\u0338 a\u0301<|im_end|>\u0323",
    " " * 300 + "x" + " " * 300,
    "\n" * 100 + "x" + "\t" * 100 + " \n" * 50,
    "a" * 500,
    "ab" * 300 + "!" * 200 + "=" * 200 + "-" * 300 + "." * 100 + "0" * 100,
]

ALPHABET = (
    list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("etaoinshrdlu " * 4)
    + list("0123456789" * 2)
    + list("        \n\n\n\t\t\r")
    + list("''''''.,;:!?-_()[]{}<>|/\\\"=+*&^%$#@~`")
    + list("stmdSTMD") * 2
    + ["re", "ve", "ll", "RE", "Ve", "lL", "'s", "'t", "'re", "'ll", "\r\n", "  ", "   ", " \n", "\n "]
    + list("中文字符日本語汉字")
    + list("ひらがなカタカナ")
    + list("한글음절")
    + list("абвгдЖЗ")
    + list("αβγé ñüß")
    + list("\u0301\u0300\u0308\u0323\u0327\u031b\u0304\u0345\u093f\u094d\u0e31\u3099")
    + list("\u1100\u1161\u11a8\uac00")
    + list("😀👍🏽❤✓→∑")
    + ["\u200d", "\ufe0f", "🇯", "🇵", "\u00a0", "\u2028", "\u3000", "\u0085", "\u017f", "\u2003"]
    + ["<", "|", ">", "im_start", "im_end", "think", "/", "<|", "|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>"]
)


def mapped_char(byte):
    """The character the ByteLevel pre-tokenizer writes for a byte (GPT-2's table)."""
    kept = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    if byte in kept:
        return chr(byte)
    return chr(0x100 + [b for b in range(256) if b not in kept].index(byte))


def mapped(text):
    return "".join(mapped_char(b) for b in text.encode("utf-8"))


def added_token_cases(contents):
    cases = []
    for i, t in enumerate(contents):
        other = contents[(i * 7 + 3) % len(contents)]
        cases += [
            t,
            "a" + t + "b",
            " " + t + " ",
            "x " + t,
            t + " y",
            "\n" + t + "\n",
            " \n" + t + "\t ",
            t + t,
            t + other + t,
            "é" + t + "\u0301" + t + "1",
            t[:-1],
            t[1:],
            t[:-2] + t,
            t[: len(t) // 2] + " " + t[len(t) // 2 :],
            "_" + t + "_ 9" + t + "9 '" + t + "'s",
        ]
    cases.append("".join(contents))
    cases.append(" ".join(contents))
    cases.append("\n".join(reversed(contents)))
    return cases


def random_strings(rng, alphabet, count, longest):
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, longest))) for _ in range(count)]


def wild_strings(rng, extra, count):
    """Strings over every plane: any code point, the corpus alphabet, whitespace, marks, and `extra`."""
    spaces = [0x9, 0xA, 0xB, 0xC, 0xD, 0x20, 0x85, 0xA0, 0x1680, 0x2000, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000]
    edge = [chr(c) for c in spaces + [0x1C, 0x1F, 0x200B, 0x200C, 0x200D, 0xFEFF, 0x27, 0x17F, 0x5F, 0x203F, 0x2040]]
    marks = [chr(c) for c in range(0x300, 0x370)] + list("\u0591\u05b0\u0610\u064b\u0670\u093c\u094d\u0e31\u0f71\u1dc0\u20d0\u3099\u302a\ufe20")

    def one():
        r = rng.random()
        if r < 0.3:
            return rng.choice(ALPHABET)
        if r < 0.5:
            c = rng.randrange(0x110000)
            return chr(c) if not 0xD800 <= c <= 0xDFFF else " "
        if r < 0.65:
            return chr(rng.randrange(0x20, 0x3400))
        if r < 0.75:
            return rng.choice(edge)
        if r < 0.85:
            return rng.choice(marks)
        if r < 0.9 and extra:
            return rng.choice(extra)
        return rng.choice("abc 'stmdrevlSTMDREVL\n\r")

    return ["".join(one() for _ in range(rng.randint(1, 30))) for _ in range(count)]


def encode_cases(tokenizer, texts, plain_must_round_trip):
    nfc = normalizers.NFC()
    added = [t.content for t in tokenizer.get_added_tokens_decoder().values()]
    cases, seen = [], set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if plain_must_round_trip and not any(a in text for a in added):
            assert decoded == nfc.normalize_str(text), repr(text)
        case = {"text": text, "ids": ids}
        if decoded != text:
            case["decoded"] = decoded
        cases.append(case)
    return cases


def decode_cases(tokenizer, id_lists):
    cases = []
    for ids in id_lists:
        shown = tokenizer.decode(ids, skip_special_tokens=False)
        skipped = tokenizer.decode(ids, skip_special_tokens=True)
        case = {"ids": ids, "text": shown}
        if skipped != shown:
            case["skipped"] = skipped
        cases.append(case)
    return cases


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    return os.path.getsize(path)


def real(checkpoint, repo, out_dir, more, more_dir):
    path = os.path.join(checkpoint, "tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    contents = [t["content"] for t in raw["added_tokens"]]
    rng = random.Random(4)

    texts = list(PROSE) + [CJK] + added_token_cases(contents)
    for name, length in (("README.md", 3000), ("zig/src/weights/numeric.zig", 3000), ("AGENTS.md", 1500)):
        with open(os.path.join(repo, name), encoding="utf-8") as f:
            texts.append(f.read()[:length])
    texts += random_strings(rng, ALPHABET, 500, 40)
    texts += random_strings(rng, ALPHABET, 40, 400)
    cases = encode_cases(tokenizer, texts, plain_must_round_trip=True)

    size = tokenizer.get_vocab_size(with_added_tokens=True)
    byte_ids = [tokenizer.token_to_id(mapped_char(b)) for b in range(0x80, 0x100)]
    id_lists = [[], [size], [size + 7, 0, 2**32 - 1, 1]]
    id_lists += [[rng.randrange(size) for _ in range(rng.randint(1, 24))] for _ in range(60)]
    id_lists += [[rng.choice(byte_ids) for _ in range(rng.randint(1, 12))] for _ in range(60)]
    id_lists += [[rng.randrange(size - len(contents), size) for _ in range(rng.randint(1, 6))] for _ in range(10)]
    # a character cut at every byte, alone and between other tokens
    for text in ("é", "中", "😀", "👩‍👩‍👧‍👦"):
        ids = [tokenizer.token_to_id(mapped_char(b)) for b in text.encode("utf-8")]
        id_lists += [ids, ids[:-1], ids[1:], ids[:-1] + ids, ids[:1] + [size - 1] + ids[1:], [64] + ids[:-1] + [65]]
    fixture = {
        "tokenizers": tokenizers.__version__,
        "revision": os.path.basename(os.path.normpath(checkpoint)),
        "vocabulary": size,
        "cases": cases,
        "decode": decode_cases(tokenizer, id_lists),
    }
    bytes_written = write_json(os.path.join(out_dir, "tokenizer_cases.json"), fixture)
    print("tokenizer_cases.json: %d cases, %d decode cases, %d bytes" % (len(cases), len(fixture["decode"]), bytes_written))
    if more:
        texts = wild_strings(rng, contents + [c[:-1] for c in contents], more)
        wild = {"cases": encode_cases(tokenizer, texts, plain_must_round_trip=True), "decode": []}
        wild["decode"] = decode_cases(tokenizer, [[rng.randrange(size + 2) for _ in range(rng.randint(1, 24))] for _ in range(more)])
        print("tokenizer_cases_more.json: %d cases, %d bytes" % (len(wild["cases"]), write_json(os.path.join(more_dir, "tokenizer_cases_more.json"), wild)))


def tiny(out_dir, more, more_dir):
    rng = random.Random(4)
    absent = {0x09, 0xF0, ord("z"), ord("Q")}
    chars = [mapped_char(b) for b in range(256) if b not in absent]
    rng.shuffle(chars)
    vocab = {c: i for i, c in enumerate(chars)}
    merges = [
        ("a", "b"),
        ("b", "c"),
        ("ab", "c"),  # abc, also reached by the next merge
        ("a", "bc"),
        ("a", "a"),
        ("aa", "a"),
        ("c", "c"),
        ("b", "a"),
        ("ab", "ab"),
        ("c", "a"),
        ("bc", "a"),
        ("cc", "a"),  # cca, also reached by the next merge
        ("c", "ca"),
        ("aa", "aa"),
        ("ba", "b"),  # bab, also reached by the next merge
        ("b", "ab"),
        ("Ġ", "t"),
        ("h", "e"),
        ("Ġt", "he"),
        ("l", "l"),
        ("e", "r"),
        ("i", "n"),
        ("Ġ", "Ġ"),
        ("ĠĠ", "ĠĠ"),
        ("Ċ", "Ċ"),
        ("'", "s"),
        ("1", "2"),
        (mapped_char(0xC3), mapped_char(0xA9)),  # é
        (mapped_char(0xE4), mapped_char(0xB8)),  # the first two bytes of 中
        (mapped_char(0x9F), mapped_char(0x98)),  # the middle of an emoji whose first byte is absent
        ("a", "b"),  # a pair listed twice: the later rank is the one in force
    ]
    for a, b in merges:
        vocab.setdefault(a + b, len(vocab))
    vocab["€uro"] = len(vocab)  # not byte-level: the decoder writes it as it is
    vocab["Ġthe" + "Ġthe"] = len(vocab) + 5  # leaves ids without a token

    built = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    built.normalizer = normalizers.NFC()
    built.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split(Regex(SPLIT), "isolated"), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    built.decoder = decoders.ByteLevel()
    built.add_special_tokens(
        [
            AddedToken("<|end|>", special=True, normalized=False),
            AddedToken("<l>", special=True, normalized=False, lstrip=True),
        ]
    )
    built.add_tokens(
        [
            AddedToken("<t>", normalized=False),
            AddedToken("<t>x", normalized=False),
            AddedToken("<r>", normalized=False, rstrip=True),
            # no `single_word`: TensorFold's tokenizer, which zig/src/text/tokenizer.zig stands on, refuses the flag
            AddedToken("<w>", normalized=False),
            AddedToken("<b>", normalized=False, lstrip=True, rstrip=True),
            AddedToken("<w>!", normalized=False),
            AddedToken(">>", normalized=False),  # starts inside "<w>>": found only if the search resumes inside
            AddedToken(" >>", normalized=False),  # starts in the whitespace an rstrip token before it has taken
        ]
    )
    pairs_path = os.path.join(out_dir, "tiny_tokenizer_pairs.json")
    with open(pairs_path, "w", encoding="utf-8") as f:
        f.write(built.to_str(pretty=False) + "\n")
    with open(pairs_path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"] == SPLIT
    # The library saves a pair once; the list as given, with the pair that is in it twice, is written back.
    assert len(raw["model"]["merges"]) == len(merges) - 1
    raw["model"]["merges"] = [list(m) for m in merges]
    write_json(pairs_path, raw)
    raw["model"]["merges"] = [" ".join(m) for m in merges]
    path = os.path.join(out_dir, "tiny_tokenizer.json")
    write_json(path, raw)
    tokenizer = Tokenizer.from_file(path)
    pairs = Tokenizer.from_file(pairs_path)

    contents = [t["content"] for t in raw["added_tokens"]]
    flagged = [
        "a <l> b",
        "a   <l>b",
        "a\u00a0\u3000<l>\u00a0b",
        "a<l>",
        "<l>",
        " <l> <l> ",
        "<t> <l>",
        "<t>  <l>  <t>",
        "a <r> b",
        "a<r>   b",
        "<r>\n\t <r> \u2028x",
        "<r>",
        "<r> ",
        "<r> <l> c",
        "<r>   <l>",
        "<r> <t>",
        "<r>   <b>  <l>",
        "x <b>  y",
        "x<b>y",
        " <b> ",
        "<b> <b>",
        "a<w>",
        "<w>a",
        "a <w> b",
        "a <w>b",
        "(<w>)",
        "é<w>",
        "<w>é",
        "_<w>",
        "<w>_",
        "1<w> <w>1",
        "<w>.",
        "-<w>-",
        "\u0301<w>",
        "<w>\u0301",
        "\u200d<w> <w>\u200c",
        "中<w> <w>中",
        "①<w> Ⅳ<w> ²<w> \u24b6<w>",
        "<w><w>",
        "<w> <w>",
        "<w><t>",
        "<t><w>",
        "a<w>!",
        "<w>! <w>!a",
        "<l><w>",
        "<w><l>",
        "<w> <l>a",
        "<r> <w>",
        "<r> <w>a",
        "a<w> <r>",
        "<t>x",
        "<t>xx",
        "<t>y<t>x<t>",
        "<t><t>x<t>",
        "<t",
        "<t ><|end|",
        "a<w>>",
        "<w>>",
        "a<w>> b<w>>>",
        "a>>b >> c  >>",
        "<r> >>",
        "<r>  >> <r>>>",
        "<l> >><l>",
        "aab aabc abab baab aaab",
        "ab<|end|>abc",
        "<|end|><|end|>",
        "<<t>>",
        "abc <|end|> the the\n\nthe",
        "zebra Quiz\tzz z'sQ",
        "😀 中 é e\u0301 ä¸",
    ]
    small = list("abc") * 6 + list(" \n'slthein12z") + ["é", "中", "😀", "<t>", "<w>", "<l>", "<r>", "x", "\u0301"]
    texts = list(PROSE) + [CJK[:400]] + flagged + added_token_cases(contents)
    texts += random_strings(rng, list("abc"), 500, 24)
    texts += random_strings(rng, small, 300, 24)
    texts += random_strings(rng, ALPHABET, 100, 30)
    cases = encode_cases(tokenizer, texts, plain_must_round_trip=False)
    for case in cases:
        assert pairs.encode(case["text"], add_special_tokens=False).ids == case["ids"]

    top = max(vocab.values()) + len(contents) + 3
    id_lists = [[], [top + 100], list(range(top))]
    id_lists += [[rng.randrange(top) for _ in range(rng.randint(1, 16))] for _ in range(150)]
    for text in ("é", "中", "👩‍👩‍👧‍👦"):
        # the emoji's first byte has no token here, so its ids are a run of continuation bytes
        ids = [vocab[mapped_char(b)] for b in text.encode("utf-8") if b not in absent]
        id_lists += [ids, ids[:-1], ids[1:], ids[:-1] + ids, ids[:1] + [top - 4] + ids[1:]]

    nfc = normalizers.NFC()
    split = pre_tokenizers.Split(Regex(SPLIT), "isolated")
    plain = [t for t in PROSE + [CJK[:400]] + random_strings(rng, ALPHABET, 150, 30) if len(t) < 400]
    marks = list("\u0301\u0300\u0308\u0323\u0327\u031b\u0304\u0345\u0315\u05ae\u093c\u094d\u3099\u309a\u0c3c\u07fd\u0338")
    bases = list("aeouAEOUcnsd \u00e9\u1e63\u1e9b\u0915\u0928か\u1100\u1161\u11a8\uac00\uac01\u212b\u0344\u0958>=<")
    nfc_texts = plain + random_strings(rng, marks + bases, 400, 10)
    fixture = {
        "tokenizers": tokenizers.__version__,
        "cases": cases,
        "decode": decode_cases(tokenizer, id_lists),
        "nfc": [{"text": t, "nfc": nfc.normalize_str(t)} for t in nfc_texts if nfc.normalize_str(t) != t],
        "nfc_unchanged": [t for t in nfc_texts if nfc.normalize_str(t) == t and any(ord(c) >= 0x300 for c in t)][:80],
        "pieces": [{"text": t, "pieces": [p for p, _ in split.pre_tokenize_str(t)]} for t in plain if t],
    }
    bytes_written = write_json(os.path.join(out_dir, "tiny_cases.json"), fixture)
    if more:
        texts = wild_strings(rng, contents + list("abc") * 3, more)
        wild = {"cases": encode_cases(tokenizer, texts, plain_must_round_trip=False), "decode": []}
        wild["decode"] = decode_cases(tokenizer, [[rng.randrange(top) for _ in range(rng.randint(1, 24))] for _ in range(more)])
        wild["nfc"] = [{"text": t, "nfc": nfc.normalize_str(t)} for t in texts]
        wild["pieces"] = [{"text": t, "pieces": [p for p, _ in split.pre_tokenize_str(t)]} for t in texts]
        print("tiny_cases_more.json: %d cases, %d bytes" % (len(wild["cases"]), write_json(os.path.join(more_dir, "tiny_cases_more.json"), wild)))
    print(
        "tiny_cases.json: %d cases, %d decode, %d nfc, %d pieces, %d bytes; tiny_tokenizer.json %d bytes"
        % (len(cases), len(fixture["decode"]), len(fixture["nfc"]), len(fixture["pieces"]), bytes_written, os.path.getsize(path))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint", help="the checkpoint's directory, with tokenizer.json")
    parser.add_argument("--repo", required=True, help="zspark's root, for the files the corpus quotes")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--more", type=int, default=0, help="this many random strings more, to --more-dir")
    parser.add_argument("--more-dir", help="where the larger, unchecked-in fixtures go")
    args = parser.parse_args()
    if args.more and not args.more_dir:
        parser.error("--more needs --more-dir")
    tiny(args.out_dir, args.more, args.more_dir)
    real(args.checkpoint, args.repo, args.out_dir, args.more, args.more_dir)


if __name__ == "__main__":
    main()
