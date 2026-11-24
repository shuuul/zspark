"""README's first measured table, from one `zspark generate` load of one checkpoint.

The cases are the eight replies of `client_bench.json`, two serial replies on the short prose and code
prompts, a short reply whose end is the device-memory sample, and the harness's 4000-token prompt. Serial
is the plain one-token loop. Nothing stops at an end token: `generate` runs the count it is given. The
prompt's seconds are the model's, and its token count is whatever the template rendered.

`prepare` renders the prompts and writes the case file. It does not load the model. `measure` is the load.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

REPO = Path(__file__).resolve().parents[2]
SET = Path(__file__).resolve().parent / "client_bench.json"
TOKEN_LINE = re.compile(r"^(\d+) tokens in ([\d.]+) s: ([\d.]+) tok/s")
PROMPT_LINE = re.compile(r"^prompt ([\d.]+) s")
LOAD_LINE = re.compile(r"^loaded in ([\d.]+) s")
NLL_LINE = re.compile(r"^all: (\d+) tokens, nll ([\d.]+),")
IDS_LINE = re.compile(r"^(\d+) ids:(.*)$")


@dataclass(frozen=True)
class CaseResult:
    """One case's printed clocks."""

    name: str
    prompt_s: float
    tokens: int
    seconds: float
    tok_s: float
    serial: bool


def _spec() -> dict[str, object]:
    raw = json.loads(SET.read_text(encoding="utf-8"))
    text = "".join((REPO / path).read_text(encoding="utf-8") for path in raw["corpus"])
    raw["text"] = text
    return raw


def _salt(seed: int, name: str, repeat: int) -> str:
    return hashlib.sha256(f"{seed}:{name}:{repeat}".encode()).hexdigest()[:16]


def _rows(checkpoint: Path) -> int:
    # the session's prompt windows (`session.promptRows`): 2048 on every checkpoint since the EXL3 packs' 512
    # went with them (spec 121)
    return 2048


def _render(zspark: Path, checkpoint: Path, messages: Path, tools: Path | None, thinking: bool) -> tuple[int, str]:
    command: list[str] = [str(zspark), "chat", str(checkpoint), "--render", "--messages", str(messages)]
    if tools is not None:
        command.extend(["--tools", str(tools)])
    if not thinking:
        command.append("--no-thinking")
    done = subprocess.run(command, check=True, capture_output=True, text=True, cwd=REPO)
    ids = ""
    count = 0
    for line in done.stdout.splitlines():
        matched = IDS_LINE.match(line)
        if matched is not None:
            count = int(matched.group(1))
            ids = matched.group(2).strip()
    if count == 0 or not ids:
        raise RuntimeError(f"chat --render printed no ids\n{done.stdout[-400:]}")
    return count, ids


def _write_messages(path: Path, content: str) -> None:
    path.write_text(json.dumps([{"role": "user", "content": content}], ensure_ascii=False), encoding="utf-8")


def prepare(checkpoint: Path, out: Path, zspark: Path, seed: int) -> None:
    """Render every prompt and write `cases.txt`. Does not load the model.

    Args:
        checkpoint: The pack or the NVFP4 directory.
        out: Where the case file and the rendered counts are written.
        zspark: The `zspark` binary. `chat --render` returns before a load.
        seed: The salt of the long prompt, the harness's seed.
    """

    out.mkdir(parents=True, exist_ok=True)
    spec = _spec()
    tools_path = out / "tools.json"
    tools_path.write_text(json.dumps(spec["tools"]), encoding="utf-8")
    messages = out / "messages.json"
    lines: list[str] = []
    counts: dict[str, int] = {}

    def add(name: str, count: int, content: str, *, plain: bool = False, tools: bool = False,
            thinking: bool = False, sample: str | None = None) -> None:
        _write_messages(messages, content)
        n, ids = _render(zspark, checkpoint, messages, tools_path if tools else None, thinking)
        print(f"rendered {name}: {n} ids", file=sys.stderr)
        counts[name] = n
        words = [str(count)]
        if plain:
            words.append("plain")
        if sample is not None:
            words.append(f"sample={sample}")
        words.append(f"name={name}")
        lines.append(" ".join(words) + " " + ids)

    warm = spec["warm"]
    assert isinstance(warm, dict)
    add("after-reply", int(warm["max_tokens"]), str(warm["user"]))
    decode = spec["decode"]
    assert isinstance(decode, list)
    short = {str(case["name"]): str(case["user"]) for case in decode}
    add("serial-prose", 128, short["prose-short"], plain=True)
    add("serial-code", 128, short["code-short"], plain=True)
    for case in decode:
        assert isinstance(case, dict)
        sample = None
        if "temperature" in case:
            sample = f"{case['seed']},{case['temperature']},{case['top_k']},{case['top_p']}"
        add(str(case["name"]), int(case["max_tokens"]), str(case["user"]), tools=bool(case.get("tools")),
            thinking=bool(case.get("thinking")), sample=sample)
    prompt = spec["prompt"]
    assert isinstance(prompt, dict)
    size = 4000
    name = f"prompt-{size}"
    room = max(0, int(size * float(spec["chars_per_token"])) - len(str(prompt["lead"])) - len(str(prompt["question"])) - 24)
    text = str(spec["text"])
    if room > len(text):
        raise RuntimeError(f"the corpus has {len(text)} characters and {room} are asked")
    content = f"[{_salt(seed, name, 0)}]\n" + str(prompt["lead"]) + text[:room] + str(prompt["question"])
    add("prompt-4054", 2, content)

    (out / "cases.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "rows.txt").write_text(str(_rows(checkpoint)) + "\n", encoding="utf-8")
    (out / "counts.json").write_text(json.dumps(counts, indent=1) + "\n", encoding="utf-8")


def _device_mib() -> int:
    done = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True)
    total = 0
    for line in done.stdout.splitlines():
        text = line.strip()
        if text.isdigit():
            total += int(text)
    return total


def _vmhwm_kib(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1])
    return 0


def _watch(pid: int, stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        try:
            peak[0] = max(peak[0], _vmhwm_kib(pid))
        except FileNotFoundError:
            return
        stop.wait(0.2)


def measure(checkpoint: Path, out: Path, zspark: Path, score: Path) -> None:
    """One generate load. Writes `generate.txt`, `meta.json` and `table.md`.

    Args:
        checkpoint: The pack or the NVFP4 directory.
        out: The directory `prepare` wrote.
        zspark: A `-Doptimize=fast` binary.
        score: Whitespace-separated token ids, scored after the cases on this load.
    """

    rows = (out / "rows.txt").read_text(encoding="utf-8").strip()
    counts: Mapping[str, int] = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    command = [str(zspark), "generate", str(checkpoint), "0", f"rows={rows}", f"score={score}",
               f"cases={out / 'cases.txt'}"]
    stderr_path = out / "generate.stderr"
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_handle, text=True, cwd=REPO)
    assert proc.stdout is not None and proc.pid is not None
    stop = threading.Event()
    peak = [0]
    thread = threading.Thread(target=_watch, args=(proc.pid, stop, peak), daemon=True)
    thread.start()
    stdout_path = out / "generate.txt"
    results: dict[str, CaseResult] = {}
    name = ""
    prompt_s = 0.0
    loaded: str | None = None
    nll: str | None = None
    scored: str | None = None
    device: int | None = None
    host_at_reply = 0
    try:
        with stdout_path.open("w", encoding="utf-8") as handle:
            for line in proc.stdout:
                handle.write(line)
                handle.flush()
                text = line.rstrip("\n")
                if text.startswith("== "):
                    name = text[3:]
                    prompt_s = 0.0
                    continue
                load = LOAD_LINE.match(text)
                if load is not None and loaded is None:
                    loaded = load.group(1)
                    continue
                prompt = PROMPT_LINE.match(text)
                if prompt is not None:
                    prompt_s = float(prompt.group(1))
                    continue
                tokens = TOKEN_LINE.match(text)
                if tokens is not None and name:
                    results[name] = CaseResult(name, prompt_s, int(tokens.group(1)), float(tokens.group(2)),
                                                float(tokens.group(3)), text.endswith("(serial)"))
                    print(f"case {name}: {tokens.group(1)} tokens, {tokens.group(3)} tok/s", file=sys.stderr)
                    if name == "after-reply" and device is None:
                        device = _device_mib()
                        host_at_reply = _vmhwm_kib(proc.pid)
                    continue
                nll_match = NLL_LINE.match(text)
                if nll_match is not None:
                    scored = nll_match.group(1)
                    nll = nll_match.group(2)
        code = proc.wait()
    finally:
        stop.set()
        thread.join(timeout=2)
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        stderr_handle.close()
    if code != 0:
        raise RuntimeError(f"generate exited {code}; see {out / 'generate.stderr'}")
    if loaded is None or device is None or nll is None or scored is None:
        raise RuntimeError(f"the load's output is missing a cell; see {stdout_path}")
    _write(out, counts, results, loaded, device, host_at_reply, peak[0], nll, scored)


def _range(values: Sequence[float]) -> str:
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 0.05:
        return f"{lo:.1f}"
    return f"{lo:.1f} to {hi:.1f}"


def _write(out: Path, counts: Mapping[str, int], results: Mapping[str, CaseResult], loaded: str, device: int,
           host_at_reply: int, host_peak: int, nll: str, scored: str) -> None:
    serial = [results["serial-prose"], results["serial-code"]]
    kinds = ("prose-short", "prose-long", "code-short", "code-long", "tool-call", "thinking",
             "prose-long-sampled", "code-long-sampled")
    drafted = [results[name] for name in kinds]
    code_cases = [results["code-short"], results["code-long"]]
    tokens = sum(case.tokens for case in drafted)
    seconds = sum(case.seconds for case in drafted)
    prompt = results["prompt-4054"]
    prompt_tokens = counts["prompt-4054"]
    prompt_rate = prompt_tokens / prompt.prompt_s
    cells = {
        "load_s": loaded,
        "device_mib_after_reply": device,
        "host_kib_at_reply": host_at_reply,
        "host_kib_peak": host_peak,
        "serial_tok_s": [case.tok_s for case in serial],
        "drafted_tok_s": tokens / seconds,
        "drafted_tokens": tokens,
        "drafted_seconds": round(seconds, 3),
        "code_tok_s": [case.tok_s for case in code_cases],
        "prompt_tokens": prompt_tokens,
        "prompt_s": prompt.prompt_s,
        "prompt_tok_s": round(prompt_rate, 1),
        "score_tokens": int(scored),
        "nll": nll,
    }
    (out / "meta.json").write_text(json.dumps(cells, indent=1) + "\n", encoding="utf-8")
    host_gb = host_peak / (1024 * 1024)
    lines = [
        f"| Load | {loaded} s |",
        f"| Device memory after a reply | {device} MiB ({device / 1024:.2f} GiB) |",
        f"| Host memory at its peak | {host_peak} KiB VmHWM ({host_gb:.2f} GB); {host_at_reply} KiB at the reply |",
        f"| Decoding, serial | {_range([case.tok_s for case in serial])} tok/s |",
        f"| Decoding, drafted | {tokens / seconds:.1f} tok/s, {_range([case.tok_s for case in code_cases])} on code |",
        f"| A prompt of {prompt_tokens} tokens first met | {prompt.prompt_s:.3f} s ({prompt_rate:.0f} tok/s) |",
        f"| score, {scored} tokens | {nll} nats |",
        "",
        "source lines in generate.txt:",
    ]
    for case in (*serial, *drafted, prompt):
        lines.append(f"- {case.name}: prompt {case.prompt_s:.3f} s; {case.tokens} tokens in {case.seconds:.2f} s: {case.tok_s:.1f} tok/s")
    lines.append(f"- loaded in {loaded} s")
    lines.append(f"- all: {scored} tokens, nll {nll}")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """`prepare CHECKPOINT OUT` or `measure CHECKPOINT OUT SCORE`."""

    if len(sys.argv) < 4:
        raise SystemExit("usage: readme_table.py prepare|measure CHECKPOINT OUT [SCORE]")
    command, checkpoint, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    zspark = Path(os.environ.get("ZSPARK", str(REPO / "zig-out" / "bin" / "zspark")))
    if command == "prepare":
        prepare(checkpoint, out, zspark, int(os.environ.get("SEED", "1")))
    elif command == "measure":
        if len(sys.argv) != 5:
            raise SystemExit("measure CHECKPOINT OUT SCORE")
        measure(checkpoint, out, zspark, Path(sys.argv[4]))
    else:
        raise SystemExit(f"unknown command {command}")


if __name__ == "__main__":
    main()
