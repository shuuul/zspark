"""A vLLM server's greedy replies and logprobs over its API, and zspark's held to them (spec 105 WS-02).

The reference of a checkpoint no TensorFold reads is the vLLM container its card names (`vllm_reference.sh`
starts it). This script is its client and the comparison, standard library only, on the host:

    python3 -I vllm_reference.py fetch URL CASES SCORE_IDS OUT [RUNS]
        each line of CASES (`zspark generate cases=` lines: `N [OPTION..] name=LABEL ID..`, options other than
        name= ignored) as a greedy completion of N tokens with the top 20 logprobs a step, and SCORE_IDS's ids
        as one prompt with the top 20 prompt logprobs a position, RUNS times (2 unless given), the prefix
        cache reset before each run where the server allows it (VLLM_SERVER_DEV_MODE=1); into OUT/NAME.json
        and OUT/score.json, each {"runs": [...]}.
    python3 -I vllm_reference.py compare OUT ZSPARK_OUTPUT
        the replies of a `zspark generate cases=... score-each=...` output against OUT's, a case by its name
        (the drafted cases': a plain case prints no tokens): for each the tokens in common, and at the first that differs vLLM's top-2 gap there; for the score text each
        token's loss against vLLM's, the means, and vLLM's own spread between its runs.

The tolerance is read from vLLM's own figures: how far its two runs differ (score and replies), and how close
its first two choices lie where zspark's reply leaves its own. A gap under twice the score's mean |zspark -
vLLM| per token is called a tie: the two are as near there as they are on an average token.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOP = 20


def post(url: str, body: dict | None, timeout: float = 3600.0) -> dict:
    data = json.dumps(body).encode() if body is not None else b""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode()
    return json.loads(text) if text.strip().startswith("{") else {"text": text}


def model_of(url: str) -> str:
    with urllib.request.urlopen(url + "/v1/models", timeout=60) as response:
        return json.loads(response.read())["data"][0]["id"]


def cases_of(path: Path) -> list[tuple[str, int, list[int]]]:
    out = []
    for line in path.read_text().splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        name, ids = "", []
        for word in words[1:]:
            if word.startswith("name="):
                name = word[5:]
            elif word.lstrip("-").isdigit():
                ids.append(int(word))
        out.append((name, int(words[0]), ids))
    return out


def token_id(token: str) -> int:
    # `return_tokens_as_token_ids`: "token_id:123"
    return int(token.split(":", 1)[1])


def reset(url: str) -> bool:
    try:
        post(url + "/reset_prefix_cache", None, timeout=60)
        return True
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False


def fetch(url: str, cases: Path, score_ids: Path, out: Path, runs: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = model_of(url)
    score = [int(w) for w in score_ids.read_text().split()]
    results: dict[str, dict] = {}
    for run in range(runs):
        cleared = reset(url)
        for name, count, ids in cases_of(cases):
            if name in results and len(results[name]["runs"]) > run:
                continue
            body = {"model": model, "prompt": ids, "max_tokens": count, "temperature": 0.0, "logprobs": TOP,
                    "ignore_eos": True, "skip_special_tokens": False, "return_tokens_as_token_ids": True}
            reply = post(url + "/v1/completions", body)["choices"][0]
            lp = reply["logprobs"]
            steps = [{"token": token_id(t), "logprob": l, "top": {str(token_id(k)): v for k, v in top.items()}}
                     for t, l, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])]
            results.setdefault(name, {"prompt": ids, "count": count, "runs": []})["runs"].append(
                {"cleared": cleared, "tokens": [s["token"] for s in steps], "steps": steps})
            print(f"{name} run {run + 1}: {len(steps)} tokens", flush=True)
        cleared = reset(url)
        body = {"model": model, "prompt": score, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": TOP,
                "skip_special_tokens": False, "return_tokens_as_token_ids": True}
        reply = post(url + "/v1/completions", body)["choices"][0]
        positions = []
        for at, entry in enumerate(reply["prompt_logprobs"]):
            if entry is None:
                positions.append(None)
                continue
            own = str(score[at])
            top = {k: v["logprob"] for k, v in entry.items()}
            positions.append({"logprob": entry[own]["logprob"] if own in entry else None,
                              "rank": entry[own].get("rank") if own in entry else None, "top": top})
        results.setdefault("score", {"prompt": score, "runs": []})["runs"].append({"cleared": cleared, "positions": positions})
        print(f"score run {run + 1}: {len(positions)} positions", flush=True)
    for name, value in results.items():
        (out / f"{name}.json").write_text(json.dumps(value))


def zspark_output(path: Path) -> tuple[dict[str, list[int]], list[tuple[int, float]]]:
    replies: dict[str, list[int]] = {}
    losses: list[tuple[int, float]] = []
    name = None
    for line in path.read_text().splitlines():
        if line.startswith("== "):
            name = line[3:].strip()
        elif name is not None and "greedy:" in line:
            # a drafted case's `prompt S s; greedy: ID..` (a plain case prints no tokens: drafted is its bits)
            replies[name] = [int(w) for w in line.split("greedy:", 1)[1].split() if w.lstrip("-").isdigit()]
        else:
            words = line.split()
            if len(words) == 3 and words[0].isdigit() and words[2] in ("0", "1"):
                try:
                    losses.append((int(words[0]), float(words[1])))
                except ValueError:
                    pass
    return replies, losses


def gap(step: dict) -> float:
    values = sorted(step["top"].values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else math.inf


def compare(out: Path, zspark: Path) -> None:
    replies, losses = zspark_output(zspark)
    score = json.loads((out / "score.json").read_text())
    runs = score["runs"]
    vllm = [[-p["logprob"] if p and p["logprob"] is not None else None for p in r["positions"]] for r in runs]
    spread = [abs(a - b) for a, b in zip(vllm[0], vllm[-1]) if a is not None and b is not None]
    print(f"vLLM score runs: {len(runs)}; between the first and last, a token's |difference| mean "
          f"{sum(spread) / max(len(spread), 1):.5f}, largest {max(spread, default=0):.5f}, "
          f"{sum(1 for d in spread if d > 0)} of {len(spread)} differ")
    mine = {at: loss for at, loss in losses}
    pairs = [(mine[at], vllm[0][at]) for at in range(len(vllm[0])) if at in mine and vllm[0][at] is not None]
    if pairs:
        d = [a - b for a, b in pairs]
        print(f"score: {len(pairs)} tokens; nll zspark {sum(a for a, _ in pairs) / len(pairs):.5f}, vLLM "
              f"{sum(b for _, b in pairs) / len(pairs):.5f}; zspark - vLLM {sum(d) / len(d):+.5f}, a token's "
              f"|difference| mean {sum(abs(x) for x in d) / len(d):.5f}, 99th percentile "
              f"{sorted(abs(x) for x in d)[int(0.99 * (len(d) - 1))]:.5f}")
    tie = 2 * (sum(abs(a - b) for a, b in pairs) / len(pairs)) if pairs else 0.0
    for path in sorted(out.glob("*.json")):
        if path.name in ("score.json", "provenance.json"):
            continue
        case = json.loads(path.read_text())
        name = path.stem
        first = case["runs"][0]["tokens"]
        again = [r["tokens"] for r in case["runs"][1:]]
        agree_runs = [next((i for i, (a, b) in enumerate(zip(first, t)) if a != b), len(first)) for t in again]
        line = f"{name}: vLLM {len(first)} tokens, its runs agree on the first {agree_runs}"
        for label, tokens in sorted(replies.items()):
            if label != name:
                continue
            same = next((i for i, (a, b) in enumerate(zip(first, tokens)) if a != b), min(len(first), len(tokens)))
            line += f"; zspark {label}: {same} in common"
            if same < min(len(first), len(tokens)):
                step = case["runs"][0]["steps"][same]
                mine_top = step["top"].get(str(tokens[same]))
                line += (f" (at {same} vLLM's top-2 gap {gap(step):.4f}, zspark's token {tokens[same]} "
                         f"{'%.4f' % (step['logprob'] - mine_top) if mine_top is not None else 'not in its top 20'} "
                         f"under vLLM's; a tie under {tie:.4f}, twice the score's mean |zspark - vLLM|)")
        print(line)


def main() -> None:
    if len(sys.argv) >= 6 and sys.argv[1] == "fetch":
        fetch(sys.argv[2].rstrip("/"), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]),
              int(sys.argv[6]) if len(sys.argv) > 6 else 2)
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        compare(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
