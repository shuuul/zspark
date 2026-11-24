"""What a long prompt costs the replies that stream beside it in the pool (spec 074, WS-04): the waits between
a stream's deltas before the prompt comes, while it is filled and after its first token, and the prompt's own
first token.

Not a fixture and never runtime: a measurement, run by hand through `fill_gaps.sh`, which starts one
`zspark serve --parallel 4`, holds the machine alone and keeps the server's log. The standard library only; the
requests are `client_bench.py`'s (`body_of`, its corpus and salts), the pool is opened as `spill_states.py`
opens it, and the server's lines are read as `pool_states.py` reads them.

    python3 tools/reference/fill_gaps.py --url http://127.0.0.1:18888 --log server.log --out gaps.json
    python3 tools/reference/fill_gaps.py --compare off=a/gaps.json on=b/gaps.json

One load: a warm request, two streams at once so that the pool is open before anything is timed, then
`--repeats` times: `--streams` replies of `--max-tokens` tokens started together (short prompts, thinking off,
greedy: the same three each repeat and each load), and `--after` seconds later a document of `--size` tokens
(about) under a salt no request before it had, with a question whose reply is short. The first repeat pays for
the graphs its rounds capture; the table reads each.

A stream's deltas are parted by the clock: `before` the prompt was sent, `during` (from there to the prompt's
first delta: the host's tokenizer, the wait for a seat's round and the fill) and `after`. A wait between two
deltas belongs to `during` when any of it lies there. Read of each phase: the waits' median, 99th percentile
and longest over the streams, and tokens a second. A round's tokens come as deltas a few microseconds apart, so
the median is of the waits of 5 ms and more, a round to the next, and the percentile and the longest are of
all. Tokens a second is a stream's `completion_tokens` shared out by the
characters that arrived in the phase (a delta does not say its tokens: a proxy, level across loads since the
replies are the same text). `during` is also read in thirds of its time, the longest wait and the tokens a
second of each: a round's cost that grows with the context filled shows there.

`--compare` sets loads side by side, a repeat a row, and says whether each request's reply is the same text in
every load (`sha`, as `client_bench.send` hashes it).
"""

import argparse, hashlib, http.client, json, math, os, statistics, sys, threading, time
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # no __pycache__ beside the scripts: a tree with one is dirty to `zspark version`
import client_bench  # noqa: E402
import pool_states  # noqa: E402
import spill_states  # noqa: E402

ESSAYS = [
    "Write an essay of about eight hundred words on the history of lighthouses, in plain prose with no lists.",
    "Write an essay of about eight hundred words on how bread is made, in plain prose with no lists.",
    "Write an essay of about eight hundred words on the rivers of Europe, in plain prose with no lists.",
    "Write an essay of about eight hundred words on the game of chess, in plain prose with no lists.",
]
ROUND = 0.005  # a wait this long is a round's, not two deltas of one round


def stream(url, body, epoch):
    """One streamed chat completion: when it was sent and when each delta came, in seconds from `epoch`, and
    each delta's characters."""

    u = urlparse(url)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
    out = {"sent": time.perf_counter() - epoch, "times": [], "chars": []}
    conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    response = conn.getresponse()
    out["status"] = response.status
    if response.status != 200:
        out["error"] = response.read().decode("utf-8", "replace")[:2000]
        conn.close()
        return out
    reasoning, content, usage, finish = [], [], None, None
    while True:
        raw = response.readline()
        if not raw:
            break
        now = time.perf_counter() - epoch
        line = raw.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            out["error"] = json.dumps(event["error"])[:2000]
            continue
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            r, c = delta.get("reasoning_content") or "", delta.get("content") or ""
            if not (r or c):
                continue
            reasoning.append(r)
            content.append(c)
            out["times"].append(now)
            out["chars"].append(len(r) + len(c))
    conn.close()
    reasoning, content = "".join(reasoning), "".join(content)
    details = (usage or {}).get("prompt_tokens_details") or {}
    out.update({"content": content, "finish": finish, "deltas": len(out["times"]),
                "first_delta": out["times"][0] - out["sent"] if out["times"] else None,
                "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": details.get("cached_tokens", 0),
                "completion_tokens": (usage or {}).get("completion_tokens"),
                "sha": hashlib.sha256(json.dumps([reasoning, content, []]).encode()).hexdigest()[:12]})
    return out


def one(run, repeat):
    """The streams, then the document beside them: every request of the repeat, the document's last."""

    args = run.args
    user = spill_states.user
    epoch = time.perf_counter()
    found = [None] * args.streams
    threads = []
    for i in range(args.streams):
        body = client_bench.body_of(args.model, [user(ESSAYS[i % len(ESSAYS)])], args.max_tokens, seed=1)

        def go(i=i, body=body):
            found[i] = stream(args.url, body, epoch)

        threads.append(threading.Thread(target=go))
    for thread in threads:
        thread.start()
    time.sleep(args.after)
    document = run.document("fill", args.size, repeat) + pool_states.QUESTIONS["q1"]
    long = stream(args.url, client_bench.body_of(args.model, [user(document)], 32, seed=1), epoch)
    for thread in threads:
        thread.join()
    said = run.lines()
    rows = []
    for name, out in [(f"stream-{i}", out) for i, out in enumerate(found)] + [("document", long)]:
        if out.get("status") != 200 or out.get("error") or not out["times"]:
            raise RuntimeError(f"{name}: {out.get('status')} {out.get('error')}")
        out.update({"repeat": repeat, "name": name})
        out.pop("content")
        rows.append(out)
        print(f"#{repeat} {name}: prompt {out['prompt_tokens']} cached {out['cached_tokens']}, first delta "
              f"{out['first_delta']:.3f} s, {out['completion_tokens']} tokens {out['finish']}, {out['sha']}", flush=True)
    return rows, said


def centile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, max(0, math.ceil(q * len(values)) - 1))]


def parted(row, sent, first):
    """A stream's waits and tokens a second in each phase, and the fill's thirds."""

    times, chars = row["times"], row["chars"]
    each = row["completion_tokens"] / max(1, sum(chars))
    last = max(times[-1], first)
    spans = {"before": (times[0], sent), "during": (sent, first), "after": (first, last)}
    third = (first - sent) / 3
    for k in range(3):
        spans[f"third-{k + 1}"] = (sent + k * third, sent + (k + 1) * third)
    out = {}
    for name, (lo, hi) in spans.items():
        waits = [b - a for a, b in zip(times, times[1:]) if b > lo and a < hi]
        if name == "before":
            waits = [b - a for a, b in zip(times, times[1:]) if b <= hi]
        if name == "after":
            waits = [b - a for a, b in zip(times, times[1:]) if a >= lo]
        tokens = each * sum(c for t, c in zip(times, chars) if lo < t <= hi)
        out[name] = {"waits": waits, "tps": tokens / (hi - lo) if hi > lo and tokens else None}
    return out


def read(requests):
    """A repeat a row: each phase's waits over the streams and their tokens a second, and the document's first
    delta."""

    rows = []
    for repeat in sorted({r["repeat"] for r in requests}):
        mine = [r for r in requests if r["repeat"] == repeat]
        long = next(r for r in mine if r["name"] == "document")
        sent, first = long["sent"], long["times"][0]
        parts = [parted(r, sent, first) for r in mine if r["name"] != "document"]
        row = {"repeat": repeat, "first": long["first_delta"], "prompt_tokens": long["prompt_tokens"],
               "cached_tokens": long["cached_tokens"]}
        for name in parts[0]:
            waits = [w for p in parts for w in p[name]["waits"]]
            rates = [p[name]["tps"] for p in parts if p[name]["tps"]]
            rounds = [w for w in waits if w >= ROUND]
            row[name] = {"n": len(waits), "rounds": len(rounds), "median": statistics.median(rounds) if rounds else None,
                         "p99": centile(waits, 0.99) if waits else None, "max": max(waits) if waits else None,
                         "tps": rates}
        rows.append(row)
    return rows


def ms(value):
    return "" if value is None else f"{value * 1000:.0f}"


def cell(part):
    rates = ", ".join(f"{r:.1f}" for r in part["tps"])
    return f"{ms(part['median'])} / {ms(part['p99'])} / {ms(part['max'])}; {rates}"


def table(label, requests):
    out = [f"| {label}: repeat | before: waits ms a round's median / p99 / max; tok/s a stream | during | after | "
           f"the fill's thirds: longest wait ms; tok/s over the streams | the document: first token s, held of prompt |",
           "| --- | --- | --- | --- | --- | --- |"]
    for row in read(requests):
        thirds = ", ".join(f"{ms(row[f'third-{k}']['max'])}; {sum(row[f'third-{k}']['tps']):.1f}" for k in (1, 2, 3))
        out.append(f"| {row['repeat']} | {cell(row['before'])} | {cell(row['during'])} | {cell(row['after'])} | "
                   f"{thirds} | {row['first']:.2f}, {row['cached_tokens']} of {row['prompt_tokens']} |")
    return "\n".join(out)


def compare(pairs):
    loads = []
    for pair in pairs:
        label, path = pair.split("=", 1)
        loads.append((label, json.load(open(path, encoding="utf-8"))["requests"]))
    out = [table(label, requests) for label, requests in loads]
    keys = []
    for _, requests in loads:
        for r in requests:
            if (r["repeat"], r["name"]) not in keys:
                keys.append((r["repeat"], r["name"]))
    differ = []
    lines = ["| request | " + " | ".join(label for label, _ in loads) + " |", "| --- |" + " --- |" * len(loads)]
    for key in keys:
        hashes = [next((r["sha"] for r in requests if (r["repeat"], r["name"]) == key), "") for _, requests in loads]
        if len(set(hashes)) > 1:
            differ.append(key)
        lines.append(f"| #{key[0]} {key[1]} | " + " | ".join(hashes) + " |")
    out.append("\n".join(lines))
    out.append(f"requests whose replies differ between loads: {len(differ)} of {len(keys)}")
    return "\n\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url")
    parser.add_argument("--model", default="m")
    parser.add_argument("--log", help="the server's log, for its own line a request")
    parser.add_argument("--out", help="every request with each delta's time, as JSON")
    parser.add_argument("--size", type=int, default=25000, help="the document in tokens, about")
    parser.add_argument("--streams", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512, help="a stream's reply")
    parser.add_argument("--after", type=float, default=2.0, help="seconds from the streams' start to the document")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compare", nargs="+", metavar="LABEL=JSON")
    args = parser.parse_args()
    if args.compare:
        print(compare(args.compare))
        return 0
    if not (args.url and args.out):
        parser.error("--url and --out are required to run")
    spec = client_bench.load_set(os.path.join(HERE, "client_bench.json"))
    if len(spec["text"]) < args.size * spec["chars_per_token"]:
        parser.error(f"the corpus is shorter than {args.size} tokens")
    run = pool_states.Run(args, spec)
    requests, said, failed = [], [], None
    try:
        run.ask("warm", 0, 0, "warm", [spill_states.user(spec["warm"]["user"])], 32)
        spill_states.open_pool(run)
        for repeat in range(args.repeats):
            rows, lines = one(run, repeat)
            requests += rows
            said += lines
    except Exception as err:  # what was taken is still written
        failed = str(err)
        print(f"FAILED: {failed}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "failed": failed, "requests": requests, "server": said}, f, indent=1)
    if requests:
        print(table("this load", requests))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
