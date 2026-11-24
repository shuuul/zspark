"""What a conversation holds in the session and in the shared pool (spec 054): the tokens a later request finds
held (`cached_tokens`) and the seconds to its first token, for the ways a request follows another, on one load.

Not a fixture and never runtime: a measurement, run by hand through `pool_states.sh`, which starts one
`zspark serve` under its defaults (`--parallel` 4), holds the machine alone and keeps the server's log. The
standard library only; the requests are `client_bench.py`'s (`send`, `body_of`, its corpus and salts) and the
hang-up is `turns_part.py`'s (`stream`).

    python3 tools/reference/pool_states.py --url http://127.0.0.1:18888 --log server.log --out states.json

A conversation is a document of `--sizes` tokens (about) under a salt no request before it had, with thinking
off and short replies that end by themselves, greedy. Its requests, in order, each alone on the server:

- `first`: the document and a question, met for the first time.
- `second`: that, its reply and a second question. `third`: the same again with a third.
- `again`: the third turn's request sent again as it was.
- `edited-last`: the third turn with its question changed.
- `hang-up`: the third turn with a question that asks for a long reply, streamed and closed by the client after
  `--cut` deltas of content. It is not a row of the table; `after-hang-up` is the turn that sends back what
  had arrived and goes on.
- `edited-earlier`: the second turn's question changed, the turns after it gone.

And a conversation of one message, under a salt of its own, after those: `one` (the first request again, not
a row to read), `one-again` (it sent again: a reply made again), `one-edited` (the question that ends the
message changed: a request that parts inside a message, far from any reply's end) and `one-edited-twice`
(changed again, from the same word on: it parts where the request before it parted). `--only single` runs
these and the `carry` phase alone.

The phases, on the one load: `session`, every conversation while one stream has been live at a time (the lone
stream is `session.zig`); then `carry`, the one opening of the pool a load has (below); then `pool`, every
conversation again under new salts, each request alone in the pool. `first` of the two phases is a first-met
prompt alone on either path.

`carry` (spec 054, WS-03) is what the hand-over costs the stream that is live when the pool opens. A long
conversation's first turn, then a turn that asks for a long reply: once alone to its end (`long-alone`, the
reply no opening touched), then sent again and, after `--open-after` of its deltas, a short request beside it
(`opener`), which opens the pool under it (`long-live`). Read: the live stream's longest wait between two
deltas from the opener's start (its stall), whether its reply is `long-alone`'s, the opener's first delta,
and then two turns of that conversation in the pool: its next (`next-after-the-pool-opened`) and its first
question edited (`edited-earlier-after`). A session holds one sequence, the last request's: a conversation
that was not the live one when the pool opened had left the session before, and is no row here.

Each conversation runs `--repeats` times in a phase. The table printed is the median and the range of the
seconds to the first delta, with `prompt_tokens`, `cached_tokens` and the tokens run again (their difference);
`--out` has every request, and the server's own line for it where `--log` is given. A second table sets the pool's
tokens run again beside the session's, a request each, and names the most the pool ran beyond it (the bound of
spec 054 is 15, a round's rows less one).
"""

import argparse, hashlib, http.client, json, os, re, statistics, sys, threading, time
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # no __pycache__ beside the scripts: a tree with one is dirty to `zspark version`
import client_bench  # noqa: E402
import turns_part  # noqa: E402

QUESTIONS = {
    "q1": "In one sentence: what is this text about?",
    "q2": "In one sentence: which part of it is the most technical?",
    "q3": "In one sentence: who would read it?",
    "q3e": "In one sentence: what does it leave out?",
    "q3h": "Write an essay of about six hundred words on what this text describes, in plain prose with no lists.",
    "q4": "In one sentence: what was the last thing you wrote about?",
    "q2e": "In one sentence: which part of it is the least technical?",
    "q1e": "In one sentence: what is this text for?",
    # it parts from q1e where q1e parted from q1, after "text": the mark kept there serves it
    "q1f": "In one sentence: what is this text made of?",
}
LINE = re.compile(r"prompt (\d+) tokens, (\d+) held, in ([\d.]+) s; reply (\d+) tokens in ([\d.]+) s, ([\d.]+) tok/s.*")


class Run:
    def __init__(self, args, spec):
        self.args, self.spec, self.rows = args, spec, []
        self.offset = os.path.getsize(args.log) if args.log else 0

    def lines(self):
        """The server's reply lines written since the last call."""

        if not self.args.log:
            return []
        time.sleep(0.15)
        with open(self.args.log, encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            text = f.read()
            self.offset = f.tell()
        return [m.group(0) for m in LINE.finditer(text)]

    def opened(self):
        """Whether the server's log says the shared pool is open."""

        if not self.args.log:
            return False
        with open(self.args.log, encoding="utf-8", errors="replace") as f:
            return "shared pool is open" in f.read()

    def note(self, phase, size, repeat, name, out, said=None):
        row = {"phase": phase, "size": size, "repeat": repeat, "name": name, "status": out.get("status"),
               "prompt_tokens": out.get("prompt_tokens"), "cached_tokens": out.get("cached_tokens"),
               "completion_tokens": out.get("completion_tokens"), "finish": out.get("finish"),
               "first_delta": out.get("first_delta"), "decode_tps": out.get("decode_tps"), "sha": out.get("sha"),
               "error": out.get("error"), "server": said}
        self.rows.append(row)
        print(f"{phase} {size} #{repeat} {name}: prompt {row['prompt_tokens']} cached {row['cached_tokens']} "
              f"first {row['first_delta'] and round(row['first_delta'], 3)} s, {row['completion_tokens']} tokens "
              f"{row['finish']}", flush=True)
        return row

    def ask(self, phase, size, repeat, name, messages, max_tokens=128):
        body = client_bench.body_of(self.args.model, messages, max_tokens, seed=1)
        out = client_bench.send(self.args.url, body)
        said = self.lines()
        self.note(phase, size, repeat, name, out, said[-1] if said else None)
        if out.get("status") != 200 or out.get("error"):
            raise RuntimeError(f"{name}: {out.get('status')} {out.get('error')}")
        return {"role": "assistant", "content": out["content"]}

    def document(self, phase, size, repeat):
        salt = client_bench.salt_of(self.args.seed, f"{phase}-{size}", repeat)
        chars = int(size * self.spec["chars_per_token"])
        return f"[{salt}] Here is a text.\n\n{self.spec['text'][:chars]}\n\n"

    def conversation(self, phase, size, repeat):
        u = lambda text: {"role": "user", "content": text}  # noqa: E731
        q = QUESTIONS
        m1 = u(self.document(phase, size, repeat) + q["q1"])
        a1 = self.ask(phase, size, repeat, "first", [m1])
        a2 = self.ask(phase, size, repeat, "second", [m1, a1, u(q["q2"])])
        head = [m1, a1, u(q["q2"]), a2]
        self.ask(phase, size, repeat, "third", head + [u(q["q3"])])
        self.ask(phase, size, repeat, "again", head + [u(q["q3"])])
        self.ask(phase, size, repeat, "edited-last", head + [u(q["q3e"])])
        body = client_bench.body_of(self.args.model, head + [u(q["q3h"])], 1024, seed=1)
        cut = turns_part.stream(self.args.url, "/v1/chat/completions", body, None, ("content", self.args.cut))
        # The server learns of the hang-up when it next writes. The next request waits for its line: one that
        # came while the reply was still live would be a second stream, and would open the pool.
        said = self.lines()
        for _ in range(60):
            if said or not self.args.log:
                break
            said = self.lines()
        self.note(phase, size, repeat, "hang-up", {"status": 200, "finish": "hung up"}, said[-1] if said else None)
        partial = {"role": "assistant", "content": cut["content"]}
        self.ask(phase, size, repeat, "after-hang-up", head + [u(q["q3h"]), partial, u(q["q4"])])
        self.ask(phase, size, repeat, "edited-earlier", [m1, a1, u(q["q2e"])])

    def single(self, phase, size, repeat):
        """A conversation of one message, under a salt of its own: where a request parts inside a message."""

        u = lambda text: {"role": "user", "content": text}  # noqa: E731
        doc = self.document(phase + "-one", size, repeat)
        self.ask(phase, size, repeat, "one", [u(doc + QUESTIONS["q1"])])
        self.ask(phase, size, repeat, "one-again", [u(doc + QUESTIONS["q1"])])
        self.ask(phase, size, repeat, "one-edited", [u(doc + QUESTIONS["q1e"])])
        self.ask(phase, size, repeat, "one-edited-twice", [u(doc + QUESTIONS["q1f"])])

    def watch(self, body, after=None, then=None):
        """One streamed chat completion with the time of every delta; `then` is called once, when `after`
        deltas have arrived. The content, the usage, and the longest wait between two deltas from that call on."""

        u = urlparse(self.args.url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=1800)
        start = time.perf_counter()
        conn.request("POST", "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        response = conn.getresponse()
        out = {"status": response.status, "first_delta": None, "stall": None, "called_at": None}
        if response.status != 200:
            out["error"] = response.read().decode("utf-8", "replace")[:2000]
            return out
        content, times, usage, finish = [], [], None, None
        while True:
            raw = response.readline()
            if not raw:
                break
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            usage = event.get("usage") or usage
            for choice in event.get("choices") or []:
                finish = choice.get("finish_reason") or finish
                text = (choice.get("delta") or {}).get("content") or ""
                if not text:
                    continue
                content.append(text)
                times.append(time.perf_counter() - start)
                if then is not None and len(times) == after:
                    out["called_at"] = times[-1]
                    then()
        conn.close()
        text = "".join(content)
        details = (usage or {}).get("prompt_tokens_details") or {}
        waits = [(b - a, a) for a, b in zip(times, times[1:]) if out["called_at"] is not None and a >= out["called_at"]]
        out.update({"content": text, "finish": finish, "first_delta": times[0] if times else None,
                    "prompt_tokens": (usage or {}).get("prompt_tokens"), "cached_tokens": details.get("cached_tokens", 0),
                    "completion_tokens": (usage or {}).get("completion_tokens"), "deltas": len(times),
                    "sha": hashlib.sha256(json.dumps(["", text, []]).encode()).hexdigest()[:12],
                    "stall": max(waits)[0] if waits else None, "end": times[-1] if times else None})
        return out

    def carry(self, size):
        u = lambda text: {"role": "user", "content": text}  # noqa: E731
        q = QUESTIONS
        m1 = u(self.document("carry", size, 0) + q["q1"])
        a1 = self.ask("carry", size, 0, "first", [m1])
        essay = [m1, a1, u(q["q3h"])]
        body = client_bench.body_of(self.args.model, essay, self.args.long, seed=1)
        alone = self.watch(body)
        said = self.lines()
        self.note("carry", size, 0, "long-alone", alone, said[-1] if said else None)
        found, threads = [None], []

        def opener():
            # beside the live stream: the second stream, which opens the pool
            def one():
                short = client_bench.body_of(self.args.model, [u("List five facts about the planet Mars.")], 200, seed=1)
                found[0] = client_bench.send(self.args.url, short)

            threads.append(threading.Thread(target=one))
            threads[0].start()

        live = self.watch(body, self.args.open_after, opener)
        for thread in threads:
            thread.join()
        said = self.lines()
        row = self.note("carry", size, 0, "long-live", live, next((line for line in said if f"reply {live.get('completion_tokens')} tokens" in line), None))
        row.update({"stall": live.get("stall"), "called_at": live.get("called_at"), "end": live.get("end"),
                    "same_reply": live.get("sha") == alone.get("sha"), "alone_end": alone.get("end")})
        if found[0] is not None:
            self.note("carry", 0, 0, "opener", found[0])
        for out in (alone, live):
            if out.get("status") != 200 or out.get("error"):
                raise RuntimeError(f"long: {out.get('status')} {out.get('error')}")
        a2 = {"role": "assistant", "content": live["content"]}
        self.ask("carry", size, 0, "next-after-the-pool-opened", essay + [a2, u(q["q4"])])
        self.ask("carry", size, 0, "edited-earlier-after", [m1, a1, u(q["q2e"])])


def opening(rows):
    """The pool's opening under a live stream, a line."""

    live = next((r for r in rows if r["name"] == "long-live"), None)
    first = next((r for r in rows if r["name"] == "opener"), None)
    if live is None or live.get("stall") is None:
        return ""
    return (f"\n\nthe opening: the live stream ({live['prompt_tokens']} prompt tokens, {live['completion_tokens']} of its reply) "
            f"waited {live['stall']:.3f} s at most between two deltas from the opener's start, and ended at {live['end']:.2f} s "
            f"where alone it ended at {live['alone_end']:.2f}; its reply is {'the same' if live['same_reply'] else 'NOT the same'} "
            f"as alone; the opener's first delta came after {first['first_delta']:.3f} s" if first else "")


def table(rows):
    out = ["| phase | size | request | prompt tokens | held | run again | first token, s (median, range) |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    keys = []
    for row in rows:
        key = (row["phase"], row["size"], row["name"])
        if key not in keys and row["first_delta"] is not None:
            keys.append(key)
    for key in keys:
        same = [r for r in rows if (r["phase"], r["size"], r["name"]) == key]
        firsts = [r["first_delta"] for r in same]
        pick = lambda field: "/".join(str(v) for v in sorted({r[field] for r in same}))  # noqa: E731
        again = "/".join(str(v) for v in sorted({r["prompt_tokens"] - r["cached_tokens"] for r in same}))
        out.append(f"| {key[0]} | {key[1]} | {key[2]} | {pick('prompt_tokens')} | {pick('cached_tokens')} | {again} | "
                   f"{statistics.median(firsts):.3f} ({min(firsts):.3f} to {max(firsts):.3f}, n={len(firsts)}) |")
    return "\n".join(out)


def bound(rows):
    """The pool's tokens run again beside the session's, a request of the two phases each (spec 054: at most 15
    more, the rows of a round less one). The widest of a request's repeats on each side; the last line says the
    most the pool ran beyond the session, and judges nothing."""

    again = {}
    for row in rows:
        if row["phase"] in ("session", "pool") and row["first_delta"] is not None and row["cached_tokens"] is not None:
            key = (row["size"], row["name"])
            again.setdefault(key, {}).setdefault(row["phase"], []).append(row["prompt_tokens"] - row["cached_tokens"])
    both = [(key, found) for key, found in again.items() if len(found) == 2]
    if not both:
        return ""
    out = ["", "| size | request | session, run again | pool, run again | pool less session |", "| --- | --- | --- | --- | --- |"]
    worst = None
    for (size, name), found in both:
        over = max(found["pool"]) - max(found["session"])
        if worst is None or over > worst[0]:
            worst = (over, size, name)
        out.append(f"| {size} | {name} | {max(found['session'])} | {max(found['pool'])} | {over:+d} |")
    out.append("")
    out.append(f"the pool ran at most {worst[0]:+d} tokens beyond the session ({worst[2]}, {worst[1]}); the bound is +15")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url")
    parser.add_argument("--model", default="m")
    parser.add_argument("--log", help="the server's log, for its own line a request")
    parser.add_argument("--out", required=True, help="every request, as JSON")
    parser.add_argument("--sizes", default="300,6000", help="document sizes in tokens, about")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cut", type=int, default=600, help="content deltas before the client hangs up")
    parser.add_argument("--long", type=int, default=700, help="the long reply's tokens at most, in `carry`")
    parser.add_argument("--open-after", type=int, default=100, help="the live reply's deltas before the opener is sent")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--only", choices=["single"], help="single: the one-message conversations alone")
    parser.add_argument("--table", action="store_true", help="print the table of an --out written before")
    args = parser.parse_args()
    if args.table:
        rows = json.load(open(args.out, encoding="utf-8"))["requests"]
        print(table(rows) + bound(rows) + opening(rows))
        return 0
    if not args.url:
        parser.error("--url is required to run")
    spec = client_bench.load_set(os.path.join(HERE, "client_bench.json"))
    sizes = [int(s) for s in args.sizes.split(",")]
    run = Run(args, spec)
    failed = None
    try:
        run.ask("warm", 0, 0, "warm", [{"role": "user", "content": spec["warm"]["user"]}], 32)
        for size in sizes if args.only is None else []:
            for repeat in range(args.repeats):
                run.conversation("session", size, repeat)
        for size in sizes:
            for repeat in range(args.repeats):
                run.single("session", size, repeat)
        if run.opened():
            raise RuntimeError("the pool opened during the session phase: its rows are not the session's")
        run.carry(max(sizes))
        if args.log and not run.opened():
            raise RuntimeError("the pool did not open: the pool phase's rows would be the session's")
        for size in sizes if args.only is None else []:
            for repeat in range(args.repeats):
                run.conversation("pool", size, repeat)
        for size in sizes:
            for repeat in range(args.repeats):
                run.single("pool", size, repeat)
    except Exception as err:  # what was taken is still written
        failed = str(err)
        print(f"FAILED: {failed}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "sizes": sizes, "repeats": args.repeats, "cut": args.cut, "seed": args.seed,
                   "failed": failed, "requests": run.rows}, f, indent=1)
    print(table(run.rows) + bound(run.rows) + opening(run.rows))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
