"""What a client sees of `zspark serve` and of TensorFold's server: time to first token and tokens a second, over
the fixed requests of `client_bench.json` (spec 024).

Not a fixture and never runtime: a measurement, run by hand through `client_bench.sh`, which starts one server,
holds the machine alone and keeps the server's log. The standard library only, so it runs wherever python3 does.

    python3 tools/reference/client_bench.py run --url http://HOST:PORT --seed 1 --out result.json
    python3 tools/reference/client_bench.py streams --url http://HOST:PORT --counts 1,2,4 --out streams.json
    python3 tools/reference/client_bench.py sparkdash --spark spark-b --model NAME --out sparkdash.json
    python3 tools/reference/client_bench.py table zspark=DIR,DIR,DIR tensorfold=DIR,DIR,DIR
    python3 tools/reference/client_bench.py fake --port 8012        # a server that decodes nothing, to try the above

`run` sends every request once, in order, one at a time, each on a connection of its own, streamed, and writes
what it timed. A request's clock starts before the connection is opened. It records the seconds to the response's
head, to the first SSE event, to the first delta that carries text (reasoning, content or a tool call: the first
token as a client sees it), to the first content, to the last delta and to the stream's end; the usage the server
reported; and a hash of the reply's text, by which two servers' replies are compared.

The requests, all with the same bytes for the same `--seed`:

- `warm`: one short reply, the server's first, reported apart.
- `decode`: short and long replies over prose and code, a tool call, a thinking reply, greedy; two long replies
  sampled. Each `decode_repeats` times; a repeat finds its prompt already seen, and these prompts are short.
- `prompt`: for each size, a document of about that many tokens under a salt the server has not seen
  (`cold`), then the same conversation with its reply and a second question (`second`: the prefix has been
  seen), then that with the last question changed (`edited`). Each `prompt_repeats` times, a new salt each.
  The salt opens the first message, so nothing of an earlier request is a prefix of a cold one but the
  template's opening tokens. The document is the head of `corpus`, cut by characters: `client_bench_corpus.txt`,
  three specs as they stood at c7cd4de, kept as a file of its own so that a recorded hash's prompt can be made
  again whatever becomes of the specs.

- `turns` (spec 030): conversations whose replies are some hundreds of tokens, over a document under a new
  salt: the first request (`first`), a second turn (`second`, a long reply again), a third (`third`), the
  third's question changed (`third-edited`: the last message edited, after a long reply), and the second's
  changed with the third turn gone (`second-edited`: an earlier message edited). Each reply goes back as the
  server sent it, with its reasoning as `reasoning_content` where the conversation thinks. What a server holds
  of the prompt is `cached_tokens`; the tokens it runs again are the prompt's less those. Each
  `turns_repeats` times. They come last, so the requests before them are the bytes they were under spec 024.

Tokens a second are the client's: a prompt's are the tokens the server did not have (`prompt_tokens` less
`cached_tokens`) over the seconds to the first token, so they include the request's parsing, the template and
the first decode step; a reply's are its tokens less one over the seconds from the first delta to the last, as
sparkDash's decode benchmark counts them. A reply the server holds to its end (content while tools are offered)
has no such figure, and `whole` (its tokens over the whole request) stands for it.

`sparkdash` starts sparkDash's two benchmarks (the owner's dashboard on the router) against the server on the
Spark's LAN address, through its API, and keeps what they return: its decode benchmark at concurrency 1 for each
of its prompt types, and its prefill benchmark at its sizes. Nothing of sparkDash is changed; a benchmark it runs
enters its own history, as one started from its page does.

`table` reads the directories `client_bench.sh` wrote (one a load) and prints Markdown: medians and ranges over
the loads, the two servers side by side, with what each server's own log said beside the client's figures.
"""

import argparse, hashlib, http.client, http.server, json, os, re, statistics, sys, threading, time
from urllib.parse import urlparse

from local_env import local_setting

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))


def load_set(path):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    spec["text"] = "".join(open(os.path.join(REPO, p), encoding="utf-8").read() for p in spec["corpus"])
    return spec


def salt_of(seed, name, repeat):
    return hashlib.sha256(f"{seed}:{name}:{repeat}".encode()).hexdigest()[:16]


def body_of(model, messages, max_tokens, *, thinking=False, tools=None, temperature=0.0, top_p=1.0, top_k=None,
            seed=None, tool_choice=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    if top_k is not None:
        body["top_k"] = top_k
    if seed is not None:
        body["seed"] = seed
    if tools:
        body["tools"] = tools
        # only a set that asks for it sends the field: the other sets' requests stay the bytes they were
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    return body


def send(url, body, timeout=1800.0):
    """One streamed chat completion: the times, the usage and the text."""

    u = urlparse(url)
    payload = json.dumps(body).encode()
    out = {"head": None, "first_event": None, "first_delta": None, "first_content": None, "last_delta": None,
           "deltas": 0}
    reasoning, content, calls, usage, finish, extra = [], [], {}, None, None, None
    start = time.perf_counter()
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=timeout)
    try:
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        response = conn.getresponse()
        out["head"] = time.perf_counter() - start
        out["status"] = response.status
        if response.status != 200:
            out["error"] = response.read().decode("utf-8", "replace")[:2000]
            out["end"] = time.perf_counter() - start
            return out
        while True:
            line = response.readline()
            if not line:
                break
            now = time.perf_counter() - start
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if out["first_event"] is None:
                out["first_event"] = now
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                out["error"] = json.dumps(event["error"])[:2000]
                continue
            if event.get("usage"):
                usage = event["usage"]
            if "tensorfold" in event:
                extra = event["tensorfold"]
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                r, c, t = delta.get("reasoning_content") or "", delta.get("content") or "", delta.get("tool_calls") or []
                if not (r or c or t):
                    continue
                out["deltas"] += 1
                if out["first_delta"] is None:
                    out["first_delta"] = now
                if c and out["first_content"] is None:
                    out["first_content"] = now
                out["last_delta"] = now
                reasoning.append(r)
                content.append(c)
                for call in t:
                    slot = calls.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    slot["name"] += function.get("name") or ""
                    slot["arguments"] += function.get("arguments") or ""
        out["end"] = time.perf_counter() - start
    finally:
        conn.close()
    reasoning, content = "".join(reasoning), "".join(content)
    called = [calls[i] for i in sorted(calls)]
    details = (usage or {}).get("prompt_tokens_details") or {}
    out.update({
        "finish": finish, "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "cached_tokens": details.get("cached_tokens", 0),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "reasoning_tokens": ((usage or {}).get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
        "reasoning": reasoning, "content": content, "calls": called,
        "sha": hashlib.sha256(json.dumps([reasoning, content, called]).encode()).hexdigest()[:12],
    })
    if extra is not None:
        out["server_stats"] = extra
    tokens, first, last = out["completion_tokens"], out["first_delta"], out["last_delta"]
    if tokens and first is not None:
        fresh = (out["prompt_tokens"] or 0) - (out["cached_tokens"] or 0)
        out["prompt_tps"] = fresh / first
        out["whole_tps"] = tokens / out["end"]
        if tokens > 1 and out["deltas"] > 2 and last > first:
            out["decode_tps"] = (tokens - 1) / (last - first)
    return out


def requests_of(spec, model, seed):
    """Every request of a run, in order, as (name, kind, repeat, a function from the earlier results to a body)."""

    plan = []
    warm = spec["warm"]
    plan.append(("warm", "warm", 0, lambda done: body_of(model, [{"role": "user", "content": warm["user"]}],
                                                         warm["max_tokens"])))
    for repeat in range(spec["decode_repeats"]):
        for case in spec["decode"]:
            def build(done, case=case):
                return body_of(model, [{"role": "user", "content": case["user"]}], case["max_tokens"],
                               thinking=case.get("thinking", False), tools=spec["tools"] if case.get("tools") else None,
                               temperature=case.get("temperature", 0.0), top_p=case.get("top_p", 1.0),
                               top_k=case.get("top_k"), seed=case.get("seed"), tool_choice=case.get("tool_choice"))
            plan.append((case["name"], "decode", repeat, build))
    p = spec["prompt"]
    for size in spec["prompt_sizes"]:
        for repeat in range(spec["prompt_repeats"]):
            name = f"prompt-{size}"
            room = max(0, int(size * spec["chars_per_token"]) - len(p["lead"]) - len(p["question"]) - 24)
            first = {"role": "user", "content": f"[{salt_of(seed, name, repeat)}]\n" + p["lead"] + spec["text"][:room]
                     + p["question"]}
            if room > len(spec["text"]):
                raise SystemExit(f"{name}: the corpus has {len(spec['text'])} characters and {room} are asked")

            def cold(done, first=first):
                return body_of(model, [first], p["max_tokens"])

            def later(done, first=first, name=name, repeat=repeat, question="second"):
                reply = next(r for r in done if r["name"] == name + "-cold" and r["repeat"] == repeat)
                return body_of(model, [first, {"role": "assistant", "content": reply.get("content") or "OK."},
                                       {"role": "user", "content": p[question]}], p["max_tokens"])
            plan.append((name + "-cold", "prompt", repeat, cold))
            plan.append((name + "-second", "prompt", repeat, later))
            plan.append((name + "-edited", "prompt", repeat, lambda done, later=later: later(done, question="edited")))
    t = spec.get("turn") or {}
    for case in spec.get("turns") or []:
        for repeat in range(spec["turns_repeats"]):
            name = case["name"]
            room = max(0, int(case["size"] * spec["chars_per_token"]) - len(p["lead"]) - len(t["question"]) - 24)
            if room > len(spec["text"]):
                raise SystemExit(f"{name}: the corpus has {len(spec['text'])} characters and {room} are asked")
            first = {"role": "user", "content": f"[{salt_of(seed, name, repeat)}]\n" + p["lead"] + spec["text"][:room]
                     + t["question"]}

            def turn(done, steps, question=None, max_tokens=None, case=case, first=first, name=name, repeat=repeat):
                """The conversation after `steps` (the requests whose replies it holds), then `question`."""

                messages = [first]
                for step, asked in steps:
                    reply = next(r for r in done if r["name"] == f"{name}-{step}" and r["repeat"] == repeat)
                    said = {"role": "assistant", "content": reply.get("content") or "OK."}
                    if case["thinking"] and reply.get("reasoning"):
                        said["reasoning_content"] = reply["reasoning"]
                    messages += [said, {"role": "user", "content": t[asked]}]
                return body_of(model, messages, max_tokens or t["max_tokens"], thinking=case["thinking"])

            long = case["reply_tokens"]
            for step, steps, tokens in (
                    ("first", [], long),
                    ("second", [("first", "second")], long),
                    ("third", [("first", "second"), ("second", "third")], long if case["thinking"] else None),
                    ("third-edited", [("first", "second"), ("second", "third_edited")], None),
                    ("second-edited", [("first", "second_edited")], None)):
                plan.append((f"{name}-{step}", "turns", repeat,
                             lambda done, turn=turn, steps=steps, tokens=tokens: turn(done, steps, max_tokens=tokens)))
    return plan


def run(args):
    spec = load_set(args.set)
    model = args.model
    if not model:
        u = urlparse(args.url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        conn.request("GET", "/v1/models")
        model = json.loads(conn.getresponse().read())["data"][0]["id"]
        conn.close()
    only = set(args.only.split(",")) if args.only else None
    done = []
    for name, kind, repeat, build in requests_of(spec, model, args.seed):
        if only and name not in only and kind not in only:
            continue
        body = build(done)
        result = send(args.url, body)
        result.update({"name": name, "kind": kind, "repeat": repeat,
                       "request_sha": hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:12]})
        done.append(result)
        shown = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()
                 if k in ("status", "first_delta", "end", "prompt_tokens", "cached_tokens", "completion_tokens",
                          "prompt_tps", "decode_tps", "finish", "sha", "error")}
        print(f"{name}#{repeat} {shown}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "model": model, "seed": args.seed, "set": os.path.basename(args.set),
                   "set_sha": hashlib.sha256(open(args.set, "rb").read()).hexdigest()[:12],
                   "corpus_sha": hashlib.sha256(spec["text"].encode()).hexdigest()[:12], "requests": done}, f, indent=1)
    failed = [r["name"] for r in done if r.get("error") or r.get("status") != 200]
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


# What each server's log says of a finished reply, in the order the replies finished.
ZSPARK_DONE = re.compile(r"prompt (\d+) tokens, (\d+) held, in ([\d.]+) s; reply (\d+) tokens in ([\d.]+) s, "
                         r"([\d.]+) tok/s; (\d+) of (\d+) drafts kept")
TENSORFOLD_DONE = re.compile(r"done req-\S+ prompt=(\d+) cached=(\d+) .*?tokens=(\d+) sha=(\S+) finish=\S+ "
                             r"tok/s=([\d.]+) ttft=([-\d.]+)s prefill=([-\d.]+)s rounds=(\d+) accepted=(\d+)/(\d+)")


def log_lines(path):
    """A server's own account of each reply: prompt and held tokens, prompt seconds, reply tokens, tok/s, drafts."""

    found = []
    if not os.path.exists(path):
        return found
    for line in open(path, encoding="utf-8", errors="replace"):
        m = ZSPARK_DONE.search(line)
        if m:
            found.append({"prompt": int(m[1]), "held": int(m[2]), "prompt_s": float(m[3]), "tokens": int(m[4]),
                          "reply_s": float(m[5]), "tps": float(m[6]), "kept": int(m[7]), "drafted": int(m[8])})
            continue
        m = TENSORFOLD_DONE.search(line)
        if m:
            found.append({"prompt": int(m[1]), "held": int(m[2]), "tokens": int(m[3]), "tps": float(m[5]),
                          "ttft_s": float(m[6]), "prompt_s": float(m[7]), "rounds": int(m[8]), "kept": int(m[9]),
                          "drafted": int(m[10])})
    return found


def load_dir(path):
    """One load's directory: the harness's requests with the server's log lines beside them, and `meta.json`."""

    with open(os.path.join(path, "harness.json"), encoding="utf-8") as f:
        result = json.load(f)
    lines = log_lines(os.path.join(path, "harness.log"))
    requests = [r for r in result["requests"] if r.get("status") == 200 and not r.get("error")]
    if len(lines) == len(requests):
        for r, line in zip(requests, lines):
            if line["prompt"] == r["prompt_tokens"]:
                r["log"] = line
    meta = {}
    if os.path.exists(os.path.join(path, "meta.json")):
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    return {"requests": result["requests"], "meta": meta, "path": path, "seed": result["seed"]}


def spread(values, digits=2):
    values = [v for v in values if v is not None]
    if not values:
        return ""
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{statistics.median(values):.{digits}f} ({min(values):.{digits}f} to {max(values):.{digits}f})"


def ratio(ours, theirs, higher_is_better):
    ours, theirs = [v for v in ours if v is not None], [v for v in theirs if v is not None]
    if not ours or not theirs:
        return ""
    a, b = statistics.median(ours), statistics.median(theirs)
    if a <= 0 or b <= 0:
        return ""
    gain = a / b if higher_is_better else b / a
    return f"{gain:.2f}x"


def table(args):
    servers = {}
    for item in args.loads:
        label, _, dirs = item.partition("=")
        servers[label] = [load_dir(d) for d in dirs.split(",") if d]
    labels = list(servers)
    if len(labels) != 2:
        raise SystemExit("table takes two servers: LABEL=DIR,DIR LABEL=DIR,DIR (the first is compared to the second)")
    ours, theirs = labels

    def rows(label, name, pick, first_only=False, where=lambda r: True):
        out = []
        for load in servers[label]:
            for r in load["requests"]:
                if r["name"] == name and r.get("status") == 200 and where(r) and (not first_only or r["repeat"] == 0):
                    out.append(pick(r))
        return out

    names = []
    for load in servers[ours]:
        for r in load["requests"]:
            if r["name"] not in names:
                names.append(r["name"])
    prompt_names = [n for n in names if n.startswith(("prompt-", "turns-"))]
    decode_names = [n for n in names if not n.startswith(("prompt-", "turns-"))]

    def get(key, sub=None):
        return lambda r: (r.get(key) if sub is None else (r.get(key) or {}).get(sub))

    print(f"### Loads\n\n| | {ours} | {theirs} |\n| --- | --- | --- |")
    for key, title, digits in (("ready_s", "Seconds from the process's start to `/v1/models`", 1),
                               ("load_s", "The server's own load seconds", 1),
                               ("device_mib_loaded", "Device memory after the load, MiB", 0),
                               ("device_mib_end", "Device memory after the requests, MiB", 0),
                               ("host_rss_mib_loaded", "Host memory (RSS) after the load, MiB", 0),
                               ("host_rss_mib_end", "Host memory (RSS) after the requests, MiB", 0)):
        cells = [spread([load["meta"].get(key) for load in servers[label]], digits) for label in labels]
        print(f"| {title} | {cells[0]} | {cells[1]} |")

    print(f"\n### Time to first token, seconds (first delta of text; median and range over loads and repeats)\n\n"
          f"| Request | prompt tokens, held ({ours} / {theirs}) | {ours} | {theirs} | {ours} is | "
          f"{ours} log: prompt s | {theirs} log: prompt s |\n| --- | --- | --- | --- | --- | --- | --- |")
    for name in prompt_names:
        a, b = rows(ours, name, get("first_delta")), rows(theirs, name, get("first_delta"))
        tokens = " / ".join(f"{spread(rows(label, name, get('prompt_tokens')), 0).split(' ')[0]}, "
                            f"{spread(rows(label, name, get('cached_tokens')), 0).split(' ')[0]}" for label in labels)
        print(f"| {name} | {tokens} | {spread(a, 3)} | {spread(b, 3)} | {ratio(a, b, False)} | "
              f"{spread(rows(ours, name, get('log', 'prompt_s')))} | {spread(rows(theirs, name, get('log', 'prompt_s')))} |")

    print(f"\n### Prompt tokens a second (tokens not held over the seconds to the first token)\n\n"
          f"| Request | {ours} | {theirs} | {ours} is |\n| --- | --- | --- | --- |")
    for name in prompt_names:
        if not name.endswith("-cold"):
            continue
        a, b = rows(ours, name, get("prompt_tps")), rows(theirs, name, get("prompt_tps"))
        print(f"| {name} | {spread(a, 0)} | {spread(b, 0)} | {ratio(a, b, True)} |")

    def same(name):
        """Whether both servers gave the same text for a request, in every load that has both."""

        a = {(load["seed"], r["repeat"]): r["sha"] for load in servers[ours] for r in load["requests"]
             if r["name"] == name and "sha" in r}
        b = {(load["seed"], r["repeat"]): r["sha"] for load in servers[theirs] for r in load["requests"]
             if r["name"] == name and "sha" in r}
        both = [k for k in a if k in b]
        return sum(a[k] == b[k] for k in both), len(both)

    print(f"\n### Replies: tokens a second after the first token, (tokens - 1) / (last delta - first delta)\n\n"
          f"| Reply | tokens ({ours} / {theirs}) | same text | {ours} | {theirs} | {ours} is | "
          f"{ours} log tok/s; kept of drafted | {theirs} log tok/s; kept of drafted, rounds | "
          f"first token s ({ours} / {theirs}) |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    totals = {label: {"tokens": 0, "seconds": 0.0} for label in labels}
    for name in decode_names:
        a, b = rows(ours, name, get("decode_tps")), rows(theirs, name, get("decode_tps"))
        key = "decode_tps"
        if not [v for v in a if v is not None] or not [v for v in b if v is not None]:
            a, b, key = rows(ours, name, get("whole_tps")), rows(theirs, name, get("whole_tps")), "whole_tps"
        tokens = " / ".join(spread(rows(label, name, get("completion_tokens")), 0).split(" ")[0] for label in labels)
        agreed = "%d of %d" % same(name)

        def drafts(label, name=name):
            logs = [r for r in rows(label, name, get("log")) if r]
            if not logs:
                return ""
            text = f"{spread([l['tps'] for l in logs], 1)}; {sum(l['kept'] for l in logs)} of {sum(l['drafted'] for l in logs)}"
            if "rounds" in logs[0]:
                text += f", {sum(l['rounds'] for l in logs)} rounds"
            return text
        firsts = " / ".join(spread(rows(label, name, get("first_delta")), 3).split(" ")[0] for label in labels)
        note = " (whole request)" if key == "whole_tps" else ""
        print(f"| {name}{note} | {tokens} | {agreed} | {spread(a, 1)} | {spread(b, 1)} | {ratio(a, b, True)} | "
              f"{drafts(ours)} | {drafts(theirs)} | {firsts} |")
        if name != "warm":
            for label in labels:
                for load in servers[label]:
                    for r in load["requests"]:
                        if r["name"] == name and r.get("decode_tps") and r["last_delta"] > r["first_delta"]:
                            totals[label]["tokens"] += r["completion_tokens"] - 1
                            totals[label]["seconds"] += r["last_delta"] - r["first_delta"]
    over = {label: (t["tokens"] / t["seconds"] if t["seconds"] else 0.0) for label, t in totals.items()}
    if over[ours] and over[theirs]:
        print(f"\nOver every streamed reply of the set (tokens after the first over their seconds, `warm` apart): "
              f"{ours} {over[ours]:.1f} tok/s ({totals[ours]['tokens']} tokens), {theirs} {over[theirs]:.1f} tok/s "
              f"({totals[theirs]['tokens']} tokens), {ours} {over[ours] / over[theirs]:.2f}x.")

    # A round is a window of one token and its drafts. TensorFold logs its rounds; zspark's are the tokens that
    # were not kept drafts (every round gives one token of its own), the first token being the prompt's.
    print(f"\n### Rounds, from the servers' logs (every load and repeat summed)\n\n"
          f"| Reply | same text | {ours} rounds | drafts a round | kept a round | ms a round | {theirs} rounds | "
          f"drafts a round | kept a round | ms a round |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name in decode_names:
        cells = []
        for label in labels:
            logs = [r for r in rows(label, name, get("log")) if r and r["tokens"] > 1 and r["tps"] > 0]
            rounds = sum(l["rounds"] if "rounds" in l else l["tokens"] - 1 - l["kept"] for l in logs)
            seconds = sum(l["reply_s"] if "reply_s" in l else (l["tokens"] - 1) / l["tps"] for l in logs)
            if rounds <= 0:
                cells += ["", "", "", ""]
                continue
            cells += [str(rounds), f"{sum(l['drafted'] for l in logs) / rounds:.2f}",
                      f"{sum(l['kept'] for l in logs) / rounds:.2f}", f"{seconds / rounds * 1e3:.1f}"]
        print(f"| {name} | {'%d of %d' % same(name)} | " + " | ".join(cells) + " |")

    def dash(label):
        found = []
        for load in servers[label]:
            path = os.path.join(load["path"], "sparkdash.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    found.append(json.load(f))
        return found
    if dash(ours) and dash(theirs):
        kinds, sizes = [], []
        for label in labels:
            for d in dash(label):
                for job in d["decode"]:
                    kind = (job.get("config") or {}).get("promptType")
                    if kind and kind not in kinds:
                        kinds.append(kind)
                for row in (d.get("prefill") or {}).get("results") or []:
                    if row["targetTokens"] not in sizes:
                        sizes.append(row["targetTokens"])

        def waves(label, kind, key):
            return [job["results"][0].get(key) for d in dash(label) for job in d["decode"]
                    if (job.get("config") or {}).get("promptType") == kind and job.get("results")
                    and not job["results"][0].get("error")]

        def fills(label, size, key):
            return [row.get(key) for d in dash(label) for row in (d.get("prefill") or {}).get("results") or []
                    if row["targetTokens"] == size and not row.get("error")]
        print(f"\n### sparkDash's decode benchmark, concurrency 1 (its median tok/s; a run a load)\n\n"
              f"| Prompt | tokens ({ours} / {theirs}) | {ours} tok/s | {theirs} tok/s | {ours} is | "
              f"first token ms ({ours} / {theirs}) |\n| --- | --- | --- | --- | --- | --- |")
        for kind in kinds:
            a, b = waves(ours, kind, "medianDecodeTps"), waves(theirs, kind, "medianDecodeTps")
            tokens = " / ".join(spread(waves(label, kind, "totalCompletionTokens"), 0).split(" ")[0] for label in labels)
            firsts = " / ".join(spread(waves(label, kind, "medianTtftMs"), 0) for label in labels)
            print(f"| {kind} | {tokens} | {spread(a, 1)} | {spread(b, 1)} | {ratio(a, b, True)} | {firsts} |")
        print(f"\n### sparkDash's prefill benchmark (a fresh salt a size, 8 tokens generated)\n\n"
              f"| Target | prompt tokens ({ours} / {theirs}) | {ours} first token s | {theirs} first token s | "
              f"{ours} is | tok/s ({ours} / {theirs}) |\n| --- | --- | --- | --- | --- | --- |")
        for size in sorted(sizes):
            a = [v / 1000 for v in fills(ours, size, "ttftMs")]
            b = [v / 1000 for v in fills(theirs, size, "ttftMs")]
            tokens = " / ".join(spread(fills(label, size, "promptTokens"), 0).split(" ")[0] for label in labels)
            rates = " / ".join(spread(fills(label, size, "prefillTps"), 0) for label in labels)
            print(f"| {size} | {tokens} | {spread(a, 2)} | {spread(b, 2)} | {ratio(a, b, False)} | {rates} |")
    return 0


def dash_call(base, method, path, body=None):
    u = urlparse(base)
    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=60)
    try:
        payload = None if body is None else json.dumps(body).encode()
        conn.request(method, path, payload, {"Content-Type": "application/json"} if payload else {})
        response = conn.getresponse()
        text = response.read().decode("utf-8", "replace")
        try:
            return response.status, json.loads(text)
        except ValueError:
            return response.status, {"error": text[:500]}
    finally:
        conn.close()


def dash_job(base, path, body, limit):
    """Starts one of sparkDash's benchmarks and waits for it; a 429 (its cooldown) is waited out."""

    waited = 0
    while True:
        status, job = dash_call(base, "POST", path, body)
        if status == 429 and waited < 600:
            time.sleep(10)
            waited += 10
            continue
        if status >= 300 or "benchId" not in job:
            return {"status": "refused", "http": status, "error": job.get("error", job), "config": body}
        break
    started = time.time()
    while job.get("status") == "running" and time.time() - started < limit:
        time.sleep(2)
        status, job = dash_call(base, "GET", f"{path}/{job['benchId']}")
        if status != 200:
            return {"status": "lost", "http": status, "error": job, "config": body}
    return job


def sparkdash(args):
    """sparkDash's decode benchmark, a prompt type at a time at concurrency 1, then its prefill benchmark."""

    out = {"dash": args.dash, "spark": args.spark, "decode": [], "prefill": None}
    for kind in args.types.split(","):
        body = {"port": args.port, "modelId": args.model, "concurrencies": [1], "maxTokens": args.max_tokens,
                "promptType": kind}
        job = dash_job(args.dash, f"/api/sparks/{args.spark}/llm/bench", body, 1200)
        out["decode"].append(job)
        wave = (job.get("results") or [{}])[0]
        print(f"decode {kind}: {job.get('status')} {wave.get('medianDecodeTps')} tok/s, ttft "
              f"{wave.get('medianTtftMs')} ms, {wave.get('totalCompletionTokens')} tokens {job.get('error') or ''}",
              flush=True)
    sizes = [int(s) for s in args.sizes.split(",") if s]
    if sizes:
        job = dash_job(args.dash, f"/api/sparks/{args.spark}/llm/prefill-bench",
                       {"port": args.port, "modelId": args.model, "contextSizes": sizes}, 3600)
        out["prefill"] = job
        for row in job.get("results") or []:
            print(f"prefill {row.get('targetTokens')}: {row.get('promptTokens')} tokens, ttft {row.get('ttftMs')} ms, "
                  f"{row.get('prefillTps')} tok/s {row.get('error') or ''}", flush=True)
        if not job.get("results"):
            print(f"prefill: {job.get('status')} {job.get('error')}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return 0


class Fake(http.server.BaseHTTPRequestHandler):
    """A server that decodes nothing: a second of 'prompt' for 2000 tokens, then a word every 10 ms."""

    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake"}]} if "models" in self.path else {"status": "ok"})
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = json.dumps(body["messages"])
        held = max((len(os.path.commonprefix([text, s])) for s in Fake.seen), default=0) // 4
        Fake.seen.append(text)
        prompt = len(text) // 4
        time.sleep((prompt - held) / 2000.0)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        count = body.get("max_tokens", 16)
        for i in range(count):
            delta = {"content": f"w{i} "}
            frame = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.01)
        end = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": prompt, "completion_tokens": count,
                         "prompt_tokens_details": {"cached_tokens": held}}}
        self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()
        print(f"info: prompt {prompt} tokens, {held} held, in {(prompt - held) / 2000.0:.2f} s; reply {count} tokens in "
              f"{count * 0.01:.2f} s, 100.0 tok/s; 0 of 0 drafts kept; 0 snapshots, 0 MiB", file=sys.stderr, flush=True)
        self.close_connection = True


# Distinct short prompts, so a shared round does not fork one stream onto another.
_STREAM_PROMPTS = (
    "Write one paragraph about a river that meets the sea.",
    "Write one paragraph about a mountain road in winter.",
    "Write one paragraph about a harbour at dawn.",
    "Write one paragraph about an orchard after rain.",
    "Write one paragraph about a night train crossing a plain.",
    "Write one paragraph about a market square at noon.",
    "Write one paragraph about a lighthouse in a storm.",
    "Write one paragraph about a library at closing time.",
)
# Replies that list what the model knows by heart, so the head's chains run long (spec 077: six drafts and
# more a window, where the prose above keeps two or three). Nothing here is in the prompt to copy from, and
# each runs past 256 tokens.
_CHAIN_PROMPTS = (
    "Count from 1 to 400, one number per line. Output only the numbers.",
    "Write the multiplication table of 7 from 7 x 1 to 7 x 80, one line each in the form 7 x 1 = 7. Output only the table.",
    "List every date of the year 2025 in order in the form 2025-01-01, one per line. Output only the dates.",
    "Count down from 900 to 0 in steps of 3, the numbers separated by commas on one line. Output only the numbers.",
)
# Four short coding tasks, a function each: a set no draft rule's figures were fitted on (spec 077 WS-02).
_CODE_PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list. Output only the code.",
    "Write a Go function that reverses a singly linked list in place. Output only the code.",
    "Write a Rust function that returns the n-th Fibonacci number with u64 arithmetic and no recursion. Output only the code.",
    "Write a JavaScript function that groups an array of objects by the value of a given key. Output only the code.",
)
_STREAM_SETS = {"prose": _STREAM_PROMPTS, "chains": _CHAIN_PROMPTS, "code": _CODE_PROMPTS}


def streams(args):
    """N clients at once, for each count: the decode rate in total and a stream.

    One process, the server left up. Each count is a fresh set of requests, started together, greedy, thinking
    off. A stream's rate is client_bench's: (completion tokens - 1) / (last delta - first delta). The total is
    those tokens over the span from the earliest first delta to the latest last delta, and a stream of that
    total is the total divided by the count. `--prompts FILE.json` is a list of prompts in place of a set, and
    `--all` runs a count over every group of that many prompts in turn (the first n, the next n; a rest short of
    n left out): the row's tokens and span are the groups' sums, so a count's rate is over the whole list and
    not over its first prompts (spec 085: a draft rule moves one reply by several percent either way).
    """

    model = args.model
    if not model:
        u = urlparse(args.url)
        conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        conn.request("GET", "/v1/models")
        model = json.loads(conn.getresponse().read())["data"][0]["id"]
        conn.close()
    counts = [int(c) for c in args.counts.split(",") if c]
    if args.prompts in _STREAM_SETS:
        prompts = _STREAM_SETS[args.prompts]
    else:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = json.load(f)
    if any(n < 1 or n > len(prompts) for n in counts):
        print(f"counts must be between 1 and {len(prompts)}", file=sys.stderr)
        return 1
    rows = []
    for n in counts:
        groups = [list(range(g, g + n)) for g in range(0, len(prompts) - n + 1, n)] if args.all else [list(range(n))]
        parts = [_streams_group(args, model, prompts, group) for group in groups]
        if any("error" in part for part in parts):
            bad = next(part for part in parts if "error" in part)
            print(f"streams {n}: failed {bad['error']}", file=sys.stderr, flush=True)
            rows.append({"count": n, "error": bad["error"], "status": bad.get("status")})
            continue
        if len(parts) == 1:
            row = dict(parts[0], count=n)
        else:
            tokens, span = sum(part["decode_tokens"] for part in parts), sum(part["span_s"] for part in parts)
            row = {"count": n, "groups": len(parts), "decode_tokens": tokens, "span_s": round(span, 4),
                   "total_tps": round(tokens / span, 2) if span else 0.0,
                   "stream_tps": round(tokens / span / n, 2) if span else 0.0,
                   "group_tps": [part["total_tps"] for part in parts]}
            for key in ("each_tps", "completion_tokens", "prompt_tokens", "sha"):
                row[key] = [v for part in parts for v in part[key]]
        rows.append(row)
        print(f"streams {n}: total {row['total_tps']} tok/s, a stream {row['stream_tps']} tok/s, "
              f"each {row['each_tps']}, tokens {row['completion_tokens']}", flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "model": model, "seed": args.seed, "max_tokens": args.max_tokens,
                   "prompts": args.prompts, "counts": rows}, f, indent=1)
    return 1 if any("error" in row for row in rows) else 0


def _streams_group(args, model, prompts, group):
    """The prompts `group` names, started together: their tokens, span and rates, or the first failure."""
    n = len(group)
    bodies = [body_of(model, [{"role": "user", "content": prompts[i]}], args.max_tokens,
                      seed=args.seed + i) for i in group]
    found = [None] * n

    def one(i, body, found=found):
        t0 = time.perf_counter()
        out = send(args.url, body)
        out["t0"] = t0
        found[i] = out

    threads = [threading.Thread(target=one, args=(i, body)) for i, body in enumerate(bodies)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    failed = [i for i, out in enumerate(found) if out.get("error") or out.get("status") != 200]
    if failed:
        return {"error": f"{failed} {found[failed[0]].get('error')}", "status": found[failed[0]].get("status")}
    tokens = 0
    firsts, lasts = [], []
    each = []
    for out in found:
        done = out["completion_tokens"] or 0
        tokens += max(done - 1, 0)
        if out["first_delta"] is not None:
            firsts.append(out["t0"] + out["first_delta"])
        if out["last_delta"] is not None:
            lasts.append(out["t0"] + out["last_delta"])
        each.append(out.get("decode_tps"))
    span = (max(lasts) - min(firsts)) if firsts and lasts and max(lasts) > min(firsts) else 0.0
    total = tokens / span if span else 0.0
    row = {"count": n, "decode_tokens": tokens, "span_s": round(span, 4),
           "total_tps": round(total, 2), "stream_tps": round(total / n, 2) if n else 0.0,
           "each_tps": [None if v is None else round(v, 2) for v in each],
           "completion_tokens": [out["completion_tokens"] for out in found],
           "prompt_tokens": [out["prompt_tokens"] for out in found],
           "sha": [hashlib.sha256((out.get("content") or "").encode()).hexdigest()[:12] for out in found]}
    return row


def fake(args):
    http.server.ThreadingHTTPServer((args.host, args.port), Fake).serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run")
    r.add_argument("--url", required=True)
    r.add_argument("--set", default=os.path.join(HERE, "client_bench.json"))
    r.add_argument("--seed", type=int, default=1, help="the salts' seed: a load's number, the same for both servers")
    r.add_argument("--model", default="", help="the model id to ask for (default: the first of /v1/models)")
    r.add_argument("--only", default="", help="request names or kinds (warm, decode, prompt, turns), comma separated")
    r.add_argument("--out", required=True)
    r.set_defaults(func=run)
    s = commands.add_parser("streams")
    s.add_argument("--url", required=True)
    s.add_argument("--model", default="")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--counts", default="1,2,4")
    s.add_argument("--max-tokens", type=int, default=128)
    s.add_argument("--prompts", default="prose",
                   help="the streams' prompts: prose (paragraphs), chains (lists whose draft chains run long, four), "
                        "code, or a JSON file that is a list of prompts")
    s.add_argument("--all", action="store_true", help="a count over every group of that many prompts, summed")
    s.add_argument("--out", required=True)
    s.set_defaults(func=streams)
    t = commands.add_parser("table")
    t.add_argument("loads", nargs=2, help="LABEL=DIR,DIR,...")
    t.set_defaults(func=table)
    d = commands.add_parser("sparkdash")
    d.add_argument("--dash", help="sparkDash URL (defaults to SPARKDASH_URL in .env.local)")
    d.add_argument("--spark", required=True, help="sparkDash's id of the machine the server runs on")
    d.add_argument("--port", type=int, default=8888)
    d.add_argument("--model", required=True)
    d.add_argument("--types", default="structured,prose,code,json")
    d.add_argument("--max-tokens", type=int, default=400)
    d.add_argument("--sizes", default="4096,8192,16384,32768")
    d.add_argument("--out", required=True)
    d.set_defaults(func=sparkdash)
    f = commands.add_parser("fake")
    f.add_argument("--host", default="127.0.0.1")
    f.add_argument("--port", type=int, default=8012)
    f.set_defaults(func=fake)
    args = parser.parse_args()
    if args.func is sparkdash:
        args.dash = args.dash or local_setting("SPARKDASH_URL")
        if not args.dash:
            parser.error("sparkdash requires --dash or SPARKDASH_URL in .env.local")
    sys.exit(args.func(args) or 0)


if __name__ == "__main__":
    main()
