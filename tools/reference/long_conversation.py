"""One conversation of more than 100,000 tokens with a compaction, and how each `--kv-dtype` holds it (spec 100).

Not a fixture and never runtime: a measurement, run by hand. The standard library only.

The conversation is an agent's session over zspark's own text: each turn hands the model one of the repository's
sources or documents (cut at 14,000 characters) and asks what it does, and every sixth turn asks instead, without
the file, what it said of one read long before. Replies are greedy (temperature 0), thinking off, 256 tokens at
most. The server compacts as TensorFold's host does (`zig/vendor/tensorfold/zig/src/server/compact.zig`,
`--compact-at auto`: above 85% of the window, with the reply's room): the middle of the conversation becomes a
note the model writes, the system message and the last 20,000 tokens or so are kept. The client then goes on from
the compacted messages, rebuilt from the reply's `compaction` stamp, for `--after` turns more.

    drive   run the conversation against a `zspark serve ... --compact-at auto` on 127.0.0.1, into OUT/conversation.json
            python3 tools/reference/long_conversation.py drive --url http://127.0.0.1:18888 --repo . --out OUT \
                --zspark BIN --checkpoint DIR
    cases   the prompts of chosen turns (by depth, before and after the compaction, and the note's own prompt) as
            `generate cases=` lines, and the whole conversation uncompacted as ids to score, rendered by
            `zspark chat --render` and checked against the server's `prompt_tokens`
            python3 tools/reference/long_conversation.py cases --out OUT --zspark BIN --checkpoint DIR
    part    from `generate ... cases=OUT/cases.txt score-each=OUT/score_ids.txt KV` outputs, a storage a file:
            where each storage's greedy replies part from the first file's, and the nll by bands of position
            python3 tools/reference/long_conversation.py part --out OUT bf16=FILE fp8=FILE fp8-nvfp4=FILE
"""

import argparse, json, os, re, subprocess, sys, time, urllib.request
from urllib.parse import urlsplit

from local_env import local_setting

MARKER = "Memory of the earlier conversation:"
SPLIT_HEAD = "## Kept reply context\n"
FILES_HEAD = "## Files read and changed\n"
NOTE_TASK = (
    "Rewrite the memory note for this conversation so it keeps everything still needed.\n"
    "Markdown sections: Goal, Constraints and preferences, Progress, Decisions, Next steps, Critical context.\n"
    "Progress lists done, in progress and blocked. Decisions keep every earlier one and its reason.\n"
    "Next steps stay in order. Critical context keeps exact names, paths, numbers and error text.\n"
    "Under 800 words."
)
SYSTEM = ("You are a coding assistant in the zspark repository (LLM decoding in Zig and CUDA for the DGX Spark). "
          "The user shows you its files one at a time. Answer briefly and exactly; name functions as they are "
          "written.")
CUT = 14000
REPLY = 256


def files(repo):
    """zspark's own sources and documents, in a fixed order that alternates the kinds."""
    picked = {"zig": [], "md": []}
    for top in ("zig/src", "docs", "zig/kernels"):
        for root, dirs, names in os.walk(os.path.join(repo, top)):
            dirs.sort()
            for name in sorted(names):
                path = os.path.relpath(os.path.join(root, name), repo)
                if name.endswith(".zig"):
                    picked["zig"].append(path)
                elif name.endswith(".md"):
                    picked["md"].append(path)
                elif name.endswith((".cu", ".cuh")):
                    picked["zig"].append(path)
    order = []
    z, m = picked["zig"], picked["md"]
    # every 4th source and every 2nd document, so the session ranges over the repository
    z, m = z[::4], m[::2]
    while z or m:
        if z:
            order.append(z.pop(0))
        if z:
            order.append(z.pop(0))
        if m:
            order.append(m.pop(0))
    return order


def read_turn(repo, path):
    text = open(os.path.join(repo, path), encoding="utf-8", errors="replace").read()
    if len(text) > CUT:
        text = text[:CUT] + "\n(cut here)\n"
    kind = "zig" if path.endswith(".zig") else "cuda" if path.endswith((".cu", ".cuh")) else "markdown"
    return (f"Here is `{path}` from the zspark repository:\n\n```{kind}\n{text}\n```\n\n"
            "In at most 120 words: what does it do, and which two functions or sections matter most?")


def recall_turn(path):
    return (f"Without reading it again: you read `{path}` earlier in this session. What did you say it does, and "
            "which function or section did you name first?")


def post(url, body):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        return json.loads(response.read())


def find_compaction(value):
    """The `compaction` stamp wherever the reply's JSON carries it."""
    if isinstance(value, dict):
        if "compaction" in value and isinstance(value["compaction"], dict):
            return value["compaction"], value
        for v in value.values():
            found = find_compaction(v)
            if found[0] is not None:
                return found
    return None, None


def compacted(messages, stamp, late_role, include_split):
    """The messages the host assembled (`assemble`): the leading system messages, the note, the kept tail."""
    note = stamp["note"]
    prefix = 0
    while prefix < len(messages) and messages[prefix]["role"] in ("system", "developer"):
        prefix += 1
    tail = len(messages) - stamp["kept_messages"]
    out = list(messages[:prefix])
    out.append({"role": late_role, "content": f"{MARKER}\n{note}"})
    if include_split and SPLIT_HEAD in note:
        split = note.split(SPLIT_HEAD, 1)[1].split("\n\n" + FILES_HEAD, 1)[0]
        out.append({"role": "user", "content": f"Earlier in this turn:\n{split}"})
    return out + list(messages[tail:]), prefix, tail


def drive(args):
    order = files(args.repo)
    history = [{"role": "system", "content": SYSTEM}]
    full = list(history)  # the same conversation never compacted
    turns, read, compaction_turn, file_at = [], [], None, 0
    while len(turns) < args.max_turns:
        n = len(turns)
        if n % 6 == 5 and len(read) > 8:
            # a file read about two thirds of the session ago
            asked = read[len(read) // 3]
            content, kind, path = recall_turn(asked), "recall", asked
        else:
            if file_at == len(order):
                break
            path = order[file_at]
            file_at += 1
            content, kind = read_turn(args.repo, path), "read"
            read.append(path)
        user = {"role": "user", "content": content}
        sent = history + [user]
        body = {"model": "m", "messages": sent, "max_tokens": REPLY, "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False}}
        start = time.time()
        reply = post(args.url, body)
        seconds = time.time() - start
        text = reply["choices"][0]["message"].get("content") or ""
        stamp, holder = find_compaction(reply)
        row = {"turn": n, "kind": kind, "path": path, "messages": sent, "reply": text, "usage": reply.get("usage"),
               "seconds": round(seconds, 2), "response": reply}
        turns.append(row)
        full += [user, {"role": "assistant", "content": text}]
        print(f"turn {n} {kind} {path}: prompt {reply.get('usage', {}).get('prompt_tokens')}, "
              f"{seconds:.1f} s{', compacted' if stamp else ''}", flush=True)
        if stamp is not None and compaction_turn is None:
            compaction_turn = n
            row["compaction"] = stamp
            used = holder.get("context_used")
            row["context_used"] = used
            # the history the host built, which the client goes on from
            history = None
            for late_role in ("user", "system"):
                for include_split in (True, False):
                    built, prefix, tail = compacted(sent, stamp, late_role, include_split)
                    row.setdefault("candidates", []).append({"late_role": late_role, "split": include_split,
                                                             "messages": built, "prefix": prefix, "tail": tail})
            # the one that renders to the host's own length; `cases` checks it again
            pick = row["candidates"][0]
            for c in row["candidates"]:
                c["rendered"] = len(render(args, c["messages"]))
                if c["rendered"] == used:
                    pick = c
            print(f"compacted to {used} tokens; rebuilt: "
                  + ", ".join(f"{c['late_role']}/{c['split']} {c['rendered']}" for c in row["candidates"]), flush=True)
            history = pick["messages"] + [{"role": "assistant", "content": text}]
        else:
            history = sent + [{"role": "assistant", "content": text}]
        json.dump({"turns": turns, "compaction_turn": compaction_turn, "full": full},
                  open(os.path.join(args.out, "conversation.json"), "w"), indent=1)
        if compaction_turn is not None and n - compaction_turn >= args.after:
            break


def render(args, messages):
    path = os.path.join(args.out, "render.json")
    json.dump(messages, open(path, "w"))
    done = subprocess.run([args.zspark, "chat", args.checkpoint, "--messages", path, "--no-thinking", "--render"],
                          capture_output=True, text=True, check=True)
    for line in done.stdout.splitlines():
        found = re.match(r"(\d+) ids: (.*)$", line)
        if found:
            ids = [int(x) for x in found.group(2).split()]
            assert len(ids) == int(found.group(1))
            return ids
    raise RuntimeError("no ids in zspark chat --render's output")


def transcript(messages):
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def cases(args):
    data = json.load(open(os.path.join(args.out, "conversation.json")))
    turns, at = data["turns"], data["compaction_turn"]
    lines, meta = [], []

    def add(name, ids, count, expected, note=""):
        lines.append(f"{count} name={name} " + " ".join(map(str, ids)))
        meta.append({"name": name, "prompt": len(ids), "count": count, "expected": expected, "note": note})
        print(f"{name}: {len(ids)} tokens {note}", flush=True)

    # each turn before the compaction is the history as sent: its prompt checked against the server's count
    mismatched = 0
    for t in turns[:at]:
        mine = len(render(args, t["messages"]))
        if mine != t["usage"]["prompt_tokens"]:
            mismatched += 1
            print(f"turn {t['turn']}: rendered {mine}, the server's {t['usage']['prompt_tokens']}", flush=True)
    print(f"{mismatched} of {at} turns' prompts rendered to another length than the server's", flush=True)

    chosen = set()
    for depth in (25000, 50000, 75000, 100000):
        for kind in ("read", "recall"):
            near = min((t for t in turns[:at] if t["kind"] == kind),
                       key=lambda t: abs(t["usage"]["prompt_tokens"] - depth))
            chosen.add(near["turn"])
    chosen.add(at - 1)
    for n in sorted(chosen):
        t = turns[n]
        add(f"t{n}-{t['kind']}", render(args, t["messages"]), REPLY, t["reply"])

    # the note's prompt (`oneNote`): the task, then the transcript of the messages between the leading ones and
    # the kept tail (or the turn the tail cuts into)
    row = turns[at]
    stamp = row["compaction"]
    best = None
    for c in row["candidates"]:
        ids = render(args, c["messages"])
        c["rendered"] = len(ids)
        if len(ids) == row.get("context_used"):
            best = (c, ids)
    if best is None:
        raise RuntimeError(f"no rebuilt compaction renders to {row.get('context_used')} tokens: "
                           + ", ".join(str(c["rendered"]) for c in row["candidates"]))
    c, ids = best
    sent, prefix, tail = row["messages"], c["prefix"], c["tail"]
    split_at = None
    if tail < len(sent) and sent[tail]["role"] == "assistant":
        for i in range(tail - 1, prefix - 1, -1):
            if sent[i]["role"] == "user":
                split_at = i
                break
            if sent[i]["role"] == "assistant":
                break
    middle = sent[prefix:split_at if split_at is not None else tail]
    note_messages = [{"role": "system", "content": NOTE_TASK},
                     {"role": "user", "content": "Messages to add:\n" + transcript(middle)}]
    body = stamp["note"].split("\n\n" + SPLIT_HEAD, 1)[0].split("\n\n" + FILES_HEAD, 1)[0]
    add("note", render(args, note_messages), 1024, body, f"({len(middle)} messages)")
    add(f"t{at}-compacted", ids, REPLY, row["reply"], f"({c['late_role']}, split {c['split']})")
    for t in turns[at + 1:]:
        if t["turn"] in (at + 1, len(turns) - 1) or t["kind"] == "recall":
            add(f"t{t['turn']}-{t['kind']}-after", render(args, t["messages"]), REPLY, t["reply"])
            mine = meta[-1]["prompt"]
            if mine != t["usage"]["prompt_tokens"]:
                print(f"turn {t['turn']}: rendered {mine}, the server's {t['usage']['prompt_tokens']}", flush=True)
    open(os.path.join(args.out, "cases.txt"), "w").write("\n".join(lines) + "\n")
    json.dump(meta, open(os.path.join(args.out, "cases.json"), "w"), indent=1)
    whole = render(args, data["full"])
    open(os.path.join(args.out, "score_ids.txt"), "w").write(" ".join(map(str, whole)) + "\n")
    print(f"the whole conversation uncompacted: {len(whole)} tokens", flush=True)


def outputs(path):
    """`generate cases=` output: each case's greedy ids; `score-each`: each token's loss by position."""
    replies, losses, name = {}, {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("== "):
            name = line[3:].strip()
        elif name is not None and "greedy:" in line and name not in replies:
            replies[name] = [int(x) for x in line.split("greedy:", 1)[1].split()]
        else:
            found = re.match(r"^(\d+) (-?[\d.]+) ([01])$", line)
            if found:
                losses[int(found.group(1))] = (float(found.group(2)), int(found.group(3)))
    return replies, losses


def part(args):
    runs = [(a.split("=", 1)[0], outputs(a.split("=", 1)[1])) for a in args.runs]
    meta = json.load(open(os.path.join(args.out, "cases.json")))
    base_name, (base_replies, base_losses) = runs[0]
    end = 248046  # <|im_end|>
    print("| Case | Prompt | " + " | ".join(f"{n}: parts at" for n, _ in runs[1:]) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in runs[1:]) + " |")
    for m in meta:
        base = base_replies.get(m["name"], [])
        stop = base.index(end) + 1 if end in base else len(base)
        cells = []
        for _, (replies, _) in runs[1:]:
            mine = replies.get(m["name"], [])
            at = next((i for i, (a, b) in enumerate(zip(base[:stop], mine)) if a != b), None)
            cells.append(f"{at} of {stop}" if at is not None else f"same {stop}")
        print(f"| {m['name']} | {m['prompt']} | " + " | ".join(cells) + " |")
    if not base_losses:
        return
    band, last = args.band, max(base_losses)

    def mean_se(d, block):
        mu = sum(d) / len(d)
        blocks = [sum(d[i:i + block]) / block for i in range(0, len(d) - block + 1, block)]
        if len(blocks) < 2:
            return mu, float("nan")
        bm = sum(blocks) / len(blocks)
        return mu, (sum((x - bm) ** 2 for x in blocks) / (len(blocks) - 1) / len(blocks)) ** 0.5

    # a cache's error is held by every later row that reads it, so a difference runs on over thousands of
    # tokens: the bands' errors are by blocks of 1,000, the whole text's by 64 and by 4,096 too
    print(f"\n| Positions | {base_name} nll | " + " | ".join(f"{n} - {base_name} ± se" for n, _ in runs[1:]) + " |")
    print("| --- | --- | " + " | ".join("---" for _ in runs[1:]) + " |")
    for lo in range(1, last + 1, band):
        keys = [k for k in range(lo, min(lo + band, last + 1)) if k in base_losses]
        mean = sum(base_losses[k][0] for k in keys) / len(keys)
        cells = []
        for _, (_, losses) in runs[1:]:
            mu, se = mean_se([losses[k][0] - base_losses[k][0] for k in keys if k in losses], 1000)
            cells.append(f"{mu:+.4f} ± {se:.4f}")
        print(f"| {lo}-{keys[-1]} | {mean:.4f} | " + " | ".join(cells) + " |")
    keys = sorted(base_losses)
    mean = sum(base_losses[k][0] for k in keys) / len(keys)
    for block in (64, 4096):
        cells = []
        for _, (_, losses) in runs[1:]:
            mu, se = mean_se([losses[k][0] - base_losses[k][0] for k in keys if k in losses], block)
            cells.append(f"{mu:+.4f} ± {se:.4f}")
        print(f"| all, blocks of {block} | {mean:.4f} | " + " | ".join(cells) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("drive")
    d.add_argument("--url", required=True)
    d.add_argument("--repo", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--after", type=int, default=6)
    d.add_argument("--zspark", required=True)
    d.add_argument("--checkpoint", required=True)
    d.add_argument("--max-turns", type=int, default=80)
    c = sub.add_parser("cases")
    c.add_argument("--out", required=True)
    c.add_argument("--zspark", required=True)
    c.add_argument("--checkpoint", required=True)
    p = sub.add_parser("part")
    p.add_argument("--out", required=True)
    p.add_argument("--band", type=int, default=10000)
    p.add_argument("runs", nargs="+")
    args = parser.parse_args()
    if args.command == "drive":
        host = urlsplit(args.url).hostname
        protected_host = local_setting("MODEL_LAN_HOST")
        if host == protected_host or (not protected_host and host not in ("127.0.0.1", "localhost", "::1")):
            sys.exit("The drive URL must be a benchmark endpoint. Set MODEL_LAN_HOST in .env.local for LAN runs.")
        drive(args)
    elif args.command == "cases":
        cases(args)
    else:
        part(args)


if __name__ == "__main__":
    main()
