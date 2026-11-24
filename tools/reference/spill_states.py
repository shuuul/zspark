"""What a conversation that left the pool costs the next prompt over it, with the states on disk and without
(`serve --spill-gib`, spec 074): the tokens the prompt finds held (`cached_tokens`) and the seconds to its first
token, after the pool let the conversation go and after the server was started again.

Not a fixture and never runtime: a measurement, run by hand through `spill_states.sh`, a load a call. The
standard library only; the requests, the documents and the salts are `pool_states.py`'s and `client_bench.py`'s.

    python3 tools/reference/spill_states.py --phase leave --url http://127.0.0.1:18888 --log server.log --out a.json
    python3 tools/reference/spill_states.py --phase restart --from a.json --url ... --out b.json
    python3 tools/reference/spill_states.py --compare LABEL=a.json LABEL=b.json ...

`leave`, on one load: a warm request; two streams at once, which open the pool (`--parallel` above 1: the
conversations then run in it, each alone); `--conversations` conversations of `--size` tokens opened in turn,
one request each (`open-0`, `open-1`, ...), which is enough of them for the pool's budget of states to have let
the first one's go; then the first one's next turn (`next-of-0`): the row to read. `--out` keeps that turn's
messages.

`restart`, on a load after it: a warm request, then `next-of-0` again as `--from` has it, alone, so the lone
session takes it.

`--compare` prints, a request a row, each load's first token and `cached_tokens`, and whether the reply's hash
is the same in every load that has the request. It judges nothing else.
"""

import argparse, json, os, sys, threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True  # no __pycache__ beside the scripts: a tree with one is dirty to `zspark version`
import client_bench  # noqa: E402
import pool_states  # noqa: E402

Q = pool_states.QUESTIONS


def user(text):
    return {"role": "user", "content": text}


def open_pool(run):
    """Two streams at once: a reply under way and a short request beside it."""

    model = run.args.model
    body = client_bench.body_of(model, [user("Write three paragraphs about the planet Mars.")], 300, seed=1)
    found, threads = [None], []

    def opener():
        def one():
            short = client_bench.body_of(model, [user("List five facts about the planet Mars.")], 64, seed=1)
            found[0] = client_bench.send(run.args.url, short)

        threads.append(threading.Thread(target=one))
        threads[0].start()

    live = run.watch(body, 20, opener)
    for thread in threads:
        thread.join()
    run.lines()
    run.note("open", 0, 0, "live", live)
    if found[0] is not None:
        run.note("open", 0, 0, "beside", found[0])
    if run.args.log and not run.opened():
        raise RuntimeError("the pool did not open: the conversations would be the session's")


def leave(run, size, count):
    run.ask("warm", 0, 0, "warm", [user(run.spec["warm"]["user"])], 32)
    open_pool(run)
    turn = None
    for i in range(count):
        first = user(run.document("spill", size, i) + Q["q1"])
        reply = run.ask("leave", size, 0, f"open-{i}", [first])
        if i == 0:
            turn = [first, reply, user(Q["q2"])]
    run.ask("leave", size, 0, "next-of-0", turn)
    return turn


def restart(run, size, turn):
    run.ask("warm", 0, 0, "warm", [user(run.spec["warm"]["user"])], 32)
    run.ask("restart", size, 0, "next-of-0", turn)


def compare(pairs):
    loads = []
    for pair in pairs:
        label, path = pair.split("=", 1)
        loads.append((label, json.load(open(path, encoding="utf-8"))["requests"]))
    names = []
    for _, rows in loads:
        for row in rows:
            key = (row["phase"], row["name"])
            if key not in names:
                names.append(key)
    out = ["| request | " + " | ".join(f"{label}: first token, s; held of prompt" for label, _ in loads) + " | replies |",
           "| --- |" + " --- |" * (len(loads) + 1)]
    differ = 0
    for key in names:
        cells, hashes = [], set()
        for _, rows in loads:
            row = next((r for r in rows if (r["phase"], r["name"]) == key), None)
            if row is None or row.get("first_delta") is None:
                cells.append("")
                continue
            hashes.add(row["sha"])
            cells.append(f"{row['first_delta']:.3f}; {row['cached_tokens']} of {row['prompt_tokens']}")
        # a hash is compared by a request's text: the same name in another phase is another load's same request
        differ += len(hashes) > 1
        out.append(f"| {key[0]} {key[1]} | " + " | ".join(cells) + f" | {'the same' if len(hashes) == 1 else 'DIFFER'} |")
    # `next-of-0` is one request in both phases
    same = {r["sha"] for _, rows in loads for r in rows if r["name"] == "next-of-0" and r.get("sha")}
    out.append("")
    out.append(f"requests whose replies differ between loads: {differ}; next-of-0 over every load and phase: "
               f"{'one reply' if len(same) == 1 else f'{len(same)} replies'}")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", choices=["leave", "restart"])
    parser.add_argument("--url")
    parser.add_argument("--model", default="m")
    parser.add_argument("--log", help="the server's log, for its own line a request")
    parser.add_argument("--out", help="every request, as JSON, and the turn `restart` sends")
    parser.add_argument("--from", dest="source", help="a `leave` run's --out")
    parser.add_argument("--size", type=int, default=16000, help="a conversation's document in tokens, about")
    parser.add_argument("--conversations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--compare", nargs="+", metavar="LABEL=JSON")
    args = parser.parse_args()
    if args.compare:
        print(compare(args.compare))
        return 0
    if not (args.phase and args.url and args.out):
        parser.error("--phase, --url and --out are required to run")
    spec = client_bench.load_set(os.path.join(HERE, "client_bench.json"))
    if len(spec["text"]) < args.size * spec["chars_per_token"]:
        parser.error(f"the corpus is shorter than {args.size} tokens")
    run = pool_states.Run(args, spec)
    failed, turn = None, None
    try:
        if args.phase == "leave":
            turn = leave(run, args.size, args.conversations)
        else:
            turn = json.load(open(args.source, encoding="utf-8"))["turn"]
            restart(run, args.size, turn)
    except Exception as err:  # what was taken is still written
        failed = str(err)
        print(f"FAILED: {failed}", file=sys.stderr, flush=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"url": args.url, "phase": args.phase, "size": args.size, "conversations": args.conversations,
                   "seed": args.seed, "failed": failed, "turn": turn, "requests": run.rows}, f, indent=1)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
