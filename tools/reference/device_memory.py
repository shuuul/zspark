#!/usr/bin/env python3
"""Where zspark's device memory goes (spec 030): reads the lines `ZSPARK_ALLOCATIONS=1 zspark serve ...` writes
to stderr, one for each device buffer made or freed, names each buffer's call sites with `addr2line -i` and sums
the buffers alive by kind.

  device_memory.py LOG --binary zig-out/bin/zspark [--at LABEL] [--sites] [--json OUT]

`--at` stops at a mark: `model`, `session`, `graphs` (the default: a server that has loaded and answered
nothing) or `stopped` (after its last request, before it frees anything). `--sites` prints every call path with its
bytes instead of the kinds. The binary must be the one that wrote the log, not stripped. The standard library and
binutils only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

MIB = 1024 * 1024
# the names a linear gives the memory it computes in, as against the weights it holds
SCRATCH = {"part", "halves", "rotated", "face.staging", "staging", "counters", "partials", "scratch", "x", "out", "sums"}


def kind(variable: str, path: str) -> str:
    """What a buffer is for, from the name its source line gives it and the functions that made it."""

    head = " (MTP head)" if ("mtp." in path or "Mtp." in path) else ""
    if "Snapshot" in path:
        return "snapshots"
    if "attention.Cache.init" in path:
        return "attention caches: keys, values, indexer keys, pooled blocks" + head
    if "gdn.Scratch.init" in path:
        return "DeltaNet scratch: each layer's for the rows it keeps of a window, one for wider windows"
    if "State.init" in path:
        return "recurrent state: DeltaNet (two sides), conv inputs, PLE tails"
    if "forward.Buffers" in path or "mtp.Buffers" in path or "mtp.Head.init" in path:
        return "window buffers" + head
    if "session." in path and "Model.loadWith" not in path:
        return "session, other"
    if variable.removeprefix("self.") in SCRATCH:
        return "linears' scratch, a part lent by capacity and the other roles by shape, and the trellis counters" + head
    if head:
        return "weights: MTP head (its layer, experts, read-out)"
    if "DraftHead" in path:
        return "weights: draft head"
    if "loadMoe" in path or "loadPackedMoe" in path:
        return "weights: experts (routed, shared, router)"
    if "ple." in path:
        return "weights: PLE projections"
    if "loadLayer" in path:
        return "weights: layers' linears, hyper-connections, norms"
    if variable == "embed":
        return "weights: embedding"
    if "Model.loadWith" in path:
        return "weights: head and mixer"
    return "unclassified"


def read(path: str, at: str):
    alive: dict[int, dict] = {}
    marks = {}
    for line in open(path, errors="replace"):
        if not line.startswith("zspark-memory "):
            continue
        words = line.split()
        if words[1] == "alloc":
            head, _, stack = line.partition("|")
            shape = [int(n) for n in re.findall(r"\d+", head.split("{", 1)[1])] if "{" in head else []
            alive[int(words[2], 16)] = {
                "bytes": int(words[3]), "dtype": words[4], "shape": shape,
                "stack": [int(n) for n in re.findall(r"\d+", stack)],
            }
        elif words[1] == "free":
            alive.pop(int(words[2], 16), None)
        elif words[1] == "mark":
            marks[words[2]] = dict(word.split("=") for word in words[3:])
            if words[2] == at:
                break
    return list(alive.values()), marks


def name(binary: str, addresses: set[int]) -> dict[int, list[tuple[str, str]]]:
    """Each address as its inlined frames, innermost first: (function, file:line)."""

    ordered = sorted(a for a in addresses if a < 1 << 40)   # the program's own text; libc's frames are left out
    if not ordered:
        return {}
    out = subprocess.run(["addr2line", "-a", "-i", "-f", "-e", binary] + [hex(a - 1) for a in ordered],
                         check=True, capture_output=True, text=True).stdout.splitlines()
    named: dict[int, list[tuple[str, str]]] = {}
    current = None
    i = 0
    while i < len(out):
        if out[i].startswith("0x"):
            current = int(out[i], 16) + 1
            named[current] = []
            i += 1
            continue
        where = out[i + 1].split(" ")[0]
        named[current].append((out[i], where.rsplit("/", 1)[-1], held(where)))
        i += 2
    return named


SOURCES: dict[str, list[str]] = {}


def held(where: str) -> str:
    """The name the source gives the buffer made at `where` (path:line): the variable or field the line sets."""

    path, _, line = where.rpartition(":")
    if not line.isdigit():
        return ""
    if path not in SOURCES:
        try:
            SOURCES[path] = open(path, errors="replace").read().splitlines()
        except OSError:
            SOURCES[path] = []
    lines = SOURCES[path]
    text = lines[int(line) - 1] if 0 < int(line) <= len(lines) else ""
    found = re.search(r"(?:const|var)\s+(\w+)|\.?([\w.\[\]]+)\s*=\s*(?:try\s+)?(?:\.init|cuda\.Buffer|try)", text)
    return (found.group(1) or found.group(2)) if found else ""


def path_of(buffer: dict, named) -> list[tuple[str, str]]:
    frames = []
    for address in buffer["stack"]:
        for function, where, variable in named.get(address, []):
            if function.startswith("cuda.") or function in ("_start", "main", "??") or function.startswith("start."):
                continue
            if frames and frames[-1][0] == function:
                if frames[-1][1].endswith("?") or frames[-1][1].startswith("cuda.zig"):
                    frames[-1] = (function, where, variable)
                continue
            frames.append((function, where, variable))
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log")
    parser.add_argument("--binary", required=True)
    parser.add_argument("--at", default="graphs")
    parser.add_argument("--sites", action="store_true")
    parser.add_argument("--depth", type=int, default=6, help="frames of a call path that --sites prints")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    buffers, marks = read(args.log, args.at)
    named = name(args.binary, {a for b in buffers for a in b["stack"]})
    sites: dict[tuple, list[int]] = {}
    kinds: dict[str, list[int]] = {}
    for buffer in buffers:
        frames = path_of(buffer, named)
        of = kinds.setdefault(kind(frames[0][2] if frames else "", " < ".join(f for f, _, _ in frames)), [0, 0])
        of[0] += buffer["bytes"]
        of[1] += 1
        shape = "x".join(str(n) for n in buffer["shape"])
        key = ((frames[0][2] if frames else "") + ": " + " < ".join(f"{f}@{w}" for f, w, _ in frames[: args.depth]), buffer["dtype"], shape)
        entry = sites.setdefault(key, [0, 0])
        entry[0] += buffer["bytes"]
        entry[1] += 1
    total = sum(b["bytes"] for b in buffers)
    rows = sorted(sites.items(), key=lambda kv: -kv[1][0])
    if args.json:
        json.dump({"total": total, "marks": marks, "kinds": {k: v[0] for k, v in kinds.items()},
                   "sites": [{"path": k[0], "dtype": k[1], "shape": k[2], "bytes": v[0], "count": v[1]} for k, v in rows]},
                  open(args.json, "w"), indent=1)
    if args.sites:
        for (path, dtype, shape), (size, count) in rows:
            print(f"{size / MIB:10.1f} MiB {count:5d}  {dtype} [{shape}]  {path}")
    else:
        for label, (size, count) in sorted(kinds.items()):
            print(f"{size / MIB:10.1f} MiB {count:5d}  {label}")
    print(f"{total / MIB:10.1f} MiB in {len(buffers)} buffers at `{args.at}`")
    for label, mark in marks.items():
        print(f"  mark {label}: buffers {int(mark['bytes']) / MIB:.1f} MiB, device free {int(mark['device_free']) / MIB:.0f} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
