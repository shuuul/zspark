#!/usr/bin/env python3
"""Rules for ending a draft chain, played over recorded chains (specs 028, 037). Not runtime; standard library only.

A draft chooses no token, so a rule changes only how many rows a round verifies and how many of them are kept.
`zspark generate DIR N chains [follow=FILE] ID..` decodes a reply one row a round and prints, at every position,
the MTP head's chain of 15 drafts with the probability it gave each. Every rule's rounds over that reply follow
from those lines alone (the head's chain from a position does not depend on the rounds before it), so a rule is
tried in milliseconds where a decoded reply takes ten seconds and a load.

  draft_policy.py costs OUT.json ROUNDS.out..    a round's cost fitted to `zspark generate DIR N rounds ..` outputs:
                                                 ms a window by its rows, ms for the head to absorb, ms a chained draft
  draft_policy.py play COSTS.json [top=N] chains-TAG-KIND-NAME.out..
                                                 every rule of the grid over the chains, by modelled tokens a second,
                                                 the replies' kinds (prose, code) weighing the same
  draft_policy.py floors COSTS.json chains-TAG-KIND-NAME.out..
                                                 the product's floors at depth 15 against the default's: each kind's
                                                 modelled gain, and every reply's, whose spread says what one reply
                                                 can show; then how often a chain of a given probability was kept
                                                 whole, which is what a floor is set against
  draft_policy.py split OUT_DIR TAG RUN.out      a `cases=` output cut into a file a reply, OUT_DIR/chains-TAG-NAME.out
                                                 (or rounds-TAG-NAME.out), as the commands above read them
  draft_policy.py depth COSTS.json chains-TAG-KIND-NAME.out..
                                                 spec 061: TensorFold 1.0.2's rules that read costs (`Paying`, its
                                                 `DepthRule`; `Accepted`, its `lanes/depth.zig`) against the default,
                                                 each kind's modelled gain, and what a rule that knew the kept
                                                 drafts would gain (`Oracle`)
  draft_policy.py glm KIND,KIND.. [costs=FILE] RUN.out..
                                                 spec 067 WS-12, GLM-5.3-Flash: `zspark generate DIR chains
                                                 drafter=D ID.. then ID.. then mtp ID..` outputs, the prompts of
                                                 each of the KINDs in turn, the same replies by both drafters
                                                 (`chains by dflash2`, `chains by mtp`). Every stop rule
                                                 of DFlash2's (a cap, the product's floor, the kit's noise-aware
                                                 confidences and its depth by cost), the MTP layer's, and a choice
                                                 of drafter a round, against `fc5:0.3` in tokens a modelled second
  `zspark generate DIR 0 cases=FILE` records many replies' chains on one load (a line a reply: N chains
  [sample=SEED,T,K,P] name=KIND-NAME ID..).

The model: a round is the window's ms at its rows, the head absorbing the kept rows, a chained draft for every
draft the head scored after its first, and a constant for the host. It is fitted, then checked against the
replies it was fitted on (`costs` prints both); a rule it prefers is still timed for real before it is taken.
"""
import json
import re
import sys
from collections import defaultdict

ROUND = re.compile(r"(round|chain) at (\d+): (\d+) drafts, (\d+) kept;(.*)")
WINDOWS = re.compile(r"by its rows:(.*)")
TOTAL = re.compile(r"(\d+) tokens in ([\d.]+) s: [\d.]+ tok/s, ([\d.]+) s of it drafting \((\d+) steps")


def read(path):
    """A `generate` output: the reply's tokens, its round or chain lines, its windows' ms by rows, its totals."""
    out = {"tokens": [], "rounds": [], "windows": {}, "path": path}
    for line in open(path):
        if "greedy:" in line:  # alone on its line, or after a `cases=` reply's `prompt S s;`
            out["tokens"] = [int(t) for t in line.split("greedy:")[1].split()]
        m = ROUND.match(line)
        if m:
            chain = [(int(a.lstrip("/").split(":")[0]), float(a.split(":")[1])) for a in m[5].split()]
            out["rounds"].append({"drafts": int(m[3]), "kept": int(m[4]), "chain": chain})
        m = WINDOWS.search(line)
        if m:
            out["windows"] = {int(r): (float(ms), int(n)) for r, ms, n in re.findall(r"(\d+): ([\d.]+) \((\d+)\)", m[1])}
        m = TOTAL.search(line)
        if m:
            out.update(made=int(m[1]), seconds=float(m[2]), drafting=float(m[3]), steps=int(m[4]))
    return out


def costs(out_path, paths):
    runs = [read(p) for p in paths]
    total, count = defaultdict(float), defaultdict(int)
    for r in runs:
        for rows, (ms, n) in r["windows"].items():
            total[rows] += ms * n
            count[rows] += n
    window = {rows: total[rows] / count[rows] for rows in sorted(total)}
    # rows no reply ran: a line through the measured ones on either side
    known = sorted(window)
    for rows in range(1, 17):
        if rows not in window:
            lo = max([k for k in known if k < rows], default=None)
            hi = min([k for k in known if k > rows], default=None)
            if lo is None:
                a, b = known[0], known[1]
            elif hi is None:
                a, b = known[-2], known[-1]
            else:
                a, b = lo, hi
            window[rows] = window[a] + (window[b] - window[a]) * (rows - a) / (b - a)
    # drafting seconds = absorb x rounds + chained x (drafts scored after a round's first), least squares
    sxx = sxy = syy = sxz = syz = 0.0
    for r in runs:
        x = float(len(r["rounds"]))
        y = float(sum(max(len(c["chain"]) - 1, 0) for c in r["rounds"]))
        z = r["drafting"] * 1e3
        sxx += x * x; sxy += x * y; syy += y * y; sxz += x * z; syz += y * z
    det = sxx * syy - sxy * sxy
    absorb = (sxz * syy - syz * sxy) / det
    chained = (syz * sxx - sxz * sxy) / det
    # what is left of a reply's seconds, a round: the host outside the window and the head
    left = n = 0.0
    for r in runs:
        modelled = sum(window[c["drafts"] + 1] for c in r["rounds"]) + r["drafting"] * 1e3
        left += r["seconds"] * 1e3 - modelled
        n += len(r["rounds"])
    model = {"window": {str(k): round(v, 3) for k, v in sorted(window.items())}, "absorb": round(absorb, 4),
             "chained": round(chained, 4), "host": round(left / n, 4)}
    json.dump(model, open(out_path, "w"), indent=1)
    print(json.dumps(model))
    for r in runs:
        ms = sum(cost(model, c["drafts"], len(c["chain"])) for c in r["rounds"])
        print(f"{r['path'].split('/')[-1]}: {r['seconds']:.2f} s measured, {ms / 1e3:.2f} modelled")


def cost(model, drafts, scored):
    return model["window"][str(drafts + 1)] + model["absorb"] + model["chained"] * max(scored - 1, 0) + model["host"]


class Threshold:
    """TensorFold's rule for this family: the first draft always, then drafts while the head gives each `floor`."""

    def __init__(self, depth, floor):
        self.depth, self.floor, self.name = depth, floor, f"each>={floor:.2f} depth {depth}"

    def chain(self, ps, most):
        n = scored = 0
        for j in range(min(self.depth, most)):
            scored += 1
            low = ps[j] < self.floor
            if low and j > 0:
                break
            n += 1
            if low:
                break
        return n, scored

    def done(self, *a):
        pass


class Product:
    """The first draft always (or only at `first`), then drafts while the chain's probability so far (the product
    of its drafts') stays at `floor` and each draft's own at `each`."""

    def __init__(self, depth, floor, each=0.0, first=0.0):
        self.depth, self.floor, self.each, self.first = depth, floor, each, first
        self.name = (f"product>={floor:.2f}" + (f" each>={each:.2f}" if each else "") +
                     (f" first>={first:.2f}" if first else "") + f" depth {depth}")

    def chain(self, ps, most):
        n = scored = 0
        run = 1.0
        for j in range(min(self.depth, most)):
            scored += 1
            run *= ps[j]
            low = run < self.floor or ps[j] < self.each
            if low and (j > 0 or ps[j] < self.first):
                break
            n += 1
            if low:
                break
        return n, scored

    def done(self, *a):
        pass


class Paying:
    """TensorFold's `DepthRule` (cuda/draft_depth.py, Nemotron's): a draft while the chain's probability so far,
    to `power`, is at least the tokens a ms the recent rounds earned times the ms its row adds; a further draft
    is chained only if a certain one would pay for its row and its level."""

    def __init__(self, model, depth, power=1.0, weight=0.125, scale=1.0):
        self.m, self.depth, self.power, self.weight, self.scale = model, depth, power, weight, scale
        self.name = f"paying x{scale:.2f} power {power:.1f} depth {depth}"
        self.reset()

    def reset(self):
        self.tokens, self.ms = 2.0, cost(self.m, 1, 1)

    def row(self, k):
        return max(self.m["window"][str(k + 1)] - self.m["window"][str(k)], 0.0)

    def chain(self, ps, most):
        rate = self.scale * self.tokens / self.ms
        n = scored = 0
        run = 1.0
        for j in range(min(self.depth, most)):
            scored += 1
            run *= ps[j]
            k = j + 1
            if k > 1 and run ** self.power < rate * self.row(k):
                break
            n += 1
            if run ** self.power < rate * (self.row(k + 1) + self.m["chained"]) if k < 15 else True:
                break
        return n, scored

    def done(self, tokens, drafts, scored):
        self.tokens += self.weight * (tokens - self.tokens)
        self.ms += self.weight * (cost(self.m, drafts, scored) - self.ms)


class Trusting(Paying):
    """`Paying` with the head's probabilities held to what the stream kept (spec 061): a chain's probability to
    `power` is multiplied by the drafts after the first the recent rounds kept over what those probabilities
    said they would. On prose the head is kept half as often as it says, on reasoning more often."""

    def __init__(self, model, depth, power=2.0, trust=0.125, low=0.25, high=2.0):
        self.trust, self.low, self.high = trust, low, high
        super().__init__(model, depth, power)
        self.name = f"trusting {trust:.3f} {low:.2f}-{high:.1f} power {power:.1f}"

    def reset(self):
        super().reset()
        self.got = self.said = 1.0
        self.last = 0.0

    def chain(self, ps, most):
        rate = self.scale * self.tokens / self.ms
        t = min(max(self.got / self.said, self.low), self.high)
        n = scored = 0
        run = 1.0
        self.last = 0.0
        for j in range(min(self.depth, most)):
            scored += 1
            run *= ps[j]
            k = j + 1
            if k > 1 and t * run ** self.power < rate * self.row(k):
                break
            n += 1
            if k > 1:
                self.last += run ** self.power
            if t * run ** self.power < rate * (self.row(k + 1) + self.m["chained"]) if k < 15 else True:
                break
        return n, scored

    def done(self, tokens, drafts, scored):
        super().done(tokens, drafts, scored)
        if drafts > 1:
            self.got += self.trust * (max(tokens - 2, 0) - self.got)
            self.said += self.trust * (self.last - self.said)


class Accepted:
    """TensorFold 1.0.2's `lanes/depth.zig`, on modelled costs: a stream's acceptance by depth (draft j was tried
    when the drafts before it were kept; its estimate moves `rate` toward whether it was), the depth with the
    most expected tokens a ms, and one deeper every `every` rounds. The head's probabilities are not read;
    with `floor` the chain also ends where the default's would."""

    def __init__(self, model, depth, floor=0.0, rate=0.15, every=8, prior=0.8):
        self.m, self.depth, self.floor, self.rate, self.every, self.prior = model, depth, floor, rate, every, prior
        self.name = f"accepted depth {depth}" + (f" product>={floor:.2f}" if floor else "")
        self.reset()

    def reset(self):
        self.p, self.rounds = [self.prior] * self.depth, 0

    def chain(self, ps, most):
        most = min(self.depth, most)
        best, best_rate, expected, run = 1, -1.0, 1.0, 1.0
        for d in range(1, most + 1):
            run *= self.p[d - 1]
            expected += run
            rate = expected / cost(self.m, d, d)
            if rate > best_rate:
                best, best_rate = d, rate
        self.rounds += 1
        if self.rounds % self.every == 0:
            best = min(best + 1, most)
        if not self.floor:
            return best, best
        n, scored = Product(best, self.floor).chain(ps, most)
        return n, scored

    def done(self, tokens, drafts, scored):
        kept = tokens - 1
        for j in range(min(drafts, len(self.p))):
            if kept < j:
                break
            self.p[j] += self.rate * ((1.0 if kept > j else 0.0) - self.p[j])


class Oracle:
    """What the head's judgement leaves: the drafts that will be kept and no other, `play` telling it."""

    name = "oracle"

    def chain(self, ps, most):
        return self.kept, self.kept

    def done(self, *a):
        pass


def play(model, run, rule):
    """The rule's rounds over one reply: (tokens, ms, rounds, drafts, kept)."""
    text, chains = run["tokens"], run["rounds"]
    if hasattr(rule, "reset"):
        rule.reset()
    i = tokens = drafts = kept_all = rounds = 0
    ms = 0.0
    while i < len(chains):
        chain = chains[i]["chain"]
        most = min(len(chain), len(text) - i - 2)
        if isinstance(rule, Oracle):
            rule.kept = 0
            while rule.kept < most and chain[rule.kept][0] == text[i + 1 + rule.kept]:
                rule.kept += 1
        n, scored = rule.chain([p for _, p in chain], most) if most > 0 else (0, 0)
        kept = 0
        while kept < n and chain[kept][0] == text[i + 1 + kept]:
            kept += 1
        ms += cost(model, n, scored)
        rule.done(kept + 1, n, scored)
        rounds += 1
        drafts += n
        kept_all += kept
        tokens += kept + 1
        i += kept + 1
    return tokens, ms, rounds, drafts, kept_all


def grid(model):
    rules = []
    for depth in (4, 6, 8, 10, 12, 15):
        for floor in (0.5, 0.6, 0.7, 0.8, 0.9):
            rules.append(Threshold(depth, floor))
        for floor in (0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7):
            rules.append(Product(depth, floor))
        for floor in (0.5, 0.6):
            for first in (0.1, 0.2, 0.3):
                rules.append(Product(depth, floor, first=first))
        for floor, each in ((0.2, 0.5), (0.3, 0.5), (0.2, 0.6), (0.3, 0.6), (0.4, 0.6)):
            rules.append(Product(depth, floor, each))
        for scale in (0.6, 0.8, 1.0, 1.25, 1.5, 2.0):
            rules.append(Paying(model, depth, scale=scale))
    return rules


DEFAULT_FLOOR = 0.65  # mtp.Head.default_chain
FLOORS = (0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8)
EDGES = (0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0001)


def kind_of(path):
    return re.sub(r".*chains-[^-]*-(\w+)-.*", r"\1", path)


def floors(model, runs):
    def rate(rule, chosen):
        t = ms = 0
        for r in chosen:
            a, b = play(model, r, rule)[:2]
            t += a; ms += b
        return 1e3 * t / ms

    names = sorted({kind_of(r["path"]) for r in runs})
    print(f"{len(runs)} replies; % of modelled tokens a second against product>={DEFAULT_FLOOR} depth 15, by floor")
    print(f"{'':<34}" + "".join(f"{f:>7.2f}" for f in FLOORS) + "   each>=0.70 depth 6")
    rows = [(k, [r for r in runs if kind_of(r["path"]) == k]) for k in names]
    rows += [(r["path"].split("/")[-1], [r]) for r in runs]
    for label, chosen in rows:
        base = rate(Product(15, DEFAULT_FLOOR), chosen)
        print(f"{label:<34}" + "".join(f"{100 * (rate(Product(15, f), chosen) / base - 1):>+7.1f}" for f in FLOORS) +
              f"   {100 * (rate(Threshold(6, 0.7), chosen) / base - 1):+.1f}   ({base:.2f} tok/s)")
    # A floor pays where the chance its draft is kept, times a round's ms, is above what the draft's row adds.
    print("chains kept whole by their probability (the product through the draft), drafts after the first: share (count)")
    for k in names:
        n, kept = [0] * (len(EDGES) - 1), [0] * (len(EDGES) - 1)
        for r in runs:
            if kind_of(r["path"]) != k:
                continue
            text = r["tokens"]
            for i, c in enumerate(r["rounds"]):
                run, whole = 1.0, True
                for j, (token, p) in enumerate(c["chain"]):
                    if i + 1 + j >= len(text):
                        break
                    run *= p
                    whole = whole and token == text[i + 1 + j]
                    if j == 0:
                        continue
                    b = max(e for e in range(len(EDGES) - 1) if EDGES[e] <= run)
                    n[b] += 1
                    kept[b] += whole
        print(f"  {k:<8}" + " ".join(f"{EDGES[b]:.2f}: {kept[b] / n[b]:.2f} ({n[b]})" for b in range(len(n)) if n[b]))


def split(out_dir, tag, path):
    name, out = None, []

    def flush():
        if name and any(TOTAL.search(l) for l in out):
            kind = "chains" if any(l.startswith("chain at") for l in out) else "rounds"
            open(f"{out_dir}/{kind}-{tag}-{name}.out", "w").writelines(out)
            print(f"{kind}-{tag}-{name}.out")

    for line in open(path):
        if line.startswith("== "):
            flush()
            name, out = line[3:].strip(), []
        else:
            out.append(line)
    flush()


def pooled(model):
    """TensorFold's `Costs.measured`: each further row's ms pooled with its neighbours' so that it never rises
    with the rows, which also takes out a size few rounds ran."""
    v = [model["window"][str(r)] for r in range(1, 17)]
    sums, counts = [], []
    for a, b in zip(v, v[1:]):
        sums.append(b - a)
        counts.append(1)
        while len(sums) > 1 and sums[-2] * counts[-1] < sums[-1] * counts[-2]:
            last, n = sums.pop(), counts.pop()
            sums[-1] += last
            counts[-1] += n
    fixed = [v[0]]
    for total, n in zip(sums, counts):
        base = fixed[-1]
        fixed += [base + total / n * (i + 1) for i in range(n)]
    return dict(model, window={str(r + 1): ms for r, ms in enumerate(fixed)})


END = 248046  # <|im_end|>: `generate` decodes past it, and what follows is no reply


def depth(model, runs):
    model = pooled(model)
    for r in runs:
        if END in r["tokens"]:
            cut = r["tokens"].index(END) + 1
            r["tokens"], r["rounds"] = r["tokens"][:cut], r["rounds"][:cut]
    print("ms a window by its rows, pooled: " + " ".join(f"{model['window'][str(r)]:.1f}" for r in range(1, 17)))
    def rate(rule, chosen):
        t = ms = 0
        for r in chosen:
            a, b = play(model, r, rule)[:2]
            t += a; ms += b
        return 1e3 * t / ms

    names = sorted({kind_of(r["path"]) for r in runs})
    rules = [Product(15, f) for f in (0.5, 0.55, 0.6, 0.7)]
    rules += [Paying(model, 15, power=w, scale=x) for w in (1.0, 2.0) for x in (0.5, 1.0, 2.0)]
    rules += [Trusting(model, 15, trust=w, low=lo, high=hi) for w in (0.125, 0.05) for lo, hi in ((0.25, 2.0), (0.25, 1.0), (0.5, 1.5))]
    rules += [Accepted(model, 15), Accepted(model, 15, DEFAULT_FLOOR), Accepted(model, 15, 0.5), Oracle()]
    base = {k: rate(Product(15, DEFAULT_FLOOR), [r for r in runs if kind_of(r["path"]) == k]) for k in names}
    print(f"{len(runs)} replies; modelled tok/s with product>={DEFAULT_FLOOR} depth 15: " + ", ".join(f"{k} {v:.1f}" for k, v in base.items()))
    print(f"{'rule, % against it':<34}" + "".join(f"{k:>8}" for k in names) + "    mean")
    for rule in rules:
        gains = [100 * (rate(rule, [r for r in runs if kind_of(r["path"]) == k]) / base[k] - 1) for k in names]
        print(f"{rule.name:<34}" + "".join(f"{g:>+8.1f}" for g in gains) + f"  {sum(gains) / len(gains):+6.1f}")
    print("each reply under paying x1.00 power 2.0 depth 15, %: " + " ".join(
        f"{r['path'].split('-', 2)[2][:-4]} {100 * (rate(Paying(model, 15, power=2.0), [r]) / rate(Product(15, DEFAULT_FLOOR), [r]) - 1):+.1f} ({len(r['tokens'])})"
        for r in runs))


# -- GLM-5.3-Flash on two ranks (spec 067 WS-12) -------------------------------------------------------------------

# A round's ms by who drafts, its base and each draft's row: spec 068's table (zspark 8690c72, both machines alone).
# A line, so a draft the MTP layer scored and did not take costs nothing here, and neither drafter's catching up
# with rows the other drafted (TensorFold's `mtp_row`, `taps_row`): a choice a round is flattered by that.
GLM_COSTS = {"mtp": [68.0, 9.4], "dflash2": [63.5, 8.8]}
GLM_MTP = (15, 0.35)  # a window's drafts, `mtp.Head.default_confidence`
GLM_DFLASH2 = (5, 0.3)  # TensorFold's `fc5:0.3`, what every rule is held against; zspark's is `fnc7:0.2` since WS-12


def glm_model(costs, arm):
    base, each = costs[arm]
    return {"window": {str(r): base + each * (r - 1) for r in range(1, 17)}, "absorb": 0.0, "chained": 0.0, "host": 0.0}


def read_glm(path):
    """The replies of one `generate ... chains` output: each its tokens and its chains, a pick's probability and,
    from DFlash2, its noise-aware confidence."""
    runs = []
    for line in open(path):
        if line.startswith("reply:"):
            runs.append({"tokens": [int(t) for t in line.split()[1:]], "rounds": [], "noisy": [], "path": path})
        if line.startswith("chains by "):
            runs[-1]["arm"] = line.split()[2].rstrip(",")
        m = ROUND.match(line)
        if m and runs:
            picks = [a.split(":") for a in m[5].split()]
            runs[-1]["rounds"].append({"chain": [(int(a[0]), float(a[1])) for a in picks]})
            runs[-1]["noisy"].append({"chain": [(int(a[0]), float(a[2])) for a in picks if len(a) > 2]})
    return runs


class Fixed:
    """A cap and no confidence."""

    def __init__(self, depth):
        self.depth, self.name = depth, f"cap {depth}"

    def chain(self, ps, most):
        n = min(self.depth, most)
        return n, n

    def done(self, *a):
        pass


class ByCost:
    """The kit's `fcostN:noisy` (`dflash2.best_depth`): the depth with the most expected tokens a round's ms, a
    draft kept with its confidence when those before it were."""

    def __init__(self, model, depth):
        self.m, self.depth, self.name = model, depth, f"by cost, depth {depth}"

    def chain(self, ps, most):
        best, best_rate, alive, expect = 0, 0.0, 1.0, 0.0
        for k in range(min(self.depth, most) + 1):
            if k:
                alive *= ps[k - 1]
            expect += alive
            if expect / cost(self.m, k, k) > best_rate:
                best, best_rate = k, expect / cost(self.m, k, k)
        return best, best

    def done(self, *a):
        pass


class Choice:
    """TensorFold's `DrafterChoice` (drafter_choice.py): two rounds of each to begin, then the drafter whose last
    six rounds gave more tokens a ms, left when the other is `margin` better, the other tried every eighth round
    and soon after a switch."""

    def __init__(self, first="f", explore=2, every=8, margin=0.03, window=6, recheck=3):
        self.first, self.explore, self.every, self.margin, self.window, self.recheck = first, explore, every, margin, window, recheck
        self.rounds, self.choice, self.run, self.since = [], first, 0, {"m": 0, "f": 0}

    def rate(self, arm):
        rs = [r for r in self.rounds[self.since[arm]:] if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self, _):
        n, other = len(self.rounds), {"m": "f", "f": "m"}
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return other[self.first]
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(other[cur])
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin
        if ro is not None and (rc is None or ro > rc * need):
            self.since[cur] = n
            self.choice = cur = other[cur]
            self.run = max(0, self.every - self.recheck) if self.recheck else 0
        if self.every and self.run >= self.every:
            self.run = 0
            return other[cur]
        self.run += 1
        return cur

    def record(self, arm, tokens, ms):
        self.rounds.append((arm, tokens, ms))


class Knowing:
    """The drafter that gives more tokens a ms this round, known beforehand: what a choice a round can reach."""

    def pick(self, rates):
        return max(rates, key=rates.get)

    def record(self, *a):
        pass


def play_both(models, runs, rules, chooser):
    """`play` with the drafter chosen each round: `runs` and `rules` by arm ("f" DFlash2, "m" the MTP layer)."""
    text = runs["f"]["tokens"]
    i = tokens = rounds = drafts = kept_all = 0
    ms = 0.0
    while i < len(runs["f"]["rounds"]):
        tried = {}
        for arm in ("f", "m"):
            chain = runs[arm]["rounds"][i]["chain"]
            most = min(len(chain), len(text) - i - 2)
            n, scored = rules[arm].chain([p for _, p in chain], most) if most > 0 else (0, 0)
            kept = 0
            while kept < n and chain[kept][0] == text[i + 1 + kept]:
                kept += 1
            tried[arm] = (n, kept, cost(models[arm], n, scored))
        arm = chooser.pick({a: (k + 1) / c for a, (n, k, c) in tried.items()})
        n, kept, c = tried[arm]
        chooser.record(arm, kept + 1, c)
        ms += c
        rounds += 1
        drafts += n
        kept_all += kept
        tokens += kept + 1
        i += kept + 1
    return tokens, ms, rounds, drafts, kept_all


def glm(args):
    kinds = args[0].split(",")
    costs = dict(GLM_COSTS)
    tf = None
    runs = []
    for a in args[1:]:
        if a.startswith("costs="):
            costs = json.load(open(a[6:]))
        elif a.startswith("tf="):
            tf = [float(x) for x in a[3:].split(",")]
        else:
            runs += read_glm(a)
    f = [r for r in runs if r["arm"] == "dflash2"]
    m = [r for r in runs if r["arm"] == "mtp"]
    # the recordings may be the MTP layer's alone: spec 092's, where no DFlash2 drafter is loaded
    if f:
        assert len(f) == len(m) and len(f) % len(kinds) == 0, (len(f), len(m), len(kinds))
        for a, b in zip(f, m):
            assert a["tokens"] == b["tokens"], "the two drafters' recordings are not the same replies"
        kind = [kinds[i % len(kinds)] for i in range(len(f))]
    else:
        assert len(m) % len(kinds) == 0, (len(m), len(kinds))
        kind = [kinds[i % len(kinds)] for i in range(len(m))]
    noisy = [dict(r, rounds=r["noisy"]) for r in f]
    mf, mm = glm_model(costs, "dflash2"), glm_model(costs, "mtp")
    names = sorted(set(kinds))

    def total(played):
        t = ms = rounds = drafts = kept = 0
        for a, b, c, d, e in played:
            t += a; ms += b; rounds += c; drafts += d; kept += e
        return t, ms, rounds, drafts, kept

    def rows(label, play_one, over):
        # `over`: the replies the rule plays over, f's or m's; kind[i] indexes the paired lists alike
        by = {k: total([play_one(i) for i in range(len(over)) if kind[i] == k]) for k in names}
        return label, by, total(by.values())

    table = []
    if f:
        table.append(rows(f"DFlash2 fc{GLM_DFLASH2[0]}:{GLM_DFLASH2[1]} (TensorFold's default)",
                          lambda i: play(mf, f[i], Product(*GLM_DFLASH2)), f))
        for depth in range(1, 8):
            table.append(rows(f"DFlash2 cap {depth}", lambda i, d=depth: play(mf, f[i], Fixed(d)), f))
        for depth in (4, 5, 6, 7):
            for floor in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
                if (depth, floor) != GLM_DFLASH2:
                    table.append(rows(f"DFlash2 fc{depth}:{floor}", lambda i, d=depth, x=floor: play(mf, f[i], Product(d, x)), f))
        for depth in (5, 6, 7):
            for floor in (0.2, 0.3, 0.4, 0.5):
                table.append(rows(f"DFlash2 fnc{depth}:{floor} (noise-aware)",
                                  lambda i, d=depth, x=floor: play(mf, noisy[i], Product(d, x)), f))
        table.append(rows("DFlash2 fcost7:noisy", lambda i: play(mf, noisy[i], ByCost(mf, 7)), f))
        table.append(rows("DFlash2 knowing the kept drafts", lambda i: play(mf, f[i], Oracle()), f))
        both = lambda i, chooser: play_both({"f": mf, "m": mm}, {"f": f[i], "m": m[i]}, {"f": Product(*GLM_DFLASH2), "m": Product(*GLM_MTP)}, chooser)
        table.append(rows("a choice a round, TensorFold's rule", lambda i: both(i, Choice()), f))
        table.append(rows("a choice a round, knowing the round", lambda i: both(i, Knowing()), f))
    # the MTP layer's rules, first its default: today's (15, 0.35), and spec 092's grid of a cap a round and
    # a floor, (3, 0.35) among them, TensorFold's rule
    table.append(rows(f"MTP, product>={GLM_MTP[1]} depth {GLM_MTP[0]} (its default)", lambda i: play(mm, m[i], Product(*GLM_MTP)), m))
    for cap in range(1, 16):
        for floor in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            if (cap, floor) != GLM_MTP:
                table.append(rows(f"MTP product>={floor} depth {cap}", lambda i, d=cap, x=floor: play(mm, m[i], Product(d, x)), m))

    rate = lambda t: 1e3 * t[0] / t[1]
    base = table[0]
    head = (f"{'rule':<44}{'tok/s':>7}{'gain %':>8}{'tok/round':>10}{'ms/round':>9}{'drafts':>7}{'kept':>6}  ")
    print(f"{len(f) or len(m)} replies, {sum(len(r['tokens']) for r in f or m)} tokens; kinds {names}; a round's ms {costs}")
    tail = f"{'tf ms':>7}{'resid':>7}" if tf else ""
    print(head + " ".join(f"{k:>8}" for k in names) + ("  " + tail if tf else ""))
    for label, by, all_ in [base] + sorted(table[1:], key=lambda row: -rate(row[2])):
        t, ms, rounds, drafts, kept = all_
        line = (f"{label:<44}{rate(all_):>7.2f}{100 * (rate(all_) / rate(base[2]) - 1):>+8.1f}{t / rounds:>10.3f}{ms / rounds:>9.1f}"
                f"{drafts / rounds:>7.2f}{kept / max(drafts, 1):>6.2f}  " + " ".join(f"{100 * (rate(by[k]) / rate(base[1][k]) - 1):>+8.1f}" for k in names))
        if tf:
            # TensorFold's round at the same drafts a round (spec 068's 63.1 + 7.7), and what no rule removes
            theirs = tf[0] + tf[1] * (drafts / rounds)
            line += f"{theirs:>7.1f}{ms / rounds - theirs:>7.1f}"
        print(line)


def main():
    if sys.argv[1] == "glm":
        return glm(sys.argv[2:])
    if sys.argv[1] == "split":
        return split(*sys.argv[2:5])
    if sys.argv[1] == "depth":
        return depth(json.load(open(sys.argv[2])), [read(p) for p in sys.argv[3:]])
    if sys.argv[1] == "costs":
        return costs(sys.argv[2], sys.argv[3:])
    if sys.argv[1] == "floors":
        return floors(json.load(open(sys.argv[2])), [read(p) for p in sys.argv[3:]])
    model = json.load(open(sys.argv[2]))
    top = next((int(a[4:]) for a in sys.argv[3:] if a.startswith("top=")), 30)
    runs = [read(p) for p in sys.argv[3:] if not p.startswith("top=")]
    base = Threshold(6, 0.7)

    def score(rule, chosen):
        t = ms = rounds = drafts = kept = 0
        for r in chosen:
            a, b, c, d, e = play(model, r, rule)
            t += a; ms += b; rounds += c; drafts += d; kept += e
        return {"tok_s": 1e3 * t / ms, "per_round": t / rounds, "ms_round": ms / rounds, "drafts": drafts / rounds,
                "kept": kept / max(drafts, 1)}

    table = []
    names = sorted({re.sub(r".*chains-[^-]*-(\w+)-.*", r"\1", r["path"]) for r in runs})
    by_kind = {k: [r for r in runs if re.sub(r".*chains-[^-]*-(\w+)-.*", r"\1", r["path"]) == k] for k in names}
    base_all = score(base, runs)
    base_kind = {k: score(base, v) for k, v in by_kind.items()}
    for rule in [base] + grid(model):
        s = score(rule, runs)
        kinds = {k: score(rule, v) for k, v in by_kind.items()}
        # the mean of the kinds' gains: a reply of prose and one of code weigh the same, whatever their seconds
        gain = sum(kinds[k]["tok_s"] / base_kind[k]["tok_s"] for k in names) / len(names)
        table.append((gain, rule.name, s, kinds))
    table.sort(key=lambda row: -row[0])
    print(f"{len(runs)} replies; kinds {names}; the rule now: " + "; ".join(
        f"{k} {v['tok_s']:.1f} tok/s, {v['per_round']:.3f} tokens and {v['ms_round']:.1f} ms a round" for k, v in base_kind.items()))
    print(f"{'rule':<40} {'gain':>6}  " + "  ".join(f"{k + ': x, tok/round, ms, drafts, kept':<40}" for k in names))
    shown = table[:top]
    for gain, name, s, kinds in shown + [row for row in table if row[1] == base.name and row not in shown]:
        print(f"{name:<40} {gain:6.4f}  " + "  ".join(
            f"{v['tok_s'] / base_kind[k]['tok_s']:.4f} {v['per_round']:.3f} {v['ms_round']:5.1f} {v['drafts']:.2f} {v['kept']:.3f}".ljust(40)
            for k, v in kinds.items()))


if __name__ == "__main__":
    main()
