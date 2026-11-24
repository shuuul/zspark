#!/usr/bin/env python3
"""A checkpoint's calibration of a draft landing, fitted from recorded chains and played through the pool's
replay (spec 085). Not runtime; standard library only, no GPU.

  lands_fit.py RUN.out CASES   the seven numbers of `mtp.Paying.Landing` fitted on the cases named `KIND-fit-N`,
                               their AUC and calibration on `KIND-held-N` beside the head's probability, and
                               the rule that reads them played on the held-out replies at 1, 2 and 4 seats,
                               then the numbers refitted while serving, as vLLM does, against them fixed

  lands_fit.py costs RUN.out   `Paying.Costs`' row and level from the load's drafted cases (spec 102)

RUN.out is one load's `zspark generate DIR 0 cases=CASES > RUN.out`, CASES `lands_cases.py`'s (about 25 minutes
a pack). A row is a draft of a chain whose earlier drafts all landed: level j of the chain at position i, landed
if it is the reply's token i + 1 + j. The fit is logit(landing) = a logit p + b[min(j, 5)], by Newton's steps:
vLLM's `OnlineAcceptanceEstimator`'s shape, fitted once a checkpoint. `mtp.Rule` then verifies a draft after a
chain's first only where the chain pays by its probability squared, as without the numbers, and by the product
of these, both (`beside` below); `alone` is the product in the square's place with its own trust, which draws
more on prose and was timed slower there. Nothing is fitted on a held-out reply. `chains` prints a probability
to four decimals; the numbers in `mtp.zig` were fitted on the probabilities whole (0.543 and 0.517 the slopes,
0.531 and 0.501 from RUN.out), so a new checkpoint's are these lines' and are timed before they are on.
"""
import math
import re
import sys

import pool_policy as P

LEVELS = 6


def clip(p):
    return min(max(p, 1e-6), 1 - 1e-6)


def logit(p):
    p = clip(p)
    return math.log(p / (1 - p))


def rows_of(replies):
    """(p, level, landed) for each draft whose chain landed before it, as `Seat.chain` counts what lands."""
    out = []
    for r in replies:
        t = r["tokens"]
        for i, ch in enumerate(r["chains"]):
            for j in range(min(len(ch), len(t) - i - 2)):
                y = ch[j][0] == t[i + 1 + j]
                out.append((ch[j][1], j, y))
                if not y:
                    break
    return out


def solve(m, v):
    """m x = v by elimination with pivots; m is small and positive definite."""
    n = len(v)
    a = [row[:] + [v[k]] for k, row in enumerate(m)]
    for c in range(n):
        pivot = max(range(c, n), key=lambda k: abs(a[k][c]))
        a[c], a[pivot] = a[pivot], a[c]
        for k in range(c + 1, n):
            f = a[k][c] / a[c][c]
            for q in range(c, n + 1):
                a[k][q] -= f * a[c][q]
    x = [0.0] * n
    for c in range(n - 1, -1, -1):
        x[c] = (a[c][n] - sum(a[c][q] * x[q] for q in range(c + 1, n))) / a[c][c]
    return x


def fit(rows):
    """The slope and the levels' intercepts: the likelihood's maximum, a ridge of 1e-6 for a level with no row."""
    w = [0.5] + [0.0] * LEVELS
    data = [(logit(p), min(j, LEVELS - 1), 1.0 if y else 0.0) for p, j, y in rows]
    for _ in range(30):
        g = [0.0] * (1 + LEVELS)
        h = [[1e-6 if a == b else 0.0 for b in range(1 + LEVELS)] for a in range(1 + LEVELS)]
        for x, level, y in data:
            s = 1 / (1 + math.exp(-(w[0] * x + w[1 + level])))
            d, v = s - y, s * (1 - s)
            g[0] += d * x
            g[1 + level] += d
            h[0][0] += v * x * x
            h[0][1 + level] += v * x
            h[1 + level][0] += v * x
            h[1 + level][1 + level] += v
        step = solve(h, g)
        w = [a - b for a, b in zip(w, step)]
        if max(abs(b) for b in step) < 1e-9:
            break
    return w[0], w[1:]


def landing(numbers, p, j):
    slope, levels = numbers
    return 1 / (1 + math.exp(-(slope * logit(p) + levels[min(j, LEVELS - 1)])))


def auc(pairs):
    """Of (score, landed) pairs; ties share their mean rank."""
    pairs = sorted(pairs)
    ranks, start, total = 0.0, 0, len(pairs)
    for end in range(1, total + 1):
        if end == total or pairs[end][0] != pairs[start][0]:
            ranks += 0.5 * (start + 1 + end) * sum(1 for k in range(start, end) if pairs[k][1])
            start = end
    pos = sum(1 for _, y in pairs if y)
    return (ranks - pos * (pos + 1) / 2) / (pos * (total - pos)) if pos and total > pos else float("nan")


def judged(pairs):
    """AUC, log loss, and the expected calibration error over ten equal-count bins."""
    pairs = sorted((clip(s), y) for s, y in pairs)
    loss = -sum(math.log(s) if y else math.log(1 - s) for s, y in pairs) / len(pairs)
    ece, size = 0.0, len(pairs) / 10
    for b in range(10):
        part = pairs[round(b * size):round((b + 1) * size)]
        ece += abs(sum(1 for _, y in part if y) - sum(s for s, _ in part))
    return auc(pairs), loss, ece / len(pairs)


class Judged(P.Seat):
    """`mtp.Rule` with a `Paying.Landing` (`kw['numbers']`). `kw['beside']`: a draft pays by the shipped square
    and by the calibrated product, both; else by the product alone, its trust within 0.5 to 1.5 (none with
    `kw['untrusted']`)."""

    def __init__(self, reply, kw):
        P.Seat.__init__(self, reply, kw)
        self.landed_got, self.landed_said = 1.0, 1.0

    def chain(self, rate, of_row):
        ch, t = self.r["chains"][self.i], self.r["tokens"]
        most = min(len(ch), len(t) - self.i - 2)
        self.lands = 0
        while self.lands < most and ch[self.lands][0] == t[self.i + 1 + self.lands]:
            self.lands += 1
        if most <= 0:
            return 0, 0
        beside = self.kw["beside"]
        row, level = rate * P.ROW * of_row, rate * (P.ROW * of_row + P.LEVEL)
        n = scored = 0
        run = product = 1.0
        self.trail, self.products = [], []
        for j in range(most):
            scored += 1
            run *= ch[j][1]
            product *= landing(self.kw["numbers"], ch[j][1], j)
            squared = self.trust(run) * run ** 2
            fitted = product if beside or self.kw.get("untrusted") else min(max(self.landed_got / self.landed_said, 0.5), 1.5) * product
            low = fitted < row or (beside and squared < row)
            if j > 0 and low:
                break
            n += 1
            self.trail.append(run)
            self.products.append(product)
            if j + 1 >= 15 or fitted < level or (beside and squared < level):
                break
        return n, scored

    def done(self, n, scored):
        if n > 1:
            kept = min(n, self.lands)
            self.landed_got += 0.05 * (min(max(kept - 1, 0), n - 1) - self.landed_got)
            self.landed_said += 0.05 * (sum(self.products[1:]) - self.landed_said)
        P.Seat.done(self, n, scored)


class Online:
    """vLLM's `OnlineAcceptanceEstimator` (`3313825`, `v1/worker/gpu/spec_decode/acceptance_estimator.py`): one
    slope and an intercept a level shared by every reply, refitted every 100 steps (rounds) by one Newton step
    over the interval's verified drafts (each kept one and the first refused), damped by n / (n + 50), a ridge
    of 1e-3, a row weighed by the prediction made at draft time."""

    def __init__(self, numbers, seats):
        self.slope, self.levels = numbers[0], list(numbers[1])
        self.every, self.calls = 100 * seats, 0
        self.clear()

    def clear(self):
        self.a, self.b, self.c = [0.0] * LEVELS, [0.0] * LEVELS, [0.0] * LEVELS
        self.g0, self.g1, self.n = [0.0] * LEVELS, [0.0] * LEVELS, [0.0] * LEVELS

    def numbers(self):
        return self.slope, self.levels

    def observe(self, ps, kept):
        """`ps` the probabilities of the drafts verified, `kept` of them landed; called once a seat's round."""
        for j in range(min(len(ps), kept + 1)):
            x, k = max(min(logit(ps[j]), 40.0), -40.0), min(j, LEVELS - 1)
            pred = 1 / (1 + math.exp(-(self.slope * x + self.levels[k])))
            w, resid = pred * (1 - pred), (1.0 if j < kept else 0.0) - pred
            self.a[k] += w * x * x
            self.b[k] += w * x
            self.c[k] += w
            self.g0[k] += resid * x
            self.g1[k] += resid
            self.n[k] += 1
        self.calls += 1
        if self.calls % self.every == 0:
            self.refit()

    def refit(self, l2=1e-3, damping=50.0):
        c = [v + l2 for v in self.c]
        step = (sum(self.g0) - sum(b * g / ck for b, g, ck in zip(self.b, self.g1, c))) / (
            sum(self.a) + l2 - sum(b * b / ck for b, ck in zip(self.b, c)))
        total_n = sum(self.n)
        step *= total_n / (total_n + damping)
        if step != step or self.slope + step < 0:
            step = 0.0
        for k in range(LEVELS):
            bias = (self.g1[k] - self.b[k] * step) / c[k] * self.n[k] / (self.n[k] + damping)
            self.levels[k] += bias if bias == bias else 0.0
        self.slope += step
        self.clear()


class Refitted(Judged):
    """`Judged` by `kw['online']`'s numbers as they stand, which it tells of every verified draft."""

    def chain(self, rate, of_row):
        self.kw["numbers"] = self.kw["online"].numbers()
        return Judged.chain(self, rate, of_row)

    def done(self, n, scored):
        if n > 0:
            ch = self.r["chains"][self.i]
            self.kw["online"].observe([ch[j][1] for j in range(n)], min(n, self.lands))
        Judged.done(self, n, scored)


def total(groups, kw):
    outs = [P.play(g, kw) for g in groups]
    tokens = sum(o["tokens"] for o in outs)
    per = "/".join(str(round(100 * sum(o[k] for o in outs) / tokens)) for k in ("drafts", "kept", "levels"))
    return tokens / sum(o["tokens"] / o["tps"] for o in outs), per


def spread(groups, base, kw, draws=400):
    """The gain over `base` played on the same groups (in order: a refit carries across them, as one server's),
    and its deviation over the groups resampled."""
    import random
    pairs = []
    for g in groups:
        b, c = P.play(g, base), P.play(g, kw)
        pairs.append((b["tokens"], b["tokens"] / b["tps"], c["tokens"], c["tokens"] / c["tps"]))

    def gain(ps):
        return 100 * (sum(p[2] for p in ps) / sum(p[3] for p in ps) * sum(p[1] for p in ps) / sum(p[0] for p in ps) - 1)
    rng = random.Random(102)
    got = [gain([rng.choice(pairs) for _ in pairs]) for _ in range(draws)]
    mean = sum(got) / draws
    return gain(pairs), (sum((d - mean) ** 2 for d in got) / (draws - 1)) ** 0.5


def online(fitting, held, numbers):
    """The refit while serving (spec 102) against the numbers fixed a pack, on the held-out replies: `pack` starts
    at the pack's numbers, `cold` at vLLM's 0.5 and -0.2 with the kind's fitting replies played first (a server's
    earlier requests); `beside` as shipped, `alone` the product in the square's place with no trust (vLLM's form,
    the refit its trust); `alone fixed` that with the pack's numbers unchanged."""
    print("the refit while serving, held-out replies, % over the shipped rule (beside, the numbers fixed), "
          "the deviation over the groups resampled, the numbers at the end:")
    cold = (0.5, [-0.2] * LEVELS)
    for kind in sorted({r["name"].split("-")[0] for r in held}):
        of_kind = [r for r in held if r["name"].startswith(kind)]
        warm = [r for r in fitting if r["name"].startswith(kind)]
        for seats in (1, 2, 4):
            groups = [of_kind[i:i + seats] for i in range(0, len(of_kind) - seats + 1, seats)]
            base = dict(seat=Judged, numbers=numbers, beside=True)
            line = f"  {kind} x{seats}:"
            for name, start, beside in (("beside pack", numbers, True), ("beside cold", cold, True),
                                        ("alone pack", numbers, False), ("alone cold", cold, False),
                                        ("alone fixed", None, False)):
                if start is None:
                    kw = dict(seat=Judged, numbers=numbers, beside=False, untrusted=True)
                else:
                    est = Online(start, seats)
                    kw = dict(seat=Refitted, online=est, beside=beside, untrusted=True)
                    if start is cold:
                        for i in range(0, len(warm) - seats + 1, seats):
                            P.play(warm[i:i + seats], kw)
                gain, dev = spread(groups, base, kw)
                line += f"; {name} {gain:+.1f} ({dev:.1f})"
                if start is not None:
                    line += " [%.2f: %s]" % (est.slope, " ".join("%.2f" % b for b in est.levels))
            print(line)


def costs(path):
    """`Paying.Costs`' `row` and `level` from a load's drafted cases (`rounds`, no `chains`, names with
    `-drafted-`), as spec 061 fitted them: the windows' ms by their rows pooled, a line by least squares weighed
    by their counts, and the head's seconds over the levels it scored; both against the one-row window."""
    cases, cur = [], None
    for line in open(path, errors="replace"):
        if line.startswith("== "):
            cur = {"name": line[3:].strip(), "levels": 0, "rounds": 0, "windows": {}}
            cases.append(cur)
        elif cur is None or "-drafted-" not in cur["name"]:
            continue
        elif line.startswith("round at "):
            cur["rounds"] += 1
            cur["levels"] += len(line.split(";", 1)[1].split())
        elif " s of it drafting (" in line:
            cur["drafting"] = float(re.search(r"([\d.]+) s of it drafting", line)[1])
        if cur is not None and "ms a window after, by its rows:" in line:
            for rows, ms, n in re.findall(r" (\d+): ([\d.]+) \((\d+)\)", line.split("by its rows:")[1]):
                cur["windows"][int(rows)] = (float(ms), int(n))
    drafted = [c for c in cases if "-drafted-" in c["name"]]
    pooled = {}
    for c in drafted:
        for rows, (ms, n) in c["windows"].items():
            t, k = pooled.get(rows, (0.0, 0))
            pooled[rows] = (t + ms * n, k + n)
    sw = sx = sy = sxx = sxy = 0.0
    for rows, (t, k) in pooled.items():
        x, y = rows - 1, t / k
        sw, sx, sy, sxx, sxy = sw + k, sx + k * x, sy + k * y, sxx + k * x * x, sxy + k * x * y
    slope = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
    one = (sy - slope * sx) / sw
    level = sum(c["drafting"] for c in drafted) * 1e3 / sum(c["levels"] for c in drafted)
    print(f"{len(drafted)} drafted replies, {sum(c['rounds'] for c in drafted)} rounds; ms a window by its rows: "
          + " ".join(f"{r}: {t / k:.2f} ({k})" for r, (t, k) in sorted(pooled.items())))
    print(f"a window of one row {one:.2f} ms, a row more {slope:.2f}, a level of the head {level:.2f}: "
          f".row = {slope / one:.3f}, .level = {level / one:.3f}")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "costs":
        return costs(sys.argv[2])
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    names = [re.search(r"name=(\S+)", line)[1] for line in open(sys.argv[2])]
    replies = P.load(sys.argv[1], names)
    fitting = [r for r in replies if "-fit-" in r["name"]]
    held = [r for r in replies if "-held-" in r["name"]]
    numbers = fit(rows_of(fitting))
    print(f"{len(fitting)} fitting replies, {len(held)} held out")
    print(".slope = %.3f, .levels = .{ %s }" % (numbers[0], ", ".join("%.2f" % b for b in numbers[1])))
    for name, part in (("fitting", fitting), ("held out", held)):
        rows = rows_of(part)
        landed = sum(1 for *_, y in rows if y) / len(rows)
        head = judged([(p, y) for p, _, y in rows])
        fitted = judged([(landing(numbers, p, j), y) for p, j, y in rows])
        print(f"{name}: {len(rows)} rows, {landed:.3f} landed; AUC, log loss, calibration error: the head's p "
              "%.3f %.3f %.3f, fitted %.3f %.3f %.3f" % (head + fitted))
    print("the replay on the held-out replies, % over the shipped rule (drafts/kept/levels a 100 tokens):")
    for kind in sorted({r["name"].split("-")[0] for r in held}):
        of_kind = [r for r in held if r["name"].startswith(kind)]
        for seats in (1, 2, 4):
            groups = [of_kind[i:i + seats] for i in range(0, len(of_kind) - seats + 1, seats)]
            base, per = total(groups, {})
            line = f"  {kind} x{seats}: shipped {base:.1f} tok/s ({per})"
            for name, kw in (("beside", {"beside": True}), ("alone", {"beside": False}), ("oracle", {"oracle": True})):
                rule = kw if "oracle" in kw else dict(kw, seat=Judged, numbers=numbers)
                value, per = total(groups, rule)
                line += f"; {name} {100 * (value / base - 1):+.1f} ({per})"
            print(line)
    online(fitting, held, numbers)


if __name__ == "__main__":
    main()
