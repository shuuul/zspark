#!/usr/bin/env python3
"""The pool's draft rule played on the host over chains recorded at every position (spec 078). Not runtime;
standard library only.

`zspark generate DIR 0 cases=FILE` (lines `N chains name=KIND-NAME ID..`) prints, for each reply, the MTP head's
chain of 15 drafts at every position. The head's chain from a position does not depend on the rounds before it,
so S replies sharing rounds under any rule follow from those lines: `mtp.Paying` with `Paying.Round` (the seats'
tokens and cost summed, the row at the round's rows) is played as `hall_engine.zig` plays it, and a round's ms
is spec 077's fit for the 3.05-bit pack (26.91 + 3.35 a seat + 2.53 a row; a chained pass 0.62 + 0.63 a seat).

  pool_policy.py RUN.out CASES [S..]   every rule below at S seats (1 2 4 unless given), the replies of a kind
                                       grouped S at a time in the cases' order: tok/s of the shipped rule, its
                                       drafts, kept and levels, and each other rule's % against it

Checked against a load (spec 078 WS-03): on the bench's four prose prompts at four streams it gives 550 tokens,
443 drafts, 257 kept and 302 levels where the server counted 546, 450, 257 and 307. Its tok/s is 8% under the
client's (the fit is of traced rounds), and a cap at two drafts gains 3.3% in it on those four where the
loads gave 2.7 to 3.1: a rule it prefers is still timed for real before it is taken. What it found: no form of the trust (its
range, the power, two bands in the chain's probability as vLLM's estimator, a ratio a depth as TensorFold's)
gains at one, two and four seats together; knowing what lands (`oracle`) is 18 to 27% away, which the head's
probabilities do not hold.
"""
import re
import sys

END = 248046
def replies(path):
    out, cur = [], None
    for line in open(path, errors='replace'):
        if 'greedy:' in line:
            cur = {'tokens': [int(t) for t in line.split('greedy:')[1].split()], 'chains': [], 'name': None}
            out.append(cur)
        m = re.match(r'chain at (\d+): \d+ drafts, \d+ kept;(.*)', line)
        if m and cur is not None:
            cur['chains'].append([(int(a.lstrip('/').split(':')[0]), float(a.split(':')[1])) for a in m[2].split()])
        m = re.search(r'name=(\S+)', line)
        if m and cur is not None and cur['name'] is None: cur['name'] = m[1]
    return out
def load(path, names):
    rs = replies(path)
    for r, n in zip(rs, names):
        r['name'] = n
        if END in r['tokens']:
            cut = r['tokens'].index(END) + 1
            r['tokens'], r['chains'] = r['tokens'][:cut], r['chains'][:cut]
    return rs

ROW, LEVEL, SHARED = 0.098, 0.043, 0.82
def band(rows): return 1.0 if rows <= 32 else 0.6 if rows <= 64 else 0.46
def round_ms(seats, rows): return 26.91 + 3.35 * seats + 2.53 * rows * band(rows)
def pass_ms(seats): return 0.62 + 0.63 * seats

class Seat:
    """`mtp.Paying` of one reply, with the knobs tried here."""
    def __init__(self, reply, kw):
        self.r, self.i, self.kw = reply, 0, kw
        self.tokens, self.cost, self.got, self.said = 2.0, 1 + ROW + LEVEL, 1.0, 1.0
        self.band = [[1.0, 1.0], [1.0, 1.0]]  # shape: got, said for chains at `split` and more, and under
        self.n_tokens = self.drafts = self.kept = self.rounds = self.levels = 0
    def live(self): return self.i < len(self.r['chains'])
    def trust(self, through):
        lo, hi = self.kw.get('range', (0.25, 1.0))
        if 'split' in self.kw:
            g, s = self.band[0 if through >= self.kw['split'] else 1]
            return min(max(g / s, self.kw.get('shape_range', (0.05, 1.5))[0]), self.kw.get('shape_range', (0.05, 1.5))[1])
        return min(max(self.got / self.said, lo), hi)
    def chain(self, rate, of_row):
        """drafts taken, levels scored (the first pass included), by the floors; `oracle` knows what lands."""
        ch = self.r['chains'][self.i]; t = self.r['tokens']
        most = min(len(ch), len(t) - self.i - 2)
        self.lands = 0
        while self.lands < most and ch[self.lands][0] == t[self.i + 1 + self.lands]: self.lands += 1
        if most <= 0: return 0, 0
        if self.kw.get('oracle'): return self.lands, max(self.lands, 1)
        cap = self.kw.get('depth', 15)
        n = scored = 0; run = 1.0; self.trail = []
        power = self.kw.get('power', 2.0)
        for j in range(min(most, cap)):
            scored += 1; run *= ch[j][1]; k = j + 1
            q = self.trust(run) * run ** power
            if k > 1 and q < rate * ROW * of_row: break
            n += 1; self.trail.append(run)
            if k >= 15 or q < rate * (ROW * of_row + LEVEL * self.kw.get('level_x', 1.0)): break
        return n, scored
    def done(self, n, scored):
        kept = min(n, self.lands); tokens = kept + 1
        self.tokens += 0.125 * (tokens - self.tokens)
        self.cost += 0.125 * (1 + ROW * n + LEVEL * scored - self.cost)
        if n > 1 and not self.kw.get('oracle'):
            power = self.kw.get('power', 2.0)
            said = sum(x ** power for x in self.trail[1:])
            self.got += 0.05 * (min(max(tokens - 2, 0), n - 1) - self.got); self.said += 0.05 * (said - self.said)
            if 'split' in self.kw:
                w = self.kw.get('shape_weight', 0.05)
                for b in (0, 1):
                    idx = [j for j in range(1, n) if (self.trail[j] >= self.kw['split']) == (b == 0)]
                    if not idx: continue
                    g = sum(1 for j in idx if kept > j); s = sum(self.trail[j] ** power for j in idx)
                    self.band[b][0] += w * (g - self.band[b][0]) * 1; self.band[b][1] += w * (s - self.band[b][1])
        self.n_tokens += tokens; self.drafts += n; self.kept += kept; self.rounds += 1; self.levels += max(scored - 1, 0)
        self.i += tokens

def play(replies, kw, priced=True):
    seats = [kw.get('seat', Seat)(r, kw) for r in replies]; ms = 0.0; rows_last = len(seats)
    while True:
        live = [s for s in seats if s.live()]
        if not live: break
        if priced and len(live) > 0:
            rate = sum(s.tokens for s in live) / (sum(s.cost for s in live) - SHARED * (len(live) - 1)); of_row = band(rows_last)
            prices = [(rate, of_row)] * len(live)
        else:
            prices = [(s.tokens / s.cost, 1.0) for s in live]
        chosen = [s.chain(*p) for s, p in zip(live, prices)]
        rows = sum(1 + n for n, _ in chosen); rows_last = rows
        ms += round_ms(len(live), rows)
        deepest = max(sc for _, sc in chosen)
        for l in range(2, deepest + 1): ms += pass_ms(sum(1 for _, sc in chosen if sc >= l))
        for s, (n, sc) in zip(live, chosen): s.done(n, sc)
    t = sum(s.n_tokens for s in seats)
    return dict(tps=1e3 * t / ms, tokens=t, drafts=sum(s.drafts for s in seats), kept=sum(s.kept for s in seats), rounds=sum(s.rounds for s in seats), levels=sum(s.levels for s in seats))

RULES = {
    "shipped": {},
    "priced alone": {"alone": True},
    "depth 1": {"depth": 1},
    "depth 2": {"depth": 2},
    "trust to 2": {"range": (0.25, 2.0)},
    "power 3": {"power": 3.0, "range": (0.25, 2.0)},
    "bands at 0.9": {"split": 0.9},
    "bands at 0.95": {"split": 0.95},
    "level x2": {"level_x": 2.0},
    "oracle": {"oracle": True},
}


def total(groups, kw):
    outs = [play(g, kw, not kw.get("alone")) for g in groups]
    tokens = sum(o["tokens"] for o in outs)
    return tokens / sum(o["tokens"] / o["tps"] for o in outs), tokens, outs


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    names = [re.search(r"name=(\S+)", line)[1] for line in open(sys.argv[2])]
    kinds = {}
    for r in load(sys.argv[1], names):
        kinds.setdefault(r["name"].split("-")[0], []).append(r)
    for seats in [int(a) for a in sys.argv[3:]] or [1, 2, 4]:
        for kind, rs in kinds.items():
            groups = [rs[i:i + seats] for i in range(0, len(rs) - seats + 1, seats)]
            if not groups:
                continue
            base, tokens, outs = total(groups, {})
            per = "/".join(str(round(100 * sum(o[k] for o in outs) / tokens)) for k in ("drafts", "kept", "levels"))
            print(f"{kind} x{seats}: {base:.1f} tok/s modelled, {per} drafts/kept/levels a 100 tokens; " + ", ".join(
                f"{name} {100 * (total(groups, kw)[0] / base - 1):+.1f}%" for name, kw in RULES.items() if kw))


if __name__ == "__main__":
    main()
