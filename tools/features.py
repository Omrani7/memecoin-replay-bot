"""Signal research: a table of decision points (coin, slot) with features observable at that slot and
forward price returns measured from the fill slot (slot + 2), then decile tables per feature.

Forward price at horizon H: the price after the coin's last trade at or before fill + H (its completion
price when it completed earlier). Returns are gross of fees (a round trip costs about 2.5% in curve fees).

    python tools/features.py <kit dir> [first|second|all]
"""

import bisect
import math
import os
import sys
from collections import defaultdict, deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402

SOL = 1_000_000_000
HORIZONS = (30, 100, 300, 1000)
SAMPLE_GAP = 10  # slots between two decision points of one coin


class Coin:
    def __init__(self, mint, slot, creator, rt0):
        self.mint = mint
        self.created = slot
        self.creator = creator
        self.trades = deque()  # recent (slot, is_buy, sol, user)
        self.slots = []  # price path: slot of each trade
        self.prices = []
        self.rq = 0
        self.vq = 0
        self.vt = 0
        self.ath = 0.0
        self.holders = defaultdict(int)  # wallet -> tokens bought minus sold
        self.complete = None
        self.last_point = -10**9
        self.n_trades = 0
        self.buyers = set()


def build_points(ds):
    coins = {}
    points = []
    for slot, evs in ds.batches:
        touched = []
        for e in evs:
            k = e["event"]
            mint = e["mint"]
            if k == "CreateEvent":
                coins[mint] = Coin(mint, slot, e["creator"], int(e["real_token_reserves"]))
                continue
            c = coins[mint]
            if k == "CompleteEvent":
                c.complete = slot
                continue
            sol = int(e["quote_amount"])
            tok = int(e["token_amount"])
            buy = e["is_buy"]
            user = e["user"]
            c.trades.append((slot, buy, sol, user))
            c.vq = int(e["virtual_quote_reserves"])
            c.vt = int(e["virtual_token_reserves"])
            c.rq = int(e["real_quote_reserves"])
            p = c.vq / c.vt
            c.slots.append(slot)
            c.prices.append(p)
            c.ath = max(c.ath, p)
            c.holders[user] += tok if buy else -tok
            c.n_trades += 1
            if buy:
                c.buyers.add(user)
            touched.append(c)
        seen = set()
        for c in touched:
            if c.mint in seen or c.complete is not None:
                continue
            seen.add(c.mint)
            if c.rq < 1 * SOL or slot - c.last_point < SAMPLE_GAP:
                continue
            c.last_point = slot
            while c.trades and c.trades[0][0] < slot - 400:
                c.trades.popleft()
            points.append((c, slot, feats(c, slot)))
    return coins, points


def feats(c, slot):
    def window(w):
        nb = ns = 0
        vb = vs = 0
        users = set()
        for s, b, sol, u in reversed(c.trades):
            if s < slot - w:
                break
            if b:
                nb += 1
                vb += sol
                users.add(u)
            else:
                ns += 1
                vs += sol
        return nb, ns, vb, vs, len(users)

    nb50, ns50, vb50, vs50, u50 = window(50)
    nb200, ns200, vb200, vs200, u200 = window(200)
    p = c.prices[-1]

    def ret(w):
        i = bisect.bisect_right(c.slots, slot - w) - 1
        if i < 0:
            return math.log(p / c.prices[0])
        return math.log(p / c.prices[i])

    circ = sum(v for v in c.holders.values() if v > 0)
    top = sorted((v for v in c.holders.values() if v > 0), reverse=True)
    top1 = top[0] / circ if circ > 0 and top else 0.0
    top5 = sum(top[:5]) / circ if circ > 0 and top else 0.0
    creator = max(0, c.holders.get(c.creator, 0)) / circ if circ > 0 else 0.0
    return {
        "rq": c.rq / SOL,
        "age": slot - c.created,
        "ntr": c.n_trades,
        "buyers": len(c.buyers),
        "net50": (vb50 - vs50) / SOL,
        "net200": (vb200 - vs200) / SOL,
        "buys50": nb50,
        "sells50": ns50,
        "bfrac200": nb200 / max(1, nb200 + ns200),
        "u200": u200,
        "r20": ret(20),
        "r100": ret(100),
        "r400": ret(400),
        "dd_ath": math.log(p / c.ath),
        "top1": top1,
        "top5": top5,
        "creator": creator,
    }


def forward(c, slot, h):
    fill = slot + 2
    i = bisect.bisect_right(c.slots, fill) - 1
    if i < 0:
        return None
    p0 = c.prices[i]
    end = fill + h
    j = bisect.bisect_right(c.slots, end) - 1
    return math.log(c.prices[j] / p0)


def decile_table(points, name, h, coins_weight=True):
    rows = [(f[name], forward(c, s, h)) for c, s, f in points]
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    out = []
    for d in range(10):
        part = rows[d * n // 10:(d + 1) * n // 10]
        if not part:
            continue
        lo, hi = part[0][0], part[-1][0]
        mean_simple = sum(math.exp(r[1]) - 1 for r in part) / len(part)
        out.append(f"{lo:9.3f}..{hi:9.3f} {mean_simple:+7.3f}")
    return out


def main():
    ds = load_dataset(os.path.join(sys.argv[1], "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    coins, points = build_points(ds)
    if which != "all":
        mid = ds.batches[len(ds.batches) // 2][0]
        points = [p for p in points if (p[1] < mid) == (which == "first")]
    print(f"{len(points)} decision points over {len({p[0].mint for p in points})} coins ({which})")
    for h in (100, 300):
        base = [forward(c, s, h) for c, s, f in points]
        base = [b for b in base if b is not None]
        print(f"\n### horizon {h} slots: mean simple return {sum(math.exp(b) - 1 for b in base) / len(base):+.4f} over {len(base)}")
        for name in points[0][2]:
            tab = decile_table(points, name, h)
            print(f"-- {name}")
            print("   " + " | ".join(tab[:5]))
            print("   " + " | ".join(tab[5:]))


if __name__ == "__main__":
    main()
