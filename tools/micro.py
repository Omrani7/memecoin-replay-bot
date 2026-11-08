"""Research: short-horizon structure around single large trades and over the 2-slot latency.

1. After a single sell (buy) of at least X% of the curve's real SOL, the return from two slots later.
2. Conditional on the price move over the latency (decision slot s to fill slot s + 2), the return after
   the fill: does a buy that would land after a jump do worse than one that lands after a dip?

    python tools/micro.py <kit dir>
"""

import bisect
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402
from smart_wallets import scan  # noqa: E402


def price_at(tr, slots, s):
    j = bisect.bisect_right(slots, s) - 1
    return tr[j][4] if j >= 0 else None


def main():
    ds = load_dataset(os.path.join(sys.argv[1], "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    created, trades = scan(ds)
    mid = ds.batches[len(ds.batches) // 2][0]
    H = (10, 30, 100, 300)
    for side in ("sell", "buy"):
        for frac in (0.05, 0.10, 0.20, 0.35):
            for min_rq in (3e9, 20e9):
                res = {0: defaultdict(list), 1: defaultdict(list)}
                for m, tr in trades.items():
                    slots = [t[0] for t in tr]
                    pre_rq = 0
                    last_ev = -10**9
                    for i, (s, buy, sol, u, p, rq, tok) in enumerate(tr):
                        before = rq + sol if not buy else rq - sol
                        if (buy == (side == "buy")) and before >= min_rq and sol >= frac * before and s - last_ev > 50:
                            last_ev = s
                            p0 = price_at(tr, slots, s + 2)
                            half = 0 if s < mid else 1
                            for h in H:
                                ph = price_at(tr, slots, s + 2 + h)
                                res[half][h].append(ph / p0 - 1)
                cells = []
                for h in H:
                    a, b = res[0][h], res[1][h]
                    if a and b:
                        cells.append(f"H{h}: {sum(a) / len(a):+.3f}/{sum(b) / len(b):+.3f}")
                n = (len(res[0][H[0]]), len(res[1][H[0]]))
                print(f"after single {side:4s} >= {frac:.0%} of real SOL (curve >= {min_rq / 1e9:.0f} SOL) n={n}: " + " ".join(cells))
    # latency-conditional
    print("\nreturn after fill, by the price move over the latency (s -> s+2), coins with >= 3 SOL real:")
    buckets = defaultdict(lambda: defaultdict(list))
    edges = (-1, -0.05, -0.01, -0.0001, 0.0001, 0.01, 0.05, 10)
    for m, tr in trades.items():
        slots = [t[0] for t in tr]
        last = -10**9
        for i, (s, buy, sol, u, p, rq, tok) in enumerate(tr):
            if rq < 3e9 or s - last < 10:
                continue
            if i + 1 < len(tr) and tr[i + 1][0] == s:
                continue  # decide on the slot's last trade
            last = s
            p2 = price_at(tr, slots, s + 2)
            lat = p2 / p - 1
            b = bisect.bisect_right(edges, lat) - 1
            for h in (30, 100, 300):
                buckets[b][h].append(price_at(tr, slots, s + 2 + h) / p2 - 1)
    for b in sorted(buckets):
        cells = " ".join(f"H{h} {sum(v) / len(v):+.4f}" for h, v in sorted(buckets[b].items()))
        print(f"  latency move {edges[b]:+.4f}..{edges[b + 1]:+.4f}: n {len(buckets[b][30]):6d} {cells}")


if __name__ == "__main__":
    main()
