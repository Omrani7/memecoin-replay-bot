"""Coin lifecycle statistics of a replay window (research): how coins grow, how many complete, how
prices move after observable states. Prints tables; writes nothing.

    python tools/explore.py <kit dir>
"""

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402

SOL = 1_000_000_000


def coins_of(ds):
    coins = {}
    for slot, evs in ds.batches:
        for e in evs:
            k = e["event"]
            mint = e["mint"]
            if k == "CreateEvent":
                coins[mint] = {"created": slot, "creator": e["creator"], "trades": [], "complete": None}
            elif k == "TradeEvent":
                c = coins[mint]
                c["trades"].append((slot, e["is_buy"], int(e["quote_amount"]), int(e["real_quote_reserves"]),
                                    int(e["virtual_quote_reserves"]), int(e["virtual_token_reserves"]), e["user"]))
            else:
                coins[mint]["complete"] = slot
    return coins


def price(vq, vt):
    return vq / vt


def main():
    ds = load_dataset(os.path.join(sys.argv[1], "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    coins = coins_of(ds)
    n = len(coins)
    done = [c for c in coins.values() if c["complete"] is not None]
    print(f"coins {n}, completed {len(done)}")
    ntr = sorted(len(c["trades"]) for c in coins.values())
    print("trades per coin: median", ntr[n // 2], "p90", ntr[int(n * 0.9)], "p99", ntr[int(n * 0.99)], "max", ntr[-1])
    maxrq = sorted(max((t[3] for t in c["trades"]), default=0) / SOL for c in coins.values())
    for th in (0.5, 1, 2, 5, 10, 20, 40, 60, 80):
        k = sum(1 for x in maxrq if x >= th)
        print(f"  coins reaching real SOL >= {th:>4}: {k:5d} ({k / n:.1%})")
    # P(complete | reached X SOL real)
    print("P(complete | reached level):")
    for th in (2, 5, 10, 20, 30, 40, 50, 60, 70):
        reached = [c for c in coins.values() if any(t[3] >= th * SOL for t in c["trades"])]
        comp = [c for c in reached if c["complete"] is not None]
        print(f"  >= {th:>3} SOL: {len(reached):4d} coins, {len(comp):3d} complete ({len(comp) / max(1, len(reached)):.1%})")
    # Lifetime in slots of active trading
    lif = sorted((c["trades"][-1][0] - c["created"]) if c["trades"] else 0 for c in coins.values())
    print("active life (slots): median", lif[n // 2], "p75", lif[int(n * .75)], "p90", lif[int(n * 0.9)])
    if done:
        tt = sorted(c["complete"] - c["created"] for c in done)
        print("slots to complete: ", tt)
    # Forward return of the price from the first time a coin reaches X SOL real, to its max / final
    print("forward price ratio after first reaching level (final = last trade or completion):")
    for th in (1, 3, 5, 10, 20, 30, 45, 60):
        rs = []
        mx = []
        for c in coins.values():
            tr = c["trades"]
            for i, t in enumerate(tr):
                if t[3] >= th * SOL:
                    p0 = price(t[4], t[5])
                    pf = price(tr[-1][4], tr[-1][5])
                    pm = max(price(x[4], x[5]) for x in tr[i:])
                    rs.append(pf / p0)
                    mx.append(pm / p0)
                    break
        if rs:
            rs.sort()
            m = len(rs)
            mean = sum(rs) / m
            print(f"  >= {th:>3} SOL: {m:4d} coins, final/entry mean {mean:.3f} median {rs[m // 2]:.3f}, "
                  f"share >1.1 {sum(1 for r in rs if r > 1.1) / m:.1%}, mean max/entry {sum(mx) / m:.2f}")


if __name__ == "__main__":
    main()
