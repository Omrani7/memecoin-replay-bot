"""Research: do some wallets trade persistently well, and does following them (two slots late) pay?

Wallet PnL per coin = SOL out - SOL in + tokens left at the coin's last price. "Smart" wallets are
picked on the coins created in the first half of the window only; the forward returns of their later
buys are then measured on the coins created in the second half (and vice versa).

    python tools/smart_wallets.py <kit dir>
"""

import bisect
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402

SOL = 1e9


def scan(ds):
    created = {}
    trades = defaultdict(list)  # mint -> [(slot, buy, sol, user, price, rq)]
    for slot, evs in ds.batches:
        for e in evs:
            if e["event"] == "CreateEvent":
                created[e["mint"]] = (slot, e["creator"])
            elif e["event"] == "TradeEvent":
                trades[e["mint"]].append((slot, e["is_buy"], int(e["quote_amount"]), e["user"],
                                          int(e["virtual_quote_reserves"]) / int(e["virtual_token_reserves"]),
                                          int(e["real_quote_reserves"]), int(e["token_amount"])))
    return created, trades


def wallet_pnl(trades, mints):
    out = defaultdict(lambda: [0.0, 0])
    for m in mints:
        tr = trades.get(m, [])
        if not tr:
            continue
        last = tr[-1][4]
        flow = defaultdict(float)
        tok = defaultdict(int)
        for s, buy, sol, u, p, rq, t in tr:
            if buy:
                flow[u] -= sol
                tok[u] += t
            else:
                flow[u] += sol
                tok[u] -= t
        for u in flow:
            out[u][0] += (flow[u] + tok[u] * last) / SOL
            out[u][1] += 1
    return out


def follow_returns(trades, mints, smart, created, horizons=(20, 60, 200, 600), min_rq=0, max_age=None):
    rows = defaultdict(list)
    for m in mints:
        tr = trades.get(m, [])
        slots = [t[0] for t in tr]
        seen_users = set()
        for i, (s, buy, sol, u, p, rq, t) in enumerate(tr):
            if not buy or u not in smart or (u, m) in seen_users or rq < min_rq:
                continue
            if max_age is not None and s - created[m][0] > max_age:
                continue
            seen_users.add((u, m))
            j = bisect.bisect_right(slots, s + 2) - 1
            p0 = tr[j][4]
            for h in horizons:
                k = bisect.bisect_right(slots, s + 2 + h) - 1
                rows[h].append(tr[k][4] / p0 - 1)
    return rows


def main():
    ds = load_dataset(os.path.join(sys.argv[1], "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    created, trades = scan(ds)
    mid = ds.batches[len(ds.batches) // 2][0]
    halves = ([m for m, (s, _) in created.items() if s < mid], [m for m, (s, _) in created.items() if s >= mid])
    for a, b, name in ((0, 1, "learn h1 -> test h2"), (1, 0, "learn h2 -> test h1")):
        pnl = wallet_pnl(trades, halves[a])
        for thr, mincoins in ((0.5, 3), (1.0, 3), (2.0, 5)):
            smart = {u for u, (v, n) in pnl.items() if v >= thr and n >= mincoins}
            for label, kw in (("any", {}), ("rq>=5", {"min_rq": 5 * 10**9}), ("age<=50", {"max_age": 50}), ("rq>=20", {"min_rq": 20 * 10**9})):
                rows = follow_returns(trades, halves[b], smart, created, **kw)
                cells = []
                for h, rs in sorted(rows.items()):
                    if rs:
                        rs2 = sorted(rs)
                        cells.append(f"H{h}: n {len(rs)} mean {sum(rs) / len(rs):+.3f} med {rs2[len(rs2) // 2]:+.3f}")
                print(f"{name} smart>= {thr} SOL ({len(smart)} wallets) {label:8s} " + " | ".join(cells))
        base = follow_returns(trades, halves[b], {u for m in halves[b] for (_, _, _, u, _, _, _) in trades.get(m, [])}, created)
        print(f"{name} ALL buyers (first buy per wallet per coin): " + " | ".join(
            f"H{h}: n {len(rs)} mean {sum(rs) / len(rs):+.3f}" for h, rs in sorted(base.items())))


if __name__ == "__main__":
    main()
