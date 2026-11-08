"""Research: can a combination of observable features pick trades that beat fees out of sample?

Builds decision points (a coin with >= MIN_RQ real SOL, at most one point every GAP slots), features
from the past only (flows, returns, holder concentration, online wallet reputation), and a label: the
net return of a TP/SL/time trade entered at the fill slot (decision + 2) on the recorded price path,
exits filled two slots after their trigger, both curve fees and tx fees charged. Then a gradient
boosting model is fit on one half of the window and scored on the other (both directions).

    python tools/ml.py <kit dir> [tp] [sl] [max_hold]
"""

import bisect
import math
import os
import sys
from collections import defaultdict, deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402

SOL = 1e9
MIN_RQ = 3 * SOL
GAP = 20
FEE = 0.0125
STAKE = 0.5 * SOL
TX = 2 * 105_000


class Wallets:
    """Online realized PnL per wallet: average-cost basis per (wallet, coin)."""

    def __init__(self):
        self.pos = {}  # (u, m) -> [tokens, cost]
        self.realized = defaultdict(float)
        self.coins = defaultdict(int)

    def trade(self, u, m, buy, sol, tok):
        k = (u, m)
        p = self.pos.get(k)
        if p is None:
            p = [0, 0.0]
            self.pos[k] = p
            self.coins[u] += 1
        if buy:
            p[0] += tok
            p[1] += sol
        elif p[0] > 0:
            frac = min(1.0, tok / p[0])
            cost = p[1] * frac
            self.realized[u] += sol - cost
            p[0] = max(0, p[0] - tok)
            p[1] -= cost


def build(ds):
    created = {}
    tr = defaultdict(list)  # mint -> list of (slot, price, rq)
    complete = {}
    tape = defaultdict(deque)
    holders = defaultdict(dict)
    ath = {}
    creator = {}
    wallets = Wallets()
    last_point = {}
    points = []  # (mint, slot, feature dict)
    n_trades = defaultdict(int)
    buyers = defaultdict(set)
    for slot, evs in ds.batches:
        touched = []
        for e in evs:
            k = e["event"]
            m = e["mint"]
            if k == "CreateEvent":
                created[m] = slot
                creator[m] = e["creator"]
                ath[m] = 0.0
                continue
            if k == "CompleteEvent":
                complete[m] = slot
                continue
            sol = int(e["quote_amount"])
            tok = int(e["token_amount"])
            buy = e["is_buy"]
            u = e["user"]
            p = int(e["virtual_quote_reserves"]) / int(e["virtual_token_reserves"])
            rq = int(e["real_quote_reserves"])
            tr[m].append((slot, p, rq))
            tape[m].append((slot, buy, sol, u))
            h = holders[m]
            h[u] = h.get(u, 0) + (tok if buy else -tok)
            ath[m] = max(ath[m], p)
            wallets.trade(u, m, buy, sol, tok)
            n_trades[m] += 1
            if buy:
                buyers[m].add(u)
            touched.append(m)
        for m in dict.fromkeys(touched):
            if m in complete:
                continue
            t = tr[m]
            rq = t[-1][2]
            if rq < MIN_RQ or slot - last_point.get(m, -10**9) < GAP:
                continue
            last_point[m] = slot
            q = tape[m]
            while q and q[0][0] < slot - 600:
                q.popleft()
            points.append((m, slot, features(m, slot, t, q, holders[m], ath[m], created[m], creator[m], wallets, n_trades[m], len(buyers[m]))))
    return points, tr, complete, created


def features(m, slot, t, q, holders, ath, created, creator, wallets, ntr, nbuyers):
    p = t[-1][1]
    rq = t[-1][2]
    slots = None

    def wnd(w):
        net = vb = vs = 0
        nb = ns = 0
        users = set()
        big = 0
        smart_b = smart_s = 0.0
        for s, buy, sol, u in reversed(q):
            if s < slot - w:
                break
            rep = wallets.realized.get(u, 0.0)
            if buy:
                vb += sol
                nb += 1
                users.add(u)
                if rep > 0.5 * SOL:
                    smart_b += sol
            else:
                vs += sol
                ns += 1
                big = max(big, sol)
                if rep > 0.5 * SOL:
                    smart_s += sol
        return vb, vs, nb, ns, len(users), big, smart_b, smart_s

    f = {}
    for w in (20, 100, 400):
        vb, vs, nb, ns, nu, big, sb, ss = wnd(w)
        f[f"net{w}"] = (vb - vs) / SOL
        f[f"bshare{w}"] = vb / (vb + vs) if vb + vs > 0 else 0.5
        f[f"ntr{w}"] = nb + ns
        f[f"users{w}"] = nu
        f[f"big{w}"] = big / max(rq, 1)
        f[f"smartb{w}"] = sb / (vb + 1)
        f[f"smarts{w}"] = ss / (vs + 1)
    sl = [x[0] for x in t]
    for w in (20, 100, 400):
        i = bisect.bisect_right(sl, slot - w) - 1
        f[f"r{w}"] = math.log(p / t[i][1]) if i >= 0 else math.log(p / t[0][1])
        f[f"rqg{w}"] = rq / max(t[i][2], 1) if i >= 0 else rq / max(t[0][2], 1)
    f["rq"] = rq / SOL
    f["age"] = math.log1p(slot - created)
    f["dd_ath"] = math.log(p / ath)
    f["ntr"] = math.log1p(ntr)
    f["nbuyers"] = math.log1p(nbuyers)
    pos = sorted((v for v in holders.values() if v > 0), reverse=True)
    circ = sum(pos)
    f["top1"] = pos[0] / circ if circ else 0
    f["top5"] = sum(pos[:5]) / circ if circ else 0
    f["nholders"] = math.log1p(len(pos))
    f["creator"] = max(0, holders.get(creator, 0)) / circ if circ else 0
    # reputation of current holders (token weighted)
    rep_tok = sum(v for u, v in holders.items() if v > 0 and wallets.realized.get(u, 0.0) > 0.5 * SOL)
    f["smart_hold"] = rep_tok / circ if circ else 0
    return f


def label(t, complete, m, slot, tp, sl, hold):
    slots = [x[0] for x in t]
    i = bisect.bisect_right(slots, slot + 2) - 1
    p0 = t[i][1]
    end = slot + 2 + hold
    comp = complete.get(m)
    exit_p = None
    for j in range(i + 1, len(t)):
        s, p, rq = t[j]
        if s > end:
            break
        r = p / p0 - 1
        if r >= tp or r <= -sl:
            k = bisect.bisect_right(slots, s + 2) - 1
            exit_p = t[k][1]
            if comp is not None and comp <= s + 2:
                exit_p = t[-1][1]
            break
    if exit_p is None:
        if comp is not None and comp <= end:
            exit_p = t[-1][1]
        else:
            k = bisect.bisect_right(slots, end) - 1
            exit_p = t[k][1]
    return exit_p / p0 * (1 - FEE) ** 2 - 1 - TX / STAKE


def main():
    kit = sys.argv[1]
    tp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.2
    slp = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
    hold = int(sys.argv[4]) if len(sys.argv) > 4 else 600
    ds = load_dataset(os.path.join(kit, "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    points, tr, complete, created = build(ds)
    names = list(points[0][2])
    X = np.array([[f[n] for n in names] for _, _, f in points])
    y = np.array([label(tr[m], complete, m, s, tp, slp, hold) for m, s, _ in points])
    mint = [m for m, _, _ in points]
    slot = np.array([s for _, s, _ in points])
    mid = ds.batches[len(ds.batches) // 2][0]
    cr = np.array([created[m] for m in mint])
    half = (cr >= mid).astype(int)
    print(f"{len(points)} points, {len(set(mint))} coins; tp {tp} sl {slp} hold {hold}; mean label {y.mean():+.4f}")
    from sklearn.ensemble import HistGradientBoostingRegressor
    for a in (0, 1):
        tr_idx = half == a
        te_idx = half == 1 - a
        model = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=200, l2_regularization=1.0)
        model.fit(X[tr_idx], np.clip(y[tr_idx], -1, 1.5))
        pred = model.predict(X[te_idx])
        yt = y[te_idx]
        order = np.argsort(-pred)
        print(f"train half {a} -> test half {1 - a}: test mean {yt.mean():+.4f}")
        for q in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
            k = max(1, int(len(order) * q))
            sel = order[:k]
            mints_sel = len({mint[i] for i in np.nonzero(te_idx)[0][sel]})
            print(f"   top {q:4.0%} by prediction: n {k:5d} ({mints_sel} coins) mean {yt[sel].mean():+.4f} win {np.mean(yt[sel] > 0):.2f} pred {pred[sel].mean():+.4f}")
        # feature importances by permutation would be slow; print correlation of pred with features
    corr = [(n, np.corrcoef(X[:, i], y)[0, 1]) for i, n in enumerate(names)]
    corr.sort(key=lambda r: -abs(r[1]) if not math.isnan(r[1]) else 0)
    print("feature/label correlations:", " ".join(f"{n} {c:+.3f}" for n, c in corr[:15]))


if __name__ == "__main__":
    main()
