"""Event study (research): coins that cross a real-SOL level while still tradable (not completed two slots
later). What share go on to complete, and how the price moves for those that do and those that don't.

    python tools/explore_grad.py <kit dir>
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset import load_dataset  # noqa: E402
from explore import coins_of, SOL  # noqa: E402


def main():
    ds = load_dataset(os.path.join(sys.argv[1], "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    coins = coins_of(ds)
    for th in (10, 20, 30, 40, 50, 60, 70, 75):
        rows = []
        for mint, c in coins.items():
            tr = c["trades"]
            for i, t in enumerate(tr):
                if t[3] >= th * SOL:
                    s = t[0]
                    if c["complete"] is not None and c["complete"] <= s + 2:
                        break  # not tradable: completes within the latency
                    # entry price: the last trade at or before s + 2
                    j = i
                    while j + 1 < len(tr) and tr[j + 1][0] <= s + 2:
                        j += 1
                    p0 = tr[j][4] / tr[j][5]
                    after = tr[j:]
                    pf = after[-1][4] / after[-1][5]
                    pmin = min(x[4] / x[5] for x in after)
                    pmax = max(x[4] / x[5] for x in after)
                    dur = (c["complete"] or after[-1][0]) - s
                    rows.append((mint, c["complete"] is not None, pf / p0, pmin / p0, pmax / p0, dur, s - c["created"]))
                    break
        if not rows:
            continue
        comp = [r for r in rows if r[1]]
        fail = [r for r in rows if not r[1]]
        print(f"== level {th} SOL: {len(rows)} tradable crossings, {len(comp)} complete ({len(comp) / len(rows):.0%})")
        if comp:
            print("   completers final/entry:", " ".join(f"{r[2]:.2f}" for r in sorted(comp, key=lambda r: r[2])),
                  "| min/entry:", " ".join(f"{r[3]:.2f}" for r in sorted(comp, key=lambda r: r[2])))
            print("   completers slots to complete:", sorted(r[5] for r in comp), " age at crossing:", sorted(r[6] for r in comp))
        if fail:
            fs = sorted(r[2] for r in fail)
            mx = sorted(r[4] for r in fail)
            print(f"   failures final/entry median {fs[len(fs) // 2]:.2f} mean {sum(fs) / len(fs):.2f}; "
                  f"max/entry median {mx[len(mx) // 2]:.2f}; share with max>1.2: {sum(1 for x in mx if x > 1.2) / len(mx):.0%}")


if __name__ == "__main__":
    main()
