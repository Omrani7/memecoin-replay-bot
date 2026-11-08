"""Evaluate strategy parameter sets on the dev window with the in-process engine (tools/replay.py).

    python tools/run.py <kit dir> [key=value ...]          one run, full window and both halves
    python tools/run.py <kit dir> --grid grid.json         every combination in the grid (JSON {key: [values]})

Values are Python literals (SOL amounts in lamports). Halves are fresh replays of the coins created in
each half of the window (tools/dataset.slice_dataset), so they are two separate samples.
"""

import ast
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "vendor")):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset import load_dataset, slice_dataset  # noqa: E402
from replay import run_strategy  # noqa: E402

SOL = 1_000_000_000
CACHE = os.path.join(ROOT, "..", "memedev", "cache", "dev.pkl")
HELLO = {"capital": {"lamports": "10000000000"},
         "caps": {"leverage": False, "max_open_positions": 5, "max_order_bps_of_real_reserve": 100, "max_orders_per_batch": 16, "shorts": False},
         "fees": {"base_fee_lamports": "5000", "creator_bps_fallback": 30, "failed_tx_bps": 200, "priority_fee_lamports": "100000", "protocol_bps_fallback": 95},
         "latency_slots": 2, "protocol": "agent-fun/replay/v2", "seed": "18384746552962903145", "type": "hello",
         "universe": {"exclude_cashback": True, "mayhem": "excluded", "program": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P", "quote": "SOL",
                      "sample": {"hash": "sha256", "modulus": 16, "residue": 15}}}

_DS = {}


STRESS = os.path.join(ROOT, "..", "memedev", "stress")


def datasets(kit):
    if not _DS:
        ds = load_dataset(os.path.join(kit, "dataset"), cache=CACHE)
        mid = ds.batches[len(ds.batches) // 2][0]
        _DS["full"] = ds
        _DS["h1"] = slice_dataset(ds, ds.batches[0][0], mid - 1)
        _DS["h2"] = slice_dataset(ds, mid, ds.end_slot)
    return _DS


def stress_dataset(name):
    """A kept block surrogate of the dev window (memedev/stress/<name>, built by grade-local --stress)."""
    key = "stress:" + name
    if key not in _DS:
        _DS[key] = load_dataset(os.path.join(STRESS, name), cache=os.path.join(STRESS, name + ".pkl"))
    return _DS[key]


def stress_names():
    if not os.path.isdir(STRESS):
        return []
    names = sorted(n for n in os.listdir(STRESS) if n.startswith("blocks-") and os.path.isdir(os.path.join(STRESS, n)))
    limit = int(os.environ.get("STRESS_MAX", "0"))
    return names[:limit] if limit else names


def evaluate(kit, params, parts=("full", "h1", "h2")):
    from tradebot.strategy import Strategy
    out = {}
    for part in parts:
        ds = stress_dataset(part) if part.startswith("blocks-") else datasets(kit)[part]
        strat = Strategy(params)
        res = run_strategy(ds, HELLO, strat.on_batch)
        out[part] = {k: res[k] for k in ("net_pnl", "max_drawdown", "fills", "round_trips", "counted_mints", "win_rate",
                                         "fees", "tx_fees", "failed", "rejected")}
        top = sorted(res["per_token"].values(), reverse=True)
        out[part]["top3"] = sum(top[:3])
        out[part]["trips"] = res["trips"]
    return out


def fmt(params, out):
    def one(r):
        return (f"{r['net_pnl'] / SOL:+8.3f} SOL dd {r['max_drawdown']:.3f} trips {r['round_trips']:4d} coins {r['counted_mints']:4d} "
                f"win {r['win_rate']:.2f} ex-top3 {(r['net_pnl'] - r['top3']) / SOL:+7.3f}")
    s = " ".join(f"{k}={v}" for k, v in params.items())
    return f"{s}\n   full {one(out['full'])}\n   h1   {one(out['h1'])}\n   h2   {one(out['h2'])}"


def _job(args):
    kit, params = args[:2]
    parts = args[2] if len(args) > 2 else ("full", "h1", "h2")
    out = evaluate(kit, params, parts)
    for part in out.values():
        part.pop("trips", None)
    return params, out


def stress_report(kit, params, workers=10):
    """Net PnL of one parameter set on every kept stress copy (in parallel), plus dev."""
    names = stress_names()
    chunks = [names[i::workers] for i in range(workers)]
    res = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _, out in ex.map(_job, [(kit, params, tuple(c)) for c in chunks if c]):
            res.update(out)
    pnls = sorted(r["net_pnl"] / SOL for r in res.values())
    n = len(pnls)
    return {"n": n, "median": pnls[n // 2] if n else None, "mean": sum(pnls) / n if n else None,
            "p10": pnls[n // 10] if n else None, "p90": pnls[(9 * n) // 10] if n else None,
            "positive": sum(1 for x in pnls if x > 0), "dd_max": max((r["max_drawdown"] for r in res.values()), default=None),
            "coins_min": min((r["counted_mints"] for r in res.values()), default=None), "runs": res}


def main():
    kit = sys.argv[1]
    args = sys.argv[2:]
    if args and args[0] == "--stress":
        params = {k: eval(v, {"SOL": SOL}) for k, v in (a.split("=", 1) for a in args[1:])}
        rep = stress_report(kit, params)
        print(" ".join(f"{k}={v}" for k, v in params.items()))
        print(f"   stress n {rep['n']} median {rep['median']:+.3f} mean {rep['mean']:+.3f} p10 {rep['p10']:+.3f} p90 {rep['p90']:+.3f} "
              f"positive {rep['positive']}/{rep['n']} worst dd {rep['dd_max']:.3f} min coins {rep['coins_min']}")
        return
    if args and args[0] == "--sgrid":
        grid = json.load(open(args[1]))
        base = {k: eval(v, {"SOL": SOL}) for k, v in (a.split("=", 1) for a in args[2:])}
        keys = list(grid)
        log = os.path.join(ROOT, "..", "memedev", "sgrid.jsonl")
        for vals in itertools.product(*(grid[k] for k in keys)):
            params = dict(base, **dict(zip(keys, vals)))
            rep = stress_report(kit, params)
            runs = rep.pop("runs")
            with open(log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"params": params, "stress": rep,
                                    "pnls": {k: v["net_pnl"] for k, v in runs.items()},
                                    "dds": {k: v["max_drawdown"] for k, v in runs.items()}}) + "\n")
            print(" ".join(f"{k}={params[k]}" for k in keys),
                  f"| median {rep['median']:+.3f} mean {rep['mean']:+.3f} pos {rep['positive']}/{rep['n']} dd {rep['dd_max']:.2f}", flush=True)
        return
    if args and args[0] == "--grid":
        grid = json.load(open(args[1]))
        base = {}
        if len(args) > 2:
            base = {k: eval(v, {"SOL": SOL}) for k, v in (a.split("=", 1) for a in args[2:])}
        keys = list(grid)
        combos = [dict(base, **dict(zip(keys, vals))) for vals in itertools.product(*(grid[k] for k in keys))]
        results = []
        with ProcessPoolExecutor(max_workers=10) as ex:
            for params, out in ex.map(_job, [(kit, c) for c in combos]):
                results.append((params, out))
                print(fmt({k: params[k] for k in keys}, out), flush=True)
        log = os.path.join(ROOT, "..", "memedev", "experiments.jsonl")
        with open(log, "a", encoding="utf-8") as f:
            for params, out in results:
                f.write(json.dumps({"params": params, "out": out}) + "\n")
        return
    params = {k: eval(v, {"SOL": SOL}) for k, v in (a.split("=", 1) for a in args)}
    out = evaluate(kit, params)
    print(fmt(params, out))
    reasons = {}
    for t in sorted(out["full"]["trips"], key=lambda t: t["pnl_lamports"])[:5]:
        print("   worst", t["mint"][:8], f"{t['pnl_lamports'] / SOL:+.3f}", "bought", f"{t['bought_lamports'] / SOL:.3f}",
              "held", t["closed_slot"] - t["opened_slot"])
    for t in sorted(out["full"]["trips"], key=lambda t: -t["pnl_lamports"])[:5]:
        print("   best ", t["mint"][:8], f"{t['pnl_lamports'] / SOL:+.3f}", "bought", f"{t['bought_lamports'] / SOL:.3f}",
              "held", t["closed_slot"] - t["opened_slot"])


if __name__ == "__main__":
    main()
