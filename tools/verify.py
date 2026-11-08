"""Check tools/replay.py against the official grader: replay a saved action log (grade-local --save) and
compare net PnL, fills, round trips and drawdown with the grader's result.json.

    python tools/verify.py <kit dir> <save dir> [<save dir>...]
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from dataset import load_dataset  # noqa: E402
from replay import Engine  # noqa: E402


def replay_log(ds, log):
    eng = Engine(ds, log["hello"])
    i = 0
    while True:
        b = eng.next_batch()
        if b is None:
            break
        eng.submit(log["replies"][i]["orders"])
        i += 1
    return eng.finish()


def main():
    kit = sys.argv[1]
    ds = load_dataset(os.path.join(kit, "dataset"), cache=os.path.join(HERE, "..", "..", "memedev", "cache", "dev.pkl"))
    print(f"{len(ds.batches)} batches, {ds.n_events} events, end_slot {ds.end_slot}")
    ok = True
    for save in sys.argv[2:]:
        log = json.load(open(os.path.join(save, "action-log.json"), encoding="utf-8"))
        res = json.load(open(os.path.join(save, "result.json"), encoding="utf-8"))
        m = res["performance"]["metrics"]
        mine = replay_log(ds, log)
        pairs = [
            ("net_pnl", int(m["net_pnl_lamports"]), mine["net_pnl"]),
            ("fills", m["fills"], mine["fills"]),
            ("round_trips", m["round_trips"], mine["round_trips"]),
            ("counted_mints", m["constraints"]["round_trips"]["counted_mints"], mine["counted_mints"]),
            ("failed", m["failed_txs"], mine["failed"]),
            ("rejected", m["rejected"], mine["rejected"]),
            ("fees", int(m["fees_lamports"]), mine["fees"]),
            ("tx_fees", int(m["tx_fees_lamports"]), mine["tx_fees"]),
            ("max_drawdown", float(m["max_drawdown"]), round(mine["max_drawdown"], 9)),
        ]
        for name, official, ours in pairs:
            same = abs(official - ours) < 1e-8 if isinstance(official, float) else official == ours
            ok &= same
            print(f"{os.path.basename(os.path.dirname(save)):24s} {name:14s} official {official!s:>16s} port {ours!s:>16s} {'ok' if same else 'MISMATCH'}")
    print("ALL MATCH" if ok else "MISMATCHES FOUND")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
