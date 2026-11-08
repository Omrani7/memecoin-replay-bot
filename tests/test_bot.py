"""Unit and integration tests: curve math, engine rules, the bot over the wire, determinism.

    python -m unittest discover -s tests          (from the repository root)

The integration tests read the kit's dev window when MAKEFUN_KIT points at an unpacked kit; they are
skipped otherwise.
"""

import io
import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "vendor"), os.path.join(ROOT, "tools")):
    if path not in sys.path:
        sys.path.insert(0, path)

import makefun_bot  # noqa: E402
import replay  # noqa: E402
from tradebot import strategy as strat_mod  # noqa: E402
from tradebot.market import Market, flow, top_holder_share  # noqa: E402

SOL = 1_000_000_000
MINT = "FdfDYUcHRar4ezebmC7dbYvFLjoFK7ceUy5WzHzvpump"
CREATOR = "E1CrPhVtP9A7XWyAZQacdG5LxZSENRDm2FVrmV3Ndz1Y"
KIT = os.environ.get("MAKEFUN_KIT")

HELLO = {"capital": {"lamports": "10000000000"},
         "caps": {"leverage": False, "max_open_positions": 5, "max_order_bps_of_real_reserve": 100,
                  "max_orders_per_batch": 16, "shorts": False},
         "fees": {"base_fee_lamports": "5000", "creator_bps_fallback": 30, "failed_tx_bps": 200,
                  "priority_fee_lamports": "100000", "protocol_bps_fallback": 95},
         "latency_slots": 2, "protocol": "agent-fun/replay/v2", "seed": "18384746552962903145", "type": "hello",
         "universe": {"exclude_cashback": True, "mayhem": "excluded", "program": "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P",
                      "quote": "SOL", "sample": {"hash": "sha256", "modulus": 16, "residue": 15}}}


def create(slot, mint=MINT):
    return {"event": "CreateEvent", "mint": mint, "creator": CREATOR, "slot": slot,
            "virtual_token_reserves": "1073000000000000", "virtual_quote_reserves": "30000000000",
            "virtual_sol_reserves": "30000000000", "real_token_reserves": "793100000000000",
            "quote_mint": replay.SOL_QUOTE, "is_mayhem_mode": False, "is_cashback_enabled": False}


class Curve:
    """A toy constant-product curve that emits TradeEvents with consistent post-trade reserves."""

    def __init__(self, mint=MINT):
        self.mint = mint
        self.vq, self.vt, self.rq, self.rt = 30 * SOL, 1_073_000_000_000_000, 0, 793_100_000_000_000

    def trade(self, slot, buy, sol, user="J72g3ptE4QZXEMBYcdRjp39LGYjRXwJvGyUYKeHH3Ew7"):
        k = self.vq * self.vt
        if buy:
            nvq = self.vq + sol
            tok = self.vt - k // nvq
            self.vq, self.vt, self.rq, self.rt = nvq, self.vt - tok, self.rq + sol, self.rt - tok
        else:
            nvq = self.vq - sol
            tok = k // nvq - self.vt
            self.vq, self.vt, self.rq, self.rt = nvq, self.vt + tok, self.rq - sol, self.rt + tok
        return {"event": "TradeEvent", "mint": self.mint, "slot": slot, "is_buy": buy, "quote_amount": str(sol),
                "token_amount": str(tok), "user": user, "virtual_quote_reserves": str(self.vq),
                "virtual_token_reserves": str(self.vt), "real_quote_reserves": str(self.rq),
                "real_token_reserves": str(self.rt), "fee_basis_points": "95", "creator_fee_basis_points": "30",
                "buyback_fee_basis_points": "0"}


class CurveMath(unittest.TestCase):
    def test_round_trip_without_other_trades_loses_only_fees(self):
        s = (30 * SOL, 1_073_000_000_000_000, 0, 793_100_000_000_000)
        q, t, fee, cfee, paid = replay.quote_buy_exact_in(SOL, s, (95, 30))
        self.assertEqual(paid, q + fee + cfee)
        self.assertLessEqual(paid, SOL)
        after = (s[0] + q, s[1] - t, s[2] + q, s[3] - t)
        sq, sfee, scfee, proceeds = replay.quote_sell(t, after, (95, 30))
        self.assertLessEqual(sq, q)
        self.assertAlmostEqual(proceeds / SOL, (1 - 0.0125) ** 2, delta=0.001)

    def test_offset_never_takes_reserves_below_zero(self):
        rec = (100 * SOL, 280_000_000_000_000, 70 * SOL, 0)
        vq, vt, rq, rt = replay.offset_reserves(rec, 5 * SOL, 10**13)
        self.assertEqual((vq, vt, rq, rt), rec)  # a completed curve (rt 0) is met as recorded

    def test_sell_value_matches_engine_close_quote(self):
        tokens = 10**12
        vq, vt, rq = 40 * SOL, 800_000_000_000_000, 10 * SOL
        q, fee, cfee, proceeds = replay.quote_sell(tokens, (vq, vt, rq, 0), (95, 30))
        self.assertEqual(strat_mod.sell_value(tokens, vq, vt, rq), proceeds)

    def test_fail_draw_is_deterministic_and_near_two_percent(self):
        key = replay.public_fail_key(HELLO["seed"])
        draws = [replay.tx_fails(key, seq, 0, 200) for seq in range(20000)]
        self.assertEqual(draws, [replay.tx_fails(key, seq, 0, 200) for seq in range(20000)])
        self.assertAlmostEqual(sum(draws) / len(draws), 0.02, delta=0.005)


class MarketState(unittest.TestCase):
    def test_flows_holders_and_levels(self):
        m = Market()
        c = Curve()
        m.apply({"slot": 1, "events": [create(1)]})
        m.apply({"slot": 2, "events": [c.trade(2, True, 2 * SOL, "A" * 32), c.trade(2, True, SOL, "B" * 32)]})
        touched = m.apply({"slot": 5, "events": [c.trade(5, False, SOL // 2, "A" * 32)]})
        coin = m.coins[MINT]
        self.assertEqual([x.mint for x in touched], [MINT])
        net, nb, ns, buyers, big = flow(coin, 5, 10)
        self.assertEqual((net, nb, ns, buyers, big), (2 * SOL + SOL - SOL // 2, 2, 1, 2, SOL // 2))
        self.assertGreater(top_holder_share(coin), 0.5)
        self.assertEqual(coin.rq, 2 * SOL + SOL - SOL // 2)
        self.assertEqual(coin.n_trades, 3)

    def test_unknown_mint_and_complete(self):
        m = Market()
        m.apply({"slot": 1, "events": [{"event": "TradeEvent", "mint": MINT}]})  # never created: ignored
        self.assertNotIn(MINT, m.coins)
        m.apply({"slot": 2, "events": [create(2), {"event": "CompleteEvent", "mint": MINT, "slot": 2}]})
        self.assertTrue(m.coins[MINT].complete)


class EngineRules(unittest.TestCase):
    """The in-process engine on a toy curve: latency, size cap, open-position cap, closes."""

    def toy(self, n_slots=40):
        c = Curve()
        batches = [(1, [create(1)])]
        for s in range(2, n_slots):
            batches.append((s, [c.trade(s, True, SOL // 10)]))
        ds = type("DS", (), {})()
        ds.batches, ds.end_slot, ds.slot_time = batches, n_slots + 5, {}
        return ds

    def test_buy_fills_two_slots_later_and_is_flattened(self):
        ds = self.toy()
        seen = []

        def bot(state, batch):
            seen.extend(batch["fills"])
            if batch["slot"] == 10:
                return [{"side": "buy", "mint": MINT, "quote_lamports": 5_000_000, "max_slippage_bps": 10000}]
            return []

        res = replay.run_strategy(ds, HELLO, bot)
        buys = [f for f in seen if f["side"] == "buy"]
        self.assertEqual(len(buys), 1)
        self.assertEqual((buys[0]["decided_slot"], buys[0]["slot"], buys[0]["status"]), (10, 12, "filled"))
        self.assertEqual(res["round_trips"], 1)  # flattened at the end
        self.assertEqual(res["trips"][0]["closed_slot"], ds.end_slot)

    def test_size_cap_rejects_oversized_buy(self):
        ds = self.toy()
        seen = []

        def bot(state, batch):
            seen.extend(batch["fills"])
            if batch["slot"] == 5:
                return [{"side": "buy", "mint": MINT, "quote_lamports": SOL, "max_slippage_bps": 10000}]
            return []

        replay.run_strategy(ds, HELLO, bot)
        self.assertEqual([f["reason"] for f in seen if f["side"] == "buy"], ["max_order_size"])


@unittest.skipUnless(KIT, "set MAKEFUN_KIT to an unpacked kit to run the dev-window tests")
class DevWindow(unittest.TestCase):
    def test_bot_over_the_wire_is_deterministic(self):
        """Run main.py as the grader does (stdin/stdout, PYTHONHASHSEED=0) on the first batches, twice."""
        from dataset import load_dataset
        ds = load_dataset(os.path.join(KIT, "dataset"))
        eng = replay.Engine(ds, HELLO)
        lines = [json.dumps(HELLO)]
        n = 0
        while n < 3000:
            b = eng.next_batch()
            if b is None:
                break
            lines.append(json.dumps(b))
            eng.submit([])  # the wire test only checks the bot answers every batch the same way
            n += 1
        lines.append(json.dumps({"type": "done", "seq": n, "reason": "end_of_data", "account": eng.account()}))
        feed = ("\n".join(lines) + "\n").encode()
        env = dict(os.environ, PYTHONHASHSEED="0")
        flags = ["-P", "-s"] if sys.version_info >= (3, 11) else ["-s"]  # -P is new in 3.11; the grader runs 3.13
        outs = [subprocess.run([sys.executable, *flags, os.path.join(ROOT, "main.py")], input=feed, env=env,
                               capture_output=True, check=True).stdout for _ in range(2)]
        self.assertEqual(outs[0], outs[1])
        replies = [json.loads(x) for x in outs[0].decode().splitlines()]
        self.assertEqual(len(replies), n)
        self.assertEqual([r["seq"] for r in replies], list(range(n)))
        for r in replies:
            self.assertLessEqual(len(r["orders"]), 16)

    def test_full_replay_meets_guardrails(self):
        from dataset import load_dataset
        ds = load_dataset(os.path.join(KIT, "dataset"))
        a = replay.run_strategy(ds, HELLO, strat_mod.Strategy().on_batch)
        b = replay.run_strategy(ds, HELLO, strat_mod.Strategy().on_batch)
        self.assertEqual(a["net_pnl"], b["net_pnl"])
        self.assertGreaterEqual(a["counted_mints"], 20)
        self.assertLessEqual(a["max_drawdown"], 0.5)


if __name__ == "__main__":
    unittest.main()
