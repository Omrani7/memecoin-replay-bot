"""Trading logic: entries, position building, exits, risk limits and guardrail probes.

Strategy (see STRATEGY.md): buy bonding curves that are already deep (50 SOL or more of real reserve,
near their high), half a SOL each built within the per-reply cap, and sell on a small take-profit, a
wide stop or a time limit; curves that complete are closed by the replay at the completion price.
Small guardrail probes make sure enough distinct coins carry a qualifying round trip. The other
switches (trailing stop, flow exit, progress schedule, pyramiding, spike exit) stay off by default:
they were tried in research (tools/) and did not help.

All decisions are pure functions of the batches received so far.
"""

from .market import SOL, Market, flow, top_holder_share

BPS = 10_000

DEFAULT_PARAMS = {
    # --- entry filter
    "entry_rq_min": 50 * SOL,      # real SOL in the curve at decision time, lower bound
    "entry_rq_max": 83 * SOL,      # ...upper bound (a curve completes near 85 SOL)
    "min_age": 0,                  # slots since the CreateEvent
    "max_age": 10**9,
    "flow_window": 150,            # slots of tape the entry flow is read over
    "min_net_flow": 0,             # net SOL bought over flow_window
    "min_buyers": 5,               # distinct buying wallets over flow_window
    "max_dd_from_ath": 0.20,       # price at most this far under its all-time high
    "max_top_holder": 1.0,         # largest wallet's share of tokens held outside the curve
    "max_big_sell_frac": 1.0,      # skip when a single recent sell took more than this share of real SOL
    "fresh_level_slots": 10**9,    # enter only within this many slots of first reaching entry_rq_min
    "max_entries_per_coin": 1,
    "cooldown_slots": 3000,
    # --- sizing
    "stake": SOL // 2,             # target SOL per position (built over several batches)
    "min_buy": 10_000_000,         # smallest buy order worth sending (also the guardrail floor)
    "cap_use": 0.98,               # share of the per-reply size cap a buy may use
    "build_slots": 60,             # keep adding to a new position only this long after its first buy
    "max_positions": 5,
    "reserve": 1 * SOL,            # cash never committed to entries
    "buy_slip_bps": 500,
    # --- exits (fractions of the position's cost)
    "tp": 0.08,
    "sl": 0.30,
    "trail_arm": 10.0,             # trailing stop off (arms only at +1000%)
    "trail": 0.10,
    "max_hold": 3000,
    "exit_flow_window": 50,
    "exit_flow": -10**15,          # flow exit off (it cut winners on deep curves)
    "sell_slip_bps": BPS,
    "spike_frac": 1.0,             # sell when one buy of at least this share of real SOL lands (1.0 = off)
    "spike_min_profit": 0.0,       # ...and the position is at least this far in profit
    # --- progress schedule: exit when the coin's real SOL lags it ([age slots, min real SOL], ascending)
    "progress": [],
    # --- pyramiding: add when real SOL has grown by this factor since the last buy (0 = off)
    "pyramid_step": 0.0,
    "pyramid_frac": 0.0,           # position may grow to this share of the curve's real SOL
    "pyramid_max": 2 * SOL,        # ...but never past this much cost
    # --- guardrail probes
    "probe_target": 26,            # coins with a qualifying round trip to reach through probes
    "probe_lamports": 12_000_000,
    "probe_min_rq": 5 * SOL,
    "probe_every": 400,            # slots between two probe entries
}

FEE_GUESS_BPS = 125  # protocol 95 + creator 30, the usual curve fee per side


def sell_value(tokens, vq, vt, rq):
    """What selling `tokens` against these reserves pays after the usual curve fees (tx fees not deducted)."""
    if tokens <= 0:
        return 0
    q = tokens * vq // (vt + tokens)
    if q > rq:
        q = rq
    return q - (q * FEE_GUESS_BPS + BPS - 1) // BPS


class Position:
    __slots__ = ("mint", "first_slot", "peak_value", "probe", "last_buy_slot", "exit_sent_slot", "entries",
                 "last_buy_rq", "last_buy_price")

    def __init__(self, mint, slot, probe, rq=0, price=0.0):
        self.mint = mint
        self.first_slot = slot
        self.peak_value = 0
        self.probe = probe
        self.last_buy_slot = slot
        self.exit_sent_slot = None
        self.entries = 1
        self.last_buy_rq = rq
        self.last_buy_price = price


class Strategy:
    def __init__(self, params=None):
        self.p = dict(DEFAULT_PARAMS)
        if params:
            self.p.update(params)
        for k, v in DEFAULT_PARAMS.items():
            if isinstance(v, int) and not isinstance(v, bool):
                self.p[k] = int(self.p[k])  # amounts and slot counts are always integers on the wire
        self.p["progress"] = [(int(a), int(b)) for a, b in self.p["progress"]]
        self.market = Market()
        self.book = {}          # mint -> Position (ours, from first buy until flat)
        self.entries = {}       # mint -> entries made
        self.last_exit = {}     # mint -> slot of last exit decision
        self.qualified = set()  # mints with a round trip of at least min_buy closed (probe accounting)
        self.bought = {}        # mint -> lamports bought in the open round trip
        self.last_probe = -10**9
        self.marks = {}

    # ------------------------------------------------------------ helpers
    def _fill_bookkeeping(self, state, batch):
        for f in batch["fills"]:
            mint = f["mint"]
            if f["status"] != "filled":
                continue
            if f["side"] == "buy":
                self.bought[mint] = self.bought.get(mint, 0) + int(f["quote_lamports"])
        for mint in list(self.bought):
            if mint not in state.positions:
                has_pending_buy = any(p.order["mint"] == mint and p.order["side"] == "buy" for p in state.pending.values())
                if not has_pending_buy:
                    if self.bought[mint] >= self.p["min_buy"]:
                        self.qualified.add(mint)
                    del self.bought[mint]
        for mint in list(self.book):
            if mint not in state.positions:
                has_pending = any(p.order["mint"] == mint for p in state.pending.values())
                if not has_pending:
                    del self.book[mint]

    def _pending_sell(self, state, mint):
        return any(p.order["mint"] == mint and p.order["side"] == "sell" for p in state.pending.values())

    def _pending_buy(self, state, mint):
        return any(p.order["mint"] == mint and p.order["side"] == "buy" for p in state.pending.values())

    def _value(self, mint, tokens, state):
        m = self.marks.get(mint)
        if m is not None:
            vq, vt, rq = int(m["virtual_quote_reserves"]), int(m["virtual_token_reserves"]), int(m["real_quote_reserves"])
        else:
            c = state.curves.get(mint)
            vq, vt, rq = c.virtual_quote_reserves, c.virtual_token_reserves, c.real_quote_reserves
        return sell_value(tokens, vq, vt, rq)

    # ------------------------------------------------------------ main
    def on_batch(self, state, batch):
        p = self.p
        slot = batch["slot"]
        touched = self.market.apply(batch)
        self.marks = {m["mint"]: m for m in batch["marks"]}
        self._fill_bookkeeping(state, batch)
        orders = []
        cap_orders = state.hello["caps"]["max_orders_per_batch"]
        tx = state.tx_fee_lamports()

        # 1. exits
        for mint, h in state.positions.items():
            if len(orders) >= cap_orders:
                break
            pos = self.book.get(mint)
            if pos is None:
                pos = Position(mint, slot, False)
                self.book[mint] = pos
            if self._pending_sell(state, mint):
                continue
            c = self.market.coins.get(mint)
            if c is None or c.complete:
                continue
            value = self._value(mint, h.token_amount, state) - tx
            cost = h.cost_lamports
            if value > pos.peak_value:
                pos.peak_value = value
            why = self._exit_reason(pos, c, value, cost, slot)
            if why is not None:
                orders.append({"side": "sell", "mint": mint, "token_amount": h.token_amount,
                               "max_slippage_bps": p["sell_slip_bps"]})
                pos.exit_sent_slot = slot
                self.last_exit[mint] = slot

        # 2. adds to positions being built
        free = state.free_cash_lamports() - p["reserve"]
        for mint, pos in self.book.items():
            if len(orders) >= cap_orders or pos.probe or pos.exit_sent_slot is not None:
                continue
            h = state.positions.get(mint)
            if h is None or self._pending_buy(state, mint):
                continue
            if p["pyramid_step"] <= 0 and slot - pos.first_slot > p["build_slots"]:
                continue
            c = self.market.coins.get(mint)
            if c is None or c.complete or c not in touched:
                continue
            pyramid = p["pyramid_step"] > 0 and c.rq >= pos.last_buy_rq * p["pyramid_step"] and c.price >= pos.last_buy_price
            if pyramid:
                want = min(p["pyramid_max"], int(p["pyramid_frac"] * c.rq)) - h.cost_lamports
            elif self._entry_ok(c, slot, adding=True):
                want = p["stake"] - h.cost_lamports
            else:
                continue
            amount = min(want, int(state.max_buy_lamports(mint) * p["cap_use"]), free - tx)
            if amount >= p["min_buy"]:
                orders.append({"side": "buy", "mint": mint, "quote_lamports": amount, "max_slippage_bps": p["buy_slip_bps"]})
                free -= amount + tx
                pos.last_buy_slot = slot
                pos.last_buy_rq = c.rq
                pos.last_buy_price = c.price

        # 3. new entries
        slots_free = min(p["max_positions"], state.hello["caps"]["max_open_positions"]) - state.committed_position_count()
        if slots_free > 0:
            cands = []
            for c in touched:
                if c.mint in self.book or c.mint in state.positions or c.complete:
                    continue
                if self.entries.get(c.mint, 0) >= p["max_entries_per_coin"]:
                    continue
                if slot - self.last_exit.get(c.mint, -10**9) < p["cooldown_slots"]:
                    continue
                if self._entry_ok(c, slot, adding=False):
                    cands.append(c)
            cands.sort(key=lambda c: (-c.rq, c.mint))
            for c in cands:
                if slots_free <= 0 or len(orders) >= cap_orders:
                    break
                amount = min(p["stake"], int(state.max_buy_lamports(c.mint) * p["cap_use"]), free - tx)
                if amount < p["min_buy"]:
                    continue
                orders.append({"side": "buy", "mint": c.mint, "quote_lamports": amount, "max_slippage_bps": p["buy_slip_bps"]})
                free -= amount + tx
                slots_free -= 1
                self.entries[c.mint] = self.entries.get(c.mint, 0) + 1
                self.book[c.mint] = Position(c.mint, slot, False, c.rq, c.price)

        # 4. guardrail probes: small round trips on liquid curves until enough coins qualify
        if (slots_free > 0 and len(orders) < cap_orders and len(self.qualified) + len(self.bought) < p["probe_target"]
                and slot - self.last_probe >= p["probe_every"]):
            for c in touched:
                if c.mint in self.book or c.mint in state.positions or c.complete or c.mint in self.qualified:
                    continue
                if c.rq < p["probe_min_rq"]:
                    continue
                amount = p["probe_lamports"]
                if amount > int(state.max_buy_lamports(c.mint) * p["cap_use"]) or free - tx < amount:
                    continue
                orders.append({"side": "buy", "mint": c.mint, "quote_lamports": amount, "max_slippage_bps": BPS})
                free -= amount + tx
                self.book[c.mint] = Position(c.mint, slot, True)
                self.last_probe = slot
                break
        return orders

    # ------------------------------------------------------------ rules
    def _entry_ok(self, c, slot, adding):
        p = self.p
        if not (p["entry_rq_min"] <= c.rq <= p["entry_rq_max"]):
            return False
        age = c.age(slot)
        if not adding and not (p["min_age"] <= age <= p["max_age"]):
            return False
        if not adding and p["fresh_level_slots"] < 10**9:
            lv_slot = self._level_slot(c, p["entry_rq_min"])
            if lv_slot is None or slot - lv_slot > p["fresh_level_slots"]:
                return False
        if c.price < c.ath * (1 - p["max_dd_from_ath"]):
            return False
        net, nb, ns, buyers, big_sell = flow(c, slot, p["flow_window"])
        if net < p["min_net_flow"] or buyers < p["min_buyers"]:
            return False
        if c.rq > 0 and big_sell > p["max_big_sell_frac"] * c.rq:
            return False
        if p["max_top_holder"] < 1.0 and top_holder_share(c) > p["max_top_holder"]:
            return False
        return True

    @staticmethod
    def _level_slot(c, level):
        from .market import LEVELS
        best = None
        for i, lv in enumerate(LEVELS):
            if lv <= level and i in c.level_slots:
                best = c.level_slots[i]
        return best

    def _exit_reason(self, pos, c, value, cost, slot):
        p = self.p
        if pos.probe:
            return "probe"
        if cost <= 0:
            return None
        building = slot - pos.first_slot <= p["build_slots"] and pos.exit_sent_slot is None
        r = value / cost - 1
        if r >= p["tp"]:
            return "tp"
        if r <= -p["sl"]:
            return "sl"
        if pos.peak_value >= cost * (1 + p["trail_arm"]) and value <= pos.peak_value * (1 - p["trail"]):
            return "trail"
        if slot - pos.first_slot >= p["max_hold"]:
            return "time"
        age = c.age(slot)
        for at, min_rq in p["progress"]:
            if age >= at and c.rq < min_rq:
                return "progress"
        if not building:
            net, nb, ns, buyers, big = flow(c, slot, p["exit_flow_window"])
            if net < p["exit_flow"]:
                return "flow"
        if p["spike_frac"] < 1.0 and r >= p["spike_min_profit"] and c.last_slot == slot:
            # a single large buy just landed: large buys tend to be faded, so sell into it
            for s, buy, sol, user in reversed(c.tape):
                if s < slot:
                    break
                if buy and sol >= p["spike_frac"] * c.rq:
                    return "spike"
        return None
