"""A Python port of the kit's pump.fun replay engine (harness sim/engine.ts + sim/curve.ts + sim/rng.ts).

Research only: it lets a strategy be replayed in-process many times a minute. The official grader
(`makefun grade-local`) stays the authority; tools/verify.py checks this port against it.

    from tools.dataset import load_dataset
    from tools.replay import run_strategy
    ds = load_dataset(kit_dir)
    res = run_strategy(ds, hello, on_batch)          # on_batch(state, batch) like makefun_bot.run's

Everything is integer math, mirroring the BigInt code: same rejections, fills, closes, fees.
"""

import hashlib

BPS = 10_000
MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
DOMAIN_FAILED_TX = 0x6661696C65642D74
DOMAIN_PUBLIC_FAIL_KEY = 0x6661696C2D6B6579
SOL_QUOTE = "11111111111111111111111111111111"


# ---------------------------------------------------------------- rng (sim/rng.ts)

def mix64(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def keyed_u64(seed, *parts):
    h = mix64((seed + GAMMA) & MASK)
    for p in parts:
        h = mix64(((h ^ (p & MASK)) + GAMMA) & MASK)
    return h


def public_fail_key(seed):
    return keyed_u64(int(seed), DOMAIN_PUBLIC_FAIL_KEY)


def tx_fails(fail_key, seq, index, failed_bps):
    if failed_bps <= 0:
        return False
    return keyed_u64(fail_key, DOMAIN_FAILED_TX, seq, index) % 10_000 < failed_bps


# ---------------------------------------------------------------- curve math (sim/curve.ts)

def ceil_div(a, b):
    return (a + b - 1) // b


def fee_of(amount, bps):
    return ceil_div(amount * bps, BPS)


def buyback_of(fee, bps):
    return fee * bps // BPS


def sell_quote_out(t, s):
    vq, vt, rq, rt = s
    return t * vq // (vt + t)


def buy_quote_in_for_tokens(t, s):
    vq, vt, rq, rt = s
    return t * vq // (vt - t) + 1


def buy_tokens_out_for_quote(q, s):
    vq, vt, rq, rt = s
    return 0 if q <= 1 else (q - 1) * vt // (vq + q - 1)


def exact_in_split(spendable, fee_bps, creator_bps):
    net0 = spendable * BPS // (BPS + fee_bps + creator_bps)
    fee = fee_of(net0, fee_bps)
    cfee = fee_of(net0, creator_bps)
    over = net0 + fee + cfee - spendable if net0 + fee + cfee > spendable else 0
    return net0 - over, fee, cfee


def quote_buy_exact_in(spendable, s, bps):
    """(q, t, fee, creator_fee, paid) of an exact-in buy."""
    pb, cb = bps
    q, fee, cfee = exact_in_split(spendable, pb, cb)
    t = buy_tokens_out_for_quote(q, s)
    rt = s[3]
    if t > rt:
        tc = rt
        if tc == 0:
            return 0, 0, 0, 0, 0
        q = buy_quote_in_for_tokens(tc, s)
        fee = fee_of(q, pb)
        cfee = fee_of(q, cb)
        return q, tc, fee, cfee, q + fee + cfee
    return q, t, fee, cfee, q + fee + cfee


def quote_sell(t, s, bps):
    """(q, fee, creator_fee, proceeds) of a sell of t tokens."""
    pb, cb = bps
    q = sell_quote_out(t, s)
    fee = fee_of(q, pb)
    cfee = fee_of(q, cb)
    return q, fee, cfee, q - fee - cfee


def offset_reserves(rec, off_q, off_t):
    """Recorded reserves shifted by the bot's own net flow (engine.ts offsetReserves)."""
    vq, vt, rq, rt = rec
    num, den = 1, 1
    if off_t > rt:
        num, den = rt, off_t
    if off_q < 0 and rq * den < num * -off_q:
        num, den = rq, -off_q

    def scale(x):
        if num == den:
            return x
        return x * num // den if x >= 0 else -((-x * num) // den)

    q = scale(off_q)
    t = scale(off_t)
    return (vq + q, vt - t, rq + q, rt - t)


# ---------------------------------------------------------------- universe (sim/universe.ts)

def in_sample(mint, modulus, residue):
    h = int.from_bytes(hashlib.sha256(mint.encode("utf-8")).digest(), "big")
    return h % modulus == residue


def universe_ok(create, sample, excluded):
    if create.get("quote_mint") != SOL_QUOTE:
        return False
    if create.get("is_mayhem_mode"):
        return False
    if create.get("is_cashback_enabled"):
        return False
    if create["mint"] in excluded:
        return False
    return in_sample(create["mint"], sample["modulus"], sample["residue"])


# ---------------------------------------------------------------- the engine

class _Mint:
    __slots__ = ("state", "state_slot", "last_trade", "last_trade_slot", "complete", "off_q", "off_t")

    def __init__(self, state, slot):
        self.state = state
        self.state_slot = slot
        self.last_trade = None
        self.last_trade_slot = -1
        self.complete = False
        self.off_q = 0
        self.off_t = 0


class _Pending:
    __slots__ = ("order", "seq", "index", "decided_slot", "fill_slot", "reject", "decision")

    def __init__(self, order, seq, index, decided_slot, fill_slot, reject, decision):
        self.order = order
        self.seq = seq
        self.index = index
        self.decided_slot = decided_slot
        self.fill_slot = fill_slot
        self.reject = reject
        self.decision = decision


class _Token:
    __slots__ = ("pnl", "fills", "fees", "tx_fees", "open")

    def __init__(self):
        self.pnl = 0
        self.fills = 0
        self.fees = 0
        self.tx_fees = 0
        self.open = None  # [pnl, opened_slot, bought]


class Engine:
    """engine.ts ReplayEngine: next_batch() / submit(orders) / finish()."""

    def __init__(self, dataset, hello, fail_key=None, fee_multiplier=1):
        self.hello = hello
        self.mult = fee_multiplier
        fees = hello["fees"]
        self.tx_fee = (int(fees["base_fee_lamports"]) + int(fees["priority_fee_lamports"])) * fee_multiplier
        self.fail_key = public_fail_key(hello["seed"]) if fail_key is None else int(fail_key)
        self.failed_bps = fees["failed_tx_bps"]
        self.fallback = (fees["protocol_bps_fallback"] * fee_multiplier, fees["creator_bps_fallback"] * fee_multiplier)
        caps = hello["caps"]
        self.max_open = caps["max_open_positions"]
        self.max_order_bps = caps["max_order_bps_of_real_reserve"]
        self.max_orders = caps["max_orders_per_batch"]
        self.latency = hello["latency_slots"]
        self.cash = int(hello["capital"]["lamports"])
        self.start_cash = self.cash
        self.batches = dataset.batches
        self.end_slot = dataset.end_slot
        self.slot_time = dataset.slot_time
        self.cursor = 0
        self.seq = 0
        self.cur_slot = -1
        self.cur_time = None
        self.mints = {}
        self.pending = []
        self.pending_head = 0
        self.used_ids = set()
        self.positions = {}  # mint -> [tokens, cost]
        self.tokens = {}
        self.resolutions = []
        self.undelivered = []
        self.equity = []
        self.round_trips = []
        self.close_counter = 0
        self.spent_seq = -1
        self.spent = {}
        self.fees_paid = 0
        self.tx_paid = 0
        self.failed = 0
        self.rejected = 0
        self.fills = 0

    # -- helpers
    def reserves(self, m):
        return offset_reserves(m.state, m.off_q, m.off_t)

    def fees_for(self, landing):
        if landing is not None and int(landing["fee_basis_points"]) > 0:
            return (int(landing["fee_basis_points"]) * self.mult, int(landing["creator_fee_basis_points"]) * self.mult,
                    int(landing["buyback_fee_basis_points"]), False)
        return (self.fallback[0], self.fallback[1], 0, True)

    def close_quote(self, m, tokens, s=None):
        if s is None:
            s = self.reserves(m)
        pb, cb, bbb, fb = self.fees_for(m.last_trade)
        q = sell_quote_out(tokens, s)
        if q > s[2]:
            q = s[2]
        fee = fee_of(q, pb)
        cfee = fee_of(q, cb)
        proceeds = q - fee - cfee
        return q, fee, cfee, proceeds, proceeds - self.tx_fee, (pb, cb, bbb, fb)

    def equity_now(self):
        e = self.cash
        for mint, p in self.positions.items():
            c = self.close_quote(self.mints[mint], p[0])
            if c[4] > 0:
                e += c[4]
        return e

    def token(self, mint):
        t = self.tokens.get(mint)
        if t is None:
            t = _Token()
            self.tokens[mint] = t
        return t

    def flow(self, tk, cash_delta, curve_fees, tx_fees, filled):
        tk.pnl += cash_delta
        tk.fees += curve_fees
        tk.tx_fees += tx_fees
        if filled:
            tk.fills += 1
        if tk.open is not None:
            tk.open[0] += cash_delta
        self.fees_paid += curve_fees
        self.tx_paid += tx_fees

    def close_round_trip(self, mint, tk, slot):
        if tk.open is None:
            return
        self.round_trips.append({"mint": mint, "opened_slot": tk.open[1], "closed_slot": slot,
                                 "pnl_lamports": tk.open[0], "bought_lamports": tk.open[2]})
        tk.open = None

    def fill_msg(self, p, status, reason=None, quote=0, tokens=0, fee=0, tx=0, fallback=False):
        return {"client_id": p.order["client_id"], "mint": p.order["mint"], "side": p.order["side"], "status": status,
                "reason": reason, "decided_slot": p.decided_slot, "slot": p.fill_slot, "quote_lamports": str(quote),
                "token_amount": str(tokens), "fee_lamports": str(fee), "tx_fee_lamports": str(tx), "fee_fallback": fallback}

    def record(self, f):
        self.resolutions.append(f)
        self.undelivered.append(f)

    def reject(self, p, reason):
        self.rejected += 1
        self.record(self.fill_msg(p, "rejected", reason))

    def open_count(self):
        return sum(1 for p in self.positions.values() if p[0] > 0)

    # -- protocol
    def next_batch(self):
        if self.cursor >= len(self.batches):
            return None
        slot, events = self.batches[self.cursor]
        return self.step_to(slot, events)

    def step_to(self, x, events):
        if not self.equity:
            self.equity.append(self.cash)
        pend = self.pending
        while self.pending_head < len(pend) and pend[self.pending_head].fill_slot < x:
            p = pend[self.pending_head]
            self.pending_head += 1
            self.cur_slot = p.fill_slot
            self.resolve(p)
        self.cur_slot = x
        t = self.slot_time.get(x)
        if t is not None:
            self.cur_time = t
        completed = []
        for e in events:
            kind = e["event"]
            if kind == "TradeEvent":
                m = self.mints[e["mint"]]
                m.state = (int(e["virtual_quote_reserves"]), int(e["virtual_token_reserves"]),
                           int(e["real_quote_reserves"]), int(e["real_token_reserves"]))
                m.state_slot = x
                m.last_trade = e
                m.last_trade_slot = x
            elif kind == "CreateEvent":
                self.mints[e["mint"]] = _Mint((int(e["virtual_quote_reserves"]), int(e["virtual_token_reserves"]), 0,
                                               int(e["real_token_reserves"])), x)
            elif kind == "CompleteEvent":
                self.mints[e["mint"]].complete = True
                completed.append(e["mint"])
        for mint in completed:
            if mint in self.positions:
                self.force_close(mint, "complete", x)
        while self.pending_head < len(pend) and pend[self.pending_head].fill_slot == x:
            p = pend[self.pending_head]
            self.pending_head += 1
            self.resolve(p)
        if self.pending_head > 4096:
            del pend[: self.pending_head]
            self.pending_head = 0
        marks = self.marks()
        fills = self.undelivered
        self.undelivered = []
        self.equity.append(self.equity_now())
        return {"type": "batch", "seq": self.seq, "slot": x, "events": events, "marks": marks, "fills": fills,
                "account": self.account()}

    def marks(self):
        want = set(self.positions)
        for i in range(self.pending_head, len(self.pending)):
            mint = self.pending[i].order["mint"]
            if mint in self.mints:
                want.add(mint)
        out = []
        for mint in sorted(want):
            m = self.mints[mint]
            s = self.reserves(m)
            out.append({"mint": mint, "slot": m.state_slot, "virtual_token_reserves": str(s[1]),
                        "virtual_quote_reserves": str(s[0]), "real_token_reserves": str(s[3]),
                        "real_quote_reserves": str(s[2])})
        return out

    def account(self):
        return {"cash_lamports": str(self.cash),
                "positions": [{"mint": mint, "token_amount": str(p[0]), "cost_lamports": str(p[1])}
                              for mint, p in sorted(self.positions.items())]}

    def submit(self, orders):
        for index, order in enumerate(orders):
            reject = None
            amount = order["quote_lamports"] if order["side"] == "buy" else order["token_amount"]
            if index >= self.max_orders:
                reject = "max_orders_per_batch"
            elif order["client_id"] in self.used_ids:
                reject = "duplicate_client_id"
            elif str(amount) == "0":
                reject = "zero_amount"
            elif order["mint"] not in self.mints:
                reject = "not_in_universe"
            self.used_ids.add(order["client_id"])
            m = self.mints.get(order["mint"])
            self.pending.append(_Pending(order, self.seq, index, self.cur_slot, self.cur_slot + self.latency, reject,
                                         self.reserves(m) if m is not None else None))
        self.seq += 1
        self.cursor += 1

    def finish(self):
        self.cur_slot = self.end_slot
        while self.pending_head < len(self.pending):
            p = self.pending[self.pending_head]
            self.pending_head += 1
            if p.fill_slot > self.end_slot:
                self.reject(p, "after_end")
            else:
                self.resolve(p)
        for mint in sorted(self.positions):
            self.force_close(mint, "flatten", self.end_slot)
        self.equity.append(self.cash)
        return self.result()

    # -- resolution
    def failed_draw(self, p):
        if not tx_fails(self.fail_key, p.seq, p.index, self.failed_bps):
            return False
        self.failed += 1
        self.cash -= self.tx_fee
        self.flow(self.token(p.order["mint"]), -self.tx_fee, 0, self.tx_fee, False)
        self.record(self.fill_msg(p, "failed", tx=self.tx_fee))
        return True

    def resolve(self, p):
        if p.reject is not None:
            return self.reject(p, p.reject)
        mint = p.order["mint"]
        m = self.mints[mint]
        if m.complete:
            return self.reject(p, "token_complete")
        landing = m.last_trade if m.last_trade_slot == p.fill_slot else None
        pb, cb, bbb, fallback = self.fees_for(landing)
        bps = (pb, cb)
        slip = p.order["max_slippage_bps"]
        decision = p.decision
        s = self.reserves(m)
        if p.order["side"] == "sell":
            pos = self.positions.get(mint)
            t = int(p.order["token_amount"])
            if pos is None or pos[0] < t:
                return self.reject(p, "insufficient_tokens")
            q, fee, cfee, proceeds = quote_sell(t, s, bps)
            if q > s[2]:
                return self.reject(p, "max_order_size")
            if proceeds <= 0:
                return self.reject(p, "zero_amount")
            expected = quote_sell(t, decision, bps)[3]
            if proceeds * BPS < expected * (BPS - slip):
                return self.reject(p, "max_slippage")
            if self.cash < self.tx_fee:
                return self.reject(p, "insufficient_cash")
            if self.failed_draw(p):
                return
            before = self.cash
            self.cash += proceeds - self.tx_fee
            removed = pos[1] if pos[0] == t else pos[1] * t // pos[0]
            pos[0] -= t
            pos[1] -= removed
            m.off_q -= q
            m.off_t -= t
            tk = self.token(mint)
            self.flow(tk, self.cash - before, fee + cfee, self.tx_fee, True)
            self.fills += 1
            if pos[0] == 0:
                del self.positions[mint]
                self.close_round_trip(mint, tk, p.fill_slot)
            return self.record(self.fill_msg(p, "filled", quote=proceeds, tokens=t, fee=fee + cfee, tx=self.tx_fee,
                                             fallback=fallback))
        spend = int(p.order["quote_lamports"])
        held = self.positions.get(mint)
        if (held is None or held[0] == 0) and self.open_count() >= self.max_open:
            return self.reject(p, "max_open_positions")
        if self.spent_seq != p.seq:
            self.spent_seq = p.seq
            self.spent = {}
        total = self.spent.get(mint, 0) + spend
        if total * BPS > decision[2] * self.max_order_bps:
            return self.reject(p, "max_order_size")
        self.spent[mint] = total
        q, t, fee, cfee, paid = quote_buy_exact_in(spend, s, bps)
        if t == 0:
            return self.reject(p, "zero_amount")
        expected = quote_buy_exact_in(spend, decision, bps)[1]
        if t * BPS < expected * (BPS - slip):
            return self.reject(p, "max_slippage")
        if self.cash < spend + self.tx_fee:
            return self.reject(p, "insufficient_cash")
        if self.failed_draw(p):
            return
        before = self.cash
        self.cash -= paid + self.tx_fee
        opening = held is None or held[0] == 0
        if held is None:
            held = [0, 0]
            self.positions[mint] = held
        held[0] += t
        held[1] += paid
        m.off_q += q
        m.off_t += t
        tk = self.token(mint)
        if opening:
            tk.open = [0, p.fill_slot, 0]
        if tk.open is not None:
            tk.open[2] += paid
        self.flow(tk, self.cash - before, fee + cfee, self.tx_fee, True)
        self.fills += 1
        self.record(self.fill_msg(p, "filled", quote=paid, tokens=t, fee=fee + cfee, tx=self.tx_fee, fallback=fallback))

    def force_close(self, mint, why, slot):
        pos = self.positions[mint]
        m = self.mints[mint]
        t = pos[0]
        q, fee, cfee, proceeds, net, bpsx = self.close_quote(m, t)
        n = self.close_counter
        self.close_counter += 1
        tk = self.token(mint)
        before = self.cash
        abandon = net <= 0
        kind = "writeoff" if abandon else why
        f = {"client_id": f"_{kind}_{n}", "mint": mint, "side": "sell", "status": "filled", "reason": None,
             "decided_slot": slot, "slot": slot, "quote_lamports": "0", "token_amount": str(t), "fee_lamports": "0",
             "tx_fee_lamports": "0", "fee_fallback": False}
        if not abandon:
            self.cash += net
            m.off_q -= q
            m.off_t -= t
            self.flow(tk, self.cash - before, fee + cfee, self.tx_fee, False)
            f.update({"quote_lamports": str(proceeds), "fee_lamports": str(fee + cfee),
                      "tx_fee_lamports": str(self.tx_fee), "fee_fallback": bpsx[3]})
        del self.positions[mint]
        self.close_round_trip(mint, tk, slot)
        self.record(f)

    def result(self):
        peak = 0
        worst = 0.0
        for e in self.equity:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > worst:
                    worst = dd
        min_buy = 10_000_000
        counted_mints = {rt["mint"] for rt in self.round_trips if rt["bought_lamports"] >= min_buy}
        wins = sum(1 for rt in self.round_trips if rt["pnl_lamports"] > 0)
        per_token = {mint: t.pnl for mint, t in self.tokens.items()}
        return {
            "net_pnl": self.cash - self.start_cash,
            "max_drawdown": worst,
            "fills": self.fills,
            "round_trips": len(self.round_trips),
            "counted_mints": len(counted_mints),
            "win_rate": wins / len(self.round_trips) if self.round_trips else 0.0,
            "fees": self.fees_paid,
            "tx_fees": self.tx_paid,
            "failed": self.failed,
            "rejected": self.rejected,
            "per_token": per_token,
            "trips": self.round_trips,
            "equity": self.equity,
        }


def run_strategy(dataset, hello, on_batch, fail_key=None, fee_multiplier=1, make_state=None):
    """Drive a makefun_bot-style strategy (on_batch(state, batch) -> orders) through the engine in process.

    make_state(hello) builds the bot's state (default makefun_bot.BotState); its to_orders/record_orders
    do exactly what makefun_bot.run does with the return value.
    """
    if make_state is None:
        import makefun_bot
        make_state = makefun_bot.BotState
    eng = Engine(dataset, hello, fail_key=fail_key, fee_multiplier=fee_multiplier)
    state = make_state(hello)
    while True:
        batch = eng.next_batch()
        if batch is None:
            break
        state.apply_batch(batch)
        orders = state.to_orders(on_batch(state, batch))
        state.record_orders(orders)
        eng.submit(orders)
    return eng.finish()
