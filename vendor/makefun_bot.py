"""agent-fun/replay/v2 bot shim for Python 3.13: standard library only, one file.

The grader runs a Python bot as ``python3 -P -s /bot/main.py`` with ``PYTHONHASHSEED=0`` in its
environment. -P leaves the script's directory off sys.path (and -s the user site directory), so copy
this file into your repository (vendor/makefun_bot.py keeps it out of the authorship counts) and put
that directory on sys.path yourself::

    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor"))
    import makefun_bot

    def on_batch(state, batch):
        # the orders decided on this slot; None or [] for none
        return [{"side": "buy", "mint": mint, "quote_lamports": makefun_bot.LAMPORTS_PER_SOL // 20,
                 "max_slippage_bps": 500}]

    makefun_bot.run(on_batch)              # optional: on_hello=f(state), on_done=f(state, done)

The shim owns the wire, exactly as sdk/ts/makefun-bot.ts does. It reads one JSON line at a time from
stdin (each at most MAX_HARNESS_LINE_BYTES), keeps ``state`` current (account, positions, every curve's
latest reserves, the orders still in flight), calls on_batch and writes exactly one reply per batch (at
most MAX_BOT_LINE_BYTES), echoing the batch's seq, and flushes it before reading the next line. Messages
are plain dicts with the protocol's keys; u64 values arrive as decimal strings (``int()`` them).
Amounts you return may be ints or u64 decimal strings; the shim writes them as decimal strings and
fills in client_id when you leave it out. An order the harness would treat as a protocol violation
(malformed mint, amount, bps or client_id, a client_id starting with ``_``, an unknown key, a reply line
over the limit) raises OrderError here, so a broken bot fails on your machine and not in grading. So
does a reply with more orders than ``hello["caps"]["max_orders_per_batch"]``: the harness rejects the
orders past the cap and fails the run on a reply far over it. Orders the market refuses (position and
size caps, cash, slippage) are not errors: they come back in a later batch's fills with status
"rejected" and a reason.

Two market rules worth knowing before you size orders: the buys on one coin in one reply share one
size cap (``state.max_buy_lamports``), and your own fills on one coin in one slot move the reserves for
each other, so splitting an order never gets you a better price than sending it whole. Client ids
starting with ``_`` are the harness's closes (``_complete_<n>``, ``_flatten_<n>``, ``_writeoff_<n>``),
reported as sell fills.

stdout carries replies only: print diagnostics to stderr. Determinism: every entry is executed twice
and the two action logs must match byte for byte, so never read the clock, the environment or an
unseeded random; seed ``random.Random`` from ``int(state.hello["seed"])``. The grader fixes
PYTHONHASHSEED=0, so str hashes, and with them the iteration order of sets of strings, are the same in
both executions. Set it when you run your bot yourself too (``PYTHONHASHSEED=0 python3 -P -s main.py``):
without it set order changes from run to run and differs from the graded one (dicts keep insertion
order whatever the seed).
"""

import json
import re
import sys

PROTOCOL = "agent-fun/replay/v2"
#: Harness -> bot lines (hello, batch, done) are at most this many UTF-8 bytes, the newline not counted (1 MiB).
MAX_HARNESS_LINE_BYTES = 1_048_576
#: Bot -> harness lines (your replies) are at most this many UTF-8 bytes, the newline not counted (64 KiB).
MAX_BOT_LINE_BYTES = 65_536
PUMP_PROGRAM_ID = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
LAMPORTS_PER_SOL = 1_000_000_000
#: Basis points per unit: 10_000 bps = 100%.
BPS = 10_000
U64_MAX = 2**64 - 1

REJECT_REASONS = (
    "max_slippage",
    "max_open_positions",
    "max_order_size",
    "max_orders_per_batch",
    "insufficient_cash",
    "insufficient_tokens",
    "not_in_universe",
    "token_complete",
    "zero_amount",
    "duplicate_client_id",
    "after_end",
)

_U64_RE = re.compile(r"0|[1-9][0-9]{0,19}")
_PUBKEY_RE = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")
#: 1-64 of [A-Za-z0-9_-], not starting with ``_``: ids starting with ``_`` are the harness's closes.
_CLIENT_ID_RE = re.compile(r"[A-Za-z0-9-][A-Za-z0-9_-]{0,63}")
_BUY_KEYS = ("side", "client_id", "mint", "quote_lamports", "max_slippage_bps")
_SELL_KEYS = ("side", "client_id", "mint", "token_amount", "max_slippage_bps")


class ProtocolError(Exception):
    """The harness sent something this shim cannot follow (wrong protocol, malformed line, bad order of messages)."""


class OrderError(Exception):
    """The bot returned something the harness would treat as a protocol violation."""


class Curve:
    """A bonding curve the bot has seen, at its latest known reserves (ints)."""

    __slots__ = (
        "mint",
        "created_slot",
        "virtual_token_reserves",
        "virtual_quote_reserves",
        "real_token_reserves",
        "real_quote_reserves",
        "reserves_slot",
        "trades",
        "complete",
    )

    def __init__(self, mint, created_slot, slot):
        self.mint = mint
        #: Slot of its CreateEvent; None when the shim saw the mint first through a trade or a mark.
        self.created_slot = created_slot
        self.virtual_token_reserves = 0
        self.virtual_quote_reserves = 0
        self.real_token_reserves = 0
        self.real_quote_reserves = 0
        #: Slot of the trade the reserves follow (the create slot until the first trade).
        self.reserves_slot = slot
        #: TradeEvents seen on it.
        self.trades = 0
        #: Its CompleteEvent arrived: no more orders (they would be rejected token_complete).
        self.complete = False

    def __repr__(self):
        fields = ", ".join(f"{k}={getattr(self, k)!r}" for k in self.__slots__)
        return f"Curve({fields})"


class Holding:
    """A held position, as the latest account reports it (ints)."""

    __slots__ = ("token_amount", "cost_lamports")

    def __init__(self, token_amount, cost_lamports):
        self.token_amount = token_amount
        self.cost_lamports = cost_lamports

    def __repr__(self):
        return f"Holding(token_amount={self.token_amount}, cost_lamports={self.cost_lamports})"


class PendingOrder:
    """An order sent and not yet reported back in a fill: the wire order and the batch it was decided on."""

    __slots__ = ("order", "seq", "decided_slot")

    def __init__(self, order, seq, decided_slot):
        self.order = order
        self.seq = seq
        self.decided_slot = decided_slot

    def __repr__(self):
        return f"PendingOrder(order={self.order!r}, seq={self.seq}, decided_slot={self.decided_slot})"


class BotState:
    """Everything the bot knows, kept current by run() before each callback.

    You can also drive it yourself to unit-test a strategy:
    ``s = BotState(hello); s.apply_batch(batch); on_batch(s, batch)``.
    """

    def __init__(self, hello):
        self.hello = hello
        #: Seq and slot of the batch being answered (-1 before the first batch).
        self.seq = -1
        self.slot = -1
        #: The account dict as of the latest batch (or done).
        self.account = {"cash_lamports": hello["capital"]["lamports"], "positions": []}
        #: account["cash_lamports"] as an int. Pending orders are not deducted: see free_cash_lamports.
        self.cash = int(hello["capital"]["lamports"])
        #: Positions with tokens, by mint.
        self.positions = {}
        #: Every curve seen, by mint, in first-seen order.
        self.curves = {}
        #: Orders sent and not yet reported in a fill, by client_id, in send order.
        self.pending = {}
        self._used = set()
        self._next = 1

    def client_id(self):
        """A fresh client_id (o1, o2, ...), never one already used in this run and never starting with ``_``."""
        cid = self._free_id(())
        self._used.add(cid)
        return cid

    def tx_fee_lamports(self):
        """Base + priority fee every sent order pays, filled or not."""
        fees = self.hello["fees"]
        return int(fees["base_fee_lamports"]) + int(fees["priority_fee_lamports"])

    def max_buy_lamports(self, mint):
        """The size cap on this mint now: the most this batch's buys on it may spend together (the sum of
        their quote_lamports); 0 when unknown or complete."""
        curve = self.curves.get(mint)
        if curve is None or curve.complete:
            return 0
        return curve.real_quote_reserves * self.hello["caps"]["max_order_bps_of_real_reserve"] // BPS

    def committed_position_count(self):
        """Mints held plus mints with a buy pending: what counts against caps.max_open_positions."""
        mints = set(self.positions)
        for p in self.pending.values():
            if p.order["side"] == "buy":
                mints.add(p.order["mint"])
        return len(mints)

    def free_cash_lamports(self):
        """Cash minus what pending orders will take if they fill (buy amounts and every tx fee). May go negative."""
        tx = self.tx_fee_lamports()
        free = self.cash
        for p in self.pending.values():
            free -= tx
            if p.order["side"] == "buy":
                free -= int(p.order["quote_lamports"])
        return free

    def apply_batch(self, batch):
        """Fold a batch in: its fills, account, marks and events (in that order). run() calls this."""
        self.seq = batch["seq"]
        self.slot = batch["slot"]
        for f in batch["fills"]:
            self.pending.pop(f["client_id"], None)
        self._set_account(batch["account"])
        for m in batch["marks"]:
            curve = self.curves.get(m["mint"])
            if curve is not None and m["slot"] < curve.reserves_slot:
                continue
            if curve is None:
                curve = self._new_curve(m["mint"], None, m["slot"])
            curve.virtual_token_reserves = int(m["virtual_token_reserves"])
            curve.virtual_quote_reserves = int(m["virtual_quote_reserves"])
            curve.real_token_reserves = int(m["real_token_reserves"])
            curve.real_quote_reserves = int(m["real_quote_reserves"])
            curve.reserves_slot = m["slot"]
        for e in batch["events"]:
            kind = e["event"]
            if kind == "CreateEvent":
                curve = self._new_curve(e["mint"], e["slot"], e["slot"])
                curve.virtual_token_reserves = int(e["virtual_token_reserves"])
                curve.virtual_quote_reserves = int(e["virtual_quote_reserves"])
                curve.real_token_reserves = int(e["real_token_reserves"])
                curve.real_quote_reserves = 0
            elif kind == "TradeEvent":
                curve = self.curves.get(e["mint"]) or self._new_curve(e["mint"], None, e["slot"])
                curve.virtual_token_reserves = int(e["virtual_token_reserves"])
                curve.virtual_quote_reserves = int(e["virtual_quote_reserves"])
                curve.real_token_reserves = int(e["real_token_reserves"])
                curve.real_quote_reserves = int(e["real_quote_reserves"])
                curve.reserves_slot = e["slot"]
                curve.trades += 1
            else:
                curve = self.curves.get(e["mint"]) or self._new_curve(e["mint"], None, e["slot"])
                curve.complete = True

    def apply_done(self, done):
        """Fold the final account in. run() calls this before on_done."""
        self.seq = done["seq"]
        self._set_account(done["account"])

    def to_orders(self, result):
        """Validate what on_batch returned and turn it into wire orders (assigning missing client_ids).

        Raises OrderError on anything the harness would call a protocol violation, and on more orders
        than ``hello["caps"]["max_orders_per_batch"]``. Does not record them.
        """
        if result is None:
            return []
        if not isinstance(result, (list, tuple)):
            raise OrderError(f"batch {self.seq}: on_batch must return a list of orders, got {_describe(result)}")
        cap = self.hello["caps"]["max_orders_per_batch"]
        if len(result) > cap:
            raise OrderError(
                f"batch {self.seq}: on_batch returned {len(result)} orders, over hello.caps.max_orders_per_batch ({cap}); "
                f"the harness rejects the orders past the cap and fails the run on a reply far over it, "
                f"so send at most {cap} per batch"
            )
        checked = [_check_order(o, f"batch {self.seq} order {i}") for i, o in enumerate(result)]
        taken = {o["client_id"] for o in checked if o["client_id"] is not None}
        orders = []
        for o in checked:
            cid = o["client_id"]
            if cid is None:
                cid = self._free_id(taken)
            taken.add(cid)
            amount_key = "quote_lamports" if o["side"] == "buy" else "token_amount"
            orders.append(
                {
                    "side": o["side"],
                    "client_id": cid,
                    "mint": o["mint"],
                    amount_key: o["amount"],
                    "max_slippage_bps": o["max_slippage_bps"],
                }
            )
        return orders

    def record_orders(self, orders):
        """Record orders as sent on the current batch: they stay in ``pending`` until their fill arrives."""
        for order in orders:
            self._used.add(order["client_id"])
            self.pending[order["client_id"]] = PendingOrder(order, self.seq, self.slot)

    def _free_id(self, taken):
        while True:
            cid = f"o{self._next}"
            self._next += 1
            if cid not in self._used and cid not in taken:
                return cid

    def _new_curve(self, mint, created_slot, slot):
        curve = Curve(mint, created_slot, slot)
        self.curves[mint] = curve
        return curve

    def _set_account(self, account):
        self.account = account
        self.cash = int(account["cash_lamports"])
        self.positions = {}
        for p in account["positions"]:
            tokens = int(p["token_amount"])
            if tokens > 0:
                self.positions[p["mint"]] = Holding(tokens, int(p["cost_lamports"]))


def _check_order(value, where):
    if not isinstance(value, dict):
        raise OrderError(f"{where}: expected an order dict, got {_describe(value)}")
    side = value.get("side")
    if side not in ("buy", "sell"):
        raise OrderError(f'{where}: side must be "buy" or "sell", got {_describe(side)}')
    allowed = _BUY_KEYS if side == "buy" else _SELL_KEYS
    for k in value:
        if k not in allowed:
            raise OrderError(f'{where}: unknown key {_describe(k)} on a {side} order')
    mint = value.get("mint")
    if not isinstance(mint, str) or not _PUBKEY_RE.fullmatch(mint):
        raise OrderError(f"{where}: mint must be a base58 address, got {_describe(mint)}")
    amount_key = "quote_lamports" if side == "buy" else "token_amount"
    amount = to_u64(value.get(amount_key), f"{where}: {amount_key}")
    bps = value.get("max_slippage_bps")
    if type(bps) is not int or not 0 <= bps <= 10_000:
        raise OrderError(f"{where}: max_slippage_bps must be an int 0..10000, got {_describe(bps)}")
    cid = value.get("client_id")
    if cid is not None and (not isinstance(cid, str) or not _CLIENT_ID_RE.fullmatch(cid)):
        raise OrderError(
            f"{where}: client_id must be 1-64 of [A-Za-z0-9_-] not starting with _ (reserved for the harness's closes), "
            f"got {_describe(cid)}"
        )
    return {"side": side, "client_id": cid, "mint": mint, "amount": amount, "max_slippage_bps": bps}


def to_u64(value, what="amount"):
    """An amount (int or u64 decimal string) as a canonical u64 decimal string; raises OrderError otherwise."""
    n = None
    if type(value) is int:
        n = value
    elif isinstance(value, str) and _U64_RE.fullmatch(value):
        n = int(value)
    if n is None or n < 0 or n > U64_MAX:
        raise OrderError(f"{what} must be a u64 (int or decimal string), got {_describe(value)}")
    return str(n)


def _describe(value):
    if isinstance(value, str):
        return json.dumps(value if len(value) <= 80 else value[:77] + "...")
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    return f"a {type(value).__name__}"


def read_lines(src, max_bytes=MAX_HARNESS_LINE_BYTES):
    """Yield the lines of a binary stream (newline-terminated, a trailing CR dropped) as strict UTF-8.

    A line longer than max_bytes raises ProtocolError; an unterminated last line is yielded.
    """
    line_no = 0
    while True:
        raw = src.readline(max_bytes + 1)
        if not raw:
            return
        line_no += 1
        if raw.endswith(b"\n"):
            raw = raw[:-1]
        elif len(raw) > max_bytes:
            raise ProtocolError(f"input line {line_no} is over {max_bytes} bytes")
        if raw.endswith(b"\r"):
            raw = raw[:-1]
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError(f"input line {line_no} is not valid UTF-8") from None
        yield text


class RunResult:
    """What run() returns after done: the final state, the done message and how many batches were answered."""

    __slots__ = ("state", "done", "batches")

    def __init__(self, state, done, batches):
        self.state = state
        self.done = done
        self.batches = batches


def encode_reply(seq, orders):
    """The reply line for a batch (no newline), byte for byte what sdk/ts/makefun-bot.ts writes."""
    return json.dumps({"type": "reply", "seq": seq, "orders": orders}, separators=(",", ":"), ensure_ascii=False)


def run(on_batch, on_hello=None, on_done=None, *, input=None, output=None):
    """Speak agent-fun/replay/v2 on stdin/stdout until done, then stop reading and return a RunResult.

    on_hello(state) runs once before the first batch, on_batch(state, batch) once per batch (its return
    value is the batch's orders), on_done(state, done) once at the end (no reply is sent). ``input`` is a
    binary stream (sys.stdin.buffer by default), ``output`` a text stream (sys.stdout by default),
    flushed after every reply. Raises ProtocolError, OrderError or whatever a callback raised; uncaught,
    that ends the process with exit code 1.
    """
    src = input if input is not None else sys.stdin.buffer
    out = output if output is not None else sys.stdout
    state = None
    batches = 0
    line_no = 0
    for line in read_lines(src):
        line_no += 1
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            raise ProtocolError(f"input line {line_no} is not JSON") from None
        kind = msg.get("type") if isinstance(msg, dict) else None
        if kind == "hello":
            if state is not None:
                raise ProtocolError(f"input line {line_no}: a second hello")
            if msg.get("protocol") != PROTOCOL:
                raise ProtocolError(f"the harness speaks {_describe(msg.get('protocol'))}; this shim speaks {PROTOCOL}")
            state = BotState(msg)
            if on_hello is not None:
                on_hello(state)
        elif kind == "batch":
            if state is None:
                raise ProtocolError(f"input line {line_no}: a batch before hello")
            _check_batch(msg, line_no)
            state.apply_batch(msg)
            orders = state.to_orders(on_batch(state, msg))
            text = encode_reply(msg["seq"], orders)
            size = len(text.encode("utf-8"))
            if size > MAX_BOT_LINE_BYTES:
                raise OrderError(
                    f"batch {msg['seq']}: the reply is {size} bytes (max {MAX_BOT_LINE_BYTES}); send fewer orders "
                    f"(caps.max_orders_per_batch is {state.hello['caps']['max_orders_per_batch']})"
                )
            state.record_orders(orders)
            out.write(text + "\n")
            out.flush()
            batches += 1
        elif kind == "done":
            if state is None:
                raise ProtocolError(f"input line {line_no}: done before hello")
            state.apply_done(msg)
            if on_done is not None:
                on_done(state, msg)
            return RunResult(state, msg, batches)
        else:
            raise ProtocolError(f"input line {line_no}: unknown message type {_describe(kind)}")
    if state is None:
        raise ProtocolError("input ended before hello")
    raise ProtocolError(f"input ended before done (after {batches} batches)")


def _check_batch(batch, line_no):
    account = batch.get("account")
    ok = (
        type(batch.get("seq")) is int
        and type(batch.get("slot")) is int
        and isinstance(batch.get("events"), list)
        and isinstance(batch.get("marks"), list)
        and isinstance(batch.get("fills"), list)
        and isinstance(account, dict)
        and isinstance(account.get("positions"), list)
    )
    if not ok:
        raise ProtocolError(f"input line {line_no}: malformed batch")
