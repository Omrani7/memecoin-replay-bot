"""Market state built from the replay's events: one CoinState per bonding curve in the universe.

Only what the batches carry is used: CreateEvents, TradeEvents (amounts, post-trade reserves, the
trader's wallet) and CompleteEvents, in delivery order. No clock, no randomness.
"""

from collections import deque

SOL = 1_000_000_000


class CoinState:
    __slots__ = (
        "mint", "created_slot", "creator", "vq", "vt", "rq", "rt", "last_slot", "complete", "n_trades",
        "tape", "ath", "ath_slot", "holders", "buyers", "first_price", "level_slots", "max_rq",
    )

    def __init__(self, mint, slot, creator, vq, vt, rt):
        self.mint = mint
        self.created_slot = slot
        self.creator = creator
        self.vq = vq
        self.vt = vt
        self.rq = 0
        self.rt = rt
        self.last_slot = slot
        self.complete = False
        self.n_trades = 0
        #: Recent trades: (slot, is_buy, lamports, user), oldest first, trimmed to TAPE_SLOTS.
        self.tape = deque()
        self.ath = vq / vt
        self.ath_slot = slot
        #: Net tokens per wallet over the coin's whole tape (buys minus sells).
        self.holders = {}
        self.buyers = 0
        self.first_price = vq / vt
        #: Slot at which real SOL first reached each tracked level (index into LEVELS).
        self.level_slots = {}
        self.max_rq = 0

    @property
    def price(self):
        return self.vq / self.vt

    def age(self, slot):
        return slot - self.created_slot


TAPE_SLOTS = 1500
LEVELS = tuple(x * SOL for x in (5, 10, 20, 30, 40, 50, 60, 70))


class Market:
    def __init__(self):
        self.coins = {}

    def apply(self, batch):
        """Fold a batch's events in; returns the coins that traded in it (each once, in event order)."""
        touched = []
        seen = set()
        slot = batch["slot"]
        for e in batch["events"]:
            kind = e["event"]
            mint = e["mint"]
            if kind == "CreateEvent":
                self.coins[mint] = CoinState(mint, slot, e.get("creator"), int(e["virtual_quote_reserves"]),
                                             int(e["virtual_token_reserves"]), int(e["real_token_reserves"]))
                continue
            c = self.coins.get(mint)
            if c is None:
                continue
            if kind == "CompleteEvent":
                c.complete = True
                continue
            if kind != "TradeEvent":
                continue
            sol = int(e["quote_amount"])
            tok = int(e["token_amount"])
            buy = bool(e["is_buy"])
            user = e.get("user")
            c.vq = int(e["virtual_quote_reserves"])
            c.vt = int(e["virtual_token_reserves"])
            c.rq = int(e["real_quote_reserves"])
            c.rt = int(e["real_token_reserves"])
            c.last_slot = slot
            c.n_trades += 1
            c.tape.append((slot, buy, sol, user))
            p = c.vq / c.vt
            if p > c.ath:
                c.ath = p
                c.ath_slot = slot
            h = c.holders.get(user, 0)
            if buy and h <= 0:
                c.buyers += 1
            c.holders[user] = h + tok if buy else h - tok
            if c.rq > c.max_rq:
                c.max_rq = c.rq
                for i, lv in enumerate(LEVELS):
                    if c.rq >= lv and i not in c.level_slots:
                        c.level_slots[i] = slot
            if mint not in seen:
                seen.add(mint)
                touched.append(c)
        for c in touched:
            tape = c.tape
            while tape and tape[0][0] < slot - TAPE_SLOTS:
                tape.popleft()
        return touched


def flow(c, slot, window):
    """(net lamports bought, buys, sells, distinct buyer wallets, largest single sell) over the last `window` slots."""
    net = 0
    nb = ns = 0
    big_sell = 0
    users = set()
    lo = slot - window
    for s, buy, sol, user in reversed(c.tape):
        if s < lo:
            break
        if buy:
            net += sol
            nb += 1
            users.add(user)
        else:
            net -= sol
            ns += 1
            if sol > big_sell:
                big_sell = sol
    return net, nb, ns, len(users), big_sell


def top_holder_share(c, exclude=()):
    """Largest wallet's share of the tokens held outside the curve (by the tape's net flows)."""
    circ = 0
    top = 0
    for u, v in c.holders.items():
        if v <= 0 or u in exclude:
            continue
        circ += v
        if v > top:
            top = v
    return top / circ if circ > 0 else 0.0
