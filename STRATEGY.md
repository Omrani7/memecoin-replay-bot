# Strategy

## What the bot measures
For every coin in the universe it keeps, from the batches alone: the latest post-trade reserves, real
SOL in the curve, the price and its all-time high, a 1,500-slot tape of trades (side, SOL, wallet),
and net token balances per wallet. Positions are valued at the batch's marks, which include our own
price impact, net of the 1.25% curve fee and the transaction fee.

## Entry
A coin that trades in the current batch is a candidate when:
- its curve holds between 50 and 83 SOL of real reserve (a curve completes at about 85 SOL);
- its price is within 20% of its all-time high;
- at least 5 distinct wallets bought it in the last 150 slots;
- it was not traded by the bot before (one entry per coin).

Candidates are taken deepest first while fewer than 5 positions are open or committed.

## Sizing
Each position targets 0.5 SOL, built over up to 60 slots, because a single reply may only spend 1% of
the curve's real SOL. 1 SOL of cash is never committed. Buys allow 5% slippage against the decision
price, so a buy that would land after a jump is refused (at no cost) instead of chasing.

## Exits
A position is sold whole, in one order with no slippage limit (an exit must not be refused):
- take-profit when its liquidation value is 8% over its cost;
- stop when it is 30% under its cost;
- after 3,000 slots in any case.
A curve that completes is closed by the replay at its final reserves.

## Guardrail probes
Until 26 coins carry a closed round trip of at least 0.01 SOL, the bot makes one probe every 400 slots:
a 0.012 SOL buy on a curve holding at least 5 SOL, sold on the next batch. On the dev window the probes
cost about 0.01 SOL in all.

## Why it may work
Curves that already hold 50+ SOL are close to completion. Buying usually keeps coming as they near
it, and a small target is often reached before the price turns. Completion closes the position at the
top of the curve.

## Known limitations
Research on the dev window (tools/, with an exact port of the replay engine) found no robust edge. The
average coin falls after every signal tried: momentum, flow, launch age, holder concentration,
copying profitable wallets, post-dump reversion and a gradient-boosting model. None of them covered
fees out of sample. This configuration was chosen on 24 of the kit's block surrogates, where it had the
best share of profitable copies (10 of 24) with a worst drawdown under 20%. Its mean there is still
slightly negative (about -0.2 SOL), and on the dev window itself it lost 0.92 SOL. It is a bet on
deep curves completing, not a proven edge. Rug pulls gap straight through the 30% stop: the worst
losses lose 80% of the position in one or two slots.
