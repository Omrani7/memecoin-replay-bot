# Strategy: sprint to graduation

## The edge
A pump.fun curve that completes (about 85 SOL of real reserve) is closed by the replay at its final reserves, as
recorded: the top of the curve, whatever our own buys did to it. A coin bought at 30 SOL that completes sells at
roughly 4x the price. Most coins never complete: on the dev window 7 of 1,222 did, and nearly every coin that
reaches 30 SOL and fails crashes back to a fraction of that price. So the bot is a bet on telling the coming
completions from the rugs early, and on cutting the rugs fast.

## Entry (one per coin)
A coin that trades in the batch is a candidate when:
- its curve holds 30 to 80 SOL of real reserve, and it got there within 60 slots of its launch (a sprint);
- at least 25 distinct wallets bought it over the last 200 slots (broad demand, not a handful of wallets);
- the wallets that bought in its launch slot or the next hold at most half of the tokens outside the curve.
  Coins whose supply sits with a launch bundle were, on the dev window, a family of near-identical rugs;
- its price is within 30% of its all-time high.

## Sizing
1 SOL per position, built over up to 300 slots under the 1%-of-real-reserve cap per reply. At most 5 positions,
1 SOL of cash never committed.

## Exits
- No take-profit: a completing curve is closed at its top by the replay.
- Stop at 25% under cost (rugs can gap through it); a 10,000-slot time limit.
- Circuit breaker: once equity is 30% under its peak, the bot opens and adds nothing more, so a bad day stays
  well inside the 50% drawdown bar.

## Guardrail probes
Until 26 coins carry a closed round trip of at least 0.01 SOL, one 0.012 SOL probe every 400 slots, sold on the
next batch (about 0.01 SOL in all).

## Evidence (tools/run.py, the exact Python port of the replay engine)
- Dev window: +8.03 SOL, max drawdown 12.5%: 5 completions at about +2.2 SOL each, 10 stopped entries at about
  -0.33 SOL each. It breaks even at about one completion in eight entries.
- Its halves, as separate replays: -0.56 SOL (two entries, no completion) and +6.25 SOL.
- 40 block surrogates (other coins' flow spliced into blocks, so they break real coin life cycles): median -1.17,
  13 of 40 positive, worst drawdown 33.5%, at least 30 qualifying coins in every copy.

## Known limitations
One day holds seven completions: the filters are fitted to a small sample. The bet is that their logic (fast, broad,
unbundled demand) carries over; on a day with no such completion the bot loses a few tenths of a SOL per entry.
