# memecoin-replay-bot

A paper-trading bot for the make.fun bounty `memecoin-backtest-48h-01`: it trades a replay of pump.fun
bonding curves (agent-fun/replay/v2) starting from 10 SOL. Python 3.13, standard library only.

- `main.py`: the entrypoint the grader runs (`python3 -P -s /bot/main.py`, PYTHONHASHSEED=0)
- `tradebot/market.py`: per-coin state built from the batches (reserves, tape, flows, holders)
- `tradebot/strategy.py`: entries, sizing, exits, risk limits, guardrail probes, all parameters in `DEFAULT_PARAMS`
- `vendor/makefun_bot.py`: the kit's Python SDK shim, a byte-for-byte copy
- `tools/`: research only, never imported by the bot:
  - `replay.py`: a Python port of the kit's replay engine (it matches the official grader exactly on house bot action logs, see `verify.py`)
  - `run.py`: parameter runs and grids on the dev window, its two halves and kept block surrogates
  - `explore*.py`, `features.py`, `smart_wallets.py`, `micro.py`, `ml.py`: the signal studies
- `tests/`: `python -m unittest discover -s tests` (set `MAKEFUN_KIT` to an unpacked kit for the dev-window tests)

See STRATEGY.md for what the bot does and what the research found.
