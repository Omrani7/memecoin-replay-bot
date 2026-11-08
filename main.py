"""make.fun memecoin-backtest-48h-01 entry: agent-fun/replay/v2 bot (Python 3.13, standard library only).

Run by the grader as `python3 -P -s /bot/main.py` with PYTHONHASHSEED=0. -P leaves this directory off
sys.path, so it is added here, with vendor/ (the kit's SDK shim, byte for byte).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.join(HERE, "vendor")):
    if path not in sys.path:
        sys.path.insert(0, path)

import makefun_bot  # noqa: E402
from tradebot.strategy import Strategy  # noqa: E402


def main():
    strategy = Strategy()
    makefun_bot.run(strategy.on_batch)


if __name__ == "__main__":
    main()
