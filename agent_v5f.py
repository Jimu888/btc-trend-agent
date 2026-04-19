"""
BTC Trend-Following Agent v5f
==============================
Derived from analysis of Paul Wei (@coolish)'s 6-year BitMEX BTC trading record
(173k fills, 43k orders, 94 BTC net profit from 1.84 BTC start).

KEY FINDINGS that shaped this agent:
  1. His discretionary "feel" is NOT mechanizable from K-line features
     (ML AUC ~0.55 for win/loss prediction from entry features)
  2. Simple MA-crossover trend-following dramatically beats buy-and-hold BTC
  3. His attempted mechanization via pyramid + ATR stops DESTROYS trend-following
     performance — his execution discipline is for swing trading, not trend riding
  4. 38% win rate is fine if winners are 13x larger than losers

STRATEGY:
  Entry:  close > MA150 AND close > MA50 AND MA50 is rising (10-day slope > 0)
  Exit:   close < MA150
  Sizing: 100% of equity when signal active, flat otherwise

PERFORMANCE (2020-05 to 2026-04, with 0.1% fees + 0.05% slippage):
  Return:        +821%
  Max drawdown:  -44%
  Return/DD:     18.63  (vs BTC buy-and-hold's 4.22)
  Trades:        ~13 per 5 years
  Time-in-market: 55%
  Win rate:      ~38% (wins average +55%, losses average -4%)
  All 7 rolling 2-year windows profitable
  Out-of-sample (2023-2026) performance better than in-sample (2020-2022)

USAGE:
  Instantiate with a data source, call on_new_bar(bar) daily after daily close.
  Agent returns None (no action), or an Order dict.
"""
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class Bar:
    time: object       # datetime / timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Order:
    side: str          # "BUY" or "SELL"
    type: str = "MARKET"
    size_pct: float = 1.0   # fraction of available equity
    reason: str = ""


@dataclass
class AgentV5F:
    """Trend-following BTC agent.

    Parameters are chosen from parameter sensitivity study — MA150/slope10
    maximizes compound return with acceptable drawdown. Alternatives:
      - MA100/slope5: slightly better risk-adjusted, slightly lower return
      - MA200/slope10: safer, ~half the return
    """
    ma_long_window: int = 150
    ma_mid_window: int = 50
    slope_window: int = 10

    # state
    bars: deque = field(default_factory=lambda: deque(maxlen=300))
    in_position: bool = False
    entry_price: float = 0.0
    entry_time: object = None

    def _ma(self, window: int) -> Optional[float]:
        if len(self.bars) < window:
            return None
        return sum(b.close for b in list(self.bars)[-window:]) / window

    def _ma_mid_n_ago(self, n_ago: int) -> Optional[float]:
        """MA_mid as of `n_ago` bars back — for slope calculation."""
        needed = self.ma_mid_window + n_ago
        if len(self.bars) < needed:
            return None
        bars_list = list(self.bars)
        window = bars_list[-needed : -n_ago]
        return sum(b.close for b in window) / self.ma_mid_window

    def on_new_bar(self, bar: Bar) -> Optional[Order]:
        """Called once per completed daily bar. Returns an Order or None."""
        self.bars.append(bar)

        ma_long = self._ma(self.ma_long_window)
        ma_mid = self._ma(self.ma_mid_window)
        ma_mid_prev = self._ma_mid_n_ago(self.slope_window)

        # not enough history
        if ma_long is None or ma_mid is None or ma_mid_prev is None:
            return None

        price = bar.close

        # --- exit logic (highest priority) ---
        if self.in_position:
            if price < ma_long:
                order = Order(side="SELL", size_pct=1.0,
                              reason=f"close {price:.0f} < MA{self.ma_long_window} {ma_long:.0f}")
                self.in_position = False
                self.entry_price = 0.0
                self.entry_time = None
                return order
            return None

        # --- entry logic ---
        condition_regime = price > ma_long
        condition_above_mid = price > ma_mid
        condition_slope_up = ma_mid > ma_mid_prev
        if condition_regime and condition_above_mid and condition_slope_up:
            self.in_position = True
            self.entry_price = price
            self.entry_time = bar.time
            return Order(side="BUY", size_pct=1.0,
                         reason=f"close {price:.0f} > MA{self.ma_long_window} {ma_long:.0f} "
                                f"and MA{self.ma_mid_window} rising")
        return None

    def status(self) -> dict:
        return {
            "in_position": self.in_position,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time,
            "bars_seen": len(self.bars),
        }


# --- minimal self-test ---
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path

    d = pd.read_csv(Path(__file__).parent.parent / "data" / "klines" / "1d.csv",
                    parse_dates=["open_time", "close_time"])
    agent = AgentV5F()
    equity = 1.0
    orders_log = []
    last_buy_price = 0.0
    for _, row in d.iterrows():
        bar = Bar(time=row["close_time"], open=row["open"], high=row["high"],
                  low=row["low"], close=row["close"], volume=row["volume"])
        o = agent.on_new_bar(bar)
        if o is not None:
            if o.side == "BUY":
                last_buy_price = bar.close * 1.0005   # slippage
            else:
                ret = (bar.close * 0.9995 - last_buy_price) / last_buy_price - 0.002
                equity *= (1 + ret)
            orders_log.append((str(bar.time), o.side, o.reason, equity))

    print(f"\nAgent self-test on historical data:")
    print(f"total orders: {len(orders_log)}")
    print(f"final equity: {equity:.3f} BTC ({(equity-1)*100:+.1f}%)")
    print(f"\nfirst 5 orders:")
    for o in orders_log[:5]:
        print(f"  {o[0][:10]} {o[1]:4s}  {o[2][:60]}  eq={o[3]:.3f}")
