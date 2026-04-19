# Strategy Specification: BTC Trend v5f

Formal specification in three equivalent forms. Use whichever matches your
execution platform's expected input format.

---

## 1. Natural Language

Every day, shortly after the UTC 00:00 daily close:

1. Fetch the last ~200 daily OHLC bars for BTC/USD (or BTC/USDT)
2. Compute the 150-day simple moving average (MA150) from closing prices
3. Compute the 50-day simple moving average (MA50)
4. Compute MA50 from 10 days ago (for slope)
5. **If currently flat (no position)**: enter a long position at market if ALL three conditions hold:
   - Close > MA150
   - Close > MA50
   - MA50(today) > MA50(10 days ago)
6. **If currently long**: exit the long position at market if:
   - Close < MA150
7. Otherwise: do nothing, wait until tomorrow

Position size is 100% of trading capital when in position, 0% when flat.
Never short. Never pyramid. Never add on loss. No stop-loss or take-profit.

---

## 2. Pseudocode

```
function on_daily_close(bars):
    if len(bars) < 160:
        return NONE            # not enough history

    close     = bars.last.close
    MA150     = mean(bars[-150:].close)
    MA50      = mean(bars[-50:].close)
    MA50_prev = mean(bars[-60:-10].close)

    if in_position:
        if close < MA150:
            return SELL(size=100%, reason="close < MA150")
        return NONE
    else:
        if close > MA150 and close > MA50 and MA50 > MA50_prev:
            return BUY(size=100%, reason="all three conditions met")
        return NONE
```

## 3. Python (canonical implementation)

See [`agent_v5f.py`](agent_v5f.py) in this repo. Core snippet:

```python
price = bar.close
ma_long     = mean(bars[-150:])              # 150-day SMA
ma_mid      = mean(bars[-50:])               # 50-day SMA
ma_mid_prev = mean(bars[-60:-10])            # 50-day SMA, 10 days back

if in_position:
    if price < ma_long:
        emit(Order("SELL", size_pct=1.0))
        in_position = False
else:
    if price > ma_long and price > ma_mid and ma_mid > ma_mid_prev:
        emit(Order("BUY", size_pct=1.0))
        in_position = True
```

---

## Parameters

All parameters are *intentionally few* and *empirically validated*:

| Parameter | Value | Justification |
|---|---|---|
| `ma_long_window` | 150 days | Optimal by return/drawdown in sensitivity grid across {100, 150, 200, 250, 300} |
| `ma_mid_window` | 50 days | Standard; also appears as filter in breakout systems |
| `slope_window` | 10 days | Standard for MA slope detection; tested {5, 10, 20, 30} |

Alternative validated parameter sets (use if you have specific preferences):

| Alias | Params | Return | Max DD | Trades |
|---|---|---|---|---|
| v5f (default) | MA150 / slope10 | +821% | -44% | 15 |
| v5e | MA200 / slope10 | +418% | -58% | 13 |
| v5_short | MA100 / slope5 | +745% | -38% | 20 |

`v5_short` has slightly lower absolute return but better risk-adjusted performance.
Do **not** use MA < 100 or MA > 300; these fall outside the validated range.

---

## Timeframe portability

Tested on 1h, 4h, daily, weekly. **Scales by calendar duration**, not by bar count:

| Timeframe | ma_long (bars) | Equivalent calendar | Return | Max DD |
|---|---|---|---|---|
| 1 week | 21 | ~147 days | +746% | -33% |
| 1 day (default) | 150 | 150 days | +821% | -44% |
| 4 hour | 900 | 150 days | +881% | -45% |
| 1 hour | 3600 | 150 days | +683% | -45% |

**Do not apply** at sub-hour timeframes (5m, 1m). Trend-following has no statistical edge there; fees and noise dominate. See `analysis/scripts/26_test_5m_recent.py` for empirical proof.

---

## Execution rules

- **Order type**: market orders (simplicity and reliability > marginal slippage savings)
- **Decision time**: once per completed daily bar. Ideally within 15 minutes of the 00:00 UTC daily close.
- **Slippage allowance**: 0.05% expected; strategy robustness tolerates up to ~0.2%.
- **Fees assumed**: 0.1% per side (typical futures taker). Higher fees reduce the +821% figure proportionally.

---

## What NOT to do

These modifications have been tested and shown to *degrade* performance. Do not add them:

- ❌ Intraday stop-loss (gets hit by normal BTC volatility, kills winners)
- ❌ Take-profit targets (cap upside; the strategy's edge is in holding trend winners)
- ❌ Pyramid additions (no measurable benefit; adds execution risk)
- ❌ Martingale / averaging down (catastrophic; historically the biggest leak)
- ❌ Short positions (no edge; long-only has all the profit)
- ❌ Multiple timeframe confirmations (no measurable benefit; adds lag)
- ❌ Sentiment / news / social signals (not in the training data; adds noise)
- ❌ Changing params based on recent performance (overfits to the most recent regime)

---

## Safety gates (for live deployment)

Independent of the core logic, a production harness should enforce:

1. **Data sanity**: no null bars, no timestamp gaps > 1 day, no intrabar price jump > 30%
2. **Circuit breakers**:
   - Halt if 7-day drawdown > 20% (suggests broken data or regime change)
   - Halt if 5 consecutive losing trades > 5% each
3. **Connection health**: if exchange API fails 3× consecutively, halt and alert
4. **State persistence**: `in_position` flag must survive restarts. Never assume "fresh start = flat".

---

## Leverage warning

All metrics above are for **1× (spot-equivalent) leverage**. If deploying on margined products:

- At 2×: theoretically doubles return, but max intrabar adverse excursion in historical trades reached -18.8% — 2× would survive
- At 3-4×: increasingly risky; funding costs compound against the position
- **At 5× or higher**: historical data has an intrabar MAE of -18.8%, whereas 5× leverage liquidates at ~-19.5%. You are 1.2 percentage points from liquidation on the worst historical trade. Future adverse moves will likely exceed this. Funding costs at 5× erase 300%+ of equity over the 6-year in-market duration. **Not recommended.**

See [`analysis/scripts/27_leverage_risk.py`](analysis/scripts/27_leverage_risk.py) for full simulation.
