"""Phase 4.4: v2 strategy + backtest.

v2 encodes the findings of phases 1-4.3:
  ENTRY (simple, mechanical — AUC shows entry is weakly predictive anyway):
    1. Volatility gate: h4_atr14_pct > 1.5%
    2. Regime: bull (close > MA200 and MA50 rising) OR bear (close < MA200 and MA50 falling)
    3. Setup A (BULL_PULLBACK_LONG): bull regime AND h4 pulled back 1-5% from 24h high
                                    AND d_dist_ath > -30% AND d_rsi14 35-65
    4. Setup B (BEAR_RALLY_SHORT): bear regime AND h4 rallied 1-5% from 24h low
                                   AND d_rsi14 35-65
    5. FORBIDDEN: bear regime + long direction; d_rsi14 > 75; d_pos_in_20d_range > 0.9
  SIZE / SCALE-IN:
    6. Initial 60% of target size at trigger
    7. Additional 20% at -0.5% only if not in broader loss — ONE TIME ONLY
    8. Additional 20% at +1% profit (pyramid, per finding 2)
    9. Never add if current unrealized < -2%  (no martingale)
  EXIT:
    10. Hard stop: -5% unrealized (from avg entry)
    11. Profit protect: once unrealized >= +3%, raise stop to breakeven
    12. Trail stop: once unrealized >= +5%, trail at 2% below best price
    13. Time stop: close at T+168h (7 days)
    14. Volatility collapse: close if h4_atr14_pct drops below 0.8% (alpha gone)

Backtest on BTCUSDT 4h klines, 2020-05-01 to 2026-04-18.
Only one concurrent position. BTC-denominated PnL (inverse-like).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

# --- load data ---
d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
h4 = pd.read_csv(KL / "4h.csv", parse_dates=["open_time", "close_time"])

for df in [d1, h4]:
    c = df["close"]
    df["ma20"]  = c.rolling(20).mean()
    df["ma50"]  = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    tr = pd.concat([df["high"]-df["low"],
                    (df["high"]-c.shift(1)).abs(),
                    (df["low"]-c.shift(1)).abs()], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr14"] / c
    d = c.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    df["high20"] = df["high"].rolling(20).max()
    df["low20"]  = df["low"].rolling(20).min()

# daily all-time high at each point (cumulative max)
d1["ath"] = d1["high"].cummax()
d1["ret_7"] = d1["close"].pct_change(7)

# helper: get latest closed daily bar at a given 4h close_time
def latest_d(h4_close_time):
    idx = d1["close_time"].searchsorted(h4_close_time, side="right") - 1
    if idx < 200: return None
    return d1.iloc[idx]

# --- backtest loop ---
# One open position at a time.
START = pd.Timestamp("2020-10-01", tz="UTC")   # allow warm-up for MA200
end_time = h4["close_time"].iloc[-1]

trades = []  # closed trades log
open_pos = None

for _, bar in h4[h4["close_time"] >= START].iterrows():
    t = bar["close_time"]
    price = bar["close"]
    if pd.isna(bar["atr_pct"]) or pd.isna(bar["ma200"]) or pd.isna(bar["rsi14"]):
        continue
    d = latest_d(t)
    if d is None: continue

    # --- update open position state ---
    if open_pos is not None:
        sign = 1 if open_pos["direction"] == "long" else -1
        unreal = sign * (price - open_pos["avg_entry"]) / open_pos["avg_entry"]

        # track high/low of favor
        fav = sign * (bar["high" if sign == 1 else "low"] - open_pos["avg_entry"]) / open_pos["avg_entry"]
        adv = sign * (bar["low" if sign == 1 else "high"] - open_pos["avg_entry"]) / open_pos["avg_entry"]
        open_pos["mfe"] = max(open_pos["mfe"], fav)
        open_pos["mae"] = min(open_pos["mae"], adv)

        # upgrade stop based on profit
        if open_pos["mfe"] >= 0.05 and not open_pos["trailing"]:
            open_pos["trailing"] = True
        if open_pos["mfe"] >= 0.03 and open_pos["stop_pct"] < 0:
            open_pos["stop_pct"] = 0.0   # breakeven

        # trail stop
        if open_pos["trailing"]:
            target_stop = open_pos["mfe"] - 0.02
            if target_stop > open_pos["stop_pct"]:
                open_pos["stop_pct"] = target_stop

        exit_reason = None
        exit_price = None

        # check stops in priority order
        hit_stop_adv = adv <= open_pos["stop_pct"] - 1e-9   # if adv less than threshold, triggered
        # convert stop_pct to a price level
        stop_price = open_pos["avg_entry"] * (1 + sign * open_pos["stop_pct"])
        # for a LONG: stop triggers when LOW <= stop_price. stop_pct is stored as
        # the unrealized% threshold (e.g., -0.05 for -5%). we detect intrabar.
        intrabar_stop_hit = ((sign == 1 and bar["low"]  <= stop_price) or
                             (sign == -1 and bar["high"] >= stop_price))
        if intrabar_stop_hit:
            exit_price = stop_price
            exit_reason = "stop"

        # time stop
        hours_in = (t - open_pos["entry_time"]).total_seconds() / 3600
        if exit_reason is None and hours_in >= 168:
            exit_price = price
            exit_reason = "time"

        # volatility collapse
        if exit_reason is None and bar["atr_pct"] < 0.008:
            exit_price = price
            exit_reason = "vol_collapse"

        # scale-in event (ADD at pyramid point +1% only if not triggered stop/time/vol)
        if exit_reason is None and not open_pos["pyramid_done"] and unreal >= 0.01 and open_pos["adds"] < 1:
            open_pos["avg_entry"] = (open_pos["avg_entry"] * open_pos["size_frac"] + price * 0.20) / (open_pos["size_frac"] + 0.20)
            open_pos["size_frac"] += 0.20
            open_pos["adds"] += 1
            open_pos["pyramid_done"] = True

        if exit_reason is not None:
            pnl_ret = sign * (exit_price - open_pos["avg_entry"]) / open_pos["avg_entry"]
            # inverse contract BTC pnl (per unit notional entered, treat size_frac as notional fraction)
            # assume 1 BTC of equity-equivalent sizing at target = 1.0; trade uses size_frac BTC
            trades.append({
                "entry_time": open_pos["entry_time"],
                "exit_time": t,
                "direction": open_pos["direction"],
                "setup": open_pos["setup"],
                "avg_entry": open_pos["avg_entry"],
                "exit_price": exit_price,
                "size_frac": open_pos["size_frac"],
                "mfe": open_pos["mfe"],
                "mae": open_pos["mae"],
                "return_pct": pnl_ret,
                "exit_reason": exit_reason,
                "hours": hours_in,
                "adds": open_pos["adds"],
            })
            open_pos = None

    # --- look for new entry ---
    if open_pos is not None:
        continue
    if bar["atr_pct"] < 0.015:   # vol gate
        continue

    bull = d["close"] > d["ma200"]
    # MA50 slope (10d)
    idx_d = d1["close_time"].searchsorted(t, side="right") - 1
    ma50_now = d1.iloc[idx_d]["ma50"]
    ma50_prev = d1.iloc[idx_d - 10]["ma50"] if idx_d >= 210 else None
    if ma50_prev is None or pd.isna(ma50_prev) or ma50_prev == 0:
        continue
    ma50_slope = (ma50_now - ma50_prev) / ma50_prev
    bull_trend = bull and ma50_slope > 0.005
    bear_trend = (not bull) and ma50_slope < -0.005

    # forbidden zones
    if d["rsi14"] > 75: continue
    if bar["high20"] > 0:
        pos_20 = (price - bar["low20"]) / (bar["high20"] - bar["low20"]) if bar["high20"] > bar["low20"] else 0.5
    else:
        continue
    if pos_20 > 0.9:
        continue

    dist_ath = (price - d["ath"]) / d["ath"]

    # Setup A: BULL_PULLBACK_LONG
    # conditions: bull_trend, near ATH (>-30%), h4 pulled back from 20-bar high (5-15%)
    dist_h4_high20 = (price - bar["high20"]) / bar["high20"]
    if (bull_trend and dist_ath > -0.30 and -0.15 <= dist_h4_high20 <= -0.02 and
        35 <= d["rsi14"] <= 65):
        open_pos = {
            "entry_time": t, "direction": "long", "setup": "bull_pullback_long",
            "avg_entry": price, "size_frac": 0.60, "mfe": 0.0, "mae": 0.0,
            "stop_pct": -0.05, "trailing": False, "adds": 0, "pyramid_done": False,
        }
        continue

    # Setup B: BEAR_RALLY_SHORT
    dist_h4_low20 = (price - bar["low20"]) / bar["low20"]
    if (bear_trend and 0.02 <= dist_h4_low20 <= 0.15 and
        35 <= d["rsi14"] <= 65):
        open_pos = {
            "entry_time": t, "direction": "short", "setup": "bear_rally_short",
            "avg_entry": price, "size_frac": 0.60, "mfe": 0.0, "mae": 0.0,
            "stop_pct": -0.05, "trailing": False, "adds": 0, "pyramid_done": False,
        }

tr = pd.DataFrame(trades)
print(f"trades: {len(tr)}")
print(f"wins: {(tr['return_pct']>0).sum()}  ({(tr['return_pct']>0).mean()*100:.1f}%)")
print(f"mean return: {tr['return_pct'].mean()*100:+.2f}%  median: {tr['return_pct'].median()*100:+.2f}%")
print(f"sum of returns (per unit size): {tr['return_pct'].sum()*100:+.1f}%")
# size-weighted sum (if size_frac is 0.6 or 0.8 — approximate notional deployed)
tr["btc_pnl_per_equity_unit"] = tr["return_pct"] * tr["size_frac"]
print(f"size-weighted pnl sum: {tr['btc_pnl_per_equity_unit'].sum()*100:+.1f}% "
      f"(per unit of allocated equity per trade)")

# compare year-by-year vs Paul
tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year
yr = tr.groupby("year").agg(
    n=("return_pct", "size"),
    win_rate=("return_pct", lambda s: (s > 0).mean()),
    sum_ret_pct=("return_pct", "sum"),
    avg_ret_pct=("return_pct", "mean"),
)
print("\nby year:")
print(yr.round(4).to_string())

print("\nby setup:")
print(tr.groupby("setup").agg(
    n=("return_pct", "size"),
    win=("return_pct", lambda s: (s > 0).mean()),
    sum_ret=("return_pct", "sum"),
    avg_ret=("return_pct", "mean"),
).round(4).to_string())

print("\nby exit reason:")
print(tr.groupby("exit_reason").agg(
    n=("return_pct", "size"),
    win=("return_pct", lambda s: (s > 0).mean()),
    avg_ret=("return_pct", "mean"),
    sum_ret=("return_pct", "sum"),
).round(4).to_string())

# convert to BTC-ish equivalent: Paul made 41 BTC net on XBTUSD.
# our simulation is per-unit of notional deployed, not per-BTC-of-equity.
# rough compounded BTC-equivalent return with 1 BTC equity and trade sizing = size_frac BTC:
eq = 1.0
for _, row in tr.iterrows():
    eq *= (1 + row["return_pct"] * row["size_frac"])
print(f"\ncompounded 1 BTC -> {eq:.3f} BTC ({(eq-1)*100:+.1f}%) over {len(tr)} trades")
print(f"Paul's XBTUSD net: ~41 BTC on ~90 BTC avg equity")

# drawdown check
tr_s = tr.sort_values("entry_time").copy()
tr_s["cum"] = (1 + tr_s["return_pct"] * tr_s["size_frac"]).cumprod()
peak = tr_s["cum"].cummax()
dd = (tr_s["cum"] - peak) / peak
print(f"max drawdown: {dd.min()*100:.1f}%")

tr.to_csv(DATA / "derived" / "backtest_v2_trades.csv", index=False)
print(f"\nsaved -> {DATA / 'derived' / 'backtest_v2_trades.csv'}")
