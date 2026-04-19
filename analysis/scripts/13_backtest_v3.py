"""v3 strategy — fixes from v2 failure.

v2 failed because:
  - Fixed -5% stop killed by normal 4h volatility (95% trades stopped out)
  - Too many entries in chop (307 trades / 5.5yr)
  - No requirement for confirmed trend

v3 changes:
  A. ATR-based stop (2 x daily ATR) instead of fixed percentage
  B. Daily-timeframe trigger (not 4h) to filter chop
  C. Double-confirmation entry: pullback + RSI reset + vol gate
  D. Skip shorts in bull regime, skip longs in bear regime (hard rule from regime table)
  E. Add "take 1/3 profit at +1x ATR, 1/3 at +2x ATR, trail 1/3"
  F. Minimum 10-day cooloff between trades in same direction (prevent over-trading)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
d1["ma20"] = c.rolling(20).mean()
d1["ma50"] = c.rolling(50).mean()
d1["ma200"] = c.rolling(200).mean()
tr = pd.concat([d1["high"]-d1["low"], (d1["high"]-c.shift(1)).abs(),
                (d1["low"]-c.shift(1)).abs()], axis=1).max(axis=1)
d1["atr14"] = tr.rolling(14).mean()
d1["atr_pct"] = d1["atr14"] / c
delta = c.diff()
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
d1["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
d1["high20"] = d1["high"].rolling(20).max()
d1["low20"]  = d1["low"].rolling(20).min()
d1["ath"] = d1["high"].cummax()
d1["ret_7"] = c.pct_change(7)
d1["ma50_slope"] = (d1["ma50"] - d1["ma50"].shift(10)) / d1["ma50"].shift(10)

# for trailing exit we need intra-trade path — use daily high/low and forward loop
trades = []
open_pos = None
last_long_exit = None
last_short_exit = None
START_IDX = 200   # skip warm-up

for i in range(START_IDX, len(d1)):
    b = d1.iloc[i]
    if any(pd.isna([b["ma200"], b["ma50"], b["atr14"], b["rsi14"]])):
        continue

    # --- manage open position intraday (using today's high/low) ---
    if open_pos is not None:
        sign = 1 if open_pos["direction"] == "long" else -1
        # MFE & MAE update
        fav_p = b["high"] if sign == 1 else b["low"]
        adv_p = b["low"]  if sign == 1 else b["high"]
        fav = sign * (fav_p - open_pos["avg_entry"]) / open_pos["avg_entry"]
        adv = sign * (adv_p - open_pos["avg_entry"]) / open_pos["avg_entry"]
        open_pos["mfe"] = max(open_pos["mfe"], fav)
        open_pos["mae"] = min(open_pos["mae"], adv)

        exit_price = None
        exit_reason = None

        # 1) stop: ATR-based
        stop_dist = open_pos["stop_atr_mult"] * open_pos["atr_at_entry"] / open_pos["avg_entry"]
        # fixed initial stop is at avg_entry - sign*stop_dist
        # after +1 ATR of profit, move to breakeven
        # after +2 ATR, trail at 1 ATR
        if open_pos["mfe"] >= 1.0 * stop_dist and open_pos["stop_pct"] < 0:
            open_pos["stop_pct"] = 0.0
        if open_pos["mfe"] >= 2.0 * stop_dist:
            new_trail = open_pos["mfe"] - stop_dist
            if new_trail > open_pos["stop_pct"]:
                open_pos["stop_pct"] = new_trail

        stop_price = open_pos["avg_entry"] * (1 + sign * open_pos["stop_pct"])
        if sign == 1 and b["low"] <= stop_price:
            exit_price = stop_price
            exit_reason = "stop"
        elif sign == -1 and b["high"] >= stop_price:
            exit_price = stop_price
            exit_reason = "stop"

        # 2) partial take-profits (only once each)
        if exit_reason is None:
            # 1/3 off at +1 ATR
            if not open_pos["tp1_done"] and open_pos["mfe"] >= stop_dist:
                open_pos["size_frac"] *= 2/3
                open_pos["tp1_done"] = True
                open_pos["realized_partial"] = open_pos.get("realized_partial", 0.0) + (1/3) * stop_dist
            # 1/3 off at +2 ATR
            if not open_pos["tp2_done"] and open_pos["mfe"] >= 2 * stop_dist:
                open_pos["size_frac"] *= 0.5   # from 2/3 to 1/3
                open_pos["tp2_done"] = True
                open_pos["realized_partial"] = open_pos.get("realized_partial", 0.0) + (1/3) * 2 * stop_dist

        # 3) time stop
        hours_in = (b["close_time"] - open_pos["entry_time"]).total_seconds() / 3600
        if exit_reason is None and hours_in >= 168:
            exit_price = b["close"]
            exit_reason = "time"

        # 4) volatility collapse
        if exit_reason is None and b["atr_pct"] < 0.015:
            exit_price = b["close"]
            exit_reason = "vol_collapse"

        if exit_reason is not None:
            pnl_ret_remain = sign * (exit_price - open_pos["avg_entry"]) / open_pos["avg_entry"]
            total_pnl = open_pos.get("realized_partial", 0.0) + pnl_ret_remain * open_pos["size_frac"]
            trades.append({
                "entry_time": open_pos["entry_time"],
                "exit_time": b["close_time"],
                "direction": open_pos["direction"],
                "setup": open_pos["setup"],
                "avg_entry": open_pos["avg_entry"],
                "exit_price": exit_price,
                "total_pnl_ret": total_pnl,
                "mfe": open_pos["mfe"],
                "mae": open_pos["mae"],
                "tp1": open_pos["tp1_done"],
                "tp2": open_pos["tp2_done"],
                "exit_reason": exit_reason,
                "hours": hours_in,
                "atr_pct_at_entry": open_pos["atr_at_entry"] / open_pos["avg_entry"],
            })
            if open_pos["direction"] == "long":
                last_long_exit = b["close_time"]
            else:
                last_short_exit = b["close_time"]
            open_pos = None

    if open_pos is not None:
        continue

    # --- new entry ---
    price = b["close"]
    # hard gates
    if b["atr_pct"] < 0.02: continue           # at least 2% daily ATR
    if b["rsi14"] > 75 or b["rsi14"] < 25: continue
    if b["high20"] > b["low20"]:
        pos_20 = (price - b["low20"]) / (b["high20"] - b["low20"])
    else:
        pos_20 = 0.5
    if pos_20 > 0.9: continue

    bull = (price > b["ma200"]) and (b["ma50_slope"] > 0.01)
    bear = (price < b["ma200"]) and (b["ma50_slope"] < -0.01)
    dist_ath = (price - b["ath"]) / b["ath"]

    # cooloff
    if last_long_exit is not None and (b["close_time"] - last_long_exit).days < 10:
        bull = False   # block longs during cooloff
    if last_short_exit is not None and (b["close_time"] - last_short_exit).days < 10:
        bear = False

    # Setup A: bull pullback long
    # requires: bull regime, close to ATH, RSI reset below 55 after being >60 recently,
    # daily close is a pullback bar (red or small body).
    if bull and dist_ath > -0.30 and 40 <= b["rsi14"] <= 58:
        # recent RSI history: did RSI hit 60+ in last 10 days?
        recent_rsi = d1.iloc[max(0, i-10):i]["rsi14"].max()
        # recent pullback: 7d return between -10% and 0%
        if pd.notna(recent_rsi) and recent_rsi >= 60 and -0.10 <= b["ret_7"] <= 0.01:
            open_pos = {
                "entry_time": b["close_time"], "direction": "long",
                "setup": "bull_pullback_long_daily",
                "avg_entry": price, "atr_at_entry": b["atr14"],
                "size_frac": 1.0, "mfe": 0.0, "mae": 0.0,
                "stop_pct": -2.0 * b["atr14"] / price, "stop_atr_mult": 2.0,
                "tp1_done": False, "tp2_done": False,
            }
            continue

    # Setup B: bear rally short
    if bear and 40 <= b["rsi14"] <= 58:
        recent_rsi_min = d1.iloc[max(0, i-10):i]["rsi14"].min()
        if pd.notna(recent_rsi_min) and recent_rsi_min <= 40 and -0.01 <= b["ret_7"] <= 0.10:
            open_pos = {
                "entry_time": b["close_time"], "direction": "short",
                "setup": "bear_rally_short_daily",
                "avg_entry": price, "atr_at_entry": b["atr14"],
                "size_frac": 1.0, "mfe": 0.0, "mae": 0.0,
                "stop_pct": -2.0 * b["atr14"] / price, "stop_atr_mult": 2.0,
                "tp1_done": False, "tp2_done": False,
            }

tr_df = pd.DataFrame(trades)
print(f"v3 trades: {len(tr_df)}")
if len(tr_df) == 0:
    print("no trades — conditions too strict")
    raise SystemExit(0)
print(f"win rate: {(tr_df['total_pnl_ret']>0).mean()*100:.1f}%")
print(f"mean pnl: {tr_df['total_pnl_ret'].mean()*100:+.2f}%  median: {tr_df['total_pnl_ret'].median()*100:+.2f}%")
print(f"sum of pnl (per full 1 unit size): {tr_df['total_pnl_ret'].sum()*100:+.1f}%")

# compound
eq = 1.0
for ret in tr_df.sort_values("entry_time")["total_pnl_ret"]:
    eq *= (1 + ret)
print(f"compounded 1 BTC -> {eq:.3f} BTC ({(eq-1)*100:+.1f}%)")

# drawdown
tr_s = tr_df.sort_values("entry_time").copy()
tr_s["cum"] = (1 + tr_s["total_pnl_ret"]).cumprod()
peak = tr_s["cum"].cummax()
dd = (tr_s["cum"] - peak) / peak
print(f"max drawdown: {dd.min()*100:.1f}%")

tr_df["year"] = pd.to_datetime(tr_df["entry_time"]).dt.year
print("\nby year:")
print(tr_df.groupby("year").agg(
    n=("total_pnl_ret", "size"),
    wr=("total_pnl_ret", lambda s: (s > 0).mean()),
    sum_ret_pct=("total_pnl_ret", lambda s: s.sum() * 100),
).round(3))

print("\nby exit reason:")
print(tr_df.groupby("exit_reason").agg(
    n=("total_pnl_ret", "size"),
    wr=("total_pnl_ret", lambda s: (s > 0).mean()),
    avg_ret=("total_pnl_ret", "mean"),
    sum_ret=("total_pnl_ret", "sum"),
).round(4).to_string())

print("\nby setup:")
print(tr_df.groupby("setup").agg(
    n=("total_pnl_ret", "size"),
    wr=("total_pnl_ret", lambda s: (s > 0).mean()),
    avg=("total_pnl_ret", "mean"),
    sum=("total_pnl_ret", "sum"),
).round(4).to_string())

# benchmark: BTC buy-and-hold over same period
start = d1[d1.index >= START_IDX].iloc[0]["close"]
end = d1.iloc[-1]["close"]
print(f"\nbenchmark: BTC buy-and-hold {start:.0f} -> {end:.0f} = +{(end/start-1)*100:.0f}%")

tr_df.to_csv(DATA / "derived" / "backtest_v3_trades.csv", index=False)
