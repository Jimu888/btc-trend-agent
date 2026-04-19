"""Test whether v5f parameters transfer to other timeframes.

Hypothesis: v5f (MA150/slope10 on daily) is tuned for BTC's weekly-monthly trend
cycles. Applying the SAME MA count to other timeframes should fail because:
  - 4h: noise-dominated, fees compound faster
  - 1h: pure noise for trend following
  - weekly: lags cycle changes, too few signals

Scale parameters TWO ways:
  A. Same numeric MA (naive): MA150 bars regardless of timeframe
  B. Same calendar duration: MA150 days → MA900 on 4h, MA3600 on 1h
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data") / "klines"
FEE = 0.001; SLIP = 0.0005

def load_tf(interval):
    df = pd.read_csv(DATA / f"{interval}.csv", parse_dates=["close_time"])
    return df.sort_values("close_time").reset_index(drop=True)

def backtest(df, ma_long, ma_mid, slope_w, label):
    c = df["close"]
    df = df.copy()
    df["ma_long"] = c.rolling(ma_long).mean()
    df["ma_mid"] = c.rolling(ma_mid).mean()
    df["ma_mid_prev"] = df["ma_mid"].shift(slope_w)
    warmup = max(ma_long, ma_mid + slope_w) + 1
    if warmup >= len(df):
        return {"label": label, "ret_pct": np.nan, "dd_pct": np.nan, "n": 0, "err": "insufficient_data"}
    eq = 1.0; pos = None; trades = 0; peak = 1.0; dd = 0.0
    for i in range(warmup, len(df)):
        b = df.iloc[i]
        if any(pd.isna([b["ma_long"], b["ma_mid"], b["ma_mid_prev"]])): continue
        if pos is not None:
            mtm = eq * (1 + (b["close"] - pos) / pos - 2 * FEE)
            peak = max(peak, mtm); dd = min(dd, (mtm - peak) / peak)
            if b["close"] < b["ma_long"]:
                ret = (b["close"] * (1 - SLIP) - pos) / pos - 2 * FEE
                eq *= (1 + ret); trades += 1; pos = None
        else:
            peak = max(peak, eq)
        if pos is None:
            if (b["close"] > b["ma_long"] and b["close"] > b["ma_mid"]
                and b["ma_mid"] > b["ma_mid_prev"]):
                pos = b["close"] * (1 + SLIP)
    return {
        "label": label,
        "ret_pct": (eq - 1) * 100,
        "dd_pct": dd * 100,
        "ratio": (eq - 1) / abs(dd + 1e-9),
        "n": trades,
    }

# --- Baseline: daily v5f ---
d1 = load_tf("1d")
h4 = load_tf("4h")
h1 = load_tf("1h")

# Approach A: SAME MA NUMBER (naive transfer)
print("=" * 70)
print("APPROACH A: same numeric params (MA150, MA50, slope=10) on all timeframes")
print("=" * 70)
r_d = backtest(d1, 150, 50, 10, "1d (baseline)")
r_4 = backtest(h4, 150, 50, 10, "4h (MA150 bars = 25 days)")
r_1 = backtest(h1, 150, 50, 10, "1h (MA150 bars = 6.25 days)")
for r in [r_d, r_4, r_1]:
    if "err" in r:
        print(f"{r['label']:40s} ERROR: {r['err']}")
    else:
        print(f"{r['label']:40s}  ret={r['ret_pct']:+7.1f}%  dd={r['dd_pct']:+6.1f}%  ratio={r['ratio']:.2f}  n={r['n']}")

# Approach B: SAME CALENDAR DURATION (scale MA proportionally)
# daily MA150 = 150 days ≈ 21 weeks
#   weekly:   MA ~= 21
#   4h:       MA ~= 150 * 6 = 900
#   1h:       MA ~= 150 * 24 = 3600
print()
print("=" * 70)
print("APPROACH B: same CALENDAR duration (MA150 days-equivalent)")
print("=" * 70)
r_d_b = backtest(d1, 150, 50, 10, "1d   MA150")
r_4_b = backtest(h4, 900, 300, 60, "4h   MA900 (=150d)  MA300 (=50d)")
r_1_b = backtest(h1, 3600, 1200, 240, "1h   MA3600 (=150d)  MA1200 (=50d)")
# weekly — resample daily to weekly
d1["week"] = d1["close_time"].dt.isocalendar().week + d1["close_time"].dt.year * 100
w = d1.groupby("week").agg(close_time=("close_time","last"), open=("open","first"),
                            high=("high","max"), low=("low","min"),
                            close=("close","last"), volume=("volume","sum")).reset_index(drop=True)
w = w.sort_values("close_time").reset_index(drop=True)
r_w = backtest(w, 21, 8, 2, "1w   MA21 (=147d)  MA8")
for r in [r_d_b, r_4_b, r_1_b, r_w]:
    if "err" in r:
        print(f"{r['label']:40s} ERROR: {r['err']}")
    else:
        print(f"{r['label']:40s}  ret={r['ret_pct']:+7.1f}%  dd={r['dd_pct']:+6.1f}%  ratio={r['ratio']:.2f}  n={r['n']}")

# Benchmark
print()
start_p = d1.iloc[0]["close"]
end_p = d1.iloc[-1]["close"]
print(f"BTC buy-and-hold (same period): ret={(end_p/start_p - 1)*100:+.0f}%")
