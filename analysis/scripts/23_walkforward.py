"""Walk-forward validation of v5e.

v5e has only 2 parameters (effectively):
  - MA window = 200 days
  - MA50 slope window = 10 days

Test robustness by:
  1. Grid-search best (ma_long, slope_window) on 2020-2022, test on 2023-2026
  2. Rolling 2-year train / 6-month test windows
  3. Parameter sensitivity heatmap

If +437% is fragile to small parameter changes, it's curve-fitting.
If robust across a range of parameters, it's real.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]

FEE = 0.001
SLIP = 0.0005

def run_v5e(ma_long, slope_window, start_idx=None, end_idx=None):
    """Backtest v5e with configurable params. Returns (final_eq, max_dd, n_trades, trade_df)."""
    data = d1.copy()
    data["ma_long"] = c.rolling(ma_long).mean()
    data["ma_mid"] = c.rolling(50).mean()
    data["ma_mid_prev"] = data["ma_mid"].shift(slope_window)
    warmup = max(ma_long, 50 + slope_window) + 1
    si = warmup if start_idx is None else max(warmup, start_idx)
    ei = len(data) - 1 if end_idx is None else min(len(data) - 1, end_idx)

    eq = 1.0; pos = None; trades = []; peak = 1.0; dd_min = 0.0
    for i in range(si, ei + 1):
        b = data.iloc[i]
        if any(pd.isna([b["ma_long"], b["ma_mid"], b["ma_mid_prev"]])):
            continue
        if pos is not None:
            mtm = eq * (1 + (b["close"] - pos["entry"]) / pos["entry"] - 2 * FEE)
            peak = max(peak, mtm); dd_min = min(dd_min, (mtm - peak) / peak)
            if b["close"] < b["ma_long"]:
                exit_p = b["close"] * (1 - SLIP)
                ret = (exit_p - pos["entry"]) / pos["entry"] - 2 * FEE
                eq *= (1 + ret)
                trades.append({"entry": pos["entry"], "exit": exit_p, "ret": ret})
                pos = None
                peak = max(peak, eq); dd_min = min(dd_min, (eq - peak) / peak)
        else:
            peak = max(peak, eq)
        if pos is None:
            if (b["close"] > b["ma_long"] and b["close"] > b["ma_mid"]
                and b["ma_mid"] > b["ma_mid_prev"]):
                pos = {"entry": b["close"] * (1 + SLIP), "entry_time": b["close_time"]}
    if pos is not None:
        last = data.iloc[ei]
        exit_p = last["close"] * (1 - SLIP)
        ret = (exit_p - pos["entry"]) / pos["entry"] - 2 * FEE
        eq *= (1 + ret)
    return eq, dd_min, len(trades)

# --------- parameter sensitivity grid ---------
print("=" * 70)
print("PARAMETER SENSITIVITY (full period 2020-2026)")
print("=" * 70)
print(f"{'ma_long':>8} {'slope_w':>8} {'final_eq':>10} {'return%':>10} {'dd%':>8} {'ret/|dd|':>10} {'n':>4}")
results = []
for ma_long in [100, 150, 200, 250, 300]:
    for slope_w in [5, 10, 20, 30]:
        eq, dd, n = run_v5e(ma_long, slope_w)
        ratio = (eq - 1) / abs(dd + 1e-9)
        print(f"{ma_long:>8} {slope_w:>8} {eq:>10.4f} {(eq-1)*100:>+9.1f} {dd*100:>+7.1f} {ratio:>10.2f} {n:>4}")
        results.append({"ma_long":ma_long,"slope_w":slope_w,"eq":eq,"ret":eq-1,"dd":dd,"ratio":ratio,"n":n})
df = pd.DataFrame(results)

print(f"\nMedian across all 20 param combos: ret={df['ret'].median()*100:.0f}%, dd={df['dd'].median()*100:.0f}%, ratio={df['ratio'].median():.2f}")
print(f"Best by ret/|dd|: ma_long={df.loc[df['ratio'].idxmax(),'ma_long']}, slope_w={df.loc[df['ratio'].idxmax(),'slope_w']}, ratio={df['ratio'].max():.2f}")

# --------- in-sample / out-of-sample split ---------
print("\n" + "=" * 70)
print("IN-SAMPLE (2020-2022) vs OUT-OF-SAMPLE (2023-2026)")
print("=" * 70)
# find split indices
split_ts = pd.Timestamp("2023-01-01", tz="UTC")
split_idx = d1[d1["close_time"] < split_ts].index[-1]
print(f"split at index {split_idx}, time {d1.iloc[split_idx]['close_time']}")
print(f"\n{'ma_long':>8} {'slope_w':>8} {'IS_ret%':>10} {'IS_dd%':>10} {'OOS_ret%':>10} {'OOS_dd%':>10}")
for ma_long in [100, 150, 200, 250, 300]:
    for slope_w in [10, 20]:
        eq_is, dd_is, _ = run_v5e(ma_long, slope_w, end_idx=split_idx)
        eq_oos, dd_oos, _ = run_v5e(ma_long, slope_w, start_idx=split_idx)
        print(f"{ma_long:>8} {slope_w:>8} {(eq_is-1)*100:>+9.1f} {dd_is*100:>+9.1f} {(eq_oos-1)*100:>+9.1f} {dd_oos*100:>+9.1f}")

# --------- rolling windows ---------
print("\n" + "=" * 70)
print("ROLLING 2-YEAR PERFORMANCE (ma_long=200, slope_w=10)")
print("=" * 70)
# start window at 2020-05, roll every 6 months
window_days = 730
step_days = 182
start_time = d1["close_time"].iloc[210]
windows = []
t = start_time
while t + pd.Timedelta(days=window_days) <= d1["close_time"].iloc[-1]:
    w_end = t + pd.Timedelta(days=window_days)
    si = d1[d1["close_time"] >= t].index[0]
    ei = d1[d1["close_time"] <= w_end].index[-1]
    eq, dd, n = run_v5e(200, 10, start_idx=si, end_idx=ei)
    windows.append({"start":t.date(), "end":w_end.date(), "ret":eq-1, "dd":dd, "n":n})
    t = t + pd.Timedelta(days=step_days)
wdf = pd.DataFrame(windows)
for _, row in wdf.iterrows():
    print(f"{row['start']} → {row['end']}: ret {row['ret']*100:+7.1f}%, dd {row['dd']*100:+6.1f}%, trades {row['n']}")
print(f"\nrolling windows: {len(wdf)}")
print(f"  positive return: {(wdf['ret'] > 0).sum()}/{len(wdf)}")
print(f"  median return: {wdf['ret'].median()*100:+.1f}%")
print(f"  worst return:  {wdf['ret'].min()*100:+.1f}%")
print(f"  median dd:     {wdf['dd'].median()*100:.1f}%")
