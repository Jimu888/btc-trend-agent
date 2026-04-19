"""Detailed performance analysis of v5e — the winning strategy.

Validates robustness: year-by-year, trade-by-trade, what kind of regime catches
the big wins vs misses.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
d1["ma200"] = c.rolling(200).mean()
d1["ma50"]  = c.rolling(50).mean()
d1["ma50_prev10"] = d1["ma50"].shift(10)

trades = []
pos = None
START = 200

for i in range(START, len(d1)):
    b = d1.iloc[i]
    if any(pd.isna([b["ma200"], b["ma50"], b["ma50_prev10"]])):
        continue

    # manage pos
    if pos is not None:
        if b["close"] < b["ma200"]:
            ret = (b["close"] - pos["entry_price"]) / pos["entry_price"]
            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": b["close_time"],
                "entry_price": pos["entry_price"],
                "exit_price": b["close"],
                "days": (b["close_time"] - pos["entry_time"]).days,
                "ret": ret,
                "reason": "ma200_break",
            })
            pos = None

    # entry
    if pos is None:
        in_regime = b["close"] > b["ma200"] and b["close"] > b["ma50"]
        slope_up = b["ma50"] > b["ma50_prev10"]
        if in_regime and slope_up:
            pos = {"entry_time": b["close_time"], "entry_price": b["close"]}

# close at end
if pos is not None:
    last = d1.iloc[-1]
    ret = (last["close"] - pos["entry_price"]) / pos["entry_price"]
    trades.append({
        "entry_time": pos["entry_time"], "exit_time": last["close_time"],
        "entry_price": pos["entry_price"], "exit_price": last["close"],
        "days": (last["close_time"] - pos["entry_time"]).days,
        "ret": ret, "reason": "end",
    })

tr = pd.DataFrame(trades)
tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year

print("=== v5e trades in detail ===")
print(tr.round(4).to_string())

print("\n=== summary ===")
print(f"n: {len(tr)}")
print(f"win: {(tr['ret']>0).sum()}/{len(tr)} ({(tr['ret']>0).mean()*100:.1f}%)")
print(f"avg win: {tr[tr['ret']>0]['ret'].mean()*100:+.1f}%")
print(f"avg loss: {tr[tr['ret']<=0]['ret'].mean()*100:+.1f}%")

# compound
eq = 1.0
for r in tr["ret"]:
    eq *= (1 + r)
print(f"compound: 1 BTC -> {eq:.3f} BTC ({(eq-1)*100:+.1f}%)")

# time-in-market
total_days_in = tr["days"].sum()
total_range = (d1.iloc[-1]["close_time"] - d1.iloc[START]["close_time"]).days
print(f"time in market: {total_days_in}/{total_range} days ({total_days_in/total_range*100:.1f}%)")

# 2020-2023 vs 2024-2026 out-of-sample split
# Paul's trades were primarily in 2020-2023, so 2024+ is a natural OOS check
tr["period"] = np.where(pd.to_datetime(tr["entry_time"]).dt.year < 2024, "train_2020-23", "oos_2024+")
print("\n=== in-sample vs out-of-sample ===")
per = tr.groupby("period").agg(
    n=("ret", "size"),
    win_rate=("ret", lambda s: (s > 0).mean()),
    avg_ret=("ret", "mean"),
    sum_ret=("ret", lambda s: ((1 + s).prod() - 1) * 100),
).round(3)
print(per)

print("\n=== top 5 biggest winners ===")
print(tr.nlargest(5, "ret")[["entry_time", "exit_time", "entry_price", "exit_price", "ret", "days"]].round(3).to_string(index=False))

print("\n=== all losers ===")
print(tr[tr["ret"]<=0][["entry_time", "exit_time", "entry_price", "exit_price", "ret", "days"]].round(3).to_string(index=False))
