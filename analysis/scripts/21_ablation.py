"""Quick ablation: what if swing engine were perfect (oracle) vs dead (100% trend)?

Establishes theoretical bounds for the hybrid agent.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
d1["ma200"] = c.rolling(200).mean()
d1["ma50"] = c.rolling(50).mean()
d1["ma50_prev10"] = d1["ma50"].shift(10)

# pure v5e at 100% capital with realistic costs
FEE = 0.001
SLIP = 0.0005
eq = 1.0
pos = None
trades = []
dd_max = 0
peak = 1.0
START = 210

for i in range(START, len(d1)):
    b = d1.iloc[i]
    if any(pd.isna([b["ma200"], b["ma50"], b["ma50_prev10"]])):
        continue

    if pos is not None:
        if b["close"] < b["ma200"]:
            exit_p = b["close"] * (1 - SLIP)
            ret = (exit_p - pos["entry"]) / pos["entry"]
            net_ret = ret - 2 * FEE
            eq *= (1 + net_ret)
            trades.append({"entry": pos["entry"], "exit": exit_p, "ret": net_ret,
                           "days": (b["close_time"] - pos["entry_time"]).days})
            pos = None

    # mark-to-market
    if pos is not None:
        mtm = eq * (1 + (b["close"] - pos["entry"]) / pos["entry"] - 2 * FEE)
    else:
        mtm = eq
    peak = max(peak, mtm)
    dd_max = min(dd_max, (mtm - peak) / peak)

    if pos is None:
        if b["close"] > b["ma200"] and b["close"] > b["ma50"] and b["ma50"] > b["ma50_prev10"]:
            pos = {"entry": b["close"] * (1 + SLIP), "entry_time": b["close_time"]}

if pos is not None:
    last = d1.iloc[-1]
    exit_p = last["close"] * (1 - SLIP)
    ret = (exit_p - pos["entry"]) / pos["entry"]
    net_ret = ret - 2 * FEE
    eq *= (1 + net_ret)

print(f"=== PURE v5e 100% CAPITAL (with costs) ===")
print(f"trades: {len(trades)}")
print(f"final eq: {eq:.4f}  ({(eq-1)*100:+.1f}%)")
print(f"max drawdown: {dd_max*100:.1f}%")
print(f"return/|dd|: {(eq-1)/abs(dd_max+1e-9):.2f}")

start_p = d1.iloc[START]["close"]
end_p = d1.iloc[-1]["close"]
bh = end_p/start_p - 1
bh_eq = d1["close"].iloc[START:] / start_p
bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()
print(f"\n=== BENCHMARK: BTC buy-and-hold ===")
print(f"{bh*100:+.0f}%  dd={bh_dd*100:.1f}%  ratio={bh/abs(bh_dd):.2f}")

print(f"\n=== COMPARISON (with realistic costs, ~0.1% fee + 0.05% slip per side) ===")
print(f"pure v5e:     {(eq-1)*100:+.1f}%  dd={dd_max*100:.1f}%")
print(f"buy-and-hold: {bh*100:+.0f}%       dd={bh_dd*100:.1f}%")
