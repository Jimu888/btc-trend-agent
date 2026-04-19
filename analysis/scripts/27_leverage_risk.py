"""Leverage risk assessment for v5f strategy.

For each of the 15 historical v5f trades, compute the worst intrabar adverse
excursion (MAE) — the lowest low during the trade vs entry price.

Then simulate leverage 2x, 3x, 4x, 5x:
  - Binance liquidation: unrealized loss consumes initial margin (minus small
    maintenance margin ~0.5%). So liquidation triggers at approx:
      adverse move >= (1/leverage) - 0.005
    5x: -19.5%, 4x: -24.5%, 3x: -32.8%, 2x: -49.5%, 1x: never

  - Funding cost: U-margined perps typically pay/receive ~0.01% per 8h.
    In strong bull markets often 0.03-0.10%/8h. We model conservative 0.02%/8h.

Output: annualized cost + liquidation events at each leverage.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
d1 = pd.read_csv(DATA / "klines" / "1d.csv", parse_dates=["close_time"])
c = d1["close"]
d1["ma150"] = c.rolling(150).mean()
d1["ma50"] = c.rolling(50).mean()
d1["ma50_prev10"] = d1["ma50"].shift(10)

# rebuild v5f trades
trades = []
pos = None
for i in range(210, len(d1)):
    b = d1.iloc[i]
    if any(pd.isna([b["ma150"], b["ma50"], b["ma50_prev10"]])): continue
    if pos is not None and b["close"] < b["ma150"]:
        trades.append({"entry_idx": pos["idx"], "exit_idx": i,
                       "entry_time": pos["time"], "exit_time": b["close_time"],
                       "entry_price": pos["price"], "exit_price": b["close"]})
        pos = None
    if pos is None:
        if b["close"] > b["ma150"] and b["close"] > b["ma50"] and b["ma50"] > b["ma50_prev10"]:
            pos = {"idx": i, "time": b["close_time"], "price": b["close"]}
if pos is not None:
    trades.append({"entry_idx": pos["idx"], "exit_idx": len(d1)-1,
                   "entry_time": pos["time"], "exit_time": d1.iloc[-1]["close_time"],
                   "entry_price": pos["price"], "exit_price": d1.iloc[-1]["close"]})

tr = pd.DataFrame(trades)

# intrabar MAE (worst low during hold)
mae_pcts = []
mfe_pcts = []
for _, t in tr.iterrows():
    window = d1.iloc[t["entry_idx"]:t["exit_idx"]+1]
    worst_low = window["low"].min()
    best_high = window["high"].max()
    mae = (worst_low - t["entry_price"]) / t["entry_price"] * 100
    mfe = (best_high - t["entry_price"]) / t["entry_price"] * 100
    mae_pcts.append(mae)
    mfe_pcts.append(mfe)
tr["mae_pct"] = mae_pcts
tr["mfe_pct"] = mfe_pcts
tr["return_pct"] = (tr["exit_price"] - tr["entry_price"]) / tr["entry_price"] * 100
tr["days"] = (tr["exit_time"] - tr["entry_time"]).dt.days

print("=" * 80)
print("v5f trades with INTRABAR worst adverse excursion")
print("=" * 80)
print(f"{'entry_date':12s} {'days':5s} {'return%':>8s} {'mae%':>8s} {'mfe%':>8s}")
for _, t in tr.iterrows():
    print(f"{str(t['entry_time'].date()):12s} {t['days']:5d} {t['return_pct']:+8.2f} {t['mae_pct']:+8.2f} {t['mfe_pct']:+8.2f}")

print(f"\n### SUMMARY ###")
print(f"worst intrabar MAE across all trades: {tr['mae_pct'].min():+.2f}%")
print(f"median intrabar MAE: {tr['mae_pct'].median():+.2f}%")
print(f"fraction of trades with MAE worse than -10%: {(tr['mae_pct'] < -10).mean()*100:.0f}%")
print(f"fraction with MAE worse than -20%: {(tr['mae_pct'] < -20).mean()*100:.0f}%")

print("\n" + "=" * 80)
print("LIQUIDATION SIMULATION by leverage")
print("=" * 80)
print(f"{'lev':>4s} {'liq_thresh':>11s} {'liq_count':>10s} {'liq_rate':>10s}  {'net_ret*lev (pre-funding)':s}")
print("-" * 80)

# liquidation threshold: adverse move that wipes out initial margin (less maintenance)
# approximate: 1/leverage - 0.005
for lev in [1, 2, 3, 4, 5]:
    liq_thresh = -(1/lev - 0.005) * 100   # as percent
    # count trades that would liquidate
    liquidated = tr[tr["mae_pct"] <= liq_thresh]
    n_liq = len(liquidated)
    # if any trade liquidates, that trade contributes -100% to equity
    # simulate compounded: start 1.0, apply each trade in order
    eq = 1.0
    liq_events = 0
    for _, t in tr.sort_values("entry_time").iterrows():
        if t["mae_pct"] <= liq_thresh:
            eq *= 0  # wipeout (simplification — really you lose margin then start again)
            liq_events += 1
            # in reality, sub-account is 0; simulate fresh restart (best case)
            eq = 1e-9  # avoid 0 for log; effectively wiped
        else:
            # amplify return by leverage, minus 0.2% roundtrip fee per side × leverage
            ret_amplified = t["return_pct"] / 100 * lev
            ret_net = ret_amplified - 0.002 * lev  # entry+exit fees scaled by leverage
            eq *= (1 + ret_net)
    final_display = f"{(eq-1)*100:+.1f}%" if eq > 1e-6 else "WIPED OUT"
    print(f"{lev:>4d}x  {liq_thresh:>+9.1f}%  {n_liq:>10d}  {n_liq/len(tr)*100:>9.0f}%   {final_display}")

print("""
Interpretation:
  - If liquidation count > 0, you'd blow up at least once over 6 years.
  - 'wiped out' means at some point your sub-account hit zero.
  - Below we also need to factor in FUNDING COSTS, which eat ~5-20% per year
    of the LEVERAGED position value in bull markets.
""")

# funding cost modeling
print("=" * 80)
print("FUNDING COST SIMULATION (assumes 0.02%/8h = 0.06%/day funding in long position)")
print("=" * 80)
total_days_in = tr["days"].sum()
print(f"total days in position across all trades: {total_days_in}")
# at 1x: funding cost = 0.06%/day × days = X%
# at 5x: funding cost amplified by leverage
for lev in [1, 2, 3, 4, 5]:
    annualized_cost = 0.06 * lev  # percent per day on equity
    # wait: funding is on NOTIONAL not equity. At 5x, notional = 5x equity.
    # So cost per day on equity = 0.06% * 5 = 0.3%/day in high funding periods
    total_funding_cost_pct = 0.06 * lev * total_days_in / 100
    print(f"  {lev}x leverage: total funding drag over 1096 in-market days ≈ {total_funding_cost_pct*100:.1f}% of equity")

print(f"""
NOTE on funding variability:
  - Neutral markets: ~0.01%/8h (≈0.03%/day)
  - Normal bull: ~0.02%/8h (≈0.06%/day)  ← used above
  - Extreme bull (late 2021): 0.05-0.10%/8h (≈0.15-0.30%/day)
  - Extreme bear: funding can flip NEGATIVE (pays longs)

v5f holds long during bull markets, so funding is a CONSISTENT DRAG.
At 5x leverage in extreme bull periods, funding alone can cost 10-20% of
equity per month.
""")
