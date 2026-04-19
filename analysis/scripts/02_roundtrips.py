"""Phase 1: reconstruct round-trip trades from XBTUSD fills.

BitMEX XBTUSD is inverse (1 contract = $1, settled in BTC).
PnL formula:
  long:  contracts * (1/entry - 1/exit)   [in BTC]
  short: contracts * (1/exit - 1/entry)   [in BTC]

realisedPnl column is only populated from 2024 onward, so we compute PnL
from first principles using volume-weighted avg entry/exit per roundtrip.
Commissions (execComm in satoshis) are subtracted to get net PnL.
Funding costs are separate and handled later from walletHistory.

Output: data/derived/roundtrips_xbtusd.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
OUT = DATA / "derived"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA / "api-v1-execution-tradeHistory.csv",
                 parse_dates=["timestamp", "transactTime"],
                 low_memory=False)

df = df[(df["execType"] == "Trade") & (df["symbol"] == "XBTUSD")].copy()
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"XBTUSD trade fills: {len(df):,}")

df["signed_qty"] = np.where(df["side"] == "Buy", df["lastQty"], -df["lastQty"])
# prefer lastPx (actual fill price); fall back to price
df["fill_price"] = df["lastPx"].fillna(df["price"])

roundtrips = []

def init_rt(ts, sign, contracts, price, comm):
    return {
        "entry_time": ts,
        "exit_time": ts,
        "direction": "long" if sign > 0 else "short",
        "entry_contracts": contracts,
        "entry_notional": contracts * price,     # USD notional, volume-weighted
        "exit_contracts": 0,
        "exit_notional": 0,
        "commission_sat": comm,
        "max_contracts": contracts,
        "n_fills": 1,
    }

position = 0
open_rt = None

for row in df.itertuples(index=False):
    fill = row.signed_qty
    price = row.fill_price
    comm = row.execComm if not pd.isna(row.execComm) else 0
    if price is None or pd.isna(price) or price == 0:
        continue

    if position == 0:
        open_rt = init_rt(row.timestamp, 1 if fill > 0 else -1, abs(fill), price, comm)
        position = fill
        continue

    same_sign = (position > 0) == (fill > 0)

    if same_sign:
        # scale in
        open_rt["entry_contracts"] += abs(fill)
        open_rt["entry_notional"] += abs(fill) * price
        open_rt["commission_sat"] += comm
        open_rt["n_fills"] += 1
        position += fill
        open_rt["max_contracts"] = max(open_rt["max_contracts"], abs(position))
        open_rt["exit_time"] = row.timestamp
    else:
        abs_fill = abs(fill)
        abs_pos = abs(position)
        if abs_fill <= abs_pos:
            # reduce or flat
            open_rt["exit_contracts"] += abs_fill
            open_rt["exit_notional"] += abs_fill * price
            open_rt["commission_sat"] += comm
            open_rt["n_fills"] += 1
            open_rt["exit_time"] = row.timestamp
            position += fill
            if position == 0:
                roundtrips.append(open_rt)
                open_rt = None
        else:
            # flip: close + reopen
            closing = abs_pos
            opening = abs_fill - abs_pos
            frac_close = closing / abs_fill
            open_rt["exit_contracts"] += closing
            open_rt["exit_notional"] += closing * price
            open_rt["commission_sat"] += comm * frac_close
            open_rt["exit_time"] = row.timestamp
            open_rt["n_fills"] += 1
            roundtrips.append(open_rt)
            new_sign = 1 if fill > 0 else -1
            open_rt = init_rt(row.timestamp, new_sign, opening, price, comm * (1 - frac_close))
            position = opening * new_sign

rt = pd.DataFrame(roundtrips)
rt["avg_entry"] = rt["entry_notional"] / rt["entry_contracts"]
rt["avg_exit"] = rt["exit_notional"] / rt["exit_contracts"]

# PnL in BTC (inverse contract formula)
sign = np.where(rt["direction"] == "long", 1, -1)
rt["gross_pnl_btc"] = sign * rt["entry_contracts"] * (1 / rt["avg_entry"] - 1 / rt["avg_exit"])
rt["commission_btc"] = rt["commission_sat"] / 1e8
rt["net_pnl_btc"] = rt["gross_pnl_btc"] - rt["commission_btc"]

# return % (USD price move in trade direction)
rt["return_pct"] = np.where(
    rt["direction"] == "long",
    (rt["avg_exit"] - rt["avg_entry"]) / rt["avg_entry"] * 100,
    (rt["avg_entry"] - rt["avg_exit"]) / rt["avg_entry"] * 100,
)

rt["holding_seconds"] = (rt["exit_time"] - rt["entry_time"]).dt.total_seconds()
rt["holding_hours"] = rt["holding_seconds"] / 3600
rt["holding_days"] = rt["holding_seconds"] / 86400
rt["year"] = rt["entry_time"].dt.year
rt["notional_btc"] = rt["entry_contracts"] / rt["avg_entry"]    # BTC-equiv size at entry

out_path = OUT / "roundtrips_xbtusd.csv"
rt.to_csv(out_path, index=False)

print(f"\nroundtrips: {len(rt):,}")
print(f"by year:")
print(rt.groupby("year").agg(
    n=("net_pnl_btc", "size"),
    gross_btc=("gross_pnl_btc", "sum"),
    net_btc=("net_pnl_btc", "sum"),
    win_rate=("net_pnl_btc", lambda s: (s > 0).mean()),
))

print(f"\nTotals:")
print(f"  gross pnl (BTC): {rt['gross_pnl_btc'].sum():.2f}")
print(f"  commissions (BTC): {rt['commission_btc'].sum():.2f}")
print(f"  net pnl (BTC): {rt['net_pnl_btc'].sum():.2f}")
print(f"  expected (wallet growth adjusted, README): ~94.5 BTC")

# sanity: cross-check against walletHistory RealisedPNL sum for XBTUSD
wh = pd.read_csv(DATA / "api-v1-user-walletHistory.csv", low_memory=False)
xbt_pnl_sat = wh[(wh["transactType"] == "RealisedPNL") &
                 (wh["orderID"] == "XBTUSD")]["amount"].sum()
funding_sat = wh[(wh["transactType"] == "Funding") &
                 (wh["orderID"] == "XBTUSD")]["amount"].sum() if "Funding" in wh["transactType"].values else 0
print(f"\n  walletHistory XBTUSD RealisedPNL total (BTC): {xbt_pnl_sat/1e8:.2f}")
print(f"  walletHistory XBTUSD Funding total (BTC): {funding_sat/1e8:.2f}")
