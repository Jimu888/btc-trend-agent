"""Verify roundtrip PnL against walletHistory ground truth."""
import pandas as pd
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
wh = pd.read_csv(DATA / "api-v1-user-walletHistory.csv", low_memory=False)
wh["timestamp"] = pd.to_datetime(wh["timestamp"])
wh["year"] = wh["timestamp"].dt.year

# address column contains symbol for trading events
print("walletHistory by transactType, summed in BTC (address=XBTUSD):")
xbt = wh[wh["address"] == "XBTUSD"]
print(xbt.groupby("transactType")["amount"].agg(["count", "sum"]).assign(btc=lambda d: d["sum"]/1e8))

print("\nAll RealisedPNL by symbol (address) and year (top symbols):")
pnl = wh[wh["transactType"] == "RealisedPNL"].copy()
pnl["btc"] = pnl["amount"] / 1e8
print(pnl.groupby("address")["btc"].agg(["count", "sum"]).sort_values("sum", ascending=False).head(15))

print("\nRealisedPNL by year (all symbols):")
print(pnl.groupby("year")["btc"].sum())

print("\nFunding by year (all symbols):")
fund = wh[wh["transactType"] == "Funding"].copy()
fund["btc"] = fund["amount"] / 1e8
print(fund.groupby("year")["btc"].sum())

print("\nTotal wallet-side trading result:")
print(f"  RealisedPNL sum (BTC): {pnl['btc'].sum():.2f}")
print(f"  Funding sum   (BTC): {fund['btc'].sum():.2f}")
print(f"  Combined      (BTC): {(pnl['btc'].sum() + fund['btc'].sum()):.2f}")
