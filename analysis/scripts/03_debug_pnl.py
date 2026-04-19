"""Debug: why is realisedPnl so sparse?"""
import pandas as pd
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
df = pd.read_csv(DATA / "api-v1-execution-tradeHistory.csv", low_memory=False)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["year"] = df["timestamp"].dt.year

print("realisedPnl populated rows by year (execType=Trade, XBTUSD):")
m = (df["execType"] == "Trade") & (df["symbol"] == "XBTUSD")
sub = df[m]
print("total XBTUSD trades:", len(sub))
print("with realisedPnl:", sub["realisedPnl"].notna().sum())
print(sub.groupby("year").agg(
    n=("realisedPnl", "size"),
    with_pnl=("realisedPnl", lambda s: s.notna().sum()),
))

print("\nSample of non-null realisedPnl entries:")
cols = ["timestamp", "side", "lastQty", "lastPx", "realisedPnl", "execComm", "execType", "ordStatus"]
print(sub[sub["realisedPnl"].notna()][cols].head(10).to_string())

print("\nSample of funding entries:")
print(df[df["execType"] == "Funding"][["timestamp", "symbol", "realisedPnl", "execComm"]].head(5).to_string())

# check walletHistory for realised PnL totals
wh = pd.read_csv(DATA / "api-v1-user-walletHistory.csv", low_memory=False)
print("\nwalletHistory transactTypes:")
print(wh["transactType"].value_counts())
if "amount" in wh.columns:
    pnl_events = wh[wh["transactType"].isin(["RealisedPNL", "Trade"])]
    print(f"\ntotal RealisedPNL events: {(wh['transactType']=='RealisedPNL').sum()}")
    print(wh[wh["transactType"] == "RealisedPNL"].head(3).to_string())
