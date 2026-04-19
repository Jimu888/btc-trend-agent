"""Phase 0: quick structural exploration — cheap summaries only, no heavy load."""
import pandas as pd
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")

print("=" * 60)
print("EXECUTION TRADE HISTORY (fills)")
print("=" * 60)

# Stream-read to check shape without loading everything at once
exec_df = pd.read_csv(DATA / "api-v1-execution-tradeHistory.csv",
                     parse_dates=["timestamp", "transactTime"],
                     low_memory=False)
print(f"rows: {len(exec_df):,}")
print(f"columns: {list(exec_df.columns)}")
print(f"time range: {exec_df['timestamp'].min()} -> {exec_df['timestamp'].max()}")

print("\n-- execType distribution --")
print(exec_df["execType"].value_counts())

print("\n-- symbol distribution (top 15) --")
print(exec_df["symbol"].value_counts().head(15))

print("\n-- settlCurrency distribution --")
print(exec_df["settlCurrency"].value_counts())

print("\n-- ordType distribution --")
print(exec_df["ordType"].value_counts())

print("\n-- side distribution --")
print(exec_df["side"].value_counts(dropna=False))

print("\n-- by year (execType=Trade, XBTUSD only) --")
trades = exec_df[(exec_df["execType"] == "Trade") & (exec_df["symbol"] == "XBTUSD")].copy()
trades["year"] = trades["timestamp"].dt.year
print(trades.groupby("year").size())

print("\n-- realisedPnl not-null rows (sample) --")
print(exec_df["realisedPnl"].notna().sum(), "rows have realisedPnl")
