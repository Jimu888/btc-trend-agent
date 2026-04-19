"""Phase 2: behavior profile — who is this trader?

Answers:
  Q1: Win rate, expectancy, profit factor, max-consecutive-loss
  Q2: Hold time distribution (scalper? swing? position?)
  Q3: Long vs short: does he have directional bias?
  Q4: Size distribution: how big does he bet?
  Q5: Time-of-day / day-of-week preferences
  Q6: Scale-in behavior: single-shot vs. layered entry?
  Q7: Year-by-year evolution
  Q8: Biggest wins and biggest losses
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
rt = pd.read_csv(DATA / "derived" / "roundtrips_xbtusd.csv",
                 parse_dates=["entry_time", "exit_time"])

# we want trades that actually made/lost something — filter tiny noise roundtrips
# (sometimes zero-crossings from intra-order fills create 1-contract round-trips).
# threshold: >= $100 notional (100 contracts). Still generous.
rt = rt[rt["entry_contracts"] >= 100].copy()
print(f"meaningful roundtrips (>=100 contracts): {len(rt)}")

# make entry_time UTC-aware for hour calcs
rt["entry_hour_utc"] = rt["entry_time"].dt.hour
rt["entry_dow"] = rt["entry_time"].dt.dayofweek  # 0=Mon
rt["win"] = rt["net_pnl_btc"] > 0

def hdr(t): print(f"\n{'='*64}\n{t}\n{'='*64}")

# ---------- Q1: overall performance ----------
hdr("Q1. Overall performance")
n = len(rt)
wins = rt[rt["win"]]
losses = rt[~rt["win"]]
print(f"trades: {n}")
print(f"win rate: {len(wins)/n*100:.1f}%")
print(f"avg win:  {wins['net_pnl_btc'].mean():+.4f} BTC  (median {wins['net_pnl_btc'].median():+.4f})")
print(f"avg loss: {losses['net_pnl_btc'].mean():+.4f} BTC  (median {losses['net_pnl_btc'].median():+.4f})")
print(f"win/loss ratio: {abs(wins['net_pnl_btc'].mean() / losses['net_pnl_btc'].mean()):.2f}")
print(f"profit factor: {wins['net_pnl_btc'].sum() / abs(losses['net_pnl_btc'].sum()):.2f}")
print(f"expectancy per trade: {rt['net_pnl_btc'].mean():+.4f} BTC")
print(f"gross: {rt['gross_pnl_btc'].sum():+.2f} BTC  net: {rt['net_pnl_btc'].sum():+.2f} BTC  comm drag: {rt['commission_btc'].sum():.2f} BTC ({rt['commission_btc'].sum()/rt['gross_pnl_btc'].sum()*100:.0f}% of gross)")

# max consecutive losses
rt_s = rt.sort_values("entry_time").reset_index(drop=True)
streak = max_streak = 0
for w in rt_s["win"]:
    if not w:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
print(f"max consecutive losses: {max_streak}")

# return distribution (in % price move, trader's direction)
print(f"\nreturn% distribution:")
print(rt["return_pct"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(2))

# ---------- Q2: hold time ----------
hdr("Q2. Hold time")
print(rt["holding_hours"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
print(f"\nhold-time buckets:")
bins = [0, 1, 6, 24, 72, 168, 720, float("inf")]
labels = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "1-4w", ">1mo"]
rt["hold_bucket"] = pd.cut(rt["holding_hours"], bins=bins, labels=labels, right=False)
bucket_stats = rt.groupby("hold_bucket", observed=True).agg(
    n=("net_pnl_btc", "size"),
    win_rate=("win", "mean"),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
)
print(bucket_stats.round(3))

# ---------- Q3: long vs short ----------
hdr("Q3. Long vs short")
dir_stats = rt.groupby("direction").agg(
    n=("net_pnl_btc", "size"),
    win_rate=("win", "mean"),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
    avg_return_pct=("return_pct", "mean"),
    median_hold_h=("holding_hours", "median"),
)
print(dir_stats.round(3))

# ---------- Q4: size ----------
hdr("Q4. Position size (notional_btc at entry)")
print(rt["notional_btc"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2))

# ---------- Q5: time of day, day of week ----------
hdr("Q5. Time preferences")
hour_stats = rt.groupby("entry_hour_utc").agg(
    n=("net_pnl_btc", "size"),
    win_rate=("win", "mean"),
    pnl=("net_pnl_btc", "sum"),
)
# find top-3 entry hours
print("top 6 entry hours (UTC):")
print(hour_stats.sort_values("n", ascending=False).head(6).round(3))
print("\nby day of week (0=Mon):")
dow_stats = rt.groupby("entry_dow").agg(
    n=("net_pnl_btc", "size"),
    win_rate=("win", "mean"),
    pnl=("net_pnl_btc", "sum"),
)
print(dow_stats.round(3))

# ---------- Q6: scale-in ----------
hdr("Q6. Scale-in behavior (n_fills per roundtrip)")
print(rt["n_fills"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.99]).round(1))
print(f"\nroundtrips by fill count:")
print(rt["n_fills"].value_counts().head(10).sort_index())
# high-fill roundtrips: are they more profitable?
rt["many_fills"] = rt["n_fills"] >= 20
print(f"\nhigh-fill (>=20) vs low-fill roundtrips:")
print(rt.groupby("many_fills").agg(
    n=("net_pnl_btc", "size"),
    win_rate=("win", "mean"),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
).round(3))

# ---------- Q7: year-by-year evolution ----------
hdr("Q7. Year-by-year evolution")
yr = rt.groupby("year").agg(
    trades=("net_pnl_btc", "size"),
    net_pnl=("net_pnl_btc", "sum"),
    win_rate=("win", "mean"),
    avg_pnl=("net_pnl_btc", "mean"),
    median_hold_h=("holding_hours", "median"),
    median_notional=("notional_btc", "median"),
    long_share=("direction", lambda s: (s == "long").mean()),
)
print(yr.round(3))

# ---------- Q8: biggest wins / losses ----------
hdr("Q8. Biggest single wins and losses")
top_cols = ["entry_time", "exit_time", "direction", "notional_btc",
            "avg_entry", "avg_exit", "return_pct", "net_pnl_btc",
            "holding_hours", "n_fills"]
print("TOP 10 WINS:")
print(rt.nlargest(10, "net_pnl_btc")[top_cols].round(3).to_string(index=False))
print("\nTOP 10 LOSSES:")
print(rt.nsmallest(10, "net_pnl_btc")[top_cols].round(3).to_string(index=False))

# save trimmed dataset for phase 3
out = DATA / "derived" / "roundtrips_xbtusd_clean.csv"
rt.to_csv(out, index=False)
print(f"\nsaved clean dataset -> {out}")
