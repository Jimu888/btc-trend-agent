"""Phase 3.2: engineer K-line features for each roundtrip entry.

All features are computed using ONLY data available before entry_time
(no lookahead bias). Features are designed to be:
  - observable in real-time by an AI agent
  - timeframe-aware (1d for regime, 4h for swing, 1h for precision)

Output: data/derived/roundtrips_with_features.csv
        data/derived/baseline_features.csv (random 4h bar sample for comparison)
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"
OUT = DATA / "derived"

# --- load klines ---
def load(interval):
    df = pd.read_csv(KL / f"{interval}.csv", parse_dates=["open_time", "close_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    return df

d1 = load("1d")
h4 = load("4h")
h1 = load("1h")

# --- enrich each kline table with indicators ---
def enrich(df, prefix):
    c = df["close"]
    df[f"{prefix}_ret_1"] = c.pct_change(1)
    df[f"{prefix}_ret_3"] = c.pct_change(3)
    df[f"{prefix}_ret_7"] = c.pct_change(7)
    df[f"{prefix}_ma20"] = c.rolling(20).mean()
    df[f"{prefix}_ma50"] = c.rolling(50).mean()
    df[f"{prefix}_ma200"] = c.rolling(200).mean()
    df[f"{prefix}_high20"] = df["high"].rolling(20).max()
    df[f"{prefix}_low20"]  = df["low"].rolling(20).min()
    df[f"{prefix}_high50"] = df["high"].rolling(50).max()
    df[f"{prefix}_low50"]  = df["low"].rolling(50).min()
    # ATR
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - c.shift(1)).abs(),
        (df["low"] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df[f"{prefix}_atr14"] = tr.rolling(14).mean()
    # realized vol (20-period stdev of log returns)
    df[f"{prefix}_vol20"] = np.log(c / c.shift(1)).rolling(20).std()
    # RSI-14
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    df[f"{prefix}_rsi14"] = 100 - 100 / (1 + rs)
    return df

d1 = enrich(d1, "d")
h4 = enrich(h4, "h4")
h1 = enrich(h1, "h1")

print(f"enriched klines: 1d={len(d1)}, 4h={len(h4)}, 1h={len(h1)}")

# --- for a given timestamp, find the MOST RECENT bar that CLOSED before entry ---
# this enforces no lookahead: at entry_time T, agent only sees bars closed <= T
def _latest(df, ts_col, ts):
    # binary search for last row where close_time <= ts
    idx = df[ts_col].searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    return df.iloc[idx]

# --- feature builder ---
def features_at(ts):
    """Build a feature row for a given entry timestamp (pd.Timestamp, UTC-aware)."""
    d  = _latest(d1, "close_time", ts)
    h4r = _latest(h4, "close_time", ts)
    h1r = _latest(h1, "close_time", ts)
    if d is None or h4r is None or h1r is None:
        return None
    p = h1r["close"]  # latest observable price
    if pd.isna(p) or p == 0:
        return None

    # --- trend / regime (daily) ---
    f = {}
    f["price"] = p
    f["d_pct_vs_ma20"]  = (p - d["d_ma20"])  / d["d_ma20"]  if pd.notna(d["d_ma20"])  else np.nan
    f["d_pct_vs_ma50"]  = (p - d["d_ma50"])  / d["d_ma50"]  if pd.notna(d["d_ma50"])  else np.nan
    f["d_pct_vs_ma200"] = (p - d["d_ma200"]) / d["d_ma200"] if pd.notna(d["d_ma200"]) else np.nan
    f["d_above_ma200"] = int(p > d["d_ma200"]) if pd.notna(d["d_ma200"]) else np.nan
    f["d_ma20_gt_ma50"] = int(d["d_ma20"] > d["d_ma50"]) if pd.notna(d["d_ma20"]) and pd.notna(d["d_ma50"]) else np.nan
    # 50-day trend slope: is MA50 rising? compare ma50 to ma50 10d earlier
    idx = d1["close_time"].searchsorted(ts, side="right") - 1
    if idx >= 10:
        ma50_now = d1.iloc[idx]["d_ma50"]
        ma50_prev = d1.iloc[idx - 10]["d_ma50"]
        if pd.notna(ma50_now) and pd.notna(ma50_prev) and ma50_prev > 0:
            f["d_ma50_slope_10d"] = (ma50_now - ma50_prev) / ma50_prev
    f["d_ret_1"] = d["d_ret_1"]
    f["d_ret_3"] = d["d_ret_3"]
    f["d_ret_7"] = d["d_ret_7"]
    f["d_rsi14"] = d["d_rsi14"]
    f["d_vol20"] = d["d_vol20"]
    f["d_atr14_pct"] = d["d_atr14"] / p if pd.notna(d["d_atr14"]) and p > 0 else np.nan

    # range/structure (daily)
    if pd.notna(d["d_high20"]) and pd.notna(d["d_low20"]) and d["d_high20"] > d["d_low20"]:
        f["d_pos_in_20d_range"] = (p - d["d_low20"]) / (d["d_high20"] - d["d_low20"])
    f["d_dist_20d_high"] = (p - d["d_high20"]) / d["d_high20"] if pd.notna(d["d_high20"]) else np.nan
    f["d_dist_20d_low"] = (p - d["d_low20"]) / d["d_low20"] if pd.notna(d["d_low20"]) else np.nan
    f["d_dist_50d_high"] = (p - d["d_high50"]) / d["d_high50"] if pd.notna(d["d_high50"]) else np.nan
    f["d_dist_50d_low"] = (p - d["d_low50"]) / d["d_low50"] if pd.notna(d["d_low50"]) else np.nan

    # all-time high at this point (from available data)
    ath = d1.loc[:idx, "high"].max() if idx >= 0 else np.nan
    f["d_dist_ath"] = (p - ath) / ath if pd.notna(ath) and ath > 0 else np.nan

    # --- swing (4h) ---
    f["h4_ret_1"] = h4r["h4_ret_1"]
    f["h4_ret_3"] = h4r["h4_ret_3"]   # 12h
    f["h4_ret_7"] = h4r["h4_ret_7"]   # ~28h
    f["h4_rsi14"] = h4r["h4_rsi14"]
    f["h4_vol20"] = h4r["h4_vol20"]
    f["h4_atr14_pct"] = h4r["h4_atr14"] / p if pd.notna(h4r["h4_atr14"]) and p > 0 else np.nan
    if pd.notna(h4r["h4_high20"]) and pd.notna(h4r["h4_low20"]) and h4r["h4_high20"] > h4r["h4_low20"]:
        f["h4_pos_in_20bar_range"] = (p - h4r["h4_low20"]) / (h4r["h4_high20"] - h4r["h4_low20"])

    # --- 1h precision ---
    f["h1_ret_1"] = h1r["h1_ret_1"]
    f["h1_ret_3"] = h1r["h1_ret_3"]
    f["h1_rsi14"] = h1r["h1_rsi14"]
    # 1h candle body & wick
    if pd.notna(h1r["open"]) and h1r["open"] > 0:
        body = (h1r["close"] - h1r["open"]) / h1r["open"]
        upper_wick = (h1r["high"] - max(h1r["close"], h1r["open"])) / h1r["open"]
        lower_wick = (min(h1r["close"], h1r["open"]) - h1r["low"]) / h1r["open"]
        f["h1_body_pct"] = body
        f["h1_upper_wick_pct"] = upper_wick
        f["h1_lower_wick_pct"] = lower_wick

    return f

# --- compute features for all roundtrips ---
rt = pd.read_csv(OUT / "roundtrips_xbtusd_clean.csv", parse_dates=["entry_time", "exit_time"])
# ensure UTC
rt["entry_time"] = rt["entry_time"].dt.tz_convert("UTC") if rt["entry_time"].dt.tz is not None else rt["entry_time"].dt.tz_localize("UTC")

print(f"\ncomputing features for {len(rt)} roundtrips...")
feats = []
for ts in rt["entry_time"]:
    f = features_at(ts)
    feats.append(f if f is not None else {})
feat_df = pd.DataFrame(feats)
print(f"features: {list(feat_df.columns)}")
out = pd.concat([rt.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)
out_path = OUT / "roundtrips_with_features.csv"
out.to_csv(out_path, index=False)
print(f"-> {out_path}")

# --- baseline: compute features for every 4h bar as "random point in time" reference ---
print("\ncomputing baseline features (every 4h bar)...")
base_times = h4["close_time"].iloc[200:].sample(n=5000, random_state=42).sort_values()
base_feats = []
for ts in base_times:
    f = features_at(ts)
    if f is not None:
        base_feats.append(f)
bdf = pd.DataFrame(base_feats)
bdf.to_csv(OUT / "baseline_features.csv", index=False)
print(f"baseline sample rows: {len(bdf)}")
print(f"\ndone.")
