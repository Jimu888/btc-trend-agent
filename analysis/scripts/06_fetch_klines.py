"""Phase 3.1: fetch BTC historical klines from Binance public API.

Pulls BTCUSDT perpetual futures klines (no auth needed).
Granularities: 1d, 4h, 1h. Coverage: 2020-05-01 -> 2026-04-18.

Output: data/klines/{1d,4h,1h}.csv
"""
import requests
import pandas as pd
import time
from pathlib import Path
from datetime import datetime, timezone

OUT = Path("/Users/jimu/btc-trader-analysis/data/klines")
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://fapi.binance.com/fapi/v1/klines"  # USDT-M perpetual

def fetch(symbol, interval, start_ms, end_ms):
    """Fetch klines in pages of 1500 (binance limit). Robust to transient drops."""
    out = []
    cur = start_ms
    while cur < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cur,
            "endTime": end_ms,
            "limit": 1500,
        }
        for attempt in range(5):
            try:
                r = requests.get(BASE, params=params, timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"    attempt {attempt+1} failed ({e.__class__.__name__}), retry in {wait}s")
                time.sleep(wait)
        else:
            raise RuntimeError("exhausted retries")
        if not batch:
            break
        out.extend(batch)
        last_open_ms = batch[-1][0]
        cur = last_open_ms + 1
        time.sleep(0.2)
        if len(batch) < 1500:
            break
    return out

cols = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"]

start = int(datetime(2020, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
end = int(datetime(2026, 4, 19, tzinfo=timezone.utc).timestamp() * 1000)

for interval in ["1h"]:   # skip the already-done ones; re-run only 1h
    print(f"\nfetching BTCUSDT {interval}...")
    rows = fetch("BTCUSDT", interval, start, end)
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base", "taker_buy_quote"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)
    path = OUT / f"{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  rows: {len(df):,}  range: {df['open_time'].min()} -> {df['open_time'].max()}  -> {path}")
