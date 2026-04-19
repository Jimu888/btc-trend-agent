"""Download BTC historical klines from Binance public futures API.

No authentication required. Saves CSVs to ./data/klines/.

Usage:
    python examples/fetch_klines.py              # fetches 1d, 4h
    python examples/fetch_klines.py 1h 5m        # fetches specific intervals
"""
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

OUT = Path(__file__).parent.parent / "data" / "klines"
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://fapi.binance.com/fapi/v1/klines"


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int):
    rows = []
    cur = start_ms
    while cur < end_ms:
        for attempt in range(5):
            try:
                r = requests.get(BASE, params={
                    "symbol": symbol, "interval": interval,
                    "startTime": cur, "endTime": end_ms, "limit": 1500,
                }, timeout=30)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                wait = 2 ** attempt
                print(f"  retry {attempt+1} in {wait}s: {e}")
                time.sleep(wait)
        else:
            raise RuntimeError("exhausted retries")
        if not batch:
            break
        rows.extend(batch)
        cur = batch[-1][0] + 1
        time.sleep(0.2)
        if len(batch) < 1500:
            break
    return rows


def save(interval: str, rows):
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.drop(columns=["ignore"]).sort_values("open_time").reset_index(drop=True)
    path = OUT / f"{interval}.csv"
    df.to_csv(path, index=False)
    print(f"  {interval}: {len(df):,} rows  range {df['open_time'].min()} → {df['open_time'].max()}  → {path}")


if __name__ == "__main__":
    intervals = sys.argv[1:] if len(sys.argv) > 1 else ["1d", "4h"]
    start = int(datetime(2020, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end = int(time.time() * 1000)
    for interval in intervals:
        print(f"\nfetching BTCUSDT {interval}…")
        rows = fetch("BTCUSDT", interval, start, end)
        save(interval, rows)
    print("\ndone.")
