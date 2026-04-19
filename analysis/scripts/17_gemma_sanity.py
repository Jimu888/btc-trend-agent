"""Sanity check: can Gemma4 read a K-line chart and make sensible observations?

Generate a chart of BTC 60 days before a known Paul entry, send to Ollama,
ask for basic pattern description. If answer is plausible, path B is viable.
"""
import base64
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
OUT = DATA / "charts"
OUT.mkdir(exist_ok=True)

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

# pick Paul's biggest win: 2022-01-20 short $37886 -> $35786, +13.2 BTC
# 60 trading days before 2022-01-20 = ~2021-11-20
d1 = pd.read_csv(DATA / "klines" / "1d.csv", parse_dates=["open_time"])
entry_ts = pd.Timestamp("2022-01-20", tz="UTC")
end_idx = d1[d1["open_time"] <= entry_ts].index[-1]
window = d1.iloc[end_idx - 60:end_idx + 1].copy()

# render candlestick-style chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
for _, row in window.iterrows():
    color = "green" if row["close"] >= row["open"] else "red"
    ax.plot([row["open_time"], row["open_time"]], [row["low"], row["high"]],
            color=color, linewidth=1)
    ax.plot([row["open_time"], row["open_time"]],
            [row["open"], row["close"]], color=color, linewidth=4)
# add 20d and 50d MAs
window["ma20"] = window["close"].rolling(20).mean()
window["ma50"] = window["close"].rolling(50).mean()
ax.plot(window["open_time"], window["ma20"], color="blue", linewidth=1, label="MA20")
ax.plot(window["open_time"], window["ma50"], color="orange", linewidth=1, label="MA50")
# mark the entry point
ax.axvline(entry_ts, color="black", linestyle="--", linewidth=1, label="decision point")
ax.set_title(f"BTC daily, 60 days up to {entry_ts.date()}")
ax.set_xlabel("date")
ax.set_ylabel("price (USD)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
plt.xticks(rotation=45)
plt.tight_layout()
chart_path = OUT / "sanity_2022-01-20.png"
plt.savefig(chart_path)
plt.close()
print(f"chart saved: {chart_path}")

# encode and send to Gemma
with open(chart_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

prompt = (
    "You are a crypto trading analyst looking at a BTC daily chart. "
    "Describe what you see in 3-5 short bullet points: "
    "(1) current trend (bull/bear/sideways), "
    "(2) price position relative to MA20 and MA50, "
    "(3) any notable recent pattern (breakout / pullback / consolidation / reversal), "
    "(4) given what you see at the decision point (black dashed line), "
    "would you go LONG, SHORT, or WAIT? Justify in one sentence."
)

print("\n--- sending to Gemma4 ---")
resp = requests.post(
    OLLAMA,
    json={
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt, "images": [img_b64]},
        ],
        "stream": False,
        "options": {"temperature": 0.3},
    },
    timeout=180,
)
resp.raise_for_status()
data = resp.json()
print(f"\nGemma response:")
print(data["message"]["content"])
print(f"\ntokens: prompt={data.get('prompt_eval_count', '?')}, "
      f"response={data.get('eval_count', '?')}, "
      f"total_s={data.get('total_duration', 0)/1e9:.1f}")

print(f"\nGround truth: Paul opened a SHORT on this date at $37886 and closed "
      f"5 days later at $35786 for his BIGGEST WIN of +13.2 BTC.")
