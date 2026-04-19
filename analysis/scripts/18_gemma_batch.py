"""Batch test: does Gemma4 make sensible trade decisions that match or beat Paul?

Workflow:
  1. Sample 50 Paul trades (mix of top wins, top losses, random)
  2. For each: render 90-day chart up to entry time (NO lookahead)
  3. Ask Gemma for structured JSON decision: LONG / SHORT / WAIT, conviction 1-5
  4. Compare to Paul's actual direction and outcome
  5. Analyze:
     - agreement rate
     - avg PnL when Gemma agrees with Paul
     - avg PnL when Gemma disagrees
     - conviction calibration
"""
import base64, json, re, time
import pandas as pd
import numpy as np
import requests
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

# load Paul's trades
rt = pd.read_csv(DATA / "derived" / "roundtrips_with_exits.csv")
rt["entry_time"] = pd.to_datetime(rt["entry_time"], utc=True, format="ISO8601")
rt = rt[rt["entry_contracts"] >= 1000].copy()   # substantial trades only (filter noise)

# sample: top 20 wins + top 20 losses + 10 random = 50
top_wins = rt.nlargest(20, "net_pnl_btc")
top_losses = rt.nsmallest(20, "net_pnl_btc")
random_sample = rt.sample(10, random_state=42)
sample = pd.concat([top_wins, top_losses, random_sample]).drop_duplicates(subset="entry_time")
print(f"sample size: {len(sample)}")

# load klines
d1 = pd.read_csv(DATA / "klines" / "1d.csv", parse_dates=["open_time"])
d1 = d1.sort_values("open_time").reset_index(drop=True)

def render_chart(entry_ts, label, window_days=90):
    """Render 90 daily candles up to (but NOT including) entry_ts.

    Keeps lookahead bias out: the agent sees only bars fully closed before entry.
    """
    entry_ts = pd.Timestamp(entry_ts).tz_convert("UTC") if pd.Timestamp(entry_ts).tz else pd.Timestamp(entry_ts).tz_localize("UTC")
    # find last daily bar that fully closed before entry
    cutoff = entry_ts.floor("D")   # entry day's 00:00 UTC
    end_idx = d1[d1["open_time"] < cutoff].index[-1]
    start_idx = max(0, end_idx - window_days + 1)
    w = d1.iloc[start_idx:end_idx + 1].copy()
    w["ma20"] = d1["close"].rolling(20).mean().iloc[start_idx:end_idx + 1]
    w["ma50"] = d1["close"].rolling(50).mean().iloc[start_idx:end_idx + 1]
    w["ma200"] = d1["close"].rolling(200).mean().iloc[start_idx:end_idx + 1]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=90,
                                   gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for _, row in w.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax.plot([row["open_time"], row["open_time"]], [row["low"], row["high"]],
                color=color, linewidth=0.8)
        ax.plot([row["open_time"], row["open_time"]], [row["open"], row["close"]],
                color=color, linewidth=3)
    ax.plot(w["open_time"], w["ma20"], color="blue", linewidth=1, label="MA20")
    ax.plot(w["open_time"], w["ma50"], color="orange", linewidth=1, label="MA50")
    if w["ma200"].notna().any():
        ax.plot(w["open_time"], w["ma200"], color="purple", linewidth=1, label="MA200")
    ax.set_ylabel("BTC / USD")
    ax.set_title(f"BTC daily — {window_days} days up to {cutoff.date()}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    # volume
    colors = ["green" if c >= o else "red" for c, o in zip(w["close"], w["open"])]
    ax2.bar(w["open_time"], w["volume"], color=colors, alpha=0.7, width=1)
    ax2.set_ylabel("Volume")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = OUT / f"{label}.png"
    plt.savefig(path)
    plt.close()
    return path

PROMPT = """You are a disciplined BTC swing trader. Analyze this BTC daily chart.
The right edge is "NOW" — you must decide whether to enter a position today.

Respond ONLY with valid JSON (no prose outside JSON). Schema:
{
  "trend": "bull" | "bear" | "sideways",
  "setup": string,          // one short phrase, e.g. "pullback in uptrend", "breakdown", "range consolidation"
  "decision": "LONG" | "SHORT" | "WAIT",
  "conviction": 1-5,        // 5 = strong signal, 1 = marginal
  "reasoning": string       // one sentence
}
"""

def ask_gemma(img_path):
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    resp = requests.post(
        OLLAMA,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
            "stream": False,
            "format": "json",   # force JSON mode
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # try to extract first {...} block
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
    return {"decision": "ERROR", "raw": content}

# run batch
results = []
t0 = time.time()
for idx, (_, row) in enumerate(sample.iterrows()):
    entry_ts = row["entry_time"]
    label = f"tr_{entry_ts.strftime('%Y%m%d_%H%M')}"
    chart = render_chart(entry_ts, label)
    try:
        ans = ask_gemma(chart)
    except Exception as e:
        ans = {"decision": "ERROR", "error": str(e)}
    results.append({
        "entry_time": entry_ts,
        "paul_direction": row["direction"],
        "paul_pnl_btc": row["net_pnl_btc"],
        "paul_return_pct": row["return_pct"] if "return_pct" in row else None,
        "paul_mfe": row["mfe_pct"] if "mfe_pct" in row else None,
        "paul_mae": row["mae_pct"] if "mae_pct" in row else None,
        "gemma_decision": ans.get("decision", "ERROR"),
        "gemma_trend": ans.get("trend"),
        "gemma_setup": ans.get("setup"),
        "gemma_conviction": ans.get("conviction"),
        "gemma_reasoning": ans.get("reasoning"),
    })
    if (idx + 1) % 5 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1) / elapsed
        remaining = (len(sample) - (idx + 1)) / rate
        print(f"  [{idx+1}/{len(sample)}]  {rate:.1f}/min  ETA {remaining:.0f}s")

df = pd.DataFrame(results)
df.to_csv(DATA / "derived" / "gemma_batch_50.csv", index=False)

# === ANALYSIS ===
print("\n" + "="*70)
print("GEMMA vs PAUL")
print("="*70)

# direction agreement
df["gemma_long"] = df["gemma_decision"] == "LONG"
df["gemma_short"] = df["gemma_decision"] == "SHORT"
df["gemma_wait"] = df["gemma_decision"] == "WAIT"
df["paul_long"] = df["paul_direction"] == "long"

# when Paul goes long vs Gemma
print("\nGemma decision distribution:")
print(df["gemma_decision"].value_counts())

print("\nAgreement matrix (rows=Paul, cols=Gemma):")
cross = pd.crosstab(df["paul_direction"], df["gemma_decision"])
print(cross)

# when Gemma agrees with Paul, what's average PnL?
df["agrees"] = ((df["paul_direction"] == "long") & (df["gemma_decision"] == "LONG")) | \
               ((df["paul_direction"] == "short") & (df["gemma_decision"] == "SHORT"))
df["disagrees"] = ((df["paul_direction"] == "long") & (df["gemma_decision"] == "SHORT")) | \
                  ((df["paul_direction"] == "short") & (df["gemma_decision"] == "LONG"))
df["wait"] = df["gemma_decision"] == "WAIT"

agree = df[df["agrees"]]
disagree = df[df["disagrees"]]
waited = df[df["wait"]]

print(f"\nGemma AGREES with Paul: n={len(agree)}")
print(f"  avg PnL: {agree['paul_pnl_btc'].mean():+.3f} BTC  median: {agree['paul_pnl_btc'].median():+.3f}")
print(f"  win rate: {(agree['paul_pnl_btc']>0).mean()*100:.0f}%")
print(f"Gemma DISAGREES with Paul: n={len(disagree)}")
print(f"  avg PnL Paul got: {disagree['paul_pnl_btc'].mean():+.3f} BTC  median: {disagree['paul_pnl_btc'].median():+.3f}")
print(f"  win rate: {(disagree['paul_pnl_btc']>0).mean()*100:.0f}%")
print(f"  → if Gemma had been right, these would be INVERTED for Paul")
print(f"Gemma says WAIT (skip trade): n={len(waited)}")
print(f"  avg PnL Paul got: {waited['paul_pnl_btc'].mean():+.3f} BTC")
print(f"  → WAIT avoids trades with avg={waited['paul_pnl_btc'].mean():+.3f} BTC")

# conviction calibration
if df["gemma_conviction"].notna().sum() > 0:
    print(f"\nGemma conviction calibration:")
    conv = df[df["agrees"] & df["gemma_conviction"].notna()].copy()
    if len(conv) > 0:
        for c in sorted(conv["gemma_conviction"].dropna().unique()):
            sub = conv[conv["gemma_conviction"] == c]
            print(f"  conviction={c}: n={len(sub)}  avg PnL={sub['paul_pnl_btc'].mean():+.3f}  win={(sub['paul_pnl_btc']>0).mean()*100:.0f}%")

# show some disagreement cases
print(f"\nSample DISAGREEMENT cases (where Paul might have been wrong):")
for _, r in disagree.head(5).iterrows():
    print(f"  {r['entry_time'].date()}  Paul={r['paul_direction']} (PnL={r['paul_pnl_btc']:+.2f})  "
          f"Gemma={r['gemma_decision']}  reason: {r.get('gemma_reasoning', '')}")

print(f"\ntotal time: {(time.time()-t0):.0f}s")
