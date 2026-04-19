"""Thread A: stratified 200-trade Gemma batch for statistical validation.

Stratification axes (ensures balanced sample):
  - year (2020-2026)
  - direction (long/short)
  - outcome tier (big-win, win, loss, big-loss)

Reuses chart rendering + Gemma query from script 18.
Output: data/derived/gemma_200.csv

After completion, evaluates:
  1. Is "Gemma AGREES" consistently correlated with higher PnL?
  2. Does the effect hold across years and directions?
  3. Is Gemma's shorts-skepticism consistently correct?
"""
import base64, json, re, time, sys
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
OUT = DATA / "charts_200"
OUT.mkdir(exist_ok=True)
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"

# --- sample selection ---
rt = pd.read_csv(DATA / "derived" / "roundtrips_with_exits.csv")
rt["entry_time"] = pd.to_datetime(rt["entry_time"], utc=True, format="ISO8601")
rt = rt[rt["entry_contracts"] >= 1000].copy()
rt["year"] = rt["entry_time"].dt.year

# outcome tier
q25, q75 = rt["net_pnl_btc"].quantile([0.25, 0.75])
def tier(p):
    if p >= q75: return "big_win"
    if p > 0:    return "win"
    if p > q25:  return "loss"
    return "big_loss"
rt["tier"] = rt["net_pnl_btc"].apply(tier)

# stratified sample: 200 trades total, proportional to year × direction
# but ensure each (year, direction) cell has at least a few
target_n = 200
grouped = rt.groupby(["year", "direction"])
samples = []
for (y, d), grp in grouped:
    # proportional: take roughly (target_n * share) from each cell, min 3 max 30
    share = len(grp) / len(rt)
    n = min(max(int(round(target_n * share)), 3), len(grp))
    samples.append(grp.sample(n=n, random_state=42))
sample = pd.concat(samples).drop_duplicates(subset="entry_time")
if len(sample) > target_n:
    sample = sample.sample(n=target_n, random_state=42)
elif len(sample) < target_n:
    extra = rt.loc[~rt.index.isin(sample.index)].sample(n=min(target_n - len(sample), len(rt) - len(sample)), random_state=42)
    sample = pd.concat([sample, extra])
print(f"sample size: {len(sample)}")
print(f"by year/direction:\n{sample.groupby(['year', 'direction']).size()}")
print(f"by tier:\n{sample.groupby('tier').size()}")

# load daily klines once
d1 = pd.read_csv(DATA / "klines" / "1d.csv", parse_dates=["open_time"])
d1 = d1.sort_values("open_time").reset_index(drop=True)

def render_chart(entry_ts, label, window_days=90):
    entry_ts = pd.Timestamp(entry_ts).tz_convert("UTC") if pd.Timestamp(entry_ts).tz else pd.Timestamp(entry_ts).tz_localize("UTC")
    cutoff = entry_ts.floor("D")
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

Respond ONLY with valid JSON:
{
  "trend": "bull" | "bear" | "sideways",
  "setup": string,
  "decision": "LONG" | "SHORT" | "WAIT",
  "conviction": 1-5,
  "reasoning": string
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
            "stream": False, "format": "json",
            "options": {"temperature": 0.2},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try: return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try: return json.loads(m.group(0))
            except: pass
    return {"decision": "ERROR", "raw": content[:200]}

# resume support: skip already-processed
OUT_CSV = DATA / "derived" / "gemma_200.csv"
done = set()
if OUT_CSV.exists():
    prev = pd.read_csv(OUT_CSV, parse_dates=["entry_time"])
    done = set(prev["entry_time"])
    print(f"resuming — {len(done)} already done")

results = []
t0 = time.time()
for idx, (_, row) in enumerate(sample.iterrows()):
    entry_ts = row["entry_time"]
    if entry_ts in done: continue
    label = f"tr_{entry_ts.strftime('%Y%m%d_%H%M%S')}"
    try:
        chart = render_chart(entry_ts, label)
        ans = ask_gemma(chart)
    except Exception as e:
        ans = {"decision": "ERROR", "error": str(e)[:100]}
    results.append({
        "entry_time": entry_ts,
        "year": row["year"],
        "paul_direction": row["direction"],
        "paul_pnl_btc": row["net_pnl_btc"],
        "paul_return_pct": row.get("return_pct", None),
        "paul_tier": row["tier"],
        "gemma_decision": ans.get("decision", "ERROR"),
        "gemma_trend": ans.get("trend"),
        "gemma_setup": ans.get("setup"),
        "gemma_conviction": ans.get("conviction"),
        "gemma_reasoning": ans.get("reasoning"),
    })
    if (idx + 1) % 10 == 0:
        elapsed = time.time() - t0
        rate = (idx + 1 - len(done)) / elapsed * 60
        remaining = (len(sample) - (idx + 1)) / max(rate, 0.1) * 60
        # incremental save every 10
        df_sofar = pd.DataFrame(results)
        if OUT_CSV.exists():
            prev = pd.read_csv(OUT_CSV, parse_dates=["entry_time"])
            df_sofar = pd.concat([prev, df_sofar]).drop_duplicates(subset="entry_time")
        df_sofar.to_csv(OUT_CSV, index=False)
        print(f"  [{idx+1}/{len(sample)}]  {rate:.1f}/min  ETA {remaining:.0f}s  saved checkpoint")
        sys.stdout.flush()

df = pd.DataFrame(results)
if OUT_CSV.exists():
    prev = pd.read_csv(OUT_CSV, parse_dates=["entry_time"])
    df = pd.concat([prev, df]).drop_duplicates(subset="entry_time")
df.to_csv(OUT_CSV, index=False)
print(f"\ntotal time: {(time.time()-t0)/60:.1f}min")
print(f"saved: {OUT_CSV}")

# === ANALYSIS ===
print("\n" + "="*70)
print("STATISTICAL VALIDATION")
print("="*70)

print(f"\nGemma decision distribution:")
print(df["gemma_decision"].value_counts())

df["agrees"] = ((df["paul_direction"] == "long") & (df["gemma_decision"] == "LONG")) | \
               ((df["paul_direction"] == "short") & (df["gemma_decision"] == "SHORT"))
df["disagrees"] = ((df["paul_direction"] == "long") & (df["gemma_decision"] == "SHORT")) | \
                  ((df["paul_direction"] == "short") & (df["gemma_decision"] == "LONG"))
df["wait"] = df["gemma_decision"] == "WAIT"

print("\n--- overall stats by Gemma verdict ---")
for name, mask in [("AGREE", df["agrees"]), ("DISAGREE", df["disagrees"]), ("WAIT", df["wait"])]:
    sub = df[mask]
    if len(sub) == 0: continue
    print(f"{name}: n={len(sub)}  win_rate={(sub['paul_pnl_btc']>0).mean()*100:.0f}%  "
          f"avg_pnl={sub['paul_pnl_btc'].mean():+.3f}  median={sub['paul_pnl_btc'].median():+.3f}  "
          f"total={sub['paul_pnl_btc'].sum():+.2f}")

# statistical significance test: Mann-Whitney U on AGREE vs rest
from scipy import stats
agree_pnl = df[df["agrees"]]["paul_pnl_btc"].values
rest_pnl = df[~df["agrees"]]["paul_pnl_btc"].values
if len(agree_pnl) > 5 and len(rest_pnl) > 5:
    u, p = stats.mannwhitneyu(agree_pnl, rest_pnl, alternative="greater")
    print(f"\nMann-Whitney U: AGREE > REST, p-value = {p:.6f}")
    print(f"  → {'SIGNIFICANT' if p < 0.05 else 'NOT significant'} at α=0.05")

print("\n--- by year ---")
for y in sorted(df["year"].unique()):
    sub_y = df[df["year"] == y]
    if len(sub_y) < 5: continue
    ag = sub_y[sub_y["agrees"]]
    rest = sub_y[~sub_y["agrees"]]
    print(f"{y}: n={len(sub_y)}  agree_n={len(ag)}  "
          f"agree_avg_pnl={ag['paul_pnl_btc'].mean() if len(ag) else 0:+.2f}  "
          f"rest_avg_pnl={rest['paul_pnl_btc'].mean() if len(rest) else 0:+.2f}")

print("\n--- Gemma's shorts skepticism ---")
paul_shorts = df[df["paul_direction"] == "short"]
shorts_gemma_long = paul_shorts[paul_shorts["gemma_decision"] == "LONG"]
shorts_gemma_wait = paul_shorts[paul_shorts["gemma_decision"] == "WAIT"]
shorts_gemma_short = paul_shorts[paul_shorts["gemma_decision"] == "SHORT"]
print(f"Paul's shorts where Gemma said LONG: n={len(shorts_gemma_long)}  avg_pnl={shorts_gemma_long['paul_pnl_btc'].mean() if len(shorts_gemma_long) else 0:+.3f}")
print(f"Paul's shorts where Gemma said WAIT: n={len(shorts_gemma_wait)}  avg_pnl={shorts_gemma_wait['paul_pnl_btc'].mean() if len(shorts_gemma_wait) else 0:+.3f}")
print(f"Paul's shorts where Gemma agreed SHORT: n={len(shorts_gemma_short)}  avg_pnl={shorts_gemma_short['paul_pnl_btc'].mean() if len(shorts_gemma_short) else 0:+.3f}")
