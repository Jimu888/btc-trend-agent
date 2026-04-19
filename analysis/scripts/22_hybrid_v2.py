"""Hybrid Agent v2 — cleaner architecture.

Key changes from v1:
  - Full 100% capital to whichever engine signals (not 70/30 sub-accounts)
  - Priority: TREND > SWING
  - Swing only activates when TREND is flat (no signal)
  - If TREND signal fires while SWING is open: close swing, open trend
  - Swing now supports Gemma gatekeeper (optional flag)

The Gemma gatekeeper re-uses decisions cached from the 200-trade batch
where available (by nearest timestamp match), otherwise calls Gemma live.
This lets us backtest with Gemma filter at negligible additional time cost.
"""
import base64, json, re
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"
OLLAMA = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"
CHART_DIR = DATA / "charts_v2"
CHART_DIR.mkdir(exist_ok=True)

FEE = 0.001
SLIP = 0.0005

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
d1["ma20"] = c.rolling(20).mean()
d1["ma50"] = c.rolling(50).mean()
d1["ma200"] = c.rolling(200).mean()
d1["ma50_prev10"] = d1["ma50"].shift(10)
tr = pd.concat([d1["high"]-d1["low"], (d1["high"]-c.shift(1)).abs(),
                (d1["low"]-c.shift(1)).abs()], axis=1).max(axis=1)
d1["atr14"] = tr.rolling(14).mean()
delta = c.diff()
up = delta.clip(lower=0).rolling(14).mean()
dn = (-delta.clip(upper=0)).rolling(14).mean()
d1["rsi14"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))

def render_chart_at(i, label):
    """Render chart of days up-to-and-including bar i (for Gemma consultation)."""
    start_i = max(0, i - 89)
    w = d1.iloc[start_i:i+1].copy()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=90,
                                  gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for _, row in w.iterrows():
        color = "green" if row["close"] >= row["open"] else "red"
        ax.plot([row["open_time"], row["open_time"]], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax.plot([row["open_time"], row["open_time"]], [row["open"], row["close"]], color=color, linewidth=3)
    ax.plot(w["open_time"], w["ma20"], color="blue", linewidth=1, label="MA20")
    ax.plot(w["open_time"], w["ma50"], color="orange", linewidth=1, label="MA50")
    if w["ma200"].notna().any():
        ax.plot(w["open_time"], w["ma200"], color="purple", linewidth=1, label="MA200")
    ax.set_ylabel("BTC/USD"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    colors = ["green" if cl>=op else "red" for cl,op in zip(w["close"],w["open"])]
    ax2.bar(w["open_time"], w["volume"], color=colors, alpha=0.7, width=1)
    ax2.set_ylabel("Vol"); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    plt.xticks(rotation=45); plt.tight_layout()
    path = CHART_DIR / f"{label}.png"
    plt.savefig(path); plt.close()
    return path

GEMMA_PROMPT = """You are a disciplined BTC swing trader. Analyze this BTC daily chart.
The right edge is "NOW" — decide whether to enter.

JSON only:
{
  "trend": "bull"|"bear"|"sideways",
  "setup": string,
  "decision": "LONG"|"SHORT"|"WAIT",
  "conviction": 1-5,
  "reasoning": string
}
"""

# cache on disk for consistency across runs
GEMMA_CACHE_FILE = DATA / "derived" / "gemma_cache.json"
try:
    gemma_cache = json.load(open(GEMMA_CACHE_FILE))
except (FileNotFoundError, json.JSONDecodeError):
    gemma_cache = {}

def ask_gemma(i):
    b = d1.iloc[i]
    key = str(b["close_time"].date())
    if key in gemma_cache:
        return gemma_cache[key]
    try:
        chart = render_chart_at(i, f"bar_{key}")
        with open(chart, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        resp = requests.post(OLLAMA, json={
            "model": MODEL,
            "messages":[{"role":"user","content":GEMMA_PROMPT,"images":[img_b64]}],
            "stream": False, "format": "json", "options": {"temperature": 0.2},
        }, timeout=180)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        try: ans = json.loads(content)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            ans = json.loads(m.group(0)) if m else {"decision":"ERROR"}
    except Exception as e:
        ans = {"decision":"ERROR","error":str(e)[:100]}
    gemma_cache[key] = ans
    # periodic save
    json.dump(gemma_cache, open(GEMMA_CACHE_FILE, "w"))
    return ans

# ---------- backtest ----------
def backtest(use_gemma_gate=False):
    eq = 1.0
    pos = None    # {"engine":"trend|swing", "direction":"long|short", ...}
    trades = []
    eq_ts = []
    dd_peak = 1.0
    dd_min = 0.0

    for i in range(210, len(d1)):
        b = d1.iloc[i]
        if any(pd.isna([b["ma200"], b["ma50"], b["ma50_prev10"], b["atr14"], b["rsi14"]])):
            eq_ts.append((b["close_time"], eq)); continue

        # mark equity
        mtm_eq = eq
        if pos is not None:
            sign = 1 if pos["direction"] == "long" else -1
            unreal = sign * (b["close"] - pos["entry"]) / pos["entry"] * pos.get("size", 1.0) - 2 * FEE * pos.get("size",1.0)
            mtm_eq = eq * (1 + unreal)
        dd_peak = max(dd_peak, mtm_eq)
        dd_min = min(dd_min, (mtm_eq - dd_peak) / dd_peak)

        # --- manage open pos ---
        if pos is not None:
            sign = 1 if pos["direction"] == "long" else -1
            atr_frac = pos["atr_entry"] / pos["entry"]
            fav_p = b["high"] if sign == 1 else b["low"]
            adv_p = b["low"] if sign == 1 else b["high"]
            fav = sign * (fav_p - pos["entry"]) / pos["entry"]
            adv = sign * (adv_p - pos["entry"]) / pos["entry"]
            pos["mfe"] = max(pos["mfe"], fav)
            pos["mae"] = min(pos["mae"], adv)

            # trend engine exits
            exit_price, reason = None, None
            if pos["engine"] == "trend":
                if b["close"] < b["ma200"]:
                    exit_price = b["close"] * (1 - SLIP)
                    reason = "ma200_break"
            else:
                # swing: ATR trail + RSI flip + time stop
                if pos["mfe"] >= 1.0 * atr_frac and pos["stop_pct"] < 0:
                    pos["stop_pct"] = 0.0
                if pos["mfe"] >= 2.0 * atr_frac:
                    trail = pos["mfe"] - 1.0 * atr_frac
                    if trail > pos["stop_pct"]: pos["stop_pct"] = trail
                stop_price = pos["entry"] * (1 + sign * pos["stop_pct"])
                if sign == 1 and b["low"] <= stop_price:
                    exit_price, reason = stop_price, "atr_stop"
                elif sign == -1 and b["high"] >= stop_price:
                    exit_price, reason = stop_price, "atr_stop"
                if exit_price is None:
                    if sign == 1 and b["rsi14"] > 60:
                        exit_price, reason = b["close"] * (1 - SLIP), "rsi_flip"
                    elif sign == -1 and b["rsi14"] < 40:
                        exit_price, reason = b["close"] * (1 + SLIP), "rsi_flip"
                if exit_price is None:
                    days_in = (b["close_time"] - pos["entry_time"]).days
                    if days_in >= 14:
                        exit_price, reason = b["close"] * (1 - SLIP * sign), "time"

            # trend signal preemption: if trend would enter and swing is open, close swing and open trend
            if exit_price is None and pos["engine"] == "swing":
                if b["close"] > b["ma200"] and b["close"] > b["ma50"] and b["ma50"] > b["ma50_prev10"]:
                    exit_price, reason = b["close"] * (1 - SLIP * sign), "trend_preempt"

            if exit_price is not None:
                ret = sign * (exit_price - pos["entry"]) / pos["entry"]
                net_ret = ret - 2 * FEE
                eq *= (1 + net_ret)
                trades.append({
                    "engine": pos["engine"], "direction": pos["direction"],
                    "entry_time": pos["entry_time"], "exit_time": b["close_time"],
                    "entry": pos["entry"], "exit": exit_price, "ret": net_ret,
                    "mfe": pos["mfe"], "mae": pos["mae"],
                    "setup": pos.get("setup", pos["engine"]), "reason": reason,
                    "days": (b["close_time"] - pos["entry_time"]).days,
                    "gemma_asked": pos.get("gemma_asked", False),
                    "gemma_decision": pos.get("gemma_decision"),
                })
                pos = None

        # --- entries ---
        if pos is None:
            # priority 1: trend engine
            if b["close"] > b["ma200"] and b["close"] > b["ma50"] and b["ma50"] > b["ma50_prev10"]:
                pos = {"engine":"trend","direction":"long","entry":b["close"]*(1+SLIP),
                       "entry_time":b["close_time"],"atr_entry":b["atr14"],"size":1.0,
                       "mfe":0.0,"mae":0.0,"stop_pct":-0.50}
                continue

            # priority 2: swing engine (with optional Gemma gate)
            bull_regime = b["close"] > b["ma200"]
            setup = None
            direction = None
            if bull_regime and b["rsi14"] < 40 and b["close"] < b["ma20"]:
                setup, direction = "bull_pullback_long", "long"
            elif (not bull_regime) and b["rsi14"] > 60 and b["close"] > b["ma20"]:
                setup, direction = "bear_rally_short", "short"

            if setup is not None:
                ok = True
                gemma_ans = None
                if use_gemma_gate:
                    gemma_ans = ask_gemma(i)
                    dec = gemma_ans.get("decision", "ERROR")
                    # only enter if Gemma agrees with direction (strict)
                    if direction == "long" and dec != "LONG": ok = False
                    if direction == "short" and dec != "SHORT": ok = False
                if ok:
                    entry_p = b["close"] * (1 + SLIP * (1 if direction == "long" else -1))
                    pos = {"engine":"swing","direction":direction,"entry":entry_p,
                           "entry_time":b["close_time"],"atr_entry":b["atr14"],
                           "setup":setup,"size":1.0,"mfe":0.0,"mae":0.0,
                           "stop_pct":-1.5 * b["atr14"] / entry_p,
                           "gemma_asked": use_gemma_gate,
                           "gemma_decision": gemma_ans.get("decision") if gemma_ans else None}

        eq_ts.append((b["close_time"], eq))

    # close at end
    if pos is not None:
        last = d1.iloc[-1]
        sign = 1 if pos["direction"] == "long" else -1
        exit_p = last["close"] * (1 - SLIP * sign)
        ret = sign * (exit_p - pos["entry"]) / pos["entry"]
        eq *= (1 + ret - 2 * FEE)

    return pd.DataFrame(trades), eq, eq_ts, dd_min

# BASELINE (no Gemma)
print("=" * 70)
print("HYBRID AGENT v2 — BASELINE (no Gemma gate)")
print("=" * 70)
tr, eq, ts, dd = backtest(use_gemma_gate=False)
print(f"trades: {len(tr)}  (trend: {(tr['engine']=='trend').sum()}  swing: {(tr['engine']=='swing').sum()})")
print(f"final eq: {eq:.4f}  ({(eq-1)*100:+.1f}%)   max DD: {dd*100:.1f}%   ret/|dd|: {(eq-1)/abs(dd+1e-9):.2f}")
print("\nby engine:")
print(tr.groupby("engine").agg(
    n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
    avg=("ret","mean"), total_ret=("ret","sum"),
).round(4).to_string())
print("\nby setup:")
if "setup" in tr.columns:
    print(tr.groupby("setup").agg(
        n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
        avg=("ret","mean"), total=("ret","sum"),
    ).round(4).to_string())

tr.to_csv(DATA / "derived" / "hybrid_v2_baseline.csv", index=False)

# GEMMA-GATED (Gemma filter on swing candidates)
print("\n" + "=" * 70)
print("HYBRID AGENT v2 — WITH GEMMA GATE on swing signals")
print("=" * 70)
print("(this will call Gemma on each swing candidate; cache persisted to disk)")
tr_g, eq_g, ts_g, dd_g = backtest(use_gemma_gate=True)
print(f"trades: {len(tr_g)}  (trend: {(tr_g['engine']=='trend').sum()}  swing: {(tr_g['engine']=='swing').sum()})")
print(f"final eq: {eq_g:.4f}  ({(eq_g-1)*100:+.1f}%)   max DD: {dd_g*100:.1f}%   ret/|dd|: {(eq_g-1)/abs(dd_g+1e-9):.2f}")
print("\nby engine:")
print(tr_g.groupby("engine").agg(
    n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
    avg=("ret","mean"), total_ret=("ret","sum"),
).round(4).to_string())
print("\nby setup (with Gemma gate):")
if "setup" in tr_g.columns:
    print(tr_g.groupby("setup").agg(
        n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
        avg=("ret","mean"), total=("ret","sum"),
    ).round(4).to_string())
tr_g.to_csv(DATA / "derived" / "hybrid_v2_gemma.csv", index=False)

# save cache
json.dump(gemma_cache, open(GEMMA_CACHE_FILE, "w"))
print(f"\nGemma cache size: {len(gemma_cache)} entries")
