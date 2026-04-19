"""Generate equity curve comparison chart for deliverable."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
FEE = 0.001; SLIP = 0.0005

def run(ma_long, slope_w):
    data = d1.copy()
    data["ma_long"] = c.rolling(ma_long).mean()
    data["ma_mid"] = c.rolling(50).mean()
    data["ma_mid_prev"] = data["ma_mid"].shift(slope_w)
    eq = 1.0; pos = None; eq_ts = []
    warmup = max(ma_long, 50 + slope_w) + 1
    for i in range(warmup, len(data)):
        b = data.iloc[i]
        if any(pd.isna([b["ma_long"], b["ma_mid"], b["ma_mid_prev"]])):
            eq_ts.append((b["close_time"], eq)); continue

        if pos is not None:
            if b["close"] < b["ma_long"]:
                exit_p = b["close"] * (1 - SLIP)
                ret = (exit_p - pos["entry"]) / pos["entry"] - 2 * FEE
                eq *= (1 + ret)
                pos = None
            else:
                # mark-to-market for plotting
                mtm = eq * (1 + (b["close"] - pos["entry"]) / pos["entry"] - 2 * FEE)
                eq_ts.append((b["close_time"], mtm)); continue

        eq_ts.append((b["close_time"], eq))
        if pos is None:
            if (b["close"] > b["ma_long"] and b["close"] > b["ma_mid"]
                and b["ma_mid"] > b["ma_mid_prev"]):
                pos = {"entry": b["close"] * (1 + SLIP)}

    return pd.DataFrame(eq_ts, columns=["t", "eq"])

v5e = run(200, 10)
v5f = run(150, 10)
v5_short = run(100, 5)

# buy-and-hold from first v5e date
start_t = v5e["t"].iloc[0]
bh_mask = d1["close_time"] >= start_t
bh = d1.loc[bh_mask, ["close_time", "close"]].copy()
bh["eq"] = bh["close"] / bh["close"].iloc[0]

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=110,
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

ax.plot(bh["close_time"], bh["eq"], color="gray", linewidth=1.5, label=f"BTC buy-and-hold (final {bh['eq'].iloc[-1]:.2f}x)")
ax.plot(v5e["t"], v5e["eq"], color="tab:blue", linewidth=1.5, label=f"v5e MA200 (final {v5e['eq'].iloc[-1]:.2f}x)")
ax.plot(v5f["t"], v5f["eq"], color="tab:green", linewidth=2.0, label=f"v5f MA150 (final {v5f['eq'].iloc[-1]:.2f}x) ★")
ax.plot(v5_short["t"], v5_short["eq"], color="tab:orange", linewidth=1.5, label=f"v5_short MA100/5 (final {v5_short['eq'].iloc[-1]:.2f}x)")

ax.set_yscale("log")
ax.set_ylabel("Equity (1 BTC start, log)")
ax.set_title("BTC trend-following strategies vs buy-and-hold (2020-2026, with 0.1% fee + 0.05% slippage)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3, which="both")

# drawdown subplot
for df, name, col in [(bh, "BH", "gray"), (v5e, "v5e", "tab:blue"), (v5f, "v5f", "tab:green"), (v5_short, "v5_short", "tab:orange")]:
    peak = df["eq"].cummax()
    dd = (df["eq"] - peak) / peak * 100
    ax2.plot(df["t" if "t" in df.columns else "close_time"], dd, color=col, linewidth=1)

ax2.set_ylabel("Drawdown (%)")
ax2.set_xlabel("Date")
ax2.grid(True, alpha=0.3)
ax2.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
out = DATA / "deliverables" / "equity_curves.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out)
plt.close()
print(f"saved -> {out}")

# print comparison
print("\nSummary:")
for df, name in [(bh, "BTC buy-and-hold"), (v5e, "v5e (MA200/10)"),
                  (v5f, "v5f (MA150/10) ★"), (v5_short, "v5_short (MA100/5)")]:
    peak = df["eq"].cummax()
    dd_pct = ((df["eq"] - peak) / peak).min() * 100
    final = df["eq"].iloc[-1]
    ret_pct = (final - 1) * 100
    print(f"  {name:30s}  ret={ret_pct:+7.1f}%  max_dd={dd_pct:+6.1f}%  ratio={ret_pct/abs(dd_pct):.2f}")
