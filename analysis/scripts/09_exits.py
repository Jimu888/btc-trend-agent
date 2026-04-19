"""Phase 4.1: exit behavior analysis.

For each roundtrip:
  - MFE (max favorable excursion): best unrealized profit reached during trade
  - MAE (max adverse excursion):   worst unrealized loss reached during trade
  - exit efficiency = actual_return / mfe  (close to 1 = took profit near peak)
  - exit_trigger classification (profit-take, stop-loss, timeout, reversal)
  - validate "holding > 7d kills edge" hypothesis with causal analysis:
    for every trade, ask what PnL would have been if exited at +1h, +1d, +3d, +7d

Uses 1h klines for intra-trade price tracking.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
OUT = DATA / "derived"

rt = pd.read_csv(OUT / "roundtrips_with_features.csv")
for c in ["entry_time", "exit_time"]:
    rt[c] = pd.to_datetime(rt[c], utc=True, format="ISO8601")

h1 = pd.read_csv(DATA / "klines" / "1h.csv")
h1["open_time"] = pd.to_datetime(h1["open_time"], utc=True)
h1["close_time"] = pd.to_datetime(h1["close_time"], utc=True)
h1 = h1.sort_values("open_time").reset_index(drop=True)
# numpy datetime64[ns] array — strip tz for searchsorted (all UTC anyway)
h1_times = h1["open_time"].dt.tz_localize(None).values

def to_naive(t):
    ts = pd.Timestamp(t)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts)

def slice_hours(t_start, t_end):
    i0 = np.searchsorted(h1_times, to_naive(t_start), side="left")
    i1 = np.searchsorted(h1_times, to_naive(t_end),   side="right")
    return h1.iloc[i0:i1]

# --- compute MFE, MAE, exit efficiency, and counterfactual-exit PnLs ---
rows = []
CHECKPOINTS_H = [1, 6, 24, 72, 168, 336]   # +1h, +6h, +1d, +3d, +7d, +14d

for i, r in rt.iterrows():
    bars = slice_hours(r["entry_time"], r["exit_time"])
    if len(bars) == 0:
        rows.append({})
        continue
    entry_p = r["avg_entry"]
    exit_p = r["avg_exit"]
    sign = 1 if r["direction"] == "long" else -1

    highs = bars["high"].values
    lows = bars["low"].values
    closes = bars["close"].values

    # for a long: MFE uses highs, MAE uses lows
    # for a short: MFE uses lows (price drops = profit), MAE uses highs
    if sign == 1:
        best_p = highs.max()
        worst_p = lows.min()
    else:
        best_p = lows.min()
        worst_p = highs.max()

    mfe_pct = sign * (best_p - entry_p) / entry_p * 100
    mae_pct = sign * (worst_p - entry_p) / entry_p * 100   # typically negative
    exit_ret = sign * (exit_p - entry_p) / entry_p * 100
    efficiency = exit_ret / mfe_pct if mfe_pct > 0 else np.nan

    # counterfactual: PnL if exited exactly at +1h, +6h, +1d, +3d, +7d, +14d from entry
    cf = {}
    for h in CHECKPOINTS_H:
        target = r["entry_time"] + pd.Timedelta(hours=h)
        if target >= r["exit_time"]:
            # if we'd still be holding — use his actual exit (capped by reality)
            cf[f"pnl_if_exit_{h}h"] = exit_ret
        else:
            # find 1h close nearest target
            idx = np.searchsorted(h1_times, to_naive(target), side="right") - 1
            if idx < 0:
                cf[f"pnl_if_exit_{h}h"] = np.nan
                continue
            cf_price = h1.iloc[idx]["close"]
            cf_ret = sign * (cf_price - entry_p) / entry_p * 100
            cf[f"pnl_if_exit_{h}h"] = cf_ret

    rows.append({
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "exit_return_pct": exit_ret,
        "exit_efficiency": efficiency,
        "bars_held_1h": len(bars),
        **cf,
    })

cols_df = pd.DataFrame(rows)
rt_e = pd.concat([rt.reset_index(drop=True), cols_df.reset_index(drop=True)], axis=1)
rt_e.to_csv(OUT / "roundtrips_with_exits.csv", index=False)

def hdr(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# === 1. MFE / MAE / efficiency summary ===
hdr("1. MFE / MAE / exit efficiency")
print("mfe_pct (best unrealized profit %):")
print(rt_e["mfe_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
print("\nmae_pct (worst unrealized drawdown %):")
print(rt_e["mae_pct"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
print("\nexit_efficiency (actual_ret / mfe, only positive-mfe trades):")
eff = rt_e[rt_e["mfe_pct"] > 0]["exit_efficiency"]
print(eff.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

# === 2. Classify exits ===
hdr("2. Exit trigger classification")
def classify(r):
    if pd.isna(r["exit_return_pct"]):
        return "unknown"
    # target zones
    ret = r["exit_return_pct"]
    mfe = r["mfe_pct"]
    mae = r["mae_pct"]
    hold_h = r["holding_hours"]
    # large win taken: return captured most of MFE
    if ret > 2 and ret / mfe > 0.6:
        return "profit_take_strong"
    # modest win with big give-back from MFE (reversal)
    if ret > 0 and mfe > 3 and ret / mfe < 0.4:
        return "reversal_kept_small_win"
    # modest win
    if 0 < ret <= 2:
        return "profit_take_small"
    # stop-loss style
    if ret < -1 and abs(ret) / abs(mae if mae < 0 else -0.001) > 0.6:
        return "stop_loss"
    # small loss, gave up
    if -1 <= ret < 0:
        return "gave_up_small"
    return "other"

rt_e["exit_class"] = rt_e.apply(classify, axis=1)
cls_stats = rt_e.groupby("exit_class").agg(
    n=("net_pnl_btc", "size"),
    share=("net_pnl_btc", lambda s: len(s) / len(rt_e)),
    median_ret=("exit_return_pct", "median"),
    median_mfe=("mfe_pct", "median"),
    median_mae=("mae_pct", "median"),
    median_hold_h=("holding_hours", "median"),
    total_btc=("net_pnl_btc", "sum"),
).sort_values("total_btc", ascending=False)
print(cls_stats.round(3).to_string())

# === 3. Causal hold-time analysis ===
hdr("3. Counterfactual: what if he had exited at +Xh?")
# for each horizon, compute aggregate PnL if everyone had exited at that time
# limitation: using price return %, not BTC PnL (same direction, comparable)
cf_agg = {}
for h in CHECKPOINTS_H:
    col = f"pnl_if_exit_{h}h"
    cf_agg[f"+{h}h"] = {
        "mean_ret_pct": rt_e[col].mean(),
        "median_ret_pct": rt_e[col].median(),
        "win_rate": (rt_e[col] > 0).mean(),
        "sum_ret_pct": rt_e[col].sum(),
    }
cf_agg["actual"] = {
    "mean_ret_pct": rt_e["exit_return_pct"].mean(),
    "median_ret_pct": rt_e["exit_return_pct"].median(),
    "win_rate": (rt_e["exit_return_pct"] > 0).mean(),
    "sum_ret_pct": rt_e["exit_return_pct"].sum(),
}
print(pd.DataFrame(cf_agg).T.round(3).to_string())

print("""
Note: counterfactuals past his actual exit fall back to his actual exit price,
so very-long horizons converge to actual. Key comparison is early horizons.
""")

# === 4. MFE utilization by trade outcome ===
hdr("4. MFE utilization by trade outcome")
rt_e["outcome"] = np.where(rt_e["net_pnl_btc"] > 0, "win", "loss")
# for wins: what fraction of MFE did he capture?
# for losses: did MFE ever go positive (i.e., was it a winner that reversed)?
win_stats = rt_e[rt_e["outcome"] == "win"].agg(
    n=("exit_return_pct", "size"),
    median_mfe=("mfe_pct", "median"),
    median_capture=("exit_efficiency", "median"),
)
# "trade went profitable at some point but closed at loss"
missed = rt_e[(rt_e["outcome"] == "loss") & (rt_e["mfe_pct"] > 1)]
print(f"losing trades that were once +1% or better: {len(missed)} / {(rt_e['outcome']=='loss').sum()} "
      f"({len(missed)/(rt_e['outcome']=='loss').sum()*100:.0f}%)")
print(f"  their median MFE: {missed['mfe_pct'].median():.2f}%")
print(f"  their median final loss: {missed['exit_return_pct'].median():.2f}%")
print(f"  → significant give-back on losers that briefly showed profit")

# === 5. MAE tolerance — how deep does he let drawdown go before cutting? ===
hdr("5. MAE tolerance — his actual pain threshold")
mae_buckets = pd.cut(rt_e["mae_pct"],
                     bins=[-30, -10, -5, -3, -2, -1, 0, 10],
                     labels=["<-10%", "-10 to -5%", "-5 to -3%", "-3 to -2%", "-2 to -1%", "-1 to 0%", "≥0%"])
mae_stats = rt_e.groupby(mae_buckets, observed=True).agg(
    n=("net_pnl_btc", "size"),
    win_rate=("net_pnl_btc", lambda s: (s > 0).mean()),
    median_ret=("exit_return_pct", "median"),
    total_btc=("net_pnl_btc", "sum"),
)
print(mae_stats.round(3).to_string())

# === 6. Exit-time K-line state ===
hdr("6. Exit-time price action state")
# simple exit features using 1h data at exit time
def exit_kline_state(t_exit):
    idx = np.searchsorted(h1_times, to_naive(t_exit), side="right") - 1
    if idx < 24:
        return {}
    window = h1.iloc[idx - 23: idx + 1]
    close_exit = window.iloc[-1]["close"]
    high24 = window["high"].max()
    low24 = window["low"].min()
    close_24h_ago = window.iloc[0]["close"]
    ret_24h = (close_exit - close_24h_ago) / close_24h_ago
    pos_in_24h = (close_exit - low24) / (high24 - low24) if high24 > low24 else 0.5
    return {"exit_ret_24h": ret_24h, "exit_pos_in_24h": pos_in_24h}

exit_feats = pd.DataFrame([exit_kline_state(t) for t in rt_e["exit_time"]])
rt_e = pd.concat([rt_e, exit_feats], axis=1)

# for profitable long exits: is he exiting near 24h high? (selling strength)
wins_long = rt_e[(rt_e["direction"] == "long") & (rt_e["outcome"] == "win")]
wins_short = rt_e[(rt_e["direction"] == "short") & (rt_e["outcome"] == "win")]
print(f"winning LONG exits: median 24h-range pos at exit = {wins_long['exit_pos_in_24h'].median():.2f}")
print(f"  → if >0.5, he sells into strength; if <0.5, sells on weakness")
print(f"winning SHORT exits: median 24h-range pos at exit = {wins_short['exit_pos_in_24h'].median():.2f}")
print(f"  → if <0.5, he covers into weakness; if >0.5, covers on a bounce")

rt_e.to_csv(OUT / "roundtrips_with_exits.csv", index=False)
print(f"\nsaved -> {OUT / 'roundtrips_with_exits.csv'}")
