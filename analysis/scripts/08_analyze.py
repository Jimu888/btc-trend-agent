"""Phase 3.3: discover his setups & extract agent rules.

Three comparisons:
  A. His entries (all 687) vs. baseline (random 4h bars): what conditions does
     he respond to that the market is *not* mostly in?
  B. Long entries vs short entries: what flips his direction?
  C. Winning entries vs losing entries: what ex-ante signal separates them?
  D. Big-win entries (top-quartile net_pnl_btc) vs normal: what marks his best setups?

Then we cluster his entries into latent "setups" (KMeans on key features)
and characterize each cluster's performance.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

DATA = Path("/Users/jimu/btc-trader-analysis/data") / "derived"
rt = pd.read_csv(DATA / "roundtrips_with_features.csv", parse_dates=["entry_time", "exit_time"])
base = pd.read_csv(DATA / "baseline_features.csv")

# numeric feature list
FEATS = [c for c in base.columns if c != "price"]

def hdr(t):
    print(f"\n{'='*70}\n{t}\n{'='*70}")

def compare(a, b, feats, name_a, name_b):
    """Mean-of-a vs mean-of-b with Mann-Whitney U for non-normality.
    Returns a dataframe sorted by absolute std-units effect size."""
    rows = []
    for f in feats:
        av = a[f].dropna()
        bv = b[f].dropna()
        if len(av) < 10 or len(bv) < 10:
            continue
        ma, mb = av.mean(), bv.mean()
        sd = np.sqrt((av.std() ** 2 + bv.std() ** 2) / 2)
        if sd == 0 or np.isnan(sd):
            continue
        d = (ma - mb) / sd   # Cohen's d-ish
        try:
            u, p = stats.mannwhitneyu(av, bv, alternative="two-sided")
        except Exception:
            p = np.nan
        rows.append({
            "feature": f,
            f"{name_a}_median": av.median(),
            f"{name_b}_median": bv.median(),
            "effect_d": d,
            "p_mw": p,
        })
    return pd.DataFrame(rows).sort_values("effect_d", key=abs, ascending=False)

hdr("A. ALL HIS ENTRIES vs BASELINE (random 4h bars)")
df = compare(rt, base, FEATS, "entry", "base")
print(df.head(15).round(4).to_string(index=False))
print("\n...bottom half (smallest differences):")
print(df.tail(8).round(4).to_string(index=False))

hdr("B. LONG vs SHORT entries")
longs = rt[rt["direction"] == "long"]
shorts = rt[rt["direction"] == "short"]
df = compare(longs, shorts, FEATS, "long", "short")
print(df.head(15).round(4).to_string(index=False))

hdr("C. WIN vs LOSS entries")
win = rt[rt["net_pnl_btc"] > 0]
loss = rt[rt["net_pnl_btc"] <= 0]
df = compare(win, loss, FEATS, "win", "loss")
print(df.head(15).round(4).to_string(index=False))

hdr("D. BIG-WIN entries (top 25% by net_pnl_btc) vs the rest")
q75 = rt["net_pnl_btc"].quantile(0.75)
big = rt[rt["net_pnl_btc"] >= q75]
rest = rt[rt["net_pnl_btc"] < q75]
print(f"threshold: net_pnl_btc >= {q75:.3f} BTC, n_big={len(big)}")
df = compare(big, rest, FEATS, "big", "rest")
print(df.head(15).round(4).to_string(index=False))

hdr("E. 2024+ entries (his evolved style) — feature profile")
modern = rt[rt["year"] >= 2024]
print(f"modern trades: {len(modern)}")
print("\nmodern entry features (medians):")
for f in ["d_pct_vs_ma200", "d_above_ma200", "d_ret_7", "d_rsi14",
          "d_pos_in_20d_range", "d_dist_20d_high", "d_dist_ath",
          "h4_ret_3", "h4_rsi14", "d_atr14_pct"]:
    if f in modern.columns:
        m_val = modern[f].median()
        a_val = rt[f].median()
        print(f"  {f:25s}  modern={m_val:+.4f}   all={a_val:+.4f}")
print(f"\nmodern long share: {(modern['direction'] == 'long').mean():.0%}")
print(f"modern median hold: {modern['holding_hours'].median():.1f}h")
print(f"modern median net PnL: {modern['net_pnl_btc'].median():+.3f} BTC")

hdr("F. CLUSTER his entries into setups (KMeans k=5)")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

CLUSTER_FEATS = [
    "d_pct_vs_ma200", "d_ma50_slope_10d", "d_ret_7", "d_rsi14",
    "d_pos_in_20d_range", "d_dist_ath", "d_atr14_pct",
    "h4_ret_3", "h4_rsi14",
]
X = rt[CLUSTER_FEATS].dropna()
rt_c = rt.loc[X.index].copy()
Xs = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs)
rt_c["cluster"] = km.labels_

print("cluster summary (median features + performance):")
agg = rt_c.groupby("cluster").agg(
    n=("cluster", "size"),
    long_share=("direction", lambda s: (s == "long").mean()),
    win_rate=("net_pnl_btc", lambda s: (s > 0).mean()),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
    med_hold_h=("holding_hours", "median"),
    med_notional=("notional_btc", "median"),
    med_d_vs_ma200=("d_pct_vs_ma200", "median"),
    med_d_ret_7=("d_ret_7", "median"),
    med_d_rsi14=("d_rsi14", "median"),
    med_d_pos_20d=("d_pos_in_20d_range", "median"),
    med_d_dist_ath=("d_dist_ath", "median"),
    med_h4_ret_3=("h4_ret_3", "median"),
)
print(agg.round(4).to_string())

# save labelled dataset for further inspection
rt_c.to_csv(DATA / "roundtrips_clustered.csv", index=False)

hdr("G. REGIME-CONDITIONED win rate")
# split by d_above_ma200 & d_ma50_slope
rt_r = rt.dropna(subset=["d_above_ma200", "d_ma50_slope_10d"]).copy()
rt_r["regime"] = np.where(
    (rt_r["d_above_ma200"] == 1) & (rt_r["d_ma50_slope_10d"] > 0), "bull_trend",
    np.where(
        (rt_r["d_above_ma200"] == 0) & (rt_r["d_ma50_slope_10d"] < 0), "bear_trend",
        "mixed",
    ),
)
reg = rt_r.groupby(["regime", "direction"]).agg(
    n=("net_pnl_btc", "size"),
    win_rate=("net_pnl_btc", lambda s: (s > 0).mean()),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
)
print(reg.round(3))
