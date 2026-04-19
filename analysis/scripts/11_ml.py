"""Phase 4.3: ML-based non-linear feature importance.

Goals (not prediction for its own sake — we want INTERPRETATION):
  Task A: predict big-win vs not (top 25% of net_pnl_btc). Gradient boosting
          + permutation importance + SHAP-like summary.
  Task B: predict direction (long vs short) — what features flip his direction?
  Task C: at each entry time, can the model beat a simple rule in separating
          winners from losers? (internal validation)
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

DATA = Path("/Users/jimu/btc-trader-analysis/data") / "derived"
rt = pd.read_csv(DATA / "roundtrips_with_features.csv")
rt["entry_time"] = pd.to_datetime(rt["entry_time"], utc=True, format="ISO8601")
rt = rt.sort_values("entry_time").reset_index(drop=True)

FEATS = [
    "d_pct_vs_ma20", "d_pct_vs_ma50", "d_pct_vs_ma200", "d_above_ma200",
    "d_ma20_gt_ma50", "d_ret_1", "d_ret_3", "d_ret_7", "d_rsi14", "d_vol20",
    "d_atr14_pct", "d_pos_in_20d_range", "d_dist_20d_high", "d_dist_20d_low",
    "d_dist_50d_high", "d_dist_50d_low", "d_dist_ath",
    "h4_ret_1", "h4_ret_3", "h4_ret_7", "h4_rsi14", "h4_vol20", "h4_atr14_pct",
    "h4_pos_in_20bar_range", "h1_ret_1", "h1_ret_3", "h1_rsi14",
    "h1_body_pct", "h1_upper_wick_pct", "h1_lower_wick_pct",
    "d_ma50_slope_10d",
]

rt_c = rt.dropna(subset=FEATS).copy()
print(f"complete-feature trades: {len(rt_c)} / {len(rt)}")

def hdr(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# =============================================================
# TASK A: big-win classifier
# =============================================================
hdr("TASK A: predict big-win (top 25% by net_pnl_btc)")
q75 = rt_c["net_pnl_btc"].quantile(0.75)
rt_c["is_big_win"] = (rt_c["net_pnl_btc"] >= q75).astype(int)
X = rt_c[FEATS].values
y = rt_c["is_big_win"].values
print(f"class balance: big={y.sum()}, not={len(y)-y.sum()}")

# time-series CV (respects temporal order — no lookahead)
tscv = TimeSeriesSplit(n_splits=5)
aucs = []
for tr, te in tscv.split(X):
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=0.05, subsample=0.8,
                                   random_state=42)
    m.fit(X[tr], y[tr])
    aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
print(f"time-series CV AUC: mean={np.mean(aucs):.3f}, folds={[f'{a:.3f}' for a in aucs]}")

# train on full for interpretation
m = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                               learning_rate=0.05, subsample=0.8, random_state=42)
m.fit(X, y)

# permutation importance (more robust than tree .feature_importances_)
imp = permutation_importance(m, X, y, n_repeats=15, random_state=42, n_jobs=-1,
                             scoring="roc_auc")
imp_df = pd.DataFrame({
    "feature": FEATS,
    "imp_mean": imp.importances_mean,
    "imp_std": imp.importances_std,
}).sort_values("imp_mean", ascending=False)
print("\nTop 15 features for 'big-win' (permutation importance on AUC):")
print(imp_df.head(15).round(4).to_string(index=False))

# partial dependence snippets for top features
from sklearn.inspection import partial_dependence
print("\nPartial dependence curves (top 6 features):")
top_feats = imp_df.head(6)["feature"].tolist()
for feat in top_feats:
    idx = FEATS.index(feat)
    pd_result = partial_dependence(m, X, [idx], grid_resolution=10)
    grid = pd_result["grid_values"][0]
    vals = pd_result["average"][0]
    print(f"\n  {feat}:")
    for g, v in zip(grid, vals):
        print(f"    x={g:+.4f}  P(big_win)={v:.3f}")

# =============================================================
# TASK B: direction predictor
# =============================================================
hdr("TASK B: predict direction (long=1, short=0)")
y_dir = (rt_c["direction"] == "long").astype(int).values
m2 = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                learning_rate=0.05, subsample=0.8, random_state=42)
aucs2 = []
for tr, te in tscv.split(X):
    m2_t = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    m2_t.fit(X[tr], y_dir[tr])
    aucs2.append(roc_auc_score(y_dir[te], m2_t.predict_proba(X[te])[:, 1]))
print(f"direction CV AUC: mean={np.mean(aucs2):.3f}, folds={[f'{a:.3f}' for a in aucs2]}")
m2.fit(X, y_dir)
imp2 = permutation_importance(m2, X, y_dir, n_repeats=15, random_state=42, n_jobs=-1,
                              scoring="roc_auc")
imp2_df = pd.DataFrame({
    "feature": FEATS,
    "imp_mean": imp2.importances_mean,
}).sort_values("imp_mean", ascending=False)
print("\nTop 10 features for direction:")
print(imp2_df.head(10).round(4).to_string(index=False))

# =============================================================
# TASK C: win-vs-loss at entry
# =============================================================
hdr("TASK C: predict win (net_pnl_btc > 0) using only entry features")
y_win = (rt_c["net_pnl_btc"] > 0).astype(int).values
aucs3 = []
for tr, te in tscv.split(X):
    m3_t = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8, random_state=42)
    m3_t.fit(X[tr], y_win[tr])
    aucs3.append(roc_auc_score(y_win[te], m3_t.predict_proba(X[te])[:, 1]))
print(f"win-vs-loss CV AUC: mean={np.mean(aucs3):.3f}, folds={[f'{a:.3f}' for a in aucs3]}")

print(f"""
Interpretation:
  AUC 0.50 = random
  AUC 0.55-0.60 = weak but usable edge
  AUC 0.65+ = meaningful signal
If win-vs-loss AUC is near 0.5, his winners are ex-ante indistinguishable from his
losers at the moment of entry using ONLY K-line features — meaning his alpha
comes from execution (adding/trimming/exiting) rather than entry selection.
""")
