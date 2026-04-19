"""v4 — trend-following longs + Paul's best execution rules.

DESIGN PHILOSOPHY (incorporating all lessons from phases 1-4):
  - No shorts. BTC has upward drift. His short edge was 6-10 BTC vs 40+ long.
  - Trend filter: weekly close > 40-week SMA (classic BTC macro trend filter).
  - Entry on *confirmation*, not anticipation: daily close breaks out of 20d high
    within an existing trend (higher high).
  - Pyramid-only scale-in: 3 tranches at +0.5 ATR each (from his 68% win-rate style).
  - Stop: 3x daily ATR initial. Tightens to breakeven at +2 ATR. Trails at 3 ATR after.
  - Partial take-profit: 1/3 at +4 ATR, 1/3 at +8 ATR, trail the remaining 1/3.
  - Exit on trend break: weekly close < 30-week SMA (faster exit than entry trend filter).

Benchmark: BTC buy-and-hold. Agent target = match return, halve the drawdown.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]

# weekly SMA on daily data — use 280d (40 weeks) and 210d (30 weeks)
d1["ma280"] = c.rolling(280).mean()
d1["ma210"] = c.rolling(210).mean()
d1["ma50"]  = c.rolling(50).mean()
d1["high20"] = d1["high"].rolling(20).max()
d1["low20"]  = d1["low"].rolling(20).min()
d1["high55"] = d1["high"].rolling(55).max()

tr = pd.concat([d1["high"]-d1["low"],
                (d1["high"]-c.shift(1)).abs(),
                (d1["low"]-c.shift(1)).abs()], axis=1).max(axis=1)
d1["atr14"] = tr.rolling(14).mean()
d1["atr20"] = tr.rolling(20).mean()

START_IDX = 280
trades = []
equity = [1.0]
equity_dates = []

pos = None

for i in range(START_IDX, len(d1)):
    b = d1.iloc[i]
    prev = d1.iloc[i-1] if i > 0 else b

    if any(pd.isna([b["ma280"], b["ma210"], b["atr14"], b["high20"]])):
        continue

    in_bull_trend = b["close"] > b["ma280"]
    trend_broken = b["close"] < b["ma210"]

    # --- update open position ---
    if pos is not None:
        sign = 1
        fav = (b["high"] - pos["avg_entry"]) / pos["avg_entry"]
        adv = (b["low"]  - pos["avg_entry"]) / pos["avg_entry"]
        pos["mfe"] = max(pos["mfe"], fav)
        pos["mae"] = min(pos["mae"], adv)

        atr_frac = pos["atr_entry"] / pos["avg_entry"]
        stop_dist = 3.0 * atr_frac

        # upgrade stop
        if pos["mfe"] >= 2 * atr_frac and pos["stop_pct"] < 0:
            pos["stop_pct"] = 0.0   # breakeven
        if pos["mfe"] >= 4 * atr_frac:
            trail = pos["mfe"] - 3 * atr_frac
            if trail > pos["stop_pct"]:
                pos["stop_pct"] = trail

        stop_price = pos["avg_entry"] * (1 + pos["stop_pct"])
        exit_price = None
        exit_reason = None

        if b["low"] <= stop_price:
            exit_price = stop_price
            exit_reason = "stop"
        elif trend_broken:
            exit_price = b["close"]
            exit_reason = "trend_break"

        # partial take-profits (not exit, just reduce size)
        if exit_reason is None:
            if not pos["tp1"] and pos["mfe"] >= 4 * atr_frac:
                # 1/3 off at +4 ATR
                realized = (4 * atr_frac) * (pos["size_frac"] / 3)
                pos["realized"] += realized
                pos["size_frac"] *= 2/3
                pos["tp1"] = True
            if not pos["tp2"] and pos["mfe"] >= 8 * atr_frac:
                realized = (8 * atr_frac) * (pos["size_frac"] / 2)
                pos["realized"] += realized
                pos["size_frac"] *= 0.5
                pos["tp2"] = True

        # pyramid scale-in (adds at +0.5 ATR and +1.0 ATR profit)
        if exit_reason is None:
            cur_pnl = (b["close"] - pos["avg_entry"]) / pos["avg_entry"]
            if not pos["add1"] and cur_pnl >= 0.5 * atr_frac and pos["size_frac"] < 0.95:
                add_size = 0.33
                new_avg = (pos["avg_entry"] * pos["size_frac"] + b["close"] * add_size) / (pos["size_frac"] + add_size)
                pos["avg_entry"] = new_avg
                pos["size_frac"] += add_size
                pos["add1"] = True
            if not pos["add2"] and cur_pnl >= 1.0 * atr_frac and pos["size_frac"] < 1.28:
                add_size = 0.33
                new_avg = (pos["avg_entry"] * pos["size_frac"] + b["close"] * add_size) / (pos["size_frac"] + add_size)
                pos["avg_entry"] = new_avg
                pos["size_frac"] += add_size
                pos["add2"] = True

        if exit_reason is not None:
            final_ret = (exit_price - pos["avg_entry"]) / pos["avg_entry"] * pos["size_frac"]
            total_ret = pos["realized"] + final_ret
            trades.append({
                "entry_time": pos["entry_time"],
                "exit_time": b["close_time"],
                "avg_entry": pos["avg_entry"],
                "exit_price": exit_price,
                "size_final": pos["size_frac"],
                "mfe": pos["mfe"],
                "mae": pos["mae"],
                "total_ret": total_ret,
                "adds": int(pos["add1"]) + int(pos["add2"]),
                "tp1": pos["tp1"],
                "tp2": pos["tp2"],
                "exit_reason": exit_reason,
                "days": (b["close_time"] - pos["entry_time"]).days,
            })
            pos = None

    # track equity timeline (per-bar, for drawdown)
    if pos is None:
        equity.append(equity[-1])
    else:
        # mark-to-market: current unrealized based on today's close
        cur_unreal = (b["close"] - pos["avg_entry"]) / pos["avg_entry"] * pos["size_frac"]
        mtm = pos["realized"] + cur_unreal
        equity.append(equity[-2] * (1 + mtm - pos.get("last_mtm", 0.0)))
        pos["last_mtm"] = mtm
    equity_dates.append(b["close_time"])

    # --- entry logic ---
    if pos is not None:
        continue

    # trend filter
    if not in_bull_trend:
        continue
    # entry trigger: breakout of 20-day high on close
    if b["close"] > prev["high20"] and b["close"] > b["ma50"]:
        pos = {
            "entry_time": b["close_time"],
            "avg_entry": b["close"],
            "atr_entry": b["atr14"],
            "size_frac": 0.33,
            "mfe": 0.0, "mae": 0.0,
            "stop_pct": -3.0 * b["atr14"] / b["close"],
            "realized": 0.0,
            "add1": False, "add2": False,
            "tp1": False, "tp2": False,
            "last_mtm": 0.0,
        }

tr = pd.DataFrame(trades)
print(f"=== v4 TREND-FOLLOWING LONGS + EXECUTION ===")
print(f"trades: {len(tr)}")
print(f"wins: {(tr['total_ret']>0).sum()} ({(tr['total_ret']>0).mean()*100:.1f}%)")
print(f"mean return per trade: {tr['total_ret'].mean()*100:+.2f}%")
print(f"median return: {tr['total_ret'].median()*100:+.2f}%")

# compound properly using equity timeline
eq_series = pd.Series(equity[1:1+len(equity_dates)], index=pd.to_datetime(equity_dates))
# simpler: compound per-trade
compound = 1.0
for ret in tr["total_ret"]:
    compound *= (1 + ret)
print(f"\ncompounded 1 BTC -> {compound:.3f} BTC ({(compound-1)*100:+.1f}%)")

# drawdown via per-trade equity
tr_s = tr.sort_values("entry_time").copy()
tr_s["eq"] = (1 + tr_s["total_ret"]).cumprod()
peak = tr_s["eq"].cummax()
dd = (tr_s["eq"] - peak) / peak
print(f"max drawdown (trade-level): {dd.min()*100:.1f}%")

# benchmark
start_idx = d1.iloc[START_IDX]["close"]
end_idx = d1.iloc[-1]["close"]
bh = end_idx / start_idx - 1
print(f"\n=== BENCHMARK: BTC buy-and-hold ===")
print(f"price {start_idx:.0f} -> {end_idx:.0f} = {bh*100:+.0f}%")

# benchmark drawdown (roughly)
bh_eq = d1["close"].iloc[START_IDX:] / start_idx
bh_peak = bh_eq.cummax()
bh_dd = (bh_eq - bh_peak) / bh_peak
print(f"BTC max drawdown: {bh_dd.min()*100:.1f}%")

print(f"\n=== RISK-ADJUSTED ===")
print(f"v4 return / |drawdown|: {(compound-1)/abs(dd.min()+1e-9):.2f}")
print(f"BTC return / |drawdown|: {bh/abs(bh_dd.min()+1e-9):.2f}")

print(f"\n=== by year ===")
tr["year"] = pd.to_datetime(tr["entry_time"]).dt.year
print(tr.groupby("year").agg(
    n=("total_ret", "size"),
    wr=("total_ret", lambda s: (s > 0).mean()),
    sum_ret_pct=("total_ret", lambda s: s.sum() * 100),
    avg_days=("days", "mean"),
).round(2))

print(f"\n=== by exit reason ===")
print(tr.groupby("exit_reason").agg(
    n=("total_ret", "size"),
    wr=("total_ret", lambda s: (s > 0).mean()),
    avg_ret=("total_ret", "mean"),
    sum_ret=("total_ret", "sum"),
).round(4))

# distribution of winners/losers
if len(tr) > 0:
    wins = tr[tr["total_ret"] > 0]["total_ret"]
    losses = tr[tr["total_ret"] <= 0]["total_ret"]
    print(f"\navg winner: +{wins.mean()*100:.1f}%  (n={len(wins)})")
    print(f"avg loser:  {losses.mean()*100:.1f}%  (n={len(losses)})")
    if len(losses) > 0:
        print(f"profit factor: {wins.sum() / abs(losses.sum()):.2f}")
        print(f"win/loss ratio: {wins.mean() / abs(losses.mean()):.2f}")

tr.to_csv(DATA / "derived" / "backtest_v4_trades.csv", index=False)
print(f"\nsaved -> {DATA / 'derived' / 'backtest_v4_trades.csv'}")
