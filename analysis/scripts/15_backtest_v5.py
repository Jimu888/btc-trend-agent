"""v5 — back to basics. MA200 trend-following.

Hypothesis: the SIMPLEST trend filter is the best for BTC. Complicated rules
keep getting chopped.

Core rule:
  LONG when close > MA200d. FLAT when close < MA200d. That's it.

Test variants:
  v5a: pure MA200 regime
  v5b: v5a + Paul's Pyramid-only + ATR profit protection
  v5c: v5a + MA50 trend confirmation (close > MA50 AND MA50 > MA200)

Benchmark: BTC buy-and-hold.
Goal: match or exceed BTC return with HALF the drawdown.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

d1 = pd.read_csv(KL / "1d.csv", parse_dates=["open_time", "close_time"])
c = d1["close"]
d1["ma200"] = c.rolling(200).mean()
d1["ma50"]  = c.rolling(50).mean()
d1["ma100"] = c.rolling(100).mean()
tr = pd.concat([d1["high"]-d1["low"], (d1["high"]-c.shift(1)).abs(),
                (d1["low"]-c.shift(1)).abs()], axis=1).max(axis=1)
d1["atr14"] = tr.rolling(14).mean()
d1["atr_pct"] = d1["atr14"] / c

def run(strategy_name, entry_fn, exit_fn, use_pyramid=False, use_atr_trail=False):
    """Generic backtester.

    entry_fn(row_today, row_prev) -> bool: open if True and no pos
    exit_fn(row_today, pos) -> (exit_price, reason) or (None, None)
    """
    pos = None
    trades = []
    equity_pts = [(d1.iloc[200]["close_time"], 1.0)]
    eq = 1.0

    for i in range(200, len(d1)):
        b = d1.iloc[i]
        prev = d1.iloc[i-1]
        if any(pd.isna([b["ma200"], b["atr14"]])): continue

        if pos is not None:
            # mark-to-market update
            fav = (b["high"] - pos["avg_entry"]) / pos["avg_entry"]
            adv = (b["low"]  - pos["avg_entry"]) / pos["avg_entry"]
            pos["mfe"] = max(pos["mfe"], fav)
            pos["mae"] = min(pos["mae"], adv)

            exit_price, reason = exit_fn(b, pos)

            if exit_price is None and use_atr_trail:
                # Paul-style ATR trail: breakeven at +2 ATR, trail at -3 ATR from peak after that
                atr_frac = pos["atr_entry"] / pos["avg_entry"]
                if pos["mfe"] >= 2 * atr_frac and pos["stop_pct"] < 0:
                    pos["stop_pct"] = 0.0
                if pos["mfe"] >= 4 * atr_frac:
                    trail = pos["mfe"] - 3 * atr_frac
                    if trail > pos["stop_pct"]:
                        pos["stop_pct"] = trail
                stop_price = pos["avg_entry"] * (1 + pos["stop_pct"])
                if b["low"] <= stop_price:
                    exit_price, reason = stop_price, "atr_stop"

            if exit_price is None and use_pyramid:
                # pyramid at +0.5 ATR, +1.0 ATR, +2.0 ATR
                atr_frac = pos["atr_entry"] / pos["avg_entry"]
                cur = (b["close"] - pos["avg_entry"]) / pos["avg_entry"]
                for key, thresh in [("add1", 0.5), ("add2", 1.0), ("add3", 2.0)]:
                    if not pos.get(key, False) and cur >= thresh * atr_frac and pos["size"] < 0.95:
                        add = 0.22
                        new_avg = (pos["avg_entry"] * pos["size"] + b["close"] * add) / (pos["size"] + add)
                        pos["avg_entry"] = new_avg
                        pos["size"] += add
                        pos[key] = True
                        break

            if exit_price is not None:
                ret = (exit_price - pos["avg_entry"]) / pos["avg_entry"] * pos["size"]
                eq *= (1 + ret)
                trades.append({
                    "entry_time": pos["entry_time"], "exit_time": b["close_time"],
                    "avg_entry": pos["avg_entry"], "exit_price": exit_price,
                    "size": pos["size"], "ret": ret,
                    "mfe": pos["mfe"], "mae": pos["mae"],
                    "reason": reason,
                    "days": (b["close_time"] - pos["entry_time"]).days,
                })
                pos = None

            # track equity daily (mark-to-market unrealized if still open)
            if pos is not None:
                unreal = (b["close"] - pos["avg_entry"]) / pos["avg_entry"] * pos["size"]
                equity_pts.append((b["close_time"], eq * (1 + unreal)))
            else:
                equity_pts.append((b["close_time"], eq))
        else:
            equity_pts.append((b["close_time"], eq))

        if pos is not None: continue
        if entry_fn(b, prev):
            pos = {
                "entry_time": b["close_time"],
                "avg_entry": b["close"],
                "atr_entry": b["atr14"],
                "size": 0.34 if use_pyramid else 1.0,
                "mfe": 0.0, "mae": 0.0,
                "stop_pct": -3.0 * b["atr14"] / b["close"] if use_atr_trail else -0.50,
            }

    # close any open at end
    if pos is not None:
        last = d1.iloc[-1]
        ret = (last["close"] - pos["avg_entry"]) / pos["avg_entry"] * pos["size"]
        eq *= (1 + ret)
        trades.append({
            "entry_time": pos["entry_time"], "exit_time": last["close_time"],
            "avg_entry": pos["avg_entry"], "exit_price": last["close"],
            "size": pos["size"], "ret": ret,
            "mfe": pos["mfe"], "mae": pos["mae"], "reason": "end_of_data",
            "days": (last["close_time"] - pos["entry_time"]).days,
        })

    tr = pd.DataFrame(trades)
    eq_df = pd.DataFrame(equity_pts, columns=["t", "eq"])
    eq_df["peak"] = eq_df["eq"].cummax()
    eq_df["dd"] = (eq_df["eq"] - eq_df["peak"]) / eq_df["peak"]
    max_dd = eq_df["dd"].min()
    final_eq = eq_df["eq"].iloc[-1]

    print(f"\n=== {strategy_name} ===")
    print(f"trades: {len(tr)}")
    if len(tr) > 0:
        print(f"win rate: {(tr['ret']>0).mean()*100:.1f}%")
        print(f"avg hold: {tr['days'].mean():.0f} days  (max {tr['days'].max()})")
    print(f"final equity (1 BTC start): {final_eq:.3f} BTC ({(final_eq-1)*100:+.1f}%)")
    print(f"max drawdown: {max_dd*100:.1f}%")
    print(f"return/|dd|: {(final_eq-1)/abs(max_dd+1e-9):.2f}")
    return tr, eq_df, final_eq, max_dd

# ============ STRATEGIES ============

# v5a: pure MA200 regime
def entry_ma200(b, prev):
    return b["close"] > b["ma200"] and prev["close"] <= prev["ma200"]   # cross up
def exit_ma200(b, pos):
    if b["close"] < b["ma200"]:
        return b["close"], "ma200_below"
    return None, None
run("v5a: pure MA200 regime", entry_ma200, exit_ma200)

# v5b: MA200 + Paul's pyramid + ATR trail
run("v5b: MA200 + pyramid + ATR trail (Paul overlay)",
    entry_ma200, exit_ma200, use_pyramid=True, use_atr_trail=True)

# v5c: tighter filter — close > MA50 > MA200 AND MA50 rising
def entry_strict(b, prev):
    # three conditions must all be true AND just became true
    now = (b["close"] > b["ma50"] and b["ma50"] > b["ma200"])
    prev_state = (prev["close"] > prev["ma50"] and prev["ma50"] > prev["ma200"])
    return now and not prev_state
def exit_strict(b, pos):
    if b["close"] < b["ma50"]:
        return b["close"], "ma50_below"
    return None, None
run("v5c: MA50 > MA200 alignment", entry_strict, exit_strict)

# v5d: strict + Paul overlay
run("v5d: MA50>MA200 + pyramid + ATR trail",
    entry_strict, exit_strict, use_pyramid=True, use_atr_trail=True)

# v5e: even simpler — always long if MA200 slope up (no cross trigger, just regime)
def entry_regime(b, prev):
    # enter ANY time we're above MA200 and price is above MA50 on upward trend
    if b["close"] > b["ma200"] and b["close"] > b["ma50"]:
        # slope check: is MA50 rising?
        # approximate by comparing to 10 days ago
        idx = d1[d1["close_time"] == b["close_time"]].index[0]
        if idx < 60: return False
        ma50_prev = d1.iloc[idx-10]["ma50"]
        if pd.isna(ma50_prev): return False
        return b["ma50"] > ma50_prev
    return False
def exit_regime(b, pos):
    if b["close"] < b["ma200"]:
        return b["close"], "regime_off"
    return None, None
run("v5e: simple regime + MA50 slope up", entry_regime, exit_regime)

# benchmark
start = d1.iloc[200]["close"]
end   = d1.iloc[-1]["close"]
bh_eq = d1["close"].iloc[200:] / start
bh_peak = bh_eq.cummax()
bh_dd = ((bh_eq - bh_peak) / bh_peak).min()
print(f"\n=== BENCHMARK: BTC BUY-AND-HOLD ===")
print(f"price {start:.0f} -> {end:.0f} = {(end/start-1)*100:+.0f}%")
print(f"max drawdown: {bh_dd*100:.1f}%")
print(f"return/|dd|: {(end/start-1)/abs(bh_dd+1e-9):.2f}")
