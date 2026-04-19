"""Thread C: hybrid agent prototype — v5e trend + rule-based swing.

ARCHITECTURE (no Gemma yet — baseline):

  Trend sub-account (70% of equity):
    Entry: close > MA200 AND close > MA50 AND MA50 rising (10d slope positive)
    Exit: close < MA200
    Full size when active.

  Swing sub-account (30% of equity):
    ONLY active when trend sub-account is flat.
    Rule-based longs AND shorts depending on regime.

  BULL_PULLBACK_LONG (close > MA200):
    Entry: RSI14 < 40 AND close < MA20 (oversold pullback in bull)
    Exit: RSI > 60 OR close crosses MA20 from below
    Stop: -1.5x ATR14
    Pyramid: +0.5 ATR → add 50%
    Max hold: 14 days

  BEAR_RALLY_SHORT (close < MA200):
    Entry: RSI14 > 60 AND close > MA20
    Exit: RSI < 40 OR close crosses MA20 from above
    Stop: +1.5x ATR14
    Max hold: 14 days

Costs: 0.1% fee per side (realistic taker). Slippage: 0.05% on market exits.

Two sub-accounts run independently on ~1 BTC total equity (0.7 + 0.3).
At report time we combine the two curves.
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
KL = DATA / "klines"

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

FEE = 0.001
SLIP = 0.0005
START_IDX = 210
END_IDX = len(d1) - 1

# ---------- TREND ENGINE ----------
def run_trend(equity0=0.70):
    eq = equity0
    pos = None
    trades = []
    eq_ts = []   # (time, equity)
    for i in range(START_IDX, END_IDX + 1):
        b = d1.iloc[i]
        if any(pd.isna([b["ma200"], b["ma50"], b["ma50_prev10"]])):
            eq_ts.append((b["close_time"], eq)); continue

        if pos is not None:
            if b["close"] < b["ma200"]:
                exit_p = b["close"] * (1 - SLIP)   # slippage on exit
                ret = (exit_p - pos["entry"]) / pos["entry"]
                net_ret = ret - 2 * FEE             # entry + exit fee
                eq *= (1 + net_ret)
                trades.append({"engine":"trend","entry_time":pos["entry_time"],"exit_time":b["close_time"],
                               "entry":pos["entry"],"exit":exit_p,"ret":net_ret,
                               "days":(b["close_time"]-pos["entry_time"]).days,"reason":"ma200_break"})
                pos = None

        # mark-to-market equity
        if pos is not None:
            unreal = (b["close"] - pos["entry"]) / pos["entry"] - 2 * FEE
            eq_ts.append((b["close_time"], equity0 * (1 + unreal) if eq == equity0 else eq * (1 + unreal - pos.get("last_unreal",0))))
        else:
            eq_ts.append((b["close_time"], eq))

        # entry
        if pos is None:
            if b["close"] > b["ma200"] and b["close"] > b["ma50"] and b["ma50"] > b["ma50_prev10"]:
                pos = {"entry": b["close"] * (1 + SLIP), "entry_time": b["close_time"]}

    # close open
    if pos is not None:
        last = d1.iloc[END_IDX]
        exit_p = last["close"] * (1 - SLIP)
        ret = (exit_p - pos["entry"]) / pos["entry"]
        net_ret = ret - 2 * FEE
        eq *= (1 + net_ret)
        trades.append({"engine":"trend","entry_time":pos["entry_time"],"exit_time":last["close_time"],
                       "entry":pos["entry"],"exit":exit_p,"ret":net_ret,
                       "days":(last["close_time"]-pos["entry_time"]).days,"reason":"end"})
    return pd.DataFrame(trades), eq, eq_ts

# ---------- SWING ENGINE ----------
def run_swing(equity0=0.30, trend_trades=None):
    # trend_trades tells us when trend engine is IN position — swing is blocked then
    in_trend = np.zeros(len(d1), dtype=bool)
    if trend_trades is not None:
        for _, t in trend_trades.iterrows():
            start = d1["close_time"].searchsorted(t["entry_time"], side="left")
            end = d1["close_time"].searchsorted(t["exit_time"], side="right")
            in_trend[start:end] = True

    eq = equity0
    pos = None
    trades = []
    eq_ts = []
    for i in range(START_IDX, END_IDX + 1):
        b = d1.iloc[i]
        if any(pd.isna([b["ma200"], b["ma20"], b["atr14"], b["rsi14"]])):
            eq_ts.append((b["close_time"], eq)); continue

        # --- manage open swing pos ---
        if pos is not None:
            sign = 1 if pos["direction"] == "long" else -1
            atr_frac = pos["atr_entry"] / pos["avg_entry"]
            # MFE
            fav_p = b["high"] if sign == 1 else b["low"]
            adv_p = b["low"]  if sign == 1 else b["high"]
            fav = sign * (fav_p - pos["avg_entry"]) / pos["avg_entry"]
            adv = sign * (adv_p - pos["avg_entry"]) / pos["avg_entry"]
            pos["mfe"] = max(pos["mfe"], fav)
            pos["mae"] = min(pos["mae"], adv)

            # breakeven at +1 ATR, trail at -1 ATR after +2 ATR
            if pos["mfe"] >= 1.0 * atr_frac and pos["stop_pct"] < 0:
                pos["stop_pct"] = 0.0
            if pos["mfe"] >= 2.0 * atr_frac:
                trail = pos["mfe"] - 1.0 * atr_frac
                if trail > pos["stop_pct"]: pos["stop_pct"] = trail

            stop_price = pos["avg_entry"] * (1 + sign * pos["stop_pct"])
            exit_price, reason = None, None
            if sign == 1 and b["low"] <= stop_price:
                exit_price, reason = stop_price, "atr_stop"
            elif sign == -1 and b["high"] >= stop_price:
                exit_price, reason = stop_price, "atr_stop"

            # mean-reversion exit: RSI crosses opposite extreme
            if exit_price is None:
                if sign == 1 and b["rsi14"] > 60:
                    exit_price, reason = b["close"] * (1 - SLIP), "rsi_flip"
                elif sign == -1 and b["rsi14"] < 40:
                    exit_price, reason = b["close"] * (1 + SLIP), "rsi_flip"

            # time stop
            days_in = (b["close_time"] - pos["entry_time"]).days
            if exit_price is None and days_in >= 14:
                exit_price = b["close"] * (1 - SLIP * sign)
                reason = "time"

            # pyramid scale-in (+0.5 ATR while not in loss)
            if exit_price is None and not pos["added"] and pos["size"] < 0.95:
                cur = sign * (b["close"] - pos["avg_entry"]) / pos["avg_entry"]
                if cur >= 0.5 * atr_frac:
                    add = 0.5
                    new_avg = (pos["avg_entry"] * pos["size"] + b["close"] * add) / (pos["size"] + add)
                    pos["avg_entry"] = new_avg
                    pos["size"] += add
                    pos["added"] = True

            if exit_price is not None:
                ret = sign * (exit_price - pos["avg_entry"]) / pos["avg_entry"] * pos["size"]
                net_ret = ret - 2 * FEE * pos["size"]
                eq *= (1 + net_ret)
                trades.append({"engine":"swing","entry_time":pos["entry_time"],"exit_time":b["close_time"],
                               "direction":pos["direction"],"entry":pos["avg_entry"],"exit":exit_price,
                               "size":pos["size"],"ret":net_ret,"mfe":pos["mfe"],"mae":pos["mae"],
                               "setup":pos["setup"],"reason":reason,
                               "days":(b["close_time"]-pos["entry_time"]).days})
                pos = None

        # --- entry candidates (only when trend flat) ---
        if pos is None and not in_trend[i]:
            bull = b["close"] > b["ma200"]
            # BULL_PULLBACK_LONG
            if bull and b["rsi14"] < 40 and b["close"] < b["ma20"]:
                entry_p = b["close"] * (1 + SLIP)
                pos = {"direction":"long","setup":"bull_pullback_long",
                       "entry_time":b["close_time"],"avg_entry":entry_p,
                       "atr_entry":b["atr14"],"size":0.5,
                       "mfe":0.0,"mae":0.0,"stop_pct":-1.5 * b["atr14"]/entry_p,
                       "added":False}
            # BEAR_RALLY_SHORT
            elif not bull and b["rsi14"] > 60 and b["close"] > b["ma20"]:
                entry_p = b["close"] * (1 - SLIP)
                pos = {"direction":"short","setup":"bear_rally_short",
                       "entry_time":b["close_time"],"avg_entry":entry_p,
                       "atr_entry":b["atr14"],"size":0.5,
                       "mfe":0.0,"mae":0.0,"stop_pct":-1.5 * b["atr14"]/entry_p,
                       "added":False}

        # track eq (MTM)
        if pos is not None:
            sign = 1 if pos["direction"] == "long" else -1
            unreal_ret = sign * (b["close"] - pos["avg_entry"]) / pos["avg_entry"] * pos["size"]
            eq_ts.append((b["close_time"], eq * (1 + unreal_ret)))
        else:
            eq_ts.append((b["close_time"], eq))

    return pd.DataFrame(trades), eq, eq_ts

# ---------- RUN ----------
print("=" * 70)
print("HYBRID AGENT BACKTEST (with realistic costs)")
print("=" * 70)

trend_tr, trend_eq, trend_ts = run_trend()
print(f"\nTREND ENGINE: {len(trend_tr)} trades, final sub-account = {trend_eq:.4f} (start 0.70)")
if len(trend_tr) > 0:
    print(f"  win rate: {(trend_tr['ret']>0).mean()*100:.0f}%  "
          f"avg ret: {trend_tr['ret'].mean()*100:+.1f}%  "
          f"sum ret: {(trend_eq/0.70-1)*100:+.0f}%")

swing_tr, swing_eq, swing_ts = run_swing(trend_trades=trend_tr)
print(f"\nSWING ENGINE: {len(swing_tr)} trades, final sub-account = {swing_eq:.4f} (start 0.30)")
if len(swing_tr) > 0:
    print(f"  win rate: {(swing_tr['ret']>0).mean()*100:.0f}%  "
          f"avg ret: {swing_tr['ret'].mean()*100:+.2f}%  "
          f"sum ret: {(swing_eq/0.30-1)*100:+.0f}%")
    print(f"\n  by setup:")
    print(swing_tr.groupby("setup").agg(
        n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
        avg=("ret","mean"), total=("ret","sum"),
    ).round(4).to_string())
    print(f"\n  by exit reason:")
    print(swing_tr.groupby("reason").agg(
        n=("ret","size"), wr=("ret", lambda s:(s>0).mean()),
        avg=("ret","mean"),
    ).round(4).to_string())

# Combine equity curves
trend_s = pd.DataFrame(trend_ts, columns=["t", "eq"]).drop_duplicates(subset="t")
swing_s = pd.DataFrame(swing_ts, columns=["t", "eq"]).drop_duplicates(subset="t")
trend_s = trend_s.set_index("t")["eq"]
swing_s = swing_s.set_index("t")["eq"]
total_eq = (trend_s + swing_s).ffill()

final = total_eq.iloc[-1]
peak = total_eq.cummax()
dd = ((total_eq - peak) / peak).min()

print(f"\n" + "="*70)
print("COMBINED AGENT")
print("="*70)
print(f"start equity: 1.00  ->  final: {final:.4f}  ({(final-1)*100:+.1f}%)")
print(f"max drawdown: {dd*100:.1f}%")
print(f"return/|dd|: {(final-1)/abs(dd+1e-9):.2f}")

# benchmarks
start_p = d1.iloc[START_IDX]["close"]
end_p = d1.iloc[END_IDX]["close"]
bh = end_p / start_p - 1
bh_eq = d1["close"].iloc[START_IDX:] / start_p
bh_dd = ((bh_eq - bh_eq.cummax()) / bh_eq.cummax()).min()
print(f"\nBenchmark BTC buy-and-hold: {bh*100:+.0f}%  max_dd={bh_dd*100:.1f}%  ratio={bh/abs(bh_dd):.2f}")

print(f"\nPaul XBTUSD (apples-to-apples, rough): ~41 BTC on ~90 BTC avg equity = +45%")
print(f"  vs agent: {(final-1)*100:+.1f}%")

# save outputs
all_trades = pd.concat([trend_tr, swing_tr], ignore_index=True, sort=False)
all_trades.to_csv(DATA / "derived" / "hybrid_agent_trades.csv", index=False)
print(f"\nsaved trade log -> {DATA / 'derived' / 'hybrid_agent_trades.csv'}")
