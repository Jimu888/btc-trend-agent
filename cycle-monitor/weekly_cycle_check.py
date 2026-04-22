"""Weekly BTC cycle check — 5-axis scoring using free public APIs.

Runs once per week. Outputs:
  - Individual axis scores (-2 to +2)
  - Weighted total (-10 to +10)
  - Mapped cycle stage
  - Recommended agent-position sizing

Data sources (all free):
  - Axis 1: halving cycle math (local)
  - Axis 2: blockchain.info + mempool.space (partial — MVRV/SOPR need paid)
  - Axis 3: Binance public API (funding, OI, basis)
  - Axis 4: FRED (Fed rate, M2, CPI, DXY, yields)
  - Axis 5: alternative.me (F&G), CoinGecko (dominance)

No auth except FRED_API_KEY in .env.
"""
import os
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
FRED_KEY = os.environ.get("FRED_API_KEY")

CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)

# ---------- helpers ----------
def http_get(url, params=None, headers=None, timeout=30):
    r = requests.get(url, params=params, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()

def clamp(x, lo=-2, hi=2):
    return max(lo, min(hi, x))

# ==================== AXIS 1: HALVING CYCLE CLOCK ====================
def axis1_cycle_clock():
    halvings = [
        ("2012-11-28", 12),
        ("2016-07-09", 660),
        ("2020-05-11", 8600),
        ("2024-04-20", 64000),
    ]
    now = datetime.now(timezone.utc).date()
    last = datetime.strptime(halvings[-1][0], "%Y-%m-%d").date()
    months_since = (now - last).days / 30.44
    # historically tops are at month 12-18 post-halving
    # score: +2 if we're in months 6-15 (bull window)
    #         0 if months 18-24 (typical top/early bear)
    #        -2 if months 24-36 (deep bear)
    #        +1 if months 30-42 (accumulation)
    if 0 <= months_since < 6:       score, label = 1, "bull 初期"
    elif 6 <= months_since < 15:    score, label = 2, "bull 主升"
    elif 15 <= months_since < 21:   score, label = 1, "bull 末段/顶部"
    elif 21 <= months_since < 30:   score, label = -2, "熊市"
    elif 30 <= months_since < 42:   score, label = 1, "积累/早期牛"
    else:                           score, label = 0, "过渡期"
    return {
        "score": score,
        "detail": {
            "months_since_halving": round(months_since, 1),
            "label": label,
            "last_halving": halvings[-1][0],
            "next_halving_est": "2028-04",
        },
    }

# ==================== AXIS 2: ON-CHAIN (LIMITED without Glassnode) ====================
def axis2_onchain():
    out = {"note": "partial coverage — MVRV/SOPR require paid Glassnode"}
    try:
        # hashrate trend from blockchain.info (network health)
        # mempool.space also provides difficulty + hashrate
        m = http_get("https://mempool.space/api/v1/mining/hashrate/3y")
        # m["currentDifficulty"], m["currentHashrate"]
        recent = m["hashrates"][-30:]  # last 30 data points
        now_hr = recent[-1]["avgHashrate"]
        past_hr = recent[0]["avgHashrate"]
        hr_change = (now_hr - past_hr) / past_hr * 100
        out["hashrate_30pt_change_pct"] = round(hr_change, 2)

        # active addresses — blockchain.info
        addr = http_get("https://api.blockchain.info/charts/n-unique-addresses",
                       params={"timespan": "90days", "format": "json", "sampled": "false"})
        vals = [v["y"] for v in addr["values"][-30:]]
        aa_recent, aa_past = sum(vals[-7:])/7, sum(vals[:7])/7
        aa_change = (aa_recent - aa_past) / aa_past * 100
        out["active_addresses_30d_change_pct"] = round(aa_change, 2)

        # price vs 200-week SMA — free proxy for cycle position via our own klines
        # use Binance public for 200w SMA ≈ 1400 daily bars
        klines = http_get("https://fapi.binance.com/fapi/v1/klines",
                         params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1500})
        closes = [float(k[4]) for k in klines]
        ma200w = sum(closes[-1400:]) / 1400  # 200 weeks ≈ 1400 days
        cur_price = closes[-1]
        ratio = cur_price / ma200w
        out["price"] = cur_price
        out["ma200w"] = round(ma200w, 0)
        out["price_vs_ma200w"] = round(ratio, 2)
        # historical: ratio < 1 = cycle bottom, 1-2 = accumulation/bull, 2-4 = bull, 4+ = top
        if   ratio < 1.0:  score_ma = -2
        elif ratio < 1.5:  score_ma = -1
        elif ratio < 2.0:  score_ma = 0
        elif ratio < 3.0:  score_ma = 1
        elif ratio < 4.0:  score_ma = 1
        else:              score_ma = -1  # too high, distribution risk
        # combine: primary score from MA200w ratio
        score = score_ma
        # adjust if hashrate dropping (bearish sign)
        if hr_change < -10:
            score -= 1
        out["score"] = clamp(score)
        return out
    except Exception as e:
        out["error"] = str(e)
        out["score"] = 0
        return out

# ==================== AXIS 3: DERIVATIVES ====================
def axis3_derivatives():
    out = {}
    try:
        # funding rate — last ~8 days (24 records of 8h each)
        fr = http_get("https://fapi.binance.com/fapi/v1/fundingRate",
                     params={"symbol": "BTCUSDT", "limit": 30})
        rates = [float(x["fundingRate"]) * 100 for x in fr]   # percent per 8h
        avg_7d = sum(rates[-21:]) / 21     # last 7 days = 21 periods
        last = rates[-1]
        out["funding_7d_avg_pct_per_8h"] = round(avg_7d, 4)
        out["funding_last_pct"] = round(last, 4)

        # open interest
        oi = http_get("https://fapi.binance.com/fapi/v1/openInterest",
                     params={"symbol": "BTCUSDT"})
        oi_usd = float(oi["openInterest"])
        # historical OI trend
        oih = http_get("https://fapi.binance.com/futures/data/openInterestHist",
                       params={"symbol": "BTCUSDT", "period": "1d", "limit": 30})
        oi_now = float(oih[-1]["sumOpenInterest"])
        oi_past = float(oih[0]["sumOpenInterest"])
        oi_change = (oi_now - oi_past) / oi_past * 100
        out["oi_30d_change_pct"] = round(oi_change, 2)
        out["oi_current_btc"] = round(oi_now, 0)

        # basis: futures premium vs spot
        spot = http_get("https://api.binance.com/api/v3/ticker/price",
                       params={"symbol": "BTCUSDT"})
        fut = http_get("https://fapi.binance.com/fapi/v1/ticker/price",
                      params={"symbol": "BTCUSDT"})
        basis_pct = (float(fut["price"]) - float(spot["price"])) / float(spot["price"]) * 100
        out["perp_spot_basis_pct"] = round(basis_pct, 4)

        # long/short ratio (top traders)
        ls = http_get("https://fapi.binance.com/futures/data/topLongShortAccountRatio",
                     params={"symbol": "BTCUSDT", "period": "1d", "limit": 1})
        ls_ratio = float(ls[-1]["longShortRatio"])
        out["top_trader_long_short_ratio"] = round(ls_ratio, 3)

        # score composition:
        # funding: >0.1%/8h = -2 (overheated), 0.02-0.05 = +1 (healthy bull),
        #          ~0 = 0, <-0.01 = +2 (shorts piled, contrarian bull)
        if avg_7d > 0.10:   fs = -2
        elif avg_7d > 0.05: fs = -1
        elif avg_7d > 0.02: fs = 1
        elif avg_7d > -0.01: fs = 0
        elif avg_7d > -0.03: fs = 1
        else:               fs = 2

        # OI trend: spiking up = crowded, flat/down = neutral
        if oi_change > 30:    os_ = -1
        elif oi_change > 10:  os_ = 0
        elif oi_change > -10: os_ = 0
        else:                 os_ = 1

        # basis: high positive = euphoria, negative = bear
        if basis_pct > 0.3:   bs = -1
        elif basis_pct > 0.1: bs = 0
        elif basis_pct > -0.05: bs = 0
        else:                 bs = 1

        score = fs + os_ + bs
        out["sub_scores"] = {"funding": fs, "oi_trend": os_, "basis": bs}
        out["score"] = clamp(score)
        return out
    except Exception as e:
        out["error"] = str(e)
        out["score"] = 0
        return out

# ==================== AXIS 4: MACRO (FRED + yfinance) ====================
def axis4_macro():
    out = {}
    if not FRED_KEY:
        out["error"] = "FRED_API_KEY not set"
        out["score"] = 0
        return out
    def fred(series_id, limit=400):
        return http_get("https://api.stlouisfed.org/fred/series/observations",
                       params={"series_id": series_id, "api_key": FRED_KEY,
                               "file_type": "json", "limit": limit,
                               "sort_order": "desc"})
    try:
        # Fed Funds Rate — current policy stance
        ffr = fred("FEDFUNDS", limit=12)
        ffr_latest = float(ffr["observations"][0]["value"])
        ffr_6mo_ago = float(ffr["observations"][6]["value"])
        ffr_trend = ffr_latest - ffr_6mo_ago  # negative = cutting = bullish
        out["fed_funds_rate_pct"] = ffr_latest
        out["fed_funds_6mo_change"] = round(ffr_trend, 2)

        # M2 money supply — liquidity
        m2 = fred("M2SL", limit=12)
        m2_latest = float(m2["observations"][0]["value"])
        m2_yr_ago = float(m2["observations"][11]["value"])
        m2_yoy = (m2_latest - m2_yr_ago) / m2_yr_ago * 100
        out["m2_yoy_pct"] = round(m2_yoy, 2)

        # CPI — inflation
        cpi = fred("CPIAUCSL", limit=13)
        cpi_latest = float(cpi["observations"][0]["value"])
        cpi_yr_ago = float(cpi["observations"][12]["value"])
        cpi_yoy = (cpi_latest - cpi_yr_ago) / cpi_yr_ago * 100
        out["cpi_yoy_pct"] = round(cpi_yoy, 2)

        # Real Fed rate
        real_rate = ffr_latest - cpi_yoy
        out["real_fed_rate_pct"] = round(real_rate, 2)

        # 10Y Treasury via FRED
        t10 = fred("DGS10", limit=30)
        t10_latest = float(next(o for o in t10["observations"] if o["value"] != "."))["value" if False else "value"] if False else float([o["value"] for o in t10["observations"] if o["value"] != "."][0])
        out["treasury_10y_pct"] = t10_latest

        # 2Y Treasury
        t2 = fred("DGS2", limit=30)
        t2_latest = float([o["value"] for o in t2["observations"] if o["value"] != "."][0])
        out["treasury_2y_pct"] = t2_latest
        out["yield_curve_10y_2y"] = round(t10_latest - t2_latest, 2)

        # DXY via yfinance (FRED has DTWEXBGS but different base)
        import yfinance as yf
        dxy = yf.download("DX-Y.NYB", period="60d", progress=False, auto_adjust=True)
        if not dxy.empty:
            dxy_now = float(dxy["Close"].iloc[-1])
            dxy_30d = float(dxy["Close"].iloc[-30]) if len(dxy) > 30 else float(dxy["Close"].iloc[0])
            dxy_change = (dxy_now - dxy_30d) / dxy_30d * 100
            out["dxy_current"] = round(dxy_now, 2)
            out["dxy_30d_change_pct"] = round(dxy_change, 2)
        else:
            dxy_change = 0

        # QQQ — nasdaq correlation
        qqq = yf.download("QQQ", period="60d", progress=False, auto_adjust=True)
        if not qqq.empty:
            qqq_now = float(qqq["Close"].iloc[-1])
            qqq_30d = float(qqq["Close"].iloc[-30]) if len(qqq) > 30 else float(qqq["Close"].iloc[0])
            qqq_change = (qqq_now - qqq_30d) / qqq_30d * 100
            out["qqq_30d_change_pct"] = round(qqq_change, 2)
        else:
            qqq_change = 0

        # scoring
        # Fed cutting = +2, hiking = -2
        if ffr_trend < -0.5:   fs = 2
        elif ffr_trend < 0:    fs = 1
        elif ffr_trend < 0.5:  fs = 0
        else:                  fs = -2
        # M2 YoY: growing = +2 (liquidity expansion), shrinking = -2
        if m2_yoy > 8:     m2s = 2
        elif m2_yoy > 4:   m2s = 1
        elif m2_yoy > 0:   m2s = 0
        elif m2_yoy > -2:  m2s = -1
        else:              m2s = -2
        # DXY: falling = bullish BTC, rising = bearish
        if dxy_change < -3:   ds = 2
        elif dxy_change < 0:  ds = 1
        elif dxy_change < 3:  ds = -1
        else:                 ds = -2
        # QQQ: positive = risk-on = bullish BTC
        if qqq_change > 5:    qs = 1
        elif qqq_change > 0:  qs = 0
        elif qqq_change > -5: qs = -1
        else:                 qs = -2

        score = (fs + m2s + ds + qs) / 4   # average, since we're combining 4 macro signals
        score = round(score)
        out["sub_scores"] = {"fed": fs, "m2": m2s, "dxy": ds, "qqq": qs}
        out["score"] = clamp(score)
        return out
    except Exception as e:
        import traceback
        out["error"] = str(e)
        out["traceback"] = traceback.format_exc()[-500:]
        out["score"] = 0
        return out

# ==================== AXIS 5: SENTIMENT ====================
def axis5_sentiment():
    out = {}
    try:
        # Fear & Greed
        fg = http_get("https://api.alternative.me/fng/?limit=30")
        fg_now = int(fg["data"][0]["value"])
        fg_7d_avg = sum(int(d["value"]) for d in fg["data"][:7]) / 7
        out["fear_greed_current"] = fg_now
        out["fear_greed_7d_avg"] = round(fg_7d_avg, 1)
        out["fear_greed_label"] = fg["data"][0]["value_classification"]

        # BTC dominance via CoinGecko
        global_d = http_get("https://api.coingecko.com/api/v3/global")
        btc_dom = global_d["data"]["market_cap_percentage"]["btc"]
        out["btc_dominance_pct"] = round(btc_dom, 2)
        out["eth_dominance_pct"] = round(global_d["data"]["market_cap_percentage"].get("eth", 0), 2)

        # scoring
        # F&G extreme values are contrarian:
        # <20 = extreme fear = buying opp = +2
        # 20-40 = fear = +1
        # 40-60 = neutral = 0
        # 60-80 = greed = -1
        # >80 = extreme greed = -2
        if fg_now < 20:    fgs = 2
        elif fg_now < 40:  fgs = 1
        elif fg_now < 55:  fgs = 0
        elif fg_now < 75:  fgs = -1
        else:              fgs = -2

        # dominance: rising BTC dom = alt bear / BTC accumulation = +1
        # falling = alt season = bull late cycle = -1
        # For scoring, we look at level: 55+ bull dom, <45 alt season
        if btc_dom > 60:    doms = 1    # BTC dominant = accumulation stage
        elif btc_dom > 55:  doms = 0
        elif btc_dom > 45:  doms = -1   # alts gaining = late bull
        else:               doms = -2   # alt season = very late bull

        score = fgs + doms
        out["sub_scores"] = {"fear_greed": fgs, "btc_dominance": doms}
        out["score"] = clamp(score)
        return out
    except Exception as e:
        out["error"] = str(e)
        out["score"] = 0
        return out

# ==================== AGGREGATION ====================
def run():
    print(f"\n{'='*70}")
    print(f"BTC Cycle Check — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*70}")

    a1 = axis1_cycle_clock()
    a2 = axis2_onchain()
    a3 = axis3_derivatives()
    a4 = axis4_macro()
    a5 = axis5_sentiment()

    axes = {
        "1_cycle_clock": a1,
        "2_onchain": a2,
        "3_derivatives": a3,
        "4_macro": a4,
        "5_sentiment": a5,
    }

    total = a1["score"] + a2["score"] + a3["score"] + a4["score"] + a5["score"]

    def stage(t):
        if t >= 7:   return "明确牛中——加仓 / 全仓追涨"
        elif t >= 3: return "上升期——持有 / 逢低买"
        elif t >= -2: return "模糊区——让机械规则(v5f)决定"
        elif t >= -6: return "下降期——减仓 / 观望"
        else:        return "熊市深处——定投 / 等投降"

    def agent_advice(t):
        if t >= 7:   return "仓位 100% 信号、杠杆可 2x"
        elif t >= 3: return "仓位 80-100% 信号、杠杆 1x"
        elif t >= -2: return "仓位 50% 信号、无杠杆"
        elif t >= -6: return "仓位 20-30% 信号、无杠杆"
        else:        return "仓位 0-20% 信号、或完全空仓"

    print(f"\n{'-'*70}")
    print(f"AXIS SCORES")
    print(f"{'-'*70}")
    for name, a in axes.items():
        print(f"  {name:20s}: {a['score']:+d}")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL':20s}: {total:+d}  (range -10 to +10)")

    print(f"\n{'-'*70}")
    print(f"CYCLE STAGE CALL")
    print(f"{'-'*70}")
    print(f"  Stage:  {stage(total)}")
    print(f"  Agent:  {agent_advice(total)}")

    print(f"\n{'-'*70}")
    print(f"DETAILED INDICATORS")
    print(f"{'-'*70}")
    for name, a in axes.items():
        print(f"\n[{name}]  score={a['score']:+d}")
        for k, v in a.items():
            if k == "score": continue
            if isinstance(v, dict):
                print(f"  {k}:")
                for k2, v2 in v.items():
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {k}: {v}")

    # save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_score": total,
        "stage": stage(total),
        "agent_advice": agent_advice(total),
        "axes": axes,
    }
    out_file = ROOT / "cache" / f"cycle_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
    out_file.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport saved: {out_file}")

if __name__ == "__main__":
    run()
