# BTC Cycle Monitor

5-axis macro cycle monitor for BTC. Produces a weekly score from -10 to +10 to inform position sizing for the [v5f trading agent](../agent_v5f.py).

## What it measures

| Axis | Data source | Free? | Indicators |
|---|---|---|---|
| 1. Halving cycle clock | Local date math | ✅ | Months since 2024-04 halving; historical top/bottom windows |
| 2. On-chain fundamentals | mempool.space, blockchain.info, Binance | ✅ partial | Hashrate trend, active addresses, price vs 200-week MA |
| 3. Derivatives structure | Binance public API | ✅ | Funding rate 7d avg, OI 30d change, perp-spot basis, top-trader L/S ratio |
| 4. Macro liquidity | FRED + yfinance | ✅ (free FRED key) | Fed funds rate, M2 YoY, CPI, real rate, 10Y-2Y, DXY, QQQ |
| 5. Sentiment | alternative.me, CoinGecko | ✅ | Fear & Greed Index + 7d avg, BTC dominance |

### Full coverage requires Glassnode (optional, paid)

For **axis 2** deeper coverage (MVRV Z-Score, SOPR, LTH supply, NUPL), a [Glassnode](https://studio.glassnode.com) subscription is recommended. The free tier lets you view charts but doesn't expose these via API.

## Setup

```bash
pip install pandas requests yfinance python-dotenv
```

Create a `.env` file (or set env var):
```bash
FRED_API_KEY=<your_key>   # Get from https://fredaccount.stlouisfed.org/apikeys (free)
```

## Usage

```bash
python weekly_cycle_check.py
```

Output:
- Individual axis scores (-2 to +2 each)
- Weighted total (-10 to +10)
- Cycle stage interpretation
- Position sizing recommendation for the v5f agent
- JSON snapshot saved to `cache/cycle_<timestamp>.json`

## Score → Stage → Agent action

| Total score | Stage | Recommended v5f position |
|---|---|---|
| +7 to +10 | 明确牛中 | 100% signal + 2x leverage ok |
| +3 to +6 | 上升期 | 80-100% signal, 1x |
| -2 to +2 | 模糊区 | 50% signal, no leverage |
| -3 to -6 | 下降期 | 20-30% signal, no leverage |
| -7 to -10 | 深熊 | 0-20% or flat |

## Integration with v5f

The cycle monitor does **not** override v5f's entry/exit signals. It only adjusts the **position size** applied to those signals. v5f says "go long" → cycle monitor decides "with how much capital".

The two systems are deliberately decoupled. v5f is mechanical (daily timeframe). Cycle monitor is macro (weekly timeframe). Trying to merge them caused significant backtest degradation — see parent [ANALYSIS_REPORT.md](../analysis/ANALYSIS_REPORT.md) § "Hybrid attempts".

## Known limitations

1. **Axis 2 is partial without Glassnode paid access**. Currently covered via proxy indicators (hashrate, price-vs-200wMA), but MVRV/SOPR — the cleanest cycle indicators — are not available via public free APIs.
2. **yfinance ticker `DX-Y.NYB` occasionally times out**. Script falls back to 0 on failure.
3. **Weighting is uniform across axes** (each contributes equally to total). Could be optimized if we had labeled historical data.
4. **Scheduled to run once per week** — faster frequencies would add noise without signal improvement.

## Planned improvements

See [`../HANDOVER.md`](../HANDOVER.md) § "Outstanding items" for the full backlog.
