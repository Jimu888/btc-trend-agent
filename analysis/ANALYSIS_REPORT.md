# Analysis Report — How We Derived v5f

Condensed synthesis of the four-phase analysis.
Full scripts in `./scripts/` are numbered in execution order.

---

## Starting point

- **Raw data**: [`bwjoke/BTC-Trading-Since-2020`](https://github.com/bwjoke/BTC-Trading-Since-2020) — 173k fills, 43k orders, 17k wallet events from a single BitMEX account, 2020-05 to 2026-04.
- **Subject**: Paul Wei (`@coolish`), listed in BitMEX's Hall of Legends. Account grew 1.84 BTC → 96 BTC across the period (94 BTC net trading gain, mostly XBTUSD inverse perpetual).

## Phase 1 — Round-trip reconstruction

Script `02_roundtrips.py` walks the XBTUSD fills, aggregates into position paths, and closes round-trips when position returns to zero.

Result: **687 round-trip trades**, ~115 per year, median hold 12 hours, 3-7 day holds being his sweet spot.

## Phase 2 — Behavioral profile

Script `05_profile.py` computes standard trading metrics:

| Metric | Value |
|---|---|
| Win rate | 59.5% |
| Avg win / avg loss | +0.40 / -0.43 BTC |
| Profit factor | 1.34 |
| Long bias | 62% of trades |
| Long win rate | 63% |
| Short win rate | 54% |
| Biggest single loss | -12.3 BTC (2021-03 oversized long before the top) |
| Biggest single win | +13.2 BTC (2022-01 top-catching short) |

**Key finding**: He evolved dramatically. By 2025 he traded only 8 times (vs. 204 in 2020) with 87.5% win rate and median hold 99h (vs. 10h in 2020). Deliberate selectivity.

## Phase 3 — Feature engineering + ML

Script `07_features.py` computes 31 K-line features at each entry time (no lookahead).
Script `11_ml.py` trains gradient-boosting classifiers to predict outcomes from those features.

| Task | Time-series CV AUC | Interpretation |
|---|---|---|
| Predict direction (long/short) | 0.56 | Weak |
| Predict win vs loss | 0.55 | Weak |
| Predict "big win" (top 25%) | 0.52 | Essentially random |

**This is the most important finding in the whole study**: his entry decisions are *not* strongly predictable from chart features. His winners and losers look nearly identical at the moment of entry. Whatever edge he has lives elsewhere (execution, conviction, maybe off-chart signals) — not in a reproducible entry pattern.

## Phase 4 — Execution analysis

### Exit behavior (`09_exits.py`)

- MFE utilization: median 25% — he routinely gives back 75% of the best unrealized profit he saw.
- 55% of his losing trades were profitable at some point (+1% or more) before reversing.
- A fixed +6h exit would have beaten his actual exits in aggregate returns.
- Exit category **"was briefly +5.6% but closed -1.1%"** cost 80 trades × ~-0.5 BTC ≈ -37 BTC. His single biggest leak.

### Scale-in behavior (`10_scaling.py`)

Classified 48,364 fill events as OPEN / ADD / TRIM / CLOSE.

| Add style | n | Win rate | Total PnL |
|---|---|---|---|
| **Pyramid only** (adds only in profit) | 66 | **68.2%** | **+32.7 BTC** |
| No meaningful adds | 284 | 60.2% | +14.2 BTC |
| Both | 149 | 59.7% | +11.2 BTC |
| **Martingale only** (adds only in loss) | **188** | **55.3%** | **-17.0 BTC** ❌ |

Single cleanest finding: **adding to losing positions costs him 17 BTC over 6 years** compared to his no-add baseline. If he had replaced every martingale with a no-add, he would have made ~30 more BTC.

## Backtest iterations

We tried several variants before landing on v5f:

| Version | Entry | Exit | Return | Max DD |
|---|---|---|---|---|
| v2 | Paul-style regimes, fixed -5% stop | Various rules | -81% | — |
| v3 | Tighter, ATR stops | ATR trail | -45% | — |
| v4 | Breakout + regime | ATR stops | -28% | -33% |
| v5a | Simple MA200 regime | Close < MA200 | +92% | -36% |
| **v5f** ★ | **Close > MA150 + MA50 rising** | **Close < MA150** | **+821%** | **-44%** |

Lesson: every attempted "Paul-style" execution layer (stops, trails, partials, pyramids) *destroyed* performance. The mechanized version works precisely because it strips away his discretionary complications.

## Walk-forward validation (`23_walkforward.py`)

- Parameter sensitivity across 20 `(ma_long, slope_w)` combinations: median return +384%, no combination lost money
- In-sample (2020-2022) vs out-of-sample (2023-2026): OOS actually *outperforms* IS at every tested parameter
- Rolling 2-year windows: 7 out of 7 profitable, median return +95%, worst window +2.8%

**Conclusion**: v5f is not curve-fit. Its edge survives parameter perturbation and temporal split.

## What about local vision models? (`18_gemma_batch.py`, `19_gemma_200.py`)

We tested whether a local Gemma-4 Vision model (8B params) could serve as a quality filter on Paul's candidate trades.

- **50-trade pilot**: Gemma's "agrees with Paul" set showed 75% win rate vs. 59% baseline. Looked promising.
- **200-trade validation**: Effect dissolved. Mann-Whitney U p=0.27 — not statistically significant.

The pilot signal was a small-sample artifact. Vision-based gatekeeping is **not** added to the final agent.

## Final recommendation

Use v5f (or v5_short / weekly variant) in its mechanical form. Do not add "his" execution rules back in — they don't help at the trend-following timeframe.

**The full irony**: a simple 150-day moving average system outperformed the legendary discretionary trader it was designed to learn from. His edge is real but lives in dimensions (conviction, sizing judgment, perhaps information) that don't transfer to code.
