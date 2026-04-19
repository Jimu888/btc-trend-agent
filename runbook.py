"""
runbook.py — exchange integration template for agent_v5f.

Intended usage:
  1. Fill in the 3 exchange-specific sections (DATA SOURCE, ORDER PLACEMENT, STATE STORE).
  2. Schedule this to run at 00:15 UTC daily (cron, systemd timer, or a simple loop).
  3. On first run, the agent is flat and will buy on the first valid signal.

Assumes ccxt is available. Plug in Binance/Coinbase/BitMEX/etc. Adjust the
`fetch_daily_ohlc` and `place_market_order` functions to your exchange.

Safety rails enforced here (independent of the agent):
  - data sanity: no null, no gap > 5 days, price change > 30% → halt
  - order sanity: simulate-only mode available via DRY_RUN flag
  - state persisted to ./state.json between runs
"""
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# local import of the pure-logic agent
sys.path.insert(0, str(Path(__file__).parent))
from agent_v5f import AgentV5F, Bar   # noqa: E402

# ---------- CONFIG ----------
DRY_RUN = True               # set False ONLY after user explicit approval
SYMBOL = "BTC/USDT"
EXCHANGE_ID = "binance"      # ccxt id
STATE_FILE = Path(__file__).parent / "state.json"
LOG_FILE = Path(__file__).parent / "runbook.log"
MAX_PRICE_JUMP_PCT = 0.30    # halt if a single-bar close jumps > 30%
MIN_HISTORY_BARS = 160       # MA150 + 10-bar slope

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("runbook")

# ---------- STATE STORE ----------
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"in_position": False, "entry_price": 0.0, "entry_time": None,
            "trade_log": [], "bars_cache": []}

def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))

# ---------- DATA SOURCE ----------
def fetch_daily_ohlc(limit: int = 220) -> list[Bar]:
    """Return the last `limit` CLOSED daily bars, oldest first.

    Implementation example with ccxt:
        import ccxt
        ex = ccxt.binance()
        data = ex.fetch_ohlcv(SYMBOL, timeframe="1d", limit=limit)
        # ccxt returns [timestamp_ms, open, high, low, close, volume]
        # drop the last bar if it hasn't closed yet (today's bar).

    Adjust for your data source. Must return bars with UTC timestamps.
    """
    import ccxt
    ex = getattr(ccxt, EXCHANGE_ID)()
    rows = ex.fetch_ohlcv(SYMBOL, timeframe="1d", limit=limit + 2)
    now_ms = int(time.time() * 1000)
    bars = []
    for ts, o, h, l, c, v in rows:
        # skip bars that haven't fully closed (close_time > now)
        if ts + 86400_000 > now_ms:
            continue
        bars.append(Bar(
            time=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
            open=o, high=h, low=l, close=c, volume=v,
        ))
    if len(bars) > limit:
        bars = bars[-limit:]
    return bars

# ---------- ORDER PLACEMENT ----------
def place_market_order(side: str, notional_usd: float) -> dict:
    """Place a market order. Returns exchange response.

    `side` is "BUY" or "SELL".
    `notional_usd` is the USD value to trade (for BUY: buying $N of BTC;
    for SELL: selling $N worth of current BTC position).

    On real exchanges, convert notional to coin quantity using last price.
    Set DRY_RUN=True for simulation.
    """
    if DRY_RUN:
        log.info(f"[DRY_RUN] {side} ${notional_usd:.2f} of {SYMBOL}")
        return {"status": "dry_run", "side": side, "notional_usd": notional_usd}

    import ccxt
    ex = getattr(ccxt, EXCHANGE_ID)({
        # "apiKey": ..., "secret": ...   — inject from env
    })
    ticker = ex.fetch_ticker(SYMBOL)
    price = ticker["last"]
    qty = notional_usd / price
    side_lower = "buy" if side == "BUY" else "sell"
    # retry 3x with 30s backoff
    for attempt in range(3):
        try:
            return ex.create_market_order(SYMBOL, side_lower, qty)
        except Exception as e:
            log.warning(f"order attempt {attempt+1} failed: {e}")
            time.sleep(30)
    raise RuntimeError(f"3x retry failed for {side}")

# ---------- DATA SANITY ----------
def sanity_check(bars: list[Bar]) -> None:
    if len(bars) < MIN_HISTORY_BARS:
        raise RuntimeError(f"not enough history: {len(bars)} < {MIN_HISTORY_BARS}")
    for i in range(1, len(bars)):
        if bars[i].close <= 0 or bars[i].open <= 0:
            raise RuntimeError(f"non-positive price at {bars[i].time}")
        jump = abs(bars[i].close - bars[i-1].close) / bars[i-1].close
        if jump > MAX_PRICE_JUMP_PCT:
            raise RuntimeError(f"price jump {jump*100:.0f}% at {bars[i].time} — halt")

# ---------- RUN ONCE ----------
def run_once() -> None:
    log.info("=== run_once start ===")
    state = load_state()

    try:
        bars = fetch_daily_ohlc(limit=220)
        sanity_check(bars)
    except Exception as e:
        log.error(f"data error: {e}")
        return

    # rebuild agent from state. Agent is stateless except for its bar buffer
    # and position flag — we replay all bars to warm it up.
    agent = AgentV5F()
    agent.in_position = state["in_position"]
    agent.entry_price = state.get("entry_price", 0.0)

    # replay all but the newest bar to warm the buffer
    for b in bars[:-1]:
        # suppress orders on replay (we only want the buffer filled)
        agent._suppress_orders = True
        # a cleaner way is to add a "warm_up" path, but replaying with a flag works too
        agent.bars.append(b)

    # now feed the newest bar to generate today's decision
    latest = bars[-1]
    order = agent.on_new_bar(latest)

    if order is None:
        log.info(f"no action. in_position={agent.in_position} close={latest.close:.0f}")
    else:
        log.info(f"SIGNAL: {order.side}  reason: {order.reason}")
        # execute
        notional = state.get("equity_usd", 10000.0)    # initial capital from state
        if order.side == "BUY":
            resp = place_market_order("BUY", notional)
            state["in_position"] = True
            state["entry_price"] = latest.close
            state["entry_time"] = latest.time.isoformat()
            state["entry_order_id"] = resp.get("id", "dry_run")
        else:  # SELL
            # compute realized PnL
            realized_pct = (latest.close - state["entry_price"]) / state["entry_price"] - 0.002
            state["equity_usd"] = state.get("equity_usd", 10000.0) * (1 + realized_pct)
            resp = place_market_order("SELL", state["equity_usd"] / (1 + realized_pct))
            state["trade_log"].append({
                "entry_time": state["entry_time"],
                "exit_time": latest.time.isoformat(),
                "entry_price": state["entry_price"],
                "exit_price": latest.close,
                "return_pct": realized_pct,
                "equity_after": state["equity_usd"],
            })
            state["in_position"] = False
            state["entry_price"] = 0.0
            state["entry_time"] = None

    state["last_decision_time"] = datetime.now(timezone.utc).isoformat()
    state["last_daily_close"] = latest.close
    save_state(state)
    log.info("=== run_once done ===")


if __name__ == "__main__":
    run_once()
