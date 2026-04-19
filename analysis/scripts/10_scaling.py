"""Phase 4.2: scale-in / scale-out behavior.

Re-walk the XBTUSD fills and, for each roundtrip, classify every fill as:
  OPEN   — first fill that establishes position
  ADD    — same-direction fill that increases position size
  TRIM   — opposite-direction fill that reduces position without closing
  CLOSE  — opposite-direction fill that brings position to 0

For each ADD: was the trade in profit (+) or loss (-) at that moment?
  → Tests pyramiding (add-to-winner) vs martingale (add-to-loser) hypothesis.

For each TRIM: was MFE already reached? (early profit-taking vs distribution).
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("/Users/jimu/btc-trader-analysis/data")
OUT = DATA / "derived"

fills = pd.read_csv(DATA / "api-v1-execution-tradeHistory.csv", low_memory=False)
fills = fills[(fills["execType"] == "Trade") & (fills["symbol"] == "XBTUSD")].copy()
fills["timestamp"] = pd.to_datetime(fills["timestamp"], utc=True)
fills = fills.sort_values("timestamp").reset_index(drop=True)
fills["signed_qty"] = np.where(fills["side"] == "Buy", fills["lastQty"], -fills["lastQty"])
fills["fill_price"] = fills["lastPx"].fillna(fills["price"])

events = []   # one dict per fill with role + state

position = 0
rt_id = 0
running_entry_notional = 0.0   # USD
running_entry_contracts = 0    # contracts
rt_start_time = None

for row in fills.itertuples(index=False):
    fill = row.signed_qty
    price = row.fill_price
    if price is None or pd.isna(price) or price == 0:
        continue

    # compute mark-to-market on the open position BEFORE this fill
    if position != 0 and running_entry_contracts > 0:
        avg_entry = running_entry_notional / running_entry_contracts
        sign = 1 if position > 0 else -1
        unrealized_pct = sign * (price - avg_entry) / avg_entry * 100
        hours_in = (row.timestamp - rt_start_time).total_seconds() / 3600
    else:
        avg_entry = None
        unrealized_pct = None
        hours_in = 0.0

    # classify
    if position == 0:
        role = "OPEN"
        rt_id += 1
        rt_start_time = row.timestamp
        running_entry_notional = abs(fill) * price
        running_entry_contracts = abs(fill)
    elif (position > 0) == (fill > 0):
        role = "ADD"
        running_entry_notional += abs(fill) * price
        running_entry_contracts += abs(fill)
    else:
        # opposite direction
        abs_fill = abs(fill)
        abs_pos = abs(position)
        if abs_fill < abs_pos:
            role = "TRIM"
        else:
            role = "CLOSE" if abs_fill == abs_pos else "CLOSE_AND_FLIP"

    events.append({
        "rt_id": rt_id,
        "timestamp": row.timestamp,
        "role": role,
        "side": row.side,
        "contracts": abs(fill),
        "price": price,
        "direction": "long" if (position > 0 or (position == 0 and fill > 0)) else "short",
        "position_before": position,
        "avg_entry_before": avg_entry,
        "unrealized_pct_before": unrealized_pct,
        "hours_in_position": hours_in,
    })

    # update position
    if role in ("OPEN", "ADD"):
        position += fill
    elif role == "TRIM":
        position += fill
    elif role == "CLOSE":
        position = 0
        running_entry_notional = 0.0
        running_entry_contracts = 0
    elif role == "CLOSE_AND_FLIP":
        # close old, open new
        position = fill + position   # equals opening-side remainder
        # determine new sign
        new_sign = 1 if fill > 0 else -1
        remain = abs(fill) - (abs(position - (fill + (0 if position == 0 else (-position + fill)))))
        # simpler: the leftover opening piece = abs_fill - abs_pos_before
        # but position_before already captured via "position" var above; since we mutated,
        # just recompute: new_position magnitude = abs(fill) - abs(position_before)
        # but we updated position already — undo. reset:
        pass

# re-run clean with flip handling (second pass)
# actually the above mixed logic — rewrite from scratch, cleaner.
events.clear()
position = 0
rt_id = 0
running_entry_notional = 0.0
running_entry_contracts = 0
rt_start_time = None

for row in fills.itertuples(index=False):
    fill = row.signed_qty
    price = row.fill_price
    if price is None or pd.isna(price) or price == 0:
        continue

    pos_before = position

    if pos_before != 0 and running_entry_contracts > 0:
        avg_entry = running_entry_notional / running_entry_contracts
        sign = 1 if pos_before > 0 else -1
        unrealized_pct = sign * (price - avg_entry) / avg_entry * 100
        hours_in = (row.timestamp - rt_start_time).total_seconds() / 3600
    else:
        avg_entry = None
        unrealized_pct = None
        hours_in = 0.0

    if pos_before == 0:
        role = "OPEN"
        rt_id += 1
        rt_start_time = row.timestamp
        running_entry_notional = abs(fill) * price
        running_entry_contracts = abs(fill)
        position = fill
    elif (pos_before > 0) == (fill > 0):
        role = "ADD"
        running_entry_notional += abs(fill) * price
        running_entry_contracts += abs(fill)
        position = pos_before + fill
    else:
        abs_fill = abs(fill)
        abs_pos = abs(pos_before)
        if abs_fill < abs_pos:
            role = "TRIM"
            position = pos_before + fill
        elif abs_fill == abs_pos:
            role = "CLOSE"
            position = 0
            running_entry_notional = 0.0
            running_entry_contracts = 0
        else:
            role = "FLIP"
            # record a CLOSE event then open new
            events.append({
                "rt_id": rt_id, "timestamp": row.timestamp, "role": "CLOSE",
                "side": row.side, "contracts": abs_pos, "price": price,
                "direction": "long" if pos_before > 0 else "short",
                "position_before": pos_before, "avg_entry_before": avg_entry,
                "unrealized_pct_before": unrealized_pct, "hours_in_position": hours_in,
            })
            rt_id += 1
            rt_start_time = row.timestamp
            opening = abs_fill - abs_pos
            running_entry_notional = opening * price
            running_entry_contracts = opening
            position = opening if fill > 0 else -opening
            events.append({
                "rt_id": rt_id, "timestamp": row.timestamp, "role": "OPEN",
                "side": row.side, "contracts": opening, "price": price,
                "direction": "long" if fill > 0 else "short",
                "position_before": 0, "avg_entry_before": None,
                "unrealized_pct_before": None, "hours_in_position": 0.0,
            })
            continue

    events.append({
        "rt_id": rt_id, "timestamp": row.timestamp, "role": role,
        "side": row.side, "contracts": abs(fill), "price": price,
        "direction": "long" if (pos_before > 0 or (pos_before == 0 and fill > 0)) else "short",
        "position_before": pos_before, "avg_entry_before": avg_entry,
        "unrealized_pct_before": unrealized_pct, "hours_in_position": hours_in,
    })

ev = pd.DataFrame(events)
print(f"total events: {len(ev):,}")
print(f"\nby role:")
print(ev["role"].value_counts())

def hdr(t): print(f"\n{'='*70}\n{t}\n{'='*70}")

# === 1. ADD behavior: winning or losing when adding? ===
hdr("1. ADD events — pyramiding vs martingale")
adds = ev[ev["role"] == "ADD"].dropna(subset=["unrealized_pct_before"])
print(f"total ADD events: {len(adds)}")
print(f"ADDs while in profit:  {(adds['unrealized_pct_before'] > 0).sum()}  ({(adds['unrealized_pct_before'] > 0).mean()*100:.1f}%)")
print(f"ADDs while in loss:    {(adds['unrealized_pct_before'] < 0).sum()}  ({(adds['unrealized_pct_before'] < 0).mean()*100:.1f}%)")
print(f"ADDs near flat (<0.3% either way): {(adds['unrealized_pct_before'].abs() < 0.3).sum()}")
print(f"\nadds' unrealized_pct_before distribution:")
print(adds["unrealized_pct_before"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

# by hours-in: does he add quickly or after thinking?
print(f"\nhours-in-position at ADD time:")
print(adds["hours_in_position"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))

# === 2. Per-roundtrip ADD analysis ===
hdr("2. Does 'adding in the loss' predict losing roundtrips?")
# classify each roundtrip by its ADD behavior
rt_meta = ev.groupby("rt_id").agg(
    n_events=("role", "size"),
    n_adds=("role", lambda s: (s == "ADD").sum()),
    n_trims=("role", lambda s: (s == "TRIM").sum()),
    added_in_loss=("unrealized_pct_before", lambda s: ((s < -0.5) & s.notna()).any()),
    added_in_profit=("unrealized_pct_before", lambda s: ((s > 0.5) & s.notna()).any()),
)
# merge with roundtrip outcomes
rt_o = pd.read_csv(OUT / "roundtrips_with_exits.csv")
rt_o["entry_time"] = pd.to_datetime(rt_o["entry_time"], utc=True, format="ISO8601")

# match by entry_time (approximate — flip events create extra rt_ids in our walk)
# simpler: recompute roundtrip-level summaries from ev directly
rt_walk = ev.groupby("rt_id").agg(
    start=("timestamp", "min"),
    end=("timestamp", "max"),
    first_side=("side", "first"),
    total_contracts=("contracts", "sum"),
    n_adds=("role", lambda s: (s == "ADD").sum()),
    n_trims=("role", lambda s: (s == "TRIM").sum()),
    max_unreal_loss=("unrealized_pct_before", "min"),
    max_unreal_gain=("unrealized_pct_before", "max"),
).reset_index()

# merge with real roundtrips by start time tolerance
rt_o = rt_o.sort_values("entry_time").reset_index(drop=True)
rt_walk = rt_walk.sort_values("start").reset_index(drop=True)

# use merge_asof with tolerance to link walks to real roundtrips
merged = pd.merge_asof(
    rt_o[["entry_time", "exit_time", "direction", "net_pnl_btc", "mfe_pct",
          "mae_pct", "exit_return_pct", "holding_hours"]].sort_values("entry_time"),
    rt_walk.rename(columns={"start": "walk_start"}).sort_values("walk_start"),
    left_on="entry_time", right_on="walk_start",
    direction="nearest", tolerance=pd.Timedelta("2min"),
)
merged = merged.dropna(subset=["rt_id"])
print(f"matched roundtrips: {len(merged)} / {len(rt_o)}")

# did adding-in-loss predict worse outcome?
merged["added_in_loss"] = merged["max_unreal_loss"] < -1   # at some point was holding -1%+ and added
# refine: did ADD events specifically happen in loss?
adds_by_rt = ev[ev["role"] == "ADD"].groupby("rt_id").agg(
    any_add_in_loss=("unrealized_pct_before", lambda s: ((s < -0.5) & s.notna()).any()),
    any_add_in_profit=("unrealized_pct_before", lambda s: ((s > 0.5) & s.notna()).any()),
    n_adds_in_loss=("unrealized_pct_before", lambda s: ((s < -0.5) & s.notna()).sum()),
    n_adds_in_profit=("unrealized_pct_before", lambda s: ((s > 0.5) & s.notna()).sum()),
)
merged = merged.merge(adds_by_rt, on="rt_id", how="left")
merged[["any_add_in_loss", "any_add_in_profit"]] = merged[["any_add_in_loss", "any_add_in_profit"]].fillna(False)

print("\n4 categories by ADD style:")
def style(row):
    if row["any_add_in_loss"] and row["any_add_in_profit"]:
        return "both"
    if row["any_add_in_loss"]:
        return "martingale_only"
    if row["any_add_in_profit"]:
        return "pyramid_only"
    return "no_meaningful_add"
merged["add_style"] = merged.apply(style, axis=1)

perf = merged.groupby("add_style").agg(
    n=("net_pnl_btc", "size"),
    win_rate=("net_pnl_btc", lambda s: (s > 0).mean()),
    avg_pnl=("net_pnl_btc", "mean"),
    total_pnl=("net_pnl_btc", "sum"),
    median_mae=("mae_pct", "median"),
    median_mfe=("mfe_pct", "median"),
)
print(perf.round(3).to_string())

# === 3. TRIM behavior ===
hdr("3. TRIM behavior — when does he scale out?")
trims = ev[ev["role"] == "TRIM"].dropna(subset=["unrealized_pct_before"])
print(f"total TRIM events: {len(trims)}")
print(f"TRIMs while in profit:  {(trims['unrealized_pct_before'] > 0).sum()}  ({(trims['unrealized_pct_before'] > 0).mean()*100:.1f}%)")
print(f"TRIMs while in loss:    {(trims['unrealized_pct_before'] < 0).sum()}  ({(trims['unrealized_pct_before'] < 0).mean()*100:.1f}%)")
print(f"\ntrims' unrealized_pct_before distribution:")
print(trims["unrealized_pct_before"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(3))

# === 4. ADD frequency over time ===
hdr("4. Timing of ADD events within a roundtrip (hours after entry)")
print(adds["hours_in_position"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2))

# what fraction of ADDs happen in the first hour vs later?
print(f"\nADD timing buckets:")
add_hr_bins = pd.cut(adds["hours_in_position"], bins=[0, 0.5, 2, 6, 24, 72, 10000],
                     labels=["<30min", "30min-2h", "2-6h", "6-24h", "1-3d", ">3d"])
print(add_hr_bins.value_counts().sort_index())

ev.to_csv(OUT / "fill_events.csv", index=False)
print(f"\nsaved fill events -> {OUT / 'fill_events.csv'}")
