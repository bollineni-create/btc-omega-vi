#!/usr/bin/env python3
"""
Live/paper trader entrypoint: fetch data → get_live_signal → execute (paper or Kraken).
Default: paper mode. Set BTCOmega_MODE=live and KRAKEN_API_KEY/SECRET for live trading.

Usage:
  python run_live_trader.py

Run once per day (e.g. cron 0 1 * * * for 01:00 UTC).
"""

import json
import os
import sys

import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc_omega6_engine import get_live_signal
from kraken_client import add_order, get_balance, get_ohlc


def _data_path() -> str:
    return os.environ.get(
        "BTCOmega_DATA_JSON",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_omega2_data.json"),
    )


def _paper_state_path() -> str:
    return os.environ.get(
        "BTCOmega_PAPER_STATE",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_state.json"),
    )


def _ensure_data(max_age_hours: float = 24) -> None:
    """If data file missing or older than max_age_hours, fetch from Kraken and write."""
    path = _data_path()
    if os.path.isfile(path):
        age = (os.path.getmtime(path) if path else 0) or 0
        if age and (__import__("time").time() - age) < max_age_hours * 3600:
            return
    # Fetch and write
    candles = get_ohlc()
    payload = {"1d_kraken": candles}
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Fetched {len(candles)} candles to {path}")


def _daily_df_from_json(path: str) -> pd.DataFrame:
    """Build daily OHLC DataFrame from engine-style JSON (1d_kraken or 365)."""
    with open(path) as f:
        raw = json.load(f)
    kraken = raw.get("1d_kraken", [])
    raw_365 = kraken if len(kraken) >= 200 else raw.get("365", [])
    if not raw_365:
        raise ValueError(f"No 1d_kraken (>=200) or 365 candles in {path}")
    rows = []
    for c in raw_365:
        ts = int(c[0])  # ms or sec
        if ts < 1e12:
            ts = ts * 1000
        o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        vol = float(c[6]) if len(c) > 6 else 100.0
        rows.append({
            "timestamp": pd.Timestamp(ts, unit="ms"),
            "open": o, "high": h, "low": l, "close": close, "volume": vol,
        })
    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def _read_paper_state() -> dict:
    path = _paper_state_path()
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    # Default: 10k USD, 0 BTC
    default = {
        "balance_usd": 10_000.0,
        "position_btc": 0.0,
        "last_price": 0.0,
    }
    _write_paper_state(default)
    return default


def _write_paper_state(state: dict) -> None:
    path = _paper_state_path()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _paper_balance_and_position() -> tuple[float, float, float]:
    """Returns (balance_usd, position_btc, last_price)."""
    s = _read_paper_state()
    return float(s["balance_usd"]), float(s["position_btc"]), float(s["last_price"])


def _paper_execute(side: str, volume_btc: float, price: float) -> None:
    """Update paper ledger as if we executed the trade."""
    bal_usd, pos_btc, _ = _paper_balance_and_position()
    if side == "buy":
        cost = volume_btc * price
        bal_usd -= cost
        pos_btc += volume_btc
    else:
        proceeds = volume_btc * price
        bal_usd += proceeds
        pos_btc -= volume_btc
    _write_paper_state({
        "balance_usd": round(bal_usd, 2),
        "position_btc": round(pos_btc, 8),
        "last_price": price,
    })
    print(f"  Paper: {side.upper()} {volume_btc:.6f} BTC @ ${price:,.2f}")


def main():
    mode = (os.environ.get("BTCOmega_MODE") or "paper").strip().lower()
    if mode not in ("paper", "live"):
        print("BTCOmega_MODE must be 'paper' or 'live'")
        sys.exit(1)
    if mode == "live" and (not os.environ.get("KRAKEN_API_KEY") or not os.environ.get("KRAKEN_API_SECRET")):
        print("For live trading set KRAKEN_API_KEY and KRAKEN_API_SECRET")
        sys.exit(1)

    sl = float(os.environ.get("BTCOmega_SL_PCT", "2.0"))
    tp = float(os.environ.get("BTCOmega_TP_PCT", "4.0"))
    thr = float(os.environ.get("BTCOmega_THRESHOLD", "0.15"))

    _ensure_data()
    path = _data_path()
    df = _daily_df_from_json(path)
    last_close = float(df["close"].iloc[-1])

    sig = get_live_signal(df, sl_pct=sl, tp_pct=tp, threshold=thr)
    direction = sig["direction"]
    size_pct = sig["size_pct"]
    print(f"Signal: direction={direction} (1=long -1=short 0=flat) size_pct={size_pct:.4f} regime={sig['regime_name']} vote={sig['vote']:.3f}")

    if mode == "paper":
        balance_usd, position_btc, _ = _paper_balance_and_position()
        capital = balance_usd + position_btc * last_close
    else:
        bal = get_balance()
        balance_usd = bal["USD"]
        position_btc = bal["BTC"]
        capital = balance_usd + position_btc * last_close

    # Desired BTC exposure: long = (capital * size_pct) / price; short = 0 (we don't do spot short)
    if direction == 1:
        desired_btc = (capital * size_pct) / last_close
    elif direction == -1:
        desired_btc = 0.0
    else:
        desired_btc = position_btc  # no change

    min_trade_btc = 0.0001
    delta_btc = desired_btc - position_btc
    if abs(delta_btc) < min_trade_btc:
        print("No trade: position already in line with signal")
        return

    if delta_btc > 0:
        side = "buy"
        max_buy_btc = balance_usd / last_close
        vol = min(delta_btc, max_buy_btc)
        if vol < min_trade_btc:
            print("No trade: insufficient size or balance")
            return
        if mode == "paper":
            _paper_execute(side, vol, last_close)
        else:
            result = add_order(side, vol)
            print(f"  Live: {side.upper()} {vol:.6f} BTC (market) — {result}")
    else:
        side = "sell"
        vol = min(-delta_btc, position_btc)
        if vol < min_trade_btc:
            print("No trade: insufficient position to sell")
            return
        if mode == "paper":
            _paper_execute(side, vol, last_close)
        else:
            add_order(side, vol)
            print(f"  Live: {side.upper()} {vol:.6f} BTC (market)")

    print("Done.")


if __name__ == "__main__":
    main()
