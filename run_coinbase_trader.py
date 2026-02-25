#!/usr/bin/env python3
"""
Autonomous Coinbase day trader: Omega VI strategy on Coinbase.

- Fetches BTC-USD data from Coinbase, computes signal, executes market orders on Coinbase
  (or updates paper state).
- Paper (default): no API keys; state in paper_state.json.
- Live: set BTCOmega_MODE=live and COINBASE_API_KEY + COINBASE_API_SECRET.

Optional: run continuously (day trading loop) with RUN_INTERVAL_MINUTES (e.g. 60).

Usage:
  python run_coinbase_trader.py              # one-shot, paper
  BTCOmega_MODE=live python run_coinbase_trader.py   # one-shot, live Coinbase
  RUN_INTERVAL_MINUTES=60 python run_coinbase_trader.py   # loop every 60 min (paper)
"""

import json
import os
import sys
import time
import logging

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from btc_omega6_engine import get_live_signal
from coinbase_api import (
    get_candles_for_engine,
    get_spot_price,
    get_balance,
    build_market_order,
    create_order,
)

# ─── Config ─────────────────────────────────────────────────────────────────

PRODUCT_ID = "BTC-USD"
DATA_PATH_ENV = "BTCOmega_DATA_JSON"
PAPER_STATE_ENV = "BTCOmega_PAPER_STATE"
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_omega2_data.json")
DEFAULT_PAPER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_state.json")
DEFAULT_TRADES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trades.json")
MIN_CANDLES = 200
MIN_TRADE_BTC = 0.0001
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"


def _trades_path() -> str:
    return os.environ.get("BTCOmega_TRADES_JSON", DEFAULT_TRADES_PATH)


def _append_trade(side: str, base_btc: float, quote_usd: float, price: float, mode: str, log) -> None:
    """Append one executed trade to trades.json for dashboard/history."""
    path = _trades_path()
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "side": side.lower(),
        "base_btc": round(base_btc, 8),
        "quote_usd": round(quote_usd, 2),
        "price": round(price, 2),
        "mode": mode,
    }
    try:
        data = []
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
        if not isinstance(data, list):
            data = []
        data.append(entry)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.warning("Could not write trades.json: %s", e)


def _data_path() -> str:
    return os.environ.get(DATA_PATH_ENV, DEFAULT_DATA_PATH)


def _paper_state_path() -> str:
    return os.environ.get(PAPER_STATE_ENV, DEFAULT_PAPER_PATH)


def _ensure_data(max_age_hours: float = 24) -> list:
    """Fetch daily candles from Coinbase if file missing or stale. Returns engine-style rows."""
    path = _data_path()
    if os.path.isfile(path):
        age = (os.path.getmtime(path) if path else 0) or 0
        if age and (time.time() - age) < max_age_hours * 3600:
            with open(path) as f:
                raw = json.load(f)
            kraken = raw.get("1d_kraken", [])
            raw_365 = kraken if len(kraken) >= MIN_CANDLES else raw.get("365", [])
            if len(raw_365) >= MIN_CANDLES:
                return raw_365
    candles = get_candles_for_engine(PRODUCT_ID, granularity=86400, num_candles=365)
    # Engine expects [ts_ms, o, h, l, c] per row; get_candles_for_engine already returns that
    payload = {"1d_kraken": candles}
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return candles


def _daily_df_from_candles(candles: list) -> pd.DataFrame:
    """Build daily OHLC DataFrame from engine-style list of [ts_ms, o, h, l, c]."""
    rows = []
    for c in candles:
        ts = int(c[0])
        if ts < 1e12:
            ts = ts * 1000
        o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        vol = float(c[5]) if len(c) > 5 else 100.0
        rows.append({
            "timestamp": pd.Timestamp(ts, unit="ms"),
            "open": o, "high": h, "low": l, "close": close, "volume": vol,
        })
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def _read_paper_state() -> dict:
    path = _paper_state_path()
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    default = {"balance_usd": 10_000.0, "position_btc": 0.0, "last_price": 0.0}
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
    s = _read_paper_state()
    return float(s["balance_usd"]), float(s["position_btc"]), float(s["last_price"])


def _paper_execute(side: str, volume_btc: float, price: float, log) -> None:
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
    log.info("Paper %s %.6f BTC @ $%.2f", side.upper(), volume_btc, price)
    _append_trade(side, volume_btc, volume_btc * price, price, "paper", log)


def run_one_cycle(log) -> None:
    mode = (os.environ.get("BTCOmega_MODE") or "paper").strip().lower()
    if mode not in ("paper", "live"):
        log.error("BTCOmega_MODE must be 'paper' or 'live'")
        sys.exit(1)
    if mode == "live":
        has_env = os.environ.get("COINBASE_API_KEY") and os.environ.get("COINBASE_API_SECRET")
        cdp_path = os.environ.get("COINBASE_CDP_KEY_JSON", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cdp_api_key.json"))
        if not has_env and not os.path.isfile(cdp_path):
            log.error("For live trading set COINBASE_API_KEY and COINBASE_API_SECRET, or put cdp_api_key.json in the project")
            sys.exit(1)

    sl = float(os.environ.get("BTCOmega_SL_PCT", "2.0"))
    tp = float(os.environ.get("BTCOmega_TP_PCT", "4.0"))
    thr = float(os.environ.get("BTCOmega_THRESHOLD", "0.15"))
    max_position_pct = float(os.environ.get("BTCOmega_MAX_POSITION_PCT", "0.15"))  # cap size

    # Data
    candles = _ensure_data(max_age_hours=24)
    if len(candles) < 50:
        log.warning("Insufficient candles: %d", len(candles))
        return
    df = _daily_df_from_candles(candles)
    last_close = float(df["close"].iloc[-1])
    # Use spot for live execution price
    try:
        spot = get_spot_price(PRODUCT_ID)
        last_price = float(spot.get("amount", last_close) or last_close)
    except Exception:
        last_price = last_close

    sig = get_live_signal(df, sl_pct=sl, tp_pct=tp, threshold=thr)
    direction = sig["direction"]
    size_pct = min(sig["size_pct"], max_position_pct)
    log.info(
        "Signal: direction=%s (1=long -1=short 0=flat) size_pct=%.4f regime=%s vote=%.3f",
        direction, size_pct, sig.get("regime_name", "?"), sig.get("vote", 0),
    )

    if mode == "paper":
        balance_usd, position_btc, _ = _paper_balance_and_position()
        capital = balance_usd + position_btc * last_price
    else:
        bal = get_balance()
        balance_usd = bal["USD"]
        position_btc = bal["BTC"]
        capital = balance_usd + position_btc * last_price

    # Desired BTC: long = (capital * size_pct) / price; short = 0 (spot only)
    if direction == 1:
        desired_btc = (capital * size_pct) / last_price
    elif direction == -1:
        desired_btc = 0.0
    else:
        desired_btc = position_btc

    delta_btc = desired_btc - position_btc
    log.info(
        "Thinking: capital=$%.2f balance_usd=$%.2f position_btc=%.6f desired_btc=%.6f delta_btc=%.6f price=$%.2f",
        capital, balance_usd, position_btc, desired_btc, delta_btc, last_price,
    )
    if abs(delta_btc) < MIN_TRADE_BTC:
        log.info("No trade: position in line with signal")
        return

    if delta_btc > 0:
        side = "buy"
        max_buy_btc = balance_usd / last_price
        vol = min(delta_btc, max_buy_btc)
        if vol < MIN_TRADE_BTC:
            log.info("No trade: insufficient size or balance")
            return
        if mode == "paper":
            _paper_execute(side, vol, last_price, log)
        else:
            dry_run = os.environ.get("BTCOmega_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
            payload = build_market_order("BUY", product_id=PRODUCT_ID, quote_size=f"{vol * last_price:.2f}")
            if dry_run:
                log.info("DRY-RUN: would BUY %.6f BTC (~$%.2f) — payload %s", vol, vol * last_price, payload)
            else:
                try:
                    result = create_order(payload)
                    log.info("Live BUY %.6f BTC (market) — %s", vol, result)
                    _append_trade("buy", vol, vol * last_price, last_price, "live", log)
                except Exception as e:
                    log.exception("Order failed: %s", e)
    else:
        side = "sell"
        vol = min(-delta_btc, position_btc)
        if vol < MIN_TRADE_BTC:
            log.info("No trade: insufficient position to sell")
            return
        if mode == "paper":
            _paper_execute(side, vol, last_price, log)
        else:
            dry_run = os.environ.get("BTCOmega_DRY_RUN", "").strip().lower() in ("1", "true", "yes")
            payload = build_market_order("SELL", product_id=PRODUCT_ID, base_size=f"{vol:.8f}")
            if dry_run:
                log.info("DRY-RUN: would SELL %.6f BTC — payload %s", vol, payload)
            else:
                try:
                    result = create_order(payload)
                    log.info("Live SELL %.6f BTC (market) — %s", vol, result)
                    _append_trade("sell", vol, vol * last_price, last_price, "live", log)
                except Exception as e:
                    log.exception("Order failed: %s", e)


def main():
    log_path = os.environ.get("BTCOmega_LOG_PATH", "").strip()
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_path if log_path else None,
        filemode="a",
    )
    log = logging.getLogger(__name__)
    if log_path:
        # Also print to stdout when logging to file (e.g. for systemd/docker)
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler(sys.stdout))

    interval_min = float(os.environ.get("RUN_INTERVAL_MINUTES", "0"))
    if interval_min <= 0:
        run_one_cycle(log)
        return

    log.info("Autonomous loop: every %.1f minutes (Ctrl+C to stop)", interval_min)
    while True:
        try:
            run_one_cycle(log)
        except Exception as e:
            log.exception("Cycle error: %s", e)
        time.sleep(interval_min * 60)


if __name__ == "__main__":
    main()
