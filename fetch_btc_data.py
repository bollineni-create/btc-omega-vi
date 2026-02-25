#!/usr/bin/env python3
"""
Fetch BTC/USD daily OHLC from Kraken public API and write JSON in the format
expected by btc_omega6_engine.py. No API key required.

Usage:
  python fetch_btc_data.py

Output path is controlled by BTCOmega_DATA_JSON (default: ./btc_omega2_data.json).
"""

import os
import json
import requests

KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
PAIR = "XBTUSD"
INTERVAL = 1440  # 1 day in minutes


def fetch_kraken_ohlc() -> list:
    """Fetch daily OHLC from Kraken. Returns list of [ts_ms, o, h, l, c, vwap, volume, count]."""
    out = []
    since = None
    while True:
        params = {"pair": PAIR, "interval": INTERVAL}
        if since is not None:
            params["since"] = since
        resp = requests.get(KRAKEN_OHLC_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise RuntimeError(f"Kraken API error: {data['error']}")
        result = data.get("result", {})
        # Key is like XXBTZUSD or XBTUSD depending on Kraken
        ohlc_key = next((k for k in result if k != "last"), None)
        if not ohlc_key:
            break
        candles = result[ohlc_key]
        if not candles:
            break
        for c in candles:
            # Kraken: time, open, high, low, close, vwap, volume, count (all strings)
            ts_sec = int(float(c[0]))
            ts_ms = ts_sec * 1000
            o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
            vwap = float(c[5]) if len(c) > 5 else 0.0
            vol = float(c[6]) if len(c) > 6 else 0.0
            cnt = int(c[7]) if len(c) > 7 else 0
            out.append([ts_ms, o, h, l, close, vwap, vol, cnt])
        last_ts = int(float(candles[-1][0]))
        if last_ts == since or len(candles) < 720:
            break
        since = last_ts
    # Engine expects oldest first; Kraken returns newest first
    out.sort(key=lambda x: x[0])
    return out


def main():
    data_path = os.environ.get(
        "BTCOmega_DATA_JSON",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "btc_omega2_data.json"),
    )
    candles = fetch_kraken_ohlc()
    payload = {"1d_kraken": candles}
    parent = os.path.dirname(os.path.abspath(data_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {len(candles)} daily candles to {data_path}")


if __name__ == "__main__":
    main()
