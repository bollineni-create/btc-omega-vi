"""
Kraken REST client: public OHLC and private Balance / AddOrder / CancelOrder.
Uses env KRAKEN_API_KEY and KRAKEN_API_SECRET for private endpoints (no key = public only).
"""

import base64
import hashlib
import hmac
import os
import time
import urllib.parse
import requests

BASE_URL = "https://api.kraken.com"
OHLC_PATH = "/0/public/OHLC"
BALANCE_PATH = "/0/private/Balance"
ADD_ORDER_PATH = "/0/private/AddOrder"
CANCEL_ORDER_PATH = "/0/private/CancelOrder"
OPEN_ORDERS_PATH = "/0/private/OpenOrders"

PAIR = "XBTUSD"
INTERVAL_1D = 1440


def _sign(urlpath: str, data: dict, secret_b64: str) -> str:
    """Kraken API-Sign: HMAC-SHA512( path + SHA256(nonce + postdata), base64decode(secret) )."""
    postdata = urllib.parse.urlencode(data)
    encoded = (str(data["nonce"]) + postdata).encode()
    message = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac = hmac.new(base64.b64decode(secret_b64), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()


def _private_request(path: str, data: dict | None = None) -> dict:
    key = os.environ.get("KRAKEN_API_KEY")
    secret = os.environ.get("KRAKEN_API_SECRET")
    if not key or not secret:
        raise RuntimeError("KRAKEN_API_KEY and KRAKEN_API_SECRET must be set for private API calls")
    data = dict(data or {})
    data["nonce"] = str(int(time.time() * 1000))
    headers = {
        "API-Key": key,
        "API-Sign": _sign(path, data, secret),
    }
    resp = requests.post(BASE_URL + path, data=data, headers=headers, timeout=30)
    resp.raise_for_status()
    out = resp.json()
    if out.get("error"):
        raise RuntimeError(f"Kraken API error: {out['error']}")
    return out.get("result", {})


def get_ohlc(since: int | None = None) -> list:
    """
    Public: fetch daily OHLC for XBTUSD. Returns list of [ts_ms, o, h, l, c, vwap, volume, count].
    Oldest first. Same shape as expected by btc_omega6_engine / fetch_btc_data.
    """
    params = {"pair": PAIR, "interval": INTERVAL_1D}
    if since is not None:
        params["since"] = since
    resp = requests.get(BASE_URL + OHLC_PATH, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if data.get("error"):
        raise RuntimeError(f"Kraken API error: {data['error']}")
    result = data.get("result", {})
    ohlc_key = next((k for k in result if k != "last"), None)
    if not ohlc_key:
        return []
    candles = result[ohlc_key]
    out = []
    for c in candles:
        ts_sec = int(float(c[0]))
        ts_ms = ts_sec * 1000
        o, h, l, close = float(c[1]), float(c[2]), float(c[3]), float(c[4])
        vwap = float(c[5]) if len(c) > 5 else 0.0
        vol = float(c[6]) if len(c) > 6 else 0.0
        cnt = int(c[7]) if len(c) > 7 else 0
        out.append([ts_ms, o, h, l, close, vwap, vol, cnt])
    out.sort(key=lambda x: x[0])
    return out


def get_balance() -> dict:
    """
    Private: get cash balances. Returns dict with keys "USD" and "BTC" (float).
    Kraken returns e.g. ZUSD, XXBT; we normalize to USD, BTC.
    """
    raw = _private_request(BALANCE_PATH)
    # Kraken uses ZUSD, XXBT (or XBT); map to USD, BTC
    usd = 0.0
    btc = 0.0
    for k, v in raw.items():
        try:
            bal = float(v)
        except (TypeError, ValueError):
            continue
        if k in ("ZUSD", "USD"):
            usd += bal
        elif k in ("XXBT", "XBT"):
            btc += bal
    return {"USD": usd, "BTC": btc}


def add_order(
    side: str,
    volume_btc: float,
    price: float | None = None,
    order_type: str = "market",
) -> dict:
    """
    Private: place order. side = "buy" | "sell", volume_btc in BTC.
    If price is None or order_type is "market", places market order; else limit at price.
    Returns Kraken result (e.g. txid list, desc).
    """
    data = {
        "pair": PAIR,
        "type": side,
        "volume": str(volume_btc),
        "ordertype": "limit" if price is not None and order_type == "limit" else "market",
    }
    if price is not None and order_type == "limit":
        data["price"] = str(price)
    return _private_request(ADD_ORDER_PATH, data)


def cancel_order(order_id: str) -> dict:
    """Private: cancel an open order by ID."""
    return _private_request(CANCEL_ORDER_PATH, {"txid": order_id})


def open_orders() -> dict:
    """Private: list open orders. Returns Kraken open orders result."""
    return _private_request(OPEN_ORDERS_PATH)
