#!/usr/bin/env python3
"""
Coinbase API — prices, market data, and buy/sell order helpers.

- Prices & candles: public, no API key.
- Orders: require API keys (Advanced Trade); use for learning and integration.

Refs:
  - Prices: https://docs.cloud.coinbase.com/sign-in-with-coinbase/docs/api-prices
  - Candles: https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles
  - Orders:  https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/create-order
"""

import os
import time
import uuid
import requests

# ─── Base URLs ─────────────────────────────────────────────────────────────
PRICES_BASE = "https://api.coinbase.com/v2/prices"
EXCHANGE_BASE = "https://api.exchange.coinbase.com"
ADVANCED_TRADE_BASE = "https://api.coinbase.com"

# Candles: allowed granularities (seconds) → 60, 300, 900, 3600, 21600, 86400
CANDLE_GRANULARITIES = {60, 300, 900, 3600, 21600, 86400}
MAX_CANDLES_PER_REQUEST = 300


# ══════════════════════════════════════════════════════════════════════════════
# PRICES (public, no auth)
# ══════════════════════════════════════════════════════════════════════════════

def get_spot_price(currency_pair: str = "BTC-USD") -> dict:
    """
    Current spot price for a pair (e.g. BTC-USD). No auth.
    Returns {"amount": str, "currency": str} or raises on error.
    """
    r = requests.get(f"{PRICES_BASE}/{currency_pair}/spot", timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def get_buy_price(currency_pair: str = "BTC-USD") -> dict:
    """
    Current buy price (includes ~1% fee). No auth.
    Returns {"amount": str, "currency": str}.
    """
    r = requests.get(f"{PRICES_BASE}/{currency_pair}/buy", timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def get_sell_price(currency_pair: str = "BTC-USD") -> dict:
    """
    Current sell price (includes ~1% fee). No auth.
    Returns {"amount": str, "currency": str}.
    """
    r = requests.get(f"{PRICES_BASE}/{currency_pair}/sell", timeout=10)
    r.raise_for_status()
    return r.json()["data"]


def get_all_prices(currency_pair: str = "BTC-USD") -> dict:
    """Fetch spot, buy, and sell in one go. No auth."""
    return {
        "spot": get_spot_price(currency_pair),
        "buy": get_buy_price(currency_pair),
        "sell": get_sell_price(currency_pair),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CANDLES / OHLCV (public, no auth — Exchange API)
# ══════════════════════════════════════════════════════════════════════════════

def get_candles(
    product_id: str = "BTC-USD",
    granularity: int = 86400,
    start: int | None = None,
    end: int | None = None,
) -> list[list]:
    """
    Historic OHLCV candles from Coinbase Exchange (public).

    product_id: e.g. "BTC-USD"
    granularity: seconds per candle — 60, 300, 900, 3600, 21600, 86400 (max 300 candles/request)
    start, end: Unix timestamps (optional). If omitted, returns last 300 candles to now.

    Returns list of [timestamp, low, high, open, close, volume] (same order as Exchange API).
    To match btc_omega6_engine expected format [ts, open, high, low, close] you can map:
      engine_candle = [c[0], c[3], c[2], c[1], c[4]]  # ts, o, h, l, c
    """
    if granularity not in CANDLE_GRANULARITIES:
        raise ValueError(f"granularity must be one of {sorted(CANDLE_GRANULARITIES)}")
    url = f"{EXCHANGE_BASE}/products/{product_id}/candles"
    params = {"granularity": granularity}
    if start is not None:
        params["start"] = start
    if end is not None:
        params["end"] = end
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    # API returns [[time, low, high, open, close, volume], ...] descending by time
    return r.json()


def get_candles_for_engine(
    product_id: str = "BTC-USD",
    granularity: int = 86400,
    num_candles: int = 300,
) -> list[list]:
    """
    Fetch candles and convert to format expected by btc_omega6_engine:
    [timestamp_ms, open, high, low, close] (no volume in engine's legacy format).
    Returns ascending by time, at most num_candles (capped at 300).
    """
    num_candles = min(max(1, num_candles), MAX_CANDLES_PER_REQUEST)
    end = int(time.time())
    start = end - num_candles * granularity
    raw = get_candles(product_id=product_id, granularity=granularity, start=start, end=end)
    # raw: [time, low, high, open, close, volume] descending
    out = []
    for c in reversed(raw[-num_candles:]):
        ts, low, high, open_, close, vol = c
        # engine style: [timestamp_ms, open, high, low, close]
        out.append([int(ts) * 1000, float(open_), float(high), float(low), float(close)])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# BUY / SELL — Order payload builders (for learning and use with authenticated client)
# ══════════════════════════════════════════════════════════════════════════════

def build_market_order(
    side: str,
    product_id: str = "BTC-USD",
    quote_size: str | None = None,
    base_size: str | None = None,
    client_order_id: str | None = None,
) -> dict:
    """
    Build a market order payload for Advanced Trade API (create order).

    side: "BUY" or "SELL"
    quote_size: amount in quote currency (e.g. "10.00" USD)
    base_size: amount in base currency (e.g. "0.001" BTC)
    Provide exactly one of quote_size or base_size.

    Returns body for POST https://api.coinbase.com/api/v3/brokerage/orders
    (requires JWT or API key + signature).
    """
    if (quote_size is None) == (base_size is None):
        raise ValueError("Provide exactly one of quote_size or base_size")
    cfg = {}
    if quote_size is not None:
        cfg["quote_size"] = str(quote_size)
    if base_size is not None:
        cfg["base_size"] = str(base_size)
    return {
        "client_order_id": client_order_id or str(uuid.uuid4()),
        "product_id": product_id,
        "side": side.upper(),
        "order_configuration": {
            "market_market_ioc": cfg,
        },
    }


def build_limit_order(
    side: str,
    limit_price: str,
    product_id: str = "BTC-USD",
    quote_size: str | None = None,
    base_size: str | None = None,
    client_order_id: str | None = None,
) -> dict:
    """
    Build a limit (GTC) order payload for Advanced Trade API.

    side: "BUY" or "SELL"
    limit_price: limit price string (e.g. "95000.00")
    quote_size or base_size: size in quote or base (exactly one).
    """
    if (quote_size is None) == (base_size is None):
        raise ValueError("Provide exactly one of quote_size or base_size")
    cfg = {"limit_price": str(limit_price)}
    if quote_size is not None:
        cfg["quote_size"] = str(quote_size)
    if base_size is not None:
        cfg["base_size"] = str(base_size)
    return {
        "client_order_id": client_order_id or str(uuid.uuid4()),
        "product_id": product_id,
        "side": side.upper(),
        "order_configuration": {
            "limit_limit_gtc": cfg,
        },
    }


def _get_client(api_key: str | None, api_secret: str | None):
    """Import and return RESTClient; uses COINBASE_API_KEY / COINBASE_API_SECRET if not passed."""
    try:
        from coinbase.rest import RESTClient
    except ImportError as e:
        raise RuntimeError(
            "Order submission requires the Coinbase SDK. Install it with:\n"
            "  pip install coinbase-advanced-py"
        ) from e
    key = api_key or os.environ.get("COINBASE_API_KEY")
    secret = api_secret or os.environ.get("COINBASE_API_SECRET")
    if not key or not secret:
        raise RuntimeError(
            "Coinbase orders require authentication. Set COINBASE_API_KEY and "
            "COINBASE_API_SECRET (CDP API key name and EC private key PEM). "
            "See https://docs.cdp.coinbase.com/advanced-trade/docs/getting-started"
        )
    # .env often stores PEM with literal \n; SDK needs real newlines
    if isinstance(secret, str) and "\\n" in secret:
        secret = secret.replace("\\n", "\n")
    return RESTClient(api_key=key, api_secret=secret)


def get_balance(
    api_key: str | None = None,
    api_secret: str | None = None,
) -> dict:
    """
    Get USD and BTC (and USDC as USD) balances from Coinbase Advanced Trade.
    Returns dict with keys "USD" and "BTC" (float), same shape as Kraken get_balance.
    Requires COINBASE_API_KEY and COINBASE_API_SECRET.
    """
    client = _get_client(api_key, api_secret)
    out = {"USD": 0.0, "BTC": 0.0}
    cursor = None
    for _ in range(20):  # max 20 pages
        try:
            resp = client.get_accounts(limit=250, cursor=cursor) if cursor else client.get_accounts(limit=250)
        except TypeError:
            resp = client.get_accounts()
        if hasattr(resp, "to_dict"):
            data = resp.to_dict()
        else:
            data = dict(resp) if resp is not None else {}
        accounts = data.get("accounts") or []
        for acc in accounts:
            currency = (acc.get("currency") or "").upper()
            bal = acc.get("available_balance") or acc.get("balance") or {}
            if hasattr(bal, "value"):
                val = float(getattr(bal, "value", 0) or 0)
            elif isinstance(bal, dict):
                val = float((bal.get("value") or 0) or 0)
            else:
                val = float(bal) if bal else 0.0
            if currency == "USD":
                out["USD"] += val
            elif currency == "USDC":
                out["USD"] += val
            elif currency == "BTC":
                out["BTC"] += val
        has_next = data.get("has_next", False)
        cursor = data.get("cursor") or ""
        if not has_next or not cursor:
            break
    return out


def create_order(
    order_payload: dict,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> dict:
    """
    Submit an order to Coinbase Advanced Trade API using the official SDK.

    Requires: pip install coinbase-advanced-py and COINBASE_API_KEY + COINBASE_API_SECRET
    (or pass api_key/api_secret). For CDP keys, see https://docs.cdp.coinbase.com/advanced-trade/docs/getting-started

    order_payload: from build_market_order() or build_limit_order().

    Returns the API response as a dict (success, order_id, etc.). Raises on auth or API errors.
    """
    client = _get_client(api_key, api_secret)
    payload = order_payload
    resp = client.create_order(
        client_order_id=payload["client_order_id"],
        product_id=payload["product_id"],
        side=payload["side"],
        order_configuration=payload["order_configuration"],
    )
    # SDK returns a response object; convert to dict for consistent interface
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    return dict(resp) if resp is not None else {}


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Print prices, candles, sample payloads; optionally place an order if --buy/--sell passed."""
    import argparse
    p = argparse.ArgumentParser(description="Coinbase API: prices, candles, and buy/sell orders")
    p.add_argument("--buy", type=str, metavar="QUOTE_SIZE", help="Place market BUY for this much USD (e.g. 10)")
    p.add_argument("--sell", type=str, metavar="BASE_SIZE", help="Place market SELL for this much BTC (e.g. 0.0001)")
    p.add_argument("--product", type=str, default="BTC-USD", help="Product ID (default: BTC-USD)")
    p.add_argument("--dry-run", action="store_true", help="With --buy/--sell, only print payload, do not submit")
    args = p.parse_args()

    # Optional: place one order
    if args.buy or args.sell:
        side = "BUY" if args.buy else "SELL"
        if args.buy:
            payload = build_market_order(side, product_id=args.product, quote_size=args.buy)
        else:
            payload = build_market_order(side, product_id=args.product, base_size=args.sell)
        if args.dry_run:
            print("Order payload (not sent):")
            for k, v in payload.items():
                print(f"  {k}: {v}")
            return
        try:
            result = create_order(payload)
            print("Order response:", result)
        except Exception as e:
            print("Order failed:", e)
        return

    # Default: demo prices + candles + sample payloads
    print("=== Coinbase API — Prices (public) ===\n")
    try:
        prices = get_all_prices("BTC-USD")
        for k, v in prices.items():
            print(f"  {k}: {v['amount']} {v['currency']}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== Last 3 daily candles (BTC-USD) ===\n")
    try:
        candles = get_candles("BTC-USD", granularity=86400)
        for c in candles[:3]:
            print(f"  ts={c[0]}  O={c[3]}  H={c[2]}  L={c[1]}  C={c[4]}  V={c[5]}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== Sample market BUY order (payload only; not sent) ===\n")
    try:
        payload = build_market_order("BUY", product_id="BTC-USD", quote_size="10.00")
        for k, v in payload.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n=== Sample limit SELL order (payload only) ===\n")
    try:
        payload = build_limit_order("SELL", limit_price="100000.00", base_size="0.0001")
        for k, v in payload.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\nTo place a real order, set COINBASE_API_KEY and COINBASE_API_SECRET, then run:")
    print("  python coinbase_api.py --buy 10          # market buy $10 of BTC")
    print("  python coinbase_api.py --sell 0.0001    # market sell 0.0001 BTC")
    print("  python coinbase_api.py --buy 10 --dry-run   # show payload only")


if __name__ == "__main__":
    main()
