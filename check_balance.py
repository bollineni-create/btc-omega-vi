#!/usr/bin/env python3
"""Load .env and print live Coinbase balance. Run from project dir: python3 check_balance.py"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass  # .env may already be in environment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coinbase_api import get_balance, get_spot_price

try:
    b = get_balance()
except Exception as e:
    print("Error fetching balance:", e)
    sys.exit(1)

spot = get_spot_price("BTC-USD")
price = float(spot.get("amount", 0))
usd = b.get("USD", 0)
btc = b.get("BTC", 0)
total = usd + btc * price

print("Live Coinbase balance:")
print("  USD:     ", usd)
print("  BTC:     ", btc)
print("  Spot:    $", price, " (BTC-USD)", sep="")
print("  Total:   $", round(total, 2), " (USD equiv)", sep="")
