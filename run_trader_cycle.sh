#!/usr/bin/env bash
# One cycle of the Coinbase trader. Load .env from project dir, then run.
# Used by launchd for 24/7 autonomous trading (runs this script every N minutes).

set -e
cd "$(dirname "$0")"

# Load .env if present (export all vars for Python)
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

# Defaults if not set in .env
export BTCOmega_MODE="${BTCOmega_MODE:-paper}"
# Always append to trader.log so the dashboard shows updates (manual or launchd)
export BTCOmega_LOG_PATH="${BTCOmega_LOG_PATH:-$PWD/trader.log}"

# Use same Python that has project deps (edit if yours is elsewhere)
PYTHON="${BTCOmega_PYTHON:-/opt/homebrew/bin/python3}"
exec "$PYTHON" run_coinbase_trader.py
