# BTC OMEGA VI — Local Setup & Run

## 1. Prerequisites
Python 3.9 or higher required.

## 2. Install dependencies
```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install numpy pandas scipy rich requests xlsxwriter
```

## 3. Data and backtest (two steps)

The engine reads OHLC data from a JSON file; it does not fetch from the exchange itself. Use the fetcher first, then run the engine.

### 3.1 Fetch BTC/USD daily candles (optional)
```bash
python fetch_btc_data.py
```
This calls Kraken’s **public** OHLC API (no API key needed) and writes `btc_omega2_data.json` in the format the engine expects. If you already have this file (e.g. from a previous run), you can skip this step.

### 3.2 Run the backtest engine
```bash
python btc_omega6_engine.py
```
The script will:
- Load daily candles from the JSON file (see **Paths** below)
- Synthesise intraday bars using Merton jump-diffusion
- Run the optimiser (~30 seconds)
- Run 6-split walk-forward validation
- Run 10,000-path Monte Carlo bootstrap
- Export the Excel report to the path set in **Paths**

## 4. Paths (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `BTCOmega_DATA_JSON` | `./btc_omega2_data.json` | Input: path to OHLC JSON (`"1d_kraken"` or `"365"` list of `[ts, o, h, l, c, ...]`, ≥200 candles). |
| `BTCOmega_REPORT_PATH` | `./btc_omega6_report.xlsx` | Output: path for the 12-sheet Excel report. |
| `BTCOmega_ENGINE_COPY_PATH` | (unset) | If set, the engine script is copied into this directory after the run (e.g. for archiving). |

Example:
```bash
export BTCOmega_DATA_JSON=./data/btc_omega2_data.json
export BTCOmega_REPORT_PATH=./outputs/btc_omega6_report.xlsx
python fetch_btc_data.py   # writes to ./data/...
python btc_omega6_engine.py
```

## 5. Output files
- **Report:** path from `BTCOmega_REPORT_PATH` (default: `btc_omega6_report.xlsx`) — 12 sheets: equity curve, capital ladder, Monte Carlo, etc.

## 6. Notes (backtest)
- The fetcher requires internet access to call Kraken’s public API.
- The engine needs at least 200 daily candles in the JSON; the fetcher provides enough for a full backtest.
- Total runtime: ~60–90 seconds on a modern machine.
- No API keys or accounts needed for backtest or data fetch.

---

## 7. Live / paper automated trading

A separate script runs the same Omega VI strategy on the latest data and either simulates orders (paper) or places real orders on Kraken (live).

### 7.1 Run the trader
```bash
python run_live_trader.py
```
- **Paper (default):** Uses `BTCOmega_PAPER_STATE` (default: `paper_state.json`) to track simulated balance and BTC position. No exchange API keys needed.
- **Live:** Set `BTCOmega_MODE=live` and `KRAKEN_API_KEY` / `KRAKEN_API_SECRET`. The script will place market orders on Kraken based on the signal.

The script will:
1. Ensure OHLC data exists (fetches from Kraken if the file is missing or older than 24h).
2. Build a daily OHLC series and compute one end-of-day signal (long/short/flat + size).
3. Compare desired position to current position (from paper state or Kraken balance).
4. Place one order if there is a meaningful delta (idempotent: running again the same day does not double-size).

### 7.2 Environment variables (trader)

| Variable | Default | Description |
|----------|---------|-------------|
| `BTCOmega_MODE` | `paper` | `paper` = simulate in `paper_state.json`; `live` = real Kraken orders. |
| `BTCOmega_PAPER_STATE` | `./paper_state.json` | Path to paper ledger (balance_usd, position_btc, last_price). |
| `BTCOmega_SL_PCT` | `2.0` | Stop-loss % used for signal (and reporting). |
| `BTCOmega_TP_PCT` | `4.0` | Take-profit % used for signal. |
| `BTCOmega_THRESHOLD` | `0.15` | Signal threshold. |
| `KRAKEN_API_KEY` | — | Kraken public API key (required only for `BTCOmega_MODE=live`). |
| `KRAKEN_API_SECRET` | — | Kraken private API secret (required only for live). **Never commit.** |

See [.env.example](.env.example) for a template. Copy to `.env` and set values as needed (load with `export $(cat .env | xargs)` or your preferred method).

### 7.3 Scheduling (cron)

Run the trader once per day after the daily candle is available (e.g. after Kraken’s daily close). Example: 01:00 UTC.

```bash
# Crontab: run daily at 01:00 UTC (adjust path and python)
0 1 * * * cd /path/to/btc-omega-vi && /usr/bin/env python3 run_live_trader.py >> /path/to/btc-omega-vi/trader.log 2>&1
```

### 7.4 Safety
- **Use paper first.** Run with `BTCOmega_MODE=paper` and verify behaviour before switching to `live`.
- **API keys:** Store only in environment or a local, gitignored file. Never commit `KRAKEN_API_SECRET`.
- **Idempotency:** The script only sends the delta between desired and current position; running it multiple times in a day does not double position size.

---

## 8. Coinbase API — prices and buy/sell (learning)

The module `coinbase_api.py` provides Coinbase data and order helpers (no keys needed for prices or candles).

### 8.1 What’s included
- **Prices (public):** spot, buy, and sell for any pair (e.g. `BTC-USD`).
- **Candles (public):** OHLCV from Coinbase Exchange; helper `get_candles_for_engine()` returns data in the same shape as the engine’s backtest input.
- **Buy/sell (learning):** payload builders for market and limit orders. Actual order submission requires API keys and the [Advanced Trade API](https://docs.cdp.coinbase.com/api-reference/advanced-trade-api/rest-api/orders/create-order).

### 8.2 Quick run (no keys)
```bash
python coinbase_api.py
```
Prints current BTC-USD prices, last few daily candles, and sample market/limit order payloads (no order is sent).

### 8.3 From your code
```python
from coinbase_api import get_all_prices, get_candles, get_candles_for_engine
from coinbase_api import build_market_order, build_limit_order

# Current prices (public)
prices = get_all_prices("BTC-USD")  # spot, buy, sell

# Daily candles for backtest-style use (public)
candles = get_candles_for_engine("BTC-USD", granularity=86400, num_candles=365)

# Build order payloads (submit with SDK or your own auth)
buy_payload  = build_market_order("BUY",  product_id="BTC-USD", quote_size="25.00")
sell_payload = build_limit_order("SELL", limit_price="98000.00", base_size="0.0005")
```

### 8.4 Placing real orders
Order submission is implemented via the official SDK (`coinbase-advanced-py`, listed in `requirements.txt`). Set `COINBASE_API_KEY` and `COINBASE_API_SECRET` (e.g. in `.env`; see `.env.example`). For CDP keys, see [getting started](https://docs.cdp.coinbase.com/advanced-trade/docs/getting-started).

**From code:** use `create_order(build_market_order(...))` or `create_order(build_limit_order(...))`; the SDK handles JWT and the POST.

**From CLI:**
```bash
python coinbase_api.py --buy 10              # market buy $10 of BTC (requires API keys)
python coinbase_api.py --sell 0.0001         # market sell 0.0001 BTC
python coinbase_api.py --buy 10 --dry-run    # print payload only, do not submit
```

---

## 9. Autonomous Coinbase day trading (recommended)

The **Coinbase-only** autonomous trader runs the same Omega VI strategy using your Coinbase account: it fetches BTC-USD data from Coinbase, computes the signal, and places market orders on Coinbase (or updates paper state).

### 9.1 Run the Coinbase trader

```bash
python run_coinbase_trader.py
```

- **Paper (default):** No API keys. Uses `paper_state.json` to track simulated USD and BTC. Safe to run repeatedly.
- **Live:** Set `BTCOmega_MODE=live` and `COINBASE_API_KEY` + `COINBASE_API_SECRET`. The script places real market orders on Coinbase.

**Autonomous loop (day trading):** run continuously and re-evaluate every N minutes:

```bash
RUN_INTERVAL_MINUTES=60 python run_coinbase_trader.py   # every hour
RUN_INTERVAL_MINUTES=240 BTCOmega_MODE=live python run_coinbase_trader.py   # live, every 4 hours
```

Use `Ctrl+C` to stop the loop.

### 9.2 Environment variables (Coinbase trader)

| Variable | Default | Description |
|----------|---------|-------------|
| `BTCOmega_MODE` | `paper` | `paper` = simulate; `live` = real Coinbase orders. |
| `BTCOmega_PAPER_STATE` | `./paper_state.json` | Path to paper ledger. |
| `BTCOmega_SL_PCT` | `2.0` | Stop-loss % for signal. |
| `BTCOmega_TP_PCT` | `4.0` | Take-profit % for signal. |
| `BTCOmega_THRESHOLD` | `0.15` | Signal threshold. |
| `BTCOmega_MAX_POSITION_PCT` | `0.15` | Cap position size (fraction of capital). |
| `BTCOmega_DRY_RUN` | — | Set to `1` or `true` to log orders without sending (live only). |
| `BTCOmega_LOG_PATH` | — | If set, append logs to this file (and still print to stdout). |
| `RUN_INTERVAL_MINUTES` | `0` | If > 0, run in a loop every N minutes (autonomous day trading). |
| `COINBASE_API_KEY` | — | Required for live; full key name `organizations/{org_id}/apiKeys/{key_id}` or key ID if `COINBASE_ORG_ID` is set. |
| `COINBASE_API_SECRET` | — | Required for live; EC private key PEM or base64 private key from CDP JSON. **Never commit.** |
| `COINBASE_ORG_ID` | — | If set and `COINBASE_API_KEY` is only the key ID (UUID), the full key name is built as `organizations/{COINBASE_ORG_ID}/apiKeys/{key_id}`. Get org ID from [CDP API keys](https://cloud.coinbase.com/access/api). |
| `COINBASE_CDP_KEY_JSON` | `cdp_api_key.json` | Path to CDP key JSON (fields `name` or `id`, and `privateKey`). If no key/secret in env, this file is used. |

**401 Unauthorized:** The Advanced Trade API expects the **full key name** in the JWT, not just the key ID. If you only have the key ID (e.g. from `cdp_api_key.json` with an `id` field), set `COINBASE_ORG_ID` to your organization ID (from the CDP portal when viewing the key), or set `COINBASE_API_KEY` to the full string `organizations/YOUR_ORG_ID/apiKeys/YOUR_KEY_ID`.

### 9.3 Safety

- **Paper first.** Run `python run_coinbase_trader.py` (paper) and confirm behaviour before enabling `BTCOmega_MODE=live`.
- **Dry-run.** With `BTCOmega_MODE=live` and `BTCOmega_DRY_RUN=1`, the script fetches balance and computes orders but only logs them; no orders are sent.
- **Position cap.** `BTCOmega_MAX_POSITION_PCT` limits how much of your capital can be in BTC (default 15%).
- **Idempotency.** Each run only trades the delta between desired and current position; multiple runs do not double the position.

### 9.4 Run 24/7 autonomously (macOS)

To have the trader run automatically every hour, survive reboots, and keep running without an open terminal:

1. **Set your environment in `.env`** (mode, Coinbase keys, etc.). The 24/7 job reads from `.env`.

2. **Install the LaunchAgent** (one time):
   ```bash
   cd /path/to/btc-omega-vi
   chmod +x install_24_7.sh run_trader_cycle.sh
   ./install_24_7.sh
   ```
   This copies `com.btcomega.trader.plist` to `~/Library/LaunchAgents/` and loads it. The job runs **once at load** and then **every 60 minutes**.

3. **Control the job:**
   - Stop: `launchctl unload ~/Library/LaunchAgents/com.btcomega.trader.plist`
   - Start: `launchctl load ~/Library/LaunchAgents/com.btcomega.trader.plist`
   - Log: `tail -f /path/to/btc-omega-vi/trader.log`

4. **Change how often it runs:** Edit `com.btcomega.trader.plist` in the project. `StartInterval` is in seconds (e.g. `3600` = 1 hour, `900` = 15 minutes). Then run `./install_24_7.sh` again to reload, or unload/load the plist manually.

### 9.5 Dashboard (thinking + orders)

A simple web UI shows the trader’s “thinking” (recent log), current paper state, and order history.

**Run the dashboard:**
```bash
cd /path/to/btc-omega-vi
pip install flask   # if not already installed
python dashboard.py
```
Then open **http://127.0.0.1:5000** in your browser. The page auto-refreshes every 15 seconds.

- **Current state:** balance_usd, position_btc, last_price (from `paper_state.json`).
- **Order history:** all executed trades (paper and live) from `trades.json`.
- **Thinking:** tail of `trader.log` (signal, capital, desired position, and “No trade” or “Paper/Live BUY/SELL” lines).

Optional: `BTCOmega_DASHBOARD_PORT=8080 python dashboard.py` to use port 8080.
