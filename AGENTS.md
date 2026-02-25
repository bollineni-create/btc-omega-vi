# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Single-file Python quantitative backtesting engine (`btc_omega6_engine.py`) for BTC/USD. No web server, no database — runs as a CLI script producing a 12-sheet Excel report.

### Running the engine

```bash
python3 btc_omega6_engine.py
```

See `README.md` for full details.

### Runtime prerequisites (non-obvious)

The engine reads data from hard-coded paths that must exist before running:

1. **`/tmp/btc_omega2_data.json`** — BTC OHLCV candle data from Kraken. Must contain a JSON object with key `"1d_kraken"` (or `"365"`) mapping to an array of candles in format `[timestamp_ms, open, high, low, close, vwap, volume, count]` with **numeric** (not string) values. Fetch from Kraken's public OHLC API (`pair=XBTUSD&interval=1440`) and convert string values to float/int before saving.
2. **`/mnt/user-data/outputs/`** — output directory for the Excel report. Create with `sudo mkdir -p /mnt/user-data/outputs/ && sudo chmod 777 /mnt/user-data/outputs/`.
3. **`/home/claude/btc_bot/omega6.py`** — the engine copies itself to the output directory from this path (line 1672). Create with `sudo mkdir -p /home/claude/btc_bot/ && sudo cp btc_omega6_engine.py /home/claude/btc_bot/omega6.py`.

### Linting

No linter is configured in the project. Use `python3 -m pyflakes btc_omega6_engine.py` for basic lint checks or `python3 -m py_compile btc_omega6_engine.py` to verify syntax.

### Testing

No automated test suite exists. The engine itself is the test — a successful end-to-end run (~25 seconds) producing `btc_omega6_report.xlsx` validates the code.
