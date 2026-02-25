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

## 3. Run the engine
```bash
python btc_omega6_engine.py
```

The script will:
- Fetch 365 days of live BTC/USD daily candles from Kraken's public API (no API key needed)
- Synthesise intraday bars using Merton jump-diffusion
- Run the optimiser (~30 seconds)
- Run 6-split walk-forward validation
- Run 10,000-path Monte Carlo bootstrap
- Export `btc_omega6_report.xlsx` to the same directory

## 4. Output files
- `btc_omega6_report.xlsx` — 12-sheet Excel report (equity curve, capital ladder, Monte Carlo, etc.)

## Notes
- Requires internet connection for the Kraken data fetch
- If Kraken is unreachable it falls back to synthesised legacy data
- Total runtime: ~60–90 seconds on a modern machine
- No API keys or accounts needed
