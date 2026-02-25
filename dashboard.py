#!/usr/bin/env python3
"""
Simple web dashboard: view trader "thinking" (log tail), current state, and order history.
Run: python dashboard.py  then open http://127.0.0.1:5000
"""

import json
import os
from flask import Flask, jsonify, render_template_string, request

APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.environ.get("BTCOmega_LOG_PATH", os.path.join(APP_DIR, "trader.log"))
PAPER_STATE_PATH = os.environ.get("BTCOmega_PAPER_STATE", os.path.join(APP_DIR, "paper_state.json"))
TRADES_PATH = os.environ.get("BTCOmega_TRADES_JSON", os.path.join(APP_DIR, "trades.json"))

app = Flask(__name__)


def _tail(path: str, n: int = 200) -> list[str]:
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
        return lines[-n:] if len(lines) > n else lines
    except Exception:
        return []


@app.route("/api/log")
def api_log():
    n = min(500, max(50, int(request.args.get("n", 200))))
    lines = _tail(LOG_PATH, n)
    return jsonify({"lines": lines, "path": LOG_PATH})


@app.route("/api/state")
def api_state():
    if not os.path.isfile(PAPER_STATE_PATH):
        return jsonify({"balance_usd": 0, "position_btc": 0, "last_price": 0})
    try:
        with open(PAPER_STATE_PATH) as f:
            data = json.load(f)
        return jsonify(data)
    except Exception:
        return jsonify({"balance_usd": 0, "position_btc": 0, "last_price": 0})


@app.route("/api/trades")
def api_trades():
    if not os.path.isfile(TRADES_PATH):
        return jsonify([])
    try:
        with open(TRADES_PATH) as f:
            data = json.load(f)
        return jsonify(data if isinstance(data, list) else [])
    except Exception:
        return jsonify([])


@app.route("/favicon.ico")
def favicon():
    return "", 204  # No content — stops browser from logging 404


@app.route("/")
def index():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC Omega VI — Trader Dashboard</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: ui-monospace, monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 12px; }
    h1 { font-size: 1.25rem; color: #58a6ff; margin: 0 0 12px 0; }
    h2 { font-size: 1rem; color: #8b949e; margin: 16px 0 8px 0; }
    section { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-bottom: 12px; }
    .log { white-space: pre-wrap; word-break: break-all; font-size: 12px; max-height: 320px; overflow-y: auto; }
    .log line { display: block; border-bottom: 1px solid #21262d; padding: 2px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #21262d; }
    th { color: #8b949e; }
    .buy { color: #3fb950; }
    .sell { color: #f85149; }
    .paper { color: #d29922; }
    .live { color: #58a6ff; }
    .state-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
    .state-item { background: #21262d; padding: 10px; border-radius: 4px; }
    .state-item span { color: #8b949e; font-size: 11px; }
    .state-item strong { display: block; font-size: 18px; color: #58a6ff; }
    .meta { font-size: 11px; color: #6e7681; margin-top: 8px; }
    button { background: #238636; color: #fff; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 12px; }
    button:hover { background: #2ea043; }
  </style>
</head>
<body>
  <h1>₿ BTC Omega VI — Trader Dashboard</h1>
  <p class="meta">Auto-refresh every 15s · Log and orders from this machine</p>

  <section>
    <h2>Current state (paper ledger)</h2>
    <div class="state-grid" id="state">—</div>
  </section>

  <section>
    <h2>Order history</h2>
    <div id="trades-wrap"><table><thead><tr><th>Time (UTC)</th><th>Side</th><th>BTC</th><th>USD</th><th>Price</th><th>Mode</th></tr></thead><tbody id="trades"></tbody></table></div>
    <p class="meta" id="trades-meta">No trades yet.</p>
  </section>

  <section>
    <h2>Thinking (recent log)</h2>
    <button onclick="refresh()">Refresh now</button>
    <div class="log" id="log"></div>
  </section>

  <script>
    function esc(s) { return (s == null ? '' : s).toString().replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
    function refresh() {
      fetch('/api/state').then(r=>r.json()).then(d=>{
        document.getElementById('state').innerHTML =
          '<div class="state-item"><span>Balance USD</span><strong>$' + (d.balance_usd != null ? Number(d.balance_usd).toLocaleString('en-US',{minFractionDigits:2}) : '—') + '</strong></div>' +
          '<div class="state-item"><span>Position BTC</span><strong>' + (d.position_btc != null ? Number(d.position_btc).toFixed(6) : '—') + '</strong></div>' +
          '<div class="state-item"><span>Last price</span><strong>$' + (d.last_price != null ? Number(d.last_price).toLocaleString('en-US',{minFractionDigits:2}) : '—') + '</strong></div>';
      });
      fetch('/api/trades').then(r=>r.json()).then(arr=>{
        const tbody = document.getElementById('trades');
        document.getElementById('trades-meta').textContent = arr.length ? arr.length + ' trade(s)' : 'No trades yet.';
        if (!arr.length) { tbody.innerHTML = '<tr><td colspan="6">No orders yet.</td></tr>'; return; }
        tbody.innerHTML = arr.slice().reverse().slice(0, 50).map(t=>
          '<tr><td>'+esc(t.ts)+'</td><td class="'+t.side+'">'+esc(t.side)+'</td><td>'+esc(t.base_btc)+'</td><td>$'+esc(t.quote_usd)+'</td><td>$'+esc(t.price)+'</td><td class="'+esc(t.mode)+'">'+esc(t.mode)+'</td></tr>'
        ).join('');
      });
      fetch('/api/log?n=150').then(r=>r.json()).then(d=>{
        const el = document.getElementById('log');
        if (!d.lines || !d.lines.length) { el.textContent = 'No log file at ' + (d.path || 'trader.log'); return; }
        el.innerHTML = d.lines.map(l=>'<line>'+esc(l.replace(/\n$/,''))+'</line>').join('');
        el.scrollTop = el.scrollHeight;
      });
    }
    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>
""")


if __name__ == "__main__":
    port = int(os.environ.get("BTCOmega_DASHBOARD_PORT", "5000"))
    print(f"Dashboard: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
