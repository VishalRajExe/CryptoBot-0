"""
Dashboard server for EveryCryptoBot.
Serves the web dashboard and real-time JSON API at http://localhost:8765

Endpoints:
  /                   -> Dashboard UI (dashboard/index.html)
  /api/live_state     -> Real-time telemetry for all altcoins
  /api/multi_state    -> Multi-coin trade history & session stats
  /api/btc_state      -> Live state of the separate Bitcoin bot
  /api/backtests      -> Summary of backtest performance reports
  /api/logs           -> Streaming logs per crypto (?symbol=BTC|ETH|SOL|DOGE|LINK|AVAX|ALL&lines=100)
"""
import glob
import http.server
import json
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PORT = 8765
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DASH_DIR = os.path.join(BASE_DIR, "dashboard")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
BACKTESTS_DIR = os.path.join(BASE_DIR, "backtests")
MULTI_STATE_FILE = os.path.join(BASE_DIR, "multi_state.json")
LIVE_STATE_FILE = os.path.join(DASH_DIR, "live_state.json")

# BTC Bot files (managed separately in user_data)
BTC_STATE_FILE = os.path.abspath(os.path.join(BASE_DIR, "..", "user_data", "alpaca_state.json"))
BTC_LOGS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "user_data", "logs"))


def _read_json_file(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _get_latest_file(pattern: str) -> str:
    files = glob.glob(pattern)
    if not files:
        return ""
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def _tail_file(filepath: str, max_lines: int = 100) -> list:
    if not filepath or not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            return [line.rstrip("\r\n") for line in lines[-max_lines:]]
    except Exception as e:
        return [f"[Error reading log: {e}]"]


def get_logs_for_symbol(symbol: str, max_lines: int = 100) -> dict:
    sym = symbol.strip().upper().replace("/", "")

    if sym in ("BTC", "BTCUSD"):
        latest = _get_latest_file(os.path.join(BTC_LOGS_DIR, "alpaca_bot_*.log"))
        lines = _tail_file(latest, max_lines)
        return {
            "symbol": "BTC/USD",
            "file": os.path.basename(latest) if latest else "none",
            "lines": lines,
            "count": len(lines),
        }

    if sym in ("ALL", ""):
        # Merge recent lines across BTC and all altcoins
        all_lines = []
        # BTC log
        btc_file = _get_latest_file(os.path.join(BTC_LOGS_DIR, "alpaca_bot_*.log"))
        if btc_file:
            for l in _tail_file(btc_file, max_lines // 2):
                all_lines.append(f"[BTC] {l}")
        # Altcoin logs
        for s in ["ETHUSD", "SOLUSD", "DOGEUSD", "LINKUSD", "AVAXUSD"]:
            s_file = _get_latest_file(os.path.join(LOGS_DIR, f"{s}_*.log"))
            if s_file:
                for l in _tail_file(s_file, max_lines // 4):
                    all_lines.append(l)

        # Sort roughly by timestamp if present
        all_lines.sort(key=lambda x: x[:30])
        return {
            "symbol": "ALL",
            "file": "aggregated",
            "lines": all_lines[-max_lines:],
            "count": len(all_lines[-max_lines:]),
        }

    # Specific altcoin (e.g. ETH, SOL, DOGE, LINK, AVAX)
    safe = sym.replace("USD", "") + "USD"
    latest = _get_latest_file(os.path.join(LOGS_DIR, f"{safe}_*.log"))
    lines = _tail_file(latest, max_lines)
    return {
        "symbol": f"{sym.replace('USD', '')}/USD",
        "file": os.path.basename(latest) if latest else "none",
        "lines": lines,
        "count": len(lines),
    }


def get_backtest_summaries() -> list:
    results = []
    if not os.path.exists(BACKTESTS_DIR):
        return results
    for fpath in glob.glob(os.path.join(BACKTESTS_DIR, "*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = {k: v for k, v in data.items() if k != "trades"}
                results.append(summary)
        except Exception:
            pass
    return results


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASH_DIR, **kwargs)

    def log_message(self, *args):
        pass  # Suppress routine request spam

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/api/live_state":
            data = _read_json_file(LIVE_STATE_FILE)
            return self._send_json(data)

        if path == "/api/multi_state":
            data = _read_json_file(MULTI_STATE_FILE)
            return self._send_json(data)

        if path == "/api/btc_state":
            data = _read_json_file(BTC_STATE_FILE)
            return self._send_json(data)

        if path == "/api/backtests":
            data = get_backtest_summaries()
            return self._send_json({"backtests": data})

        if path == "/api/logs":
            qs = urllib.parse.parse_qs(parsed.query)
            symbol = qs.get("symbol", ["ALL"])[0]
            try:
                lines = int(qs.get("lines", [80])[0])
            except ValueError:
                lines = 80
            data = get_logs_for_symbol(symbol, max_lines=lines)
            return self._send_json(data)

        # Fallback to static files in dashboard/ directory
        super().do_GET()


def start_dashboard(auto_open: bool = True):
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        httpd.allow_reuse_address = True
        url = f"http://localhost:{PORT}"
        print(f"\n============================================================")
        print(f"  Live Dashboard running at: {url}")
        print(f"  API endpoints active: /api/live_state, /api/logs, /api/btc_state")
        print(f"============================================================\n")
        if auto_open:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()
        httpd.serve_forever()


if __name__ == "__main__":
    start_dashboard()
