"""
EveryCryptoBot — Master Launcher
Starts: multi_bot.py (all alt-coin bots) + dashboard server (browser panel)

Usage:
  python run.py              # Start everything
  python run.py --no-browser # Start without opening the browser
"""
import argparse
import subprocess
import sys
import threading
import time
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = os.path.dirname(os.path.abspath(__file__))


def start_dashboard(auto_open: bool):
    from dashboard_server import start_dashboard
    start_dashboard(auto_open=auto_open)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the browser")
    parser.add_argument("--symbol", default=None, help="Only run one asset bot (e.g. BTC/USD, SOL/USD). Omit for all 11.")
    args = parser.parse_args()

    print("=" * 60)
    print("  EveryCryptoBot v3.0 — Autonomous Paper Trading Engine")
    print("=" * 60)
    print("  🚀  Monitored   : 11 Crypto Assets (BTC, ETH, SOL, DOGE, LINK, AAVE, AVAX, LTC, XRP, UNI, MATIC)")
    print("  🧠  AI Engine   : Walk-Forward ML + Mistake Memory Active")
    print("  🛡️  Risk Mgt    : Dynamic Fractional Sizing + Trailing ATR Stops")
    port = os.getenv("PORT", "8765")
    print(f"  🌐  Dashboard   : http://0.0.0.0:{port}")
    print("  📱  Telegram    : Real-Time Trade Fills & Cycle Telemetry")
    print("=" * 60 + "\n")

    # Start dashboard server in background thread
    dash_thread = threading.Thread(
        target=start_dashboard,
        args=(not args.no_browser,),
        daemon=True,
        name="DashboardServer"
    )
    dash_thread.start()
    time.sleep(1)

    # Start bot engines in the same process (saves ~250MB RAM on Render 512MB free tier)
    target_sym = getattr(args, "symbol", None)
    try:
        from multi_bot import start_all_bots
        start_all_bots(symbol=target_sym)
    except Exception as e:
        print(f"❌ Error starting bot supervisor from run.py: {e}")


    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")


if __name__ == "__main__":
    main()
