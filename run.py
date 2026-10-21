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
    parser.add_argument("--symbol", default=None, help="Only run one asset bot (e.g. SOL/USD)")
    args = parser.parse_args()

    print("=" * 60)
    print("  EveryCryptoBot — Multi-Asset Paper Trading System")
    print("=" * 60)
    print("  📊  BTC bot     : Running separately (alpaca_btc_bot.py)")
    print("  🚀  Alt bots    : ETH / SOL / DOGE / LINK / AVAX")
    print("  🌐  Dashboard   : http://localhost:8765")
    print("  📱  Telegram    : @UnboundCycle12Bot")
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

    # Start multi_bot
    cmd = [sys.executable, os.path.join(BASE, "multi_bot.py")]
    if args.symbol:
        cmd += ["--symbol", args.symbol]

    print(f"🚀  Launching multi_bot: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")


if __name__ == "__main__":
    main()
