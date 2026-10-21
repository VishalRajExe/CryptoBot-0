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
    parser.add_argument("--one-bot", action="store_true", help="Run ONE bot (BTC/USD 1m Turbo Scalper with $50 goal)")
    parser.add_argument("--symbol", default=None, help="Only run one asset bot (e.g. SOL/USD)")
    args = parser.parse_args()

    print("=" * 60)
    print("  EveryCryptoBot — Autonomous Paper Trading System")
    print("=" * 60)
    if args.one_bot:
        print("  ⚡  Mode        : ONE BOT (BTC/USD 1m Turbo Scalper)")
        print("  🎯  Goal        : +$50.00 Profit in 2 Hours ($100 Starting Wallet)")
        print("  💰  Stake       : 90% Capital Allocation (~$90.00 per trade)")
    else:
        print("  📊  BTC bot     : alpaca_btc_bot.py")
        print("  🚀  Alt bots    : ETH / SOL / DOGE / LINK / AAVE / AVAX")
    port = os.getenv("PORT", "8765")
    print(f"  🌐  Dashboard   : http://0.0.0.0:{port}")
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

    # Start bot engines in the same process (saves ~250MB RAM on Render 512MB free tier)
    target_sym = "BTC/USD" if args.one_bot else args.symbol
    from multi_bot import start_all_bots
    start_all_bots(symbol=target_sym)

    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑  Stopped by user.")


if __name__ == "__main__":
    main()
