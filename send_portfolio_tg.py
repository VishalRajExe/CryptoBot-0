"""
Send Fresh Portfolio Notification to Telegram.
"""
import os
import urllib.parse
import urllib.request
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
key = os.environ.get("ALPACA_API_KEY", "")
sec = os.environ.get("ALPACA_SECRET_KEY", "")

if not key or not sec:
    print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY is not set.")
    exit(1)

headers = {
    "APCA-API-KEY-ID": key,
    "APCA-API-SECRET-KEY": sec
}

acc_req = urllib.request.Request("https://paper-api.alpaca.markets/v2/account", headers=headers)
with urllib.request.urlopen(acc_req) as resp:
    acc = json.loads(resp.read().decode())

pos_req = urllib.request.Request("https://paper-api.alpaca.markets/v2/positions", headers=headers)
with urllib.request.urlopen(pos_req) as resp:
    pos = json.loads(resp.read().decode())

cash = float(acc.get("cash", 100))
equity = float(acc.get("equity", 100))
buying_power = float(acc.get("buying_power", 100))
pos_count = len(pos)

msg = (
    "🚀 <b>EVERYCRYPTOBOT v3.0 — AUTONOMOUS PORTFOLIO</b> 🚀\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "🏦 <b>Broker:</b> Alpaca Paper Trading v2\n"
    "💼 <b>Current Portfolio Status:</b>\n"
    f"  • <b>Cash Balance:</b> ${cash:,.2f}\n"
    f"  • <b>Total Equity:</b> ${equity:,.2f}\n"
    f"  • <b>Buying Power:</b> ${buying_power:,.2f}\n"
    f"  • <b>Open Positions:</b> {pos_count}\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "⚙️ <b>ENGINE CONFIGURATION:</b>\n"
    "  • <b>Monitored Assets:</b> 11 Pairs (BTC, ETH, SOL, DOGE, LINK, AAVE, AVAX, LTC, XRP, UNI, MATIC)\n"
    "  • <b>Mode:</b> Autonomous Unconstrained Trading (No Artificial Goal)\n"
    "  • <b>AI Engine:</b> Walk-Forward Ensemble + Mistake Memory\n"
    "  • <b>Risk Mgt:</b> Dynamic Fractional Compounding + Trailing ATR Stops\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "🌐 <b>Live Dashboard:</b> https://everycryptobot.onrender.com\n"
    f"🕒 <i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
)

url = f"https://api.telegram.org/bot{token}/sendMessage"
data = urllib.parse.urlencode({
    "chat_id": chat_id,
    "text": msg,
    "parse_mode": "HTML"
}).encode("utf-8")

req = urllib.request.Request(url, data=data)
with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read().decode())
    print("Telegram Notification Status:", resp.status, res.get("ok"))
