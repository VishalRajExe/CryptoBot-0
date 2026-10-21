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

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")
key = os.environ.get("ALPACA_API_KEY")
sec = os.environ.get("ALPACA_SECRET_KEY")

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
    "🚀 <b>EVERYCRYPTOBOT — FRESH INITIALIZATION</b> 🚀\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "🏦 <b>Broker:</b> Alpaca Paper Trading v2\n"
    "💼 <b>Current Portfolio Status:</b>\n"
    f"  • <b>Cash Balance:</b> ${cash:,.2f}\n"
    f"  • <b>Total Equity:</b> ${equity:,.2f}\n"
    f"  • <b>Buying Power:</b> ${buying_power:,.2f}\n"
    f"  • <b>Open Positions:</b> {pos_count} (FLAT)\n"
    f"  • <b>Total Realized Profits:</b> $0.0000 (Fresh Session)\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "🎯 <b>SESSION TARGET:</b>\n"
    "  • <b>Profit Goal:</b> <b>+$50.00 in 2 Hours</b>\n"
    "  • <b>Target Account Value:</b> <b>$150.00</b>\n"
    "  • <b>Execution Mode:</b> ⚡ 1-Minute Turbo Scalper\n"
    "  • <b>Allocation:</b> 90% Dynamic Capital Deployment\n"
    "━━━━━━━━━━━━━━━━━━━━━\n"
    "🌐 <b>Live Dashboard:</b> http://localhost:8765\n"
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
