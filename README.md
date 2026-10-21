# EveryCryptoBot 🚀

An automated algorithmic cryptocurrency trading bot built for **Alpaca Trading API** (Paper & Live) with real-time **Telegram notifications** and a proven **Triple Confluence Strategy**.

---

## 📌 Strategy Overview

The bot operates on a **Triple Confluence Strategy** designed to filter out market noise and trade high-probability momentum breakouts:

```
                  ┌──────────────────────────────┐
                  │       MARKET CANDLES         │
                  └──────────────┬───────────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   TREND FILTER   │   │ MOMENTUM TRIGGER │   │ EXHAUSTION FILTER│
│   EMA 50 > EMA 200│   │ MACD Crossover   │   │ RSI Sweet Spot   │
└────────┬─────────┘   └─────────┬────────┘   └─────────┬────────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
                        [ ALL 3 ALIGNED? ]
                                 │
                       YES ──────┴────── NO
                        │                 │
                        ▼                 ▼
                  [ EXECUTE BUY ]      [ HOLD ]
```

### 1. Buy Conditions (All 3 Must Align)
- **Trend Filter**: Price > EMA(50) > EMA(200) *(Confirmed Bullish Trend)*
- **Momentum Trigger**: MACD Line crosses above Signal Line *(Bullish Momentum Crossover)*
- **Exhaustion Filter**: RSI(14) between 50 and 70 *(Bullish control without being overbought)*

### 2. Sell / Exit Conditions
- **Dynamic Stop-Loss**: $1.5 \times \text{ATR}(14)$ below entry price *(adapts dynamically to volatility)*
- **Take-Profit Target**: $2.0 \times \text{Risk}$ *(positive 1:2 Risk-to-Reward ratio)*
- **Trend Invalidation**: Bearish confluence crossover triggers early protection

---

## 📱 Telegram Notifications

The bot sends formatted real-time updates directly to your Telegram chat:

- **🟢 Buy Alert**: Quantity, Entry Price, Stop-Loss, Take-Profit, and Risk/Reward ratio.
- **🎯 Take-Profit / 🛑 Stop-Loss Alert**: Exit Price, Realized PnL ($ and %), and cumulative session stats.
- **🟡 Cycle Report**: Polled market snapshot every cycle with Price, RSI, MACD, Trend, and live Unrealized PnL.

---

## ⚙️ Project Structure

```
EveryCryptoBot/
├── bot.py             # Main bot engine & indicator computations
├── requirements.txt   # Python dependencies
├── .env.example       # Template for environment credentials
├── .gitignore         # Git ignore rules (protects credentials and logs)
└── README.md          # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/VishalRajExe/EveryCryptoBot.git
cd EveryCryptoBot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Credentials
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
# Alpaca Trading API Credentials
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_PAPER=true

# Trading Settings
TRADING_SYMBOL=BTC/USD
STAKE_USD=5000.0
POLL_SECS=60
CANDLE_LIMIT=250

# Telegram Bot Notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
TELEGRAM_ENABLED=true
```

### 4. Run the Bot
```bash
python bot.py
```

---

## 🛡️ Risk Management

- **Dynamic Position Sizing**: Stakes a fixed dollar amount per trade (e.g. $5,000 USD).
- **Hard Volatility Bounds**: Dynamic ATR stop-loss ensures trades are cut before large pullbacks.
- **Persistent State**: Open positions, trades, and PnL are persisted to `alpaca_state.json` to safely survive process restarts.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
