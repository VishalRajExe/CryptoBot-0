# EveryCryptoBot 🚀

An autonomous, multi-asset algorithmic cryptocurrency trading and paper-trading engine built for the **Alpaca Trading API** with real-time **Telegram bot alerts**, interactive **web dashboard**, **crypto log inspector**, and a modular **backtesting engine**.

---

## 📌 Architecture & Supported Assets

EveryCryptoBot runs dedicated algorithmic strategies tailored to each asset's volatility and liquidity characteristics:

| Asset | Timeframe | Strategy | Characteristics | Risk/Reward |
| :--- | :--- | :--- | :--- | :--- |
| **BTC/USD** | 5-min | **Triple Confluence** | EMA 50/200 + MACD momentum + RSI exhaustion filter | Dynamic ATR Stop, 1:2 R:R |
| **SOL/USD** | 1-min | **ScalpMacdRsi** | High-frequency momentum scalping with EMA 200 trend anchor | 1.0% SL, 1.5% TP |
| **DOGE/USD** | 1-min | **ScalpMacdRsi** | Rapid volatility scalping on meme momentum surges | 1.0% SL, 1.5% TP |
| **LINK/USD** | 15-min | **VwapBreakout** | Institutional anchored VWAP breakout with volume surge | 1.5% SL, 3.0% TP |
| **ETH/USD** | 5-min | **MomentumEmaRibbon** | Trend-riding across 8/13/21/55 EMA ribbon expansion | 2x ATR trailing stop |
| **AVAX/USD** | 5-min | **MomentumEmaRibbon** | Intraday trend breakout & EMA expansion | 2x ATR trailing stop |

---

## 🧠 SQLite Trade Memory & Reboot Resilience

EveryCryptoBot is equipped with an **ACID-compliant SQLite trade database** (`trading_memory.db`) using **WAL (Write-Ahead Logging)** mode:

- **Reboot & Crash Survival**: Positions and trade telemetry persist across bot restarts, computer reboots, or network disconnects.
- **Ground-Truth Alpaca Reconciliation**: When the bot boots up, it automatically queries Alpaca's API ground-truth. If an open position exists, it **immediately adopts and reconciles** it—preventing duplicate buying or orphaned positions.
- **Detailed Trade Telemetry**: Stores full indicator states at entry (RSI, MACD, ATR, volume, confidence) alongside exit triggers and outcomes for retrospective analysis.

---

## 🤖 Adaptive Self-Learning & Circuit Breakers

The bot features active feedback loops to learn from past trade mistakes:

1. **Machine Learning Signal Forecasting (`learner.py`)**:
   - Uses `scikit-learn` (RandomForest classifier) pre-trained on 570+ historical trade samples.
   - Evaluates prospective BUY setups against past winning/losing indicator clusters.
   - **AI Veto**: Vetoes or warns on setups where predicted win probability is below 40%.
2. **Consecutive Loss Circuit Breaker**:
   - If an asset experiences 2 consecutive losses, the bot triggers an automated **20-minute cool-down freeze** to avoid choppy regime traps.
3. **Adaptive Stake Scaling**:
   - Dynamically scales position sizing (0.5x to 1.2x) based on the asset's rolling win rate and recent performance.

---

## 🖥️ Live Dashboard & Real-Time Log Viewer

EveryCryptoBot features a dark-themed control center running at `http://localhost:8765`:

- **Interactive Crypto Log Inspector**: Filter and view live terminal logs by crypto symbol (`[ALL]` `[BTC]` `[ETH]` `[SOL]` `[DOGE]` `[LINK]` `[AVAX]`) in real-time with auto-scroll and color-coded signals.
- **Bitcoin Master Banner**: Live position monitoring, entry price, trailing stops, and realized/unrealized PnL.
- **Active Asset Grid**: Real-time prices, live technical indicators (RSI, MACD, ATR, VWAP), confidence meters, and buy/sell/hold badges.
- **Backtest Benchmarks Panel**: Direct side-by-side comparison between backtested win rates / profit factors and live performance.
- **Execution History Table**: Chronological journal of fills with entry, exit, PnL ($ and %), and exit triggers.

---

## 📱 Telegram Notifications

Integrated Telegram alerts with HTML formatting:
- **🟢 Buy Orders**: Asset, quantity, entry price, stop-loss, and take-profit targets.
- **🎯 Take-Profit / 🛑 Stop-Loss**: Exit price, realized PnL ($ and %), and updated session total.
- **📊 Cycle Telemetry**: Polled technical indicator status, trend conditions, and unrealized positions.

---

## 🧪 Backtesting Engine

Run historical simulations against Alpaca historical market data:

```bash
# Backtest SOL/USD 1-minute scalping over 7 days
python backtest.py --symbol SOL/USD --strategy ScalpMacdRsi --days 7 --stake 300

# Backtest ETH/USD 5-minute ribbon over 14 days
python backtest.py --symbol ETH/USD --strategy MomentumEmaRibbon --days 14 --stake 500

# Backtest LINK/USD 15-minute VWAP breakout over 14 days
python backtest.py --symbol LINK/USD --strategy VwapBreakout --days 14 --stake 300
```

Results are saved to `backtests/<symbol>_<strategy>_<days>d.json` with Win Rate, Profit Factor, Sharpe Ratio, and Drawdowns.

---

## 🚀 Quick Start

### 1. Requirements
- Python 3.9+
- Alpaca API Paper or Live credentials

```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Copy `.env.example` to `.env` and fill in your keys:
```ini
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true

TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ENABLED=true
```

### 3. Launch System
```bash
# Start all altcoin bots and live dashboard:
python run.py

# Run headless (no browser auto-open):
python run.py --no-browser

# Run a single asset bot:
python multi_bot.py --symbol SOL/USD
```

---

## 📂 Project Structure

```
EveryCryptoBot/
├── backtest.py           # Historical backtesting engine
├── backtests/            # Backtest benchmark reports (.json)
├── dashboard/
│   ├── index.html        # Real-time web control center
│   └── live_state.json   # Live telemetry state feed
├── dashboard_server.py   # HTTP API & log streaming server
├── multi_bot.py          # Multi-asset threaded trading engine
├── run.py                # Master launcher
├── strategies/
│   ├── base.py           # BaseStrategy abstract interface
│   ├── momentum_ema_ribbon.py
│   ├── scalp_macd_rsi.py
│   └── vwap_breakout.py
├── requirements.txt      # Python dependencies
├── .env.example          # Environment template
└── README.md             # Documentation
```
