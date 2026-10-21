"""
Backtesting engine for EveryCryptoBot strategies.
Usage:
    python backtest.py --symbol SOL/USD --strategy ScalpMacdRsi --days 30
    python backtest.py --symbol ETH/USD --strategy MomentumEmaRibbon --days 60
    python backtest.py --symbol LINK/USD --strategy VwapBreakout --days 30
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

# -- path fix so we can import strategies ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strategies import STRATEGY_REGISTRY

from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("ALPACA_API_KEY", "")
API_SECRET = os.getenv("ALPACA_SECRET_KEY", "")

TF_MAP = {
    1:  TimeFrame(1,  TimeFrameUnit.Minute),
    5:  TimeFrame(5,  TimeFrameUnit.Minute),
    15: TimeFrame(15, TimeFrameUnit.Minute),
    60: TimeFrame(1,  TimeFrameUnit.Hour),
}


def fetch_bars(symbol: str, tf_minutes: int, days: int) -> pd.DataFrame:
    client = CryptoHistoricalDataClient(API_KEY, API_SECRET)
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(days=days)
    tf     = TF_MAP.get(tf_minutes, TF_MAP[5])
    req    = CryptoBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
    bars   = client.get_crypto_bars(req)
    df     = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def run_backtest(symbol: str, strategy_name: str, days: int,
                 stake_usd: float = 25.0) -> dict:
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_REGISTRY.keys())}")

    strat = STRATEGY_REGISTRY[strategy_name](symbol=symbol)
    tf    = strat.timeframe_minutes()

    print(f"\n{'='*60}")
    print(f"  BACKTEST: {symbol}  |  {strategy_name}  |  {days} days  |  {tf}m bars")
    print(f"{'='*60}")
    print(f"  Fetching {days} days of {tf}-minute bars from Alpaca...")

    df = fetch_bars(symbol, tf, days)
    print(f"  Fetched {len(df)} bars  ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")

    if len(df) < strat.MIN_CANDLES + 10:
        print(f"  !!  Not enough data. Got {len(df)}, need {strat.MIN_CANDLES + 10}")
        return {}

    df = strat.add_indicators(df)

    # -- Simulation loop ------------------------------------------------------
    trades        = []
    position      = None  # dict when in a trade
    total_bars    = len(df)

    for i in range(strat.MIN_CANDLES, total_bars):
        window  = df.iloc[:i + 1]
        bar     = df.iloc[i]
        price   = float(bar["close"])

        # Check SL/TP on open position
        if position:
            pnl = 0.0
            reason = None
            if price <= position["sl"]:
                pnl    = (position["sl"] - position["entry"]) * position["qty"]
                reason = "stop_loss"
            elif price >= position["tp"]:
                pnl    = (position["tp"] - position["entry"]) * position["qty"]
                reason = "take_profit"

            if reason:
                trades.append({
                    "entry_time":  position["entry_time"],
                    "exit_time":   bar.name,
                    "entry_price": position["entry"],
                    "exit_price":  price,
                    "qty":         position["qty"],
                    "pnl":         round(pnl, 4),
                    "pnl_pct":     round(pnl / (position["entry"] * position["qty"]) * 100, 3),
                    "reason":      reason,
                    "win":         pnl > 0,
                })
                position = None
                continue

        # Generate new signal only when flat
        if not position:
            result = strat.get_signal(window, price)
            if result.signal == "BUY" and result.stop_loss > 0:
                qty = round(stake_usd / price, 6)
                position = {
                    "entry":      price,
                    "sl":         result.stop_loss,
                    "tp":         result.take_profit,
                    "qty":        qty,
                    "entry_time": bar.name,
                }

    # Close any open position at last bar price
    if position:
        last_price = float(df.iloc[-1]["close"])
        pnl = (last_price - position["entry"]) * position["qty"]
        trades.append({
            "entry_time":  position["entry_time"],
            "exit_time":   df.index[-1],
            "entry_price": position["entry"],
            "exit_price":  last_price,
            "qty":         position["qty"],
            "pnl":         round(pnl, 4),
            "pnl_pct":     round(pnl / (position["entry"] * position["qty"]) * 100, 3),
            "reason":      "end_of_data",
            "win":         pnl > 0,
        })

    # -- Metrics --------------------------------------------------------------
    if not trades:
        print("  i  No trades triggered in backtest period.")
        return {"trades": 0}

    tr_df   = pd.DataFrame(trades)
    total   = len(tr_df)
    wins    = int(tr_df["win"].sum())
    losses  = total - wins
    wr      = wins / total * 100
    total_p = tr_df["pnl"].sum()
    avg_win = tr_df.loc[tr_df["win"], "pnl"].mean() if wins else 0
    avg_los = tr_df.loc[~tr_df["win"], "pnl"].mean() if losses else 0
    profit_factor = (
        abs(tr_df.loc[tr_df["win"], "pnl"].sum()) /
        max(abs(tr_df.loc[~tr_df["win"], "pnl"].sum()), 0.001)
    )
    max_draw = tr_df["pnl"].cumsum().cummax().sub(tr_df["pnl"].cumsum()).max()
    sharpe_num = tr_df["pnl_pct"].mean()
    sharpe_den = tr_df["pnl_pct"].std()
    sharpe = sharpe_num / sharpe_den * (252 ** 0.5) if sharpe_den > 0 else 0

    results = {
        "symbol": symbol, "strategy": strategy_name, "days": days,
        "timeframe_minutes": tf, "stake_usd": stake_usd,
        "total_trades": total, "wins": wins, "losses": losses,
        "win_rate_pct": round(wr, 2),
        "total_pnl": round(total_p, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_los, 4),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown": round(max_draw, 4),
        "sharpe_ratio": round(sharpe, 3),
        "trades": trades,
    }

    # -- Print Summary ---------------------------------------------------------
    print(f"\n  {'-'*54}")
    print(f"  {'RESULTS':^54}")
    print(f"  {'-'*54}")
    print(f"  Total Trades   : {total:>6}  |  Win Rate  : {wr:>6.1f}%")
    print(f"  Wins           : {wins:>6}  |  Losses    : {losses:>6}")
    print(f"  Total PnL      : ${total_p:>10.4f}")
    print(f"  Avg Win        : ${avg_win:>10.4f}  |  Avg Loss  : ${avg_los:>10.4f}")
    print(f"  Profit Factor  : {profit_factor:>6.3f}")
    print(f"  Max Drawdown   : ${max_draw:>10.4f}")
    print(f"  Sharpe Ratio   : {sharpe:>6.3f}")
    print(f"  {'-'*54}")

    verdict = " PROFITABLE" if total_p > 0 and wr > 45 else "!!  MARGINAL" if total_p > 0 else " UNPROFITABLE"
    print(f"\n  Verdict: {verdict}")

    # Save to file
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtests")
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir,
                         f"{symbol.replace('/', '_')}_{strategy_name}_{days}d.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n    Results saved -> {fname}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EveryCryptoBot Backtesting Engine")
    parser.add_argument("--symbol",   default="ETH/USD",         help="e.g. ETH/USD")
    parser.add_argument("--strategy", default="MomentumEmaRibbon", help="ScalpMacdRsi | MomentumEmaRibbon | VwapBreakout")
    parser.add_argument("--days",     type=int, default=30,       help="Historical days to test on")
    parser.add_argument("--stake",    type=float, default=25.0, help="USD per trade in simulation (default: $25.0 for $100 balance)")
    args = parser.parse_args()

    run_backtest(args.symbol, args.strategy, args.days, args.stake)
