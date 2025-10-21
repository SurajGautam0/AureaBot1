
import numpy as np

def metrics_from_equity(equity_curve):
    if len(equity_curve) < 2:
        return {"sharpe": 0.0, "max_drawdown": 1.0, "winrate": 0.0, "profit_factor": 1.0}
    rets = np.diff(equity_curve)
    mean = float(np.mean(rets))
    std = float(np.std(rets) + 1e-9)
    sharpe = mean / std if std > 0 else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        dd = (peak - e) / (abs(peak) + 1e-9)
        max_dd = max(max_dd, dd)

    wins = (rets > 0).sum()
    losses = (rets < 0).sum()
    winrate = wins / max(1, (wins + losses))
    gross_win = float(rets[rets > 0].sum())
    gross_loss = float(-rets[rets < 0].sum())
    pf = gross_win / max(1e-9, gross_loss)

    return {"sharpe": sharpe, "max_drawdown": max_dd, "winrate": winrate, "profit_factor": pf}

def simple_backtest(signals, prices):
    """
    signals: list of dicts with "side" in {1,-1} and "conf" in [0,1]
    prices: list of floats matching length
    Very simple PnL: side * price_change * conf
    """
    if len(signals) < 2 or len(signals) != len(prices):
        return [0.0]
    eq = [0.0]
    for i in range(1, len(prices)):
        d = prices[i] - prices[i-1]
        side = signals[i-1]["side"]
        conf = max(0.0, min(1.0, signals[i-1].get("conf", 0.5)))
        pnl = side * d * conf
        eq.append(eq[-1] + pnl)
    return eq
