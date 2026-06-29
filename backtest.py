"""
Backtesting engine: IC calculation, quintile portfolio construction, performance metrics.

IC (Information Coefficient) is the standard evaluation metric for alpha signals
at systematic funds. It measures the cross-sectional rank correlation between
a signal and subsequent returns — a good signal has a consistently positive IC.

ICIR (IC Information Ratio) = mean(IC) / std(IC) measures signal consistency.
Practitioners generally look for ICIR > 0.5 as a threshold for a tradeable signal.
"""
import numpy as np
import pandas as pd
from scipy import stats


def compute_ic(signal: pd.DataFrame, monthly_returns: pd.DataFrame) -> pd.Series:
    """
    Compute monthly Information Coefficient (IC) via Spearman rank correlation.

    IC(t) = Spearman correlation between signal(t) and return(t+1),
    computed cross-sectionally across all stocks at each date.

    A positive IC means high-signal stocks outperformed low-signal stocks
    on average that month. We require at least 50 common observations
    per period to compute a reliable estimate.

    Args:
        signal: Signal DataFrame (dates x tickers)
        monthly_returns: Monthly returns DataFrame (dates x tickers)

    Returns:
        Series of monthly IC values.
    """
    forward_returns = monthly_returns.shift(-1)
    ic_values = {}

    for date in signal.index:
        if date not in forward_returns.index:
            continue
        sig = signal.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        common = sig.index.intersection(fwd.index)

        if len(common) < 50:
            continue

        rho, _ = stats.spearmanr(sig[common], fwd[common])
        ic_values[date] = rho

    return pd.Series(ic_values, name="IC")


def compute_icir(ic: pd.Series, window: int = None):
    """
    Information Coefficient Information Ratio = mean(IC) / std(IC).

    Args:
        ic: Series of monthly IC values
        window: If provided, return rolling ICIR Series. Otherwise return scalar.
    """
    if window:
        return ic.rolling(window).mean() / ic.rolling(window).std()
    return ic.mean() / ic.std()


def build_quintile_portfolios(
    signal: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    n_quintiles: int = 5,
    transaction_cost: float = 0.0010,
) -> tuple:
    """
    Build equal-weighted quintile portfolios with monthly rebalancing.

    At each month end t:
      1. Rank all stocks by signal(t) into n_quintiles buckets
      2. Each quintile's gross return = equal-weighted return during t+1
      3. Net return = gross return minus turnover * transaction_cost

    Turnover is computed as the fraction of the portfolio that changes
    hands each month (two-way), averaged across quintiles.

    Args:
        signal: Signal DataFrame (dates x tickers)
        monthly_returns: Monthly returns DataFrame (dates x tickers)
        n_quintiles: Number of buckets (default 5)
        transaction_cost: One-way cost per unit of turnover (default 10bps)

    Returns:
        quintile_returns: DataFrame of net monthly returns (dates x quintiles)
        turnover: Series of average monthly two-way portfolio turnover
    """
    forward_returns = monthly_returns.shift(-1)
    labels = [f"Q{i}" for i in range(1, n_quintiles + 1)]

    returns_list = []
    turnover_list = []
    prev_holdings = {q: set() for q in labels}

    common_dates = signal.index.intersection(forward_returns.index)

    for date in common_dates:
        sig = signal.loc[date].dropna()
        fwd = forward_returns.loc[date].dropna()
        common_tickers = sig.index.intersection(fwd.index)

        if len(common_tickers) < n_quintiles * 10:
            continue

        quintile_assigns = pd.qcut(sig[common_tickers], q=n_quintiles, labels=labels)
        row = {}
        period_turnover = 0.0

        for q in labels:
            current = set(sig[quintile_assigns == q].index)
            prev = prev_holdings[q]

            # Two-way turnover: stocks that entered + stocks that exited
            if len(prev) > 0:
                q_turnover = (len(current - prev) + len(prev - current)) / (2 * max(len(current), 1))
            else:
                q_turnover = 1.0  # first period: full turnover

            gross_ret = fwd[list(current)].mean()
            net_ret = gross_ret - q_turnover * transaction_cost
            row[q] = net_ret
            period_turnover += q_turnover / n_quintiles
            prev_holdings[q] = current

        returns_list.append(pd.Series(row, name=date))
        turnover_list.append(period_turnover)

    quintile_returns = pd.DataFrame(returns_list)
    turnover = pd.Series(turnover_list, index=quintile_returns.index, name="Turnover")

    return quintile_returns, turnover


def compute_performance_metrics(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """
    Standard performance metrics for a monthly return series.

    Returns a dict suitable for display as a summary table.
    """
    clean = returns.dropna()
    n = len(clean)

    ann_return = (1 + clean).prod() ** (periods_per_year / n) - 1
    ann_vol = clean.std() * np.sqrt(periods_per_year)
    sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

    wealth = (1 + clean).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    max_dd = drawdown.min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else np.nan

    hit_rate = (clean > 0).mean()

    return {
        "Ann. Return": f"{ann_return:.1%}",
        "Ann. Volatility": f"{ann_vol:.1%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.1%}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Hit Rate": f"{hit_rate:.1%}",
        "Months": n,
    }
