import pandas as pd
import numpy as np


def amm_gross_return(gross_returns, weights):
    """
    Compute the AMM gross return at each time step using the formula:
        R_AMM = R_AAPL^w_A * R_MSFT^w_M * R_cash^w_C

    Since R_cash = 1 (cash doesn't grow), any cash weight contributes nothing
    to the product and is safely ignored.

    Parameters
    ----------
    gross_returns : pd.DataFrame
        Columns are ticker names (e.g. 'AAPL', 'MSFT').
        Each value is a gross daily return (e.g. 1.02 means +2%).
    weights : dict
        Keys are ticker names or 'CASH', values are non-negative floats
        that sum to 1. Example: {"AAPL": 0.5, "MSFT": 0.3, "CASH": 0.2}

    Returns
    -------
    pd.Series
        AMM gross return for each date in gross_returns.
    """
    # Start with an array of ones (multiplicative identity).
    result = pd.Series(1.0, index=gross_returns.index)

    for ticker, weight in weights.items():
        if ticker == "CASH":
            # Cash gross return is 1, so 1^anything = 1. Skip it.
            continue
        if ticker not in gross_returns.columns:
            raise ValueError(f"Ticker '{ticker}' not found in gross_returns columns.")
        # Raise each day's gross return to the power of its weight, then multiply in.
        result = result * (gross_returns[ticker] ** weight)

    return result


def cumulative_growth(gross_return_series, initial_value=1.0):
    """
    Convert a series of gross returns into a cumulative growth curve.

    For example, if gross returns are [1.02, 0.99, 1.03], the cumulative
    growth is [1.02, 1.0098, 1.040094] starting from initial_value=1.0.

    Parameters
    ----------
    gross_return_series : pd.Series
        Daily gross returns (values near 1.0).
    initial_value : float
        Starting portfolio value (default 1.0 = normalized to $1).

    Returns
    -------
    pd.Series
        Cumulative product of gross returns scaled by initial_value.
    """
    return initial_value * gross_return_series.cumprod()


def square_root_return_table():
    """
    Build a demonstration table showing how a 50/50 AAPL-cash AMM dampens
    AAPL's simple returns via the square-root formula:

        R_AMM_simple = sqrt(1 + r_AAPL) - 1

    This illustrates the fractional exposure property: the AMM absorbs only
    half the volatility of holding AAPL outright.

    Returns
    -------
    pd.DataFrame
        Columns: 'AAPL Simple Return', 'AMM Simple Return (50/50 AAPL-cash)'
    """
    aapl_simple_returns = [-0.50, -0.25, 0.25, 0.50, 0.75]

    amm_simple_returns = [
        np.sqrt(1 + r) - 1 for r in aapl_simple_returns
    ]

    df = pd.DataFrame({
        "AAPL Simple Return": [f"{r:.0%}" for r in aapl_simple_returns],
        "AMM Simple Return (50/50 AAPL-cash)": [f"{r:.2%}" for r in amm_simple_returns],
    })

    return df


def build_fractional_scenarios(gross_returns):
    """
    Compute cumulative growth curves for several weight scenarios.

    Each scenario is a dictionary specifying how much weight to put on
    AAPL, MSFT, and CASH. The remaining weight always goes to CASH.

    Parameters
    ----------
    gross_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', gross daily returns.

    Returns
    -------
    dict
        Keys are scenario names (strings), values are pd.Series of
        cumulative growth starting from 1.0.
    """
    scenarios = {
        "AAPL 100%":               {"AAPL": 1.00, "MSFT": 0.00, "CASH": 0.00},
        "MSFT 100%":               {"AAPL": 0.00, "MSFT": 1.00, "CASH": 0.00},
        "AAPL 25% / Cash 75%":     {"AAPL": 0.25, "MSFT": 0.00, "CASH": 0.75},
        "AAPL 50% / Cash 50%":     {"AAPL": 0.50, "MSFT": 0.00, "CASH": 0.50},
        "AAPL 75% / Cash 25%":     {"AAPL": 0.75, "MSFT": 0.00, "CASH": 0.25},
        "AAPL 50% / MSFT 50%":     {"AAPL": 0.50, "MSFT": 0.50, "CASH": 0.00},
        "AAPL 40% / MSFT 40% / Cash 20%": {"AAPL": 0.40, "MSFT": 0.40, "CASH": 0.20},
    }

    results = {}
    for name, weights in scenarios.items():
        amm_returns = amm_gross_return(gross_returns, weights)
        results[name] = cumulative_growth(amm_returns, initial_value=1.0)

    return results
