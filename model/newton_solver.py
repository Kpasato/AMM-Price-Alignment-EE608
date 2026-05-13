import numpy as np
import pandas as pd


# Sign convention used throughout this file:
#   delta_i > 0  means a trader deposits token i into the AMM.
#   delta_j > 0  means the AMM pays out token j to the trader.
#   The AMM never pays out more than it holds: delta_j < x_j always.


def realized_delta_j(x_i, x_j, w_i, w_j, delta_i):
    """
    Compute how many units of token j the AMM sends out when a trader
    deposits delta_i units of token i.

    Formula (zero-fee weighted AMM swap invariant):
        delta_j = x_j * (1 - (x_i / (x_i + delta_i)) ^ (w_i / w_j))

    Parameters
    ----------
    x_i : float   Current AMM reserve of token i (the token being deposited).
    x_j : float   Current AMM reserve of token j (the token being received).
    w_i : float   Pool weight of token i  (e.g. 0.5).
    w_j : float   Pool weight of token j  (e.g. 0.5).
    delta_i : float  Amount of token i deposited (must keep x_i + delta_i > 0).

    Returns
    -------
    float  Amount of token j paid out by the AMM.
    """
    ratio = x_i / (x_i + delta_i)          # how the reserve ratio shifts
    exponent = w_i / w_j                    # weight ratio drives the curve shape
    delta_j = x_j * (1.0 - ratio ** exponent)
    return delta_j


def amm_price(x_i, x_j, w_i, w_j):
    """
    Return the current marginal price of token i denominated in token j.

    Formula:
        p_ij = (x_j * w_i) / (x_i * w_j)

    Intuitively: a heavier weight on token i means the AMM prices it higher
    relative to token j.

    Parameters
    ----------
    x_i, x_j : float  Current reserves of tokens i and j.
    w_i, w_j : float  Pool weights of tokens i and j.

    Returns
    -------
    float  Marginal price of token i in units of token j.
    """
    return (x_j * w_i) / (x_i * w_j)


def post_trade_price(x_i, x_j, w_i, w_j, delta_i):
    """
    Compute the AMM price AFTER a trade of size delta_i has been executed.

    Steps:
      1. Compute the tokens paid out: delta_j = realized_delta_j(...)
      2. Update reserves: x_i_new = x_i + delta_i,  x_j_new = x_j - delta_j
      3. Compute new price using amm_price(x_i_new, x_j_new, w_i, w_j)

    Parameters
    ----------
    x_i, x_j : float  Current reserves before the trade.
    w_i, w_j : float  Pool weights.
    delta_i   : float  Size of the deposit into the pool.

    Returns
    -------
    float  Post-trade marginal price of token i in units of token j.
    """
    dj = realized_delta_j(x_i, x_j, w_i, w_j, delta_i)
    x_i_new = x_i + delta_i
    x_j_new = x_j - dj
    return amm_price(x_i_new, x_j_new, w_i, w_j)


def numerical_derivative(f, x, eps=1e-6):
    """
    Estimate the derivative of f at x using the central-difference formula:

        f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)

    Central differences are more accurate than one-sided differences for
    the same step size, which matters when the function is nearly flat.

    Parameters
    ----------
    f   : callable  A function R -> R.
    x   : float     Point at which to evaluate the derivative.
    eps : float     Small step size (default 1e-6).

    Returns
    -------
    float  Numerical approximation of f'(x).
    """
    return (f(x + eps) - f(x - eps)) / (2.0 * eps)


def newton_solve_trade(x_i, x_j, w_i, w_j, external_price_ratio,
                       delta_i_init=0.0, max_iter=50, tol=1e-10):
    """
    Use Newton's method to find the trade size delta_i that aligns the
    AMM post-trade price with the external market price.

    We solve the root-finding problem:
        f(delta_i) = post_trade_price(delta_i) - external_price_ratio = 0

    Newton update rule:
        delta_i_new = delta_i - f(delta_i) / f'(delta_i)

    Safety checks at every iteration:
      - x_i + delta_i must remain positive (no draining the pool).
      - delta_j must be less than x_j (AMM cannot pay out more than it holds).
      - If |f'(delta_i)| is tiny, stop to avoid division-by-zero.

    Parameters
    ----------
    x_i, x_j           : float  Current AMM reserves.
    w_i, w_j           : float  Pool weights.
    external_price_ratio: float  Target price p* (token i in terms of token j).
    delta_i_init        : float  Starting guess for the trade size (default 0).
    max_iter            : int    Maximum Newton iterations (default 50).
    tol                 : float  Convergence tolerance on |f(delta_i)| (default 1e-10).

    Returns
    -------
    dict with keys:
        'delta_i'      : optimal trade size found
        'delta_j'      : tokens paid out at that trade size
        'final_price'  : post-trade AMM price
        'converged'    : True if |f| < tol at the final iterate
        'iterations'   : pd.DataFrame logging each Newton step
    """
    # The root function: zero when AMM price matches the external price.
    def f(d):
        return post_trade_price(x_i, x_j, w_i, w_j, d) - external_price_ratio

    delta = float(delta_i_init)
    records = []

    for iteration in range(max_iter):
        f_val = f(delta)
        df_val = numerical_derivative(f, delta)

        records.append({
            "iteration": iteration,
            "delta_i":   round(delta, 8),
            "f(delta_i)": round(f_val, 10),
            "f'(delta_i)": round(df_val, 10),
        })

        # Converged when the price gap is smaller than the tolerance.
        if abs(f_val) < tol:
            break

        # Stop safely if the derivative is essentially zero.
        if abs(df_val) < 1e-14:
            break

        # Newton step.
        delta_new = delta - f_val / df_val

        # Safety: x_i + delta must stay strictly positive.
        if x_i + delta_new <= 0:
            delta_new = -x_i * 0.99   # pull back to 99% of max withdrawal

        # Safety: delta_j must not exceed the pool's j-reserve.
        dj_check = realized_delta_j(x_i, x_j, w_i, w_j, delta_new)
        if dj_check >= x_j:
            delta_new = delta   # freeze if the trade would drain the pool
            break

        delta = delta_new

    converged = abs(f(delta)) < tol
    final_dj = realized_delta_j(x_i, x_j, w_i, w_j, delta)
    final_price = post_trade_price(x_i, x_j, w_i, w_j, delta)

    return {
        "delta_i":     round(delta, 8),
        "delta_j":     round(final_dj, 8),
        "final_price": round(final_price, 8),
        "converged":   converged,
        "iterations":  pd.DataFrame(records),
    }


def run_example_newton_case():
    """
    Demonstrate the Newton solver on a concrete example.

    Setup:
      - AMM holds 1000 units of AAPL (token i) and 1000 units of MSFT (token j).
      - Both tokens have equal weight (0.5 / 0.5), so the initial price is 1.0.
      - The external market moves: AAPL drops to 0.90 MSFT.
      - Arbitrageurs deposit AAPL (delta_i > 0) and receive MSFT (delta_j > 0)
        until the AMM price falls to match the external price.
      - Newton finds the exact trade size needed.

    Prints a summary and returns the full result dictionary.
    """
    x_i = 1000.0    # AAPL reserve
    x_j = 1000.0    # MSFT reserve
    w_i = 0.5       # AAPL pool weight
    w_j = 0.5       # MSFT pool weight

    initial_price = amm_price(x_i, x_j, w_i, w_j)
    external_price = 0.90   # AAPL fell 10% vs MSFT externally

    print("Newton Solver — Example AMM Rebalancing")
    print(f"  Initial AMM price  : {initial_price:.6f}")
    print(f"  External price     : {external_price:.6f}")
    print(f"  Price gap to close : {external_price - initial_price:.6f}")
    print()

    result = newton_solve_trade(
        x_i, x_j, w_i, w_j,
        external_price_ratio=external_price,
        delta_i_init=0.0,
    )

    print("Newton convergence table:")
    print(result["iterations"].to_string(index=False))
    print()
    print(f"  Optimal delta_i  : {result['delta_i']:>12.6f}  (AAPL deposited)")
    print(f"  Resulting delta_j: {result['delta_j']:>12.6f}  (MSFT received)")
    print(f"  Final AMM price  : {result['final_price']:>12.8f}")
    print(f"  Converged        : {result['converged']}")

    return result


if __name__ == "__main__":
    run_example_newton_case()
