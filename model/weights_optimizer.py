import numpy as np
import pandas as pd
from scipy.optimize import minimize


def make_target_log_return(gross_returns, log_returns, target_type):
    """
    Build a target log-return series to fit the AMM weights against.

    Three target types are supported:

    - "sqrt_aapl"   : target is 50% AAPL log-return, equivalent to sqrt(R_AAPL).
    - "mixed_40_40" : target is 40% AAPL + 40% MSFT log-return.
    - "simple_50_50": target is the log of a simple 50/50 arithmetic blend of
                      AAPL and MSFT simple returns.  This is NOT a geometric mean,
                      so it cannot be matched exactly by the AMM — that tension is
                      interesting and worth showing in the slides.

    Parameters
    ----------
    gross_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', daily gross returns.
    log_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', daily log returns (= log of gross returns).
    target_type : str
        One of "sqrt_aapl", "mixed_40_40", "simple_50_50".

    Returns
    -------
    pd.Series
        Target log-return series aligned to the same date index.
    """
    if target_type == "sqrt_aapl":
        # Geometric mean with w_A=0.5 is equivalent to taking the square root.
        return 0.5 * log_returns["AAPL"]

    elif target_type == "mixed_40_40":
        # Exact geometric blend: 40% AAPL, 40% MSFT, 20% cash.
        return 0.4 * log_returns["AAPL"] + 0.4 * log_returns["MSFT"]

    elif target_type == "simple_50_50":
        # Arithmetic (simple) average of the two simple returns, then log.
        # This blends the two stocks equally in dollar terms, not log terms.
        target_gross = (
            1
            + 0.5 * (gross_returns["AAPL"] - 1)
            + 0.5 * (gross_returns["MSFT"] - 1)
        )
        return np.log(target_gross)

    else:
        raise ValueError(
            f"Unknown target_type '{target_type}'. "
            "Choose from: 'sqrt_aapl', 'mixed_40_40', 'simple_50_50'."
        )


def optimize_weights(log_returns, target_log_return):
    """
    Find the AMM weights w_A and w_M that minimise the sum of squared
    log-return errors:

        minimize  sum_t ( target_t - w_A * log_AAPL_t - w_M * log_MSFT_t )^2

    subject to:
        w_A >= 0
        w_M >= 0
        w_A + w_M <= 1

    The cash weight is recovered as:  w_cash = 1 - w_A - w_M

    Uses SciPy's SLSQP solver, which handles both inequality constraints
    and bounds efficiently.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', daily log returns.
    target_log_return : pd.Series
        Target log-return series (same index as log_returns).

    Returns
    -------
    dict with keys:
        'w_AAPL'     : optimal AAPL weight
        'w_MSFT'     : optimal MSFT weight
        'w_cash'     : implied cash weight
        'objective'  : final objective value (sum of squared errors)
        'success'    : True if the solver converged
    """
    log_aapl = log_returns["AAPL"].values
    log_msft = log_returns["MSFT"].values
    target = target_log_return.values

    def objective(w):
        w_a, w_m = w
        fitted = w_a * log_aapl + w_m * log_msft
        residuals = target - fitted
        return np.sum(residuals ** 2)

    # Initial guess: equal split between AAPL and MSFT.
    w0 = np.array([0.5, 0.5])

    # Each weight must be >= 0.
    bounds = [(0.0, 1.0), (0.0, 1.0)]

    # The two stock weights together must not exceed 1.
    constraints = [
        {"type": "ineq", "fun": lambda w: 1.0 - w[0] - w[1]}
    ]

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    w_a = float(result.x[0])
    w_m = float(result.x[1])
    w_c = 1.0 - w_a - w_m

    return {
        "w_AAPL":    round(w_a, 6),
        "w_MSFT":    round(w_m, 6),
        "w_cash":    round(w_c, 6),
        "objective": round(float(result.fun), 8),
        "success":   bool(result.success),
    }


def compute_tracking_metrics(target_log_return, fitted_log_return):
    """
    Measure how closely the AMM log returns track the target log returns.

    Parameters
    ----------
    target_log_return : pd.Series or np.ndarray
        The desired log-return series.
    fitted_log_return : pd.Series or np.ndarray
        The AMM log-return series produced by the optimised weights.

    Returns
    -------
    dict with keys:
        'MSE'         : mean squared error
        'MAE'         : mean absolute error
        'max_abs_err' : maximum absolute error across all days
        'correlation' : Pearson correlation between target and fitted
    """
    target = np.asarray(target_log_return)
    fitted = np.asarray(fitted_log_return)
    errors = target - fitted

    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    max_err = float(np.max(np.abs(errors)))
    corr = float(np.corrcoef(target, fitted)[0, 1])

    return {
        "MSE":         round(mse, 8),
        "MAE":         round(mae, 6),
        "max_abs_err": round(max_err, 6),
        "correlation": round(corr, 6),
    }


def run_all_optimizations(gross_returns, log_returns):
    """
    Run the weight optimisation for all three target profiles and collect
    results in a tidy summary DataFrame.

    Parameters
    ----------
    gross_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', daily gross returns.
    log_returns : pd.DataFrame
        Columns 'AAPL' and 'MSFT', daily log returns.

    Returns
    -------
    results_df : pd.DataFrame
        One row per target, columns for weights and tracking metrics.
    fitted_gross : dict
        Keys are target names, values are pd.Series of fitted daily gross
        returns (= exp of fitted log returns).
    target_gross : dict
        Keys are target names, values are pd.Series of target daily gross
        returns (= exp of target log returns).
    """
    target_types = ["sqrt_aapl", "mixed_40_40", "simple_50_50"]

    rows = []
    fitted_gross = {}
    target_gross = {}

    for target_type in target_types:
        # Build the target log-return series for this profile.
        target_lr = make_target_log_return(gross_returns, log_returns, target_type)

        # Solve for the best weights.
        opt = optimize_weights(log_returns, target_lr)

        # Reconstruct the fitted log-return using the solved weights.
        fitted_lr = (
            opt["w_AAPL"] * log_returns["AAPL"]
            + opt["w_MSFT"] * log_returns["MSFT"]
        )

        # Compute tracking quality metrics.
        metrics = compute_tracking_metrics(target_lr, fitted_lr)

        # Convert log returns back to gross returns for plotting later.
        fitted_gross[target_type] = np.exp(fitted_lr)
        target_gross[target_type] = np.exp(target_lr)

        # Collect everything into one flat row.
        row = {"target": target_type}
        row.update(opt)
        row.update(metrics)
        rows.append(row)

    results_df = pd.DataFrame(rows).set_index("target")

    return results_df, fitted_gross, target_gross
