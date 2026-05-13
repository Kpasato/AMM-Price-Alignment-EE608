import os
import matplotlib.pyplot as plt


def plot_fractional_scenarios(scenarios, output_path):
    """
    Plot cumulative growth of $1 for multiple AMM weight scenarios.

    Parameters
    ----------
    scenarios : dict
        Keys are scenario names (str), values are pd.Series of cumulative
        growth (starting from 1.0) indexed by date.
    output_path : str
        File path where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, series in scenarios.items():
        ax.plot(series.index, series.values, label=name)

    ax.set_title("Fractional AMM Return Scenarios")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value of $1")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_target_vs_fitted(target_cum, fitted_cum, title, output_path):
    """
    Plot target cumulative return vs the AMM-fitted cumulative return.

    Parameters
    ----------
    target_cum : pd.Series
        Cumulative growth of the target return profile.
    fitted_cum : pd.Series
        Cumulative growth of the optimized AMM return.
    title : str
        Plot title (include the scenario name here).
    output_path : str
        File path where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(target_cum.index, target_cum.values, label="Target", linewidth=2)
    ax.plot(fitted_cum.index, fitted_cum.values, label="AMM Fitted",
            linestyle="--", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Value of $1")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_newton_convergence(iterations_df, output_path):
    """
    Plot how quickly Newton's method converges by showing |f(delta_i)|
    at each iteration on a log scale.

    Parameters
    ----------
    iterations_df : pd.DataFrame
        The 'iterations' DataFrame returned by newton_solve_trade().
        Must have columns 'iteration' and 'f(delta_i)'.
    output_path : str
        File path where the PNG will be saved.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    abs_f = iterations_df["f(delta_i)"].abs()
    ax.plot(iterations_df["iteration"], abs_f, marker="o")

    # Use log scale only if all values are positive (avoids log(0) error).
    if (abs_f > 0).all():
        ax.set_yscale("log")
        ax.set_ylabel("|f(delta_i)|  (log scale)")
    else:
        ax.set_ylabel("|f(delta_i)|")

    ax.set_title("Newton's Method Convergence")
    ax.set_xlabel("Iteration")
    ax.grid(True)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


def save_summary_table(df, output_path):
    """
    Save a DataFrame to a CSV file, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Any summary table to persist.
    output_path : str
        File path for the CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved: {output_path}")
