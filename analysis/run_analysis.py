import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from data.fetch_data import download_prices, compute_gross_returns, save_dataframe
from model.amm_returns import build_fractional_scenarios, square_root_return_table, cumulative_growth
from model.weights_optimizer import run_all_optimizations
from model.newton_solver import run_example_newton_case
from analysis.plots import (
    plot_fractional_scenarios,
    plot_target_vs_fitted,
    plot_newton_convergence,
    save_summary_table,
)


# Step 1: Download price data
print("Downloading AAPL and MSFT prices...")
tickers = ["AAPL", "MSFT"]
prices = download_prices(tickers, "2020-01-01", "2025-12-31")
gross_returns = compute_gross_returns(prices)

save_dataframe(prices, "outputs/prices.csv")
save_dataframe(gross_returns, "outputs/gross_returns.csv")

# Step 2: Compute log returns
log_returns = np.log(gross_returns)

# Step 3: Fractional AMM scenarios
print("Building fractional AMM scenarios...")
scenarios = build_fractional_scenarios(gross_returns)
plot_fractional_scenarios(scenarios, "outputs/fractional_return_scenarios.png")

# Step 4: Square-root return table
sqrt_table = square_root_return_table()
save_summary_table(sqrt_table, "outputs/square_root_return_table.csv")

# Step 5: Weight optimization
print("Running weight optimizations...")
results_df, fitted_gross, target_gross = run_all_optimizations(gross_returns, log_returns)
save_summary_table(results_df, "outputs/optimization_results.csv")
print(results_df.to_string())

# Step 6: Target vs fitted plots for each optimization target
target_labels = {
    "sqrt_aapl":    "Target vs Fitted: sqrt(R_AAPL)  [w=0.5 AAPL / 0.5 Cash]",
    "mixed_40_40":  "Target vs Fitted: 40% AAPL + 40% MSFT",
    "simple_50_50": "Target vs Fitted: Simple 50/50 Blend",
}

for target_type, title in target_labels.items():
    target_cum = cumulative_growth(target_gross[target_type])
    fitted_cum = cumulative_growth(fitted_gross[target_type])
    out_path = f"outputs/target_vs_fitted_{target_type}.png"
    plot_target_vs_fitted(target_cum, fitted_cum, title, out_path)

# Step 7: Newton solver example
print("Running Newton solver example...")
newton_result = run_example_newton_case()

# Save the iteration-by-iteration convergence table
save_summary_table(newton_result["iterations"], "outputs/newton_iterations.csv")

# Save a one-row summary of the final result
newton_summary = pd.DataFrame([{
    "delta_i":     newton_result["delta_i"],
    "delta_j":     newton_result["delta_j"],
    "final_price": newton_result["final_price"],
    "converged":   newton_result["converged"],
}])
save_summary_table(newton_summary, "outputs/newton_summary.csv")

# Save the convergence plot
plot_newton_convergence(newton_result["iterations"], "outputs/newton_convergence.png")

print("Done. All outputs saved to outputs/")
