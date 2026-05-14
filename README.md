Fractional-Return AMMs with Newton-Based Arbitrage Rebalancing

EE/CpE 608 Applied Modeling and Optimization — Final Project


Main Question

Can AMM weights be chosen to create controlled fractional exposure to AAPL/MSFT returns, and can Newton's method compute the trade needed to realign AMM prices with external market prices?


Overview

Weighted automated market makers (AMMs) have a useful property: by choosing the pool weights carefully, you can make the AMM's return behave like a fractional version of an underlying asset's return.

For example, a pool with 50% AAPL and 50% cash produces a return that looks like the square root of AAPL's gross return. More generally, the AMM gross return is:

    R_AMM = R_AAPL^w_A * R_MSFT^w_M

This project does two things:

1. Weight optimization — fit AAPL/MSFT/cash weights so the AMM tracks a target return profile as closely as possible.
2. Newton price alignment — after external market prices move, use Newton's method to find the trade size that brings the AMM price back in line with the market.


Optimization Model

We choose weights by minimizing the sum of squared log-return errors:

    min_{w_A, w_M}  sum_t ( log(R_target,t) - w_A*log(R_AAPL,t) - w_M*log(R_MSFT,t) )^2

    subject to:
      w_A >= 0
      w_M >= 0
      w_A + w_M <= 1

    w_cash = 1 - w_A - w_M

This is a constrained least-squares problem in log-return space. We solve it with scipy.optimize.minimize using the SLSQP method.


Dataset

- AAPL and MSFT adjusted close prices
- Date range: 2020-01-01 to 2025-12-31
- Source: downloaded via yfinance with auto_adjust=True


How to Run

    pip install -r requirements.txt
    python3 analysis/run_analysis.py

All outputs are saved to the outputs/ folder.


File Overview

- data/fetch_data.py        — downloads prices, computes gross returns, saves CSVs
- model/amm_returns.py      — AMM return formula, cumulative growth, scenario builder
- model/weights_optimizer.py — constrained least-squares weight fitting
- model/newton_solver.py    — Newton's method for AMM price alignment
- analysis/plots.py         — matplotlib plotting and CSV saving helpers
- analysis/run_analysis.py  — runs the full pipeline end to end


Generated Outputs

Running the analysis script produces:

- fractional_return_scenarios.png — cumulative growth curves for several weight scenarios
- optimization_results.csv — fitted weights and tracking metrics for each target
- target_vs_fitted_*.png — target vs optimized AMM cumulative returns for each scenario
- newton_iterations.csv and newton_convergence.png — Newton solver convergence per iteration
- newton_summary.csv — final trade size, price, and convergence status


Limitations

- This is a simplified educational model, not a live AMM implementation.
- The weight optimization uses historical AAPL/MSFT returns, not real AMM pool data.
- The Newton solver assumes a zero-fee AMM and ignores slippage, gas costs, and liquidity depth.
- Long-only weight constraints produce fractional exposure, not true leverage or short positions.
- Results depend on the chosen date range and may not generalize to other periods.
