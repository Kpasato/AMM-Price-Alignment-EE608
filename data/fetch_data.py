import yfinance as yf
import pandas as pd
import os


def download_prices(tickers, start_date, end_date):
    """
    Download daily adjusted close prices for a list of tickers.
    auto_adjust=True means splits and dividends are already baked in.
    Returns a DataFrame with one column per ticker.
    """
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)

    # yfinance returns a MultiIndex when multiple tickers are requested.
    # We only want the 'Close' level.
    prices = raw["Close"]

    # Drop any rows where all values are NaN (e.g. non-trading days).
    prices = prices.dropna(how="all")

    return prices


def compute_gross_returns(prices):
    """
    Compute gross daily returns: R_t = P_t / P_{t-1}.
    A gross return of 1.02 means a 2% gain that day.
    The first row is dropped because there is no previous price for it.
    """
    gross_returns = prices / prices.shift(1)

    # Remove the first row (it will be all NaN after the shift).
    gross_returns = gross_returns.dropna(how="all")

    return gross_returns


def save_dataframe(df, output_path):
    """
    Save a DataFrame to a CSV file.
    Creates any missing parent directories automatically.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT"]
    prices = download_prices(tickers, "2020-01-01", "2025-12-31")
    gross_returns = compute_gross_returns(prices)
    save_dataframe(prices, "outputs/prices.csv")
    save_dataframe(gross_returns, "outputs/gross_returns.csv")
    print(prices.head())
    print(gross_returns.head())
