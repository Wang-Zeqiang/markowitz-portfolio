"""Run Markowitz optimization and an empirical CAPM comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from markowitz import efficient_frontier, tangency_portfolio
from markowitz.data import download_historical_market_data
from markowitz.metrics import capm_expected_returns, historical_beta
from markowitz.plotting import plot_efficient_frontier

TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "PG"]
START = "2019-01-01"
END = "2024-01-01"


def main() -> None:
    data = download_historical_market_data(TICKERS, START, END)
    frontier = efficient_frontier(data.expected_returns, data.covariance)
    tangency = tangency_portfolio(
        data.expected_returns,
        data.covariance,
        data.risk_free_rate,
    )

    beta = historical_beta(data.asset_returns, data.market_returns)
    capm_returns = capm_expected_returns(
        beta,
        data.risk_free_rate,
        data.market_expected_return,
    )

    print(f"Average risk-free proxy: {data.risk_free_rate:.4f}")
    print(f"Historical S&P 500 mean return: {data.market_expected_return:.4f}")
    print("\nTicker  Beta    Historical mean  CAPM-implied mean")
    for ticker, asset_beta, historical, predicted in zip(
        TICKERS,
        beta,
        data.expected_returns,
        capm_returns,
    ):
        print(f"{ticker:<7} {asset_beta:>6.3f} {historical:>16.4f} {predicted:>18.4f}")

    print("\nTangency portfolio weights")
    for ticker, weight in zip(TICKERS, tangency.weights):
        print(f"{ticker:<7} {weight:>8.4f}")

    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "outputs" / "historical_efficient_frontier.png"
    figure, _ = plot_efficient_frontier(
        data.expected_returns,
        data.covariance,
        frontier,
        asset_labels=TICKERS,
        tangency=tangency,
        risk_free_rate=data.risk_free_rate,
        output_path=output,
    )
    plt.close(figure)
    print(f"\nSaved figure to {output}")


if __name__ == "__main__":
    main()
