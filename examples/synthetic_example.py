"""Generate a deterministic efficient-frontier example."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from markowitz import (
    efficient_frontier,
    global_minimum_variance,
    portfolio_with_risk_free_asset,
    tangency_portfolio,
)
from markowitz.plotting import plot_efficient_frontier


def main() -> None:
    expected_returns = np.array([0.10, 0.12, 0.15])
    covariance = np.array(
        [
            [0.005, -0.010, 0.004],
            [-0.010, 0.040, -0.002],
            [0.004, -0.002, 0.023],
        ]
    )
    risk_free_rate = 0.04

    frontier = efficient_frontier(expected_returns, covariance, maximum_return=0.17)
    gmv = global_minimum_variance(expected_returns, covariance)
    tangency = tangency_portfolio(expected_returns, covariance, risk_free_rate)
    target = portfolio_with_risk_free_asset(
        expected_returns,
        covariance,
        risk_free_rate,
        target_return=0.13,
    )

    print("Global minimum-variance weights:", np.round(gmv.weights, 4))
    print("Tangency weights:", np.round(tangency.weights, 4))
    print("Target risky weights:", np.round(target.risky_weights, 4))
    print("Target risk-free weight:", round(target.risk_free_weight, 4))

    project_root = Path(__file__).resolve().parents[1]
    output = project_root / "outputs" / "efficient_frontier.png"
    figure, _ = plot_efficient_frontier(
        expected_returns,
        covariance,
        frontier,
        asset_labels=["Asset 1", "Asset 2", "Asset 3"],
        tangency=tangency,
        risk_free_rate=risk_free_rate,
        output_path=output,
    )
    plt.close(figure)
    print(f"Saved figure to {output}")


if __name__ == "__main__":
    main()
