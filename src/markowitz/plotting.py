"""Visualization helpers for Markowitz analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

from .optimization import Frontier, Portfolio


def plot_efficient_frontier(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    frontier: Frontier,
    *,
    asset_labels: Sequence[str] | None = None,
    tangency: Portfolio | None = None,
    risk_free_rate: float | None = None,
    output_path: str | Path | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the efficient branch, assets, and optional Capital Market Line."""

    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    if asset_labels is None:
        asset_labels = [f"Asset {index + 1}" for index in range(mu.size)]
    if len(asset_labels) != mu.size:
        raise ValueError("asset_labels must match the number of assets")

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        frontier.volatilities,
        frontier.expected_returns,
        color="#17365D",
        linewidth=2.2,
        label="Efficient frontier",
    )

    asset_volatilities = np.sqrt(np.diag(sigma))
    axis.scatter(asset_volatilities, mu, color="#333333", marker="x", s=65, label="Assets")
    for label, volatility, expected_return in zip(asset_labels, asset_volatilities, mu):
        axis.annotate(label, (volatility, expected_return), xytext=(5, 4), textcoords="offset points")

    if tangency is not None:
        axis.scatter(
            tangency.volatility,
            tangency.expected_return,
            color="#C62828",
            s=85,
            zorder=3,
            label="Tangency portfolio",
        )
        if risk_free_rate is not None:
            slope = (tangency.expected_return - risk_free_rate) / tangency.volatility
            visible_return = max(float(frontier.expected_returns.max()), float(mu.max()))
            if slope > 0:
                maximum_sigma = (visible_return * 1.05 - risk_free_rate) / slope
            else:
                maximum_sigma = max(float(frontier.volatilities.max()), tangency.volatility)
            cml_sigma = np.linspace(0.0, maximum_sigma, 200)
            axis.plot(
                cml_sigma,
                risk_free_rate + slope * cml_sigma,
                color="#E07A1F",
                linestyle="--",
                linewidth=1.8,
                label="Capital Market Line",
            )
            axis.scatter(0.0, risk_free_rate, color="#2E7D32", s=75, zorder=3, label="Risk-free asset")

    axis.set_title("Markowitz Efficient Frontier")
    axis.set_xlabel("Annualized volatility")
    axis.set_ylabel("Annualized expected return")
    right_edge = max(float(frontier.volatilities.max()), float(asset_volatilities.max()))
    axis.set_xlim(left=-0.005, right=right_edge * 1.10)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return figure, axis
