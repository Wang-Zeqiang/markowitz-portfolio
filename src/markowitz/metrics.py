"""Portfolio and CAPM metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def portfolio_return(weights: ArrayLike, expected_returns: ArrayLike) -> float:
    """Calculate portfolio expected return."""

    return float(np.asarray(weights, dtype=float) @ np.asarray(expected_returns, dtype=float))


def portfolio_volatility(weights: ArrayLike, covariance: ArrayLike) -> float:
    """Calculate portfolio standard deviation."""

    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    variance = float(w @ sigma @ w)
    return float(np.sqrt(max(variance, 0.0)))


def sharpe_ratio(expected_return: float, volatility: float, risk_free_rate: float) -> float:
    """Calculate the portfolio Sharpe ratio."""

    if volatility <= 0:
        raise ValueError("volatility must be positive")
    return (expected_return - risk_free_rate) / volatility


def historical_beta(asset_returns: ArrayLike, market_returns: ArrayLike) -> NDArray[np.float64]:
    """Estimate each asset's beta against observed market returns."""

    assets = np.asarray(asset_returns, dtype=float)
    market = np.asarray(market_returns, dtype=float).reshape(-1)
    if assets.ndim == 1:
        assets = assets.reshape(-1, 1)
    if assets.ndim != 2 or assets.shape[0] != market.size:
        raise ValueError("asset and market returns must contain the same observations")
    if not np.all(np.isfinite(assets)) or not np.all(np.isfinite(market)):
        raise ValueError("returns must contain only finite values")
    market_variance = float(np.var(market, ddof=1))
    if market_variance <= 0:
        raise ValueError("market returns must have positive sample variance")

    centered_assets = assets - assets.mean(axis=0)
    centered_market = market - market.mean()
    covariance = centered_assets.T @ centered_market / (market.size - 1)
    return covariance / market_variance


def capm_expected_returns(
    beta: ArrayLike,
    risk_free_rate: float,
    market_expected_return: float,
) -> NDArray[np.float64]:
    """Calculate CAPM-implied expected returns."""

    beta_array = np.asarray(beta, dtype=float)
    return risk_free_rate + beta_array * (market_expected_return - risk_free_rate)
