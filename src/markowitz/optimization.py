"""Analytical solutions for unconstrained Markowitz optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Portfolio:
    """A fully invested risky-asset portfolio."""

    weights: FloatArray
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class RiskFreePortfolio:
    """A portfolio combining a risk-free asset with risky assets."""

    risk_free_weight: float
    risky_weights: FloatArray
    expected_return: float
    volatility: float


@dataclass(frozen=True)
class Frontier:
    """Points and weights on the efficient upper branch."""

    expected_returns: FloatArray
    volatilities: FloatArray
    weights: FloatArray


def _validate_inputs(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    mu = np.asarray(expected_returns, dtype=float)
    sigma = np.asarray(covariance, dtype=float)

    if mu.ndim != 1 or mu.size == 0:
        raise ValueError("expected_returns must be a nonempty one-dimensional array")
    if sigma.shape != (mu.size, mu.size):
        raise ValueError("covariance must have shape (n_assets, n_assets)")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sigma)):
        raise ValueError("inputs must contain only finite values")
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")

    try:
        np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc

    return mu, sigma


def _portfolio(weights: FloatArray, mu: FloatArray, sigma: FloatArray) -> Portfolio:
    expected_return = float(mu @ weights)
    variance = float(weights @ sigma @ weights)
    return Portfolio(
        weights=weights,
        expected_return=expected_return,
        volatility=float(np.sqrt(max(variance, 0.0))),
    )


def _frontier_constants(
    mu: FloatArray,
    sigma: FloatArray,
) -> tuple[FloatArray, FloatArray, float, float, float, float]:
    ones = np.ones(mu.size)
    solved_ones = np.linalg.solve(sigma, ones)
    solved_mu = np.linalg.solve(sigma, mu)
    a = float(ones @ solved_ones)
    b = float(ones @ solved_mu)
    c = float(mu @ solved_mu)
    determinant = a * c - b**2
    if determinant <= _TOLERANCE:
        raise ValueError("expected returns do not identify a nondegenerate frontier")
    return solved_ones, solved_mu, a, b, c, determinant


def global_minimum_variance(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
) -> Portfolio:
    """Return the fully invested global minimum-variance portfolio."""

    mu, sigma = _validate_inputs(expected_returns, covariance)
    ones = np.ones(mu.size)
    solved_ones = np.linalg.solve(sigma, ones)
    denominator = float(ones @ solved_ones)
    if denominator <= _TOLERANCE:
        raise ValueError("invalid covariance matrix for a fully invested portfolio")
    return _portfolio(solved_ones / denominator, mu, sigma)


def minimum_variance_for_return(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    target_return: float,
) -> Portfolio:
    """Return the minimum-variance portfolio with return equal to the target."""

    mu, sigma = _validate_inputs(expected_returns, covariance)
    if not np.isfinite(target_return):
        raise ValueError("target_return must be finite")

    solved_ones, solved_mu, a, b, c, determinant = _frontier_constants(mu, sigma)
    coefficient_ones = (c - b * target_return) / determinant
    coefficient_mu = (a * target_return - b) / determinant
    weights = coefficient_ones * solved_ones + coefficient_mu * solved_mu
    return _portfolio(weights, mu, sigma)


def minimum_variance_with_floor(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    minimum_return: float,
) -> Portfolio:
    """Minimize variance subject to a minimum-return inequality."""

    gmv = global_minimum_variance(expected_returns, covariance)
    if minimum_return <= gmv.expected_return:
        return gmv
    return minimum_variance_for_return(expected_returns, covariance, minimum_return)


def efficient_frontier(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    *,
    maximum_return: float | None = None,
    number_of_points: int = 200,
) -> Frontier:
    """Return only the efficient branch above the global minimum-variance point."""

    mu, sigma = _validate_inputs(expected_returns, covariance)
    if number_of_points < 2:
        raise ValueError("number_of_points must be at least 2")

    gmv = global_minimum_variance(mu, sigma)
    if maximum_return is None:
        return_scale = max(float(np.ptp(mu)), float(np.max(np.abs(mu))), 0.01)
        maximum_return = max(float(np.max(mu)), gmv.expected_return + 0.5 * return_scale)
    if maximum_return <= gmv.expected_return:
        raise ValueError("maximum_return must exceed the GMV portfolio return")

    targets = np.linspace(gmv.expected_return, maximum_return, number_of_points)
    portfolios = [minimum_variance_for_return(mu, sigma, target) for target in targets]
    weights = np.vstack([portfolio.weights for portfolio in portfolios])
    volatilities = np.array([portfolio.volatility for portfolio in portfolios])
    return Frontier(targets, volatilities, weights)


def tangency_portfolio(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    risk_free_rate: float,
) -> Portfolio:
    """Return the maximum-Sharpe fully invested risky-asset portfolio."""

    mu, sigma = _validate_inputs(expected_returns, covariance)
    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    excess_returns = mu - risk_free_rate
    direction = np.linalg.solve(sigma, excess_returns)
    denominator = float(np.sum(direction))
    if abs(denominator) <= _TOLERANCE:
        raise ValueError("tangency portfolio cannot be normalized to sum to one")
    return _portfolio(direction / denominator, mu, sigma)


def portfolio_with_risk_free_asset(
    expected_returns: ArrayLike,
    covariance: ArrayLike,
    risk_free_rate: float,
    target_return: float,
) -> RiskFreePortfolio:
    """Return the minimum-risk allocation attaining at least the target return."""

    mu, sigma = _validate_inputs(expected_returns, covariance)
    if not np.isfinite(risk_free_rate) or not np.isfinite(target_return):
        raise ValueError("risk_free_rate and target_return must be finite")

    if target_return <= risk_free_rate:
        return RiskFreePortfolio(
            risk_free_weight=1.0,
            risky_weights=np.zeros(mu.size),
            expected_return=risk_free_rate,
            volatility=0.0,
        )

    excess_returns = mu - risk_free_rate
    direction = np.linalg.solve(sigma, excess_returns)
    squared_slope = float(excess_returns @ direction)
    if squared_slope <= _TOLERANCE:
        raise ValueError("risky assets offer no positive mean-variance tradeoff")

    risky_weights = ((target_return - risk_free_rate) / squared_slope) * direction
    risk_free_weight = float(1.0 - np.sum(risky_weights))
    expected_return = float(risk_free_weight * risk_free_rate + mu @ risky_weights)
    variance = float(risky_weights @ sigma @ risky_weights)
    return RiskFreePortfolio(
        risk_free_weight=risk_free_weight,
        risky_weights=risky_weights,
        expected_return=expected_return,
        volatility=float(np.sqrt(max(variance, 0.0))),
    )
