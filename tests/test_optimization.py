import unittest

import numpy as np

from markowitz import (
    efficient_frontier,
    global_minimum_variance,
    minimum_variance_for_return,
    minimum_variance_with_floor,
    portfolio_with_risk_free_asset,
    tangency_portfolio,
)

EXPECTED_RETURNS = np.array([0.10, 0.12, 0.15])
COVARIANCE = np.array(
    [
        [0.005, -0.010, 0.004],
        [-0.010, 0.040, -0.002],
        [0.004, -0.002, 0.023],
    ]
)


class OptimizationTests(unittest.TestCase):
    def test_global_minimum_variance_is_fully_invested(self) -> None:
        result = global_minimum_variance(EXPECTED_RETURNS, COVARIANCE)
        self.assertTrue(np.isclose(result.weights.sum(), 1.0))
        self.assertTrue(np.isclose(result.expected_return, EXPECTED_RETURNS @ result.weights))
        self.assertGreater(result.volatility, 0)

    def test_target_return_portfolio_meets_both_equalities(self) -> None:
        result = minimum_variance_for_return(EXPECTED_RETURNS, COVARIANCE, 0.13)
        self.assertTrue(np.isclose(result.weights.sum(), 1.0))
        self.assertTrue(np.isclose(result.expected_return, 0.13))

    def test_return_floor_uses_gmv_when_constraint_is_inactive(self) -> None:
        gmv = global_minimum_variance(EXPECTED_RETURNS, COVARIANCE)
        result = minimum_variance_with_floor(EXPECTED_RETURNS, COVARIANCE, 0.05)
        self.assertTrue(np.allclose(result.weights, gmv.weights))
        self.assertTrue(np.isclose(result.expected_return, gmv.expected_return))
        self.assertTrue(np.isclose(result.volatility, gmv.volatility))

    def test_tangency_portfolio_is_fully_invested(self) -> None:
        result = tangency_portfolio(EXPECTED_RETURNS, COVARIANCE, 0.04)
        self.assertTrue(np.isclose(result.weights.sum(), 1.0))
        self.assertGreater(result.expected_return, 0.04)
        self.assertGreater(result.volatility, 0)

    def test_risk_free_portfolio_meets_target_and_budget(self) -> None:
        result = portfolio_with_risk_free_asset(EXPECTED_RETURNS, COVARIANCE, 0.04, 0.13)
        total_weight = result.risk_free_weight + result.risky_weights.sum()
        self.assertTrue(np.isclose(total_weight, 1.0))
        self.assertTrue(np.isclose(result.expected_return, 0.13))

    def test_risk_free_only_case(self) -> None:
        result = portfolio_with_risk_free_asset(EXPECTED_RETURNS, COVARIANCE, 0.04, 0.03)
        self.assertTrue(np.isclose(result.risk_free_weight, 1.0))
        self.assertTrue(np.allclose(result.risky_weights, 0.0))
        self.assertTrue(np.isclose(result.volatility, 0.0))

    def test_frontier_contains_only_efficient_branch(self) -> None:
        gmv = global_minimum_variance(EXPECTED_RETURNS, COVARIANCE)
        frontier = efficient_frontier(
            EXPECTED_RETURNS,
            COVARIANCE,
            maximum_return=0.17,
            number_of_points=25,
        )
        self.assertTrue(np.isclose(frontier.expected_returns[0], gmv.expected_return))
        self.assertTrue(np.all(np.diff(frontier.expected_returns) > 0))
        self.assertTrue(np.all(np.diff(frontier.volatilities) >= -1e-10))
        self.assertTrue(np.allclose(frontier.weights.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
