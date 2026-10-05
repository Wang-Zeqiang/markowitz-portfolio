import unittest

import numpy as np

from markowitz import efficient_frontier, global_minimum_variance
from markowitz.metrics import capm_expected_returns, historical_beta


class ValidationTests(unittest.TestCase):
    def test_rejects_nonsymmetric_covariance(self) -> None:
        mu = np.array([0.1, 0.2])
        covariance = np.array([[0.04, 0.01], [0.02, 0.09]])
        with self.assertRaisesRegex(ValueError, "symmetric"):
            global_minimum_variance(mu, covariance)

    def test_rejects_singular_covariance(self) -> None:
        mu = np.array([0.1, 0.2])
        covariance = np.array([[0.04, 0.04], [0.04, 0.04]])
        with self.assertRaisesRegex(ValueError, "positive definite"):
            global_minimum_variance(mu, covariance)

    def test_rejects_bad_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            global_minimum_variance([0.1, 0.2], np.eye(3))

    def test_rejects_too_few_frontier_points(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            efficient_frontier(
                [0.1, 0.2],
                [[0.04, 0.0], [0.0, 0.09]],
                number_of_points=1,
            )

    def test_historical_beta_and_capm_returns(self) -> None:
        market = np.array([-0.02, 0.01, 0.03, -0.01, 0.02])
        assets = np.column_stack((market, 2.0 * market))
        beta = historical_beta(assets, market)
        predicted = capm_expected_returns(beta, 0.03, 0.09)
        self.assertTrue(np.allclose(beta, [1.0, 2.0]))
        self.assertTrue(np.allclose(predicted, [0.09, 0.15]))


if __name__ == "__main__":
    unittest.main()
