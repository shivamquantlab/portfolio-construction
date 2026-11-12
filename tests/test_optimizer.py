import numpy as np
import pytest

from portfolio_construction.optimizer import minimum_variance_weights


def test_diagonal_minimum_variance_has_inverse_variance_weights():
    weights = minimum_variance_weights(np.diag([1.0, 4.0]))
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-7)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_identical_assets_return_a_valid_simplex_solution():
    weights = minimum_variance_weights(np.eye(3))
    np.testing.assert_allclose(weights, np.full(3, 1 / 3), atol=1e-7)


def test_indefinite_covariance_is_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        minimum_variance_weights(np.array([[1.0, 2.0], [2.0, 1.0]]))

