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


def test_weights_are_invariant_to_covariance_scale():
    weights = minimum_variance_weights(np.diag([1e-16, 4e-16]))
    np.testing.assert_allclose(weights, [0.8, 0.2], atol=1e-7)


def test_material_asymmetry_is_rejected_relative_to_matrix_scale():
    covariance = np.array([[1.0, 1e-6], [0.0, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        minimum_variance_weights(covariance)


def test_psd_check_is_scale_relative():
    covariance = np.array([[1e-16, 2e-16], [2e-16, 1e-16]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        minimum_variance_weights(covariance)


def test_nonconvergence_is_not_returned_as_a_successful_solution():
    with pytest.raises(RuntimeError, match="did not converge"):
        minimum_variance_weights(np.diag([1.0, 4.0]), max_iterations=1)


@pytest.mark.parametrize("tolerance,max_iterations", [(np.nan, 10), (1e-8, 1.5), (0.0, 10)])
def test_invalid_optimizer_controls_are_rejected(tolerance, max_iterations):
    with pytest.raises(ValueError, match="tolerance and max_iterations"):
        minimum_variance_weights(np.eye(2), tolerance=tolerance, max_iterations=max_iterations)

