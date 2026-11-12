"""Projected-gradient minimum-variance optimization on the long-only simplex."""

from numbers import Integral, Real

import numpy as np


def _project_simplex(vector: np.ndarray) -> np.ndarray:
    ordered = np.sort(vector)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, vector.size + 1)
    valid = ordered - cumulative / indices > 0
    rho = np.flatnonzero(valid)[-1]
    threshold = cumulative[rho] / (rho + 1)
    return np.maximum(vector - threshold, 0.0)


def minimum_variance_weights(
    covariance: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 20_000,
) -> np.ndarray:
    """Minimize w'Σw subject to w>=0 and sum(w)=1.

    Covariance inputs must be estimated from data available before the
    portfolio holding period. This function performs no return estimation.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite and symmetric")
    scale = float(np.max(np.abs(matrix)))
    symmetry_tolerance = np.finfo(float).eps * max(1, matrix.shape[0]) * scale * 10.0
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=symmetry_tolerance):
        raise ValueError("covariance must be finite and symmetric")
    if (
        isinstance(tolerance, (bool, np.bool_))
        or not isinstance(tolerance, Real)
        or not np.isfinite(tolerance)
        or float(tolerance) <= 0
        or isinstance(max_iterations, (bool, np.bool_))
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError("tolerance and max_iterations must be positive")
    if scale == 0.0:
        return np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    normalized = 0.5 * (matrix / scale + (matrix / scale).T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    psd_tolerance = np.finfo(float).eps * max(1, matrix.shape[0]) * 100.0
    if eigenvalues[0] < -psd_tolerance:
        raise ValueError("covariance must be positive semidefinite")
    largest = float(eigenvalues[-1])
    step_size = 1.0 / (2.0 * largest)
    weights = np.full(matrix.shape[0], 1.0 / matrix.shape[0])
    for _ in range(max_iterations):
        updated = _project_simplex(weights - step_size * (2.0 * normalized @ weights))
        if np.linalg.norm(updated - weights, ord=2) <= tolerance:
            return updated
        weights = updated
    raise RuntimeError("minimum-variance optimizer did not converge within max_iterations")

