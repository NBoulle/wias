"""Linear algebra helpers used by weak-SINDy workflows."""

from __future__ import annotations

import numpy as np
import scipy.linalg


def remove_dependent_columns(A, rtol=1e-10):
    """Remove numerically dependent columns using pivoted QR."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    _, R, piv = scipy.linalg.qr(A, pivoting=True, mode="economic")
    diagonal = np.abs(np.diag(R))
    if diagonal.size == 0 or diagonal[0] == 0.0:
        raise ValueError("System is rank-deficient with no usable columns.")

    rank = int(np.sum(diagonal > rtol * diagonal[0]))
    independent_cols = np.sort(piv[:rank])
    return A[:, independent_cols], independent_cols


def solve_reduced_system(A_reduced, b, independent_cols, n_cols):
    """Solve reduced least-squares system and expand to full coefficient vector."""
    A_reduced = np.asarray(A_reduced, dtype=float)
    b = np.asarray(b, dtype=float)
    independent_cols = np.asarray(independent_cols, dtype=int)

    if A_reduced.ndim != 2:
        raise ValueError("A_reduced must be a 2D array")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array")
    if A_reduced.shape[0] != b.shape[0]:
        raise ValueError("A_reduced and b must have the same number of rows")

    coeffs_reduced, *_ = np.linalg.lstsq(A_reduced, b, rcond=None)
    coeffs = np.zeros(int(n_cols), dtype=float)
    coeffs[independent_cols] = coeffs_reduced
    return coeffs


def CPQR_solver(A, b, rtol=1e-10):
    """
    Solve A @ c = b using Column-Pivoted QR decomposition.

    Algorithm:
        1. Compute pivoted QR: A P = Q R
        2. Identify first r pivots (independent columns)
        3. Form reduced triangular system: R_{1:r, 1:r} y = Q_{:, 1:r}^T b
        4. Recover full coefficient vector by undoing permutation

    Parameters
    ----------
    A : ndarray
        Design matrix of shape (m, n)
    b : ndarray
        Right-hand side of shape (m,)
    rtol : float
        Relative tolerance for rank determination

    Returns
    -------
    c : ndarray
        Coefficient vector of shape (n,) with dependent columns set to zero
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b must have the same number of rows")

    A_reduced, independent_cols = remove_dependent_columns(A, rtol=rtol)
    print("Condition number of reduced system:", np.linalg.cond(A_reduced))
    return solve_reduced_system(A_reduced, b, independent_cols, A.shape[1])


def recover_homogeneous_relation(A, normalization_index=0):
    """Return smallest-singular-vector relation for homogeneous A @ theta ~= 0."""
    _, singular_values, vh = scipy.linalg.svd(A, full_matrices=False)
    print(f"Singular values: {singular_values}")
    coeffs = vh[-1, :]

    if normalization_index is None:
        pivot_index = int(np.argmax(np.abs(coeffs)))
        if coeffs[pivot_index] < 0.0:
            coeffs *= -1.0
    else:
        if abs(coeffs[normalization_index]) < 1e-12:
            raise ValueError("Cannot normalize with near-zero reference coefficient.")

        coeffs /= coeffs[normalization_index]
        if coeffs[normalization_index] < 0.0:
            coeffs *= -1.0

    residual = np.linalg.norm(A @ coeffs)
    return coeffs, singular_values, residual
