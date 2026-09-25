"""Tests for spdee.backends.fd2.stencils."""

import numpy as np
import scipy.sparse.linalg as spla

from spdee.backends.fd2.stencils import (
    standard_d1,
    standard_d1_uniform,
    standard_d1_nonuniform)


# ===========================================================================
# Helpers
# ===========================================================================

def _apply(A, B, f):
    """Compute f' = A^{-1} B f."""
    return spla.spsolve(A.tocsr(), B @ f)


def _convergence_rate(errors, hs):
    """Estimate the order of convergence from a list of errors and step sizes."""
    # Fit log(error) = p * log(h) + c
    logs_h = np.log(hs)
    logs_e = np.log(np.array(errors).clip(min=1e-300))
    slope, _ = np.polyfit(logs_h, logs_e, 1)
    return slope


# ===========================================================================
# Compact 4th-order on uniform grid
# ===========================================================================

# ===========================================================================
# Standard 4th-order on uniform grid
# ===========================================================================

def test_standard_d1_uniform_exact_on_cubic():
    N = 20
    h = 0.1
    x = np.linspace(0, (N - 1) * h, N)
    f = x ** 3
    A, B = standard_d1_uniform(N, h)
    fprime = _apply(A, B, f)
    err = np.max(np.abs(fprime - 3 * x ** 2))
    assert err < 1e-10


def test_standard_d1_uniform_4th_order_convergence_sin():
    errors, hs = [], []
    for N in [16, 32, 64, 128]:
        h = 2 * np.pi / (N - 1)
        x = np.linspace(0, 2 * np.pi, N)
        f = np.sin(x)
        A, B = standard_d1_uniform(N, h)
        fprime = _apply(A, B, f)
        interior = slice(3, -3)
        err = np.max(np.abs(fprime[interior] - np.cos(x[interior])))
        errors.append(err)
        hs.append(h)
    rate = _convergence_rate(errors, hs)
    assert rate > 3.5, f"Expected ~4th order; got rate = {rate:.3f}"


def test_standard_d1_A_is_identity():
    A, B = standard_d1_uniform(20, 0.1)
    A_d = A.toarray()
    assert np.allclose(A_d, np.eye(20))


def _max_offdiag_band(M):
    """Largest |i - j| at which M has any nonzero entry."""
    max_band = 0
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] != 0 and abs(i - j) > max_band:
                max_band = abs(i - j)
    return max_band


# ===========================================================================
# Non-uniform-grid stencils
# ===========================================================================

def test_standard_d1_nonuniform_uniform_agrees():
    N = 20
    h = 0.1
    x = np.linspace(0, (N - 1) * h, N)
    A_u, B_u = standard_d1_uniform(N, h)
    A_n, B_n = standard_d1_nonuniform(x)
    assert np.allclose(A_u.toarray(), A_n.toarray(), atol=1e-12)
    assert np.allclose(B_u.toarray(), B_n.toarray(), atol=1e-10)


def test_standard_d1_nonuniform_on_chebyshev_grid():
    N = 20
    i = np.arange(N)
    x = 0.5 * (1 - np.cos(i * np.pi / (N - 1)))
    f = x ** 3
    A, B = standard_d1_nonuniform(x)
    fprime = _apply(A, B, f)
    err = np.max(np.abs(fprime - 3 * x ** 2))
    assert err < 1e-9, f"Cubic should be exact; err = {err:.2e}"


# ===========================================================================
# Auto-dispatching wrappers
# ===========================================================================

def test_standard_d1_auto_uniform():
    x = np.linspace(0, 1, 20)
    A, B = standard_d1(x)
    f = x ** 3
    fprime = _apply(A, B, f)
    assert np.max(np.abs(fprime - 3 * x ** 2)) < 1e-10


# ===========================================================================
# Compact vs standard: bandwidth advantage in eigenvalue context
# ===========================================================================
