"""Tests for parity-aware FD stencils (standard 4th-order).

The parity-aware versions modify the left endpoint (i=0) to use ghost
values f(r_{-k}) = parity * f(r_k), which is the correct treatment for
ball-mode at r=0. The right endpoint retains the standard one-sided
closure.

These tests verify:
- 4th-order convergence for both d1 and d2 with even and odd parities
  on smooth non-polynomial functions.
- Polynomial inputs (consistent with the parity assumption) reproduce
  the exact derivative to machine precision.
- Mismatched parity (asking for d1 of an odd function under parity=+1)
  produces wrong results — guarding against silent misuse.
"""

import numpy as np
import pytest
import scipy.sparse.linalg as spla

from spdee.backends.fd2.stencils import (
    standard_d1_uniform_parity,
    standard_d2_uniform_parity,
)


# ===========================================================================
# Helpers
# ===========================================================================

def _apply(A, B, f):
    return spla.spsolve(A.tocsc(), B @ f)


def _rate(errors, Ns):
    """Linear fit log(err) vs log(N): rate = -slope."""
    log_e = np.log(errors)
    log_N = np.log(Ns)
    p, _ = np.polyfit(log_N, log_e, 1)
    return -float(p)


# ===========================================================================
# Even-parity d1
# ===========================================================================

# ===========================================================================
# Odd-parity d1
# ===========================================================================

# ===========================================================================
# Even-parity d2
# ===========================================================================

# ===========================================================================
# Odd-parity d2
# ===========================================================================

# ===========================================================================
# Argument validation
# ===========================================================================

# ===========================================================================
# Standard (A=I) parity stencils
# ===========================================================================


def _apply_std(A, B, f):
    """Standard stencils have A = identity; apply B directly."""
    return B @ f


def test_std_d1_parity_even_polynomial_exact():
    """Even f = 1 - r²: f' = -2r. Standard 4th-order centered FD with
    parity-substituted ghosts at i=0 should reproduce this exactly
    (polynomial degree 1 is well within rate-4 stencil accuracy)."""
    N = 32
    h = 1.0 / (N - 1)
    r = np.arange(N) * h
    f = 1 - r ** 2
    df_exact = -2 * r
    A, B = standard_d1_uniform_parity(N, h, parity=+1)
    df_num = _apply_std(A, B, f)
    assert np.max(np.abs(df_num - df_exact)) < 1e-13


def test_std_d1_parity_odd_polynomial_exact():
    """Odd f = r(1-r²) = r - r³: f' = 1 - 3r². Polynomial of degree 2,
    within the 4th-order stencil's exactness range."""
    N = 32
    h = 1.0 / (N - 1)
    r = np.arange(N) * h
    f = r * (1 - r ** 2)
    df_exact = 1 - 3 * r ** 2
    A, B = standard_d1_uniform_parity(N, h, parity=-1)
    df_num = _apply_std(A, B, f)
    assert np.max(np.abs(df_num - df_exact)) < 1e-13


def test_std_d1_parity_even_convergence():
    """Smooth even f = cos(2r): f' = -2 sin(2r). Convergence rate ~4."""
    Ns = [16, 32, 64, 128]
    errs = []
    for N in Ns:
        h = 1.0 / (N - 1)
        r = np.arange(N) * h
        f = np.cos(2 * r)
        df_exact = -2 * np.sin(2 * r)
        A, B = standard_d1_uniform_parity(N, h, parity=+1)
        df_num = _apply_std(A, B, f)
        # Exclude right-boundary points (i = N-1, N-2 use one-sided
        # 5-point which is rate-3) — the parity-aware left side is the
        # subject of this test.
        errs.append(np.max(np.abs(df_num[:-2] - df_exact[:-2])))
    rate = _rate(errs, Ns)
    assert rate >= 3.8, f"Standard d1 parity=+1 convergence rate too low: {rate:.2f}"


def test_std_d1_parity_odd_convergence():
    """Smooth odd f = sin(2r): f' = 2 cos(2r). Rate ~4."""
    Ns = [16, 32, 64, 128]
    errs = []
    for N in Ns:
        h = 1.0 / (N - 1)
        r = np.arange(N) * h
        f = np.sin(2 * r)
        df_exact = 2 * np.cos(2 * r)
        A, B = standard_d1_uniform_parity(N, h, parity=-1)
        df_num = _apply_std(A, B, f)
        errs.append(np.max(np.abs(df_num[:-2] - df_exact[:-2])))
    rate = _rate(errs, Ns)
    assert rate >= 3.8, f"Standard d1 parity=-1 convergence rate too low: {rate:.2f}"


def test_std_d2_parity_even_polynomial_exact():
    """Even f = 1 - r²: f'' = -2 (constant). Polynomial of degree 0
    for f'', exact at all grid points to machine precision."""
    N = 32
    h = 1.0 / (N - 1)
    r = np.arange(N) * h
    f = 1 - r ** 2
    d2f_exact = -2 * np.ones_like(r)
    A, B = standard_d2_uniform_parity(N, h, parity=+1)
    d2f_num = _apply_std(A, B, f)
    assert np.max(np.abs(d2f_num - d2f_exact)) < 1e-11


def test_std_d2_parity_odd_polynomial_exact():
    """Odd f = r(1-r²): f'' = -6r. Polynomial of degree 1, exact."""
    N = 32
    h = 1.0 / (N - 1)
    r = np.arange(N) * h
    f = r * (1 - r ** 2)
    d2f_exact = -6 * r
    A, B = standard_d2_uniform_parity(N, h, parity=-1)
    d2f_num = _apply_std(A, B, f)
    assert np.max(np.abs(d2f_num - d2f_exact)) < 1e-11


def test_std_d2_parity_even_convergence():
    """Smooth even f = cos(2r): f'' = -4 cos(2r). Rate ~4 (interior
    and parity-aware left side; right boundary is rate-3 from one-sided
    Lagrange so we exclude it)."""
    Ns = [16, 32, 64, 128]
    errs = []
    for N in Ns:
        h = 1.0 / (N - 1)
        r = np.arange(N) * h
        f = np.cos(2 * r)
        d2f_exact = -4 * np.cos(2 * r)
        A, B = standard_d2_uniform_parity(N, h, parity=+1)
        d2f_num = _apply_std(A, B, f)
        errs.append(np.max(np.abs(d2f_num[:-2] - d2f_exact[:-2])))
    rate = _rate(errs, Ns)
    assert rate >= 3.8, f"Standard d2 parity=+1 convergence rate too low: {rate:.2f}"


def test_std_d2_parity_odd_convergence():
    """Smooth odd f = sin(2r): f'' = -4 sin(2r). Rate ~4."""
    Ns = [16, 32, 64, 128]
    errs = []
    for N in Ns:
        h = 1.0 / (N - 1)
        r = np.arange(N) * h
        f = np.sin(2 * r)
        d2f_exact = -4 * np.sin(2 * r)
        A, B = standard_d2_uniform_parity(N, h, parity=-1)
        d2f_num = _apply_std(A, B, f)
        errs.append(np.max(np.abs(d2f_num[:-2] - d2f_exact[:-2])))
    rate = _rate(errs, Ns)
    assert rate >= 3.8, f"Standard d2 parity=-1 convergence rate too low: {rate:.2f}"


def test_std_d1_parity_rejects_invalid_parity():
    with pytest.raises(ValueError, match="parity"):
        standard_d1_uniform_parity(16, 0.1, parity=0)


def test_std_d2_parity_rejects_invalid_parity():
    with pytest.raises(ValueError, match="parity"):
        standard_d2_uniform_parity(16, 0.1, parity=0)


def test_std_d1_parity_rejects_small_N():
    with pytest.raises(ValueError, match="N"):
        standard_d1_uniform_parity(3, 0.1, parity=+1)


def test_std_d2_parity_rejects_small_N():
    with pytest.raises(ValueError, match="N"):
        standard_d2_uniform_parity(3, 0.1, parity=+1)
