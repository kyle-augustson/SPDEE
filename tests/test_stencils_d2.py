"""Tests for 2nd-derivative compact and standard FD stencils."""

import numpy as np
import scipy.sparse.linalg as spla

from spdee.backends.fd2.stencils import (
    standard_d2,
    standard_d2_uniform,
    standard_d2_nonuniform)


def _apply(A, B, f):
    return spla.spsolve(A.tocsc(), B @ f)


def _convergence_rate(errors, hs):
    logs_h = np.log(hs)
    logs_e = np.log(np.array(errors).clip(min=1e-300))
    slope, _ = np.polyfit(logs_h, logs_e, 1)
    return slope


# ---------------------------------------------------------------------------
# Compact 2nd derivative
# ---------------------------------------------------------------------------

def test_standard_d2_uniform_exact_on_cubic():
    N = 20
    h = 0.1
    x = np.linspace(0, (N - 1) * h, N)
    f = x ** 3
    A, B = standard_d2_uniform(N, h)
    fpp = _apply(A, B, f)
    err = np.max(np.abs(fpp - 6 * x))
    assert err < 1e-9


def test_standard_d2_uniform_4th_order_convergence_sin():
    errors, hs = [], []
    for N in [16, 32, 64, 128]:
        h = 2 * np.pi / (N - 1)
        x = np.linspace(0, 2 * np.pi, N)
        f = np.sin(x)
        A, B = standard_d2_uniform(N, h)
        fpp = _apply(A, B, f)
        interior = slice(5, -5)
        err = np.max(np.abs(fpp[interior] - (-np.sin(x[interior]))))
        errors.append(err)
        hs.append(h)
    rate = _convergence_rate(errors, hs)
    assert rate > 3.5, f"Expected ~4th order; got rate = {rate:.3f}"


def test_standard_d2_A_is_identity():
    A, B = standard_d2_uniform(20, 0.1)
    assert np.allclose(A.toarray(), np.eye(20))


# ---------------------------------------------------------------------------
# Non-uniform 2nd derivative
# ---------------------------------------------------------------------------

def test_standard_d2_nonuniform_uniform_agrees():
    N = 20
    h = 0.1
    x = np.linspace(0, (N - 1) * h, N)
    A_u, B_u = standard_d2_uniform(N, h)
    A_n, B_n = standard_d2_nonuniform(x)
    assert np.allclose(A_u.toarray(), A_n.toarray(), atol=1e-12)
    assert np.allclose(B_u.toarray(), B_n.toarray(), atol=1e-9)


def test_standard_d2_auto_uniform():
    x = np.linspace(0, 1, 20)
    A, B = standard_d2(x)
    f = x ** 3
    fpp = _apply(A, B, f)
    assert np.max(np.abs(fpp - 6 * x)) < 1e-10

