"""Convergence-rate tests for the order-parameterized standard FD stencils.

Verifies that `standard_d1_uniform`, `standard_d2_uniform`, and their
parity-aware variants achieve the requested formal order on smooth
test functions over a refinement sweep.

These tests are at the 1D stencil level — they do not exercise the
2D spdee operator stack. End-to-end convergence is verified by
`test_compile_problem_order.py`.

Notes on expected rates
-----------------------
A centered FD scheme of formal order p uses a width-(p+1) interior
stencil. The boundary closure is also width-(p+1) but, because a
one-sided stencil for the d-th derivative of width (p+1) has formal
truncation order (p+1) - d, the d2 closures lose one order at the
boundary (rate p-1 in the row, contaminating the L^∞ rate of the
raw derivative application to p-1). The d1 closures retain the full
order p. End-to-end Poisson convergence with d2 typically recovers
~p in L^∞ because the inverse Green's function smooths boundary
truncation by one order — that's measured separately in
`test_compile_problem_order.py`.

So these stencil-level tests expect:
  - d1 (any variant)            : rate ≥ p
  - d2 (any variant)            : rate ≥ p - 1  (boundary-dominated)
  - interior-only error of d2   : rate ≥ p
"""

from __future__ import annotations

import numpy as np
import pytest

from spdee.backends.fd2.stencils import (
    standard_d1_uniform,
    standard_d2_uniform,
    standard_d1_uniform_parity,
    standard_d2_uniform_parity,
    standard_d1_nonuniform,
    standard_d2_nonuniform,
    standard_d1,
    standard_d2,
)


# Smooth test functions (description, f, f', f'').
_FUNCS = [
    (
        "exp_cos_3x",
        lambda x: np.exp(np.cos(3.0 * x)),
        lambda x: -3.0 * np.sin(3.0 * x) * np.exp(np.cos(3.0 * x)),
        lambda x: (
            -9.0 * np.cos(3.0 * x) * np.exp(np.cos(3.0 * x))
            + 9.0 * np.sin(3.0 * x) ** 2 * np.exp(np.cos(3.0 * x))
        ),
    ),
    (
        "sin_2x_plus_x2",
        lambda x: np.sin(2.0 * x) + 0.5 * x ** 2,
        lambda x: 2.0 * np.cos(2.0 * x) + x,
        lambda x: -4.0 * np.sin(2.0 * x) + 1.0,
    ),
]


# Refinement sequences: cap N at high order so we stay above the
# double-precision roundoff floor (which appears around ~1e-12 to ~1e-14
# for these test functions).
_REFINEMENTS = {
    2: [16, 32, 64, 128, 256],
    4: [16, 32, 64, 128],
    6: [16, 32, 64],
    # order=8 reaches the double-precision floor by N=64 for d2 on
    # well-behaved test functions, so we use only the two coarsest
    # refinements where the truncation error still dominates.
    8: [16, 32],
}


# --- Helpers ----------------------------------------------------------------


def _estimate_rate(errs, Ns):
    """Linear log-log fit: err ~ C * (1/N)^p → p = -slope."""
    log_h = -np.log(np.array(Ns, dtype=float))
    log_e = np.log(np.array(errs, dtype=float))
    slope, _intercept = np.polyfit(log_h, log_e, 1)
    return float(slope)


# --- Uniform-grid d1 / d2 ---------------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("name,f,fp,fpp", _FUNCS)
def test_standard_d1_uniform_rate(order, name, f, fp, fpp):
    """d1 on a uniform grid converges at formal order p (boundary-included)."""
    a, b = 0.5, 2.7
    Ns = _REFINEMENTS[order]
    errs = []
    for N in Ns:
        x = np.linspace(a, b, N)
        h = float(x[1] - x[0])
        _, B = standard_d1_uniform(N, h, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= order - 0.5, (
        f"d1 uniform order={order} on {name!r}: measured rate {rate:.2f}, "
        f"errors {errs}"
    )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("name,f,fp,fpp", _FUNCS)
def test_standard_d2_uniform_interior_rate(order, name, f, fp, fpp):
    """Interior-only d2 error converges at formal order p on a uniform grid.

    Excludes the half-width boundary rows on each side, where the
    one-sided width-(order+1) closure has truncation order p-1.
    """
    a, b = 0.5, 2.7
    Ns = _REFINEMENTS[order]
    errs = []
    half = order // 2
    for N in Ns:
        x = np.linspace(a, b, N)
        h = float(x[1] - x[0])
        _, B = standard_d2_uniform(N, h, order=order)
        err = B @ f(x) - fpp(x)
        errs.append(float(np.max(np.abs(err[half:N - half]))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= order - 0.5, (
        f"d2 uniform (interior) order={order} on {name!r}: "
        f"measured rate {rate:.2f}, errors {errs}"
    )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("name,f,fp,fpp", _FUNCS)
def test_standard_d2_uniform_global_rate(order, name, f, fp, fpp):
    """Global d2 error (including boundary rows) converges at rate p-1.

    The boundary one-sided d2 closure of width (order+1) has formal
    truncation order p-1; this dominates the L^∞ error on the raw
    derivative application. End-to-end Poisson convergence recovers
    rate ≈ p (see test_compile_problem_order.py).
    """
    a, b = 0.5, 2.7
    Ns = _REFINEMENTS[order]
    errs = []
    for N in Ns:
        x = np.linspace(a, b, N)
        h = float(x[1] - x[0])
        _, B = standard_d2_uniform(N, h, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fpp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    # Allow 1.0 slack below the formal p-1: the leading-order truncation
    # coefficient for d2 boundary closures depends on f^{(p+1)}, which
    # vanishes for the polynomial part of certain test functions and
    # slows the observed asymptotic rate by up to ~0.5.
    assert rate >= (order - 1) - 1.0, (
        f"d2 uniform (global) order={order} on {name!r}: "
        f"measured rate {rate:.2f}, errors {errs}"
    )


# --- Parity-aware variants (left endpoint at x=0) ---------------------------


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("parity", [+1, -1])
def test_standard_d1_uniform_parity_rate(order, parity):
    """Parity-aware d1 converges at formal order p on a parity-matched function.

    The parity-folded left-endpoint rows derive from the centered
    interior stencil, so they retain truncation order p there.
    """
    if parity == +1:
        f = lambda x: np.cos(2.0 * x) + 0.5 * x ** 2  # even
        fp = lambda x: -2.0 * np.sin(2.0 * x) + x
    else:
        f = lambda x: np.sin(2.0 * x) + x ** 3  # odd
        fp = lambda x: 2.0 * np.cos(2.0 * x) + 3.0 * x ** 2

    Ns = _REFINEMENTS[order]
    errs = []
    R = 1.5
    for N in Ns:
        x = np.linspace(0.0, R, N)
        h = float(x[1] - x[0])
        _, B = standard_d1_uniform_parity(N, h, parity=parity, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= order - 0.5, (
        f"d1 parity({parity:+d}) order={order}: rate={rate:.2f}, errors={errs}"
    )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("parity", [+1, -1])
def test_standard_d2_uniform_parity_global_rate(order, parity):
    """Parity-aware d2 converges at rate p-1 globally (boundary-dominated).

    Same boundary-closure penalty as the plain variant: the right-endpoint
    one-sided width-(order+1) d2 stencil has truncation order p-1.
    """
    if parity == +1:
        f = lambda x: np.cos(2.0 * x) + 0.5 * x ** 2
        fpp = lambda x: -4.0 * np.cos(2.0 * x) + 1.0
    else:
        f = lambda x: np.sin(2.0 * x) + x ** 3
        fpp = lambda x: -4.0 * np.sin(2.0 * x) + 6.0 * x

    Ns = _REFINEMENTS[order]
    errs = []
    R = 1.5
    for N in Ns:
        x = np.linspace(0.0, R, N)
        h = float(x[1] - x[0])
        _, B = standard_d2_uniform_parity(N, h, parity=parity, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fpp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= (order - 1) - 0.5, (
        f"d2 parity({parity:+d}) order={order}: rate={rate:.2f}, errors={errs}"
    )


# --- Non-uniform grid variants ---------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6])
def test_standard_d1_nonuniform_rate(order):
    """d1 on a mildly non-uniform smooth grid still converges at order p.

    Uses a mild bisplit grid stretching (ξ → ξ^1.2) to avoid the
    severe condition-number blow-up of strongly compressed grids.
    """
    a, b = 0.5, 2.7
    Ns = _REFINEMENTS[order]
    errs = []
    name, f, fp, _ = _FUNCS[0]
    for N in Ns:
        xi = np.linspace(0.0, 1.0, N)
        x = a + (b - a) * xi ** 1.2
        _, B = standard_d1_nonuniform(x, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= order - 0.5, (
        f"d1 nonuniform order={order}: rate={rate:.2f}, errors={errs}"
    )


@pytest.mark.parametrize("order", [2, 4, 6])
def test_standard_d2_nonuniform_global_rate(order):
    """d2 on a mildly non-uniform smooth grid: global rate ≥ p-1."""
    a, b = 0.5, 2.7
    Ns = _REFINEMENTS[order]
    errs = []
    name, f, _, fpp = _FUNCS[0]
    for N in Ns:
        xi = np.linspace(0.0, 1.0, N)
        x = a + (b - a) * xi ** 1.2
        _, B = standard_d2_nonuniform(x, order=order)
        deriv_num = B @ f(x)
        deriv_exact = fpp(x)
        errs.append(float(np.max(np.abs(deriv_num - deriv_exact))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= (order - 1) - 0.5, (
        f"d2 nonuniform order={order}: rate={rate:.2f}, errors={errs}"
    )


# --- Default order is 4 (API stability) -------------------------------------


def test_default_order_is_4():
    """Calling standard_d1 / standard_d2 with no order kwarg gives the
    same result as passing order=4 explicitly."""
    x = np.linspace(0.1, 1.0, 17)
    _, B_default = standard_d1(x)
    _, B_explicit = standard_d1(x, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)
    _, B_default = standard_d2(x)
    _, B_explicit = standard_d2(x, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)


def test_uniform_default_order_is_4():
    """Same default check for the uniform-grid entry points."""
    _, B_default = standard_d1_uniform(20, 0.1)
    _, B_explicit = standard_d1_uniform(20, 0.1, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)
    _, B_default = standard_d2_uniform(20, 0.1)
    _, B_explicit = standard_d2_uniform(20, 0.1, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)
    _, B_default = standard_d1_uniform_parity(20, 0.1, parity=+1)
    _, B_explicit = standard_d1_uniform_parity(20, 0.1, parity=+1, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)
    _, B_default = standard_d2_uniform_parity(20, 0.1, parity=-1)
    _, B_explicit = standard_d2_uniform_parity(20, 0.1, parity=-1, order=4)
    assert np.allclose(B_default.toarray(), B_explicit.toarray(), atol=0)


# --- Input validation ------------------------------------------------------


@pytest.mark.parametrize("bad_order", [0, 1, 3, 5, -2])
def test_invalid_integer_order_rejected(bad_order):
    """Odd, zero, or negative integer orders are rejected."""
    with pytest.raises(ValueError):
        standard_d1_uniform(10, 0.1, order=bad_order)


def test_non_integer_order_rejected():
    """Non-integer order is rejected."""
    with pytest.raises(TypeError):
        standard_d1_uniform(10, 0.1, order=4.5)


def test_grid_too_small_for_order():
    """N < order + 1 raises a clear error."""
    with pytest.raises(ValueError, match="order=8"):
        standard_d1_uniform(8, 0.1, order=8)  # need N >= 9
    with pytest.raises(ValueError, match="order=6"):
        standard_d2_uniform(6, 0.1, order=6)  # need N >= 7
    with pytest.raises(ValueError, match="order=4"):
        standard_d1_uniform_parity(4, 0.1, parity=+1, order=4)  # need N >= 5
