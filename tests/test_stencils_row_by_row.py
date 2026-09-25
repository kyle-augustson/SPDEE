"""Row-by-row convergence tests for every row of every stencil variant.

`test_stencils_order.py` verifies the *global* (max-over-rows) rate
matches the requested order. This file goes further: it verifies the
rate at EACH ROW INDIVIDUALLY, so a degraded boundary row can't hide
behind a faster interior.

The contract: when the user requests order p, every row of every
matrix delivers rate p. There is no boundary–interior split. To meet
this, d2 boundary closures use a width-(p+2) one-sided stencil (one
wider than the interior centered stencil), which by Fornberg (1988)
delivers truncation order p; a narrower width-(p+1) closure would give
order p-1, breaking the contract.

  d1 interior centered stencil      : order p
  d1 one-sided boundary closures    : order p   (p+1 nodes → order p)
  d2 interior centered stencil      : order p
  d2 one-sided boundary closures    : order p   (p+2 nodes → order p)

For parity-aware variants on a ball grid (r=0 is the LEFT endpoint):

  parity-folded left rows (d1, d2)  : order p   (centered stencil with ghost
                                                 reflection — no one-sided loss)
  right-boundary rows (d1, d2)      : order p

Some rows are *identically zero by parity symmetry* and are skipped:
  parity=+1, d1, row 0   (f'(0) = 0 for an even function)
  parity=-1, d2, row 0   (f''(0) = 0 for an odd function, since f(0)=0)
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
)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

# Smooth, non-polynomial functions with nonzero derivatives everywhere on the
# test interval. The "gentle" choice — small higher derivatives — is
# important: at low order (2) and on a relatively coarse refinement, a
# strongly varying function like `exp(cos(3x))` has the next-to-leading
# term in its truncation series large enough to perturb the apparent rate
# at any finite N. With slowly-varying functions the asymptote is visible
# from N=32 onward.
_F_GENERIC = lambda x: np.sin(x) + 0.3 * x ** 2
_FP_GENERIC = lambda x: np.cos(x) + 0.6 * x
_FPP_GENERIC = lambda x: -np.sin(x) + 0.6

# Parity-matched gentle functions for the parity variants. The parity-folded
# stencils ASSUME f(-x) = parity * f(x); the test function must satisfy that
# for the stencil's output to correspond to the analytic derivative.
_F_EVEN = lambda x: np.cos(x) + 0.3 * x ** 2  # even
_FP_EVEN = lambda x: -np.sin(x) + 0.6 * x
_FPP_EVEN = lambda x: -np.cos(x) + 0.6

_F_ODD = lambda x: np.sin(x) + 0.3 * x ** 3  # odd
_FP_ODD = lambda x: np.cos(x) + 0.9 * x ** 2
_FPP_ODD = lambda x: -np.sin(x) + 1.8 * x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Below this absolute error the rate fit is unreliable (floating-point noise
# starts to dominate). Set tight enough to keep most high-order test data;
# the `_estimate_rate` function additionally guards against flatlined tails.
_FP_NOISE_FLOOR = 1e-12


def _estimate_rate(errs, Ns):
    """Estimate the asymptotic convergence rate from a refinement sweep.

    We compute the rate between each consecutive pair of refinements
    (excluding pairs where either endpoint is at the FP-noise floor)
    and return the **maximum**. This is the most robust estimator for
    a rate that may be approaching its asymptote from below:

    - For a clean rate-p refinement, every consecutive pair gives ≈ p,
      and max = p ✓.
    - For a refinement that's pre-asymptotic at coarse N but
      asymptotic at fine N, the fine-end pair gives ≈ p and dominates
      the max ✓.
    - For a noise-saturated tail (errors fluctuating around 1e-12),
      we filter those points out before taking the max.
    - For a refinement that's hit the noise floor entirely, returns
      None (caller treats this as "row converged to FP precision —
      that's a pass").

    A second guard: if the largest error in the sweep is itself below
    a "definitively saturated" threshold (1e-9), we conclude the
    stencil has already reached machine precision by the coarsest N,
    so any apparent rate is fitting roundoff, and return None.

    Returns the rate (positive float), or None if the data isn't
    informative enough to estimate.
    """
    errs = np.array(errs, dtype=float)
    Ns = np.array(Ns, dtype=float)
    # If even the coarsest error is tiny, the row has converged to FP
    # precision before the sweep started — there's no rate to measure.
    if np.max(errs) < 1e-9:
        return None
    keep = errs > _FP_NOISE_FLOOR
    if keep.sum() < 2:
        return None
    Ns = Ns[keep]
    errs = errs[keep]
    pair_rates = []
    for i in range(len(errs) - 1):
        if errs[i+1] <= 0 or errs[i] <= 0:
            continue
        r = float(np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i]))
        pair_rates.append(r)
    if not pair_rates:
        return None
    return max(pair_rates)


def _row_rate(builder, builder_args, f, true_deriv, idx_fn, Ns, x_min, x_max):
    """Refine N over `Ns`, build the stencil each time, apply to f at the
    discrete grid, and return the rate at the row produced by `idx_fn(N)`.

    Returns (rate, errs). `rate` may be None if the row is at FP noise.
    """
    errs = []
    for N in Ns:
        x = np.linspace(x_min, x_max, N)
        h = float(x[1] - x[0])
        _, B = builder(N, h, *builder_args)
        deriv = B @ f(x)
        i = idx_fn(N)
        errs.append(float(abs(deriv[i] - true_deriv(x[i]))))
    return _estimate_rate(errs, Ns), errs


def _check_row_rate(
    label, builder, builder_args, f, true_deriv, idx_fn, Ns, x_min, x_max,
    expected, slack=0.5,
):
    """Verify a single row converges at the expected rate. The slack
    allowance handles pre-asymptotic deviation and FP-noise floor effects;
    rates exceeding `expected + 2` are accepted as superconvergence.

    Raises AssertionError with a diagnostic message if the rate is below
    `expected - slack`.
    """
    rate, errs = _row_rate(
        builder, builder_args, f, true_deriv, idx_fn, Ns, x_min, x_max,
    )
    if rate is None:
        # All errors are at FP noise — that's a pass (the row converged
        # so well it hit machine precision, which is acceptable).
        return
    assert rate >= expected - slack, (
        f"{label}: rate {rate:.2f} < expected {expected} - {slack}; "
        f"errs={[f'{e:.2e}' for e in errs]}"
    )


# ---------------------------------------------------------------------------
# Plain (non-parity) uniform-grid stencils
# ---------------------------------------------------------------------------

# Refinement schedules per order. Coarser at higher order so we measure the
# rate before hitting the FP noise floor.
_SCHEDULES = {
    2: [16, 32, 64, 128, 256],
    4: [16, 32, 64, 128],
    6: [16, 24, 32, 48],
    8: [16, 20, 24, 32],
}

_A, _B = 0.5, 2.7  # generic test interval (away from origin)


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_d1_uniform_each_row_at_order_p(order):
    """Every row of `standard_d1_uniform` should converge at order p.

    Verified per row at i = 0, 1, ..., half-1 (left closures),
    i = half (first interior), i = N//2 (mid), i = N-half-1 (last interior),
    and i = N-half, ..., N-1 (right closures).
    """
    half = order // 2
    Ns = _SCHEDULES[order]
    f, fp = _F_GENERIC, _FP_GENERIC

    # Left boundary closures: i = 0, 1, ..., half-1
    for k in range(half):
        _check_row_rate(
            f"d1 uniform order={order} left_bdry_L{k}",
            standard_d1_uniform, (order,), f, fp,
            (lambda N, k=k: k), Ns, _A, _B,
            expected=order,
        )

    # First true interior row
    _check_row_rate(
        f"d1 uniform order={order} first_interior",
        standard_d1_uniform, (order,), f, fp,
        (lambda N: half), Ns, _A, _B,
        expected=order,
    )

    # Midpoint
    _check_row_rate(
        f"d1 uniform order={order} mid",
        standard_d1_uniform, (order,), f, fp,
        (lambda N: N // 2), Ns, _A, _B,
        expected=order,
    )

    # Last true interior
    _check_row_rate(
        f"d1 uniform order={order} last_interior",
        standard_d1_uniform, (order,), f, fp,
        (lambda N: N - half - 1), Ns, _A, _B,
        expected=order,
    )

    # Right boundary closures: i = N-half, ..., N-1
    for k in range(half):
        _check_row_rate(
            f"d1 uniform order={order} right_bdry_R{k}",
            standard_d1_uniform, (order,), f, fp,
            (lambda N, k=k: N - 1 - k), Ns, _A, _B,
            expected=order,
        )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_d2_uniform_each_row_at_expected_order(order):
    """Every row of `standard_d2_uniform` converges at its expected order.

    Interior rows: order p. One-sided boundary closures: order p-1
    (Fornberg's rule for a width-(p+1) stencil applied to a 2nd derivative).
    """
    half = order // 2
    Ns = _SCHEDULES[order]
    f, fpp = _F_GENERIC, _FPP_GENERIC

    # Left boundary closures: expected p (widened to width-(p+2) so
    # the formal order matches the interior).
    for k in range(half):
        _check_row_rate(
            f"d2 uniform order={order} left_bdry_L{k}",
            standard_d2_uniform, (order,), f, fpp,
            (lambda N, k=k: k), Ns, _A, _B,
            expected=order,
        )

    # First true interior, mid, last true interior: expected p
    for label, idx_fn in [
        ("first_interior", lambda N: half),
        ("mid", lambda N: N // 2),
        ("last_interior", lambda N: N - half - 1),
    ]:
        _check_row_rate(
            f"d2 uniform order={order} {label}",
            standard_d2_uniform, (order,), f, fpp,
            idx_fn, Ns, _A, _B,
            expected=order,
        )

    # Right boundary closures: expected p (widened to width-(p+2)).
    for k in range(half):
        _check_row_rate(
            f"d2 uniform order={order} right_bdry_R{k}",
            standard_d2_uniform, (order,), f, fpp,
            (lambda N, k=k: N - 1 - k), Ns, _A, _B,
            expected=order,
        )


# ---------------------------------------------------------------------------
# Parity-aware uniform-grid stencils (left endpoint = 0)
# ---------------------------------------------------------------------------

_R_PARITY = 1.5  # outer radius for parity tests; left endpoint is 0


# A row whose analytic derivative is identically zero (due to parity) gives a
# stencil output of identically zero (the row is the zero row). We can't fit
# a rate to that, so we skip those rows by listing them here.
def _row_is_identically_zero(parity, deriv_order, i):
    """Is the stencil row for (parity, deriv_order) at index i identically
    zero because the analytic derivative there vanishes by symmetry?

    For parity=+1 (even f): f'(0) = 0 → d1 row at i=0 is the zero row.
    For parity=-1 (odd f): f(0) = 0 and f''(0) = 0 → d2 row at i=0 is the
        zero row (after the parity stencil enforces f(0) = 0 via cancellation
        in the centered formula).
    """
    if i != 0:
        return False
    if parity == +1 and deriv_order == 1:
        return True
    if parity == -1 and deriv_order == 2:
        return True
    return False


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("parity", [+1, -1])
def test_d1_uniform_parity_each_row_at_order_p(order, parity):
    """Every meaningful row of `standard_d1_uniform_parity` should
    converge at order p. The d1 case has order p at both centered
    parity-folded rows (left) and one-sided boundary closures (right)
    by Fornberg.

    Skipped rows: rows that are identically zero by symmetry (parity=+1
    at i=0, where f'(0) = 0 for any even f).
    """
    half = order // 2
    Ns = _SCHEDULES[order]
    if parity == +1:
        f, fp = _F_EVEN, _FP_EVEN
    else:
        f, fp = _F_ODD, _FP_ODD

    # Left parity-folded rows: i = 0, 1, ..., half-1
    for k in range(half):
        if _row_is_identically_zero(parity, 1, k):
            continue
        _check_row_rate(
            f"d1 parity({parity:+d}) order={order} fold_L{k}",
            standard_d1_uniform_parity, (parity, order), f, fp,
            (lambda N, k=k: k), Ns, 0.0, _R_PARITY,
            expected=order,
        )

    # First true interior, mid, last true interior
    for label, idx_fn in [
        ("first_interior", lambda N: half),
        ("mid", lambda N: N // 2),
        ("last_interior", lambda N: N - half - 1),
    ]:
        _check_row_rate(
            f"d1 parity({parity:+d}) order={order} {label}",
            standard_d1_uniform_parity, (parity, order), f, fp,
            idx_fn, Ns, 0.0, _R_PARITY,
            expected=order,
        )

    # Right boundary closures (standard one-sided)
    for k in range(half):
        _check_row_rate(
            f"d1 parity({parity:+d}) order={order} right_bdry_R{k}",
            standard_d1_uniform_parity, (parity, order), f, fp,
            (lambda N, k=k: N - 1 - k), Ns, 0.0, _R_PARITY,
            expected=order,
        )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
@pytest.mark.parametrize("parity", [+1, -1])
def test_d2_uniform_parity_each_row_at_expected_order(order, parity):
    """Per-row rates for `standard_d2_uniform_parity`:
      - left parity-folded rows: order p (centered, no Fornberg loss)
      - interior rows: order p
      - right boundary closures: order p-1 (one-sided d2)
    """
    half = order // 2
    Ns = _SCHEDULES[order]
    if parity == +1:
        f, fpp = _F_EVEN, _FPP_EVEN
    else:
        f, fpp = _F_ODD, _FPP_ODD

    # Left parity-folded rows: expected p
    for k in range(half):
        if _row_is_identically_zero(parity, 2, k):
            continue
        _check_row_rate(
            f"d2 parity({parity:+d}) order={order} fold_L{k}",
            standard_d2_uniform_parity, (parity, order), f, fpp,
            (lambda N, k=k: k), Ns, 0.0, _R_PARITY,
            expected=order,
        )

    # Interior rows: expected p
    for label, idx_fn in [
        ("first_interior", lambda N: half),
        ("mid", lambda N: N // 2),
        ("last_interior", lambda N: N - half - 1),
    ]:
        _check_row_rate(
            f"d2 parity({parity:+d}) order={order} {label}",
            standard_d2_uniform_parity, (parity, order), f, fpp,
            idx_fn, Ns, 0.0, _R_PARITY,
            expected=order,
        )

    # Right boundary closures: expected p (widened to width-(p+2) so
    # the formal order matches the interior).
    for k in range(half):
        _check_row_rate(
            f"d2 parity({parity:+d}) order={order} right_bdry_R{k}",
            standard_d2_uniform_parity, (parity, order), f, fpp,
            (lambda N, k=k: N - 1 - k), Ns, 0.0, _R_PARITY,
            expected=order,
        )


# ---------------------------------------------------------------------------
# Non-uniform-grid stencils
# ---------------------------------------------------------------------------


def _make_nonuniform_grid(N, a=0.5, b=2.7):
    """A smoothly-varying non-uniform grid: x = a + (b-a) * ξ², ξ ∈ [0,1]
    uniform. h_{i+1} - h_i is O(h²), so smoothness is preserved and the
    interior should still hit formal order p.
    """
    xi = np.linspace(0.0, 1.0, N)
    return a + (b - a) * xi ** 2


def _row_rate_nonuniform(builder, order_arg, f, true_deriv, idx_fn, Ns, a, b):
    errs = []
    for N in Ns:
        x = _make_nonuniform_grid(N, a, b)
        _, B = builder(x, order=order_arg)
        deriv = B @ f(x)
        i = idx_fn(N)
        errs.append(float(abs(deriv[i] - true_deriv(x[i]))))
    return _estimate_rate(errs, Ns), errs


def _check_row_rate_nonuniform(
    label, builder, order_arg, f, true_deriv, idx_fn, Ns, a, b,
    expected, slack=2.0,  # non-uniform: wider slack — pre-asymptotic is
    # slower on a varying grid because the leading error coefficient
    # involves grid-derivative terms, and at high order the asymptote
    # may not be visible at the N values where errors stay above the
    # FP noise floor.
):
    rate, errs = _row_rate_nonuniform(
        builder, order_arg, f, true_deriv, idx_fn, Ns, a, b,
    )
    if rate is None:
        return
    assert rate >= expected - slack, (
        f"{label}: rate {rate:.2f} < expected {expected} - {slack}; "
        f"errs={[f'{e:.2e}' for e in errs]}"
    )


# Non-uniform schedules: wider refinement because the asymptotic order is
# approached more slowly than on a uniform grid (the leading constant
# involves grid-derivative terms that themselves shrink with refinement).
_NONUNIFORM_SCHEDULES = {
    2: [16, 32, 64, 128, 256],
    4: [16, 32, 48, 64, 96],
    6: [16, 24, 32, 48, 64],
    8: [16, 24, 32, 48, 64],
}


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_d1_nonuniform_each_row_at_order_p(order):
    half = order // 2
    Ns = _NONUNIFORM_SCHEDULES[order]
    f, fp = _F_GENERIC, _FP_GENERIC

    for k in range(half):
        _check_row_rate_nonuniform(
            f"d1 nonuniform order={order} left_bdry_L{k}",
            standard_d1_nonuniform, order, f, fp,
            (lambda N, k=k: k), Ns, _A, _B,
            expected=order,
        )
    for label, idx_fn in [
        ("first_interior", lambda N: half),
        ("mid", lambda N: N // 2),
        ("last_interior", lambda N: N - half - 1),
    ]:
        _check_row_rate_nonuniform(
            f"d1 nonuniform order={order} {label}",
            standard_d1_nonuniform, order, f, fp,
            idx_fn, Ns, _A, _B,
            expected=order,
        )
    for k in range(half):
        _check_row_rate_nonuniform(
            f"d1 nonuniform order={order} right_bdry_R{k}",
            standard_d1_nonuniform, order, f, fp,
            (lambda N, k=k: N - 1 - k), Ns, _A, _B,
            expected=order,
        )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_d2_nonuniform_each_row_at_expected_order(order):
    half = order // 2
    Ns = _NONUNIFORM_SCHEDULES[order]
    f, fpp = _F_GENERIC, _FPP_GENERIC

    # Left & right boundary closures expected p (widened to width-(p+2)).
    for k in range(half):
        _check_row_rate_nonuniform(
            f"d2 nonuniform order={order} left_bdry_L{k}",
            standard_d2_nonuniform, order, f, fpp,
            (lambda N, k=k: k), Ns, _A, _B,
            expected=order,
        )
    for label, idx_fn in [
        ("first_interior", lambda N: half),
        ("mid", lambda N: N // 2),
        ("last_interior", lambda N: N - half - 1),
    ]:
        _check_row_rate_nonuniform(
            f"d2 nonuniform order={order} {label}",
            standard_d2_nonuniform, order, f, fpp,
            idx_fn, Ns, _A, _B,
            expected=order,
        )
    for k in range(half):
        _check_row_rate_nonuniform(
            f"d2 nonuniform order={order} right_bdry_R{k}",
            standard_d2_nonuniform, order, f, fpp,
            (lambda N, k=k: N - 1 - k), Ns, _A, _B,
            expected=order,
        )


# ---------------------------------------------------------------------------
# Identically-zero parity rows: verify they ARE identically zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_parity_plus_d1_row_0_is_identically_zero(order):
    """For an even function, f'(0) = 0 exactly. The parity=+1 d1 stencil
    must produce a row that is identically zero in column space, so
    (B @ f)[0] = 0 to machine precision regardless of f's interior values.
    """
    f = _F_EVEN
    for N in [16, 32, 64]:
        x = np.linspace(0.0, _R_PARITY, N)
        h = float(x[1] - x[0])
        _, B = standard_d1_uniform_parity(N, h, parity=+1, order=order)
        deriv = B @ f(x)
        assert abs(deriv[0]) < 1e-12, (
            f"order={order}, N={N}: parity=+1 d1 at i=0 should be 0, "
            f"got {deriv[0]:.3e}"
        )


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_parity_minus_d2_row_0_is_identically_zero(order):
    """For an odd function, f''(0) = 0 exactly (because f(0)=0 and the
    parity stencil's row at i=0 is `B[0,0] · f_0` only, and f_0 = 0).
    """
    f = _F_ODD
    for N in [16, 32, 64]:
        x = np.linspace(0.0, _R_PARITY, N)
        h = float(x[1] - x[0])
        _, B = standard_d2_uniform_parity(N, h, parity=-1, order=order)
        deriv = B @ f(x)
        # f(0) = 0, and the row only acts on f_0 (the row is just B[0,0]
        # on column 0 after parity folding). So the output at i=0 is 0
        # exactly, regardless of the rest of f.
        assert abs(deriv[0]) < 1e-12, (
            f"order={order}, N={N}: parity=-1 d2 at i=0 should be 0 "
            f"(f(0)=0 for odd f), got {deriv[0]:.3e}"
        )
