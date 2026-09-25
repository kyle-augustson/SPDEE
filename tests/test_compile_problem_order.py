"""End-to-end convergence tests for the `order_r` / `order_theta`
parameters of `compile_problem`.

Solves manufactured Poisson problems on shell and ball grids and
checks that the empirical convergence rate matches the requested
formal order in each axis independently.

These tests exercise the full operator stack: stencil construction,
sparse operator builders (Lap, Grad, Div), boundary-condition
dispatch, and origin regularity at r=0 for ball grids.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Background, Unknown, r, theta,
    Lap,
    Equation, BoundaryCondition, EquationSystem, Boundary,
)
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spherical_lap(u_expr):
    """Symbolic spherical-Laplacian (m=0 axisymmetric) of u_expr(r, θ)."""
    return sp.simplify(
        (1 / r ** 2) * sp.diff(r ** 2 * sp.diff(u_expr, r), r)
        + (1 / (r ** 2 * sp.sin(theta)))
        * sp.diff(sp.sin(theta) * sp.diff(u_expr, theta), theta)
    )


def _estimate_rate(errs, Ns):
    """Estimate the asymptotic convergence rate from a refinement sweep.

    We compute the rate between each consecutive pair of refinements
    and return the **maximum**. This is more robust than a polyfit at
    moderate N where pre-asymptotic deviations can pull the fitted
    slope below the true asymptotic order.

    For a rate-p sequence err_i = C / N_i^p, every consecutive pair
    gives the same rate p; max = p ✓. For a sequence with strong
    pre-asymptotic deviation (e.g., higher-order corrections in the
    truncation series at coarse N), the fine-end pairs converge to
    rate p while coarse-end pairs may not — max captures the
    asymptote ✓. The estimator does not protect against noise
    saturation; tests using it should choose schedules that stay
    well above FP precision.
    """
    errs = np.array(errs, dtype=float)
    Ns = np.array(Ns, dtype=float)
    pair_rates = []
    for i in range(len(errs) - 1):
        if errs[i] <= 0 or errs[i+1] <= 0:
            continue
        r = float(np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i]))
        pair_rates.append(r)
    if not pair_rates:
        return 0.0
    return max(pair_rates)


def _build_shell_poisson(u_sym):
    """Set up u, f, and the Equation for manufactured Lap u = f on a
    shell with full Dirichlet BCs. Returns (sys_factory, u_exact_fn).

    sys_factory(grid) → EquationSystem.
    u_exact_fn(R, T) → np.ndarray.
    """
    lap_u = _spherical_lap(u_sym)
    u = Unknown.Scalar("u")
    f = Background.Scalar("f", expr=lap_u)
    eq = Equation(Lap(u) - f, name="poisson")

    r_in_val = sp.Rational(3, 10)
    r_out_val = sp.Integer(1)
    u_in = u_sym.subs(r, r_in_val)
    u_out = u_sym.subs(r, r_out_val)

    def sys_factory():
        bcs = [
            BoundaryCondition(Boundary.R_INNER, u - u_in, name="in"),
            BoundaryCondition(Boundary.R_OUTER, u - u_out, name="out"),
        ]
        return EquationSystem([eq], bcs)

    # Build the exact-solution evaluator from u_sym.
    u_lam = sp.lambdify((r, theta), u_sym, modules="numpy")

    def u_exact(R, T):
        return u_lam(R, T)

    return sys_factory, u_exact


# ---------------------------------------------------------------------------
# Radial order: refine N_r at fixed N_θ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_r", [2, 4, 6])
def test_order_r_convergence_shell(order_r):
    """Manufactured Poisson with strong radial structure; refine N_r
    while holding N_θ fixed (and order_theta high enough that θ error
    doesn't pollute the rate measurement).

    Asserts measured rate ≥ order_r - 0.5.
    """
    u_sym = sp.sin(sp.pi * r) * sp.cos(theta)
    sys_factory, u_exact = _build_shell_poisson(u_sym)

    # Refinement schedule per order: coarser for higher orders to stay
    # well above the precision floor. N_θ and order_theta are large
    # enough that the angular error contribution stays below the radial
    # error across the whole sweep.
    schedules = {
        2: [16, 32, 64, 128],
        4: [16, 24, 32, 48],
        6: [12, 16, 20, 24],
    }
    Ns = schedules[order_r]
    errs = []
    for N in Ns:
        grid = make_shell_grid(N_r=N, N_theta=64, r_inner=0.3, r_outer=1.0)
        sys_ = sys_factory()
        cp = compile_problem(sys_, grid, m=0, order_r=order_r, order_theta=8)
        res = cp.solve_bvp()
        R, T = grid.mesh()
        err = float(np.max(np.abs(res["u"] - u_exact(R, T))))
        errs.append(err)
    rate = _estimate_rate(errs, Ns)
    assert rate >= order_r - 0.5, (
        f"order_r={order_r} measured rate {rate:.2f}; errs={errs}"
    )


# ---------------------------------------------------------------------------
# Angular order: refine N_θ at fixed N_r
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_theta", [2, 4, 6])
def test_order_theta_convergence_shell(order_theta):
    """Manufactured Poisson with strong θ structure; refine N_θ while
    holding N_r large and order_r high so r-error is negligible.

    Asserts measured rate ≥ order_theta - 0.5.
    """
    # Higher-frequency θ structure so the rate is observable in the
    # asymptotic regime.
    u_sym = sp.sin(sp.pi * r) * sp.cos(4 * theta)
    sys_factory, u_exact = _build_shell_poisson(u_sym)

    schedules = {
        2: [24, 48, 96, 192],
        4: [24, 32, 48, 64],
        6: [20, 28, 36, 44],
    }
    Ns = schedules[order_theta]
    errs = []
    for Nth in Ns:
        grid = make_shell_grid(N_r=128, N_theta=Nth, r_inner=0.3, r_outer=1.0)
        sys_ = sys_factory()
        cp = compile_problem(
            sys_, grid, m=0, order_r=6, order_theta=order_theta,
        )
        res = cp.solve_bvp()
        R, T = grid.mesh()
        err = float(np.max(np.abs(res["u"] - u_exact(R, T))))
        errs.append(err)
    rate = _estimate_rate(errs, Ns)
    assert rate >= order_theta - 0.5, (
        f"order_theta={order_theta} measured rate {rate:.2f}; errs={errs}"
    )


# ---------------------------------------------------------------------------
# Independence of radial and angular orders
# ---------------------------------------------------------------------------


def test_anisotropic_orders_independent():
    """The radial and angular orders can be chosen independently.

    Concretely: with (order_r=6, order_theta=2), refining N_θ should
    produce a rate consistent with the θ-order (≈ 2). Refining N_r
    alone with this (order_r, order_theta) pair would be limited by
    the much larger θ-error of order_theta=2; that's the expected
    behavior and isn't asserted here.

    The complementary check — that the r-order parameter is actually
    being used downstream — is covered by
    `test_order_r_convergence_shell` at the various `order_r` values.
    """
    u_sym = sp.sin(sp.pi * r) * sp.cos(3 * theta)
    sys_factory, u_exact = _build_shell_poisson(u_sym)

    # θ-refinement at order_theta=2: expect rate ≈ 2.
    Nths = [32, 64, 128, 256]
    errs_theta = []
    for Nth in Nths:
        grid = make_shell_grid(N_r=96, N_theta=Nth, r_inner=0.3, r_outer=1.0)
        sys_ = sys_factory()
        cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=2)
        res = cp.solve_bvp()
        R, T = grid.mesh()
        errs_theta.append(
            float(np.max(np.abs(res["u"] - u_exact(R, T))))
        )
    rate_theta = _estimate_rate(errs_theta, Nths)
    assert rate_theta >= 1.5, (
        f"order_theta=2 anisotropic rate {rate_theta:.2f}; errs={errs_theta}"
    )


# ---------------------------------------------------------------------------
# Defaults stay at order=4
# ---------------------------------------------------------------------------


def test_default_order_matches_explicit_4():
    """compile_problem with no order kwargs gives the same matrices and
    solution as compile_problem(..., order_r=4, order_theta=4)."""
    u_sym = sp.sin(sp.pi * r) * sp.cos(2 * theta)
    sys_factory, _ = _build_shell_poisson(u_sym)
    grid = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)

    cp_default = compile_problem(sys_factory(), grid, m=0)
    cp_explicit = compile_problem(
        sys_factory(), grid, m=0, order_r=4, order_theta=4,
    )

    A_d = cp_default.matrices.A_total.toarray()
    A_e = cp_explicit.matrices.A_total.toarray()
    # Bit-exact match — at order=4 the new stencils were tuned to
    # reproduce the canonical hardcoded coefficient tables exactly,
    # which is load-bearing for cancellation-sensitive solves.
    assert np.array_equal(A_d, A_e), (
        f"default vs order=4 matrices differ; max diff "
        f"{np.max(np.abs(A_d - A_e))}"
    )


# ---------------------------------------------------------------------------
# Ball geometry: convergence at r=0 with parity-aware origin treatment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order_r", [2, 4, 6])
def test_order_r_convergence_ball(order_r):
    """Manufactured Poisson on a ball grid; refine N_r at fixed N_θ.

    The exact solution must be Cartesian-smooth at r=0. We use
    u = sin(r cos θ) · (1 - r²) = sin(z) · (1 - r²), which is
    analytic in Cartesian coordinates (a function of z and r²)
    and is non-polynomial — so the error doesn't collapse to
    machine precision at low N_r for high orders.
    """
    u_sym = sp.sin(r * sp.cos(theta)) * (1 - r ** 2)
    lap_u = _spherical_lap(u_sym)

    u = Unknown.Scalar("u")
    f = Background.Scalar("f", expr=lap_u)
    eq = Equation(Lap(u) - f, name="poisson")

    # u(r=1) = 0 (the radial cutoff vanishes); the origin gets the
    # parity-aware origin regularity treatment automatically.
    u_out_expr = u_sym.subs(r, sp.Integer(1))

    u_lam = sp.lambdify((r, theta), u_sym, modules="numpy")

    schedules = {
        2: [16, 32, 64, 128],
        4: [16, 24, 32, 48],
        6: [12, 16, 20, 24],
    }
    Ns = schedules[order_r]
    errs = []
    for N in Ns:
        grid = make_ball_grid(N_r=N, N_theta=64, r_outer=1.0)
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, u - u_out_expr, name="out"),
        ]
        sys_ = EquationSystem([eq], bcs)
        cp = compile_problem(sys_, grid, m=0, order_r=order_r, order_theta=8)
        res = cp.solve_bvp()
        R, T = grid.mesh()
        errs.append(float(np.max(np.abs(res["u"] - u_lam(R, T)))))
    rate = _estimate_rate(errs, Ns)
    assert rate >= order_r - 0.5, (
        f"ball order_r={order_r} measured rate {rate:.2f}; errs={errs}"
    )


# ---------------------------------------------------------------------------
# Validation: invalid orders rejected at compile time
# ---------------------------------------------------------------------------


def test_invalid_order_r_rejected():
    """compile_problem with odd or zero order_r raises."""
    u_sym = sp.sin(sp.pi * r) * sp.cos(theta)
    sys_factory, _ = _build_shell_poisson(u_sym)
    grid = make_shell_grid(N_r=16, N_theta=16, r_inner=0.3, r_outer=1.0)
    with pytest.raises((ValueError, TypeError)):
        compile_problem(sys_factory(), grid, m=0, order_r=3)
    with pytest.raises((ValueError, TypeError)):
        compile_problem(sys_factory(), grid, m=0, order_theta=0)


def test_grid_too_small_for_order():
    """compile_problem on a grid with N < order + 1 raises."""
    u_sym = sp.sin(sp.pi * r) * sp.cos(theta)
    sys_factory, _ = _build_shell_poisson(u_sym)
    # N_r = 8 is too small for order_r = 8 (need ≥ 9).
    grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.3, r_outer=1.0)
    with pytest.raises(ValueError, match="order_r=8"):
        compile_problem(sys_factory(), grid, m=0, order_r=8)
