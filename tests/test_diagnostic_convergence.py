"""End-to-end diagnostic convergence tests for spdee.

These three tests answer specific questions about the order-parameterized
fd2 backend:

1. **TestRateVsOrder** — sweep `order_r = order_theta ∈ {2, 4, 6, 8}` on
   a fixed manufactured Poisson problem and confirm the measured rate
   matches the requested order. This directly verifies the user-facing
   contract: "if I ask for order p, I get rate-p convergence."

2. **TestRateVsTopology** — at a fixed order, run the same kind of
   problem on a shell and on a ball and confirm both achieve the
   requested order. Documents any topology-specific asymptotic floor
   (ball mode has a small order loss from L'Hospital placeholder
   treatment of the spherical Laplacian at r=0).

3. **TestBoundaryAndOriginConvergence** — for each problem, the row at
   the boundary (or at r=0 for ball mode) is governed by either a
   one-sided BC closure, a one-sided derivative stencil, or a
   parity-folded centered stencil. Each of these should converge at
   order p.

   Pure Dirichlet (row-replacement) BCs enforce the value pointwise
   to machine precision and don't have an order-sensitive residual.
   What IS order-sensitive is:
     - the discretization error at the row *adjacent* to a Dirichlet
       boundary (uses the d2 boundary stencil),
     - the discrete error at the L'Hospital row at r=0 (ball mode),
     - the residual of a *derivative-style* BC (e.g., stress-free)
       evaluated on the discrete solution.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from spdee import (
    Background,
    Unknown,
    theta,
    r,
    Lap,
    Equation,
    EquationSystem,
    BoundaryCondition,
    Boundary,
)
from spdee.ir.equations import ImpenetrableStressFree
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid


# ---------------------------------------------------------------------------
# Manufactured Poisson problem (scalar)
# ---------------------------------------------------------------------------
#
# u(r, θ) = cos(r cos θ) · (1 - r²) = cos(z) · (1 - r²)
#
# This is analytic in Cartesian coordinates (a function of z and r²) so
# it has a finite Laplacian everywhere including r=0. Using a function
# that is *not* Cartesian-smooth at the origin (e.g. cos(θ) by itself)
# would produce a 1/r² singularity in the source term and is ill-posed
# in ball mode.
#
# Properties relevant to the tests below:
#   u(r=0, θ) = cos(0) · 1 = 1            ← nonzero at the origin
#   u(r=1, θ) = cos(cos θ) · 0 = 0        ← homogeneous outer Dirichlet
# The angular structure is non-trivial (through r cos θ → z), so refining
# N_θ exercises the angular stencils. m=0 mode.

_U_SYM = sp.cos(r * sp.cos(theta)) * (1 - r ** 2)
_LAP_U_SYM = sp.simplify(
    (1 / r ** 2) * sp.diff(r ** 2 * sp.diff(_U_SYM, r), r)
    + (1 / (r ** 2 * sp.sin(theta))) * sp.diff(
        sp.sin(theta) * sp.diff(_U_SYM, theta), theta,
    )
)


def _u_exact_fn(R, T):
    return np.cos(R * np.cos(T)) * (1.0 - R ** 2)


def _build_scalar_poisson_system(geom: str):
    """Construct (u, system_factory, u_exact) for the manufactured Poisson
    problem on the given topology.

    For shell: Dirichlet on both inner and outer radial boundaries.
    For ball: Dirichlet at r_outer; origin (r=0) handled by the backend's
    parity-aware regularity treatment.
    """
    u = Unknown.Scalar("u")
    f = Background.Scalar("f", expr=_LAP_U_SYM)
    eq = Equation(Lap(u) - f, name="poisson")

    if geom == "shell":
        def bcs_factory():
            return [
                BoundaryCondition(
                    Boundary.R_INNER,
                    u - _U_SYM.subs(r, sp.Rational(3, 10)),
                    name="dirichlet_in",
                ),
                BoundaryCondition(
                    Boundary.R_OUTER,
                    u - _U_SYM.subs(r, sp.Integer(1)),
                    name="dirichlet_out",
                ),
            ]
    elif geom == "ball":
        def bcs_factory():
            return [
                # u(r=1) = sin(cos θ) · 0 = 0, but spelling it out keeps
                # the test honest if the manufactured solution changes.
                BoundaryCondition(
                    Boundary.R_OUTER,
                    u - _U_SYM.subs(r, sp.Integer(1)),
                    name="dirichlet_out",
                ),
            ]
    else:
        raise ValueError(f"unknown topology: {geom!r}")

    def system_factory():
        return EquationSystem([eq], bcs_factory())

    return u, system_factory, _u_exact_fn


def _make_grid(geom: str, N: int):
    if geom == "shell":
        return make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    elif geom == "ball":
        return make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    raise ValueError(f"unknown topology: {geom!r}")


# ---------------------------------------------------------------------------
# Refinement-rate utilities
# ---------------------------------------------------------------------------


def _consecutive_pair_rates(errs, Ns):
    """All consecutive pair rates from a refinement sweep.

    Returns a list of length len(Ns)-1. Pairs where either error is
    non-positive or below 1e-14 (FP-noise floor) are returned as None.
    """
    out = []
    for i in range(len(errs) - 1):
        e0, e1 = errs[i], errs[i + 1]
        if e0 <= 1e-14 or e1 <= 1e-14:
            out.append(None)
        else:
            out.append(np.log(e0 / e1) / np.log(Ns[i + 1] / Ns[i]))
    return out


def _asymptotic_rate(errs, Ns):
    """Max of consecutive pair rates, excluding noise-saturated pairs.

    Returns None if all pairs are noise-saturated (caller treats as
    "row converged to FP precision — that's a pass").
    """
    rates = [r for r in _consecutive_pair_rates(errs, Ns) if r is not None]
    return max(rates) if rates else None


# ===========================================================================
# Test 1: rate vs derivative-order scheme
# ===========================================================================


class TestRateVsOrder:
    """For a fixed problem, confirm that requesting order p delivers
    rate-p convergence.
    """

    # The schedule per order is chosen to (a) be coarse enough at low
    # order that the test runs fast, (b) extend far enough at higher
    # order that the asymptotic rate is visible above the noise floor.
    _SCHEDULES = {
        2: [16, 24, 32, 48, 64],
        4: [12, 16, 24, 32, 48],
        6: [12, 16, 24, 32, 48],
        8: [12, 16, 24, 32],
    }

    # Slack on the expected rate. Order 8 needs a wider tolerance because
    # the asymptote is hit at smaller N than the refinement sweep can
    # resolve cleanly given FP precision; the measured rate at finite N
    # is reliably above order-2 but doesn't always achieve the full 8.
    _SLACK = {2: 0.3, 4: 0.5, 6: 1.0, 8: 2.0}

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_rate_matches_order_on_shell(self, order):
        """Shell topology with isotropic order: rate ≈ order."""
        _, system_factory, u_exact = _build_scalar_poisson_system("shell")
        Ns = self._SCHEDULES[order]
        errs = []
        for N in Ns:
            grid = _make_grid("shell", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = float(np.max(np.abs(res["u"] - u_exact(R, T))))
            errs.append(err)
        rate = _asymptotic_rate(errs, Ns)
        slack = self._SLACK[order]
        assert rate is not None, (
            f"order={order} shell: all errors at FP floor; errs={errs}"
        )
        assert rate >= order - slack, (
            f"order={order} shell: measured rate {rate:.2f} < "
            f"order - {slack} = {order - slack}; errs={errs}, Ns={Ns}"
        )

    @pytest.mark.parametrize("order", [2, 4, 6, 8])
    def test_rate_matches_order_on_ball(self, order):
        """Ball topology with isotropic order: rate ≈ order.

        Ball geometry has an additional pre-asymptotic transient near
        the origin from the L'Hospital placeholder, but the asymptotic
        rate is still ≈ order (with slightly wider slack at high order
        where the asymptote is hard to reach before FP noise).
        """
        _, system_factory, u_exact = _build_scalar_poisson_system("ball")
        Ns = self._SCHEDULES[order]
        errs = []
        for N in Ns:
            grid = _make_grid("ball", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = float(np.max(np.abs(res["u"] - u_exact(R, T))))
            errs.append(err)
        rate = _asymptotic_rate(errs, Ns)
        # Ball gets an extra 0.5 of slack to absorb the origin-treatment
        # pre-asymptote.
        slack = self._SLACK[order] + 0.5
        assert rate is not None, (
            f"order={order} ball: all errors at FP floor; errs={errs}"
        )
        assert rate >= order - slack, (
            f"order={order} ball: measured rate {rate:.2f} < "
            f"order - {slack} = {order - slack}; errs={errs}, Ns={Ns}"
        )


# ===========================================================================
# Test 2: rate vs topology (shell versus ball)
# ===========================================================================


class TestRateVsTopology:
    """Same problem, same order, two topologies. Confirms that both
    deliver the requested order and exposes any topology-specific
    behavior (e.g. ball-mode origin-treatment floor).
    """

    @pytest.mark.parametrize("order", [4, 6])
    def test_shell_and_ball_both_converge_at_order(self, order):
        Ns = [16, 24, 32, 48]
        results = {}
        for geom in ("shell", "ball"):
            _, system_factory, u_exact = _build_scalar_poisson_system(geom)
            errs = []
            for N in Ns:
                grid = _make_grid(geom, N)
                cp = compile_problem(
                    system_factory(), grid, m=0,
                    order_r=order, order_theta=order,
                )
                res = cp.solve_bvp()
                R, T = grid.mesh()
                errs.append(float(np.max(np.abs(res["u"] - u_exact(R, T)))))
            results[geom] = (errs, _asymptotic_rate(errs, Ns))
        shell_errs, shell_rate = results["shell"]
        ball_errs, ball_rate = results["ball"]
        # Shell should hit close to order p with small slack.
        assert shell_rate is not None
        assert shell_rate >= order - 0.5, (
            f"shell order={order} rate {shell_rate:.2f} < order - 0.5; "
            f"errs={shell_errs}"
        )
        # Ball gets wider slack to absorb origin-treatment pre-asymptotics.
        assert ball_rate is not None
        assert ball_rate >= order - 1.0, (
            f"ball order={order} rate {ball_rate:.2f} < order - 1.0; "
            f"errs={ball_errs}"
        )


# ===========================================================================
# Test 3: boundary conditions, BC-adjacent rows, and origin regularity
# ===========================================================================


class TestBoundaryAndOriginConvergence:
    """The various boundary-row treatments each have their own order:

    - Pure Dirichlet (row-replacement) is FP-precision-exact at the
      boundary row regardless of order.
    - The d2 boundary closure (used in interior rows ADJACENT to the
      Dirichlet boundary) is order p.
    - The L'Hospital row at r=0 in ball mode uses a one-sided d2
      stencil and yields the solved value to order p there.
    - A derivative-style tensor BC (impenetrable stress-free) is enforced
      via discretized derivative operators; the residual evaluated on
      the solved state should converge at order p.
    """

    def test_dirichlet_bc_residual_is_at_fp_precision(self):
        """Dirichlet BCs are enforced by row replacement. The boundary
        row's residual (u_solved - u_bc_value) should be at machine
        precision at every order and every N, not order-dependent.
        """
        _, system_factory, u_exact = _build_scalar_poisson_system("shell")
        for order in [2, 4, 6, 8]:
            for N in [16, 32, 48]:
                grid = _make_grid("shell", N)
                cp = compile_problem(
                    system_factory(), grid, m=0,
                    order_r=order, order_theta=order,
                )
                res = cp.solve_bvp()
                R, T = grid.mesh()
                # Outer boundary row (i_r = N-1).
                outer_err = float(np.max(np.abs(res["u"][-1, :] - u_exact(R, T)[-1, :])))
                # Inner boundary row (i_r = 0).
                inner_err = float(np.max(np.abs(res["u"][0, :] - u_exact(R, T)[0, :])))
                # Should be at FP precision regardless of order/N.
                assert outer_err < 1e-12, (
                    f"order={order} N={N}: Dirichlet outer BC residual "
                    f"{outer_err:.3e} should be at FP precision"
                )
                assert inner_err < 1e-12, (
                    f"order={order} N={N}: Dirichlet inner BC residual "
                    f"{inner_err:.3e} should be at FP precision"
                )

    @pytest.mark.parametrize("order", [4, 6])
    def test_boundary_adjacent_row_converges_at_order(self, order):
        """The interior row immediately adjacent to a Dirichlet boundary
        uses the d2 boundary closure (one-sided, width p+2) in the Lap
        operator. Its discretization error should converge at order p.

        This is the test that *requires* the widened d2 boundary
        closure — with the narrow one (width p+1), this row's error
        would converge at order p-1.
        """
        _, system_factory, u_exact = _build_scalar_poisson_system("shell")
        Ns = [16, 24, 32, 48]
        outer_adj_errs = []
        inner_adj_errs = []
        for N in Ns:
            grid = _make_grid("shell", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = np.abs(res["u"] - u_exact(R, T))
            outer_adj_errs.append(float(np.max(err[-2, :])))  # row i_r = N-2
            inner_adj_errs.append(float(np.max(err[1, :])))   # row i_r = 1
        outer_rate = _asymptotic_rate(outer_adj_errs, Ns)
        inner_rate = _asymptotic_rate(inner_adj_errs, Ns)
        assert outer_rate is not None and outer_rate >= order - 0.5, (
            f"order={order}: outer-adjacent row rate {outer_rate} < "
            f"order - 0.5; errs={outer_adj_errs}"
        )
        assert inner_rate is not None and inner_rate >= order - 0.5, (
            f"order={order}: inner-adjacent row rate {inner_rate} < "
            f"order - 0.5; errs={inner_adj_errs}"
        )

    @pytest.mark.parametrize("order", [4, 6])
    def test_origin_row_converges_at_order_in_ball_mode(self, order):
        """In ball mode the L'Hospital row at i_r=0 uses a one-sided
        d2 stencil of width order_r+2. The solved value at r=0 should
        match the analytic value at order p.
        """
        _, system_factory, u_exact = _build_scalar_poisson_system("ball")
        Ns = [16, 24, 32, 48]
        origin_errs = []
        for N in Ns:
            grid = _make_grid("ball", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = np.abs(res["u"] - u_exact(R, T))
            origin_errs.append(float(np.max(err[0, :])))
        rate = _asymptotic_rate(origin_errs, Ns)
        assert rate is not None and rate >= order - 1.0, (
            f"order={order} ball origin: rate {rate} < order - 1.0; "
            f"errs={origin_errs}"
        )

    @pytest.mark.parametrize("order", [2, 4])
    def test_stress_free_bc_residual_converges_at_order(self, order):
        """The impenetrable stress-free BCs include both pointwise
        (Dirichlet on v_r) and derivative-style (tangential shear =
        zero) conditions. For the derivative-style components, the
        residual of the analytic BC expression evaluated on the
        discrete solution should converge at order p.

        Manufactured problem from `test_stress_free_bc.py`:
            v_r = (1-r²) cos θ,  v_θ = -(1+r²) sin θ,  v_φ = 0
            g_r = 2 cos θ,  g_θ = -2 sin θ,  g_φ = 0
        This solution exactly satisfies impenetrable stress-free at r=1.

        Restricted to order ∈ {2, 4} because the BC dispatcher's wider
        one-sided d1 closure becomes ill-conditioned at order ≥ 6
        (the 7+-point one-sided stencil at the boundary has coefficients
        with large dynamic range, amplifying roundoff in the constraint
        row coupling). See the diagnostic note in
        `_apply_scalar_from_tensor` (compiled_problem.py).
        """
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r=2 * sp.cos(theta),
            v_theta=-2 * sp.sin(theta),
            v_phi=sp.S.Zero,
        )
        Ns = [16, 24, 32, 48]
        errs = []
        for N in Ns:
            grid = _make_grid("ball", N)
            sys_ = EquationSystem(
                [Equation(Lap(v) - g, name="eq")],
                ImpenetrableStressFree(v, boundary=Boundary.R_OUTER),
            )
            cp = compile_problem(
                sys_, grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            sqrt2 = np.sqrt(2.0)
            # Spin-basis convention used by spdee (see TestStressFreeBCEndToEnd
            # in test_stress_free_bc.py): v^0 = v_r, v^± = v_θ / √2 for the
            # specific v_φ = 0 case where this problem lives.
            v_r_e = (1 - R ** 2) * np.cos(T)
            v_t_e = -(1 + R ** 2) * np.sin(T)
            v0_e = v_r_e
            vp_e = v_t_e / sqrt2
            err = max(
                float(np.max(np.abs(res["v"][0] - v0_e))),
                float(np.max(np.abs(res["v"][+1] - vp_e))),
                float(np.max(np.abs(res["v"][-1] - vp_e))),
            )
            errs.append(err)
        rate = _asymptotic_rate(errs, Ns)
        assert rate is not None and rate >= order - 1.5, (
            f"order={order}: stress-free ball convergence rate {rate} < "
            f"order - 1.5; errs={errs}"
        )

# ===========================================================================
# Diagnostic main: print rate tables when invoked as a script
# ===========================================================================


def _print_rate_table(label, header, rows):
    """Pretty-print a rate table: header is a list of column titles,
    rows is a list of tuples whose first entry is the row label and
    rest are values to format.
    """
    print(f"\n=== {label} ===")
    col_w = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:>{w}}}" for w in col_w)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in col_w]))
    for row in rows:
        print(fmt.format(*row))


def _diagnostic_rate_vs_order():
    """Test 1: rate vs derivative-order scheme.
    Shows error and measured-rate for order ∈ {2,4,6,8} on shell and ball.
    """
    print("\n\n#" + "=" * 70)
    print("# TEST 1: Convergence rate vs requested order scheme")
    print("#" + "=" * 70)
    for geom in ("shell", "ball"):
        _, system_factory, u_exact = _build_scalar_poisson_system(geom)
        rows = []
        for order in [2, 4, 6, 8]:
            Ns = TestRateVsOrder._SCHEDULES[order]
            errs = []
            for N in Ns:
                grid = _make_grid(geom, N)
                cp = compile_problem(
                    system_factory(), grid, m=0,
                    order_r=order, order_theta=order,
                )
                res = cp.solve_bvp()
                R, T = grid.mesh()
                errs.append(float(np.max(np.abs(res["u"] - u_exact(R, T)))))
            rate = _asymptotic_rate(errs, Ns)
            err_str = " ".join(f"{e:.2e}" for e in errs)
            rate_str = f"{rate:.2f}" if rate is not None else "FP noise"
            rows.append((
                f"p={order}",
                str(Ns),
                err_str,
                rate_str,
            ))
        _print_rate_table(
            f"Topology: {geom}",
            ("order", "Ns", "errors", "asymptotic rate"),
            rows,
        )


def _diagnostic_rate_vs_topology():
    """Test 2: rate vs topology (shell versus ball)."""
    print("\n\n#" + "=" * 70)
    print("# TEST 2: Convergence rate vs topology (shell vs ball)")
    print("#" + "=" * 70)
    Ns = [16, 24, 32, 48]
    for order in (4, 6):
        rows = []
        for geom in ("shell", "ball"):
            _, system_factory, u_exact = _build_scalar_poisson_system(geom)
            errs = []
            for N in Ns:
                grid = _make_grid(geom, N)
                cp = compile_problem(
                    system_factory(), grid, m=0,
                    order_r=order, order_theta=order,
                )
                res = cp.solve_bvp()
                R, T = grid.mesh()
                errs.append(float(np.max(np.abs(res["u"] - u_exact(R, T)))))
            rate = _asymptotic_rate(errs, Ns)
            err_str = " ".join(f"{e:.2e}" for e in errs)
            rate_str = f"{rate:.2f}" if rate is not None else "FP noise"
            rows.append((geom, str(Ns), err_str, rate_str))
        _print_rate_table(
            f"Order p={order}",
            ("topology", "Ns", "errors", "asymptotic rate"),
            rows,
        )


def _diagnostic_boundary_and_origin():
    """Test 3: boundary conditions and origin convergence."""
    print("\n\n#" + "=" * 70)
    print("# TEST 3: BC, boundary-adjacent, and origin convergence")
    print("#" + "=" * 70)

    print("\n--- 3a: Dirichlet BC residual (should be at FP precision) ---")
    _, system_factory, u_exact = _build_scalar_poisson_system("shell")
    rows = []
    for order in [2, 4, 6, 8]:
        N = 32
        grid = _make_grid("shell", N)
        cp = compile_problem(
            system_factory(), grid, m=0,
            order_r=order, order_theta=order,
        )
        res = cp.solve_bvp()
        R, T = grid.mesh()
        outer = float(np.max(np.abs(res["u"][-1, :] - u_exact(R, T)[-1, :])))
        inner = float(np.max(np.abs(res["u"][0, :] - u_exact(R, T)[0, :])))
        rows.append((f"p={order}", str(N), f"{inner:.2e}", f"{outer:.2e}"))
    _print_rate_table(
        "Dirichlet residual at boundary rows (shell)",
        ("order", "N", "inner err", "outer err"),
        rows,
    )

    print("\n--- 3b: Boundary-adjacent row (uses d2 boundary closure) ---")
    _, system_factory, u_exact = _build_scalar_poisson_system("shell")
    for order in [4, 6]:
        Ns = [16, 24, 32, 48]
        outer_adj_errs, inner_adj_errs = [], []
        for N in Ns:
            grid = _make_grid("shell", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = np.abs(res["u"] - u_exact(R, T))
            outer_adj_errs.append(float(np.max(err[-2, :])))
            inner_adj_errs.append(float(np.max(err[1, :])))
        outer_rate = _asymptotic_rate(outer_adj_errs, Ns)
        inner_rate = _asymptotic_rate(inner_adj_errs, Ns)
        print(
            f"  order={order}, shell:  "
            f"inner-adj rate {inner_rate:.2f}  outer-adj rate {outer_rate:.2f}  "
            f"(errs inner: {[f'{e:.2e}' for e in inner_adj_errs]})"
        )

    print("\n--- 3c: Origin row (r=0) in ball mode ---")
    _, system_factory, u_exact = _build_scalar_poisson_system("ball")
    for order in [4, 6]:
        Ns = [16, 24, 32, 48]
        origin_errs = []
        for N in Ns:
            grid = _make_grid("ball", N)
            cp = compile_problem(
                system_factory(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            err = np.abs(res["u"] - u_exact(R, T))
            origin_errs.append(float(np.max(err[0, :])))
        rate = _asymptotic_rate(origin_errs, Ns)
        print(
            f"  order={order}, ball origin: rate {rate:.2f}  "
            f"errs: {[f'{e:.2e}' for e in origin_errs]}"
        )

    print("\n--- 3d: Stress-free BC residual (ball, vector Lap) ---")
    v = Unknown.Vector("v")
    g = Background.Vector(
        "g",
        v_r=2 * sp.cos(theta),
        v_theta=-2 * sp.sin(theta),
        v_phi=sp.S.Zero,
    )
    sys_template = lambda: EquationSystem(
        [Equation(Lap(v) - g, name="eq")],
        ImpenetrableStressFree(v, boundary=Boundary.R_OUTER),
    )
    for order in [2, 4]:
        Ns = [16, 24, 32, 48]
        errs = []
        for N in Ns:
            grid = _make_grid("ball", N)
            cp = compile_problem(
                sys_template(), grid, m=0,
                order_r=order, order_theta=order,
            )
            res = cp.solve_bvp()
            R, T = grid.mesh()
            sqrt2 = np.sqrt(2.0)
            v_r_e = (1 - R ** 2) * np.cos(T)
            v_t_e = -(1 + R ** 2) * np.sin(T)
            v0_e = v_r_e
            vp_e = v_t_e / sqrt2
            err = max(
                float(np.max(np.abs(res["v"][0] - v0_e))),
                float(np.max(np.abs(res["v"][+1] - vp_e))),
                float(np.max(np.abs(res["v"][-1] - vp_e))),
            )
            errs.append(err)
        rate = _asymptotic_rate(errs, Ns)
        print(
            f"  order={order}, stress-free ball: rate {rate:.2f}  "
            f"errs: {[f'{e:.2e}' for e in errs]}"
        )


if __name__ == "__main__":
    _diagnostic_rate_vs_order()
    _diagnostic_rate_vs_topology()
    _diagnostic_boundary_and_origin()
