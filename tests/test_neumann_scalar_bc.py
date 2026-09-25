"""Tests for the radial Neumann scalar BC ``Dot(r_hat, Grad(s)) = g``
(``BC_NEUMANN_SCALAR_RADIAL``).

Coverage:

* Walker-side recognition: bare Dot pattern (both argument orders),
  Sum with background-scalar RHS, mixed Dirichlet+Neumann systems.
* Backend-side numerics: manufactured-solution BVP with a non-
  polynomial solution, demonstrates rate-4 convergence of the
  one-sided radial-derivative stencil at the boundary (matching the
  4th-order interior).
* Host-row collision: Dirichlet + Neumann on the same scalar at the
  same boundary raises (they share a host row by design).
"""
import numpy as np
import pytest
import sympy as sp

from spdee import (
    Background, Unknown, r, theta, Grad, Dot, Lap,
    Equation, BoundaryCondition, EquationSystem, Boundary, r_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.ir.walker import _parse_one_bc
from spdee.ir.tasks import BC_NEUMANN_SCALAR_RADIAL


# ---------------------------------------------------------------------------
# Walker-side: pattern recognition
# ---------------------------------------------------------------------------

class TestNeumannWalker:
    """Walker recognizes ``Dot(r_hat, Grad(scalar_unknown))`` and the
    inhomogeneous form ``Dot(r_hat, Grad(s)) - g``."""

    def test_bare_homogeneous(self):
        s = Unknown.Scalar("s")
        bc = BoundaryCondition(
            Boundary.R_OUTER, Dot(r_hat, Grad(s)), name="neumann_out",
        )
        bct = _parse_one_bc(bc, m=0)
        assert bct.kind == BC_NEUMANN_SCALAR_RADIAL
        assert bct.unknown_name == "s"
        assert bct.location == "r=1"
        assert bct.value is None

    def test_bare_homogeneous_arg_order_reversed(self):
        """Dot is commutative on scalars; Dot(Grad(s), r_hat) should
        be recognized identically to Dot(r_hat, Grad(s))."""
        s = Unknown.Scalar("s")
        bc = BoundaryCondition(
            Boundary.R_INNER, Dot(Grad(s), r_hat), name="neumann_in_rev",
        )
        bct = _parse_one_bc(bc, m=0)
        assert bct.kind == BC_NEUMANN_SCALAR_RADIAL
        assert bct.unknown_name == "s"
        assert bct.location == "r=0"

    def test_inhomogeneous_with_background_rhs(self):
        s = Unknown.Scalar("s")
        g = Background.Scalar("g", expr=sp.sin(theta))
        bc = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(r_hat, Grad(s)) - g, name="neumann_inhom",
        )
        bct = _parse_one_bc(bc, m=0)
        assert bct.kind == BC_NEUMANN_SCALAR_RADIAL
        assert bct.value is not None
        # Background scalar -g moved to RHS: value's SymPy expression
        # should equal +g (Sum unwrap puts background on the RHS with
        # appropriate sign).
        assert bct.value.expr == sp.sin(theta)

    def test_inhomogeneous_with_background_lhs(self):
        """``g + Dot(r_hat, Grad(s)) = 0`` means ``∂_r s = -g``."""
        s = Unknown.Scalar("s")
        g = Background.Scalar("g", expr=sp.cos(theta))
        bc = BoundaryCondition(
            Boundary.R_OUTER,
            g + Dot(r_hat, Grad(s)), name="neumann_inhom_lhs",
        )
        bct = _parse_one_bc(bc, m=0)
        assert bct.kind == BC_NEUMANN_SCALAR_RADIAL
        assert bct.value is not None
        # ``g + ∂_r s = 0`` →  ∂_r s = -g.
        # Sum unwrap moves the bg side to RHS with negation.
        assert sp.expand(bct.value.expr + sp.cos(theta)) == sp.S.Zero


# ---------------------------------------------------------------------------
# Numerical: BVP convergence rate
# ---------------------------------------------------------------------------

class TestNeumannBVP:
    """Solve a Helmholtz-type BVP with mixed Dirichlet + Neumann BCs
    and verify rate-4 convergence."""

    @staticmethod
    def _build_helmholtz_problem(r_in=0.5, r_out=1.0):
        """3D axisymmetric m=0 Poisson with manufactured solution
        ``s_true(r) = sin(2 r)``.

        Lap(s) = s'' + (2/r) s' = -4 sin(2r) + (4/r) cos(2r).
        ∂_r s = 2 cos(2 r).

        Mixed BCs: Dirichlet at r_in (pins the constant), Neumann at
        r_out (exercises the new BC).
        """
        s = Unknown.Scalar("s")
        lap_s_sym = -4 * sp.sin(2 * r) + (4 / r) * sp.cos(2 * r)
        src_bg = Background.Scalar("src", expr=lap_s_sym)
        eq = Equation(Lap(s) - src_bg, name="poisson")

        s_in_bg = Background.Scalar(
            "s_in", expr=sp.Float(np.sin(2 * r_in)),
        )
        g_out_bg = Background.Scalar(
            "g_out", expr=sp.Float(2 * np.cos(2 * r_out)),
        )
        bcs = [
            BoundaryCondition(
                Boundary.R_INNER, s - s_in_bg, name="dirichlet_in",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(r_hat, Grad(s)) - g_out_bg, name="neumann_out",
            ),
        ]
        return EquationSystem([eq], bcs), s

    def test_rate_4_convergence(self):
        """Solve at increasing N_r and check that the max error
        decreases at rate 4 (4th-order interior + 4th-order one-sided
        Neumann BC)."""
        r_in, r_out = 0.5, 1.0
        sys_, _s = self._build_helmholtz_problem(r_in, r_out)
        N_theta = 16
        errs = []
        for N_r in [16, 24, 32, 48, 64]:
            grid = make_shell_grid(
                N_r=N_r, N_theta=N_theta,
                r_inner=r_in, r_outer=r_out,
            )
            cp = compile_problem(
                sys_, grid, m=0, order_r=4, order_theta=4,
            )
            sol = cp.solve_bvp()
            s_field = sol["s"].real
            R, _ = grid.mesh()
            s_true = np.sin(2 * R)
            err = float(np.max(np.abs(s_field - s_true)))
            errs.append((N_r, err))

        # Convergence rate between successive resolutions.
        rates = []
        for i in range(1, len(errs)):
            N0, e0 = errs[i - 1]
            N1, e1 = errs[i]
            rate = float(np.log(e0 / e1) / np.log(N1 / N0))
            rates.append(rate)

        # Each pair-rate should be ≥ 3.8 (allow some slack from
        # leading-coefficient drift); the asymptotic rate is 4.
        for r_pair, (N0, _) , (N1, _) in zip(
            rates, errs[:-1], errs[1:]
        ):
            assert r_pair > 3.8, (
                f"Neumann BC convergence rate {r_pair:.2f} between "
                f"N_r={N0} and N_r={N1} is below 3.8 — expected ≥ 4 "
                f"asymptotically. All rates: {rates}"
            )
        # Final-decade rate should be very close to 4.
        assert abs(rates[-1] - 4.0) < 0.3, (
            f"Asymptotic rate {rates[-1]:.2f} too far from 4. "
            f"All rates: {rates}"
        )

    def test_homogeneous_neumann_recovers_constant(self):
        """If the source is zero and both BCs are homogeneous Neumann
        (∂_r s = 0 at both boundaries), the solution should be a
        constant (the trivial solution of Laplace with pure Neumann).

        Without an integral constraint the system is singular (constant
        kernel). We instead use Dirichlet at r_in (s = s_in_val) +
        Neumann at r_out (∂_r s = 0). The solution to Lap(s) = 0 with
        these BCs is *not* generally constant for a 3D radial problem
        (the radial Laplacian Lap(s) = s'' + (2/r) s' has non-trivial
        solutions even for ∂_r s = 0). So this test instead checks
        a slightly different property: with the manufactured solution
        s_true(r) = constant, the homogeneous Neumann BC at r_out is
        satisfied trivially, and the system reduces to a Dirichlet
        problem with constant source.
        """
        r_in, r_out = 0.5, 1.0
        s_const = 3.7  # arbitrary
        s = Unknown.Scalar("s")
        eq = Equation(Lap(s), name="laplace")  # source = 0
        s_in_bg = Background.Scalar(
            "s_in", expr=sp.Float(s_const),
        )
        bcs = [
            BoundaryCondition(
                Boundary.R_INNER, s - s_in_bg, name="dirichlet_in",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(r_hat, Grad(s)), name="neumann_out_zero",
            ),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=32, N_theta=16, r_inner=r_in, r_outer=r_out,
        )
        cp = compile_problem(sys_, grid, m=0, order_r=4, order_theta=4)
        sol = cp.solve_bvp()
        s_field = sol["s"].real
        err = float(np.max(np.abs(s_field - s_const)))
        # The constant solution satisfies both BCs and the equation
        # exactly, so the error should be at roundoff.
        assert err < 1e-9, (
            f"Constant solution not recovered: max err = {err:.2e}"
        )


# ---------------------------------------------------------------------------
# Collision: Dirichlet + Neumann on the same scalar boundary
# ---------------------------------------------------------------------------

class TestNeumannCollision:
    """Two BCs on the same scalar at the same boundary share a host
    row and must collide loudly."""

    def test_dirichlet_plus_neumann_same_boundary_raises(self):
        s = Unknown.Scalar("s")
        eq = Equation(Lap(s), name="laplace")
        # Both BCs target r=r_outer of s — row collision.
        bcs = [
            BoundaryCondition(
                Boundary.R_OUTER, s, name="dirichlet_out",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(r_hat, Grad(s)), name="neumann_out",
            ),
            BoundaryCondition(
                Boundary.R_INNER, s, name="dirichlet_in",
            ),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=24, N_theta=16, r_inner=0.5, r_outer=1.0,
        )
        cp = compile_problem(
            sys_, grid, m=0, order_r=4, order_theta=4,
        )
        with pytest.raises(ValueError, match="host-row collision"):
            cp.solve_bvp()
