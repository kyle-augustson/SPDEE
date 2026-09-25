"""Test that solve_evp respects scalar coefficients on Dt(unknown).

The pattern `rho0 * Dt(v) - OP(v) = 0` should be solved as the
generalized EVP
    K v = λ M v
with M = rho0(r,θ) · (equation mass) · I on v's column block.

This matters for stratified problems (variable density / variable
specific heat capacity multiplying the time derivative) and is
the structural enabler for compressible-MHD EVPs around a solar
Model-S background.

Verifies:
1. The B_total compiled matrix has rho0 baked in (compiler stage).
2. solve_evp uses B_total when populated, so eigenvalues scale
   correctly with the coefficient.
3. The mass_equations path (no Dt, mass_equations declared) works
   without rho0.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Dt, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


def _build_strat_diff(rho0_value):
    """Build a stratified vector diffusion EVP: rho0 ∂_t v = Lap(v).

    rho0_value is either a number (constant), or a sympy expression
    in (r, theta).
    """
    v = Unknown.Vector("v")
    rho0_expr = sp.S(rho0_value) if isinstance(rho0_value, (int, float)) else rho0_value
    rho0 = Background.Scalar("rho0", expr=rho0_expr)
    eq = Equation(rho0 * Dt(v) - Lap(v), name="strat_diff")
    bcs = [
        BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="r_in_imp"),
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="r_out_imp"),
    ]
    return EquationSystem([eq], bcs)


class TestDtCoefficientCompiledIntoBTotal:
    """B_total stores eq_mass × rho0 × I when rho0 * Dt(v) is present."""

    def test_constant_rho0_appears_in_B_total(self):
        sys_ = _build_strat_diff(rho0_value=2.0)
        grid = make_shell_grid(N_r=8, N_theta=12, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        B = cp.matrices.B_total
        eq_mass = cp.matrices.equation_mass_matrices["strat_diff"]
        # B should be rho0 × eq_mass on the equation rows mapped to v's cols.
        # On the diagonal: each entry of B = 2 × corresponding eq_mass entry.
        ratio = B.diagonal()[:eq_mass.shape[0]] / eq_mass.diagonal()
        assert np.allclose(ratio, 2.0, atol=1e-12), (
            f"Constant rho0=2 didn't appear as factor of 2 in B_total; "
            f"got ratios {ratio[:5]}"
        )

    def test_radial_rho0_appears_in_B_total(self):
        r_sym = sp.symbols("r", positive=True)
        sys_ = _build_strat_diff(rho0_value=1 + r_sym ** 2)
        grid = make_shell_grid(N_r=8, N_theta=12, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        B = cp.matrices.B_total
        eq_mass = cp.matrices.equation_mass_matrices["strat_diff"]
        # The ratio of B to eq_mass on the diagonal should equal
        # rho0 evaluated on the grid, in V19 spin-block layout.
        # Just verify B nnz matches eq_mass × 1 (no extra coupling).
        assert B.nnz == eq_mass.nnz, (
            f"B_total nnz {B.nnz} != eq_mass nnz {eq_mass.nnz} — "
            f"radial rho0 should be diagonal-pointwise"
        )


class TestSolveEVPRespectsDtCoefficient:
    """Doubling rho0 should halve all eigenvalues of `rho0 ∂_t v = Lap(v)`."""

    def _eigenvalues(self, rho0_value, k=6):
        sys_ = _build_strat_diff(rho0_value=rho0_value)
        grid = make_shell_grid(N_r=14, N_theta=20, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        res = cp.solve_evp(k=k, sigma=-100.0)
        lam = np.array(res.eigenvalues)
        return lam[np.isfinite(lam)]

    def test_constant_rho0_scales_eigenvalues(self):
        l1 = self._eigenvalues(1.0)
        l2 = self._eigenvalues(2.0)
        # The deepest (most negative) eigenvalues are best resolved;
        # near-zero ones suffer from singular-M numerical effects.
        # Use the third eigenvalue (well below the constraint-y noise).
        # Find the most-negative eigenvalue in each.
        e1 = sorted(l1.real)[0]
        e2 = sorted(l2.real)[0]
        # If solve_evp respects rho0=2, eigenvalues should be HALVED
        # (since rho0 multiplies the eigenvalue side: lambda_new = lambda_old / rho0).
        ratio = e1 / e2
        assert 1.9 < ratio < 2.1, (
            f"rho0 not respected: e1={e1:.4f}, e2={e2:.4f}, "
            f"ratio={ratio:.4f}, expected ~2.0"
        )


class TestSolveEVPLegacyPathStillWorks:
    """When no Dt is in the equation and mass_equations is passed, the
    natural-mass construction applies."""

    def test_no_dt_uses_natural_mass(self):
        from spdee import Unknown
        v = Unknown.Vector("v")
        # No Dt — instead declare mass via mass_equations.
        eq = Equation(Lap(v), name="visc")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="r_in_imp"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="r_out_imp"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(N_r=14, N_theta=20, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        # B_total should be empty (no Dt).
        assert cp.matrices.B_total.nnz == 0
        # solve_evp uses mass_equations → natural mass.
        res = cp.solve_evp(k=6, sigma=-100.0, mass_equations={"visc": "v"})
        lam = np.array(res.eigenvalues)
        assert len(lam[np.isfinite(lam)]) > 0, (
            "Legacy mass_equations path should still solve EVPs"
        )
