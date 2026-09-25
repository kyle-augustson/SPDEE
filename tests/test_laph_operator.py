"""Tests for LapH (horizontal Laplacian) — the angular part of Lap on a scalar.

In spherical coordinates with fixed azimuthal mode m:
    r² · ∇_H² f = ∂²_θ f + cot θ · ∂_θ f - m²/sin²θ · f

This is an angular-only operator (no radial derivatives), used in
toroidal-scalar formulations of incompressible velocity problems
(Rieutord & Valdettaro 1997 §2.2 / Appendix B).

These tests verify the IR → walker → compiler → sparse-builder
pipeline gives the same result as the manually-built LapH operator
used in `test_evp_inertial_appendix_b.py`.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee import Unknown, LapH, Equation, EquationSystem, BoundaryCondition, Boundary
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


class TestLapHCompiles:
    """The IR/compiler pipeline accepts LapH(scalar_unknown) cleanly."""

    def test_ball_compile(self):
        u = Unknown.Scalar("u")
        eq = Equation(LapH(u), name="laph_eq")
        sys_ = EquationSystem([eq], [])
        grid = make_ball_grid(N_r=8, N_theta=24, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=2, order_r=6, order_theta=6)
        assert cp.matrices.A_total.shape[0] == 8 * 24

    def test_shell_compile(self):
        u = Unknown.Scalar("u")
        eq = Equation(LapH(u), name="laph_eq")
        bcs = [BoundaryCondition(Boundary.R_INNER, u, name="diri_in"),
               BoundaryCondition(Boundary.R_OUTER, u, name="diri_out")]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(N_r=8, N_theta=24,
                               r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=2, order_r=6, order_theta=6)
        assert cp.matrices.A_total.shape[0] == 8 * 24

    def test_rejects_vector_operand(self):
        v = Unknown.Vector("v")
        with pytest.raises(TypeError, match="LapH requires a scalar"):
            LapH(v)


class TestLapHOperatorAction:
    """The compiled LapH operator acts correctly on spherical
    harmonic test functions.

    For T = R(r) · Y^m_ℓ(θ,φ) (with ∂_φ → im baked in by the m
    parameter), the angular Laplacian satisfies
        ∇_H² T = -ℓ(ℓ+1)/r² · T
    so r²·∇_H² T = -ℓ(ℓ+1) · T.

    The compiled A_total for `Equation(LapH(u))` should give
        A_total · u = r² · ∇_H² u
    at interior θ rows (away from the singularities at θ=0, π).
    """

    @pytest.mark.parametrize("m,ell", [(1, 1), (1, 2), (2, 2), (2, 3), (3, 3)])
    def test_acts_correctly_on_spherical_harmonic(self, m, ell):
        u = Unknown.Scalar("u")
        eq = Equation(LapH(u), name="laph_eq")
        sys_ = EquationSystem([eq], [])
        # Use shell to avoid r=0 origin treatment muddying the interior check
        grid = make_shell_grid(N_r=12, N_theta=48,
                               r_inner=0.3, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=m, order_r=6, order_theta=6)
        A = cp.matrices.A_total

        # Build T = R(r) · associated-Legendre(ℓ, m, cos θ)
        # For ℓ ≥ m, the angular part is a true spherical harmonic.
        # Use a simple polynomial form: (sin θ)^m * P_ℓ^m(cos θ) component.
        # For ℓ=m exactly: Y^m_m ∝ (sin θ)^m e^{im φ}; just (sin θ)^m here.
        # For ℓ > m we'd need the proper associated Legendre — for this
        # test use ℓ=m only.
        from scipy.special import lpmv
        theta = grid.theta
        # Use scipy.special.lpmv for associated Legendre P_ℓ^m(cos θ)
        ang = lpmv(m, ell, np.cos(theta))  # shape (N_theta,)
        # Pick R(r) = r² so it's a smooth radial profile.
        R_profile = grid.r ** 2
        T_2d = R_profile[:, None] * ang[None, :]  # shape (N_r, N_theta)
        T_flat = T_2d.ravel(order='F').astype(complex)

        # Apply A and compare to expected
        Ax = A @ T_flat
        expected = -ell * (ell + 1) * T_flat  # r²·∇_H² = -ℓ(ℓ+1) on Y^m_ℓ

        # Check interior (skip near θ=0, π and i_r=0 / i_r=N_r-1)
        N_r, N_theta = grid.N_r, grid.N_theta
        mask = np.zeros((N_r, N_theta), dtype=bool)
        mask[2:-2, 4:-4] = True
        mask_flat = mask.ravel(order='F')

        # Relative error
        denom = np.linalg.norm(expected[mask_flat])
        rel_err = np.linalg.norm(Ax[mask_flat] - expected[mask_flat]) / max(denom, 1e-30)
        assert rel_err < 1e-6, (
            f"m={m}, ℓ={ell}: ||A T - (-ℓ(ℓ+1)) T|| / ||target|| = "
            f"{rel_err:.3e} (expected < 1e-6)"
        )


class TestLapHEVPMatchesManualBuild:
    """The compiled LapH operator matches the manually-built operator
    in `test_evp_inertial_appendix_b.py` to high precision.

    This is the consistency check that IR-routed LapH produces the
    same matrix as the hand-rolled angular-Laplacian builder.
    """

    def test_matches_manual_builder(self):
        # Manual: replicate _build_r2_lap_h
        from spdee.backends.fd2.operators_core import (
            lift_theta_op, diag_from_2d_array,
        )
        from spdee.backends.fd2.stencils import standard_d1, standard_d2

        m = 2
        N_r, N_theta = 8, 32
        order = 6
        grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0)
        _, B_1_t = standard_d1(grid.theta, order=order)
        _, B_2_t = standard_d2(grid.theta, order=order)
        B_1_t_lift = lift_theta_op(B_1_t, N_r)
        B_2_t_lift = lift_theta_op(B_2_t, N_r)
        cot_2d = (np.ones(N_r)[:, None]
                  * (np.cos(grid.theta) / np.sin(grid.theta))[None, :])
        csc2_2d = (np.ones(N_r)[:, None]
                   * (1.0 / np.sin(grid.theta) ** 2)[None, :])
        cot_diag = diag_from_2d_array(cot_2d.astype(complex))
        csc2_diag = diag_from_2d_array(csc2_2d.astype(complex))
        manual = (B_2_t_lift + cot_diag @ B_1_t_lift - m**2 * csc2_diag).tocsr()

        # IR-routed
        u = Unknown.Scalar("u")
        eq = Equation(LapH(u), name="laph_eq")
        sys_ = EquationSystem([eq], [])
        cp = compile_problem(sys_, grid, m=m, order_r=order, order_theta=order)
        ir = cp.matrices.A_total.tocsr()

        # Compare on interior rows (skip i_r=0 in ball mode — origin
        # treatment may differ since LapH doesn't have a radial part).
        # Actually: the LapH builder shouldn't be affected by origin
        # regularity since it's not in unknowns_with_2nd_order_op.
        # But ball mode mass diagonal still has identity at i_r=0.
        diff = (manual - ir).toarray()
        # Skip i_r=0 rows where ball-mode origin placeholders live
        N = N_r * N_theta
        keep = np.ones(N, dtype=bool)
        for j in range(N_theta):
            keep[j * N_r + 0] = False  # i_r=0 placeholder row
        diff_keep = diff[keep, :]
        rel = np.linalg.norm(diff_keep) / max(np.linalg.norm(manual.toarray()[keep, :]), 1e-30)
        assert rel < 1e-12, (
            f"IR-routed LapH differs from manual builder: rel = {rel:.3e}"
        )


class TestLapHInertialWaveEVP:
    """End-to-end inertial-wave EVP test using IR-routed LapH operator.

    Solves the Appendix B inertial-wave equation
        λ ∇_H² T + (im/r²) T = 0
    using the IR pipeline to build the M-matrix (= compiled r²·∇_H²).
    This is the analog of `test_evp_inertial_appendix_b.py` but
    obtains the M-operator from spdee's IR/compile rather than from
    a hand-rolled construction.

    Note: K = -im·I and M = (compiled r²·LapH) are built directly
    here rather than going through `solve_evp`, because the latter
    currently has no API for "M comes from a *different* operator
    than the equation's natural mass". Extending solve_evp to
    support that is a separate future change.
    """

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_lowest_eigenvalue_matches_one_over_m_plus_one(self, m):
        T = Unknown.Scalar("T")
        eq = Equation(LapH(T), name="laph_eq")
        sys_ = EquationSystem([eq], [])
        grid = make_ball_grid(N_r=8, N_theta=48, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=m, order_r=6, order_theta=6)

        # M = the IR-compiled r²·∇_H²
        M_op = cp.matrices.A_total.tocsr()
        N = M_op.shape[0]
        # K = -im · I (the Coriolis "pointwise" operator)
        K_op = sp.diags(np.full(N, -1j * m, dtype=complex)).tocsr()

        # Shift-invert near the expected eigenvalue
        sigma = 1j / (m + 1)
        KmM = (K_op - sigma * M_op).tocsc()
        LU = spla.splu(KmM)

        def matvec(x):
            return LU.solve(M_op @ x)

        LinOp = spla.LinearOperator(K_op.shape, matvec=matvec, dtype=complex)
        mu = spla.eigs(LinOp, k=8, which="LM", return_eigenvectors=False)
        finite = np.abs(mu) > 1e-10
        lam = 1.0 / mu[finite] + sigma

        target = 1j / (m + 1)
        rel_err = np.abs(lam - target) / abs(target)
        best = float(rel_err.min())
        assert best < 1e-3, (
            f"m={m}: no eigenvalue within 0.1% of i/{m+1}. "
            f"Best λ = {lam[np.argmin(rel_err)]}, rel err {best:.3e}"
        )
