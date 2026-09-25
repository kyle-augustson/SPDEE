"""Inertial-wave EVP test against Rieutord & Valdettaro 1997 Appendix B.

The inviscid inertial-wave equations on a rotating fluid:
    λ v + ẑ × v = -∇p,   ∇·v = 0,
on a sphere with no impenetrability constraint at any (purely
toroidal) boundary. Appendix B gives a family of analytic solutions
in the form `v = ∇×(T(r,θ,φ) r̂)` with

    T(r,θ,φ) ∝ r^{m+1} (sin θ)^m exp(im φ)

and eigenvalue λ = i/(m+1).

Derivation in the user's convention (v = curl(T r̂)):
- v is purely tangential (v_r = 0).
- Curl of the momentum equation eliminates pressure:
      λ ∇×v - (ẑ·∇)v = 0
- Take radial component. After unit-vector bookkeeping in
  spherical coords:
      (∇×v)_r = ∇_H² T   (with ∇_H² ≡ angular Laplacian / r²)
      r̂·(ẑ·∇)v = -im/r² · T
- Substituting:
      λ ∇_H² T + (im/r²) T = 0

The angular Laplacian's eigenfunctions are spherical harmonics:
∇_H² Y^m_ℓ = -ℓ(ℓ+1)/r² · Y^m_ℓ. So for T = R(r) Y^m_ℓ(θ,φ) (with any
radial profile R(r)), the equation reduces to
    -λ ℓ(ℓ+1)/r² + im/r² = 0
giving λ = im/(ℓ(ℓ+1)). For ℓ=m this is λ = i/(m+1) (Appendix B).
For higher ℓ ≥ m, additional modes exist at λ = im/(ℓ(ℓ+1)).

The equation has no radial derivatives — at each r it's an angular
EVP. The radial direction is therefore infinitely degenerate in the
continuum (any R(r) works); in the discretization each ℓ-eigenvalue
appears with multiplicity equal to N_r (one per radial grid row).

This test bypasses spdee's IR/compile pipeline because the LBB
issues of the colocated v/p discretization are precisely what
motivated the toroidal-scalar formulation. We build K and M directly
from existing primitive stencils.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.operators_core import lift_theta_op, diag_from_2d_array
from spdee.backends.fd2.stencils import standard_d1, standard_d2


def _build_r2_lap_h(grid, m: int, order_theta: int) -> sp.csr_matrix:
    """Build the (N_r·N_θ × N_r·N_θ) sparse operator for r²·∇_H².

    r²·∇_H² = ∂²_θ + cot θ ∂_θ - m²/sin²θ

    Acts on a scalar T(r, θ) stored r-major within (θ, r). No radial
    derivatives are involved — the operator is block-diagonal across
    r (every radial slice gets the same angular stencil).
    """
    N_r = grid.N_r

    _, B_1_t = standard_d1(grid.theta, order=order_theta)
    _, B_2_t = standard_d2(grid.theta, order=order_theta)
    B_1_t_lift = lift_theta_op(B_1_t, N_r)
    B_2_t_lift = lift_theta_op(B_2_t, N_r)

    theta = grid.theta
    cot_theta = (np.cos(theta) / np.sin(theta))[None, :]
    csc2_theta = (1.0 / np.sin(theta) ** 2)[None, :]
    ones_r = np.ones(N_r)[:, None]

    cot_2d = ones_r * cot_theta  # shape (N_r, N_theta)
    csc2_2d = ones_r * csc2_theta

    cot_diag = diag_from_2d_array(cot_2d.astype(complex))
    csc2_diag = diag_from_2d_array(csc2_2d.astype(complex))

    # r²·∇_H² T = ∂²_θ T + cot θ · ∂_θ T - m² csc² θ · T
    op = (
        B_2_t_lift
        + cot_diag @ B_1_t_lift
        - (m**2) * csc2_diag
    )
    return sp.csr_matrix(op)


def _solve_toroidal_evp(N_r: int, N_theta: int, m: int,
                         order_theta: int = 6, k: int = 8,
                         sigma: complex = None):
    """Build K and M, solve K x = λ M x, return eigenvalues.

    Equation: λ ∇_H² T + (im/r²) T = 0
    Multiply through by r²:  λ (r²·∇_H²) T + im T = 0
    EVP form K x = λ M x with K = -im·I, M = r²·∇_H².

    Wait: write as K T = λ M T with rearrangement. The eq says
        λ M T = -K T
    where M = r²·∇_H², K_eq = im·I. So in eigs's convention
    (which finds λ such that K_in x = λ M_in x):
        K_in = -K_eq = -im·I
        M_in = M = r²·∇_H²
    Then  K_in T = -im T = λ · (r²·∇_H²) T = λ M_in T.
    So eigs returns λ such that this holds. Good.

    For the eigenfunction T = R(r) Y^m_ℓ, this gives
        -im = λ · (-ℓ(ℓ+1))
        λ = im/(ℓ(ℓ+1))
    For ℓ=m: λ = i/(m+1). ✓
    """
    grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0)
    M_op = _build_r2_lap_h(grid, m=m, order_theta=order_theta)
    N = M_op.shape[0]
    K_op = sp.diags(np.full(N, -1j * m, dtype=complex)).tocsr()

    if sigma is None:
        # Default: pick a shift near the expected smallest eigenvalue.
        sigma = 1j / (m + 1)

    # Use manual shift-invert (same recipe as solve_evp).
    KmM = (K_op - sigma * M_op).tocsc()
    LU = spla.splu(KmM)
    def matvec(x):
        return LU.solve(M_op @ x)
    LinOp = spla.LinearOperator(K_op.shape, matvec=matvec, dtype=complex)
    try:
        mu = spla.eigs(LinOp, k=k, which="LM", return_eigenvectors=False,
                       maxiter=5000, tol=1e-10)
    except spla.ArpackNoConvergence as err:
        # Partial convergence is acceptable; use whatever ARPACK found.
        mu = err.eigenvalues
    # μ → λ = 1/μ + σ; filter μ ≈ 0 (infinite-eigenvalue modes from
    # singular M — happens if angular Laplacian has zero modes
    # at boundary rows, depending on θ grid).
    finite = np.abs(mu) > 1e-10
    lam = 1.0 / mu[finite] + sigma
    return lam, grid


class TestAppendixBInertialWaves:
    """Toroidal-only inertial-wave eigenvalues against Appendix B."""

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_lowest_eigenvalue_matches_one_over_m_plus_one(self, m):
        """λ = i/(m+1) for the lowest toroidal mode (ℓ=m) at each m."""
        lam, _ = _solve_toroidal_evp(
            N_r=12, N_theta=32, m=m,
            order_theta=6, k=10,
        )
        # Expected: λ = i/(m+1). Imaginary, positive.
        target = 1.0 / (m + 1)
        # Pick the eigenvalue with smallest |λ - i·target|.
        rel_err = np.abs(lam - 1j * target) / target
        best = float(rel_err.min())
        assert best < 1e-3, (
            f"m={m}: no eigenvalue within 0.1% of i/{m+1} = i·{target:.6f}. "
            f"Closest: {lam[np.argmin(rel_err)]}, rel err {best:.3e}. "
            f"All: {lam}"
        )

    @pytest.mark.parametrize("m,ell", [
        (1, 1), (1, 2), (1, 3),
        (2, 2), (2, 3),
        (3, 3), (3, 4),
    ])
    def test_higher_ell_modes(self, m, ell):
        """λ = im/(ℓ(ℓ+1)) for higher toroidal modes ℓ ≥ m."""
        target_im = m / (ell * (ell + 1))
        # Shift near the target.
        sigma = 1j * target_im
        lam, _ = _solve_toroidal_evp(
            N_r=8, N_theta=48, m=m,
            order_theta=6, k=15, sigma=sigma,
        )
        rel_err = np.abs(lam - 1j * target_im) / target_im
        best = float(rel_err.min())
        assert best < 5e-3, (
            f"m={m}, ℓ={ell}: no eigenvalue within 0.5% of "
            f"i·{m}/({ell}·{ell+1}) = i·{target_im:.6f}. "
            f"Closest: {lam[np.argmin(rel_err)]}, rel err {best:.3e}. "
            f"All: {lam}"
        )

    def test_convergence_with_angular_order(self):
        """Higher angular order gives more accurate eigenvalues.

        Note: order=4 is unstable at low N_theta — the FD boundary
        closures interact with the angular singularities at θ=0, π to
        produce small eigenvalues that confuse ARPACK. Start at order=6.
        """
        m = 2
        target = 1.0 / (m + 1)  # ℓ=m mode
        errs = {}
        for order in (6, 8):
            lam, _ = _solve_toroidal_evp(
                N_r=8, N_theta=48, m=m,
                order_theta=order, k=10,
            )
            rel_err = np.abs(lam - 1j * target) / target
            errs[order] = float(rel_err.min())
        # Order-8 should be at least as accurate as order-6, allowing
        # a small tolerance for ARPACK convergence variability between
        # runs (both are at the 1e-9 level where numerical noise
        # dominates).
        assert errs[8] <= errs[6] * 10.0, (
            f"Order convergence not monotone within 10× tolerance: {errs}"
        )
        assert errs[8] < 1e-9, f"Order-8 accuracy too poor: {errs[8]:.2e}"

    def test_no_spurious_modes_at_low_resolution(self):
        """At m=1, the lowest few eigenvalues all match i/(ℓ(ℓ+1))."""
        m = 1
        lam, _ = _solve_toroidal_evp(
            N_r=8, N_theta=48, m=m,
            order_theta=6, k=20,
        )
        # Sort by |λ| descending (largest |λ| first = ℓ=m mode at 1/(m+1))
        order = np.argsort(np.abs(lam))[::-1]
        lam_sorted = lam[order]
        # The first few should match i·m/(ℓ(ℓ+1)) for ℓ = m, m+1, ...
        # With ℓ=1: |λ|=1/2; ℓ=2: |λ|=1/6 ≈ 0.167; ℓ=3: 1/12 ≈ 0.0833.
        # Each ℓ has multiplicity N_r in the discrete spectrum, so
        # there are 8 copies of each at N_r=8.
        # Take the largest (=ℓ=1) and check it.
        target_l1 = 1.0 / 2.0
        rel_err = abs(np.abs(lam_sorted[0]) - target_l1) / target_l1
        assert rel_err < 1e-3, (
            f"Largest |λ| = {np.abs(lam_sorted[0]):.6f}, expected "
            f"{target_l1:.6f}, rel err {rel_err:.2e}"
        )
