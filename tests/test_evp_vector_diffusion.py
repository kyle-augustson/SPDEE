"""Vector diffusion EVP tests against Lecoanet et al. 2019 §4.3.

The constrained linear diffusion problem:
    ∂t u + ∇p - ∇²u = 0,    ∇·u = 0,    u(r=1) = 0  (no-slip).

Taking ∂t u = -λ u gives the EVP
    Lap(u) - Grad(p) = λ u,    Div(u) = 0,    u|_{r=1} = 0.

Analytic dispersion (Lecoanet Table 7, no-slip row):
    Toroidal (ℓ ≥ 1):   j_ℓ(√λ)     = 0  →  λ = κ²
    Poloidal (ℓ ≥ 1):   j_{ℓ+1}(√λ) = 0  →  λ = κ²
Each Bessel zero appears with double multiplicity (tor at ℓ, pol at
ℓ+1 give same κ). The smallest physical eigenvalue is λ = κ² with
κ = first zero of j_1 ≈ 4.4934, so λ ≈ -20.19 (in spdee's `λ_spdee =
-κ²` convention).

This test uses the `mass_equations={'momentum': 'v'}` form of
`solve_evp`, which correctly maps the momentum equation's λ-action
onto the velocity unknown's state columns. Without that mapping the
M-matrix columns are misaligned and the discrete EVP returns a
contaminated spectrum.

The discrete spectrum also contains a few N-dependent "extra" modes
not matching any analytic value — these are LBB-type colocated-FD
artifacts. The tests verify the physical Bessel-zero modes are
*present* in the spectrum to good accuracy, but don't demand the
entire spectrum be free of extras.
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

from spdee import (
    Unknown, Lap, Grad, Div, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_ball_grid


def _bessel_zero(ell: int, n: int) -> float:
    """The n-th positive zero of j_ℓ (n=0 is the smallest)."""
    zeros = []
    x_left = 1e-6
    while len(zeros) <= n:
        x_right = x_left + 0.5
        f_left = spherical_jn(ell, x_left)
        while True:
            f_right = spherical_jn(ell, x_right)
            if f_left * f_right < 0:
                root = brentq(lambda x: spherical_jn(ell, x), x_left, x_right)
                zeros.append(root)
                x_left = x_right
                break
            x_left = x_right
            x_right += 0.5
            f_left = f_right
    return zeros[n]


def _make_compiled(N=24, order=6):
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    eq_mom = Equation(Lap(v) - Grad(p), name="momentum")
    eq_cont = Equation(Div(v), name="continuity")
    bcs = [
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
        BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
        BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp"),
    ]
    sys_ = EquationSystem([eq_mom, eq_cont], bcs, gauge_unknowns=["p"])
    grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    return compile_problem(sys_, grid, m=0, order_r=order, order_theta=order)


def _find_match(measured_kappas, target, tol_rel=1e-3):
    """Find the measured κ closest to target; return (κ, rel_err) or None."""
    if len(measured_kappas) == 0:
        return None
    diffs = np.abs(measured_kappas - target)
    i = int(np.argmin(diffs))
    rel = diffs[i] / target
    if rel < tol_rel:
        return float(measured_kappas[i]), float(rel)
    return None


class TestVectorDiffusionEVP:
    """Lecoanet 2019 §4.3: no-slip linear diffusion EVP, m=0."""

    def test_smallest_eigenvalue_is_first_j1_zero(self):
        """Smallest physical κ = first zero of j_1 (toroidal ℓ=1, n=0)."""
        cp = _make_compiled(N=24, order=6)
        kappa_target = _bessel_zero(1, 0)
        # Shift slightly above the target in λ-space (less negative).
        sigma = -(kappa_target ** 2) * 0.9
        result = cp.solve_evp(
            k=10, sigma=sigma, mass_equations={"momentum": "v"},
        )
        lam = np.real(result.eigenvalues)
        finite = lam[np.isfinite(lam)]
        kappa = np.sqrt(-finite[finite < -1.0])
        match = _find_match(kappa, kappa_target, tol_rel=1e-3)
        assert match is not None, (
            f"No measured κ within 0.1% of analytic {kappa_target:.6f}. "
            f"Measured: {sorted(kappa)[:10]}"
        )
        kappa_meas, rel_err = match
        assert rel_err < 1e-3, (
            f"Best measured κ = {kappa_meas:.6f}, target {kappa_target:.6f}, "
            f"rel err {rel_err:.2e}"
        )

    def test_first_six_bessel_zeros_in_spectrum(self):
        """The first 6 distinct analytic Bessel-zero eigenvalues all
        appear in the measured spectrum to good accuracy."""
        cp = _make_compiled(N=24, order=6)
        # Targets: j_ℓ(κ)=0 for ℓ ∈ {1, 2, 3, 4, 5} (first zero each)
        # plus j_1 second zero. Collect first 6 unique values.
        targets = sorted(set([
            _bessel_zero(1, 0),  # 4.4934
            _bessel_zero(2, 0),  # 5.7635
            _bessel_zero(3, 0),  # 6.9879
            _bessel_zero(1, 1),  # 7.7253
            _bessel_zero(4, 0),  # 8.1826
            _bessel_zero(2, 1),  # 9.0950
        ]))
        # Choose sigma well below the smallest target to capture
        # the bottom of the spectrum.
        sigma = -25.0
        result = cp.solve_evp(
            k=25, sigma=sigma, mass_equations={"momentum": "v"},
        )
        lam = np.real(result.eigenvalues)
        finite = lam[np.isfinite(lam)]
        kappa = np.sqrt(-finite[(finite < -1.0) & (finite > -200.0)])

        misses = []
        for t in targets:
            match = _find_match(kappa, t, tol_rel=2e-3)
            if match is None:
                misses.append(t)
        assert not misses, (
            f"Missed analytic eigenvalues: {misses}. "
            f"Measured κ values: {sorted(kappa)[:20]}"
        )

    def test_higher_order_improves_accuracy(self):
        """Going from order=4 to order=6 reduces the error in the
        smallest physical eigenvalue."""
        kappa_target = _bessel_zero(1, 0)
        sigma = -(kappa_target ** 2) * 0.9

        errs = {}
        for order in (4, 6):
            cp = _make_compiled(N=24, order=order)
            result = cp.solve_evp(
                k=10, sigma=sigma, mass_equations={"momentum": "v"},
            )
            lam = np.real(result.eigenvalues)
            finite = lam[np.isfinite(lam)]
            kappa = np.sqrt(-finite[finite < -1.0])
            match = _find_match(kappa, kappa_target, tol_rel=1e-1)
            assert match is not None, (
                f"order={order}: no measured κ within 10% of target. "
                f"Measured: {sorted(kappa)[:10]}"
            )
            _, errs[order] = match

        assert errs[6] < errs[4], (
            f"Higher order didn't improve accuracy: "
            f"order=4 rel err {errs[4]:.2e}, order=6 rel err {errs[6]:.2e}"
        )
