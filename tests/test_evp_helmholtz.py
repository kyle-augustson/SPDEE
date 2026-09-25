"""Eigenvalue tests against analytic spherical Bessel zeros.

Tracks Lecoanet et al. 2019, §4.2 (scalar Helmholtz):
    ∇²f + κ² f = 0,    f(r=1) = 0,
on the unit ball. Eigenvalues are κ² where j_ℓ(κ) = 0 for each ℓ.

For m=0, the spectrum of the discretized operator on the (r, θ) tensor
grid is the union over ℓ ≥ 0 of the spherical-Bessel zeros squared:
    spectrum = {κ_{ℓ,n}²} where j_ℓ(κ_{ℓ,n}) = 0.

The lowest eigenvalue is κ² = π² ≈ 9.8696 (the first zero of j_0).
The next several are zeros of j_1 (4.4934, 7.7253, ...), j_2 (5.7635,
9.0950, ...), etc.

spdee compiles `Lap(u)` as `r²·Lap u` and uses mass `M = diag(r²)`. So
the discrete generalized eigenvalue problem is:
    (r² Lap_disc) @ x = λ (r² I) @ x
with λ_spdee = -κ². The analytic κ is recovered as √(-λ_spdee).
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn

from spdee import (
    Unknown, Lap, Equation, EquationSystem,
    BoundaryCondition, Boundary,
)
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_ball_grid


def _bessel_zero(ell: int, n: int) -> float:
    """The n-th positive zero of j_ℓ (n=0 is the smallest)."""
    # Search in a window. Spherical-bessel zeros are approximately
    # (n + ℓ/2 + 1)π for large n; bracket with the analytic asymptotic.
    from scipy.optimize import brentq
    # First few zeros found by bisection in a generous range.
    zeros = []
    # Start each bracket just past the previous zero (or 0).
    x_left = 1e-6
    while len(zeros) <= n:
        # Step forward in small increments to find sign changes.
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
            if x_right > 200:
                raise RuntimeError(
                    f"Couldn't find {n}-th zero of j_{ell} below x=200"
                )
    return zeros[n]


def _collect_low_spectrum(num_per_ell=3, max_ell=4):
    """Return sorted list of (κ, ℓ, n) for the lowest spherical-Bessel zeros."""
    out = []
    for ell in range(max_ell + 1):
        for n in range(num_per_ell):
            kappa = _bessel_zero(ell, n)
            out.append((kappa, ell, n))
    out.sort(key=lambda t: t[0])
    return out


class TestScalarHelmholtzEVP:
    """Lecoanet §4.2 scalar Helmholtz EVP on the unit ball, m=0."""

    def test_lowest_eigenvalue_is_pi(self):
        """The smallest eigenvalue is κ² = π² (ground state of j_0)."""
        u = Unknown.Scalar("u")
        eq = Equation(Lap(u), name="helmholtz")
        bcs = [BoundaryCondition(Boundary.R_OUTER, u, name="dirichlet")]
        sys_ = EquationSystem([eq], bcs)
        grid = make_ball_grid(N_r=32, N_theta=24, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=6)

        # Shift-invert near 0 — we want the smallest-magnitude eigenvalues.
        # The target is λ = -π² ≈ -9.87, so sigma slightly negative pulls
        # us to that cluster.
        result = cp.solve_evp(k=6, sigma=-9.0)
        kappa_sq = -result.eigenvalues  # λ_spdee = -κ²
        # Real-part comparison — eigs returns complex even for real spectra.
        kappa = np.sqrt(np.real(kappa_sq))
        kappa_sorted = np.sort(kappa)
        # The smallest κ should be π.
        rel_err = abs(kappa_sorted[0] - np.pi) / np.pi
        assert rel_err < 1e-4, (
            f"Smallest κ = {kappa_sorted[0]:.6f}, expected π = {np.pi:.6f}, "
            f"rel err {rel_err:.2e}"
        )

    def test_lowest_six_eigenvalues_match_bessel_zeros(self):
        """The lowest six eigenvalues match the first six Bessel zeros
        across ℓ ∈ {0, 1, 2, 3}."""
        u = Unknown.Scalar("u")
        eq = Equation(Lap(u), name="helmholtz")
        bcs = [BoundaryCondition(Boundary.R_OUTER, u, name="dirichlet")]
        sys_ = EquationSystem([eq], bcs)
        grid = make_ball_grid(N_r=48, N_theta=32, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=6)

        # Get the analytic targets.
        targets = _collect_low_spectrum(num_per_ell=3, max_ell=4)
        analytic = np.array([t[0] for t in targets])

        # Shift-invert near the smallest expected eigenvalue (λ = -π²).
        # `eigs` returns the k eigenvalues closest to sigma in the
        # shift-invert sense; sigma slightly below the smallest λ
        # captures the bottom of the spectrum.
        sigma = -np.pi ** 2 * 0.9  # ≈ -8.88

        result = cp.solve_evp(k=12, sigma=sigma)
        lam = np.real(result.eigenvalues)
        # We want the SMALLEST κ values, i.e., LARGEST (least negative) λ.
        lam_desc = np.sort(lam)[::-1]  # most positive (= least negative) first
        kappa_meas = np.sqrt(-lam_desc)

        # Check the lowest 6.
        n_check = 6
        for i in range(n_check):
            rel_err = abs(kappa_meas[i] - analytic[i]) / analytic[i]
            assert rel_err < 1e-3, (
                f"Eigenvalue {i}: measured κ = {kappa_meas[i]:.6f}, "
                f"analytic κ = {analytic[i]:.6f} (ℓ={targets[i][1]}, n={targets[i][2]}), "
                f"rel err {rel_err:.2e}"
            )
