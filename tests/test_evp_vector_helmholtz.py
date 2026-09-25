"""Vector EVP tests against spherical Bessel zeros.

We solve the *unconstrained* vector Helmholtz EVP:
    Lap(v) = λ v,    v(r=1) = 0,
on the unit ball. This is the natural vector generalization of the
scalar Helmholtz problem and gives the same Bessel-zero spectrum but
with multiplicity from the three Cartesian/spin components.

This is NOT the full Lecoanet §4.3 problem, which adds a divergence
constraint (∇·v = 0) enforced by a pressure Lagrange multiplier. The
saddle-point structure of the constrained system makes the mass
matrix singular on the continuity rows, which produces spurious
eigenvalues from `eigs` in shift-invert mode. Handling the
constrained EVP cleanly requires either dense eigenvalue solvers
(viable only for small problems) or a more careful filter on the
returned spectrum (eliminate eigenvectors with substantial
divergence). For now we test the unconstrained vector form, which
is a clean test of `solve_evp` on a coupled (spin-mixing) operator.

The eigenvalues match the scalar Helmholtz spectrum (zeros of
spherical Bessel functions) with the multiplicities expected from
the three Cartesian components of v:
  - κ = π (j_0):   multiplicity 1 (one Cartesian component contributes per ℓ=0 mode)
  - κ = 4.493 (j_1): multiplicity 3 (full ℓ=1 vector spherical harmonics)
  - κ = 5.763 (j_2): multiplicity 3 (full ℓ=2)
  - κ = 6.283 (j_0, n=1): multiplicity 1
  ...

(The factor-of-3 multiplicity reduces in the m=0 sector but the
eigenvalues themselves are unchanged.)
"""

from __future__ import annotations

import numpy as np
from scipy.special import spherical_jn
from scipy.optimize import brentq

from spdee import (
    Unknown, Lap, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_ball_grid


def _bessel_zero(ell: int, n: int) -> float:
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


class TestVectorHelmholtzEVP:
    """Unconstrained vector Helmholtz EVP on the unit ball, m=0."""

    def _make_compiled(self, order=6, N=32):
        v = Unknown.Vector("v")
        eq = Equation(Lap(v), name="vec_helmholtz")
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
            BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
            BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        return compile_problem(sys_, grid, m=0, order_r=order, order_theta=order)

    def test_smallest_eigenvalue_matches_j0(self):
        """The smallest λ matches the first zero of j_0, κ = π."""
        cp = self._make_compiled(order=6, N=32)
        result = cp.solve_evp(k=4, sigma=-9.0)
        lam = np.real(result.eigenvalues)
        kappa = np.sqrt(-lam[lam < -1.0])
        kappa_min = float(np.min(kappa))
        rel_err = abs(kappa_min - np.pi) / np.pi
        assert rel_err < 1e-4, (
            f"Smallest κ = {kappa_min:.6f}, expected π = {np.pi:.6f}, "
            f"rel err {rel_err:.2e}"
        )

    def test_lowest_eigenvalues_match_bessel_zeros(self):
        """The lowest distinct eigenvalues match the scalar Helmholtz
        Bessel-zero spectrum.

        In m=0 the multiplicities reduce, but the distinct eigenvalues
        are the same: π, 4.4934 (j_1), 5.7635 (j_2), 6.2832 (2π = 2nd j_0),
        6.9879 (j_3), 7.7253 (2nd j_1), ...
        """
        cp = self._make_compiled(order=6, N=32)
        result = cp.solve_evp(k=12, sigma=-9.0)
        lam = np.real(result.eigenvalues)
        kappa_all = np.sort(np.sqrt(-lam[lam < -1.0]))

        # Compact to unique values (within rel-tol 1e-3).
        unique = []
        for k in kappa_all:
            if not unique or abs(k - unique[-1]) / k > 1e-3:
                unique.append(float(k))

        # Build the analytic target list.
        targets = []
        for (ell, n) in [
            (0, 0), (1, 0), (2, 0), (0, 1), (3, 0), (1, 1),
        ]:
            targets.append(_bessel_zero(ell, n))
        targets.sort()

        # Compare the first len(targets) measured values.
        for i, target in enumerate(targets):
            assert i < len(unique), (
                f"Only {len(unique)} unique eigenvalues found; need {len(targets)}: "
                f"{unique}"
            )
            rel_err = abs(unique[i] - target) / target
            assert rel_err < 1e-3, (
                f"Distinct eigenvalue {i}: measured κ = {unique[i]:.6f}, "
                f"target κ = {target:.6f}, rel err {rel_err:.2e}. "
                f"All unique measured: {unique}"
            )

    def test_convergence_with_order(self):
        """Eigenvalue accuracy improves dramatically with stencil order."""
        target = _bessel_zero(1, 0)  # first j_1 zero ≈ 4.4934
        rel_errs = {}
        for order in [4, 6, 8]:
            cp = self._make_compiled(order=order, N=32)
            result = cp.solve_evp(k=8, sigma=-9.0)
            lam = np.real(result.eigenvalues)
            kappa = np.sort(np.sqrt(-lam[lam < -1.0]))
            # Find the eigenvalue closest to the j_1 first zero.
            idx = int(np.argmin(np.abs(kappa - target)))
            rel_errs[order] = abs(kappa[idx] - target) / target

        # Higher order is strictly better.
        assert rel_errs[8] < rel_errs[6] < rel_errs[4], (
            f"Convergence not monotone in order: {rel_errs}"
        )
        # And the order-8 result is very accurate.
        assert rel_errs[8] < 1e-9, (
            f"Order-8 rel err {rel_errs[8]:.2e}; expected < 1e-9"
        )

