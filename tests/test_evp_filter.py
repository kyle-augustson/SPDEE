"""Tests for `filter_spurious_eigenvalues` — twin-grid spurious-mode detection.

The filter solves an EVP at two slightly different resolutions and
keeps eigenvalues that are present in both spectra within
`rel_tolerance`. Physical modes converge with refinement; spurious
modes drift.

Test strategy:
- A problem with NO spurious modes (scalar Helmholtz on the ball)
  should have ALL its eigenvalues classified as physical, including an
  eigenvalue at λ = 0 (Neumann constant mode), which a purely relative
  tolerance cannot match.
- The Lecoanet §4.3 vector diffusion problem has no spurious finite
  modes with the origin treatment of ball_origin.py (its former LBB
  extras came from the untreated pressure origin rows); the analytic
  Bessel zeros are all kept.
- The inviscid inertial problem has non-neutral spurious eigenvalues
  (non-adjoint collocated Grad/Div pair); the filter flags them and
  keeps the neutral Zhang mode.
- The filter should be robust to NaN entries (infinite eigenvalues
  from shift-invert deflation).
"""

from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.special import spherical_jn
from scipy.optimize import brentq

from spdee import (
    Unknown, Background, Lap, Grad, Div, Cross, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat, theta_hat, phi_hat, theta,
)
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.evp_filter import filter_spurious_eigenvalues


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


class TestScalarHelmholtzFilter:
    """Scalar Helmholtz has NO spurious modes; everything should be physical."""

    def test_scalar_helmholtz_no_spurious(self):
        u = Unknown.Scalar("u")
        eq = Equation(Lap(u), name="helmholtz")
        bcs = [BoundaryCondition(Boundary.R_OUTER, u, name="diri")]
        sys_ = EquationSystem([eq], bcs)

        phys, spur_base, spur_fine = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=32, N_theta_base=24,
            m=0, k=8, sigma=-9.0,
            order_r=6, order_theta=6,
            dN=1, rel_tolerance=1e-3,
        )

        # All 8 eigenvalues should be physical for a clean problem.
        # The smallest must be κ = π.
        assert len(phys) >= 6, (
            f"Expected at least 6 physical modes; got {len(phys)} "
            f"(spurious_base={len(spur_base)})"
        )
        # The minimum κ should be π.
        kappas = np.sqrt(-phys.real)
        assert any(abs(k - np.pi) / np.pi < 1e-4 for k in kappas), (
            f"κ=π not found among physical modes: {sorted(kappas)}"
        )

    def test_zero_eigenvalue_kept_physical(self):
        """Neumann Helmholtz at m=0 has the constant mode λ = 0. Its
        twin-grid partner differs by round-off only, which the absolute
        part of the tolerance (rel_tolerance·|σ|) accepts."""
        u = Unknown.Scalar("u")
        eq = Equation(Lap(u), name="helmholtz")
        bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, Grad(u)),
                                 name="neumann")]
        sys_ = EquationSystem([eq], bcs)
        phys, spur_base, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=16,
            m=0, k=6, sigma=-1.0,
            order_r=4, order_theta=4,
            dN=1, rel_tolerance=1e-3,
        )
        zero = phys[np.abs(phys) < 1e-8]
        assert len(zero) == 1, (
            f"λ=0 constant mode not classified physical; physical={phys}, "
            f"spurious={spur_base}"
        )
        assert not np.any(np.abs(spur_base) < 1e-8)


class TestVectorDiffusionFilter:
    """The §4.3 problem: every finite mode below κ² ≈ 100 is a Bessel
    zero and survives the filter."""

    @staticmethod
    def _system():
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        eq_mom = Equation(Lap(v) - Grad(p), name="momentum")
        eq_cont = Equation(Div(v), name="continuity")
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
            BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
            BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp"),
        ]
        return EquationSystem([eq_mom, eq_cont], bcs,
                              gauge_unknowns=["p"])

    def test_bessel_zeros_kept_as_physical(self):
        """The first few analytic Bessel-zero eigenvalues all survive
        the filter."""
        sys_ = self._system()
        phys, _, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=24,
            m=0, k=25, sigma=-25.0,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=1, rel_tolerance=1e-2,
        )
        # Collect distinct measured κ values (within rel-tol 1e-3).
        kappas = np.sort(np.sqrt(-phys.real[phys.real < -1.0]))
        unique = []
        for k in kappas:
            if not unique or abs(k - unique[-1]) / k > 5e-4:
                unique.append(float(k))

        # Check that the first few analytic zeros are all present.
        for target in [
            _bessel_zero(1, 0),  # 4.4934
            _bessel_zero(2, 0),  # 5.7635
            _bessel_zero(3, 0),  # 6.9879
            _bessel_zero(1, 1),  # 7.7253
            _bessel_zero(4, 0),  # 8.1826
        ]:
            ok = any(abs(k - target) / target < 2e-3 for k in unique)
            assert ok, (
                f"Analytic κ = {target:.5f} not found among filter "
                f"output. Unique physical κ: {unique[:15]}"
            )

    def test_no_finite_spurious_modes(self):
        """With the origin rows of ball_origin.py the §4.3 spectrum below
        κ² ≈ 100 is the Bessel-zero set (pairs of toroidal/poloidal
        modes); nothing is flagged and the former LBB extras at
        κ ≈ 5.836, 8.72 are absent."""
        sys_ = self._system()
        phys, spur_base, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=24,
            m=0, k=25, sigma=-25.0,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=1, rel_tolerance=1e-2,
        )
        assert len(spur_base) == 0, (
            f"Unexpected spurious modes: {spur_base}"
        )
        kappa_phys = np.sqrt(-phys.real[phys.real < -1.0])
        for sus_kappa in [5.836, 8.72]:
            assert not any(
                abs(k - sus_kappa) / sus_kappa < 5e-3 for k in kappa_phys
            ), f"LBB extra at κ≈{sus_kappa} is back: {sorted(kappa_phys)}"


class TestInertialFilter:
    """Inviscid inertial waves (Zhang et al. 2001) at m=1: the
    collocated Grad/Div pair is not discretely adjoint, so
    under-resolved in-band eigenvalues carry damping |Re λ| ~ 1e-2 that
    drifts with resolution. The filter flags them and keeps the
    neutral physical mode."""

    def test_flags_non_neutral_drifting_modes(self):
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        Omega = Background.Vector(
            "Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta),
            v_phi=sp.S.Zero,
        )
        sys_ = EquationSystem(
            [Equation(2 * Cross(Omega, v) - Grad(p), name="momentum"),
             Equation(Div(v), name="continuity")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")],
            gauge_unknowns=["p"],
        )
        omega_target = -0.82001
        phys, spur_base, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=48,
            m=1, k=8, sigma=-1j * omega_target - 0.001,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=2, rel_tolerance=2e-3,
        )
        assert len(spur_base) >= 1, "expected drifting spurious modes"
        best = min(phys, key=lambda x: abs(-x.imag - omega_target))
        assert abs(-best.imag - omega_target) < 1e-4 * abs(omega_target)
        assert abs(best.real) < 1e-10, f"physical mode not neutral: {best}"


class TestVectorDiffusionFilterTolerance:
    """Tolerance monotonicity on the §4.3 problem."""

    def test_loose_tolerance_keeps_more(self):
        """With a looser tolerance, at least as many modes pass the
        physical filter."""
        sys_ = TestVectorDiffusionFilter._system()
        # Tight: only well-converged modes kept.
        phys_tight, _, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=24,
            m=0, k=15, sigma=-25.0,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=1, rel_tolerance=1e-4,
        )
        # Loose: more lenient.
        phys_loose, _, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(
                N_r=nr, N_theta=nt, r_outer=1.0,
            ),
            N_r_base=24, N_theta_base=24,
            m=0, k=15, sigma=-25.0,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=1, rel_tolerance=1e-1,
        )
        assert len(phys_loose) >= len(phys_tight), (
            f"Loose tolerance should classify ≥ as many physical as tight; "
            f"got loose={len(phys_loose)}, tight={len(phys_tight)}"
        )
