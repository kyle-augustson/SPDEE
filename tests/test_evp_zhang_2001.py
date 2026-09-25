"""Inertial-wave EVP test against Zhang, Earnshaw, Liao & Busse 2001.

The paper "On inertial waves in a rotating fluid sphere" (J. Fluid
Mech. 437, 103–119) gives explicit polynomial eigenfunctions for the
inviscid inertial-wave problem on a full rotating sphere with
impenetrability BC at r=1:

    ∂_t u + 2 k × u = -∇P,  ∇ · u = 0,
    r · u = 0 at r = 1

where k = ẑ is the rotation axis. With time-dependence e^{i(mφ + 2σt)},
this is a generalized EVP for the half-frequency σ (frequency ω = 2σ).

Zhang et al. give explicit eigenvalue equations (Eqs. 2.15 for
equatorially symmetric and 2.16 for antisymmetric modes) — each a
polynomial in σ whose roots correspond to a family of modes indexed
by (N, m, n) where N is the polynomial degree controlling axial
complexity, m is the azimuthal wavenumber, and n indexes radial
structure. For each N there are 2N symmetric and 2N+1 antisymmetric
roots; tabulated explicit examples in their Figures 1-2 give targets
for N=2, m∈{1, 8}.

spdee solves the EVP as follows:
    K = compiled `2·Cross(Ω, v) - Grad(p)` with Div(v)=0 constraint
    M = natural equation mass (diag(r) per equation) on v
    BC: Dot(r̂, v) = 0 at r=r_outer

Sign convention: spdee's equation `2 Cross(Ω, v) - Grad(p) = λ v`
with Ω = ẑ matches Zhang's `2 ẑ × u + ∇P = -2iσ u` if λ = -2iσ.
Therefore ω = 2σ = -Im(λ_spdee) for the spdee-reported eigenvalues.

Limitations of the colocated FD discretization:
The Div(v)=0 incompressibility constraint paired with Grad(p) creates
LBB-style extras in the spectrum. The twin-grid filter
(`filter_spurious_eigenvalues`) removes these by demanding agreement
between two resolutions, which works robustly for well-resolved
modes. Modes with very small |σ| and equatorial-symmetric parity
(the "geostrophic" subclass) converge slowly at moderate resolution
and are excluded from this test; they are the same modes that
trouble Rieutord & Valdettaro 1997's spurious-mode filter.

What IS tested: a curated set of 28 well-resolved (m, N, parity, ω)
targets spanning m ∈ {1, 2, 8} and both equatorial parities.
Empirically these recover to rel err < 5e-3 at N_r=24, N_theta=48
(m≤2) or N_r=28, N_theta=64 (m=8), with most matching to better
than 1e-4.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Cross, Grad, Div, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat, Background, theta,
)
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.evp_filter import filter_spurious_eigenvalues


def _build_system():
    """Build the Zhang inertial-wave EquationSystem.

    Inviscid momentum equation `2·k×v - Grad(p) = 0` with `Div(v)=0`
    and impenetrability `Dot(r̂, v) = 0` at r=r_outer. The rotation
    axis Ω = ẑ in spherical coords is `(cos θ, -sin θ, 0)`.
    """
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Omega = Background.Vector(
        "Omega",
        v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    eq_mom = Equation(2 * Cross(Omega, v) - Grad(p), name="momentum")
    eq_cont = Equation(Div(v), name="continuity")
    bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_imp")]
    return EquationSystem([eq_mom, eq_cont], bcs, gauge_unknowns=["p"])


def _find_eigenvalue_near(sys_, m, omega_target, N_r, N_theta, dN=2,
                          rel_tol=2e-3, k=8):
    """Search for a physical eigenvalue near ω_target.

    Returns the closest ω from the filtered spectrum, or None if no
    physical eigenvalue is found near the target.
    """
    sigma_shift = -1j * omega_target - 0.001  # tiny real offset for stability
    try:
        phys, _, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(N_r=nr, N_theta=nt, r_outer=1.0),
            N_r_base=N_r, N_theta_base=N_theta,
            m=m, k=k, sigma=sigma_shift,
            mass_equations={"momentum": "v"},
            order_r=6, order_theta=6,
            dN=dN, rel_tolerance=rel_tol,
        )
    except Exception:
        return None
    phys = list(phys)
    if not phys:
        return None
    best = min(phys, key=lambda x: abs(-x.imag - omega_target))
    return -best.imag  # convert to ω = -Im(λ_spdee)


# Targets curated from a full survey of N=1,2 modes at m=1,2,8 (both parities).
# Each (m, label, ω_target) was verified to recover to rel err < 5e-3 at
# the resolution N_r, N_theta. The "label" encodes (N, parity, classification).
ZHANG_TARGETS_M1 = [
    # (label, ω from Zhang Eq. 2.15/2.16, expected tolerance)
    ("N=1 SYM ω=+1.51", +1.50994, 1e-2),
    ("N=2 SYM ω=+1.806", +1.80600, 1e-3),
    ("N=1 ANTI ω=-0.820", -0.82001, 1e-4),
    ("N=1 ANTI ω=+0.612", +0.61198, 1e-4),
    ("N=1 ANTI ω=+1.708", +1.70802, 1e-4),
    ("N=2 ANTI ω=-1.404", -1.40422, 1e-4),
    ("N=2 ANTI ω=-0.537", -0.53733, 1e-4),
    ("N=2 ANTI ω=+0.440", +0.44045, 1e-4),
    ("N=2 ANTI ω=+1.306", +1.30608, 1e-4),
    ("N=2 ANTI ω=+1.862", +1.86168, 1e-4),
]

ZHANG_TARGETS_M2 = [
    ("N=1 SYM ω=+1.232", +1.23193, 5e-3),
    ("N=2 SYM ω=+1.643", +1.64344, 5e-3),
    ("N=1 ANTI ω=-0.763", -0.76334, 1e-4),
    ("N=1 ANTI ω=+0.467", +0.46690, 1e-4),
    ("N=1 ANTI ω=+1.496", +1.49644, 1e-4),
    # These two sit at 2.2e-4 and 1.7e-3 on 24×48 and converge to 2e-6 on
    # 32×64 (the tighter values of an earlier survey came with the
    # spurious m≠0 gauge row, since removed).
    ("N=2 ANTI ω=-1.308", -1.30800, 5e-4),
    ("N=2 ANTI ω=-0.510", -0.50999, 2e-3),
    ("N=2 ANTI ω=+0.360", +0.35954, 1e-4),
    ("N=2 ANTI ω=+1.153", +1.15306, 1e-3),
    ("N=2 ANTI ω=+1.734", +1.73397, 1e-4),
]

# m=8 needs higher angular resolution; Figure 1's N=2 m=8 quadruple is
# the published reference.
ZHANG_TARGETS_M8 = [
    # Paper Figure 1: ω ∈ {-0.8107, -0.1450, 0.5037, 1.1187} for N=2 m=8 SYM.
    # The two slow ones (-0.145, +0.504) are geostrophic-class and converge
    # poorly at moderate resolution.
    ("N=2 SYM ω=-0.811", -0.81074, 1e-3),
    ("N=2 SYM ω=+1.119", +1.11871, 5e-3),
    ("N=1 ANTI ω=-0.577", -0.57697, 1e-3),
    ("N=1 ANTI ω=+0.194", +0.19394, 5e-3),
    ("N=1 ANTI ω=+0.928", +0.92848, 1e-3),
    ("N=2 ANTI ω=-0.989", -0.98893, 1e-3),
    # 2.1e-3 on 28×64, 3.1e-4 on 36×80 (the 5.4e-4 of an earlier survey
    # came with the spurious m≠0 gauge row, since removed).
    ("N=2 ANTI ω=-0.408", -0.40777, 5e-3),
    ("N=2 ANTI ω=+0.172", +0.17219, 1e-3),
    ("N=2 ANTI ω=+0.734", +0.73423, 5e-3),
    ("N=2 ANTI ω=+1.260", +1.25952, 1e-3),
]


class TestZhang2001InertialWaves:
    """Recover Zhang et al. 2001 polynomial inertial-wave eigenvalues."""

    @pytest.mark.parametrize("label,omega_target,tol", ZHANG_TARGETS_M1,
                             ids=[t[0] for t in ZHANG_TARGETS_M1])
    def test_m1(self, label, omega_target, tol):
        """m=1 N=1,2 modes (both parities, well-resolved subset)."""
        sys_ = _build_system()
        omega = _find_eigenvalue_near(
            sys_, m=1, omega_target=omega_target,
            N_r=24, N_theta=48,
        )
        assert omega is not None, (
            f"m=1 {label}: no physical eigenvalue found near ω={omega_target:+.5f}"
        )
        rel = abs(omega - omega_target) / max(abs(omega_target), 0.05)
        assert rel < tol, (
            f"m=1 {label}: spdee ω={omega:+.5f}, "
            f"Zhang ω={omega_target:+.5f}, rel err {rel:.2e} > tol {tol:.0e}"
        )

    @pytest.mark.parametrize("label,omega_target,tol", ZHANG_TARGETS_M2,
                             ids=[t[0] for t in ZHANG_TARGETS_M2])
    def test_m2(self, label, omega_target, tol):
        """m=2 N=1,2 modes."""
        sys_ = _build_system()
        omega = _find_eigenvalue_near(
            sys_, m=2, omega_target=omega_target,
            N_r=24, N_theta=48,
        )
        assert omega is not None, (
            f"m=2 {label}: no physical eigenvalue found near ω={omega_target:+.5f}"
        )
        rel = abs(omega - omega_target) / max(abs(omega_target), 0.05)
        assert rel < tol, (
            f"m=2 {label}: spdee ω={omega:+.5f}, "
            f"Zhang ω={omega_target:+.5f}, rel err {rel:.2e} > tol {tol:.0e}"
        )

    @pytest.mark.parametrize("label,omega_target,tol", ZHANG_TARGETS_M8,
                             ids=[t[0] for t in ZHANG_TARGETS_M8])
    def test_m8(self, label, omega_target, tol):
        """m=8 (high azimuthal): subset of N=1,2 modes from Figure 1.

        Needs higher N_theta to resolve the (sin θ)^m angular dependence.
        """
        sys_ = _build_system()
        omega = _find_eigenvalue_near(
            sys_, m=8, omega_target=omega_target,
            N_r=28, N_theta=64,
        )
        assert omega is not None, (
            f"m=8 {label}: no physical eigenvalue found near ω={omega_target:+.5f}"
        )
        rel = abs(omega - omega_target) / max(abs(omega_target), 0.05)
        assert rel < tol, (
            f"m=8 {label}: spdee ω={omega:+.5f}, "
            f"Zhang ω={omega_target:+.5f}, rel err {rel:.2e} > tol {tol:.0e}"
        )

    def test_spinover_at_m1_N1ANTI(self):
        """Sanity: the m=1 N=1 ANTI ω=-0.82 mode recovers to high precision.

        The N=1 antisymmetric modes are the simplest non-trivial
        inertial waves in the sphere and exercise the basic
        Cross(Ω, v)-Grad(p) machinery without the higher-order
        polynomial structure of N ≥ 2.
        """
        sys_ = _build_system()
        omega = _find_eigenvalue_near(
            sys_, m=1, omega_target=-0.82001,
            N_r=24, N_theta=48, rel_tol=5e-4,
        )
        assert omega is not None
        rel = abs(omega + 0.82001) / 0.82001
        assert rel < 1e-5, f"Expected high precision; got rel={rel:.2e}"


# ---------------------------------------------------------------------
# Mode identification by eigenfunction structure
# ---------------------------------------------------------------------

def _zhang_pressure_basis(R, T, m, N, parity):
    """Monomials of the pressure of a Zhang (N, m, parity) mode.

    In a full sphere the inertial eigenfunctions are polynomials in the
    Cartesian coordinates; a mode of Zhang's class N has pressure
    s^m e^{i m phi} times a polynomial in (s^2, z) of degree 2N (SYM,
    even in z) or 2N + 1 (ANTI, odd in z), s = r sin(theta), z = r cos(theta).
    """
    s, z = R * np.sin(T), R * np.cos(T)
    degree = 2 * N + (1 if parity == "ANTI" else 0)
    cols = []
    for j in range(degree // 2 + 1):
        for ell in range(degree - 2 * j + 1):
            if (ell % 2 == 0) == (parity == "SYM"):
                cols.append((s ** (m + 2 * j) * z ** ell).ravel())
    return np.array(cols).T


def _polynomial_residual(p, basis, weight):
    """Relative residual of the weighted least-squares fit of p on basis."""
    sw = np.sqrt(weight)
    A, b = basis * sw[:, None], p.ravel() * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.linalg.norm(A @ c - b) / np.linalg.norm(b))


# The three m = 2 targets whose census error did not fall between 12x18
# and 18x27 at order 4 (docs/analytic_benchmarks.md, finding 4).
_M2_SLOW = [t for t in ZHANG_TARGETS_M2 if t[1] in (1.64344, 1.15306, 1.73397)]


@pytest.mark.parametrize("label,omega_target,tol", _M2_SLOW, ids=[t[0] for t in _M2_SLOW])
def test_m2_slow_targets_identified_by_pressure_polynomial(label, omega_target, tol):
    """The three m = 2 targets are the core's modes, identified by the
    eigenvector rather than by the nearest eigenvalue.

    On 24x48 at order 6 the eigenvector nearest each target has a
    pressure that lies in the analytic polynomial space of its Zhang
    class to better than 2 % (the other seven targets fit to 1e-3 or
    better) and its frequency is within 1e-4 of the target. At the
    dense-census resolutions (12x18, 18x27, order 4) these three modes
    are not neutral: each has collided with a neighbouring band mode
    into a complex pair with |Im omega| of 1e-3 to 1e-2 (the
    non-adjoint collocated Grad/Div), so a neutral gate drops them and
    the nearest neutral survivor is a band neighbour at the local
    spacing, whose distance does not fall with N. The real parts of
    the pairs converge (2e-4 to 4e-4 at 24x48, order 4) and the pairs
    close with resolution, which is what this test pins: at 24x48 the
    identified mode is neutral to 1e-8 at order 6 and its imaginary
    part is below 3e-3 at order 4.
    """
    parity = "SYM" if "SYM" in label else "ANTI"
    n_class = int(label.split()[0].split("=")[1])
    sys_ = _build_system()
    grid = make_ball_grid(N_r=24, N_theta=48, r_outer=1.0)
    R, T = grid.mesh()
    weight = (R ** 2 * np.sin(T)).ravel() + 1e-3
    basis = _zhang_pressure_basis(R, T, 2, n_class, parity)
    found = {}
    from spdee.backends.fd2.compiled_problem import compile_problem
    for order in (6, 4):
        cp = compile_problem(sys_, grid, m=2, order_r=order, order_theta=order)
        res = cp.solve_evp(k=8, sigma=-1j * omega_target - 0.001,
                           mass_equations={"momentum": "v"}, return_eigenvectors=True)
        lam = res.eigenvalues
        ok = np.isfinite(lam)
        idx = int(np.argmin(np.where(ok, np.abs(-lam.imag - omega_target), np.inf)))
        omega = -lam[idx].imag
        resid = _polynomial_residual(res.eigenvectors[idx]["p"], basis, weight)
        found[order] = (omega, float(lam[idx].real), resid)
        print(f"BENCH | zhang m=2 {label} order {order} 24x48: omega {omega:+.6f} "
              f"(rel {abs(omega - omega_target) / abs(omega_target):.1e}), "
              f"Im omega {lam[idx].real:+.1e}, polynomial residual {resid:.2e}")
    omega, growth, resid = found[6]
    assert abs(omega - omega_target) / abs(omega_target) < 1e-4, found
    assert resid < 2e-2, found
    assert abs(growth) < 1e-8, found
    omega4, growth4, _ = found[4]
    assert abs(omega4 - omega_target) / abs(omega_target) < 1e-3, found
    assert abs(growth4) < 3e-3, found


# =====================================================================
# STAGGERED backend: Zhang-2001 full-band inertial waves.
#
# The collocated tests above validate the inertial spectrum across
# |ω| ∈ (0, 1.86). This section runs the SAME analytic Zhang targets
# through the STAGGERED (Arakawa-C) backend, across the FULL band.
#
# The staggered assembler must build the Coriolis coupling `c·Cross(Ω,v)`
# at the momentum equation's coefficient c=2. At UNIT strength (half the
# Coriolis force) the high-|ω| pure-inertial modes,
# which track the coupling strength, cap at
# 2Ω_eff = 1 and the band ceiling sits at |ω| ≈ 1.1, while the low-|ω|
# geostrophic modes (set by the Coriolis↔pressure balance) keep their
# correct frequencies. Only the Coriolis blocks scale with c — a global ×2
# would move the whole operator. These tests guard that coefficient.
# =====================================================================


def _staggered_inertial_band(m, N_r, N_theta):
    """Dense staggered inertial-wave solve; the neutral band omega = -Im(lambda)
    with |omega| < 5."""
    from spdee.backends.fd2.grid import make_ball_grid
    from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
    from spdee.solvers import dense
    grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0)
    cp = compile_fd2_staggered(_build_system(), grid, m=m)
    A, B = cp.assemble_evp_matrices()
    res = dense(A, B, sigma=0.3 + 0.2j, backend="auto", vectors=True)
    omega = -res.lam[res.resid < 1e-8].imag       # omega = -Im(lambda), Zhang
    return omega[np.abs(omega) < 5.0]


# The band-coverage tolerance: a mode must sit this close to every curated
# target. It is many times the mean mode spacing, so it is a statement about
# which frequencies the band reaches, not about how well a mode is resolved.
_BAND_MATCH = 0.03

_ZHANG_STAG = [(1, ZHANG_TARGETS_M1), (2, ZHANG_TARGETS_M2)]


class TestZhang2001StaggeredInertialWaves:
    """Band coverage of the staggered core across the full Zhang band.

    With the Coriolis coefficient 2 the staggered operator reaches the
    continuum band |omega| <= 2*Omega = 2, so a mode lands within 0.03 of
    every curated target. A unit-strength coefficient truncates the band at
    |omega| ~ 1.1 and every target above that disappears; that is what
    these tests guard.

    They do not assert the frequency itself. The staggered inviscid band is
    dense -- 960 neutral modes at 16x32 cells, a mean spacing of about
    4e-3 -- so a per-target threshold of 1.5e-2 or 6e-3 is at or below the
    spacing and would pass on any spectrum of that density. The accuracy
    claim the data does support (the RMS error over the target set, its
    convergence order and the discrimination against random control
    frequencies) is in tests/test_ball_staggered.py::TestZhangBall.
    """

    @staticmethod
    def _assert_covered(m, targets, N_r, N_theta):
        om = _staggered_inertial_band(m, N_r, N_theta)
        assert om.size, f"staggered m={m}: the census returned no neutral mode"
        misses = [(lbl, w, float(np.abs(om - w).min())) for lbl, w, _ in targets
                  if np.abs(om - w).min() > _BAND_MATCH]
        assert not misses, (
            f"staggered m={m} at {N_r - 1}x{N_theta}: no mode within "
            f"{_BAND_MATCH} of {misses}; if the missing targets all have "
            f"|omega| > 1.1 the Coriolis coefficient has regressed to unit "
            f"strength -- check the cross_vector dispatch")

    @pytest.mark.parametrize("m,targets", _ZHANG_STAG, ids=["m1", "m2"])
    def test_full_band_covered(self, m, targets):
        """Every curated target has a mode within 0.03 on 16x32 cells."""
        self._assert_covered(m, targets, 17, 32)

    @pytest.mark.slow
    @pytest.mark.parametrize("m,targets", _ZHANG_STAG, ids=["m1", "m2"])
    def test_full_band_covered_24x48(self, m, targets):
        """The same on 24x48 cells."""
        self._assert_covered(m, targets, 25, 48)


class TestStaggeredCoriolisBandReachesContinuum:
    """Guard that the staggered inertial band reaches the continuum 2Ω=2,
    as a measured number (a half-strength Coriolis gives a ceiling ≈1.1)."""

    @pytest.mark.slow
    def test_band_reaches_two(self):
        """The staggered inertial band tops out near |ω| ≈ 2 (continuum
        2Ω). A unit-strength Coriolis coefficient would drop this to ≈1.1."""
        from spdee.backends.fd2.grid import make_ball_grid
        from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
        sys_ = _build_system()
        grid = make_ball_grid(N_r=20, N_theta=32, r_outer=1.0)
        cp = compile_fd2_staggered(sys_, grid, m=1)
        A, B = cp.assemble_evp_matrices()
        from spdee.solvers import dense
        res = dense(A, B, sigma=0.3 + 0.2j, backend="auto", vectors=True)
        lam = res.lam[res.resid < 1e-8]
        omega = -lam.imag
        omega = omega[np.abs(omega) < 5.0]
        ceiling = float(np.abs(omega).max())
        # Continuum band is |ω| ≤ 2Ω = 2; discrete overshoots slightly.
        assert ceiling > 1.8, (
            f"band ceiling {ceiling:.3f} — if this dropped back near 1.1 the "
            f"Coriolis coefficient likely regressed to unit strength "
            f"(should be 2; see _extract_coriolis_coeff).")
        assert ceiling < 2.5, (
            f"band ceiling {ceiling:.3f} unexpectedly high — check the "
            f"Coriolis coefficient is 2, not larger.")


# =====================================================================
# MECHANISM diagnostic (fast: no full eigensolve).
#
# A fast probe of the divergence-free Coriolis spectral radius (max |ω| the
# operator can host), computed from the assembler's operator pieces without a
# dense generalized eigensolve. It matches the full-solve band ceiling.
#
# The div-free ceiling is directly proportional to the Coriolis coefficient
# (coeff 1 → ~1.1, coeff 2 → ~2.2): with half the Coriolis force the band
# caps at 2Ω_eff = 1 (×coupling). That linear scaling is the signature of the
# coupling strength; averaging-ordering, Gilman flux-form and
# transpose-independence all leave the ceiling unchanged, so it is not an
# order/averaging effect.
# =====================================================================


def _divfree_coriolis_band_ceiling(N_r, N_theta, m, order, coriolis_coeff=2.0):
    """Spectral radius (max |Im λ| ≡ max |ω|) of the staggered Coriolis skew
    operator restricted to the discrete divergence-free velocity subspace.

    Built directly from the assembler's own operator pieces
    (_build_staggered_gradient_blocks + the Coriolis interpolation recipe),
    so it tracks any change to those operators. ``coriolis_coeff`` mirrors
    the assembler's coupling coefficient (the `c` in `c·Cross(Ω,v)`); the
    ceiling scales linearly with it. No dense generalized eigensolve — only
    an SVD for the null space and a small reduced eig.
    """
    import scipy.sparse as sp
    from spdee.backends.fd2.grid import make_ball_grid
    from spdee.backends.fd2.staggered_compile import (
        StaggeredGrids, _build_staggered_gradient_blocks, _avg_1d,
        _stag_interp_1d_lagrange)

    grids = StaggeredGrids.from_grid(
        make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0))
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_e, Nr_c = len(r_edge), len(r_center)
    Nt_e, Nt_c = len(th_edge), len(th_center)
    hr = r_edge[1] - r_edge[0]
    ht = th_center[1] - th_center[0]
    n_vr, n_vt, n_vp = Nr_e * Nt_c, Nr_c * Nt_e, Nr_c * Nt_c

    I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
    I_rc = sp.eye(Nr_c, format="csr", dtype=complex)

    def vol(rv, tv):
        a = (rv ** 2)[:, None] * np.sin(tv)[None, :]
        return a.ravel().astype(complex) * hr * ht
    W_vr, W_vt, W_vp = (vol(r_edge, th_center), vol(r_center, th_edge),
                        vol(r_center, th_center))
    W_p = vol(r_center, th_center)
    W_v = np.concatenate([W_vr, W_vt, W_vp]).real

    if order <= 2:
        P_vp2vr = sp.kron(_avg_1d(Nr_e, Nr_c), I_tc, format="csr")
        A_t = np.zeros((Nt_e, Nt_c), dtype=complex)
        for j in range(1, Nt_e - 1):
            A_t[j, j - 1] = 0.5
            A_t[j, j] = 0.5
        P_vp2vt = sp.kron(I_rc, sp.csr_matrix(A_t), format="csr")
    else:
        P_vp2vr = sp.kron(
            _stag_interp_1d_lagrange(r_center, r_edge, order), I_tc,
            format="csr")
        P_vp2vt = sp.kron(
            I_rc, _stag_interp_1d_lagrange(th_center, th_edge, order),
            format="csr")
    P_vr2vp = sp.diags(1.0 / W_vp) @ P_vp2vr.conj().T @ sp.diags(W_vr)
    P_vt2vp = sp.diags(1.0 / W_vp) @ P_vp2vt.conj().T @ sp.diags(W_vt)

    def diagrc(rv, tv, fn):
        a = fn(rv[:, None], tv[None, :])
        return sp.diags(np.asarray(a).ravel().astype(complex), format="csr")
    sin_vr = diagrc(r_edge, th_center, lambda r, t: np.sin(t) + 0 * r)
    cos_vt = diagrc(r_center, th_edge, lambda r, t: np.cos(t) + 0 * r)
    sin_vp = diagrc(r_center, th_center, lambda r, t: np.sin(t) + 0 * r)
    cos_vp = diagrc(r_center, th_center, lambda r, t: np.cos(t) + 0 * r)
    Cor_r_vp = -coriolis_coeff * (sin_vr @ P_vp2vr)
    Cor_t_vp = -coriolis_coeff * (cos_vt @ P_vp2vt)
    Cor_p_vr = coriolis_coeff * (sin_vp @ P_vr2vp)
    Cor_p_vt = coriolis_coeff * (cos_vp @ P_vt2vp)

    Z = lambda a, b: sp.csr_matrix((a, b), dtype=complex)
    S = sp.bmat([
        [Z(n_vr, n_vr), Z(n_vr, n_vt), Cor_r_vp],
        [Z(n_vt, n_vr), Z(n_vt, n_vt), Cor_t_vp],
        [Cor_p_vr, Cor_p_vt, Z(n_vp, n_vp)],
    ], format="csr").toarray()

    G_r, G_t, G_p = _build_staggered_gradient_blocks(grids, m, order)
    G = sp.vstack([G_r, G_t, G_p], format="csr")
    Div = -(sp.diags(1.0 / W_p) @ G.conj().T
            @ sp.diags(np.concatenate([W_vr, W_vt, W_vp]))).toarray()

    # restrict to weightful DOFs (θ-edge poles carry zero volume weight)
    sqrtW = np.sqrt(W_v)
    ok = sqrtW > 1e-14
    Sok = S[np.ix_(ok, ok)]
    Divok = Div[:, ok]
    sqrtWok = sqrtW[ok]
    Dw = Divok @ np.diag(1.0 / sqrtWok)
    _, s_, Vh = np.linalg.svd(Dw)
    tol = max(Dw.shape) * (np.max(s_) if len(s_) else 0.0) * 1e-12
    rank = int((s_ > tol).sum())
    Nz = Vh.conj().T[:, rank:]                      # whitened null(Div) basis
    Sw = np.diag(sqrtWok) @ Sok @ np.diag(1.0 / sqrtWok)
    Sr = Nz.conj().T @ Sw @ Nz                      # reduced skew operator
    ev = np.linalg.eigvals(Sr)
    return float(np.max(np.abs(ev.imag)))


class TestStaggeredCoriolisCouplingStrength:
    """Localize the band ceiling to the Coriolis COUPLING STRENGTH — a
    measured number, computed without a full eigensolve.

    Unit-strength Coriolis caps the band at ~1.1; the equation's
    coefficient is 2, which reaches the continuum ~2.2. These check the
    coefficient directly and cheaply.
    """

    def test_full_strength_reaches_continuum(self):
        """With the correct coefficient (2), the div-free Coriolis ceiling
        reaches the continuum band ~2.2 (2Ω, with discrete overshoot)."""
        ceil = _divfree_coriolis_band_ceiling(16, 28, m=1, order=2,
                                              coriolis_coeff=2.0)
        assert 2.0 < ceil < 2.4, (
            f"coeff-2 div-free Coriolis band ceiling {ceil:.4f}; expected "
            f"~2.2. If this dropped near 1.1 the coupling strength regressed "
            f"to unit (the original bug); see _extract_coriolis_coeff.")

    def test_ceiling_scales_linearly_with_coefficient(self):
        """The band ceiling is PROPORTIONAL to the
        Coriolis coefficient (unit→~1.1, double→~2.2). This is what
        distinguishes a coupling-strength error from an averaging/order
        effect (which would not scale linearly)."""
        c1 = _divfree_coriolis_band_ceiling(16, 28, m=1, order=2,
                                            coriolis_coeff=1.0)
        c2 = _divfree_coriolis_band_ceiling(16, 28, m=1, order=2,
                                            coriolis_coeff=2.0)
        assert 1.05 < c1 < 1.15, f"unit-coeff ceiling {c1:.4f}, expected ~1.1"
        assert abs(c2 / c1 - 2.0) < 1e-6, (
            f"ceiling should scale linearly with coefficient; "
            f"got c1={c1:.4f}, c2={c2:.4f}, ratio {c2/c1:.4f} (expected 2).")

    def test_ceiling_is_resolution_independent(self):
        """The full-strength ceiling is ~constant across resolution — the
        signature of a structural (coupling) property, not a Nyquist cutoff."""
        ceils = [
            _divfree_coriolis_band_ceiling(Nr, Nt, m=1, order=2,
                                           coriolis_coeff=2.0)
            for (Nr, Nt) in [(14, 24), (20, 32), (24, 44)]
        ]
        spread = max(ceils) - min(ceils)
        assert spread < 0.04, (
            f"ceilings {['%.4f' % c for c in ceils]} vary by {spread:.4f} "
            f"across resolution — should be ~constant.")
        assert all(2.0 < c < 2.4 for c in ceils)
