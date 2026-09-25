"""Acceptance tests for the Bekki + background magnetic field system.

These tests use a small grid (N=24) to keep CI cost down. They check:

1. **Field profile**: `make_dipolar_B0` produces the requested
   amplitude ratio on the equator.

2. **Le=0 reduces to hydro**: the MHD system with Le=0 has a
   spectrum that contains the pure Bekki Step 4 spectrum as a
   subset (the additional modes are pure-magnetic decay).

3. **Le scaling is quadratic**: doubling Le quadruples the shift
   Δω, Δγ of the dominant Rossby mode (since the Lorentz force is
   Le²-proportional in our normalization).

4. **Aux-free**: the layout has exactly the four primary unknowns
   (v, b, rho1, s1), no auxiliaries.

5. **Nondimensionalisation**: the polytrope builder, the polytrope object and
   the certified string system agree on N²/Ω₀², Ω₀ is one parameter, the
   ball builder's pressure coefficient carries the density unit of its
   continuity equation, and a dipole on a ball is regular at the origin.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

# Add tests dir for test_solar_shell access.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from spdee.backgrounds.solar import SolarBackgrounds
from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.applications.solar_eigenmodes import (
    OMEGA0_NHZ, make_solar_shell_system, make_polytrope_system,
    make_solar_ball_system, make_dipolar_B0, make_dipolar_B0_curl,
)
from spdee.applications import solar_strings
from spdee.ir.fields import r as r_sym, theta as t_sym


# The Model S tests need the solar data file; the polytrope tests do not.
_DATA_DIR = _TESTS_DIR.parent / "data" / "solar"
_HAS_SOLAR = (_DATA_DIR / "model_s_corrected.txt").exists()

needs_solar = pytest.mark.skipif(
    not _HAS_SOLAR,
    reason="Solar data file not available in this environment.",
)


@pytest.fixture(scope="module")
def solar():
    """Solar backgrounds in the Bekki shell domain."""
    return SolarBackgrounds(
        _DATA_DIR, ('shell', 0.71, 0.985),
        nondimensional=True, uniform_core_blend=True,
    )


@pytest.fixture(scope="module")
def grid():
    """Small shell grid for cheap tests."""
    return make_shell_grid(N_r=24, N_theta=24, r_inner=0.71, r_outer=0.985)


# ---------------------------------------------------------------------------
# Field profile sanity
# ---------------------------------------------------------------------------

class TestDipolarBackground:
    """`make_dipolar_B0` should produce the specified amplitude ratio."""

    def test_equatorial_amplitude_ratio(self):
        """|B_θ(r_in)/B_θ(r_out)| should equal amplitude_ratio."""
        r_in, r_out = 0.71, 0.985
        for ratio in (10.0, 1000.0, 30000.0):
            B0 = make_dipolar_B0(
                "B0", r_in, r_out,
                amplitude=1.0, amplitude_ratio=ratio,
            )
            bt_in = float(B0.v_theta.subs({r_sym: r_in, t_sym: sp.pi/2}))
            bt_out = float(B0.v_theta.subs({r_sym: r_out, t_sym: sp.pi/2}))
            obs = abs(bt_in / bt_out)
            assert abs(obs - ratio) / ratio < 1e-9, (
                f"ratio {obs} differs from target {ratio}"
            )

    def test_phi_component_zero(self):
        """B_φ should be identically zero (purely poloidal)."""
        B0 = make_dipolar_B0("B0", 0.71, 0.985)
        assert B0.v_phi == sp.S.Zero

    def test_curl_is_purely_toroidal(self):
        """`make_dipolar_B0_curl` should have only v_phi nonzero."""
        J0 = make_dipolar_B0_curl("J0", 0.71, 0.985)
        assert J0.v_r == sp.S.Zero
        assert J0.v_theta == sp.S.Zero
        # v_phi should be a non-trivial expression of r and θ.
        assert J0.v_phi != sp.S.Zero

    def test_field_is_divergence_free_numerically(self):
        """Pointwise: ∇·B at a few sample points should be ~ 0.

        Uses the analytic stream-function expressions plus SymPy
        differentiation, so this is a check of the construction, not
        a discretization test.
        """
        B0 = make_dipolar_B0("B0", 0.71, 0.985, amplitude_ratio=100.0)
        # ∇·B for an axisymmetric field B = (B_r, B_θ, 0) is:
        #   (1/r²) ∂_r(r² B_r) + (1/(r sin θ)) ∂_θ(sin θ B_θ)
        Br = B0.v_r
        Bt = B0.v_theta
        div = (
            sp.diff(r_sym**2 * Br, r_sym) / r_sym**2
            + sp.diff(sp.sin(t_sym) * Bt, t_sym) / (r_sym * sp.sin(t_sym))
        )
        for r_v in (0.75, 0.85, 0.95):
            for t_v in (np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3):
                val = float(div.subs({r_sym: r_v, t_sym: t_v}))
                assert abs(val) < 1e-10, (
                    f"∇·B = {val:.2e} at (r={r_v}, θ={t_v}) — not zero"
                )


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@needs_solar
class TestCompile:

    def test_compiles_at_Le_zero(self, solar, grid):
        sys_ = make_solar_shell_system(
            solar, m_val=3, Ek=3e-3, Le=0.0, geometry="shell",
        )
        cp = compile_problem(
            sys_, grid, m=3, order_r=4, order_theta=4,
        )
        # 4 unknowns: v, b, rho1, s1. State size = N²·(3 + 3 + 1 + 1) = 4608.
        assert cp.matrices.A_total.shape[0] == 24*24 * 8
        # Layout should contain exactly the four primaries (aux-free).
        assert set(cp.matrices.layout.all_names) == {"v", "b", "rho1", "s1"}

    def test_compiles_at_nonzero_Le(self, solar, grid):
        sys_ = make_solar_shell_system(
            solar, m_val=3, Ek=3e-3, Le=0.05, geometry="shell",
        )
        cp = compile_problem(
            sys_, grid, m=3, order_r=4, order_theta=4,
        )
        assert cp.matrices.A_total.shape[0] == 24*24 * 8
        assert set(cp.matrices.layout.all_names) == {"v", "b", "rho1", "s1"}


# ---------------------------------------------------------------------------
# Le=0 recovers hydro spectrum
# ---------------------------------------------------------------------------

@needs_solar
class TestLeZeroMatchesHydro:
    """At Le=0, all hydro Bekki Step 4 eigenvalues should appear in
    the MHD spectrum (plus extra purely-magnetic decay modes)."""

    def test_hydro_spectrum_is_subset(self, solar, grid):
        from test_solar_shell import TestCompressibleStep4

        m, Ek = 3, 3e-3
        # Pure hydro reference.
        sys_h = TestCompressibleStep4._build(
            solar, m_val=m, Ek=Ek, entropy_bc="neumann", geometry="shell",
        )
        cp_h = compile_problem(sys_h, grid, m=m, order_r=4, order_theta=4)
        res_h = cp_h.solve_evp(k=10, sigma=-1j*-0.5)
        eigs_h = np.array(res_h.eigenvalues)

        # MHD at Le=0.
        sys_m = make_solar_shell_system(
            solar, m_val=m, Ek=Ek, Le=0.0, geometry="shell",
        )
        cp_m = compile_problem(sys_m, grid, m=m, order_r=4, order_theta=4)
        # Need more modes since spectrum is denser.
        res_m = cp_m.solve_evp(k=40, sigma=-1j*-0.5)
        eigs_m = np.array(res_m.eigenvalues)

        # Every hydro eigenvalue should appear in eigs_m to roundoff.
        for ev_h in eigs_h:
            dists = np.abs(eigs_m - ev_h)
            best = float(np.min(dists))
            assert best < 1e-10, (
                f"hydro eigenvalue {ev_h} has no MHD match within 1e-10; "
                f"closest distance = {best:.2e}"
            )


# ---------------------------------------------------------------------------
# Le scaling
# ---------------------------------------------------------------------------

@needs_solar
class TestLeScaling:
    """The Rossby-mode shift should be quadratic in Le for small Le."""

    def _dominant_rossby(self, solar, grid, m, Ek, Le, target=-0.466):
        sys_ = make_solar_shell_system(
            solar, m_val=m, Ek=Ek, Le=Le, geometry="shell",
        )
        cp = compile_problem(sys_, grid, m=m, order_r=4, order_theta=4)
        res = cp.solve_evp(k=20, sigma=-1j*target)
        eigs = np.array(res.eigenvalues)
        omega = -eigs.imag
        gamma = -eigs.real
        # Pick the n=0 Rossby: high γ damping (~0.3), ω near target.
        keep = (gamma < -0.05) & (omega < -0.1) & (omega > -0.8)
        if not np.any(keep):
            raise AssertionError("No Rossby mode found")
        idx = np.where(keep)[0]
        sub_idx = idx[np.argsort(np.abs(omega[idx] - target))]
        return float(omega[sub_idx[0]]), float(gamma[sub_idx[0]])

    def test_quadratic_scaling_small_Le(self, solar, grid):
        """Doubling Le should quadruple Δω for small Le."""
        m, Ek = 3, 3e-3
        om0, ga0 = self._dominant_rossby(solar, grid, m, Ek, Le=0.0)
        om_a, ga_a = self._dominant_rossby(solar, grid, m, Ek, Le=0.05)
        om_b, ga_b = self._dominant_rossby(solar, grid, m, Ek, Le=0.10)
        d_om_a = abs(om_a - om0)
        d_om_b = abs(om_b - om0)
        # Le doubled from 0.05 → 0.10, so Le² goes 0.0025 → 0.01, a
        # factor of 4. Δω should scale similarly.
        ratio = d_om_b / max(d_om_a, 1e-20)
        assert 3.0 < ratio < 5.5, (
            f"Δω ratio = {ratio:.2f} not in expected ~4× window. "
            f"d_om_a={d_om_a:.2e}, d_om_b={d_om_b:.2e}"
        )

    def test_zero_Le_zero_shift(self, solar, grid):
        """At Le=0, the Rossby eigenvalue must equal the hydro value."""
        from test_solar_shell import TestCompressibleStep4
        m, Ek = 3, 3e-3
        # Hydro reference.
        sys_h = TestCompressibleStep4._build(
            solar, m_val=m, Ek=Ek, entropy_bc="neumann", geometry="shell",
        )
        cp_h = compile_problem(sys_h, grid, m=m, order_r=4, order_theta=4)
        res_h = cp_h.solve_evp(k=10, sigma=-1j*-0.5)
        eigs_h = np.array(res_h.eigenvalues)
        omega_h = -eigs_h.imag
        idx_h = np.argmin(np.abs(omega_h - (-0.466)))
        om_h_target = omega_h[idx_h]

        # MHD Le=0.
        om_m, ga_m = self._dominant_rossby(solar, grid, m, Ek, Le=0.0)
        assert abs(om_m - om_h_target) < 1e-12


# ---------------------------------------------------------------------------
# Nondimensionalisation: N²/Ω₀², Ω₀, density units, the ball dipole
# ---------------------------------------------------------------------------

def _bg_r(system, name, R, T, comp="v_r"):
    """Real values of a background scalar or of one vector component."""
    bg = system.backgrounds[name]
    c = bg.component if bg.kind == "scalar" else bg.components[comp]
    return np.real(np.asarray(c.evaluate_on_grid(R, T)))


class TestPolytropeNondimensionalisation:
    """Entropy in units of c_p, ds₀/dr = -δ/H_p: the builder's buoyancy ×
    source loop, the polytrope's N² and the string system's gacc δ/H_p are
    one N²/Ω₀² profile."""

    def test_n2_profile_agrees_across_builder_polytrope_and_strings(self):
        delta = 1e-6
        poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=delta)
        sys_ = make_polytrope_system(poly, m_val=8, Ek=0.0)
        R = np.array([[0.72], [0.8], [0.9], [0.98]])
        T = np.full_like(R, np.pi / 2)
        loop = _bg_r(sys_, "g_over_cp_vec", R, T) * _bg_r(sys_, "grad_s0_optA", R, T)
        omega0 = 2 * np.pi * OMEGA0_NHZ * 1e-9
        g_dyn = poly.g_bc / (poly.R_sun * omega0 ** 2)
        n2_poly = np.real(poly.N2.component.evaluate_on_grid(R, T)) * g_dyn
        ex = solar_strings.polytrope_exprs(r_sym)
        n2_str = sp.lambdify(r_sym, -delta * ex["gacc"] * (-ex["dlnp"]), "numpy")(R)
        assert np.allclose(loop, n2_poly, rtol=1e-12), (loop, n2_poly)
        assert np.allclose(loop, n2_str, rtol=1e-12), (loop, n2_str)
        # δ > 0 is convectively unstable: N² < 0 throughout the shell.
        assert np.all(loop < 0)

    def test_grad_s0_matches_builder_source(self):
        poly = PolytropeBackground(delta=1e-6)
        sys_ = make_polytrope_system(poly, m_val=8, Ek=0.0)
        R = np.linspace(0.71, 0.985, 7).reshape(-1, 1)
        T = np.full_like(R, 1.0)
        assert np.allclose(_bg_r(sys_, "grad_s0_optA", R, T),
                           np.real(poly.grad_s0.components["v_r"].evaluate_on_grid(R, T)),
                           rtol=1e-14)

    def test_omega0_is_one_parameter_with_the_carrington_default(self):
        """The pressure and buoyancy coefficients scale as Ω₀⁻²; the default is
        the Carrington rate of the string system and of the ball builder."""
        assert OMEGA0_NHZ == 456.0
        assert np.isclose(2 * np.pi * OMEGA0_NHZ * 1e-9, solar_strings.OMEGA_CARR)
        poly = PolytropeBackground(delta=1e-6)
        ref = make_polytrope_system(poly, m_val=8, Ek=0.0)
        fast = make_polytrope_system(poly, m_val=8, Ek=0.0, omega0_nHz=2 * OMEGA0_NHZ)
        R = np.array([[0.75], [0.95]])
        T = np.full_like(R, 0.7)
        for name in ("cs2_over_rho0", "cs2_over_cp", "g_over_cp_vec"):
            assert np.allclose(_bg_r(fast, name, R, T), _bg_r(ref, name, R, T) / 4.0)


class TestBallDipole:
    """A dipolar B₀ on a ball starts at the origin and is regular there."""

    def test_ball_dipole_is_uniform_z_at_the_origin(self):
        B0 = make_dipolar_B0("B0", 0.0, 0.985)
        J0 = make_dipolar_B0_curl("J0", 0.0, 0.985)
        R = np.array([[0.0], [1e-4], [0.5]])
        T = np.full_like(R, 0.4)
        b_r = np.real(B0.components["v_r"].evaluate_on_grid(R, T)).ravel()
        b_t = np.real(B0.components["v_theta"].evaluate_on_grid(R, T)).ravel()
        j_p = np.real(J0.components["v_phi"].evaluate_on_grid(R, T)).ravel()
        assert np.all(np.isfinite(b_r)) and np.all(np.isfinite(b_t)) and np.all(np.isfinite(j_p))
        assert np.isclose(b_r[0], np.cos(0.4)) and np.isclose(b_t[0], -np.sin(0.4))
        # the closed form at r = 1e-4 continues the origin limit
        assert np.isclose(b_r[1], b_r[0], rtol=2e-3)
        assert np.isclose(j_p[1], j_p[0], rtol=2e-3)

    def test_figure2_ball_with_field_needs_origin_inner_radius(self):
        poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=1e-6)
        with pytest.raises(ValueError, match="r_inner = 0"):
            make_polytrope_system(poly, m_val=8, Ek=7e-6, Le=0.05, geometry="ball")

    @needs_solar
    def test_model_s_ball_with_field_needs_origin_inner_radius(self, solar):
        with pytest.raises(ValueError, match="r_inner = 0"):
            make_solar_shell_system(solar, m_val=3, Ek=3e-3, Le=0.05, geometry="ball")


@needs_solar
class TestSolarBallUnits:
    """ρ₁ is in units of ρ₀(0) in the ball builder's continuity equation, so
    the ρ₁ pressure coefficient is cs²/(ρ₀/ρ₀(0)) and cs2_over_rho0 × poly_rho
    is the dynamical cs² everywhere (the polytrope builder's arrangement)."""

    def test_pressure_coefficient_carries_the_continuity_density_unit(self):
        solar = SolarBackgrounds(_DATA_DIR, ('ball', 0.985), nondimensional=True)
        solarD = SolarBackgrounds(_DATA_DIR, ('ball', 0.985), nondimensional=False)
        sys_ = make_solar_ball_system(solar, m_val=2, Ek=0.0)
        R = np.array([[0.0], [0.3], [0.7], [0.9]])
        T = np.full_like(R, np.pi / 2)
        omega0 = 2 * np.pi * OMEGA0_NHZ * 1e-9
        cs_dyn = np.real(solarD.sound_speed.component.evaluate_on_grid(R, T)) / (
            solarD.normalization["R_sun"] * omega0)
        prod = _bg_r(sys_, "cs2_over_rho0", R, T) * _bg_r(sys_, "poly_rho", R, T)
        assert np.allclose(prod, cs_dyn ** 2, rtol=1e-12)
        # at the centre both pressure coefficients are the dynamical cs²
        assert np.isclose(_bg_r(sys_, "cs2_over_rho0", R, T)[0, 0],
                          _bg_r(sys_, "cs2_over_cp", R, T)[0, 0], rtol=1e-12)
        assert np.isclose(_bg_r(sys_, "poly_rho", R, T)[0, 0], 1.0)
