"""Tests for the spdee.backgrounds.solar adapter.

Validates that Model S and HMI Ω data load correctly, interpolators
produce sensible values, geometry handling works for both shell and
ball, and the resulting `Background` instances integrate with spdee's
equation compiler.

Test data is under `data/solar/` in the repo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp


# Path to the bundled solar data directory.
DATA_DIR = Path(__file__).parent.parent / "data" / "solar"


@pytest.fixture(scope="module")
def have_solar_data():
    """Skip the module if test data is absent (e.g., not bundled)."""
    if not (DATA_DIR / "model_s_corrected.txt").exists():
        pytest.skip(f"Solar test data missing under {DATA_DIR}")
    if not (DATA_DIR / "omega_fit.npy").exists():
        pytest.skip(f"omega_fit.npy missing under {DATA_DIR}")
    return DATA_DIR


# ---------------------------------------------------------------------------
# Parser / loader tests
# ---------------------------------------------------------------------------

class TestModelSParser:
    def test_basic_load(self, have_solar_data):
        from spdee.backgrounds.solar import _parse_model_s
        ms = _parse_model_s(have_solar_data / "model_s_corrected.txt")
        # Has 2482 raw points; after dedup we keep nearly all (the
        # near-zero innermost point is snapped to 0 but kept).
        assert len(ms["r"]) > 2400
        # r ordered ascending, starting at exactly 0.
        assert ms["r"][0] == 0.0
        # Outer radius should be very close to R_sun.
        assert 1.0 < ms["r"][-1] < 1.01
        # All profiles same length.
        for key in ("density", "pressure", "gamma1", "sound_speed",
                    "gravity", "N2"):
            assert ms[key].shape == ms["r"].shape

    def test_gravity_at_origin_is_zero(self, have_solar_data):
        """The raw Model S table has a numerical singularity at r → 0
        (G·M(r)/r² evaluated at r ≈ 1e-49 produces ~1e64 cm/s²). The
        parser must replace this with the physical limit g(0) = 0.
        """
        from spdee.backgrounds.solar import _parse_model_s
        ms = _parse_model_s(have_solar_data / "model_s_corrected.txt")
        assert ms["gravity"][0] == 0.0
        assert ms["N2"][0] == 0.0
        # The values just outside r=0 should be small but finite.
        assert 0.0 < ms["gravity"][1] < 1e6
        assert ms["gravity"].max() < 3e5  # solar interior max ≈ 2.4e5 cm/s²

    def test_central_density_matches_literature(self, have_solar_data):
        """Solar central density is ~150 g/cm³ (standard Model S value)."""
        from spdee.backgrounds.solar import _parse_model_s
        ms = _parse_model_s(have_solar_data / "model_s_corrected.txt")
        rho_central = ms["density"][0]
        assert 140 < rho_central < 160, (
            f"Central density {rho_central:.2f} not near standard "
            f"~150 g/cm³"
        )

    def test_central_pressure_matches_literature(self, have_solar_data):
        """Solar central pressure ≈ 2.3e17 dyn/cm²."""
        from spdee.backgrounds.solar import _parse_model_s
        ms = _parse_model_s(have_solar_data / "model_s_corrected.txt")
        p_central = ms["pressure"][0]
        assert 2e17 < p_central < 3e17, (
            f"Central pressure {p_central:.2e} not near 2.3e17 dyn/cm²"
        )


class TestOmegaLoader:
    """Validate the raw-HMI Ω loader produces a sane evaluator.

    The loader takes the data directory (not a single file path) so
    it can choose between raw HMI tables and the pre-fit fallback.
    The raw path is preferred when both are present.
    """

    def test_raw_source_preferred(self, have_solar_data):
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data)
        # Raw HMI tables ship with the test data; loader should use them.
        assert od["source"] == "raw"
        # Evaluator is a callable, not an array.
        assert callable(od["evaluator"])

    def test_data_extents(self, have_solar_data):
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data)
        # The raw HMI radial grid covers r ∈ [0, 1].
        assert od["r_min"] == 0.0 or od["r_min"] < 1e-9
        assert 0.999 < od["r_max"] <= 1.001

    def test_surface_equator_rotation_rate(self, have_solar_data):
        """At the surface equator, Ω should be ≈ 452 nHz. The peak of
        Ω anywhere in the convection zone is ≈ 469 nHz (slightly below
        the surface, in the NSSL).
        """
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data)
        ev = od["evaluator"]
        nHz = lambda x: x / (2 * np.pi * 1e-9)
        # Surface equator (r=1, θ=π/2)
        val_eq = ev(np.array(1.0), np.array(np.pi / 2)).item()
        assert 440 < nHz(val_eq) < 460, (
            f"Surface equator Ω = {nHz(val_eq):.1f} nHz, expected ~452"
        )
        # Global max anchor should be near 469 nHz.
        assert 460 < nHz(od["Omega_max"]) < 480, (
            f"Omega_max = {nHz(od['Omega_max']):.1f} nHz, expected ~469"
        )

    def test_north_south_symmetric(self, have_solar_data):
        """Mirroring across the equator: Ω(r, θ) = Ω(r, π−θ)."""
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data)
        ev = od["evaluator"]
        rs = np.array([[0.6, 0.6], [0.8, 0.8], [1.0, 1.0]])
        ts = np.array([[0.3, np.pi - 0.3], [0.8, np.pi - 0.8],
                       [1.2, np.pi - 1.2]])
        vals = ev(rs, ts)
        # North and south columns should be (numerically) equal.
        err = np.max(np.abs(vals[:, 0] - vals[:, 1]))
        assert err < 1e-12, (
            f"N/S symmetry broken; max diff = {err:.3e} rad/s"
        )

    def test_core_uniform_with_blend(self, have_solar_data):
        """With uniform_core_blend=True, Ω deep in the core is
        latitudinally uniform (the tanh asymptote).
        """
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data, uniform_core_blend=True)
        ev = od["evaluator"]
        # Deep core: r=0.2, varied θ
        rs = np.full(8, 0.2)
        ts = np.linspace(0.1, np.pi - 0.1, 8)
        vals = ev(rs, ts)
        spread = vals.max() - vals.min()
        mean = vals.mean()
        assert spread / mean < 1e-6, (
            f"With blend, core Ω should be uniform; spread/mean "
            f"= {spread/mean:.3e}"
        )

    def test_core_nonuniform_without_blend(self, have_solar_data):
        """Without the blend, the raw inversion shows finite (if
        unreliable) latitudinal variation deep in the core. Verify
        the raw spline preserves this rather than flattening it.
        """
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data, uniform_core_blend=False)
        ev = od["evaluator"]
        rs = np.full(8, 0.2)
        ts = np.linspace(0.1, np.pi - 0.1, 8)
        vals = ev(rs, ts)
        spread = vals.max() - vals.min()
        # Spread is small but nonzero — the raw HMI inversion gives
        # ≈ 1-2 nHz of latitudinal variation at r=0.2.
        nHz_to = 2 * np.pi * 1e-9
        assert spread / nHz_to > 1e-3, (
            f"Without blend, expected some latitudinal variation; "
            f"got spread = {spread/nHz_to:.3e} nHz"
        )

    def test_blend_radius_moves_transition(self, have_solar_data):
        """Increasing ``blend_radius`` shifts the transition outward,
        so the uniform-core asymptote extends to larger r.
        """
        from spdee.backgrounds.solar import _load_omega
        od_inner = _load_omega(
            have_solar_data, uniform_core_blend=True,
            blend_radius=0.55, blend_steepness=200.0,
        )
        od_outer = _load_omega(
            have_solar_data, uniform_core_blend=True,
            blend_radius=0.75, blend_steepness=200.0,
        )
        # At r=0.65, equator: with r_c=0.55 we're past the transition
        # (data-dominated); with r_c=0.75 we're still in the core regime.
        r = np.array(0.65)
        t = np.array(np.pi / 2)  # equator (varies most with latitude)
        v_inner = od_inner["evaluator"](r, t).item()
        v_outer = od_outer["evaluator"](r, t).item()
        # The outer-blend evaluator at r=0.65 should sit very close to
        # the core-uniform asymptote.
        nHz = 2 * np.pi * 1e-9
        core_value = od_inner["Omega_core_uniform"]
        assert abs(v_outer - core_value) / nHz < 0.5, (
            f"With blend_radius=0.75, Ω(r=0.65) should be ~ Ω_core "
            f"({core_value/nHz:.1f} nHz); got {v_outer/nHz:.1f} nHz"
        )
        # The inner-blend evaluator at the same point should be
        # noticeably different — it's seeing the data.
        assert abs(v_inner - v_outer) / nHz > 0.5, (
            f"Expected blend_radius to materially change Ω at r=0.65; "
            f"v_inner={v_inner/nHz:.1f}, v_outer={v_outer/nHz:.1f} nHz"
        )

    def test_blend_steepness_controls_width(self, have_solar_data):
        """Sharper steepness ⇒ narrower transition. With s=1000 the
        flip from core-value to data-value happens within ~0.005 R_sun;
        with s=20 it spreads over ~0.05 R_sun.
        """
        from spdee.backgrounds.solar import _load_omega
        od_sharp = _load_omega(
            have_solar_data, uniform_core_blend=True,
            blend_radius=0.55, blend_steepness=1000.0,
        )
        od_wide = _load_omega(
            have_solar_data, uniform_core_blend=True,
            blend_radius=0.55, blend_steepness=20.0,
        )
        ev_sharp = od_sharp["evaluator"]
        ev_wide = od_wide["evaluator"]
        core_value = od_sharp["Omega_core_uniform"]
        # The two evaluators share the same Ω_core asymptote.
        assert abs(od_wide["Omega_core_uniform"] - core_value) < 1e-12

        nHz = 2 * np.pi * 1e-9
        t = np.array(np.pi / 2)

        # At r = blend_radius exactly (tanh(0) = 0), the two evaluators
        # give the same 50/50 mix regardless of steepness.
        r_center = np.array(0.55)
        v_sharp_c = ev_sharp(r_center, t).item()
        v_wide_c = ev_wide(r_center, t).item()
        assert abs(v_sharp_c - v_wide_c) / nHz < 0.5, (
            f"At the blend center, sharp and wide should agree; "
            f"got {v_sharp_c/nHz:.2f} vs {v_wide_c/nHz:.2f} nHz"
        )

        # Below the center (r=0.54): the SHARP transition is essentially
        # complete (tanh(-10) ≈ -1 → pure core), while the WIDE
        # transition is still partly mixed with data.
        r_below = np.array(0.54)
        v_sharp_b = ev_sharp(r_below, t).item()
        v_wide_b = ev_wide(r_below, t).item()
        d_sharp = abs(v_sharp_b - core_value)
        d_wide = abs(v_wide_b - core_value)
        # The wide variant should be further from the core asymptote
        # than the sharp one (sharp has finished transitioning to core,
        # wide is still en route).
        assert d_sharp < d_wide, (
            f"Below blend center, sharp should be at core ({core_value/nHz:.2f} "
            f"nHz) but wide should still carry data influence. "
            f"sharp dist to core = {d_sharp/nHz:.3f}, wide dist = "
            f"{d_wide/nHz:.3f} nHz"
        )

        # Symmetric check ABOVE the center (r=0.56): sharp ≈ pure data,
        # wide partly mixed with core.
        r_above = np.array(0.56)
        v_sharp_a = ev_sharp(r_above, t).item()
        v_wide_a = ev_wide(r_above, t).item()
        # Sharp at r=0.56 should be further from the core (closer to
        # data) than wide.
        assert abs(v_sharp_a - core_value) > abs(v_wide_a - core_value), (
            f"Above blend center, sharp should be further from core "
            f"than wide. sharp = {v_sharp_a/nHz:.3f}, wide = "
            f"{v_wide_a/nHz:.3f}, core = {core_value/nHz:.3f} nHz"
        )

    def test_blend_params_ignored_when_blend_off(self, have_solar_data):
        """blend_radius/blend_steepness have no effect when
        uniform_core_blend=False.
        """
        from spdee.backgrounds.solar import _load_omega
        ev_a = _load_omega(
            have_solar_data, uniform_core_blend=False,
            blend_radius=0.55, blend_steepness=200.0,
        )["evaluator"]
        ev_b = _load_omega(
            have_solar_data, uniform_core_blend=False,
            blend_radius=0.99, blend_steepness=1.0,  # absurd
        )["evaluator"]
        r = np.linspace(0.1, 1.0, 10)
        t = np.full(10, np.pi / 3)
        a = ev_a(r, t)
        b = ev_b(r, t)
        assert np.allclose(a, b, atol=1e-12), (
            "Blend params should not change anything when blend is off"
        )

    def test_blend_radius_auto_detect_lands_at_tachocline(
        self, have_solar_data,
    ):
        """When blend_radius=None, the loader auto-detects the
        transition radius as the largest r where the latitudinally-
        averaged Ω̄(r) crosses Ω_core. For solar inversions this is
        the tachocline, around r ≈ 0.69-0.71 R_sun (0.709 with the
        sin θ-weighted mean).
        """
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(have_solar_data, uniform_core_blend=True)
        r_used = od["blend_radius_used"]
        assert 0.65 < r_used < 0.72, (
            f"Auto-detected blend_radius = {r_used:.4f} not at the "
            f"tachocline (expected ~0.71)"
        )

    def test_blend_radius_explicit_override(self, have_solar_data):
        """An explicit numeric blend_radius is used as-is, ignoring
        auto-detection.
        """
        from spdee.backgrounds.solar import _load_omega
        od = _load_omega(
            have_solar_data, uniform_core_blend=True,
            blend_radius=0.45,
        )
        assert od["blend_radius_used"] == 0.45

    def test_auto_detect_consistent_raw_vs_prefit(
        self, have_solar_data, tmp_path,
    ):
        """Raw and pre-fit data should auto-detect to similar
        tachocline radii (within ~0.01 R_sun). The two paths differ
        only in interpolation method, not in the underlying physics.
        """
        import shutil
        from spdee.backgrounds.solar import _load_omega
        # Set up a temp dir with only the prefit file to force fallback.
        shutil.copy(have_solar_data / "omega_fit.npy",
                    tmp_path / "omega_fit.npy")
        raw = _load_omega(have_solar_data, uniform_core_blend=True)
        pf = _load_omega(tmp_path, uniform_core_blend=True)
        assert raw["source"] == "raw"
        assert pf["source"] == "prefit"
        diff = abs(raw["blend_radius_used"] - pf["blend_radius_used"])
        assert diff < 0.01, (
            f"Auto-detect should give similar r_c for raw and prefit; "
            f"raw={raw['blend_radius_used']:.4f}, "
            f"prefit={pf['blend_radius_used']:.4f}"
        )


# ---------------------------------------------------------------------------
# Geometry parsing
# ---------------------------------------------------------------------------

class TestGeometryParse:
    def test_shell_tuple(self):
        from spdee.backgrounds.solar import _Geometry
        g = _Geometry.parse(('shell', 0.5, 0.9))
        assert g.kind == 'shell'
        assert g.r_min == 0.5 and g.r_max == 0.9

    def test_ball_tuple(self):
        from spdee.backgrounds.solar import _Geometry
        g = _Geometry.parse(('ball', 0.95))
        assert g.kind == 'ball'
        assert g.r_min == 0.0 and g.r_max == 0.95

    def test_dict_form(self):
        from spdee.backgrounds.solar import _Geometry
        assert _Geometry.parse({'shell': (0.4, 0.8)}).kind == 'shell'
        assert _Geometry.parse({'ball': 0.6}).kind == 'ball'

    def test_invalid_raises(self):
        from spdee.backgrounds.solar import _Geometry
        with pytest.raises(ValueError):
            _Geometry.parse(('annulus', 0.4, 0.8))
        with pytest.raises(ValueError):
            _Geometry.parse({'foo': 1.0})


# ---------------------------------------------------------------------------
# SolarBackgrounds end-to-end
# ---------------------------------------------------------------------------

class TestSolarBackgroundsShell:
    """Shell geometry covering the convection zone."""

    @pytest.fixture
    def bg(self, have_solar_data):
        from spdee.backgrounds.solar import SolarBackgrounds
        return SolarBackgrounds(
            have_solar_data, ('shell', 0.715, 0.99),
        )

    def test_scalar_fields_evaluate_on_mesh(self, bg):
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(
            N_r=16, N_theta=12, r_inner=0.715, r_outer=0.99,
        )
        R, T = grid.mesh()
        for name in ('rho', 'pressure', 'sound_speed', 'gravity',
                     'gamma1', 'N2'):
            f = getattr(bg, name)
            vals = f.component.evaluate_on_grid(R, T)
            assert vals.shape == R.shape
            assert np.all(np.isfinite(vals))

    def test_density_decreases_outward_in_cz(self, bg):
        """ρ decreases monotonically through the CZ."""
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(
            N_r=20, N_theta=4, r_inner=0.715, r_outer=0.99,
        )
        R, T = grid.mesh()
        rho_vals = bg.rho.component.evaluate_on_grid(R, T)
        # Take a fixed θ slice.
        rho_radial = rho_vals.real[:, 0]
        assert np.all(np.diff(rho_radial) < 0), (
            "Density not monotone-decreasing through CZ"
        )

    def test_N2_negative_in_cz(self, bg):
        """In the convection zone, N² < 0 (unstable stratification)."""
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(
            N_r=12, N_theta=4, r_inner=0.75, r_outer=0.95,
        )
        R, T = grid.mesh()
        N2_vals = bg.N2.component.evaluate_on_grid(R, T)
        # Interior of CZ should have N² < 0 (top and bottom can
        # transition through 0).
        interior = N2_vals.real[2:-2, :]
        assert np.all(interior < 0), (
            f"N² should be negative in CZ interior; got "
            f"min={interior.min():.3e}, max={interior.max():.3e}"
        )

    def test_g_vec_points_inward(self, bg):
        """g_vec.v_r is the radial component of gravity-as-a-vector,
        which by convention points inward, so values should be negative.
        """
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(
            N_r=10, N_theta=6, r_inner=0.8, r_outer=0.95,
        )
        R, T = grid.mesh()
        gr = bg.g_vec.components['v_r'].evaluate_on_grid(R, T)
        assert np.all(gr.real < 0), "g_vec.v_r should be negative"
        # The other components should be exactly zero.
        for comp in ('v_theta', 'v_phi'):
            other = bg.g_vec.components[comp].evaluate_on_grid(R, T)
            assert np.all(other == 0)

    def test_U0_phi_matches_omega_r_sintheta(self, bg):
        """U₀.v_φ ≡ Ω(r,θ) · r · sin θ."""
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(
            N_r=12, N_theta=10, r_inner=0.8, r_outer=0.99,
        )
        R, T = grid.mesh()
        Om = bg.Omega.component.evaluate_on_grid(R, T)
        U0_phi = bg.U0.components['v_phi'].evaluate_on_grid(R, T)
        expected = Om.real * R * np.sin(T)
        err = np.max(np.abs(U0_phi.real - expected))
        denom = np.max(np.abs(expected))
        assert err / denom < 1e-12

    def test_shell_spanning_blend_transition(self, have_solar_data):
        """A shell from r=0.4 to 0.99 spans the tanh-blend transition
        at r≈0.55. With uniform_core_blend=True, Ω at small r should
        be (nearly) uniform; at large r it varies with latitude.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        bg = SolarBackgrounds(
            have_solar_data, ('shell', 0.4, 0.99),
            uniform_core_blend=True,
        )
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(N_r=20, N_theta=14, r_inner=0.4, r_outer=0.99)
        R, T = grid.mesh()
        Om = bg.Omega.component.evaluate_on_grid(R, T).real
        # Well below the blend center, latitudinal variation should be
        # tiny (the blend dominates).
        deep = R < 0.45
        if np.any(deep):
            vals_deep = Om[deep]
            assert vals_deep.std() / vals_deep.mean() < 1e-3, (
                "Ω well below the blend transition should be nearly "
                "uniform"
            )
        # Near the surface, variation is the full ~200 nHz pole-equator
        # contrast.
        near_surface = R > 0.95
        if np.any(near_surface):
            vals_top = Om[near_surface]
            rel_spread = (vals_top.max() - vals_top.min()) / vals_top.mean()
            assert rel_spread > 0.2, (
                f"Near surface, expected substantial differential "
                f"rotation; got relative spread {rel_spread:.3f}"
            )

    def test_invalid_geometry_outside_modelS_raises(self, have_solar_data):
        """Requesting r_max > Model S data extent raises ValueError."""
        from spdee.backgrounds.solar import SolarBackgrounds
        with pytest.raises(ValueError, match="exceeds"):
            bg = SolarBackgrounds(
                have_solar_data, ('shell', 0.5, 1.5),
            )
            # The error is raised lazily when an interpolator is built,
            # which happens at first property access.
            _ = bg.rho

    def test_grad_rho_is_radial_only(self, have_solar_data):
        """``grad_rho`` returns a Background.Vector with only a radial
        component; v_θ and v_φ are identically zero (ρ_0 is
        spherically symmetric).
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        bg = SolarBackgrounds(
            have_solar_data, ('shell', 0.71, 0.985),
        )
        gr = bg.grad_rho
        assert gr.components["v_theta"].expr == 0
        assert gr.components["v_phi"].expr == 0

    def test_grad_rho_negative_in_cz(self, have_solar_data):
        """Density falls outward in the convection zone, so dρ_0/dr < 0
        everywhere in the shell.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        bg = SolarBackgrounds(
            have_solar_data, ('shell', 0.71, 0.985),
        )
        r = np.linspace(0.72, 0.98, 30).reshape(-1, 1)
        t = np.full_like(r, np.pi / 2)
        vals = bg.grad_rho.components["v_r"].evaluate_on_grid(r, t)
        vals = vals.real.flatten()
        assert np.all(vals < 0), (
            f"dρ_0/dr should be negative in CZ; got "
            f"min={vals.min():.3e}, max={vals.max():.3e}"
        )

    def test_grad_rho_matches_hydrostatic_balance(self, have_solar_data):
        """Numerical dρ_0/dr from ``grad_rho`` agrees with the
        hydrostatic prediction -R_sun · ρ_0 g / c_s² to ~1% in the
        bulk of the CZ. This is a self-consistency check on the Model S
        background — the hydrostatic-balance equation
            dp_0/dr = -ρ_0 g
        combined with the adiabatic relation dp/dρ = c_s² gives
            dρ_0/dr = -ρ_0 g / c_s²    (per unit physical distance)
        Multiplied by R_sun for our R_sun-normalized radial coordinate.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        bg = SolarBackgrounds(
            have_solar_data, ('shell', 0.71, 0.985),
            nondimensional=False,
        )
        # Avoid the edge points where FD endpoint artifacts dominate.
        r = np.linspace(0.74, 0.95, 50).reshape(-1, 1)
        t = np.full_like(r, np.pi / 2)
        rho = bg.rho.component.evaluate_on_grid(r, t).real.flatten()
        cs = bg.sound_speed.component.evaluate_on_grid(r, t).real.flatten()
        # gravity vector v_r is negative (points inward); magnitude |g|.
        g_r = bg.g_vec.components["v_r"].evaluate_on_grid(r, t).real.flatten()
        g_mag = -g_r
        # spdee value
        drho_dr = bg.grad_rho.components["v_r"].evaluate_on_grid(r, t).real.flatten()
        # hydrostatic prediction
        R_sun_cm = bg.normalization["R_sun"]
        drho_dr_hydro = -R_sun_cm * rho * g_mag / cs**2
        # Relative discrepancy
        rel = np.abs(drho_dr - drho_dr_hydro) / np.abs(drho_dr)
        assert np.max(rel) < 0.01, (
            f"Bulk-CZ hydrostatic check failed: max rel err = {np.max(rel):.2%}"
        )


class TestSolarBackgroundsBall:
    """Ball geometry with various outer radii."""

    @pytest.fixture
    def bg_cz_top(self, have_solar_data):
        from spdee.backgrounds.solar import SolarBackgrounds
        return SolarBackgrounds(have_solar_data, ('ball', 0.95))

    @pytest.fixture
    def bg_radiative(self, have_solar_data):
        """Radiative-zone ball using the uniform-core blend, so that
        Ω is identically uniform across the domain.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        return SolarBackgrounds(
            have_solar_data, ('ball', 0.5),
            uniform_core_blend=True,
        )

    @pytest.fixture
    def bg_radiative_raw(self, have_solar_data):
        """Same ball geometry, but using the raw (non-blended) Ω.
        Used to check that the spline retains some latitudinal
        variation deep in the core.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        return SolarBackgrounds(
            have_solar_data, ('ball', 0.5),
            uniform_core_blend=False,
        )

    def test_ball_includes_origin(self, bg_cz_top):
        """Ball mesh includes r=0 in its grid; backgrounds must evaluate
        there without error.
        """
        from spdee.backends.fd2.grid import make_ball_grid
        grid = make_ball_grid(N_r=14, N_theta=8, r_outer=0.95)
        R, T = grid.mesh()
        # Some r should be very close to 0.
        assert R.min() < 0.1
        for name in ('rho', 'gravity', 'N2', 'pressure', 'sound_speed'):
            vals = getattr(bg_cz_top, name).component.evaluate_on_grid(R, T)
            assert np.all(np.isfinite(vals)), f"{name} has non-finite values"

    def test_central_values_match_modelS_table(self, bg_cz_top):
        """At r=0 the evaluated central density should match Model S."""
        from spdee.backgrounds.solar import _parse_model_s
        ms = _parse_model_s(bg_cz_top.data_dir / "model_s_corrected.txt")
        rho_center_table = ms["density"][0]
        # Evaluate at a small mesh containing r=0.
        R = np.array([[0.0, 0.001]])
        T = np.array([[1.0, 1.0]])
        vals = bg_cz_top.rho.component.evaluate_on_grid(R, T)
        # Center value should match the table (this is exact since
        # PCHIP exactly interpolates at data points).
        assert abs(vals[0, 0].real - rho_center_table) < 1e-9

    def test_radiative_ball_has_uniform_rotation_with_blend(
        self, bg_radiative,
    ):
        """In a ball with r1=0.5 AND uniform_core_blend=True, the
        whole domain is below the tanh transition, so Ω is uniform.
        """
        from spdee.backends.fd2.grid import make_ball_grid
        grid = make_ball_grid(N_r=10, N_theta=8, r_outer=0.5)
        R, T = grid.mesh()
        Om = bg_radiative.Omega.component.evaluate_on_grid(R, T)
        spread = Om.real.max() - Om.real.min()
        assert spread / Om.real.mean() < 1e-6

    def test_radiative_ball_has_variation_without_blend(
        self, bg_radiative_raw,
    ):
        """Without the blend, the raw inversion shows finite latitudinal
        variation in the deep interior — small but nonzero. This is
        physically meaningful: even though the inversion is poorly
        constrained there, the underlying data isn't uniform.
        """
        from spdee.backends.fd2.grid import make_ball_grid
        grid = make_ball_grid(N_r=10, N_theta=12, r_outer=0.5)
        R, T = grid.mesh()
        Om = bg_radiative_raw.Omega.component.evaluate_on_grid(R, T)
        spread = Om.real.max() - Om.real.min()
        # Spread is small (~1% relative) but well above machine precision.
        rel_spread = spread / Om.real.mean()
        assert 1e-4 < rel_spread < 0.2, (
            f"Without blend, expected modest core Ω variation; "
            f"got relative spread {rel_spread:.3e}"
        )


# ---------------------------------------------------------------------------
# Nondimensional mode
# ---------------------------------------------------------------------------

class TestNondimensional:
    def test_nondimensional_rho_has_max_one(self, have_solar_data):
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.backends.fd2.grid import make_shell_grid
        bg = SolarBackgrounds(
            have_solar_data, ('shell', 0.715, 0.99), nondimensional=True,
        )
        grid = make_shell_grid(
            N_r=200, N_theta=4, r_inner=0.715, r_outer=0.99,
        )
        R, T = grid.mesh()
        rho = bg.rho.component.evaluate_on_grid(R, T)
        # Sampled max should be very close to 1.0 (the anchor is the
        # max within the data sampled on the Model S grid).
        assert 0.95 < rho.real.max() <= 1.0

    def test_nondimensional_omega_max_below_one(self, have_solar_data):
        """Ω anchor is the max of ALL Ω values (over both r and θ).
        Within any specific geometry the max ≤ 1.
        """
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.backends.fd2.grid import make_ball_grid
        bg = SolarBackgrounds(
            have_solar_data, ('ball', 0.95), nondimensional=True,
        )
        grid = make_ball_grid(N_r=12, N_theta=12, r_outer=0.95)
        R, T = grid.mesh()
        Om = bg.Omega.component.evaluate_on_grid(R, T)
        assert Om.real.max() <= 1.0 + 1e-12
        assert Om.real.min() > 0.5  # roughly Ω_core / Ω_max ≈ 0.6


# ---------------------------------------------------------------------------
# End-to-end: compile and solve a g-mode EVP using solar backgrounds
# ---------------------------------------------------------------------------

class TestSolarGModeEVP:
    """Validates the full compile path: solar backgrounds + spdee
    operators + EVP solver produce sensible solar g-mode frequencies.
    """

    def test_radiative_zone_gmodes_have_correct_max_frequency(
        self, have_solar_data,
    ):
        """In the radiative zone, the fundamental g-mode angular
        frequency is bounded by the local maximum of N. Compile a
        Boussinesq system on a radiative-zone shell with solar N²(r)
        and verify the highest g-mode frequency is at most √N²_max
        and within ~10% of it (matching the asymptotic ω ≤ N bound).
        """
        from spdee import (
            Unknown, Lap, Dt, Grad, Div, Equation,
            EquationSystem, BoundaryCondition, Boundary, Dot as DotOp,
            r_hat, theta_hat, phi_hat,
        )
        from spdee.backends.fd2.grid import make_shell_grid
        from spdee.backends.fd2.compiled_problem import compile_problem
        from spdee.backgrounds.solar import SolarBackgrounds
        solar = SolarBackgrounds(
            have_solar_data, ('shell', 0.2, 0.6), nondimensional=True,
        )

        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        nu = sp.S(1e-4)
        eq_mom = Equation(
            Dt(v) + Grad(p) - nu * Lap(v) + s * solar.g_vec,
            name="momentum",
        )
        eq_cont = Equation(Div(v), name="continuity")
        eq_energy = Equation(
            Dt(s) + DotOp(solar.grad_s0, v) - nu * Lap(s),
            name="energy",
        )

        bcs = [
            BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
            BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
        ]
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(
                    loc, DotOp(unit, v), name=f"v{c}{lbl}",
                ))

        sys_ = EquationSystem(
            [eq_mom, eq_cont, eq_energy], bcs, gauge_unknowns=["p"],
        )
        grid = make_shell_grid(
            N_r=20, N_theta=20, r_inner=0.2, r_outer=0.6,
        )
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        res = cp.solve_evp(k=20, sigma=0.01 + 1j)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        # Filter to oscillatory positive-frequency modes.
        osc = [x for x in finite if 0.3 < x.imag < 1.5]
        assert osc, "No g-modes found in expected band"
        # Highest-frequency mode should be ≤ 1.0 in nondim units
        # (since N²_max anchor was used → max N = 1 in nondim).
        top_omega = max(x.imag for x in osc)
        assert top_omega <= 1.0 + 1e-3, (
            f"Highest g-mode ω = {top_omega:.4f} exceeds N₀ = 1"
        )
        # Within 10% of 1.0 (the asymptotic limit for the fundamental).
        assert top_omega > 0.9, (
            f"Highest g-mode ω = {top_omega:.4f} too far below the "
            f"Brunt limit; expected fundamental near 1"
        )


class TestSolarBallStaggered:
    """Model S as an origin-regular ball background driving the staggered
    compressible assembler (mimetic viscous/thermal + r^n origin regularity).

    This is the infrastructure test: a REAL solar background on the full ball
    r∈[0, r_outer], reusing the verified Bekki non-dimensionalization, with the
    physical N²(r) (convective CZ + stable radiative core) replacing Bekki's
    constant-δ source, and uniform Ω₀. Low resolution here; production runs
    (high N_r, shift-invert) are for a production run.
    """

    def test_solar_ball_compiles_and_spectrum_is_sane(self, have_solar_data):
        import numpy as np
        import scipy.linalg as sla
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.applications.solar_eigenmodes import make_solar_ball_system
        from spdee.backends.fd2.grid import make_ball_grid
        from spdee.backends.fd2.compiled_problem import compile_problem, EvpResult
        solar = SolarBackgrounds(have_solar_data, ('ball', 0.985),
                                 nondimensional=True)
        for Ek in (0.0, 7e-6):
            sys_ = make_solar_ball_system(solar, m_val=8, Ek=Ek)
            grid = make_ball_grid(N_r=20, N_theta=20, r_outer=0.985)
            cp = compile_problem(sys_, grid, m=8, order_r=2, order_theta=2,
                                 discretization="staggered")
            A, B = cp.A_total.toarray(), cp.B_total.toarray()
            assert np.all(np.isfinite(A)) and np.all(np.isfinite(B))
            # r^n origin regularity keeps entries bounded: the largest entry
            # is the ρ₁ pressure coefficient cs²/(ρ₀/ρ₀(0)) ≈ 3.4e9 at the
            # outer wall, where ρ₀/ρ₀(0) ≈ 1e-6 (the m²/r² barrier is ~1e7);
            # NOT the ~1e21 W_v⁻¹ blowup.
            assert np.abs(A).max() < 1e10, np.abs(A).max()
            lam = sla.eig(A, B, right=False)
            lam = lam[np.isfinite(lam)]
            om = EvpResult.eigenvalues_to_omega(lam)
            band = om[np.abs(om.real) < 3]
            assert len(band) > 50
            # inviscid: bounded near-neutral band; viscous: physical damping
            assert band.imag.max() < 10.0
            if Ek > 0:
                assert band.imag.min() > -1e3   # no 1e15 garbage modes

    def test_solar_ball_loop_recovers_physical_N2(self, have_solar_data):
        """The buoyancy×entropy-source product reproduces the PHYSICAL Model S
        N²(r) (c_p kept in the source, per Bekki eq. 10) — positive (stable) in
        the radiative interior, negative (convective) in the CZ."""
        import numpy as np
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.applications.solar_eigenmodes import make_solar_ball_system
        solar = SolarBackgrounds(have_solar_data, ('ball', 0.985),
                                 nondimensional=True)
        solarD = SolarBackgrounds(have_solar_data, ('ball', 0.985),
                                  nondimensional=False)
        sys_ = make_solar_ball_system(solar, m_val=8, Ek=7e-6)
        OM = 2 * np.pi * 456e-9

        def ev(bg, c=None):
            comp = bg.components[c] if c else bg.component
            return lambda R, T: np.real(np.asarray(
                comp.evaluate_on_grid(R, T)).ravel()[0])
        gcp = ev(sys_.backgrounds["g_over_cp_vec"], "v_r")
        ds0 = ev(sys_.backgrounds["grad_s0_optA"], "v_r")
        n2 = ev(solarD.N2)
        sign_seen = set()
        for r in (0.1, 0.5, 0.85, 0.985):
            R = np.array([[r]]); T = np.array([[np.pi / 2]])
            loop = gcp(R, T) * ds0(R, T)
            n2_nd = n2(R, T) / OM ** 2
            assert np.isclose(loop, n2_nd, rtol=1e-5), (r, loop, n2_nd)
            sign_seen.add(np.sign(n2_nd) > 0)
        # both stable (radiative) and convective (CZ) regions present
        assert sign_seen == {True, False}
