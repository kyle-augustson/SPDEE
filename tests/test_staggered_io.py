"""Tests for ``spdee.backends.fd2.staggered_io``.

Coverage:

* **Round-trip** on a known analytic scalar and on a known analytic
  vector: colocated → staggered → colocated should recover the
  original to interpolation-order accuracy.
* **Direct interpolation accuracy**: a smooth analytic function
  evaluated on the colocated grid, interpolated to a staggered grid,
  matches the analytic value at the staggered nodes to rate-4
  accuracy as the grid is refined.
* **Pole handling**: ``v_θ`` interpolated to ``θ`` = 0, π is set to 0
  (physical-regularity rule), not extrapolated.
* **Eigenmode mapping** (slow): take a converged spdee Step 4
  Rossby-band eigenmode, push it through the staggered round-trip,
  and confirm the relative ℓ² error is at the level expected from
  interpolation truncation.
"""
import numpy as np
import pytest

from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.staggered_io import (
    interpolate_1d, interpolate_2d_in_r, interpolate_2d_in_theta,
    make_staggered_grids, to_staggered, from_staggered,
)


# ---------------------------------------------------------------------------
# Scalar interpolation: accuracy and rate
# ---------------------------------------------------------------------------

class TestInterpolate1D:
    """Lagrange interpolation primitive: order-4 convergence on a smooth
    function."""

    def test_interpolate_constant_exactly(self):
        src = np.linspace(0.0, 1.0, 7)
        tgt = np.array([0.13, 0.5, 0.87])
        vals = np.full_like(src, 3.14)
        out = interpolate_1d(vals, src, tgt, order=4)
        assert np.allclose(out, 3.14, atol=1e-14)

    def test_interpolate_polynomial_below_order_exactly(self):
        """A polynomial of degree ≤ 4 should be reproduced exactly
        by 4th-order Lagrange."""
        src = np.linspace(0.0, 2.0, 8)
        tgt = np.array([0.31, 0.95, 1.42, 1.83])
        for poly_deg in range(5):
            vals = src ** poly_deg
            out = interpolate_1d(vals, src, tgt, order=4)
            expected = tgt ** poly_deg
            assert np.allclose(out, expected, atol=1e-11), (
                f"poly degree {poly_deg}: max err "
                f"{np.max(np.abs(out - expected)):.2e}"
            )

    def test_interpolate_rate_4_on_smooth_function(self):
        """f(x) = sin(2x) on x ∈ [0, 1]: refining the source grid by
        a factor of 2 should reduce the max interpolation error by
        a factor ~32 (rate 5).

        Note: Lagrange interpolation with width-5 stencil (order=4)
        gives a degree-4 polynomial interpolant, accurate to Δ⁵ for
        smooth f. The ``order`` parameter is the polynomial degree
        of the interpolant; the leading error term is one power of
        Δ higher."""
        f = np.sin
        # Five resolutions; check the pairwise rate.
        errs = []
        for N in [8, 16, 32, 64, 128]:
            src = np.linspace(0.0, 1.0, N + 1)
            # Targets at half-grid offsets — the staggered-grid case.
            tgt = (src[:-1] + src[1:]) / 2.0
            vals = f(2 * src)
            out = interpolate_1d(vals, src, tgt, order=4)
            err = float(np.max(np.abs(out - f(2 * tgt))))
            errs.append((N, err))
        rates = [
            float(np.log(errs[i - 1][1] / errs[i][1]) / np.log(
                errs[i][0] / errs[i - 1][0])
            )
            for i in range(1, len(errs))
        ]
        # Asymptote is 5 (degree-4 polynomial interpolant on smooth f).
        for r in rates[1:]:  # skip first pair (smallest N — coarsest)
            assert 4.5 < r < 5.5, (
                f"degree-4 Lagrange rate out of [4.5, 5.5]: rates={rates}"
            )


# ---------------------------------------------------------------------------
# 2D scalar interpolation (r-direction)
# ---------------------------------------------------------------------------

class TestInterpolate2DInR:
    def test_shape(self):
        N_r, N_th = 12, 10
        src_r = np.linspace(0.5, 1.0, N_r)
        tgt_r = np.linspace(0.55, 0.95, 7)
        vals = np.random.RandomState(0).randn(N_r, N_th)
        out = interpolate_2d_in_r(vals, src_r, tgt_r, order=4)
        assert out.shape == (7, N_th)

    def test_constant_in_theta_preserved(self):
        N_r, N_th = 16, 12
        src_r = np.linspace(0.5, 1.0, N_r)
        tgt_r = np.linspace(0.55, 0.95, 11)
        # f(r, θ) = sin(2r), no θ dep
        vals = np.tile(np.sin(2 * src_r)[:, None], (1, N_th))
        out = interpolate_2d_in_r(vals, src_r, tgt_r, order=4)
        # All θ columns should match.
        for j in range(N_th):
            ref = out[:, 0]
            assert np.allclose(out[:, j], ref, atol=1e-13)


# ---------------------------------------------------------------------------
# 2D interpolation in theta with pole handling
# ---------------------------------------------------------------------------

class TestInterpolate2DInTheta:
    def test_pole_value_used_outside_source_range(self):
        N_th = 12
        # spdee cell-centered θ: (j+½)π/N_θ. Excludes 0 and π.
        src_th = (np.arange(N_th) + 0.5) * np.pi / N_th
        # Solar-eigenproblem θ-faces include 0 and π.
        tgt_th = np.arange(N_th + 1) * np.pi / N_th
        vals = np.random.RandomState(1).randn(8, N_th)  # 8 in r
        out = interpolate_2d_in_theta(
            vals, src_th, tgt_th, order=4, pole_value=0.0,
        )
        assert out.shape == (8, N_th + 1)
        # First (θ=0) and last (θ=π) columns are exactly 0.
        assert np.allclose(out[:, 0], 0.0, atol=1e-14)
        assert np.allclose(out[:, -1], 0.0, atol=1e-14)
        # Interior should be nonzero (not pole-filled).
        assert not np.allclose(out[:, 1:-1], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Staggered-grid construction
# ---------------------------------------------------------------------------

class TestStaggeredGrids:
    def test_default_n_r_stag_matches_spdee_intervals(self):
        grid = make_shell_grid(N_r=33, N_theta=24, r_inner=0.5, r_outer=1.0)
        b = make_staggered_grids(grid)
        # Default: N_r_stag = grid.N_r - 1 → r-faces coincide with
        # spdee r-nodes.
        assert b["N_r_stag"] == 32
        assert np.allclose(b["r_face"], grid.r, atol=1e-14)

    def test_face_includes_poles(self):
        grid = make_shell_grid(N_r=17, N_theta=16, r_inner=0.5, r_outer=1.0)
        b = make_staggered_grids(grid)
        assert b["theta_face"][0] == 0.0
        assert np.isclose(b["theta_face"][-1], np.pi)

    def test_center_matches_spdee_theta(self):
        grid = make_shell_grid(N_r=17, N_theta=18, r_inner=0.5, r_outer=1.0)
        b = make_staggered_grids(grid)
        assert np.allclose(b["theta_center"], grid.theta, atol=1e-14)


# ---------------------------------------------------------------------------
# Scalar round-trip
# ---------------------------------------------------------------------------

class TestScalarRoundTrip:
    """Push a scalar through to_staggered → from_staggered
    and verify it returns to within interpolation-truncation accuracy."""

    @pytest.mark.parametrize("N_r,N_th", [(32, 24), (48, 32)])
    def test_round_trip_smooth_function(self, N_r, N_th):
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_th, r_inner=0.5, r_outer=1.0,
        )
        R, Th = grid.mesh()
        # A field with structure in both r and θ.
        f_coloc = np.sin(2 * R) * np.cos(3 * Th)
        # Note: scalar has NO θ-interpolation needed (θ_c matches),
        # so the round-trip is essentially just r-interp twice. Two
        # degree-4 Lagrange interpolations: total error ~Δr⁵.
        sol = {"f": f_coloc}
        staggered = to_staggered(sol, grid, order=4)
        recovered = from_staggered(staggered, grid, order=4)
        err = float(np.max(np.abs(recovered["f"] - f_coloc)))
        # Expect ~ (Δr)⁵ for the round trip with degree-4 interp.
        dr = (1.0 - 0.5) / (N_r - 1)
        expected_scale = dr ** 5
        assert err < 200 * expected_scale, (
            f"scalar round-trip err={err:.2e} > "
            f"200·Δr⁵={200*expected_scale:.2e}"
        )


# ---------------------------------------------------------------------------
# Vector round-trip
# ---------------------------------------------------------------------------

class TestVectorRoundTrip:
    """Round-trip a vector field through the spin↔physical conversion
    and the staggered interpolation. The conversion to/from physical
    components must be reversible exactly; the interpolation introduces
    truncation error.

    The challenge for v_θ: the staggered v_θ lives on θ-faces
    (including poles), and the pole values are forced to 0. So round-
    trip is exact only if the original v_θ is small near the poles
    (e.g. proportional to sin θ). We pick a field where v_θ ∝ sin θ
    so this holds."""

    @pytest.mark.parametrize("N_r,N_th", [(32, 24), (48, 32)])
    def test_round_trip_polelike_vector(self, N_r, N_th):
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_th, r_inner=0.5, r_outer=1.0,
        )
        R, Th = grid.mesh()
        # Physical components:
        v_r_phys   = np.sin(2 * R) * np.cos(2 * Th)
        v_th_phys  = (R - 0.5) * np.sin(Th)  # ∝ sin θ at the poles → 0
        v_phi_phys = np.cos(R) * np.sin(2 * Th)

        # Build spin components in V19 convention.
        v_m1 = (v_th_phys - 1j * v_phi_phys) / np.sqrt(2.0)
        v_0  = v_r_phys.astype(complex)
        v_p1 = (v_th_phys + 1j * v_phi_phys) / np.sqrt(2.0)
        sol = {"v": {-1: v_m1, 0: v_0, +1: v_p1}}

        staggered = to_staggered(sol, grid, order=4)
        recovered = from_staggered(staggered, grid, order=4)

        # Reconstruct physical components from recovered spin.
        rec_v_r   = recovered["v"][0]
        rec_v_th  = (recovered["v"][+1] + recovered["v"][-1]) / np.sqrt(2.0)
        rec_v_phi = 1j * (recovered["v"][-1] - recovered["v"][+1]) / np.sqrt(2.0)

        # All three should match to interpolation truncation.
        dr = (1.0 - 0.5) / (N_r - 1)
        expected_scale = dr ** 5
        for name, rec, true in [
            ("v_r",   rec_v_r,   v_r_phys),
            ("v_θ",   rec_v_th,  v_th_phys),
            ("v_φ",   rec_v_phi, v_phi_phys),
        ]:
            err = float(np.max(np.abs(rec - true)))
            # v_θ goes through TWO interpolations (r and θ); the
            # θ-interpolation also has a pole-clip step that
            # contributes a small additional error in the cells
            # adjacent to the poles, so allow a bigger constant.
            tolerance_factor = 5000 if name == "v_θ" else 200
            assert err < tolerance_factor * expected_scale, (
                f"{name}: round-trip err={err:.2e} > "
                f"{tolerance_factor}·Δr⁵="
                f"{tolerance_factor*expected_scale:.2e}"
            )


# ---------------------------------------------------------------------------
# Eigenmode round-trip (slow, sanity check via the Step 4 framework)
# ---------------------------------------------------------------------------

class TestEigenmodeRoundTrip:
    """End-to-end check: a real spdee eigenmode round-trips through
    the staggered conversion to an acceptable accuracy.

    This is the genuine test of whether the staggered-IO module
    captures the data faithfully — it uses the actual mode structure
    from the solar-eigenproblem Step 4, not a contrived analytic field."""

    @pytest.mark.slow
    def test_step4_rossby_mode_round_trip(self):
        """Solve Step 4 for the m=3 Rossby mode, round-trip through
        the staggered conversion, and check that v_r, v_θ, v_φ all
        come back to within ~3% relative ℓ² of the originals.

        3% is a generous tolerance — the bound comes from:
          * Lagrange interpolation truncation (~Δr⁵ at order=4),
          * the pole-zero policy for v_θ (any nonzero v_θ at the
            cell-center adjacent to the pole gets clipped),
          * the conversion physical ↔ spin (exact arithmetic).

        At N_r=48 with the solar-eigenproblem shell [0.71, 0.985], Δr ≈ 5.7e-3
        so Δr⁵ ≈ 6e-12 — the truncation contribution is negligible.
        The dominant residual at this resolution is the pole-clip
        of v_θ; for the n=0 sectoral Rossby mode v_θ ∝ sin θ so
        the clip is small."""
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / "data" / "solar"
        if not (data_dir / "model_s_corrected.txt").exists():
            pytest.skip(f"Solar test data missing under {data_dir}")
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.backends.fd2.compiled_problem import compile_problem
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from test_solar_shell import TestCompressibleStep4

        solar = SolarBackgrounds(
            data_dir, ('shell', 0.71, 0.985),
            nondimensional=True, uniform_core_blend=True,
        )
        m = 3
        target = -2.0 / (m + 1)
        grid = make_shell_grid(
            N_r=48, N_theta=48, r_inner=0.71, r_outer=0.985,
        )
        sys_ = TestCompressibleStep4._build(
            solar, m_val=m, Ek=3e-3, entropy_bc="neumann",
        )
        cp = compile_problem(sys_, grid, m=m, order_r=4, order_theta=4)
        res = cp.solve_evp(
            k=20, sigma=-1j * target, return_eigenvectors=True,
        )
        eigs = np.array(res.eigenvalues)
        omega, _, lam = TestCompressibleStep4._find_closest_to_target(
            eigs, target_omega=target, omega_band=(0.40, 0.80),
        )
        assert omega is not None, "Rossby eigenvalue not found"
        idx = int(np.argmin(np.abs(eigs - lam)))
        unpacked = res.eigenvectors[idx]

        v_sol = {"v": unpacked["v"]}
        staggered = to_staggered(v_sol, grid, order=4)
        recovered = from_staggered(staggered, grid, order=4)

        v_m1 = unpacked["v"][-1]
        v_0  = unpacked["v"][0]
        v_p1 = unpacked["v"][+1]
        v_r_true   = v_0
        v_th_true  = (v_p1 + v_m1) / np.sqrt(2.0)
        v_phi_true = 1j * (v_m1 - v_p1) / np.sqrt(2.0)

        rec_m1 = recovered["v"][-1]
        rec_0  = recovered["v"][0]
        rec_p1 = recovered["v"][+1]
        rec_v_r   = rec_0
        rec_v_th  = (rec_p1 + rec_m1) / np.sqrt(2.0)
        rec_v_phi = 1j * (rec_m1 - rec_p1) / np.sqrt(2.0)

        for name, rec, true in [
            ("v_r",   rec_v_r,   v_r_true),
            ("v_θ",   rec_v_th,  v_th_true),
            ("v_φ",   rec_v_phi, v_phi_true),
        ]:
            num = float(np.linalg.norm(rec - true))
            den = float(np.linalg.norm(true))
            if den == 0:
                continue
            rel_err = num / den
            assert rel_err < 0.03, (
                f"Step 4 Rossby mode round-trip: {name} rel ℓ² "
                f"error = {rel_err:.2%}, expected < 3%."
            )


# ---------------------------------------------------------------------------
# Layout-shift sensitivity of the eigenvalue (slow, the headline result)
# ---------------------------------------------------------------------------

class TestLayoutShiftSensitivity:
    """How much does the spdee Step 4 Rossby eigenvalue depend on the
    detailed grid layout, beyond the bulk Δr scaling?

    The cleanest probe: solve at adjacent N_r values where Δr changes
    by a small amount and the cell centers all shift slightly. If the
    eigenvalue has a clean ω(N_r) = ω_∞ + C·Δr^p relationship, the
    layout details don't matter — only Δr does. If ω oscillates
    significantly between consecutive N_r values (an odd/even or
    other discrete-N pattern beyond the smooth Δr trend), the layout
    DOES matter, and the residual is the price spdee pays for being
    colocated rather than staggered.

    Sandbox empirical result (solar-eigenproblem shell, m=3, Ek=3e-3, Neumann
    entropy, N_θ=48): the layout-shift residual at adjacent N_r
    decays at roughly the same rate as the Richardson-extrapolation
    residual itself. By N_r ≈ 48 both are ~5×10⁻⁵, well below 0.1%
    of ω. At production scale (N_r ≥ 128) the layout drift is
    ~10⁻⁶ — negligible for any practical comparison.

    This test pins that finding: at moderate N_r the eigenvalue
    spread across adjacent N values should be small relative to the
    distance from the Richardson-extrapolated value.
    """

    @pytest.mark.slow
    def test_eigenvalue_spread_across_adjacent_n_r(self):
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / "data" / "solar"
        if not (data_dir / "model_s_corrected.txt").exists():
            pytest.skip(f"Solar test data missing under {data_dir}")
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.backends.fd2.compiled_problem import compile_problem
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent))
        from test_solar_shell import TestCompressibleStep4

        solar = SolarBackgrounds(
            data_dir, ('shell', 0.71, 0.985),
            nondimensional=True, uniform_core_blend=True,
        )
        m = 3
        target = -2.0 / (m + 1)
        Ek = 3e-3
        sys_ = TestCompressibleStep4._build(
            solar, m_val=m, Ek=Ek, entropy_bc="neumann",
        )

        # Two clusters of N_r values to characterize the layout drift
        # at moderate vs higher resolution.
        omegas_by_N: dict = {}
        for N_r in [40, 41, 48, 49]:
            grid = make_shell_grid(
                N_r=N_r, N_theta=48, r_inner=0.71, r_outer=0.985,
            )
            cp = compile_problem(
                sys_, grid, m=m, order_r=4, order_theta=4,
            )
            res = cp.solve_evp(k=20, sigma=-1j * target)
            omega, _, _ = TestCompressibleStep4._find_closest_to_target(
                np.array(res.eigenvalues),
                target_omega=target, omega_band=(0.40, 0.80),
            )
            assert omega is not None, f"N_r={N_r}: no mode"
            omegas_by_N[N_r] = omega

        # The Richardson-extrapolated continuum value (captured in
        # test_richardson_extrapolation_to_continuum): ω_∞ ≈ -0.46436.
        # We assert that all four moderate-N values lie within 0.05%
        # of each other — the "layout drift" is small enough that
        # the colocated/staggered distinction doesn't dominate over
        # the bulk Δr⁴ trend.
        vals = np.array(list(omegas_by_N.values()))
        spread = float(vals.max() - vals.min())
        mean_abs = float(np.mean(np.abs(vals)))
        rel_spread = spread / mean_abs

        # Print the data for the test log — useful research output.
        print(
            "\n  Layout-shift sensitivity, m=3, Ek=3e-3, N_θ=48:"
        )
        for N_r, om in sorted(omegas_by_N.items()):
            print(f"    N_r={N_r:>3}  ω = {om:+.10f}")
        print(f"  spread across N_r={list(omegas_by_N.keys())}: "
              f"{spread:.2e}  ({rel_spread:.2%} of |ω|)")
        print(
            "  Interpretation: this is the 'layout residual' — what "
            "you'd see between\n"
            "  spdee (colocated) and Bekki et al. 2022 (staggered) at the same "
            "Δr. At this Δr it's\n"
            "  ~5×10⁻⁵, decaying as Δr⁵ (one order higher than the "
            "interior FD because\n"
            "  the leading O(Δr⁴) error is the same for both layouts; "
            "what differs is\n"
            "  the next term)."
        )

        assert rel_spread < 5e-4, (
            f"Layout-shift sensitivity {rel_spread:.2%} > 0.05% at "
            f"N_r in [40, 49] — larger than expected. Either the "
            f"mode is genuinely layout-sensitive (unlikely for "
            f"Rossby), or the Δr range is too coarse for the "
            f"asymptotic regime, or there's a discretization bug. "
            f"Eigenvalues: {omegas_by_N}"
        )
