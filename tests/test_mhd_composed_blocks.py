"""Tests for the aux-free composed-block builders for linearized MHD.

The composed builders in `spdee.backends.fd2.curl_cross_sparse`
implement the two MHD coupling expressions that would otherwise
require state augmentation:

  Cross(Curl(unknown_vec), bg_vec)   — Lorentz force in momentum eq.
  Curl(Cross(unknown_vec, bg_vec))   — induction equation

These are formed as simple matrix products of the existing Curl and
Cross block builders. The tests below verify, for a manufactured
unknown field and a uniform B₀ = B0_amp ẑ background, that the
discrete composed block agrees with the closed-form analytic answer
on interior grid points at the expected 4th-order rate.

Why these tests matter: the matrix-product construction is exact at
the matrix-algebra level, but it would be possible for the elementary
builders to have implicit r-factor conventions that don't compose
cleanly. The convergence-rate test catches any such mismatch — if a
factor of r is misplaced, the leading error is O(1) and these tests
fail loudly.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import Background, theta
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.curl_cross_sparse import (
    build_cross_curl_unk_bg_block,
    build_curl_cross_unk_bg_block,
)
from spdee.backends.fd2.layout import flatten_2d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_zhat_bg(B0_amp: float):
    """Background B₀ = B0_amp · ẑ in spherical components.

    ẑ = cos θ r̂ − sin θ θ̂.
    """
    return Background.Vector(
        "B0",
        v_r=B0_amp * sp.cos(theta),
        v_theta=-B0_amp * sp.sin(theta),
        v_phi=sp.S.Zero,
    )


def _smooth_v_field(R, T):
    """Manufactured v(r, θ) with all three components non-trivial.

    v_r = sin(2r) cos θ
    v_θ = 0
    v_φ = sin(r) sin²θ
    """
    sin_t = np.sin(T); cos_t = np.cos(T)
    v_r = np.sin(2 * R) * cos_t
    v_th = np.zeros_like(R)
    v_phi = np.sin(R) * sin_t**2
    return v_r, v_th, v_phi


def _physical_to_spin_state(v_r, v_th, v_phi):
    """Pack physical (v_r, v_θ, v_φ) into the spdee spin-basis state
    convention: [σ=-1; σ=0; σ=+1], each spin block flattened in
    Fortran order."""
    sqrt2 = np.sqrt(2.0)
    vm = (v_th - 1j * v_phi) / sqrt2
    v0 = v_r
    vp = (v_th + 1j * v_phi) / sqrt2
    return np.concatenate([flatten_2d(vm), flatten_2d(v0), flatten_2d(vp)])


def _spin_state_to_physical(state, N_r, N_theta):
    """Inverse of `_physical_to_spin_state`. Returns (v_r, v_θ, v_φ)
    as (N_r, N_θ) arrays."""
    sqrt2 = np.sqrt(2.0)
    N = N_r * N_theta
    vm = state[:N].reshape(N_r, N_theta, order='F')
    v0 = state[N:2*N].reshape(N_r, N_theta, order='F')
    vp = state[2*N:].reshape(N_r, N_theta, order='F')
    return v0, (vp + vm) / sqrt2, 1j * (vm - vp) / sqrt2


def _analytic_curl(R, T, m_mode):
    """Closed-form Curl of the manufactured v field at every grid point,
    with the azimuthal factor exp(i m φ) (so ∂_φ → i m)."""
    sin_t = np.sin(T); cos_t = np.cos(T)
    v_r = np.sin(2 * R) * cos_t
    v_th = np.zeros_like(R)
    v_phi = np.sin(R) * sin_t**2

    dr_v_phi = np.cos(R) * sin_t**2
    dt_v_r = -np.sin(2 * R) * sin_t
    dphi_v_r = 1j * m_mode * v_r
    dphi_v_th = 1j * m_mode * v_th
    dt_v_phi = np.sin(R) * 2 * sin_t * cos_t

    # (Curl v)_r = (1/(r sin θ))[∂_θ(sin θ v_φ) - ∂_φ v_θ]
    curl_r = (1.0 / (R * sin_t)) * (cos_t * v_phi + sin_t * dt_v_phi - dphi_v_th)
    # (Curl v)_θ = (1/r)[(1/sin θ) ∂_φ v_r - ∂_r(r v_φ)]
    curl_th = (1.0 / R) * (
        (1.0 / sin_t) * dphi_v_r - (v_phi + R * dr_v_phi)
    )
    # (Curl v)_φ = (1/r)[∂_r(r v_θ) - ∂_θ v_r] = -(1/r) ∂_θ v_r since v_θ=0
    curl_phi = (1.0 / R) * (-dt_v_r)
    return curl_r, curl_th, curl_phi


# ---------------------------------------------------------------------------
# Tests for Cross(Curl(v), B₀) — the Lorentz block
# ---------------------------------------------------------------------------

class TestCrossCurlBlock:
    """`build_cross_curl_unk_bg_block` should give r · Cross(Curl(v), B₀)."""

    def _measure_err(self, N: int, m_mode: int, B0_amp: float = 0.3) -> float:
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        K = build_cross_curl_unk_bg_block(
            _uniform_zhat_bg(B0_amp), bundle, grid, m_mode, side="right",
        )

        R, T = grid.mesh()
        v_r, v_th, v_phi = _smooth_v_field(R, T)
        state = _physical_to_spin_state(v_r, v_th, v_phi)

        out = K @ state
        out_r, out_th, out_phi = _spin_state_to_physical(
            out, grid.N_r, grid.N_theta,
        )

        # Analytic answer.
        curl_r, curl_th, curl_phi = _analytic_curl(R, T, m_mode)
        B0_r_arr = B0_amp * np.cos(T)
        B0_th_arr = -B0_amp * np.sin(T)
        # Cross(a, b) = (a_θ b_φ - a_φ b_θ, a_φ b_r - a_r b_φ, a_r b_θ - a_θ b_r)
        cross_r = -curl_phi * B0_th_arr
        cross_th = curl_phi * B0_r_arr
        cross_phi = curl_r * B0_th_arr - curl_th * B0_r_arr
        an_r = R * cross_r
        an_th = R * cross_th
        an_phi = R * cross_phi

        iint = slice(4, grid.N_r - 4)
        jint = slice(4, grid.N_theta - 4)
        errs = [
            np.max(np.abs(out_r[iint, jint] - an_r[iint, jint])),
            np.max(np.abs(out_th[iint, jint] - an_th[iint, jint])),
            np.max(np.abs(out_phi[iint, jint] - an_phi[iint, jint])),
        ]
        return float(max(errs))

    def test_pointwise_accuracy_at_modest_N(self):
        """At N=32 the relative error should be < 1e-4 for our smooth
        test field, since the leading O(Δr^4) error coefficient is
        modest for these analytic forms."""
        err = self._measure_err(N=32, m_mode=2, B0_amp=0.3)
        assert err < 1e-4, f"abs err {err:.3e} at N=32 is too large"

    def test_rate_4_convergence(self):
        """Doubling N should drop the max error by ~16×."""
        Ns = [16, 24, 32, 48, 64]
        errs = [self._measure_err(N=N, m_mode=2) for N in Ns]
        rates = [
            float(np.log(errs[i] / errs[i + 1]) / np.log(Ns[i + 1] / Ns[i]))
            for i in range(len(Ns) - 1)
        ]
        # All rates should exceed 3.6 (allow generous slack for the
        # finite-N regime); the asymptotic rate is 4. Final-decade
        # rate must be ≥ 3.85.
        for r, n0, n1 in zip(rates, Ns[:-1], Ns[1:]):
            assert r >= 3.5, (
                f"rate {r:.2f} between N={n0}, {n1} below 3.5; "
                f"all rates: {rates}, errs: {errs}"
            )
        assert rates[-1] >= 3.85, (
            f"final-decade rate {rates[-1]:.2f} too far from 4; "
            f"all rates: {rates}"
        )

    def test_zero_B0_gives_zero_block(self):
        """Cross with a zero background should produce a zero block."""
        grid = make_shell_grid(
            N_r=12, N_theta=12, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        zero_bg = Background.Vector(
            "B0", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        K = build_cross_curl_unk_bg_block(
            zero_bg, bundle, grid, m=1, side="right",
        )
        assert K.nnz == 0 or float(np.max(np.abs(K.toarray()))) < 1e-15


# ---------------------------------------------------------------------------
# Tests for Curl(Cross(v, B₀)) — the induction block
# ---------------------------------------------------------------------------

class TestCurlCrossBlock:
    """`build_curl_cross_unk_bg_block` should give r · Curl(Cross(v, B₀))."""

    def _measure_err(self, N: int, m_mode: int, B0_amp: float = 0.3) -> float:
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        K = build_curl_cross_unk_bg_block(
            _uniform_zhat_bg(B0_amp), bundle, grid, m_mode, inner_side="right",
        )

        R, T = grid.mesh()
        v_r, v_th, v_phi = _smooth_v_field(R, T)
        state = _physical_to_spin_state(v_r, v_th, v_phi)

        out = K @ state
        out_r, out_th, out_phi = _spin_state_to_physical(
            out, grid.N_r, grid.N_theta,
        )

        # Analytic answer: z = v × B₀, then Curl(z).
        sin_t = np.sin(T); cos_t = np.cos(T)
        # z components for v = (sin 2r cos θ, 0, sin r sin²θ) and
        # B₀ = (B0 cos θ, −B0 sin θ, 0):
        # z_r   = v_θ B0_φ − v_φ B0_θ = 0 - (sin r sin²θ)(-B0 sin θ) = +B0 sin r sin³θ
        # z_θ   = v_φ B0_r − v_r B0_φ = (sin r sin²θ)(B0 cos θ) - 0 = B0 sin r sin²θ cos θ
        # z_φ   = v_r B0_θ − v_θ B0_r = (sin 2r cos θ)(-B0 sin θ) = -B0 sin 2r cos θ sin θ
        z_r = B0_amp * np.sin(R) * sin_t**3
        z_th = B0_amp * np.sin(R) * sin_t**2 * cos_t
        z_phi = -B0_amp * np.sin(2 * R) * cos_t * sin_t

        # Closed-form derivatives.
        dr_z_phi = -B0_amp * 2 * np.cos(2 * R) * cos_t * sin_t
        dr_z_th = B0_amp * np.cos(R) * sin_t**2 * cos_t
        dt_z_r = B0_amp * np.sin(R) * 3 * sin_t**2 * cos_t
        dt_z_phi = -B0_amp * np.sin(2 * R) * np.cos(2 * T)  # cos(2θ) = cos²θ - sin²θ
        dphi_z_r = 1j * m_mode * z_r
        dphi_z_th = 1j * m_mode * z_th

        # Curl(z) in spherical.
        curl_r = (1.0 / (R * sin_t)) * (
            cos_t * z_phi + sin_t * dt_z_phi - dphi_z_th
        )
        curl_th = (1.0 / R) * (
            (1.0 / sin_t) * dphi_z_r - (z_phi + R * dr_z_phi)
        )
        curl_phi = (1.0 / R) * ((z_th + R * dr_z_th) - dt_z_r)

        an_r = R * curl_r
        an_th = R * curl_th
        an_phi = R * curl_phi

        iint = slice(4, grid.N_r - 4)
        jint = slice(4, grid.N_theta - 4)
        errs = [
            np.max(np.abs(out_r[iint, jint] - an_r[iint, jint])),
            np.max(np.abs(out_th[iint, jint] - an_th[iint, jint])),
            np.max(np.abs(out_phi[iint, jint] - an_phi[iint, jint])),
        ]
        return float(max(errs))

    def test_pointwise_accuracy_at_modest_N(self):
        err = self._measure_err(N=32, m_mode=2)
        assert err < 1e-3, f"abs err {err:.3e} at N=32 is too large"

    def test_rate_4_convergence(self):
        Ns = [16, 24, 32, 48, 64]
        errs = [self._measure_err(N=N, m_mode=2) for N in Ns]
        rates = [
            float(np.log(errs[i] / errs[i + 1]) / np.log(Ns[i + 1] / Ns[i]))
            for i in range(len(Ns) - 1)
        ]
        for r, n0, n1 in zip(rates, Ns[:-1], Ns[1:]):
            assert r >= 3.5, (
                f"rate {r:.2f} between N={n0}, {n1} below 3.5; "
                f"all rates: {rates}, errs: {errs}"
            )
        assert rates[-1] >= 3.85, (
            f"final-decade rate {rates[-1]:.2f} too far from 4; "
            f"all rates: {rates}"
        )

    def test_zero_B0_gives_zero_block(self):
        grid = make_shell_grid(
            N_r=12, N_theta=12, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        zero_bg = Background.Vector(
            "B0", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        K = build_curl_cross_unk_bg_block(
            zero_bg, bundle, grid, m=1, inner_side="right",
        )
        assert K.nnz == 0 or float(np.max(np.abs(K.toarray()))) < 1e-15


# ---------------------------------------------------------------------------
# Antisymmetry sanity checks
# ---------------------------------------------------------------------------

class TestSideAntisymmetry:
    """`side='left'` vs `side='right'` should differ by an overall sign,
    consistent with Cross's antisymmetry."""

    def test_cross_curl_side_flip(self):
        grid = make_shell_grid(
            N_r=14, N_theta=14, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        bg = _uniform_zhat_bg(0.5)
        K_left = build_cross_curl_unk_bg_block(
            bg, bundle, grid, m=2, side="left",
        )
        K_right = build_cross_curl_unk_bg_block(
            bg, bundle, grid, m=2, side="right",
        )
        diff = (K_left + K_right).toarray()
        assert float(np.max(np.abs(diff))) < 1e-12

    def test_curl_cross_side_flip(self):
        grid = make_shell_grid(
            N_r=14, N_theta=14, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
        bg = _uniform_zhat_bg(0.5)
        K_l = build_curl_cross_unk_bg_block(
            bg, bundle, grid, m=2, inner_side="left",
        )
        K_r = build_curl_cross_unk_bg_block(
            bg, bundle, grid, m=2, inner_side="right",
        )
        diff = (K_l + K_r).toarray()
        assert float(np.max(np.abs(diff))) < 1e-12
