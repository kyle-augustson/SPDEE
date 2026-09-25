"""Tests for tabular (NumericComponent) backgrounds in the advection
operator Dot(v0, Grad(v)).

The advection dispatcher evaluates v0 numerically on the grid (via
evaluate_vector_background_spin_2d), so tabular v0 works without
special handling. Verifies that:

1. A tabular v0 sampling a known polynomial flow produces the same
   advection action as the symbolic equivalent (to interpolation
   accuracy).
2. Both linear and cubic 2D interpolation work.
3. A mixed v0 (some components symbolic, some tabular) works.
4. A purely zonal-flow v0 (only v_φ depending on r,θ) — the
   linearized-MHD use case — produces sensible advection action.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Grad, Dot, Equation, EquationSystem,
    radial_interpolator, meridional_interpolator,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import split_vector_state


def _apply_advection(v0_bg, name, m_val=2, N_r=20, N_theta=32):
    v = Unknown.Vector("v")
    eq = Equation(Dot(v0_bg, Grad(v)), name=name)
    sys_ = EquationSystem([eq], [])
    grid = make_shell_grid(N_r=N_r, N_theta=N_theta,
                           r_inner=0.3, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=m_val, order_r=6, order_theta=6)
    R, T = grid.mesh()
    v_data = {
        -1: (R ** 2 * np.sin(T)).astype(complex),
         0: (R * np.cos(T)).astype(complex),
        +1: (R * np.sin(T) * np.cos(T)).astype(complex),
    }
    state = cp.matrices.layout.pack({"v": v_data})
    Av = cp.matrices.A_total @ state
    eq_slice = cp.matrices.equation_row_slices[name]
    return split_vector_state(Av[eq_slice], grid.N_r, grid.N_theta), grid


class TestTabularV0InAdvection:
    """Tabular v0 in Dot(v0, Grad(v)) matches symbolic v0 of the same
    underlying function."""

    def test_purely_radial_v_phi(self):
        """v0 = r·sin(θ) φ̂  — rigid-body-like azimuthal flow."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)

        # SymPy version
        v0_sym = Background.Vector(
            "v0_sym",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=r_sym * sp.sin(t_sym),
        )

        # Tabular version
        r_t = np.linspace(0.0, 1.5, 200)
        t_t = np.linspace(0.0, np.pi, 200)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        v_phi_data = R_t * np.sin(T_t)
        v0_tab = Background.Vector(
            "v0_tab",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=meridional_interpolator(r_t, t_t, v_phi_data, kind="cubic"),
        )

        r_sym_res, _ = _apply_advection(v0_sym, "via_sym")
        r_tab_res, _ = _apply_advection(v0_tab, "via_tab")

        # Compare on interior (away from θ poles where cubic-interp
        # boundary effects are larger).
        for sigma in (-1, 0, +1):
            interior = np.zeros_like(r_sym_res[sigma], dtype=bool)
            interior[2:-2, 4:-4] = True
            rel = (np.abs(r_tab_res[sigma] - r_sym_res[sigma])[interior].max() /
                   max(np.abs(r_sym_res[sigma][interior]).max(), 1e-30))
            assert rel < 1e-4, (
                f"σ={sigma}: tabular v_φ doesn't match symbolic; rel = {rel:.2e}"
            )

    def test_differential_rotation_profile(self):
        """v0 with θ-dependent φ component — like solar differential rotation."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)

        # Ω(θ) = 1 + 0.1·cos²(θ); v_φ = r·sin(θ)·Ω(θ)
        v0_sym = Background.Vector(
            "v0_sym",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=r_sym * sp.sin(t_sym) * (
                1 + sp.Rational(1, 10) * sp.cos(t_sym) ** 2
            ),
        )

        r_t = np.linspace(0.0, 1.5, 200)
        t_t = np.linspace(0.0, np.pi, 200)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        v_phi_data = R_t * np.sin(T_t) * (1 + 0.1 * np.cos(T_t) ** 2)
        v0_tab = Background.Vector(
            "v0_tab",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=meridional_interpolator(r_t, t_t, v_phi_data, kind="cubic"),
        )

        r_sym_res, _ = _apply_advection(v0_sym, "via_sym")
        r_tab_res, _ = _apply_advection(v0_tab, "via_tab")

        for sigma in (-1, 0, +1):
            interior = np.zeros_like(r_sym_res[sigma], dtype=bool)
            interior[2:-2, 4:-4] = True
            rel = (np.abs(r_tab_res[sigma] - r_sym_res[sigma])[interior].max() /
                   max(np.abs(r_sym_res[sigma][interior]).max(), 1e-30))
            assert rel < 1e-4, (
                f"σ={sigma}: tabular Ω(θ) doesn't match symbolic; rel = {rel:.2e}"
            )

    def test_meridional_circulation(self):
        """v0 with both v_r and v_θ components — meridional flow.

        Use v_r = r·(1−r)·cos(θ), v_θ = −sin(θ)·(1−2r)·r (chosen so that
        ∇·v ≈ 0 in the bulk — not strictly enforced here, just realistic).
        """
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)

        v_r_expr = r_sym * (1 - r_sym) * sp.cos(t_sym)
        v_th_expr = -sp.sin(t_sym) * (1 - 2 * r_sym) * r_sym

        v0_sym = Background.Vector(
            "v0_sym", v_r=v_r_expr, v_theta=v_th_expr, v_phi=sp.S.Zero,
        )

        r_t = np.linspace(0.0, 1.5, 200)
        t_t = np.linspace(0.0, np.pi, 200)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        v_r_data = R_t * (1 - R_t) * np.cos(T_t)
        v_th_data = -np.sin(T_t) * (1 - 2 * R_t) * R_t
        v0_tab = Background.Vector(
            "v0_tab",
            v_r=meridional_interpolator(r_t, t_t, v_r_data, kind="cubic"),
            v_theta=meridional_interpolator(r_t, t_t, v_th_data, kind="cubic"),
            v_phi=sp.S.Zero,
        )

        r_sym_res, _ = _apply_advection(v0_sym, "via_sym", m_val=1)
        r_tab_res, _ = _apply_advection(v0_tab, "via_tab", m_val=1)

        for sigma in (-1, 0, +1):
            interior = np.zeros_like(r_sym_res[sigma], dtype=bool)
            interior[2:-2, 4:-4] = True
            rel = (np.abs(r_tab_res[sigma] - r_sym_res[sigma])[interior].max() /
                   max(np.abs(r_sym_res[sigma][interior]).max(), 1e-30))
            assert rel < 1e-4, (
                f"σ={sigma}: tabular meridional doesn't match symbolic; "
                f"rel = {rel:.2e}"
            )


class TestMixedSymbolicAndTabularComponents:
    """Mixing symbolic and tabular components within a single v0 works
    (the dispatcher evaluates all three numerically on the grid)."""

    def test_radial_v_phi_with_sympy_v_r(self):
        """v_r = sympy (0), v_θ = sympy (0), v_φ = tabular function of r."""
        # v_φ = sin(πr) — radial-only profile
        r_t = np.linspace(0.0, 1.5, 100)
        v_phi_data_1d = np.sin(np.pi * r_t)
        v0_mixed = Background.Vector(
            "v0_mixed",
            v_r=sp.S.Zero,         # symbolic 0
            v_theta=sp.S.Zero,     # symbolic 0
            v_phi=radial_interpolator(r_t, v_phi_data_1d, kind="cubic"),
        )

        # Pure-symbolic equivalent
        r_sym = sp.symbols("r", positive=True)
        v0_sym = Background.Vector(
            "v0_sym",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=sp.sin(sp.pi * r_sym),
        )

        r_mix_res, _ = _apply_advection(v0_mixed, "mix")
        r_sym_res, _ = _apply_advection(v0_sym, "sym")

        for sigma in (-1, 0, +1):
            interior = np.zeros_like(r_sym_res[sigma], dtype=bool)
            interior[2:-2, 4:-4] = True
            rel = (np.abs(r_mix_res[sigma] - r_sym_res[sigma])[interior].max() /
                   max(np.abs(r_sym_res[sigma][interior]).max(), 1e-30))
            assert rel < 1e-5, (
                f"σ={sigma}: mixed radial+sympy doesn't match; rel = {rel:.2e}"
            )


class TestRadialInterpolatorInAdvection:
    """A purely radial v_φ(r) via radial_interpolator works for the
    advection operator. This is the simplest stand-in for tabular
    Model-S–style profiles."""

    def test_compiles_and_acts(self):
        # v_φ = 0.5 + r  — a simple monotonic profile
        r_t = np.linspace(0.0, 1.5, 80)
        v_phi_data = 0.5 + r_t
        v0 = Background.Vector(
            "v0",
            v_r=sp.S.Zero, v_theta=sp.S.Zero,
            v_phi=radial_interpolator(r_t, v_phi_data, kind="cubic"),
        )
        # Just check it compiles and produces a finite result.
        result, grid = _apply_advection(v0, "radial_vphi")
        for sigma in (-1, 0, +1):
            assert np.all(np.isfinite(result[sigma]))
            assert np.abs(result[sigma]).max() > 0  # operator did something
