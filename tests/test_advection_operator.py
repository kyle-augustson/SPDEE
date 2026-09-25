"""Test the v0·∇v advection operator end-to-end.

Strategy:
  1. Pick a smooth v0 and v.
  2. Compute (v0·∇v) analytically in spin basis using the derived formula.
  3. Apply spdee's compiled A_total to v_state and divide by r to recover
     the un-r-multiplied operator value.
  4. Compare on interior grid points.

Special cases verified analytically:
  - v0 = r̂ → (v0·∇v) = ∂_r v (component by component, no Christoffels).
  - v0 = ẑ, v = ẑ → 0 (constant Cartesian vector advecting itself).
  - v0 = ẑ, v = r̂ → -sin(θ)/r · θ̂.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Grad, Dot, Equation, EquationSystem, theta,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.ir.tensor_reduce import grad_vector_spin_components


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _analytic_advection_spin(v_spin_sym, v0_spin_sym, m_val):
    """Compute (v0·∇v)^σ symbolically for given v and v0.

    v_spin_sym, v0_spin_sym : dict {σ: sp.Expr} in (r, θ).
    """
    r_sym, t_sym = sp.symbols("r theta", positive=True)
    T_cov = grad_vector_spin_components(v_spin_sym, m_val)
    result = {}
    for sigma_3 in (-1, 0, +1):
        expr = sum(v0_spin_sym[sa] * T_cov[(sa, -sigma_3)] for sa in (-1, 0, +1))
        result[sigma_3] = sp.simplify(sp.expand(expr))
    return result


def _v_to_state_array(v_spin_sym, grid, layout, m_val):
    """Build the spdee state vector for unknown v given symbolic spin
    components in (r, θ). e^{imφ} is factored out.
    """
    r_sym, t_sym = sp.symbols("r theta", positive=True)
    N_r, N_theta = grid.N_r, grid.N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    T = np.ones((N_r, 1)) * grid.theta[None, :]
    v_data = {}
    for sigma in (-1, 0, +1):
        fn = sp.lambdify((r_sym, t_sym), v_spin_sym[sigma], modules="numpy")
        v_data[sigma] = np.broadcast_to(
            np.asarray(fn(R, T), dtype=complex), (N_r, N_theta)
        ).copy()
    return layout.pack({"v": v_data})


# -------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------

class TestAdvectionCompiles:
    def test_compile_ball(self):
        """Smoke: equation containing only Dot(bg_vec, Grad(v)) compiles."""
        from spdee.backends.fd2.grid import make_ball_grid
        v = Unknown.Vector("v")
        v0 = Background.Vector(
            "v0",
            v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
        )
        eq = Equation(Dot(v0, Grad(v)), name="advection")
        sys_ = EquationSystem([eq], [])
        grid = make_ball_grid(N_r=8, N_theta=16, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        assert cp.matrices.A_total.shape == (3 * 8 * 16, 3 * 8 * 16)


class TestAdvectionAgainstAnalytic:
    """Apply A_total to a state and compare to (r·formula) evaluated."""

    def _residual(self, v_spin_sym, v0_spin_phys_sym, m_val,
                  N_r=24, N_theta=48, order=6):
        """Compute the relative max residual on interior grid points."""
        r_sym, t_sym = sp.symbols("r theta", positive=True)

        # v0 in physical components for the Background.Vector constructor.
        v0_r, v0_th, v0_ph = v0_spin_phys_sym
        v = Unknown.Vector("v")
        v0 = Background.Vector(
            "v0", v_r=v0_r, v_theta=v0_th, v_phi=v0_ph,
        )
        eq = Equation(Dot(v0, Grad(v)), name="advection")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(N_r=N_r, N_theta=N_theta,
                               r_inner=0.3, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=m_val,
                             order_r=order, order_theta=order)
        A = cp.matrices.A_total

        # Convert v0_physical to spin (analytically, for the formula)
        sqrt2 = sp.sqrt(2)
        v0_spin_sym = {
            -1: (v0_th - sp.I * v0_ph) / sqrt2,
             0: v0_r,
            +1: (v0_th + sp.I * v0_ph) / sqrt2,
        }
        formula = _analytic_advection_spin(v_spin_sym, v0_spin_sym, m_val)

        # Multiply each formula by r to align with what spdee's A represents.
        formula_times_r = {s: sp.expand(r_sym * formula[s]) for s in (-1, 0, +1)}

        # Build state, apply A
        v_state = _v_to_state_array(v_spin_sym, grid, cp.matrices.layout, m_val)
        Av = A @ v_state

        # Compare on interior. Get the equation row slice and split into
        # vector spin components using the same shape as the unknown.
        from spdee.backends.fd2.layout import split_vector_state
        eq_slice = cp.matrices.equation_row_slices["advection"]
        Av_block = Av[eq_slice]
        Av_2d = split_vector_state(Av_block, N_r, N_theta)
        # Av_2d is dict {sigma: (N_r, N_theta) array}.

        # Evaluate the analytic formula on the same grid.
        R = grid.r[:, None] * np.ones((1, N_theta))
        T = np.ones((N_r, 1)) * grid.theta[None, :]

        max_rel_err = 0.0
        for sigma in (-1, 0, +1):
            fn = sp.lambdify((r_sym, t_sym), formula_times_r[sigma],
                              modules="numpy")
            target_2d = np.broadcast_to(
                np.asarray(fn(R, T), dtype=complex), (N_r, N_theta)
            ).copy()

            # Interior mask: skip i_r=0 (origin row) and θ near 0/π.
            interior = np.zeros_like(target_2d, dtype=bool)
            interior[3:-3, 4:-4] = True

            err = np.abs(Av_2d[sigma] - target_2d)
            tar = np.abs(target_2d)
            denom = max(np.max(tar[interior]), 1e-30)
            max_rel_err = max(max_rel_err, float(np.max(err[interior]) / denom))
        return max_rel_err, None

    def test_pure_radial_v0(self):
        """v0 = r̂: (v0·∇v) = ∂_r v per spin component, no Christoffels."""
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        # v = (smooth function), e.g. v^0 = r², v^± = r sin(θ).
        v_spin_sym = {
            -1: r_sym * sp.sin(t_sym),
             0: r_sym ** 2,
            +1: r_sym * sp.sin(t_sym),
        }
        # v0 = r̂ in physical: v0_r = 1.
        v0_phys = (sp.S.One, sp.S.Zero, sp.S.Zero)
        rel, _ = self._residual(v_spin_sym, v0_phys, m_val=1,
                                  N_r=20, N_theta=32, order=6)
        assert rel < 1e-6, f"Pure radial v0 case rel err {rel:.2e}"

    def test_z_hat_v0(self):
        """v0 = ẑ: the standard rotation-axis background.

        Uses a nontrivial test v and just checks the closed-form vs
        the formula evaluated on the grid agree.
        """
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        v_spin_sym = {
            -1: r_sym ** 2 * sp.sin(t_sym),
             0: r_sym * sp.cos(t_sym),
            +1: r_sym ** 2 * sp.cos(t_sym),
        }
        # v0 = ẑ
        v0_phys = (sp.cos(t_sym), -sp.sin(t_sym), sp.S.Zero)
        rel, _ = self._residual(v_spin_sym, v0_phys, m_val=2,
                                  N_r=24, N_theta=48, order=6)
        assert rel < 1e-5, f"ẑ background rel err {rel:.2e}"

    def test_meridional_v0(self):
        """v0 with both v_r and v_θ components (meridional flow)."""
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        v_spin_sym = {
            -1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
             0: r_sym ** 2 * sp.sin(t_sym),
            +1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        }
        # v0 = r sin(θ) r̂ - r cos(θ) θ̂ (some smooth meridional flow)
        v0_phys = (
            r_sym * sp.sin(t_sym),
            -r_sym * sp.cos(t_sym),
            sp.S.Zero,
        )
        rel, _ = self._residual(v_spin_sym, v0_phys, m_val=1,
                                  N_r=28, N_theta=56, order=6)
        assert rel < 1e-5, f"Meridional v0 rel err {rel:.2e}"

    def test_azimuthal_v0(self):
        """v0 with only v_φ component (zonal flow)."""
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        v_spin_sym = {
            -1: r_sym ** 2 * sp.sin(t_sym),
             0: r_sym * sp.cos(t_sym),
            +1: r_sym ** 2 * sp.sin(t_sym),
        }
        # v0 = r sin θ · φ̂ (rigid-body-like azimuthal)
        v0_phys = (
            sp.S.Zero,
            sp.S.Zero,
            r_sym * sp.sin(t_sym),
        )
        rel, _ = self._residual(v_spin_sym, v0_phys, m_val=3,
                                  N_r=24, N_theta=48, order=6)
        assert rel < 1e-5, f"Azimuthal v0 rel err {rel:.2e}"
