"""Test the pointwise matvec operator Dot(background_tensor, unknown_vector).

The operator: at each grid point, with T_phys a 3x3 matrix and v a
contravariant spin vector, return w = T·v with the convention
    w_phys = T_phys · v_phys
and w expressed back in spin.

In V19 spin form: M^{σ_out, σ_in} = U · T_phys · U†, and
    w^{σ_out} = Σ_{σ_in} M^{σ_out, σ_in} v^{σ_in}.

Use cases:
  - User pre-computes ∇v_0 as a Background.Tensor for the term
    v · ∇v_0 = (Transpose(∇v_0)) · v (so they'd use Transpose if
    they want the natural contraction).
  - Pre-computed pointwise linear operators (Hall MHD coefficients,
    anisotropic mobility tensors, etc.) on a vector unknown.

Cross-validation cases:
  - Identity tensor T_phys = I → Dot(T, v) = v (the operator is
    identity on the vector).
  - Skew-symmetric T from Ω̂ × · (the Coriolis matrix): should match
    Cross(Ω̂, v).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Dot, Cross, Equation, EquationSystem,
)
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import split_vector_state


def _identity_tensor():
    """Build a Background.Tensor whose components form the identity matrix."""
    return Background.Tensor("I_tensor", components={
        ("r", "r"): sp.S.One,
        ("r", "theta"): sp.S.Zero,
        ("r", "phi"): sp.S.Zero,
        ("theta", "r"): sp.S.Zero,
        ("theta", "theta"): sp.S.One,
        ("theta", "phi"): sp.S.Zero,
        ("phi", "r"): sp.S.Zero,
        ("phi", "theta"): sp.S.Zero,
        ("phi", "phi"): sp.S.One,
    })


def _skew_z_tensor():
    """Skew-symmetric tensor implementing ẑ × ·.

    ẑ = (cos θ, -sin θ, 0)_phys. For K·v = ẑ × v in physical components:
        (ẑ × v)_r = z_θ v_φ - z_φ v_θ = -sin θ · v_φ
        (ẑ × v)_θ = z_φ v_r - z_r v_φ = -cos θ · v_φ
        (ẑ × v)_φ = z_r v_θ - z_θ v_r = cos θ v_θ + sin θ v_r
    So
        K_{rr} = 0, K_{rθ} = 0, K_{rφ} = -sin θ
        K_{θr} = 0, K_{θθ} = 0, K_{θφ} = -cos θ
        K_{φr} = sin θ, K_{φθ} = cos θ, K_{φφ} = 0
    """
    t = sp.symbols("theta", positive=True)
    return Background.Tensor("K_z", components={
        ("r", "r"): sp.S.Zero,
        ("r", "theta"): sp.S.Zero,
        ("r", "phi"): -sp.sin(t),
        ("theta", "r"): sp.S.Zero,
        ("theta", "theta"): sp.S.Zero,
        ("theta", "phi"): -sp.cos(t),
        ("phi", "r"): sp.sin(t),
        ("phi", "theta"): sp.cos(t),
        ("phi", "phi"): sp.S.Zero,
    })


def _build_v_state(v_spin_sym, grid, layout):
    """Build state vector from symbolic spin components."""
    r_sym, t_sym = sp.symbols("r theta", positive=True)
    R, T = grid.mesh()
    v_data = {}
    for sigma in (-1, 0, +1):
        fn = sp.lambdify((r_sym, t_sym), v_spin_sym[sigma], modules="numpy")
        v_data[sigma] = np.broadcast_to(
            np.asarray(fn(R, T), dtype=complex), R.shape
        ).copy()
    return layout.pack({"v": v_data})


class TestDotBgTensorCompiles:
    def test_compile_ball(self):
        v = Unknown.Vector("v")
        T = _identity_tensor()
        eq = Equation(Dot(T, v), name="matvec")
        sys_ = EquationSystem([eq], [])
        grid = make_ball_grid(N_r=6, N_theta=12, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=0, order_r=4, order_theta=4)
        assert cp.matrices.A_total.shape == (3 * 6 * 12, 3 * 6 * 12)


class TestDotBgTensorIdentity:
    """Dot(I, v) = v: applying the identity tensor should give back v."""

    def test_identity_tensor_acts_as_identity(self):
        v = Unknown.Vector("v")
        T = _identity_tensor()
        eq = Equation(Dot(T, v), name="matvec")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(N_r=12, N_theta=24, r_inner=0.3, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=2, order_r=6, order_theta=6)

        # Build a nontrivial v
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        v_spin_sym = {
            -1: r_sym ** 2 * sp.sin(t_sym),
             0: r_sym * sp.cos(t_sym),
            +1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        }
        v_state = _build_v_state(v_spin_sym, grid, cp.matrices.layout)
        Av = cp.matrices.A_total @ v_state

        # The matvec output should equal v (since T = I), BUT note the
        # equation mass: spdee compiles `A_eq · OP(v) = block · v_state`
        # with A_eq = identity for pointwise tasks (r-power 0). So the
        # block IS the operator directly, and Av should equal v_state's
        # vector-block (interpreting the equation rows in spin layout).
        eq_slice = cp.matrices.equation_row_slices["matvec"]
        Av_split = split_vector_state(Av[eq_slice], grid.N_r, grid.N_theta)
        v_entry = cp.matrices.layout.entry("v")
        v_state_vec = v_state[v_entry.slice]
        v_split = split_vector_state(v_state_vec, grid.N_r, grid.N_theta)

        for sigma in (-1, 0, +1):
            err = np.abs(Av_split[sigma] - v_split[sigma]).max()
            denom = max(np.abs(v_split[sigma]).max(), 1e-30)
            assert err / denom < 1e-13, (
                f"σ={sigma}: identity matvec not pointwise identity; "
                f"rel err {err/denom:.2e}"
            )


class TestDotBgTensorMatchesCross:
    """For the skew-symmetric tensor K_ẑ implementing ẑ × ·,
    Dot(K_ẑ, v) should equal Cross(ẑ, v).
    """

    def test_skew_z_matches_cross_z(self):
        v = Unknown.Vector("v")
        K = _skew_z_tensor()
        omega_bg = Background.Vector(
            "omega",
            v_r=sp.cos(sp.symbols("theta", positive=True)),
            v_theta=-sp.sin(sp.symbols("theta", positive=True)),
            v_phi=sp.S.Zero,
        )
        eq_dot = Equation(Dot(K, v), name="via_dot")
        eq_cross = Equation(Cross(omega_bg, v), name="via_cross")

        sys_dot = EquationSystem([eq_dot], [])
        sys_cross = EquationSystem([eq_cross], [])
        grid = make_shell_grid(N_r=12, N_theta=24, r_inner=0.3, r_outer=1.0)
        cp_dot = compile_problem(sys_dot, grid, m=1, order_r=6, order_theta=6)
        cp_cross = compile_problem(sys_cross, grid, m=1, order_r=6, order_theta=6)

        r_sym, t_sym = sp.symbols("r theta", positive=True)
        v_spin_sym = {
            -1: r_sym * sp.cos(t_sym),
             0: r_sym ** 2,
            +1: r_sym * sp.sin(t_sym),
        }
        state_dot = _build_v_state(v_spin_sym, grid, cp_dot.matrices.layout)
        state_cross = _build_v_state(v_spin_sym, grid, cp_cross.matrices.layout)

        Av_dot = cp_dot.matrices.A_total @ state_dot
        Av_cross = cp_cross.matrices.A_total @ state_cross

        slice_dot = cp_dot.matrices.equation_row_slices["via_dot"]
        slice_cross = cp_cross.matrices.equation_row_slices["via_cross"]
        d = split_vector_state(Av_dot[slice_dot], grid.N_r, grid.N_theta)
        c = split_vector_state(Av_cross[slice_cross], grid.N_r, grid.N_theta)

        for sigma in (-1, 0, +1):
            err = np.abs(d[sigma] - c[sigma]).max()
            denom = max(np.abs(c[sigma]).max(), 1e-30)
            assert err / denom < 1e-13, (
                f"σ={sigma}: Dot(K_ẑ, v) ≠ Cross(ẑ, v); rel err {err/denom:.2e}"
            )


class TestDotBgTensorExplicitMatvec:
    """Dot(T, v) for an explicit T agrees with the closed-form matvec."""

    def test_diagonal_tensor(self):
        """T = diag(a, b, c) in physical components.
        Then T·v in physical = (a v_r, b v_θ, c v_φ).
        """
        v = Unknown.Vector("v")
        r_sym, t_sym = sp.symbols("r theta", positive=True)
        a, b, c = 1 + r_sym, 2 + sp.cos(t_sym), r_sym * sp.sin(t_sym) + 1
        T = Background.Tensor("T", components={
            ("r", "r"): a, ("r", "theta"): sp.S.Zero, ("r", "phi"): sp.S.Zero,
            ("theta", "r"): sp.S.Zero, ("theta", "theta"): b, ("theta", "phi"): sp.S.Zero,
            ("phi", "r"): sp.S.Zero, ("phi", "theta"): sp.S.Zero, ("phi", "phi"): c,
        })
        eq = Equation(Dot(T, v), name="diag_matvec")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(N_r=12, N_theta=24, r_inner=0.3, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=2, order_r=6, order_theta=6)

        # Build v in physical, then convert to spin for state, AND
        # build T·v in physical from a, b, c → then convert to spin
        # for comparison.
        v_r_sym = r_sym ** 2 * sp.cos(t_sym)
        v_th_sym = r_sym * sp.sin(t_sym)
        v_ph_sym = r_sym ** 3
        sqrt2 = sp.sqrt(2)
        v_spin_sym = {
            -1: (v_th_sym - sp.I * v_ph_sym) / sqrt2,
             0: v_r_sym,
            +1: (v_th_sym + sp.I * v_ph_sym) / sqrt2,
        }
        # T·v in physical:
        Tv_r_sym = a * v_r_sym
        Tv_th_sym = b * v_th_sym
        Tv_ph_sym = c * v_ph_sym
        Tv_spin_sym = {
            -1: (Tv_th_sym - sp.I * Tv_ph_sym) / sqrt2,
             0: Tv_r_sym,
            +1: (Tv_th_sym + sp.I * Tv_ph_sym) / sqrt2,
        }

        state = _build_v_state(v_spin_sym, grid, cp.matrices.layout)
        Av = cp.matrices.A_total @ state
        eq_slice = cp.matrices.equation_row_slices["diag_matvec"]
        Av_split = split_vector_state(Av[eq_slice], grid.N_r, grid.N_theta)

        R, T_grid = grid.mesh()
        for sigma in (-1, 0, +1):
            fn = sp.lambdify((r_sym, t_sym), Tv_spin_sym[sigma], modules="numpy")
            expected = np.asarray(fn(R, T_grid), dtype=complex)
            err = np.abs(Av_split[sigma] - expected).max()
            denom = max(np.abs(expected).max(), 1e-30)
            assert err / denom < 1e-13, (
                f"σ={sigma}: diag matvec rel err {err/denom:.2e}"
            )
