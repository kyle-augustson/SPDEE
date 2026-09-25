"""Test that scalar background coefficients (e.g. ρ_0, η) work as
pointwise prefactors on existing operator tasks.

The pattern `Background.Scalar('rho0', expr=...) * Lap(v)` should
compile such that the resulting matrix acts as `ρ_0(r,θ) · Lap(v)`
pointwise. This is structurally what the walker already does for any
coefficient that reduces to a scalar SymPy expression — the
`_materialize_coefficient` machinery captures the prefactor and the
dispatcher's `_apply_scalar_coefficient_*` helpers apply it on the
grid.

This test exists to:
  (a) make the behavior explicit and testable;
  (b) prove that for variable-coefficient PDE applications (e.g.
      stratified momentum equation ρ_0(r) ∂_t v = ..., variable
      resistivity η(r) ∇²B, etc.) you don't need any new
      infrastructure — just write the equation naturally with the
      Background.Scalar as a Python multiplication.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Grad, Div, Equation,
    EquationSystem,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import split_vector_state


def _matches_pointwise_product(eq_with_coef, eq_bare, coef_expr, v_data,
                               m_val=0, N_r=16, N_theta=24, order=6):
    """Compile both equations and verify
        A_with_coef @ v_state = (coef on grid) · (A_bare @ v_state)
    pointwise on each spin component.
    """
    sys_with = EquationSystem([eq_with_coef], [])
    sys_bare = EquationSystem([eq_bare], [])
    grid = make_shell_grid(N_r=N_r, N_theta=N_theta,
                           r_inner=0.3, r_outer=1.0)
    cp_with = compile_problem(sys_with, grid, m=m_val,
                              order_r=order, order_theta=order)
    cp_bare = compile_problem(sys_bare, grid, m=m_val,
                              order_r=order, order_theta=order)
    state = cp_with.matrices.layout.pack({"v": v_data})
    A1v = cp_with.matrices.A_total @ state
    A2v = cp_bare.matrices.A_total @ state

    R, T = grid.mesh()
    r_sym, t_sym = sp.symbols("r theta", positive=True)
    coef_fn = sp.lambdify((r_sym, t_sym), coef_expr, modules="numpy")
    coef_grid = np.broadcast_to(np.asarray(coef_fn(R, T), dtype=complex),
                                R.shape).copy()

    eq_with_slice = cp_with.matrices.equation_row_slices[
        eq_with_coef.name]
    eq_bare_slice = cp_bare.matrices.equation_row_slices[eq_bare.name]
    A1v_split = split_vector_state(A1v[eq_with_slice], grid.N_r, grid.N_theta)
    A2v_split = split_vector_state(A2v[eq_bare_slice], grid.N_r, grid.N_theta)

    max_rel = 0.0
    for sigma in (-1, 0, +1):
        expected = coef_grid * A2v_split[sigma]
        err = np.abs(A1v_split[sigma] - expected)
        denom = max(float(np.max(np.abs(A2v_split[sigma]))), 1e-30)
        max_rel = max(max_rel, float(np.max(err) / denom))
    return max_rel


class TestScalarCoefficientOnLapVector:
    """rho0(r,θ) · Lap(v) is exactly pointwise · Lap(v)."""

    def test_radial_rho0(self):
        v = Unknown.Vector("v")
        r_sym = sp.symbols("r", positive=True)
        coef = 1 + r_sym ** 2
        rho0 = Background.Scalar("rho0", expr=coef)
        eq_with = Equation(rho0 * Lap(v), name="visc_with_rho0")
        eq_bare = Equation(Lap(v), name="visc_bare")
        # Use same axisymmetric v
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(N_r=16, N_theta=24, r_inner=0.3, r_outer=1.0)
        R, T = grid.mesh()
        v_data = {-1: np.zeros_like(R, dtype=complex),
                   0: (R * np.cos(T)).astype(complex),
                  +1: np.zeros_like(R, dtype=complex)}
        rel = _matches_pointwise_product(eq_with, eq_bare, coef, v_data, m_val=0)
        assert rel < 1e-12, (
            f"rho0·Lap(v) doesn't act pointwise; rel = {rel:.2e}"
        )

    def test_theta_dependent_rho0(self):
        v = Unknown.Vector("v")
        t_sym = sp.symbols("theta", positive=True)
        coef = 2 + sp.cos(t_sym)
        rho0 = Background.Scalar("rho0", expr=coef)
        eq_with = Equation(rho0 * Lap(v), name="visc_with_rho0")
        eq_bare = Equation(Lap(v), name="visc_bare")
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(N_r=16, N_theta=24, r_inner=0.3, r_outer=1.0)
        R, T = grid.mesh()
        v_data = {-1: np.zeros_like(R, dtype=complex),
                   0: (R * np.cos(T)).astype(complex),
                  +1: np.zeros_like(R, dtype=complex)}
        rel = _matches_pointwise_product(eq_with, eq_bare, coef, v_data, m_val=0)
        assert rel < 1e-11, (
            f"θ-dependent rho0·Lap(v) doesn't act pointwise; rel = {rel:.2e}"
        )


class TestScalarCoefficientOnGradScalar:
    """rho0 · Grad(p) is exactly pointwise · Grad(p).

    To get a square system, add a continuity-like row Div(v) for the
    vector unknown v. The resulting system has Grad(p) on row block
    'g_with_rho0' (vector, 3N rows, hosting v as gauge) and the Div(v)
    on a scalar row block — and we compare just the Grad rows.
    """

    def test_compiles_and_acts_pointwise(self):
        p = Unknown.Scalar("p")
        v = Unknown.Vector("v")
        r_sym = sp.symbols("r", positive=True)
        coef = 1 + sp.Rational(1, 2) * r_sym
        rho0 = Background.Scalar("rho0", expr=coef)
        # System: rho0·Grad(p) = 0  (vector eq, 3N rows)
        #         Div(v) = 0       (scalar eq, N rows)
        # Total rows: 3N + N = 4N. Unknowns: 3N (v) + N (p) = 4N. ✓
        eq_with = Equation(rho0 * Grad(p), name="g_with_rho0")
        eq_cont = Equation(Div(v), name="cont")
        eq_bare = Equation(Grad(p), name="g_bare")
        sys_with = EquationSystem([eq_with, eq_cont], [])
        sys_bare = EquationSystem([eq_bare, eq_cont], [])
        grid = make_shell_grid(N_r=16, N_theta=24, r_inner=0.3, r_outer=1.0)
        cp_with = compile_problem(sys_with, grid, m=0,
                                  order_r=6, order_theta=6)
        cp_bare = compile_problem(sys_bare, grid, m=0,
                                  order_r=6, order_theta=6)

        # p_state: a smooth scalar
        R, T = grid.mesh()
        p_data = (R**2 * np.cos(T)).astype(complex)
        state_with = cp_with.matrices.layout.pack({"p": p_data})
        state_bare = cp_bare.matrices.layout.pack({"p": p_data})

        A1v = cp_with.matrices.A_total @ state_with
        A2v = cp_bare.matrices.A_total @ state_bare

        r_sym2, t_sym2 = sp.symbols("r theta", positive=True)
        coef_fn = sp.lambdify((r_sym2, t_sym2), coef, modules="numpy")
        coef_grid = np.broadcast_to(np.asarray(coef_fn(R, T), dtype=complex),
                                    R.shape).copy()
        slice_with = cp_with.matrices.equation_row_slices["g_with_rho0"]
        slice_bare = cp_bare.matrices.equation_row_slices["g_bare"]
        A1v_split = split_vector_state(A1v[slice_with],
                                        grid.N_r, grid.N_theta)
        A2v_split = split_vector_state(A2v[slice_bare],
                                        grid.N_r, grid.N_theta)

        for sigma in (-1, 0, +1):
            expected = coef_grid * A2v_split[sigma]
            err = float(np.max(np.abs(A1v_split[sigma] - expected)))
            denom = max(float(np.max(np.abs(A2v_split[sigma]))), 1e-30)
            assert err / denom < 1e-12, (
                f"σ={sigma}: rho0·Grad(p) not pointwise; rel err {err/denom:.2e}"
            )
