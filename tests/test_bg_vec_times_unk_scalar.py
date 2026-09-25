"""Test the pointwise scalar-vector product operator
  unknown_scalar * background_vector → vector.

This is the operator needed for the buoyancy term `-s · ĝ` in
linearized stratified momentum equations. The unknown is a scalar;
the background is a vector. At each grid point, the vector output's
σ component gets s(r,θ) · bg^σ(r,θ).

In V19 spin components this is straightforward — the operator is
diagonal in spin (no cross-spin coupling), since multiplication by a
scalar respects spin.

The dispatcher builds a sparse (3N, N) block — vector output ×
scalar input. Three diagonal sub-blocks stacked vertically, each
diag(bg^σ(grid)) for σ ∈ {-1, 0, +1}.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Dt, Lap, Grad, Div, Dot, Equation, EquationSystem,
    BoundaryCondition, Boundary, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import split_vector_state


def _build_system(g_phys, N=16):
    """Build a system where mom has +s*g term, isolating the new dispatcher:
        ∂_t v + ∇p - ∇²v + s·g = 0
        ∇·v = 0
        ∂_t s - ∇²s = 0
    """
    g_field = Background.Vector(
        "g_field",
        v_r=g_phys[0], v_theta=g_phys[1], v_phi=g_phys[2],
    )
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    s = Unknown.Scalar("s")
    eq_mom = Equation(
        Dt(v) + Grad(p) - Lap(v) + s * g_field,
        name="momentum",
    )
    eq_cont = Equation(Div(v), name="continuity")
    eq_energy = Equation(Dt(s) - Lap(s), name="energy")
    bcs = [
        BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
        BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
    ]
    for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
        for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
            bcs.append(BoundaryCondition(loc, Dot(unit, v), name=f"v{c}{lbl}"))
    sys_ = EquationSystem([eq_mom, eq_cont, eq_energy], bcs)
    grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=6)
    return cp, grid


def _expected_mom_contribution(g_phys_grid, s_data, R, grid):
    """Independent computation: expected = r² · s · g_σ at each grid
    point, where σ runs through V19 spin components.

    g_phys_grid: tuple (g_r, g_th, g_ph) numpy arrays on grid.
    s_data: numpy array of s values on grid, shape (N_r, N_theta).
    Returns: dict {σ: 2D array (N_r, N_theta)} with the expected
             contribution to the momentum row block.
    """
    sqrt2 = np.sqrt(2.0)
    g_r, g_th, g_ph = g_phys_grid
    g_minus = (g_th - 1j * g_ph) / sqrt2
    g_zero = g_r
    g_plus = (g_th + 1j * g_ph) / sqrt2
    # Mom equation has Lap so A_eq = diag(r²)
    return {
        -1: R**2 * s_data * g_minus,
         0: R**2 * s_data * g_zero,
        +1: R**2 * s_data * g_plus,
    }


class TestBgVecTimesUnkScalar:
    """End-to-end tests of the buoyancy-style dispatcher."""

    def test_radial_g_gives_radial_buoyancy(self):
        """g = -r̂ → buoyancy contributes only to σ=0 (radial) component.
        With g_θ = g_φ = 0, the σ=±1 contributions vanish.
        """
        g_phys = (sp.S(-1), sp.S.Zero, sp.S.Zero)
        cp, grid = _build_system(g_phys, N=14)
        R, T = grid.mesh()
        s_data = (R * np.cos(T)).astype(complex)
        # Pack state with only s nonzero
        state = cp.matrices.layout.pack({
            "v": {sigma: np.zeros((grid.N_r, grid.N_theta), dtype=complex)
                  for sigma in (-1, 0, +1)},
            "p": np.zeros((grid.N_r, grid.N_theta), dtype=complex),
            "s": s_data,
        })
        result = cp.matrices.A_total @ state
        mom_slice = cp.matrices.equation_row_slices["momentum"]
        mom_split = split_vector_state(
            result[mom_slice], grid.N_r, grid.N_theta,
        )
        # σ=0: should be r²·s·(-1)
        expected_0 = -R**2 * s_data.real
        interior = np.zeros_like(R, dtype=bool)
        interior[1:-1, :] = True
        err_0 = float(np.max(np.abs(mom_split[0].real - expected_0)[interior]))
        denom = float(np.max(np.abs(expected_0)[interior]))
        assert err_0 / denom < 1e-13, (
            f"σ=0 (radial): rel err {err_0/denom:.2e}"
        )
        # σ=±1: should be exactly zero (g has no θ or φ component)
        for sigma in (-1, +1):
            max_abs = float(np.max(np.abs(mom_split[sigma])))
            assert max_abs < 1e-13, (
                f"σ={sigma}: expected zero, got max |value| = {max_abs:.2e}"
            )

    def test_general_g_matches_phys_product(self):
        """g with all three components nonzero. Compare to an
        independent physical-component product s · g.
        """
        t_sym = sp.symbols("theta", positive=True)
        g_phys = (
            sp.cos(t_sym), -sp.sin(t_sym), sp.sin(t_sym) * sp.cos(t_sym),
        )
        cp, grid = _build_system(g_phys, N=16)
        R, T = grid.mesh()
        s_data = (R * np.cos(T) + R**2 * np.sin(T)).astype(complex)
        state = cp.matrices.layout.pack({
            "v": {sigma: np.zeros((grid.N_r, grid.N_theta), dtype=complex)
                  for sigma in (-1, 0, +1)},
            "p": np.zeros((grid.N_r, grid.N_theta), dtype=complex),
            "s": s_data,
        })
        result = cp.matrices.A_total @ state
        mom_slice = cp.matrices.equation_row_slices["momentum"]
        mom_split = split_vector_state(
            result[mom_slice], grid.N_r, grid.N_theta,
        )
        g_phys_grid = (
            np.cos(T), -np.sin(T), np.sin(T) * np.cos(T),
        )
        expected = _expected_mom_contribution(g_phys_grid, s_data, R, grid)
        interior = np.zeros_like(R, dtype=bool)
        interior[1:-1, :] = True
        for sigma in (-1, 0, +1):
            err = float(np.max(np.abs(mom_split[sigma] - expected[sigma])[interior]))
            denom = max(float(np.max(np.abs(expected[sigma])[interior])), 1e-30)
            assert err / denom < 1e-13, (
                f"σ={sigma}: rel err {err/denom:.2e}"
            )

    def test_zero_g_produces_zero_block(self):
        """When g = 0 vector, the dispatcher produces zero contributions
        in all spin components.
        """
        g_phys = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
        cp, grid = _build_system(g_phys, N=12)
        R, T = grid.mesh()
        s_data = (R * np.cos(T)).astype(complex)
        state = cp.matrices.layout.pack({
            "v": {sigma: np.zeros((grid.N_r, grid.N_theta), dtype=complex)
                  for sigma in (-1, 0, +1)},
            "p": np.zeros((grid.N_r, grid.N_theta), dtype=complex),
            "s": s_data,
        })
        result = cp.matrices.A_total @ state
        mom_slice = cp.matrices.equation_row_slices["momentum"]
        mom_split = split_vector_state(
            result[mom_slice], grid.N_r, grid.N_theta,
        )
        for sigma in (-1, 0, +1):
            max_abs = float(np.max(np.abs(mom_split[sigma])))
            assert max_abs < 1e-14, (
                f"σ={sigma}: zero-g should give zero, got max = {max_abs:.2e}"
            )

    def test_operand_flip_gives_same_result(self):
        """s * g_field and g_field * s should produce identical operators
        (scalar-vector product is commutative).
        """
        t_sym = sp.symbols("theta", positive=True)
        g_field = Background.Vector(
            "g_field",
            v_r=sp.cos(t_sym), v_theta=-sp.sin(t_sym), v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        s = Unknown.Scalar("s")
        # Build two systems differing only in operand order of s * g.
        eq_mom_a = Equation(
            Dt(v) + Grad(p) - Lap(v) + s * g_field, name="momentum",
        )
        eq_mom_b = Equation(
            Dt(v) + Grad(p) - Lap(v) + g_field * s, name="momentum",
        )
        eq_cont = Equation(Div(v), name="continuity")
        eq_energy = Equation(Dt(s) - Lap(s), name="energy")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
            BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
        ]
        for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
            for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
                bcs.append(BoundaryCondition(loc, Dot(unit, v), name=f"v{c}{lbl}"))
        grid = make_shell_grid(N_r=12, N_theta=12, r_inner=0.3, r_outer=1.0)
        cp_a = compile_problem(
            EquationSystem([eq_mom_a, eq_cont, eq_energy], bcs),
            grid, m=0, order_r=6, order_theta=6,
        )
        cp_b = compile_problem(
            EquationSystem([eq_mom_b, eq_cont, eq_energy], bcs),
            grid, m=0, order_r=6, order_theta=6,
        )
        # Their A_total should be identical.
        diff = cp_a.matrices.A_total - cp_b.matrices.A_total
        max_diff = abs(diff).max() if diff.nnz > 0 else 0.0
        assert max_diff < 1e-14, (
            f"s*g vs g*s: max |diff| = {max_diff:.2e}"
        )
