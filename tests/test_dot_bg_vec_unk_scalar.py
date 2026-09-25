"""Test the pointwise inner-product operator
Dot(background_vector, unknown_vector) → scalar.

This is the operator needed for the linearized energy/entropy equation:
    ∂_t s + v · ∇s_0 = κ ∇²s
where ∇s_0 is pre-computed as a background vector field. The operator
at each grid point computes a scalar = a_r·b_r + a_θ·b_θ + a_φ·b_φ
where a is the bg vector and b is the unknown vector.

In V19 spin components this is:
    s = a^0 b^0 + a^+ b^- + a^- b^+
(so columns of the unknown couple to bg spin components as
σ=0 ↔ a^0, σ=-1 ↔ a^+, σ=+1 ↔ a^-).

The dispatcher builds a sparse (N, 3N) block — scalar output × vector
input. Pointwise → no derivative stencils, natural r-power 0.

Cross-validation cases:
  - Radial bg (e^r̂): Dot(r̂, v) extracts v_r. Matches against direct
    extraction.
  - General-direction bg: matches against an independent numpy-level
    physical-component contraction.
  - Operand-flipped form Dot(v, bg) gives the same result as Dot(bg, v).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Dot, Dt, Lap, Equation, EquationSystem,
    BoundaryCondition, Boundary, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import reshape_2d


def _build_system(bg_phys, flip_operands=False, N=16):
    """Build a coupled system:
        Dt(s) + Dot(bg, v) - Lap(s) = 0     (energy)
        Dt(v)             - Lap(v)  = 0     (momentum, decoupled from s)
    so the energy row block is the only place the new pattern is exercised.
    """
    bg = Background.Vector(
        "bg_field",
        v_r=bg_phys[0], v_theta=bg_phys[1], v_phi=bg_phys[2],
    )
    s = Unknown.Scalar("s")
    v = Unknown.Vector("v")
    dot_term = Dot(v, bg) if flip_operands else Dot(bg, v)
    eq_energy = Equation(Dt(s) + dot_term - Lap(s), name="energy")
    eq_mom = Equation(Dt(v) - Lap(v), name="momentum")
    bcs = [
        BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
        BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
    ]
    for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
        for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
            bcs.append(BoundaryCondition(loc, Dot(unit, v), name=f"v{c}{lbl}"))
    sys_ = EquationSystem([eq_energy, eq_mom], bcs)
    grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=6)
    return cp, grid


def _independent_dot_phys(bg_phys_grid, v_data, grid):
    """Independent computation of (bg · v)_phys on the grid.

    bg_phys_grid: tuple (bg_r, bg_th, bg_ph) of numpy arrays on the grid.
    v_data: dict {-1, 0, +1: numpy arrays} of v spin components.
    Returns: 2D array of (bg · v)_phys shape (N_r, N_theta).
    """
    sqrt2 = np.sqrt(2.0)
    v_r = v_data[0]
    v_th = (v_data[+1] + v_data[-1]) / sqrt2
    v_ph = 1j * (v_data[-1] - v_data[+1]) / sqrt2
    bg_r, bg_th, bg_ph = bg_phys_grid
    return bg_r * v_r + bg_th * v_th + bg_ph * v_ph


def _extract_energy_block(cp, grid, v_data):
    """Apply A_total to a state with s=0 and given v; return the energy
    equation row block, reshaped to 2D (N_r, N_theta).
    """
    state = cp.matrices.layout.pack({
        "s": np.zeros((grid.N_r, grid.N_theta), dtype=complex),
        "v": v_data,
    })
    result = cp.matrices.A_total @ state
    energy_slice = cp.matrices.equation_row_slices["energy"]
    return reshape_2d(result[energy_slice], grid.N_r, grid.N_theta)


class TestDotBgVecUnkVecScalar:
    """End-to-end tests of the scalar Dot dispatcher."""

    def test_radial_bg_extracts_vr(self):
        """bg = r̂ (radial unit vector). Then bg · v = v_r.
        The compiled energy block should equal r² · v_r at each point.
        """
        bg_phys = (sp.S.One, sp.S.Zero, sp.S.Zero)
        cp, grid = _build_system(bg_phys, N=16)

        R, T = grid.mesh()
        v_data = {
            -1: (R**2 * np.sin(T)).astype(complex),
             0: (R * np.cos(T)).astype(complex),   # this is v_r
            +1: (R**2 * np.sin(T) * np.cos(T)).astype(complex),
        }
        energy_2d = _extract_energy_block(cp, grid, v_data)
        # A_eq for energy is diag(r²) — so output is r²·(bg·v) = r²·v_r.
        expected = R**2 * v_data[0].real  # v_r is real
        # Compare in interior (away from boundary stencils — though
        # pointwise has no stencil, BCs DO overwrite some boundary rows)
        interior = np.zeros_like(R, dtype=bool)
        interior[1:-1, :] = True
        err = float(np.max(np.abs(energy_2d.real - expected)[interior]))
        denom = float(np.max(np.abs(expected)[interior]))
        assert err / denom < 1e-13, (
            f"radial bg: rel err {err/denom:.2e}"
        )

    def test_general_bg_matches_phys_contraction(self):
        """bg with all three components nonzero. Compare to an
        independent physical-component contraction.
        """
        t_sym = sp.symbols("theta", positive=True)
        bg_phys = (
            sp.cos(t_sym),
            -sp.sin(t_sym),
            sp.sin(t_sym) * sp.cos(t_sym),
        )
        cp, grid = _build_system(bg_phys, N=18)

        R, T = grid.mesh()
        v_data = {
            -1: (R * np.cos(T) + 1j * R**2 * np.sin(T)).astype(complex),
             0: (R**2 * np.sin(T)).astype(complex),
            +1: (R * np.sin(T) * np.cos(T) - 1j * R).astype(complex),
        }
        energy_2d = _extract_energy_block(cp, grid, v_data)

        bg_phys_grid = (
            np.cos(T), -np.sin(T), np.sin(T) * np.cos(T),
        )
        dot_phys = _independent_dot_phys(bg_phys_grid, v_data, grid)
        expected = R**2 * dot_phys

        interior = np.zeros_like(R, dtype=bool)
        interior[1:-1, :] = True
        err = float(np.max(np.abs(energy_2d - expected)[interior]))
        denom = float(np.max(np.abs(expected)[interior]))
        assert err / denom < 1e-13, (
            f"general bg: rel err {err/denom:.2e}"
        )

    def test_operand_flip_gives_same_result(self):
        """Dot is commutative on physical components: Dot(v, bg) should
        produce the same operator as Dot(bg, v).
        """
        t_sym = sp.symbols("theta", positive=True)
        bg_phys = (
            sp.cos(t_sym), -sp.sin(t_sym), sp.sin(t_sym) * sp.cos(t_sym),
        )
        cp_normal, grid = _build_system(bg_phys, flip_operands=False, N=14)
        cp_flipped, _ = _build_system(bg_phys, flip_operands=True, N=14)

        R, T = grid.mesh()
        v_data = {
            -1: (R**2 * np.sin(T)).astype(complex),
             0: (R * np.cos(T)).astype(complex),
            +1: (R**2 * np.cos(T)).astype(complex),
        }
        e_normal = _extract_energy_block(cp_normal, grid, v_data)
        e_flipped = _extract_energy_block(cp_flipped, grid, v_data)

        # Should be exactly equal (same operator, same machine ops).
        err = float(np.max(np.abs(e_normal - e_flipped)))
        denom = max(float(np.max(np.abs(e_normal))), 1e-30)
        assert err / denom < 1e-14, (
            f"Dot(bg,v) vs Dot(v,bg) differ by rel {err/denom:.2e}"
        )

    def test_zero_bg_produces_zero_operator(self):
        """When bg is identically zero, the dispatcher produces a zero
        contribution (zero-bg dropping happens in the augmentation pass,
        not here).
        """
        bg_phys = (sp.S.Zero, sp.S.Zero, sp.S.Zero)
        cp, grid = _build_system(bg_phys, N=12)

        R, T = grid.mesh()
        v_data = {
            -1: (R**2 * np.sin(T)).astype(complex),
             0: (R * np.cos(T)).astype(complex),
            +1: (R**2 * np.cos(T)).astype(complex),
        }
        energy_2d = _extract_energy_block(cp, grid, v_data)
        # All entries should be ~0.
        assert float(np.max(np.abs(energy_2d))) < 1e-14
