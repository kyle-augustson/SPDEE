"""Cross-validation tests: aux-free MHD dispatch matches aux-augmented.

The aux-free path adds two new task kinds — TASK_CROSS_CURL_UNK_BG and
TASK_CURL_CROSS_UNK_BG — that the walker emits when it sees the
composed expressions

    Cross(Curl(unknown_vec), bg_vec)
    Curl(Cross(unknown_vec, bg_vec))

(in either argument order). The backend builds the corresponding
sparse blocks as matrix products of the elementary Curl and Cross
blocks (already validated against analytic answers in
``tests/test_mhd_composed_blocks.py``).

This file checks the end-to-end equivalence by solving the same
linearized MHD problem two ways — aux-augmented and aux-free
(default) — and confirming the **physical** part of the spectrum
matches. The aux-augmented path produces extra spurious modes from
the constraint equations (`λ = 0` modes corresponding to
auxiliary-unknown definitions); these are filtered out before
comparison.
"""
from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Dt, Grad, Div, Curl, Cross, Equation,
    EquationSystem, BoundaryCondition, Boundary, Dot, r_hat,
    theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mhd_system(B0_amp: float, m: int):
    """Stage-2 incompressible MHD with uniform B0 = B0_amp ẑ.

    ∂_t v + ∇p = ν ∇²v + Cross(Curl(B), B0)
    ∂_t B      = η ∇²B + Curl(Cross(v, B0))
    ∇·v = 0

    No-slip on v, B=0 on B at both boundaries; ν = η = 1.
    """
    v = Unknown.Vector("v")
    B = Unknown.Vector("B")
    p = Unknown.Scalar("p")
    t_sym = sp.symbols("theta", positive=True)
    B0 = Background.Vector(
        "B0",
        v_r=sp.S(B0_amp) * sp.cos(t_sym),
        v_theta=-sp.S(B0_amp) * sp.sin(t_sym),
        v_phi=sp.S.Zero,
    )
    eq_mom = Equation(
        Dt(v) + Grad(p) - Lap(v) - Cross(Curl(B), B0),
        name="momentum",
    )
    eq_ind = Equation(
        Dt(B) - Lap(B) - Curl(Cross(v, B0)),
        name="induction",
    )
    eq_cont = Equation(Div(v), name="continuity")
    bcs = []
    for loc, lbl in [(Boundary.R_INNER, "i"), (Boundary.R_OUTER, "o")]:
        for unit, c in [(r_hat, "r"), (theta_hat, "t"), (phi_hat, "p")]:
            bcs.append(BoundaryCondition(loc, Dot(unit, v), name=f"v{c}{lbl}"))
            bcs.append(BoundaryCondition(loc, Dot(unit, B), name=f"B{c}{lbl}"))
    return EquationSystem([eq_mom, eq_ind, eq_cont], bcs)


def _solve_both_paths(B0_amp, N=12, m=1, k=20, sigma=20.0):
    """Solve the same MHD problem aux-augmented and aux-free."""
    sys_ = _build_mhd_system(B0_amp, m)
    grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    cp_aux = compile_problem(
        sys_, grid, m=m, order_r=4, order_theta=4,
        prefer_aux_free=False,
    )
    cp_free = compile_problem(
        sys_, grid, m=m, order_r=4, order_theta=4,
        prefer_aux_free=True,
    )
    eigs_aux = np.array(cp_aux.solve_evp(k=k, sigma=sigma).eigenvalues)
    eigs_free = np.array(cp_free.solve_evp(k=k, sigma=sigma).eigenvalues)
    return eigs_aux, eigs_free, cp_aux, cp_free


def _filter_physical(eigs, abs_tol=1.0):
    """Keep finite eigenvalues with |real| > abs_tol (filters out the
    aux-constraint λ≈0 modes that show up in the aux-augmented path)."""
    finite = eigs[np.isfinite(eigs)]
    return finite[np.abs(finite.real) > abs_tol]


def _match_sets(a, b, atol=1e-6):
    """For each element of `a`, find the closest element of `b`. Return
    the array of per-element distances (in absolute eigenvalue terms)."""
    a = np.asarray(a); b = np.asarray(b)
    if len(a) == 0:
        return np.array([])
    diffs = []
    for x in a:
        d = np.min(np.abs(b - x))
        diffs.append(d)
    return np.array(diffs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAuxFreeMatchesAuxAugmented:
    """End-to-end cross-validation."""

    def test_state_size_smaller_aux_free(self):
        """Aux-free state should have exactly 3 unknowns × 3 N_r N_θ.
        Aux-augmented state has 5 unknowns: v, B, p, aux_curl_B,
        aux_cross_v_x_B0 (each vector → 3 spin blocks; p is 1 block).
        """
        N = 10
        sys_ = _build_mhd_system(B0_amp=0.5, m=1)
        grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
        cp_aux = compile_problem(
            sys_, grid, m=1, order_r=4, order_theta=4,
            prefer_aux_free=False,
        )
        cp_free = compile_problem(
            sys_, grid, m=1, order_r=4, order_theta=4,
            prefer_aux_free=True,
        )
        n_aux = cp_aux.matrices.A_total.shape[0]
        n_free = cp_free.matrices.A_total.shape[0]
        # aux-free: 2 vectors (v, B) × 3 spin × N_r·N_θ + 1 scalar p × N_r·N_θ
        #         = 6 N_r N_θ + N_r N_θ = 7 N_r N_θ.
        # aux-augmented: 4 vectors × 3 + 1 scalar = 13 N_r N_θ.
        assert n_free == 7 * N * N
        assert n_aux == 13 * N * N
        # aux-augmented should be larger.
        assert n_aux > n_free

    def test_eigenvalues_agree_zero_B0(self):
        """With B0=0 the v and B sectors decouple in both paths; the
        eigenvalues should match (no spurious aux modes when B0=0
        because the aux defining equations decouple cleanly).
        """
        eigs_aux, eigs_free, _, _ = _solve_both_paths(B0_amp=0.0, N=10)
        phys_aux = _filter_physical(eigs_aux)
        phys_free = _filter_physical(eigs_free)

        # Each aux-free eigenvalue should match some aux eigenvalue.
        diffs = _match_sets(phys_free, phys_aux)
        # Allow a generous tolerance (~1e-6) — ARPACK doesn't return
        # eigenvalues to machine precision, but the two systems
        # mathematically agree.
        max_diff = float(np.max(diffs)) if len(diffs) else 0.0
        assert max_diff < 1e-6, (
            f"With B0=0, max aux-free/aux distance is {max_diff:.3e}; "
            f"physical aux-free: {sorted(phys_free, key=lambda x: x.real)}, "
            f"physical aux: {sorted(phys_aux, key=lambda x: x.real)}"
        )

    def test_eigenvalues_agree_nonzero_B0_modest(self):
        """With B0 nonzero the v and B sectors couple via the composed
        operators; the aux-free dispatch and aux-augmented dispatch
        should give matching physical eigenvalues.
        """
        eigs_aux, eigs_free, _, _ = _solve_both_paths(
            B0_amp=0.3, N=10, k=20, sigma=20.0,
        )
        phys_aux = _filter_physical(eigs_aux)
        phys_free = _filter_physical(eigs_free)

        assert len(phys_aux) >= 3, (
            f"Expected ≥3 physical eigenvalues in aux path, got {len(phys_aux)}"
        )
        assert len(phys_free) >= 3, (
            f"Expected ≥3 physical eigenvalues in aux-free path, got {len(phys_free)}"
        )

        # Each aux-free eigenvalue should match an aux eigenvalue
        # closely. ARPACK precision plus the implicit numerical
        # differences (e.g. column-block ordering) put the floor at
        # ~1e-6 or so; we use 1e-5 to be safe.
        diffs = _match_sets(phys_free, phys_aux)
        max_diff = float(np.max(diffs))
        assert max_diff < 1e-5, (
            f"With B0=0.3, max aux-free/aux distance is {max_diff:.3e}; "
            f"aux-free physical: "
            f"{sorted(phys_free, key=lambda x: x.real)[:6]}; "
            f"aux physical: {sorted(phys_aux, key=lambda x: x.real)[:6]}"
        )

    @pytest.mark.parametrize("B0_amp", [0.05, 0.5])
    def test_eigenvalues_agree_across_B0_amplitudes(self, B0_amp):
        """Cross-validate at small and moderate B0 amplitudes."""
        eigs_aux, eigs_free, _, _ = _solve_both_paths(
            B0_amp=B0_amp, N=10, k=15, sigma=20.0,
        )
        phys_aux = _filter_physical(eigs_aux)
        phys_free = _filter_physical(eigs_free)

        diffs = _match_sets(phys_free, phys_aux)
        max_diff = float(np.max(diffs)) if len(diffs) else 0.0
        assert max_diff < 1e-5, (
            f"B0={B0_amp}: max aux-free/aux distance {max_diff:.3e}; "
            f"aux-free: {sorted(phys_free, key=lambda x: x.real)[:6]}; "
            f"aux: {sorted(phys_aux, key=lambda x: x.real)[:6]}"
        )

    def test_aux_free_state_layout_clean(self):
        """The aux-free path should have no aux unknown names in the
        layout — only the user's three unknowns."""
        sys_ = _build_mhd_system(B0_amp=0.3, m=1)
        grid = make_shell_grid(N_r=10, N_theta=10, r_inner=0.3, r_outer=1.0)
        cp = compile_problem(
            sys_, grid, m=1, order_r=4, order_theta=4,
            prefer_aux_free=True,
        )
        names = cp.matrices.layout.all_names
        # Just the user's unknowns: v, B, p.
        assert set(names) == {"v", "B", "p"}, f"unexpected names: {names}"
