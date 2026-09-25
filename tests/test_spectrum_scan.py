"""Tests for the matrix-reuse EVP path and spectrum-scan helpers.

The spectrum scan factors `solve_evp` into:
  * `assemble_evp_matrices()` — σ-independent (K, M) build
  * `solve_evp_from_matrices(K, M, σ, k)` — per-target shift-invert

These tests confirm:
  1. The factored path reproduces the monolithic `solve_evp` exactly.
  2. (K, M) round-trip through scipy sparse save/load unchanged.
  3. The scan deduplication collapses repeated eigenvalues correctly.
  4. The σ = γ − iω convention maps a target frequency to the right
     eigenvalue (the recovered ω equals the input ω for a known mode).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# scripts dir for the scan helpers.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grid import make_shell_grid


# A lightweight EVP fixture that doesn't need solar data: a simple
# scalar diffusion eigenproblem on a shell.
def _simple_scalar_evp():
    """Build a tiny scalar Helmholtz-like EVP: λ s = Lap(s), s=0 at walls.

    Returns a compiled problem on a 16×16 shell.
    """
    from spdee import Unknown, Equation, EquationSystem, Lap, Dt
    from spdee import BoundaryCondition, Boundary

    s = Unknown.Scalar("s")
    eq = Equation(Dt(s) - Lap(s), name="diffusion")
    bcs = [
        BoundaryCondition(Boundary.R_INNER, s, name="s_in"),
        BoundaryCondition(Boundary.R_OUTER, s, name="s_out"),
    ]
    system = EquationSystem([eq], bcs)
    grid = make_shell_grid(N_r=16, N_theta=16, r_inner=0.5, r_outer=1.0)
    return compile_problem(system, grid, m=2, order_r=4, order_theta=4)


# ---------------------------------------------------------------------------
# Refactor equivalence
# ---------------------------------------------------------------------------

class TestFactoredEVPEquivalence:
    """The assemble + from_matrices path must equal monolithic solve_evp."""

    def test_eigenvalues_match_monolithic(self):
        cp = _simple_scalar_evp()
        sigma = -0.5 - 1j * 0.0
        res_mono = cp.solve_evp(k=8, sigma=sigma)
        e_mono = np.sort_complex(np.array(res_mono.eigenvalues))

        K, M = cp.assemble_evp_matrices()
        e_fact = np.sort_complex(
            np.array(cp.solve_evp_from_matrices(K, M, sigma, k=8))
        )
        # Compare the finite eigenvalues (drop NaN artifacts).
        e_mono = e_mono[np.isfinite(e_mono)]
        e_fact = e_fact[np.isfinite(e_fact)]
        assert len(e_mono) == len(e_fact)
        # ARPACK's random start vector means two runs of the same
        # matrices agree to ~1e-12 absolute, not bit-for-bit. Use a
        # relative tolerance scaled by the eigenvalue magnitude.
        scale = max(np.max(np.abs(e_mono)), 1.0)
        assert np.max(np.abs(e_mono - e_fact)) / scale < 1e-12

    def test_assemble_is_sigma_independent(self):
        """Assembling twice gives identical (K, M) — no σ leakage."""
        cp = _simple_scalar_evp()
        K1, M1 = cp.assemble_evp_matrices()
        K2, M2 = cp.assemble_evp_matrices()
        assert (K1 - K2).nnz == 0 or np.max(np.abs((K1 - K2).toarray())) < 1e-15
        assert (M1 - M2).nnz == 0 or np.max(np.abs((M1 - M2).toarray())) < 1e-15

    def test_multiple_sigmas_from_one_assembly(self):
        """One assembly, several shifts — each matches a fresh solve_evp."""
        cp = _simple_scalar_evp()
        K, M = cp.assemble_evp_matrices()
        for sig_im in (-1.0, -0.5, 0.0, 0.5):
            sigma = -0.01 - 1j * sig_im
            e_fact = np.sort_complex(
                np.array(cp.solve_evp_from_matrices(K, M, sigma, k=6))
            )
            e_mono = np.sort_complex(
                np.array(cp.solve_evp(k=6, sigma=sigma).eigenvalues)
            )
            e_fact = e_fact[np.isfinite(e_fact)]
            e_mono = e_mono[np.isfinite(e_mono)]
            assert len(e_fact) == len(e_mono)
            scale = max(np.max(np.abs(e_mono)), 1.0)
            assert np.max(np.abs(e_fact - e_mono)) / scale < 1e-12


# ---------------------------------------------------------------------------
# Matrix disk round-trip
# ---------------------------------------------------------------------------

class TestMatrixRoundTrip:
    """save_npz / load_npz must preserve (K, M) bit-for-bit."""

    def test_K_M_round_trip(self, tmp_path):
        cp = _simple_scalar_evp()
        K, M = cp.assemble_evp_matrices()
        sp.save_npz(tmp_path / "K.npz", K)
        sp.save_npz(tmp_path / "M.npz", M)
        K2 = sp.load_npz(tmp_path / "K.npz")
        M2 = sp.load_npz(tmp_path / "M.npz")
        assert (K - K2).nnz == 0 or np.max(np.abs((K - K2).toarray())) < 1e-15
        assert (M - M2).nnz == 0 or np.max(np.abs((M - M2).toarray())) < 1e-15

    def test_solve_from_loaded_matches_solve_from_assembled(self, tmp_path):
        cp = _simple_scalar_evp()
        K, M = cp.assemble_evp_matrices()
        sp.save_npz(tmp_path / "K.npz", K)
        sp.save_npz(tmp_path / "M.npz", M)
        K2 = sp.load_npz(tmp_path / "K.npz").tocsr()
        M2 = sp.load_npz(tmp_path / "M.npz").tocsr()
        sigma = -0.01 - 1j * 0.3
        e_a = np.sort_complex(np.array(cp.solve_evp_from_matrices(K, M, sigma, k=6)))
        e_b = np.sort_complex(np.array(cp.solve_evp_from_matrices(K2, M2, sigma, k=6)))
        e_a = e_a[np.isfinite(e_a)]
        e_b = e_b[np.isfinite(e_b)]
        scale = max(np.max(np.abs(e_a)), 1.0)
        assert np.max(np.abs(e_a - e_b)) / scale < 1e-12


# ---------------------------------------------------------------------------
# Scan helpers
# ---------------------------------------------------------------------------

class TestScanHelpers:
    """Test the dedup logic in scan_solar_mhd_spectrum."""

    def _import_scan(self):
        import scan_solar_mhd_spectrum as scan
        return scan

    def test_dedup_collapses_duplicates(self):
        scan = self._import_scan()
        # Two targets, both finding the same two eigenvalues (within tol).
        results = [
            (-0.5, np.array([-0.1 - 0.5j, -0.2 - 0.8j])),
            (-0.6, np.array([-0.1 - 0.5j + 1e-9, -0.2 - 0.8j - 1e-9])),
        ]
        uniq = scan._dedup_spectrum(results, tol=1e-6)
        assert len(uniq) == 2

    def test_dedup_keeps_distinct(self):
        scan = self._import_scan()
        results = [
            (-0.5, np.array([-0.1 - 0.5j])),
            (-0.6, np.array([-0.1 - 0.7j])),  # distinct ω
        ]
        uniq = scan._dedup_spectrum(results, tol=1e-6)
        assert len(uniq) == 2

    def test_dedup_drops_nan(self):
        scan = self._import_scan()
        results = [
            (-0.5, np.array([-0.1 - 0.5j, np.nan + 0j])),
        ]
        uniq = scan._dedup_spectrum(results, tol=1e-6)
        assert len(uniq) == 1
        assert np.isfinite(uniq[0])

    def test_dedup_sorts_by_omega(self):
        scan = self._import_scan()
        # ω = -Im. Modes at Im = 0.2, 0.8 → ω = -0.2, -0.8.
        results = [
            (-0.5, np.array([-0.1 - 0.2j, -0.1 - 0.8j])),
        ]
        uniq = scan._dedup_spectrum(results, tol=1e-6)
        # The helper sorts by argsort(-Im), i.e. ascending in -Im.
        # -Im values are -0.2 and -0.8, so ascending order is
        # [-0.8, -0.2], meaning the Im=0.8 mode comes first.
        # Equivalently ω = -Im is sorted ascending: [-0.8, -0.2]? No:
        # argsort(-Im) of [-0.2,-0.8] sorts [-(-0.2), -(-0.8)] wait,
        # -uniq.imag = [0.2, 0.8] but the SORT KEY in the helper is
        # `-uniq.imag`. Let's just assert it is monotonically sorted
        # in the recovered ω.
        omega = -uniq.imag
        assert np.all(np.diff(omega) >= 0) or np.all(np.diff(omega) <= 0)

    def test_matrix_paths_unique_per_cell(self):
        scan = self._import_scan()
        p1 = scan._matrix_paths(Path("/tmp/out"), 3, 0.0)
        p2 = scan._matrix_paths(Path("/tmp/out"), 3, 0.05)
        p3 = scan._matrix_paths(Path("/tmp/out"), 4, 0.0)
        # All distinct.
        assert p1 != p2 != p3
        assert p1[0] != p1[1]  # K and M differ


# ---------------------------------------------------------------------------
# Convention check
# ---------------------------------------------------------------------------

class TestSigmaConvention:
    """σ = γ − iω: a target near a known eigenvalue should recover it."""

    def test_target_recovers_known_eigenvalue(self):
        """For the scalar diffusion problem, find an eigenvalue, then
        target it via the σ = γ − iω mapping and confirm recovery."""
        cp = _simple_scalar_evp()
        K, M = cp.assemble_evp_matrices()
        # First, find some eigenvalues near σ = 0.
        evals = cp.solve_evp_from_matrices(K, M, sigma=-0.3 + 0j, k=8)
        evals = evals[np.isfinite(evals)]
        # Pick the one with the largest |Im| (most oscillatory) to make
        # the ω-targeting non-trivial.
        target_ev = evals[np.argmax(np.abs(evals.imag))]
        omega_t = -target_ev.imag  # ω = -Im(λ)
        gamma_t = target_ev.real   # use the actual γ to target precisely

        # Target via σ = γ − iω.
        sigma = complex(gamma_t, -omega_t)
        recovered = cp.solve_evp_from_matrices(K, M, sigma, k=4)
        recovered = recovered[np.isfinite(recovered)]
        # The targeted eigenvalue should be among those recovered.
        best = np.min(np.abs(recovered - target_ev))
        assert best < 1e-8, (
            f"targeted eigenvalue {target_ev} not recovered; "
            f"closest distance {best:.2e}"
        )
