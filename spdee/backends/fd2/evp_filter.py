"""Twin-grid spurious-mode detection for generalized eigenvalue problems.

A discretized EVP yields two kinds of finite eigenvalues: physical
modes, which converge to fixed values under grid refinement, and
spurious modes (LBB / inf-sup artifacts on collocated grids, pressure
checkerboards, ...), which drift by O(1) when the grid changes.

The filter solves the EVP twice, at (N_r, N_θ) and at (N_r + dN,
N_θ + dN), and keeps the eigenvalues that appear in both spectra to
within a tolerance. Physical modes move by O(h^p) between the two
grids, spurious ones by O(1). The fine grid asks for more eigenvalues
than the base grid so its window around σ contains the base window
(the k nearest eigenvalues of the two grids are not the same set at
the edge of the window).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.ir.equations import EquationSystem


def filter_spurious_eigenvalues(
    system: EquationSystem,
    grid_factory: Callable[[int, int], object],
    N_r_base: int,
    N_theta_base: int,
    m: int,
    k: int,
    sigma: complex,
    mass_equations: Optional[Union[Dict[str, str], List[str]]] = None,
    order_r: int = 4,
    order_theta: int = 4,
    dN: int = 1,
    rel_tolerance: float = 1e-2,
    abs_tolerance: Optional[float] = None,
    k_fine: Optional[int] = None,
    return_eigenvectors: bool = False,
):
    """Separate physical from spurious eigenvalues by twin-grid comparison.

    Parameters
    ----------
    system : EquationSystem
        IR system compiled on both grids.
    grid_factory : callable
        `grid_factory(N_r, N_theta) -> Grid`; carries the geometry and
        radii, e.g. `lambda nr, nt: make_ball_grid(nr, nt, r_outer=1.0)`.
    N_r_base, N_theta_base : int
        Base resolution; the probe grid is (N_r_base+dN, N_theta_base+dN).
    m : int
        Azimuthal mode number.
    k : int
        Number of eigenvalues requested per grid (`CompiledProblem.solve_evp`).
    sigma : complex
        Shift-invert target, shared by both solves.
    mass_equations : dict, list, or None
        Passed through to `CompiledProblem.solve_evp`.
    order_r, order_theta : int
        FD orders for both grids.
    dN : int, default 1
        Resolution increment of the probe grid.
    rel_tolerance : float, default 1e-2
        A base eigenvalue λ_b is physical when some fine eigenvalue
        satisfies ``|λ_f − λ_b| < rel_tolerance·|λ_b| + abs_tolerance``.
    abs_tolerance : float, optional
        Absolute part of the tolerance; defaults to
        ``rel_tolerance·|sigma|`` so that eigenvalues near λ = 0 are
        judged on the scale of the shift instead of on their own
        (vanishing) magnitude.
    k_fine : int, optional
        Eigenvalues requested on the fine grid; defaults to
        ``k + max(2, k // 2)`` so the fine window covers the base one.
    return_eigenvectors : bool, default False
        Also return the base-grid eigenvectors of the physical modes.

    Returns
    -------
    (physical_evals, spurious_base, spurious_fine), or with
    `return_eigenvectors` (physical_evals, physical_evecs, spurious_base,
    spurious_fine). `physical_evals` is sorted by ascending real part.
    NaN eigenvalues (deflated infinite modes) are dropped before matching.

    Matching is greedy and not bijective: each base eigenvalue only needs
    some fine partner within tolerance, so closely spaced base modes may
    share one fine partner. `spurious_fine` lists the fine eigenvalues
    that are not the nearest partner of any physical mode.
    """
    if abs_tolerance is None:
        abs_tolerance = rel_tolerance * abs(sigma)
    if k_fine is None:
        k_fine = k + max(2, k // 2)
    grid_base = grid_factory(N_r_base, N_theta_base)
    cp_base = compile_problem(
        system, grid_base, m=m,
        order_r=order_r, order_theta=order_theta,
    )
    res_base = cp_base.solve_evp(
        k=k, sigma=sigma, mass_equations=mass_equations,
        return_eigenvectors=return_eigenvectors,
    )
    evals_base = np.asarray(res_base.eigenvalues)

    grid_fine = grid_factory(N_r_base + dN, N_theta_base + dN)
    cp_fine = compile_problem(
        system, grid_fine, m=m,
        order_r=order_r, order_theta=order_theta,
    )
    res_fine = cp_fine.solve_evp(
        k=k_fine, sigma=sigma, mass_equations=mass_equations,
        return_eigenvectors=False,
    )
    evals_fine = np.asarray(res_fine.eigenvalues)

    valid_b = np.isfinite(evals_base)
    valid_f = np.isfinite(evals_fine)
    base_clean = evals_base[valid_b]
    fine_clean = evals_fine[valid_f]
    if return_eigenvectors:
        evecs_base_clean = [
            res_base.eigenvectors[i] for i in np.where(valid_b)[0]
        ]

    # Base eigenvalue is physical when some fine eigenvalue lies within
    # the tolerance.
    def tol(lam):
        return rel_tolerance * abs(lam) + abs_tolerance

    physical_mask = np.zeros(len(base_clean), dtype=bool)
    if len(fine_clean) > 0:
        for i, lam_b in enumerate(base_clean):
            if np.abs(fine_clean - lam_b).min() < tol(lam_b):
                physical_mask[i] = True
    physical_evals = base_clean[physical_mask]
    spurious_base = base_clean[~physical_mask]

    # Fine eigenvalues that are the nearest partner of no physical mode.
    fine_matched = np.zeros(len(fine_clean), dtype=bool)
    if len(physical_evals) > 0:
        for lam_p in physical_evals:
            dist = np.abs(fine_clean - lam_p)
            nearest = int(np.argmin(dist))
            if dist[nearest] < tol(lam_p):
                fine_matched[nearest] = True
    spurious_fine = fine_clean[~fine_matched]

    order = np.argsort(physical_evals.real)
    physical_evals = physical_evals[order]
    if return_eigenvectors:
        keep_idx = np.where(physical_mask)[0]
        physical_evecs = [evecs_base_clean[keep_idx[i]] for i in order]
        return (physical_evals, physical_evecs,
                spurious_base, spurious_fine)
    return physical_evals, spurious_base, spurious_fine
