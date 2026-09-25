"""spdee.backends.fd2.staggered_io — interpolation between the collocated
grid and Arakawa-C (Bekki-style) staggered grids.

The collocated grid stores every unknown at the same `(r, θ)` nodes:
``r`` includes both endpoints (`N_r` nodes) and ``θ`` is cell-centered
at `(j+½)Δθ` (`N_θ` nodes, poles excluded).

The staggered layout on the same domain with ``N_r_stag`` radial cells
and the same ``N_θ`` angular cells is::

    radial:
        cell centers:  r_c[i] = r_inner + (i + ½) Δr,  i ∈ [0, N_r_stag)
        r-faces:       r_f[i] = r_inner + i Δr,        i ∈ [0, N_r_stag]

    angular:
        cell centers:  θ_c[j] = (j + ½) Δθ,           j ∈ [0, N_θ)
                       (the collocated θ grid)
        θ-faces:       θ_f[j] = j Δθ,                  j ∈ [0, N_θ]
                       (includes both poles)

    fields:
        ρ_1, s_1, v_φ  at (r_c, θ_c)
        v_r            at (r_f, θ_c)
        v_θ            at (r_c, θ_f)

With ``N_r_stag = N_r − 1`` (the default) the r-faces coincide with
the collocated r nodes. The θ-faces include the poles, which lie
outside the collocated θ range; ``v_θ`` is set to ``0`` there rather
than extrapolated.

Both directions are provided: `to_staggered` for comparing a
collocated eigenmode with staggered-grid snapshots, and
`from_staggered` for the inverse. Interpolation is local
Lagrange of even order; ``order=4`` (5-point stencil) matches the
default FD order.
"""
from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from spdee.backends.fd2.grid import Grid
from spdee.backends.fd2.stencils import _lagrange_derivative_weights


def _local_stencil(source_nodes: np.ndarray, target: float, width: int):
    """Indices of `width` source nodes around `target`.

    The stencil is centered on the gap containing the target (or on the
    node itself when the target coincides with one) and shifted inward
    at the ends of the grid. `width` is the polynomial order + 1.
    """
    N = len(source_nodes)
    if width > N:
        raise ValueError(
            f"stencil width {width} exceeds source grid size {N}"
        )
    i_left = int(np.searchsorted(source_nodes, target)) - 1
    i_left = max(0, min(i_left, N - 1))
    half_before = (width - 1) // 2
    start = i_left - half_before + 1
    end = start + width
    if start < 0:
        start = 0
        end = width
    if end > N:
        end = N
        start = N - width
    return np.arange(start, end)


def interpolate_1d(
    values: np.ndarray,
    source_nodes: np.ndarray,
    target_nodes: np.ndarray,
    *,
    order: int = 4,
) -> np.ndarray:
    """Interpolate `values` from `source_nodes` to `target_nodes` with a
    local Lagrange stencil of width ``order + 1``. Targets outside the
    source range are extrapolated with the one-sided end stencil.

    Parameters
    ----------
    values : array of shape (N_source, ...)
        Values at ``source_nodes``; trailing axes are carried along.
    source_nodes : 1D increasing array.
    target_nodes : 1D array.
    order : int, default 4
        Polynomial order.

    Returns
    -------
    np.ndarray of shape (N_target, ...).
    """
    source_nodes = np.asarray(source_nodes, dtype=float)
    target_nodes = np.asarray(target_nodes, dtype=float)
    values = np.asarray(values)
    if values.shape[0] != len(source_nodes):
        raise ValueError(
            f"values.shape[0]={values.shape[0]} must equal "
            f"len(source_nodes)={len(source_nodes)}"
        )
    width = order + 1
    out_shape = (len(target_nodes),) + values.shape[1:]
    out = np.empty(out_shape, dtype=values.dtype)
    for k, t in enumerate(target_nodes):
        idx = _local_stencil(source_nodes, float(t), width)
        w = _lagrange_derivative_weights(source_nodes[idx], float(t), order=0)
        out[k] = np.tensordot(w, values[idx], axes=(0, 0))
    return out


def interpolate_2d_in_r(
    values_2d: np.ndarray,
    source_r: np.ndarray,
    target_r: np.ndarray,
    *,
    order: int = 4,
) -> np.ndarray:
    """Interpolate an (N_source, N_θ) array along r to (N_target, N_θ)."""
    return interpolate_1d(values_2d, source_r, target_r, order=order)


def interpolate_2d_in_theta(
    values_2d: np.ndarray,
    source_theta: np.ndarray,
    target_theta: np.ndarray,
    *,
    order: int = 4,
    pole_value: float = 0.0,
) -> np.ndarray:
    """Interpolate an (N_r, N_source) array along θ to (N_r, N_target).

    Any target θ outside the source range (below source[0] or above
    source[-1]) is treated as a pole and filled with ``pole_value``
    instead of being extrapolated; for ``v_θ`` the pole value is 0.
    """
    source_theta = np.asarray(source_theta, dtype=float)
    target_theta = np.asarray(target_theta, dtype=float)
    values_2d = np.asarray(values_2d)
    if values_2d.shape[1] != len(source_theta):
        raise ValueError(
            f"values_2d.shape[1]={values_2d.shape[1]} must equal "
            f"len(source_theta)={len(source_theta)}"
        )
    tposed = values_2d.T
    width = order + 1
    out_shape = (len(target_theta), values_2d.shape[0])
    out = np.empty(out_shape, dtype=values_2d.dtype)
    source_lo = float(source_theta[0])
    source_hi = float(source_theta[-1])
    for k, t in enumerate(target_theta):
        if t < source_lo or t > source_hi:
            out[k] = pole_value
            continue
        idx = _local_stencil(source_theta, float(t), width)
        w = _lagrange_derivative_weights(
            source_theta[idx], float(t), order=0,
        )
        out[k] = np.tensordot(w, tposed[idx], axes=(0, 0))
    return out.T


def make_staggered_grids(
    grid: Grid,
    *,
    N_r_stag: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Staggered (r, θ) auxiliary grids matching a collocated grid.

    Parameters
    ----------
    grid : Grid
        Collocated grid with inclusive r endpoints and cell-centered θ.
    N_r_stag : int, optional
        Radial cell count; default ``grid.N_r - 1`` so the r-faces
        coincide with the collocated r nodes.

    Returns
    -------
    dict with keys ``r_center`` (N_r_stag,), ``r_face`` (N_r_stag + 1,),
    ``theta_center`` (N_θ,), ``theta_face`` (N_θ + 1,), ``N_r_stag``,
    ``N_theta``.
    """
    if N_r_stag is None:
        N_r_stag = grid.N_r - 1
    if N_r_stag <= 0:
        raise ValueError(f"N_r_stag must be > 0; got {N_r_stag}")
    r_in = float(grid.r[0])
    r_out = float(grid.r[-1])
    dr_b = (r_out - r_in) / N_r_stag
    r_face = r_in + dr_b * np.arange(N_r_stag + 1)
    r_center = r_in + dr_b * (np.arange(N_r_stag) + 0.5)
    N_theta = grid.N_theta
    dtheta = np.pi / N_theta
    theta_center = grid.theta.copy()
    theta_face = dtheta * np.arange(N_theta + 1)
    return {
        "r_center": r_center,
        "r_face": r_face,
        "theta_center": theta_center,
        "theta_face": theta_face,
        "N_r_stag": N_r_stag,
        "N_theta": N_theta,
    }


def to_staggered(
    solution: Dict[str, np.ndarray],
    grid: Grid,
    *,
    N_r_stag: Optional[int] = None,
    order: int = 4,
) -> Dict[str, np.ndarray]:
    """Convert a collocated solution dict to staggered fields.

    Parameters
    ----------
    solution : dict
        Field name → ``(N_r, N_θ)`` array for scalars, or
        ``{-1: arr, 0: arr, +1: arr}`` of spin components for vectors
        (``v^- = (v_θ - i v_φ)/√2``, ``v^0 = v_r``,
        ``v^+ = (v_θ + i v_φ)/√2``).
    grid : Grid
        Collocated grid of the solution.
    N_r_stag : int, optional
        Radial cell count of the target grid; default ``grid.N_r - 1``.
    order : int, default 4
        Lagrange interpolation order.

    Returns
    -------
    dict of staggered fields: scalars → ``(N_r_stag, N_θ)`` at (r_c, θ_c);
    for each vector ``name``, ``name_r`` → ``(N_r_stag + 1, N_θ)`` at
    (r_f, θ_c), ``name_theta`` → ``(N_r_stag, N_θ + 1)`` at (r_c, θ_f),
    ``name_phi`` → ``(N_r_stag, N_θ)`` at (r_c, θ_c).

    Vectors are first converted to physical ``(v_r, v_θ, v_φ)`` on the
    collocated grid, then each component is interpolated to its
    staggered position.
    """
    stag_grids = make_staggered_grids(grid, N_r_stag=N_r_stag)
    r_src = grid.r
    th_src = grid.theta
    r_c_tgt = stag_grids["r_center"]
    r_f_tgt = stag_grids["r_face"]
    th_f_tgt = stag_grids["theta_face"]
    out: Dict[str, np.ndarray] = {}
    for name, val in solution.items():
        if isinstance(val, dict):
            v_m1 = val[-1]
            v_0 = val[0]
            v_p1 = val[+1]
            # v_r = v^0, v_θ = (v^+ + v^-)/√2, v_φ = i (v^- − v^+)/√2
            v_r_coloc = v_0
            v_th_coloc = (v_p1 + v_m1) / np.sqrt(2.0)
            v_phi_coloc = 1j * (v_m1 - v_p1) / np.sqrt(2.0)
            v_r_bekki = interpolate_2d_in_r(
                v_r_coloc, r_src, r_f_tgt, order=order,
            )
            v_th_at_rc = interpolate_2d_in_r(
                v_th_coloc, r_src, r_c_tgt, order=order,
            )
            v_th_bekki = interpolate_2d_in_theta(
                v_th_at_rc, th_src, th_f_tgt, order=order,
                pole_value=0.0,
            )
            v_phi_bekki = interpolate_2d_in_r(
                v_phi_coloc, r_src, r_c_tgt, order=order,
            )
            out[f"{name}_r"] = v_r_bekki
            out[f"{name}_theta"] = v_th_bekki
            out[f"{name}_phi"] = v_phi_bekki
        else:
            out[name] = interpolate_2d_in_r(
                val, r_src, r_c_tgt, order=order,
            )
    return out


def from_staggered(
    stag_solution: Dict[str, np.ndarray],
    target_grid: Grid,
    *,
    order: int = 4,
) -> Dict[str, np.ndarray]:
    """Inverse of :func:`to_staggered`: staggered fields to
    collocated fields on ``target_grid``.

    Parameters
    ----------
    stag_solution : dict
        In the output format of :func:`to_staggered`: vectors as
        the three keys ``name_r``, ``name_theta``, ``name_phi``; scalars
        under their own name. A key ending in ``_r`` whose ``_theta``
        and ``_phi`` partners are present is taken as a vector.
    target_grid : Grid
        Collocated grid to interpolate onto. The source staggered grid
        is rebuilt from the array shapes and this grid's radii.
    order : int, default 4

    Returns
    -------
    dict: scalars → ``(N_r, N_θ)``; vectors → ``{-1, 0, +1}`` spin
    component dicts of ``(N_r, N_θ)`` arrays.
    """
    vector_names = set()
    for key in stag_solution.keys():
        if key.endswith("_r"):
            stem = key[:-2]
            if (f"{stem}_theta" in stag_solution
                    and f"{stem}_phi" in stag_solution):
                vector_names.add(stem)
    scalar_keys = [
        k for k in stag_solution.keys()
        if not (
            (k.endswith("_r") and k[:-2] in vector_names)
            or (k.endswith("_theta") and k[:-6] in vector_names)
            or (k.endswith("_phi") and k[:-4] in vector_names)
        )
    ]
    # Recover N_r_stag from any field stored at (r_c, θ_c).
    sample_arr = None
    if scalar_keys:
        sample_arr = stag_solution[scalar_keys[0]]
    else:
        for stem in vector_names:
            sample_arr = stag_solution[f"{stem}_phi"]
            break
    if sample_arr is None:
        return {}
    N_r_stag = sample_arr.shape[0]
    stag_grids = make_staggered_grids(
        target_grid, N_r_stag=N_r_stag,
    )
    r_c_src = stag_grids["r_center"]
    r_f_src = stag_grids["r_face"]
    th_f_src = stag_grids["theta_face"]
    r_tgt = target_grid.r
    th_tgt = target_grid.theta
    out: Dict[str, np.ndarray] = {}
    for k in scalar_keys:
        out[k] = interpolate_2d_in_r(
            stag_solution[k], r_c_src, r_tgt, order=order,
        )
    for stem in vector_names:
        v_r_b = stag_solution[f"{stem}_r"]
        v_th_b = stag_solution[f"{stem}_theta"]
        v_phi_b = stag_solution[f"{stem}_phi"]
        v_r_coloc = interpolate_2d_in_r(
            v_r_b, r_f_src, r_tgt, order=order,
        )
        # θ first (drops the poles), then r.
        v_th_at_th_tgt = interpolate_2d_in_theta(
            v_th_b, th_f_src, th_tgt, order=order, pole_value=0.0,
        )
        v_th_coloc = interpolate_2d_in_r(
            v_th_at_th_tgt, r_c_src, r_tgt, order=order,
        )
        v_phi_coloc = interpolate_2d_in_r(
            v_phi_b, r_c_src, r_tgt, order=order,
        )
        # v^0 = v_r, v^∓ = (v_θ ∓ i v_φ) / √2
        out[stem] = {
            0: v_r_coloc,
            -1: (v_th_coloc - 1j * v_phi_coloc) / np.sqrt(2.0),
            +1: (v_th_coloc + 1j * v_phi_coloc) / np.sqrt(2.0),
        }
    return out
