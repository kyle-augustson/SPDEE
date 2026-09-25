"""spdee.backends.common.field_eval — evaluate background fields on grids.

A background field carries its components as BackgroundComponent instances
(see spdee.ir.backgrounds), each of which evaluates itself on an (R, Theta)
mesh. This module combines those evaluations into spin-basis grids for
vector and tensor fields. The physical-to-spin conversion is done numerically
on the grid, so it works for SymPy-backed, interpolator-backed, or any other
BackgroundComponent subclass.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from spdee.ir.fields import ScalarField, VectorField, TensorField


# ---------------------------------------------------------------------------
# Scalar evaluation
# ---------------------------------------------------------------------------

def evaluate_scalar_background_2d(
    field: ScalarField, R: np.ndarray, Theta: np.ndarray
) -> np.ndarray:
    """Evaluate a background ScalarField on a 2D (R, Theta) mesh.

    Parameters
    ----------
    field : ScalarField
        Must be a background; field.component is a BackgroundComponent.
    R, Theta : np.ndarray
        Same-shape 2D arrays (typically (N_r, N_theta) from Grid.mesh()).

    Returns
    -------
    np.ndarray of the same shape holding the complex-valued field values.
    """
    if not field.is_background:
        raise ValueError(
            f"evaluate_scalar_background_2d requires a background field; "
            f"{field.name!r} is an unknown"
        )
    return field.component.evaluate_on_grid(R, Theta)


# ---------------------------------------------------------------------------
# Vector evaluation
# ---------------------------------------------------------------------------

def evaluate_vector_background_spin_2d(
    field: VectorField, R: np.ndarray, Theta: np.ndarray
) -> Dict[int, np.ndarray]:
    """Evaluate a background VectorField and return its spin components on a
    2D (R, Theta) mesh.

    The physical components (v_r, v_theta, v_phi) are evaluated on the grid
    and combined into V19 spin components:

        v^- = (v_theta - i v_phi) / sqrt(2)
        v^0 = v_r
        v^+ = (v_theta + i v_phi) / sqrt(2)

    Returns
    -------
    dict {sigma: np.ndarray} for sigma in {-1, 0, +1}, each of shape R.shape.
    """
    if not field.is_background:
        raise ValueError(
            f"evaluate_vector_background_spin_2d requires a background field; "
            f"{field.name!r} is an unknown"
        )
    v_r = field.components["v_r"].evaluate_on_grid(R, Theta)
    v_th = field.components["v_theta"].evaluate_on_grid(R, Theta)
    v_ph = field.components["v_phi"].evaluate_on_grid(R, Theta)
    sqrt2 = np.sqrt(2.0)
    return {
        -1: (v_th - 1j * v_ph) / sqrt2,
         0: v_r,
        +1: (v_th + 1j * v_ph) / sqrt2,
    }


# ---------------------------------------------------------------------------
# Tensor evaluation
# ---------------------------------------------------------------------------

def evaluate_tensor_background_matvec_spin_2d(
    field: TensorField, R: np.ndarray, Theta: np.ndarray
) -> Dict[Tuple[int, int], np.ndarray]:
    """Evaluate a background TensorField in matvec spin form.

    Returns the matrix M^{σ_out, σ_in} such that for a contravariant spin
    vector v^σ_in, M @ v is the contravariant spin vector of
    w_physical = T_physical @ v_physical. This is the form for a pointwise
    `Dot(T, v)` with a background tensor T acting on an unknown vector v.

    M = U · T_phys · U†, where U is the rank-1 V19 spin transform (rows σ,
    columns physical (r, θ, φ)). `evaluate_tensor_background_spin_2d` instead
    returns T^{σ_1, σ_2} = U · T_phys · Uᵀ, the doubly contravariant form used
    when T enters a Div(T)-style operator as a 2-tensor.
    """
    if not field.is_background:
        raise ValueError(
            f"evaluate_tensor_background_matvec_spin_2d requires a "
            f"background field; {field.name!r} is an unknown"
        )
    phys_idx = ("r", "theta", "phi")
    phys_grid: Dict[Tuple[str, str], np.ndarray] = {}
    for i in phys_idx:
        for j in phys_idx:
            comp = field.components[(i, j)]
            phys_grid[(i, j)] = comp.evaluate_on_grid(R, Theta)
    s2 = 1.0 / np.sqrt(2.0)
    U = np.array([
        [0,  s2, -1j * s2],
        [1,  0,   0       ],
        [0,  s2,  1j * s2],
    ], dtype=complex)
    Udag = U.conj().T
    spin_labels = (-1, 0, +1)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    # M[σ_out, σ_in] = U[σ_out, i] T[i, j] Udag[j, σ_in]
    for k1, s_out in enumerate(spin_labels):
        for k2, s_in in enumerate(spin_labels):
            acc = np.zeros(R.shape, dtype=complex)
            for i, i_lab in enumerate(phys_idx):
                for j, j_lab in enumerate(phys_idx):
                    acc += U[k1, i] * Udag[j, k2] * phys_grid[(i_lab, j_lab)]
            out[(s_out, s_in)] = acc
    return out


def evaluate_tensor_background_spin_2d(
    field: TensorField, R: np.ndarray, Theta: np.ndarray
) -> Dict[Tuple[int, int], np.ndarray]:
    """Evaluate a background TensorField and return its 9 spin components on
    a 2D (R, Theta) mesh.

    The physical components T_{i, j} (i, j in {r, theta, phi}) are evaluated
    and converted with the rank-1 V19 unitary applied on both indices:

        T^{s1, s2} = sum_{i, j} U[s1, i] U[s2, j] T_{i, j}.
    """
    if not field.is_background:
        raise ValueError(
            f"evaluate_tensor_background_spin_2d requires a background field; "
            f"{field.name!r} is an unknown"
        )
    phys_idx = ("r", "theta", "phi")
    phys_grid: Dict[Tuple[str, str], np.ndarray] = {}
    for i in phys_idx:
        for j in phys_idx:
            comp = field.components[(i, j)]
            phys_grid[(i, j)] = comp.evaluate_on_grid(R, Theta)
    # U: rows σ ∈ (-1, 0, +1), columns physical (r, θ, φ).
    s2 = 1.0 / np.sqrt(2.0)
    U = np.array([
        [0,  s2, -1j * s2],
        [1,  0,   0     ],
        [0,  s2,  1j * s2],
    ], dtype=complex)
    spin_labels = (-1, 0, +1)
    out: Dict[Tuple[int, int], np.ndarray] = {}
    for k1, s1 in enumerate(spin_labels):
        for k2, s2_lab in enumerate(spin_labels):
            acc = np.zeros(R.shape, dtype=complex)
            for i, i_lab in enumerate(phys_idx):
                for j, j_lab in enumerate(phys_idx):
                    acc += U[k1, i] * U[k2, j] * phys_grid[(i_lab, j_lab)]
            out[(s1, s2_lab)] = acc
    return out
