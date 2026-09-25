"""spdee.backends.fd2.layout — flat state-vector layout of a single field.

A 2D field on an (N_r, N_theta) grid is stored r-major, meaning r is
the fastest-varying index so that radial neighbours are adjacent:

    flat[i + j * N_r] = value at (r_i, theta_j)

This is the Fortran-order flatten of a (N_r, N_theta) array. A radial
operator D_r acts on each contiguous slice flat[j*N_r : (j+1)*N_r], so
its 2D form is kron(I_{N_theta}, D_r); an angular operator D_theta has
stride N_r and its 2D form is kron(D_theta, I_{N_r}).

Vectors carry three spin components stored block-by-component in the
order SPIN_COMPONENTS_VECTOR = (-1, 0, +1), total length
3 * N_r * N_theta. Rank-2 tensors carry nine components (s1, s2) in
the lexicographic order SPIN_COMPONENTS_TENSOR, total length
9 * N_r * N_theta.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

SPIN_COMPONENTS_VECTOR: Tuple[int, ...] = (-1, 0, +1)
SPIN_COMPONENTS_TENSOR: Tuple[Tuple[int, int], ...] = tuple(
    (s1, s2) for s1 in SPIN_COMPONENTS_VECTOR for s2 in SPIN_COMPONENTS_VECTOR
)
COMPONENTS_PER_KIND = {
    "scalar": 1,
    "vector": 3,
    "tensor": 9,
}


def state_size(kind: str, N_r: int, N_theta: int) -> int:
    """Total length of the flat state vector for a single field."""
    return COMPONENTS_PER_KIND[kind] * N_r * N_theta


def flatten_2d(arr: np.ndarray) -> np.ndarray:
    """Flatten a (N_r, N_theta) array r-major (Fortran order)."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array; got shape {arr.shape}")
    return np.asfortranarray(arr).ravel(order="F")


def reshape_2d(flat: np.ndarray, N_r: int, N_theta: int) -> np.ndarray:
    """Inverse of flatten_2d; returns a (N_r, N_theta) array."""
    expected = N_r * N_theta
    if flat.size != expected:
        raise ValueError(
            f"flat array has {flat.size} elements; expected {expected} "
            f"(N_r={N_r}, N_theta={N_theta})"
        )
    return np.asarray(flat).reshape((N_r, N_theta), order="F")


def vector_component_slice(
    component_sigma: int, N_r: int, N_theta: int
) -> slice:
    """Slice of one spin component within a 3 * N_r * N_theta vector state."""
    block = N_r * N_theta
    if component_sigma not in SPIN_COMPONENTS_VECTOR:
        raise ValueError(
            f"sigma must be in {SPIN_COMPONENTS_VECTOR}; got {component_sigma}"
        )
    idx = SPIN_COMPONENTS_VECTOR.index(component_sigma)
    return slice(idx * block, (idx + 1) * block)


def tensor_component_slice(
    component_sigmas: Tuple[int, int], N_r: int, N_theta: int
) -> slice:
    """Slice of one (s1, s2) component within a 9 * N_r * N_theta tensor state."""
    block = N_r * N_theta
    if component_sigmas not in SPIN_COMPONENTS_TENSOR:
        raise ValueError(
            f"sigma pair must be in {SPIN_COMPONENTS_TENSOR}; got "
            f"{component_sigmas}"
        )
    idx = SPIN_COMPONENTS_TENSOR.index(component_sigmas)
    return slice(idx * block, (idx + 1) * block)


def split_vector_state(
    state: np.ndarray, N_r: int, N_theta: int
) -> Dict[int, np.ndarray]:
    """Split a vector state into {sigma: (N_r, N_theta) array}."""
    block = N_r * N_theta
    if state.size != 3 * block:
        raise ValueError(
            f"vector state has {state.size} elements; expected {3 * block}"
        )
    out = {}
    for idx, sigma in enumerate(SPIN_COMPONENTS_VECTOR):
        flat = state[idx * block:(idx + 1) * block]
        out[sigma] = reshape_2d(flat, N_r, N_theta)
    return out


def assemble_vector_state(
    components: Dict[int, np.ndarray], N_r: int, N_theta: int
) -> np.ndarray:
    """Inverse of split_vector_state; returns a complex flat vector state."""
    if set(components.keys()) != set(SPIN_COMPONENTS_VECTOR):
        raise ValueError(
            f"components must have keys {SPIN_COMPONENTS_VECTOR}; got "
            f"{list(components.keys())}"
        )
    out = np.empty(3 * N_r * N_theta, dtype=complex)
    for idx, sigma in enumerate(SPIN_COMPONENTS_VECTOR):
        sl = slice(idx * N_r * N_theta, (idx + 1) * N_r * N_theta)
        out[sl] = flatten_2d(components[sigma])
    return out


def split_tensor_state(
    state: np.ndarray, N_r: int, N_theta: int
) -> Dict[Tuple[int, int], np.ndarray]:
    """Split a tensor state into {(s1, s2): (N_r, N_theta) array}."""
    block = N_r * N_theta
    if state.size != 9 * block:
        raise ValueError(
            f"tensor state has {state.size} elements; expected {9 * block}"
        )
    out = {}
    for idx, pair in enumerate(SPIN_COMPONENTS_TENSOR):
        flat = state[idx * block:(idx + 1) * block]
        out[pair] = reshape_2d(flat, N_r, N_theta)
    return out


def assemble_tensor_state(
    components: Dict[Tuple[int, int], np.ndarray], N_r: int, N_theta: int
) -> np.ndarray:
    """Inverse of split_tensor_state; returns a complex flat tensor state."""
    if set(components.keys()) != set(SPIN_COMPONENTS_TENSOR):
        raise ValueError(
            f"components must have keys {SPIN_COMPONENTS_TENSOR}; got "
            f"{list(components.keys())}"
        )
    out = np.empty(9 * N_r * N_theta, dtype=complex)
    for idx, pair in enumerate(SPIN_COMPONENTS_TENSOR):
        sl = slice(idx * N_r * N_theta, (idx + 1) * N_r * N_theta)
        out[sl] = flatten_2d(components[pair])
    return out
