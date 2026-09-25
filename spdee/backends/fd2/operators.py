"""spdee.backends.fd2.operators — derivative bundle and equation mass
matrices.

`DerivativeBundle` caches the 1D first-derivative stencils of a grid at
user-selectable even orders (`order_r`, `order_theta`, default 4):
  - `A_r`, `B_r`: radial pair with A_r = I and B_r the explicit D1.
  - `A_theta`, `B_theta`: the same for the angular direction.
  - `d1_theta_dense`: dense N_θ × N_θ angular D1, built lazily for
    tensor-derived boundary conditions.

The equation mass matrices multiply the left-hand side of an equation
by the power of r that clears the 1/r factors of its operators:
  - `r1_vector_mass`, `r2_vector_mass`, `r_sqrt2_vector_mass`:
    (3N, 3N) block-diagonal diag(r), diag(r²), diag(r·√2).
  - `r1_scalar_mass`: (N, N) diag(r).
On ball grids the i_r=0 rows of each block are identity placeholders
(so the mass is invertible); `CompiledProblem` replaces those rows with
origin-regularity constraints.

Spin basis (V19):
    e_+ = (e_theta - i e_phi) / sqrt(2)
    e_0 = e_r
    e_- = (e_theta + i e_phi) / sqrt(2)
Components (v^sigma = e_{-sigma} · v):
    v^+ = (v_theta + i v_phi) / sqrt(2)
    v^0 = v_r
    v^- = (v_theta - i v_phi) / sqrt(2)

Vector layout: three (N_r × N_θ) blocks in spin order (-1, 0, +1),
each r-major flattened.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.backends.fd2.grid import Grid
from spdee.backends.fd2.operators_core import diag_from_2d_array
from spdee.backends.fd2.stencils import standard_d1, _check_order


class DerivativeBundle:
    """Cache of 1D radial and angular first-derivative stencils on a Grid.

    Both pairs have A = I and B the explicit derivative matrix. The
    radial and angular orders are independent.

    Parameters
    ----------
    grid : Grid
    order_r : int, default 4
        Even formal order ≥ 2 for the radial stencils.
    order_theta : int, default 4
        Even formal order ≥ 2 for the angular stencils.
    """

    def __init__(self, grid: Grid, order_r: int = 4, order_theta: int = 4):
        self.grid = grid
        self.order_r = _check_order(order_r, "order_r")
        self.order_theta = _check_order(order_theta, "order_theta")
        # The d2 boundary closure needs order + 2 nodes; requiring that
        # here lets every second-order builder assemble without a
        # mid-compile error.
        if grid.N_r < self.order_r + 2:
            raise ValueError(
                f"Grid N_r={grid.N_r} is too small for order_r={self.order_r}; "
                f"need N_r >= {self.order_r + 2}."
            )
        if grid.N_theta < self.order_theta + 2:
            raise ValueError(
                f"Grid N_theta={grid.N_theta} is too small for "
                f"order_theta={self.order_theta}; "
                f"need N_theta >= {self.order_theta + 2}."
            )
        self.A_r, self.B_r = standard_d1(grid.r, order=self.order_r)
        self.A_theta, self.B_theta = standard_d1(grid.theta, order=self.order_theta)
        self._d1_theta_dense = None

    @property
    def d1_theta_dense(self) -> np.ndarray:
        """Dense N_θ × N_θ matrix D1_θ = A_θ⁻¹ B_θ, cached lazily.

        Row j holds the ∂_θ weights at θ_j across the θ grid; the
        tensor-derived boundary conditions read rows from it.
        """
        if self._d1_theta_dense is None:
            self._d1_theta_dense = spla.spsolve(
                self.A_theta.tocsc(), self.B_theta.toarray()
            )
        return self._d1_theta_dense

    @property
    def N_r(self) -> int:
        return self.grid.N_r

    @property
    def N_theta(self) -> int:
        return self.grid.N_theta


def r2_vector_mass(bundle: "DerivativeBundle") -> sp.csr_matrix:
    """(3N, 3N) vector mass diag(r²) on each spin block, for the r²
    equation forms (Div(tensor), r²·Lap(v), r²·Grad(Div v)).

    On ball grids the i_r=0 row of each block is an identity
    placeholder. Cached on the bundle.
    """
    if hasattr(bundle, "_r2_vector_mass"):
        return bundle._r2_vector_mass
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    diag_r2 = diag_from_2d_array(R ** 2)
    diag_r2_lil = sp.lil_matrix(diag_r2, dtype=complex)
    A_op = sp.block_diag([diag_r2_lil.copy() for _ in range(3)],
                         format="lil", dtype=complex)
    if grid.is_ball and grid.r[0] == 0.0:
        for block_idx in range(3):
            block_offset = block_idx * block
            for j in range(N_theta):
                row = block_offset + j * N_r + 0
                A_op.rows[row] = [row]
                A_op.data[row] = [1.0]
    bundle._r2_vector_mass = A_op.tocsr()
    return bundle._r2_vector_mass


def r1_vector_mass(bundle: "DerivativeBundle") -> sp.csr_matrix:
    """(3N, 3N) vector mass diag(r) on each spin block, for the r
    equation forms (Curl, Cross, advection, Dot(bg_vec, tensor)).

    On ball grids the i_r=0 row of each block is an identity
    placeholder. Cached on the bundle.
    """
    if hasattr(bundle, "_r1_vector_mass"):
        return bundle._r1_vector_mass
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    diag_r = diag_from_2d_array(R)
    diag_r_lil = sp.lil_matrix(diag_r, dtype=complex)
    A_op = sp.block_diag([diag_r_lil.copy() for _ in range(3)],
                         format="lil", dtype=complex)
    if grid.is_ball and grid.r[0] == 0.0:
        for block_idx in range(3):
            block_offset = block_idx * block
            for j in range(N_theta):
                row = block_offset + j * N_r + 0
                A_op.rows[row] = [row]
                A_op.data[row] = [1.0]
    bundle._r1_vector_mass = A_op.tocsr()
    return bundle._r1_vector_mass


def r_sqrt2_vector_mass(bundle: "DerivativeBundle") -> sp.csr_matrix:
    """(3N, 3N) vector mass diag(r·√2) on each spin block, for the
    r·√2 form of Grad(scalar): the σ=±1 components of Grad carry a
    1/(r·√2) prefactor, which this mass clears.

    On ball grids the i_r=0 row of each block is an identity
    placeholder. Cached on the bundle.
    """
    if hasattr(bundle, "_r_sqrt2_vector_mass"):
        return bundle._r_sqrt2_vector_mass
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    diag_rs = diag_from_2d_array(np.sqrt(2.0) * R)
    diag_rs_lil = sp.lil_matrix(diag_rs, dtype=complex)
    A_op = sp.block_diag([diag_rs_lil.copy() for _ in range(3)],
                         format="lil", dtype=complex)
    if grid.is_ball and grid.r[0] == 0.0:
        for block_idx in range(3):
            block_offset = block_idx * block
            for j in range(N_theta):
                row = block_offset + j * N_r + 0
                A_op.rows[row] = [row]
                A_op.data[row] = [1.0]
    bundle._r_sqrt2_vector_mass = A_op.tocsr()
    return bundle._r_sqrt2_vector_mass


def r1_scalar_mass(bundle: "DerivativeBundle") -> sp.csr_matrix:
    """(N, N) scalar mass diag(r), for the r form of Div(vector).

    On ball grids the i_r=0 rows are identity placeholders; if no other
    equation constrains the scalar there, the row reduces to
    `p(0, θ) = 0` for the scalar unknown in that row block. Cached on
    the bundle.
    """
    if hasattr(bundle, "_r1_scalar_mass"):
        return bundle._r1_scalar_mass
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    diag_r = diag_from_2d_array(R)
    A_op = sp.lil_matrix(diag_r, dtype=complex)
    if grid.is_ball and grid.r[0] == 0.0:
        for j in range(N_theta):
            row = j * N_r + 0
            A_op.rows[row] = [row]
            A_op.data[row] = [1.0]
    bundle._r1_scalar_mass = A_op.tocsr()
    return bundle._r1_scalar_mass
