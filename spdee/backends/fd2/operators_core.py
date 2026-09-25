"""spdee.backends.fd2.operators_core — 2D lifts of 1D operators.

In the r-major flat layout (r varies fastest), a 1D operator acting
along r becomes a Kronecker product with the identity on θ, and a 1D
operator acting along θ becomes the reverse product:

    op_2d_r     = kron(I_{N_theta}, op_r)
    op_2d_theta = kron(op_theta, I_{N_r})

Pointwise multipliers evaluated on the grid become sparse diagonals in
the same flat ordering.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from spdee.backends.fd2.layout import flatten_2d


def lift_r_op(op_r: sp.spmatrix, N_theta: int) -> sp.csr_matrix:
    """Lift a (N_r, N_r) radial operator to (N_r N_theta, N_r N_theta)."""
    return sp.kron(sp.identity(N_theta), op_r, format="csr")


def lift_theta_op(op_theta: sp.spmatrix, N_r: int) -> sp.csr_matrix:
    """Lift a (N_theta, N_theta) angular operator to the flat 2D state."""
    return sp.kron(op_theta, sp.identity(N_r), format="csr")


def diag_from_2d_array(arr: np.ndarray) -> sp.dia_matrix:
    """Sparse diagonal of a (N_r, N_theta) array in r-major flat order."""
    return sp.diags(flatten_2d(arr), format="csr")
