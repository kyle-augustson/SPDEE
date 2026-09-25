"""spdee.backends.fd2.stencils — 1D finite-difference derivative matrices.

Explicit centered finite differences of even order p. Every public
builder returns a pair

    (A, B) = stencil(grid, ..., order=p)

with A = I and B the derivative matrix, so that f^(k) ≈ B @ f. First
and second derivatives are provided.

Order conventions
-----------------
`order` is the formal truncation order p of the interior stencil:
order=2 is the 3-point centered stencil, order=4 the 5-point stencil
(default), order=6 the 7-point stencil, and so on. A centered stencil
of order p spans p+1 nodes, so the p/2 rows nearest each boundary need
a one-sided closure. A one-sided stencil of N nodes is exact for
polynomials of degree ≤ N-1 and therefore has order N-k for the k-th
derivative, so the closures use p+k nodes to keep order p:

  - d1 boundary closure: p+1 nodes → order p
  - d2 boundary closure: p+2 nodes → order p

Weights are obtained by solving the Vandermonde moment system rather
than from coefficient tables, so any even order ≥ 2 works as long as
N is at least p+k.

Parity-folded variants
----------------------
`standard_d1_uniform_parity` and `standard_d2_uniform_parity` treat
the left endpoint as r=0 and close the rows nearest it with ghost
values f(r_{-k}) = parity · f(r_k):

  parity = +1:  f(-r) = +f(r)   (scalars, v_θ and v_φ of an m=0 vector)
  parity = -1:  f(-r) = -f(r)   (v_r of an m=0 vector)

The right endpoint keeps the one-sided closure. The collocated
operator pipeline does not call these; its ball-mode origin treatment
lives in `ball_origin.py` and `CompiledProblem`.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import scipy.sparse as sp


def _lagrange_derivative_weights(
    nodes: np.ndarray, x_eval: float, order: int = 1
) -> np.ndarray:
    """Weights w_j with f^(order)(x_eval) ≈ sum_j w_j f(nodes[j]).

    Solves the moment system

        sum_j w_j (nodes[j] - x_eval)^k = delta(k, order) * order!
        for k = 0, 1, ..., n-1,   n = len(nodes)

    in floating point, for any derivative order and any node spacing.
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    if order < 0 or order >= n:
        raise ValueError(
            f"derivative order {order} must be in [0, {n - 1}] for a "
            f"stencil of {n} nodes"
        )
    offs = nodes - x_eval
    V = np.array([[off ** k for off in offs] for k in range(n)], dtype=float)
    rhs = np.zeros(n)
    rhs[order] = float(math.factorial(order))
    return np.linalg.solve(V, rhs)


def _uniform_lagrange_weights_rational(
    integer_nodes, integer_x_eval, deriv_order: int, h: float,
):
    """Weights w_j with f^(deriv_order)(x_eval) ≈ sum_j w_j f(nodes[j])
    on a uniform grid of spacing h, for integer node offsets.

    The moment system is solved exactly in `fractions.Fraction`
    arithmetic in dimensionless form. Each weight is then converted as
    `(num · L/den) / (L · h^p)`, where L is the LCM of the rational
    weights' denominators in the row, so the floating-point rounding
    matches the canonical tabulated form `coef_int / (L · h^p)` (e.g.
    `[-25, 48, -36, 16, -3] / (12 h)`). That keeps the constant mode
    exactly in the kernel of d1, which matters for gauge-constrained
    solves.

    Parameters
    ----------
    integer_nodes : sequence of int
        Node positions in units of h.
    integer_x_eval : int
        Evaluation position in units of h.
    deriv_order : int
        Derivative order p.
    h : float
        Grid spacing.

    Returns
    -------
    weights : float array of length len(integer_nodes), including 1/h^p.
    """
    from fractions import Fraction
    nodes_int = [int(k) for k in integer_nodes]
    x0 = int(integer_x_eval)
    offs = [Fraction(k - x0, 1) for k in nodes_int]
    n = len(offs)
    if deriv_order < 0 or deriv_order >= n:
        raise ValueError(
            f"derivative order {deriv_order} must be in [0, {n - 1}] "
            f"for a stencil of {n} nodes"
        )
    V = [[off ** k for off in offs] for k in range(n)]
    rhs = [Fraction(0)] * n
    rhs[deriv_order] = Fraction(math.factorial(deriv_order))
    w_rat = _solve_rational(V, rhs)
    common_denom = 1
    for wj in w_rat:
        d = wj.denominator
        common_denom = common_denom * d // math.gcd(common_denom, d)
    denom_factor = float(common_denom) * (h ** deriv_order)
    inv = 1.0 / denom_factor
    weights = np.empty(n, dtype=float)
    for j, wj in enumerate(w_rat):
        if wj == 0:
            weights[j] = 0.0
        else:
            scaled_num = wj.numerator * (common_denom // wj.denominator)
            weights[j] = float(scaled_num) * inv
    return weights


def _solve_rational(A, b):
    """Solve A @ x = b in `fractions.Fraction` arithmetic by Gauss
    elimination with partial pivoting. Inputs and outputs are Fractions.
    """
    from fractions import Fraction
    n = len(b)
    M = [list(row) + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        if M[pivot_row][col] == 0:
            raise ValueError("Vandermonde matrix is singular (duplicate nodes?)")
        if pivot_row != col:
            M[col], M[pivot_row] = M[pivot_row], M[col]
        piv = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= piv
        for r in range(col + 1, n):
            factor = M[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n]
        for j in range(i + 1, n):
            s -= M[i][j] * x[j]
        x[i] = s
    return x


def _check_order(order: int, name: str = "order") -> int:
    """Validate that `order` is an even integer ≥ 2 and return it as int."""
    if not isinstance(order, (int, np.integer)):
        raise TypeError(f"{name} must be an integer; got {order!r}")
    order = int(order)
    if order < 2 or order % 2 != 0:
        raise ValueError(
            f"{name} must be an even integer ≥ 2; got {order}"
        )
    return order


def _stencil_width(order: int) -> int:
    """Nodes in a centered stencil of order p: p+1."""
    return order + 1


def _half_width(order: int) -> int:
    """Rows on each side that need a non-centered closure: p/2."""
    return order // 2


def _boundary_stencil_width(order: int, deriv_order: int) -> int:
    """Nodes in a one-sided closure of order p for the k-th derivative:
    p+k (a one-sided stencil of N nodes has order N-k).
    """
    return order + deriv_order


def _check_grid_size(
    N: int, order: int, fn_name: str, deriv_order: int = 1,
) -> None:
    """Require N ≥ max(p+1, p+k): room for the interior stencil and for
    the one-sided closure of the k-th derivative.
    """
    min_required = max(
        _stencil_width(order),
        _boundary_stencil_width(order, deriv_order),
    )
    if N < min_required:
        raise ValueError(
            f"{fn_name} requires N >= {min_required} for order={order} "
            f"(d{deriv_order}); got N={N}"
        )


def _build_uniform_derivative_matrix(
    N: int, h: float, order: int, deriv_order: int,
) -> sp.csr_matrix:
    """Derivative matrix B of order p for the `deriv_order`-th derivative
    on a uniform grid of N points with spacing h.

    Interior rows use the centered (p+1)-point stencil. The `half = p/2`
    rows at each end are block-anchored: all of them share one node set
    of `p + deriv_order` nodes — {0, ..., width-1} on the left,
    {N-width, ..., N-1} on the right — and differ only in the evaluation
    point, which is the row's own node. Rows i ≥ 1 of the block then
    evaluate inside the node set, which keeps their leading error
    constant small.

    Weights come from `_uniform_lagrange_weights_rational`.
    """
    boundary_width = _boundary_stencil_width(order, deriv_order)
    half = _half_width(order)
    B = sp.lil_matrix((N, N))

    # Interior rows: nodes at integer offsets -half..+half.
    interior_w = _uniform_lagrange_weights_rational(
        list(range(-half, half + 1)), 0, deriv_order, h,
    )
    for i in range(half, N - half):
        for k, w in enumerate(interior_w):
            j = i - half + k
            B[i, j] = w

    # Left boundary rows i = 0..half-1: nodes 0..boundary_width-1,
    # evaluated at node i.
    for i in range(half):
        w = _uniform_lagrange_weights_rational(
            list(range(boundary_width)), i, deriv_order, h,
        )
        for k, wk in enumerate(w):
            B[i, k] = wk

    # Right boundary rows i = N-1-row_offset: nodes N-boundary_width..N-1,
    # evaluated at local position boundary_width-1-row_offset.
    for row_offset in range(half):
        i = N - 1 - row_offset
        x_eval_int = (boundary_width - 1) - row_offset
        w = _uniform_lagrange_weights_rational(
            list(range(boundary_width)), x_eval_int, deriv_order, h,
        )
        for k, wk in enumerate(w):
            B[i, N - boundary_width + k] = wk

    return B.tocsr()


def standard_d1_uniform(
    N: int, h: float, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """First derivative on a uniform grid: (A=I, B).

    Interior rows use the centered (order+1)-point stencil; boundary
    rows use one-sided (order+1)-point closures of the same order.

    Parameters
    ----------
    N : int
        Number of grid points, at least order+1.
    h : float
        Grid spacing.
    order : int, default 4
        Even formal order.
    """
    order = _check_order(order)
    _check_grid_size(N, order, "standard_d1_uniform", deriv_order=1)
    A = sp.identity(N, format="csr")
    B = _build_uniform_derivative_matrix(N, h, order, deriv_order=1)
    return A, B


def standard_d2_uniform(
    N: int, h: float, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Second derivative on a uniform grid: (A=I, B).

    Interior rows use the centered (order+1)-point stencil; boundary
    rows use one-sided (order+2)-point closures so every row has
    order `order`.

    Parameters
    ----------
    N : int
        Number of grid points, at least order+2.
    h : float
        Grid spacing.
    order : int, default 4
        Even formal order.
    """
    order = _check_order(order)
    _check_grid_size(N, order, "standard_d2_uniform", deriv_order=2)
    A = sp.identity(N, format="csr")
    B = _build_uniform_derivative_matrix(N, h, order, deriv_order=2)
    return A, B


def _build_uniform_parity_derivative_matrix(
    N: int, h: float, order: int, deriv_order: int, parity: int,
) -> sp.csr_matrix:
    """Derivative matrix B of order p with parity-folded ghosts at the
    left endpoint (node 0 at r=0) and a one-sided closure on the right.

    Rows i = 0..half-1 apply the centered interior stencil with the
    ghost values f_{-k} = parity · f_k substituted: the coefficient of
    column j is the interior weight at offset j-i plus parity times the
    interior weight at offset -j-i whenever that ghost lies in the
    window. Column 0 (the axis) is never folded.

    For parity=+1 the d1 row at i=0 is identically zero (f'(0)=0 for an
    even function) and the d2 row is symmetric in f_1..f_half. For
    parity=-1 the d2 row at i=0 reduces to the single entry w_0·f_0,
    which is consistent with f(0)=0 for an odd function.
    """
    interior_width = _stencil_width(order)
    boundary_width = _boundary_stencil_width(order, deriv_order)
    half = _half_width(order)
    if parity not in (+1, -1):
        raise ValueError(f"parity must be +1 or -1; got {parity!r}")
    B = sp.lil_matrix((N, N))
    interior_w = _uniform_lagrange_weights_rational(
        list(range(-half, half + 1)), 0, deriv_order, h,
    )

    # Parity-folded rows i = 0..half-1. Column i-half+k < 0 is a ghost
    # and folds onto column |i-half+k| with the parity factor.
    for i in range(half):
        row_coefs = {}
        for k in range(interior_width):
            col_signed = i - half + k
            if col_signed < 0:
                col_in = -col_signed
                w = parity * interior_w[k]
            else:
                col_in = col_signed
                w = interior_w[k]
            row_coefs[col_in] = row_coefs.get(col_in, 0.0) + w
        for col, w in row_coefs.items():
            if w != 0.0:
                B[i, col] = w

    # Interior rows.
    for i in range(half, N - half):
        for k, w in enumerate(interior_w):
            j = i - half + k
            B[i, j] = w

    # Right boundary rows: one-sided closure of width boundary_width.
    for row_offset in range(half):
        i = N - 1 - row_offset
        x_eval_int = (boundary_width - 1) - row_offset
        w = _uniform_lagrange_weights_rational(
            list(range(boundary_width)), x_eval_int, deriv_order, h,
        )
        for k, wk in enumerate(w):
            B[i, N - boundary_width + k] = wk

    return B.tocsr()


def standard_d1_uniform_parity(
    N: int, h: float, parity: int, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """First derivative on a uniform grid whose node 0 lies at r=0, with
    ghost values f(r_{-k}) = parity · f(r_k) folded into the `order/2`
    rows nearest the axis. The right endpoint keeps the one-sided
    closure. Returns (A=I, B) like `standard_d1_uniform`.

    Parameters
    ----------
    N : int
    h : float
    parity : int
        ±1.
    order : int, default 4
        Even formal order.
    """
    order = _check_order(order)
    _check_grid_size(N, order, "standard_d1_uniform_parity", deriv_order=1)
    A = sp.identity(N, format="csr")
    B = _build_uniform_parity_derivative_matrix(
        N, h, order, deriv_order=1, parity=parity
    )
    return A, B


def standard_d2_uniform_parity(
    N: int, h: float, parity: int, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Second derivative with parity-folded ghosts at r=0; see
    `standard_d1_uniform_parity`. The folded rows keep the centered
    interior stencil and so have full order p; the right boundary uses
    the (order+2)-point one-sided closure. Returns (A=I, B).

    Parameters
    ----------
    N : int
        Number of grid points, at least order+2.
    h : float
    parity : int
        ±1.
    order : int, default 4
        Even formal order.
    """
    order = _check_order(order)
    _check_grid_size(N, order, "standard_d2_uniform_parity", deriv_order=2)
    A = sp.identity(N, format="csr")
    B = _build_uniform_parity_derivative_matrix(
        N, h, order, deriv_order=2, parity=parity
    )
    return A, B


def standard_d1_nonuniform(
    x: np.ndarray, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """First derivative on a non-uniform grid x: (A=I, B).

    Interior rows use Lagrange weights on the (order+1)-node window
    centered on each point; boundary rows use the first / last
    (order+1) nodes. The order is asymptotically `order` when the grid
    is smooth (h_{i+1} - h_i = O(h²)).
    """
    order = _check_order(order)
    N = len(x)
    _check_grid_size(N, order, "standard_d1_nonuniform", deriv_order=1)
    width = _stencil_width(order)
    half = _half_width(order)
    A = sp.identity(N, format="csr")
    B = sp.lil_matrix((N, N))

    for i in range(half, N - half):
        idx = list(range(i - half, i + half + 1))
        weights = _lagrange_derivative_weights(x[idx], x[i], order=1)
        for j, w in zip(idx, weights):
            B[i, j] = w

    left_idx = list(range(width))
    for i in range(half):
        weights = _lagrange_derivative_weights(x[left_idx], x[i], order=1)
        for j, w in zip(left_idx, weights):
            B[i, j] = w

    right_idx = list(range(N - width, N))
    for i in range(N - half, N):
        weights = _lagrange_derivative_weights(x[right_idx], x[i], order=1)
        for j, w in zip(right_idx, weights):
            B[i, j] = w

    return A, B.tocsr()


def standard_d2_nonuniform(
    x: np.ndarray, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Second derivative on a non-uniform grid x: (A=I, B).

    Interior rows use Lagrange weights on the (order+1)-node centered
    window; boundary rows use the first / last (order+2) nodes so the
    boundary order matches the interior.
    """
    order = _check_order(order)
    N = len(x)
    _check_grid_size(N, order, "standard_d2_nonuniform", deriv_order=2)
    boundary_width = _boundary_stencil_width(order, 2)
    half = _half_width(order)
    A = sp.identity(N, format="csr")
    B = sp.lil_matrix((N, N))

    for i in range(half, N - half):
        idx = list(range(i - half, i + half + 1))
        weights = _lagrange_derivative_weights(x[idx], x[i], order=2)
        for j, w in zip(idx, weights):
            B[i, j] = w

    left_idx = list(range(boundary_width))
    for i in range(half):
        weights = _lagrange_derivative_weights(x[left_idx], x[i], order=2)
        for j, w in zip(left_idx, weights):
            B[i, j] = w

    right_idx = list(range(N - boundary_width, N))
    for i in range(N - half, N):
        weights = _lagrange_derivative_weights(x[right_idx], x[i], order=2)
        for j, w in zip(right_idx, weights):
            B[i, j] = w

    return A, B.tocsr()


def standard_d1(
    x: np.ndarray, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """First-derivative pair (A=I, B) on grid x, dispatching to the
    uniform builder when the spacing is uniform and to the non-uniform
    builder otherwise.
    """
    x = np.asarray(x, dtype=float)
    if _is_uniform(x):
        return standard_d1_uniform(len(x), float(x[1] - x[0]), order=order)
    return standard_d1_nonuniform(x, order=order)


def standard_d2(
    x: np.ndarray, order: int = 4,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """Second-derivative pair (A=I, B) on grid x, dispatching to the
    uniform builder when the spacing is uniform and to the non-uniform
    builder otherwise.
    """
    x = np.asarray(x, dtype=float)
    if _is_uniform(x):
        return standard_d2_uniform(len(x), float(x[1] - x[0]), order=order)
    return standard_d2_nonuniform(x, order=order)


def _is_uniform(x: np.ndarray, rtol: float = 1e-12) -> bool:
    """True when all spacings of x equal the first one within np.allclose
    tolerances (rtol here, default atol).
    """
    diffs = np.diff(x)
    return bool(np.allclose(diffs, diffs[0], rtol=rtol))
