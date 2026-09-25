"""Inner products for the analysis of a pencil ``K x = lambda M x``.

Every quantity in ``spdee.analysis`` is taken in a weighted inner product

    <x, y>_W = x^H W y,    ||x||_W = sqrt(x^H W x),

with W Hermitian positive semidefinite.  W may be singular: the constraint
unknowns of a saddle-point pencil (a pressure, a divergence multiplier)
carry no mass and no weight, so ||.||_W is a seminorm on the state and a
norm on the dynamical unknowns, the columns of M that are not identically
zero.  The M-inner product, W = (M + M^H)/2 when that is positive
semidefinite (a real nonnegative diagonal mass), is the default for a bare
pencil; a compiled problem supplies its volume weight instead
(``natural_weight``):

    fd2 collocated   r_i^2 sin(theta_j) dr_i dtheta on every spin component
                     (trapezoid in r, midpoint in theta),
    staggered        the C-grid cell volume r^2 sin(theta) dr dtheta of the
                     sub-grid each component lives on (the norm under which
                     the assembler's operator pairs are adjoint),
    sh_fd            r_i^2 dr_i per (component, l, r) coefficient (the
                     spin-weighted harmonics are orthonormal on the sphere),
    dedalus          the identity on the dynamical coefficients (the Jacobi
                     bases are not orthonormal in L2, so this is the
                     coefficient-space norm, not the energy),

each zeroed on the unknowns without mass.  ``Weight`` wraps a diagonal
(1-D array) or a sparse Hermitian W and provides the norm, the inner
product, W^(1/2) and its pseudo-inverse (diagonal only) and W^+ y.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["Weight", "default_weight", "natural_weight", "mass_support"]

ArrayLike = Union[np.ndarray, sp.spmatrix, "Weight", None]


def mass_support(M: Optional[sp.spmatrix], N: int, tol: float = 0.0) -> np.ndarray:
    """Boolean mask of the dynamical unknowns: the columns of M with an
    entry above ``tol`` in magnitude (all True for a standard problem)."""
    if M is None:
        return np.ones(N, dtype=bool)
    col = np.asarray(abs(M).max(axis=0).todense()).ravel()
    return col > tol


class Weight:
    """A Hermitian positive semidefinite weight W, diagonal (``w`` 1-D) or
    sparse.  A sparse W must be nonsingular; a diagonal one may carry
    zeros, which mark the unknowns outside the (semi)norm's support."""

    def __init__(self, w, N: Optional[int] = None):
        if isinstance(w, Weight):
            self.w, self.mat, self.N = w.w, w.mat, w.N
            self._lu = w._lu
            return
        self._lu = None
        if sp.issparse(w):
            self.mat = w.tocsc()
            self.N = self.mat.shape[0]
            d = self.mat.diagonal()
            if (self.mat - sp.diags(d)).nnz == 0:
                self.w = np.real(d).astype(float)
                self.mat = None
            else:
                self.w = None
        else:
            self.w = np.asarray(w, dtype=float).ravel()
            self.mat = None
            self.N = self.w.size
        if self.w is not None and np.any(self.w < 0):
            raise ValueError("a diagonal weight must be nonnegative")
        if N is not None and N != self.N:
            raise ValueError(f"weight of size {self.N} for a pencil of size {N}")

    @classmethod
    def identity(cls, N: int, support: Optional[np.ndarray] = None) -> "Weight":
        w = np.ones(N)
        if support is not None:
            w[~np.asarray(support, dtype=bool)] = 0.0
        return cls(w)

    @property
    def is_diagonal(self) -> bool:
        return self.w is not None

    @property
    def support(self) -> np.ndarray:
        """Mask of the unknowns inside the norm's support."""
        if self.is_diagonal:
            return self.w > 0
        return np.ones(self.N, dtype=bool)

    def apply(self, x):
        """W x (columns of a 2-D array are treated independently)."""
        if self.is_diagonal:
            return x * (self.w[:, None] if np.ndim(x) == 2 else self.w)
        return self.mat @ x

    def pinv_apply(self, y):
        """W^+ y (the pseudo-inverse on the support of a diagonal W)."""
        if self.is_diagonal:
            inv = np.where(self.w > 0, 1.0 / np.where(self.w > 0, self.w, 1.0), 0.0)
            return y * (inv[:, None] if np.ndim(y) == 2 else inv)
        if self._lu is None:
            self._lu = spla.splu(self.mat)
        return self._lu.solve(np.ascontiguousarray(y))

    def sqrt_apply(self, x):
        """W^(1/2) x (diagonal W only)."""
        s = np.sqrt(self._diag())
        return x * (s[:, None] if np.ndim(x) == 2 else s)

    def pinv_sqrt_apply(self, x):
        """W^(+1/2) x, zero outside the support (diagonal W only)."""
        w = self._diag()
        s = np.where(w > 0, 1.0 / np.sqrt(np.where(w > 0, w, 1.0)), 0.0)
        return x * (s[:, None] if np.ndim(x) == 2 else s)

    def _diag(self):
        if not self.is_diagonal:
            raise NotImplementedError("W^(1/2) needs a diagonal weight")
        return self.w

    def inner(self, x, y) -> complex:
        return complex(np.vdot(x, self.apply(y)))

    def norm(self, x) -> float:
        """||x||_W, columnwise for a 2-D array."""
        Wx = self.apply(x)
        if np.ndim(x) == 2:
            return np.sqrt(np.abs(np.einsum("ij,ij->j", np.conj(x), Wx)))
        return float(np.sqrt(abs(np.vdot(x, Wx))))

    def normalize(self, x):
        n = self.norm(x)
        return x / np.where(n == 0, 1.0, n) if np.ndim(x) == 2 else x / (n or 1.0)

    def orthonormalize(self, V, tol: float = 1e-12):
        """A W-orthonormal basis of the columns of V by modified Gram-Schmidt
        with one reorthogonalization; columns that fall below ``tol`` of
        their original W-norm are dropped.  Returns (Q, R) with V = Q R."""
        V = np.array(V, dtype=complex, copy=True)
        n, k = V.shape
        Q = np.zeros((n, 0), dtype=complex)
        R = np.zeros((k, k), dtype=complex)
        keep = []
        for j in range(k):
            v = V[:, j]
            n0 = self.norm(v)
            for _ in range(2):
                if Q.shape[1]:
                    c = np.conj(Q).T @ self.apply(v)
                    v = v - Q @ c
                    R[keep, j] += c
            nv = self.norm(v)
            if nv <= tol * (n0 or 1.0):
                continue
            R[j, j] = nv
            Q = np.hstack([Q, (v / nv)[:, None]])
            keep.append(j)
        return Q, R[np.ix_(keep, keep)] if keep else R[:0, :0]


def default_weight(M: Optional[sp.spmatrix], N: int) -> Weight:
    """The M-inner product when M is a real nonnegative diagonal (possibly
    with zero rows), else the identity on the dynamical unknowns."""
    if M is None:
        return Weight.identity(N)
    d = M.diagonal()
    off = (M - sp.diags(d)).nnz
    if off == 0 and np.all(np.abs(d.imag) <= 1e-14 * (np.abs(d.real) + 1e-300)) \
            and np.all(d.real >= 0):
        return Weight(d.real.astype(float))
    return Weight.identity(N, mass_support(M, N))


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 1:
        return np.ones(1)
    w = np.empty_like(x)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def _cell_weights(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Cell widths of cell-centred nodes between the walls lo and hi."""
    x = np.asarray(x, float)
    edges = np.concatenate([[lo], 0.5 * (x[1:] + x[:-1]), [hi]])
    return np.diff(edges)


def natural_weight(cp, M: Optional[sp.spmatrix] = None) -> Weight:
    """The volume weight of a compiled problem (module docstring), zero on
    the unknowns without mass in its pencil."""
    if M is None:
        _, M = cp.assemble_evp_matrices()
    N = M.shape[0]
    disc = getattr(cp, "discretization", None)
    if disc == "sh_fd":
        w = _sh_fd_weight(cp)
    elif disc == "dedalus":
        w = np.ones(N)
    elif disc == "staggered":
        w = _staggered_weight(cp)
    elif hasattr(cp, "matrices"):
        w = _collocated_weight(cp)
    else:
        raise TypeError(f"no natural weight for {type(cp).__name__}")
    w = np.asarray(w, float)
    w[~mass_support(M, N)] = 0.0
    return Weight(w)


def _collocated_weight(cp) -> np.ndarray:
    cm = cp.matrices
    grid, layout = cm.grid, cm.layout
    wr = _trapezoid_weights(grid.r) * grid.r ** 2
    th = grid.theta
    wt = _cell_weights(th, 0.0, np.pi) * np.sin(th)
    node = (wr[:, None] * wt[None, :]).ravel(order="F")
    w = np.zeros(layout.size)
    for name in layout.all_names:
        e = layout.entry(name)
        ncomp = e.size // node.size
        w[e.slice] = np.tile(node, ncomp)
    return w


def _sh_fd_weight(cp) -> np.ndarray:
    r = cp.r
    if cp.ball:
        wr = _cell_weights(r, 0.0, cp.grid.r_outer) * r ** 2
    else:
        wr = _trapezoid_weights(r) * r ** 2
    w = np.zeros(cp.N)
    for _name, kind, off, size in cp.layout:
        w[off:off + size] = np.tile(wr, size // len(r))
    return w


def _staggered_weight(cp) -> np.ndarray:
    from spdee.analysis.modes import staggered_node_index
    g = cp.grids
    hr = g.r_edge[1] - g.r_edge[0]
    ht = g.theta_center[1] - g.theta_center[0] if len(g.theta_center) > 1 else np.pi
    nodes = {"center": (g.r_center, g.theta_center), "r_edge": (g.r_edge, g.theta_center),
             "theta_edge": (g.r_center, g.theta_edge), "corner": (g.r_edge, g.theta_edge)}
    w = np.zeros(cp.layout.total_size)
    for f in cp.layout.fields:
        rv, tv = nodes[f.location]
        vol = ((rv ** 2)[:, None] * np.sin(tv)[None, :]).ravel() * hr * ht
        full_index = staggered_node_index(cp, f)
        w[f.slice] = vol[full_index]
    return w
