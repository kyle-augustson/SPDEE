"""Preconditioning and convergence aids for the shift-invert and
Jacobi-Davidson solves of a pencil K x = lambda M x.

Equilibration.  Diagonal scalings D1 (rows) and D2 (columns) applied to
both matrices,

    K' = D1 K D2,   M' = D1 M D2,   K' x' = lambda M' x'  with  x = D2 x',

leave every eigenvalue unchanged (the pencil is equivalent) and map the
eigenvectors by D2.  The scalings are Ruiz iterations on the shifted
matrix P = K - sigma M: each sweep divides every row and every column by
the square root of its largest entry in magnitude, so that after
convergence every row and column of D1 P D2 has max-norm one; with
``row_blocks`` / ``col_blocks`` (index sets per unknown or per equation)
one scale per block is used instead of one per row or column, so the
scaling is the physical rescaling of the unknowns and equations.  The
improvement is measured by the 1-norm condition number estimate
cond_1(P) = ||P||_1 ||P^(-1)||_1 (Hager's estimator on the LU) before and
after.

Deflation.  Given a subspace V (converged eigenvectors, null modes of a
singular pencil, pinned rings), the orthogonal projector
P_R = I - Q Q^H (Q an orthonormal basis of V in the Euclidean or a
user-supplied inner product) and the left projector P_L = I - Z Z^H
(Z an orthonormal basis of [M V, K V]) give the deflated pencil

    K' = P_L K P_R,   M' = P_L M P_R,

whose spectrum is the remaining spectrum of (K, M) plus the singular
directions V (annihilated by both matrices): for an invariant V with
K V = M V T, a remaining eigenpair (lambda, u) gives
K' P_R u = lambda M' P_R u exactly, and the shift-invert operator
P_R (K - sigma M)^(-1) M P_R has the remaining eigenvalues
mu = 1/(lambda - sigma) with the deflated ones sent to mu = 0.  The
eigenvector of the original pencil is recovered from P_R u by the least
squares ``restore``.  ARPACK iterates the projected shift-invert operator;
bbjd receives the projected matvecs and a projected correction-equation
preconditioner P_R (K - sigma M)^(-1) P_L (the Jacobi-Davidson deflated
preconditioner with the exact projector replaced by its cheap
approximation, adequate for an inner GMRES at loose tolerance).
``pressure_null_modes`` finds the vectors supported on the massless
unknowns that K nearly annihilates (the collocated pressure
checkerboard, a 0 = lambda 0 direction of the pencil).

Coarse-grid preconditioner.  The same problem compiled on a coarser grid
of the same core gives (K_c, M_c); with the restriction R (fine -> coarse)
and prolongation P (coarse -> fine), both local Lagrange interpolation
between the grids applied blockwise to every unknown / equation, the
correction-equation preconditioner for bbjd is the coarse-grid correction

    B^(-1) b = P (K_c - sigma M_c)^(-1) R b,

an approximation of (K - sigma M)^(-1) whose cost is the coarse
factorization (the two-grid step of a multigrid method without a fine
smoother); the fd2 collocated core interpolates in r and theta, the
sh_fd core in r at fixed l.  With a smoother S the two-level step is
x = S b, x += P (K_c - sigma M_c)^(-1) R (b - (K - sigma M) x), then one
more smoothing of the remaining residual.

Schwarz preconditioner.  The fd2 collocated grid cut into patches in
theta and r, each extended by ``overlap`` points on every side; the
shifted matrix restricted to a patch, all unknowns and all equations at
its points, is factorized exactly, so the block structure of the pencil
(massless constraint rows, zero diagonals) and the radial stratification
are handled inside every patch, where an incomplete factorization of the
whole matrix breaks down on the zero pivots.  The preconditioner solves
every patch and keeps each solution on the points the patch owns
(restricted additive Schwarz); alone it is a smoother, and as the
smoother of the coarse-grid preconditioner it gives the two-level method
whose memory is the patch factors and the coarse factorization.  Patches
are independent, so the solves parallelize over them.

Refinement.  A mode found on a coarse grid is carried to a fine grid of
the same problem: the coarse eigenvector interpolated by the prolongation
starts bbjd on the fine pencil, at a shift offset from the coarse
eigenvalue (the coarse factorization inside the two-level preconditioner
is singular at the coarse eigenvalue itself), with the correction
equation solved by preconditioned GMRES; the fine eigenpair is the one
of largest overlap with the interpolated start.

Continuation.  Walking a parameter (Ek, Ra, B0) in steps, the previous
eigenvector is the ARPACK start vector and the previous eigenvalue
(offset by ``shift_offset`` relative) the shift; at each step the mode is
followed by the largest W-overlap |<x_prev, x_j>_W| / (||x_prev|| ||x_j||)
among the returned eigenpairs, not by the nearest eigenvalue, and the
path records value, eigenvalue, overlap and residual.

Diagnosis.  ``diagnose`` reads a stalled or slow solve and reports the
likely causes with evidence and a remedy: a near-singular shift
(cond_1(K - sigma M) large or an eigenvalue within rounding of sigma), a
dense constraint row (a massless row of K with many more entries than the
median, the integral gauge on which bbjd stalls), an ill-conditioned
target (kappa of the eigenvalues nearest sigma above a threshold), a dense
band (eigenvalue spacing near sigma comparable to the residual level, so
nearest-eigenvalue identification fails), and an unequilibrated pencil
(entry magnitudes spanning many decades across blocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.solvers import factorize, _shifted, shift_invert
from spdee.analysis.weights import Weight, default_weight, mass_support

__all__ = ["equilibrate", "Equilibration", "condition_estimate", "Deflation",
           "pressure_null_modes", "CoarseGridPreconditioner", "SchwarzPreconditioner", "refine",
           "Refinement", "continuation", "ContinuationPath", "diagnose", "Diagnosis",
           "interpolation_matrix"]


# ------------------------------------------------------------ equilibration

def condition_estimate(P, factor: str = "scipy") -> float:
    """cond_1(P) = ||P||_1 ||P^(-1)||_1 with Hager's estimator on the LU."""
    P = P.tocsc()
    n1 = spla.norm(P, 1)
    fac = factorize(P, factor)
    try:
        op = spla.LinearOperator(P.shape, matvec=fac.solve, rmatvec=fac.solve_H, dtype=complex)
        inv1 = spla.onenormest(op)
    finally:
        fac.close()
    return float(n1 * inv1)


@dataclass
class Equilibration:
    d1: np.ndarray
    d2: np.ndarray
    K: sp.spmatrix
    M: Optional[sp.spmatrix]
    sigma: complex
    cond_before: Optional[float] = None
    cond_after: Optional[float] = None

    def vectors(self, X_eq: np.ndarray) -> np.ndarray:
        """Eigenvectors of the original pencil from those of the scaled one."""
        X_eq = np.asarray(X_eq)
        return X_eq * (self.d2[:, None] if X_eq.ndim == 2 else self.d2)

    def to_scaled(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return X / (self.d2[:, None] if X.ndim == 2 else self.d2)

    @property
    def improvement(self) -> Optional[float]:
        if self.cond_before is None or self.cond_after is None:
            return None
        return self.cond_before / self.cond_after

    def __str__(self) -> str:
        s = (f"equilibration: row scales in [{self.d1.min():.2e}, {self.d1.max():.2e}], "
             f"column scales in [{self.d2.min():.2e}, {self.d2.max():.2e}]")
        if self.improvement is not None:
            s += (f"; cond_1(K - sigma M) {self.cond_before:.3e} -> {self.cond_after:.3e}"
                  f" (factor {self.improvement:.3g})")
        return s


def _block_max(v: np.ndarray, blocks) -> np.ndarray:
    if blocks is None:
        return v
    out = np.zeros_like(v)
    for b in blocks:
        idx = np.arange(b.start, b.stop) if isinstance(b, slice) else np.asarray(b)
        out[idx] = v[idx].max() if len(idx) else 0.0
    return out


def equilibrate(K, M, sigma: complex = 0.0, row_blocks=None, col_blocks=None,
                iterations: int = 10, report: bool = True, factor: str = "scipy") -> Equilibration:
    """Ruiz-equilibrate the pencil on P = K - sigma M (module docstring)."""
    K = K.tocsr()
    Mm = None if M is None else M.tocsr()
    N = K.shape[0]
    P = _shifted(K, Mm, sigma).tocsr()
    cond_before = condition_estimate(P, factor) if report else None
    d1 = np.ones(N)
    d2 = np.ones(N)
    A = abs(P).astype(float)
    for _ in range(iterations):
        r = _block_max(np.asarray(A.max(axis=1).todense()).ravel(), row_blocks)
        c = _block_max(np.asarray(A.max(axis=0).todense()).ravel(), col_blocks)
        r = 1.0 / np.sqrt(np.where(r > 0, r, 1.0))
        c = 1.0 / np.sqrt(np.where(c > 0, c, 1.0))
        d1 *= r
        d2 *= c
        A = sp.diags(r) @ A @ sp.diags(c)
        if np.max(np.abs(np.log(r))) < 1e-3 and np.max(np.abs(np.log(c))) < 1e-3:
            break
    D1, D2 = sp.diags(d1), sp.diags(d2)
    K_eq = (D1 @ K @ D2).tocsr()
    M_eq = None if Mm is None else (D1 @ Mm @ D2).tocsr()
    cond_after = condition_estimate(_shifted(K_eq, M_eq, sigma), factor) if report else None
    return Equilibration(d1=d1, d2=d2, K=K_eq, M=M_eq, sigma=complex(sigma),
                         cond_before=cond_before, cond_after=cond_after)


# ----------------------------------------------------------------- deflation

class Deflation:
    """A subspace to remove from a solve (module docstring).  ``V`` (N x k)
    spans it; ``W`` is the inner product of the right projector (None:
    Euclidean).  Columns that both K and M annihilate (relative to
    ``null_tol``) are 0 = lambda 0 directions of a singular pencil; for
    them ``regularize`` makes the shifted matrix nonsingular."""

    def __init__(self, K, M, V: np.ndarray, W=None, null_tol: float = 1e-10):
        K = K.tocsr()
        self.K = K
        self.M = None if M is None else M.tocsr()
        V = np.asarray(V, dtype=complex)
        if V.ndim == 1:
            V = V[:, None]
        self.N = K.shape[0]
        Wt = Weight.identity(self.N) if W is None else Weight(W, self.N)
        self.Wt = Wt
        self.Q, _ = Wt.orthonormalize(V)
        self.V = V
        MQ = self.Q if self.M is None else self.M @ self.Q
        KQ = K @ self.Q
        nK = spla.norm(K, 1)
        nM = nK if self.M is None else (spla.norm(self.M, 1) or nK)
        self.null = ((np.linalg.norm(KQ, axis=0) < null_tol * nK)
                     & (np.linalg.norm(MQ, axis=0) < null_tol * nM))
        Z = np.hstack([MQ, KQ])
        Z = Z[:, np.linalg.norm(Z, axis=0) > null_tol * nK]
        eye = Weight.identity(self.N)
        self.Z = eye.orthonormalize(Z, tol=1e-10)[0] if Z.shape[1] else np.zeros((self.N, 0),
                                                                                 dtype=complex)
        self.Z0 = None
        self.WQ = Wt.apply(self.Q)

    @property
    def k(self) -> int:
        return self.Q.shape[1]

    def regularize(self, P, factor: str = "scipy", eps: float = 1e-8):
        """P + c Z_0 Q_0^H for the 0 = lambda 0 directions Q_0 of the
        subspace, Z_0 their left null vectors (inverse iteration on
        P^H + eps ||P|| I), c = ||P||_1: nonsingular, and equal to P on the
        deflated pencil, since (P + c Z_0 Q_0^H) x = b with b orthogonal to
        Z_0 gives Q_0^H x = 0 and P x = b.  The left null vectors join the
        left projector.  P itself when the subspace has no null direction."""
        if not self.null.any():
            return P
        P = P.tocsc()
        n1 = spla.norm(P, 1)
        Q0 = self.Q[:, self.null]
        shifted = (P + eps * n1 * sp.identity(self.N, dtype=complex, format="csc")).tocsc()
        fac = factorize(shifted, factor)
        try:
            rng = np.random.default_rng(0)
            Y = rng.standard_normal(Q0.shape) + 1j * rng.standard_normal(Q0.shape)
            for _ in range(3):
                Y = fac.solve_H(Y)
                Y /= np.linalg.norm(Y, axis=0)
        finally:
            fac.close()
        Z0, _ = Weight.identity(self.N).orthonormalize(Y)
        self.Z0 = Z0
        self.Z = Weight.identity(self.N).orthonormalize(np.hstack([self.Z, Z0]), tol=1e-10)[0]
        update = sp.csr_matrix(Z0) @ sp.csr_matrix(np.conj(Q0).T)
        update.data[np.abs(update.data) < 1e-15] = 0.0
        update.eliminate_zeros()
        return (P + n1 * update).tocsc()

    def project_right(self, x):
        """P_R x = x - Q (Q^H W x), columnwise for a 2-D array."""
        return x - self.Q @ (np.conj(self.WQ).T @ x)

    def project_left(self, b):
        return b - self.Z @ (np.conj(self.Z).T @ b)

    def shift_invert_operator(self, fac) -> spla.LinearOperator:
        """P_R (K - sigma M)^(-1) M P_R from a factorization of (K - sigma M)."""
        M = self.M

        def mv(x):
            x = self.project_right(x)
            return self.project_right(fac.solve(x if M is None else M @ x))
        return spla.LinearOperator((self.N, self.N), matvec=mv, dtype=complex)

    def deflated_pencil(self, prec=None):
        """(K', M', prec') as LinearOperators / a solve object for bbjd."""
        K, M = self.K, self.M
        N = self.N
        Kd = spla.LinearOperator((N, N), dtype=complex,
                                 matvec=lambda x: self.project_left(K @ self.project_right(x)))
        Md = spla.LinearOperator((N, N), matvec=lambda x: self.project_left(
            self.project_right(x) if M is None else M @ self.project_right(x)), dtype=complex)
        pd = None
        if prec is not None:
            solve = prec.solve if hasattr(prec, "solve") else prec

            class _Prec:
                @staticmethod
                def solve(b, _s=solve, _d=self):
                    return _d.project_right(_s(_d.project_left(b)))
            pd = _Prec()
        return Kd, Md, pd

    def residual(self, lam, U: np.ndarray) -> np.ndarray:
        """Relative residual of the pairs (lam, U) in the deflated pencil,
        ||P_L (K - lambda M) P_R u|| / (||P_L K P_R u|| + |lambda| ||P_L M P_R u||):
        for a singular pencil the eigenpairs of its regular part satisfy
        this, not the residual of the full pencil, whose left singular
        direction they may excite."""
        U = np.asarray(U, dtype=complex)
        if U.ndim == 1:
            U = U[:, None]
        lam = np.atleast_1d(np.asarray(lam, dtype=complex))
        PU = self.project_right(U)
        KU = self.project_left(self.K @ PU)
        MU = self.project_left(PU if self.M is None else self.M @ PU)
        num = np.linalg.norm(KU - MU * lam[None, :], axis=0)
        den = np.linalg.norm(KU, axis=0) + np.abs(lam) * np.linalg.norm(MU, axis=0)
        return num / np.where(den == 0, 1.0, den)

    def restore(self, lam: complex, u: np.ndarray) -> np.ndarray:
        """The eigenvector u + V c of the original pencil closest in residual
        to the deflated eigenvector u (least squares in c)."""
        A = self.K - lam * (sp.identity(self.N, dtype=complex, format="csr")
                            if self.M is None else self.M)
        AV = A @ self.V
        c, *_ = np.linalg.lstsq(AV, -(A @ u), rcond=None)
        return u + self.V @ c


def pressure_null_modes(K, M, k: int = 2, rtol: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """(sigma, N_modes): the k smallest singular values of K restricted to
    the massless columns, relative to ||K||_1, and the corresponding unit
    vectors (zero on the dynamical unknowns); those with a relative
    singular value below ``rtol`` are directions the pencil annihilates
    for every lambda."""
    K = K.tocsr()
    N = K.shape[0]
    cols = np.nonzero(~mass_support(M, N))[0]
    if len(cols) == 0:
        return np.empty(0), np.empty((N, 0), dtype=complex)
    KC = K[:, cols]
    if len(cols) <= 4000:
        _, s, Vh = np.linalg.svd(KC.toarray(), full_matrices=False)
        s, Vh = s[::-1][:k], Vh[::-1][:k]
    else:
        _, s, Vh = spla.svds(KC.tocsc(), k=k, which="SM")
        order = np.argsort(s)
        s, Vh = s[order], Vh[order]
    modes = np.zeros((N, len(s)), dtype=complex)
    modes[cols, :] = np.conj(Vh).T
    return s / spla.norm(K, 1), modes


# ------------------------------------------------------- coarse-grid solve

def interpolation_matrix(src: np.ndarray, dst: np.ndarray, order: int = 4) -> sp.csr_matrix:
    """Local Lagrange interpolation from the nodes ``src`` to ``dst``
    (``order`` + 1 nearest source nodes, clamped at the ends)."""
    from spdee.backends.fd2.stencils import _lagrange_derivative_weights
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    width = min(order + 1, len(src))
    rows, cols, vals = [], [], []
    for i, x in enumerate(dst):
        j = int(np.searchsorted(src, x))
        j0 = int(np.clip(j - width // 2, 0, len(src) - width))
        idx = np.arange(j0, j0 + width)
        w = _lagrange_derivative_weights(src[idx], float(x), order=0)
        rows += [i] * width
        cols += list(idx)
        vals += list(w)
    return sp.csr_matrix((vals, (rows, cols)), shape=(len(dst), len(src)), dtype=complex)


class CoarseGridPreconditioner:
    """The two-grid preconditioner for a fine and a coarse compiled problem
    of the same core (module docstring): the coarse-grid correction
    P (K_c - sigma M_c)^(-1) R, wrapped in a pre- and post-smoothing step by
    an incomplete LU of the fine shifted matrix (``smoother='ilu'``, SuperLU
    ILUTP with ``drop_tol`` and ``fill_factor``) or by any object with
    ``solve(b)`` approximating (K - sigma M)^(-1) on the fine grid (a
    ``SchwarzPreconditioner``), or the coarse correction alone
    (``smoother=None``), which has rank N_c and leaves the fine-grid
    complement uncorrected."""

    def __init__(self, cp_fine, cp_coarse, sigma: complex, factor: str = "scipy", order: int = 4,
                 smoother="ilu", drop_tol: float = 1e-3, fill_factor: float = 10.0):
        K_c, M_c = cp_coarse.assemble_evp_matrices()
        self.sigma = complex(sigma)
        self.fac = factorize(_shifted(K_c, M_c, sigma), factor)
        self.P, self.R = self._transfer(cp_fine, cp_coarse, order)
        self.shape = (self.P.shape[0], self.P.shape[0])
        self.calls = 0
        self.fine = None
        self.smoother = None
        if smoother is not None:
            K_f, M_f = cp_fine.assemble_evp_matrices()
            self.fine = _shifted(K_f, M_f, sigma).tocsc()
            if smoother == "ilu":
                self.smoother = spla.spilu(self.fine, drop_tol=drop_tol, fill_factor=fill_factor)
            elif hasattr(smoother, "solve"):
                self.smoother = smoother
            else:
                raise ValueError("smoother must be 'ilu', None or an object with solve(b)")

    def coarse_correction(self, r):
        return self.P @ self.fac.solve(self.R @ np.asarray(r, dtype=complex))

    def solve(self, b):
        self.calls += 1
        b = np.asarray(b, dtype=complex)
        if self.smoother is None:
            return self.coarse_correction(b)
        x = self.smoother.solve(b)
        x = x + self.coarse_correction(b - self.fine @ x)
        return x + self.smoother.solve(b - self.fine @ x)

    matvec = solve

    def close(self):
        self.fac.close()

    @staticmethod
    def _transfer(cf, cc, order):
        if hasattr(cf, "matrices") and hasattr(cc, "matrices"):
            return _collocated_transfer(cf.matrices, cc.matrices, order)
        if getattr(cf, "discretization", None) == "sh_fd" and getattr(cc, "discretization",
                                                                       None) == "sh_fd":
            if cf.nl != cc.nl:
                raise ValueError("sh_fd coarse grid must keep ell_max")
            Pr = interpolation_matrix(cc.r, cf.r, order)
            Rr = interpolation_matrix(cf.r, cc.r, order)
            nb = cf.N // cf.Nr
            return (sp.kron(sp.identity(nb), Pr, format="csr"),
                    sp.kron(sp.identity(nb), Rr, format="csr"))
        raise NotImplementedError("coarse-grid transfer exists for the fd2 collocated and sh_fd "
                                  "cores only")


def _collocated_transfer(cmf, cmc, order):
    gf, gc = cmf.grid, cmc.grid
    Pr = interpolation_matrix(gc.r, gf.r, order)
    Pt = interpolation_matrix(gc.theta, gf.theta, order)
    Rr = interpolation_matrix(gf.r, gc.r, order)
    Rt = interpolation_matrix(gf.theta, gc.theta, order)
    P2 = sp.kron(Pt, Pr, format="csr")
    R2 = sp.kron(Rt, Rr, format="csr")
    nf, nc = gf.N_r * gf.N_theta, gc.N_r * gc.N_theta
    lf, lc = cmf.layout, cmc.layout
    if lf.all_names != lc.all_names:
        raise ValueError("the two compiled problems carry different unknowns")
    Pcols, Rcols = [], []
    for name in lf.all_names:
        ncomp = lf.entry(name).size // nf
        Pcols.append(sp.kron(sp.identity(ncomp), P2, format="csr"))
        Rcols.append(sp.kron(sp.identity(ncomp), R2, format="csr"))
    P = sp.block_diag(Pcols, format="csr")
    Prow, Rrow = [], []
    for eq, sl in cmf.equation_row_slices.items():
        ncomp = (sl.stop - sl.start) // nf
        slc = cmc.equation_row_slices[eq]
        if (slc.stop - slc.start) // nc != ncomp:
            raise ValueError(f"equation {eq!r} differs between the grids")
        Prow.append(sp.kron(sp.identity(ncomp), P2, format="csr"))
        Rrow.append(sp.kron(sp.identity(ncomp), R2, format="csr"))
    R = sp.block_diag(Rrow, format="csr")
    return P.tocsr(), R.tocsr()


# ------------------------------------------------------------------- schwarz

def _collocated_point_maps(cp):
    """For an fd2 collocated compiled problem: N_r, N_theta and two maps
    from an array of grid-point indices (i + N_r j) to the unknown columns
    and the equation rows at those points, every component of every field
    or equation, in the layout's order."""
    cm = cp.matrices
    grid = cm.grid
    N_r, N_t = grid.N_r, grid.N_theta
    npt = N_r * N_t
    col_blocks = [(cm.layout.entry(name).offset, cm.layout.entry(name).size // npt)
                  for name in cm.layout.all_names]
    row_blocks = [(sl.start, (sl.stop - sl.start) // npt) for sl in cm.equation_row_slices.values()]

    def expand(blocks, pts):
        return np.concatenate([off + c * npt + pts for off, nc in blocks for c in range(nc)])

    return N_r, N_t, (lambda pts: expand(col_blocks, pts)), (lambda pts: expand(row_blocks, pts))


class SchwarzPreconditioner:
    """Restricted additive Schwarz on patches of the fd2 collocated grid
    (module docstring): ``n_theta_blocks`` x ``n_r_blocks`` patches, each
    extended by ``overlap`` points on every side, the shifted matrix
    restricted to the unknowns and equations at a patch's points factorized
    exactly (``factor`` and ``factor_options`` as in ``shift_invert``)."""

    def __init__(self, cp, sigma: complex, n_theta_blocks: int = 4, n_r_blocks: int = 1,
                 overlap: int = 2, factor: str = "scipy",
                 factor_options: Optional[Dict] = None):
        if not hasattr(cp, "matrices"):
            raise NotImplementedError("the Schwarz preconditioner exists for the fd2 collocated "
                                      "core only")
        K, M = cp.assemble_evp_matrices()
        P = _shifted(K, M, sigma).tocsr()
        self.sigma = complex(sigma)
        self.shape = P.shape
        self.calls = 0
        N_r, N_t, cols_of, rows_of = _collocated_point_maps(cp)
        edges_t = np.linspace(0, N_t, n_theta_blocks + 1).astype(int)
        edges_r = np.linspace(0, N_r, n_r_blocks + 1).astype(int)
        self.patches = []
        for t0, t1 in zip(edges_t[:-1], edges_t[1:]):
            for r0, r1 in zip(edges_r[:-1], edges_r[1:]):
                tt = np.arange(max(0, t0 - overlap), min(N_t, t1 + overlap))
                rr = np.arange(max(0, r0 - overlap), min(N_r, r1 + overlap))
                pts = (tt[:, None] * N_r + rr[None, :]).ravel()
                own_pts = (((tt >= t0) & (tt < t1))[:, None]
                           & ((rr >= r0) & (rr < r1))[None, :]).ravel()
                cols, rows = cols_of(pts), rows_of(pts)
                own = np.concatenate([own_pts] * (cols.size // pts.size))
                fac = factorize(P[rows][:, cols].tocsc(), factor, **(factor_options or {}))
                self.patches.append((rows, cols[own], own, fac))

    def solve(self, b):
        self.calls += 1
        b = np.asarray(b, dtype=complex)
        x = np.zeros_like(b)
        for rows, owned_cols, own, fac in self.patches:
            x[owned_cols] = fac.solve(b[rows])[own]
        return x

    matvec = solve

    def close(self):
        for *_, fac in self.patches:
            fac.close()


# ---------------------------------------------------------------- refinement

@dataclass
class Refinement:
    lam: complex
    x: np.ndarray
    overlap: float
    residual: float
    iters: int
    nconv: int


def refine(cp_fine, cp_coarse, lam: complex, x: np.ndarray, sigma: Optional[complex] = None,
           preconditioner=None, shift_offset: float = 1e-2, k: int = 1, order: int = 4,
           inner_maxiter: int = 20, inner_tol: float = 1e-2, tol: float = 1e-10,
           **solver_options) -> Refinement:
    """Carry the coarse eigenpair (lam, x) to the fine problem (module
    docstring).  ``sigma`` defaults to lam offset by ``shift_offset``
    relative; ``preconditioner`` to the two-level Schwarz preconditioner
    (4 theta patches with the coarse problem); ``solver_options`` go to
    bbjd (maxsub, minsub, maxiter, verbose)."""
    from spdee.analysis.modes import pencil_residual
    lam = complex(lam)
    if sigma is None:
        sigma = lam + shift_offset * (1.0 + abs(lam)) * (1.0 + 1.0j) / np.sqrt(2)
    sigma = complex(sigma)
    K, M = cp_fine.assemble_evp_matrices()
    P, _ = CoarseGridPreconditioner._transfer(cp_fine, cp_coarse, order)
    v0 = P @ np.asarray(x, dtype=complex)
    own = preconditioner is None
    if own:
        preconditioner = CoarseGridPreconditioner(
            cp_fine, cp_coarse, sigma, order=order,
            smoother=SchwarzPreconditioner(cp_fine, sigma))
    info: Dict = {}
    try:
        lam_f, V = shift_invert(K, M, sigma, k, method="bbjd", return_vectors=True, v0=v0,
                                preconditioner=preconditioner, inner_maxiter=inner_maxiter,
                                inner_tol=inner_tol, tol=tol, info=info, **solver_options)
    finally:
        if own:
            preconditioner.close()
            preconditioner.smoother.close()
    ok = np.isfinite(lam_f)
    if not ok.any():
        raise RuntimeError(f"refine: bbjd converged nothing in {info.get('iters')} iterations")
    ov = np.abs(np.conj(v0) @ V) / (np.linalg.norm(v0) * np.linalg.norm(V, axis=0))
    j = int(np.argmax(np.where(ok, ov, -1.0)))
    res = float(pencil_residual(K, M, lam_f[j], V[:, j])[0])
    return Refinement(complex(lam_f[j]), V[:, j], float(ov[j]), res,
                      int(info.get("iters", 0)), int(info.get("nconv", 0)))


# -------------------------------------------------------------- continuation

@dataclass
class ContinuationPath:
    values: List[float] = field(default_factory=list)
    lam: List[complex] = field(default_factory=list)
    overlap: List[float] = field(default_factory=list)
    residual: List[float] = field(default_factory=list)
    lost: List[bool] = field(default_factory=list)
    vectors: List[np.ndarray] = field(default_factory=list)

    def __str__(self) -> str:
        rows = [f"{'value':>12} {'lambda':>28} {'overlap':>8} {'resid':>8}"]
        for v, l, o, r, lost in zip(self.values, self.lam, self.overlap, self.residual, self.lost):
            rows.append(f"{v:12.5g} {l.real:+13.6e}{l.imag:+13.6e}j {o:8.4f} {r:8.1e}"
                        + (" lost" if lost else ""))
        return "\n".join(rows)


def continuation(build: Callable, values: Sequence[float], sigma0: complex, k: int = 6,
                 W=None, method: str = "arpack", factor: str = "scipy", shift_offset: float = 1e-4,
                 min_overlap: float = 0.5, keep_vectors: bool = False,
                 **solver_options) -> ContinuationPath:
    """Follow one mode along ``values`` (module docstring).  ``build(value)``
    returns a compiled problem or a pair (K, M); the first step takes the
    eigenvalue nearest ``sigma0``."""
    from spdee.analysis.modes import pencil_residual
    from spdee.analysis.weights import natural_weight
    path = ContinuationPath()
    x_prev, sigma = None, complex(sigma0)
    for value in values:
        built = build(value)
        if isinstance(built, tuple):
            K, M = built
            Wt = default_weight(M, K.shape[0]) if W is None else Weight(W, K.shape[0])
        else:
            K, M = built.assemble_evp_matrices()
            Wt = natural_weight(built, M) if W is None else Weight(W, K.shape[0])
        lam, V = shift_invert(K, M, sigma, k, method=method, factor=factor, return_vectors=True,
                              v0=x_prev, **solver_options)
        ok = np.isfinite(lam)
        if x_prev is None:
            j = int(np.argmin(np.where(ok, np.abs(lam - sigma), np.inf)))
            ov = 1.0
        else:
            nv = Wt.norm(V)
            ov_all = np.abs(np.conj(x_prev) @ Wt.apply(V))
            ov_all /= Wt.norm(x_prev) * np.where(nv > 0, nv, 1)
            ov_all = np.where(ok, ov_all, -1.0)
            j = int(np.argmax(ov_all))
            ov = float(ov_all[j])
        x = V[:, j]
        path.values.append(float(value))
        path.lam.append(complex(lam[j]))
        path.overlap.append(ov)
        path.residual.append(float(pencil_residual(K, M, lam[j], x)[0]))
        path.lost.append(bool(ov < min_overlap))
        if keep_vectors:
            path.vectors.append(x)
        x_prev = x
        sigma = lam[j] + shift_offset * (1.0 + abs(lam[j])) * (1.0 + 1.0j) / np.sqrt(2)
    return path


# ------------------------------------------------------------------ diagnosis

@dataclass
class Finding:
    cause: str
    evidence: str
    remedy: str


@dataclass
class Diagnosis:
    findings: List[Finding]
    facts: Dict[str, object]

    def __str__(self) -> str:
        if not self.findings:
            return "diagnose: nothing conspicuous; see facts"
        out = []
        for f in self.findings:
            out.append(f"- {f.cause}\n    evidence: {f.evidence}\n    remedy:   {f.remedy}")
        return "\n".join(out)


def diagnose(K, M, sigma: complex, lam=None, V=None, W=None, info: Optional[dict] = None,
             kappa_threshold: float = 1e3, cond_threshold: float = 1e13,
             dense_row_factor: float = 5.0, factor: str = "scipy") -> Diagnosis:
    """Likely causes of a stalled or slow solve at ``sigma`` (module docstring);
    ``lam`` / ``V`` are the eigenpairs the solve returned, ``info`` the
    solver's info dict (bbjd: nconv, iters)."""
    from spdee.analysis.conditioning import condition_report
    K = K.tocsr()
    Mm = None if M is None else M.tocsr()
    N = K.shape[0]
    facts: Dict[str, object] = {"N": N}
    findings: List[Finding] = []
    if info:
        facts.update({k: info[k] for k in ("nconv", "iters", "status") if k in info})
    P = _shifted(K, Mm, sigma)
    try:
        cond = condition_estimate(P, factor)
        eq = equilibrate(K, Mm, sigma, report=False)
        cond_eq = condition_estimate(_shifted(eq.K, eq.M, sigma), factor)
    except RuntimeError as exc:
        cond = cond_eq = np.inf
        facts["factorization_error"] = str(exc)
    facts["cond_1(K - sigma M)"] = cond
    facts["cond_1 equilibrated"] = cond_eq
    if cond_eq > cond_threshold:
        findings.append(Finding(
            "near-singular shift", f"cond_1(K - sigma M) = {cond_eq:.2e} after equilibration",
            "move sigma off the eigenvalue (a relative offset of 1e-4 suffices) or deflate the "
            "mode at sigma"))
    nnz_row = np.diff(K.indptr)
    massless = (np.asarray(abs(Mm).sum(axis=1)).ravel() == 0) if Mm is not None \
        else np.zeros(N, dtype=bool)
    facts["massless_rows"] = int(massless.sum())
    med = float(np.median(nnz_row[nnz_row > 0])) if (nnz_row > 0).any() else 1.0
    dense_rows = np.nonzero(massless & (nnz_row > dense_row_factor * med))[0]
    facts["dense_constraint_rows"] = dense_rows.tolist()
    if len(dense_rows):
        findings.append(Finding(
            "dense constraint row",
            f"rows {dense_rows.tolist()} have {nnz_row[dense_rows].tolist()} entries against a "
            f"median of {med:.0f} and no mass (an integral gauge)",
            "bbjd stalls on such pencils (NOTES.md of the BBJD library): use method='arpack', or "
            "deflate the gauge direction, or compile at m != 0 where no gauge row is written"))
    data = np.abs(K.data[K.data != 0])
    if Mm is not None and Mm.nnz:
        data = np.concatenate([data, np.abs(Mm.data[Mm.data != 0])])
    span = float(np.log10(data.max() / data.min())) if data.size else 0.0
    facts["entry_decades"] = span
    if span > 8 and cond > 1e3 * cond_eq:
        findings.append(Finding(
            "unequilibrated pencil",
            f"entries span {span:.1f} decades; cond_1(K - sigma M) {cond:.1e} -> {cond_eq:.1e} "
            f"when equilibrated",
            "equilibrate(K, M, sigma) before factoring; undo with Equilibration.vectors"))
    if lam is not None:
        lam = np.asarray(lam, dtype=complex)
        ok = np.isfinite(lam)
        facts["returned"] = int(ok.sum())
        if ok.sum() >= 2:
            d = np.abs(lam[ok] - sigma)
            order = np.argsort(d)
            near = lam[ok][order]
            spacing = np.min(np.abs(near[1:] - near[0]))
            facts["nearest_spacing"] = float(spacing)
            if np.min(d) < 1e-10 * (1 + abs(sigma)):
                findings.append(Finding(
                    "shift on an eigenvalue", f"|lambda - sigma| = {np.min(d):.1e}",
                    "offset sigma; the factorization of (K - sigma M) is singular there"))
            if V is not None:
                rep = condition_report(K, Mm, lam[ok], np.asarray(V)[:, ok], W=W, factor=factor,
                                       threshold=kappa_threshold)
                facts["kappa_nearest"] = float(rep.kappa[order[0]])
                facts["kappa_max"] = float(np.nanmax(rep.kappa))
                bad = rep.ill_conditioned
                if len(bad):
                    findings.append(Finding(
                        "ill-conditioned target",
                        f"kappa = {rep.kappa[bad].max():.2e} for {len(bad)} of the {ok.sum()} "
                        f"returned modes (threshold {kappa_threshold:g})",
                        "these modes are spurious or unresolved: deflate them (Deflation), refine "
                        "the grid or raise the stencil order, and identify the target by "
                        "structure (modes.identify)"))
                res = rep.resid_right
                if spacing < 1e3 * np.max(res):
                    findings.append(Finding(
                        "dense band",
                        f"nearest spacing {spacing:.2e} against residuals up to {np.max(res):.1e}",
                        "nearest-eigenvalue matching is not meaningful; identify by eigenvector "
                        "(modes.identify) and use the dense census as the completeness oracle"))
    if info and info.get("nconv", 1) == 0 and not findings:
        findings.append(Finding(
            "no convergence without a visible cause", f"info = {info}",
            "try a larger subspace (maxsub), the exact LU as preconditioner, or arpack"))
    return Diagnosis(findings=findings, facts=facts)
