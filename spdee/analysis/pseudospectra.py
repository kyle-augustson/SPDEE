"""Pseudospectra and resolvent norms of a pencil K x = lambda M x.

Resolvent.  The finite spectrum of the pencil is the spectrum of the
operator A = M^+ K on the dynamical unknowns (``spdee.analysis.weights``),
and its resolvent is

    R(z) = (K - z M)^(-1) M      (M = I for a standard problem: (K - z)^(-1)),

whose norm in the W inner product is

    ||R(z)||_W = sup_v ||R(z) v||_W / ||v||_W
               = sigma_max( W^(1/2) (K - zM)^(-1) M W^(+1/2) ),

W^(+1/2) the pseudo-inverse square root, zero on the unknowns without
weight.  For a singular M this is the response of the state, measured on
the dynamical unknowns, to a forcing of the dynamical equations; a forcing
along a discrete gradient is absorbed by the multiplier and gives zero
response, so on a pencil whose gradient and divergence are W-adjoint
(the staggered core) the quantity is the resolvent of the evolution
operator on the constraint manifold, and on the collocated core it also
sees the non-adjoint remainder.  For a standard problem it is the usual
resolvent norm.

Pseudospectrum.  sigma_eps = {z : ||R(z)||_W > 1/eps}, the set of
eigenvalues of every W-perturbation dK of K with ||dK||_W < eps (in the
operator sense, A -> A + M^+ dK).  A W-normal operator has
||R(z)||_W = 1/dist(z, spectrum) and its pseudospectra are unions of discs
of radius eps about the eigenvalues; a non-normal one has larger
pseudospectra.

Methods (``pseudospectrum(..., method=)``):

  'lu'        exact: one sparse LU of (K - zM) per grid point and a Lanczos
              iteration (ARPACK svds, k = 1) for the largest singular value
              of W^(1/2) (K - zM)^(-1) M W^(+1/2), each step one solve and one
              conjugate-transpose solve.  Any N the factorization affords.
  'dense'     exact for moderate N: the dense shift-invert transform
              S = (K - sigma M)^(-1) M is scaled to S_W = W^(1/2) S W^(+1/2)
              and put in Schur form S_W = Q T Q^H once; since
              R(z) = (I - (z - sigma) S)^(-1) S,
              ||R(z)||_W = ||(I - (z - sigma) T)^(-1) T||_2 is a Lanczos
              iteration on triangular solves per grid point, O(N^2) each.
              With ``restrict=True`` the Schur form is reordered so the
              finite spectrum comes first and only that block is used: the
              resolvent of the evolution operator restricted to its finite
              invariant subspace (the constraint manifold), in the W-norm.
  'projected' approximate for large N: a W-orthonormal basis V of a
              subspace (converged eigenvectors near the region, or any
              Krylov / shift-invert basis) gives the projected pencil
              K_V = V^H W K V, M_V = V^H W M V and the estimate
              ||(K_V - z M_V)^(-1) M_V||_2.  On an exactly invariant subspace
              this is the resolvent norm of the operator restricted to that
              subspace, hence a lower bound on ||R(z)||_W that is exact when
              the subspace carries the supremum; on a general subspace it is
              the Rayleigh-Ritz approximation with no bound.

Transient growth.  With omega = -i lambda and dt x = -A x on the finite
part (spdee's convention: Re lambda > 0 is damped), the growth half-plane
is Re z < 0 and the Kreiss constant is

    K = sup_{Re z < 0} (-Re z) ||R(z)||_W,

with sup_t ||exp(-tA)||_W >= K (and <= e N K by the Kreiss matrix
theorem); ``kreiss_constant`` takes the supremum over the grid points of a
pseudospectrum, a lower bound.  ``resolvent_along_line`` evaluates
||R(z)||_W on a user line, e.g. the imaginary axis (the response to
harmonic forcing at frequency omega = Im z) or a line Re z = -delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.solvers import factorize, _shifted
from spdee.analysis.weights import Weight, default_weight

__all__ = ["resolvent_norm", "pseudospectrum", "resolvent_along_line", "kreiss_constant",
           "Pseudospectrum", "SchurResolvent", "ProjectedResolvent"]


def _weight(W, M, N) -> Weight:
    Wt = default_weight(M, N) if W is None else Weight(W, N)
    if not Wt.is_diagonal:
        raise NotImplementedError("pseudospectra need a diagonal weight")
    return Wt


def _sigma_max(matvec, rmatvec, n: int, tol: float) -> float:
    if n <= 2:
        B = np.array([matvec(e) for e in np.eye(n, dtype=complex)]).T
        return float(np.linalg.norm(B, 2))
    op = spla.LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=complex)
    try:
        s = spla.svds(op, k=1, which="LM", tol=tol, return_singular_vectors=False)
        return float(s[0])
    except (spla.ArpackNoConvergence, spla.ArpackError):
        B = np.array([matvec(e) for e in np.eye(n, dtype=complex)]).T
        return float(np.linalg.norm(B, 2))


def resolvent_norm(K, M, z: complex, W=None, factor: str = "scipy", tol: float = 1e-6) -> float:
    """||(K - zM)^(-1) M||_W by one sparse LU and a Lanczos iteration."""
    K = K.tocsr()
    N = K.shape[0]
    Wt = _weight(W, M, N)
    Mm = None if M is None else M.tocsr()
    MH = None if Mm is None else Mm.conj().T.tocsr()
    try:
        fac = factorize(_shifted(K, Mm, z), factor)
    except RuntimeError as exc:
        if "singular" in str(exc).lower():
            return np.inf
        raise
    try:
        def mv(v):
            u = Wt.pinv_sqrt_apply(v)
            return Wt.sqrt_apply(fac.solve(u if Mm is None else Mm @ u))

        def rmv(v):
            u = fac.solve_H(Wt.sqrt_apply(v))
            return Wt.pinv_sqrt_apply(u if MH is None else MH @ u)
        return _sigma_max(mv, rmv, N, tol)
    finally:
        fac.close()


class SchurResolvent:
    """The dense route: S_W = W^(1/2) (K - sigma M)^(-1) M W^(+1/2) in Schur
    form, ``norm(z)`` = ||(I - (z - sigma) T)^(-1) T||_2."""

    def __init__(self, K, M, sigma: complex, W=None, restrict: bool = False, mu_min: float = 1e-8):
        K = K.tocsr()
        N = K.shape[0]
        Wt = _weight(W, M, N)
        Mm = sp.identity(N, dtype=complex, format="csr") if M is None else M.tocsr()
        fac = factorize(_shifted(K, Mm, sigma), "scipy")
        try:
            MD = Mm @ sp.diags(Wt.pinv_sqrt_apply(np.ones(N)))
            S = fac.solve(np.asfortranarray(MD.toarray()))
        finally:
            fac.close()
        S = Wt.sqrt_apply(S)
        self.sigma, self.N = complex(sigma), N
        if restrict:
            T, Q, ndim = sla.schur(S, output="complex", sort=lambda x: abs(x) > mu_min)
            T, Q = T[:ndim, :ndim], Q[:, :ndim]
        else:
            T, Q = sla.schur(S, output="complex")
        self.T, self.Q = T, Q
        self.mu = np.diag(T).copy()
        self.lam = self.sigma + 1.0 / np.where(self.mu == 0, np.nan, self.mu)

    def norm(self, z: complex, tol: float = 1e-6) -> float:
        T = self.T
        n = T.shape[0]
        if n == 0:
            return 0.0
        A = np.eye(n, dtype=complex) - (z - self.sigma) * T

        def mv(v):
            return sla.solve_triangular(A, T @ v, check_finite=False)

        def rmv(v):
            return np.conj(T).T @ sla.solve_triangular(A, v, trans="C", check_finite=False)
        return _sigma_max(mv, rmv, n, tol)


class ProjectedResolvent:
    """The subspace route: K_V = V^H W K V, M_V = V^H W M V on a W-orthonormal
    basis of V; ``norm(z)`` = ||(K_V - z M_V)^(-1) M_V||_2."""

    def __init__(self, K, M, V: np.ndarray, W=None):
        K = K.tocsr()
        N = K.shape[0]
        Wt = _weight(W, M, N)
        Q, _ = Wt.orthonormalize(np.asarray(V, dtype=complex))
        WQ = Wt.apply(Q)
        self.KV = np.conj(WQ).T @ (K @ Q)
        self.MV = np.conj(WQ).T @ (Q if M is None else M @ Q)
        self.Q = Q
        self.lam = sla.eigvals(self.KV, self.MV)

    def norm(self, z: complex) -> float:
        A = self.KV - z * self.MV
        try:
            return float(np.linalg.norm(np.linalg.solve(A, self.MV), 2))
        except np.linalg.LinAlgError:
            return np.inf


@dataclass
class Pseudospectrum:
    re: np.ndarray                   # grid abscissae (Re z)
    im: np.ndarray                   # grid ordinates (Im z)
    resolvent: np.ndarray            # ||R(z)||_W on the grid, shape (len(im), len(re))
    method: str
    lam: Optional[np.ndarray] = None

    @property
    def log10_inverse(self) -> np.ndarray:
        """log10 of eps(z) = 1 / ||R(z)||_W; the eps-pseudospectrum is the
        region below log10(eps)."""
        with np.errstate(divide="ignore"):
            return -np.log10(self.resolvent)

    def pseudospectral_abscissa(self, eps: float) -> float:
        """sup{-Re z : ||R(z)||_W > 1/eps} on the grid (the growth rate the
        eps-perturbed operator can reach), -inf when the set is empty."""
        Z = self.re[None, :] + 0j * self.im[:, None]
        mask = self.resolvent > 1.0 / eps
        return float(np.max(-Z.real[mask])) if mask.any() else -np.inf

    def plot(self, filename: str, levels: Sequence[float] = (1e-1, 1e-2, 1e-3, 1e-4),
             title: str = "", omega: bool = False):
        """Contours log10(eps) of the pseudospectrum saved to ``filename``;
        ``omega`` draws the omega = -i lambda plane instead."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        X, Y = np.meshgrid(self.re, self.im)
        Zv = self.log10_inverse
        if omega:
            X, Y = Y, -X
        fig, ax = plt.subplots(figsize=(6, 5))
        lv = sorted(np.log10(levels))
        cs = ax.contour(X, Y, Zv, levels=lv, cmap="viridis")
        ax.clabel(cs, fmt=lambda v: f"1e{v:.0f}", fontsize=8)
        ax.contourf(X, Y, Zv, levels=lv + [max(lv) + 1], alpha=0.15, cmap="viridis")
        if self.lam is not None:
            lam = self.lam[np.isfinite(self.lam)]
            px, py = (lam.imag, -lam.real) if omega else (lam.real, lam.imag)
            ax.plot(px, py, "k.", ms=4)
        ax.set_xlabel("Re omega" if omega else "Re lambda")
        ax.set_ylabel("Im omega" if omega else "Im lambda")
        ax.set_xlim(X.min(), X.max())
        ax.set_ylim(Y.min(), Y.max())
        ax.set_title(title or f"pseudospectrum ({self.method})")
        fig.tight_layout()
        fig.savefig(filename, dpi=130)
        plt.close(fig)
        return filename


def pseudospectrum(K, M, re: Sequence[float], im: Sequence[float], W=None, method: str = "lu",
                   sigma: Optional[complex] = None, V: Optional[np.ndarray] = None,
                   factor: str = "scipy", restrict: bool = False, lam=None,
                   tol: float = 1e-6) -> Pseudospectrum:
    """||R(z)||_W on the grid z = re + i im by the named method (module docstring)."""
    re = np.asarray(re, float)
    im = np.asarray(im, float)
    out = np.empty((len(im), len(re)))
    if method == "lu":
        for a, y in enumerate(im):
            for b, x in enumerate(re):
                out[a, b] = resolvent_norm(K, M, complex(x, y), W, factor=factor, tol=tol)
    elif method == "dense":
        if sigma is None:
            sigma = complex(re.mean(), im.mean()) + 1e-3j * (im.max() - im.min() + 1.0)
        sr = SchurResolvent(K, M, sigma, W, restrict=restrict)
        for a, y in enumerate(im):
            for b, x in enumerate(re):
                out[a, b] = sr.norm(complex(x, y), tol)
        if lam is None:
            lam = sr.lam
    elif method == "projected":
        if V is None:
            raise ValueError("method='projected' needs the subspace V")
        pr = ProjectedResolvent(K, M, V, W)
        for a, y in enumerate(im):
            for b, x in enumerate(re):
                out[a, b] = pr.norm(complex(x, y))
        if lam is None:
            lam = pr.lam
    else:
        raise ValueError("method must be 'lu', 'dense' or 'projected'")
    return Pseudospectrum(re=re, im=im, resolvent=out, method=method,
                          lam=None if lam is None else np.asarray(lam, dtype=complex))


def resolvent_along_line(K, M, points: Sequence[complex], W=None, method: str = "lu",
                         sigma: Optional[complex] = None, V=None, factor: str = "scipy",
                         restrict: bool = False) -> np.ndarray:
    """||R(z)||_W at each z of ``points``."""
    points = np.asarray(points, dtype=complex)
    if method == "lu":
        return np.array([resolvent_norm(K, M, z, W, factor=factor) for z in points])
    if method == "dense":
        if sigma is None:
            sigma = points.mean() + 1e-3j * (np.ptp(points.imag) + 1.0)
        sr = SchurResolvent(K, M, sigma, W, restrict=restrict)
        return np.array([sr.norm(z) for z in points])
    if method == "projected":
        pr = ProjectedResolvent(K, M, V, W)
        return np.array([pr.norm(z) for z in points])
    raise ValueError("method must be 'lu', 'dense' or 'projected'")


def kreiss_constant(ps: Pseudospectrum) -> float:
    """sup over the grid points with Re z < 0 of (-Re z) ||R(z)||_W, a lower
    bound on the Kreiss constant and on the transient growth
    sup_t ||exp(-tA)||_W; zero when the grid has no such point."""
    X = ps.re[None, :] + 0.0 * ps.im[:, None]
    mask = X < 0
    if not mask.any():
        return 0.0
    return float(np.max((-X * ps.resolvent)[mask]))
