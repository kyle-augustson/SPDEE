"""Eigenvalue condition numbers and sensitivities of a pencil K x = lambda M x.

Left eigenvectors.  y is a left eigenvector when y^H K = lambda y^H M, i.e.
K^H y = conj(lambda) M^H y.  ``left_eigenvectors`` computes y by inverse
iteration on the adjoint of the shifted pencil at the computed eigenvalue,

    y <- (K - z M)^(-H) M^H y,   z = lambda (1 + delta) + delta,

two steps from the right eigenvector (a normal pencil has y = x), one
sparse LU per mode; ``two_sided_dense`` takes both vectors from a dense
two-sided eigensolve of the shift-invert transform S = (K - sigma M)^(-1) M
(y = (K - sigma M)^(-H) y_S for a left eigenvector y_S of S).

Condition number.  In the W inner product on the dynamical unknowns
(``spdee.analysis.weights``) the operator is A = M^+ K; the eigenvector of
its W-adjoint for conj(lambda) is u = W^+ M^H y, and <u, x>_W = y^H M x.
The generalized Wilkinson condition number of a simple eigenvalue is

    kappa(lambda) = ||x||_W ||u||_W / |y^H M x|,
    ||u||_W^2 = y^H M W^+ M^H y,

which is ||x||_M ||y||_M / |y^H M x| in the M-inner product (W = M
positive semidefinite) and ||x||_2 ||y||_2 / |y^H x| for a standard
problem.  kappa = 1 for every eigenvalue of a W-normal operator (a
Hermitian pencil in the M-norm), and a perturbation of the pencil with
||dK||_W <= eps, ||dM||_W <= eps moves lambda by at most
kappa (1 + |lambda|) eps to first order.  For a singular M the constraint
unknowns carry no weight, and kappa is the condition number of the finite
eigenvalue on the constraint manifold.

Sensitivity.  For a perturbation (dK, dM) of the pencil the first-order
change of a simple eigenvalue is

    dlambda = y^H (dK - lambda dM) x / (y^H M x),

independent of W; with (dK, dM) the difference of two compiled pencils at
parameter values p and p + dp this is (dlambda/dp) dp, the parameter
sensitivity of the mode.

Eigenvector angles.  cos(theta_ij) = |<x_i, x_j>_W| / (||x_i||_W ||x_j||_W);
the departure from W-orthogonality is the largest off-diagonal cosine and
the condition number of the W-Gram matrix X^H W X (1 for a W-orthonormal
set, infinite for a defective pair).

``condition_report`` ranks the modes by kappa and flags those above a
threshold as ill-conditioned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp

from spdee.solvers import factorize, _shifted
from spdee.analysis.weights import Weight, default_weight

__all__ = ["left_eigenvectors", "two_sided_dense", "condition_numbers", "sensitivity",
           "eigenvector_angles", "ConditioningReport", "condition_report"]


def _as_weight(W, M, N) -> Weight:
    if W is None:
        return default_weight(M, N)
    return Weight(W, N)


def _apply_M(M, x):
    return x if M is None else M @ x


def _apply_MH(M, y):
    return y if M is None else M.conj().T @ y


def left_eigenvectors(K, M, lam: Sequence[complex], X: Optional[np.ndarray] = None,
                      factor: str = "scipy", iterations: int = 2, delta: float = 1e-9,
                      refine_right: bool = False, seed: int = 0):
    """Left eigenvectors of the pairs (lam, X) by inverse iteration on the
    adjoint pencil (module docstring).  Returns (Y, X, resid_left) with the
    columns of Y normalized in the 2-norm and resid_left the relative
    residual ||K^H y - conj(lambda) M^H y|| / (||K^H y|| + |lambda| ||M^H y||);
    ``refine_right`` also applies the iteration to X.  ``factor`` must
    provide a conjugate-transpose solve ('scipy' or 'lapack')."""
    K = K.tocsr()
    M = None if M is None else M.tocsr()
    N = K.shape[0]
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    rng = np.random.default_rng(seed)
    if X is None:
        X = rng.standard_normal((N, len(lam))) + 1j * rng.standard_normal((N, len(lam)))
    X = np.array(X, dtype=complex, copy=True)
    if X.ndim == 1:
        X = X[:, None]
    Y = np.empty_like(X)
    resid = np.empty(len(lam))
    KH = K.conj().T.tocsr()
    MH = None if M is None else M.conj().T.tocsr()
    for j, lj in enumerate(lam):
        z = lj * (1.0 + delta) + delta
        fac = factorize(_shifted(K, M, z), factor)
        try:
            y = X[:, j].copy()
            for _ in range(iterations):
                y = fac.solve_H(_apply_MH(M, y))
                y /= np.linalg.norm(y)
            if refine_right:
                x = X[:, j]
                for _ in range(iterations):
                    x = fac.solve(_apply_M(M, x))
                    x /= np.linalg.norm(x)
                X[:, j] = x
        finally:
            fac.close()
        Y[:, j] = y
        KHy = KH @ y
        MHy = y if MH is None else MH @ y
        num = np.linalg.norm(KHy - np.conj(lj) * MHy)
        resid[j] = num / (np.linalg.norm(KHy) + abs(lj) * np.linalg.norm(MHy) or 1.0)
    return Y, X, resid


def two_sided_dense(K, M, sigma: complex = 0.0, mu_min: float = 1e-8):
    """Every finite eigenvalue with its right and left eigenvectors from a
    dense two-sided eigensolve of S = (K - sigma M)^(-1) M; the ker(M)
    images at |mu| <= mu_min are dropped.  Returns (lam, X, Y)."""
    import scipy.linalg as sla
    K = K.tocsr()
    N = K.shape[0]
    Mm = sp.identity(N, dtype=complex, format="csr") if M is None else M.tocsr()
    fac = factorize(_shifted(K, Mm, sigma), "scipy")
    try:
        S = fac.solve(np.asfortranarray(Mm.toarray()))
        mu, YS, XS = sla.eig(S, left=True, right=True)
        keep = np.abs(mu) > mu_min
        lam = sigma + 1.0 / mu[keep]
        X = XS[:, keep]
        Y = fac.solve_H(YS[:, keep])
    finally:
        fac.close()
    Y /= np.linalg.norm(Y, axis=0)
    X /= np.linalg.norm(X, axis=0)
    return lam, X, Y


def condition_numbers(K, M, X: np.ndarray, Y: np.ndarray, W=None) -> np.ndarray:
    """kappa(lambda_j) = ||x_j||_W ||u_j||_W / |y_j^H M x_j| per column pair."""
    N = X.shape[0]
    Wt = _as_weight(W, M, N)
    MX = _apply_M(M, X)
    MHY = _apply_MH(M, Y)
    nx = Wt.norm(X)
    nu = np.sqrt(np.abs(np.einsum("ij,ij->j", np.conj(MHY), Wt.pinv_apply(MHY))))
    denom = np.abs(np.einsum("ij,ij->j", np.conj(Y), MX))
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 0, nx * nu / np.where(denom > 0, denom, 1.0), np.inf)
    return kappa


def sensitivity(K, M, lam, X: np.ndarray, Y: np.ndarray, dK, dM=None) -> np.ndarray:
    """dlambda_j = y_j^H (dK - lambda_j dM) x_j / (y_j^H M x_j) for a pencil
    perturbation (dK, dM); dM None means no mass perturbation."""
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    if X.ndim == 1:
        X, Y = X[:, None], Y[:, None]
    dKX = dK @ X
    if dM is not None:
        dKX = dKX - (dM @ X) * lam[None, :]
    num = np.einsum("ij,ij->j", np.conj(Y), dKX)
    den = np.einsum("ij,ij->j", np.conj(Y), _apply_M(M, X))
    return num / den


def eigenvector_angles(X: np.ndarray, W=None, M=None):
    """(cosines, gram_condition): the matrix of |cos theta_ij| in the W inner
    product and the 2-norm condition number of the W-Gram matrix."""
    N = X.shape[0]
    Wt = _as_weight(W, M, N)
    G = np.conj(X).T @ Wt.apply(X)
    d = np.sqrt(np.abs(np.diag(G)))
    d = np.where(d == 0, 1.0, d)
    C = np.abs(G) / np.outer(d, d)
    s = np.linalg.svd(G, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else np.inf
    return C, cond


@dataclass
class ConditioningReport:
    lam: np.ndarray
    kappa: np.ndarray
    resid_right: np.ndarray
    resid_left: np.ndarray
    order: np.ndarray            # indices sorted by decreasing kappa
    threshold: float
    max_cosine: float
    gram_condition: float
    X: Optional[np.ndarray] = None
    Y: Optional[np.ndarray] = None

    @property
    def ill_conditioned(self) -> np.ndarray:
        """Indices with kappa above the threshold, worst first."""
        return self.order[self.kappa[self.order] > self.threshold]

    def __str__(self) -> str:
        rows = [f"{'rank':>4} {'index':>5} {'lambda':>28} {'kappa':>10} {'res_x':>8} {'res_y':>8}"]
        for rank, j in enumerate(self.order):
            flag = " *" if self.kappa[j] > self.threshold else ""
            rows.append(f"{rank:4d} {j:5d} {self.lam[j].real:+13.6e}{self.lam[j].imag:+13.6e}j "
                        f"{self.kappa[j]:10.3e} {self.resid_right[j]:8.1e} "
                        f"{self.resid_left[j]:8.1e}{flag}")
        rows.append(f"* kappa > {self.threshold:g}: {len(self.ill_conditioned)} of {len(self.lam)};"
                    f" max |cos| between eigenvectors {self.max_cosine:.3f},"
                    f" Gram condition {self.gram_condition:.3e}")
        return "\n".join(rows)


def condition_report(K, M, lam, X, W=None, threshold: float = 100.0, factor: str = "scipy",
                     Y: Optional[np.ndarray] = None, keep_vectors: bool = False):
    """Rank the eigenpairs (lam, X) by kappa; the left vectors are computed
    by ``left_eigenvectors`` unless given."""
    from spdee.analysis.modes import pencil_residual
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    X = np.asarray(X, dtype=complex)
    if X.ndim == 1:
        X = X[:, None]
    finite = np.isfinite(lam)
    if Y is None:
        Y = np.zeros_like(X)
        resid_left = np.full(len(lam), np.nan)
        if finite.any():
            Yf, _, rl = left_eigenvectors(K, M, lam[finite], X[:, finite], factor=factor)
            Y[:, finite] = Yf
            resid_left[finite] = rl
    else:
        Y = np.asarray(Y, dtype=complex)
        KH = K.conj().T
        MH = None if M is None else M.conj().T
        resid_left = pencil_residual(KH, MH, np.conj(lam), Y)
    kappa = np.full(len(lam), np.nan)
    kappa[finite] = condition_numbers(K, M, X[:, finite], Y[:, finite], W)
    resid_right = pencil_residual(K, M, lam, X)
    order = np.argsort(-np.where(finite, kappa, -np.inf))
    if finite.sum() > 1:
        C, gram = eigenvector_angles(X[:, finite], W, M)
        max_cos = float(np.max(C - np.eye(C.shape[0])))
    else:
        max_cos, gram = 0.0, 1.0
    return ConditioningReport(lam=lam, kappa=kappa, resid_right=resid_right, resid_left=resid_left,
                              order=order, threshold=threshold, max_cosine=max_cos,
                              gram_condition=gram, X=X if keep_vectors else None,
                              Y=Y if keep_vectors else None)
