"""Non-normality measures of the finite part of a pencil K x = lambda M x.

Reduced operator.  In the W inner product (``spdee.analysis.weights``,
diagonal W) the shift-invert transform S = (K - sigma M)^(-1) M becomes
S_W = W^(1/2) S W^(+1/2); its Schur form is reordered so the finite
spectrum (|mu| > mu_min, mu = 1/(lambda - sigma)) leads,
S_W = Q T Q^H with T_11 (k x k) the finite block.  On that invariant
subspace S is invertible and the evolution operator of the pencil is

    A = sigma I + T_11^(-1)      (k x k, upper triangular, eigenvalues lambda),

the operator M^+ K restricted to the finite invariant subspace (the
constraint manifold), expressed in the W-orthonormal basis Q_1: a
similarity by a W-unitary map, so every unitarily invariant quantity
below is the W-norm quantity of the pencil.  ``reduced_operator`` returns
A, T_11 and the basis.

Henrici's departure from normality, in the Frobenius norm,

    dep_F(A) = sqrt(||A||_F^2 - sum_j |lambda_j|^2) = ||strict upper part of A||_F,

zero iff A is normal (W-normal for the pencil), reported both for A and
for the shift-inverted operator T_11 (``dep_F(T_11)``, which weights the
modes near sigma), and relative to ||A||_F.

Numerical abscissa.  For dt x = -A x on the finite part (omega = -i lambda,
growth rate -Re lambda) the numerical abscissa is

    omega_num = max eig( (-A - A^H)/2 ) = max_x Re <x, -A x>_W / <x, x>_W,

the largest instantaneous growth rate d/dt ln ||x||_W at t = 0; the
spectral abscissa alpha = max_j (-Re lambda_j) is the asymptotic rate.
omega_num >= alpha, with equality for a normal operator; the difference
is the potential for transient growth, ||exp(-tA)||_W <= exp(omega_num t).

Numerical range (field of values) F(A) = {<x, A x>_W / <x, x>_W}: its
boundary is sampled by the eigenvectors of the largest eigenvalue of the
Hermitian parts (e^{i theta} A + e^{-i theta} A^H)/2 over theta in [0, 2 pi),
each giving the boundary point x^H A x (Johnson's algorithm).  F(A)
contains the convex hull of the spectrum, with equality iff A is normal.

Per-mode non-normality.  ||R(z)||_W >= 1/dist(z, spectrum) with equality
everywhere iff A is W-normal; around lambda_j the excess

    c_j = max over |z - lambda_j| = rho_j of ||R(z)||_W dist(z, spectrum),

rho_j a fraction of the distance to the nearest other eigenvalue, measures
how much of the resolvent norm near lambda_j is beyond the normal value
1/dist: c_j = 1 for a mode of a normal operator, and c_j -> kappa(lambda_j)
as rho_j -> 0 for a simple eigenvalue.  Computed from the same reduced
operator when one is given, else by ``pseudospectra.resolvent_norm``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg as sla

from spdee.analysis.pseudospectra import SchurResolvent, resolvent_norm

__all__ = ["ReducedOperator", "reduced_operator", "henrici", "numerical_abscissa",
           "numerical_range", "mode_nonnormality"]


@dataclass
class ReducedOperator:
    A: np.ndarray          # sigma I + T11^{-1}, upper triangular, eigenvalues lam
    T: np.ndarray          # the finite Schur block of the shift-inverted operator
    Q: np.ndarray          # W-orthonormal basis of the finite invariant subspace (W-coordinates)
    lam: np.ndarray
    sigma: complex

    @property
    def k(self) -> int:
        return self.A.shape[0]


def reduced_operator(K, M, sigma: complex, W=None, mu_min: float = 1e-8) -> ReducedOperator:
    """The evolution operator on the finite invariant subspace in a
    W-orthonormal basis (module docstring)."""
    sr = SchurResolvent(K, M, sigma, W, restrict=True, mu_min=mu_min)
    T = sr.T
    k = T.shape[0]
    Tinv = sla.solve_triangular(T, np.eye(k, dtype=complex), check_finite=False)
    A = sigma * np.eye(k, dtype=complex) + Tinv
    return ReducedOperator(A=A, T=T, Q=sr.Q, lam=np.diag(A).copy(), sigma=complex(sigma))


def henrici(A: np.ndarray) -> dict:
    """{'departure': dep_F(A), 'relative': dep_F(A)/||A||_F, 'frobenius': ||A||_F}
    for a square matrix (its Schur form is taken when A is not triangular)."""
    A = np.asarray(A, dtype=complex)
    if not np.allclose(A, np.triu(A)):
        A = sla.schur(A, output="complex")[0]
    fro = float(np.linalg.norm(A))
    dep = float(np.linalg.norm(np.triu(A, 1)))
    return {"departure": dep, "relative": dep / (fro or 1.0), "frobenius": fro}


def numerical_abscissa(A: np.ndarray) -> dict:
    """{'numerical': omega_num, 'spectral': alpha, 'excess': omega_num - alpha}
    for dt x = -A x."""
    A = np.asarray(A, dtype=complex)
    H = -(A + np.conj(A).T) / 2
    w = sla.eigvalsh(H)
    lam = np.linalg.eigvals(A) if not np.allclose(A, np.triu(A)) else np.diag(A)
    alpha = float(np.max(-lam.real))
    return {"numerical": float(w[-1]), "spectral": alpha, "excess": float(w[-1] - alpha)}


def numerical_range(A: np.ndarray, n_angles: int = 90) -> np.ndarray:
    """Boundary points of the field of values of A (Johnson's algorithm)."""
    A = np.asarray(A, dtype=complex)
    pts = []
    for th in np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False):
        H = (np.exp(1j * th) * A + np.exp(-1j * th) * np.conj(A).T) / 2
        w, V = sla.eigh(H)
        v = V[:, -1]
        pts.append(np.vdot(v, A @ v))
    return np.array(pts)


def mode_nonnormality(K, M, lam, W=None, reduced: Optional[ReducedOperator] = None,
                      lam_all=None, fraction: float = 0.25, n_points: int = 8,
                      factor: str = "scipy") -> np.ndarray:
    """c_j for each eigenvalue in ``lam`` (module docstring); ``lam_all`` is
    the spectrum the distances are taken to (default: ``lam`` itself, or the
    reduced operator's)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=complex))
    if lam_all is None:
        lam_all = reduced.lam if reduced is not None else lam
    lam_all = np.asarray(lam_all, dtype=complex)
    lam_all = lam_all[np.isfinite(lam_all)]
    sr = None
    if reduced is not None:
        sr = SchurResolvent.__new__(SchurResolvent)
        sr.T, sr.sigma = reduced.T, reduced.sigma
    out = np.empty(len(lam))
    for j, lj in enumerate(lam):
        others = lam_all[np.abs(lam_all - lj) > 1e-12 * (1 + abs(lj))]
        gap = np.min(np.abs(others - lj)) if len(others) else 1.0
        rho = fraction * gap
        best = 0.0
        for th in np.linspace(0.0, 2 * np.pi, n_points, endpoint=False):
            z = lj + rho * np.exp(1j * th)
            rn = sr.norm(z) if sr is not None else resolvent_norm(K, M, z, W, factor=factor)
            best = max(best, rn * np.min(np.abs(lam_all - z)))
        out[j] = best
    return out
