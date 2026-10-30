"""The generalized-eigenvalue solver layer.

Every backend compiles to a sparse pencil ``K x = lambda M x`` (``M`` may be
None for a standard problem).  Two entry points:

``shift_invert(K, M, sigma, k, method=, factor=)``
    The k eigenvalues nearest sigma.  ``method`` is the outer eigensolver,
    ``factor`` the inner factorization of (K - sigma M):

      method   'arpack'  scipy ARPACK on OP = (K - sigma M)^-1 M
               'bbjd'    Jacobi-Davidson QZ with (K - sigma M)^-1 as preconditioner;
                         the BBJD library when SPDEE_BBJD_PATH names a build
      factor   'scipy'   SuperLU
               'lapack'  dense LU on the host, for pencils whose fill is dense anyway
               'fplus'   FPLUS: int64 LU, selectable pivoting, Ruiz scaling; the
                         library build that SPDEE_FPLUS_PATH names
                         (spdee.solvers.external)
               'cudss'   NVIDIA cuDSS through nvmath-python (CUDA only)
               'cupy'    cupyx splu: SuperLU on the host, triangular solves on the
                         device, measured 33x slower than 'scipy'; explicit only
               'auto'    cudss on a discrete NVIDIA GPU, else scipy

``dense(K, M, sigma, backend=)``
    The full spectrum through the shift-invert transform: LAPACK geev on
    host or cuSOLVER Xgeev on the GPU.  Returns a ``CensusResult``.

Nothing here knows about omega; the caller maps lambda to omega.
``available()`` reports the optional components present.  FPLUS and BBJD
are separate builds of their libraries (``spdee.solvers.external``).  A GPU factor or backend that the installed
stack cannot provide (no device, the ROCm build, no nvmath, or
``SPDEE_NO_GPU=1``) degrades to the host path with one
``gpu.GPUFallbackWarning`` naming the missing capability; a GPU path that
is present but fails numerically raises, so 'auto' can move on.
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from spdee.solvers import gpu

__all__ = ["shift_invert", "dense", "factorize", "available", "gpu_factor_order", "Factor",
           "METHODS", "FACTORS"]

METHODS = ("arpack", "bbjd")
FACTORS = ("scipy", "lapack", "fplus", "cudss", "cupy", "auto")


class Factor:
    """Uniform wrapper: ``solve(b)`` solves P x = b, ``solve_H(b)`` solves P^H x = b
    where supported, ``close()`` releases memory."""

    def __init__(self, impl, kind: str, solve, solve_H=None, close=None):
        self._impl = impl
        self.kind = kind
        self._solve = solve
        self._solve_H = solve_H
        self._close = close
        self.library = None

    def solve(self, b):
        return self._solve(b)

    def solve_H(self, b):
        if self._solve_H is None:
            raise NotImplementedError(f"factor {self.kind!r} has no conjugate-transpose solve")
        return self._solve_H(b)

    def close(self):
        if self._close is not None:
            self._close()


def _factor_scipy(P: sp.spmatrix, **kw) -> Factor:
    lu = spla.splu(P.tocsc())
    return Factor(lu, "scipy", lu.solve, lambda b: lu.solve(b, trans="H"))


def _factor_lapack(P: sp.spmatrix, **kw) -> Factor:
    """Dense LU of the full matrix (LAPACK getrf, threaded through BLAS):
    N^2 memory and N^3 work, which a sparse LU also reaches when a
    non-constant coefficient couples every angular degree."""
    import scipy.linalg as sla
    lu = sla.lu_factor(P.toarray(), overwrite_a=True, check_finite=False)
    return Factor(lu, "lapack", lambda b: sla.lu_solve(lu, b, check_finite=False),
                  lambda b: sla.lu_solve(lu, b, trans=2, check_finite=False))


def _factor_fplus(P: sp.spmatrix, library: str = "auto", pivot_strategy: str = "partial",
                  colamd: bool = True, pin_dense_rows: bool = True, equilibrate=True,
                  verbose: int = 0, check: Optional[bool] = None, check_tol: float = 1e-11, **kw) -> Factor:
    """FPLUS: the library build that ``SPDEE_FPLUS_PATH`` / ``FPLUS_PATH``
    names; see ``spdee.solvers.external``.

    It takes its own options through ``**kw``: relax_rows
    and panel (default -1, the automatic rule; the meridional MHD pencils of
    the solar paper used 32 and 512), pivot_rel (default 0: only an exactly
    zero pivot is singular, because shift-invert drivers place the shift on
    converged eigenvalues), ordering, postorder, and gpu=True (cuBLAS panel
    updates on a device with coherent host memory, e.g. the GH200).  With
    pivot_rel 0 the factorization is gated instead: both solve directions
    on one random vector must have backward error below ``check_tol``
    (``check`` defaults to on)."""
    from spdee.solvers.external import load_fplus
    SparseLU, which, _ = load_fplus(library)
    opts = dict(colamd=colamd, pin_dense_rows=pin_dense_rows, pivot_strategy=pivot_strategy,
                equilibrate=equilibrate, verbose=verbose)
    opts.update(relax_rows=-1, panel=-1, pivot_rel=0.0)
    opts.update(kw)
    A = P.tocsc()
    lu = SparseLU(A, **opts)
    fac = Factor(lu, "fplus", lambda b: lu.solve(b, trans="N"), lambda b: lu.solve(b, trans="H"))
    fac.library = which
    if check is not False:
        rng = np.random.default_rng(0)
        n = P.shape[0]
        b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        nP = spla.norm(A, np.inf)
        for name, solve, op in (("forward", fac.solve, A), ("conjugate-transposed", fac.solve_H,
                                                             A.conj().T)):
            x = solve(b)
            be = np.linalg.norm(op @ x - b, np.inf) / (nP * np.linalg.norm(x, np.inf)
                                                       + np.linalg.norm(b, np.inf))
            if not np.isfinite(be) or be > check_tol:
                raise RuntimeError(f"FPLUS ({which}) {name} solve failed the backward-error gate "
                                   f"({be:.2e} > {check_tol:.0e})")
    return fac


def _factor_gpu(P: sp.spmatrix, backend: str, check: bool = True, check_tol: float = 1e-8,
                **kw) -> Factor:
    """GPU factorization verified on one random right-hand side, because
    cuDSS can report success and still return wrong solves on pencils with
    zero diagonal entries; a failed check raises so 'auto' moves on."""
    from spdee.solvers import gpu_si
    fac = gpu_si.SparseFactor(P, None, backend=backend).factor(0.0)
    if check:
        rng = np.random.default_rng(0)
        b = rng.standard_normal(P.shape[0]) + 1j * rng.standard_normal(P.shape[0])
        x = fac.solve(b)
        res = np.linalg.norm(P @ x - b) / np.linalg.norm(b)
        if not np.isfinite(res) or res > check_tol:
            fac.close()
            raise gpu_si.BackendUnavailable(f"{fac.backend_used} factorization failed the residual "
                                            f"gate ({res:.2e} > {check_tol:.0e})")
    return Factor(fac, fac.backend_used, fac.solve, None, fac.close)


_GPU_FACTOR_FEATURE = {"cudss": "cudss", "cupy": "sparse_lu"}


def gpu_factor_order() -> Tuple[str, ...]:
    """The GPU factors 'auto' tries before SuperLU: cuDSS on a discrete
    NVIDIA GPU, nothing otherwise ('cupy' factors on the host and solves
    slower than SuperLU, so it is never chosen automatically)."""
    if gpu.prefer_device()[0] and gpu.features()["cudss"][0]:
        return ("cudss",)
    return ()


def factorize(P: sp.spmatrix, factor: str = "scipy", **kw) -> Factor:
    """Factor the shifted matrix P with the named backend."""
    if factor == "scipy":
        return _factor_scipy(P, **kw)
    if factor == "lapack":
        return _factor_lapack(P, **kw)
    if factor == "fplus":
        return _factor_fplus(P, **kw)
    if factor in _GPU_FACTOR_FEATURE:
        ok, why = gpu.features()[_GPU_FACTOR_FEATURE[factor]]
        if not ok:
            gpu.fallback(f"factor={factor!r}", why, "factoring with factor='scipy' (SuperLU)")
            return _factor_scipy(P, **kw)
        return _factor_gpu(P, factor, **kw)
    if factor == "auto":
        errs = []
        for name in gpu_factor_order() + ("scipy",):
            try:
                return factorize(P, name, **kw)
            except Exception as exc:  # noqa: BLE001
                errs.append(f"{name}: {exc!r}")
        raise RuntimeError("no factorization backend worked:\n  " + "\n  ".join(errs))
    raise ValueError(f"unknown factor {factor!r}; choose from {FACTORS}")


def _shifted(K, M, sigma):
    K = K.tocsr()
    if M is None:
        return (K - sigma * sp.identity(K.shape[0], dtype=complex, format="csr")).tocsc()
    return (K - sigma * M.tocsr()).tocsc()


def _arpack(K, M, sigma, k, fac: Factor, return_vectors, tol, maxiter, v0, deflation=None):
    N = K.shape[0]
    if deflation is not None:
        OP = deflation.shift_invert_operator(fac)
    else:
        if M is None:
            mv = fac.solve
        else:
            Mc = M.tocsr()
            mv = lambda x: fac.solve(Mc @ x)  # noqa: E731
        OP = spla.LinearOperator((N, N), matvec=mv, dtype=complex)
    out = spla.eigs(OP, k=k, which="LM", tol=tol, maxiter=maxiter, v0=v0,
                    return_eigenvectors=return_vectors)
    mu, V = out if return_vectors else (out, None)
    small = np.abs(mu) <= 1e-12
    lam = np.where(small, np.nan + 0j, sigma + 1.0 / np.where(small, 1.0, mu))
    return lam, V


def _bbjd(K, M, sigma, k, fac: Optional[Factor], return_vectors, tol, maxiter, v0,
          preconditioner=None, deflation=None, info=None, **kw):
    from spdee.solvers.external import load_bbjd
    jdqz, which, _ = load_bbjd(kw.pop("library", "auto"))
    if info is not None:
        info["library"] = which

    prec_solve = fac.solve if preconditioner is None else (
        preconditioner.solve if hasattr(preconditioner, "solve") else preconditioner)

    class _P:
        solve = staticmethod(prec_solve)

    A, B, prec = K.tocsr(), None if M is None else M.tocsr(), _P()
    if deflation is not None:
        A, B, prec = deflation.deflated_pencil(prec)
    lam, V, inf = jdqz(A, B, k=k, sigma=sigma, tol=tol, maxiter=maxiter, preconditioner=prec,
                       v0=v0, return_info=True, **kw)
    if deflation is not None and V is not None and V.size:
        V = deflation.project_right(V)
    if info is not None:
        info.update(inf)
    if inf.get("nconv", len(lam)) < k:
        warnings.warn(f"bbjd converged {inf.get('nconv')}/{k} eigenvalues in "
                      f"{inf.get('iters')} iterations", RuntimeWarning)
    return lam, (V if return_vectors else None)


def shift_invert(K, M, sigma: complex, k: int, method: str = "arpack", factor: str = "scipy",
                 return_vectors: bool = False, tol: float = 0.0, maxiter: Optional[int] = None,
                 v0=None, factor_options: Optional[Dict[str, Any]] = None,
                 preconditioner=None, deflation=None, info: Optional[dict] = None,
                 **method_options) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """The k eigenvalues of ``K x = lambda M x`` nearest sigma.

    ``tol=0`` is machine precision for ARPACK; bbjd uses 1e-10 then.
    ``factor_options`` go to the factorization (FPLUS: library, pivot_strategy,
    colamd, pin_dense_rows, equilibrate, and with the external library
    relax_rows, panel, pivot_rel, ordering, gpu; see ``_factor_fplus``);
    ``method_options`` go to the eigensolver (bbjd: library, maxsub, minsub,
    inner_maxiter, inner_tol, verbose; ``v0`` starts both solvers).

    ``preconditioner`` (bbjd only): an object with ``solve(b)`` approximating
    (K - sigma M)^-1 for the correction equation, in place of the exact
    factorization, which is then not formed (e.g.
    ``spdee.analysis.precondition.CoarseGridPreconditioner``).
    ``deflation``: a ``spdee.analysis.precondition.Deflation`` whose subspace
    is removed from the iteration, so the solve targets the remaining
    spectrum.  ``info``: a dict filled with the eigensolver's report (bbjd:
    nconv, iters, status, residuals).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if preconditioner is not None and method != "bbjd":
        raise ValueError("an approximate preconditioner applies to method='bbjd' only")
    sigma = complex(sigma)
    fac = None
    if preconditioner is None:
        P = _shifted(K, M, sigma)
        if deflation is not None:
            P = deflation.regularize(P)
        try:
            fac = factorize(P, factor, **(factor_options or {}))
        except RuntimeError as exc:
            raise RuntimeError(f"shift_invert: factorization of (K - sigma M) at sigma={sigma} "
                               f"failed: {exc}") from exc
    try:
        if method == "arpack":
            return _arpack(K, M, sigma, k, fac, return_vectors, tol, maxiter, v0,
                           deflation=deflation)
        return _bbjd(K, M, sigma, k, fac, return_vectors, tol if tol > 0 else 1e-10,
                     maxiter or 200, v0, preconditioner=preconditioner, deflation=deflation,
                     info=info, **method_options)
    finally:
        if fac is not None:
            fac.close()


def dense(K, M=None, sigma: complex = 0.0, backend: str = "auto", precision: str = "double",
          vectors: bool = True, keep_vectors: bool = False, **kw):
    """Full spectrum through the shift-invert transform and dense geev; see
    ``gpu_dense.dense_spectrum`` for every option."""
    from spdee.solvers.gpu_dense import dense_spectrum
    return dense_spectrum(K.tocsr(), None if M is None else M.tocsr(), sigma=sigma, backend=backend,
                          precision=precision, vectors=vectors, keep_vectors=keep_vectors, **kw)


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def available() -> Dict[str, Any]:
    """Which optional solver components work in this installation.

    Host keys are booleans except ``fplus_external`` / ``bbjd_external``,
    the path of a loadable external build or None; ``gpu_vendor`` is "cuda", "rocm" or None and
    ``gpu_device`` the device name or None.  ``cupy`` means a CuPy build
    with a visible device (any vendor), ``cudss`` that cuDSS can factor,
    ``dense_cupy`` that the dense census can run on the device.  The reason
    a GPU capability is missing is in ``gpu.features()``.
    """
    out: Dict[str, Any] = {"arpack": True, "scipy": True, "lapack": True, "dense_numpy": True}
    from spdee.solvers.external import describe
    ext = describe()
    out["fplus_external"] = ext["fplus"]["path"] if ext["fplus"]["ok"] else None
    out["bbjd_external"] = ext["bbjd"]["path"] if ext["bbjd"]["ok"] else None
    out["fplus"] = out["fplus_c_ext"] = out["fplus_external"] is not None   # a build loads only with its C extension
    out["bbjd"] = out["bbjd_external"] is not None
    dev = gpu.device()
    feat = gpu.features()
    out["gpu_vendor"] = None if dev is None else dev.vendor
    out["gpu_device"] = None if dev is None else dev.name
    out["cupy"] = dev is not None
    out["gpu_dense_lu"] = feat["dense_lu"][0]
    out["gpu_sparse_lu"] = feat["sparse_lu"][0]
    out["gpu_geev"] = feat["xgeev"][0] or feat["cupy_geev"][0]
    out["cudss"] = feat["cudss"][0]
    out["dense_cupy"] = out["gpu_dense_lu"]
    return out
