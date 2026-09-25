"""Sparse factorization of the shifted pencil (K - sigma M) on a device.

The shift-invert cost per target is one sparse LU plus a few triangular
solves per ARPACK matvec; ARPACK itself stays on the host (``spdee.solvers``)
and only the factorization and solve move to a device backend.  Backends,
'auto' trying cudss then host:

  cudss   NVIDIA cuDSS through nvmath-python (>= 0.5); ``_CudssFactor`` is
          the only nvmath contact, and ``probe_backends()`` shows the exact
          failing call when the installed API differs.  cuDSS returns wrong
          solves on pencils with zero diagonal entries unless matching is on,
          so the adapter enables it and ``spdee.solvers`` gates the factor
          on a residual.
  cupy    cupyx.scipy.sparse.linalg.splu: the factorization runs on the HOST
          with scipy's SuperLU and only the triangular solves move to the
          device, where they measured 33x slower than the host (Radeon
          890M; CuPy's design on CUDA too).  An explicit option, never
          chosen by 'auto'.
  host    scipy SuperLU, the correctness reference.

Everything is complex128.  ``python -m spdee.solvers.gpu_si`` runs the probe.
Which of cudss and cupy the installed stack can run at all (vendor, nvmath,
``SPDEE_NO_GPU``) is decided by ``spdee.solvers.gpu``; the adapters here
report the failing call when a runnable backend still fails.
"""
from __future__ import annotations
import os
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

__all__ = ["SparseFactor", "probe_backends", "BackendUnavailable"]


class BackendUnavailable(RuntimeError):
    pass


# --------------------------------------------------------------- cuDSS adapter
class _CudssFactor:
    """(A - sigma B) factorization via nvmath-python's cuDSS direct solver.

    All nvmath API contact lives here (the DirectSolver pattern plan /
    factorize / solve with reset_operands for RHS reuse, nvmath 0.5 through
    1.0); if the installed nvmath differs, this class is the only thing to
    touch and probe_backends() prints the failing call."""

    def __init__(self, P_host_csr):
        try:
            import nvmath.sparse.advanced as adv
            import logging
            logging.getLogger("nvmath").setLevel(logging.WARNING)
            logging.getLogger().setLevel(max(logging.getLogger().level,
                                             logging.WARNING))
        except Exception as e:
            raise BackendUnavailable(f"nvmath-python not usable: {e!r}")
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx
        except Exception as e:
            raise BackendUnavailable(f"cupy not usable: {e!r}")
        self._adv = adv
        self._cp = cp
        P = P_host_csr.tocsr().astype(np.complex128)
        self._A = cpx.csr_matrix(P)
        n = P.shape[0]
        # nvmath >= 0.8 requires a col-major RHS; a 1-D vector satisfies it
        # (a C-ordered (n, 1) array does not).
        self._b = cp.zeros(n, dtype=cp.complex128)
        self._slv = None
        opts = None
        mt = os.environ.get("CUDSS_THREADING_LIB")
        if mt:
            # nvmath only honors the threading layer via options, not the env
            try:
                opts = adv.DirectSolverOptions(multithreading_lib=mt)
            except TypeError:
                try:
                    opts = adv.DirectSolverOptions(threading_lib=mt)
                except TypeError:
                    opts = None
        try:
            if opts is not None:
                slv = adv.DirectSolver(self._A, self._b, options=opts)
            else:
                slv = adv.DirectSolver(self._A, self._b)
            # Maximum-weight matching before reordering: without it cuDSS
            # returns wrong solves on pencils with zero diagonal entries.
            # CUDSS_USE_MATCHING=0 restores the library default.
            if os.environ.get("CUDSS_USE_MATCHING", "1") != "0":
                _enable_matching(slv)
            slv.plan()
            slv.factorize()
        except Exception as e:
            raise BackendUnavailable(f"cuDSS DirectSolver failed: {e!r}")
        self._slv = slv

    def solve(self, rhs_host):
        cp = self._cp
        self._b[:] = cp.asarray(rhs_host).ravel()
        slv = self._slv
        if hasattr(slv, "reset_operands"):
            try:
                slv.reset_operands(b=self._b)
            except Exception:
                pass  # same-buffer reuse
        x = slv.solve()
        # nvmath >= 1.0 solves non-blocking on its own stream and asnumpy
        # syncs CuPy's stream only; a device-wide sync is needed before x
        # is read.
        cp.cuda.runtime.deviceSynchronize()
        return cp.asnumpy(x).ravel()

    def close(self):
        if self._slv is not None and hasattr(self._slv, "free"):
            try:
                self._slv.free()
            except Exception:
                pass
        self._slv = None


def _enable_matching(slv):
    """Turn on the maximum-weight (diagonal-product) matching of a cuDSS
    DirectSolver: ``plan_config.use_matching`` through nvmath 0.x,
    ``plan_config.matching_algorithm = MAX_DIAG_PRODUCT`` from nvmath 1.0
    (where the default, NONE, and the count / min-diagonal choices all
    return garbage on the spdee pencils)."""
    pc = slv.plan_config
    if hasattr(pc, "use_matching"):
        pc.use_matching = True
        return
    if hasattr(pc, "matching_algorithm"):
        from nvmath.bindings.cudss import MatchingAlg
        pc.matching_algorithm = MatchingAlg.MAX_DIAG_PRODUCT
        return
    raise BackendUnavailable(
        "cuDSS plan_config offers neither use_matching nor matching_algorithm; "
        "without matching the solves are wrong on pencils with zero diagonal entries")


# --------------------------------------------------------------- cupy adapter
class _CupySpluFactor:
    """cupyx splu: SuperLU factorization on the host, solves on the device
    (slower than the host solve); explicit use only."""

    def __init__(self, P_host_csr):
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx
            import cupyx.scipy.sparse.linalg as cpsla
        except Exception as e:
            raise BackendUnavailable(f"cupy sparse not usable: {e!r}")
        self._cp = cp
        P = P_host_csr.tocsr().astype(np.complex128)
        try:
            self._lu = cpsla.splu(cpx.csr_matrix(P).tocsc())
        except Exception as e:
            raise BackendUnavailable(f"cupyx splu failed: {e!r}")

    def solve(self, rhs_host):
        cp = self._cp
        x = self._lu.solve(cp.asarray(rhs_host))
        return cp.asnumpy(x).ravel()

    def close(self):
        self._lu = None


# --------------------------------------------------------------- host adapter
class _HostFactor:
    """SuperLU with COLAMD applied to the transpose by default, which
    roughly halves the L+U fill of a tau-bordered spectral pencil (the
    solve goes through trans='T', the plain transpose, no conjugation).
    Set GPU_SI_HOST_TRANSPOSE=0 to factor the matrix directly."""
    def __init__(self, P_host_csr):
        self._trans = os.environ.get("GPU_SI_HOST_TRANSPOSE", "1") != "0"
        P = P_host_csr.tocsc().astype(np.complex128)
        self._lu = splu(P.T.tocsc()) if self._trans else splu(P)

    def solve(self, rhs_host):
        if self._trans:
            return self._lu.solve(rhs_host, trans='T')
        return self._lu.solve(rhs_host)

    def close(self):
        self._lu = None


_ADAPTERS = {"cudss": _CudssFactor, "cupy": _CupySpluFactor, "host": _HostFactor}


class SparseFactor:
    """Factor P = (A - sigma B) (B=None -> identity) with backend fallback.
    backend='auto' tries cudss where ``spdee.solvers.gpu`` prefers the
    device, then host, remembering the first that works for the rest of
    the process."""
    _auto_choice = None

    def __init__(self, A, B=None, backend="auto"):
        self.A = A.tocsr()
        self.B = B.tocsr() if B is not None else None
        self.backend = backend
        self._fac = None
        self.backend_used = None
        self.sigma = None

    def factor(self, sigma):
        if self._fac is not None:
            self._fac.close()
            self._fac = None
        P = (self.A - sigma * self.B) if self.B is not None else \
            (self.A - sigma * sp.identity(self.A.shape[0],
                                          dtype=np.complex128, format="csr"))
        if self.backend != "auto":
            order = [self.backend]
        elif SparseFactor._auto_choice:
            order = [SparseFactor._auto_choice]
        else:
            from spdee.solvers import gpu_factor_order
            order = list(gpu_factor_order()) + ["host"]
        errs = []
        for key in order:
            try:
                self._fac = _ADAPTERS[key](P)
                self.backend_used = key
                if self.backend == "auto":
                    SparseFactor._auto_choice = key
                self.sigma = sigma
                return self
            except BackendUnavailable as e:
                errs.append(str(e))
                continue
        raise BackendUnavailable("no factorization backend worked:\n  " +
                                 "\n  ".join(errs))

    def solve(self, rhs):
        return self._fac.solve(rhs)

    def close(self):
        if self._fac is not None:
            self._fac.close()


def probe_backends(N=2000, density_stencil=True, verbose=True):
    """Try each backend on a small complex test system; report availability,
    correctness (vs host), and factor+solve timing."""
    rng = np.random.default_rng(0)
    if density_stencil:
        main = 4.0 + rng.standard_normal(N) + 1j * rng.standard_normal(N)
        off = 0.5 * (rng.standard_normal(N - 1) + 1j * rng.standard_normal(N - 1))
        P = sp.diags([off, main, off.conj()], [-1, 0, 1], format="csr")
        P = (P + 0.05 * sp.random(N, N, density=5.0 / N, dtype=float,
                                  random_state=1) * (1 + 0.3j)).tocsr()
    b = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    ref = None
    out = {}
    for key in ("host", "cupy", "cudss"):
        try:
            t0 = time.time()
            f = _ADAPTERS[key](P)
            t_fac = time.time() - t0
            t0 = time.time()
            x = f.solve(b)
            for _ in range(9):
                x = f.solve(b)
            t_sol = (time.time() - t0) / 10
            f.close()
            if key == "host":
                ref = x
            err = (np.linalg.norm(x - ref) / np.linalg.norm(ref)
                   if ref is not None else 0.0)
            out[key] = dict(ok=True, factor_s=t_fac, solve_s=t_sol, err=err)
            if verbose:
                print(f"  {key:6s}: factor {t_fac*1e3:8.1f} ms   "
                      f"solve {t_sol*1e3:7.2f} ms   vs host {err:.1e}")
        except BackendUnavailable as e:
            out[key] = dict(ok=False, reason=str(e))
            if verbose:
                print(f"  {key:6s}: unavailable -- {e}")
    return out


if __name__ == "__main__":
    print("backend probe (N=2000 complex test system):")
    probe_backends()
