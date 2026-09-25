"""Dense full-spectrum census of a pencil through the shift-invert
transform: the completeness oracle of the sparse shift-invert solves.

Two problem shapes, one entry point (``dense_spectrum``):

  standard     K x = lambda x
  generalized  K x = lambda M x with M singular (the boundary, gauge and
               constraint rows of M are zero)

Both go through C = (K - sigma M)^{-1} M (M = I for a standard problem),
then a dense geev on C: lambda = sigma + 1/mu, and the ker(M) images of a
generalized pencil land at mu = 0, where they are filtered.  Only
(K - sigma M) needs to be nonsingular; a zero pivot is reported so the
caller can nudge sigma.

Backends:
  numpy   host scipy LU + LAPACK geev, the correctness reference.
  cupy    device LU (cuSOLVER or rocSOLVER getrf/getrs through cupyx) and
          cusolverDnXgeev (CUDA >= 12.8) through ctypes, else cp.linalg.eig;
          without either (CUDA < 12.8, or the ROCm build, which has no
          device geev) the transform stays on the device and the geev runs
          on the host ('cupy+hosteig').  Without a usable device at all the
          census runs on the host after one ``gpu.GPUFallbackWarning``.
  auto    cupy on a discrete or datacenter GPU of either vendor (the LU on
          the device; on ROCm the geev then runs on the host, silently),
          numpy on an integrated device, where the LU measured 0.4-0.5x
          host speed; an explicit backend='cupy' is honoured there with a
          ``gpu.GPUPerformanceWarning``.  ``SPDEE_NO_GPU=1`` gives numpy.

Precision: 'double' (complex128), 'single' (complex64, half the memory, about
1e-5 relative error on the eigenvalues) or 'mixed' (double transform, single
geev).  Memory in complex double is about 2-3 N^2 words at the peak;
``managed_memory()`` lets CuPy allocations spill from HBM into host memory.

Nothing here knows about omega; the caller maps lambda to omega.
"""
from __future__ import annotations
import time
import ctypes
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

from spdee.solvers import gpu

__all__ = ["dense_spectrum", "CensusResult", "XgeevUnavailable", "managed_memory"]


# ------------------------------------------------------------------ utilities
def managed_memory():
    """Switch CuPy to managed memory (cudaMallocManaged, hipMallocManaged)
    so device arrays can exceed HBM (they spill into host memory). Call
    before the first allocation; a no-op, with one warning, when the device
    stack cannot provide it (no CuPy, no device, SPDEE_NO_GPU)."""
    ok, why = gpu.features()["managed_memory"]
    if not ok:
        gpu.fallback("managed memory", why, "keeping the default allocator")
        return False
    try:
        import cupy as cp
        pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
        cp.cuda.set_allocator(pool.malloc)
        return True
    except Exception:
        return False


def _cdtype(precision):
    if precision in ("double", "mixed"):
        return np.complex128          # mixed: transform in double
    if precision == "single":
        return np.complex64
    raise ValueError(f"precision must be 'double', 'single', or 'mixed', got {precision!r}")


def _default_mu_min(precision):
    # ker(B) images sit at |mu| ~ N eps cond, the deepest physical modes at
    # |mu| = 1/|lambda|; the gap is many decades wide in double and narrower
    # in single / mixed (single geev)
    return 1e-8 if precision == "double" else 1e-4


# ------------------------------------------------- cuSOLVER Xgeev via ctypes
class XgeevUnavailable(RuntimeError):
    pass


_CUSOLVER_EIG_MODE_NOVECTOR = 0
_CUSOLVER_EIG_MODE_VECTOR = 1
_CUDA_C_32F = 4
_CUDA_C_64F = 5


def _load_libcusolver():
    # Bind to the instance CuPy has ALREADY mapped: two libcusolver copies can
    # coexist in one process (pip nvidia wheels + an HPC-SDK tree on
    # LD_LIBRARY_PATH), and a handle created by one copy is invalid in the
    # other -- the symptom is cusolverDnSetStream returning 7 (INTERNAL_ERROR)
    # on a perfectly good handle. /proc/self/maps has the exact mapped path;
    # CDLL on that path returns the same instance.
    try:
        seen = set()
        with open("/proc/self/maps") as fh:
            for line in fh:
                p = line.rstrip().split(" ")[-1]
                if "libcusolver.so" in p and p.startswith("/") and p not in seen:
                    seen.add(p)
                    try:
                        return ctypes.CDLL(p)
                    except OSError:
                        continue
    except OSError:
        pass
    for name in ("libcusolver.so", "libcusolver.so.12", "libcusolver.so.13",
                 "libcusolver.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise XgeevUnavailable("libcusolver shared library not found")


def _xgeev_device(C, want_vectors=True):
    """cusolverDnXgeev on a square Fortran-ordered CuPy array C (destroyed).
    Returns (w, VR) with VR None when want_vectors is False; raises
    XgeevUnavailable when the toolkit predates Xgeev (CUDA < 12.8)."""
    import cupy as cp
    from cupy.cuda import device as _cpdev

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be square")
    if not C.flags.f_contiguous:
        raise ValueError("C must be Fortran-ordered (build with order='F')")
    n = C.shape[0]
    if C.dtype == cp.complex128:
        dt = _CUDA_C_64F
    elif C.dtype == cp.complex64:
        dt = _CUDA_C_32F
    else:
        raise ValueError(f"unsupported dtype {C.dtype}")

    lib = _load_libcusolver()
    for sym in ("cusolverDnXgeev", "cusolverDnXgeev_bufferSize",
                "cusolverDnCreateParams", "cusolverDnDestroyParams",
                "cusolverDnSetStream"):
        if not hasattr(lib, sym):
            raise XgeevUnavailable(
                f"{sym} missing from libcusolver -- Xgeev needs a CUDA >= 12.8 "
                "toolkit runtime; load a newer cuda module or fall back to "
                "backend='cupy+hosteig'/'numpy'.")

    c_ip = ctypes.c_void_p
    c_i64 = ctypes.c_int64
    c_int = ctypes.c_int
    c_sz = ctypes.c_size_t

    handle = c_ip(_cpdev.get_cusolver_handle())
    stream = cp.cuda.get_current_stream().ptr
    st = lib.cusolverDnSetStream(handle, c_ip(stream))
    if st != 0:
        raise RuntimeError(f"cusolverDnSetStream status {st}")

    params = c_ip()
    st = lib.cusolverDnCreateParams(ctypes.byref(params))
    if st != 0:
        raise RuntimeError(f"cusolverDnCreateParams status {st}")

    try:
        jobvl = c_int(_CUSOLVER_EIG_MODE_NOVECTOR)
        jobvr = c_int(_CUSOLVER_EIG_MODE_VECTOR if want_vectors
                      else _CUSOLVER_EIG_MODE_NOVECTOR)
        W = cp.empty(n, dtype=C.dtype)
        if want_vectors:
            VR = cp.empty((n, n), dtype=C.dtype, order="F")
            vr_ptr, ldvr = c_ip(VR.data.ptr), c_i64(n)
        else:
            VR = None
            vr_ptr, ldvr = c_ip(None), c_i64(1)
        vl_ptr, ldvl = c_ip(None), c_i64(1)

        dev_bytes, host_bytes = c_sz(0), c_sz(0)
        st = lib.cusolverDnXgeev_bufferSize(
            handle, params, jobvl, jobvr, c_i64(n),
            c_int(dt), c_ip(C.data.ptr), c_i64(n),
            c_int(dt), c_ip(W.data.ptr),
            c_int(dt), vl_ptr, ldvl,
            c_int(dt), vr_ptr, ldvr,
            c_int(dt), ctypes.byref(dev_bytes), ctypes.byref(host_bytes))
        if st != 0:
            raise RuntimeError(f"cusolverDnXgeev_bufferSize status {st}")

        dwork = cp.empty(max(1, dev_bytes.value), dtype=cp.uint8)
        hwork = np.empty(max(1, host_bytes.value), dtype=np.uint8)
        dinfo = cp.zeros(1, dtype=cp.int32)

        st = lib.cusolverDnXgeev(
            handle, params, jobvl, jobvr, c_i64(n),
            c_int(dt), c_ip(C.data.ptr), c_i64(n),
            c_int(dt), c_ip(W.data.ptr),
            c_int(dt), vl_ptr, ldvl,
            c_int(dt), vr_ptr, ldvr,
            c_int(dt), c_ip(dwork.data.ptr), c_sz(dwork.nbytes),
            c_ip(hwork.ctypes.data), c_sz(hwork.nbytes),
            c_ip(dinfo.data.ptr))
        cp.cuda.get_current_stream().synchronize()
        if st != 0:
            raise RuntimeError(f"cusolverDnXgeev status {st}")
        info = int(dinfo.get()[0])
        if info < 0:
            raise RuntimeError(f"cusolverDnXgeev: illegal argument {-info}")
        if info > 0:
            # LAPACK geev semantics: QR failed; W[info:] converged, W[:info] not.
            raise RuntimeError(
                f"cusolverDnXgeev: QR iteration failed to converge (info={info}); "
                "retry with a different sigma or precision")
        return W, VR
    finally:
        lib.cusolverDnDestroyParams(params)


# ------------------------------------------------------------ dense builders
def _densify(A_csr, cdtype, xp, spx, order="F"):
    """Sparse (host scipy CSR) -> dense array on the target backend."""
    A_csr = A_csr.tocsr().astype(cdtype)
    if xp is np:
        return np.asfortranarray(A_csr.toarray()) if order == "F" else A_csr.toarray()
    dA = spx.csr_matrix(A_csr)          # host -> device transfer of CSR triplets
    try:
        out = dA.toarray(order=order)
    except TypeError:
        out = xp.asfortranarray(dA.toarray())
    return out


def _transform_generalized(A, B, sigma, cdtype, xp, spx, timings):
    """C = (A - sigma B)^{-1} B, dense Fortran-ordered on the backend."""
    t0 = time.time()
    P = (A - sigma * B).tocsr()
    Pd = _densify(P, cdtype, xp, spx)
    Bd = _densify(B, cdtype, xp, spx)
    timings["densify"] = time.time() - t0

    t0 = time.time()
    if xp is np:
        lu, piv = sla.lu_factor(Pd, overwrite_a=True, check_finite=False)
        C = sla.lu_solve((lu, piv), Bd, overwrite_b=True, check_finite=False)
    else:
        import cupyx.scipy.linalg as cpla
        lu, piv = cpla.lu_factor(Pd, overwrite_a=True, check_finite=False)
        C = cpla.lu_solve((lu, piv), Bd, overwrite_b=True, check_finite=False)
        xp.cuda.get_current_stream().synchronize()
    del lu, Pd, Bd
    timings["transform_lu_solve"] = time.time() - t0
    if not C.flags.f_contiguous:
        C = xp.asfortranarray(C)
    return C


def _geev(C, want_vectors, backend, timings, warn=True):
    """Standard dense eigensolve on C (destroyed where possible).
    Returns (w, VR, backend_used) as host-or-device arrays matching backend.
    On the cupy backend: Xgeev through ctypes where the probe says the
    toolkit has it, else CuPy's own geev, else the host; the ROCm build
    never touches libcusolver.  The host landing warns when the device
    was requested explicitly (``warn``), not when 'auto' chose it knowing
    the geev would run on the host; a device geev that was attempted and
    failed always warns."""
    t0 = time.time()
    if backend == "numpy":
        w, VR = sla.eig(C, right=True, overwrite_a=True, check_finite=False)
        timings["geev"] = time.time() - t0
        return w, (VR if want_vectors else None), "numpy"
    import cupy as cp
    feat = gpu.features()
    ok, why = feat["xgeev"]
    if ok:
        warn = True
        try:
            w, VR = _xgeev_device(C, want_vectors=want_vectors)
            timings["geev"] = time.time() - t0
            return w, VR, "cupy(Xgeev)"
        except (XgeevUnavailable, RuntimeError) as e:
            why = str(e)
    ok, why2 = feat["cupy_geev"]
    if ok:
        # CuPy >= 14 wraps Xgeev natively (same library instance as the rest
        # of CuPy, immune to loader mismatches; may make one extra N^2 copy)
        warn = True
        try:
            if want_vectors:
                w, VR = cp.linalg.eig(C)
            else:
                w, VR = cp.linalg.eigvals(C), None
            timings["geev"] = time.time() - t0
            return w, VR, "cupy(cp.linalg.eig)"
        except Exception as e2:  # noqa: BLE001
            why2 = str(e2)
    if warn:
        gpu.fallback("device geev", f"Xgeev: {why}; cp.linalg.eig: {why2}",
                     "running geev on host LAPACK ('cupy+hosteig')")
    Ch = cp.asnumpy(C)
    del C
    cp.get_default_memory_pool().free_all_blocks()
    w, VR = sla.eig(Ch, right=True, overwrite_a=True, check_finite=False)
    timings["geev"] = time.time() - t0
    return w, (VR if want_vectors else None), "cupy+hosteig"


def _pencil_residuals_blocked(A, B, lam, V, block=512):
    """Relative pencil residuals per eigenpair, blocked over columns.
    Standard problem when B is None. Host (numpy/scipy) or device (cupy)
    decided by where V lives."""
    xp = np
    spmod = sp
    is_dev = V.__class__.__module__.startswith("cupy")
    if is_dev:
        import cupy as xp                     # noqa: F811
        import cupyx.scipy.sparse as spmod    # noqa: F811
    Ad = spmod.csr_matrix(A.tocsr()) if is_dev else A.tocsr()
    Bd = None if B is None else (spmod.csr_matrix(B.tocsr()) if is_dev else B.tocsr())
    lam_d = xp.asarray(lam)
    n = V.shape[1]
    out = xp.empty(n, dtype=float)
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        Vb = V[:, j0:j1]
        AV = Ad @ Vb
        if Bd is None:
            num = xp.linalg.norm(AV - Vb * lam_d[None, j0:j1], axis=0)
            den = xp.linalg.norm(AV, axis=0) + xp.abs(lam_d[j0:j1]) * xp.linalg.norm(Vb, axis=0)
        else:
            BV = Bd @ Vb
            num = xp.linalg.norm(AV - BV * lam_d[None, j0:j1], axis=0)
            den = xp.linalg.norm(AV, axis=0) + xp.abs(lam_d[j0:j1]) * xp.linalg.norm(BV, axis=0)
        den = xp.where(den == 0, 1.0, den)
        out[j0:j1] = num / den
    return out if not is_dev else out.get()


def _bnorm_ratio(B, VR, block=512):
    """s_j = ||B v_j|| / ||v_j|| in DOUBLE, per eigenvector column. ker(B)
    members collapse by the accuracy of the eigenVECTOR subspace (well
    separated in the transform spectrum, so ~eps even when the eigenVALUES
    are smeared by conditioning), which is what makes this the robust
    classifier in single precision."""
    Bd = B.tocsr().astype(np.complex128)
    n = VR.shape[1]
    on_dev = not isinstance(VR, np.ndarray)
    out = np.empty(n, dtype=float)
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block)
        Vb = VR[:, j0:j1]
        if on_dev:
            Vb = Vb.get()
        Vb = Vb.astype(np.complex128, copy=False)
        num = np.linalg.norm(Bd @ Vb, axis=0)
        den = np.linalg.norm(Vb, axis=0)
        den = np.where(den == 0, 1.0, den)
        out[j0:j1] = num / den
    return out


def _gap_split(s, floor, cap=None, min_ratio=1e3):
    """Split a positive statistic into (large = finite/physical) vs (small =
    ker(B)) at the TOPMOST multiplicative gap >= min_ratio, scanning downward
    from the cap (default: median/10, i.e. inside the lower tail). Deeper gaps
    are internal structure of the null cluster (exact zeros vs eps-scale).
    Falls back to the fixed `floor` when no qualifying gap exists (e.g. B
    nonsingular: nothing clustered near zero)."""
    s = np.asarray(s, dtype=float)
    sa = np.sort(s)
    if cap is None:
        cap = max(np.median(sa) / 10.0, floor * 10.0)
    k_cap = int(np.searchsorted(sa, cap))
    cut = None
    if k_cap > 0:
        lo = np.maximum(sa[:k_cap + 1], np.finfo(float).tiny)
        for j in range(lo.size - 2, -1, -1):
            if lo[j + 1] / lo[j] >= min_ratio:
                cut = np.sqrt(lo[j] * lo[j + 1])   # geometric midpoint
                break
    if cut is None:
        cut = floor
    return s > cut


# ---------------------------------------------------------------- entry point
@dataclass
class CensusResult:
    lam: np.ndarray                 # finite pencil eigenvalues, native convention
    resid: Optional[np.ndarray]     # relative pencil residuals (same order), or None
    mu: Optional[np.ndarray]        # transform eigenvalues (generalized only)
    n_infinite: int                 # ker(B) images filtered at |mu| <= mu_min
    N: int
    sigma: complex
    backend: str
    precision: str
    timings: dict = field(default_factory=dict)
    V: Optional[np.ndarray] = None  # host eigenvectors (N x k) if keep_vectors

    def in_window(self, re_lo, re_hi, im_lo, im_hi):
        w = self.lam
        m = ((w.real >= re_lo) & (w.real <= re_hi) &
             (w.imag >= im_lo) & (w.imag <= im_hi))
        return np.nonzero(m)[0]


def dense_spectrum(A, B=None, sigma=0.0, backend="auto", precision="double",
                 vectors=True, mu_min=None, compute_resid=True,
                 keep_vectors=False, sigma_retry=1e-3j, transform="auto"):
    """
    Dense full-spectrum solve of the standard problem (B=None) or the
    generalized pencil A x = lambda B x with (possibly singular) B.

    transform='auto' runs BOTH shapes through the shift-invert resolvent:
      generalized: C = (A - sigma B)^{-1} B   mu = 1/(lambda - sigma)
      standard:    C = (A - sigma I)^{-1}     mu = 1/(lambda - sigma)
    and geev's backward error eps*||C|| then lands on the modes NEAREST sigma:
    place sigma in the empty growth half-plane (e.g. +0.5j in omega units) and
    the physical window is the best-conditioned part of the spectrum, while the
    deep diffusive tail is crushed toward mu=0 where its (large) lambda error
    is irrelevant to a census. Raw geev on A itself (transform='none',
    standard problems only) puts eps*||A|| of backward error on everything and
    ||A|| here is dominated by the diffusive blocks -- measured ~1e-4 eigenvalue
    error on window modes, vs ~1e-10 through the resolvent.

    Only generalized pencils get the |mu| <= mu_min "infinite mode" filter
    (ker(B) images); a standard problem's small-|mu| modes are real, far modes
    and are kept.

    vectors=False skips right eigenvectors entirely (halves geev output memory;
    residuals then unavailable). keep_vectors=True returns them on host --
    N x n_finite complex, mind the size.

    On an exactly singular (A - sigma B) factorization (sigma sitting on an
    eigenvalue), retries once at sigma + sigma_retry before raising.
    """
    timings = {}
    t_all = time.time()
    cdtype = _cdtype(precision)
    if backend not in ("auto", "numpy", "cupy"):
        raise ValueError(f"backend must be auto/numpy/cupy, got {backend!r}")
    explicit = backend == "cupy"
    if backend == "auto":
        backend = "cupy" if gpu.prefer_device()[0] and gpu.features()["dense_lu"][0] else "numpy"
    elif backend == "cupy":
        ok, why = gpu.features()["dense_lu"]
        if not ok:
            gpu.fallback("backend='cupy'", why, "running the census on the host (numpy)")
            backend = "numpy"
        else:
            dev = gpu.device()
            if dev is not None and dev.integrated:
                warnings.warn(f"backend='cupy' on {dev.name}: {gpu.prefer_device()[1]}",
                              gpu.GPUPerformanceWarning, stacklevel=2)
    if backend == "cupy":
        import cupy as xp
        import cupyx.scipy.sparse as spx
    else:
        xp, spx = np, None
    if compute_resid and not vectors:
        compute_resid = False

    A = A.tocsr()
    N = A.shape[0]
    generalized = B is not None
    if transform not in ("auto", "none"):
        raise ValueError(f"transform must be 'auto' or 'none', got {transform!r}")
    if generalized and transform == "none":
        raise ValueError("generalized pencil requires the transform (B singular)")
    use_transform = transform == "auto"

    if use_transform:
        if generalized:
            B = B.tocsr()
            if mu_min is None:
                mu_min = _default_mu_min(precision)
            Bmat = B
        else:
            Bmat = sp.identity(N, dtype=cdtype, format="csr")

        def _build(sig):
            return _transform_generalized(A, Bmat, sig, cdtype, xp, spx, timings)

        try:
            C = _build(sigma)
        except Exception as e:
            if "singular" not in str(e).lower() or sigma_retry == 0:
                raise
            warnings.warn(f"(A - sigma B) singular at sigma={sigma}; "
                          f"retrying at sigma={sigma + sigma_retry}")
            sigma = sigma + sigma_retry
            C = _build(sigma)
    else:
        t0 = time.time()
        C = _densify(A, cdtype, xp, spx)
        timings["densify"] = time.time() - t0

    if precision == "mixed":
        # accuracy is set by the transform's LU at cond(A - sigma B) ~ 1e10+;
        # doing that in double and only the geev in single keeps window-mode
        # errors at the eps_single * cond(C) level (~1e-5) instead of
        # eps_single * cond(P) (~1e-2 measured at N=5e4), while halving the
        # geev-phase memory. Cast C down after the double transform:
        t0 = time.time()
        C32 = C.astype(np.complex64)
        del C
        if backend == "cupy":
            import cupy as _cp
            _cp.get_default_memory_pool().free_all_blocks()
        C = C32 if C32.flags.f_contiguous else (
            np.asfortranarray(C32) if isinstance(C32, np.ndarray) else C32)
        timings["downcast"] = time.time() - t0
    w, VR, backend_used = _geev(C, vectors, backend, timings, warn=explicit)

    # ---- postprocess: filter/map, residuals, host landing
    t0 = time.time()
    w_host = w if isinstance(w, np.ndarray) else w.get()
    if use_transform:
        if generalized:
            if VR is not None:
                s = _bnorm_ratio(B, VR)
                finite = _gap_split(s, floor=mu_min)
            else:
                finite = _gap_split(np.abs(w_host), floor=mu_min)
            n_inf = int(N - finite.sum())
        else:
            finite = w_host != 0            # exact zeros only (none expected)
            n_inf = int(N - finite.sum())
        mu = w_host[finite]
        lam = sigma + 1.0 / mu
    else:
        finite = np.ones(N, dtype=bool)
        mu = None
        lam = w_host
        n_inf = 0

    resid = None
    if compute_resid and VR is not None:
        idx = np.nonzero(finite)[0]
        if backend_used.startswith("cupy") and not isinstance(VR, np.ndarray):
            import cupy as cp
            Vf = VR[:, cp.asarray(idx)]
        else:
            Vf = VR[:, idx]
        resid = _pencil_residuals_blocked(A, B if generalized else None, lam, Vf)
    Vout = None
    if keep_vectors and VR is not None:
        idx = np.nonzero(finite)[0]
        if isinstance(VR, np.ndarray):
            Vout = VR[:, idx]
        else:
            import cupy as cp
            # a one-shot VR[:, idx] materializes an N x n_sel device copy on
            # top of VR; chunked slices bound the extra footprint to ~1 GB
            cp.get_default_memory_pool().free_all_blocks()
            Vout = np.empty((VR.shape[0], idx.size), dtype=complex)
            step = max(1, int(1e9 // (16 * VR.shape[0])))
            for j0 in range(0, idx.size, step):
                sel = cp.asarray(idx[j0:j0 + step])
                Vout[:, j0:j0 + len(sel)] = cp.asnumpy(VR[:, sel])
                del sel
    del VR
    timings["postprocess"] = time.time() - t0
    timings["total"] = time.time() - t_all

    return CensusResult(lam=lam, resid=resid, mu=mu, n_infinite=n_inf, N=N,
                        sigma=sigma, backend=backend_used, precision=precision,
                        timings=timings, V=Vout)
