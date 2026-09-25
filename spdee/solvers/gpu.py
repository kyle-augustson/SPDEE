"""The device probe: which GPU vendor is present and which capabilities of
the solver layer it supports.

CuPy is the only device library SPDEE speaks.  Its CUDA build and its ROCm
build share the ``cupy`` namespace, so one probe serves both:
``cupy.cuda.runtime.is_hip`` names the vendor (hipSOLVER is reported under
cuSOLVER's name, so library names are never used for that; the mapped
librocsolver/librocblas are shown as a secondary signal only) and the
capability table below says what each vendor can run.

  capability      CUDA                          ROCm
  dense_lu        cuSOLVER getrf through cupyx  rocSOLVER through cupyx
  sparse_lu       cupyx splu: SuperLU on the host, triangular solves on the
                  device (slower than the host; explicit option only)
  xgeev           cusolverDnXgeev, CUDA >= 12.8 never attempted
  cupy_geev       cp.linalg.eig (a trial call)  raises 'geev is not available'
  cudss           nvmath-python                 no equivalent
  managed_memory  cudaMallocManaged             hipMallocManaged through the
                                                same cupy.cuda.malloc_managed

Automatic choices are decided by device CLASS, not vendor
(``prefer_device()``): an integrated device (the ``integrated`` field of
``getDeviceProperties``, shares system memory) never wins and every
'auto' stays on the host there; a discrete or datacenter device of either
vendor is preferred for the dense LU.  Measured on an integrated Radeon
890M (gfx1150, ROCm 7.1, CuPy 14.2 built from source): the dense LU on the
device runs at 0.4-0.5x host speed from N = 2000 to 10000 with no
crossover, the cupyx splu solve is 33x slower than the host, the sparse
CSR matvec is 2.2x faster than the host at N = 30000 (3.8x slower at 3000)
and is the matrix-free operation worth building on.  MI300-class parts
are the AMD production target and are expected to win on the dense LU and
the matvec; untested here.  Vendor facts stay vendor facts: no device geev
on ROCm today (host LAPACK; MAGMA-HIP is the future route), no device
sparse direct solver through CuPy on either vendor, managed memory on
both.  Explicit requests are honoured, with a ``GPUPerformanceWarning`` on
an integrated device.  ``SPDEE_NO_GPU=1`` hides the device entirely: the
probe reports no vendor and every automatic choice goes to the CPU.

The hardware probe runs once per process (``reset()`` clears it, for tests
that patch the runtime); the environment switch is read on every call.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = ["Device", "GPUFallbackWarning", "GPUPerformanceWarning", "vendor", "device",
           "features", "disabled", "prefer_device", "mapped_libraries", "reset", "fallback"]

XGEEV_MIN_RUNTIME = 12080

_FEATURE_NAMES = ("dense_lu", "sparse_lu", "xgeev", "cupy_geev", "cudss", "managed_memory")


class GPUFallbackWarning(RuntimeWarning):
    """A requested GPU capability is missing and the CPU path was taken."""


class GPUPerformanceWarning(RuntimeWarning):
    """An explicit GPU request is honoured where the host is known to be faster."""


_LIBRARIES = ("libcusolver", "libcublas", "libcudss", "libcudart", "librocsolver", "librocblas",
              "libhipsolver", "libhipblas", "libamdhip64")


def mapped_libraries() -> Tuple[str, ...]:
    """The solver and runtime libraries mapped into this process, from
    /proc/self/maps (a diagnostic, not a vendor signal)."""
    found = []
    try:
        with open("/proc/self/maps") as fh:
            for line in fh:
                path = line.rstrip().split(" ")[-1]
                base = path.rsplit("/", 1)[-1]
                for lib in _LIBRARIES:
                    if base.startswith(lib + ".") and lib not in found:
                        found.append(lib)
    except OSError:
        pass
    return tuple(found)


@dataclass(frozen=True)
class Device:
    vendor: str                 # "cuda" or "rocm"
    name: str
    index: int
    count: int
    free_bytes: int
    total_bytes: int
    cupy_version: str
    runtime_version: int        # CUDA runtime (e.g. 13020) or HIP version
    integrated: bool = False    # shares system memory (an APU / iGPU)

    @property
    def klass(self) -> str:
        return "integrated" if self.integrated else "discrete"


_CACHE: Dict[str, object] = {}


def reset() -> None:
    """Forget the hardware probe so the next call re-runs it."""
    _CACHE.clear()


def disabled() -> bool:
    """True when SPDEE_NO_GPU asks for a CPU-only process."""
    return os.environ.get("SPDEE_NO_GPU", "").strip().lower() in ("1", "true", "yes", "on")


def _probe_device() -> Tuple[Optional[Device], str]:
    """(device, reason) with device None when no usable one is found."""
    if "device" in _CACHE:
        return _CACHE["device"]  # type: ignore[return-value]
    dev, why = None, ""
    try:
        import cupy as cp
        rt = cp.cuda.runtime
        try:
            count = int(rt.getDeviceCount())
        except Exception as exc:  # noqa: BLE001
            count, why = 0, f"CuPy {cp.__version__} imports but sees no device ({exc!r})"
        if count > 0:
            idx = int(rt.getDevice())
            props = rt.getDeviceProperties(idx)
            name = props["name"]
            name = name.decode() if isinstance(name, bytes) else str(name)
            try:
                free, total = rt.memGetInfo()
            except Exception:  # noqa: BLE001
                free, total = 0, int(props.get("totalGlobalMem", 0))
            try:
                version = int(rt.runtimeGetVersion())
            except Exception:  # noqa: BLE001
                version = 0
            dev = Device(vendor="rocm" if bool(rt.is_hip) else "cuda", name=name, index=idx,
                         count=count, free_bytes=int(free), total_bytes=int(total),
                         cupy_version=str(cp.__version__), runtime_version=version,
                         integrated=bool(props.get("integrated", 0)))
        elif not why:
            why = f"CuPy {cp.__version__} imports but sees no device"
    except ImportError as exc:
        why = f"CuPy not installed ({exc})"
    except Exception as exc:  # noqa: BLE001
        why = f"CuPy device probe failed ({exc!r})"
    _CACHE["device"] = (dev, why)
    return dev, why


def device() -> Optional[Device]:
    """The current device, or None (no CuPy, no device, or SPDEE_NO_GPU)."""
    if disabled():
        return None
    return _probe_device()[0]


def vendor() -> Optional[str]:
    """"cuda", "rocm" or None."""
    dev = device()
    return None if dev is None else dev.vendor


def prefer_device() -> Tuple[bool, str]:
    """Whether an automatic choice may pick the device: (True, "") for a
    discrete or datacenter GPU of either vendor, (False, why) for an
    integrated one, no device, or SPDEE_NO_GPU.  Callers add the capability
    the choice needs (a dense LU for the census, cuDSS for the factor)."""
    if disabled():
        return False, "SPDEE_NO_GPU set"
    dev, why = _probe_device()
    if dev is None:
        return False, why
    if dev.integrated:
        return False, (f"{dev.name} is an integrated device sharing system memory; on such a "
                       "part the dense LU measured 0.4-0.5x host speed and the cupyx solves "
                       "33x slower; pass backend='cupy' or factor='cupy' to use it anyway")
    return True, ""


def _importable(name: str) -> Tuple[bool, str]:
    try:
        __import__(name)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{exc!r}"


def _xgeev_symbol() -> Tuple[bool, str]:
    """cusolverDnXgeev in the libcusolver CuPy has mapped (CUDA only)."""
    try:
        import cupy as cp
        from spdee.solvers.gpu_dense import _load_libcusolver
        cp.cuda.device.get_cusolver_handle()
        lib = _load_libcusolver()
        if hasattr(lib, "cusolverDnXgeev"):
            return True, ""
        return False, "cusolverDnXgeev missing from libcusolver"
    except Exception as exc:  # noqa: BLE001
        return False, f"libcusolver probe failed ({exc!r})"


def _probe_features(dev: Device) -> Dict[str, Tuple[bool, str]]:
    if "features" in _CACHE:
        return _CACHE["features"]  # type: ignore[return-value]
    feat: Dict[str, Tuple[bool, str]] = {}
    ok, why = _importable("cupyx.scipy.linalg")
    feat["dense_lu"] = (ok, "" if ok else f"cupyx.scipy.linalg not importable ({why})")
    ok, why = _importable("cupyx.scipy.sparse.linalg")
    feat["sparse_lu"] = (ok, "" if ok else f"cupyx.scipy.sparse.linalg not importable ({why})")
    if dev.vendor == "rocm":
        rocm = "ROCm: rocSOLVER has no nonsymmetric eigensolver; geev runs on the host"
        feat["xgeev"] = (False, rocm)
        feat["cupy_geev"] = (False, "ROCm: cp.linalg.eig raises 'geev is not available' on HIP")
        feat["cudss"] = (False, "ROCm: cuDSS is NVIDIA-only, no sparse direct solver on the device")
        feat["managed_memory"] = _managed_memory()
    else:
        if dev.runtime_version and dev.runtime_version < XGEEV_MIN_RUNTIME:
            feat["xgeev"] = (False, f"CUDA runtime {dev.runtime_version} predates Xgeev "
                                    f"(needs >= {XGEEV_MIN_RUNTIME})")
        else:
            feat["xgeev"] = _xgeev_symbol()
        feat["cupy_geev"] = _cupy_geev()
        ok, why = _importable("nvmath.sparse.advanced")
        feat["cudss"] = (ok, "" if ok else f"nvmath-python (cuDSS) not importable ({why})")
        feat["managed_memory"] = _managed_memory()
    _CACHE["features"] = feat
    return feat


def _cupy_geev() -> Tuple[bool, str]:
    """cp.linalg.eig by a trial call on a 2 x 2 matrix: the attribute exists
    on builds whose call raises ('geev is not available')."""
    try:
        import cupy as cp
        if not hasattr(cp.linalg, "eig"):
            return False, f"cupy {cp.__version__} has no cp.linalg.eig (needs >= 14)"
        w, _ = cp.linalg.eig(cp.eye(2, dtype=cp.complex128))
        cp.cuda.get_current_stream().synchronize()
        if w.shape != (2,):
            return False, "cp.linalg.eig trial call returned the wrong shape"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"cp.linalg.eig trial call failed ({exc!r})"


def _managed_memory() -> Tuple[bool, str]:
    """cupy.cuda.malloc_managed backs a pool on CUDA and on HIP alike."""
    try:
        import cupy as cp
        if not hasattr(cp.cuda, "malloc_managed"):
            return False, "cupy.cuda.malloc_managed missing"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"cupy not importable ({exc!r})"


def features() -> Dict[str, Tuple[bool, str]]:
    """Each GPU capability -> (available, reason when not)."""
    if disabled():
        why = "SPDEE_NO_GPU set"
        return {k: (False, why) for k in _FEATURE_NAMES}
    dev, why = _probe_device()
    if dev is None:
        return {k: (False, why) for k in _FEATURE_NAMES}
    return dict(_probe_features(dev))


def fallback(requested: str, why: str, taken: str) -> None:
    """The one warning of a degraded GPU path: what was asked, why it is
    unavailable, what runs instead."""
    warnings.warn(f"{requested} unavailable ({why}); {taken}", GPUFallbackWarning, stacklevel=3)
