"""The maintained FPLUS and BBJD libraries, loaded from a build outside the tree.

SPDEE carries copies of both (``spdee.solvers.fplus``, ``spdee.solvers.bbjd``).
The maintained releases (github.com/kyle-augustson/FPLUS and .../BBJD) are
used instead when the environment names a build of them:

  SPDEE_FPLUS_PATH (else FPLUS_PATH)   the directory that CONTAINS the FPLUS package
  SPDEE_BBJD_PATH  (else BBJD_PATH)    the directory that contains the BBJD package

The FPLUS release is the C library (int64 supernodal partial pivoting,
blocked multi-right-hand-side solves, automatic relaxation for small
matrices, ordering cache, the optional cuBLAS update path ``gpu=True`` on a
device with coherent host memory such as the GH200).  SPDEE's copy predates
it and has none of those.  The BBJD release and SPDEE's copy are the same
code; the switch exists so a site build (its own BLAS) can be used.

``library`` (a factor option for FPLUS, an eigensolver option for bbjd):
  'auto'      the external build when its variable is set, else the copy
  'external'  the external build; raises when it is absent or not compiled
  'vendored'  SPDEE's copy

A named build must have its C extension loaded: an external FPLUS on its
pure-Python fallback, or one that could not find SuiteSparse's libcolamd
(it would then order with SuperLU's COLAMD, the lost-fill trigger seen in
production), raises rather than run silently slow or wrong.
"""
from __future__ import annotations

import ctypes
import importlib
import os
import sys
from typing import Optional, Tuple

__all__ = ["fplus_path", "bbjd_path", "load_fplus", "load_bbjd", "LIBRARIES"]

LIBRARIES = ("auto", "external", "vendored")

_ENV = {"FPLUS": ("SPDEE_FPLUS_PATH", "FPLUS_PATH"), "BBJD": ("SPDEE_BBJD_PATH", "BBJD_PATH")}
_cache = {}


class ExternalLibraryError(ImportError):
    """A requested external solver library is absent or not usable."""


def _path(pkg: str) -> Optional[str]:
    for var in _ENV[pkg]:
        p = os.environ.get(var)
        if p:
            return os.path.abspath(os.path.expanduser(p))
    return None


def fplus_path() -> Optional[str]:
    """The directory named for an external FPLUS build, or None."""
    return _path("FPLUS")


def bbjd_path() -> Optional[str]:
    """The directory named for an external BBJD build, or None."""
    return _path("BBJD")


def _import_from(pkg: str, path: str):
    """Import package ``pkg`` from ``path`` (the directory containing it).

    A package of that name already imported from elsewhere is an error: two
    builds in one process would mix their extension modules."""
    key = (pkg, path)
    if key in _cache:
        return _cache[key]
    if not os.path.isfile(os.path.join(path, pkg, "__init__.py")):
        raise ExternalLibraryError(f"{_ENV[pkg][0]}={path}: no {pkg}/__init__.py there (set it to the "
                                   f"directory that contains the {pkg} package)")
    mod = sys.modules.get(pkg)
    if mod is not None:
        have = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__)))
        if have != path:
            raise ExternalLibraryError(f"{pkg} is already imported from {have}, not {path}")
        _cache[key] = mod
        return mod
    sys.path.insert(0, path)
    try:
        mod = importlib.import_module(pkg)
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass
    _cache[key] = mod
    return mod


def _check_colamd() -> None:
    for name in ("libcolamd.so.3", "libcolamd.so", "libcolamd.dylib"):
        try:
            ctypes.CDLL(name)
            return
        except OSError:
            pass
    raise ExternalLibraryError("external FPLUS: SuiteSparse's libcolamd is not on the library path, so "
                               "FPLUS would order with SuperLU's COLAMD; add its lib directory to "
                               "LD_LIBRARY_PATH")


def _choose(pkg: str, library: str) -> Optional[str]:
    if library not in LIBRARIES:
        raise ValueError(f"unknown library {library!r}; choose from {LIBRARIES}")
    if library == "vendored":
        return None
    p = _path(pkg)
    if p is None and library == "external":
        raise ExternalLibraryError(f"library='external' but neither {' nor '.join(_ENV[pkg])} is set")
    return p


def load_fplus(library: str = "auto") -> Tuple[type, str, Optional[str]]:
    """(SparseLU class, 'external' | 'vendored', path or None)."""
    p = _choose("FPLUS", library)
    if p is None:
        from spdee.solvers.fplus import SparseLU
        return SparseLU, "vendored", None
    mod = _import_from("FPLUS", p)
    impl = sys.modules.get("FPLUS.sparse_lu_full_pivot")
    if impl is None or not getattr(impl, "_HAVE_C_EXT", False):
        raise ExternalLibraryError(f"external FPLUS at {p}: the C extension did not load (pure-Python "
                                   f"fallback); build it there")
    _check_colamd()
    return mod.SparseLU, "external", p


def load_bbjd(library: str = "auto"):
    """(jdqz function, 'external' | 'vendored', path or None)."""
    p = _choose("BBJD", library)
    if p is None:
        from spdee.solvers.bbjd import jdqz, HAVE_C_EXT
        if not HAVE_C_EXT:
            raise ImportError("spdee.solvers.bbjd: the C extension is not built; spdee/solvers/build_ext.sh")
        return jdqz, "vendored", None
    mod = _import_from("BBJD", p)
    impl = sys.modules.get("BBJD.bbjd")
    if impl is None or getattr(impl, "_bbjd_c", None) is None:
        raise ExternalLibraryError(f"external BBJD at {p}: the C extension _bbjd_c did not load; build it there")
    return mod.jdqz, "external", p


def describe() -> dict:
    """What each external variable names and whether it loads (for ``spdee.check``)."""
    out = {}
    for pkg, loader in (("FPLUS", load_fplus), ("BBJD", load_bbjd)):
        p = _path(pkg)
        if p is None:
            out[pkg.lower()] = {"path": None, "ok": False, "why": f"{_ENV[pkg][0]} not set"}
            continue
        try:
            loader("external")
            out[pkg.lower()] = {"path": p, "ok": True, "why": ""}
        except Exception as exc:  # noqa: BLE001
            out[pkg.lower()] = {"path": p, "ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return out
