"""spdee.backends — pluggable discretization backends.

Registered backends:

  - 'fd2'     — explicit centered finite differences in r and theta
                (collocated, or the staggered Arakawa-C grid with
                ``discretization='staggered'``).
  - 'sh_fd'   — spin-weighted spherical harmonics in (theta, phi),
                finite differences in r.
  - 'dedalus' — hand-off to Dedalus.

The dispatcher:

    >>> from spdee.backends import compile_problem
    >>> cp = compile_problem(system, backend='fd2', grid=grid, m=2, **backend_kwargs)
    >>> result = cp.solve_evp(k=10, sigma=0.0)

A backend's compile function returns a compiled-problem object that
assembles the pencil K x = lambda M x (``assemble_evp_matrices``) and
exposes ``solve_evp``.

To add a backend:

  1. Create a subpackage under spdee/backends/.
  2. Implement a 'compile(system, **kwargs)' function returning such an object.
  3. Register it with `register_backend` at the bottom of this module.
"""

from __future__ import annotations

from typing import Any, Dict

# Backend registry: name -> compile function.
_BACKENDS: Dict[str, Any] = {}


def register_backend(name: str, compile_fn) -> None:
    """Register a backend's compile function under `name`."""
    if name in _BACKENDS:
        raise ValueError(f"Backend {name!r} already registered")
    _BACKENDS[name] = compile_fn


def available_backends() -> tuple:
    """Sorted names of all registered backends."""
    return tuple(sorted(_BACKENDS.keys()))


def compile_problem(system, backend: str, **kwargs):
    """Compile an EquationSystem with the named backend.

    Parameters
    ----------
    system : EquationSystem
        The IR-level problem specification.
    backend : str
        One of available_backends().
    **kwargs : backend-specific
        Configuration passed through to the backend's compile function
        (grid type, resolution, geometry options, etc.).

    Returns
    -------
    A backend-specific compiled problem with ``assemble_evp_matrices`` and
    ``solve_evp``.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}; available: {available_backends()}"
        )
    return _BACKENDS[backend](system, **kwargs)


from spdee.backends.fd2 import compile as _fd2_compile
from spdee.backends.sh_fd import compile as _sh_fd_compile
from spdee.backends.dedalus import compile as _dedalus_compile

register_backend("fd2", _fd2_compile)
register_backend("sh_fd", _sh_fd_compile)
register_backend("dedalus", _dedalus_compile)
