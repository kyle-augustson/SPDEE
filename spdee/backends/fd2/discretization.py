"""spdee.backends.fd2.discretization — the discretization-method option.

The fd2 backend offers two spatial discretizations of the same IR
problem, selected by the ``discretization`` argument of
``compile_problem``. The default is ``COLLOCATED``.

Methods
-------
COLLOCATED
    All unknowns (scalars, vector and tensor components) share the same
    ``(r, θ)`` nodes: r includes both endpoints, θ is cell-centered with
    the poles excluded. Handled by ``compiler`` / ``compiled_problem``.

STAGGERED
    Arakawa-C layout: scalars and v_φ at cell centers, v_r on r-edges,
    v_θ on θ-edges. Removes the velocity-pressure checkerboard; the
    adjoint-consistent assembly under the sinθ volume norm gives a
    skew (purely imaginary) inertial spectrum with parity-based pole
    regularity. Assembled by ``staggered_general`` from the walker's
    tasks, on the blocks of ``staggered_compile`` / ``staggered_mhd``.

SH_FD
    Spherical-harmonic angular basis with finite-difference radial
    discretization. Lives in the separate package
    ``spdee.backends.sh_fd`` and is selected through the backend
    registry, not through this enum; requesting it here raises.

DEDALUS
    Fully spectral (Chebyshev × spherical harmonics) discretization.
    Lives in the separate package ``spdee.backends.dedalus`` and is
    selected through the backend registry, not through this enum;
    requesting it here raises.
"""

from __future__ import annotations

from enum import Enum


class Discretization(Enum):
    """Selectable spatial discretization method."""

    COLLOCATED = "collocated"
    STAGGERED = "staggered"
    SH_FD = "sh_fd"
    DEDALUS = "dedalus"

    @classmethod
    def coerce(cls, value) -> "Discretization":
        """Normalize a Discretization or a string alias to a member.

        Strings are matched case-insensitively against the member name,
        the member value, and the alias table below.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            key = value.strip().lower()
            aliases = {
                "colocated": cls.COLLOCATED,
                "collocated": cls.COLLOCATED,
                "staggered": cls.STAGGERED,
                "arakawa-c": cls.STAGGERED,
                "arakawa_c": cls.STAGGERED,
                "cgrid": cls.STAGGERED,
                "c-grid": cls.STAGGERED,
                "gilman": cls.STAGGERED,
                "solar_eigenproblem": cls.STAGGERED,
                "sh_fd": cls.SH_FD,
                "sh-fd": cls.SH_FD,
                "shfd": cls.SH_FD,
                "dedalus": cls.DEDALUS,
                "spectral": cls.DEDALUS,
            }
            if key in aliases:
                return aliases[key]
            for member in cls:
                if key in (member.name.lower(), member.value.lower()):
                    return member
        raise ValueError(
            f"unknown discretization {value!r}; valid options are "
            f"{[m.value for m in cls]}"
        )

    @property
    def is_implemented(self) -> bool:
        """True for the methods the fd2 pipeline can build."""
        return self in (Discretization.COLLOCATED, Discretization.STAGGERED)


DEFAULT_DISCRETIZATION = Discretization.COLLOCATED


class DiscretizationNotImplementedError(NotImplementedError):
    """Raised when a method the fd2 pipeline cannot build is selected."""


def require_implemented(method: Discretization) -> None:
    """Raise for SH_FD and DEDALUS, which are not built by fd2."""
    if method.is_implemented:
        return
    msg = {
        Discretization.SH_FD: (
            "the 'sh_fd' discretization (spherical-harmonic angular basis "
            "+ FD radial) is provided by the separate spdee.backends.sh_fd "
            "package, selected through the backend registry; it cannot be "
            "built by the fd2 pipeline."
        ),
        Discretization.DEDALUS: (
            "the 'dedalus' discretization (fully spectral Chebyshev × "
            "spherical harmonics) is provided by the separate "
            "spdee.backends.dedalus package, selected through the backend "
            "registry; it cannot be built by the fd2 pipeline."
        ),
    }.get(method, f"discretization {method.value!r} is not implemented.")
    raise DiscretizationNotImplementedError(msg)
