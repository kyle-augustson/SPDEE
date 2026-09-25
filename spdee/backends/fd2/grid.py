"""spdee.backends.fd2.grid — tensor-product grids in (r, theta).

Two geometries:

  - 'shell': r in [r_inner, r_outer] with r_inner > 0. Both endpoints
    are grid points, since boundary conditions live there.

  - 'ball': r in [0, r_outer]; r[0] == 0.0 exactly. The 1/r and 1/r²
    factors of the spherical operators are singular there, so the
    operator builders multiply each equation through by a power of r,
    write the operators' r→0 limits into the i_r=0 rows (ball_origin.py)
    and `CompiledProblem._apply_ball_origin_regularity` reduces the
    i_r=0 unknowns to the regular degrees of freedom.

Theta grid: theta in (0, pi), poles excluded. The default sampling is
cell-centered uniform, theta_j = (j + 1/2) * pi / N_theta, so that
sin(theta) never vanishes at a node and cot(theta), 1/sin(theta) stay
finite.

`Grid` is a passive data container; stencils.py and the operator
modules build the operators on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A tensor-product (r, theta) grid.

    Attributes
    ----------
    r : np.ndarray
        1D radial grid points, strictly increasing.
    theta : np.ndarray
        1D colatitude grid points, strictly increasing, in (0, pi).
    geometry : str
        'ball' (r[0] == 0.0 exactly) or 'shell' (r[0] > 0).

    A 2D field on the grid has shape (N_r, N_theta); the flat storage
    order is defined in layout.py, not here.
    """
    r: np.ndarray
    theta: np.ndarray
    geometry: str

    def __post_init__(self):
        if self.geometry not in ("ball", "shell"):
            raise ValueError(
                f"geometry must be 'ball' or 'shell'; got {self.geometry!r}"
            )
        r = np.asarray(self.r, dtype=float)
        if r.ndim != 1:
            raise ValueError(f"r must be 1D; got shape {r.shape}")
        if len(r) < 2:
            raise ValueError(f"r must have at least 2 points; got {len(r)}")
        if not np.all(np.diff(r) > 0):
            raise ValueError("r must be strictly monotonically increasing")
        if self.geometry == "ball":
            if r[0] != 0.0:
                raise ValueError(
                    f"ball geometry requires r[0] == 0.0 exactly (the "
                    f"masses and origin rows test for it); got {r[0]!r}"
                )
        else:
            if r[0] <= 0:
                raise ValueError(
                    f"shell geometry requires r[0] > 0; got {r[0]}"
                )
        theta = np.asarray(self.theta, dtype=float)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D; got shape {theta.shape}")
        if len(theta) < 2:
            raise ValueError(
                f"theta must have at least 2 points; got {len(theta)}"
            )
        if not np.all(np.diff(theta) > 0):
            raise ValueError("theta must be strictly monotonically increasing")
        if theta[0] <= 0:
            raise ValueError(
                f"theta grid must exclude theta=0 (pole); got theta[0]={theta[0]}. "
                f"Use uniform_theta() or pass cell-centered theta values."
            )
        if theta[-1] >= np.pi:
            raise ValueError(
                f"theta grid must exclude theta=pi (pole); got "
                f"theta[-1]={theta[-1]}."
            )
        object.__setattr__(self, "r", r.copy())
        object.__setattr__(self, "theta", theta.copy())

    @property
    def N_r(self) -> int:
        return len(self.r)

    @property
    def N_theta(self) -> int:
        return len(self.theta)

    @property
    def shape(self) -> Tuple[int, int]:
        """(N_r, N_theta)."""
        return (self.N_r, self.N_theta)

    @property
    def size(self) -> int:
        """N_r * N_theta."""
        return self.N_r * self.N_theta

    @property
    def r_inner(self) -> float:
        return float(self.r[0])

    @property
    def r_outer(self) -> float:
        return float(self.r[-1])

    @property
    def is_ball(self) -> bool:
        return self.geometry == "ball"

    @property
    def is_shell(self) -> bool:
        return self.geometry == "shell"

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Broadcast (R, Theta) arrays of shape (N_r, N_theta)."""
        R, Theta = np.meshgrid(self.r, self.theta, indexing="ij")
        return R, Theta

    def __repr__(self) -> str:
        return (
            f"Grid({self.geometry}, N_r={self.N_r}, N_theta={self.N_theta}, "
            f"r in [{self.r_inner:.3g}, {self.r_outer:.3g}])"
        )


def uniform_theta(N_theta: int) -> np.ndarray:
    """Cell-centered uniform theta grid, theta_j = (j + 1/2) * pi / N_theta."""
    if N_theta < 2:
        raise ValueError(f"N_theta must be >= 2; got {N_theta}")
    j = np.arange(N_theta)
    return (j + 0.5) * np.pi / N_theta


def uniform_r(N_r: int, r_inner: float, r_outer: float) -> np.ndarray:
    """Uniform radial grid of N_r points including both endpoints."""
    if N_r < 2:
        raise ValueError(f"N_r must be >= 2; got {N_r}")
    if not (r_inner < r_outer):
        raise ValueError(
            f"r_inner < r_outer required; got [{r_inner}, {r_outer}]"
        )
    return np.linspace(r_inner, r_outer, N_r)


def make_ball_grid(
    N_r: int,
    N_theta: int,
    r_outer: float = 1.0,
    r: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
) -> Grid:
    """Ball grid: uniform r in [0, r_outer] and cell-centered theta.

    Explicit `r` (must start at 0) or `theta` (must exclude the poles)
    arrays override N_r / r_outer and N_theta respectively.
    """
    if r is None:
        r = uniform_r(N_r, 0.0, r_outer)
    if theta is None:
        theta = uniform_theta(N_theta)
    return Grid(r=r, theta=theta, geometry="ball")


def make_shell_grid(
    N_r: int,
    N_theta: int,
    r_inner: float,
    r_outer: float = 1.0,
    r: Optional[np.ndarray] = None,
    theta: Optional[np.ndarray] = None,
) -> Grid:
    """Shell grid: uniform r in [r_inner, r_outer] (r_inner > 0) and
    cell-centered theta. Explicit `r` / `theta` arrays override the
    resolution and radii arguments.
    """
    if r is None:
        r = uniform_r(N_r, r_inner, r_outer)
    if theta is None:
        theta = uniform_theta(N_theta)
    return Grid(r=r, theta=theta, geometry="shell")
