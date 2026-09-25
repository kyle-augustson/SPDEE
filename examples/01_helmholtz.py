"""Scalar Helmholtz eigenvalues in a ball and a shell on the four cores.

Physics.  The heat equation dt(u) = lap(u) separates as u = R(r) Y_lm e^{lambda t}
with R = a j_l(k r) + b y_l(k r), so every pencil eigenvalue is lambda = k^2 with
k a root of the wall condition: Dirichlet u = 0, Neumann du/dr = 0 or Robin
du/dn + h u = 0 (outward normal, so the inner wall of the shell reads
du/dr - h u = 0).  In a ball regularity drops y_l and k solves the single wall
condition on j_l; in the shell 0.5 <= r <= 1 it solves the 2 x 2 determinant of
the two walls.  The reference values are these spherical Bessel roots, found by
bracketing (docs/analytic_benchmarks.md, "Radial eigenvalue conditions").

Demonstrates.  One Problem stated once from strings and compiled on every core
(collocated FD, staggered Arakawa-C, spin-weighted harmonics x radial FD, Dedalus);
spdee.solvers.dense for a full census of a small pencil; the eigenvalue convention
omega = -i lambda (here lambda = k^2 > 0 is a pure decay, omega = -i k^2); the
measured order of each core against resolution (4 for the collocated and sh_fd
stencils used, 2 for the C-grid, spectral for Dedalus); and how each core hands an
eigenvector back onto a meridional grid (layout.unpack, unpack + to_grid,
eigenvector_fields).

Run from the repository root: PYTHONPATH=. python examples/01_helmholtz.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import matplotlib
import numpy as np
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from spdee import Problem
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.sh_fd import ShFdGrid
from spdee.solvers import dense

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("subsystems").setLevel(logging.WARNING)

CORES = ("fd2", "staggered", "sh_fd", "dedalus")
WALLS = ("dirichlet", "neumann", "robin")
R_INNER = 0.5
H_ROBIN = 2.0
M = 1
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

# three resolutions per core: (N_r, N_theta) for the FD cores, (N_r, ell_max - m) for
# sh_fd and (N_r, N_theta) for Dedalus; the order is measured over the last two levels,
# whose radial ratio is RATIO (the staggered ladder is not uniform between its first two)
RESOLUTIONS = {
    "fd2": ((12, 18), (18, 27), (27, 40)),
    "staggered": ((12, 18), (24, 36), (36, 54)),
    "sh_fd": ((16, 6), (32, 6), (64, 6)),
    "dedalus": ((8, 8), (12, 12), (16, 16)),
}
RATIO = {"fd2": 1.5, "staggered": 1.5, "sh_fd": 2.0, "dedalus": None}


# ---------------------------------------------------------------------------
# The problem, stated once
# ---------------------------------------------------------------------------

def helmholtz_problem(wall: str, geometry: str, h: float = H_ROBIN) -> Problem:
    """dt(u) = lap(u) with the named wall on every boundary of the geometry."""
    pb = Problem()
    pb.add_unknown("u", "scalar")
    pb.add_parameter("h", h)
    pb.add_equation("dt(u) - lap(u) = 0", name="heat")
    rows = {
        "dirichlet": {"outer": "u = 0", "inner": "u = 0"},
        "neumann": {"outer": "dot(r_hat, grad(u)) = 0", "inner": "dot(r_hat, grad(u)) = 0"},
        "robin": {"outer": "dot(r_hat, grad(u)) + h*u = 0",
                  "inner": "dot(r_hat, grad(u)) - h*u = 0"},
    }[wall]
    pb.add_bc(rows["outer"], at="outer")
    if geometry == "shell":
        pb.add_bc(rows["inner"], at="inner")
    return pb


# ---------------------------------------------------------------------------
# The closed form: spherical Bessel roots of the wall condition
# ---------------------------------------------------------------------------

def _wall(f, l, k, radius, wall, h, outward):
    x = k * radius
    if wall == "dirichlet":
        return f(l, x)
    if wall == "neumann":
        return k * f(l, x, derivative=True)
    return k * f(l, x, derivative=True) + outward * h * f(l, x)


def _determinant(k, l, wall, geometry, h):
    if geometry == "ball":
        return _wall(spherical_jn, l, k, 1.0, wall, h, +1)
    return (_wall(spherical_jn, l, k, R_INNER, wall, h, -1)
            * _wall(spherical_yn, l, k, 1.0, wall, h, +1)
            - _wall(spherical_jn, l, k, 1.0, wall, h, +1)
            * _wall(spherical_yn, l, k, R_INNER, wall, h, -1))


def bessel_rate(l: int, wall: str, geometry: str, h: float = H_ROBIN) -> float:
    """lambda = k^2 of the gravest radial mode of degree l."""
    ks = np.linspace(0.2, 40.0, 4000)
    vals = np.array([_determinant(k, l, wall, geometry, h) for k in ks])
    for i in range(len(ks) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            k = brentq(lambda x: _determinant(x, l, wall, geometry, h), ks[i], ks[i + 1],
                       xtol=1e-14)
            return k ** 2
    raise RuntimeError(f"no root for l={l} {wall} {geometry}")


# ---------------------------------------------------------------------------
# One Problem, four cores
# ---------------------------------------------------------------------------

def compile_on(pb: Problem, core: str, geometry: str, m: int, level: int):
    """Compile ``pb`` on one core at one resolution level."""
    a, b = RESOLUTIONS[core][level]
    ri = 0.0 if geometry == "ball" else R_INNER
    if core == "sh_fd":
        return pb.compile(ShFdGrid(N_r=a, ell_max=m + b, r_inner=ri, r_outer=1.0, fd_order=4),
                          m=m, backend="sh_fd")
    if core == "dedalus":
        from spdee.backends.dedalus import DedalusGrid
        return pb.compile(DedalusGrid(N_r=a, N_theta=b, r_inner=ri, r_outer=1.0),
                          m=m, backend="dedalus")
    # the C-grid takes the N_r nodes as radial edges, so N_r + 1 nodes give N_r cells
    n_r = a + 1 if core == "staggered" else a
    grid = (make_ball_grid(N_r=n_r, N_theta=b) if geometry == "ball"
            else make_shell_grid(N_r=n_r, N_theta=b, r_inner=ri))
    if core == "staggered":
        return pb.compile(grid, m=m, backend="fd2", discretization="staggered",
                          order_r=2, order_theta=2)
    return pb.compile(grid, m=m, backend="fd2", order_r=4, order_theta=4)


def census(cp, sigma: float):
    """Every finite eigenvalue of the compiled pencil, with the eigenvectors.

    The dense census runs through the shift-invert transform at ``sigma``;
    the infinite eigenvalues of the wall rows (zero rows of M) are filtered.
    """
    K, M_ = cp.assemble_evp_matrices()
    out = dense(K, M_, sigma=sigma, backend="auto", vectors=True, keep_vectors=True)
    keep = np.isfinite(out.lam)
    return out.lam[keep], out.V[:, keep]


def eigenfunction(cp, core: str, x: np.ndarray, theta_plot: np.ndarray):
    """The scalar unknown of a raw eigenvector on a meridional (theta, r) grid.

    Each core exposes its own readback: the collocated layout unpacks to
    (N_r, N_theta) arrays; the C-grid stores cell-centre values theta fastest;
    sh_fd unpacks to spectral coefficients that ``to_grid`` synthesizes on any
    theta; Dedalus loads the state into its fields and returns grid data.
    """
    if core == "fd2":
        u = cp.matrices.layout.unpack(x, role="primary")["u"]
        return cp.matrices.grid.theta, cp.matrices.grid.r, u.T
    if core == "staggered":
        g = cp.grids
        u = x[cp.layout.slice_of("u")].reshape(len(g.r_center), len(g.theta_center))
        return g.theta_center, g.r_center, u.T
    if core == "sh_fd":
        coeffs = cp.unpack(x)["u"][0]
        return theta_plot, cp.grid.r, cp.to_grid(coeffs, theta_plot)
    return cp.theta_grid, cp.r_grid, cp.eigenvector_fields(x)["u"]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def nearest(lam: np.ndarray, target: float) -> complex:
    return complex(lam[np.argmin(np.abs(lam - target))])


def run(walls=WALLS, geometries=("ball", "shell"), cores=CORES, levels=(0, 1, 2), m=M,
        verbose=True):
    """The table of every (wall, geometry, core, level) against the closed form.

    Returns {(wall, geometry, core): (targets, values[level][target])} and the
    eigenvector data of the finest Robin shell solve per core for the figure.
    """
    results, pictures = {}, {}
    for wall in walls:
        for geometry in geometries:
            pb = helmholtz_problem(wall, geometry)
            ells = (m, m + 1)
            targets = [bessel_rate(l, wall, geometry) for l in ells]
            if verbose:
                print(f"\n{wall} {geometry} m={m}: lambda = k^2 for l = {ells}")
                print("  analytic " + "  ".join(f"{t:12.6f}" for t in targets)
                      + "   (omega = -i lambda: decay rates, no oscillation)")
            for core in cores:
                values = []
                for level in levels:
                    t0 = time.time()
                    cp = compile_on(pb, core, geometry, m, level)
                    lam, V = census(cp, sigma=float(np.mean(targets)))
                    values.append([nearest(lam, t).real for t in targets])
                    if verbose:
                        errs = [abs(v - t) / t for v, t in zip(values[-1], targets)]
                        res = "x".join(str(n) for n in RESOLUTIONS[core][level])
                        print(f"  {core:9s} {res:7s} {lam.size:5d} finite  "
                              + "  ".join(f"{v:12.6f}" for v in values[-1])
                              + "   rel err " + " ".join(f"{e:.1e}" for e in errs)
                              + f"   {time.time() - t0:4.1f} s")
                    if wall == "robin" and geometry == "shell" and level == levels[-1]:
                        j = int(np.argmin(np.abs(lam - targets[0])))
                        pictures[core] = (cp, V[:, j], lam[j])
                if verbose and RATIO[core]:
                    e = np.array([[abs(v - t) / t for v, t in zip(vals, targets)]
                                  for vals in values])
                    orders = np.log(e[-2] / e[-1]) / np.log(RATIO[core])
                    print(f"  {core:9s} order over the last two levels: "
                          + " ".join(f"{o:.2f}" for o in orders))
                results[(wall, geometry, core)] = (targets, values)
    return results, pictures


def figure(results, pictures, path):
    """Convergence of the Robin shell l = 1 rate per core, and its eigenfunction
    read back from each core on the meridional plane."""
    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 4, height_ratios=(1.2, 1.0))
    ax = fig.add_subplot(grid[0, :2])
    for core in CORES:
        targets, values = results[("robin", "shell", core)]
        n = [RESOLUTIONS[core][lv][0] for lv in range(len(values))]
        err = [abs(v[0] - targets[0]) / targets[0] for v in values]
        ax.loglog(n, np.maximum(err, 1e-16), "o-", label=core)
    ax.set_xlabel("N_r")
    ax.set_ylabel("relative error, Robin shell l = 1")
    ax.legend()
    ax = fig.add_subplot(grid[0, 2:])
    ax.axis("off")
    ax.text(0.0, 0.5, "dt(u) = lap(u)\nlambda = k^2, omega = -i k^2\n"
            "Robin walls du/dn + h u = 0, h = 2\nshell 0.5 <= r <= 1, m = 1\n"
            "bottom: Re u of the l = 1 mode read back from each core",
            fontsize=11, va="center")
    theta_plot = np.linspace(0.02, np.pi - 0.02, 91)
    for k, core in enumerate(CORES):
        ax = fig.add_subplot(grid[1, k])
        cp, x, lam = pictures[core]
        th, r, u = eigenfunction(cp, core, x, theta_plot)
        u = np.real(u / u.ravel()[np.argmax(np.abs(u))])
        R, T = np.meshgrid(r, th)
        ax.pcolormesh(R * np.sin(T), R * np.cos(T), u, cmap="RdBu_r", vmin=-1, vmax=1,
                      shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"{core}: lambda = {lam.real:.5f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    results, pictures = run()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "01_helmholtz.png")
    figure(results, pictures, path)
    print(f"\nfigure: {path}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
