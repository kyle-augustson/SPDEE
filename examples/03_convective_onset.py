"""Onset of Boussinesq convection in a rotating spherical shell: the critical
Rayleigh number at fixed m on two cores, from a growth-rate scan.

Physics.  A Boussinesq fluid in the shell 0.5 <= r <= 1 rotating at Omega about ez,
with uniform gravity g r_hat and a uniform conducting temperature gradient
dT0/dr = -beta.  Length r_o, time 1/Omega, velocity Omega r_o, temperature beta r_o:

    dt(v) + 2 ez x v + grad(p) - Ra T r_hat - Ek lap(v) = 0
    dt(T) - v_r - (Ek/Pr) lap(T) = 0                        (v.grad(T0) = -v_r)
    div(v) = 0

with Ek = nu / (Omega r_o^2), Pr = nu / kappa and Ra = alpha g beta / Omega^2, the
Rayleigh number of the rotational scaling (the classical one is Ra Pr / Ek^2).
Walls: impenetrable stress-free (add_stress_free) or no-slip, fixed temperature T = 0.
A mode grows when Im omega = -Re lambda > 0; at fixed m the critical Rayleigh number
Ra_c(m) is the zero of the growth rate of the least-damped mode, and the marginal
mode drifts at the frequency Re omega there.  Reference: the Dedalus core on the same
strings, at Ra = 2.5, Ek = 1e-2, Pr = 1, m = 2, stress-free, gives
lambda = -0.02510814 + 0.04842610 i, stable to 1e-8 between 16 and 24 radial modes
(tests/test_staggered_general.py); the C-grid converges to it at second order.

Demonstrates.  A symbolic parameter bound at compile time (add_parameter("Ra") with
no value, params={"Ra": value} at compile), and that a parameter entering the operator
linearly needs only two assemblies for a whole scan: K(Ra) = K0 + Ra K1, checked
against a third assembly to rounding; a growth-rate scan and the bracketing of its
zero (brentq on the least-damped eigenvalue, shift-invert seeded by the previous
eigenpair); the same problem on the Dedalus core (spectral) and on the staggered C-grid
at two resolutions (second order), with the marginal mode's temperature read back from
each (eigenvector_fields; the C-grid's cell-centre layout).

Run from the repository root: PYTHONPATH=. python examples/03_convective_onset.py
"""

from __future__ import annotations

import logging
import os
import sys
import time

import matplotlib
import numpy as np
from scipy.optimize import brentq

from spdee import Problem
from spdee.backends.fd2.grid import make_shell_grid
from spdee.solvers import shift_invert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("subsystems").setLevel(logging.WARNING)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
EK = 1e-2
PR = 1.0
M = 2
R_INNER = 0.5
RA_SCAN = (2.0, 2.5, 3.0, 3.5, 4.0)
DEDALUS_RES = (16, 32)
STAGGERED_RES = ((12, 24), (24, 48))


# ---------------------------------------------------------------------------
# The problem
# ---------------------------------------------------------------------------

def boussinesq_problem(Ek: float = EK, Pr: float = PR, wall: str = "stress_free") -> Problem:
    """Rotating Boussinesq convection with Ra left symbolic."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("T", "scalar")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("grad_T0", v_r=-1)
    pb.add_parameter("Ra")
    pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_parameter("Pr", Pr, symbolic=False)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - Ra*T*r_hat - Ek*lap(v) = 0",
                    name="momentum")
    pb.add_equation("dt(T) + dot(v, grad_T0) - (Ek/Pr)*lap(T) = 0", name="energy")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in ("inner", "outer"):
        if wall == "stress_free":
            pb.add_stress_free("v", at=w)
        else:
            for unit in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({unit}, v) = 0", at=w)
        pb.add_bc("T = 0", at=w)
    return pb


def compile_on(pb: Problem, core: str, res, Ra: float, m: int = M):
    """One core at one resolution with Ra bound; the C-grid takes N_r + 1 edges."""
    n_r, n_t = res
    if core == "dedalus":
        from spdee.backends.dedalus import DedalusGrid
        return pb.compile(DedalusGrid(N_r=n_r, N_theta=n_t, r_inner=R_INNER, r_outer=1.0),
                          m=m, backend="dedalus", gauge_unknowns=["p"], params={"Ra": Ra})
    grid = make_shell_grid(N_r=n_r + 1, N_theta=n_t, r_inner=R_INNER, r_outer=1.0)
    return pb.compile(grid, m=m, backend="fd2", discretization="staggered",
                      order_r=2, order_theta=2, params={"Ra": Ra})


class Pencil:
    """K(Ra) = K0 + Ra K1 and M of one core, from two compiles."""

    def __init__(self, pb, core, res, check=2.5):
        t0 = time.time()
        self.cp = compile_on(pb, core, res, 0.0)
        K0, self.M = self.cp.assemble_evp_matrices()
        K1, _ = compile_on(pb, core, res, 1.0).assemble_evp_matrices()
        self.K0, self.K1 = K0.tocsr(), (K1 - K0).tocsr()
        Kc, _ = compile_on(pb, core, res, check).assemble_evp_matrices()
        self.linearity = float(abs(Kc - self.K(check)).max() / abs(Kc).max())
        self.seconds = time.time() - t0
        self.N = K0.shape[0]

    def K(self, Ra):
        return self.K0 + Ra * self.K1


def least_damped(pencil: Pencil, Ra: float, guess: complex, v0=None, k: int = 6):
    """The eigenpair near ``guess`` of largest growth (smallest Re lambda), or,
    when the previous eigenvector ``v0`` is given, the one on its branch (largest
    overlap); v0 also seeds ARPACK."""
    lam, V = shift_invert(pencil.K(Ra), pencil.M, guess, k, return_vectors=True, v0=v0)
    keep = np.isfinite(lam)
    lam, V = lam[keep], V[:, keep]
    if v0 is None:
        j = int(np.argmin(lam.real))
    else:
        j = int(np.argmax(np.abs(v0.conj() @ V) / np.linalg.norm(V, axis=0)))
    return lam[j], V[:, j]


# ---------------------------------------------------------------------------
# Scan and bracket
# ---------------------------------------------------------------------------

def growth_scan(pencil: Pencil, ras=RA_SCAN, guess=-0.02 + 0.05j):
    """(Ra, lambda) of the least-damped mode along the scan, each solve seeded
    by the previous eigenpair."""
    out, v0 = [], None
    for Ra in ras:
        lam, v0 = least_damped(pencil, Ra, guess, v0=v0)
        guess = lam + 0.01
        out.append((Ra, lam))
    return out


def critical_rayleigh(pencil: Pencil, scan):
    """Ra_c and the marginal eigenpair by bracketing Re lambda = 0 on the scan."""
    ras = [Ra for Ra, _ in scan]
    growth = [-lam.real for _, lam in scan]
    i = next(i for i in range(len(ras) - 1) if growth[i] < 0 <= growth[i + 1])
    state = {"guess": scan[i][1] + 0.01, "v0": None}

    def f(Ra):
        lam, v = least_damped(pencil, Ra, state["guess"], v0=state["v0"])
        state["guess"], state["v0"], state["lam"] = lam + 0.01, v, lam
        return -lam.real

    Ra_c = brentq(f, ras[i], ras[i + 1], xtol=1e-6)
    return Ra_c, state["lam"], state["v0"]


def temperature_on_grid(pencil: Pencil, core: str, x):
    """(theta, r, T) of the marginal mode from the core's own readback."""
    cp = pencil.cp
    if core == "dedalus":
        return cp.theta_grid, cp.r_grid, cp.eigenvector_fields(x)["T"]
    g = cp.grids
    T = x[cp.layout.slice_of("T")].reshape(len(g.r_center), len(g.theta_center))
    return g.theta_center, g.r_center, T.T


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(cores=(("dedalus", DEDALUS_RES),) + tuple(("staggered", r) for r in STAGGERED_RES),
        verbose=True):
    pb = boussinesq_problem()
    results = {}
    for core, res in cores:
        pencil = Pencil(pb, core, res)
        scan = growth_scan(pencil)
        Ra_c, lam_c, x_c = critical_rayleigh(pencil, scan)
        results[(core, res)] = (pencil, scan, Ra_c, lam_c, x_c)
        if verbose:
            print(f"\n{core} {res[0]}x{res[1]}: N = {pencil.N}, two compiles + check in "
                  f"{pencil.seconds:.1f} s, |K(2.5) - K0 - 2.5 K1| / |K| = {pencil.linearity:.1e}")
            print("    Ra      growth -Re lambda   frequency Im lambda")
            for Ra, lam in scan:
                print(f"  {Ra:5.2f}    {-lam.real:+.6f}          {lam.imag:+.6f}")
            print(f"  Ra_c = {Ra_c:.6f}: the marginal mode drifts at Re omega = Im lambda = "
                  f"{lam_c.imag:+.6f} with growth {-lam_c.real:+.1e}")
    if verbose:
        ref = results[("dedalus", DEDALUS_RES)]
        print(f"\nRa_c against the Dedalus value {ref[2]:.6f}:")
        for (core, res), (_, _, Ra_c, lam_c, _) in results.items():
            print(f"  {core:9s} {res[0]:3d}x{res[1]:<3d}  Ra_c = {Ra_c:.6f}  "
                  f"err {abs(Ra_c - ref[2]):.1e}  drift {lam_c.imag:+.6f}  "
                  f"err {abs(lam_c.imag - ref[3].imag):.1e}")
        e = [abs(results[("staggered", r)][2] - ref[2]) for r in STAGGERED_RES]
        print(f"  C-grid order in Ra_c: {np.log(e[0] / e[1]) / np.log(2):.2f}")
    return results


def figure(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    for (core, res), (_, scan, Ra_c, _, _) in results.items():
        ax.plot([Ra for Ra, _ in scan], [-lam.real for _, lam in scan], "o-",
                label=f"{core} {res[0]}x{res[1]}: Ra_c = {Ra_c:.4f}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Ra")
    ax.set_ylabel("growth rate  Im omega = -Re lambda")
    ax.set_title(f"m = {M}, Ek = {EK}, Pr = {PR}, stress-free")
    ax.legend(fontsize=8)
    shown = (("dedalus", DEDALUS_RES), ("staggered", STAGGERED_RES[-1]))
    for ax, (core, res) in zip(axes[1:], shown):
        pencil, _, Ra_c, lam_c, x = results[(core, res)]
        th, r, T = temperature_on_grid(pencil, core, x)
        T = T / T.ravel()[np.argmax(np.abs(T))]
        R, TH = np.meshgrid(r, th)
        ax.pcolormesh(R * np.sin(TH), R * np.cos(TH), T.real, cmap="RdBu_r", vmin=-1, vmax=1,
                      shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"T of the marginal mode, {core} {res[0]}x{res[1]}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    results = run()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "03_convective_onset.png")
    figure(results, path)
    print(f"\nfigure: {path}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
