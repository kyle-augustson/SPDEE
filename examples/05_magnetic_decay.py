"""Free decay of a magnetic field in a perfectly conducting shell, written three
ways: the field with a divergence constraint, the vector potential in the Coulomb
gauge, and the field closed by the conducting wall alone.

Physics.  In a conductor at rest the field obeys dt(b) = eta lap(b) with div(b) = 0,
and a perfectly conducting wall imposes b_r = 0 and a vanishing tangential electric
field, i.e. the tangential components of curl(b) = 0 (eta = 1 here).  Writing
b = curl(curl(S r)) + curl(T r), both potentials obey dt X = lap X, so at m = 0 the
toroidal and poloidal families decouple and every decay rate is lambda = k^2 with
X = a j_l(k r) + b y_l(k r) in 0.5 <= r <= 1:

    poloidal   b_r = l(l+1) S / r^2 gives S = 0 at both walls (which also kills the
               toroidal wall electric field, proportional to lap S = -k^2 S)
    toroidal   the poloidal wall electric field is proportional to d(r T)/dr, so the
               family closes on d(r T)/dr = 0 at both walls.

Targets: toroidal l = 1 at k = 1.984570 and 6.618465, poloidal l = 1 and 2 at
k = 6.572013 and 7.111576 (docs/analytic_benchmarks.md, section 4).

Three spellings.  'b' evolves the field and writes div(b) = 0 with no multiplier;
'A' evolves the vector potential, dt(A) - lap(A) - grad(psi) = 0 with div(A) = 0 as a
saddle point on the multiplier psi, tangential A = 0 and psi = 0 at the walls, so that
b = curl(A) satisfies the conducting conditions and the same decay equation; 'b_free'
drops the constraint and closes the vector Laplacian with the conducting wall alone.

Demonstrates.  How SPDEE classifies the constraints of a system before any core is
chosen (spdee.ir.constraints.classify on the built system, printed for a mimetic and
a non-mimetic core): the written div(b) = 0 of 'b' is an involution, preserved by the
curl evolution, which the staggered C-grid keeps exactly by constrained transport
(the row is dropped, b lives on the face fluxes) and which the three non-mimetic cores
enforce by divergence cleaning (an auto-introduced multiplier __div_b, a saddle point
with the integral gauge at m = 0); the Coulomb gauge of 'A' is a saddle point on every
core; 'b_free' carries no constraint and the C-grid refuses it (a tangential electric
field on a non-flux unknown) while the other cores build it and carry, next to the
physical modes, extra modes with div(b) != 0.  All spellings converge to the same
rates; div(b) of the eigenvectors is reported per core (the C-grid's own discrete
divergence, second-order finite differences of the grid data elsewhere, Dedalus's own
div operator on its core).

Run from the repository root: PYTHONPATH=. python examples/05_magnetic_decay.py
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
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.sh_fd import ShFdGrid
from spdee.ir.constraints import classify
from spdee.ir.walker import walk_equation_system
from spdee.solvers import dense

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logging.getLogger("subsystems").setLevel(logging.WARNING)

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CORES = ("fd2", "staggered", "sh_fd", "dedalus")
FORMULATIONS = ("b", "A", "b_free")
R_INNER = 0.5
M = 0
RESOLUTION = {"fd2": (18, 27), "staggered": (20, 32), "sh_fd": (48, 6), "dedalus": (20, 16)}


# ---------------------------------------------------------------------------
# The three spellings
# ---------------------------------------------------------------------------

def decay_problem(formulation: str, eta: float = 1.0) -> Problem:
    pb = Problem()
    pb.add_parameter("eta", eta, symbolic=False)
    if formulation == "A":
        pb.add_unknown("A", "vector")
        pb.add_unknown("psi", "scalar")
        pb.add_equation("dt(A) - eta*lap(A) - grad(psi) = 0", name="induction")
        pb.add_equation("div(A) = 0", name="coulomb")
        for wall in ("inner", "outer"):
            pb.add_bc("dot(theta_hat, A) = 0", at=wall)
            pb.add_bc("dot(phi_hat, A) = 0", at=wall)
            pb.add_bc("psi = 0", at=wall)
        return pb
    pb.add_unknown("b", "vector")
    pb.add_equation("dt(b) - eta*lap(b) = 0", name="induction")
    if formulation == "b":
        pb.add_equation("div(b) = 0", name="solenoidal")
    for wall in ("inner", "outer"):
        pb.add_bc("dot(r_hat, b) = 0", at=wall)
        pb.add_bc("dot(theta_hat, curl(b)) = 0", at=wall)
        pb.add_bc("dot(phi_hat, curl(b)) = 0", at=wall)
    return pb


def classification(pb: Problem, m: int = M):
    """The classifier's record for a mimetic and a non-mimetic core."""
    system = pb.build()
    tasks = walk_equation_system(system, m)
    return {mimetic: classify(system, tasks, geometry="shell", mimetic=mimetic)
            for mimetic in (True, False)}


def print_classification(formulation, records):
    print(f"\n[{formulation}] constraint classification at m = {M}")
    for mimetic, rec in records.items():
        core = "mimetic (C-grid)" if mimetic else "non-mimetic (fd2, sh_fd, dedalus)"
        cls = {name: info["class"] for name, info in rec.equations.items()}
        print(f"  {core:36s} equations {cls}")
        print(f"  {'':36s} constraints {rec.constraints}, involutions {rec.involutions}, "
              f"multipliers {rec.multipliers}, gauges {rec.gauges}")
        if rec.involutions:
            print(f"  {'':36s} involution treatment: {rec.involution_treatment}"
                  + "".join(f"; {dc.vector}: multiplier {dc.multiplier} in {dc.evolution}, "
                            f"gauge {dc.gauge}, walls {dc.walls}"
                            for dc in rec.divergence_cleaning.values()))


# ---------------------------------------------------------------------------
# The closed form
# ---------------------------------------------------------------------------

def _wall(f, l, x, kind):
    if kind == "S":
        return f(l, x)
    if kind == "N":
        return f(l, x, derivative=True)
    return f(l, x) + x * f(l, x, derivative=True)


def _det(k, l, kind):
    return (_wall(spherical_jn, l, k * R_INNER, kind) * _wall(spherical_yn, l, k, kind)
            - _wall(spherical_jn, l, k, kind) * _wall(spherical_yn, l, k * R_INNER, kind))


def bessel_roots(kind, l, n):
    ks = np.linspace(0.5, 30.0, 3000)
    v = np.array([_det(k, l, kind) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        if v[i] * v[i + 1] < 0:
            out.append(brentq(lambda x: _det(x, l, kind), ks[i], ks[i + 1], xtol=1e-14))
            if len(out) == n:
                break
    return out


def decay_targets():
    """[(label, k^2)] of the four gravest rates."""
    tor = bessel_roots("rT", 1, 2)
    return [("toroidal l=1 n=1", tor[0] ** 2), ("poloidal l=1", bessel_roots("S", 1, 1)[0] ** 2),
            ("poloidal l=2", bessel_roots("S", 2, 1)[0] ** 2), ("toroidal l=1 n=2", tor[1] ** 2)]


def gradient_rates(l_max=6, n=2, below=60.0):
    """k^2 of the gradient modes b = grad(phi) that 'b_free' admits: phi obeys the
    heat equation with Neumann walls (b_r = d phi/dr = 0), so R' = 0 at both radii.
    The l = 0 overtone k = 6.572013 coincides with the poloidal l = 1 rate because
    j_1 = -j_0'; that pair is degenerate in the 'b_free' pencil."""
    rates = sorted(k ** 2 for l in range(l_max + 1) for k in bessel_roots("N", l, n))
    return [x for x in rates if x < below]


# ---------------------------------------------------------------------------
# Cores, censuses and the divergence of the eigenvectors
# ---------------------------------------------------------------------------

def compile_on(pb, core, m=M):
    a, b = RESOLUTION[core]
    if core == "sh_fd":
        return pb.compile(ShFdGrid(N_r=a, ell_max=m + b, r_inner=R_INNER, r_outer=1.0),
                          m=m, backend="sh_fd")
    if core == "dedalus":
        from spdee.backends.dedalus import DedalusGrid
        return pb.compile(DedalusGrid(N_r=a, N_theta=b, r_inner=R_INNER, r_outer=1.0),
                          m=m, backend="dedalus")
    n_r = a + 1 if core == "staggered" else a
    grid = make_shell_grid(N_r=n_r, N_theta=b, r_inner=R_INNER, r_outer=1.0)
    if core == "staggered":
        return pb.compile(grid, m=m, backend="fd2", discretization="staggered",
                          order_r=2, order_theta=2)
    return pb.compile(grid, m=m, backend="fd2", order_r=4, order_theta=4)


def census(cp, sigma):
    """Finite eigenpairs whose relative residual is below 100 times the
    pencil's median (and below 1e-8 at least): the Dedalus 'b' pencil is
    ill-conditioned, with a median near 1e-9, and the device transform puts
    its slowest decay mode at 1.6e-8, so a fixed 1e-8 dropped it on a GPU."""
    K, M_ = cp.assemble_evp_matrices()
    out = dense(K, M_, sigma=sigma, backend="auto", vectors=True, keep_vectors=True)
    fin = np.isfinite(out.lam)
    cut = max(1e-8, 100.0 * float(np.median(out.resid[fin])))
    keep = fin & (out.resid < cut)
    return out.lam[keep], out.V[:, keep]


def _fd_divergence(r, th, b_r, b_t, b_p, m):
    """Second-order divergence of physical components on an (r, theta) grid."""
    R, T = np.meshgrid(r, th, indexing="ij")
    d_r = np.gradient(R ** 2 * b_r, r, axis=0) / R ** 2
    d_t = np.gradient(np.sin(T) * b_t, th, axis=1) / (R * np.sin(T))
    return d_r + d_t + 1j * m * b_p / (R * np.sin(T))


def relative_divergence(cp, core, x, lam, m=M) -> float:
    """||div b|| / (k ||b||) with k = sqrt(lambda), the scale of curl b."""
    k = np.sqrt(abs(lam))
    if core == "staggered":
        d = cp.divergence(x)
        b = np.concatenate([x[f.slice] for f in cp.layout.fields if f.name.startswith("b_")])
        return float(np.linalg.norm(d) / (k * np.linalg.norm(b)))
    if core == "dedalus":
        cp.set_state(x)
        div = cp.d3.div(cp.fields["b"]).evaluate()
        div.change_scales(1)
        b = cp.fields["b"]
        b.change_scales(1)
        return float(np.linalg.norm(div["g"]) / (k * np.linalg.norm(b["g"])))
    if core == "fd2":
        v = cp.matrices.layout.unpack(x, role="primary")["b"]
        r, th = cp.matrices.grid.r, cp.matrices.grid.theta
        b_r, b_t, b_p = v[0], (v[1] + v[-1]) / np.sqrt(2), (v[1] - v[-1]) / (1j * np.sqrt(2))
    else:
        th = np.linspace(0.01, np.pi - 0.01, 96)
        c = cp.unpack(x)["b"]
        v_m, v_p = cp.to_grid(c[0], th, spin=-1).T, cp.to_grid(c[1], th, spin=+1).T
        r = cp.grid.r
        b_r = cp.to_grid(c[2], th, spin=0).T
        b_t, b_p = (v_p + v_m) / np.sqrt(2), (v_p - v_m) / (1j * np.sqrt(2))
    div = _fd_divergence(r, th, b_r, b_t, b_p, m)
    mag = np.sqrt(abs(b_r) ** 2 + abs(b_t) ** 2 + abs(b_p) ** 2)
    return float(np.linalg.norm(div[1:-1, 1:-1]) / (k * np.linalg.norm(mag[1:-1, 1:-1])))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run(formulations=FORMULATIONS, cores=CORES, verbose=True):
    targets = decay_targets()
    sigma = float(np.mean([t for _, t in targets]))
    results = {}
    for formulation in formulations:
        pb = decay_problem(formulation)
        if verbose:
            print_classification(formulation, classification(pb))
            print("  core     modes   " + "  ".join(f"{lab:>18s}" for lab, _ in targets)
                  + "   real rates in [3, 60]  time")
            print("  analytic         " + "  ".join(f"{t:18.6f}" for _, t in targets))
        for core in cores:
            t0 = time.time()
            try:
                cp = compile_on(pb, core)
                lam, V = census(cp, sigma)
            except (NotImplementedError, ValueError) as exc:
                results[(formulation, core)] = None
                if verbose:
                    print(f"  {core:9s} refused: {str(exc).splitlines()[0][:90]}")
                continue
            got, divs = [], []
            for _, t in targets:
                j = int(np.argmin(np.abs(lam - t)))
                got.append(lam[j].real)
                if formulation != "A":
                    divs.append(relative_divergence(cp, core, V[:, j], lam[j]))
            real = lam[(np.abs(lam.imag) < 1e-6) & (lam.real > 3) & (lam.real < 60)].real
            results[(formulation, core)] = (cp, lam, V, got, divs, real)
            if verbose:
                cells = "  ".join(f"{g:12.6f} {abs(g - t) / t:5.0e}"
                                  for g, (_, t) in zip(got, targets))
                print(f"  {core:9s} {lam.size:5d} {cells}   {real.size:3d}"
                      f"                    {time.time() - t0:4.1f} s")
                if divs:
                    print(f"  {'':9s} div b / (k |b|) of those eigenvectors: "
                          + "  ".join(f"{d:.1e}" for d in divs))
        if verbose and formulation == "b_free":
            extra = np.array(sorted(set(np.round(results[("b_free", "sh_fd")][5], 2))
                                    - set(np.round(results[("b", "sh_fd")][5], 2))))
            print("  extra real rates of b_free on sh_fd (not in the 'b' census): "
                  + " ".join(f"{x:.2f}" for x in extra))
            print("  gradient modes b = grad(phi), Neumann rates for l <= 6 below 60:     "
                  + " ".join(f"{x:.2f}" for x in gradient_rates()))
            print("  (the l = 0 overtone at 43.19 is degenerate with the poloidal l = 1 rate,"
                  " so that census eigenvector is a mixture: div b of order one)")
    return targets, results


def figure(targets, results, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for k, (_, t) in enumerate(targets):
        ax.axvline(t, color="k", lw=0.8, alpha=0.5, label="Bessel targets" if k == 0 else None)
    for k, t in enumerate(gradient_rates()):
        ax.axvline(t, color="C3", lw=0.8, ls="--", alpha=0.6,
                   label="gradient modes" if k == 0 else None)
    for i, formulation in enumerate(FORMULATIONS):
        real = results[(formulation, "sh_fd")][5]
        ax.plot(real, np.full(real.size, i), "|", ms=22, mew=2.5, label=formulation, zorder=3)
    ax.set_yticks(range(3))
    ax.set_yticklabels(FORMULATIONS)
    ax.set_xlim(0, 60)
    ax.set_xlabel("decay rate lambda = k^2 (sh_fd census)")
    ax.set_title("b_free carries the non-solenoidal gradient modes")
    ax.legend(fontsize=7, loc="center")
    ax = axes[1]
    width = 0.35
    for i, formulation in enumerate(("b", "b_free")):
        vals = [results[(formulation, c)][4][0] if results[(formulation, c)] else np.nan
                for c in CORES]
        ax.bar(np.arange(len(CORES)) + (i - 0.5) * width, np.maximum(vals, 1e-17), width,
               label=formulation)
    ax.set_yscale("log")
    ax.set_xticks(range(len(CORES)))
    ax.set_xticklabels(CORES)
    ax.set_ylabel(f"div b / (k |b|), {targets[0][0]} mode")
    ax.set_title("own operator (staggered, dedalus); grid estimate (fd2, sh_fd)")
    ax.legend()
    ax = axes[2]
    cp, lam, V, got, _, _ = results[("b", "fd2")]
    j = int(np.argmin(np.abs(lam - targets[0][1])))
    v = cp.matrices.layout.unpack(V[:, j], role="primary")["b"]
    b_phi = (v[1] - v[-1]) / (1j * np.sqrt(2))
    b_phi = np.real(b_phi / b_phi.ravel()[np.argmax(np.abs(b_phi))])
    R, T = cp.matrices.grid.mesh()
    ax.pcolormesh(R * np.sin(T), R * np.cos(T), b_phi, cmap="RdBu_r", vmin=-1, vmax=1,
                  shading="gouraud")
    ax.set_aspect("equal")
    ax.set_title(f"b_phi, {targets[0][0]} (fd2), lambda = {got[0]:.4f}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    targets, results = run()
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "05_magnetic_decay.png")
    figure(targets, results, path)
    print(f"\nfigure: {path}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
