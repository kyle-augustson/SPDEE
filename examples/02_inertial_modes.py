"""Inviscid inertial modes of a rotating full sphere: the Zhang et al. (2001)
frequencies, the exact Doppler shift under a change of rotating frame, and how a
mode is told apart from its neighbours in a dense band.

Physics.  Poincare's problem in the unit ball rotating at Omega0 = 1,

    dt(v) + 2 ez x v + grad(p) = 0,   div(v) = 0,   v_r = 0 at r = 1,

with v ~ e^{i m phi - i omega t} (omega = -i lambda).  The spectrum is neutral and
fills |omega| < 2 densely; in a full sphere the eigenfunctions are polynomials in the
Cartesian coordinates and Zhang, Earnshaw, Liao & Busse (J. Fluid Mech. 437, 2001)
give closed polynomial eigenvalue equations (their 2.15 and 2.16) whose roots are
tabulated below for the N = 1, 2 classes at m = 1 and 2.  Zhang et al. write the modes
as e^{i (m phi + 2 sigma t)} and tabulate omega_Z = 2 sigma; in the convention here
omega = -omega_Z.  The pressure of a mode of class (N, m, parity) is
s^m e^{i m phi} times a polynomial in (s^2, z) of degree 2N (equatorially symmetric,
even in z) or 2N + 1 (antisymmetric, odd in z), s = r sin(theta), z = r cos(theta).

The Doppler check.  Seen from a frame rotating at Omega0 - rate the same fluid carries
the rigid mean flow U0 = rate * r sin(theta) phi_hat, and the linearised operator is
dt(v) + U0.grad(v) + v.grad(U0) + 2 (Omega0 - rate) ez x v + grad(p).  For rigid
rotation U0.grad(v) = rate (d/dphi v + ez x v) and v.grad(U0) = rate ez x v, so the
frame terms collapse to i m rate v + 2 Omega0 ez x v: every eigenvalue shifts by
exactly m * rate and every eigenvector is unchanged.  Comparing two nonzero rates on
the same grid cancels the discretisation error entirely, so the shift
m (rate2 - rate1) is a rounding-level check of the advection operator against the
Coriolis operator (docs/analytic_benchmarks.md, "Doppler shift: the exact statement").

Demonstrates.  A saddle-point constraint with the pressure declared a gauge unknown
(gauge_unknowns=["p"], the integral gauge at m = 0, nothing at m != 0); a dense
census read through the pencil residual and a neutrality gate; why a dense band needs
eigenvector-based identification (the nearest eigenvalue to a target is not evidence
when the mean spacing is comparable to the match, so the candidates within a window
are ranked by the fit of their pressure to the Zhang polynomial space); shift-invert
near a target with v0 seeding (the eigenvector of one frame seeds the solve in the
other, where it is exact); and the meridional readback of an sh_fd eigenvector
(unpack + to_grid per spin component).  The core is sh_fd (spin-weighted harmonics
times radial FD), whose inertial frequencies are accurate to 1e-5 at this size.

Run from the repository root: PYTHONPATH=. python examples/02_inertial_modes.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
import numpy as np

from spdee import Problem
from spdee.backends.sh_fd import ShFdGrid
from spdee.solvers import dense, shift_invert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
N_R = 48
ELL_EXTRA = 8
NEUTRAL = 1e-6
RESID = 1e-8
WINDOW = 0.05
RATES = (0.2, 0.5)

# (label, omega_Z) from Zhang et al. (2001), equations 2.15 (SYM) and 2.16 (ANTI)
ZHANG = {
    1: [("N=1 SYM", +1.50994), ("N=2 SYM", +1.80600),
        ("N=1 ANTI", -0.82001), ("N=1 ANTI", +0.61198), ("N=1 ANTI", +1.70802),
        ("N=2 ANTI", -1.40422), ("N=2 ANTI", -0.53733), ("N=2 ANTI", +0.44045),
        ("N=2 ANTI", +1.30608), ("N=2 ANTI", +1.86168)],
    2: [("N=1 SYM", +1.23193), ("N=2 SYM", +1.64344),
        ("N=1 ANTI", -0.76334), ("N=1 ANTI", +0.46690), ("N=1 ANTI", +1.49644),
        ("N=2 ANTI", -1.30800), ("N=2 ANTI", -0.50999), ("N=2 ANTI", +0.35954),
        ("N=2 ANTI", +1.15306), ("N=2 ANTI", +1.73397)],
}


# ---------------------------------------------------------------------------
# The problem in the co-rotating frame and in a frame rotating at Omega0 - rate
# ---------------------------------------------------------------------------

def inertial_problem(rate: float = 0.0, Omega0: float = 1.0) -> Problem:
    """Inviscid inertial modes; ``rate`` > 0 changes frame and adds the mean flow."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if rate:
        pb.add_background("U0", v_r=0, v_theta=0, v_phi=f"{rate!r}*r*sin(theta)")
        # the Coriolis term of the new frame, and v.grad(U0) = rate ez x v, an exact
        # identity for rigid rotation, spelled as the Cross it is
        pb.add_parameter("frame", 2.0 * (Omega0 - rate), symbolic=False)
        pb.add_parameter("rate", rate, symbolic=False)
        pb.add_equation("dt(v) + dot(U0, grad(v)) + rate*cross(ez, v)"
                        " + frame*cross(ez, v) + grad(p) = 0", name="momentum")
    else:
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


def compile_ball(pb: Problem, m: int, n_r: int = N_R, ell_extra: int = ELL_EXTRA):
    grid = ShFdGrid(N_r=n_r, ell_max=m + ell_extra, r_inner=0.0, r_outer=1.0, fd_order=4)
    return pb.compile(grid, m=m, backend="sh_fd", gauge_unknowns=["p"])


def census(cp, sigma=0.3j + 0.02):
    """Finite eigenvalues that solve the pencil (residual gate), with vectors."""
    K, M = cp.assemble_evp_matrices()
    out = dense(K, M, sigma=sigma, backend="auto", vectors=True, keep_vectors=True)
    keep = np.isfinite(out.lam) & (out.resid < RESID)
    return out.lam[keep], out.V[:, keep]


def neutral(lam):
    """omega of the neutral modes (|Im omega| < NEUTRAL) of a census."""
    om = -1j * lam
    return om.real[np.abs(om.imag) < NEUTRAL]


# ---------------------------------------------------------------------------
# Identification by the eigenvector
# ---------------------------------------------------------------------------

def zhang_pressure_basis(R, T, m, N, parity):
    """Monomials s^(m+2j) z^l of the Zhang (N, m, parity) pressure space."""
    s, z = R * np.sin(T), R * np.cos(T)
    degree = 2 * N + (1 if parity == "ANTI" else 0)
    cols = [(s ** (m + 2 * j) * z ** ell).ravel()
            for j in range(degree // 2 + 1)
            for ell in range(degree - 2 * j + 1)
            if (ell % 2 == 0) == (parity == "SYM")]
    return np.array(cols).T


def pressure_on_grid(cp, x, theta):
    """Pressure of an sh_fd eigenvector on (theta, r), synthesized from its
    spin-0 coefficients."""
    return cp.to_grid(cp.unpack(x)["p"][0], theta, spin=0)


def polynomial_fit(p, basis, weight):
    """Relative residual of the weighted least-squares fit of p on the basis."""
    sw = np.sqrt(weight)
    A, b = basis * sw[:, None], p.ravel() * sw
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.linalg.norm(A @ c - b) / np.linalg.norm(b))


def identify(cp, lam, V, m, target, label, theta=None):
    """Among the census modes within WINDOW of ``target``, the one whose pressure
    lies closest to the Zhang polynomial space of ``label``; returns
    (omega, fit, [(omega, fit) of every candidate])."""
    theta = np.linspace(0.02, np.pi - 0.02, 64) if theta is None else theta
    R, T = np.meshgrid(cp.grid.r, theta)
    weight = (R ** 2 * np.sin(T)).ravel() + 1e-3
    N, parity = int(label[2]), label.split()[1]
    basis = zhang_pressure_basis(R, T, m, N, parity)
    om = -1j * lam
    cand = np.nonzero(np.abs(om.real - target) < WINDOW)[0]
    fits = sorted((polynomial_fit(pressure_on_grid(cp, V[:, i], theta), basis, weight),
                   float(om[i].real)) for i in cand)
    return fits[0][1], fits[0][0], [(w, f) for f, w in fits]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def zhang_table(m, verbose=True):
    """The census at one m against the Zhang table: nearest neutral eigenvalue,
    the band density, and the identification by pressure polynomial."""
    t0 = time.time()
    cp = compile_ball(inertial_problem(), m)
    lam, V = census(cp)
    band = neutral(lam)
    inband = band[np.abs(band) < 2.0]
    spacing = 4.0 / inband.size
    controls = np.random.default_rng(3).uniform(-1.9, 1.9, 40)
    miss = np.median([np.abs(inband - c).min() for c in controls])
    rows = []
    for label, omega_z in ZHANG[m]:
        target = -omega_z
        nearest_ = inband[np.argmin(np.abs(inband - target))]
        om_id, fit, cands = identify(cp, lam, V, m, target, label)
        rows.append((label, target, nearest_, om_id, fit, cands))
    if verbose:
        print(f"\nm = {m}: sh_fd {N_R} x (ell_max = m + {ELL_EXTRA}), N = {cp.N}, "
              f"{lam.size} finite modes, {inband.size} neutral in |omega| < 2 "
              f"(mean spacing {spacing:.1e}), {time.time() - t0:.1f} s")
        print(f"  a random frequency in the band is matched to {miss:.1e} by some eigenvalue;"
              " the targets:")
        print("  class      omega_Zhang   nearest        err     identified     fit  "
              "candidates in the window (omega: polynomial fit)")
        for label, target, nearest_, om_id, fit, cands in rows:
            others = ", ".join(f"{w:+.4f}: {f:.2f}" for w, f in cands[1:4])
            print(f"  {label:9s} {target:+.5f}   {nearest_:+.6f} {abs(nearest_ - target):8.1e}"
                  f" {om_id:+.6f} {fit:8.1e}  {others}")
    return cp, lam, V, rows


def doppler_check(m=1, targets=("N=1 ANTI", -0.82001), verbose=True):
    """The N = 1 antisymmetric modes of m = 1 in two rotating frames: the shift
    m (rate2 - rate1) to rounding, with the eigenvector of the first frame
    seeding ARPACK in the second."""
    labels = [(lab, -wz) for lab, wz in ZHANG[m] if lab == "N=1 ANTI"]
    r1, r2 = RATES
    cp1 = compile_ball(inertial_problem(r1), m)
    cp2 = compile_ball(inertial_problem(r2), m)
    K1, M1 = cp1.assemble_evp_matrices()
    K2, M2 = cp2.assemble_evp_matrices()
    out = []
    for label, omega_i in labels:
        t0 = time.time()
        lam1, V1 = shift_invert(K1, M1, 1j * (omega_i + m * r1) + 0.01, 6, return_vectors=True)
        j1 = int(np.argmin(np.abs(-1j * lam1 - (omega_i + m * r1))))
        t1 = time.time()
        # the same eigenvector in the other frame: ARPACK starts on the answer
        lam2, V2 = shift_invert(K2, M2, 1j * (omega_i + m * r2) + 0.01, 6, return_vectors=True,
                                v0=V1[:, j1])
        j2 = int(np.argmin(np.abs(-1j * lam2 - (omega_i + m * r2))))
        w1, w2 = -1j * lam1[j1], -1j * lam2[j2]
        overlap = abs(np.vdot(V1[:, j1], V2[:, j2])) / (np.linalg.norm(V1[:, j1])
                                                       * np.linalg.norm(V2[:, j2]))
        out.append((label, omega_i, w1, w2, overlap, t1 - t0, time.time() - t1))
    if verbose:
        print(f"\nDoppler shift at m = {m}: frames rotating at Omega0 - {r1} and Omega0 - {r2}, "
              f"expected shift m (rate2 - rate1) = {m * (r2 - r1):.6f}")
        print("  class      omega_i     omega(rate1) - m rate1   omega(rate2) - m rate2   "
              "shift error   |<v1|v2>|  t1     t2 (seeded)")
        for label, omega_i, w1, w2, overlap, t1, t2 in out:
            print(f"  {label:9s} {omega_i:+.5f}   {(w1 - m * r1).real:+.8f}{(w1).imag:+.1e}i"
                  f"   {(w2 - m * r2).real:+.8f}{(w2).imag:+.1e}i   "
                  f"{abs((w2 - w1) - m * (r2 - r1)):.1e}   {overlap:.6f}   "
                  f"{t1:4.2f}s  {t2:4.2f}s")
    return out


def velocity_on_grid(cp, x, theta):
    """(v_r, v_theta, v_phi) of an sh_fd eigenvector on (theta, r): the unpack
    gives the spin components (-, +, r), each synthesized with its own spin."""
    c = cp.unpack(x)["v"]
    v_minus = cp.to_grid(c[0], theta, spin=-1)
    v_plus = cp.to_grid(c[1], theta, spin=+1)
    v_r = cp.to_grid(c[2], theta, spin=0)
    return v_r, (v_plus + v_minus) / np.sqrt(2), (v_plus - v_minus) / (1j * np.sqrt(2))


def figure(cp, lam, V, rows, m, path):
    theta = np.linspace(0.01, np.pi - 0.01, 121)
    om = -1j * lam
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    band = neutral(lam)
    ax.hist(band, bins=np.arange(-2.2, 2.25, 0.05), color="0.7", label="neutral census modes")
    for k, (label, target, _, _, _, _) in enumerate(rows):
        ax.axvline(target, color="C3" if "ANTI" in label else "C0", lw=0.8,
                   label={2: "Zhang ANTI", 0: "Zhang SYM"}.get(k))
    ax.set_xlabel("Re omega")
    ax.set_ylabel("modes per 0.05")
    ax.set_title(f"m = {m}: {band.size} neutral modes in the band")
    ax.legend(fontsize=8)
    label, target, _, om_id, _, _ = rows[2]
    j = int(np.argmin(np.abs(om.real - om_id)))
    v_r, v_th, v_ph = velocity_on_grid(cp, V[:, j], theta)
    p = pressure_on_grid(cp, V[:, j], theta)
    phase = np.exp(-1j * np.angle(p.ravel()[np.argmax(np.abs(p))]))
    R, T = np.meshgrid(cp.grid.r, theta)
    for ax, field, name in ((axes[1], p, "p"), (axes[2], v_ph, "v_phi")):
        f = np.real(field * phase)
        f /= np.abs(f).max()
        ax.pcolormesh(R * np.sin(T), R * np.cos(T), f, cmap="RdBu_r", vmin=-1, vmax=1,
                      shading="gouraud")
        ax.set_aspect("equal")
        ax.set_title(f"{name}: {label} at m = {m}, omega = {om[j].real:+.5f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    tables = {m: zhang_table(m) for m in (1, 2)}
    doppler_check()
    cp, lam, V, rows = tables[1]
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "02_inertial_modes.png")
    figure(cp, lam, V, rows, 1, path)
    print(f"\nfigure: {path}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
