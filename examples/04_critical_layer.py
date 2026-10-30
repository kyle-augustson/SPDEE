"""A Rossby mode with a critical layer in a differentially rotating shell, and
the workflow that polishes one eigenpair: locate, refine, polish, census, converge.

Physics.  Incompressible flow in the shell 0.7 <= r <= 1 seen from a frame rotating
at the equatorial rate Omega0 = 1, with the latitudinal differential rotation
Omega(theta) = Omega0 (1 - eps cos^2 theta) entered as the background flow
U0 = r sin(theta) (Omega(theta) - Omega0) phi_hat = -eps r sin(theta) cos^2(theta) phi_hat.
Linearising the momentum equation about U0 in the rotating frame gives

    dt(v) + U0.grad(v) + v.grad(U0) + 2 Omega0 ez x v + grad(p) - Ek lap(v) = 0,  div(v) = 0,

with Ek = nu / (Omega0 r_o^2), stress-free impenetrable walls, and the gauge on p.
U0.grad(v) is the advection operator dot(U0, grad(v)); v.grad(U0) is the contraction
of v with the gradient tensor of the background, which is known and is entered as
the background tensor gradU0 with (gradU0)_ij = d_i U0_j in the orthonormal basis:
for U0 = U phi_hat the nonzero entries are (r, phi) = dU/dr, (theta, phi) =
(1/r) dU/dtheta, (phi, r) = -U/r and (phi, theta) = -U cot(theta)/r; the string
dot(v, gradU0) contracts on the first index.  For rigid rotation U = eps r sin(theta)
this tensor gives exactly eps ez x v, which is the check of
tests/test_analytic_evp.py's Doppler benchmark; the same identity is asserted here on
the assembled pencil before the differential rotation is switched on.

A mode of frequency omega (e^{i m phi - i omega t}, so omega = -i lambda) has a
critical latitude where its phase speed matches the local flow,
omega = m (Omega(theta_c) - Omega0), i.e. cos^2(theta_c) = -omega / (m eps): the
inviscid operator is singular there and at finite Ek the eigenfunction develops a
critical layer of width scaling as Ek^(1/3).  The mode followed here is the sectoral
Rossby mode (omega = -2/(m+1) at eps = 0, m = 4): as eps grows it is Doppler shifted
retrograde, its critical latitude enters the shell once |omega| < m eps (near
eps = 0.12) and its damping rises by an order of magnitude; at eps = 0.2 it sits at
omega = -0.5274 - 0.0192 i with the layer at colatitude 35.7 degrees.  The profile is
the equator-fast (solar-like) one because a retrograde mode can only match a flow that
is retrograde in its frame: with the pole-fast profile Omega0 (1 - eps sin^2 theta)
in the polar frame the sectoral mode is more retrograde than every point of the flow
and has no critical latitude.

Demonstrates.  The polishing workflow of spdee.solvers on one collocated pencil:
(a) a coarse ARPACK shift-invert to locate the mode; (b) the same on a finer grid with
the coarse eigenvalue as the shift, on a GPU factorization when one passes the
residual gate, else SuperLU; (c) polishing the eigenpair with method="bbjd" started
from the ARPACK vector (v0=), the pencil residual before and after, and the same from
the coarse eigenvector interpolated onto the fine grid, where bbjd does the whole
refinement without a fine ARPACK solve; (d) a dense census (backend="auto", the GPU
when present) at the coarse resolution as the ground truth of the whole window,
showing what a k-nearest shift-invert leaves out; (e) convergence of the eigenvalue
and of the layer width with resolution and with Ek.  bbjd converges on this pencil
(m = 4, no integral gauge row); it is known to stall on pencils that carry the dense
integral gauge row of m = 0.  FPLUS is built for the spectral cores' pencils and is far
slower than SuperLU on this 2-D finite-difference pencil (minutes against a second at
N = 6144), so it is not used.  ARPACK's own shift-invert already reaches rounding on
this pencil, so bbjd from its vector confirms rather than improves; from the
interpolated coarse vector it does the whole refinement.

Run from the repository root: PYTHONPATH=. python examples/04_critical_layer.py
"""

from __future__ import annotations

import os
import sys
import time

import matplotlib
import numpy as np
import sympy as sp
from scipy.interpolate import RegularGridInterpolator

from spdee import Problem
from spdee.backends.fd2.grid import make_shell_grid
from spdee.io import pencil_residuals
from spdee.ir import r, theta
from spdee.solvers import available, dense, shift_invert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

M = 4
EPS = 0.2
EK = 1e-4
R_INNER = 0.7
COARSE = (16, 96)
FINE = (24, 192)
FINER = (32, 256)
# (Ek, N_r, N_theta): the layer narrows as Ek^(1/3), the grid with it
EK_LADDER = ((1e-4, 24, 192), (3e-5, 24, 288), (1e-5, 32, 384))
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ---------------------------------------------------------------------------
# The linearised operator
# ---------------------------------------------------------------------------

def gradient_of_azimuthal_flow(U):
    """(grad U0)_ij = d_i U0_j for U0 = U(r, theta) phi_hat, physical components."""
    return {("r", "phi"): sp.diff(U, r),
            ("theta", "phi"): sp.diff(U, theta) / r,
            ("phi", "r"): -U / r,
            ("phi", "theta"): -U * sp.cos(theta) / (r * sp.sin(theta))}


def rossby_problem(U, Ek: float = EK) -> Problem:
    """Viscous rotating flow about the azimuthal background U(r, theta) phi_hat."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("U0", v_r=0, v_theta=0, v_phi=U)
    pb.add_background("gradU0", components=gradient_of_azimuthal_flow(U))
    pb.add_parameter("Ek", Ek)
    pb.add_equation("dt(v) + dot(U0, grad(v)) + dot(v, gradU0) + 2*cross(ez, v)"
                    " + grad(p) - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    return pb


def differential_rotation(eps: float = EPS):
    """U0 = r sin(theta) (Omega(theta) - Omega0) for Omega = 1 - eps cos^2(theta)."""
    return -eps * r * sp.sin(theta) * sp.cos(theta) ** 2


def pencil(n_r: int, n_theta: int, eps: float = EPS, Ek: float = EK, m: int = M):
    grid = make_shell_grid(N_r=n_r, N_theta=n_theta, r_inner=R_INNER, r_outer=1.0)
    cp = rossby_problem(differential_rotation(eps), Ek).compile(
        grid, m=m, backend="fd2", gauge_unknowns=["p"], order_r=4, order_theta=4)
    K, Mm = cp.assemble_evp_matrices()
    return cp, K, Mm


def check_rigid_rotation_identity(rate: float = 0.3, m: int = M) -> float:
    """|K(dot(v, gradU0)) - K(rate * cross(ez, v))| for U0 = rate r sin(theta) phi_hat:
    the gradient tensor of rigid rotation is the Coriolis matrix."""
    grid = make_shell_grid(N_r=8, N_theta=12, r_inner=R_INNER, r_outer=1.0)
    U = rate * r * sp.sin(theta)
    K1, _ = rossby_problem(U).compile(grid, m=m, gauge_unknowns=["p"]).assemble_evp_matrices()
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("U0", v_r=0, v_theta=0, v_phi=U)
    pb.add_parameter("Ek", EK)
    pb.add_parameter("rate", rate)
    pb.add_equation("dt(v) + dot(U0, grad(v)) + rate*cross(ez, v) + 2*cross(ez, v)"
                    " + grad(p) - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    K2, _ = pb.compile(grid, m=m, gauge_unknowns=["p"]).assemble_evp_matrices()
    return float(abs(K1 - K2).max() / abs(K1).max())


# ---------------------------------------------------------------------------
# Solver helpers
# ---------------------------------------------------------------------------

def nearest(lam, V, target):
    """The finite eigenpair nearest ``target``."""
    lam = np.where(np.isfinite(lam), lam, np.inf)
    j = int(np.argmin(np.abs(lam - target)))
    return lam[j], V[:, j]


def residual(K, Mm, lam, x) -> float:
    """||K x - lam M x|| / (||K x|| + |lam| ||M x||)."""
    return float(pencil_residuals(K, Mm, np.array([lam]), x[:, None])[0])


def solve_near(K, Mm, sigma, k, factor="scipy", **kw):
    """ARPACK shift-invert at sigma; a GPU factor that fails its residual gate
    (cuDSS on a nearly singular K - sigma M) falls back to SuperLU."""
    try:
        return shift_invert(K, Mm, sigma, k, return_vectors=True, factor=factor, **kw), factor
    except RuntimeError as exc:
        if factor == "scipy":
            raise
        print(f"      {factor} declined ({str(exc).splitlines()[-1][:60]}...); using scipy")
        return shift_invert(K, Mm, sigma, k, return_vectors=True, factor="scipy", **kw), "scipy"


def physical_components(cp, x):
    """(v_r, v_theta, v_phi, p) of a raw eigenvector as (N_r, N_theta) arrays."""
    f = cp.matrices.layout.unpack(x, role="primary")
    v = f["v"]
    v_theta = (v[+1] + v[-1]) / np.sqrt(2)
    v_phi = (v[+1] - v[-1]) / (1j * np.sqrt(2))
    return v[0], v_theta, v_phi, f["p"]


def layer_width(cp, x, theta_c: float) -> float:
    """Full width at half maximum, in radians, of |v_r| at mid-depth around the
    critical colatitude theta_c: the measure of the layer thickness."""
    v_r, _, _, _ = physical_components(cp, x)
    th = cp.matrices.grid.theta
    prof = np.abs(v_r[len(cp.matrices.grid.r) // 2])
    window = np.abs(th - theta_c) < 0.35
    jpk = np.nonzero(window)[0][np.argmax(prof[window])]
    half = 0.5 * prof[jpk]
    lo = jpk
    while lo > 0 and prof[lo] > half:
        lo -= 1
    hi = jpk
    while hi < len(th) - 1 and prof[hi] > half:
        hi += 1
    return float(th[hi] - th[lo])


def critical_colatitude(omega: complex, eps: float = EPS, m: int = M) -> float:
    return float(np.arccos(np.sqrt(-omega.real / (m * eps))))


# ---------------------------------------------------------------------------
# The workflow
# ---------------------------------------------------------------------------

def locate(coarse=COARSE, verbose=True):
    """(a) coarse ARPACK shift-invert near the Doppler-shifted Rossby frequency."""
    cp, K, Mm = pencil(*coarse)
    sigma = 1j * (-0.52) + 0.02
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    t0 = time.time()
    lam, V = shift_invert(K, Mm, sigma, 8, return_vectors=True, v0=v0)
    lam_c, x_c = nearest(lam, V, 1j * (-0.53))
    om = -1j * lam[np.isfinite(lam)]
    if verbose:
        print(f"(a) coarse {coarse[0]}x{coarse[1]} N={K.shape[0]}: ARPACK k=8 at omega = "
              f"{-1j * sigma:.3f} in {time.time() - t0:.1f} s")
        print("    omega found: " + ", ".join(f"{w.real:+.4f}{w.imag:+.4f}i" for w in
                                              sorted(om, key=lambda w: -w.imag)))
        print(f"    Rossby mode: omega = {-1j * lam_c:.6f}, residual "
              f"{residual(K, Mm, lam_c, x_c):.1e}, critical colatitude "
              f"{np.degrees(critical_colatitude(-1j * lam_c)):.1f} deg")
    return cp, K, Mm, lam_c, x_c, om


def refine(lam_c, fine=FINE, verbose=True):
    """(b) ARPACK on a finer grid, the coarse eigenvalue as the shift."""
    cp, K, Mm = pencil(*fine)
    factor = "cudss" if available()["cudss"] else "scipy"
    t0 = time.time()
    # the shift sits a little off the coarse value so K - sigma M is not nearly singular
    (lam, V), used = solve_near(K, Mm, lam_c + 0.01, 4, factor=factor)
    lam_f, x_f = nearest(lam, V, lam_c)
    res = residual(K, Mm, lam_f, x_f)
    if verbose:
        print(f"(b) fine {fine[0]}x{fine[1]} N={K.shape[0]}: ARPACK k=4, factor {used}, "
              f"{time.time() - t0:.1f} s")
        print(f"    omega = {-1j * lam_f:.8f}, residual {res:.1e}")
    return cp, K, Mm, lam_f, x_f, res


def interpolate(cp_from, cp_to, x):
    """A collocated eigenvector carried onto another collocated grid by cubic
    interpolation of every component; the state is repacked by the target layout."""
    f = cp_from.matrices.layout.unpack(x, role="primary")
    g_from, g_to = cp_from.matrices.grid, cp_to.matrices.grid
    R, T = g_to.mesh()
    points = np.stack([R.ravel(), T.ravel()], axis=-1)

    def onto(a):
        it = RegularGridInterpolator((g_from.r, g_from.theta), a, method="cubic",
                                     bounds_error=False, fill_value=None)
        return it(points).reshape(R.shape)

    return cp_to.matrices.layout.pack({"v": {s: onto(f["v"][s]) for s in f["v"]},
                                       "p": onto(f["p"])})


def polish(K, Mm, sigma, x0, label, verbose=True):
    """(c) Jacobi-Davidson (bbjd) from the pair (sigma, x0) to machine precision;
    without a BBJD build (SPDEE_BBJD_PATH) the pair is returned unpolished."""
    before = residual(K, Mm, sigma, x0)
    if not available()["bbjd"]:
        if verbose:
            print(f"(c) bbjd polish from {label} skipped: no BBJD build "
                  f"(SPDEE_BBJD_PATH); residual stays {before:.1e}")
        return sigma, x0, before, before
    t0 = time.time()
    lam, V = shift_invert(K, Mm, sigma + 0.01, 1, method="bbjd", return_vectors=True,
                          v0=x0, tol=1e-13, maxiter=60)
    lam_b, x_b = lam[0], V[:, 0]
    after = residual(K, Mm, lam_b, x_b)
    if verbose:
        print(f"(c) bbjd polish from {label}, {time.time() - t0:.1f} s: "
              f"omega = {-1j * lam_b:.8f}, residual {before:.1e} -> {after:.1e}")
    return lam_b, x_b, before, after


def census_window(K, Mm, lam_c, found, coarse=COARSE, verbose=True):
    """(d) every eigenvalue of the coarse pencil in a window around the mode,
    against the k = 8 the locating ARPACK solve returned."""
    t0 = time.time()
    out = dense(K, Mm, sigma=1j * (-0.52) + 0.05, backend="auto", vectors=True)
    lam = out.lam[out.resid < 1e-8]
    om = -1j * lam
    window = (np.abs(om.real + 0.52) < 0.2) & (om.imag > -0.2)
    om_w = om[window][np.argsort(-om[window].imag)]
    hit = np.array([np.abs(found - w).min() < 1e-6 for w in om_w])
    if verbose:
        print(f"(d) dense census of the {coarse[0]}x{coarse[1]} pencil on {out.backend}: "
              f"{lam.size} finite eigenvalues, {out.n_infinite} infinite, "
              f"{time.time() - t0:.1f} s")
        print(f"    {om_w.size} modes with |Re omega + 0.52| < 0.2 and Im omega > -0.2, "
              f"{hit.sum()} of them among the k = 8 nearest the ARPACK shift (*):")
        print("    " + ", ".join(f"{w.real:+.4f}{w.imag:+.4f}i{'*' if h else ''}"
                                 for w, h in zip(om_w, hit)))
    return om_w, hit


def converge(lam_ref, verbose=True):
    """(e) the eigenvalue and layer width against resolution and Ek."""
    rows = []
    theta_c = critical_colatitude(-1j * lam_ref)
    for (n_r, n_t) in (COARSE, FINE, FINER):
        cp, K, Mm = pencil(n_r, n_t)
        (lam, V), _ = solve_near(K, Mm, lam_ref + 0.01, 2, tol=1e-10)
        lam_n, x = nearest(lam, V, lam_ref)
        rows.append(("res", EK, n_r, n_t, -1j * lam_n, layer_width(cp, x, theta_c), cp, x))
    target = lam_ref
    for Ek, n_r, n_t in EK_LADDER:
        cp, K, Mm = pencil(n_r, n_t, Ek=Ek)
        (lam, V), _ = solve_near(K, Mm, target + 0.01, 2, tol=1e-10)
        lam_n, x = nearest(lam, V, target)
        target = lam_n
        width = layer_width(cp, x, critical_colatitude(-1j * lam_n))
        rows.append(("Ek", Ek, n_r, n_t, -1j * lam_n, width, cp, x))
    if verbose:
        print("(e) convergence: resolution at Ek = 1e-4, then Ek at a grid following the layer")
        print("    scan   Ek      grid     omega                      width(rad) width/Ek^(1/3)")
        for kind, Ek, n_r, n_t, om, w, _, _ in rows:
            print(f"    {kind:4s} {Ek:8.1e} {n_r:3d}x{n_t:<4d} {om.real:+.7f}{om.imag:+.7f}i "
                  f"{w:8.4f}  {w / Ek ** (1 / 3):8.3f}")
        ek = [(row[1], row[5]) for row in rows if row[0] == "Ek"]
        slopes = [np.log(w2 / w1) / np.log(e2 / e1) for (e1, w1), (e2, w2) in zip(ek, ek[1:])]
        print("    local exponents d ln(width) / d ln(Ek): "
              + ", ".join(f"{s:.2f}" for s in slopes) + "  (asymptotic law 1/3)")
    return rows


def figure(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for kind, Ek, n_r, n_t, om, w, cp, x in rows:
        if kind != "Ek":
            continue
        _, _, v_phi, _ = physical_components(cp, x)
        th = cp.matrices.grid.theta
        prof = np.abs(v_phi[len(cp.matrices.grid.r) // 2])
        ax.plot(np.degrees(th), prof / prof.max(), label=f"Ek = {Ek:g}")
    theta_c = np.degrees(critical_colatitude(rows[-1][4]))
    for tc in (theta_c, 180 - theta_c):
        ax.axvline(tc, color="k", ls="--", lw=0.8)
    ax.set_xlabel("colatitude (deg)")
    ax.set_ylabel("|v_phi| at mid-depth")
    ax.set_title("critical layer at cos^2(theta_c) = -omega/(m eps)")
    ax.legend()
    ax = axes[1]
    eks = [row[1] for row in rows if row[0] == "Ek"]
    ws = [row[5] for row in rows if row[0] == "Ek"]
    ax.loglog(eks, ws, "o-", label="FWHM of |v_r| at the layer")
    ax.loglog(eks, ws[-1] * (np.array(eks) / eks[-1]) ** (1 / 3), "k--", label="Ek^(1/3)")
    ax.set_xlabel("Ek")
    ax.set_ylabel("layer width (rad)")
    ax.legend()
    ax = axes[2]
    kind, Ek, n_r, n_t, om, w, cp, x = rows[-1]
    _, _, v_phi, _ = physical_components(cp, x)
    R, T = cp.matrices.grid.mesh()
    ax.pcolormesh(R * np.sin(T), R * np.cos(T), np.abs(v_phi), shading="gouraud")
    ax.set_aspect("equal")
    ax.set_title(f"|v_phi|, Ek = {Ek:g}, omega = {om.real:.4f}{om.imag:+.4f}i")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    t0 = time.time()
    print(f"solver components: {available()}")
    print(f"rigid-rotation identity dot(v, gradU0) = rate ez x v on the pencil: "
          f"{check_rigid_rotation_identity():.1e}")
    print(f"m = {M}, eps = {EPS}, Ek = {EK}, shell {R_INNER} <= r <= 1, stress-free walls")
    cp_c, K_c, M_c, lam_c, x_c, found = locate()
    cp_f, K_f, M_f, lam_f, x_f, _ = refine(lam_c)
    lam_b, _, _, _ = polish(K_f, M_f, lam_f, x_f, "the fine ARPACK pair")
    print("    (ARPACK shift-invert already sits at rounding on this pencil; bbjd confirms it)")
    polish(K_f, M_f, lam_c, interpolate(cp_c, cp_f, x_c),
           "the coarse pair interpolated onto the fine grid, no fine ARPACK")
    census_window(K_c, M_c, lam_c, found)
    rows = converge(lam_b)
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, "04_critical_layer.png")
    figure(rows, path)
    print(f"\nfigure: {path}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
