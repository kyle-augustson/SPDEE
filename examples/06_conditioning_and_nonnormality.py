"""Judging eigenpairs: condition numbers, deflation, pseudospectra and
non-normality with spdee.analysis.

Part (i), the collocated inertial pencil in a ball.  2 z_hat x v + grad(p) =
-dt(v), div(v) = 0, r_hat . v = 0 at r = 1, compiled on the collocated fd2
core at m = 0 with the textbook wall closure (impenetrability_host='momentum').
That closure leaves the pressure checkerboard: a p-only vector n with
K n = 0 and M n = 0, a 0 = lambda 0 direction of the pencil, so the pencil is
singular and a shift-invert solve returns eigenvalues that track the shift.
The dense two-sided census (spdee.analysis.two_sided_dense) gives every
finite eigenpair with its left vector; the condition numbers
kappa = ||x||_W ||y||_W / |y^H M x| in the volume norm W rank the modes, and
not one of them is neutral.  pressure_null_modes finds the checkerboard
(singular value 6e-17 of K on the massless columns), Deflation removes it
and regularizes the shifted matrix, and the deflated shift-invert solve
returns neutral inertial modes with kappa of order 10, independent of the
shift.  The m = 2 pencil has no checkerboard; there the sectoral r-mode
omega = -2/3 has kappa = 1.1 while the spurious non-neutral pressure modes of
the non-adjoint Grad/Div pair top the ranking.

Part (ii), a genuinely non-normal operator.  A Rossby mode in the shell
0.5 <= r <= 1 with the differential rotation Omega(theta) = Omega0 (1 -
eps sin^2 theta) at Ek = 1e-3.  In the frame rotating at Omega0 the mean
flow is U = r sin(theta) (Omega - Omega0) phi_hat and the linearised momentum
equation, with the identity U . grad v + v . grad U = i m (Omega - Omega0) v +
2 (Omega - Omega0) z_hat x v + sin(theta) Omega'(theta) v_theta phi_hat, is

    dt(v) + 2 Omega(theta) z_hat x v + i m (Omega - Omega0) v
          + sin(theta) Omega'(theta) v_theta phi_hat + grad(p) - Ek lap(v) = 0,

div(v) = 0, stress-free walls, on the sh_fd core (spin-weighted harmonics in
theta, second-order differences in r, whose diffusion operator is exactly
symmetric in the r^2 dr weight).  The shear term is a background tensor
S_{phi theta} = sin(theta) Omega'(theta) entered as dot(S, v).  The mode is
followed from eps = 0 to eps = 0.4 by continuation (largest W-overlap, the
previous eigenvector as start vector); around it the eps-pseudospectrum in
the energy norm (dense Schur route on the finite invariant subspace), the
numerical against the spectral abscissa, the numerical range, the resolvent
norm along the imaginary axis (the response to harmonic forcing) and the
Kreiss constant lower bound quantify the departure from normality, with the
eps = 0 pencil as the reference.  Finally bbjd polishes the mode with the
exact LU and with the two-grid preconditioner (ILU smoother on the fine grid,
coarse solve on half the radial nodes), iteration counts reported.

Figures: examples/figures/06_conditioning.png and 06_nonnormality.png.
Run from the repository root: PYTHONPATH=. python examples/06_conditioning_and_nonnormality.py
"""

from __future__ import annotations

import os
import time
import warnings

import matplotlib
import numpy as np

from spdee import Problem, analysis as an, solvers
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.sh_fd import ShFdGrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def inertial_ball(m, host, N_r=12, N_theta=18):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_ball_grid(N_r=N_r, N_theta=N_theta)
    return pb.compile(grid, m=m, order_r=4, order_theta=4, impenetrability_host=host,
                      gauge_unknowns=["p"] if m == 0 else ())


def part_i():
    print("=== (i) collocated inertial ball: kappa ranking, checkerboard, deflation ===")
    t0 = time.time()
    cp = inertial_ball(0, "momentum")
    K, M = cp.assemble_evp_matrices()
    W = an.natural_weight(cp, M)
    lam, X, Y = an.two_sided_dense(K, M, sigma=0.3 + 0.2j)
    rep = an.condition_report(K, M, lam, X, W=W, Y=Y, threshold=100.0)
    om = -1j * lam
    neutral = np.abs(om.imag) < 1e-6
    print(f"m = 0, momentum host, N = {K.shape[0]}: {len(lam)} finite eigenvalues, "
          f"{neutral.sum()} neutral; kappa median {np.median(rep.kappa):.1e}, "
          f"max {rep.kappa.max():.1e}; {len(rep.ill_conditioned)} modes above kappa = 100")
    s, null = an.pressure_null_modes(K, M, k=3)
    print(f"pressure null directions: relative singular values {s}")
    n = null[:, 0]
    p_cb = cp.matrices.layout.unpack(n)["p"]
    mean_part = abs(np.sum(p_cb)) / (np.linalg.norm(p_cb) * np.sqrt(p_cb.size))
    print(f"null direction: p-only, {mean_part:.2f} of it the constant the centred gradient "
          f"annihilates, the rest the sawtooth that cancels it on the gauge row")

    defl = an.Deflation(K, M, n)
    sigma = 1.2j + 0.01
    lam_d, V_d = solvers.shift_invert(K, M, sigma, 12, return_vectors=True, deflation=defl)
    ok = np.isfinite(lam_d)
    lam_d, V_d = lam_d[ok], V_d[:, ok]
    res_d = defl.residual(lam_d, V_d)
    rep_d = an.condition_report(K, M, lam_d, V_d, W=W, threshold=100.0)
    om_d = -1j * lam_d
    print(f"deflated solve at sigma = {sigma}: {np.sum(np.abs(om_d.imag) < 1e-8)} neutral of "
          f"{len(lam_d)}, deflated residuals <= {res_d.max():.1e}, kappa median "
          f"{np.median(rep_d.kappa):.1f}, max {rep_d.kappa.max():.1f}")
    lam_u, _ = solvers.shift_invert(K, M, sigma, 12)
    print(f"undeflated solve at the same shift: {np.sum(np.abs(lam_u.real) < 1e-8)} neutral of "
          f"{np.isfinite(lam_u).sum()} (the singular pencil tracks the shift)")

    cp2 = inertial_ball(2, "momentum")
    K2, M2 = cp2.assemble_evp_matrices()
    lam2, X2, Y2 = an.two_sided_dense(K2, M2, sigma=0.3 + 0.2j)
    rep2 = an.condition_report(K2, M2, lam2, X2, W=an.natural_weight(cp2, M2), Y=Y2)
    om2 = -1j * lam2
    j_r = int(np.argmin(np.abs(om2.real + 2 / 3)))
    worst = rep2.order[0]
    print(f"m = 2: r-mode omega {om2[j_r]:.5f} kappa {rep2.kappa[j_r]:.2f}; worst mode omega "
          f"{om2[worst]:.4f} kappa {rep2.kappa[worst]:.1e} (p energy fraction "
          f"{an.energy_fractions(an.to_grid_fields(cp2, X2[:, worst]))['p']['total']:.2f} against "
          f"{an.energy_fractions(an.to_grid_fields(cp2, X2[:, j_r]))['p']['total']:.2f} "
          f"for the r-mode); median kappa {np.median(rep2.kappa):.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    sc = ax.scatter(om2.real, rep2.kappa, c=np.log10(np.abs(om2.imag) + 1e-16), s=10,
                    cmap="plasma", vmin=-16, vmax=-1)
    ax.plot(om2[j_r].real, rep2.kappa[j_r], "o", ms=12, mfc="none", mec="k", label="r-mode")
    ax.set_yscale("log")
    ax.set_xlabel("Re omega")
    ax.set_ylabel("kappa(lambda)")
    ax.set_title("m = 2: condition numbers of the census")
    ax.legend(loc="upper left")
    fig.colorbar(sc, ax=ax, label="log10 |Im omega|")
    ax = axes[1]
    R, T = cp.matrices.grid.mesh()
    saw = p_cb.real - np.mean(p_cb.real)
    ax.pcolormesh(R * np.sin(T), R * np.cos(T), saw, cmap="RdBu", shading="auto",
                  vmin=-np.abs(saw).max(), vmax=np.abs(saw).max())
    ax.set_aspect("equal")
    ax.set_title("m = 0: null direction n (K n = M n = 0), mean removed")
    ax.set_xlabel("s")
    ax.set_ylabel("z")
    ax = axes[2]
    ax.scatter(om.real, rep.kappa, s=8, c=np.log10(np.abs(om.imag) + 1e-16), cmap="plasma",
               vmin=-16, vmax=-1, label="singular pencil, census (none neutral)")
    neut = np.abs(om_d.imag) < 1e-8
    ax.scatter(om_d.real[~neut], rep_d.kappa[~neut], s=40, marker="s", c="C3",
               label="deflated shift-invert, damped")
    ax.scatter(om_d.real[neut], rep_d.kappa[neut], s=60, marker="*", c="C2",
               label="deflated shift-invert, neutral")
    ax.set_yscale("log")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(5, 1e3)
    ax.set_xlabel("Re omega")
    ax.set_ylabel("kappa(lambda)")
    ax.set_title("m = 0: deflating the checkerboard")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIG, "06_conditioning.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure {path}  ({time.time() - t0:.1f} s)")


def shear_problem(m, eps, Ek):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_parameter("Ek", Ek)
    om = f"(1 - {eps}*sin(theta)**2)"
    pb.add_background("Omega", v_r=f"{om}*cos(theta)", v_theta=f"-{om}*sin(theta)", v_phi=0)
    pb.add_background("adv", expr=f"I*{m}*({om} - 1)")
    pb.add_background("shear",
                      components={("phi", "theta"): f"-2*{eps}*sin(theta)**2*cos(theta)"})
    pb.add_equation("dt(v) + 2*cross(Omega, v) + adv*v + dot(shear, v) + grad(p)"
                    " - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    return pb


def part_ii(m=2, Ek=1e-3, eps_max=0.4, N_r=24, ell=11):
    print("\n=== (ii) Rossby mode under differential rotation: non-normality ===")
    t0 = time.time()
    grid = ShFdGrid(N_r=N_r, ell_max=m + ell, r_inner=0.5, r_outer=1.0, fd_order=2)

    def build(eps):
        return shear_problem(m, eps, Ek).compile(grid, m=m, backend="sh_fd")
    eps_path = np.linspace(0.0, eps_max, 9)
    path = an.continuation(build, eps_path, -2j / (m + 1) + 0.01, k=8, keep_vectors=True)
    print("continuation of the sectoral Rossby mode in eps (omega = -i lambda):")
    for e, l_, o in zip(path.values, path.lam, path.overlap):
        print(f"   eps = {e:.3f}  omega = {-1j * l_:+.5f}  overlap {o:.4f}")
    results = {}
    for eps, lam_r in ((0.0, path.lam[0]), (eps_max, path.lam[-1])):
        cp = build(eps)
        K, M = cp.assemble_evp_matrices()
        W = an.natural_weight(cp, M)
        red = an.reduced_operator(K, M, sigma=lam_r + 0.3 - 0.1j, W=W)
        na = an.numerical_abscissa(red.A)
        h = an.henrici(red.A)
        fov = an.numerical_range(red.A, 72)
        results[eps] = (cp, K, M, W, red, na, h, fov)
        print(f"eps = {eps}: {red.k} finite modes; Henrici departure {h['relative']:.3f} of "
              f"||A||_F; spectral abscissa {na['spectral']:+.4f}, numerical abscissa "
              f"{na['numerical']:+.4f} (transient growth potential {na['excess']:.4f})")
    cp, K, M, W, red, na, h, fov = results[eps_max]
    lam_r = path.lam[-1]
    x_r = path.vectors[-1]
    feat = an.features(cp, lam_r, x_r, K=K, M=M)
    print("the followed mode: " + feat.summary())
    Y, _, _ = an.left_eigenvectors(K, M, [lam_r], x_r[:, None])
    kappa = an.condition_numbers(K, M, x_r[:, None], Y, W)[0]
    c_mode = an.mode_nonnormality(K, M, [lam_r], W=W, reduced=red)[0]
    print(f"kappa(lambda) = {kappa:.2f}, excess resolvent on a circle at a quarter gap "
          f"{c_mode:.2f}")

    re = lam_r.real + np.linspace(-0.25, 0.25, 25)
    im = lam_r.imag + np.linspace(-0.25, 0.25, 25)
    t1 = time.time()
    sr = an.pseudospectra.SchurResolvent.__new__(an.pseudospectra.SchurResolvent)
    sr.T, sr.sigma = red.T, red.sigma
    grid_norm = np.array([[sr.norm(complex(x, y)) for x in re] for y in im])
    ps = an.Pseudospectrum(re=re, im=im, resolvent=grid_norm, method="dense", lam=red.lam)
    kc = an.kreiss_constant(ps)
    print(f"pseudospectrum on {len(re)}x{len(im)} points in {time.time() - t1:.1f} s; "
          f"eps-pseudospectral abscissa at eps = 1e-2: {ps.pseudospectral_abscissa(1e-2):+.4f}; "
          f"Kreiss constant lower bound {kc:.3f}")
    w_line = np.linspace(lam_r.imag - 0.6, lam_r.imag + 0.6, 61)
    axis = np.array([sr.norm(complex(0.0, w)) for w in w_line])
    sr0 = an.pseudospectra.SchurResolvent.__new__(an.pseudospectra.SchurResolvent)
    sr0.T, sr0.sigma = results[0.0][4].T, results[0.0][4].sigma
    axis0 = np.array([sr0.norm(complex(0.0, w)) for w in w_line])
    peak = w_line[np.argmax(axis)]
    print(f"resolvent norm on the imaginary axis: max {axis.max():.1f} at omega = {peak:+.3f} "
          f"(eps = {eps_max}) against {axis0.max():.1f} (eps = 0); 1/dist bound at the peak "
          f"{1 / np.min(np.abs(red.lam - 1j * peak)):.1f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma = lam_r + 0.02 * (1 + abs(lam_r))
        coarse = shear_problem(m, eps_max, Ek).compile(
            ShFdGrid(N_r=N_r // 2, ell_max=m + ell, r_inner=0.5, r_outer=1.0, fd_order=2),
            m=m, backend="sh_fd")
        counts = {}
        for label, prec in (("exact LU", None),
                            ("two-grid", an.CoarseGridPreconditioner(cp, coarse, sigma))):
            info = {}
            lam_b, _ = solvers.shift_invert(K, M, sigma, 1, method="bbjd", preconditioner=prec,
                                            maxsub=40, maxiter=200, info=info)
            counts[label] = (info["nconv"], info["iters"],
                             abs(lam_b[0] - lam_r) if info["nconv"] else np.inf)
    for label, (nconv, iters, err) in counts.items():
        print(f"bbjd polish with {label:9s}: converged {nconv}, {iters} outer iterations, "
              f"|lambda - lambda_ref| = {err:.1e}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    X_, Y_ = np.meshgrid(ps.im, -ps.re)
    lv = [-3, -2.5, -2, -1.5, -1]
    cs = ax.contour(X_, Y_, ps.log10_inverse, levels=lv, cmap="viridis")
    ax.clabel(cs, fmt=lambda v: f"1e{v:.1f}", fontsize=7)
    lam_all = red.lam
    ax.plot(lam_all.imag, -lam_all.real, "k.", ms=4, label="eigenvalues")
    ax.plot(lam_r.imag, -lam_r.real, "r*", ms=12, label="followed Rossby mode")
    ax.set_xlim(im.min(), im.max())
    ax.set_ylim(-re.max(), -re.min())
    ax.set_xlabel("Re omega")
    ax.set_ylabel("Im omega (growth)")
    ax.set_title(f"eps-pseudospectrum, eps_shear = {eps_max}")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    for eps_, ls in ((0.0, "--"), (eps_max, "-")):
        red_ = results[eps_][4]
        fov_ = results[eps_][7]
        ax.plot(fov_.imag, -fov_.real, ls, color="C0", lw=1,
                label=f"numerical range boundary, eps = {eps_}")
        ax.plot(red_.lam.imag, -red_.lam.real, ".", ms=3, color="C1" if eps_ else "gray")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylim(-1.5, 0.2)
    ax.set_xlabel("Re omega")
    ax.set_ylabel("Im omega")
    ax.set_title("spectrum inside the numerical range (top edge = numerical abscissa)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axes[2]
    ax.semilogy(w_line, axis, label=f"||R(i omega)||_W, eps = {eps_max}")
    ax.semilogy(w_line, axis0, "--", label="eps = 0")
    ax.semilogy(w_line, 1 / np.min(np.abs(red.lam[None, :] - 1j * w_line[:, None]), axis=1), ":",
                color="k", label="1/dist(i omega, spectrum)")
    ax.set_xlabel("forcing frequency omega")
    ax.set_ylabel("resolvent norm on the imaginary axis")
    ax.set_title("harmonic response")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path_fig = os.path.join(FIG, "06_nonnormality.png")
    fig.savefig(path_fig, dpi=130)
    plt.close(fig)
    print(f"figure {path_fig}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    t_all = time.time()
    part_i()
    part_ii()
    print(f"\ntotal {time.time() - t_all:.1f} s")
