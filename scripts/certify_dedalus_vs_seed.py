#!/usr/bin/env python
"""Operator-level comparison of spdee's Dedalus core against the Bekki
harness Dedalus problem (bekki_problem.py) for the BCG22 hydro system.

    python scripts/certify_dedalus_vs_seed.py --nr 16 --nt 72 --m 8

Reports, at one matched resolution: the pencil sizes; the generalized
eigenvalues of both pencils paired in the physical window; and the
discretized operator applied to identical states, equation by equation and
unknown by unknown, compared in grid space.  The harness premultiplies its
equations (momentum by r^2 H_rho, continuity by H_rho, entropy by H_p) and
reduces grad(u), grad(S) to first order with extra taus, so the two pencils
are not permutations of each other; the grid-space comparison divides the
premultipliers out and compares the physical operators directly.

Seed tree: $SPDEE_SEED_TREE (read-only); the
commit it was run against is printed in the banner.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
SEED = os.path.expanduser(os.environ.get("SPDEE_SEED_TREE", ""))

WINDOW = dict(re_max=2.2, im_min=-0.6, im_max=0.1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nr", type=int, default=16)
    p.add_argument("--nt", type=int, default=72)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--nu", type=float, default=2e12)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--skip-eig", action="store_true", help="skip the dense eigenvalue pairing")
    return p.parse_args(argv)


def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "log", "--oneline", "-1"], text=True).strip()
    except Exception:
        return "unknown"


def load_seed(args):
    os.environ.update(DEDALUS_COMM_SELF="1", DEDALUS_NR=str(args.nr), DEDALUS_NT=str(args.nt),
                      DEDALUS_M=str(args.m), DELTA=str(args.delta))
    for p in (SEED, os.path.join(SEED, "bcg22")):
        if p not in sys.path:
            sys.path.insert(0, p)
    BM = __import__("bekki_problem")
    BM.solver.build_matrices([BM.sp], ["M", "L"])
    return BM


def load_spdee(args, n_phi):
    from spdee.applications.solar_strings import solar_hydro_problem, RMIN, RMAX
    from spdee.backends.dedalus import DedalusGrid
    pb = solar_hydro_problem(nu_cgs=args.nu, delta=args.delta)
    grid = DedalusGrid(N_r=args.nr, N_theta=args.nt, r_inner=RMIN, r_outer=RMAX, N_phi=n_phi)
    return pb.compile(grid, m=args.m, backend="dedalus")


def window_eigs(L, M):
    """omega = -i lambda for L x = lambda M x, restricted to the physical window."""
    from spdee import solvers
    lam = solvers.dense(L, M, sigma=0.5j, backend="auto").lam
    om = -1j * lam[np.isfinite(lam)]
    keep = (np.abs(om.real) <= WINDOW["re_max"]) & (om.imag >= WINDOW["im_min"]) & (om.imag <= WINDOW["im_max"])
    return om[keep]


def pair(om_a, om_b):
    """Nearest-neighbour pairing of a into b; returns |d omega| per a-mode."""
    d = np.abs(om_a[:, None] - om_b[None, :])
    return d.min(axis=1), d.argmin(axis=1)


def random_state(cp, rng, only=None):
    """Smooth random coefficients in every unknown (or one), taus zero."""
    for tau in cp.tau_fields:
        tau["c"] = 0.0
    for name in cp.unknown_names:
        f = cp.fields[name]
        f["c"] = 0.0
        if only is not None and name != only:
            continue
        c = f["c"]
        amp = rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape)
        decay = np.exp(-0.15 * np.arange(c.shape[-1]))
        f["c"] = amp * decay
        f["g"]


def copy_state(cp, BM):
    for name, seed_name in (("v", "u"), ("rhat", "rhat"), ("S", "S")):
        getattr(BM, seed_name)["g"] = np.array(cp.fields[name]["g"])
    for tau in (BM.tau_u1, BM.tau_u2, BM.tau_S1, BM.tau_S2):
        tau["c"] = 0.0


def eq_images(problem, s_eig, s_val):
    s_eig["g"] = s_val
    return [np.array(eq["LHS"].evaluate()["g"]) for eq in problem.equations[:3]]


def main(argv=None):
    args = parse_args(argv)
    print(f"seed: {git_head(SEED)}\nspdee: {git_head(_ROOT)}")
    print(f"m={args.m} N_r={args.nr} N_theta={args.nt} nu=kappa={args.nu:g} delta={args.delta:g}")
    BM = load_seed(args)
    cp = load_spdee(args, n_phi=BM.Nphi)
    L_s, M_s = BM.sp.L_min, BM.sp.M_min
    L_p, M_p = cp.A_total, cp.B_total
    from spdee.applications.solar_strings import ekman
    print(f"Ek seed {BM.Ek:.10e}  spdee {ekman(args.nu):.10e}")
    print(f"pencil seed  N={L_s.shape[0]} nnz(L)={L_s.nnz} nnz(M)={M_s.nnz}  variables={[v.name for v in BM.problem.variables]}")
    print(f"pencil spdee N={L_p.shape[0]} nnz(L)={L_p.nnz} nnz(M)={M_p.nnz}  variables={[v.name for v in cp.problem.variables]}")

    if not args.skip_eig:
        t0 = time.time()
        om_s = window_eigs(L_s, M_s)
        om_p = window_eigs(L_p, M_p)
        d_ps, _ = pair(om_p, om_s)
        d_sp, _ = pair(om_s, om_p)
        target = -2.0 / (args.m + 1)
        r_s = om_s[np.argmin(np.abs(om_s - target))]
        r_p = om_p[np.argmin(np.abs(om_p - target))]
        print(f"window eigenvalues: seed {len(om_s)}  spdee {len(om_p)}   dense time {time.time() - t0:.0f}s")
        print(f"  Rossby: seed {r_s:.8f}  spdee {r_p:.8f}  |d|={abs(r_s - r_p):.2e}")
        for tag, d in (("spdee->seed", d_ps), ("seed->spdee", d_sp)):
            print(f"  pairing {tag}: median |d omega| {np.median(d):.2e}  90% {np.quantile(d, 0.9):.2e}  "
                  f"max {d.max():.2e}  n(|d|<1e-8)={int((d < 1e-8).sum())}  n(|d|<1e-6)={int((d < 1e-6).sum())}")

    rng = np.random.default_rng(0)
    s_val = 1j * (-2.0 / (args.m + 1))
    R = BM.r2["g"]
    premult = [R * BM.Hrho["g"], BM.Hrho["g"], BM.Hp["g"]]
    eq_names = ["momentum", "continuity", "entropy"]
    print("operator images on identical random states (taus zero), relative max difference per equation:")
    print(f"{'state':>10s} " + " ".join(f"{n:>12s}" for n in eq_names))
    for only in (None, "v", "rhat", "S"):
        random_state(cp, rng, only=only)
        copy_state(cp, BM)
        img_p = eq_images(cp.problem, cp.s_eig, s_val)
        img_s = eq_images(BM.problem, BM.s_eig, s_val)
        row = []
        for ip, is_, pm in zip(img_p, img_s, premult):
            phys_s = is_ / pm
            scale = max(np.max(np.abs(ip)), np.max(np.abs(phys_s)), 1e-300)
            row.append(np.max(np.abs(ip - phys_s)) / scale)
        label = "all" if only is None else f"only {only}"
        print(f"{label:>10s} " + " ".join(f"{x:12.3e}" for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
