#!/usr/bin/env python
"""Operator-level comparison of spdee's Dedalus core against the Bekki
harness Dedalus MHD problem (solar_eigenmodes.py) for the BCG22 hydro system with
the toroidal eqpeak x sqrtrho background field and conducting walls.

    python scripts/certify_dedalus_mhd_vs_seed.py --nr 16 --nt 48 --m 20 --b0 4534

The harness evolves the vector potential A (b = curl A, Coulomb gauge psi,
tangential A = 0 and psi = 0 at the walls) and scales A by Le; spdee is
compiled twice, in the same vector-potential form (a direct comparison of
every equation) and in the production flux form b with b_r = 0 and total
tangential E = 0 (momentum compared with b = curl A, induction compared
with the curl of the harness's A equation).  The harness takes J0 as the
spectral curl of its B0 field; spdee carries the analytic J0 of
`toroidal_eqpeak_field` (--j0 analytic) or the same spectral curl
(--j0 spectral).  The premultipliers of the harness (r^2 H_rho, H_rho, H_p,
Le) are divided out and the images of identical random states compared in
grid space, each equation scaled by the size of its image on the full
state.  The flux-form induction comparison goes through the curl of the
harness's image, whose terms Dedalus sums on the k = 2 radial basis by
truncating conversions, so it is exact only for states resolved to
roundoff: --decay sets the radial coefficient decay of the random states
(the vector-potential rows are exact at any decay).  With --eig the
eigenvalue nearest the given target is also solved by shift-invert on all
three pencils.

Seed tree: $SPDEE_SEED_TREE (read-only).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from certify_dedalus_vs_seed import SEED, git_head  # noqa: E402

NHZ = 456.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nr", type=int, default=16)
    p.add_argument("--nt", type=int, default=48)
    p.add_argument("--m", type=int, default=20)
    p.add_argument("--nu", type=float, default=2e12)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--b0", type=float, default=4534.0, help="B_phi at the base, gauss")
    p.add_argument("--j0", default="analytic", choices=["analytic", "spectral"],
                   help="spdee's background current: the analytic curl or Curl(B0) on the core")
    p.add_argument("--decay", type=float, default=0.5,
                   help="radial coefficient decay rate exp(-decay n) of the random states")
    p.add_argument("--eig", type=float, default=None,
                   help="omega target in nHz for a shift-invert check on the three pencils")
    return p.parse_args(argv)


def load_seed(args):
    os.environ.update(DEDALUS_COMM_SELF="1", DEDALUS_NR=str(args.nr), DEDALUS_NT=str(args.nt),
                      DEDALUS_M=str(args.m), DELTA=str(args.delta), NU_CGS=repr(args.nu),
                      DEDALUS_PROBLEM="solar_eigenmodes", FIELD_TYPE="toroidal", TOR_PROFILE="sqrtrho",
                      TOR_THETA="eqpeak", B0_SURF_GAUSS=repr(args.b0), MAG_BC="conducting",
                      PM="1.0")
    for p in (SEED, os.path.join(SEED, "bcg22")):
        if p not in sys.path:
            sys.path.insert(0, p)
    BM = __import__("solar_eigenmodes")
    BM.solver.build_matrices([BM.sp], ["M", "L"])
    return BM


def load_spdee(args, n_phi, formulation):
    from spdee.applications.solar_strings import solar_hydro_problem, ekman, RMIN, RMAX
    from spdee.applications.solar_eigenmodes import add_magnetic_field, toroidal_eqpeak_field
    from spdee.backends.dedalus import DedalusGrid, compile_dedalus
    from spdee import Curl
    from spdee.ir.expressions import FieldRef
    F = toroidal_eqpeak_field(args.b0)
    J0 = F.J0 if args.j0 == "analytic" else Curl(FieldRef(F.B0))
    hydro = solar_hydro_problem(nu_cgs=args.nu, delta=args.delta).build()
    system = add_magnetic_field(hydro, F.B0, Le=F.Le, eta=ekman(args.nu), J0=J0,
                                inv_rho_ratio=F.inv_rho_ratio, bc="conducting",
                                lorentz_sign=-1.0, formulation=formulation,
                                b_name="A" if formulation == "A" else "b")
    grid = DedalusGrid(N_r=args.nr, N_theta=args.nt, r_inner=RMIN, r_outer=RMAX, N_phi=n_phi)
    return compile_dedalus(system, grid, m=args.m), F


def random_state(cp, rng, decay, only=None):
    """Random coefficients in every unknown (or one), decaying radially as
    exp(-decay n), taus zero."""
    for tau in cp.tau_fields:
        tau["c"] = 0.0
    for name in cp.unknown_names:
        f = cp.fields[name]
        f["c"] = 0.0
        if only is not None and name != only:
            continue
        c = f["c"]
        amp = rng.normal(size=c.shape) + 1j * rng.normal(size=c.shape)
        f["c"] = amp * np.exp(-decay * np.arange(c.shape[-1]))
        f["g"]


def copy_state(cp, BM, Le):
    """The spdee A-form state into the harness fields; the harness's A and
    psi carry the factor Le."""
    for name, seed_name, scale in (("v", "u", 1.0), ("rhat", "rhat", 1.0), ("S", "S", 1.0),
                                   ("A", "A", Le), ("psi", "psi", Le)):
        getattr(BM, seed_name)["g"] = scale * np.array(cp.fields[name]["g"])
    for tau in (BM.tau_u1, BM.tau_u2, BM.tau_S1, BM.tau_S2, BM.tau_A1, BM.tau_A2):
        tau["c"] = 0.0


def copy_to_flux(cp_A, cp_b):
    """The same state on the flux-form problem, b = curl A on the grid."""
    for tau in cp_b.tau_fields:
        tau["c"] = 0.0
    for name in ("v", "rhat", "S"):
        cp_b.fields[name]["g"] = np.array(cp_A.fields[name]["g"])
    cp_b.fields["b"]["g"] = np.array(cp_A.d3.curl(cp_A.fields["A"]).evaluate()["g"])


def eq_images(problem, s_eig, s_val, n):
    s_eig["g"] = s_val
    return [np.array(eq["LHS"].evaluate()["g"]) for eq in problem.equations[:n]]


def nearest_eig(K, M, omega, k=6):
    from spdee.solvers import shift_invert
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    lam, _ = shift_invert(K, M, 1j * omega + 0.02, k, v0=v0)
    om = -1j * lam[np.isfinite(lam)]
    return om[np.argmin(np.abs(om - omega))]


def main(argv=None):
    args = parse_args(argv)
    print(f"seed: {git_head(SEED)}\nspdee: {git_head(_ROOT)}")
    print(f"m={args.m} N_r={args.nr} N_theta={args.nt} nu=kappa=eta={args.nu:g} "
          f"delta={args.delta:g} B0={args.b0:g} G toroidal sqrtrho x eqpeak, conducting walls")
    BM = load_seed(args)
    cp_A, F = load_spdee(args, BM.Nphi, "A")
    cp_b, _ = load_spdee(args, BM.Nphi, "b")
    print(f"Le seed {BM.Le:.12e}  spdee {F.Le:.12e}  rho0_ref seed {BM.rho0_ref:.6f} @ r={BM.r_ref}"
          f"  spdee {F.rho0_ref:.6f} @ r={F.r_ref}  Em seed {BM.Em:.10e}")
    for tag, L, M_ in (("seed  ", BM.sp.L_min, BM.sp.M_min),
                       ("spdee A", cp_A.A_total, cp_A.B_total),
                       ("spdee b", cp_b.A_total, cp_b.B_total)):
        print(f"pencil {tag} N={L.shape[0]} nnz(L)={L.nnz} nnz(M)={M_.nnz}")

    rng = np.random.default_rng(0)
    s_val = 1j * (-2.0 / (args.m + 1))
    R = BM.r2["g"]
    premult = [R * BM.Hrho["g"], BM.Hrho["g"], BM.Hp["g"], BM.Le, BM.Le]
    names = ["momentum", "continuity", "entropy", "induction", "gauge"]
    d3 = cp_A.d3
    print(f"operator images on identical random states (taus zero, radial decay "
          f"exp(-{args.decay:g} n)), max difference over the size of the full-state image:")
    print(f"{'state':>10s} " + " ".join(f"{n:>12s}" for n in names)
          + f" | {'b momentum':>12s} {'b induction':>12s}")
    scale = None
    worst = 0.0
    for only in (None, "v", "rhat", "S", "A", "psi"):
        random_state(cp_A, rng, args.decay, only=only)
        copy_state(cp_A, BM, BM.Le)
        copy_to_flux(cp_A, cp_b)
        img_A = eq_images(cp_A.problem, cp_A.s_eig, s_val, 5)
        img_b = eq_images(cp_b.problem, cp_b.s_eig, s_val, 4)
        img_s = eq_images(BM.problem, BM.s_eig, s_val, 5)
        phys_s = [i / pm for i, pm in zip(img_s, premult)]
        ind_s = BM.problem.equations[3]["LHS"].evaluate()
        curl_ind = np.array(d3.curl(ind_s).evaluate()["g"]) / BM.Le
        if scale is None:
            scale = [np.max(np.abs(x)) for x in phys_s] + [np.max(np.abs(curl_ind))]
        row = [np.max(np.abs(a - b)) / sc for a, b, sc in zip(img_A, phys_s, scale)]
        if only == "A":
            lorentz = np.max(np.abs(img_A[0] - phys_s[0])) / np.max(np.abs(phys_s[0]))
        row_b = [np.max(np.abs(img_b[0] - phys_s[0])) / scale[0],
                 np.max(np.abs(img_b[3] - curl_ind)) / scale[5]]
        worst = max([worst] + row)
        label = "all" if only is None else f"only {only}"
        print(f"{label:>10s} " + " ".join(f"{x:12.3e}" for x in row)
              + " | " + " ".join(f"{x:12.3e}" for x in row_b))
    print(f"worst relative difference, vector-potential form: {worst:.3e}; "
          f"Lorentz force against its own size (only A, J0 {args.j0}): {lorentz:.3e}")

    if args.eig is not None:
        omega = args.eig / NHZ
        for tag, L, M_ in (("seed A ", BM.sp.L_min, BM.sp.M_min),
                           ("spdee A", cp_A.A_total, cp_A.B_total),
                           ("spdee b", cp_b.A_total, cp_b.B_total)):
            t0 = time.time()
            om = nearest_eig(L, M_, omega)
            print(f"  {tag}: omega/Omega0 = {om.real:+.7f}{om.imag:+.7f}j = "
                  f"{om.real * NHZ:+.4f}{om.imag * NHZ:+.4f}j nHz  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
