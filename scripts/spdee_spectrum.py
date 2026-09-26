#!/usr/bin/env python
"""Solve the solar-eigenproblem hydro system on a spdee core and write the spectrum
file (spdee.io), one file per solve.

    python scripts/spdee_spectrum.py --core sh_fd --m 8 --nr 48 --lmax 32 --nu 1e11 --k 64 --out runs/
    python scripts/spdee_spectrum.py --core dedalus --m 8 --nr 32 --ntheta 144 --nu 2e12 --dense --out runs/
    python scripts/spdee_spectrum.py --core sh_fd --m-range 1:40 --nr 48 --lmax-offset 24 --nu 2e12 --k 16 --out runs/
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

from spdee.applications.solar_strings import solar_hydro_problem, RMIN, RMAX, ekman   # noqa: E402
from spdee.io import save_spectrum, spectrum_filename                                   # noqa: E402
from spdee import solvers                                                                # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--core", choices=["sh_fd", "dedalus"], default="sh_fd")
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--m-range", default=None, help="a:b inclusive; overrides --m")
    p.add_argument("--nr", type=int, default=48)
    p.add_argument("--lmax", type=int, default=32, help="sh_fd ell_max")
    p.add_argument("--lmax-offset", type=int, default=None, help="sh_fd: ell_max = m + offset")
    p.add_argument("--ntheta", type=int, default=48, help="dedalus N_theta")
    p.add_argument("--fd-order", type=int, default=4)
    p.add_argument("--nu", type=float, default=2e12, help="cm^2/s")
    p.add_argument("--kappa", type=float, default=None, help="cm^2/s (default nu)")
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--k", type=int, default=32, help="eigenvalues per shift-invert solve")
    p.add_argument("--sigma", type=complex, default=None,
                   help="target omega/Omega0 (complex); default -2/(m+1)")
    p.add_argument("--dense", action="store_true", help="dense full-spectrum census")
    p.add_argument("--dense-backend", default="auto", help="numpy | cupy | auto")
    p.add_argument("--factor", default="scipy", help="shift-invert factorization: scipy|fplus|cudss|cupy|auto")
    p.add_argument("--out", default=".")
    p.add_argument("--tag", default="hydro")
    return p.parse_args(argv)


def solve_one(args, m):
    pb = solar_hydro_problem(nu_cgs=args.nu, kappa_cgs=args.kappa, delta=args.delta)
    if args.core == "sh_fd":
        from spdee.backends.sh_fd import ShFdGrid
        ell = (m + args.lmax_offset) if args.lmax_offset is not None else args.lmax
        grid = ShFdGrid(N_r=args.nr, ell_max=ell, r_inner=RMIN, r_outer=RMAX, fd_order=args.fd_order)
        cp = pb.compile(grid, m=m, backend="sh_fd")
        res_tag = ell
    else:
        from spdee.backends.dedalus import DedalusGrid
        grid = DedalusGrid(N_r=args.nr, N_theta=args.ntheta, r_inner=RMIN, r_outer=RMAX)
        cp = pb.compile(grid, m=m, backend="dedalus")
        res_tag = args.ntheta
    K, M = cp.assemble_evp_matrices()
    extra = dict(nu_cgs=args.nu, kappa_cgs=args.kappa if args.kappa is not None else args.nu,
                 delta=args.delta, Ek=ekman(args.nu), rotation="uniform",
                 bc=dict(v="impenetrable stress-free (both walls)", S="Neumann (both walls)"),
                 system="spdee.applications.solar_strings.solar_hydro_problem",
                 fd_order=args.fd_order if args.core == "sh_fd" else None)
    t0 = time.time()
    if args.dense:
        cen = solvers.dense(K, M, sigma=0.5j, backend=args.dense_backend, keep_vectors=True)
        lam, V, resid = cen.lam, cen.V, cen.resid
        extra.update(solver="dense", dense_backend=cen.backend, sigma=None)
    else:
        omega_t = args.sigma if args.sigma is not None else (-2.0 / (m + 1))
        sigma = 1j * omega_t + 0.02
        lam, V = solvers.shift_invert(K, M, sigma, args.k, factor=args.factor, return_vectors=True)
        keep = np.isfinite(lam)
        lam, V = lam[keep], V[:, keep]
        resid = None
        extra.update(solver="arpack shift-invert", factor=args.factor, k=args.k,
                     sigma=str(sigma), sigma_omega=str(omega_t))
    extra["wall_s"] = time.time() - t0
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, spectrum_filename(args.core, args.tag, m, args.nr, res_tag, args.nu))
    save_spectrum(path, cp, lam, V, meta_extra=extra, resid=resid)
    om = -1j * lam
    j = int(np.argmin(np.abs(om - (-2.0 / (m + 1)))))
    print(f"m={m:2d} {args.core} N={K.shape[0]} n={len(lam)} wall={extra['wall_s']:.1f}s "
          f"nearest Rossby: omega/Omega0 = {om[j]:.6f} ({om[j].real*456:.3f}{om[j].imag*456:+.3f}i nHz) -> {path}")
    return path


def main(argv=None):
    args = parse_args(argv)
    if args.m_range:
        a, b = (int(x) for x in args.m_range.split(":"))
        ms = range(a, b + 1)
    else:
        ms = [args.m]
    return [solve_one(args, m) for m in ms]


if __name__ == "__main__":
    main()
