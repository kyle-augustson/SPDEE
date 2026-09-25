#!/usr/bin/env python
"""Solve the headline cases of the magneto-inertial-mode paper with SPDEE, on
every core, and print the values with the commit, for the reproduction table.

    python scripts/reproduce_paper_cases.py --case hydro_m8 --cores dedalus,staggered
    python scripts/reproduce_paper_cases.py --case mhd_eqpeak --cores dedalus --m 20,30

mhd_eqpeak: the toroidal sqrtrho x eqpeak band, B_phi = 4534 G at the base,
conducting walls, at m = 20 and 30, by shift-invert near the harness's
magnetic branch; the field-free mode at each m as a check.
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

from spdee.applications.solar_strings import (  # noqa: E402
    solar_hydro_problem, ekman, RMIN, RMAX, OMEGA_CARR)
from spdee.applications.solar_eigenmodes import add_magnetic_field, toroidal_eqpeak_field  # noqa: E402
from spdee.solvers import shift_invert  # noqa: E402

NHZ = 1e9 / (2 * np.pi) * OMEGA_CARR

# harness values (sigma in nHz) of the eqpeak x sqrtrho 4534 G branch and
# the field-free sectoral modes at the same m, Dedalus 32 x 144
HARNESS = {20: dict(hydro=-44.1 - 40.3j, mhd=-161.8), 30: dict(hydro=-30.0 - 81.5j, mhd=-223.6)}


def rossby(K, M, m, k=8):
    """omega/Omega0 of the sectoral Rossby mode (nearest -2/(m+1)) and the
    relative pencil residual of its eigenpair."""
    target = -2.0 / (m + 1)
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    lam, V = shift_invert(K, M, 1j * target + 0.02, k, v0=v0, return_vectors=True)
    keep = np.isfinite(lam)
    lam, V = lam[keep], V[:, keep]
    om = -1j * lam
    j = int(np.argmin(np.abs(om - target)))
    x = V[:, j]
    resid = np.linalg.norm(K @ x - lam[j] * (M @ x)) / np.linalg.norm(K @ x)
    return om[j], resid


def near(K, M, target, k=8, offset=0.02, factor="scipy"):
    """The k eigenpairs of the pencil nearest omega = target (lambda = i omega,
    the shift ``offset`` below it in Im omega), as (omega, residual) sorted
    by distance from target."""
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    lam, V = shift_invert(K, M, 1j * target + offset, k, v0=v0, return_vectors=True,
                          factor=factor)
    keep = np.isfinite(lam)
    lam, V = lam[keep], V[:, keep]
    om = -1j * lam
    out = []
    for j in np.argsort(np.abs(om - target)):
        x = V[:, j]
        out.append((om[j], np.linalg.norm(K @ x - lam[j] * (M @ x)) / np.linalg.norm(K @ x)))
    return out


def default_factor(K, dense_row=500):
    """Dense LU when the pencil is dense enough that a sparse LU fills it
    anyway (the toroidal NCC couples every l on the Dedalus core: about 2700
    entries per row at 32 x 144, where SuperLU runs out of memory)."""
    return "lapack" if K.nnz > dense_row * K.shape[0] else "scipy"


def compile_system(system, core, nr, nt, m):
    if core == "dedalus":
        from spdee.backends.dedalus import compile_dedalus
        return compile_dedalus(system, grid_for(core, nr, nt)[0], m=m)
    if core == "staggered":
        from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
        return compile_fd2_staggered(system, grid_for(core, nr, nt)[0], m=m, order_r=2,
                                     order_theta=2)
    raise ValueError(f"mhd_eqpeak runs on the dedalus or staggered core, not {core}")


def fmt(om):
    return (f"omega/Omega0 = {om.real:+.6f}{om.imag:+.6f}j = "
            f"{om.real * NHZ:+.3f}{om.imag * NHZ:+.3f}j nHz")


def mhd_eqpeak(cores, nr, nt, nu, ms, b0, k, factor):
    F = toroidal_eqpeak_field(b0)
    print(f"toroidal sqrtrho x eqpeak, B_phi(base) = {b0:g} G, rho0_ref = {F.rho0_ref:g} g/cm^3 "
          f"at r = {F.r_ref:g}, Le = {F.Le:.6e}, eta = nu = kappa = {nu:g}")
    hydro = solar_hydro_problem(nu_cgs=nu).build()
    mhd = add_magnetic_field(hydro, F.B0, Le=F.Le, eta=ekman(nu), J0=F.J0,
                             inv_rho_ratio=F.inv_rho_ratio, bc="conducting",
                             lorentz_sign=-1.0, formulation="b")
    for core in cores:
        for m in ms:
            h = HARNESS.get(m, {})
            for tag, system in (("hydro", hydro), ("mhd", mhd)):
                t0 = time.time()
                K, M = compile_system(system, core, nr, nt, m).assemble_evp_matrices()
                t_build = time.time() - t0
                ref = h.get(tag)
                target = (-2.0 / (m + 1)) if ref is None else (complex(ref) - 3.0) / NHZ
                fac = factor or default_factor(K)
                pairs = near(K, M, target, k=k, factor=fac,
                             offset=0.0 if isinstance(ref, complex) else 0.02)
                om, resid = pairs[0]
                if ref is not None:
                    om, resid = min(pairs, key=lambda p: abs(p[0] - complex(ref) / NHZ))
                print(f"{tag:5s} m={m:2d} {core:9s} {nr}x{nt} N={K.shape[0]:6d}  {fmt(om)}  "
                      f"resid {resid:.1e}  harness {ref}  (build {t_build:.0f} s, "
                      f"total {time.time() - t0:.0f} s, factor {fac})", flush=True)
                others = ", ".join(f"{o.real * NHZ:+.1f}{o.imag * NHZ:+.1f}j" for o, _ in pairs[1:])
                print(f"      other eigenvalues returned (nHz): {others}", flush=True)


def grid_for(core, nr, nt):
    if core == "dedalus":
        from spdee.backends.dedalus import DedalusGrid
        return DedalusGrid(N_r=nr, N_theta=nt, r_inner=RMIN, r_outer=RMAX), {"backend": "dedalus"}
    if core == "sh_fd":
        from spdee.backends.sh_fd import ShFdGrid
        return (ShFdGrid(N_r=nr, ell_max=8 + nt // 6, r_inner=RMIN, r_outer=RMAX, fd_order=4),
                {"backend": "sh_fd"})
    from spdee.backends.fd2.grid import make_shell_grid
    if core == "staggered":
        return (make_shell_grid(N_r=nr + 1, N_theta=nt, r_inner=RMIN, r_outer=RMAX),
                {"backend": "fd2", "discretization": "staggered"})
    return make_shell_grid(N_r=nr, N_theta=nt, r_inner=RMIN, r_outer=RMAX), {"backend": "fd2"}


def hydro_m8(cores, nr, nt, nu):
    m = 8
    pb = solar_hydro_problem(nu_cgs=nu)
    for core in cores:
        grid, kw = grid_for(core, nr, nt)
        t0 = time.time()
        cp = pb.compile(grid, m=m, **kw)
        K, M = cp.assemble_evp_matrices()
        om, resid = rossby(K, M, m)
        print(f"hydro m=8 nu={nu:g} {core:10s} {nr}x{nt} N={K.shape[0]:6d}  omega/Omega0 = "
              f"{om.real:+.6f}{om.imag:+.6f}j  = {om.real * NHZ:+.4f}{om.imag * NHZ:+.4f}j nHz  "
              f"resid {resid:.1e}  ({time.time() - t0:.0f} s)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="hydro_m8", choices=["hydro_m8", "mhd_eqpeak"])
    p.add_argument("--cores", default="dedalus,staggered,sh_fd,collocated")
    p.add_argument("--nr", type=int, default=32)
    p.add_argument("--nt", type=int, default=144)
    p.add_argument("--nu", type=float, default=2e12)
    p.add_argument("--m", default="20,30", help="mhd_eqpeak: azimuthal orders")
    p.add_argument("--b0", type=float, default=4534.0, help="mhd_eqpeak: B_phi at the base, G")
    p.add_argument("--k", type=int, default=8, help="eigenvalues per shift")
    p.add_argument("--factor", default=None,
                   help="shift-invert factorization (default: by the pencil's density)")
    a = p.parse_args(argv)
    commit = subprocess.check_output(["git", "-C", _ROOT, "rev-parse", "--short", "HEAD"],
                                     text=True).strip()
    print(f"spdee {commit}; Omega0 = 456 nHz")
    cores = a.cores.split(",")
    if a.case == "hydro_m8":
        hydro_m8(cores, a.nr, a.nt, a.nu)
    else:
        if a.cores == "dedalus,staggered,sh_fd,collocated":
            cores = ["dedalus"]
        mhd_eqpeak(cores, a.nr, a.nt, a.nu, [int(x) for x in a.m.split(",")], a.b0, a.k, a.factor)


if __name__ == "__main__":
    main()
