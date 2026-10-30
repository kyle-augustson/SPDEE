"""``python -m spdee.check``: what this installation of SPDEE can run.

Prints the package versions by requirement tier, each numerical core, each
solver method, factor and dense backend with the reason it is missing,
the GPU vendor, device and capabilities, the thread settings and the
environment switches, then compiles and solves one small pencil on every
importable host core.  Exit code 0 when the minimum tier (numpy, scipy,
sympy; the collocated, staggered and sh_fd cores; ARPACK, SuperLU and the
dense host census) works, 1 otherwise.  ``--json`` prints the same report
as one JSON object.
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["report", "main"]

TIERS = (
    ("numpy", "required"), ("scipy", "required"), ("sympy", "required"),
    ("dedalus", "dedalus"), ("cupy", "gpu-cuda / gpu-rocm"), ("nvmath", "gpu-cuda"),
    ("matplotlib", "viz"), ("mpi4py", "mpi"), ("h5py", "dedalus"), ("pytest", "test"),
)
CORES = (
    ("collocated", "spdee.backends.fd2", "fd2, discretization='collocated'"),
    ("staggered", "spdee.backends.fd2.staggered_general", "fd2, discretization='staggered'"),
    ("sh_fd", "spdee.backends.sh_fd", "spin-weighted harmonics x radial FD"),
    ("dedalus", "dedalus.public", "Dedalus v3 spectral shell / ball"),
)
SWITCHES = ("SPDEE_NO_GPU", "CUDSS_USE_MATCHING", "SPDEE_USE_PARDISO", "GPU_SI_HOST_TRANSPOSE",
            "CUDSS_THREADING_LIB")
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS")


def _version(name: str) -> Tuple[Optional[str], str]:
    """(version, reason) with version None when the import fails."""
    try:
        if name.split(".")[0] == "dedalus":
            logging.getLogger("dedalus").setLevel(logging.ERROR)
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    return str(getattr(mod, "__version__", "?")), ""


def _importable(name: str) -> Tuple[bool, str]:
    v, why = _version(name)
    return v is not None, why


def _fmt_bytes(n: Optional[int]) -> str:
    return "?" if n is None else f"{n / 2**30:.1f} GiB"


def _host_memory() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None


def _test_problem():
    from spdee import Problem
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="inner")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


def _solve_cores(cores: Dict[str, Any]) -> Dict[str, Any]:
    """Compile the m=2 inviscid inertial problem on each importable host
    core and find the Rossby mode omega = -2/3 with ARPACK + SuperLU; the
    sh_fd pencil is also censused densely on the host."""
    import numpy as np
    from spdee import solvers
    out: Dict[str, Any] = {}
    target = -2.0 / 3.0
    sigma = 1j * target + 0.02
    try:
        pb = _test_problem()
        from spdee.backends.fd2.grid import make_shell_grid
        grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.71, r_outer=0.985)
    except Exception as exc:  # noqa: BLE001
        return {"problem": {"ok": False, "why": f"{type(exc).__name__}: {exc}"}}
    jobs = []
    if cores["collocated"]["ok"]:
        jobs.append(("collocated", lambda: pb.compile(grid, m=2, gauge_unknowns=["p"])))
    if cores["staggered"]["ok"]:
        jobs.append(("staggered", lambda: pb.compile(grid, m=2, gauge_unknowns=["p"],
                                                     discretization="staggered")))
    if cores["sh_fd"]["ok"]:
        def _sh():
            from spdee.backends.sh_fd import ShFdGrid
            g = ShFdGrid(N_r=8, ell_max=8, r_inner=0.71, r_outer=0.985)
            return pb.compile(g, m=2, backend="sh_fd", gauge_unknowns=["p"])
        jobs.append(("sh_fd", _sh))
    for name, build in jobs:
        try:
            K, M = build().assemble_evp_matrices()
            lam, _ = solvers.shift_invert(K, M, sigma, 4, factor="scipy")
            om = -1j * lam[np.isfinite(lam)]
            err = float(np.min(np.abs(om - target)))
            rec: Dict[str, Any] = {"ok": err < 5e-2, "N": int(K.shape[0]), "rossby_err": err}
            if name == "sh_fd":
                res = solvers.dense(K, M, sigma=sigma, backend="numpy")
                omd = -1j * res.lam
                rec["dense_err"] = float(np.min(np.abs(omd - target)))
                rec["ok"] = rec["ok"] and rec["dense_err"] < 5e-2
            out[name] = rec
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return out


def report(solve: bool = True) -> Dict[str, Any]:
    """The capability report as nested dicts (what ``main`` prints)."""
    rep: Dict[str, Any] = {
        "python": f"{platform.python_version()} ({sys.platform}, {platform.machine()})",
        "platform": platform.platform(),
        "executable": sys.executable,
        "host_memory_bytes": _host_memory(),
    }
    try:
        import spdee
        rep["spdee"] = {"version": spdee.__version__,
                        "path": os.path.dirname(os.path.abspath(spdee.__file__))}
    except Exception as exc:  # noqa: BLE001
        rep["spdee"] = {"version": None, "why": f"{type(exc).__name__}: {exc}"}
        rep["packages"] = {n: dict(zip(("version", "why"), _version(n)), tier=t)
                           for n, t in TIERS}
        rep["minimum_tier_ok"] = False
        return rep
    rep["packages"] = {n: dict(zip(("version", "why"), _version(n)), tier=t) for n, t in TIERS}

    cores: Dict[str, Any] = {}
    for name, module, desc in CORES:
        ok, why = _importable(module)
        cores[name] = {"ok": ok, "module": module, "desc": desc, "why": why}
    rep["cores"] = cores

    from spdee import solvers
    from spdee.solvers import gpu
    avail = solvers.available()
    feat = gpu.features()
    fplus_note = ("C extension" if avail.get("fplus_c_ext")
                  else "pure-Python fallback (spdee/solvers/build_ext.sh builds the C extension)")
    from spdee.solvers.external import describe as _ext_describe
    ext = _ext_describe()
    if ext["fplus"]["ok"]:
        fplus_note = f"FPLUS library at {ext['fplus']['path']} (copy: {fplus_note})"
    elif ext["fplus"]["path"]:
        fplus_note += f"; SPDEE_FPLUS_PATH unusable: {ext['fplus']['why']}"
    bbjd_ok, bbjd_why = _importable("spdee.solvers.bbjd")
    rep["methods"] = {
        "arpack": {"ok": True, "note": "scipy ARPACK"},
        "bbjd": {"ok": bbjd_ok or ext["bbjd"]["ok"],
                 "note": (f"BBJD library at {ext['bbjd']['path']}" if ext["bbjd"]["ok"] else
                          "C extension" if bbjd_ok else
                          f"not built ({bbjd_why}); spdee/solvers/build_ext.sh")
                 + (f"; SPDEE_BBJD_PATH unusable: {ext['bbjd']['why']}"
                    if ext["bbjd"]["path"] and not ext["bbjd"]["ok"] else "")},
    }
    gpu_order = solvers.gpu_factor_order()
    prefer, prefer_why = gpu.prefer_device()
    auto_why = "" if prefer or gpu.device() is None else f"  ({prefer_why})"
    rep["factors"] = {
        "scipy": {"ok": True, "note": "SuperLU"},
        "lapack": {"ok": True, "note": "dense host LU"},
        "fplus": {"ok": bool(avail.get("fplus")), "note": fplus_note},
        "cudss": {"ok": feat["cudss"][0], "note": feat["cudss"][1] or "cuDSS via nvmath-python"},
        "cupy": {"ok": feat["sparse_lu"][0], "note": feat["sparse_lu"][1] or
                 "cupyx splu: SuperLU on the host, slower solves on the device; explicit only"},
        "auto": {"ok": True, "note": "-> " + (gpu_order[0] if gpu_order else "scipy" + auto_why)},
    }
    if feat["dense_lu"][0]:
        if feat["xgeev"][0]:
            geev = "device LU + cusolverDnXgeev"
        elif feat["cupy_geev"][0]:
            geev = "device LU + cp.linalg.eig"
        else:
            geev = f"device LU + host geev ({feat['xgeev'][1]})"
    else:
        geev = feat["dense_lu"][1]
    rep["dense"] = {
        "numpy": {"ok": True, "note": "host LU + LAPACK geev"},
        "cupy": {"ok": feat["dense_lu"][0], "note": geev},
        "auto": {"ok": True, "note": "-> " + ("cupy" if prefer and feat["dense_lu"][0]
                                                else "numpy" + auto_why)},
    }

    dev = gpu.device()
    g: Dict[str, Any] = {"disabled": gpu.disabled(), "vendor": avail["gpu_vendor"],
                         "device": avail["gpu_device"]}
    if dev is not None:
        g.update(count=dev.count, free_bytes=dev.free_bytes, total_bytes=dev.total_bytes,
                 cupy=dev.cupy_version, runtime=dev.runtime_version, integrated=dev.integrated,
                 device_class=dev.klass, auto_prefers_device=prefer)
    elif gpu.disabled():
        g["why"] = "SPDEE_NO_GPU set"
    else:
        g["why"] = feat["dense_lu"][1]
    if dev is not None and not prefer:
        g["auto_why"] = prefer_why
    g["features"] = {k: {"ok": ok, "why": why} for k, (ok, why) in feat.items()}
    g["mapped_libraries"] = list(gpu.mapped_libraries())
    rep["gpu"] = g

    threads: Dict[str, Any] = {v: os.environ.get(v) for v in THREAD_VARS}
    threads["cpu_count"] = os.cpu_count()
    try:
        import numpy as np
        cfg = getattr(np.__config__, "CONFIG", {})
        threads["numpy_blas"] = cfg.get("Build Dependencies", {}).get("blas", {}).get("name")
    except Exception:  # noqa: BLE001
        threads["numpy_blas"] = None
    try:
        from threadpoolctl import threadpool_info
        threads["pools"] = [f"{p['internal_api']} {p.get('num_threads')}"
                            for p in threadpool_info()]
    except Exception:  # noqa: BLE001
        pass
    rep["threads"] = threads
    rep["switches"] = {v: os.environ.get(v) for v in SWITCHES}

    if solve:
        rep["solve"] = _solve_cores(cores)
    host_cores = all(cores[n]["ok"] for n in ("collocated", "staggered", "sh_fd"))
    solved = all(r.get("ok") for r in rep.get("solve", {}).values()) if solve else True
    required = all(rep["packages"][n]["version"] for n in ("numpy", "scipy", "sympy"))
    rep["minimum_tier_ok"] = bool(required and host_cores and solved)
    return rep


def _print(rep: Dict[str, Any]) -> None:
    def row(label: str, ok: Optional[bool], text: str) -> None:
        mark = "  " if ok is None else ("ok" if ok else "--")
        print(f"  {mark} {label:<14s} {text}")

    print("SPDEE capability check")
    print(f"  python       {rep['python']}")
    print(f"  platform     {rep['platform']}   host memory {_fmt_bytes(rep['host_memory_bytes'])}")
    print(f"  executable   {rep['executable']}")
    sp = rep["spdee"]
    print(f"  spdee        {sp.get('version') or 'not importable: ' + sp.get('why', '')}"
          f"{'  ' + sp['path'] if sp.get('path') else ''}")
    print("\npackages (tier)")
    for name, p in rep["packages"].items():
        row(name, p["version"] is not None,
            f"{p['version'] or 'not installed'}  [{p['tier']}]"
            + (f"  {p['why']}" if p["version"] is None and p["tier"] == "required" else ""))
    if "cores" not in rep:
        print("\nminimum tier: FAILED (spdee does not import)")
        return
    print("\ncores")
    for name, c in rep["cores"].items():
        row(name, c["ok"], c["desc"] + ("" if c["ok"] else f"  -- {c['why']}"))
    print("\nsolver methods")
    for name, m in rep["methods"].items():
        row(name, m["ok"], m["note"])
    print("\nfactors (shift_invert factor=)")
    for name, f in rep["factors"].items():
        row(name, f["ok"], f["note"])
    print("\ndense census (dense backend=)")
    for name, d in rep["dense"].items():
        row(name, d["ok"], d["note"])
    g = rep["gpu"]
    print("\nGPU")
    if g["vendor"]:
        print(f"  vendor       {g['vendor']}   device {g['device']} ({g['count']} visible)")
        print(f"  memory       {_fmt_bytes(g['free_bytes'])} free of "
              f"{_fmt_bytes(g['total_bytes'])}")
        print(f"  class        {g['device_class']}"
              f"{' (shares system memory)' if g['integrated'] else ''}   "
              f"auto -> {'device' if g['auto_prefers_device'] else 'host'}")
        print(f"  cupy         {g['cupy']}   runtime {g['runtime']}")
    else:
        print(f"  none         {g.get('why', '')}")
    for k, f in g["features"].items():
        row(k, f["ok"], "" if f["ok"] else f["why"])
    if g.get("auto_why"):
        print(f"  auto -> host: {g['auto_why']}")
    if g["mapped_libraries"]:
        print(f"  mapped       {', '.join(g['mapped_libraries'])}")
    t = rep["threads"]
    print("\nthreads")
    print("  " + "  ".join(f"{v}={t[v] or 'unset'}" for v in THREAD_VARS))
    print(f"  cpu_count={t['cpu_count']}  numpy BLAS={t.get('numpy_blas') or '?'}"
          + (f"  pools: {', '.join(t['pools'])}" if t.get("pools") else ""))
    print("\nenvironment switches")
    print("  " + "  ".join(f"{k}={v}" if v is not None else f"{k}=unset"
                           for k, v in rep["switches"].items()))
    if "solve" in rep:
        print("\ntest pencil (m=2 inertial modes, Rossby omega=-2/3, ARPACK + SuperLU)")
        for name, r in rep["solve"].items():
            if r.get("ok"):
                extra = f"  dense numpy err {r['dense_err']:.1e}" if "dense_err" in r else ""
                row(name, True, f"N={r['N']}  err {r['rossby_err']:.1e}{extra}")
            else:
                row(name, False, r.get("why") or f"Rossby mode off by {r.get('rossby_err')}")
    print(f"\nminimum tier: {'OK' if rep['minimum_tier_ok'] else 'FAILED'}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m spdee.check", description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    ap.add_argument("--no-solve", action="store_true", help="skip the test pencil")
    args = ap.parse_args(argv)
    rep = report(solve=not args.no_solve)
    if args.json:
        print(json.dumps(rep, indent=2, default=str))
    else:
        _print(rep)
    return 0 if rep["minimum_tier_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
