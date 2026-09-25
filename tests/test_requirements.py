"""The requirement tiers and the vendor-neutral GPU layer.

The minimum tier (numpy, scipy, sympy) must give a working solver whatever
else is missing: with the GPU switched off, with CuPy absent, with Dedalus
absent, and on CuPy's ROCm build.  Absence is simulated two ways: an
environment switch or a runtime patch in this process, and a meta-path
finder that hides a package from a fresh interpreter (the GPU stack is
already imported here, so only a subprocess can prove that nothing
hard-requires it).
"""
import dataclasses
import os
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from spdee import Problem, solvers
from spdee.backends.fd2.grid import make_shell_grid
from spdee.solvers import gpu

REPO_ROOT = Path(__file__).resolve().parent.parent



def _fallbacks(rec):
    """The GPUFallbackWarning records only: an integrated device adds a
    GPUPerformanceWarning to the same block."""
    return [w for w in rec if issubclass(w.category, gpu.GPUFallbackWarning)]

def _pencil():
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.71, r_outer=0.985)
    return pb.compile(grid, m=2, gauge_unknowns=["p"]).assemble_evp_matrices()


K, M = _pencil()
SIGMA = 1j * (-2.0 / 3.0) + 0.02
KK = 6


def _nearest(lam, n=4):
    lam = np.asarray(lam)
    lam = lam[np.isfinite(lam)]
    near = lam[np.argsort(np.abs(lam - SIGMA))[:n]]
    return near[np.argsort(near.imag)]


REF_SI = _nearest(solvers.shift_invert(K, M, SIGMA, KK, factor="scipy")[0])
REF_DENSE = _nearest(solvers.dense(K, M, sigma=SIGMA, backend="numpy").lam)


def _run(script: str, env_extra=None) -> subprocess.CompletedProcess:
    """A fresh interpreter on the repository with the given script."""
    env = dict(os.environ)
    env.pop("SPDEE_NO_GPU", None)          # absence is what these scripts test
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, "-c", textwrap.dedent(script)], cwd=REPO_ROOT,
                          env=env, capture_output=True, text=True, timeout=600)


def _ok(proc: subprocess.CompletedProcess):
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


HIDE = '''
    import sys
    HIDDEN = {names}

    class _Hide:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in HIDDEN:
                raise ImportError(f"{{name}} hidden by the test")
            return None

    sys.meta_path.insert(0, _Hide())
'''


@pytest.fixture
def probe_reset():
    gpu.reset()
    yield
    gpu.reset()


# ---------------------------------------------------------------- (a) no GPU
def test_no_gpu_switch(monkeypatch, probe_reset):
    """SPDEE_NO_GPU=1: no GPU reported, every 'auto' choice on the CPU,
    same eigenvalues, no warning (nothing was asked for that is missing)."""
    monkeypatch.setenv("SPDEE_NO_GPU", "1")
    gpu.reset()
    rep = solvers.available()
    assert rep["gpu_vendor"] is None and rep["gpu_device"] is None
    assert not rep["cupy"] and not rep["cudss"] and not rep["dense_cupy"]
    assert not rep["gpu_dense_lu"] and not rep["gpu_geev"]
    assert rep["arpack"] and rep["scipy"] and rep["dense_numpy"]
    assert all(not ok and "SPDEE_NO_GPU" in why for ok, why in gpu.features().values())
    assert solvers.gpu_factor_order() == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error", gpu.GPUFallbackWarning)
        res = solvers.dense(K, M, sigma=SIGMA, backend="auto")
        lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="auto")
    assert res.backend == "numpy"
    assert np.allclose(_nearest(res.lam), REF_DENSE, rtol=1e-10)
    assert np.allclose(_nearest(lam), REF_SI, rtol=1e-10)


def test_no_gpu_explicit_requests_degrade_with_one_warning(monkeypatch, probe_reset):
    """An explicit GPU factor or backend without a GPU runs on the host
    after exactly one warning naming the request and the fallback."""
    monkeypatch.setenv("SPDEE_NO_GPU", "1")
    gpu.reset()
    for factor in ("cudss", "cupy"):
        with pytest.warns(gpu.GPUFallbackWarning, match=f"factor={factor!r}.*scipy") as rec:
            lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor=factor)
        assert len(_fallbacks(rec)) == 1
        assert np.allclose(_nearest(lam), REF_SI, rtol=1e-10)
    with pytest.warns(gpu.GPUFallbackWarning, match="backend='cupy'.*host") as rec:
        res = solvers.dense(K, M, sigma=SIGMA, backend="cupy")
    assert len(_fallbacks(rec)) == 1 and res.backend == "numpy"
    assert np.allclose(_nearest(res.lam), REF_DENSE, rtol=1e-10)
    from spdee.solvers.gpu_dense import managed_memory
    with pytest.warns(gpu.GPUFallbackWarning, match="managed memory"):
        assert managed_memory() is False


# ----------------------------------------------------------- (b) no CuPy stack
def test_import_tree_without_cupy():
    """With cupy, cupyx and nvmath unimportable every spdee module imports,
    the probe reports no GPU and the CPU paths solve the pencil."""
    script = HIDE.format(names='{"cupy", "cupyx", "cupy_backends", "nvmath"}') + '''
    import importlib, pkgutil, warnings
    import numpy as np
    import spdee
    skipped = []
    for info in pkgutil.walk_packages(spdee.__path__, "spdee."):
        leaf = info.name.rsplit(".", 1)[-1]
        if ".tests" in info.name or leaf.startswith("test_") or leaf == "setup":
            skipped.append(info.name)
            continue
        importlib.import_module(info.name)
    from spdee import solvers
    from spdee.solvers import gpu
    rep = solvers.available()
    assert rep["gpu_vendor"] is None and not rep["cupy"] and not rep["cudss"], rep
    assert all("CuPy not installed" in why for _, why in gpu.features().values())
    from spdee import Problem
    from spdee.backends.fd2.grid import make_shell_grid
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.71, r_outer=0.985)
    K, M = pb.compile(grid, m=2, gauge_unknowns=["p"]).assemble_evp_matrices()
    sigma = 1j * (-2.0 / 3.0) + 0.02
    with warnings.catch_warnings():
        warnings.simplefilter("error", gpu.GPUFallbackWarning)
        lam, _ = solvers.shift_invert(K, M, sigma, 6, factor="auto")
        res = solvers.dense(K, M, sigma=sigma, backend="auto")
    assert res.backend == "numpy"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res2 = solvers.dense(K, M, sigma=sigma, backend="cupy")
        lam2, _ = solvers.shift_invert(K, M, sigma, 6, factor="cudss")
    kinds = [w.category for w in rec]
    assert kinds.count(gpu.GPUFallbackWarning) == 2, [str(w.message) for w in rec]
    ref = np.sort(res.lam[np.argsort(np.abs(res.lam - sigma))[:4]])
    got = np.sort(res2.lam[np.argsort(np.abs(res2.lam - sigma))[:4]])
    assert np.allclose(ref, got)
    assert "cupy" not in sys.modules and "nvmath" not in sys.modules
    print("imported without cupy; skipped", skipped)
    '''
    _ok(_run(script))


# ------------------------------------------------------------- (c) no Dedalus
def test_import_tree_without_dedalus():
    """With dedalus unimportable spdee imports, the three host cores compile
    and the dedalus backend raises one ImportError naming the tier."""
    script = HIDE.format(names='{"dedalus"}') + '''
    import importlib, pkgutil
    import spdee
    for info in pkgutil.walk_packages(spdee.__path__, "spdee."):
        leaf = info.name.rsplit(".", 1)[-1]
        if ".tests" in info.name or leaf.startswith("test_") or leaf == "setup":
            continue
        importlib.import_module(info.name)
    from spdee import Problem, compile_problem
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.sh_fd import ShFdGrid
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.71, r_outer=0.985)
    for kw in ({}, {"discretization": "staggered"}):
        K, M = pb.compile(grid, m=2, gauge_unknowns=["p"], **kw).assemble_evp_matrices()
        assert K.shape[0] > 0
    sh = ShFdGrid(N_r=8, ell_max=8, r_inner=0.71, r_outer=0.985)
    K, M = pb.compile(sh, m=2, backend="sh_fd", gauge_unknowns=["p"]).assemble_evp_matrices()
    assert K.shape[0] > 0
    assert "dedalus" in spdee.available_backends()
    try:
        pb.compile(grid, m=2, backend="dedalus", gauge_unknowns=["p"])
    except ImportError as exc:
        msg = str(exc)
        assert "Dedalus v3" in msg and "docs/requirements.md" in msg, msg
    else:
        raise AssertionError("dedalus backend compiled without Dedalus")
    assert "dedalus" not in sys.modules
    print("ok")
    '''
    _ok(_run(script))


# -------------------------------------------------------------- (d) the check
def test_check_entry_point():
    proc = subprocess.run([sys.executable, "-m", "spdee.check"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    for line in ("SPDEE capability check", "cores", "factors", "dense census", "GPU",
                 "threads", "environment switches", "minimum tier: OK"):
        assert line in out, out
    proc = subprocess.run([sys.executable, "-m", "spdee.check", "--json", "--no-solve"],
                          cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
                          env={**os.environ, "PYTHONPATH": str(REPO_ROOT), "SPDEE_NO_GPU": "1"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    import json
    rep = json.loads(proc.stdout)
    assert rep["minimum_tier_ok"] and rep["gpu"]["vendor"] is None
    assert rep["factors"]["auto"]["note"] == "-> scipy"
    assert rep["dense"]["auto"]["note"] == "-> numpy"


def test_check_report_in_process():
    from spdee import check
    rep = check.report(solve=False)
    assert rep["minimum_tier_ok"]
    assert set(rep["cores"]) == {"collocated", "staggered", "sh_fd", "dedalus"}
    assert set(rep["factors"]) == set(solvers.FACTORS)
    assert set(rep["gpu"]["features"]) == set(gpu.features())


# --------------------------------------------------------- (e) ROCm simulation
@pytest.mark.skipif(not solvers.available().get("cupy"), reason="no CuPy device")
def test_rocm_simulation(monkeypatch, probe_reset):
    """CuPy's ROCm build reports is_hip; the probe then says 'rocm', marks
    Xgeev, cp.linalg.eig and cuDSS unavailable with the ROCm reason, keeps
    managed memory, and on this discrete device 'auto' takes the dense LU
    on the device with the geev on the host silently while the sparse
    factor stays SuperLU; an explicit census does the same with one warning,
    never loading libcusolver.  The hardware itself is CUDA here, so this
    exercises the branch, not rocSOLVER."""
    import cupy as cp
    from spdee.solvers import gpu_dense
    # The flag is patched only while the probe runs and caches its answer:
    # CuPy's own kernel compiler consults it too and would call hipcc.
    with monkeypatch.context() as mp:
        mp.setattr(cp.cuda.runtime, "is_hip", True)
        gpu.reset()
        assert gpu.vendor() == "rocm"
        feat = gpu.features()
    assert gpu.vendor() == "rocm"
    assert feat["dense_lu"][0] and feat["sparse_lu"][0] and feat["managed_memory"][0]
    for key in ("xgeev", "cupy_geev", "cudss"):
        ok, why = feat[key]
        assert not ok and why.startswith("ROCm"), (key, ok, why)
    rep = solvers.available()
    assert rep["gpu_vendor"] == "rocm" and rep["cupy"] and rep["dense_cupy"]
    assert not rep["cudss"] and not rep["gpu_geev"]
    if gpu.device().klass == "discrete":
        assert gpu.prefer_device() == (True, "")
    else:
        assert gpu.prefer_device()[0] is False
    assert solvers.gpu_factor_order() == ()

    def _never():
        raise AssertionError("libcusolver must not be touched on ROCm")

    monkeypatch.setattr(gpu_dense, "_load_libcusolver", _never)
    with warnings.catch_warnings():
        warnings.simplefilter("error", gpu.GPUFallbackWarning)
        auto = solvers.dense(K, M, sigma=SIGMA, backend="auto")
        lam_auto, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="auto")
    assert auto.backend == ("cupy+hosteig" if gpu.device().klass == "discrete" else "numpy")
    assert np.allclose(_nearest(auto.lam), REF_DENSE, rtol=1e-6)
    assert np.allclose(_nearest(lam_auto), REF_SI, rtol=1e-10)
    with pytest.warns(gpu.GPUFallbackWarning, match="ROCm.*host LAPACK") as rec:
        res = solvers.dense(K, M, sigma=SIGMA, backend="cupy")
    assert len(_fallbacks(rec)) == 1
    assert res.backend == "cupy+hosteig"
    # rocSOLVER's blocked getrf pivots differently from LAPACK, so only
    # invariants are compared: eigenvalues here, residuals below
    assert np.allclose(_nearest(res.lam), REF_DENSE, rtol=1e-6)
    with pytest.warns(gpu.GPUFallbackWarning, match="factor='cudss'.*ROCm") as rec:
        lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="cudss")
    assert len(_fallbacks(rec)) == 1
    assert np.allclose(_nearest(lam), REF_SI, rtol=1e-10)


@pytest.mark.skipif(not solvers.available().get("cupy"), reason="no CuPy device")
def test_geev_that_raises_reaches_host(monkeypatch, probe_reset):
    """On ROCm ``cp.linalg.eig`` exists and raises 'geev is not available';
    the probe's trial call reports that, and even when a probe says yes
    the census chain catches the raise and lands on host LAPACK with one
    warning."""
    import cupy as cp
    from spdee.solvers import gpu_dense

    def _raise(*a, **k):
        raise RuntimeError("geev is not available")

    monkeypatch.setattr(cp.linalg, "eig", _raise)
    monkeypatch.setattr(cp.linalg, "eigvals", _raise)
    gpu.reset()
    ok, why = gpu.features()["cupy_geev"]
    assert not ok and "geev is not available" in why
    # the probe still believes in cp.linalg.eig: the chain must survive the raise
    feat = dict(gpu.features())
    feat["cupy_geev"] = (True, "")
    feat["xgeev"] = (True, "")
    monkeypatch.setitem(gpu._CACHE, "features", feat)

    def _no_xgeev(C, want_vectors=True):
        raise gpu_dense.XgeevUnavailable("simulated: toolkit predates Xgeev")

    monkeypatch.setattr(gpu_dense, "_xgeev_device", _no_xgeev)
    with pytest.warns(gpu.GPUFallbackWarning, match="geev is not available.*host LAPACK") as rec:
        res = solvers.dense(K, M, sigma=SIGMA, backend="cupy")
    assert len(_fallbacks(rec)) == 1
    assert res.backend == "cupy+hosteig"
    assert np.allclose(_nearest(res.lam), REF_DENSE, rtol=1e-6)


def test_probe_reports_this_machine(probe_reset):
    """Whatever the hardware, the report is self-consistent."""
    rep = solvers.available()
    feat = gpu.features()
    dev = gpu.device()
    if dev is None:
        assert rep["gpu_vendor"] is None and not rep["cupy"]
        assert all(not ok and why for ok, why in feat.values())
    else:
        assert rep["gpu_vendor"] == dev.vendor in ("cuda", "rocm")
        assert rep["gpu_device"] == dev.name and dev.total_bytes > 0
        assert rep["dense_cupy"] == feat["dense_lu"][0]
        assert rep["cudss"] == feat["cudss"][0]
        assert rep["gpu_geev"] == (feat["xgeev"][0] or feat["cupy_geev"][0])
    for key in ("cupy", "cudss", "dense_cupy", "gpu_dense_lu", "gpu_geev"):
        assert isinstance(rep[key], bool)


@pytest.mark.skipif(not solvers.available().get("cupy"), reason="no CuPy device")
def test_integrated_device_auto_prefers_host(monkeypatch, probe_reset):
    """On an integrated GPU (shares system memory, slower than the host
    BLAS) 'auto' stays on the host without a warning whatever the vendor;
    an explicit request still uses the device, with a performance warning."""
    dev = gpu.device()
    gpu._CACHE["device"] = (dataclasses.replace(dev, integrated=True), "")
    assert gpu.device().klass == "integrated"
    ok, why = gpu.prefer_device()
    assert not ok and "integrated" in why
    assert gpu.vendor() == dev.vendor and gpu.features()["dense_lu"][0]
    assert solvers.gpu_factor_order() == ()
    with warnings.catch_warnings():
        warnings.simplefilter("error", gpu.GPUFallbackWarning)
        warnings.simplefilter("error", gpu.GPUPerformanceWarning)
        res = solvers.dense(K, M, sigma=SIGMA, backend="auto")
        lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="auto")
    with pytest.warns(gpu.GPUPerformanceWarning, match="integrated"):
        explicit = solvers.dense(K, M, sigma=SIGMA, backend="cupy")
    assert res.backend == "numpy"
    assert explicit.backend.startswith("cupy")
    assert np.allclose(_nearest(lam), REF_SI, rtol=1e-10)
    assert np.allclose(_nearest(explicit.lam), REF_DENSE, rtol=1e-6)


@pytest.mark.skipif(not solvers.available().get("dense_cupy"), reason="no device dense LU")
def test_device_dense_lu_by_invariants():
    """The device LU behind dense(backend='cupy') (cuSOLVER or rocSOLVER
    through cupyx) is checked by what every valid LU satisfies: P L U
    reconstructs the matrix and the solve residual against the pencil is
    small.  Pivot sequences and packed factors are not compared with
    scipy's: a blocked getrf may pivot differently and still be right."""
    import cupy as cp
    import cupyx.scipy.linalg as cpla
    import scipy.linalg as sla
    P = (K - SIGMA * M).toarray()
    n = P.shape[0]
    Pd = cp.asfortranarray(cp.asarray(P))
    lu, piv = cpla.lu_factor(Pd, check_finite=False)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x = cp.asnumpy(cpla.lu_solve((lu, piv), cp.asarray(b), check_finite=False))
    assert np.linalg.norm(P @ x - b) <= 1e-10 * np.linalg.norm(b) * n
    x_host = sla.lu_solve(sla.lu_factor(P, check_finite=False), b, check_finite=False)
    assert np.allclose(x, x_host, rtol=1e-8, atol=1e-10 * np.abs(x_host).max())
    lu_h, piv_h = cp.asnumpy(lu), cp.asnumpy(piv).astype(np.int64)
    assert piv_h.dtype.kind == "i" and piv_h.min() >= 0 and piv_h.max() < n
    L = np.tril(lu_h, -1) + np.eye(n)
    U = np.triu(lu_h)
    PA = P.copy()
    for i, p in enumerate(piv_h):
        if p != i:
            PA[[i, p]] = PA[[p, i]]
    assert np.linalg.norm(L @ U - PA) <= 1e-12 * np.linalg.norm(P) * n
