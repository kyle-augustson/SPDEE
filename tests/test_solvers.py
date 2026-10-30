"""The solver layer: every method x factor combination returns the same
eigenvalues of the same pencil (reference ARPACK + SuperLU, agreement to
1e-8 relative).  GPU and C-extension backends skip when unavailable.
"""

import numpy as np
import pytest

from spdee import Problem
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import CompiledProblem
from spdee import solvers


def _pencil(m=2):
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_parameter("Ek", 1e-3)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    grid = make_shell_grid(N_r=10, N_theta=20, r_inner=0.71, r_outer=0.985)
    cp = pb.compile(grid, m=m, gauge_unknowns=["p"])
    return cp.assemble_evp_matrices()


K, M = _pencil()
SIGMA = 1j * (-2.0 / 3.0) + 0.02      # near the m=2 Rossby mode, off the axis
KK = 6
AVAIL = solvers.available()


def _nearest(lam, n=4):
    """The n eigenvalues nearest SIGMA, ordered by imaginary part (the real
    parts of the inviscid modes are rounding noise)."""
    lam = np.asarray(lam)
    lam = lam[np.isfinite(lam)]
    near = lam[np.argsort(np.abs(lam - SIGMA))[:n]]
    return near[np.argsort(near.imag)]


REF, _ = solvers.shift_invert(K, M, SIGMA, KK)
REF4 = _nearest(REF)


def _check(lam, rtol=1e-8):
    got = _nearest(lam)
    assert got.shape == REF4.shape
    assert np.max(np.abs(got - REF4) / np.abs(REF4)) < rtol, (got, REF4)


def test_reference_is_sane():
    assert np.isfinite(REF4).all()
    assert np.min(np.abs(REF4 - SIGMA)) < 0.1


def _legacy_shift_invert(K_csr, M_csr, sigma, k):
    """splu + LinearOperator + eigs, written out."""
    import scipy.sparse.linalg as spla
    LU = spla.splu((K_csr - sigma * M_csr).tocsc())
    LinOp = spla.LinearOperator(K_csr.shape, matvec=lambda x: LU.solve(M_csr @ x), dtype=complex)
    mu = spla.eigs(LinOp, k=k, which="LM", return_eigenvectors=False)
    keep = np.abs(mu) > 1e-12
    return np.where(keep, 1.0 / np.where(keep, mu, 1.0) + sigma, np.nan + 0j)


def test_arpack_scipy_matches_legacy_path():
    lam = CompiledProblem.solve_evp_from_matrices(K, M, SIGMA, k=KK)
    _check(lam, rtol=1e-12)
    legacy = _legacy_shift_invert(K, M, SIGMA, KK)
    assert np.allclose(_nearest(legacy), REF4, rtol=1e-10)


def test_arpack_lapack():
    """The dense host factorization solves the same shifted pencil."""
    lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="lapack")
    _check(lam, rtol=1e-10)
    fac = solvers.factorize(solvers._shifted(K, M, SIGMA), "lapack")
    b = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    P = (K - SIGMA * M).tocsr()
    assert np.linalg.norm(P @ fac.solve(b) - b) < 1e-10 * np.linalg.norm(b)
    assert np.linalg.norm(P.conj().T @ fac.solve_H(b) - b) < 1e-10 * np.linalg.norm(b)


@pytest.mark.skipif(not AVAIL.get("fplus"), reason="no FPLUS build (SPDEE_FPLUS_PATH)")
@pytest.mark.parametrize("pivot", ["partial", "rook", "full"])
def test_arpack_fplus(pivot):
    lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="fplus",
                                  factor_options=dict(pivot_strategy=pivot))
    _check(lam)


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="no BBJD build (SPDEE_BBJD_PATH)")
@pytest.mark.parametrize("factor", ["scipy", "fplus"])
def test_bbjd(factor):
    """bbjd on the m=2 pencil: singular M (continuity and wall rows) but no
    dense constraint row."""
    if factor == "fplus" and not AVAIL.get("fplus"):
        pytest.skip("FPLUS not built")
    lam, V = solvers.shift_invert(K, M, SIGMA, KK, method="bbjd", factor=factor,
                                  return_vectors=True, maxsub=60, maxiter=400)
    _check(lam, rtol=1e-7)
    # residual of the returned pairs in the pencil
    for j in range(len(lam)):
        x = V[:, j]
        r = np.linalg.norm(K @ x - lam[j] * (M @ x)) / (np.linalg.norm(K @ x) + 1e-300)
        assert r < 1e-7


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="no BBJD build (SPDEE_BBJD_PATH)")
def test_bbjd_gauge_row():
    """bbjd on the m=0 pencil: singular M and the dense integral gauge row,
    with sigma in a gap of the spectrum (the nearest eigenvalue 0.27 away),
    the case whose Galerkin extraction produced spurious Ritz values."""
    K0, M0 = _pencil(m=0)
    sigma = 0.02 + 0.3j
    ref, _ = solvers.shift_invert(K0, M0, sigma, KK)
    lam, _ = solvers.shift_invert(K0, M0, sigma, KK, method="bbjd", maxsub=60, maxiter=400)
    ref, lam = ref[np.isfinite(ref)], lam[np.isfinite(lam)]
    assert len(lam) == KK
    for x in lam:
        # one of the six is numerically zero, so the tolerance is on the scale of the shift
        assert np.min(np.abs(x - ref)) < 1e-7 * max(abs(x), abs(sigma))


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="no BBJD build (SPDEE_BBJD_PATH)")
def test_bbjd_start_vector():
    """ARPACK's eigenvector as v0: bbjd accepts the pair at once."""
    ref, V = solvers.shift_invert(K, M, SIGMA, KK, return_vectors=True)
    j = int(np.argmin(np.abs(ref - SIGMA)))
    info = {}
    lam, _ = solvers.shift_invert(K, M, SIGMA, 1, method="bbjd", v0=V[:, j], info=info)
    assert info["nconv"] == 1 and info["iters"] <= 2
    assert abs(lam[0] - ref[j]) < 1e-10 * abs(ref[j])


def test_standard_problem_path():
    """M=None must mean M=I: compare against the explicit identity."""
    import scipy.sparse as sps
    I = sps.identity(K.shape[0], dtype=complex, format="csr")
    a, _ = solvers.shift_invert(K, None, SIGMA, KK)
    b, _ = solvers.shift_invert(K, I, SIGMA, KK)
    assert np.allclose(_nearest(a), _nearest(b), rtol=1e-10)


def test_dense_numpy():
    res = solvers.dense(K, M, sigma=SIGMA, backend="numpy")
    _check(res.lam, rtol=1e-8)
    assert res.resid is not None and np.max(res.resid[np.argsort(np.abs(res.lam - SIGMA))[:4]]) < 1e-8


@pytest.mark.skipif(not AVAIL.get("cupy"), reason="no CuPy device")
def test_factor_cupy():
    lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="cupy")
    _check(lam)


@pytest.mark.skipif(not AVAIL.get("cudss"), reason="no cuDSS")
def test_factor_cudss():
    """cuDSS with matching solves this pencil; without it the residual gate
    must reject the factorization and 'auto' fall through."""
    lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="cudss")
    _check(lam)
    import os
    os.environ["CUDSS_USE_MATCHING"] = "0"
    try:
        with pytest.raises(RuntimeError, match="residual gate"):
            solvers.shift_invert(K, M, SIGMA, KK, factor="cudss")
        lam, _ = solvers.shift_invert(K, M, SIGMA, KK, factor="auto")
        _check(lam)
    finally:
        del os.environ["CUDSS_USE_MATCHING"]


@pytest.mark.skipif(not AVAIL.get("cupy"), reason="no CuPy device")
def test_dense_cupy():
    res = solvers.dense(K, M, sigma=SIGMA, backend="cupy")
    _check(res.lam, rtol=1e-6)


def test_solve_evp_method_kwargs():
    """CompiledProblem.solve_evp forwards method/factor."""
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = make_shell_grid(N_r=8, N_theta=16, r_inner=0.71, r_outer=0.985)
    cp = pb.compile(grid, m=2, gauge_unknowns=["p"])
    v0 = np.random.default_rng(0).standard_normal(cp.matrices.A_total.shape[0]).astype(complex)
    a = cp.solve_evp(k=4, sigma=SIGMA, v0=v0)
    kw = dict(factor="fplus", factor_options=dict(pivot_strategy="rook")) if AVAIL.get("fplus") else {}
    b = cp.solve_evp(k=4, sigma=SIGMA, v0=v0, **kw)
    assert np.allclose(_nearest(a.eigenvalues, 2), _nearest(b.eigenvalues, 2), rtol=1e-6)


def test_available_report():
    rep = solvers.available()
    assert rep["arpack"] and rep["scipy"] and rep["dense_numpy"]
