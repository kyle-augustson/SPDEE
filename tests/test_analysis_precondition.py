"""Preconditioning and convergence aids (spdee.analysis.precondition):
equilibration of the certified Bekki staggered pencil leaves the eigenvalues
unchanged and cuts the condition number of (K - sigma M) by a stated factor;
deflating a converged mode removes it from the next solve; the two-grid
preconditioner lowers bbjd's iteration count on the viscous inertial pencil
(counts with and without reported); continuation across Ek follows the
Rossby mode; the diagnosis names the dense gauge row on which bbjd stalls.
"""

import warnings

import numpy as np
import pytest

from spdee import Problem, analysis as an, solvers
from spdee.backends.fd2.grid import make_shell_grid

AVAIL = solvers.available()


def _viscous_shell(Ek=1e-3, m=2, N_r=10, N_theta=20):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_parameter("Ek", Ek)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    grid = make_shell_grid(N_r=N_r, N_theta=N_theta, r_inner=0.71, r_outer=0.985)
    return pb.compile(grid, m=m, gauge_unknowns=["p"])


SIGMA = -2j / 3 + 0.02


def test_equilibration_solar_pencil():
    """The Bekki Figure-2 hydro system on the C-grid at m = 8: entries from
    1e-5 to 1e7 across the (v, rho1, s1) blocks."""
    from test_staggered_mhd import hydro_system, compile_stag, EK, M as M_AZ
    cp = compile_stag(hydro_system(EK), 12, 24)
    K, M = cp.assemble_evp_matrices()
    sigma = 1j * (-2.0 / (M_AZ + 1)) + 0.01
    data = np.abs(K.data[K.data != 0])
    assert data.max() / data.min() > 1e10
    eq = an.equilibrate(K, M, sigma)
    blocks = [f.slice for f in cp.layout.fields]
    eq_block = an.equilibrate(K, M, sigma, row_blocks=blocks, col_blocks=blocks)
    print(f"BENCH | Bekki staggered 12x24 m=8: {eq}\nBENCH | per block: {eq_block}")
    assert eq.improvement > 1e4 and eq_block.improvement > 1e3
    ref, _ = solvers.shift_invert(K, M, sigma, 6)
    lam, V = solvers.shift_invert(eq.K, eq.M, sigma, 6, return_vectors=True)
    for x in lam:
        assert np.min(np.abs(ref - x)) < 1e-9 * abs(x)
    assert an.pencil_residual(K, M, lam, eq.vectors(V)).max() < 1e-10


def test_deflation_removes_converged_mode():
    cp = _viscous_shell()
    K, M = cp.assemble_evp_matrices()
    lam, V = solvers.shift_invert(K, M, SIGMA, 6, return_vectors=True)
    j = int(np.argmin(np.abs(lam - SIGMA)))
    defl = an.Deflation(K, M, V[:, j])
    assert not defl.null.any()
    lam2, V2 = solvers.shift_invert(K, M, SIGMA, 6, return_vectors=True, deflation=defl)
    ok = np.isfinite(lam2)
    assert np.min(np.abs(lam2[ok] - lam[j])) > 1e-3 * abs(lam[j])
    others = np.delete(lam, j)
    for x in lam2[ok][:4]:
        assert np.min(np.abs(others - x)) < 1e-8 * abs(x)
    res = an.pencil_residual(K, M, lam2[ok], np.column_stack(
        [defl.restore(l_, V2[:, i]) for i, l_ in zip(np.nonzero(ok)[0], lam2[ok])]))
    assert res[:4].max() < 1e-8


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="bbjd not built")
def test_coarse_grid_preconditioner_bbjd_counts():
    """bbjd on the viscous inertial pencil (10 x 20) for the nearest
    eigenvalue: outer iterations with the exact LU, the two-grid
    preconditioner (ILU(1e-3) smoother + 6 x 12 coarse solve), the ILU
    smoother alone, the coarse correction alone and no preconditioner."""
    import scipy.sparse.linalg as spla
    cpf, cpc = _viscous_shell(), _viscous_shell(N_r=6, N_theta=12)
    K, M = cpf.assemble_evp_matrices()
    ref, _ = solvers.shift_invert(K, M, SIGMA, 6)
    P = solvers._shifted(K, M, SIGMA)
    ilu = spla.spilu(P, drop_tol=1e-3, fill_factor=10)
    cases = {
        "exact LU": None,
        "two-grid": an.CoarseGridPreconditioner(cpf, cpc, SIGMA, drop_tol=1e-3),
        "ILU alone": ilu,
        "coarse only": an.CoarseGridPreconditioner(cpf, cpc, SIGMA, smoother=None),
        "none": lambda b: b,
    }
    counts = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for label, prec in cases.items():
            info = {}
            lam, _ = solvers.shift_invert(K, M, SIGMA, 1, method="bbjd", preconditioner=prec,
                                          maxsub=60, maxiter=300, info=info)
            ok = np.isfinite(lam)
            err = np.min(np.abs(ref - lam[ok])) if ok.any() and info["nconv"] else np.inf
            counts[label] = (info["nconv"], info["iters"], err)
    for label, (nconv, iters, err) in counts.items():
        print(f"BENCH | bbjd k=1 {label:12s}: converged {nconv}, iterations {iters}, "
              f"error {err:.1e}")
    assert counts["exact LU"][0] == 1 and counts["two-grid"][0] == 1 and counts["ILU alone"][0] == 1
    assert counts["two-grid"][2] < 1e-8 and counts["ILU alone"][2] < 1e-8
    assert counts["two-grid"][1] < counts["ILU alone"][1]
    assert counts["coarse only"][0] == 0 and counts["none"][0] == 0


def test_continuation_follows_rossby_mode_across_Ek():
    values = 1e-3 * np.array([1.0, 1.5, 2.2, 3.3, 5.0])
    path = an.continuation(lambda Ek: _viscous_shell(Ek), values, -2j / 3 + 0.005, k=6)
    print("BENCH | continuation in Ek:\n" + str(path))
    assert not any(path.lost)
    assert min(path.overlap[1:]) > 0.95
    assert max(path.residual) < 1e-8
    lam = np.array(path.lam)
    assert np.all(np.diff(lam.real) > 0)
    assert np.max(np.abs(lam.imag + 2.0 / 3.0)) < 0.05


def test_diagnose_names_the_gauge_row_and_the_band():
    cp0 = _viscous_shell(m=0)
    K0, M0 = cp0.assemble_evp_matrices()
    sigma = 0.02 + 0.3j
    lam, V = solvers.shift_invert(K0, M0, sigma, 6, return_vectors=True)
    d = an.diagnose(K0, M0, sigma, lam=lam, V=V, info={"nconv": 0, "iters": 400, "status": 1})
    causes = [f.cause for f in d.findings]
    print("BENCH | diagnosis of the m=0 gauge-row pencil:\n" + str(d))
    assert "dense constraint row" in causes
    assert d.facts["dense_constraint_rows"]
    cp2 = _viscous_shell(m=2)
    K2, M2 = cp2.assemble_evp_matrices()
    d2 = an.diagnose(K2, M2, SIGMA)
    assert "dense constraint row" not in [f.cause for f in d2.findings]


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="bbjd not built")
def test_schwarz_two_level_bbjd_counts():
    """The Schwarz smoother (4 theta patches, exact patch factors) with the
    6 x 12 coarse correction on the 10 x 20 viscous pencil: GMRES on the
    shifted system to 1e-8, then bbjd for the nearest eigenvalue with the
    single-step correction and with GMRES(20) on the correction equation,
    and the Schwarz patches alone with GMRES(20) (counts reported; from a
    random start the inner solve at a fixed tolerance need not save outer
    iterations, the pay-off is with a start vector, see ``refine``)."""
    import scipy.sparse.linalg as spla
    cpf, cpc = _viscous_shell(), _viscous_shell(N_r=6, N_theta=12)
    K, M = cpf.assemble_evp_matrices()
    ref, _ = solvers.shift_invert(K, M, SIGMA, 6)
    sw = an.SchwarzPreconditioner(cpf, SIGMA, n_theta_blocks=4)
    tl = an.CoarseGridPreconditioner(cpf, cpc, SIGMA, smoother=sw)
    P = solvers._shifted(K, M, SIGMA).tocsr()
    b = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    its = {}
    for label, prec in (("Schwarz", sw), ("two-level", tl)):
        n_it = [0]
        x, flag = spla.gmres(P, b, M=spla.LinearOperator(P.shape, matvec=prec.solve, dtype=complex),
                             rtol=1e-8, restart=100, maxiter=4,
                             callback=lambda r: n_it.__setitem__(0, n_it[0] + 1),
                             callback_type="pr_norm")
        assert flag == 0 and np.linalg.norm(P @ x - b) < 1e-6 * np.linalg.norm(b)
        its[label] = n_it[0]
    print(f"BENCH | GMRES on (K - sigma M) x = b to 1e-8: {its}")
    assert its["two-level"] < its["Schwarz"] <= 400
    cases = {"two-level, single step": (tl, {}),
             "two-level, GMRES(20)": (tl, dict(inner_maxiter=20)),
             "Schwarz alone, GMRES(20)": (sw, dict(inner_maxiter=20))}
    counts = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for label, (prec, kw) in cases.items():
            info = {}
            lam, _ = solvers.shift_invert(K, M, SIGMA, 1, method="bbjd", preconditioner=prec,
                                          maxsub=60, maxiter=300, info=info, **kw)
            ok = np.isfinite(lam)
            err = np.min(np.abs(ref - lam[ok])) if ok.any() and info["nconv"] else np.inf
            counts[label] = (info["nconv"], info["iters"], err)
    for label, (nconv, iters, err) in counts.items():
        print(f"BENCH | bbjd k=1 {label:26s}: converged {nconv}, iterations {iters}, "
              f"error {err:.1e}")
    for nconv, iters, err in counts.values():
        assert nconv == 1 and err < 1e-8
    tl.close()
    sw.close()


@pytest.mark.skipif(not AVAIL.get("bbjd"), reason="bbjd not built")
def test_refine_from_coarse_grid():
    """The Rossby mode of the 10 x 20 pencil carried to 20 x 40 by
    ``refine``: the fine eigenvalue is ARPACK's, the interpolated start
    saves iterations against a random start with the same preconditioner,
    and no fine-grid factorization is formed."""
    cpc, cpf = _viscous_shell(), _viscous_shell(N_r=20, N_theta=40)
    Kc, Mc = cpc.assemble_evp_matrices()
    lam_c, V_c = solvers.shift_invert(Kc, Mc, SIGMA, 1, return_vectors=True)
    K, M = cpf.assemble_evp_matrices()
    ref, _ = solvers.shift_invert(K, M, SIGMA, 6)
    target = ref[np.argmin(np.abs(ref - lam_c[0]))]
    r = an.refine(cpf, cpc, lam_c[0], V_c[:, 0], maxsub=60)
    sigma = lam_c[0] + 1e-2 * (1.0 + abs(lam_c[0])) * (1.0 + 1.0j) / np.sqrt(2)
    tl = an.CoarseGridPreconditioner(cpf, cpc, sigma, smoother=an.SchwarzPreconditioner(cpf, sigma))
    info = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solvers.shift_invert(K, M, sigma, 1, method="bbjd", preconditioner=tl, inner_maxiter=20,
                             maxsub=60, maxiter=300, info=info)
    print(f"BENCH | refine 10x20 -> 20x40: coarse {lam_c[0]:.6f}, fine {r.lam:.8f} "
          f"(arpack {target:.8f}), overlap {r.overlap:.4f}, residual {r.residual:.1e}, "
          f"{r.iters} iterations; random start {info['iters']} iterations")
    assert abs(r.lam - target) < 1e-8 * abs(target)
    assert r.overlap > 0.9 and r.residual < 1e-8
    assert r.iters < info["iters"]
