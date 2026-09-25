"""Mode classification (spdee.analysis.modes): the diagnostics on eigenvectors
of known structure on the sh_fd core (exact harmonics), and ``identify`` on
the staggered core, whose inviscid inertial band is dense enough that the
nearest eigenvalue to a target is a band mode
(docs/analytic_benchmarks.md): the sectoral r-mode is recovered by its
toroidal energy and a Zhang mode by its pressure polynomial where the
nearest eigenvalue is not the mode, with the discrimination ratio of
order one reported.
"""

import numpy as np
import pytest

from spdee import Problem, analysis as an, solvers
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.sh_fd import ShFdGrid
from test_evp_zhang_2001 import _build_system, _zhang_pressure_basis, _polynomial_residual


def _inertial_problem(walls):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in walls:
        pb.add_bc("dot(r_hat, v) = 0", at=w)
    return pb


def test_sh_fd_features_of_synthetic_vectors():
    """A pure toroidal l = 3, m = 2 coefficient vector is 100 % toroidal,
    l = 3 and an equatorially symmetric flow (its potential is odd in z,
    (-1)^(l + m) = -1, so v_theta is odd and v_phi even); a pure v_r of the
    same l is poloidal and antisymmetric; the parity of a scalar is
    (-1)^(l + m)."""
    pb = _inertial_problem(("inner", "outer"))
    grid = ShFdGrid(N_r=16, ell_max=8, r_inner=0.5, r_outer=1.0)
    cp = pb.compile(grid, m=2, backend="sh_fd")
    x = np.zeros(cp.N, dtype=complex)
    name, kind, off, size = cp.layout[0]
    assert name == "v"
    il = list(cp.ls).index(3)
    prof = cp.r * (1 - cp.r)
    x[off + 0 * cp.nlr + il * cp.Nr: off + 0 * cp.nlr + (il + 1) * cp.Nr] = prof
    x[off + 1 * cp.nlr + il * cp.Nr: off + 1 * cp.nlr + (il + 1) * cp.Nr] = prof
    gf = an.to_grid_fields(cp, x)
    tp = an.toroidal_poloidal(gf, "v")
    assert tp["toroidal"] > 1 - 1e-10
    ls, fr = an.l_content(gf, "v")
    assert ls[np.argmax(fr)] == 3 and fr.max() > 1 - 1e-10
    assert an.parity(gf, "v") > 1 - 1e-10
    e = an.energy_fractions(gf)
    assert abs(e["v"]["r"]) < 1e-12 and abs(e["v"]["horizontal"] - 1) < 1e-12
    x[:] = 0
    x[off + 2 * cp.nlr + il * cp.Nr: off + 2 * cp.nlr + (il + 1) * cp.Nr] = prof
    gf = an.to_grid_fields(cp, x)
    assert an.toroidal_poloidal(gf, "v")["poloidal"] > 1 - 1e-10
    assert an.parity(gf, "v") < -1 + 1e-10
    loc = an.localisation(gf, "v")
    rr = np.linspace(0.5, 1.0, 20001)
    dens = rr ** 2 * (rr * (1 - rr)) ** 2
    assert abs(loc["r_centroid"] - np.trapezoid(rr * dens, rr) / np.trapezoid(dens, rr)) < 2e-3
    x[:] = 0
    _, _, offp, _ = cp.layout[1]
    il4 = list(cp.ls).index(4)
    x[offp + il4 * cp.Nr: offp + (il4 + 1) * cp.Nr] = prof
    gf = an.to_grid_fields(cp, x)
    assert an.parity(gf, "p") > 1 - 1e-10


def test_collocated_grid_fields_and_residuals():
    pb = _inertial_problem(("outer",))
    cp = pb.compile(make_ball_grid(N_r=12, N_theta=18), m=2)
    K, M = cp.assemble_evp_matrices()
    lam, V = solvers.shift_invert(K, M, -2j / 3 + 0.01, 4, return_vectors=True)
    j = int(np.argmin(np.abs(lam + 2j / 3)))
    f = an.features(cp, lam[j], V[:, j], K=K, M=M)
    assert f.residual < 1e-10
    assert f.toroidal > 0.95 and f.l_dominant == 2 and f.parity < -0.9
    cres = an.constraint_residual(cp, V[:, j], K, M)
    assert cres["continuity"] < 1e-10
    W = an.natural_weight(cp, M)
    massless = np.asarray(abs(M).sum(axis=1)).ravel() == 0
    assert W.support.sum() == np.sum(~massless)
    assert not W.support[cp.matrices.layout.field_slice("p")].any()


def _staggered_shell(N, m):
    pb = _inertial_problem(("inner", "outer"))
    grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    return pb.compile(grid, m=m, discretization="staggered", order_r=2, order_theta=2)


@pytest.mark.parametrize("m", [1, 2])
def test_identify_rmode_by_toroidal_energy(m):
    """The sectoral r-mode omega = -2/(m + 1) from a guess of its frequency
    1e-2 off: the nearest eigenvalue is a band mode with 20-30 % poloidal
    energy, the identified one is toroidal to 1e-3 (the face averaging of the
    staggered v_r) and within 1e-3 of the
    exact frequency."""
    cp = _staggered_shell(24, m)
    K, M = cp.assemble_evp_matrices()
    exact = -2.0 / (m + 1)
    guess = exact + 0.01
    lam, V = solvers.shift_invert(K, M, 1j * guess + 1e-4, 40, return_vectors=True, tol=1e-12,
                                  maxiter=8000)
    target = an.ModeTarget(omega=guess, window=0.05, toroidal_min=0.99)
    idn = an.identify(cp, lam, V, target, K=K, M=M)
    near = an.features(cp, lam[idn.nearest_index], V[:, idn.nearest_index])
    print(f"BENCH | staggered shell m={m}: identified omega {idn.omega.real:+.6f} "
          f"(exact {exact:+.6f}, toroidal {idn.features.toroidal:.6f}), nearest to the guess "
          f"{guess:+.2f}: omega {near.omega.real:+.6f} toroidal {near.toroidal:.3f}; "
          f"discrimination {idn.discrimination:.2f}")
    assert idn.features.toroidal > 1 - 1e-3
    assert abs(idn.omega.real - exact) < 1e-3 * abs(exact)
    assert idn.features.residual < 1e-10
    assert idn.differs_from_nearest and near.toroidal < 0.95


def test_identify_zhang_mode_by_pressure_polynomial():
    """Zhang's m = 1, N = 1 ANTI mode omega = 0.82001 on the staggered ball: the
    nearest eigenvalue has a pressure orthogonal to the polynomial space
    (misfit ~ 1) while the identified mode fits it to 3e-3, converges at
    the scheme's order between 16x24 and 24x40, and the discrimination
    ratio is below one: the band spacing is finer than the error."""
    sys_ = _build_system()
    from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
    omega_zhang, parity, n_class = -0.82001, "ANTI", 1
    omega_t = -omega_zhang
    errs, discs = [], []
    for N_r, N_theta in ((16, 24), (24, 40)):
        grid = make_ball_grid(N_r=N_r + 1, N_theta=N_theta)
        cp = compile_fd2_staggered(sys_, grid, m=1, order_r=2, order_theta=2)
        K, M = cp.assemble_evp_matrices()
        R, T = np.meshgrid(cp.grids.r_center, cp.grids.theta_center, indexing="ij")
        weight = (R ** 2 * np.sin(T)).ravel() + 1e-3
        basis = _zhang_pressure_basis(R, T, 1, n_class, parity)
        fp = cp.layout.field("p")

        def score(f, x, fp=fp, basis=basis, weight=weight, shape=R.shape):
            return _polynomial_residual(x[fp.slice].reshape(shape), basis, weight)
        lam, V = solvers.shift_invert(K, M, 1j * omega_t - 1e-3, 30, return_vectors=True)
        idn = an.identify(cp, lam, V, an.ModeTarget(omega=omega_t, window=0.1, parity=parity,
                                                    score=score), K=K, M=M)
        near_misfit = score(None, V[:, idn.nearest_index])
        errs.append(abs(idn.omega.real - omega_t))
        discs.append(idn.discrimination)
        print(f"BENCH | staggered ball m=1 {N_r}x{N_theta} Zhang N=1 ANTI: identified omega "
              f"{idn.omega.real:+.5f} misfit {idn.misfit:.1e} err {errs[-1]:.1e}; nearest omega "
              f"{lam[idn.nearest_index].imag:+.5f} misfit {near_misfit:.2f}; discrimination "
              f"{idn.discrimination:.2f} (control spacing {idn.control_spacing:.1e})")
        assert idn.differs_from_nearest
        assert idn.misfit < 5e-2 and near_misfit > 0.5
    assert errs[1] < 0.6 * errs[0]
    assert discs[1] < 1.0
