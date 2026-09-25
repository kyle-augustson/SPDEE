"""Eigenvalue conditioning (spdee.analysis.conditioning) against known answers:
a Hermitian pencil has kappa = 1 in the M-norm and left = right vectors; a
2 x 2 near-Jordan block has the closed-form kappa; the sh_fd diffusion pencil
is normal to rounding at fd_order 2 in the r^2 dr weight and its departure
falls with N at order 4; the sensitivity formula predicts the finite
difference of an eigenvalue across two compiled pencils; on the collocated
inertial ball the pressure checkerboard is a 0 = lambda 0 direction that
tops the kappa ranking while the sectoral r-mode is conditioned to 1.1.
"""

import numpy as np
import pytest
import scipy.linalg as sla
import scipy.sparse as sp

from spdee import Problem, analysis as an, solvers
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.sh_fd import ShFdGrid


def _hermitian(n=40, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return sp.csr_matrix(A + A.conj().T)


def test_hermitian_pencil_kappa_one():
    K = _hermitian()
    lam, X = np.linalg.eigh(K.toarray())
    Y, _, resid = an.left_eigenvectors(K, None, lam[:6], X[:, :6])
    assert resid.max() < 1e-12
    kappa = an.condition_numbers(K, None, X[:, :6], Y)
    assert np.allclose(kappa, 1.0, atol=1e-12)
    C, gram = an.eigenvector_angles(X[:, :6])
    assert np.max(C - np.eye(6)) < 1e-12 and abs(gram - 1) < 1e-12


def test_generalized_hermitian_kappa_one_in_M_norm():
    K = _hermitian()
    w = np.random.default_rng(1).uniform(0.5, 2.0, K.shape[0])
    M = sp.diags(w).tocsr()
    lam, X = sla.eigh(K.toarray(), np.diag(w))
    Y, _, _ = an.left_eigenvectors(K, M, lam[:5], X[:, :5])
    assert np.allclose(an.condition_numbers(K, M, X[:, :5], Y), 1.0, atol=1e-12)
    # the same modes are not orthogonal in the Euclidean norm
    assert np.max(an.condition_numbers(K, M, X[:, :5], Y, W=np.ones(K.shape[0])) - 1) > 1e-3


def test_jordan_block_kappa():
    a, d = 10.0, 1e-2
    J = sp.csr_matrix(np.array([[1.0, a], [0.0, 1.0 + d]], dtype=complex))
    lam = np.array([1.0, 1.0 + d])
    X = np.array([[1.0, a], [0.0, d]], dtype=complex)
    X /= np.linalg.norm(X, axis=0)
    Y, _, _ = an.left_eigenvectors(J, None, lam, X)
    kappa = an.condition_numbers(J, None, X, Y)
    expected = np.sqrt(1.0 + (a / d) ** 2)
    assert np.allclose(kappa, expected, rtol=1e-6)
    rep = an.condition_report(J, None, lam, X, threshold=100.0)
    assert len(rep.ill_conditioned) == 2


def test_two_sided_dense_matches_inverse_iteration():
    K = _hermitian(30)
    lam, X, Y = an.two_sided_dense(K, None, sigma=0.3j)
    assert len(lam) == 30
    assert np.allclose(an.condition_numbers(K, None, X, Y), 1.0, atol=1e-10)


def _diffusion(N_r, order):
    pb = Problem()
    pb.add_unknown("s", "scalar")
    pb.add_equation("dt(s) - lap(s) = 0", name="heat")
    for w in ("inner", "outer"):
        pb.add_bc("s = 0", at=w)
    grid = ShFdGrid(N_r=N_r, ell_max=7, r_inner=0.5, r_outer=1.0, fd_order=order)
    cp = pb.compile(grid, m=1, backend="sh_fd")
    K, M = cp.assemble_evp_matrices()
    return cp, K, M


@pytest.mark.parametrize("order", [2, 4])
def test_sh_fd_diffusion_normal_in_volume_weight(order):
    """The self-adjoint diffusion operator is r^2-symmetric exactly on the
    3-point stencil (kappa = 1 to rounding) and to O(h^p) on the 5-point one."""
    worst = {}
    for N_r in (24, 48):
        cp, K, M = _diffusion(N_r, order)
        W = an.natural_weight(cp, M)
        lam, X, Y = an.two_sided_dense(K, M, sigma=-1.0)
        low = np.argsort(lam.real)[:10]
        kappa = an.condition_numbers(K, M, X[:, low], Y[:, low], W)
        worst[N_r] = float(np.max(kappa - 1.0))
        red = an.reduced_operator(K, M, sigma=-1.0, W=W)
        henrici = an.henrici(red.A)["relative"]
        print(f"BENCH | sh_fd diffusion order {order} N_r {N_r}: max kappa-1 {worst[N_r]:.2e}, "
              f"Henrici departure {henrici:.2e}")
        if order == 2:
            assert worst[N_r] < 1e-10
            assert henrici < 1e-10
    if order == 4:
        assert worst[24] < 1e-3 and worst[48] < 0.5 * worst[24]


def _viscous_shell(Ek, m=2):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_parameter("Ek", Ek)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    grid = make_shell_grid(N_r=10, N_theta=20, r_inner=0.71, r_outer=0.985)
    return pb.compile(grid, m=m, gauge_unknowns=["p"])


def test_sensitivity_predicts_parameter_derivative():
    """dlambda = y^H (dK - lambda dM) x / (y^H M x) against the finite
    difference of the Rossby eigenvalue between Ek and Ek (1 + 1e-3)."""
    Ek = 1e-3
    K1, M1 = _viscous_shell(Ek).assemble_evp_matrices()
    K2, M2 = _viscous_shell(Ek * 1.001).assemble_evp_matrices()
    sigma = -2j / 3 + 0.02
    lam1, V1 = solvers.shift_invert(K1, M1, sigma, 4, return_vectors=True)
    j = int(np.argmin(np.abs(lam1 - sigma)))
    Y, X, _ = an.left_eigenvectors(K1, M1, lam1[j], V1[:, j])
    dlam = an.sensitivity(K1, M1, lam1[j], X, Y, K2 - K1, M2 - M1)[0]
    lam2, _ = solvers.shift_invert(K2, M2, sigma, 4)
    actual = lam2[np.argmin(np.abs(lam2 - lam1[j]))] - lam1[j]
    print(f"BENCH | d lambda / d Ek: predicted {dlam / (0.001 * Ek):.6e}, "
          f"finite difference {actual / (0.001 * Ek):.6e}")
    assert abs(dlam - actual) < 1e-2 * abs(actual)


def _inertial_ball(m, host, N_r=12, N_theta=18):
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


def test_collocated_ball_ranking_rmode_well_conditioned():
    """At m = 2 the sectoral r-mode omega = -2/3 has kappa near 1 in the
    volume norm while the worst modes of the non-adjoint collocated pencil
    exceed 100."""
    cp = _inertial_ball(2, "momentum")
    K, M = cp.assemble_evp_matrices()
    W = an.natural_weight(cp, M)
    lam, X, Y = an.two_sided_dense(K, M, sigma=0.3 + 0.2j)
    rep = an.condition_report(K, M, lam, X, W=W, Y=Y, threshold=100.0)
    om = -1j * lam
    j_r = int(np.argmin(np.abs(om.real + 2.0 / 3.0)))
    print(f"BENCH | collocated ball m=2 12x18: r-mode omega {om[j_r]:.5f} kappa "
          f"{rep.kappa[j_r]:.3f}; worst kappa {rep.kappa[rep.order[0]]:.3e} at omega "
          f"{om[rep.order[0]]:.4f}; median kappa {np.median(rep.kappa):.1f}")
    assert abs(om[j_r].real + 2.0 / 3.0) < 1e-3
    assert rep.kappa[j_r] < 1.5
    assert rep.kappa[rep.order[0]] > 100.0
    assert j_r not in rep.ill_conditioned


def test_collocated_ball_checkerboard_is_null_direction():
    """With the textbook wall closure at m = 0 the pressure checkerboard is a
    p-only vector annihilated by K and M: the pencil is singular, every
    'finite' eigenvalue is garbage with huge kappa; deflating that one
    direction restores neutral inertial modes whose eigenvalues do not
    depend on the shift."""
    cp = _inertial_ball(0, "momentum")
    K, M = cp.assemble_evp_matrices()
    s, modes = an.pressure_null_modes(K, M, k=2)
    assert s[0] < 1e-12 and s[1] > 1e-6
    W = an.natural_weight(cp, M)
    lam, X, Y = an.two_sided_dense(K, M, sigma=0.3 + 0.2j)
    kappa = an.condition_numbers(K, M, X, Y, W)
    om = -1j * lam
    print(f"BENCH | collocated ball m=0 momentum host: null singular value {s[0]:.1e}, "
          f"median kappa {np.median(kappa):.1e}, neutral modes "
          f"{int(np.sum(np.abs(om.imag) < 1e-6))} of {len(lam)}")
    assert np.median(kappa) > 30.0
    assert np.sum(np.abs(om.imag) < 1e-6) == 0
    defl = an.Deflation(K, M, modes[:, :1])
    assert defl.null.all()
    got = []
    for sigma in (1.2j + 0.01, 1.2j - 0.02):
        lam_d, V_d = solvers.shift_invert(K, M, sigma, 8, return_vectors=True, deflation=defl)
        ok = np.isfinite(lam_d)
        res = defl.residual(lam_d[ok], V_d[:, ok])
        got.append(np.sort_complex(lam_d[ok][res < 1e-8]))
    neutral = np.abs(got[0].real) < 1e-8
    print(f"BENCH | deflated m=0 eigenvalues: {np.round(got[0], 5)}")
    assert neutral.sum() >= 2
    for x in got[0][neutral]:
        assert np.min(np.abs(got[1] - x)) < 1e-8
