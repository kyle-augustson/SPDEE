"""The Dedalus core on the full sphere (ball, r_inner = 0).

Analytic benchmarks at two resolutions each, all through the string entry
``Problem.compile(grid, m, backend='dedalus')`` on a ball grid:

  * scalar Helmholtz with Dirichlet and Neumann walls: spherical Bessel zeros,
    a coefficient bound at compile time, the infinite-eigenvalue count;
  * Robin walls: the Helmholtz condition dot(r_hat, grad(p)) + h p = 0 and
    the radiative (Biot) condition on thermal diffusion give
    k j_l'(k) + h j_l(k) = 0, whichever way the coefficient is spelled;
  * stress-free vector diffusion: toroidal decay rates from the Bessel
    condition k j_l'(k) = j_l(k), divergence-free eigenvectors; Navier slip
    (1 - l_s) j_l(k) + l_s k j_l'(k) = 0 from a theta/phi pair of mixed rows;
  * the traction-free Stokes ball, dot(r_hat, dot(strain, r_hat)) - c p = 0:
    row linearity, tau layout, the poloidal roots of a 2 x 2 Bessel
    determinant, the l = 0 gauge artefact lambda = 10/c;
  * inviscid inertial modes: Zhang et al. (2001) frequencies and the
    sectoral Rossby family omega = -2/(m+1) (sign as in the shell test);
  * the same Helmholtz problem on the fd2 collocated ball grid;
  * MHD with a uniform axial field (solar_eigenmodes.add_magnetic_field on a ball);
  * unit-vector projections: refused in the bulk, folded into a radial
    vector background under an f(r) coefficient; zero backgrounds drop.

Conventions: the pencil is K x = lambda M x with lambda = -(growth rate), so a
decaying mode has lambda > 0, and omega = -i lambda; Zhang's e^{i(m phi + 2 sigma t)}
frequencies appear at omega = -2 sigma.

Skips when Dedalus is not importable.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

pytest.importorskip("dedalus")

import sympy as sp

import spdee
from spdee import Background, Problem, theta
from spdee.applications.solar_eigenmodes import add_magnetic_field
from spdee.backends.dedalus import DedalusGrid
from spdee.backends.fd2.grid import make_ball_grid
from spdee.solvers import dense, shift_invert
from test_evp_zhang_2001 import ZHANG_TARGETS_M1, ZHANG_TARGETS_M2

LOW = DedalusGrid(N_r=10, N_theta=12)
HIGH = DedalusGrid(N_r=14, N_theta=16)


def _roots(f, kmax=20.0, n=3, kmin=1e-3):
    """The first n positive roots of f by sign scan and bisection."""
    ks = np.linspace(kmin, kmax, 6000)
    v = f(ks)
    out = []
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        out.append(brentq(f, ks[i], ks[i + 1]))
    return np.array(out[:n])


def _census(cp, sigma):
    """Finite pencil eigenvalues and the count of infinite ones."""
    K, M = cp.assemble_evp_matrices()
    res = dense(K, M, sigma=sigma, backend="numpy")
    return res.lam, res.n_infinite


def _lowest(lam, n):
    """The n smallest real eigenvalues; the spectrum must be real."""
    assert np.abs(lam.imag).max() < 1e-8 * max(1.0, np.abs(lam).max())
    return np.sort(lam.real)[:n]


def _helmholtz(bc="p = 0", coefficient=None, params=None, backgrounds=None):
    pb = Problem()
    pb.add_unknown("p", "scalar")
    for name, value in (params or {}).items():
        pb.add_parameter(name, value)
    for name, expr in (backgrounds or {}).items():
        pb.add_background(name, expr=expr)
    if coefficient is None:
        pb.add_equation("dt(p) - lap(p) = 0")
    else:
        pb.add_parameter("Ek", coefficient)
        pb.add_equation("dt(p) - Ek*lap(p) = 0")
    pb.add_bc(bc, at="outer")
    return pb


def _inertial():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


def _omega(cp, sigma=0.0):
    lam, _ = _census(cp, sigma)
    return -1j * lam


def _bessel_targets(m, n_per_ell, derivative, max_ell=6):
    """lambda = k^2 over l >= m with j_l(k) = 0 (or j_l'(k) = 0)."""
    out = []
    for ell in range(m, max_ell + 1):
        ks = _roots(lambda k: spherical_jn(ell, k, derivative=derivative), n=n_per_ell)
        out.extend(ks ** 2)
    if derivative and m == 0:
        out.append(0.0)
    return np.sort(out)


# ---- scalar Helmholtz

@pytest.mark.parametrize("m", [0, 1, 2])
def test_helmholtz_dirichlet(m):
    """lambda = k^2, j_l(k) = 0 over l >= m (l=0: pi, 2pi; l=1: 4.493409,
    7.725252; l=2: 5.763459), exact to roundoff and improving with N."""
    target = _bessel_targets(m, 3, False)[:6]
    errs = []
    for grid in (LOW, HIGH):
        cp = _helmholtz().compile(grid, m=m, backend="dedalus")
        lam, n_inf = _census(cp, -1.0)
        n_blocks = grid.N_theta - m
        assert n_inf == 2 * n_blocks                      # tau column + constraint row per l
        assert lam.size == cp.A_total.shape[0] - 2 * n_blocks
        errs.append(np.abs(_lowest(lam, 6) - target) / target)
    assert errs[1].max() < 1e-9, errs
    assert errs[1].max() <= errs[0].max() + 1e-12


@pytest.mark.parametrize("m", [0, 1, 2])
def test_helmholtz_neumann(m):
    """dot(r_hat, grad(p)) = 0: j_l'(k) = 0 (l=1: 2.081576; l=2: 3.342094;
    l=0: 4.493409) plus the constant mode at m = 0."""
    target = _bessel_targets(m, 3, True)[:6]
    errs = []
    for grid in (LOW, HIGH):
        cp = _helmholtz("dot(r_hat, grad(p)) = 0").compile(grid, m=m, backend="dedalus")
        lam, n_inf = _census(cp, -1.0)
        assert n_inf == 2 * (grid.N_theta - m)
        got = _lowest(lam, 6)
        errs.append(np.abs(got - target) / np.maximum(target, 1.0))
    assert errs[1].max() < 1e-9, errs
    assert errs[1].max() <= errs[0].max() + 1e-12


def _robin_targets(m, h, n_per_ell=3, max_ell=6):
    """p = j_l(k r) Y_lm e^{-k^2 t} with k j_l'(k) + h j_l(k) = 0 at r = 1:
    lambda = k^2 over l >= m.  For h > 0 the wall is dissipative
    (d/dt int p^2/2 = -int |grad p|^2 - h oint p^2), so no constant mode
    and every lambda is positive."""
    out = []
    for ell in range(m, max_ell + 1):
        ks = _roots(lambda k: k * spherical_jn(ell, k, derivative=True)
                    + h * spherical_jn(ell, k), n=n_per_ell)
        out.extend(ks ** 2)
    return np.sort(out)


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("h", [1, 3])
def test_helmholtz_robin(m, h):
    """dot(r_hat, grad(p)) + h*p = 0 with h a symbolic parameter: the
    projection is radial(grad(p)(r=1)), the other term p(r=1), the row
    their sum; lambda = k^2 with k j_l'(k) + h j_l(k) = 0 (h = 1, l = 0:
    cos k = 0, k = pi/2, 3pi/2, ...; h = 3, l = 0: 2.288930, 3.405608) to
    roundoff at both resolutions, one tau per l."""
    target = _robin_targets(m, h)[:6]
    errs = []
    for grid in (LOW, HIGH):
        pb = _helmholtz("dot(r_hat, grad(p)) + h*p = 0", params={"h": float(h)})
        cp = pb.compile(grid, m=m, backend="dedalus")
        lam, n_inf = _census(cp, -1.0)
        assert n_inf == 2 * (grid.N_theta - m)
        errs.append(np.abs(_lowest(lam, 6) - target) / target)
    assert errs[1].max() < 1e-9, errs
    assert errs[1].max() <= errs[0].max() + 1e-12


@pytest.mark.parametrize("bc,background", [
    ("dot(r_hat, grad(p)) + h*p = 0", None),
    ("r**2*dot(r_hat, grad(p)) + (2*r**2 + 1)*p = 0", None),
    ("f0*p + dot(r_hat, grad(p)) = 0", "2*r**2 + 1"),
    ("-(dot(r_hat, grad(p)) + 3*p) = 0", None),
    ("dot(grad(p), r_hat) - (-3)*p = 0", None),
], ids=["symbol", "sympy_r", "background", "negated", "reversed"])
def test_robin_coefficient_at_the_wall(bc, background):
    """Every spelling of the coefficient is taken at the wall: a number, a
    symbolic parameter, a SymPy expression in r evaluated at r = 1, a
    radial background interpolated there, a coefficient on the projection
    itself, a negated row, the unit vector on the right; all give the
    h = 3 pencil."""
    ref = _helmholtz("dot(r_hat, grad(p)) + 3*p = 0").compile(LOW, m=1, backend="dedalus")
    backgrounds = None if background is None else {"f0": background}
    pb = _helmholtz(bc, params={"h": 3.0}, backgrounds=backgrounds)
    cp = pb.compile(LOW, m=1, backend="dedalus")
    scale = abs(ref.A_total).max()
    A, B = cp.A_total, cp.B_total
    if bc.startswith("-"):                                          # only the wall rows flip
        A, B = abs(A), abs(B)
        ref.A_total, ref.B_total = abs(ref.A_total), abs(ref.B_total)
    assert abs(A - ref.A_total).max() < 1e-13 * scale
    assert abs(B - ref.B_total).max() < 1e-13 * scale


@pytest.mark.parametrize("m", [0, 2])
def test_thermal_diffusion_radiative_wall(m):
    """dt(S) - kappa*lap(S) = 0 with the Biot condition
    dot(r_hat, grad(S)) + Bi*S = 0: lambda = kappa k^2 with
    k j_l'(k) + Bi j_l(k) = 0."""
    kappa, Bi = 0.1, 2.0
    target = kappa * _robin_targets(m, Bi)[:6]
    errs = []
    for grid in (LOW, HIGH):
        pb = Problem()
        pb.add_unknown("S", "scalar")
        pb.add_parameter("kappa", kappa)
        pb.add_parameter("Bi", Bi)
        pb.add_equation("dt(S) - kappa*lap(S) = 0")
        pb.add_bc("dot(r_hat, grad(S)) + Bi*S = 0", at="outer")
        cp = pb.compile(grid, m=m, backend="dedalus")
        lam, n_inf = _census(cp, -1.0)
        assert n_inf == 2 * (grid.N_theta - m)
        errs.append(np.abs(_lowest(lam, 6) - target) / target)
    assert errs[1].max() < 1e-9, errs
    assert errs[1].max() <= errs[0].max() + 1e-12


def test_helmholtz_coefficient_bound_at_compile():
    """dt(p) - Ek*lap(p) = 0 with the symbolic Ek bound at compile time:
    lambda = Ek k^2; rebinding Ek rescales the spectrum."""
    pb = _helmholtz(coefficient=0.01)
    target = 0.01 * _bessel_targets(1, 2, False)[:4]
    a = _lowest(_census(pb.compile(HIGH, m=1, backend="dedalus"), -1.0)[0], 4)
    assert np.abs(a - target).max() < 1e-10 * target.max()
    pb.set_parameter("Ek", 0.02)
    b = _lowest(_census(pb.compile(HIGH, m=1, backend="dedalus"), -1.0)[0], 4)
    assert np.abs(b - 2 * a).max() < 1e-10 * target.max()
    pb = Problem()
    pb.add_unknown("p", "scalar")
    pb.add_parameter("c", 0.5, symbolic=False)
    pb.add_equation("dt(p) - c*lap(p) = 0")
    pb.add_bc("p = 0", at="outer")
    c = _lowest(_census(pb.compile(HIGH, m=1, backend="dedalus"), -1.0)[0], 4)
    assert np.abs(c - 50 * a).max() < 1e-10 * c.max()


def test_helmholtz_on_fd2_ball_grid_object():
    """A ball Grid from make_ball_grid selects the ball basis."""
    cp = _helmholtz().compile(make_ball_grid(N_r=12, N_theta=12), m=0, backend="dedalus")
    assert cp.grid.is_ball and cp.grid.N_r == 12
    lam, _ = _census(cp, -1.0)
    assert abs(_lowest(lam, 1)[0] - np.pi ** 2) < 1e-9


# ---- vector diffusion

def _toroidal_roots(ell, n=2):
    """Toroidal modes v = T(r) r_hat x grad(Y_lm) of dt(v) = Ek lap(v) have
    T = j_l(k r), lambda = Ek k^2; zero tangential stress at r = 1 is
    d/dr (T/r) = 0, i.e. k j_l'(k) = j_l(k) (l = 1: j_2(k) = 0, and the
    rigid rotation k = 0)."""
    return _roots(lambda k: k * spherical_jn(ell, k, derivative=True) - spherical_jn(ell, k), n=n)


def _diffusion(Ek):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_parameter("Ek", Ek)
    pb.add_equation("dt(v) - Ek*lap(v) = 0")
    pb.add_stress_free("v", at="outer")
    return pb


@pytest.mark.parametrize("m", [0, 1, 2])
def test_vector_diffusion_stress_free_toroidal(m):
    """Every toroidal decay rate Ek k^2 with k j_l'(k) = j_l(k), l = max(m, 1)..3,
    is in the spectrum to roundoff at both resolutions; the l = 1 rigid
    rotation (lambda = 0) is present for m <= 1; the tau layout is one
    vector tau for the three stress-free rows.  Without div(v) = 0 the
    stress-free wall is not dissipative for compressible motion (the energy
    boundary term is +Ek |v_h|^2), so one growing l = 1 mode exists at
    m <= 1; only the toroidal family is compared."""
    Ek = 0.01
    for grid in (LOW, HIGH):
        cp = _diffusion(Ek).compile(grid, m=m, backend="dedalus")
        assert [t.name for t in cp.tau_fields] == ["tau_v1"]
        lam, n_inf = _census(cp, -0.5)
        assert n_inf == 6 * (grid.N_theta - m) - (4 if m == 0 else 0)   # 3 rows per l, 1 at l = 0
        assert np.abs(lam.imag).max() < 1e-8
        lam = np.sort(lam.real)
        for ell in range(max(m, 1), 4):
            for k in _toroidal_roots(ell):
                assert abs(lam[np.argmin(np.abs(lam - Ek * k ** 2))] - Ek * k ** 2) < 1e-10 * Ek * k ** 2
        if m <= 1:
            assert np.abs(lam).min() < 1e-9
        assert lam[lam < -1e-9].size == (1 if m <= 1 else 0)


def _slip_roots(ell, ls, n=2):
    """Navier slip v_t + l_s (d/dr v_t - v_t/r) = 0 on the toroidal
    profile T = j_l(k r): (1 - l_s) j_l(k) + l_s k j_l'(k) = 0."""
    return _roots(lambda k: (1 - ls) * spherical_jn(ell, k)
                  + ls * k * spherical_jn(ell, k, derivative=True), n=n)


@pytest.mark.parametrize("m", [0, 1])
def test_vector_diffusion_navier_slip(m):
    """A theta/phi pair of mixed rows, dot(theta_hat, v) + ls*dot(theta_hat,
    dot(strain, r_hat)) = 0 and its phi twin, is one angular row: the
    toroidal decay rates are Ek k^2 with (1 - l_s) j_l(k) + l_s k j_l'(k) = 0
    (l_s = 1/2: j_l(k) + k j_l'(k) = 0), the tau layout that of the
    stress-free wall."""
    Ek, ls = 0.01, 0.5
    for grid in (LOW, HIGH):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_parameter("Ek", Ek)
        pb.add_parameter("ls", ls)
        pb.add_equation("dt(v) - Ek*lap(v) = 0")
        pb.add_bc("dot(r_hat, v) = 0", at="outer")
        for hat in ("theta_hat", "phi_hat"):
            pb.add_bc(f"dot({hat}, v) + ls*dot({hat}, dot(grad(v) + trans(grad(v)), r_hat)) = 0",
                      at="outer")
        cp = pb.compile(grid, m=m, backend="dedalus")
        assert [t.name for t in cp.tau_fields] == ["tau_v1"]
        lam, n_inf = _census(cp, -0.5)
        assert n_inf == 6 * (grid.N_theta - m) - (4 if m == 0 else 0)
        assert np.abs(lam.imag).max() < 1e-8
        lam = np.sort(lam.real)
        for ell in range(max(m, 1), 4):
            for k in _slip_roots(ell, ls):
                rate = Ek * k ** 2
                assert abs(lam[np.argmin(np.abs(lam - rate))] - rate) < 1e-10 * rate


def test_vector_diffusion_toroidal_eigenvector_is_solenoidal():
    Ek = 0.01
    cp = _diffusion(Ek).compile(HIGH, m=1, backend="dedalus")
    k = _toroidal_roots(2, n=1)[0]
    lam, V = shift_invert(cp.A_total, cp.B_total, Ek * k ** 2 + 1e-4, 3, return_vectors=True)
    j = int(np.argmin(np.abs(lam - Ek * k ** 2)))
    assert abs(lam[j] - Ek * k ** 2) < 1e-10
    cp.set_state(V[:, j])
    v = cp.fields["v"]
    div = cp.d3.div(v).evaluate()
    div.change_scales(1)
    vg = v["g"]
    assert np.abs(div["g"]).max() < 1e-10 * np.abs(vg).max()
    assert np.abs(vg[2]).max() < 1e-10 * np.abs(vg).max()          # purely toroidal


# ---- inviscid inertial modes

@pytest.mark.parametrize("m,targets", [(1, ZHANG_TARGETS_M1), (2, ZHANG_TARGETS_M2)],
                         ids=["m1", "m2"])
def test_inertial_modes_zhang_2001(m, targets):
    """Zhang et al. (2001) polynomial inertial modes of the full sphere with
    dot(r_hat, v) = 0 and p gauged.  The tabulated omega (e^{i(m phi + 2 sigma t)})
    appear at omega = -2 sigma here, the convention fixed by the sectoral
    Rossby mode at -2/(m+1) as in test_dedalus_backend; polynomial modes are
    exact, so the two resolutions agree to roundoff and the residual is the
    5-decimal tabulation."""
    om = [_omega(_inertial().compile(g, m=m, backend="dedalus", gauge_unknowns=["p"]), 0.3j)
          for g in (LOW, HIGH)]
    for label, target, _tol in targets:
        found = [w.real[np.argmin(np.abs(w.real + target))] for w in om]
        assert abs(found[1] + target) < 1e-5, (label, found[1], -target)
        assert abs(found[1] - found[0]) < 1e-9, (label, found)
    for w in om:
        near = w[np.abs(w.real) < 2.0]
        assert np.abs(near.imag).max() < 1e-9                      # inertial modes are neutral


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_rossby_family(m):
    """The l = m toroidal mode v ~ r^m r_hat x grad(Y_mm) is exact for the
    inviscid rotating sphere: omega = -2/(m+1), same sign as on the shell."""
    cp = _inertial().compile(LOW, m=m, backend="dedalus", gauge_unknowns=["p"])
    om = _omega(cp, 0.3j)
    target = -2.0 / (m + 1)
    assert np.abs(om - target).min() < 1e-10
    assert np.abs(om + target).min() > 1e-3                          # the mirror is not a mode
    res = cp.solve_evp(k=4, sigma=1j * target + 0.02, return_eigenvectors=True)
    j = int(np.nanargmin(np.abs(res.omega() - target)))
    assert abs(res.omega()[j] - target) < 1e-9
    v = res.eigenvectors[j]["v"]
    assert v.shape == (3, LOW.N_theta, LOW.N_r)
    assert np.abs(v[2]).max() < 1e-9 * np.abs(v[0]).max()             # toroidal


# ---- cross-core

@pytest.mark.parametrize("m", [0, 1, 2])
def test_helmholtz_matches_fd2(m):
    """The fd2 collocated core on make_ball_grid converges to the spectral
    eigenvalues at its finite-difference order."""
    pb = _helmholtz()
    ref = _lowest(_census(pb.compile(HIGH, m=m, backend="dedalus"), -1.0)[0], 6)
    errs = []
    for N_r, N_theta in ((24, 16), (32, 24)):
        cp = pb.compile(make_ball_grid(N_r=N_r, N_theta=N_theta), m=m)
        K, M = cp.assemble_evp_matrices()
        lam = dense(K, M, sigma=-1.0, backend="numpy").lam
        got = np.sort(lam.real[lam.real > 0])[:6]
        errs.append(np.abs(got - ref) / ref)
    assert errs[1].max() < 2e-3, errs
    assert np.all(errs[1] <= errs[0] + 1e-12), errs


# ---- MHD

def _mhd(Le, bc="conducting", geometry="ball"):
    B0 = Background.Vector("B0", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero)
    return add_magnetic_field(_inertial().build(gauge_unknowns=["p"]), B0, Le=Le, eta=1e-2,
                              geometry=geometry, bc=bc, solenoidal=True)


def test_mhd_ball_uniform_axial_field():
    """b-formulation with a uniform axial field: the pencil assembles with one
    vector tau per vector unknown and three probed terms (Coriolis, Lorentz,
    v x B0 shared by the induction equation and the conducting wall), the
    eigenvectors are solenoidal, the field-free limit keeps the hydro
    Rossby mode, and a finite Le changes the pencil."""
    cps = {Le: spdee.compile_problem(_mhd(Le), backend="dedalus", grid=LOW, m=1) for Le in (0.0, 0.3)}
    cp = cps[0.0]
    assert [t.name for t in cp.tau_fields] == ["tau_v1", "tau_b1", "tau_p_gauge", "tau_psi_gauge"]
    assert [(t.kind, [f.name for f in t.refs]) for t in cp._aux] == \
        [("cross", ["v"]), ("cross", ["b"]), ("cross", ["v"])]
    for Le, c in cps.items():
        lam, V = shift_invert(c.A_total, c.B_total, 1j * (-1.0) + 0.02, 4, return_vectors=True)
        om = -1j * lam
        j = int(np.argmin(np.abs(om + 1.0)))
        if Le == 0.0:
            assert abs(om[j] + 1.0) < 1e-9
        c.set_state(V[:, j])
        b = c.fields["b"]
        div = c.d3.div(b).evaluate()
        div.change_scales(1)
        assert np.abs(div["g"]).max() < 1e-9 * np.abs(b["g"]).max()
    assert abs(cps[0.3].A_total - cps[0.0].A_total).max() > 0
    pinned = spdee.compile_problem(_mhd(0.3, bc="pinned"), backend="dedalus", grid=LOW, m=1)
    assert pinned.A_total.shape == cp.A_total.shape


# ---- normal stress

def _stokes(*bcs):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(p) = 0")
    pb.add_equation("div(v) = 0")
    for bc in bcs:
        pb.add_bc(bc, at="outer")
    return pb


STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def _traction_free_roots(ell, c, n=2):
    """Poloidal modes of the Stokes ball with a traction-free wall,
    lap(v) + k^2 v = grad(p), div(v) = 0, lambda = k^2: p = a k^2 r^l Y is
    harmonic and v = a grad(r^l Y) + b curl(curl(r j_l(k r) Y)), so
    v_r = R(r) Y with R = a l r^(l-1) + b l(l+1) j_l(k r)/r and
    v_theta = S(r) dY/dtheta with S = a r^(l-1) + b (j_l' + j_l/r).  The
    rows 2 R' - c a k^2 = 0 (normal stress, ``c`` the coefficient of p) and
    S' - S + R = 0 (tangential stress) at r = 1 have a solution when their
    2 x 2 determinant vanishes."""
    def det(k):
        f = spherical_jn(ell, k)
        f1 = k * spherical_jn(ell, k, derivative=True)
        f2 = -2 * f1 - (k ** 2 - ell * (ell + 1)) * f
        n = (2 * ell * (ell - 1) - c * k ** 2, 2 * ell * (ell + 1) * (f1 - f))
        t = (2 * (ell - 1), f2 + (ell * (ell + 1) - 2) * f)
        return n[0] * t[1] - n[1] * t[0]
    return _roots(det, n=n)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("c", [1, 2])
def test_normal_stress_row(m, c):
    """The traction-free Stokes ball, dot(r_hat, dot(strain, r_hat)) - c*p = 0
    with the two tangential stress rows (c = 1 is the normal stress of
    dt(v) - lap(v) + grad(p)): the nested projection is
    radial(radial(strain(r=1), index=1), index=0) and the row is linear in
    the IR (the rows with -c p and +c p average to the row without p).  The
    tau layout is that of the stress-free wall; the wall row constrains p
    instead of v_r, so each l has one more finite eigenvalue than the
    impenetrable ball.  The spectrum holds the toroidal family
    k j_l'(k) = j_l(k) (p = 0, no normal stress), the poloidal roots of
    _traction_free_roots (l = 2, c = 2: 8, 28.211083; l = 1: 14.978747,
    the same for every c), the rigid rotation and translation at
    lambda = 0 for m <= 1, and at m = 0 the l = 0 uniform expansion
    v = alpha r that the gauge tau in div(v) + tau = 0 admits once v_r(1)
    is free: p = lambda alpha r^2/2 + p0 with integ(p) = 0 gives
    p(1) = lambda alpha/5 and the row 2 alpha - c p(1) = 0 gives
    lambda = 10/c."""
    tangential = [f"dot({hat}, {STRAIN}) = 0" for hat in ("theta_hat", "phi_hat")]
    ref = _stokes("dot(r_hat, v) = 0", *tangential).compile(LOW, m=m, backend="dedalus",
                                                              gauge_unknowns=["p"])
    cps = {}
    for sign in ("-", "+", ""):
        term = f"{sign} {c}*p" if sign else ""
        pb = _stokes(f"dot(r_hat, {STRAIN}) {term} = 0", *tangential)
        cps[sign] = pb.compile(LOW, m=m, backend="dedalus", gauge_unknowns=["p"])
    cp = cps["-"]
    assert [t.name for t in cp.tau_fields] == ["tau_v1", "tau_p_gauge"]
    assert cp.A_total.shape == ref.A_total.shape
    scale = abs(cp.A_total).max()
    assert abs((cps["-"].A_total + cps["+"].A_total) / 2 - cps[""].A_total).max() < 1e-13 * scale
    assert abs(cps["-"].A_total - cps[""].A_total).max() > 1e-3 * scale
    lam, n_inf = _census(cp, -0.5)
    assert n_inf == _census(ref, -0.5)[1] - (LOW.N_theta - m)
    assert np.abs(lam.imag).max() < 1e-8
    lam = np.sort(lam.real)
    assert np.count_nonzero(np.abs(lam) < 1e-8) == 2
    targets = [10.0 / c] if m == 0 else []
    for ell in range(max(m, 1), 4):
        targets.extend(k ** 2 for k in _toroidal_roots(ell))
        targets.extend(k ** 2 for k in _traction_free_roots(ell, c))
    for t in targets:
        assert abs(lam[np.argmin(np.abs(lam - t))] - t) < 1e-10 * t, t


# ---- rejected inputs

def test_inner_bc_rejected_on_ball():
    pb = _helmholtz()
    pb.add_bc("p = 0", at="inner")
    with pytest.raises(ValueError, match="no inner boundary"):
        pb.compile(LOW, m=0, backend="dedalus")
    with pytest.raises(ValueError, match="no inner boundary"):
        spdee.compile_problem(_mhd(0.1, geometry="shell"), backend="dedalus", grid=LOW, m=1)


def test_single_tangential_bc_rejected_on_ball():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_equation("dt(v) - lap(v) = 0")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    pb.add_bc("dot(theta_hat, v) = 0", at="outer")
    with pytest.raises(NotImplementedError, match="phi_hat"):
        pb.compile(LOW, m=1, backend="dedalus")


@pytest.mark.parametrize("row", ["dot(theta_hat, v) + p = 0",
                                 "dot(theta_hat, v) + dot(phi_hat, v) = 0",
                                 "dot(theta_hat, v) + dot(r_hat, grad(p)) = 0"])
def test_mixed_tangential_row_rejected_on_ball(row):
    """A tangential projection is a vector on the sphere: it shares a row
    only with projections on the same unit vector."""
    with pytest.raises(NotImplementedError, match="same unit vector"):
        _stokes("dot(r_hat, v) = 0", row).compile(LOW, m=1, backend="dedalus", gauge_unknowns=["p"])


def test_theta_dependent_wall_coefficient_rejected_on_ball():
    pb = _helmholtz("dot(r_hat, grad(p)) + g0*p = 0", backgrounds={"g0": "1 + cos(theta)**2"})
    with pytest.raises(NotImplementedError, match="theta-dependent coefficient"):
        pb.compile(LOW, m=1, backend="dedalus")


def test_unit_vector_product_rejected_on_ball():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_equation("dt(v) - lap(v) + cross(r_hat, v) = 0")
    pb.add_stress_free("v", at="outer")
    with pytest.raises(NotImplementedError, match="r_hat"):
        pb.compile(LOW, m=1, backend="dedalus")


def _projection_problem(term):
    """A scalar probe equation carrying ``term`` next to a vector unknown."""
    pb = Problem()
    pb.add_unknown("u", "vector")
    pb.add_unknown("w", "scalar")
    pb.add_background("f0", expr="r*(1 - r**2)")
    pb.add_background("D0", v_r="r*(1 - r**2)")
    pb.add_equation("dt(u) - lap(u) = 0")
    pb.add_equation(f"dt(w) - lap(w) + ({term}) = 0")
    pb.add_stress_free("u", at="outer")
    pb.add_bc("w = 0", at="outer")
    return pb


@pytest.mark.parametrize("term", [
    "dot(r_hat, u)", "-dot(r_hat, u)", "dot(r_hat, u) + w", "2*dot(r_hat, u)",
    "dt(dot(r_hat, u))", "dot(theta_hat, u)", "(1 - r**2)*dot(theta_hat, u)",
    "(1 - r**2)*dot(phi_hat, u)",
])
def test_unit_vector_projection_refused_in_the_bulk(term):
    """A unit-vector projection at equation level reaches the guard whatever
    wraps it.  v_r of a regular vector is r^(l-1) Y_l at the origin (v = e_z
    has v_r = cos(theta)), not a regular scalar, and theta/phi components
    are singular at the poles, so the projection is a wall row only."""
    with pytest.raises(NotImplementedError, match="boundary rows"):
        _projection_problem(term).compile(LOW, m=1, backend="dedalus")


@pytest.mark.parametrize("term", ["f0*dot(r_hat, u)", "dot(r_hat, u)*f0", "f0*dot(u, r_hat)",
                                  "r*(1 - r**2)*dot(r_hat, u)"])
def test_radial_coefficient_folds_into_r_hat(term):
    """f(r)*dot(r_hat, X) is regular when f(0) = 0: it compiles as
    dot(f(r) r_hat, X), the same radial-basis NCC pencil as the explicit
    vector background, with no probed auxiliary."""
    ref = _projection_problem("dot(D0, u)").compile(LOW, m=1, backend="dedalus")
    cp = _projection_problem(term).compile(LOW, m=1, backend="dedalus")
    assert cp._aux == [] and ref._aux == []
    assert abs(cp.A_total - ref.A_total).max() == 0.0
    assert abs(cp.B_total - ref.B_total).max() == 0.0


def _zero_background_problem(host, term):
    """Identically zero scalar, vector and tensor backgrounds; ``term`` sits
    in the scalar or the vector equation."""
    pb = Problem()
    pb.add_unknown("s", "scalar")
    pb.add_unknown("v", "vector")
    pb.add_background("rho0", expr="0*r")
    pb.add_background("U0", v_r=0, v_theta=0, v_phi="0.0*r*sin(theta)")
    pb.add_background("T0", components={("r", "r"): "0*r", ("theta", "phi"): 0})
    scalar, vector = ("dt(s) - lap(s)", "dt(v) - lap(v)")
    if host == "scalar":
        scalar += f" + ({term})"
    else:
        vector += f" + ({term})"
    pb.add_equation(f"{scalar} = 0")
    pb.add_equation(f"{vector} = 0")
    pb.add_bc("s = 0", at="outer")
    pb.add_stress_free("v", at="outer")
    return pb


@pytest.mark.parametrize("host,term", [
    ("scalar", "rho0*s"), ("scalar", "s*rho0"), ("scalar", "dot(U0, grad(s))"),
    ("scalar", "dot(grad(s), U0)"), ("scalar", "dot(U0, v)"), ("scalar", "dot(v, U0)"),
    ("scalar", "dt(dot(U0, grad(s)))"), ("scalar", "-dot(U0, grad(s))"),
    ("vector", "rho0*v"), ("vector", "U0*s"), ("vector", "s*U0"), ("vector", "U0*div(v)"),
    ("vector", "cross(U0, v)"), ("vector", "cross(v, U0)"), ("vector", "dot(U0, grad(v))"),
    ("vector", "dot(grad(v), U0)"), ("vector", "dot(U0, grad(v) + trans(grad(v)))"),
    ("vector", "dot(T0, v)"), ("vector", "dot(v, T0)"), ("vector", "rho0*grad(s)"),
])
def test_zero_background_drops_the_term(host, term):
    """An identically zero background reaches the ball product as the number
    0 and drops the term, as Dedalus's own Multiply does: the pencil is the
    one without the term, for every rank and product position."""
    ref = _zero_background_problem(host, "0*s" if host == "scalar" else "0*v")
    ref = ref.compile(LOW, m=1, backend="dedalus")
    cp = _zero_background_problem(host, term).compile(LOW, m=1, backend="dedalus")
    assert cp._aux == []
    assert abs(cp.A_total - ref.A_total).max() == 0.0
    assert abs(cp.B_total - ref.B_total).max() == 0.0


def test_radial_background_is_an_ncc_and_singular_one_warns():
    """f(r) backgrounds become radial-basis NCCs (no probing); a background
    singular at the origin is flagged."""
    pb = _helmholtz()
    pb.add_background("rho", expr="1 - r**2/2")
    pb.add_equation("rho*dt(p) - lap(p) = 0")
    pb.equations.pop(0)
    cp = pb.compile(LOW, m=1, backend="dedalus")
    assert cp._aux == []
    lam, _ = _census(cp, -1.0)
    k2 = 4.493409 ** 2
    assert k2 < _lowest(lam, 1)[0] < 2 * k2                         # 1/rho in [1, 2] speeds the decay
    pb = _helmholtz()
    pb.add_background("rho", expr="1/r")
    pb.add_equation("rho*dt(p) - lap(p) = 0")
    pb.equations.pop(0)
    with pytest.warns(RuntimeWarning, match="not spectrally resolved"):
        pb.compile(LOW, m=1, backend="dedalus")
