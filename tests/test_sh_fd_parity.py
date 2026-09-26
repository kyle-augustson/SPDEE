"""The sh_fd core on the shared contracts (docs/ir_boundary_and_constraints.md).

Composite boundary rows (Robin, Biot, Navier slip, the normal stress) are
translated directly and hosted where the walker's record says, and give the
closed forms at the stencil order on the shell and the ball; tensor unknowns
are carried on nine spin blocks or eliminated when defined; a gauge scalar
present through Grad alone (psi of the vector potential) puts its wall rows
on its constraint equation; a written Div(b) = 0 with a curl evolution is
cleaned by the classifier's multiplier.  The last two are the assertions of
tests/test_analytic_evp.py that flip from a refusal to a benchmark.
"""

import numpy as np
import pytest
import scipy.sparse as sps
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from spdee import Problem
from spdee.backends.sh_fd import ShFdGrid, compile_sh_fd
from spdee.ir.operators import Grad, Transpose
from spdee.solvers import dense
import test_analytic_evp as tae

RI = 0.5
GEOMETRIES = ("shell", "ball")


def _grid(geom, N, ell_max):
    return ShFdGrid(N_r=N, ell_max=ell_max, r_inner=0.0 if geom == "ball" else RI, r_outer=1.0,
                    fd_order=4)


def _walls(geom):
    return ("outer",) if geom == "ball" else ("inner", "outer")


def _census(cp, sigma, resid=None):
    """Finite eigenvalues (and the eigenvectors with ``resid``)."""
    K, M = cp.assemble_evp_matrices()
    res = dense(K, M, sigma=sigma, backend="numpy", vectors=resid is not None,
                compute_resid=resid is not None, keep_vectors=resid is not None)
    keep = np.isfinite(res.lam)
    if resid is not None:
        keep &= res.resid < resid
        return res.lam[keep], res.V[:, keep]
    return res.lam[keep]


def _nearest(lam, t):
    return lam[np.argmin(np.abs(lam - t))]


def _errors(cp_of, targets, sigma, Ns=(24, 48)):
    """Relative errors of the real rates nearest each target at two
    resolutions, (2, n_targets)."""
    out = []
    for N in Ns:
        lam = _census(cp_of(N), sigma)
        assert np.abs(lam.imag).max() < 1e-8 * max(1.0, np.abs(lam).max())
        out.append([abs(_nearest(lam.real, t) - t) / t for t in targets])
    return np.array(out)


def _check(errs, tol, order=3.0, floor=1e-4):
    """Every target within ``tol`` at the finer resolution and the RMS
    error decreasing at least at ``order`` when the coarse RMS is above
    ``floor`` (the individual targets straddle the sign change of the
    eigenvalue error and a two-point order below the floor is not
    meaningful, docs/ball_sh_fd.md 3a)."""
    assert errs[1].max() < tol, errs
    rms = np.sqrt((errs ** 2).mean(axis=1))
    if rms[0] > floor:
        assert np.log(rms[0] / rms[1]) / np.log(2) > order, (rms, errs)


def _roots(f, n=2, kmax=20.0):
    ks = np.linspace(1e-3, kmax, 6000)
    v = f(ks)
    out = []
    for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0]:
        out.append(brentq(f, ks[i], ks[i + 1], xtol=1e-14))
    return np.array(out[:n])


def _wall(kind, l, k, r, h=None, ls=None):
    """r times the wall functional on j_l(k r) and y_l(k r): Robin
    k j_l' + h j_l, Navier slip (1 - ls/r) j_l + ls k j_l' (the toroidal
    profile T = j_l(k r) with v_t + ls (d_r v_t - v_t / r) = 0)."""
    x = k * r
    j, jp = spherical_jn(l, x), spherical_jn(l, x, True)
    y, yp = spherical_yn(l, x), spherical_yn(l, x, True)
    if kind == "robin":
        return x * jp + h * r * j, x * yp + h * r * y
    return (r - ls) * j + ls * x * jp, (r - ls) * y + ls * x * yp


def _det(kind, l, k, geom, **kw):
    if geom == "ball":
        return _wall(kind, l, k, 1.0, **kw)[0]
    a1, b1 = _wall(kind, l, k, RI, **kw)
    a2, b2 = _wall(kind, l, k, 1.0, **kw)
    return a1 * b2 - a2 * b1


def _targets(kind, geom, m, lmax=4, n=2, **kw):
    """k^2 over l = m (l >= 1 for the toroidal families) .. lmax."""
    out = []
    for l in range(m if kind == "robin" else max(m, 1), lmax + 1):
        out.extend(k ** 2 for k in _roots(lambda k: _det(kind, l, k, geom, **kw), n=n))
    return np.sort(out)


# ---- 1. composite boundary rows

def _helmholtz(bc, geom, params=None, backgrounds=None):
    pb = Problem()
    pb.add_unknown("p", "scalar")
    for name, expr in (backgrounds or {}).items():
        pb.add_background(name, expr=expr)
    for name, value in (params or {}).items():
        pb.add_parameter(name, value)
    pb.add_equation("dt(p) - lap(p) = 0", name="helm")
    for at in _walls(geom):
        pb.add_bc(bc, at=at)
    return pb


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_helmholtz_robin(geom, m):
    """dot(r_hat, grad(p)) + h*p = 0 at every wall with h symbolic: the row
    is the direct translation on p's wall node; lambda = k^2 with
    k j_l'(k) + h j_l(k) = 0 on the ball (h = 2, l = 0: 2.0288, 4.9132) and
    the j_l / y_l determinant with k f' + h r f at r = 1/2 and 1 on the
    shell (l = 0: 1.3065, 2.4156), at the stencil order."""
    h = 2.0
    targets = _targets("robin", geom, m, h=h)[:6]
    errs = _errors(lambda N: _helmholtz("dot(r_hat, grad(p)) + h*p = 0", geom, {"h": h}).compile(
        _grid(geom, N, 4), m=m, backend="sh_fd"), targets, float(np.mean(targets)))
    _check(errs, 3e-5)


@pytest.mark.parametrize("m", [0, 2])
def test_thermal_diffusion_radiative_wall(m):
    """dt(S) - kappa*lap(S) = 0 with the Biot condition
    dot(r_hat, grad(S)) + Bi*S = 0 on the ball: lambda = kappa k^2 with
    k j_l'(k) + Bi j_l(k) = 0."""
    kappa, Bi = 0.1, 2.0
    targets = kappa * _targets("robin", "ball", m, h=Bi)[:6]

    def cp_of(N):
        pb = Problem()
        pb.add_unknown("S", "scalar")
        pb.add_parameter("kappa", kappa)
        pb.add_parameter("Bi", Bi)
        pb.add_equation("dt(S) - kappa*lap(S) = 0")
        pb.add_bc("dot(r_hat, grad(S)) + Bi*S = 0", at="outer")
        return pb.compile(_grid("ball", N, 4), m=m, backend="sh_fd")
    _check(_errors(cp_of, targets, float(np.mean(targets))), 3e-5)


@pytest.mark.parametrize("bc,background", [
    ("dot(r_hat, grad(p)) + h*p = 0", None),
    ("r**2*dot(r_hat, grad(p)) + (2*r**2 + 1)*p = 0", None),
    ("f0*p + dot(r_hat, grad(p)) = 0", "2*r**2 + 1"),
    ("-(dot(r_hat, grad(p)) + 3*p) = 0", None),
    ("dot(grad(p), r_hat) - (-3)*p = 0", None),
], ids=["symbol", "sympy_r", "background", "negated", "reversed"])
def test_robin_coefficient_at_the_wall(bc, background):
    """Every spelling of the h = 3 Robin wall on the ball gives the same
    pencil to rounding: a symbol, a SymPy expression in r at r = 1, a radial
    background, a negated row (the wall rows flip sign), the unit on the
    right."""
    grid = _grid("ball", 16, 3)
    ref = _helmholtz("dot(r_hat, grad(p)) + 3*p = 0", "ball").compile(grid, m=1, backend="sh_fd")
    backgrounds = None if background is None else {"f0": background}
    cp = _helmholtz(bc, "ball", {"h": 3.0}, backgrounds).compile(grid, m=1, backend="sh_fd")
    scale = abs(ref.A_total).max()
    A, B, Ar, Br = cp.A_total, cp.B_total, ref.A_total, ref.B_total
    if bc.startswith("-"):
        A, B, Ar, Br = abs(A), abs(B), abs(Ar), abs(Br)
    assert abs(A - Ar).max() < 1e-12 * scale
    assert abs(B - Br).max() < 1e-12 * scale


def _slip_problem(geom, ls):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_parameter("Ek", 0.01)
    pb.add_parameter("ls", ls)
    pb.add_equation("dt(v) - Ek*lap(v) = 0")
    for at in _walls(geom):
        pb.add_bc("dot(r_hat, v) = 0", at=at)
        for hat in ("theta_hat", "phi_hat"):
            pb.add_bc(f"dot({hat}, v) + ls*dot({hat}, dot(grad(v) + trans(grad(v)), r_hat)) = 0",
                      at=at)
    return pb


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_vector_diffusion_navier_slip(geom, m):
    """The theta / phi pair of Navier-slip rows is hosted on the spin +
    and spin - blocks of v (the record's host_sigma) and imposes the two
    spin components of v_t + ls (d_r v_t - v_t/r) = 0 exactly: the toroidal
    rates are Ek k^2 with (1 - ls) j_l(k) + ls k j_l'(k) = 0 on the ball
    (ls = 1/2, l = 1: 2.7437, 6.1168) and the j_l / y_l determinant with
    (r - ls) f + ls k r f' at both radii on the shell, at the stencil order."""
    ls = 0.5
    targets = 0.01 * _targets("slip", geom, m, lmax=3, ls=ls)
    errs = _errors(lambda N: _slip_problem(geom, ls).compile(_grid(geom, N, 3), m=m,
                                                              backend="sh_fd"),
                   targets, float(np.mean(targets)))
    _check(errs, 3e-5)


@pytest.mark.parametrize("ls,kind", [(0.0, "dirichlet"), (1e8, "T_over_r")])
def test_navier_slip_limits(ls, kind):
    """ls = 0 is no-slip (j_l(k) = 0 on the toroidal profile) and ls -> oo
    is stress-free (k j_l'(k) = j_l(k)) on the ball at m = 1."""
    targets = [0.01 * tae._radial_roots(kind, l, 1)[0] ** 2 for l in (1, 2)]
    lam = _census(_slip_problem("ball", ls).compile(_grid("ball", 48, 3), m=1, backend="sh_fd"),
                  float(np.mean(targets)))
    for t in targets:
        assert abs(_nearest(lam.real, t) - t) < 1e-5 * t, (kind, t)


STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def _stokes(c, term, geom="ball"):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    for at in _walls(geom):
        pb.add_bc(f"dot(r_hat, {STRAIN}) {term} = 0", at=at)
        for hat in ("theta_hat", "phi_hat"):
            pb.add_bc(f"dot({hat}, {STRAIN}) = 0", at=at)
    return pb


def _traction_free_roots(ell, c, n=2):
    """tests/test_dedalus_ball.py: the poloidal Stokes modes with a
    traction-free wall, the 2 x 2 determinant of the normal-stress row
    2 R' - c a k^2 = 0 and the tangential row S' - S + R = 0 at r = 1."""
    def det(k):
        f = spherical_jn(ell, k)
        f1 = k * spherical_jn(ell, k, derivative=True)
        f2 = -2 * f1 - (k ** 2 - ell * (ell + 1)) * f
        nn = (2 * ell * (ell - 1) - c * k ** 2, 2 * ell * (ell + 1) * (f1 - f))
        t = (2 * (ell - 1), f2 + (ell * (ell + 1) - 2) * f)
        return nn[0] * t[1] - nn[1] * t[0]
    return _roots(det, n=n)


@pytest.mark.parametrize("m", [0, 1])
@pytest.mark.parametrize("c", [1, 2])
def test_normal_stress_row(m, c):
    """The traction-free Stokes ball dot(r_hat, dot(strain, r_hat)) - c*p = 0:
    a row naming v and p is one row on v's radial wall slot (the record's
    host, not the strongest column) with the columns of both, linear in the
    IR (the rows with -c p and +c p average to the row without p).  The
    spectrum is real with the rigid rotation and translation at zero, the
    toroidal family k j_l'(k) = j_l(k) and the poloidal roots of
    _traction_free_roots (l = 2, c = 2: 8, 28.211083; l = 1: 14.978747) at
    the stencil order; at m = 0 the mean-zero gauge row of p sits on the
    continuity equation.  The l = 0 mode lambda = 10/c of the Dedalus ball
    is the uniform expansion v = alpha r that the gauge tau in
    div(v) + tau = 0 admits; a row-replacement core imposes div(v) = 0 at
    every node but one and has no such mode."""
    targets = []
    for l in range(max(m, 1), 4):
        targets.extend(k ** 2 for k in tae._radial_roots("T_over_r", l, 2))
        targets.extend(k ** 2 for k in _traction_free_roots(l, c))
    targets = np.sort(targets)

    def cp_of(N, term=f"- {c}*p"):
        return _stokes(c, term).compile(_grid("ball", N, 3), m=m, backend="sh_fd",
                                        gauge_unknowns=["p"])
    cps = {s: cp_of(16, f"{s} {c}*p" if s else "") for s in ("-", "+", "")}
    cp = cps["-"]
    scale = abs(cp.A_total).max()
    assert abs((cps["-"].A_total + cps["+"].A_total) / 2 - cps[""].A_total).max() < 1e-13 * scale
    assert abs(abs(cps["-"].A_total - cps[""].A_total).max() - c) < 1e-12
    Nr, nlr = cp.Nr, cp.nlr
    p_off = cp._offset["p"]
    row = cp.A_total[cp._eq_row_offset["momentum"] + 2 * nlr + Nr - 1].toarray().ravel()
    assert np.abs(row[p_off:p_off + nlr]).max() > 0
    assert cp.B_total[cp._eq_row_offset["momentum"] + 2 * nlr + Nr - 1].nnz == 0
    lam = _census(cp, -0.5)
    assert np.abs(lam.imag).max() < 1e-8
    assert np.count_nonzero(np.abs(lam) < 1e-6) == 2
    _check(_errors(cp_of, targets, -0.5), 2e-4)


def test_mixed_tangential_row_is_a_galerkin_row():
    """This core multiplies by the unit vectors through quadrature, so a
    tangential row is the Galerkin projection of the wall function on the
    spin-0 harmonics and a mixed row (theta_hat and phi_hat in one
    expression) is accepted: the mixed no-slip wall v_theta + v_phi = 0,
    v_phi = 0 spans the rows of the theta / phi pair and gives the same
    spectrum to rounding."""
    def cp_of(rows):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_parameter("Ek", 0.01)
        pb.add_equation("dt(v) - Ek*lap(v) = 0")
        pb.add_bc("dot(r_hat, v) = 0", at="outer")
        for row in rows:
            pb.add_bc(row, at="outer")
        return pb.compile(_grid("ball", 24, 4), m=1, backend="sh_fd")
    pair = cp_of(["dot(theta_hat, v) = 0", "dot(phi_hat, v) = 0"])
    mixed = cp_of(["dot(theta_hat, v) + dot(phi_hat, v) = 0", "dot(phi_hat, v) = 0"])
    target = 0.01 * tae._radial_roots("dirichlet", 1, 1)[0] ** 2
    a, b = _census(pair, target), _census(mixed, target)
    a, b = np.sort(a[np.abs(a) < 5.0].real), np.sort(b[np.abs(b) < 5.0].real)
    assert a.size == b.size >= 6
    assert np.abs(a - b).max() < 1e-9 * a.max()
    assert abs(_nearest(a, target) - target) < 1e-5 * target


# ---- 2. tensor unknowns and definitions

def _viscous(viscous, geom, dt_T=False):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    if "T" in viscous:
        pb.add_unknown("T", "tensor")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_parameter("Ek", 1e-2, symbolic=False)
    pb.add_equation(f"dt(v) + 2*cross(ez, v) + grad(p) - Ek*({viscous}) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    if "T" in viscous:
        pb.add_equation(("dt(T) + " if dt_T else "") + "T - grad(v) - trans(grad(v)) = 0",
                        name="def")
    for at in _walls(geom):
        pb.add_stress_free("v", at=at)
    return pb


def _spectrum_near(cp, sigma, radius=0.6):
    lam, _V = _census(cp, sigma, resid=1e-8)
    return np.sort_complex(lam[np.abs(lam - sigma) < radius])


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_definition_is_eliminated(geom, m):
    """T - grad(v) - trans(grad(v)) = 0 classifies as a definition, T gets
    no blocks, and the pencil of dt(v) + ... - Ek div(T) = 0 is entrywise the
    strain-divergence pencil with the same spectrum to rounding."""
    grid = _grid(geom, 20, m + 4)
    cpT = _viscous("div(T)", geom).compile(grid, m=m, backend="sh_fd", gauge_unknowns=["p"])
    cpS = _viscous("div(grad(v) + trans(grad(v)))", geom).compile(grid, m=m, backend="sh_fd",
                                                                   gauge_unknowns=["p"])
    rec = cpT.classification
    assert rec.equations["def"]["class"] == "definition"
    assert rec.definitions == {"T": "def"}
    assert [(name, kind) for name, kind, _o, _s in cpT.layout] == [("v", "vector"), ("p", "scalar")]
    assert abs(cpT.A_total - cpS.A_total).max() <= 1e-14 * abs(cpS.A_total).max()
    assert abs(cpT.B_total - cpS.B_total).max() == 0.0
    sT, sS = _spectrum_near(cpT, -0.05 + 0.5j), _spectrum_near(cpS, -0.05 + 0.5j)
    assert len(sT) == len(sS) >= 1
    assert np.abs(sT - sS).max() <= 1e-12


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_tensor_under_dt_keeps_its_blocks(geom, m):
    """With dt(T) the tensor stays in the state on nine spin blocks
    T^{ab}: its rows carry I on T and -(Grad v + Grad v^T) on v, M is the
    identity on the T blocks, and the components absent at l < |spin|
    (spin pm1 at l = 0, spin pm2 at l < 2) are pinned."""
    grid = _grid(geom, 12, m + 3)
    cp = _viscous("div(T)", geom, dt_T=True).compile(grid, m=m, backend="sh_fd",
                                                      gauge_unknowns=["p"])
    assert cp.classification.equations["def"]["class"] == "evolution"
    layout = {name: (kind, off, size) for name, kind, off, size in cp.layout}
    assert layout["T"] == ("tensor", 4 * cp.nlr, 9 * cp.nlr)
    assert cp.N == 13 * cp.nlr
    nlr, Nr = cp.nlr, cp.Nr
    K, M = cp.assemble_evp_matrices()
    r0 = cp._eq_row_offset["def"]
    KT = K[r0:r0 + 9 * nlr, 4 * nlr:13 * nlr]
    MT = M[r0:r0 + 9 * nlr, 4 * nlr:13 * nlr]
    v = cp.system.unknowns["v"]
    strain = cp._lin(Grad(v)).blocks["v"] + cp._lin(Transpose(Grad(v))).blocks["v"]
    Kv = K[r0:r0 + 9 * nlr, :3 * nlr]
    live = np.ones(9 * nlr, bool)
    for c in range(9):
        n = max(0, abs(cp._spin_of("T", c)) - m)
        live[c * nlr:c * nlr + n * Nr] = False
    assert np.count_nonzero(~live) == Nr * {0: 4 + 2 * 2, 1: 2, 2: 0}[m]
    on, off = np.nonzero(live)[0], np.nonzero(~live)[0]
    eye = sps.identity(9 * nlr, format="csr")
    assert abs(KT - eye).max() == 0.0
    assert abs((MT - eye)[on]).max() == 0.0 and MT[off].nnz == 0
    assert abs((Kv + strain)[on]).max() < 1e-12 and Kv[off].nnz == 0


# ---- 3. the gauge scalar of the vector potential

def _decay_problem(formulation, geom):
    """tests/test_analytic_evp.py's free decay in a perfect conductor: the
    vector potential with psi and the Coulomb gauge, or b with a written
    Div(b) = 0 and no multiplier."""
    pb = Problem()
    pb.add_parameter("eta", 1.0, symbolic=False)
    if formulation == "A":
        pb.add_unknown("A", "vector")
        pb.add_unknown("psi", "scalar")
        pb.add_equation("dt(A) - eta*lap(A) - grad(psi) = 0", name="induction")
        pb.add_equation("div(A) = 0", name="coulomb")
        for wall in _walls(geom):
            pb.add_bc("dot(theta_hat, A) = 0", at=wall)
            pb.add_bc("dot(phi_hat, A) = 0", at=wall)
            pb.add_bc("psi = 0", at=wall)
        return pb
    pb.add_unknown("b", "vector")
    pb.add_equation("dt(b) - eta*lap(b) = 0", name="induction")
    pb.add_equation("div(b) = 0", name="solenoidal")
    for wall in _walls(geom):
        pb.add_bc("dot(r_hat, b) = 0", at=wall)
        pb.add_bc("dot(theta_hat, curl(b)) = 0", at=wall)
        pb.add_bc("dot(phi_hat, curl(b)) = 0", at=wall)
    return pb


def _decay_targets(geom, m):
    """k^2 of the toroidal (d(r T)/dr = 0) and poloidal (S = 0) families,
    l = max(m, 1) and the next."""
    ri = 0.0 if geom == "ball" else RI
    out = []
    for l in range(max(m, 1), max(m, 1) + 2):
        out.extend(k ** 2 for k in tae._radial_roots("rT", l, 2, ri=ri))
        out.extend(k ** 2 for k in tae._radial_roots("dirichlet", l, 2, ri=ri))
    return np.sort(out)[:6]


def _decay_run(formulation, geom, m, gauge=()):
    """(errors (2, 6), multiplier size (2, 6), divergence (2, 6)) of the
    target modes at N_r = 24 and 48."""
    targets = _decay_targets(geom, m)
    vec, mult = ("A", "psi") if formulation == "A" else ("b", "__div_b")
    errs, mults, divs = [], [], []
    for N in (24, 48):
        cp = _decay_problem(formulation, geom).compile(_grid(geom, N, m + 3), m=m, backend="sh_fd",
                                                       gauge_unknowns=list(gauge))
        assert [name for name, *_ in cp.layout] == [vec, mult]
        lam, V = _census(cp, float(np.mean(targets)), resid=1e-8)
        assert np.abs(lam.imag[np.abs(lam) < 200.0]).max() < 1e-8
        div = cp._div_vector()
        o, om = cp._offset[vec], cp._offset[mult]
        e, mu, dv = [], [], []
        for t in targets:
            j = np.argmin(np.abs(lam - t))
            e.append(abs(lam[j] - t) / t)
            x = V[o:o + 3 * cp.nlr, j]
            mu.append(np.abs(V[om:om + cp.nlr, j]).max() / np.abs(x).max())
            dv.append(np.abs(div @ x).max() / np.abs(x).max())
        errs.append(e); mults.append(mu); divs.append(dv)
    return np.array(errs), np.array(mults), np.array(divs)


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_vector_potential_free_decay(geom, m):
    """dt(A) - eta lap(A) - grad(psi) = 0, div(A) = 0 with tangential A = 0
    and psi = 0 at the walls compiles: psi has no equation of its own and
    its wall rows go on the Coulomb rows at the wall nodes (handing them to
    A's radial block, the tau owner's slot, loses an order and 10% of the
    poloidal rates at N_r = 48).  The decay rates are the toroidal
    d(rT)/dr = 0 and poloidal S = 0 families (shell l = 1: 3.9385, 43.8041
    and 43.1914, 50.5745) at the stencil order, with psi and div(A) of the
    target modes at the discretisation error."""
    errs, mults, divs = _decay_run("A", geom, m)
    _check(errs, 3e-4)
    assert mults[1].max() < 1e-3 and divs[1].max() < 1e-3
    assert mults[1].max() < mults[0].max() / 4 and divs[1].max() < divs[0].max() / 4


def test_vector_potential_gauge_row_is_skipped_with_walls():
    """psi declared a gauge unknown at m = 0 with psi = 0 walls: the walls
    fix its constant, so no mean-zero row replaces a Coulomb row (one
    would leave a spurious l = 0 mode with A_r content), and the spectrum
    is that of the undeclared problem."""
    a = _decay_problem("A", "shell").compile(_grid("shell", 24, 3), m=0, backend="sh_fd")
    b = _decay_problem("A", "shell").compile(_grid("shell", 24, 3), m=0, backend="sh_fd",
                                             gauge_unknowns=["psi"])
    assert abs(a.A_total - b.A_total).max() == 0.0


def test_gauge_row_at_m0_on_the_constraint_equation():
    """A declared gauge scalar without wall rows (p of the inertial ball)
    gets int p dV = 0 at m = 0 on the continuity row at the first node
    inside the wall: the pencil is regular and p of every finite mode has
    zero mean; without the row the constant p is a null vector."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    grid = _grid("ball", 12, 3)
    cp = pb.compile(grid, m=0, backend="sh_fd", gauge_unknowns=["p"])
    Nr = cp.Nr
    target = cp._eq_row_offset["continuity"] + Nr - 2
    w = cp.A_total[target].toarray().ravel()
    assert np.count_nonzero(w) == Nr and np.all(w[cp._offset["p"]:cp._offset["p"] + Nr] > 0)
    assert cp.B_total[target].nnz == 0
    K, M = cp.assemble_evp_matrices()
    assert np.linalg.matrix_rank((K - 0.7j * M).toarray()) == cp.N
    lam, V = _census(cp, 0.5j, resid=1e-8)
    p0 = V[cp._offset["p"]:cp._offset["p"] + Nr]
    assert np.abs(w[cp._offset["p"]:cp._offset["p"] + Nr] @ p0).max() < 1e-10 * np.abs(V).max()
    bare = pb.compile(grid, m=0, backend="sh_fd")
    x = np.zeros(bare.N)
    x[bare._offset["p"]:bare._offset["p"] + Nr] = 1.0
    assert np.abs(bare.A_total @ x).max() < 1e-12 * abs(bare.A_total).max()


def test_dipole_shift_A_matches_b():
    """The 2 kG conducting-dipole Rossby shift of tests/test_sh_fd_backend.py
    in the vector-potential formulation (b = Curl(A), tangential A = 0 and
    psi = 0) against the b formulation on this core at N_r = 16,
    ell_max = 16: -0.06889-0.00919j against -0.07060-0.01039j (the b value
    is -0.06957-0.01037j at 24 x 20 and -0.06912-0.01035j at 32 x 24, the
    Dedalus core -0.0686-0.0102j)."""
    from spdee.applications.solar_eigenmodes import add_magnetic_field
    import test_sh_fd_backend as T
    import test_staggered_mhd as S
    grid = ShFdGrid(N_r=16, ell_max=16, r_inner=T.R_LO, r_outer=T.R_HI, fd_order=4)
    h = T.solar_hydro_strings().build()
    om0 = T._rossby(compile_sh_fd(h, grid, m=8), 8)
    shift = {}
    for form in ("b", "A"):
        sysB = add_magnetic_field(h, S.potential_dipole(), Le=S.lehnert(2000.0), eta=T.EK,
                                  inv_rho_ratio=S.inv_rho_ratio_dipole(), bc="conducting",
                                  lorentz_sign=-1.0, formulation=form)
        cp = compile_sh_fd(sysB, grid, m=8)
        assert cp.classification_error is None
        shift[form] = T._rossby(cp, 8, guess=om0) - om0
    assert abs(shift["A"] - shift["b"]) < 0.05 * abs(shift["b"]), shift
    assert abs(shift["A"] - (-0.0686 - 0.0102j)) < 0.1 * 0.07, shift


# ---- 4. the involution as divergence cleaning

@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("geom", GEOMETRIES)
def test_magnetic_free_decay_b(geom, m):
    """dt(b) - eta lap(b) = 0 with Div(b) = 0 written and no multiplier:
    the classifier's DivergenceCleaning adds the multiplier __div_b to the
    state, + Grad of it to the induction equation, hosts the written row on
    its blocks and pins it to zero at every wall (on the divergence rows at
    the wall nodes).  The decay rates are the conducting-wall families at
    the stencil order; div(b) and the multiplier of the target modes are at
    the discretisation error.  The record's mean-zero gauge at m = 0 is not
    imposed: the Dirichlet walls fix the constant, and the extra row would
    replace a divergence row and admit a spurious l = 0 mode with b_r."""
    cp = _decay_problem("b", geom).compile(_grid(geom, 12, m + 3), m=m, backend="sh_fd")
    dc = cp.classification.divergence_cleaning["b"]
    assert (dc.multiplier, dc.evolution, dc.constraint) == ("__div_b", "induction", "solenoidal")
    assert cp.classification.involution_treatment == "saddle_point"
    Nr = cp.Nr
    om = cp._offset["__div_b"]
    for wall in dc.walls:
        i = 0 if wall == "r=0" else Nr - 1
        for a_l in range(cp.nl):
            row = cp._eq_row_offset["solenoidal"] + a_l * Nr + i
            assert cp.A_total[row].nnz == 1 and cp.A_total[row, om + a_l * Nr + i] == 1.0
    grad = cp._grad_scalar()
    ind = cp._eq_row_offset["induction"]
    nlr = cp.nlr
    interior = [c * nlr + a_l * Nr + i for c in range(3) for a_l in range(cp.nl)
                for i in range(1, Nr - 1) if a_l >= (1 if m == 0 else 0)]
    diff = cp.A_total[ind:ind + 3 * nlr, om:om + nlr] - grad
    assert abs(diff[interior]).max() == 0.0
    errs, mults, divs = _decay_run("b", geom, m)
    _check(errs, 3e-5)
    assert mults[1].max() < 1e-3 and divs[1].max() < 1e-3
    assert mults[1].max() < mults[0].max() / 4 and divs[1].max() < divs[0].max() / 4


def test_implied_involution_is_left_alone():
    """An induction equation without a written Div(b) = 0 (the Malkus and
    solar-eigenproblem MHD systems) keeps its pencil: the classifier records the implied
    involution but no multiplier is added."""
    pb = tae.malkus_problem(1e-2)
    cp = pb.compile(_grid("ball", 12, 4), m=1, backend="sh_fd", gauge_unknowns=["p"])
    assert cp.classification.involutions == {"b": "induction"}
    assert "b" in cp.classification.divergence_cleaning
    assert sorted(name for name, *_ in cp.layout) == ["b", "p", "v"]


@pytest.mark.parametrize("formulation", ["b", "A"])
def test_analytic_evp_free_decay_shell(formulation):
    """tests/test_analytic_evp.py::test_magnetic_free_decay_shell asserts the
    sh_fd refusals 'not square' (b) and "no equation hosts unknown 'psi'"
    (A); both formulations now compile, and this is the benchmark that file
    runs for the other cores, with its own targets, resolutions, tolerance
    and RMS order."""
    core = "sh_fd"
    targets = tae._decay_targets()
    spectra = tae._spectra(core, tae.decay_problem(formulation), "shell", 0,
                           sigma=float(np.mean(targets)), res=tae._DECAY_RES)
    errs = tae._run(f"free decay {formulation} shell m=0", core, targets, spectra, tae._rate,
                    tae._DECAY_TOL[core], res=tae._DECAY_RES)
    tae._rms_order(f"free decay {formulation} shell m=0", core, errs, res=tae._DECAY_RES)


def test_classifier_refusal_is_reported():
    """A system outside the classifier's vocabulary still compiles by count
    when square, and a non-square one reports the classifier's reason."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("f", "scalar")
    pb.add_equation("lap(v) = 0", name="a")
    pb.add_equation("lap(f) = 0", name="b")
    pb.add_bc("f = 0", at="outer")
    cp = pb.compile(_grid("ball", 12, 3), m=1, backend="sh_fd")
    assert cp.classification is None and "algebraic" in cp.classification_error
    pb = Problem()
    pb.add_unknown("b", "vector")
    pb.add_equation("dt(b) - lap(b) + b = 0", name="induction")
    pb.add_equation("div(b) = 0", name="solenoidal")
    pb.add_bc("dot(r_hat, b) = 0", at="outer")
    with pytest.raises(ValueError, match="not square.*not curls of the state"):
        pb.compile(_grid("ball", 12, 3), m=1, backend="sh_fd")


def test_symbolic_parameters_bind_in_composite_rows():
    """A symbolic coefficient in a composite row is bound at compile time
    through params, like the operators."""
    pb = _helmholtz("dot(r_hat, grad(p)) + h*p = 0", "ball", {"h": None})
    cp = pb.compile(_grid("ball", 16, 3), m=1, backend="sh_fd", params={"h": 3.0})
    ref = _helmholtz("dot(r_hat, grad(p)) + 3*p = 0", "ball").compile(_grid("ball", 16, 3), m=1,
                                                                        backend="sh_fd")
    assert abs(cp.A_total - ref.A_total).max() < 1e-12 * abs(ref.A_total).max()
    with pytest.raises(ValueError, match="unbound"):
        pb.compile(_grid("ball", 16, 3), m=1, backend="sh_fd")
