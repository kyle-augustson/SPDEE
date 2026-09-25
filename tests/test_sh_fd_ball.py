"""The sh_fd core in the full sphere (r_inner = 0): parity closure of the
radial stencils across the origin (docs/ball_sh_fd.md).

Operator identities on polynomial regular fields are exact, the rank-2
closure of a tensor unknown included; the scalar
Helmholtz, vector-diffusion and inertial-mode benchmarks converge at the
stencil order, measured on resolution pairs inside the asymptotic range
(coarser pairs straddle a sign change of the eigenvalue error, where the
h^(p+1) term of the off-centred wall stencils competes with the origin
term) and only on modes clear of that crossing.  The compressible viscous
stress decays at the analytic toroidal rates and the collocated core's
spheroidal ones.
"""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

from spdee import Problem
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.sh_fd import ShFdGrid, compile_sh_fd, kcoef
from spdee.ir.operators import Div, Dot, Grad, Lap, LapH, Transpose
from spdee.solvers import dense
from test_sh_fd_backend import decay_rates, nearest_error, strain_problem, EK_VISC


def _root(fun, lo, hi):
    return brentq(fun, lo, hi, xtol=1e-15, rtol=1e-15)


# Dirichlet j_l(k) = 0, Neumann j_l'(k) = 0, stress-free toroidal k j_l'(k) = j_l(k)
DIRICHLET = {0: [np.pi, 2 * np.pi],
             1: [_root(lambda k: spherical_jn(1, k), 4, 5),
                 _root(lambda k: spherical_jn(1, k), 7, 8.5)],
             2: [_root(lambda k: spherical_jn(2, k), 5, 6.5)]}
NEUMANN = {0: [_root(lambda k: spherical_jn(0, k, True), 4, 5)],
           1: [_root(lambda k: spherical_jn(1, k, True), 1.5, 2.5)],
           2: [_root(lambda k: spherical_jn(2, k, True), 3, 4)]}


def _tor(l, lo, hi):
    return _root(lambda k: k * spherical_jn(l, k, True) - spherical_jn(l, k), lo, hi)


TOROIDAL = {1: [_tor(1, 5, 6.5), _tor(1, 8.5, 9.5)], 2: [_tor(2, 2, 3), _tor(2, 6.5, 7.5)],
            3: [_tor(3, 3.5, 4.2)]}


def _grid(N_r, ell_max, fd_order=4, **kw):
    return ShFdGrid(N_r=N_r, ell_max=ell_max, r_inner=0.0, r_outer=1.0, fd_order=fd_order, **kw)


def _helmholtz(bc, symbolic=False):
    pb = Problem(); pb.add_unknown("p", "scalar")
    if symbolic:
        pb.add_parameter("Ek")
        pb.add_equation("dt(p) - Ek*lap(p) = 0", name="helm")
    else:
        pb.add_equation("dt(p) - lap(p) = 0", name="helm")
    pb.add_bc("p = 0" if bc == "dirichlet" else "dot(r_hat, grad(p)) = 0", at="outer")
    return pb


def _diffusion(Ek):
    pb = Problem(); pb.add_unknown("v", "vector"); pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_equation("dt(v) - Ek*lap(v) = 0", name="momentum"); pb.add_stress_free("v", at="outer")
    return pb


def _inertial():
    pb = Problem(); pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


def _nearest(cp, target, k=6):
    """Relative error of the eigenvalue nearest ``target`` (pencil convention
    x ~ exp(-lambda t))."""
    res = cp.solve_evp(k=k, sigma=target + 0.05 * abs(target))
    lam = res.eigenvalues[np.isfinite(res.eigenvalues)]
    return abs(lam[np.argmin(np.abs(lam - target))] - target) / abs(target)


def _order(e_coarse, e_fine, ratio):
    return np.log(e_coarse / e_fine) / np.log(ratio)


def _check_order(p, fd_order, where):
    """Order 4 within 0.8; order 6 is measured on coarser pairs (finer ones
    reach the 5e-11 floor) where the same-signed h^7 term inflates it."""
    lo, hi = (0.8, 0.8) if fd_order == 4 else (0.8, 1.2)
    assert -lo < p - fd_order < hi, (where, p)


def test_grid_and_inner_bc():
    g = _grid(16, 4)
    assert g.ball and g.r[0] > 0 and np.isclose(g.r[-1], 1.0)
    assert np.allclose(np.diff(g.r), g.r[0] * 2)
    assert not ShFdGrid(N_r=16, ell_max=4, r_inner=0.5, r_outer=1.0).ball
    with pytest.raises(ValueError, match="r_inner"):
        ShFdGrid(N_r=16, ell_max=4, r_inner=-0.1, r_outer=1.0)
    with pytest.raises(ValueError, match="positive"):
        _grid(16, 4, nodes=np.linspace(0.0, 1.0, 16))
    pb = _helmholtz("dirichlet"); pb.add_bc("p = 0", at="inner")
    with pytest.raises(ValueError, match="inner wall"):
        pb.compile(g, m=1, backend="sh_fd")


@pytest.mark.parametrize("m", [0, 1, 2])
def test_operators_exact_on_polynomial_fields(m):
    """f = r^l (1 - r^2) Y_lm: Lap f = -(4l+6) r^l Y_lm; the spheroidal field
    grad f and the toroidal field curl(r^(l+1) (1 - r^2) Y_lm r_hat) obey
    div grad = Lap, curl grad = 0, Lap grad = grad Lap, div curl = 0,
    Lap t = curl(-(4l+6) r^(l+1) Y r_hat), curl curl t = -Lap t,
    grad(grad f) symmetric, LapH f = -l(l+1) f/r^2, and through the rank-2
    ghost closure Div(Grad u) = Lap u and Div(Grad u^T) = Grad(Div u), all
    exactly at fd_order 6 (degree <= 6)."""
    pb = Problem(); pb.add_unknown("v", "vector"); pb.add_unknown("f", "scalar")
    pb.add_equation("lap(v) = 0", name="a"); pb.add_equation("lap(f) = 0", name="b")
    pb.add_bc("f = 0", at="outer")
    cp = compile_sh_fd(pb.build(), _grid(16, m + 5, fd_order=6), m=m)
    r, Nr, nlr = cp.r, cp.Nr, cp.nlr
    f, v = cp.system.unknowns["f"], cp.system.unknowns["v"]
    grad_f = cp._lin(Grad(f)).blocks["f"]; lap_f = cp._lin(Lap(f)).blocks["f"]
    laph_f = cp._lin(LapH(f)).blocks["f"]
    div_v, curl_v, lap_v, grad_v = cp._div_vector(), cp._curl(), cp._lap_vector(), cp._grad_vector()
    div_grad = cp._lin(Div(Grad(v))).blocks["v"]
    div_grad_t = cp._lin(Div(Transpose(Grad(v)))).blocks["v"]
    for j, l in enumerate(cp.ls):
        if l + 3 > 6:
            continue
        fc = np.zeros(nlr, complex); fc[j * Nr:(j + 1) * Nr] = r ** l * (1 - r ** 2)
        ref = np.zeros(nlr, complex); ref[j * Nr:(j + 1) * Nr] = -(4 * l + 6) * r ** l
        assert np.abs(lap_f @ fc - ref).max() < 1e-9
        ang = np.zeros(nlr, complex)
        ang[j * Nr:(j + 1) * Nr] = -l * (l + 1.0) * r ** (l - 2) * (1 - r ** 2)
        assert np.abs(laph_f @ fc - ang).max() < 1e-9
        g = grad_f @ fc
        assert np.abs(div_v @ g - ref).max() < 1e-9
        assert np.abs(curl_v @ g).max() < 1e-9
        assert np.abs(lap_v @ g - grad_f @ ref).max() < 1e-9
        T = (grad_v @ g).reshape(3, 3, nlr)
        assert np.abs(T - T.transpose(1, 0, 2)).max() < 1e-9
        fields = [g]
        if l >= 1:
            psi = np.zeros(3 * nlr, complex); psi2 = np.zeros(3 * nlr, complex)
            sl = slice(2 * nlr + j * Nr, 2 * nlr + (j + 1) * Nr)
            psi[sl] = r ** (l + 1) * (1 - r ** 2); psi2[sl] = -(4 * l + 6) * r ** (l + 1)
            t = curl_v @ psi
            assert np.abs(div_v @ t).max() < 1e-9
            assert np.abs(lap_v @ t - curl_v @ psi2).max() < 1e-9
            assert np.abs(curl_v @ (curl_v @ t) + lap_v @ t).max() < 1e-9
            fields.append(t)
        for u in fields:
            assert np.abs(div_grad @ u - lap_v @ u).max() < 1e-9
            assert np.abs(div_grad_t @ u - grad_f @ (div_v @ u)).max() < 1e-9


@pytest.mark.parametrize("m", [0, 1, 2])
def test_rank2_regularity_at_the_origin(m):
    """A tensor unknown T on nine spin blocks with the rank-2 ghost closure
    (-1)^(l + n_r) onto the mirrored blocks: for u = grad(r^l (1 - r^2) Y_lm),
    l >= 2, the spin pm2 components of Grad(u) are
    k^mu_{l,mu} k^mu_{l,0} r^(l-2) (1 - r^2) and their closed derivative is
    the analytic one, Div(T) on those components is Lap(u), and Dot(B0, T),
    Dot(T, B0) with B0 = r r_hat pick r T^{r b} and r T^{a r}, all to
    roundoff at fd_order 6."""
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("f", "scalar"); pb.add_unknown("T", "tensor")
    pb.add_background("B0", v_r="r")
    pb.add_equation("dt(v) - div(T) = 0", name="a"); pb.add_equation("dt(f) - lap(f) = 0", name="b")
    pb.add_equation("dt(T) + T - grad(v) = 0", name="c")
    pb.add_bc("f = 0", at="outer")
    cp = compile_sh_fd(pb.build(), _grid(16, m + 5, fd_order=6), m=m)
    assert cp.N == 13 * cp.nlr and cp.classification.equations["c"]["class"] == "evolution"
    r, Nr, nlr = cp.r, cp.Nr, cp.nlr
    f, T, B0 = cp.system.unknowns["f"], cp.system.unknowns["T"], pb.namespace["B0"]
    grad_f = cp._lin(Grad(f)).blocks["f"]; grad_v = cp._grad_vector(); lap_v = cp._lap_vector()
    div_T = cp._lin(Div(T)).blocks["T"]
    dot_BT = cp._lin(Dot(B0, T)).blocks["T"]; dot_TB = cp._lin(Dot(T, B0)).blocks["T"]
    rr = np.tile(r, cp.nl)
    for j, l in enumerate(cp.ls):
        if l < 2 or l + 3 > 6:
            continue
        fc = np.zeros(nlr, complex); fc[j * Nr:(j + 1) * Nr] = r ** l * (1 - r ** 2)
        u = grad_f @ fc
        Tc = grad_v @ u
        for comp, mu in ((0, -1), (4, +1)):
            kk = kcoef(mu, l, mu) * kcoef(mu, l, 0)
            sl = slice(comp * nlr + j * Nr, comp * nlr + (j + 1) * Nr)
            assert np.abs(Tc[sl] - kk * r ** (l - 2) * (1 - r ** 2)).max() < 1e-12
            dT = np.zeros(nlr, complex)
            for c, blk in cp._dr_blocks(c1=1, comp=comp, rank=2).items():
                dT += blk @ Tc[c * nlr:(c + 1) * nlr]
            exact = kk * ((l - 2) * r ** (l - 3) - l * r ** (l - 1))
            assert np.abs(dT[j * Nr:(j + 1) * Nr] - exact).max() < 1e-9
        assert np.abs(div_T @ Tc - lap_v @ u).max() < 1e-9
        assert np.abs(dot_BT @ Tc - np.concatenate([rr] * 3) * Tc[6 * nlr:]).max() < 1e-12
        rhs = np.concatenate([rr * Tc[(3 * a + 2) * nlr:(3 * a + 3) * nlr] for a in range(3)])
        assert np.abs(dot_TB @ Tc - rhs).max() < 1e-12


@pytest.mark.parametrize("bc,fd_order,pair,tol", [
    ("dirichlet", 4, (48, 96), 5e-7), ("neumann", 4, (48, 96), 1e-6),
    ("dirichlet", 6, (16, 32), 1e-6), ("neumann", 6, (24, 48), 1e-7),
])
def test_scalar_helmholtz(bc, fd_order, pair, tol):
    """Lap(p) = -k^2 p on the unit ball at m = 0, 1, 2; the measured order on
    the modes clear of the error sign change matches fd_order."""
    targets = DIRICHLET if bc == "dirichlet" else NEUMANN
    clean = {("dirichlet", 4): [(0, 0), (0, 1), (1, 0)], ("neumann", 4): [(0, 0), (1, 0), (2, 0)],
             ("dirichlet", 6): [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
             ("neumann", 6): [(0, 0), (2, 0)]}[(bc, fd_order)]
    for m in (0, 1, 2):
        errs = {}
        for N_r in pair:
            cp = _helmholtz(bc).compile(_grid(N_r, 5, fd_order), m=m, backend="sh_fd")
            for l, ks in targets.items():
                for n, k in enumerate(ks):
                    if l >= m:
                        errs.setdefault((l, n), []).append(_nearest(cp, k ** 2))
        for (l, n), (e0, e1) in errs.items():
            assert e1 < tol, (m, l, n, e1)
            if (l, n) in clean:
                _check_order(_order(e0, e1, pair[1] / pair[0]), fd_order, (m, l, n, e0, e1))


def test_symbolic_parameter_dt_evp():
    """dt(p) - Ek lap(p) = 0 with Ek bound at compile time: lambda = Ek k^2."""
    Ek = 0.37
    cp = _helmholtz("dirichlet", symbolic=True).compile(_grid(48, 4), m=1, backend="sh_fd",
                                                        params={"Ek": Ek})
    for l, ks in DIRICHLET.items():
        for k in ks:
            if l >= 1:
                assert _nearest(cp, Ek * k ** 2) < 1e-5
    with pytest.raises(ValueError, match="unbound"):
        _helmholtz("dirichlet", symbolic=True).compile(_grid(16, 4), m=1, backend="sh_fd")


@pytest.mark.parametrize("fd_order,pair", [(4, (48, 96)), (6, (16, 32))])
def test_vector_diffusion_toroidal(fd_order, pair):
    """dt(v) = Ek lap(v), stress-free wall: toroidal rates Ek k^2 with
    k j_l'(k) = j_l(k) at m = 0, 1, 2, divergence-free and radial-free
    eigenvectors; the order is measured on the modes clear of the sign change."""
    Ek = 0.1
    clean = {4: [(2, 0), (3, 0)], 6: [(2, 0), (3, 0)]}[fd_order]
    for m in (0, 1, 2):
        errs = {}
        for N_r in pair:
            cp = _diffusion(Ek).compile(_grid(N_r, 5, fd_order), m=m, backend="sh_fd")
            K, M = cp.assemble_evp_matrices()
            div = cp._div_vector()
            for l, ks in TOROIDAL.items():
                for n, k in enumerate(ks):
                    if l < m:
                        continue
                    t = Ek * k ** 2
                    res = cp.solve_evp(k=4, sigma=t + 0.05 * t, return_eigenvectors=True)
                    lam = res.eigenvalues
                    j = int(np.nanargmin(np.abs(lam - t)))
                    errs.setdefault((l, n), []).append(abs(lam[j] - t) / t)
                    v = res.eigenvectors[j]["v"]; x = v.reshape(-1)
                    assert np.abs(div @ x).max() < 1e-8 * np.abs(x).max()
                    assert np.abs(v[2]).max() < 1e-8 * np.abs(v).max()
                    # u_+ = u_-: toroidal
                    assert np.abs(v[0] - v[1]).max() < 1e-8 * np.abs(v).max()
        for (l, n), (e0, e1) in errs.items():
            assert e1 < (1e-6 if fd_order == 4 else 2e-5), (m, l, n, e1)
            if (l, n) in clean:
                _check_order(_order(e0, e1, pair[1] / pair[0]), fd_order, (m, l, n, e0, e1))


@pytest.mark.parametrize("m", [1, 2])
def test_strain_viscous_decay(m):
    """dt(v) = Ek Div(Grad v + Grad v^T), stress-free wall: real non-negative
    spectrum with the rigid rotation at zero (m = 1), the toroidal rates
    Ek k^2 of k j_l'(k) = j_l(k) to 1e-6 at N_r = 48, and the six lowest
    rates, spheroidal included, against the collocated core (24 x 36,
    order 4) to its 5e-4."""
    pb = strain_problem(("outer",))
    sh = decay_rates(pb.compile(_grid(48, m + 4), m=m, backend="sh_fd"))
    fd = decay_rates(pb.compile(make_ball_grid(N_r=24, N_theta=36, r_outer=1.0), m=m,
                                order_r=4, order_theta=4))
    for l, ks in TOROIDAL.items():
        for k in ks:
            if l >= m and EK_VISC * k ** 2 < 3.0:
                assert nearest_error(sh, EK_VISC * k ** 2) < 1e-6, (l, k)
    if m == 1:
        assert sh[0] < 1e-9 and fd[0] < 1e-6
        sh, fd = sh[1:], fd[1:]
    assert np.abs(sh[:6] - fd[:6]).max() / fd[:6].max() < 5e-4, (sh[:6], fd[:6])


def test_m0_pins_absent_harmonics():
    """At m = 0 the spin pm1 coefficients of l = 0 are unit rows without mass,
    and the pencil is nonsingular for odd and even ell_max."""
    for ell_max in (3, 4):
        cp = _diffusion(0.1).compile(_grid(12, ell_max), m=0, backend="sh_fd")
        K, M = cp.assemble_evp_matrices()
        Nr, nlr = cp.Nr, cp.nlr
        for c in (0, 1):
            rows = K[c * nlr:c * nlr + Nr, :].toarray()
            assert np.allclose(rows, np.eye(Nr, K.shape[1], k=c * nlr))
            assert M[c * nlr:c * nlr + Nr, :].nnz == 0
        assert np.linalg.matrix_rank((K - (2 + 1j) * M).toarray()) == K.shape[0]


@pytest.mark.parametrize("m", [1, 2, 3, 4])
def test_rmode(m):
    """Sectoral r-mode omega = -2/(m+1) in the EvpResult convention of the shell test."""
    cp = _inertial().compile(_grid(24, m + 12), m=m, backend="sh_fd", gauge_unknowns=["p"])
    res = cp.solve_evp(k=8, sigma=1j * (-2.0 / (m + 1)) + 0.02)
    om = res.omega(); om = om[np.isfinite(om)]
    om = om[np.argmin(np.abs(om + 2.0 / (m + 1)))]
    assert abs(om + 2.0 / (m + 1)) < 1e-6, om


@pytest.mark.parametrize("m", [1, 2])
def test_zhang_2001_inertial_modes(m):
    """Zhang et al. (2001) polynomial inertial modes; their omega (time
    dependence exp(i(m phi + omega t))) is minus the EvpResult omega."""
    from tests.test_evp_zhang_2001 import ZHANG_TARGETS_M1, ZHANG_TARGETS_M2
    targets = {1: ZHANG_TARGETS_M1, 2: ZHANG_TARGETS_M2}[m]
    cp = _inertial().compile(_grid(24, m + 12), m=m, backend="sh_fd", gauge_unknowns=["p"])
    K, M = cp.assemble_evp_matrices()
    res = dense(K, M, sigma=0.5j, backend="auto")
    om = -1j * res.lam
    om = om.real[np.abs(om.imag) < 1e-3]
    for label, omega_z, tol in targets:
        got = -om[np.argmin(np.abs(om + omega_z))]
        assert abs(got - omega_z) < max(1e-4, 0.1 * tol) * max(abs(omega_z), 0.05), \
            (label, got, omega_z)


def test_cross_core_dedalus_helmholtz():
    """The Dedalus ball basis and sh_fd on the same Dirichlet Helmholtz
    problem at m = 1, eigenvalue by eigenvalue."""
    from spdee.backends.dedalus import DedalusGrid
    pb = _helmholtz("dirichlet")
    try:
        cpd = pb.compile(DedalusGrid(N_r=32, N_theta=16, r_inner=0.0, r_outer=1.0), m=1,
                         backend="dedalus")
    except Exception as exc:
        pytest.skip(f"the Dedalus core cannot compile a ball problem: {exc!r}")
    cps = pb.compile(_grid(96, 7, fd_order=6), m=1, backend="sh_fd")
    for sigma in (20.0, 40.0):
        ld = cpd.solve_evp(k=6, sigma=sigma).eigenvalues
        ls_ = cps.solve_evp(k=6, sigma=sigma).eigenvalues
        ld = np.sort(ld.real[np.isfinite(ld)]); ls_ = np.sort(ls_.real[np.isfinite(ls_)])
        common = [x for x in ld if abs(x - sigma) < 12]
        assert common
        for x in common:
            assert abs(ls_[np.argmin(np.abs(ls_ - x))] - x) < 1e-6 * x, (sigma, x)
