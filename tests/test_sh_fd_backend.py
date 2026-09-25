"""The sh_fd core (spin-weighted harmonics x radial FD).

The inviscid inertial problem gives the sectoral Rossby mode -2/(m+1); the
Bekki hydro system reproduces the certified m=8 Rossby mode; the
conducting-dipole shift agrees with the other cores; the tensor
divergence, LapH and tensor backgrounds satisfy exact identities and the
compressible viscous stress decays at the analytic toroidal rates and the
collocated core's spheroidal ones.
"""

import numpy as np
import pytest
import sympy as sp
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from spdee import Problem
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.sh_fd import ShFdGrid, compile_sh_fd
from spdee.ir.operators import Div, Dot, Grad, Lap, LapH, Transpose


def _rossby(cp, m, guess=None, k=8):
    omega_t = -2.0 / (m + 1) if guess is None else guess
    res = cp.solve_evp(k=k, sigma=1j * omega_t + 0.02)
    om = res.omega(); om = om[np.isfinite(om)]
    return om[np.argmin(np.abs(om - omega_t))]


def inertial_problem(Ek=None):
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if Ek is None:
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    else:
        pb.add_parameter("Ek", Ek, symbolic=False)
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    return pb


@pytest.mark.parametrize("m", [1, 2, 4])
def test_inviscid_rossby(m):
    grid = ShFdGrid(N_r=24, ell_max=m + 12, r_inner=0.71, r_outer=0.985, fd_order=4)
    cp = inertial_problem().compile(grid, m=m, backend="sh_fd", gauge_unknowns=["p"])
    om = _rossby(cp, m)
    assert abs(om.real + 2.0 / (m + 1)) < 2e-3, om
    assert abs(om.imag) < 2e-3


# Bekki constants (cgs) as in the seed's constants.py
RSUN, OMEGA0 = 7.0e10, 2 * np.pi * 456e-9
GAMMA, RHO_BC, P_BC, G_BC = 5 / 3, 0.2, 6e13, 5.2e4
R_LO, R_HI = 0.71, 0.985
H_BC = P_BC / (RHO_BC * G_BC) / RSUN
EK = 2e12 / (RSUN ** 2 * OMEGA0)
U = RSUN * OMEGA0


def solar_hydro_strings():
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("rhat", "scalar"); pb.add_unknown("S", "scalar")
    rr = pb.namespace["r"]
    zeta = 1 + (GAMMA - 1) / GAMMA * (R_LO / H_BC) * (R_LO / rr - 1)
    rho0 = RHO_BC * zeta ** (1 / (GAMMA - 1)); p0 = P_BC * zeta ** (GAMMA / (GAMMA - 1))
    pb.add_background("Cs2", expr=(GAMMA * p0 / rho0) / U ** 2)
    pb.add_background("gacc", expr=(G_BC * (rr / R_LO) ** -2) / (RSUN * OMEGA0 ** 2))
    dlnrho = sp.diff(sp.log(rho0), rr); dlnp = sp.diff(sp.log(p0), rr)
    pb.add_background("dlnrho", expr=dlnrho); pb.add_background("dlnp", expr=dlnp)
    pb.add_background("invHp", expr=-dlnp)
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("er_", v_r=1)
    pb.add_parameter("Ek", EK, symbolic=False); pb.add_parameter("delta", 1e-6, symbolic=False)
    pb.add_equation("dt(v) + grad(Cs2*(rhat+S)) - gacc*S*er_ + 2*cross(ez, v)"
                    " - Ek*(lap(v) + (1/3)*grad(div(v)) + dlnrho*dot(er_, grad(v) + trans(grad(v)))"
                    " - (2/3)*dlnrho*div(v)*er_) = 0", name="momentum")
    pb.add_equation("dt(rhat) + div(v) + dlnrho*dot(er_, v) = 0", name="continuity")
    pb.add_equation("dt(S) - delta*invHp*dot(er_, v) - Ek*(lap(S) + dlnp*dot(er_, grad(S))) = 0",
                    name="energy")
    pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="inner")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="outer")
    return pb


def test_solar_rossby_matches_certified():
    """N_r=32/order 4: -0.220586-0.019366j; N_r=48/order 4: -0.220593-0.019341j;
    N_r=48/order 2: -0.220597-0.019329j; the Bekki project's dense Dedalus
    32x144 value is -0.220599-0.019329j."""
    grid = ShFdGrid(N_r=48, ell_max=8 + 24, r_inner=R_LO, r_outer=R_HI, fd_order=4)
    cp = solar_hydro_strings().compile(grid, m=8, backend="sh_fd")
    om = _rossby(cp, 8)
    ref = -0.220599 - 0.019329j
    assert abs(om - ref) < 5e-5, (om, ref)


def test_layout_and_unpack():
    grid = ShFdGrid(N_r=12, ell_max=10, r_inner=0.71, r_outer=0.985, fd_order=2)
    cp = inertial_problem().compile(grid, m=2, backend="sh_fd", gauge_unknowns=["p"])
    assert cp.N == 4 * cp.nl * cp.Nr
    res = cp.solve_evp(k=4, sigma=1j * (-2 / 3) + 0.02, return_eigenvectors=True)
    j = int(np.nanargmin(np.abs(res.omega() + 2 / 3)))
    v = res.eigenvectors[j]["v"]
    assert v.shape == (3, cp.nl, cp.Nr)
    # sectoral Rossby: l = m dominates, and it is nearly toroidal (|v_r| small)
    energy_l = np.abs(v[:2]).sum(axis=(0, 2))
    assert energy_l[0] > 0.9 * energy_l.sum()
    assert np.abs(v[2]).max() < 0.05 * np.abs(v[:2]).max()


@pytest.mark.slow
def test_dipole_shift_matches_other_cores():
    """The 2 kG conducting-dipole Rossby shift on sh_fd (N_r=32, ell_max=24:
    -0.06912-0.01035j) against spdee Dedalus -0.06860-0.01023j, seed Dedalus
    -0.06763-0.00912j and staggered -0.06639-0.00843j.  Slow (~5 min): the
    dipole couples all l at every radius."""
    from spdee.applications.solar_eigenmodes import add_magnetic_field
    import tests.test_staggered_mhd as T
    grid = ShFdGrid(N_r=32, ell_max=24, r_inner=R_LO, r_outer=R_HI, fd_order=4)
    h = solar_hydro_strings().build()
    om0 = _rossby(compile_sh_fd(h, grid, m=8), 8)
    sysB = add_magnetic_field(h, T.potential_dipole(), Le=T.lehnert(2000.0), eta=EK,
                              inv_rho_ratio=T.inv_rho_ratio_dipole(), bc="conducting",
                              lorentz_sign=-1.0)
    d = _rossby(compile_sh_fd(sysB, grid, m=8), 8, guess=om0) - om0
    assert abs(d - (-0.0686 - 0.0102j)) < 0.05 * 0.07, d


# ---- tensors and LapH

def _tensor_problem(m, ell_max=6, N_r=16, fd_order=6, r_inner=0.4):
    pb = Problem(); pb.add_unknown("v", "vector"); pb.add_unknown("f", "scalar")
    pb.add_background("I0", components={("r", "r"): 1, ("theta", "theta"): 1, ("phi", "phi"): 1})
    pb.add_background("Rr", components={("r", "r"): 1})
    pb.add_background("T0", components={("r", "theta"): "r*sin(theta)",
                                        ("theta", "r"): "cos(theta)",
                                        ("phi", "phi"): "r**2", ("theta", "phi"): "1 + r"})
    pb.add_equation("lap(v) = 0", name="a"); pb.add_equation("lap(f) = 0", name="b")
    pb.add_bc("f = 0", at="outer")
    grid = ShFdGrid(N_r=N_r, ell_max=ell_max, r_inner=r_inner, r_outer=1.0, fd_order=fd_order)
    return compile_sh_fd(pb.build(), grid, m=m), pb.namespace


@pytest.mark.parametrize("m", [0, 1, 2])
def test_tensor_divergence_identities(m):
    """Div(Grad u) = Lap u and Div(Grad u^T) = Grad(Div u) on the polynomial
    fields grad(r^l (1 - r^2) Y_lm) and curl(r^(l+1) (1 - r^2) Y_lm r_hat),
    exact at fd_order 6; LapH is the -l(l+1)/r^2 part of Lap."""
    cp, ns = _tensor_problem(m)
    r, Nr, nlr = cp.r, cp.Nr, cp.nlr
    f, v = ns["f"], ns["v"]
    grad_f = cp._lin(Grad(f)).blocks["f"]
    lap_f, laph_f = cp._lin(Lap(f)).blocks["f"], cp._lin(LapH(f)).blocks["f"]
    div_v, curl_v, lap_v = cp._div_vector(), cp._curl(), cp._lap_vector()
    div_grad = cp._lin(Div(Grad(v))).blocks["v"]
    div_grad_t = cp._lin(Div(Transpose(Grad(v)))).blocks["v"]
    for j, l in enumerate(cp.ls):
        if l + 3 > 6:
            continue
        fc = np.zeros(nlr, complex); fc[j * Nr:(j + 1) * Nr] = r ** l * (1 - r ** 2)
        ref = np.zeros(nlr, complex)
        ref[j * Nr:(j + 1) * Nr] = -l * (l + 1.0) * r ** (l - 2) * (1 - r ** 2)
        assert np.abs(laph_f @ fc - ref).max() < 1e-9
        radial = np.zeros(nlr, complex)
        radial[j * Nr:(j + 1) * Nr] = (l * (l + 1.0) - (l + 2) * (l + 3.0) * r ** 2) * r ** (l - 2)
        assert np.abs((lap_f - laph_f) @ fc - radial).max() < 1e-9
        fields = [grad_f @ fc]
        if l >= 1:
            psi = np.zeros(3 * nlr, complex)
            psi[2 * nlr + j * Nr:2 * nlr + (j + 1) * Nr] = r ** (l + 1) * (1 - r ** 2)
            fields.append(curl_v @ psi)
        for u in fields:
            assert np.abs(div_grad @ u - lap_v @ u).max() < 1e-9
            assert np.abs(div_grad_t @ u - grad_f @ (div_v @ u)).max() < 1e-9


@pytest.mark.parametrize("m", [0, 1, 2])
def test_tensor_backgrounds(m):
    """dot(I, v) = dot(v, I) = v; dot(r_hat r_hat, v) keeps v_r only;
    dot(v, T0) = dot(trans(T0), v) for a non-symmetric T0; T0 * f is a
    tensor whose divergence contracts with a vector background."""
    cp, ns = _tensor_problem(m)
    nlr, Nr = cp.nlr, cp.Nr
    v, f, I0, Rr, T0 = (ns[n] for n in ("v", "f", "I0", "Rr", "T0"))
    live = np.ones(3 * nlr, bool)
    if m == 0:
        live[:Nr] = live[nlr:nlr + Nr] = False       # spin pm1 has no l = 0
    eye = np.eye(3 * nlr)[live][:, live]
    for e in (Dot(I0, v), Dot(v, I0)):
        assert np.abs(cp._lin(e).blocks["v"].toarray()[live][:, live] - eye).max() < 1e-12
    proj = cp._lin(Dot(Rr, v)).blocks["v"].toarray()
    ref = np.zeros((3 * nlr, 3 * nlr)); ref[2 * nlr:, 2 * nlr:] = np.eye(nlr)
    assert np.abs(proj - ref).max() < 1e-12
    left = cp._lin(Dot(Transpose(T0), v)).blocks["v"]
    right = cp._lin(Dot(v, T0)).blocks["v"]
    assert abs(left - right).max() < 1e-12
    assert abs(cp._lin(Dot(T0, v)).blocks["v"] - right).max() > 1e-3
    prod = cp._lin(T0 * f)
    assert prod.kind == "tensor" and prod.blocks["f"].shape == (9 * nlr, nlr)
    assert cp._lin(Div(T0 * f)).kind == "vector"


# ---- compressible viscous stress

EK_VISC = 0.1


def strain_problem(walls):
    """dt(v) = Ek Div(Grad v + Grad v^T) with stress-free walls: the wall
    term of the energy balance vanishes, so the spectrum is real and
    non-negative with the l = 1 rigid rotation at zero."""
    pb = Problem(); pb.add_unknown("v", "vector"); pb.add_parameter("Ek", EK_VISC, symbolic=False)
    pb.add_equation("dt(v) - Ek*div(grad(v) + trans(grad(v))) = 0", name="momentum")
    for at in walls:
        pb.add_stress_free("v", at=at)
    return pb


def toroidal_rates_shell(l, a, n=2):
    """Ek k^2 with T = alpha j_l(k r) + beta y_l(k r) and d/dr (T/r) = 0 at
    r = a and r = 1: the zero-tangential-stress toroidal modes."""
    def g(fn, k, r):
        return k * r * fn(l, k * r, derivative=True) - fn(l, k * r)

    def det(k):
        return (g(spherical_jn, k, a) * g(spherical_yn, k, 1.0)
                - g(spherical_yn, k, a) * g(spherical_jn, k, 1.0))
    ks = np.linspace(0.5, 40, 4000); val = det(ks); out = []
    for i in np.nonzero(np.sign(val[:-1]) != np.sign(val[1:]))[0][:n]:
        out.append(brentq(det, ks[i], ks[i + 1], xtol=1e-14))
    return EK_VISC * np.array(out) ** 2


def decay_rates(cp, k=16, sigma=0.4):
    lam = cp.solve_evp(k=k, sigma=sigma).eigenvalues
    lam = lam[np.isfinite(lam)]
    assert np.abs(lam.imag).max() < 1e-9
    assert lam.real.min() > -1e-9
    return np.sort(lam.real)


def nearest_error(rates, target):
    return abs(rates[np.argmin(np.abs(rates - target))] - target) / target


@pytest.mark.parametrize("m", [1, 2])
def test_strain_viscous_decay_shell(m):
    """Toroidal rates to 1e-6 at N_r = 48 and the six lowest rates, spheroidal
    included, against the collocated core (24 x 36, order 4) to its 5e-4."""
    a = 0.5
    pb = strain_problem(("inner", "outer"))
    sh = decay_rates(pb.compile(ShFdGrid(N_r=48, ell_max=m + 4, r_inner=a, r_outer=1.0), m=m,
                                backend="sh_fd"))
    fd = decay_rates(pb.compile(make_shell_grid(N_r=24, N_theta=36, r_inner=a, r_outer=1.0), m=m,
                                order_r=4, order_theta=4))
    for l in range(max(m, 2), m + 3):
        for target in toroidal_rates_shell(l, a):
            if target < 3.0:
                assert nearest_error(sh, target) < 1e-6, (l, target)
    if m == 1:
        assert sh[0] < 1e-9 and fd[0] < 1e-6
        sh, fd = sh[1:], fd[1:]
    assert np.abs(sh[:6] - fd[:6]).max() / fd[:6].max() < 5e-4, (sh[:6], fd[:6])

