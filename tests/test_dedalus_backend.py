"""The Dedalus spectral backend: IR -> Dedalus EVP -> (L_min, M_min) pencil.

Skips when Dedalus is not importable.  Checks: the inviscid inertial
problem entered as strings gives the sectoral Rossby mode omega = -2/(m+1)
to spectral accuracy; the viscous stress-free problem agrees with the fd2
collocated backend; eigenvectors come back on the meridional grid; the
registry and ``Problem.compile(backend='dedalus')`` route correctly.
"""

import numpy as np
import pytest

pytest.importorskip("dedalus")

import spdee
from spdee import Problem
from spdee.backends.dedalus import DedalusGrid
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import EvpResult


def _inertial(with_Ek=None, stress_free=False):
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if with_Ek is not None:
        pb.add_parameter("Ek", with_Ek)
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
    else:
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    if stress_free:
        pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    else:
        pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


GRID = DedalusGrid(N_r=12, N_theta=24, r_inner=0.71, r_outer=0.985)


def _rossby(cp, m):
    omega_t = -2.0 / (m + 1)
    res = cp.solve_evp(k=6, sigma=1j * omega_t + 0.02)
    assert isinstance(res, EvpResult)
    om = res.omega()
    om = om[np.isfinite(om)]
    return om[np.argmin(np.abs(om - omega_t))], omega_t


@pytest.mark.parametrize("m", [1, 2, 4])
def test_inviscid_rossby_spectral(m):
    cp = _inertial().compile(GRID, m=m, backend="dedalus", gauge_unknowns=["p"])
    om, omega_t = _rossby(cp, m)
    assert abs(om.real - omega_t) < 1e-6, (om, omega_t)
    assert abs(om.imag) < 1e-6


def test_tau_layout_inviscid():
    cp = _inertial().compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    names = [t.name for t in cp.tau_fields]
    assert names == ["tau_v1", "tau_v2", "tau_p_gauge"]


def test_viscous_stress_free_matches_fd2():
    """Viscous m=2 Rossby mode: Dedalus vs the fd2 collocated backend."""
    Ek = 1e-3
    cpd = _inertial(Ek, stress_free=True).compile(GRID, m=2, backend="dedalus",
                                                  gauge_unknowns=["p"])
    assert [t.name for t in cpd.tau_fields] == ["tau_v1", "tau_v2", "tau_p_gauge"]
    omd, omega_t = _rossby(cpd, 2)
    grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.71, r_outer=0.985)
    cpf = _inertial(Ek, stress_free=True).compile(grid, m=2, gauge_unknowns=["p"])
    omf, _ = _rossby(cpf, 2)
    assert abs(omd - omf) < 2e-2 * abs(omega_t), (omd, omf)
    assert omd.imag < 0 and omf.imag < 0            # damped in both


def test_eigenvector_fields():
    cp = _inertial().compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    res = cp.solve_evp(k=4, sigma=1j * (-2 / 3) + 0.02, return_eigenvectors=True)
    j = int(np.nanargmin(np.abs(res.omega() + 2 / 3)))
    fields = res.eigenvectors[j]
    assert set(fields) == {"v", "p"}
    assert fields["v"].shape == (3, GRID.N_theta, GRID.N_r)
    assert fields["p"].shape == (GRID.N_theta, GRID.N_r)
    # the sectoral Rossby mode is nearly toroidal: |v_r| << |v_phi|
    vr = np.abs(fields["v"][2]).max(); vphi = np.abs(fields["v"][0]).max()
    assert vr < 0.05 * vphi


def test_registry_and_grid_from_fd2():
    system = _inertial().build(gauge_unknowns=["p"])
    grid = make_shell_grid(N_r=12, N_theta=24, r_inner=0.71, r_outer=0.985)
    cp = spdee.compile_problem(system, backend="dedalus", grid=grid, m=2)
    assert cp.grid.N_r == 12 and cp.grid.geometry == "shell"
    K, M = cp.assemble_evp_matrices()
    assert K.shape == M.shape and K.shape[0] > 0


def test_symbolic_parameter_binding():
    pb = _inertial(1e-3, stress_free=True)
    a = pb.compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    pb.set_parameter("Ek", 2e-3)
    b = pb.compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    assert abs(a.A_total - b.A_total).max() > 0


def test_scalar_background_and_numeric_component():
    """Stratified inertial problem rho0(r)*dt(v): a SymPy scalar background and
    the same background given as a radial interpolator must agree, and the
    Dedalus result must match fd2."""
    from spdee.ir.backgrounds import radial_interpolator
    rr = np.linspace(0.71, 0.985, 200)
    def make(rho_spec, drho_spec):
        pb = Problem()
        pb.add_unknown("v", "vector"); pb.add_unknown("p", "scalar")
        pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_background("rho0", expr=rho_spec)
        pb.add_background("grad_rho0", v_r=drho_spec)
        pb.add_equation("rho0*dt(v) + 2*rho0*cross(Omega, v) + grad(p) = 0", name="momentum")
        # div(rho0*v) written out: fd2 wants Div of a bare unknown
        pb.add_equation("rho0*div(v) + dot(grad_rho0, v) = 0", name="continuity")
        pb.add_bc("dot(r_hat, v) = 0", at="inner"); pb.add_bc("dot(r_hat, v) = 0", at="outer")
        return pb
    sym = make("exp(-3*r)", "-3*exp(-3*r)")
    num = make(radial_interpolator(rr, np.exp(-3 * rr)),
               radial_interpolator(rr, -3 * np.exp(-3 * rr)))
    a = sym.compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    b = num.compile(GRID, m=2, backend="dedalus", gauge_unknowns=["p"])
    oma, t = _rossby(a, 2); omb, _ = _rossby(b, 2)
    assert abs(oma - omb) < 1e-6
    grid = make_shell_grid(N_r=24, N_theta=48, r_inner=0.71, r_outer=0.985)
    omf, _ = _rossby(sym.compile(grid, m=2, gauge_unknowns=["p"]), 2)
    assert abs(oma - omf) < 2e-2 * abs(t), (oma, omf)


def test_shell_helmholtz_decoupled_in_l():
    """A problem without theta-dependent coefficients decouples in l and
    Dedalus keys its subproblems by l alone; the pencil of m is the block
    diagonal over l >= m: Dirichlet Helmholtz on a shell, lambda = k^2 with
    j_l(k r_i) y_l(k r_o) = j_l(k r_o) y_l(k r_i)."""
    from scipy.optimize import brentq
    from scipy.special import spherical_jn, spherical_yn
    from spdee.solvers import dense
    r_i, r_o = 0.5, 1.0
    pb = Problem()
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(p) - lap(p) = 0")
    pb.add_bc("p = 0", at="inner")
    pb.add_bc("p = 0", at="outer")
    cp = pb.compile(DedalusGrid(N_r=16, N_theta=12, r_inner=r_i, r_outer=r_o), m=1, backend="dedalus")
    assert [t.name for t in cp.tau_fields] == ["tau_p1", "tau_p2"]
    K, M = cp.assemble_evp_matrices()
    lam = np.sort(dense(K, M, sigma=-1.0, backend="numpy").lam.real)[:4]
    target = []
    for ell in (1, 2, 3, 4):
        def f(k, ell=ell):
            return (spherical_jn(ell, k * r_i) * spherical_yn(ell, k * r_o)
                    - spherical_jn(ell, k * r_o) * spherical_yn(ell, k * r_i))
        ks = np.linspace(1.0, 20.0, 2000)
        v = f(ks)
        for i in np.nonzero(np.sign(v[:-1]) != np.sign(v[1:]))[0][:2]:
            target.append(brentq(f, ks[i], ks[i + 1]) ** 2)
    target = np.sort(target)[:4]
    assert np.abs(lam - target).max() < 1e-8 * target.max(), (lam, target)


def test_equation_weight_shell_helmholtz():
    """A radial equation weight w(r) multiplies the equation before the taus
    are added: a different pencil, the same resolved eigenvalues (Dirichlet
    Helmholtz on a shell, as above); a weight in theta is refused."""
    from spdee.solvers import dense
    r_i, r_o = 0.5, 1.0
    grid = DedalusGrid(N_r=16, N_theta=12, r_inner=r_i, r_outer=r_o)
    lams, Ks = [], []
    for weight in (None, "r**2*(2 - r)"):
        pb = Problem()
        pb.add_unknown("p", "scalar")
        eq = pb.add_equation("dt(p) - lap(p) = 0", weight=weight)
        assert (eq.weight is None) == (weight is None)
        pb.add_bc("p = 0", at="inner")
        pb.add_bc("p = 0", at="outer")
        K, M = pb.compile(grid, m=1, backend="dedalus").assemble_evp_matrices()
        Ks.append(K)
        lams.append(np.sort(dense(K, M, sigma=-1.0, backend="numpy").lam.real)[:4])
    assert abs(Ks[0] - Ks[1]).max() > 1e-3
    assert np.abs(lams[0] - lams[1]).max() < 1e-8 * lams[0].max(), lams
    pb = Problem()
    pb.add_unknown("p", "scalar")
    with pytest.raises(ValueError):
        pb.add_equation("dt(p) - lap(p) = 0", weight="r*cos(theta)")


def test_equation_weight_folds_into_coefficients():
    """The weight multiplies each term's coefficient before Dedalus expands
    it: with w = (1 + r)^2 and the coefficient c = 30 / (1 + r)^2 the
    weighted coefficients are the quadratic w and the constant 30, exact at
    an ncc_cutoff of 1e-2, which truncates the unweighted c."""
    from spdee.solvers import dense
    grid = DedalusGrid(N_r=24, N_theta=12, r_inner=0.5, r_outer=1.0)

    def lams(weight, cutoff):
        pb = Problem()
        pb.add_unknown("p", "scalar")
        pb.add_equation("dt(p) - lap(p) + 30*(1 + r)**(-2)*p = 0", weight=weight)
        pb.add_bc("p = 0", at="inner")
        pb.add_bc("p = 0", at="outer")
        K, M = pb.compile(grid, m=1, backend="dedalus", ncc_cutoff=cutoff).assemble_evp_matrices()
        return np.sort(dense(K, M, sigma=-1.0, backend="numpy").lam.real)[:4]

    ref = lams(None, 1e-14)
    scale = np.abs(ref).max()
    assert np.abs(lams(None, 1e-2) - ref).max() > 1e-6 * scale
    assert np.abs(lams("(1 + r)**2", 1e-2) - ref).max() < 1e-9 * scale
