"""Constrained-transport MHD on the staggered (Arakawa-C) backend.

Checks, on the solar-eigenproblem Figure-2 compressible shell at m=8:

  * routing: an MHD system is recognized and compiles on the staggered path;
  * div(b) = 0 is an exact discrete identity of the induction and resistive
    blocks (the whole point of constrained transport);
  * Le -> 0 recovers the hydro Rossby mode to rounding; the shift is
    quadratic in Le;
"""

import numpy as np
import pytest
import sympy as sp

from spdee import Background, r, theta
from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.applications.solar_eigenmodes import make_polytrope_system, add_magnetic_field
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import EvpResult
from spdee.backends.fd2.staggered_mhd import CTGrid
from spdee.solvers import shift_invert

R_LO, R_HI, M = 0.71, 0.985, 8
# Solar-eigenproblem cross-validation parameters: nu = kappa = 2e12 cm^2/s, delta = 1e-6
RSUN, OMEGA0 = 7.0e10, 2 * np.pi * 456e-9
EK = 2e12 / (RSUN ** 2 * OMEGA0)
GAMMA, RHO_BC, P_BC, G_BC = 5 / 3, 0.2, 6e13, 5.2e4
H_BC = P_BC / (RHO_BC * G_BC) / RSUN

def _phi(rr):
    return 1 + (GAMMA - 1) / GAMMA * (R_LO / H_BC) * (R_LO / rr - 1)


def potential_dipole():
    """Unit-surface potential dipole (rmax/r)^3 (2 cos, sin, 0), as in the
    CT and Dedalus codes of the reference implementation."""
    f = (R_HI / r) ** 3
    return Background.Vector("B0", v_r=2 * sp.cos(theta) * f, v_theta=sp.sin(theta) * f, v_phi=0)


def inv_rho_ratio_dipole():
    """rho0(rmax)/rho0(r) for the polytrope (density reference at the
    surface, where the dipole amplitude is defined)."""
    rho = _phi(r) ** (1 / (GAMMA - 1))
    rho_ref = _phi(sp.Float(R_HI)) ** (1 / (GAMMA - 1))
    return Background.Scalar("inv_rho_ratio", expr=rho_ref / rho)


def lehnert(B0_gauss):
    """Le = v_A(rmax) / (Rsun Omega0) with cgs rho0 at the surface."""
    rho_cgs = RHO_BC * float(_phi(sp.Float(R_HI)) ** (1 / (GAMMA - 1)))
    vA = B0_gauss / np.sqrt(4 * np.pi * rho_cgs)
    return vA / (RSUN * OMEGA0)


def hydro_system(Ek=EK):
    """Solar-eigenproblem Figure-2 hydro system with the stratified viscous strain term,
    the form that matches the certified codes."""
    poly = PolytropeBackground(r_inner=R_LO, r_outer=R_HI, delta=1e-6)
    return make_polytrope_system(poly, m_val=M, Ek=Ek, Le=0.0,
                                     entropy_bc="neumann", geometry="shell",
                                     with_grad_lnrho_strain=(Ek != 0.0))


def mhd_system(B0_gauss, Ek=EK, Pm=1.0, bc="conducting", lorentz_sign=-1.0):
    Le = lehnert(B0_gauss)
    return add_magnetic_field(hydro_system(Ek), potential_dipole(), Le=Le, eta=Ek / Pm,
                              inv_rho_ratio=inv_rho_ratio_dipole(), bc=bc,
                              lorentz_sign=lorentz_sign), Le


def compile_stag(system, Nr, Nt, pairing="center"):
    """Nr, Nt are CELL counts (the staggered grid takes the collocated nodes
    as r-edges, so N_r = Nr + 1); ``pairing`` selects the Coriolis
    v_phi <- v_theta block (the reference implementation's recipe or the exact adjoint)."""
    from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
    grid = make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=R_LO, r_outer=R_HI)
    return compile_fd2_staggered(system, grid, m=M, order_r=2, order_theta=2,
                                 coriolis_pairing=pairing)


def _v0(n, seed=0):
    """Deterministic ARPACK start vector."""
    return np.random.default_rng(seed).standard_normal(n).astype(complex)


def rossby(A, B, omega_guess=-2 / (M + 1), k=8, damping=0.01):
    """omega of the mode nearest omega_guess (lambda = i omega). The shift
    sits ``damping`` below the guess in Im(omega): a shift on top of the
    Rossby mode itself stalls ARPACK on the finer MHD pencils, and the mode
    is still the nearest of the k returned."""
    lam, _ = shift_invert(A, B, 1j * (omega_guess - 1j * damping), k,
                          v0=_v0(A.shape[0]))
    om = EvpResult.eigenvalues_to_omega(lam)
    om = om[np.isfinite(om)]
    return om[np.argmin(np.abs(om - omega_guess))]


# ---------------------------------------------------------------------------

def test_constraints_from_tasks():
    """The CT inputs are read from the task list, not from names: b is the
    involution unknown of the induction equation, its walls are conducting
    (the tangential electric field of that same equation), and the hydro
    system carries no involution at all."""
    system, _ = mhd_system(1000.0)
    cp = compile_stag(system, 8, 16)
    assert cp.constraints["constraints"] == {"b": "involution"}
    assert cp.constraints["involutions"] == {"b": "induction"}
    assert cp.constraints["placement"]["b"] == "face"
    walls = cp.constraints["walls"]["vectors"]["b"]
    assert walls["kind"] == "conducting" and walls["tangential"] == "electric"
    assert len(walls["electric"]) == 2
    assert compile_stag(hydro_system(), 8, 16).constraints["involutions"] == {}


def test_pinned_bc_rejected():
    system, _ = mhd_system(1000.0, bc="pinned")
    with pytest.raises(NotImplementedError, match="pins its tangential components"):
        compile_stag(system, 8, 16)


def test_compiles_and_layout():
    system, _ = mhd_system(1000.0)
    cp = compile_stag(system, 8, 16)
    names = [f.name for f in cp.layout.fields]
    assert names == ["v_r", "v_t", "v_p", "rho1", "s1", "b_r", "b_t", "b_p"]
    assert cp.A_total.shape[0] == cp.layout.total_size


def test_div_b_exact_identity():
    """div(b) of the flux tendency produced by the induction and resistive
    blocks vanishes to rounding for every velocity / flux input that
    respects the pinned rings: b_r = 0 at the walls and the zero-area
    pole theta-faces (``constraints['pinned']``)."""
    # inviscid hydro (no DOF elimination) with a resistive magnetic sector
    Le = lehnert(1000.0)
    system = add_magnetic_field(hydro_system(Ek=0.0), potential_dipole(), Le=Le, eta=1e-4,
                                inv_rho_ratio=inv_rho_ratio_dipole(), bc="conducting")
    cp = compile_stag(system, 10, 20)
    A = cp.A_total.tocsr()
    lay = cp.layout
    ct = CTGrid(cp.grids, M)
    D = ct.divergence()                                # (n_cells, n_b)
    sl_b = slice(lay.slice_of("b_r").start, lay.slice_of("b_p").stop)
    Ab = A[sl_b, :]                                    # dt Phi rows (all columns)
    _, _, rf_wall = ct.wall_masks()
    wall = lay.slice_of("b_r").start + rf_wall - sl_b.start   # wall Phi_r within the b block
    pinned = np.asarray(cp.constraints["pinned"]) - sl_b.start
    pinned = pinned[(pinned >= 0) & (pinned < Ab.shape[0])]   # pole Phi_theta ring
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])
        x[sl_b] = 0.0                                  # velocity-driven (induction)
        tend = Ab @ x
        assert np.linalg.norm(D @ tend) < 1e-12 * max(1.0, np.linalg.norm(tend))
        y = np.zeros_like(x)
        yb = rng.standard_normal(Ab.shape[0]) + 0j
        yb[wall] = 0.0                                 # b_r = 0 at the walls
        yb[pinned] = 0.0                               # zero-area pole faces
        y[sl_b] = yb
        tend = Ab @ y                                  # flux-driven (resistive)
        assert np.linalg.norm(D @ tend) < 1e-12 * max(1.0, np.linalg.norm(tend))


def test_weak_field_limit_and_quadratic_scaling():
    hyd = compile_stag(hydro_system(), 10, 24)
    Kh, Mh = hyd.assemble_evp_matrices()
    om0 = rossby(Kh, Mh)
    shifts = []
    for B in (25.0, 50.0):                   # weak field: Le ~ 1e-3, 2e-3
        sysB, _ = mhd_system(B)
        cp = compile_stag(sysB, 10, 24)
        K, Mm = cp.assemble_evp_matrices()
        shifts.append(rossby(K, Mm, omega_guess=om0) - om0)
    # tiny field: hydro recovered
    sys0, _ = mhd_system(1.0)
    K, Mm = compile_stag(sys0, 10, 24).assemble_evp_matrices()
    assert abs(rossby(K, Mm, omega_guess=om0) - om0) < 1e-6
    # quadratic in B0 (the ratio drifts from 4 as Le^2 corrections grow:
    # 3.99 at 25 -> 50 G, 3.79 at 100 -> 200 G on this grid)
    ratio = abs(shifts[1]) / abs(shifts[0])
    assert 3.95 < ratio < 4.05, (shifts, ratio)


def solar_eigenproblem_hydro_strings():
    """The solar-eigenproblem hydro system in the solar-eigenproblem project's spelling, entered as
    strings; the certified configuration for the spectral core."""
    from spdee import Problem
    U = RSUN * OMEGA0
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
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="inner"); pb.add_bc("dot(r_hat, grad(S)) = 0", at="outer")
    return pb.build()


@pytest.mark.parametrize("formulation", ["A", "b"])
def test_shift_matches_dedalus_backend(formulation):
    """The same magnetic layer on the Dedalus core (vector potential with
    Coulomb gauge and integ(psi) = 0, or b with b_r = 0 and total tangential
    E = 0) gives the same field-induced Rossby shift as the staggered CT
    assembler (2 kG dipole: reference Dedalus -0.0676-0.0091j, staggered
    -0.0664-0.0084j, spdee Dedalus -0.0686-0.0102j)."""
    pytest.importorskip("dedalus")
    from spdee.backends.dedalus import DedalusGrid, compile_dedalus
    B0 = 2000.0
    Le = lehnert(B0)
    Nr, Nt = 12, 48
    K0, M0 = compile_stag(hydro_system(), Nr, Nt).assemble_evp_matrices()
    om_s0 = rossby(K0, M0)
    sysB, _ = mhd_system(B0)
    K1, M1 = compile_stag(sysB, Nr, Nt).assemble_evp_matrices()
    d_s = rossby(K1, M1, omega_guess=om_s0) - om_s0

    dg = DedalusGrid(N_r=16, N_theta=48, r_inner=R_LO, r_outer=R_HI)
    h = solar_eigenproblem_hydro_strings()
    cd0 = compile_dedalus(h, dg, m=M)
    om_d0 = rossby(cd0.A_total, cd0.B_total)
    sysD = add_magnetic_field(h, potential_dipole(), Le=Le, eta=EK,
                              inv_rho_ratio=inv_rho_ratio_dipole(), bc="conducting",
                              lorentz_sign=-1.0, formulation=formulation)
    cd1 = compile_dedalus(sysD, dg, m=M)
    d_d = rossby(cd1.A_total, cd1.B_total, omega_guess=om_d0) - om_d0
    assert abs(om_d0 - (-0.22060 - 0.01933j)) < 5e-5, om_d0
    assert abs(om_s0 - om_d0) < 1e-3, (om_s0, om_d0)
    assert abs(d_s - d_d) < 0.06 * abs(d_d), (d_s, d_d)


def potential_system(B0_gauss, Ek=EK, Pm=1.0):
    """The same magnetic layer in the vector-potential formulation: A on
    the faces with the Coulomb gauge Div(A) = 0 and the electric potential
    psi as its multiplier, b = Curl(A), the Lorentz force
    Cross(Curl(Curl(A)), B0), tangential A = 0 and psi = 0 at the walls."""
    Le = lehnert(B0_gauss)
    return add_magnetic_field(hydro_system(Ek), potential_dipole(), Le=Le, eta=Ek / Pm,
                              inv_rho_ratio=inv_rho_ratio_dipole(),
                              formulation="A"), Le


def test_b_and_A_formulations_agree_on_the_field_induced_shift():
    """The 2 kG dipole through the general assembler: the flux (b) and the
    vector-potential (A) formulations of the same physics give the same
    Rossby shift as the grid is refined, and each lands on the Dedalus
    core's value for its own formulation (test_shift_matches_dedalus_backend:
    reference Dedalus -0.0676-0.0091j for b, spdee Dedalus -0.0686-0.0102j for
    the vector potential)."""
    diffs, shifts = [], {}
    for Nr, Nt in ((10, 24), (20, 48)):
        Kh, Mh = compile_stag(hydro_system(), Nr, Nt).assemble_evp_matrices()
        om0 = rossby(Kh, Mh)
        out = {}
        for tag, system in (("b", mhd_system(2000.0)[0]),
                            ("A", potential_system(2000.0)[0])):
            cp = compile_stag(system, Nr, Nt)
            K, Mm = cp.assemble_evp_matrices()
            out[tag] = rossby(K, Mm, omega_guess=om0) - om0
        diffs.append(abs(out["b"] - out["A"]))
        shifts[(Nr, Nt)] = out
    assert diffs[1] < diffs[0] / 2.5, diffs
    fine = shifts[(20, 48)]
    assert abs(fine["b"] - (-0.0676 - 0.0091j)) < 5e-3, fine
    assert abs(fine["A"] - (-0.0686 - 0.0102j)) < 5e-3, fine


def test_flux_is_divergence_free():
    """The involution unknown carries a divergence helper; the vector
    potential's Coulomb gauge is exact instead."""
    system, _ = mhd_system(2000.0)
    cp = compile_stag(system, 10, 24)
    K, Mm = cp.assemble_evp_matrices()
    lam, V = shift_invert(K, Mm, 1j * (-2 / (M + 1)), 8, return_vectors=True,
                          v0=_v0(K.shape[0]))
    j = int(np.argmin(np.abs(EvpResult.eigenvalues_to_omega(lam) - (-2 / (M + 1)))))
    x = V[:, j] / np.linalg.norm(V[:, j])
    flux = np.concatenate([x[cp.layout.field(f"b_{c}").slice] for c in "rtp"])
    assert abs(cp.divergence(x)).max() < 1e-11 * abs(flux).max()
