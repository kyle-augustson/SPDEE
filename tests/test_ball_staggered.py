"""Full-sphere (ball, r_inner = 0) checks of the staggered Arakawa-C core.

Resolutions are quoted as radial cells × θ cells (``make_ball_grid`` takes
the r-edges, so N_r = cells + 1). The audit of the origin treatment is in
docs/ball_staggered.md; this module pins it numerically:

  * structure at 8×16 and 12×24: the origin ring rows, the receive-only
    origin v_r, the exact identities (curl∘grad = 0 on the weightful
    curl grid, Div = −W_p⁻¹Gᵀ W_v), the Coriolis skew structure, the
    definiteness of the viscous and thermal Gram operators, no empty
    rows in the compressible pencil, the m = 0 pressure gauge;
  * Zhang et al. (2001) inertial modes at m = 1, 2 on 16×32 and 24×48:
    the RMS error over the target set and its order, with the
    discrimination against random control frequencies reported (the band
    is dense here, so a per-target threshold is not evidence);
  * the compressible ball on a regular polytrope-like background:
    inertial modes against the incompressible assembler at Ek = 0, δ = 0
    and the viscous convective ridge converging over three resolutions;
  * thermal decay rates against j_l / j_l' zeros (Dirichlet / Neumann);
  * toroidal viscous decay rates against the stress-free and no-slip
    Bessel conditions, with the divergence-free check;
  * the composite wall rows at second order: Robin against
    k j_l'(k) + h j_l(k) = 0, Navier slip against
    (1 - ls) j_l(k) + ls k j_l'(k) = 0 (its limits the no-slip and
    stress-free pencils bitwise), the normal-stress wall against the
    traction-free Bessel determinant;
  * free magnetic decay in a perfectly conducting ball at m = 0, 1, 2 in
    the flux (constrained-transport) and the vector-potential
    formulation, against the Bessel conditions, at second order;
  * constrained-transport MHD on the ball: the compressible Bekki system
    with a magnetic layer compiles with the origin ring pinned and its
    eigenvectors divergence-free.

Frequencies: the pencil convention is ω = −iλ (``EvpResult``); the Zhang
targets use ω_Z = −Im λ.
"""

import numpy as np
import scipy.sparse as sp
import sympy as smp
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn

from spdee import (
    Unknown, Cross, Curl, Grad, Div, Dot, Lap, Dt, Equation, EquationSystem,
    BoundaryCondition, Boundary, Background, r_hat, theta_hat, phi_hat,
    r, theta,
)
from spdee.ir.equations import ImpenetrableStressFree
from spdee.ir.operators import Transpose
from spdee.applications.solar_eigenmodes import add_magnetic_field
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.compiled_problem import EvpResult
from spdee.backends.fd2.staggered_compile import (
    StaggeredGrids, compile_fd2_staggered,
    _assemble_curl_blocks, _curl_output_norm, _eliminate_wall_vr,
    build_staggered_grad_div_pair,
    build_staggered_scalar_laplacian, build_staggered_weighted_scalar_laplacian,
    build_staggered_weighted_viscous_gram,
)
from spdee.solvers import dense, shift_invert
from test_evp_zhang_2001 import (
    _build_system as zhang_system, ZHANG_TARGETS_M1, ZHANG_TARGETS_M2,
)

RES_STRUCT = ((8, 16), (12, 24))
RES_BENCH = ((12, 24), (24, 48))


def ball(Nr, Nt):
    return make_ball_grid(N_r=Nr + 1, N_theta=Nt, r_outer=1.0)


def order_of(e_coarse, e_fine, ratio):
    return np.log(e_coarse / e_fine) / np.log(ratio)


def roots(f, n=2, lo=0.05, hi=16.0):
    """First n positive roots of a scalar function by sign change."""
    f = np.vectorize(f)
    ks = np.linspace(lo, hi, 6000)
    v = f(ks)
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
            out.append(brentq(f, ks[i], ks[i + 1]))
    return out[:n]


def make_regular_ball_system(m, Ek=0.0, delta=0.0, cs2_0=1.2e4, beta=0.5,
                             gamma=5.0 / 3.0, strain=True, rotation=True,
                             entropy_bc="neumann", pressure="expanded"):
    """Bekki-form compressible system (the equations and background names
    of ``make_polytrope_system``) on a Cartesian-regular polytrope-like
    ball: Φ = 1 − β r² (even in r), ρ₀ = Φⁿ with n = 1/(γ−1), c_s² = cs2_0 Φ,
    hydrostatic gravity g = 2β(n+1)(cs2_0/γ) r (odd in r), c_p = 1,
    ds₀/dr = −δ/H_p with H_p = Φ/(2β(n+1) r), uniform rotation Ω = ẑ.
    ``pressure`` spells the pressure gradient by the product rule
    ('expanded', as ``make_polytrope_system``) or as Grad(c q)
    ('grad', the form the general assembler builds as G diag(c))."""
    n = 1.0 / (gamma - 1.0)
    Phi = 1 - beta * r**2
    rho0 = Phi**n
    cs2 = cs2_0 * Phi
    g = 2 * beta * (n + 1) * (cs2_0 / gamma) * r
    Hp = Phi / (2 * beta * (n + 1) * r)

    def S(name, e):
        return Background.Scalar(name, smp.sympify(e))

    def Vr(name, e):
        return Background.Vector(name, v_r=smp.sympify(e), v_theta=0, v_phi=0)

    cs2_over_rho0 = S("cs2_over_rho0", cs2 / rho0)
    cs2_over_cp = S("cs2_over_cp", cs2)
    grad_cs2_over_rho0 = Vr("grad_cs2_over_rho0", smp.diff(cs2 / rho0, r))
    grad_cs2_over_cp = Vr("grad_cs2_over_cp", smp.diff(cs2, r))
    g_over_cp_vec = Vr("g_over_cp_vec", g)
    poly_rho = S("poly_rho", rho0)
    poly_grad_rho = Vr("poly_grad_rho", smp.diff(rho0, r))
    grad_s0 = Vr("grad_s0_optA", -delta / Hp)
    grad_ln_rho = Vr("grad_ln_rho", smp.diff(rho0, r) / rho0)
    grad_ln_P0 = Vr("grad_ln_P0", (n + 1) * smp.diff(Phi, r) / Phi)
    Omega = Background.Vector("poly_Omega_vec", v_r=smp.cos(theta),
                              v_theta=-smp.sin(theta), v_phi=0)
    v = Unknown.Vector("v")
    rho1 = Unknown.Scalar("rho1")
    s1 = Unknown.Scalar("s1")
    if pressure == "grad":
        mom = (Dt(v) + Grad(cs2_over_rho0 * rho1) + Grad(cs2_over_cp * s1)
               - g_over_cp_vec * s1)
    else:
        mom = (Dt(v) + grad_cs2_over_rho0 * rho1 + cs2_over_rho0 * Grad(rho1)
               + grad_cs2_over_cp * s1 + cs2_over_cp * Grad(s1)
               - g_over_cp_vec * s1)
    if rotation:
        mom = mom + 2 * Cross(Omega, v)
    if Ek > 0:
        mom = (mom - smp.S(Ek) * Lap(v)
               - smp.Rational(1, 3) * smp.S(Ek) * Grad(Div(v)))
        if strain:
            st = Grad(v) + Transpose(Grad(v))
            mom = (mom - smp.S(Ek) * Dot(grad_ln_rho, st)
                   + smp.Rational(2, 3) * smp.S(Ek) * grad_ln_rho * Div(v))
    cont = Dt(rho1) + poly_rho * Div(v) + Dot(poly_grad_rho, v)
    en = Dt(s1) + Dot(grad_s0, v)
    if Ek > 0:
        en = en - smp.S(Ek) * Lap(s1) - smp.S(Ek) * Dot(grad_ln_P0, Grad(s1))
    bcs = list(ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    if entropy_bc == "neumann":
        bcs.append(BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, Grad(s1)),
                                     name="s1_out"))
    else:
        bcs.append(BoundaryCondition(Boundary.R_OUTER, s1, name="s1_out"))
    return EquationSystem([Equation(mom, name="momentum"),
                           Equation(cont, name="continuity"),
                           Equation(en, name="energy")], bcs)


def incompressible_system(Ek=0.0, coriolis=2, grad_sign=+1, wall="stress_free",
                          with_dt=True):
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Om = Background.Vector("Omega", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                           v_phi=smp.S.Zero)
    terms = grad_sign * Grad(p)
    if coriolis:
        terms = terms + coriolis * Cross(Om, v)
    if Ek:
        terms = terms - smp.S(Ek) * Lap(v)
    if with_dt:
        terms = Dt(v) + terms
    if wall == "stress_free":
        bcs = list(ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    elif wall == "noslip":
        bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(u, v))
               for u in (r_hat, theta_hat, phi_hat)]
    else:
        bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v))]
    return EquationSystem([Equation(terms, name="mom"),
                           Equation(Div(v), name="cont")], bcs)


def inertial_pencil(grid, m, pairing="center"):
    """(A, B, layout) of the Zhang inertial operator 2 Cross(Omega, v) -
    Grad(p) with Div(v) = 0 and impenetrable walls: the inviscid
    saddle-point pencil whose origin closures the structure tests read."""
    cp = compile_fd2_staggered(
        incompressible_system(coriolis=2, grad_sign=-1, wall="impenetrable",
                              with_dt=False),
        grid, m=m, coriolis_pairing=pairing)
    return cp.A_total.tocsr(), cp.B_total.tocsr(), cp.layout


def good_modes(res, resid=1e-8):
    return res.lam[res.resid < resid]


# ---------------------------------------------------------------------------
# Structure of the origin treatment
# ---------------------------------------------------------------------------

class TestBallOriginStructure:

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    def test_origin_rows_and_receive_only_ring(self, Nr, Nt, m):
        """The origin ring of v_r: Dirichlet-zero rows for |m| ≥ 2, the
        single-DOF profile anchor for m = 0 (cosθ) / |m| = 1 (sinθ), no
        mass on the constraint rows; and nothing outside the ring depends
        on it (zero volume weight, so every norm-adjoint operator ignores
        it)."""
        sg = StaggeredGrids.from_grid(ball(Nr, Nt))
        Nt_c = len(sg.theta_center)
        A, B, _ = inertial_pencil(ball(Nr, Nt), m)
        Al, Bl = A.tolil(), B.tolil()
        if abs(m) >= 2:
            for it in range(Nt_c):
                assert Al.rows[it] == [it] and Bl.rows[it] == []
        else:
            prof = np.cos(sg.theta_center) if m == 0 else np.sin(sg.theta_center)
            assert Bl.rows[0] == [0]
            for it in range(1, Nt_c):
                assert Al.rows[it] == [0, it]
                assert np.isclose(Al.data[it][0], -prof[it] / prof[0])
                assert Bl.rows[it] == []
        col = np.abs(A[:, :Nt_c]).toarray()
        col[:Nt_c, :] = 0.0
        assert col.max() == 0.0

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    def test_exact_identities(self, Nr, Nt, m):
        """curl∘grad vanishes on the weightful rows of the curl grid (the
        pole rows carry zero norm) and Div is the exact negative
        W-adjoint of G, with the origin closures in place."""
        sg = StaggeredGrids.from_grid(ball(Nr, Nt))
        G, Dv, W_v, W_p = build_staggered_grad_div_pair(sg, m, order=2)
        Cp = _assemble_curl_blocks(sg, m, 2, dual=False)
        W_c = _curl_output_norm(sg)
        wc = np.abs(W_c) > 1e-12 * np.abs(W_c).max()
        CG = np.abs((Cp @ G).toarray())
        assert CG[wc].max() < 1e-9 * np.abs(Cp).max()
        X = sp.diags(W_v) @ G + (sp.diags(W_p) @ Dv).conj().T
        assert np.abs(X).max() < 1e-14

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_coriolis_skew_structure(self, Nr, Nt, m):
        """The r–φ Coriolis pair is exactly W-skew; the θ–φ pair of the
        seed recipe is skew only to O(Δθ) (cosθ on the θ-edge in one
        block, on the θ-center in the other), and exactly skew with
        ``coriolis_pairing='adjoint'``."""
        sg = StaggeredGrids.from_grid(ball(Nr, Nt))
        Nr_e, Nr_c = len(sg.r_edge), len(sg.r_center)
        Nt_e, Nt_c = len(sg.theta_edge), len(sg.theta_center)
        n_vr, n_vt = Nr_e * Nt_c, Nr_c * Nt_e
        n_v = n_vr + n_vt + Nr_c * Nt_c
        _, _, W_v, _ = build_staggered_grad_div_pair(sg, m, order=2)
        wv = np.abs(W_v) > 1e-12 * np.abs(W_v).max()
        wv[:Nt_c] = False
        wv[(Nr_e - 1) * Nt_c:n_vr] = False
        defects = {}
        for pairing in ("seed", "adjoint"):
            A, _, _ = inertial_pencil(ball(Nr, Nt), m, pairing)
            WA = (sp.diags(W_v) @ A[:n_v, :n_v]).toarray()
            S = WA[np.ix_(wv, wv)]
            defects[pairing] = np.abs(S + S.conj().T).max() / np.abs(S).max()
            kr = wv[:n_vr]
            rp = WA[:n_vr, n_vr + n_vt:][kr]
            pr = WA[n_vr + n_vt:, :n_vr][:, kr]
            assert np.abs(rp + pr.conj().T).max() < 1e-14
        assert defects["adjoint"] < 1e-12
        assert 1e-3 < defects["seed"] < 0.3

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    def test_viscous_gram_self_adjoint_dissipative(self, Nr, Nt, m):
        """The stress-free strain Gram with a regular density is
        ρ₀W-self-adjoint and negative-semidefinite on the weightful
        subspace after the wall elimination (m = 0 keeps the rigid
        rotation as its zero mode)."""
        sg = StaggeredGrids.from_grid(ball(Nr, Nt))
        rho = lambda rr, tt: np.exp(-rr**2) + 0 * tt
        F = build_staggered_weighted_viscous_gram(sg, m, rho, 2,
                                                  wall_bc="stress_free")
        Fr, kp = _eliminate_wall_vr(F, sg)
        _, _, W_v, _ = build_staggered_grad_div_pair(sg, m, order=2)
        re_, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
        rho_v = np.concatenate([rho(re_[:, None], tc[None, :]).ravel(),
                                rho(rc[:, None], te[None, :]).ravel(),
                                rho(rc[:, None], tc[None, :]).ravel()])[kp]
        WF = (sp.diags(rho_v * W_v[kp]) @ Fr).toarray()
        wf = np.abs(W_v[kp]) > 1e-12 * np.abs(W_v).max()
        H = WF[np.ix_(wf, wf)]
        assert np.abs(H - H.conj().T).max() < 1e-12 * np.abs(H).max()
        ev = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
        assert ev.max() < 1e-10 * np.abs(ev).max()

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    @pytest.mark.parametrize("m", [0, 1, 2, 3])
    @pytest.mark.parametrize("wall_bc", ["neumann", "dirichlet"])
    def test_thermal_laplacians_self_adjoint_dissipative(self, Nr, Nt, m,
                                                         wall_bc):
        """Plain and P₀-weighted scalar Laplacians on the ball are
        self-adjoint under the (weighted) center norm and negative
        semidefinite (the Neumann m = 0 constant is the zero mode)."""
        sg = StaggeredGrids.from_grid(ball(Nr, Nt))
        _, _, _, W_c = build_staggered_grad_div_pair(sg, m, order=2)
        wfn = lambda rr, tt: np.exp(-2.0 * rr**2) + 0 * tt
        L = build_staggered_scalar_laplacian(sg, m, order=2, wall_bc=wall_bc)
        Lw = build_staggered_weighted_scalar_laplacian(sg, m, wfn, order=2,
                                                       wall_bc=wall_bc)
        Ww = wfn(sg.r_center[:, None], sg.theta_center[None, :]).ravel() * W_c
        for op, W in ((L, W_c), (Lw, Ww)):
            WL = (sp.diags(W) @ op).toarray()
            assert np.abs(WL - WL.conj().T).max() < 1e-12 * np.abs(WL).max()
            ev = np.linalg.eigvalsh(0.5 * (WL + WL.conj().T))
            assert ev.max() < 1e-10 * np.abs(ev).max()
            if wall_bc == "dirichlet" or m != 0:
                assert ev.max() < 0

    @pytest.mark.parametrize("Ek", [0.0, 1e-3])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_compressible_pencil_has_no_empty_rows(self, m, Ek):
        """The r^n row premultiply keeps the origin ring weightful (the
        half-cell radius), so no row of the compressible pencil is empty in
        both A and B. The 2·Nr_c weightless pole v_θ DOFs carry an explicit
        Dirichlet row and no mass (``constraints['pinned']``) instead of an
        empty row against a unit mass, which would pin them accidentally as
        0 = λ x and add that many spurious zero eigenvalues; the only empty
        A rows left are the frozen entropy rows when δ = 0 and Ek = 0."""
        grid = ball(8, 16)
        sg = StaggeredGrids.from_grid(grid)
        cp = compile_fd2_staggered(make_regular_ball_system(m, Ek=Ek), grid, m=m)
        A, B = cp.A_total.tocsr(), cp.B_total.tocsr()
        ra = np.asarray(np.abs(A).sum(axis=1)).ravel() == 0
        rb = np.asarray(np.abs(B).sum(axis=1)).ravel() == 0
        assert not np.any(ra & rb)
        vt = cp.layout.field("v_t")
        poles = vt.offset + np.flatnonzero(
            np.isin(np.arange(vt.size) % len(sg.theta_edge),
                    [0, len(sg.theta_edge) - 1]))
        assert np.array_equal(np.sort(cp.constraints["pinned"]), poles)
        assert A[poles].nnz == len(poles)
        assert np.array_equal(A[poles][:, poles].toarray(), np.eye(len(poles)))
        assert B[poles].nnz == 0
        expected = np.zeros(A.shape[0], dtype=bool)
        if Ek == 0.0:
            expected[cp.layout.field("s1").slice] = True
        assert np.array_equal(ra, expected)

    @pytest.mark.parametrize("Nr,Nt", RES_STRUCT)
    def test_m0_pressure_gauge_regularizes_pencil(self, Nr, Nt):
        """At m = 0 the constant pressure is a null vector unless one
        continuity row is replaced by the gauge; with it the dense census
        returns accurate eigenpairs."""
        A, B, lay = inertial_pencil(ball(Nr, Nt), 0)
        n_v = lay.field("p").offset
        assert A.tolil().rows[n_v] == [n_v]
        res = dense(A, B, sigma=0.05 + 0.1j, backend="auto", vectors=True)
        band = np.abs(res.lam.imag) < 2.5
        assert np.median(res.resid[band]) < 1e-10


# ---------------------------------------------------------------------------
# Origin closure of the vector operators (docs/ball_staggered.md, section 2)
# ---------------------------------------------------------------------------

def _origin_field(m):
    """A Cartesian-regular vector field of azimuthal order m with both
    parity families (finite at the origin at m = 0, |m| = 1; linear in r
    at |m| = 2) and its vector Laplacian, as SymPy components."""
    from analytic import vector_calculus as vc
    z, s = r * smp.cos(theta), smp.sin(theta)
    f = (r * s) ** m * (smp.cos(z) + r ** 2 / 3)
    g = (r * s) ** m * smp.sin(z + smp.Rational(3, 10) * r ** 2)
    u = (vc.grad(f, m) + vc.curl(vc.Vec(r * g, 0, 0), m)) * (1 - r ** 2)
    return u, vc.lap(u, m)


def _state(sg, vec):
    from analytic.manufactured import evaluate
    parts = []
    for comp, (rv, tv) in zip(vec.c, [(sg.r_edge, sg.theta_center),
                                      (sg.r_center, sg.theta_edge),
                                      (sg.r_center, sg.theta_center)]):
        R, T = np.repeat(rv, len(tv)), np.tile(tv, len(rv))
        parts.append(evaluate(comp, R, T))
    return np.concatenate(parts)


class TestBallOriginClosure:

    def test_grad_div_flux_form_exact_on_constant_and_linear(self):
        """The radial divergence inside ``build_staggered_grad_div`` is
        exact on v_r = 1 and v_r = r in every cell of a ball, the origin
        ring entering the first cell as v_r(0)/h; the norm-adjoint form is
        exact on the constant but 33 % off on r in the first cell."""
        from spdee.backends.fd2.staggered_compile import _origin_flux_divergence
        sg = StaggeredGrids.from_grid(ball(8, 4))
        _, Div, _, _ = build_staggered_grad_div_pair(sg, 0)
        Div_A = _origin_flux_divergence(sg, Div)
        Nt = len(sg.theta_center)
        rc = sg.r_center
        for v_fn, exact in ((lambda rr: np.ones_like(rr), 2.0 / rc),
                            (lambda rr: rr, 3.0 * np.ones_like(rc))):
            x = np.zeros(Div.shape[1], dtype=complex)
            x[:len(sg.r_edge) * Nt] = np.repeat(v_fn(sg.r_edge), Nt)
            got = (Div_A @ x).reshape(len(rc), Nt)[:, 0]
            assert np.allclose(got, exact, rtol=1e-12, atol=1e-12)
        x = np.zeros(Div.shape[1], dtype=complex)
        x[:len(sg.r_edge) * Nt] = np.repeat(sg.r_edge, Nt)
        mid = (Div @ x).reshape(len(rc), Nt)[:, 0]
        assert abs(mid[0] - 4.0) < 1e-12 and abs(mid[1] - 3.0 - 1.0 / 9.0) < 1e-12

    @pytest.mark.parametrize("m", [0, 1])
    def test_strain_shear_vanishes_on_the_origin_vector(self, m):
        """A constant Cartesian vector (z-hat at m = 0, x-hat + i y-hat at
        |m| = 1) has zero strain; its C-grid shear e_rθ, e_rφ on the first
        interior face is O(Δθ²/h) with the ball spelling
        ∂_r v_tan − v_tan/r, not the O(1/h) of the two-point difference of
        v_tan/r, and halves from 12×24 to 24×48."""
        from spdee.backends.fd2.staggered_compile import (
            _pole_vtheta_prolongation, build_staggered_strain)
        peaks = []
        for Nr, Nt in RES_BENCH:
            sg = StaggeredGrids.from_grid(ball(Nr, Nt))
            S, comp = build_staggered_strain(sg, m)
            if m == 1:
                S = S @ _pole_vtheta_prolongation(sg)
            tc, te = sg.theta_center, sg.theta_edge
            if m == 0:
                vec = (np.cos(tc), -np.sin(te), np.zeros(len(tc)))
            else:
                vec = (np.sin(tc), np.cos(te), 1j * np.ones(len(tc)))
            x = np.concatenate([np.tile(vec[0], len(sg.r_edge)),
                                np.tile(vec[1], len(sg.r_center)),
                                np.tile(vec[2], len(sg.r_center))])
            e = S @ x
            off = np.cumsum([0] + [c[0] for c in comp])
            e_rt = e[off[3]:off[4]].reshape(len(sg.r_edge), len(te))[1, 1:-1]
            e_rp = e[off[4]:off[5]].reshape(len(sg.r_edge), len(tc))[1]
            peaks.append(max(np.abs(e_rt).max(), np.abs(e_rp).max()))
            h = sg.r_edge[1]
            assert peaks[-1] < 0.5 * (tc[1] - tc[0]) ** 2 / h, (m, Nr, peaks[-1])
        assert peaks[1] < 0.6 * peaks[0]

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_vector_laplacian_first_rings_consistent(self, m):
        """The general vector Laplacian (the strain Gram less (1/3) Grad
        Div) applied to a smooth field of both parity families is O(1)
        consistent on the innermost rings over the middle half of the
        colatitudes: the truncation there does not grow from 12×24 to
        24×48 (it grew like 1/h² at |m| = 1 and 1/h at m = 0, 2 with the
        midpoint stencils, an h²/r error, first order on the ring). The
        pole-adjacent columns keep the O(Δθ²/sin²θ) pole truncation of
        the two-point stencils, which the sinθ weight makes O(h²)."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div, build_staggered_weighted_viscous_gram)
        u, lap_u = _origin_field(m)
        worst = []
        for Nr, Nt in RES_BENCH:
            sg = StaggeredGrids.from_grid(ball(Nr, Nt))
            F = build_staggered_weighted_viscous_gram(sg, m, lambda rr, t: np.ones_like(rr * t))
            GD = build_staggered_grad_div(sg, m)
            d = (F - GD / 3.0) @ _state(sg, u) - _state(sg, lap_u)
            n_vr, n_vt = len(sg.r_edge) * len(sg.theta_center), \
                len(sg.r_center) * len(sg.theta_edge)
            lo, hi = Nt // 4, 3 * Nt // 4
            dr = np.abs(d[:n_vr].reshape(len(sg.r_edge), -1))[1:3, lo:hi]
            dt = np.abs(d[n_vr:n_vr + n_vt].reshape(len(sg.r_center), -1))[:2, lo:hi]
            dp = np.abs(d[n_vr + n_vt:].reshape(len(sg.r_center), -1))[:2, lo:hi]
            worst.append(max(dr.max(), dt.max(), dp.max()))
        assert worst[1] < 1.5 * worst[0], (m, worst)
        assert worst[1] < 5.0, (m, worst)

    @pytest.mark.parametrize("m", [0, 1])
    def test_folded_origin_master_carries_the_viscous_balance(self, m):
        """The folded master of the origin ring has a viscous row that
        couples it to the tangential DOFs and scales like the other
        momentum rows, 1/h² against its mass (it was empty at m = 0, the
        amplitude pinned to zero by 0 = λ x, and the 1/h-growing Grad Div
        remainder alone at |m| = 1)."""
        ratios = []
        for Nr, Nt in RES_BENCH:
            cp = compile_fd2_staggered(incompressible_system(Ek=1e-2), ball(Nr, Nt), m=m)
            A, B = cp.A_total.tocsr(), cp.B_total.tocsr()
            n_vr = cp.layout.field("v_t").offset
            n_v = cp.layout.field("p").offset
            row = A[0].toarray().ravel()
            assert np.abs(row[n_vr:n_v]).max() > 0
            assert B[0, 0] != 0
            ratios.append(np.abs(row[:n_v]).max() / abs(B[0, 0]))
        assert 3.0 < ratios[1] / ratios[0] < 5.0, ratios

    def test_primal_curl_reads_the_pole_through_the_prolongation_at_m1(self):
        """At |m| = 1 the primal curl has no column on the pinned pole
        v_theta DOFs and reads the adjacent interior edge instead, so the
        pole closure of curl_r sees (sinθ v_φ)' − i m v_θ → 0 for a regular
        field; at m = 2 the pole DOF enters the guarded rows directly."""
        from spdee.backends.fd2.staggered_general import _CGridOps
        sg = StaggeredGrids.from_grid(ball(6, 8))
        Nt_e, n_vr = len(sg.theta_edge), len(sg.r_edge) * len(sg.theta_center)
        poles = [n_vr + ir * Nt_e + j for ir in range(len(sg.r_center))
                 for j in (0, Nt_e - 1)]
        Cp1 = _CGridOps(sg, 1, 2).primal_curl(None).tocsc()
        assert Cp1[:, poles].nnz == 0
        assert Cp1[:, [p + 1 for p in poles[::2]]].nnz > 0
        Cp2 = _CGridOps(sg, 2, 2).primal_curl(None).tocsc()
        assert Cp2[:, poles].nnz > 0


# ---------------------------------------------------------------------------
# Zhang et al. 2001 inertial modes
# ---------------------------------------------------------------------------

def zhang_census(m, Nr, Nt, pairing="seed"):
    cp = compile_fd2_staggered(zhang_system(), ball(Nr, Nt), m=m,
                               coriolis_pairing=pairing)
    A, B = cp.assemble_evp_matrices()
    res = dense(A, B, sigma=0.3 + 0.2j, backend="auto", vectors=True)
    lam = good_modes(res)
    return lam, -lam.imag


def zhang_errors(om, targets):
    return np.array([abs(om[np.argmin(np.abs(om - w))] - w) / max(abs(w), 0.05)
                     for _, w, _ in targets])


# Poincare's operator is not compact: the inviscid band |omega| < 2 fills with
# as many modes as the discretisation has degrees of freedom, so a nearest
# match to a curated target is evidence only when it is much sharper than the
# mean spacing. Each census is therefore matched against random control
# frequencies of the same band as well, and the ratio of the two median
# distances is reported (docs/analytic_benchmarks.md).
ZHANG_CONTROLS = np.random.default_rng(3).uniform(-1.9, 1.9, 40)


def discrimination(om, targets):
    """(modes in band, median target distance, median control distance)."""
    band = om[np.abs(om) < 2.0]
    hit = float(np.median([np.abs(band - w).min() for _, w, _ in targets]))
    miss = float(np.median([np.abs(band - c).min() for c in ZHANG_CONTROLS]))
    return band.size, hit, miss


class TestZhangBall:
    """What the curated m = 1, 2 targets measure on this core.

    The staggered inviscid band is dense -- 960 neutral modes at 16×32 and
    2206 at 24×48 -- so a per-target threshold at or below the mean spacing
    would pass on any spectrum of that density. What the data supports is
    the RMS error over the ten targets (below 5e-3 at 16×32 and 1.5e-3 at
    24×48) and its convergence between the two grids (order above 1.3;
    measured ≈ 2.6 at m = 1 and ≈ 3.7 at m = 2). The discrimination ratio
    (median distance of random control frequencies over median distance of
    the targets) is reported and not asserted: it is 0.8 to 1.6 here, so
    this core does not resolve individual inertial modes, while the
    collocated core reaches 5.5-13.6 and sh_fd and Dedalus ~3300
    (docs/analytic_benchmarks.md).
    """

    @pytest.mark.parametrize("m,targets", [(1, ZHANG_TARGETS_M1),
                                           (2, ZHANG_TARGETS_M2)],
                             ids=["m1", "m2"])
    def test_rms_error_and_order(self, m, targets):
        _, om16 = zhang_census(m, 16, 32)
        _, om24 = zhang_census(m, 24, 48)
        e16 = zhang_errors(om16, targets)
        e24 = zhang_errors(om24, targets)
        for om, res in ((om16, "16x32"), (om24, "24x48")):
            n_band, hit, miss = discrimination(om, targets)
            print(f"ZHANG | staggered m={m} | {res} | {n_band:5d} modes in band "
                  f"| target {hit:.2e} | control {miss:.2e} | ratio {miss / hit:.2f}")
        rms16 = float(np.sqrt(np.mean(e16**2)))
        rms24 = float(np.sqrt(np.mean(e24**2)))
        assert rms16 < 5e-3, (rms16, e16)
        assert rms24 < 1.5e-3, (rms24, e24)
        assert order_of(rms16, rms24, 1.5) > 1.3, (rms16, rms24)

    @pytest.mark.parametrize("m", [1, 2])
    def test_adjoint_pairing_is_undamped(self, m):
        """With the exact W-adjoint θ–φ Coriolis partner the inviscid
        spectrum is imaginary to rounding; the seed recipe leaves real
        parts of order 1e-3."""
        lam_s, _ = zhang_census(m, 16, 32, pairing="seed")
        lam_a, _ = zhang_census(m, 16, 32, pairing="adjoint")
        band_s = lam_s[np.abs(lam_s.imag) < 2.5]
        band_a = lam_a[np.abs(lam_a.imag) < 2.5]
        assert np.abs(band_a.real).max() < 1e-10
        assert 1e-4 < np.abs(band_s.real).max() < 1e-2


# ---------------------------------------------------------------------------
# Compressible ball
# ---------------------------------------------------------------------------

class TestCompressibleBall:

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_inertial_modes_match_incompressible_assembler(self, m):
        """With δ = 0, Ek = 0 and a nearly uniform density (β = 1e-3,
        c_s² = 1.2e4) the compressible ball reproduces the incompressible
        assembler's inertial modes at the same grid: the Zhang targets at
        m = 1, 2 (Rossby-type retrograde and prograde modes included), the
        band modes at m = 0.

        The band is dense (about 530 neutral modes at 12x24), so the match
        is read against random control frequencies of the same band: the
        matched modes agree to 5e-5 while a control frequency lands 2e-3
        from its nearest compressible mode, a ratio above 100, so the
        correspondence is mode by mode and not an accident of density."""
        grid = ball(12, 24)
        cpi = compile_fd2_staggered(zhang_system(), grid, m=m)
        Ai, Bi = cpi.assemble_evp_matrices()
        om_i = -good_modes(dense(Ai, Bi, sigma=0.3 + 0.2j, backend="auto",
                                 vectors=True)).imag
        cpc = compile_fd2_staggered(
            make_regular_ball_system(m, Ek=0.0, delta=0.0, beta=1e-3), grid, m=m)
        Ac, Bc = cpc.assemble_evp_matrices()
        om_c = -good_modes(dense(Ac, Bc, sigma=0.3 + 0.2j, backend="auto",
                                 vectors=True)).imag
        if m == 0:
            targets = np.sort(np.abs(om_i[(np.abs(om_i) > 0.5)
                                          & (np.abs(om_i) < 1.9)]))[:4]
        else:
            targets = [w for _, w, _ in (ZHANG_TARGETS_M1 if m == 1
                                         else ZHANG_TARGETS_M2)]
        band_c = om_c[np.abs(om_c) < 2.0]
        d = []
        for w in targets:
            wi = om_i[np.argmin(np.abs(om_i - w))]
            wc = band_c[np.argmin(np.abs(band_c - wi))]
            d.append(abs(wc - wi))
            assert abs(wc - wi) < 1e-4, (w, wi, wc)
        ctrl = float(np.median([np.abs(band_c - c).min() for c in ZHANG_CONTROLS]))
        ratio = ctrl / float(np.median(d))
        print(f"ZHANG | compressible vs inertial m={m} | 12x24 | "
              f"{band_c.size:5d} modes in band | target {np.median(d):.2e} "
              f"| control {ctrl:.2e} | ratio {ratio:.0f}")
        assert ratio > 100.0, (ratio, np.median(d), ctrl)

    def test_convective_ridge_converges(self):
        """With δ > 0 and viscous/thermal diffusion selecting a finite
        horizontal scale, the peak growth rate of the convective ridge
        converges over 8×16, 12×24, 18×36 with a Richardson order above
        1.3."""
        peaks = []
        for Nr, Nt in ((8, 16), (12, 24), (18, 36)):
            cp = compile_fd2_staggered(
                make_regular_ball_system(1, Ek=3e-3, delta=2e-5, strain=True),
                ball(Nr, Nt), m=1)
            A, B = cp.assemble_evp_matrices()
            res = dense(A, B, sigma=0.05 + 0.5j, backend="auto", vectors=True)
            om = EvpResult.eigenvalues_to_omega(good_modes(res, 1e-6))
            grow = om[om.imag > 0.02]
            assert len(grow)
            peaks.append(grow.imag.max())
        d = np.abs(np.diff(peaks))
        assert order_of(d[0], d[1], 1.5) > 1.3, peaks

    def test_entropy_dirichlet_wall_is_wired(self):
        """The entropy wall condition of the system reaches the thermal
        operator: Dirichlet and Neumann give different, both dissipative,
        s₁–s₁ blocks."""
        grid = ball(8, 16)
        blocks = {}
        for bc in ("neumann", "dirichlet"):
            cp = compile_fd2_staggered(
                make_regular_ball_system(1, Ek=1e-3, entropy_bc=bc), grid, m=1)
            s = cp.layout.field("s1").slice
            blocks[bc] = cp.A_total.tocsr()[s, s].toarray()
        assert np.abs(blocks["neumann"] - blocks["dirichlet"]).max() > 1e-3
        for blk in blocks.values():
            ev = np.linalg.eigvals(blk)
            assert ev.real.min() > -1e-12 * np.abs(ev).max()


# ---------------------------------------------------------------------------
# Thermal diffusion decay rates
# ---------------------------------------------------------------------------

THERMAL_ROOTS = {
    ("dirichlet", 0): np.pi, ("dirichlet", 1): 4.493409, ("dirichlet", 2): 5.763459,
    ("dirichlet", 3): 6.987932,
    ("neumann", 1): 2.081576, ("neumann", 2): 3.342094, ("neumann", 3): 4.514100,
    ("neumann", 0): 4.493409,
}


class TestThermalBall:
    """∂t s = Ek ∇²s on the ball: decay rates Ek k² with j_l(k) = 0
    (Dirichlet) or j_l'(k) = 0 (Neumann), l ≥ |m|, converging at 2nd order
    with the origin-parity and wall closures of the flux-form Laplacian."""

    @pytest.mark.parametrize("wall_bc", ["dirichlet", "neumann"])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_decay_rates_second_order(self, wall_bc, m):
        errs = {}
        for Nr, Nt in RES_BENCH:
            sg = StaggeredGrids.from_grid(ball(Nr, Nt))
            L = build_staggered_scalar_laplacian(sg, m, order=2, wall_bc=wall_bc)
            res = dense(L, None, sigma=1.0 + 0.0j, backend="auto", vectors=False,
                        compute_resid=False)
            ev = np.sort(-res.lam.real)
            ev = ev[ev > 1e-6]
            for l in range(m, m + 2):
                k2 = THERMAL_ROOTS[(wall_bc, l)]**2
                errs.setdefault(l, []).append(abs(ev[np.argmin(np.abs(ev - k2))] - k2) / k2)
        for l, (e1, e2) in errs.items():
            assert e2 < 3e-3, (l, e1, e2)
            assert order_of(e1, e2, 2.0) > 1.9, (l, e1, e2)


# ---------------------------------------------------------------------------
# Toroidal viscous decay
# ---------------------------------------------------------------------------

def toroidal_k(l, wall):
    """Toroidal mode v = T(r) r × ∇Y_lm, T = j_l(kr), decay rate Ek k².
    Stress-free: ∂_r(T/r) = 0 at r = 1 → k j_l'(k) = j_l(k), i.e. with
    j_l' = j_{l−1} − (l+1)/k j_l:  k j_{l−1}(k) = (l+2) j_l(k) (l = 1 reduces
    to j_2(k) = 0). No-slip: T(1) = 0 → j_l(k) = 0."""
    if wall == "stress_free":
        f = lambda k: k * spherical_jn(l - 1, k) - (l + 2) * spherical_jn(l, k)
    else:
        f = lambda k: spherical_jn(l, k)
    return roots(f, 1, lo=0.3)[0]


class TestToroidalViscousBall:
    """Toroidal decay modes of ∂t v = Ek ∇²v − ∇p, ∇·v = 0 in the ball
    (Ek = 1e-2), at m = 0, 1, 2 and l = max(m, 1), max(m, 1) + 1, against
    the Bessel conditions of ``toroidal_k``; the eigenvectors satisfy the
    discrete continuity constraint and are toroidal-dominated (their v_r
    energy is the O(h²) C-grid admixture)."""

    EK = 1e-2

    def _rates(self, wall, m, Nr, Nt):
        sysm = incompressible_system(Ek=self.EK, coriolis=0, wall=wall)
        cp = compile_fd2_staggered(sysm, ball(Nr, Nt), m=m)
        A, B = cp.assemble_evp_matrices()
        fs = {f.name: f for f in cp.layout.fields}
        n_v = fs["p"].offset
        Dv = A[n_v:, :n_v]
        out = {}
        for l in range(max(m, 1), max(m, 1) + 2):
            k2 = toroidal_k(l, wall)**2
            v0 = np.random.default_rng(l).standard_normal(A.shape[0]).astype(complex)
            lam, V = shift_invert(A, B, self.EK * k2 * (1 + 1e-3), 16,
                                  return_vectors=True, tol=1e-10, v0=v0)
            vv = V[:n_v]
            fr = (np.linalg.norm(V[fs["v_r"].slice], axis=0)
                  / np.linalg.norm(vv, axis=0))**2
            fp = (np.linalg.norm(V[fs["v_p"].slice], axis=0)
                  / np.linalg.norm(vv, axis=0))**2
            tor = (fr < 1e-2) & (fp > 1e-3) & np.isfinite(lam)
            assert tor.any()
            j = np.flatnonzero(tor)[np.argmin(np.abs(lam[tor].real - self.EK * k2))]
            div = np.linalg.norm(Dv @ vv[:, j]) / np.linalg.norm(vv[:, j])
            assert div < 1e-8
            out[l] = abs(lam[j].real / self.EK - k2) / k2
        return out

    @pytest.mark.parametrize("wall", ["stress_free", "noslip"])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_decay_rates_second_order(self, wall, m):
        e1 = self._rates(wall, m, *RES_BENCH[0])
        e2 = self._rates(wall, m, *RES_BENCH[1])
        for l in e1:
            assert e2[l] < 4e-3, (l, e1[l], e2[l])
            assert order_of(e1[l], e2[l], 2.0) > 1.7, (l, e1[l], e2[l])


# ---------------------------------------------------------------------------
# MHD and routing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Free magnetic decay
# ---------------------------------------------------------------------------

def magnetic_k(family, l):
    """Free decay in a perfectly conducting ball: b = Curl(Curl(S r)) +
    Curl(T r) with both potentials obeying Dt X = Lap X, X = j_l(k r) and
    the rate k². The wall carries b_r = 0 and a vanishing tangential
    electric field E = Curl(b): b_r = l(l+1) S / r gives S(1) = 0 for the
    poloidal family (which also kills the toroidal part of E, proportional
    to Lap S = −k² S), and the poloidal part of E is proportional to
    d(r T)/dr, so the toroidal family closes on (x j_l(x))' = 0, i.e.
    j_l(k) + k j_l'(k) = 0. Regularity at the origin selects j_l."""
    if family == "pol":
        f = lambda k: spherical_jn(l, k)
    else:
        f = lambda k: spherical_jn(l, k) + k * spherical_jn(l, k, derivative=True)
    return roots(f, 2, lo=0.3, hi=12.0)


def magnetic_decay_system(formulation):
    """'b': the flux formulation, Div(b) = 0 an involution, the conducting
    wall b_r = 0 with tangential Curl(b) = 0; 'A': the vector potential in
    the Coulomb gauge (multiplier psi), tangential A = 0 and psi = 0."""
    b = Unknown.Vector("b")
    if formulation == "A":
        psi = Unknown.Scalar("psi")
        return EquationSystem(
            [Equation(Dt(b) - Lap(b) - Grad(psi), name="induction"),
             Equation(Div(b), name="coulomb")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, b)),
             BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, b)),
             BoundaryCondition(Boundary.R_OUTER, psi)])
    return EquationSystem(
        [Equation(Dt(b) - Lap(b), name="induction"),
         Equation(Div(b), name="solenoidal")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, b)),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, Curl(b))),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, Curl(b)))])


def _full_potential(cp, x):
    """The face components of a vector potential on a ball from the reduced
    state: the origin ring unfolded from its master (m = 0, |m| = 1) or
    zero (|m| >= 2)."""
    g, m = cp.grids, cp.m
    Nr_e, Nt_c = len(g.r_edge), len(g.theta_center)
    a_r = x[cp.layout.field("b_r").slice]
    full = np.zeros(Nr_e * Nt_c, dtype=complex)
    if abs(m) >= 2:
        full[Nt_c:] = a_r
    else:
        prof = np.cos(g.theta_center) if m == 0 else np.sin(g.theta_center)
        full[:Nt_c] = a_r[0] * prof / prof[0]
        full[Nt_c:] = a_r[1:]
    tail = x[cp.layout.field("b_r").slice.stop:cp.layout.field("b_p").slice.stop]
    return np.concatenate([full, tail])


def _radial_share(cp, x, formulation, ops):
    """The share of the magnetic energy in b_r: zero for a toroidal mode
    (b = Curl(T r) has no radial component at any m) up to the O(h²)
    admixture of the potential's origin fold and pole prolongation at
    |m| = 1, a third and more for a poloidal one. The vector potential is
    curled first."""
    if formulation == "A":
        edge = ops.primal_curl("noslip") @ _full_potential(cp, x)
        n_ce, n_rc = ops.n["CE"], ops.n["RC"]
        parts = [edge[:n_ce], edge[n_ce:n_ce + n_rc], edge[n_ce + n_rc:]]
    else:
        parts = [x[cp.layout.field(f"b_{c}").slice] for c in "rtp"]
    e = [float(np.linalg.norm(p)) ** 2 for p in parts]
    return e[0] / sum(e)


class TestMagneticDecayBall:
    """Free decay of a magnetic field in a perfectly conducting ball at
    m = 0, 1, 2 (l = max(m, 1): two toroidal rates, the gravest poloidal
    rate, and the gravest poloidal rate of l + 1) against ``magnetic_k``,
    in the flux formulation (constrained transport with the origin ring
    pinned, docs/ball_staggered.md section 7) and the vector-potential
    formulation (the regular vector's origin fold), both at second order
    and agreeing with each other to the discretization error."""

    def _rates(self, formulation, m, Nr, Nt):
        from spdee.backends.fd2.staggered_general import _CGridOps
        grid = ball(Nr, Nt)
        cp = compile_fd2_staggered(magnetic_decay_system(formulation), grid, m=m)
        assert cp.constraints["constraints"] == {
            "b": "involution" if formulation == "b" else "saddle_point"}
        ops = _CGridOps(cp.grids, m, 2)
        A, B = cp.assemble_evp_matrices()
        v0 = np.random.default_rng(0).standard_normal(A.shape[0]).astype(complex)
        l0 = max(m, 1)
        out = {}
        for family, l, n in (("tor", l0, 0), ("tor", l0, 1), ("pol", l0, 0),
                             ("pol", l0 + 1, 0)):
            k2 = magnetic_k(family, l)[n] ** 2
            lam, V = shift_invert(A, B, k2 + 0.0j, 12, return_vectors=True, tol=1e-10,
                                  v0=v0)
            found = None
            for j in np.argsort(np.abs(lam - k2)):
                if not np.isfinite(lam[j]):
                    continue
                share = _radial_share(cp, V[:, j], formulation, ops)
                if (share < 2e-2) if family == "tor" else (share > 0.1):
                    found = lam[j]
                    break
            assert found is not None, (formulation, m, family, l, n, lam)
            if formulation == "b":
                x = V[:, j]
                flux = np.concatenate([x[cp.layout.field(f"b_{c}").slice] for c in "rtp"])
                assert abs(cp.divergence(x)).max() <= 1e-10 * abs(flux).max()
            out[(family, l, n)] = (found.real, k2)
        return out

    @pytest.mark.parametrize("formulation", ["b", "A"])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_decay_rates_second_order(self, formulation, m):
        coarse = self._rates(formulation, m, *RES_BENCH[0])
        fine = self._rates(formulation, m, *RES_BENCH[1])
        for key, (got, k2) in fine.items():
            e1 = abs(coarse[key][0] - k2) / k2
            e2 = abs(got - k2) / k2
            print(f"BENCH | magnetic decay {formulation} ball m={m} {key[0]} l={key[1]} "
                  f"n={key[2]} | analytic {k2:.6f} | computed {got:.6f} | "
                  f"err {e1:.2e} {e2:.2e} | order {order_of(e1, e2, 2.0):.2f}")
            assert e2 < 5e-3, (formulation, m, key, e1, e2)
            assert order_of(e1, e2, 2.0) > 1.7, (formulation, m, key, e1, e2)


# ---------------------------------------------------------------------------
# MHD and routing
# ---------------------------------------------------------------------------

class TestBallMHDAndRouting:

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_ct_mhd_on_ball(self, m):
        """Constrained-transport MHD on the ball: the compressible system
        with a uniform axial field compiles with b an involution, the
        origin and wall rings of b_r dropped by the viscous reduction (the
        velocity's origin ring folded or dropped by regularity), no row
        empty in both A and B, and the flux of an eigenvector near the
        Rossby mode divergence-free."""
        B0 = Background.Vector("B0", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                               v_phi=0)
        sysm = add_magnetic_field(make_regular_ball_system(m, Ek=1e-3), B0,
                                  Le=1e-2, eta=1e-3, bc="conducting",
                                  geometry="ball")
        Nr, Nt = 6, 12
        cp = compile_fd2_staggered(sysm, ball(Nr, Nt), m=m)
        assert cp.constraints["constraints"] == {"b": "involution"}
        assert cp.constraints["walls"]["vectors"]["b"]["kind"] == "conducting"
        assert cp.layout.field("b_r").size == (Nr + 1) * Nt - 2 * Nt
        assert cp.layout.field("v_r").size == (Nr + 1) * Nt - Nt - (Nt if m >= 2 else Nt - 1)
        A, B = cp.assemble_evp_matrices()
        empty = (abs(A).sum(axis=1).A1 == 0) & (abs(B).sum(axis=1).A1 == 0)
        assert not empty.any()
        v0 = np.random.default_rng(0).standard_normal(A.shape[0]).astype(complex)
        lam, V = shift_invert(A, B, 0.3j if m else 0.1, 6, return_vectors=True, v0=v0)
        j = int(np.argmax(np.isfinite(lam)))
        x = V[:, j] / np.linalg.norm(V[:, j])
        flux = np.concatenate([x[cp.layout.field(f"b_{c}").slice] for c in "rtp"])
        assert abs(cp.divergence(x)).max() <= 1e-10 * abs(flux).max()

    def test_viscous_system_needs_a_tangential_wall_condition(self):
        """A Lap(v) with only Dot(r_hat, v) = 0 leaves the strain stencils
        at the wall unclosed and is refused rather than guessed."""
        with pytest.raises(NotImplementedError, match="tangential"):
            compile_fd2_staggered(incompressible_system(Ek=1e-2, wall="impenetrable"),
                                  ball(6, 12), m=1)

    def test_drag_and_a_tilted_rotation_axis_are_built(self):
        """Terms the recogniser path could not read are ordinary tasks for
        the dispatcher: a background drag f v is a coefficient identity and
        a rotation axis off z is the general cross pairing, so both build
        instead of being refused."""
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        Om = Background.Vector("Omega", v_r=smp.cos(theta),
                               v_theta=-smp.sin(theta), v_phi=0)
        f = Background.Scalar("f", r)
        drag = EquationSystem(
            [Equation(Dt(v) + 2 * Cross(Om, v) + Grad(p) + f * v, name="mom"),
             Equation(Div(v), name="cont")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v))])
        tilted = Background.Vector("Omega", v_r=smp.sin(theta),
                                   v_theta=smp.cos(theta), v_phi=0)
        axis = EquationSystem(
            [Equation(Dt(v) + 2 * Cross(tilted, v) + Grad(p), name="mom"),
             Equation(Div(v), name="cont")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v))])
        base = compile_fd2_staggered(
            incompressible_system(coriolis=2, wall="impenetrable"), ball(6, 12), m=1)
        for sysm in (drag, axis):
            cp = compile_fd2_staggered(sysm, ball(6, 12), m=1)
            assert cp.constraints["constraints"] == {"v": "saddle_point"}
            assert cp.A_total.shape == base.A_total.shape
            assert abs(cp.A_total - base.A_total).max() > 0.0

    def test_sign_conventions(self):
        """The Zhang form 2·Cross(Ω, v) − Grad(p) = λ v and the Bekki form
        Dt(v) + 2·Cross(Ω, v) + Grad(p) = 0 give the same spectrum (the
        pressure sign is a gauge); reversing the Coriolis sign maps
        ω → −ω."""
        grid = ball(8, 16)

        def spectrum(sysm):
            cp = compile_fd2_staggered(sysm, grid, m=1)
            A, B = cp.assemble_evp_matrices()
            lam = good_modes(dense(A, B, sigma=0.3 + 0.2j, backend="auto",
                                   vectors=True))
            return np.sort(lam.imag[np.abs(lam.imag) < 2.5])

        s_z = spectrum(incompressible_system(coriolis=2, grad_sign=-1,
                                             wall="impenetrable", with_dt=False))
        s_b = spectrum(incompressible_system(coriolis=2, grad_sign=+1,
                                             wall="impenetrable"))
        s_r = spectrum(incompressible_system(coriolis=-2, grad_sign=+1,
                                             wall="impenetrable"))
        assert np.allclose(s_z, s_b, atol=1e-9)
        assert np.allclose(s_r, -s_b[::-1], atol=1e-9)


# ---------------------------------------------------------------------------
# Composite wall rows: Robin, Navier slip, the normal stress
# ---------------------------------------------------------------------------

def _problem(unknowns, equations, params, bcs):
    from spdee import Problem
    pb = Problem()
    for name, kind in unknowns:
        pb.add_unknown(name, kind)
    for name, value in params.items():
        pb.add_parameter(name, value)
    for eq in equations:
        pb.add_equation(eq)
    for bc in bcs:
        pb.add_bc(bc, at="outer")
    return pb


def _nearest(cp, target, seed=0, k=8):
    """The pencil eigenvalue nearest ``target`` by a seeded shift-invert."""
    A, B = cp.assemble_evp_matrices()
    v0 = np.random.default_rng(seed).standard_normal(A.shape[0]).astype(complex)
    lam, _ = shift_invert(A, B, target * 0.97, k, v0=v0)
    lam = lam[np.isfinite(lam)]
    return lam[np.argmin(np.abs(lam - target))]


def _pencils_agree(cp1, cp2):
    A, B = cp1.A_total.tocsr(), cp2.A_total.tocsr()
    A.sort_indices()
    B.sort_indices()
    return (A.shape == B.shape and np.array_equal(A.indptr, B.indptr)
            and np.array_equal(A.indices, B.indices) and np.array_equal(A.data, B.data))


STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def traction_free_k(l, c, n=2):
    """Poloidal modes of the Stokes ball with the normal-stress wall
    2 e_rr = c p and stress-free tangential rows (the 2 x 2 Bessel
    determinant of tests/test_dedalus_ball.py): lambda = k^2."""
    def det(k):
        f = spherical_jn(l, k)
        f1 = k * spherical_jn(l, k, derivative=True)
        f2 = -2 * f1 - (k**2 - l * (l + 1)) * f
        n_row = (2 * l * (l - 1) - c * k**2, 2 * l * (l + 1) * (f1 - f))
        t_row = (2 * (l - 1), f2 + (l * (l + 1) - 2) * f)
        return n_row[0] * t_row[1] - n_row[1] * t_row[0]
    return roots(det, n, lo=0.3, hi=12.0)


class TestCompositeWallsBall:
    """The composite boundary rows on the ball at order 2, against the
    closed forms of tests/test_dedalus_ball.py: the Robin wall as the
    mirror-ghost flux closure, Navier slip as the strain closure with a
    slip length, the normal stress as an open wall whose v_r faces stay
    degrees of freedom."""

    @pytest.mark.parametrize("h", [1, 3])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_robin_decay_rates_second_order(self, m, h):
        """dt(p) - lap(p) = 0 with dot(r_hat, grad(p)) + h p = 0: lambda = k^2
        with k j_l'(k) + h j_l(k) = 0, l = m, m + 1, at second order."""
        targets = {l: roots(lambda k: k * spherical_jn(l, k, derivative=True)
                            + h * spherical_jn(l, k), 1, lo=0.3)[0]**2
                   for l in range(m, m + 2)}
        errs = {}
        for Nr, Nt in RES_BENCH:
            pb = _problem([("p", "scalar")], ["dt(p) - lap(p) = 0"], {"h": float(h)},
                          ["dot(r_hat, grad(p)) + h*p = 0"])
            cp = pb.compile(ball(Nr, Nt), m=m, backend="fd2", discretization="staggered",
                            order_r=2, order_theta=2)
            assert cp.constraints["walls"]["scalars"]["p"]["closure"] == ("neumann", "robin")
            for l, k2 in targets.items():
                errs.setdefault(l, []).append(abs(_nearest(cp, k2, l).real - k2) / k2)
        for l, (e1, e2) in errs.items():
            assert e2 < 3e-3, (l, e1, e2)
            assert order_of(e1, e2, 2.0) > 1.7, (l, e1, e2)

    @pytest.mark.parametrize("m", [0, 1])
    def test_navier_slip_toroidal_rates_second_order(self, m):
        """dot(t_hat, v) + ls dot(t_hat, dot(strain, r_hat)) = 0 on the theta
        and phi rows: the toroidal decay rates Ek k^2 with (1 - ls) j_l(k) +
        ls k j_l'(k) = 0, l = max(m, 1), max(m, 1) + 1, at second order."""
        Ek, ls = 0.01, 0.5
        row = "dot({hat}, v) + ls*dot({hat}, " + STRAIN + ") = 0"
        targets = {l: roots(lambda k: (1 - ls) * spherical_jn(l, k)
                            + ls * k * spherical_jn(l, k, derivative=True), 1, lo=0.3)[0]**2
                   for l in range(max(m, 1), max(m, 1) + 2)}
        errs = {}
        for Nr, Nt in RES_BENCH:
            pb = _problem([("v", "vector")], ["dt(v) - Ek*lap(v) = 0"], {"Ek": Ek, "ls": ls},
                          ["dot(r_hat, v) = 0"] + [row.format(hat=h)
                                                   for h in ("theta_hat", "phi_hat")])
            cp = pb.compile(ball(Nr, Nt), m=m, backend="fd2", discretization="staggered",
                            order_r=2, order_theta=2)
            wall = cp.constraints["walls"]["vectors"]["v"]["tangential"]
            assert np.all(np.asarray(wall.slip[1][0]) == ls) and wall.open == (False, False)
            for l, k2 in targets.items():
                rate = Ek * k2
                errs.setdefault(l, []).append(abs(_nearest(cp, rate, l).real - rate) / rate)
        for l, (e1, e2) in errs.items():
            assert e2 < 4e-3, (l, e1, e2)
            assert order_of(e1, e2, 2.0) > 1.7, (l, e1, e2)

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_navier_slip_limits_are_the_no_slip_and_stress_free_pencils(self, m):
        """ls = 0 gives the no-slip pencil and a vanishing velocity
        coefficient (ls = inf) the stress-free one, bitwise: the three
        closures are one code path."""
        grid = ball(*RES_STRUCT[0])
        eqs = ["dt(v) - Ek*lap(v) = 0"]
        kw = dict(m=m, backend="fd2", discretization="staggered", order_r=2, order_theta=2)
        slip0 = _problem([("v", "vector")], eqs, {"Ek": 0.01, "ls": 0.0},
                         ["dot(r_hat, v) = 0"]
                         + [f"dot({h}, v) + ls*dot({h}, {STRAIN}) = 0"
                            for h in ("theta_hat", "phi_hat")]).compile(grid, **kw)
        noslip = _problem([("v", "vector")], eqs, {"Ek": 0.01},
                          [f"dot({h}, v) = 0" for h in ("r_hat", "theta_hat", "phi_hat")]
                          ).compile(grid, **kw)
        assert _pencils_agree(slip0, noslip)
        slip_inf = _problem([("v", "vector")], eqs, {"Ek": 0.01, "eps": 0.0},
                            ["dot(r_hat, v) = 0"]
                            + [f"eps*dot({h}, v) + dot({h}, {STRAIN}) = 0"
                               for h in ("theta_hat", "phi_hat")]).compile(grid, **kw)
        free = _problem([("v", "vector")], eqs, {"Ek": 0.01},
                        ["dot(r_hat, v) = 0"]
                        + [f"dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
                        ).compile(grid, **kw)
        assert _pencils_agree(slip_inf, free)
        assert free.constraints["walls"]["vectors"]["v"]["tangential"] == "stress_free"

    @pytest.mark.parametrize("c", [1, 2])
    @pytest.mark.parametrize("m", [0, 1])
    def test_normal_stress_bessel_determinant(self, m, c):
        """The traction-free Stokes ball, dot(r_hat, dot(strain, r_hat)) - c p
        = 0 with the two tangential stress rows: the wall v_r faces stay
        degrees of freedom, the m = 0 gauge is lifted (the closed gradient
        wall row breaks the constant pressure), and the poloidal roots of
        the 2 x 2 Bessel determinant (l = 2, c = 2: 8, 28.211083; l = 1:
        14.978747 for every c) and the toroidal family k j_l'(k) = j_l(k)
        converge at second order."""
        targets = [("pol", 2, k**2) for k in traction_free_k(2, c)]
        targets += [("pol", 3, traction_free_k(3, c)[0]**2)]
        targets += [("tor", 2, roots(lambda k: k * spherical_jn(2, k, derivative=True)
                                     - spherical_jn(2, k), 1, lo=0.3)[0]**2)]
        rows = [f"dot(r_hat, {STRAIN}) - {c}*p = 0"]
        rows += [f"dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
        errs = []
        for Nr, Nt in RES_BENCH:
            pb = _problem([("v", "vector"), ("p", "scalar")],
                          ["dt(v) - lap(v) + grad(p) = 0", "div(v) = 0"], {}, rows)
            cp = pb.compile(ball(Nr, Nt), m=m, backend="fd2", discretization="staggered",
                            order_r=2, order_theta=2)
            rec = cp.constraints
            assert list(rec["walls"]["vectors"]["v"]["open"]) == [Boundary.R_OUTER]
            assert rec["walls"]["vectors"]["v"]["normal"] == ()
            assert all(info["gauge"] is None for info in rec["equations"].values()
                       if info["class"] == "saddle_point")
            v_r = cp.layout.field("v_r")
            assert v_r.size == (Nr + 1) * Nt - (Nt if abs(m) >= 2 else Nt - 1)
            errs.append([abs(_nearest(cp, t, l).real - t) / t for _, l, t in targets])
        for (kind, l, t), e1, e2 in zip(targets, *errs):
            assert e2 < 6e-3, (kind, l, t, e1, e2)
            assert order_of(e1, e2, 2.0) > 1.5, (kind, l, t, e1, e2)
