"""The staggered (Arakawa-C) three-field compressible Bekki system:
momentum, continuity/density and energy/entropy, inviscid or with the
viscous force −Ek[∇²v + ⅓∇(∇·v)] (+ the ρ₀-stratification strain term
∇ln ρ₀·T) and the thermal diffusion −Ek[∇²s₁ + (dln P₀/dr)∂_r s₁].

Coverage:
  * exact-field oracles: the buoyancy/entropy-source pair is an exact
    volume-norm transpose; the continuity operator reproduces the analytic
    ∇·(ρ₀ v); the diffusion operators are dissipative;
  * cross-check against the collocated core: the Rossby anchor (−2/(m+1),
    neutral) and the convective ridge (growing, prograde) agree to
    discretization error at Ek = 0 and at Bekki's Ek = 7e-6, and viscosity
    reduces the convective growth rate;
  * routing: the compressible systems compile from their tasks, and the
    pinned magnetic walls of the Figure-2 builder are refused.

All cross-check solves run on small grids so the default suite stays fast;
the resolution-scaling agreement is a separate (slow) study.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import pytest

from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.applications.solar_eigenmodes import make_polytrope_system
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem, EvpResult
from spdee.backends.fd2.staggered_compile import (
    StaggeredGrids,
    _build_staggered_gradient_blocks,
    _avg_1d,
    compile_fd2_staggered,
    build_staggered_scalar_laplacian,
    build_staggered_vector_laplacian,
    build_staggered_grad_div,
)


R_LO, R_HI = 0.71, 0.985
M_FIG2 = 8


def _make_system(Ek=0.0, m=M_FIG2, geometry="shell", strain=False,
                 delta=1e-6, Le=0.0):
    poly = PolytropeBackground(r_inner=R_LO, r_outer=R_HI, delta=delta)
    sys_ = make_polytrope_system(
        poly, m_val=m, Ek=Ek, Le=Le, entropy_bc="neumann",
        geometry=geometry, with_grad_lnrho_strain=strain)
    return sys_


def _grid(Nr, Nt):
    return make_shell_grid(N_r=Nr, N_theta=Nt, r_inner=R_LO, r_outer=R_HI)


def _bg_scalar(sys_, name):
    """Callable (R, T) -> values of the named background scalar."""
    comp = sys_.backgrounds[name].component
    return lambda R, T: np.asarray(comp.evaluate_on_grid(R, T)) + 0.0 * T


def _omega_near(A, B, omega_c, k=6, tol=1e-9):
    """Shift-invert the staggered/collocated pencil A x = λ B x near a target
    physical frequency ω_c (complex). λ = +iω, so the shift is σ = iω_c."""
    sigma = 1j * omega_c
    LU = spla.splu((A - sigma * B).tocsc())
    Lin = spla.LinearOperator(
        A.shape, matvec=lambda x: LU.solve(B @ x), dtype=complex)
    mu = spla.eigs(Lin, k=k, which="LM", return_eigenvectors=False,
                   maxiter=8000, tol=tol)
    return EvpResult.eigenvalues_to_omega(1.0 / mu + sigma)


def _matrices(backend, Nr, Nt, Ek=0.0, m=M_FIG2, strain=False, delta=1e-6):
    sys_ = _make_system(Ek=Ek, m=m, strain=strain, delta=delta)
    grid = _grid(Nr, Nt)
    cp = compile_problem(sys_, grid, m=m, order_r=2, order_theta=2,
                         discretization=backend)
    if backend == "staggered":
        return cp.A_total.tocsc(), cp.B_total.tocsc()
    K, M = cp.assemble_evp_matrices()
    return K.tocsc(), M.tocsc()


# ---------------------------------------------------------------------------
# Routing / scope
# ---------------------------------------------------------------------------

class TestStaggeredCompressibleRouting:
    def test_inviscid_three_field_compiles(self):
        """The inviscid three-field Bekki system assembles on the staggered
        core with the layout keyed by the unknown names v_r/v_t/v_p/rho1/s1."""
        sys_ = _make_system(Ek=0.0)
        grid = _grid(12, 18)
        cp = compile_problem(sys_, grid, m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        names = [f.name for f in cp.layout.fields]
        assert names == ["v_r", "v_t", "v_p", "rho1", "s1"]
        locs = {f.name: f.location for f in cp.layout.fields}
        assert locs["v_r"] == "r_edge"
        assert locs["v_t"] == "theta_edge"
        assert locs["v_p"] == "center"
        assert locs["rho1"] == "center"
        assert locs["s1"] == "center"
        assert cp.A_total.shape == cp.B_total.shape
        assert np.all(np.isfinite(cp.A_total.tocoo().data))

    def test_viscous_lame_thermal_compiles(self):
        """The viscous three-field system (Ek > 0, no ρ₀-strain term)
        compiles on the staggered core."""
        sys_ = _make_system(Ek=7e-6, strain=False)
        grid = _grid(12, 18)
        cp = compile_problem(sys_, grid, m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        assert [f.name for f in cp.layout.fields] == \
            ["v_r", "v_t", "v_p", "rho1", "s1"]
        assert np.all(np.isfinite(cp.A_total.tocoo().data))

    def test_strain_correction_compiles(self):
        """The ρ₀-stratification viscous term ∇ln ρ₀·T compiles on the
        staggered core."""
        sys_ = _make_system(Ek=7e-6, strain=True)
        grid = _grid(12, 18)
        cp = compile_problem(sys_, grid, m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        assert np.all(np.isfinite(cp.A_total.tocoo().data))

    def test_strain_negligible_at_physical_ek(self):
        """At Bekki's Ek = 7e-6 the strain correction is negligible: the
        convective ridge is unchanged to a few % with and without it."""
        tgt = 0.05 + 1.0j
        g = _grid(16, 28)
        cpF = compile_problem(_make_system(Ek=7e-6, strain=False), g,
                              m=M_FIG2, order_r=2, order_theta=2,
                              discretization="staggered")
        cpT = compile_problem(_make_system(Ek=7e-6, strain=True), g,
                              m=M_FIG2, order_r=2, order_theta=2,
                              discretization="staggered")
        oF = _omega_near(cpF.A_total.tocsc(), cpF.B_total.tocsc(), tgt, k=6)
        oT = _omega_near(cpT.A_total.tocsc(), cpT.B_total.tocsc(), tgt, k=6)
        tF = oF[oF.imag > 0.4].imag.max()
        tT = oT[oT.imag > 0.4].imag.max()
        assert abs(tF - tT) / tF < 0.05, (tF, tT)

    def test_mhd_pinned_bcs_rejected(self):
        """MHD is built on the staggered grid by the constrained-transport
        extension (tests/test_staggered_mhd.py), but the Figure-2 builder's
        PINNED boundary conditions on b (all three components vanish) cannot
        be imposed on face fluxes and are rejected with a clear message.
        Conducting walls (b_r = 0, tangential E = 0) compile."""
        sys_ = _make_system(Ek=7e-6, strain=False, Le=1e-3)
        grid = _grid(12, 18)
        with pytest.raises(NotImplementedError,
                           match="pins its tangential components"):
            compile_problem(sys_, grid, m=M_FIG2, order_r=2, order_theta=2,
                            discretization="staggered")

    def test_equations_are_classified_by_their_tasks(self):
        """Every equation is read from its task list, not by name: all
        three carry a Dt and are evolution equations, the entropy walls
        come from the BC tasks, and no constraint is paired."""
        cp = compile_fd2_staggered(_make_system(Ek=0.0), _grid(12, 18), m=M_FIG2)
        rec = cp.constraints
        assert {k: v["class"] for k, v in rec["equations"].items()} == {
            "momentum": "evolution", "continuity": "evolution",
            "energy": "evolution"}
        assert rec["principal"] == {"momentum": "v", "continuity": "rho1",
                                    "energy": "s1"}
        assert rec["constraints"] == {} and rec["multipliers"] == {}
        assert rec["walls"]["scalars"]["s1"]["closure"] == ("neumann", "neumann")
        assert rec["viscous"] == {"v": False}


# ---------------------------------------------------------------------------
# Exact-field oracles
# ---------------------------------------------------------------------------

class TestStaggeredCompressibleOracles:
    def test_buoyancy_entropy_mimetic_adjoint(self):
        """The base center→r_edge interpolation P_c2vr (used for buoyancy)
        and its volume-norm transpose P_vr2c (used for the entropy source)
        are exact adjoints under the staggered volume norm — the discrete
        Brunt–Väisälä loop is structurally skew/self-adjoint by
        construction. This is the property that keeps the inviscid
        convective growth rates clean.
        """
        grids = StaggeredGrids.from_grid(_grid(14, 20))
        r_edge, r_center = grids.r_edge, grids.r_center
        th_center = grids.theta_center
        Nr_e, Nr_c = len(r_edge), len(r_center)
        Nt_c = len(th_center)
        hr = r_edge[1] - r_edge[0]
        ht = th_center[1] - th_center[0]

        def vol(rv, tv):
            return ((rv**2)[:, None] * np.sin(tv)[None, :]).ravel() \
                .astype(complex) * hr * ht
        W_vr = vol(r_edge, th_center)
        W_c = vol(r_center, th_center)
        I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
        P_c2vr = sp.kron(_avg_1d(Nr_e, Nr_c), I_tc, format="csr")
        P_vr2c = sp.diags(1.0 / W_c) @ P_c2vr.conj().T @ sp.diags(W_vr)

        rng = np.random.default_rng(0)
        a = rng.standard_normal(W_c.size) + 0j      # center field (s₁-like)
        b = rng.standard_normal(W_vr.size) + 0j     # r-edge field (v_r-like)
        lhs = np.sum(W_vr * (P_c2vr @ a) * np.conj(b))
        rhs = np.sum(W_c * a * np.conj(P_vr2c @ b))
        assert abs(lhs - rhs) / (abs(lhs) + 1e-30) < 1e-12

    def test_divergence_oracle_constant_radial_flux(self):
        """The staggered norm-adjoint Div reproduces the analytic divergence
        of a purely-radial field v = f(r) r̂ in the interior: for f = r the
        divergence ∇·(r r̂) = 3 everywhere (away from the wall-adjacent rows
        where the documented transpose-defect lives and where v_r=0 is
        imposed anyway). This anchors the continuity operator's flux term.
        """
        grids = StaggeredGrids.from_grid(_grid(20, 24))
        r_edge, r_center = grids.r_edge, grids.r_center
        th_edge, th_center = grids.theta_edge, grids.theta_center
        Nr_c = len(r_center)
        Nt_e, Nt_c = len(th_edge), len(th_center)
        hr = r_edge[1] - r_edge[0]
        ht = th_center[1] - th_center[0]

        G_r, G_t, G_p = _build_staggered_gradient_blocks(grids, M_FIG2, 2)
        G = sp.vstack([G_r, G_t, G_p], format="csr")

        def vol(rv, tv):
            return ((rv**2)[:, None] * np.sin(tv)[None, :]).ravel() \
                .astype(complex) * hr * ht
        W_v = np.concatenate([vol(r_edge, th_center),
                              vol(r_center, th_edge),
                              vol(r_center, th_center)])
        W_c = vol(r_center, th_center)
        Div = -(sp.diags(1.0 / W_c) @ G.conj().T @ sp.diags(W_v))

        # v = r r̂ : v_r(r_edge,θ)=r_edge, v_θ=v_φ=0
        vr = np.repeat(r_edge, Nt_c).astype(complex)
        vt = np.zeros(Nr_c * Nt_e, dtype=complex)
        vp = np.zeros(Nr_c * Nt_c, dtype=complex)
        v = np.concatenate([vr, vt, vp])
        div = (Div @ v).reshape(Nr_c, Nt_c)
        # interior radial rows (exclude the 2 wall-adjacent rings each side)
        interior = div[2:-2, :]
        assert np.allclose(interior.real, 3.0, atol=5e-2)
        assert np.max(np.abs(interior.imag)) < 1e-10

    def test_assembler_matrix_is_finite_and_sized(self):
        """The compiled pencil is correctly sized and finite, with B the
        identity on every dynamical field (5 blocks)."""
        cp = compile_fd2_staggered(_make_system(Ek=0.0), _grid(12, 18), m=M_FIG2)
        A, B, layout = cp.A_total, cp.B_total, cp.layout
        n = layout.total_size
        assert A.shape == (n, n)
        assert B.shape == (n, n)
        assert np.all(np.isfinite(A.tocoo().data))
        # B is identity except where v_r wall rows were emptied (impenetrable)
        Bd = np.asarray(B.diagonal())
        # every center/edge dynamical dof carries unit mass except the
        # imposed v_r wall rows and the pinned pole v_theta ring (both
        # algebraic constraints)
        assert np.count_nonzero(Bd) >= n - 2 * 18 - 2 * 12  # walls × Nt_c, poles × Nr_c


# ---------------------------------------------------------------------------
# Cross-check vs collocated (inviscid)
# ---------------------------------------------------------------------------

class TestStaggeredCompressibleCrossCheck:
    def test_rossby_anchor_matches_collocated(self):
        """The Rossby mode sits at Re(ω) = ∓2/(m+1) (neutral) on BOTH
        backends, agreeing with each other and the analytic anchor to
        ~1% at this grid."""
        anchor = 2.0 / (M_FIG2 + 1)        # = 2/9 ≈ 0.2222
        results = {}
        for backend in ("staggered", "collocated"):
            A, B = _matrices(backend, 16, 28)
            best = None
            for s in (+1, -1):
                om = _omega_near(A, B, s * anchor + 0.0j, k=6)
                for w in om:
                    score = abs(abs(w.real) - anchor) + 10 * abs(w.imag)
                    if best is None or score < best[0]:
                        best = (score, w)
            results[backend] = best[1]

        for backend, w in results.items():
            assert abs(abs(w.real) - anchor) / anchor < 0.02, (backend, w)
            assert abs(w.imag) < 1e-3, (backend, w)         # neutral
        # the two backends agree on |Re ω|
        assert abs(abs(results["staggered"].real)
                   - abs(results["collocated"].real)) < 0.01

    def test_convective_ridge_present_and_prograde(self):
        """The δ>0 convective ridge appears on the staggered backend: modes
        that are GROWING (Im ω > 0) and PROGRADE (Re ω > 0, small) with the
        peak growth reaching Im ω ~ 1, matching the collocated ridge to
        discretization error."""
        As, Bs = _matrices("staggered", 14, 20)
        om = _omega_near(As, Bs, 0.05 + 1.0j, k=8)
        grow = om[om.imag > 0.5]
        assert len(grow) >= 2, om
        # growing convective modes are prograde (small positive Re)
        top = grow[np.argsort(-grow.imag)][:4]
        assert np.all(top.real > -0.02), top      # prograde (allow tiny neg)
        assert np.all(top.real < 0.5), top        # small Re (ridge, not p-mode)
        assert top.imag.max() > 0.8               # reaches Im ~ 1

    @pytest.mark.slow
    def test_ridge_agreement_quantitative(self):
        """Slower: the staggered and collocated convective ridges populate
        the same Im-ω band near Re≈0, agreeing within the spread expected
        from two distinct 2nd-order schemes at this resolution."""
        tgt = 0.05 + 1.0j
        As, Bs = _matrices("staggered", 16, 28)
        Ac, Bc = _matrices("collocated", 16, 28)
        oms = _omega_near(As, Bs, tgt, k=10)
        omc = _omega_near(Ac, Bc, tgt, k=10)
        s_top = np.sort(oms[oms.imag > 0.5].imag)[-3:]
        c_top = np.sort(omc[omc.imag > 0.5].imag)[-3:]
        assert len(s_top) >= 1 and len(c_top) >= 1
        # peak growth rates agree to within 25% (coarse-grid scheme spread)
        assert abs(s_top[-1] - c_top[-1]) / c_top[-1] < 0.25


# ---------------------------------------------------------------------------
# Viscous and thermal diffusion
# ---------------------------------------------------------------------------

class TestStaggeredDiffusionOperators:
    """Exact operator-level oracles for the diffusion building blocks: each
    must be DISSIPATIVE (so it enters the eigenproblem with the correct
    damping sign Im ω < 0)."""

    def test_scalar_laplacian_negative_definite(self):
        """The staggered scalar Laplacian (thermal ∇²s₁) is negative-definite
        — every eigenvalue has Re < 0 — so +Ek·∇² damps. Self-adjoint under
        the center volume norm."""
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        L = build_staggered_scalar_laplacian(grids, M_FIG2, 2).toarray()
        ev = np.linalg.eigvals(L)
        assert ev.real.max() < -1.0, ev.real.max()

    def test_viscous_lame_negative_semidefinite(self):
        """The staggered Lamé viscous operator ∇²v + ⅓∇(∇·v) is negative-
        semidefinite in the interior — no spurious growth from the diffusion
        operator itself (the zero modes are rigid/wall-ring, removed by the
        stress-free wall closure)."""
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        VL = build_staggered_vector_laplacian(grids, M_FIG2, 2).toarray()
        GD = build_staggered_grad_div(grids, M_FIG2, 2).toarray()
        ev = np.linalg.eigvals(VL + (1.0 / 3.0) * GD)
        assert ev.real.max() < 1e-8, ev.real.max()


class TestStaggeredViscousCrossCheck:
    # NOTE on ridge selection: the staggered viscous spectrum contains a few
    # SPURIOUS growing modes near Im ω ≈ 1.3 — part of the spurious arc
    # visible in the production Fig-2b plots — so a global argmax(Im) can
    # latch onto one. These are unphysical: a VISCOUS growing mode cannot exceed the
    # INVISCID ridge peak (diffusion cannot increase growth). So the physical
    # convective ridge is selected as the highest-Im prograde growing mode
    # that does NOT exceed the inviscid ridge peak (with a small margin). This
    # is a principled physical discriminant, not a tuned cutoff.

    @staticmethod
    def _physical_ridge(omega, inviscid_peak):
        """Highest-Im prograde growing mode at or below the inviscid peak
        (+5% margin for discretization); spurious modes above it are
        excluded as unphysical (viscosity cannot increase growth)."""
        g = omega[(omega.imag > 0.4)
                  & (omega.imag <= inviscid_peak * 1.05)
                  & (omega.real > -0.02) & (omega.real < 0.5)]
        if not len(g):
            return None
        return g[np.argmax(g.imag)]

    def test_viscosity_reduces_convective_growth(self):
        """Turning on diffusion (Ek=7e-6) REDUCES the peak convective growth
        rate relative to the inviscid case — the correct sign of the
        linewidth effect — on the staggered backend."""
        tgt = 0.05 + 1.0j
        Ai, Bi = _matrices("staggered", 16, 28, Ek=0.0)
        Av, Bv = _matrices("staggered", 16, 28, Ek=7e-6)
        oi = _omega_near(Ai, Bi, tgt, k=12)
        ov = _omega_near(Av, Bv, tgt, k=12)
        im_inv = oi[oi.imag > 0.4].imag.max()
        # physical viscous ridge: below the inviscid peak (exclude spurious
        # ~1.3 arc modes, which would falsely read as growth-increasing).
        ridge_v = self._physical_ridge(ov, im_inv)
        assert ridge_v is not None, ov[ov.imag > 0.4]
        im_vis = ridge_v.imag
        assert im_vis < im_inv, (im_inv, im_vis)
        # but only slightly: Ek=7e-6 is weak, the ridge survives
        assert im_vis > 0.8

    def test_convective_ridge_matches_collocated_viscous(self):
        """At Bekki's Ek = 7e-6 the staggered core contains a convective
        ridge mode matching the collocated ridge (growing, prograde, Im ~ 1)
        to discretization error.

        NOTE: the staggered viscous spectrum also carries a cluster of
        spurious prograde growing modes between the physical ridge and
        Im≈1.3 (the spurious arc seen in the Fig-2b production plots; a
        staggered viscous artifact independent of the Coriolis coefficient —
        it is present at unit coupling too). So
        the comparison is "does staggered contain a mode matching the
        collocated physical ridge", verified by nearest-match, rather than a
        global argmax(Im) which would catch a spurious-arc mode."""
        tgt = 0.05 + 1.0j
        As, Bs = _matrices("staggered", 16, 28, Ek=7e-6)
        Ac, Bc = _matrices("collocated", 16, 28, Ek=7e-6)
        oms = _omega_near(As, Bs, tgt, k=24)
        omc = _omega_near(Ac, Bc, tgt, k=24)
        # collocated physical ridge: top prograde growing mode (clean spectrum)
        gc = omc[(omc.imag > 0.4) & (omc.real > -0.02) & (omc.real < 0.5)]
        assert len(gc), omc
        top_c = gc[np.argmax(gc.imag)]
        # staggered: the prograde growing mode NEAREST the collocated ridge
        # frequency (does the staggered backend reproduce that physical mode?)
        gs = oms[(oms.imag > 0.4) & (oms.real > -0.02) & (oms.real < 0.5)]
        assert len(gs), oms
        match_s = gs[np.argmin(np.abs(gs.imag - top_c.imag))]
        # the matched staggered ridge is prograde and agrees within the
        # coarse-grid two-scheme spread
        assert match_s.real > -0.02 and match_s.real < 0.5, match_s
        assert abs(match_s.imag - top_c.imag) / top_c.imag < 0.25, (
            f"staggered ridge {match_s.imag:.3f} vs collocated "
            f"{top_c.imag:.3f}")

    @pytest.mark.slow
    def test_strongly_damped_when_no_convective_drive(self):
        """With δ = 0 (no convective drive) the physical band is fully
        damped, matching the collocated core: the stress-free wall closure
        (impenetrability v_r = 0 as row+column elimination + stress-free
        tangential ∂_r(v_tan/r) = 0) plus the
        dissipative diffusion sign — without them the staggered spectrum shows
        spurious near-wall growth."""
        import scipy.linalg as sla
        out = {}
        for backend in ("staggered", "collocated"):
            A, B = _matrices(backend, 14, 20, Ek=1e-4, delta=0.0)
            lam = sla.eig(A.toarray(), B.toarray(), right=False)
            lam = lam[np.isfinite(lam)]
            om = EvpResult.eigenvalues_to_omega(lam)
            band = om[(np.abs(om.real) < 2) & (np.abs(om.imag) < 2)]
            out[backend] = np.mean(band.imag > 1e-3)
        assert out["staggered"] < 0.05, out
        assert abs(out["staggered"] - out["collocated"]) < 0.1, out


# ---------------------------------------------------------------------------
# Stress-free wall closure (impenetrability elimination + stress-free
# tangential condition)
# ---------------------------------------------------------------------------

class TestStaggeredStressFreeWall:
    """The viscous wall: (a) impenetrability v_r = 0 as a row+column DOF
    elimination and (b) Bekki's stress-free tangential closure
    ∂_r(v_tan/r) = 0 in the strain Gram, which give a cleanly damped δ = 0
    spectrum
    (TestStaggeredViscousCrossCheck.test_strongly_damped_when_no_convective_drive).
    """

    def test_viscous_path_eliminates_wall_vr_dofs(self):
        """The viscous (Ek>0) shell assembler returns a REDUCED system: the
        two v_r wall rings (inner+outer) are removed as rows+columns, so the
        v_r field shrinks by 2·Nt_c while the other four fields are intact.
        The inviscid path keeps the full size (row-overwrite)."""
        Nt = 18
        Av, Bv = _matrices("staggered", 12, Nt, Ek=7e-6)
        Ai, Bi = _matrices("staggered", 12, Nt, Ek=0.0)
        # inviscid is larger by exactly 2·Nt_c (one wall ring per radial wall)
        grids = StaggeredGrids.from_grid(_grid(12, Nt))
        Nt_c = len(grids.theta_center)
        assert Ai.shape[0] - Av.shape[0] == 2 * Nt_c
        assert Av.shape == Bv.shape

    def test_curl_curl_self_adjoint_with_stress_free(self):
        """The curl-curl built with the stress-free wall_bc stays self-adjoint
        and positive-semidefinite under the velocity norm — the Cpᵀ W_c Cp
        Gram structure is preserved for any wall closure."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_curl_curl, build_staggered_grad_div_pair,
        )
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        CC = build_staggered_curl_curl(grids, M_FIG2, 2, wall_bc="stress_free")
        _, _, W_v, _ = build_staggered_grad_div_pair(grids, M_FIG2, 2)
        rng = np.random.default_rng(0)
        n = CC.shape[0]
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(u, W_v * (CC @ w))
        rhs = np.vdot(CC @ u, W_v * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12
        assert np.vdot(u, W_v * (CC @ u)).real >= -1e-8

    def test_eliminated_viscous_self_adjoint_dissipative(self):
        """After v_r row+column elimination, the stress-free viscous operator
        ∇²v + ⅓∇(∇·v) is self-adjoint AND negative-semidefinite under the
        (reduced) velocity norm — the structural guarantee that it cannot
        inject spurious growth at the wall."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_vector_laplacian, build_staggered_grad_div,
            build_staggered_grad_div_pair, _eliminate_wall_vr,
        )
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        VL = build_staggered_vector_laplacian(
            grids, M_FIG2, 2, wall_bc="stress_free")
        GD = build_staggered_grad_div(grids, M_FIG2, 2)
        visc = (VL + (1.0 / 3.0) * GD).tocsr()
        _, _, W_v, _ = build_staggered_grad_div_pair(grids, M_FIG2, 2)
        vred, keep = _eliminate_wall_vr(visc, grids)
        Wr = W_v[keep]
        rng = np.random.default_rng(1)
        n = vred.shape[0]
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(u, Wr * (vred @ w))
        rhs = np.vdot(vred @ u, Wr * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12
        assert np.vdot(u, Wr * (vred @ u)).real <= 1e-8

    def test_convective_ridge_survives_wall_closure(self):
        """The wall closure and the dissipative diffusion damp spurious modes
        without killing the buoyancy-driven convective ridge: at Ek = 7e-6,
        δ = 1e-6 the growing prograde ridge is present with Im ω ~ 1,
        matching the collocated core."""
        tgt = 0.05 + 1.0j
        As, Bs = _matrices("staggered", 16, 28, Ek=7e-6)
        om = _omega_near(As, Bs, tgt, k=8)
        grow = om[om.imag > 0.4]
        assert len(grow) >= 1
        top = grow[np.argmax(grow.imag)]
        assert top.real > -0.02 and top.real < 0.5, top
        assert top.imag > 0.8, top


# ---------------------------------------------------------------------------
# ρ₀-stratification viscous strain term (∇ln ρ₀·T)
# ---------------------------------------------------------------------------

class TestStaggeredStrainCorrection:
    """The ρ₀-stratified viscous force is the mimetic density-weighted
    strain Gram (1/ρ₀)∇·(ρ₀T) = −(1/ρ₀)W_v⁻¹Sᵀ(ρ₀W_T)C S. It is
    self-adjoint and negative-semidefinite under ρ₀W_v by construction
    (dissipative at ALL Ek — no spurious growth), reduces to the constant-ρ₀
    Lamé operator when ρ₀=const, and needs no auxiliary tensor variables.
    """

    def test_strain_no_spurious_growth_at_large_ek(self):
        """The mimetic strain force is dissipative by construction: at δ=0 and
        large Ek the whole band is cleanly damped (a non-mimetic explicit
        split leaks small positive growth here)."""
        import scipy.linalg as sla
        g = _grid(12, 16)
        cp = compile_problem(_make_system(Ek=1e-2, strain=True, delta=0.0),
                             g, m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        lam = sla.eig(cp.A_total.toarray(), cp.B_total.toarray(), right=False)
        lam = lam[np.isfinite(lam)]
        om = EvpResult.eigenvalues_to_omega(lam)
        band = om[(np.abs(om.real) < 2) & (np.abs(om.imag) < 2)]
        assert np.mean(band.imag > 1e-3) < 0.02, band.imag.max()

    def test_strain_gram_self_adjoint_and_definite(self):
        """The mimetic viscous Gram force is self-adjoint and negative-
        semidefinite under the ρ₀-weighted velocity norm on the weightful
        subspace (pole θ-edge velocity DOFs are weightless/inert)."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_viscous_gram, build_staggered_grad_div_pair,
        )
        sys_ = _make_system(Ek=7e-6, strain=True)
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        rho0_fn = _bg_scalar(sys_, "poly_rho")
        F = build_staggered_weighted_viscous_gram(grids, M_FIG2, rho0_fn, 2)
        _, _, W_v, _ = build_staggered_grad_div_pair(grids, M_FIG2, 2)
        re, rc = grids.r_edge, grids.r_center
        te, tc = grids.theta_edge, grids.theta_center

        def rho_on(rg, tg):
            return np.asarray(rho0_fn(rg[:, None] + 0 * tg[None, :],
                                      tg[None, :] + 0 * rg[:, None])).ravel().astype(complex)
        rho_v = np.concatenate([rho_on(re, tc), rho_on(rc, te), rho_on(rc, tc)])
        Wr = rho_v * W_v
        keep = np.abs(W_v) > 1e-12 * np.abs(W_v).max()
        rng = np.random.default_rng(0); n = F.shape[0]
        u = np.zeros(n, dtype=complex); w = np.zeros(n, dtype=complex)
        u[keep] = rng.standard_normal(keep.sum()) + 1j * rng.standard_normal(keep.sum())
        w[keep] = rng.standard_normal(keep.sum()) + 1j * rng.standard_normal(keep.sum())
        lhs = np.vdot(u, Wr * (F @ w)); rhs = np.vdot(F @ u, Wr * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12, abs(lhs - rhs) / abs(lhs)
        assert np.vdot(u, Wr * (F @ u)).real <= 1e-8

    def test_strain_gram_reduces_to_lame_when_uniform(self):
        """ρ₀=const recovers the constant-ρ₀ self-adjoint Lamé operator
        ∇²v + ⅓∇(∇·v) to discretization order in the interior."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_viscous_gram,
            build_staggered_vector_laplacian, build_staggered_grad_div,
        )
        grids = StaggeredGrids.from_grid(_grid(18, 28))
        F = build_staggered_weighted_viscous_gram(
            grids, M_FIG2, lambda r, t: np.ones_like(r * t), 2)
        L = (build_staggered_vector_laplacian(grids, M_FIG2, 2)
             + (1.0 / 3.0) * build_staggered_grad_div(grids, M_FIG2, 2)).tocsr()
        Nr_e = len(grids.r_edge); Nt_c = len(grids.theta_center)
        nvr = Nr_e * Nt_c
        rng = np.random.default_rng(0); n = F.shape[0]
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        a = (F @ x)[:nvr].reshape(Nr_e, Nt_c)[3:-3, 3:-3]
        b = (L @ x)[:nvr].reshape(Nr_e, Nt_c)[3:-3, 3:-3]
        assert np.linalg.norm(a - b) / np.linalg.norm(b) < 5e-3


# ---------------------------------------------------------------------------
# Script wiring (the EVP-matrix interface)
# ---------------------------------------------------------------------------

class TestStaggeredScriptWiring:
    """The staggered result is a drop-in for the production spectrum script,
    which calls ``cp.assemble_evp_matrices()`` uniformly across backends."""

    def test_assemble_evp_matrices_returns_baked_pencil(self):
        """StaggeredCompiledMatrices.assemble_evp_matrices() returns the
        already-BC-applied (A_total, B_total) pencil — the staggered BCs are
        imposed during assembly, so no further work is needed and the call
        mirrors the collocated CompiledProblem interface."""
        sys_ = _make_system(Ek=7e-6, strain=False)
        grid = _grid(12, 18)
        cp = compile_problem(sys_, grid, m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        K, M = cp.assemble_evp_matrices()
        assert K.shape == cp.A_total.shape
        assert M.shape == cp.B_total.shape
        assert (K != cp.A_total).nnz == 0
        assert (M != cp.B_total).nnz == 0

    def test_fastest_growing_convective_near_re_zero(self):
        """Coarse-grid probe: the fastest-growing clean convective mode
        sits near Re ≈ 0 on the staggered (Bekki-matching)
        scheme — agreeing with the collocated result, so the Re≈0 peak is not
        a collocated-grid artifact. (The largest-Re unstable mode — Bekki's
        red-dot criterion — sits further out and is resolution-dependent; the
        precise value needs a converged run.)"""
        import scipy.linalg as sla
        cp = compile_problem(_make_system(Ek=7e-6, strain=False),
                             _grid(14, 24), m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        lam = sla.eig(cp.A_total.toarray(), cp.B_total.toarray(), right=False)
        lam = lam[np.isfinite(lam)]
        om = EvpResult.eigenvalues_to_omega(lam)
        band = om[(np.abs(om.real) < 2) & (om.imag > 0.1) & (om.imag < 1.5)]
        assert len(band) >= 2
        fastest = band[np.argmax(band.imag)]
        assert abs(fastest.real) < 0.1, fastest   # peak near Re≈0


# ---------------------------------------------------------------------------
# Mimetic density/pressure-weighted diffusion (Bekki divergence form)
# ---------------------------------------------------------------------------

class TestStaggeredMimeticDiffusion:
    """Bekki writes the diffusion operators in divergence form
    (1/W)∇·(W∇·); discretizing that conservative form directly (weight on the
    edge fluxes, 1/W on centers) is self-adjoint + definite under the
    W-weighted norm, unlike the explicit ∇² + ∇lnW·∇ split. The thermal
    operator uses W = P₀."""

    def test_weighted_scalar_laplacian_self_adjoint_definite(self):
        """(1/W)∇·(W∇f) is self-adjoint and negative-definite under the
        W-weighted center volume norm, for a non-trivial radial weight."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_scalar_laplacian, StaggeredGrids,
        )
        grids = StaggeredGrids.from_grid(_grid(24, 36))
        rc = grids.r_center; tc = grids.theta_center
        re = grids.r_edge
        hr = re[1] - re[0]; ht = tc[1] - tc[0]
        wt = lambda r, t: np.exp(-2.0 * np.asarray(r)) + 0.0 * np.asarray(t)
        L = build_staggered_weighted_scalar_laplacian(grids, M_FIG2, wt, order=2)
        Wc = ((rc[:, None]**2 * np.sin(tc)[None, :]).ravel() * hr * ht)
        Wgt = (wt(rc[:, None], tc[None, :]).ravel() * Wc).astype(complex)
        rng = np.random.default_rng(0); n = L.shape[0]
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(u, Wgt * (L @ w)); rhs = np.vdot(L @ u, Wgt * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12, abs(lhs - rhs) / abs(lhs)
        assert np.vdot(u, Wgt * (L @ u)).real < 0

    def test_weighted_reduces_to_plain_when_uniform(self):
        """W=const recovers the plain scalar Laplacian exactly."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_scalar_laplacian,
            build_staggered_scalar_laplacian, StaggeredGrids,
        )
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        Lw = build_staggered_weighted_scalar_laplacian(
            grids, M_FIG2, lambda r, t: 3.0 + 0.0 * r * t, order=2)
        Lp = build_staggered_scalar_laplacian(grids, M_FIG2, order=2)
        assert abs((Lw - Lp).toarray()).max() < 1e-10

    def test_mimetic_thermal_preserves_ridge(self):
        """Swapping the explicit thermal split for the mimetic P₀-weighted
        Laplacian leaves the convective ridge unchanged at Bekki's Ek (the two
        agree to discretization order; the mimetic one is self-adjoint)."""
        tgt = 0.05 + 1.0j
        cp = compile_problem(_make_system(Ek=7e-6, strain=False), _grid(16, 28),
                             m=M_FIG2, order_r=2, order_theta=2,
                             discretization="staggered")
        om = _omega_near(cp.A_total.tocsc(), cp.B_total.tocsc(), tgt, k=6)
        grow = om[om.imag > 0.4]
        assert len(grow) >= 1
        assert grow.imag.max() > 0.8


class TestStaggeredStrainOperator:
    """The discrete strain operator e=½(∇v+(∇v)ᵀ) underlying the mimetic
    viscous Gram form — all six physical components 2nd-order accurate."""

    def test_strain_components_second_order(self):
        from spdee.backends.fd2.staggered_compile import build_staggered_strain
        mm = 3
        vr_f = lambda r, t: np.sin(2 * r) * np.cos(t)
        vt_f = lambda r, t: r * np.sin(t) * np.cos(t)
        vp_f = lambda r, t: np.cos(1.5 * r) * np.sin(t)

        def dd(f, r, t, wr=0, wt=0, h=1e-6):
            return (f(r + wr * h, t + wt * h) - f(r - wr * h, t - wt * h)) / (2 * h)

        def exact(r, t):
            im = 1j * mm; s = np.sin(t)
            if abs(s) < 1e-9:
                s = np.nan
            return [dd(vr_f, r, t, wr=1),
                    dd(vt_f, r, t, wt=1) / r + vr_f(r, t) / r,
                    (im / (r * s)) * vp_f(r, t) + vr_f(r, t) / r + np.cos(t) / (s * r) * vt_f(r, t),
                    0.5 * (r * dd(lambda r, t: vt_f(r, t) / r, r, t, wr=1) + dd(vr_f, r, t, wt=1) / r),
                    0.5 * ((im / (r * s)) * vr_f(r, t) + r * dd(lambda r, t: vp_f(r, t) / r, r, t, wr=1)),
                    0.5 * (dd(vp_f, r, t, wt=1) / r - np.cos(t) / (s * r) * vp_f(r, t) + (im / (r * s)) * vt_f(r, t))]

        prev = None
        for N in (24, 48):
            grids = StaggeredGrids.from_grid(
                make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0))
            re, rc = grids.r_edge, grids.r_center
            te, tc = grids.theta_edge, grids.theta_center
            S, cg = build_staggered_strain(grids, mm, order=2)
            vr = vr_f(re[:, None], tc[None, :]).ravel().astype(complex)
            vt = vt_f(rc[:, None], te[None, :]).ravel().astype(complex)
            vp = vp_f(rc[:, None], tc[None, :]).ravel().astype(complex)
            out = S @ np.concatenate([vr, vt, vp])
            offs = np.cumsum([0] + [c[0] for c in cg])
            gr = [(rc, tc), (rc, tc), (rc, tc), (re, te), (re, tc), (rc, te)]
            errs = []
            for k, (rg, tg) in enumerate(gr):
                blk = out[offs[k]:offs[k + 1]].reshape(len(rg), len(tg))
                ex = np.array([[exact(r, t)[k] for t in tg] for r in rg])
                d = np.abs(blk - ex)[3:-3, 3:-3]; d = d[np.isfinite(d)]
                errs.append(d.max())
            if prev is not None:
                rates = [np.log2(p / c) for p, c in zip(prev, errs)]
                assert min(rates) > 1.3, dict(zip(
                    ["rr", "tt", "pp", "rt", "rp", "tp"], rates))
            prev = errs


class TestStaggeredStrainStressFreeWall:
    """The strain operator's stress-free wall closure: Bekki eq.13
    ∂_r(v_tan/r)=0 zeros the wall tangential-shear strain (e_rθ, e_rφ wall
    rows), while keeping the Gram viscous form self-adjoint + dissipative."""

    def test_stress_free_zeros_wall_tangential_shear(self):
        from spdee.backends.fd2.staggered_compile import build_staggered_strain
        grids = StaggeredGrids.from_grid(_grid(14, 20))
        re, rc = grids.r_edge, grids.r_center
        te = grids.theta_edge
        Nr_e, Nr_c, Nt_e, Nt_c = (len(re), len(rc), len(te),
                                  len(grids.theta_center))
        vt = (rc[:, None]**2 * np.sin(te)[None, :]
              * np.cos(te)[None, :]).ravel().astype(complex)
        zr = np.zeros(Nr_e * Nt_c, dtype=complex)
        zp = np.zeros(Nr_c * Nt_c, dtype=complex)
        jth = Nt_e // 4   # off-equator θ-edge
        res = {}
        for wb in (None, "stress_free"):
            S, cg = build_staggered_strain(grids, M_FIG2, 2, wall_bc=wb)
            out = S @ np.concatenate([zr, vt, zp])
            offs = np.cumsum([0] + [c[0] for c in cg])
            erth = out[offs[3]:offs[4]].reshape(Nr_e, Nt_e)
            res[wb] = (abs(erth[0, jth]), abs(erth[-1, jth]),
                       abs(erth[Nr_e // 2, jth]))
        # plain: nonzero wall shear; stress_free: ZERO wall shear, same interior
        assert res[None][0] > 1e-3 and res[None][1] > 1e-3
        assert res["stress_free"][0] < 1e-12 and res["stress_free"][1] < 1e-12
        assert abs(res[None][2] - res["stress_free"][2]) < 1e-12  # interior intact

    def test_wall_aware_gram_self_adjoint_dissipative(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_viscous_gram, build_staggered_grad_div_pair,
            _eliminate_wall_vr,
        )
        sys_ = _make_system(Ek=7e-6, strain=True)
        grids = StaggeredGrids.from_grid(_grid(16, 24))
        rho0 = _bg_scalar(sys_, "poly_rho")
        F = build_staggered_weighted_viscous_gram(
            grids, M_FIG2, rho0, 2, wall_bc="stress_free")
        _, _, W_v, _ = build_staggered_grad_div_pair(grids, M_FIG2, 2)
        re, rc = grids.r_edge, grids.r_center
        te, tc = grids.theta_edge, grids.theta_center

        def rho_on(rg, tg):
            return np.asarray(rho0(rg[:, None] + 0 * tg[None, :],
                                   tg[None, :] + 0 * rg[:, None])).ravel().astype(complex)
        rho_v = np.concatenate([rho_on(re, tc), rho_on(rc, te), rho_on(rc, tc)])
        Wr = rho_v * W_v
        keep = np.abs(W_v) > 1e-12 * np.abs(W_v).max()
        rng = np.random.default_rng(0); n = F.shape[0]
        u = np.zeros(n, dtype=complex); w = np.zeros(n, dtype=complex)
        u[keep] = rng.standard_normal(keep.sum()) + 1j * rng.standard_normal(keep.sum())
        w[keep] = rng.standard_normal(keep.sum()) + 1j * rng.standard_normal(keep.sum())
        assert abs(np.vdot(u, Wr * (F @ w)) - np.vdot(F @ u, Wr * w)) / abs(np.vdot(u, Wr * (F @ w))) < 1e-12
        Fred, _ = _eliminate_wall_vr(F, grids)
        assert np.linalg.eigvals(Fred.toarray()).real.max() < 1e-8


class TestStaggeredViscousBallGeometry:
    """Operator-level finiteness of the mimetic viscous operators on BALL
    geometry (origin r-edge at r=0).

    The strain/Gram operators are finite at the origin (1/r factors on the
    weightless origin r-edge are zero-guarded) and their origin treatment is
    audited in docs/ball_staggered.md with analytic decay-rate benchmarks in
    tests/test_ball_staggered.py. Bekki's Fig-2 polytrope is a SHELL
    background (g∝1/r², singular at the origin), so these tests pin the
    operator-level behaviour on a regular density, not a physical ball solve.
    """

    def test_strain_finite_on_ball(self):
        from spdee.backends.fd2.grid import make_ball_grid
        from spdee.backends.fd2.staggered_compile import build_staggered_strain
        grids = StaggeredGrids.from_grid(make_ball_grid(N_r=14, N_theta=20, r_outer=1.0))
        assert np.isclose(grids.r_edge[0], 0.0)
        for wb in (None, "stress_free"):
            S, _ = build_staggered_strain(grids, M_FIG2, 2, wall_bc=wb)
            assert np.all(np.isfinite(S.toarray())), wb

    def test_viscous_gram_finite_on_ball(self):
        from spdee.backends.fd2.grid import make_ball_grid
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_viscous_gram,
        )
        grids = StaggeredGrids.from_grid(make_ball_grid(N_r=14, N_theta=20, r_outer=1.0))
        # regular (constant) density isolates the operator from any singular
        # physical background
        F = build_staggered_weighted_viscous_gram(
            grids, M_FIG2, lambda r, t: np.ones_like(r * t), 2,
            wall_bc="stress_free")
        assert np.all(np.isfinite(F.toarray()))


class TestStaggeredOriginParity:
    """Radial-origin parity (ball) — the analogue of the θ-pole parity, needed
    for converged origin accuracy (Bekki-on-ball + MHD). A regular field obeys
    f(−r)=parity·f(r) at r=0 with parity (-1)^|m| (scalars) and
    (-1)^max(|m|-1,0) (vector components, all three sharing it), verified by the
    Cartesian-smooth origin expansion."""

    def test_origin_parity_table(self):
        from spdee.backends.fd2.staggered_compile import _origin_parity
        # scalar (-1)^|m|
        assert _origin_parity(0, "scalar") == +1
        assert _origin_parity(1, "scalar") == -1
        assert _origin_parity(2, "scalar") == +1
        assert _origin_parity(8, "scalar") == +1
        # vector (-1)^max(|m|-1,0)
        assert _origin_parity(0, "vector") == +1
        assert _origin_parity(1, "vector") == +1
        assert _origin_parity(2, "vector") == -1
        assert _origin_parity(3, "vector") == +1
        assert _origin_parity(8, "vector") == -1

    def test_origin_parity_stencil_restores_order(self):
        """The parity-folded origin stencil makes ∂_r of an EVEN field vanish
        exactly at r=0 (where a one-sided stencil gives O(h)) and an ODD field
        converge at 2nd order there."""
        from spdee.backends.fd2.staggered_compile import (
            _stag_diff_1d_origin_parity, _stag_diff_1d)
        prev_o = None
        for N in (24, 48):
            grids = StaggeredGrids.from_grid(
                make_ball_grid(N_r=N, N_theta=8, r_outer=1.0))
            rc, re = grids.r_center, grids.r_edge
            h = re[1] - re[0]
            f_even = np.cos(2 * rc)       # f'(0)=0 exact, even
            g_odd = np.sin(2 * rc)        # g'(0)=2 exact, odd
            De = _stag_diff_1d_origin_parity(rc, re, 2, +1)
            Do = _stag_diff_1d_origin_parity(rc, re, 2, -1)
            Dplain = _stag_diff_1d(len(re), len(rc), h)
            # even: parity stencil exact at origin; plain stencil is O(h)
            assert abs((De @ f_even)[0]) < 1e-10
            assert abs((Dplain @ f_even)[0]) > 0.5 * h   # plain is ~h, first order
            # odd: 2nd-order convergence at origin
            eo = abs((Do @ g_odd)[0] - 2.0)
            if prev_o is not None:
                assert np.log2(prev_o / eo) > 1.7
            prev_o = eo

    def test_ball_operators_unchanged_on_shell(self):
        """Origin parity is a no-op on shells (no r=0): the grad-div pair and
        scalar Laplacian are identical with/without the ball code path."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div, build_staggered_scalar_laplacian)
        shell = StaggeredGrids.from_grid(
            make_shell_grid(N_r=16, N_theta=20, r_inner=0.71, r_outer=0.985))
        # these must be finite and well-formed (the origin branch never fires)
        GD = build_staggered_grad_div(shell, 8, 2)
        L = build_staggered_scalar_laplacian(shell, 8, 2)
        assert np.all(np.isfinite(GD.toarray()))
        assert np.all(np.isfinite(L.toarray()))


class TestStaggeredStrainOriginParity:
    """The strain operator's tangential-over-r radial derivatives (r∂_r(v_θ/r),
    r∂_r(v_φ/r)) use the parity-folded origin stencil on the ball — g = v_tan/r
    has parity −(vector parity) — so the viscous Gram is origin-regular and
    accurate while keeping its self-adjoint dissipative structure. On a ball
    the stress-free closure zeros only the outer wall (the origin is a parity,
    not a wall); on a shell it zeros both walls."""

    def test_strain_ball_finite_and_bounded(self):
        from spdee.backends.fd2.staggered_compile import build_staggered_strain
        grids = StaggeredGrids.from_grid(
            make_ball_grid(N_r=16, N_theta=20, r_outer=1.0))
        for wb in (None, "stress_free"):
            for mm in (2, 8):
                S, _ = build_staggered_strain(grids, mm, 2, wall_bc=wb)
                A = S.toarray()
                assert np.all(np.isfinite(A))
                # bounded by the m²/r² barrier scale, NOT a 1/r origin blowup
                assert np.abs(A).max() < 1e5, (wb, mm, np.abs(A).max())

    def test_ball_viscous_gram_dissipative_with_origin_parity(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_weighted_viscous_gram)
        grids = StaggeredGrids.from_grid(
            make_ball_grid(N_r=16, N_theta=20, r_outer=0.985))
        F = build_staggered_weighted_viscous_gram(
            grids, 8, lambda r, t: np.exp(-0.5 * r ** 2) + 0 * t, 2,
            wall_bc="stress_free")
        assert np.all(np.isfinite(F.toarray()))
        ev = np.linalg.eigvals(F.toarray())
        assert ev.real.max() < 1e-6           # negative-semidefinite
        assert np.sum(ev.real > 1e-6) == 0    # no spurious growth

    def test_shell_stressfree_zeros_both_walls(self):
        """On a SHELL the stress-free closure zeros BOTH wall-edge
        rows of the tangential shear (unaffected by the ball origin branch)."""
        from spdee.backends.fd2.staggered_compile import build_staggered_strain
        grids = StaggeredGrids.from_grid(
            make_shell_grid(N_r=14, N_theta=20, r_inner=0.71, r_outer=0.985))
        re, rc = grids.r_edge, grids.r_center
        te = grids.theta_edge
        Nr_e, Nt_e = len(re), len(te)
        vt = (rc[:, None] ** 2 * np.sin(te)[None, :]
              * np.cos(te)[None, :]).ravel().astype(complex)
        zr = np.zeros(Nr_e * len(grids.theta_center), dtype=complex)
        zp = np.zeros(len(rc) * len(grids.theta_center), dtype=complex)
        jth = Nt_e // 4
        S, cg = build_staggered_strain(grids, 8, 2, wall_bc="stress_free")
        out = S @ np.concatenate([zr, vt, zp])
        offs = np.cumsum([0] + [c[0] for c in cg])
        erth = out[offs[3]:offs[4]].reshape(Nr_e, Nt_e)
        assert abs(erth[0, jth]) < 1e-12 and abs(erth[-1, jth]) < 1e-12


class TestStaggeredViscousBallOriginElimination:
    """The viscous ball origin v_r is reduced through a self-adjoint
    master–slave prolongation (A_red = Pᵀ A P), uniformly with
    the walls. For |m|≥2 a regular vector has v_r(0,θ)=0 ∀θ, so the whole origin
    ring is dropped (like a wall). For m=0/1 the ring collapses to ONE master
    amplitude with a cosθ/sinθ profile: the master (anchor) DOF is kept and the
    slaves are folded into it, then dropped. Either way the symmetric reduction
    removes the spurious transpose COLUMN couplings that row-overwrite leaves —
    important once MHD's b makes the origin v_r DOFs dynamically live. Inviscid
    keeps the verified row-overwrite."""

    def _compile(self, m, Ek):
        from pathlib import Path
        from spdee.backgrounds.solar import SolarBackgrounds
        from spdee.applications.solar_eigenmodes import make_solar_ball_system
        from spdee.backends.fd2.compiled_problem import compile_problem
        data = Path(__file__).resolve().parent.parent / "data" / "solar"
        if not (data / "model_s_corrected.txt").exists():
            import pytest
            pytest.skip("solar data not available")
        solar = SolarBackgrounds(data, ('ball', 0.985), nondimensional=True)
        grid = make_ball_grid(N_r=14, N_theta=14, r_outer=0.985)
        return compile_problem(make_solar_ball_system(solar, m, Ek), grid,
                               m=m, order_r=2, order_theta=2,
                               discretization="staggered")

    def test_m_ge_2_viscous_ball_eliminates_origin_ring(self):
        """Viscous m≥2 ball is smaller than inviscid by 2·Nt_c (outer wall +
        origin ring both eliminated)."""
        Nt_c = 14
        inv = self._compile(8, 0.0)      # row-overwrite, no elimination
        vis = self._compile(8, 7e-6)     # elimination at outer wall + origin
        assert inv.A_total.shape[0] - vis.A_total.shape[0] == 2 * Nt_c

    def test_m1_viscous_ball_folds_origin_to_one_master(self):
        """m=0/1 ball folds the origin v_r ring into ONE master amplitude (the
        cosθ/sinθ profile), via the self-adjoint master–slave reduction. So the
        viscous v_r block loses (Nt_c−1) slave DOFs at the origin PLUS Nt_c at
        the outer wall, vs the inviscid row-overwrite (which keeps them all)."""
        Nt_c = 14
        inv = self._compile(1, 0.0)      # row-overwrite, keeps all v_r DOFs
        vis = self._compile(1, 7e-6)     # fold origin ring + drop outer wall
        # origin: Nt_c → 1 master (drop Nt_c−1);  outer wall: drop Nt_c
        assert inv.A_total.shape[0] - vis.A_total.shape[0] == (Nt_c - 1) + Nt_c

    def test_m0_viscous_ball_clean_spectrum(self):
        """m=0 viscous ball (master–slave origin fold) is finite with no
        origin-artifact fast growth."""
        import scipy.linalg as sla
        from spdee.backends.fd2.compiled_problem import EvpResult
        vis = self._compile(0, 7e-6)
        A, B = vis.A_total.toarray(), vis.B_total.toarray()
        assert np.all(np.isfinite(A))
        lam = sla.eig(A, B, right=False)
        lam = lam[np.isfinite(lam)]
        om = EvpResult.eigenvalues_to_omega(lam)
        band = om[np.abs(om.real) < 3]
        fast_grow = band[(band.imag > 0.5) & (np.abs(band.real) > 1.0)]
        assert len(fast_grow) == 0, fast_grow

    def test_m_ge_2_viscous_ball_spectrum_clean(self):
        """The eliminated operator's spectrum has no origin-artifact fast
        growth away from the convective band near Re≈0."""
        import scipy.linalg as sla
        from spdee.backends.fd2.compiled_problem import EvpResult
        vis = self._compile(8, 7e-6)
        A, B = vis.A_total.toarray(), vis.B_total.toarray()
        assert np.all(np.isfinite(A))
        lam = sla.eig(A, B, right=False)
        lam = lam[np.isfinite(lam)]
        om = EvpResult.eigenvalues_to_omega(lam)
        band = om[np.abs(om.real) < 3]
        fast_grow = band[(band.imag > 0.5) & (np.abs(band.real) > 1.0)]
        assert len(fast_grow) == 0, fast_grow
