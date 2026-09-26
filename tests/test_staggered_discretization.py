"""Tests for the STAGGERED discretization option (Arakawa-C path).

Covers: the discretization-choice enum/coercion/stubs; the staggered
sub-grid construction; the velocity-pressure assembler recovering the
analytic inertial-wave spectrum; and that selecting COLLOCATED (default)
is unaffected.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sympy as smp
import pytest

from spdee import (
    Background, Boundary, BoundaryCondition, Cross, Div, Dot, Equation,
    EquationSystem, Grad, Unknown, r_hat, theta,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.discretization import (
    Discretization,
    DEFAULT_DISCRETIZATION,
    require_implemented,
    DiscretizationNotImplementedError,
)
from spdee.backends.fd2.staggered_compile import (
    StaggeredGrids,
    compile_fd2_staggered,
)


def inertial_pencil(grid, m, order=2):
    """(A, B, layout) of the inviscid rotating pair 2 Cross(Omega, v) -
    Grad(p) = lambda v with Div(v) = 0 and impenetrable walls, compiled on
    the C-grid: the saddle-point pencil these tests read."""
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Om = Background.Vector("Omega", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                           v_phi=0)
    walls = ((Boundary.R_OUTER,) if grid.r[0] == 0.0
             else (Boundary.R_INNER, Boundary.R_OUTER))
    system = EquationSystem(
        [Equation(2 * Cross(Om, v) - Grad(p), name="momentum"),
         Equation(Div(v), name="continuity")],
        [BoundaryCondition(w, Dot(r_hat, v)) for w in walls])
    cp = compile_fd2_staggered(system, grid, m=m, order_r=order, order_theta=order)
    return cp.A_total.tocsr(), cp.B_total.tocsr(), cp.layout


class TestDiscretizationOption:
    def test_default_is_collocated(self):
        assert DEFAULT_DISCRETIZATION is Discretization.COLLOCATED

    def test_coercion_aliases(self):
        assert Discretization.coerce("collocated") is Discretization.COLLOCATED
        assert Discretization.coerce("colocated") is Discretization.COLLOCATED
        assert Discretization.coerce("cgrid") is Discretization.STAGGERED
        assert Discretization.coerce("solar_eigenproblem") is Discretization.STAGGERED
        assert Discretization.coerce("DEDALUS") is Discretization.DEDALUS
        assert Discretization.coerce(Discretization.SH_FD) is Discretization.SH_FD

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            Discretization.coerce("nonsense")

    def test_implemented_flags(self):
        assert Discretization.COLLOCATED.is_implemented
        assert Discretization.STAGGERED.is_implemented
        assert not Discretization.SH_FD.is_implemented
        assert not Discretization.DEDALUS.is_implemented

    def test_stub_methods_raise_clear_error(self):
        for stub in (Discretization.SH_FD, Discretization.DEDALUS):
            with pytest.raises(DiscretizationNotImplementedError):
                require_implemented(stub)

    def test_implemented_methods_pass(self):
        require_implemented(Discretization.COLLOCATED)
        require_implemented(Discretization.STAGGERED)


class TestStaggeredGrids:
    def test_subgrid_sizes_and_poles(self):
        g = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        assert len(sg.r_center) == len(sg.r_edge) - 1
        # θ-edges include both poles
        assert np.isclose(sg.theta_edge[0], 0.0)
        assert np.isclose(sg.theta_edge[-1], np.pi)
        # sizes consistent
        assert sg.n_vr == len(sg.r_edge) * len(sg.theta_center)
        assert sg.n_vt == len(sg.r_center) * len(sg.theta_edge)
        assert sg.n_center == len(sg.r_center) * len(sg.theta_center)


def _toroidal_rmode(A, B, layout, m, k=40):
    """The sectoral r-mode λ = −2i/(m+1) (A = −A_phys, ω = −iλ) identified by
    its eigenvector, toroidal to 10 percent in v_r, among the k eigenvalues
    nearest the target; the inviscid band is dense, so proximity alone
    identifies nothing."""
    fs = {f.name: f for f in layout.fields}
    target = -2j / (m + 1)
    shift = target + 1e-4j
    LU = spla.splu((A - shift * B).tocsc())
    Lin = spla.LinearOperator(A.shape, matvec=lambda x: LU.solve(B @ x), dtype=complex)
    mu, V = spla.eigs(Lin, k=k, which="LM", maxiter=8000, tol=1e-12)
    lam = 1.0 / mu + shift
    fr = (np.linalg.norm(V[fs["v_r"].slice], axis=0)
          / np.linalg.norm(V[:fs["p"].offset], axis=0))
    keep = (fr < 0.1) & np.isfinite(lam)
    assert keep.any(), "no toroidal eigenvector near the r-mode"
    j = np.argmin(np.abs(lam - target) + 1e9 * (~keep))
    return lam[j], target


class TestStaggeredInertialAssembly:
    """The staggered assembler recovers the sectoral r-modes of the rotating
    shell, exactly neutral, through the production module."""

    def _solve(self, N, m=2):
        grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
        return inertial_pencil(grid, m)

    def test_layout_fields(self):
        _, _, layout = self._solve(24)
        names = [f.name for f in layout.fields]
        assert names == ["v_r", "v_t", "v_p", "p"]
        locs = {f.name: f.location for f in layout.fields}
        assert locs["v_r"] == "r_edge"
        assert locs["v_t"] == "theta_edge"
        assert locs["v_p"] == "center"
        assert locs["p"] == "center"

    @pytest.mark.parametrize("m", [1, 2, 3])
    def test_rmode_neutral_and_accurate(self, m):
        lam, target = _toroidal_rmode(*self._solve(24, m=m), m)
        assert abs(lam.real) < 1e-9, lam
        assert abs(lam - target) / abs(target) < 1e-3, (lam, target)

    def test_rmode_converges(self):
        errs = []
        for N in (24, 48):
            lam, target = _toroidal_rmode(*self._solve(N, m=2), 2)
            errs.append(abs(lam - target))
        assert errs[1] < 0.4 * errs[0], errs


class TestStaggeredBallGeometry:
    """The staggered assembler on BALL geometry (r_inner=0).

    The origin is the inner r-edge: v_r lives there with zero volume
    weight, so it enters no norm-adjoint operator; the assembler installs
    the Cartesian regularity rows of ``_apply_ball_origin_vr`` (Dirichlet
    zero for |m| ≥ 2, the cosθ / sinθ single-DOF anchor for m = 0 / |m| = 1)
    and the origin-parity closures of the radial stencils. The full audit
    and the analytic ball benchmarks are in docs/ball_staggered.md and
    tests/test_ball_staggered.py.
    """

    def _ball_err(self, N, m=2):
        from spdee.backends.fd2.grid import make_ball_grid
        g = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        A, B, layout = inertial_pencil(g, m)
        lam, target = _toroidal_rmode(A, B, layout, m)
        return abs(lam - target) / abs(target), lam.real

    def test_ball_assembles_finite(self):
        from spdee.backends.fd2.grid import make_ball_grid
        g = make_ball_grid(N_r=24, N_theta=24, r_outer=1.0)
        assert g.r[0] == 0.0  # origin is a grid point
        A, B, _ = inertial_pencil(g, 2)
        assert np.all(np.isfinite(A.data)), "1/r at origin produced non-finite"
        assert np.all(np.isfinite(B.data))

    def test_ball_recovers_and_converges(self):
        e_coarse, re_c = self._ball_err(24)
        e_fine, re_f = self._ball_err(48)
        assert e_coarse < 1e-3, f"ball r-mode too inaccurate at N=24: {e_coarse:.2e}"
        assert e_fine < 0.4 * e_coarse, f"ball did not converge: {e_coarse:.2e}->{e_fine:.2e}"
        assert abs(re_f) < 1e-9, f"ball mode spuriously damped: Re={re_f:.2e}"

    def test_origin_regularity_rows_installed(self):
        """The ball origin (ir=0) gets explicit regularity rows on v_r:
        Dirichlet-zero for |m|≥2; an angular-profile anchor for m∈{0,1}.
        Verify the row STRUCTURE matches the intended treatment."""
        from spdee.backends.fd2.grid import make_ball_grid
        N = 16
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        sg = StaggeredGrids.from_grid(grid)
        Nt_c = len(sg.theta_center)

        # m≥2: every origin v_r row is Dirichlet-zero (single unit entry).
        A2, B2, _ = inertial_pencil(grid, 2)
        A2 = A2.tolil()
        B2 = B2.tolil()
        for it in range(Nt_c):
            row = it  # ir=0 block, v_r starts at offset 0
            assert A2.rows[row] == [row]
            assert np.isclose(A2.data[row][0], 1.0)
            assert B2.rows[row] == [], "origin constraint must carry no mass"

        # m=1: j>0 origin rows are 2-entry anchor constraints (self + j=0);
        # the j=0 row is left dynamical (more than the trivial unit row).
        A1, B1, _ = inertial_pencil(grid, 1)
        A1 = A1.tolil()
        for it in range(1, Nt_c):
            row = it
            assert len(A1.rows[row]) == 2, (
                f"m=1 origin anchor row should have 2 entries, got "
                f"{A1.rows[row]}")
            assert 0 in A1.rows[row], "anchor must reference j=0 origin node"


class TestStaggeredCurl:
    """The staggered curl (primal v→dual, and dual→v) is verified against
    the analytic curl. Primal: 2nd order all three components. Dual fed
    analytic dual fields: 2nd order. (The composed curl∘curl vector
    Laplacian has a known pole-closure gap in θ/φ — see the
    build_staggered_curl docstring; tracked separately.)"""

    def _curl_truth(self, M):
        import sympy as smp
        rr, th = smp.symbols("r theta", positive=True)
        vr = rr**2 * smp.sin(th)
        vth = rr * smp.sin(th) * smp.cos(th)
        vph = rr * smp.sin(th)
        cr = (1 / (rr * smp.sin(th))) * (smp.diff(smp.sin(th) * vph, th) - smp.I * M * vth)
        cth = smp.I * M * vr / (rr * smp.sin(th)) - (1 / rr) * smp.diff(rr * vph, rr)
        cph = (1 / rr) * (smp.diff(rr * vth, rr) - smp.diff(vr, th))
        L = lambda e: smp.lambdify((rr, th), smp.simplify(e), "numpy")
        return L(cr), L(cth), L(cph)

    @staticmethod
    def _ev(f, rg, tg):
        return np.broadcast_to(f(rg[:, None], tg[None, :]),
                               (len(rg), len(tg))).astype(complex)

    def test_primal_curl_second_order(self):
        from spdee.backends.fd2.staggered_compile import build_staggered_curl
        M = 2
        fcr, fcth, fcph = self._curl_truth(M)
        errs_r = []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            B = build_staggered_curl(sg, M, order=2, dual=False)
            re, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
            vr = (re[:, None]**2 * np.sin(tc)[None, :]).ravel().astype(complex)
            vth = (rc[:, None] * np.sin(te)[None, :] * np.cos(te)[None, :]).ravel().astype(complex)
            vph = (rc[:, None] * np.sin(tc)[None, :]).ravel().astype(complex)
            cr_n = (B["cr_aphi"] @ vph + B["cr_ath"] @ vth).reshape(len(rc), len(te))
            cph_n = (B["cphi_ath"] @ vth + B["cphi_ar"] @ vr).reshape(len(re), len(te))
            er = np.abs(cr_n - self._ev(fcr, rc, te))[2:-2, 2:-2].max()
            ep = np.abs(cph_n - self._ev(fcph, re, te))[2:-2, 2:-2].max()
            errs_r.append(max(er, ep))
        rate = np.log2(errs_r[0] / errs_r[-1]) / 2
        assert rate > 1.7, f"primal curl not 2nd order: rate {rate:.1f}, {errs_r}"

    def test_primal_curl_second_order_at_poles(self):
        """Pole-INCLUSIVE curl convergence — guards the θ=0-vs-θ=π symmetry
        of the 1/(r sinθ) pole closure (_inv_sin_with_pole_closure). The
        L'Hôpital N'(θ) row must use a one-sided real-node derivative
        oriented away from each pole, identical in structure at both; a
        centered parity-extended stencil straddles the pole at θ=0 and gives
        a sign-flipped one-sided derivative there (θ=π converges, θ=0 stays
        flat at ~5.6 error). This test windows only the radial walls and
        keeps ALL θ (both pole columns), and additionally asserts the two
        pole columns have matching error.

        Uses a POLE-REGULAR m=2 field (v_r ~ sin^2θ, v_θ,v_φ ~ sin^1θ) so the
        analytic curl is finite and smooth at the poles; a non-pole-regular
        field would inject a spurious pole error of its own (the curl operator
        correctly enforces the m=2 spin parity, which such a field violates).
        """
        from spdee.backends.fd2.staggered_compile import build_staggered_curl
        import sympy as smp
        M = 2
        rr, th = smp.symbols("r theta", positive=True)
        vr_s = rr**2 * smp.sin(th)**2
        vth_s = rr * smp.sin(th) * smp.cos(th)
        vph_s = rr * smp.sin(th)
        cr_s = (1 / (rr * smp.sin(th))) * (
            smp.diff(smp.sin(th) * vph_s, th) - smp.I * M * vth_s)
        cph_s = (1 / rr) * (smp.diff(rr * vth_s, rr) - smp.diff(vr_s, th))
        fcr = smp.lambdify((rr, th), smp.simplify(cr_s), "numpy")
        fcph = smp.lambdify((rr, th), smp.simplify(cph_s), "numpy")

        errs, asym = [], []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            B = build_staggered_curl(sg, M, order=2, dual=False)
            re, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
            vr = (re[:, None]**2 * np.sin(tc)[None, :]**2).ravel().astype(complex)
            vth = (rc[:, None] * np.sin(te)[None, :] * np.cos(te)[None, :]).ravel().astype(complex)
            vph = (rc[:, None] * np.sin(tc)[None, :]).ravel().astype(complex)
            cr_n = (B["cr_aphi"] @ vph + B["cr_ath"] @ vth).reshape(len(rc), len(te))
            cph_n = (B["cphi_ath"] @ vth + B["cphi_ar"] @ vr).reshape(len(re), len(te))
            er_grid = np.abs(cr_n - self._ev(fcr, rc, te))
            ep_grid = np.abs(cph_n - self._ev(fcph, re, te))
            # window only the radial walls; keep ALL theta incl. both poles
            er = er_grid[2:-2, :].max()
            ep = ep_grid[2:-2, :].max()
            errs.append(max(er, ep))
            # pole-column symmetry of curl_r: θ=0 vs θ=π error must match
            e0 = er_grid[2:-2, 0].max()
            epi = er_grid[2:-2, -1].max()
            asym.append(abs(e0 - epi) / (max(e0, epi) + 1e-300))

        rate = np.log2(errs[0] / errs[-1]) / 2
        assert rate > 1.7, (
            f"pole-inclusive curl not 2nd order: rate {rate:.2f}, {errs}")
        assert max(asym) < 1e-6, (
            f"θ=0 vs θ=π pole error asymmetric (the fixed bug): {asym}")


class TestStaggeredVectorLaplacian:
    """The composed vector Laplacian ∇²v = ∇(∇·v) − ∇×(∇×v), built from the
    W-adjoint curl-curl composition. Both pieces are norm-adjoint
    compositions, so ∇²v is self-adjoint and negative-semidefinite under the
    velocity norm W_v — the property that lets _eliminate_wall_vr preserve
    the structure on the constrained (v_r=0 wall) space. (The interior is
    2nd-order incl. both poles; the tangential radial-WALL closure for
    curl_curl is not covered here.)"""

    def test_curl_curl_is_self_adjoint(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_curl_curl, build_staggered_grad_div_pair)
        g = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        for M in (0, 1, 2):
            CC = build_staggered_curl_curl(sg, M, order=2)
            _, _, W_v, _ = build_staggered_grad_div_pair(sg, M, order=2)
            rng = np.random.default_rng(M)
            nv = CC.shape[0]
            u = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            w = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            lhs = np.vdot(u, W_v * (CC @ w))
            rhs = np.vdot(CC @ u, W_v * w)
            assert abs(lhs - rhs) / (abs(lhs) + 1e-30) < 1e-10, (
                f"curl_curl not W-self-adjoint at m={M}")
            # positive-semidefinite (curl_curl is +∇×∇×, dissipative sign is
            # carried by the MINUS in ∇²v)
            quad = np.vdot(u, W_v * (CC @ u)).real
            assert quad >= -1e-8, f"curl_curl not PSD at m={M}: {quad:.2e}"

    def test_vector_laplacian_self_adjoint_and_dissipative(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_vector_laplacian, build_staggered_grad_div_pair,
            _eliminate_wall_vr)
        g = make_shell_grid(N_r=20, N_theta=20, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        for M in (1, 2):
            L = build_staggered_vector_laplacian(sg, M, order=2)
            _, _, W_v, _ = build_staggered_grad_div_pair(sg, M, order=2)
            # self-adjoint under W_v
            rng = np.random.default_rng(10 + M)
            nv = L.shape[0]
            u = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            w = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            assert abs(np.vdot(u, W_v * (L @ w)) - np.vdot(L @ u, W_v * w)) \
                / (abs(np.vdot(u, W_v * (L @ w))) + 1e-30) < 1e-10
            # dissipative: ⟨u, L u⟩_W ≤ 0 for the viscous operator
            assert np.vdot(u, W_v * (L @ u)).real <= 1e-8
            # wall elimination preserves self-adjointness on the reduced space
            Lw, keep = _eliminate_wall_vr(L, sg)
            Wvw = W_v[keep]
            nv2 = Lw.shape[0]
            u2 = rng.standard_normal(nv2) + 1j * rng.standard_normal(nv2)
            w2 = rng.standard_normal(nv2) + 1j * rng.standard_normal(nv2)
            assert abs(np.vdot(u2, Wvw * (Lw @ w2))
                       - np.vdot(Lw @ u2, Wvw * w2)) \
                / (abs(np.vdot(u2, Wvw * (Lw @ w2))) + 1e-30) < 1e-10


class TestStaggeredScalarLaplacian:
    """The staggered scalar Laplacian (flux-form ∇·∇) is correct: it
    annihilates harmonic functions r^ℓ Y_ℓm at 2nd order and gives
    ∇²(r²)=6 at m=0. Needed for thermal diffusion κ∇²s₁."""

    def test_annihilates_harmonic(self):
        from scipy.special import lpmv
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian,
        )
        errs = []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            L = build_staggered_scalar_laplacian(sg, m=2)
            rc, tc = sg.r_center, sg.theta_center
            Nt_c = len(tc)
            f = (rc[:, None]**2) * lpmv(2, 2, np.cos(tc))[None, :]
            res = (L @ f.ravel()).reshape(len(rc), Nt_c)
            scale = np.max(np.abs(f)) / rc.mean()**2
            errs.append(np.max(np.abs(res[2:-2, 2:-2])) / scale)
        assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
        assert errs[-1] < 2e-3

    def test_radial_high_order(self):
        """The composed Div·Grad Laplacian preserves stencil order in the
        radial direction: order=2 → 2nd order, order=4 → 4th order. This
        confirms composition does NOT cap accuracy at 2nd order."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian,
        )
        rates_by_order = {}
        for order in (2, 4):
            errs = []
            for N in (20, 40, 80):
                g = make_shell_grid(N_r=N, N_theta=12, r_inner=0.3, r_outer=1.0)
                sg = StaggeredGrids.from_grid(g)
                L = build_staggered_scalar_laplacian(sg, m=0, order=order)
                rc = sg.r_center
                Nt = len(sg.theta_center)
                f = (rc[:, None]**4 * np.ones(Nt)[None, :]).ravel()
                res = (L @ f).reshape(len(rc), Nt)
                exact = 20 * rc**2  # ∇²(r⁴) = 20 r²
                errs.append(np.max(np.abs(
                    res[order:-order, Nt // 2] - exact[order:-order])))
            rate = np.log2(errs[0] / errs[-1]) / 2  # per doubling
            rates_by_order[order] = rate
        assert rates_by_order[2] > 1.7, f"order-2 rate {rates_by_order[2]:.1f}"
        assert rates_by_order[4] > 3.3, (
            f"order-4 did not achieve high order: rate {rates_by_order[4]:.1f}")

    def test_interior_theta_high_order(self):
        """High order in θ on the interior rows via the parity-extended θ
        stencil: ∇²(r²Y₂₂) → 0 converges at rate = order for orders 4 and 6
        away from the walls and poles. The divergence is the negative
        transpose of the gradient, so the operator is self-adjoint with a
        real, non-positive spectrum at every order; the pole- and
        wall-adjacent rows are then weakly (Galerkin) consistent only, and
        the eigenvalues converge at 2nd order (see docs/ball_staggered.md).
        """
        from scipy.special import lpmv
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian, build_staggered_grad_div_pair,
        )
        for order, min_rate in ((4, 3.5), (6, 5.0)):
            errs = []
            for N in (24, 48, 96):
                g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
                sg = StaggeredGrids.from_grid(g)
                L = build_staggered_scalar_laplacian(sg, m=2, order=order)
                rc, tc = sg.r_center, sg.theta_center
                Nt = len(tc)
                f = (rc[:, None]**2) * lpmv(2, 2, np.cos(tc))[None, :]
                scale = np.max(np.abs(f)) / rc.mean()**2
                res = np.abs((L @ f.ravel()).reshape(len(rc), Nt)) / scale
                errs.append(res[order:-order, order:-order].max())
            rate = np.log2(errs[0] / errs[-1]) / 2
            assert rate > min_rate, (
                f"order={order} not high order in the interior: "
                f"rate {rate:.1f}, errs {errs}")
        g = make_shell_grid(N_r=20, N_theta=20, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        _, _, _, W_c = build_staggered_grad_div_pair(sg, m=2, order=2)
        for order in (4, 6):
            L = build_staggered_scalar_laplacian(sg, m=2, order=order)
            WL = (sp.diags(W_c) @ L).toarray()
            assert np.abs(WL - WL.conj().T).max() < 1e-12 * np.abs(WL).max()
            ev = np.linalg.eigvals(L.toarray())
            assert np.abs(ev.imag).max() < 1e-8 * np.abs(ev).max()
            assert ev.real.max() < 0

    def test_laplacian_of_r_squared(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian,
        )
        g = make_shell_grid(N_r=64, N_theta=64, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        L = build_staggered_scalar_laplacian(sg, m=0)
        rc, tc = sg.r_center, sg.theta_center
        Nt_c = len(tc)
        f = (rc[:, None]**2 * np.ones(Nt_c)[None, :]).ravel()
        res = (L @ f).reshape(len(rc), Nt_c)
        # ∇²(r²) = 6 in the interior
        assert abs(res[len(rc)//2, Nt_c//2] - 6.0) < 1e-2

    def test_dirichlet_wall_at_order_4(self):
        """The Dirichlet closure at order 4 reflects the source nodes with
        the odd parity about the wall, so the composed operator stays
        self-adjoint and consistent through the wall cells: the radial
        boundary-value problem (L - 1) p = f converges at order 2 (the
        mirror ghost's own order) with errors below those of order 2,
        while a one-sided wall row alone would leave the transposed
        divergence inconsistent there (order 1)."""
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian, build_staggered_grad_div_pair,
        )
        r_in, r_out = 0.5, 1.0
        rs = smp.symbols("r", positive=True)
        p = smp.sin(smp.pi * (rs - r_in) / (r_out - r_in)) * (1 + rs**3)
        lap = smp.diff(rs**2 * smp.diff(p, rs), rs) / rs**2
        pf, lf = smp.lambdify(rs, p), smp.lambdify(rs, lap)
        errs = {}
        for order in (2, 4):
            for N in (16, 32, 64):
                g = make_shell_grid(N_r=N + 1, N_theta=4, r_inner=r_in, r_outer=r_out)
                sg = StaggeredGrids.from_grid(g)
                L = build_staggered_scalar_laplacian(sg, m=0, order=order, wall_bc="dirichlet")
                _, _, _, W_c = build_staggered_grad_div_pair(sg, m=0, order=order)
                WL = (sp.diags(W_c) @ L).toarray()
                assert np.abs(WL - WL.conj().T).max() < 1e-12 * np.abs(WL).max()
                rc, Nt = sg.r_center, len(sg.theta_center)
                x = spla.spsolve((L - sp.eye(L.shape[0])).tocsc(),
                                 np.repeat(lf(rc) - pf(rc), Nt))
                errs.setdefault(order, []).append(
                    abs(x - np.repeat(pf(rc), Nt)).max() / abs(pf(rc)).max())
        for order, e in errs.items():
            rates = [np.log2(e[i] / e[i + 1]) for i in range(2)]
            assert min(rates) > 1.9, (order, e, rates)
        assert errs[4][-1] < errs[2][-1] / 4


class TestOrderThreadedThroughout:
    """The order/parity machinery is threaded through ALL staggered
    operators via the shared 1D stencil primitives and the shared
    gradient builder: the inertial assembler, grad-div, and scalar
    Laplacian all honor `order` and preserve their structural properties
    (skew/self-adjointness) at high order."""

    def test_inertial_assembler_honors_order_and_stays_undamped(self):
        target = 2j / 3  # m=2 analytic inertial mode, coeff-2 (2·Cross(Ω,v))
        for order in (2, 4):
            g = make_shell_grid(N_r=40, N_theta=40, r_inner=0.3, r_outer=1.0)
            A, B, _ = inertial_pencil(g, 2, order=order)
            shift = target + 1e-4j
            LU = spla.splu((A - shift * B).tocsc())
            Lin = spla.LinearOperator(
                A.shape, matvec=lambda x: LU.solve(B @ x), dtype=complex)
            mu = spla.eigs(Lin, k=4, which="LM", return_eigenvectors=False,
                           maxiter=20000, tol=1e-12)
            lam = 1.0 / mu + shift
            lam = lam[np.isfinite(lam)]
            best = lam[np.argmin(np.abs(lam - target))]
            assert abs(best.imag - target.imag) / target.imag < 1e-2
            assert abs(best.real) < 1e-6, (
                f"order={order} mode spuriously damped: Re={best.real:.1e}")

    def test_grad_div_self_adjoint_at_high_order(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div, build_staggered_grad_div_pair,
        )
        g = make_shell_grid(N_r=32, N_theta=32, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        for order in (2, 4):
            GD = build_staggered_grad_div(sg, m=2, order=order)
            _, _, W_v, _ = build_staggered_grad_div_pair(sg, m=2, order=order)
            rng = np.random.default_rng(0)
            nv = GD.shape[0]
            u = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            w = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
            lhs = np.vdot(u, W_v * (GD @ w))
            rhs = np.vdot(GD @ u, W_v * w)
            assert abs(lhs - rhs) / abs(lhs) < 1e-12
            assert np.vdot(u, W_v * (GD @ u)).real <= 0

    def test_coriolis_interp_high_order_and_adjoint_paired(self):
        """The Coriolis cross-component interpolation can be raised to high
        order, AND the adjoint pairing (partner = weighted transpose) is
        preserved exactly at every order — so the skew-adjoint Coriolis
        structure (undamped spectrum) survives high-order interpolation."""
        from spdee.backends.fd2.staggered_compile import (
            _stag_interp_1d_lagrange, _avg_1d,
        )
        # (a) the interpolation primitive converges at rate=order
        rates = {}
        for order in (2, 4, 6):
            errs = []
            for N in (24, 48, 96):
                r_edge = np.linspace(0.3, 1.0, N)
                r_center = 0.5 * (r_edge[:-1] + r_edge[1:])
                P = _stag_interp_1d_lagrange(r_center, r_edge, order).toarray()
                f = lambda r: np.exp(r) * np.cos(3 * r)
                err = np.max(np.abs((P @ f(r_center) - f(r_edge))[order:-order]))
                errs.append(err)
            rates[order] = np.log2(errs[0] / errs[-1]) / 2
        assert rates[2] > 1.7 and rates[4] > 3.5 and rates[6] > 5.0, rates

        # (b) adjoint pairing holds to machine precision at all orders
        g = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        re, rc, tc = sg.r_edge, sg.r_center, sg.theta_center
        hr = re[1] - re[0]
        ht = tc[1] - tc[0]

        def vol(rv, tv):
            return ((rv**2)[:, None] * np.sin(tv)[None, :]).ravel() * hr * ht
        W_vr, W_vp = vol(re, tc), vol(rc, tc)
        Nt_c = len(tc)
        for order in (2, 4, 6):
            if order <= 2:
                P = sp.kron(_avg_1d(len(re), len(rc)), sp.eye(Nt_c),
                            format="csr")
            else:
                P = sp.kron(_stag_interp_1d_lagrange(rc, re, order),
                            sp.eye(Nt_c), format="csr")
            Padj = sp.diags(1.0 / W_vp) @ P.conj().T @ sp.diags(W_vr)
            rng = np.random.default_rng(0)
            u = rng.standard_normal(P.shape[1]) + 1j * rng.standard_normal(P.shape[1])
            w = rng.standard_normal(P.shape[0]) + 1j * rng.standard_normal(P.shape[0])
            lhs = np.vdot(P @ u, W_vr * w)
            rhs = np.vdot(u, W_vp * (Padj @ w))
            assert abs(lhs - rhs) / abs(lhs) < 1e-12, (
                f"order={order} adjoint pairing broken")


class TestComposedLaplacianSelfAdjoint:
    """The composed (Div·Grad) scalar Laplacian is discretely SELF-ADJOINT
    under the volume norm, so -∇² has a real, positive spectrum. A direct
    second-derivative stencil with the advective-form metric terms
    (2/r ∂_r, cotθ/r² ∂_θ) is NOT self-adjoint and throws eigenvalues off
    the real axis — the diffusive analog of the spurious damping the
    staggered scheme was built to avoid. This is why the composition
    (Route A) is the right construction for the viscous operators.
    """

    def _direct_d2_laplacian(self, sg, m):
        r_c, th_c = sg.r_center, sg.theta_center
        Nr, Nt = len(r_c), len(th_c)

        def diagc(rv, tv, fn):
            return sp.diags(fn(rv[:, None], tv[None, :]).ravel().astype(complex))

        def d1(x):
            n = len(x); h = x[1] - x[0]; D = np.zeros((n, n))
            for i in range(1, n - 1):
                D[i, i - 1] = -0.5 / h; D[i, i + 1] = 0.5 / h
            D[0, 0] = -1 / h; D[0, 1] = 1 / h
            D[-1, -1] = 1 / h; D[-1, -2] = -1 / h
            return D

        def d2(x):
            n = len(x); h = x[1] - x[0]; D = np.zeros((n, n))
            for i in range(1, n - 1):
                D[i, i - 1] = 1 / h**2; D[i, i] = -2 / h**2; D[i, i + 1] = 1 / h**2
            D[0, 0] = 1 / h**2; D[0, 1] = -2 / h**2; D[0, 2] = 1 / h**2
            D[-1, -1] = 1 / h**2; D[-1, -2] = -2 / h**2; D[-1, -3] = 1 / h**2
            return D

        Dr = sp.kron(sp.csr_matrix(d1(r_c)), sp.eye(Nt))
        Drr = sp.kron(sp.csr_matrix(d2(r_c)), sp.eye(Nt))
        Dt = sp.kron(sp.eye(Nr), sp.csr_matrix(d1(th_c)))
        Dtt = sp.kron(sp.eye(Nr), sp.csr_matrix(d2(th_c)))
        invr = diagc(r_c, th_c, lambda r, t: 1 / r + 0 * t)
        invr2 = diagc(r_c, th_c, lambda r, t: 1 / r**2 + 0 * t)
        cot = diagc(r_c, th_c, lambda r, t: np.cos(t) / np.sin(t) + 0 * r)
        mterm = diagc(r_c, th_c, lambda r, t: -m**2 / (r**2 * np.sin(t)**2) + 0j)
        return Drr + 2 * invr @ Dr + invr2 @ Dtt + invr2 @ cot @ Dt + mterm

    def test_composed_real_spectrum_direct_complex(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_scalar_laplacian,
        )
        g = make_shell_grid(N_r=20, N_theta=20, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        L_comp = build_staggered_scalar_laplacian(sg, m=2)
        L_dir = self._direct_d2_laplacian(sg, m=2)
        ev_comp = np.linalg.eigvals((-L_comp).toarray())
        ev_dir = np.linalg.eigvals((-L_dir).toarray())
        # composed: essentially real spectrum (self-adjoint)
        assert np.max(np.abs(ev_comp.imag)) < 1e-8, (
            f"composed -∇² not real: max|Im|={np.max(np.abs(ev_comp.imag)):.1e}")
        # direct: significantly complex spectrum (not self-adjoint)
        assert np.max(np.abs(ev_dir.imag)) > 1.0, (
            "expected direct-d2 -∇² to be non-self-adjoint (complex spectrum)")
        # composed: no negative eigenvalues (positive operator)
        assert np.sum(ev_comp.real < -1e-6) == 0


class TestStaggeredGradDiv:
    """Staggered grad-div ∇(∇·v) = G∘Div is self-adjoint and negative-
    semidefinite under W_v (the conservative-form payoff), and Div is
    physically correct. This is the ⅓∇(∇·v) piece of ∇·D."""

    def test_div_physical(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div_pair,
        )
        errs = []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            _, Div, _, _ = build_staggered_grad_div_pair(sg, m=0)
            Nt_c = len(sg.theta_center)
            Nr_c, Nt_e = len(sg.r_center), len(sg.theta_edge)
            n_vt, n_vp = Nr_c * Nt_e, Nr_c * Nt_c
            vr = (sg.r_edge[:, None] * np.ones(Nt_c)[None, :]).ravel()
            v = np.concatenate([vr, np.zeros(n_vt), np.zeros(n_vp)]).astype(complex)
            divv = (Div @ v).reshape(Nr_c, Nt_c)
            errs.append(abs(divv[Nr_c // 2, Nt_c // 2] - 3.0))  # ∇·(r ê_r)=3
        assert errs[0] > errs[1] > errs[2], f"Div not converging: {errs}"
        assert errs[-1] < 1e-3

    def test_grad_div_physical_interior_correct(self):
        """grad(div v) matches the analytic ∇(∇·v) in the radial INTERIOR
        at 2nd order. (The radial-wall rows have a known consistency
        defect from the norm-adjoint transpose — see
        test_grad_div_boundary_defect_known and the builder docstring.)"""
        import sympy as smp
        rr, th = smp.symbols("r theta", positive=True)
        M = 2
        vr_s = rr**2 * smp.sin(th)
        vth_s = rr * smp.sin(th) * smp.cos(th)
        vph_s = rr * smp.sin(th)
        divV = ((1 / rr**2) * smp.diff(rr**2 * vr_s, rr)
                + (1 / (rr * smp.sin(th))) * smp.diff(smp.sin(th) * vth_s, th)
                + smp.I * M * vph_s / (rr * smp.sin(th)))
        fgdr = smp.lambdify((rr, th), smp.simplify(smp.diff(divV, rr)), "numpy")
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div_pair,
        )
        errs = []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            G, Div, _, _ = build_staggered_grad_div_pair(sg, M, order=2)
            re, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
            vr = (re[:, None]**2 * np.sin(tc)[None, :]).ravel().astype(complex)
            vth = (rc[:, None] * np.sin(te)[None, :] * np.cos(te)[None, :]).ravel().astype(complex)
            vph = (rc[:, None] * np.sin(tc)[None, :]).ravel().astype(complex)
            v = np.concatenate([vr, vth, vph])
            gr = (G @ (Div @ v))[:len(vr)].reshape(len(re), len(tc))
            ex = fgdr(re[:, None], tc[None, :])
            # radial interior, θ interior
            errs.append(np.max(np.abs(gr - ex)[5:-5, 3:-3]))
        assert errs[0] > errs[1] > errs[2], f"interior not converging: {errs}"
        assert errs[-1] < 1e-3

    def test_grad_div_wall_elimination_consistent_and_selfadjoint(self):
        """The radial-wall defect is CURED by eliminating the wall v_r DOFs
        (rows AND columns) — the solar eigenproblem's impenetrability BC. On the reduced
        space, grad-div is both consistent to 2nd order at the wall-adjacent
        rows AND self-adjoint / negative-semidefinite (both to machine
        precision)."""
        import sympy as smp
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div_pair, _eliminate_wall_vr,
        )
        rr, th = smp.symbols("r theta", positive=True)
        M = 2
        # field with v_r vanishing at the walls (impenetrable-admissible)
        vr_s = (rr - smp.Rational(3, 10)) * (1 - rr) * smp.sin(th)
        vth_s = rr * smp.sin(th) * smp.cos(th)
        vph_s = rr * smp.sin(th)
        divV = ((1 / rr**2) * smp.diff(rr**2 * vr_s, rr)
                + (1 / (rr * smp.sin(th))) * smp.diff(smp.sin(th) * vth_s, th)
                + smp.I * M * vph_s / (rr * smp.sin(th)))
        fgdr = smp.lambdify((rr, th), smp.simplify(smp.diff(divV, rr)), "numpy")

        # (a) consistency at wall-adjacent rows (full operator, no-pen field)
        errs = []
        for N in (24, 48, 96):
            g = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
            sg = StaggeredGrids.from_grid(g)
            G, Div, _, _ = build_staggered_grad_div_pair(sg, M, order=2)
            re, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
            Nr_e, Nt_c = len(re), len(tc)
            vr = ((re[:, None] - 0.3) * (1 - re[:, None]) * np.sin(tc)[None, :]).ravel().astype(complex)
            vr2 = vr.reshape(Nr_e, Nt_c); vr2[0, :] = 0; vr2[-1, :] = 0; vr = vr2.ravel()
            vth = (rc[:, None] * np.sin(te)[None, :] * np.cos(te)[None, :]).ravel().astype(complex)
            vph = (rc[:, None] * np.sin(tc)[None, :]).ravel().astype(complex)
            v = np.concatenate([vr, vth, vph])
            gr = (G @ (Div @ v))[:len(vr)].reshape(Nr_e, Nt_c)
            ex = fgdr(re[:, None], tc[None, :])
            err = np.abs(gr - ex)
            errs.append(max(err[1, Nt_c // 2], err[2, Nt_c // 2]))  # wall-adjacent
        assert errs[0] > errs[1] > errs[2], f"wall-adjacent not converging: {errs}"
        assert errs[-1] < 1e-3

        # (b) reduced operator is self-adjoint & negative-semidefinite
        g = make_shell_grid(N_r=32, N_theta=32, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        G, Div, W_v, _ = build_staggered_grad_div_pair(sg, M, order=2)
        GD = (G @ Div).tocsr()
        GD_red, keep = _eliminate_wall_vr(GD, sg)
        W_red = W_v[keep]
        rng = np.random.default_rng(0)
        n = GD_red.shape[0]
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        w = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(u, W_red * (GD_red @ w))
        rhs = np.vdot(GD_red @ u, W_red * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12, "reduced grad-div not self-adjoint"
        assert np.vdot(u, W_red * (GD_red @ u)).real <= 0, "not neg-semidef"

    def test_grad_div_self_adjoint_negsemidef(self):
        from spdee.backends.fd2.staggered_compile import (
            build_staggered_grad_div, build_staggered_grad_div_pair,
        )
        g = make_shell_grid(N_r=32, N_theta=32, r_inner=0.3, r_outer=1.0)
        sg = StaggeredGrids.from_grid(g)
        GD = build_staggered_grad_div(sg, m=2)
        _, _, W_v, _ = build_staggered_grad_div_pair(sg, m=2)
        rng = np.random.default_rng(0)
        nv = GD.shape[0]
        u = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
        w = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
        lhs = np.vdot(u, W_v * (GD @ w))
        rhs = np.vdot(GD @ u, W_v * w)
        assert abs(lhs - rhs) / abs(lhs) < 1e-12, "grad_div not self-adjoint"
        assert np.vdot(u, W_v * (GD @ u)).real <= 0, "grad_div not neg-semidef"


class TestStaggeredHonestScope:
    """The staggered path builds what it can FAITHFULLY build and rejects
    the rest, rather than silently dropping physics.

    With Layers 1–3 + 2b, the staggered path builds the full hydrodynamic
    solar eigenproblem: inviscid compressible (momentum + continuity + energy),
    constant-coefficient viscous (Lamé) + thermal diffusion, the stress-free
    wall closure, and the ρ₀-stratification viscous strain correction. Only
    MHD coupling (induction / magnetic unknown / Curl) is still rejected. See
    tests/test_staggered_compressible.py for the full coverage.
    """

    def test_compiles_viscous_three_field(self):
        """A 3-field system carrying viscous + thermal diffusion (Ek>0) now
        COMPILES (Layers 2/2b/3) rather than raising — the diffusion is built,
        not silently dropped."""
        from spdee.backgrounds.polytrope import PolytropeBackground
        from spdee.applications.solar_eigenmodes import make_polytrope_system

        poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=1e-6)
        sys_ = make_polytrope_system(
            poly, m_val=8, Ek=7e-6, entropy_bc="neumann", geometry="shell")
        grid = make_shell_grid(N_r=12, N_theta=18, r_inner=0.71, r_outer=0.985)
        cm = compile_fd2_staggered(sys_, grid, m=8)
        assert [f.name for f in cm.layout.fields] == \
            ["v_r", "v_t", "v_p", "rho1", "s1"]

    def test_rejects_mhd_three_field(self):
        """MHD coupling (Le>0 → induction equation + magnetic unknown) is
        still rejected — not yet supported on the staggered grid."""
        from spdee.backgrounds.polytrope import PolytropeBackground
        from spdee.applications.solar_eigenmodes import make_polytrope_system

        poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=1e-6)
        sys_ = make_polytrope_system(
            poly, m_val=8, Ek=7e-6, Le=1e-3, with_grad_lnrho_strain=False,
            entropy_bc="neumann", geometry="shell")
        grid = make_shell_grid(N_r=12, N_theta=18, r_inner=0.71, r_outer=0.985)
        with pytest.raises(NotImplementedError):
            compile_fd2_staggered(sys_, grid, m=8)

    def test_accepts_inviscid_three_field(self):
        """The inviscid 3-field compressible system now COMPILES (Layer 1),
        with the layout keyed by unknown names."""
        from spdee.backgrounds.polytrope import PolytropeBackground
        from spdee.applications.solar_eigenmodes import make_polytrope_system

        poly = PolytropeBackground(r_inner=0.71, r_outer=0.985, delta=1e-6)
        sys_ = make_polytrope_system(
            poly, m_val=8, Ek=0.0, entropy_bc="neumann", geometry="shell")
        grid = make_shell_grid(N_r=12, N_theta=18, r_inner=0.71, r_outer=0.985)
        cm = compile_fd2_staggered(sys_, grid, m=8)
        assert [f.name for f in cm.layout.fields] == \
            ["v_r", "v_t", "v_p", "rho1", "s1"]
