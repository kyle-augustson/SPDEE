"""Acceptance tests of the general staggered assembler
(docs/staggered_general_design.md).

  * the rotating incompressible pair: saddle-point classification and
    placement on shell and ball, m = 0, 1, 2, inviscid and viscous with
    stress-free and no-slip walls; the IR objects and the ``Problem``
    strings of one system give one pencil entry for entry; the Dt-less
    (Zhang) spelling carries the implicit unit mass;
  * Boussinesq convection: saddle-point classification, exactly W-skew
    Coriolis blocks, an exactly W-adjoint buoyancy / advection pair, no
    empty pencil rows, and the leading eigenvalues near the onset of
    convection against the Dedalus core at second order;
  * anelastic (solar-eigenproblem polytrope, Div(rho0 v) = 0): the multiplier weighting
    and the Grad / Div adjointness under the rho0-weighted norm;
  * the refusal messages of the unrecognised shapes;
  * the compressible system: the solar-eigenproblem hydro strings and the same system
    as IR objects give one pencil, the ball's r^n row premultiply and
    origin ring, the tensor placement (Grad then Div is the strain Gram,
    Transpose a permutation, Trace consistent with Div, Dot(r_hat, T)
    consistent with the hand gradlnrho blocks), tensor unknowns with the
    first-order reduction, the strain-divergence form of the viscous
    force, and the tensor refusals;
  * background advection: Dot(B0, Grad(v)) with a theta-dependent B0
    through the Hodge-paired contraction, exact against the Coriolis
    block for a solid-body flow (the Doppler shift to rounding), second
    order for a general B0; Div(T) of a non-symmetric tensor unknown at
    second order through the pole-adjacent ring; the strain's pole
    closure at even |m| >= 2 behind ``pole_closure``;
  * constrained transport: the flux rows of the compiled pencil are
    annihilated by the flux divergence (Div Curl = 0) on shell and ball,
    the involution classification, free decay in a conducting shell
    against the Bessel conditions in both the flux and the
    vector-potential formulation, the ideal Alfven band of a uniform
    axial field, the ball's origin closure (the pinned origin ring, the
    parity-closed first-centre averages, the free wall of an ideal
    flux), and the wall refusals;
  * the composite wall rows on the shell: the Robin closure against the
    two-wall Bessel determinant and its Dirichlet / Neumann limits, the
    Navier-slip limits (the no-slip and stress-free pencils bitwise) and
    mixed walls, the open (normal-stress) wall's structure, and the
    refusals of the other composite shapes.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import sympy as smp

from spdee import (
    Unknown, Cross, Grad, Div, Dot, Lap, Dt, Equation, EquationSystem,
    BoundaryCondition, Boundary, Background, r_hat, theta_hat, phi_hat, r, theta,
)
from spdee.ir.equations import ImpenetrableStressFree
from spdee.ir.walker import walk_equation_system
from spdee.problem import Problem
from spdee.ir.operators import Transpose
from spdee.applications.solar_strings import (
    RMAX, RMIN, solar_hydro_problem, ekman, polytrope_exprs,
)
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.staggered_compile import (
    StaggeredGrids, TangentialWall, _avg_1d,
    _reduce_velocity_dofs, _strain_operator_and_norms,
    build_staggered_angular_laplacian, build_staggered_grad_div,
    build_staggered_weighted_viscous_gram, compile_fd2_staggered,
)
from spdee.backends.fd2.staggered_general import (
    _CGridOps, _classify_walls, _layout_for, classify_constraints,
    compile_staggered_general, general_path_reason,
)
from spdee.backends.fd2.staggered_mhd import ct_induction_blocks, ct_resistive_blocks
from spdee.solvers import dense, shift_invert
from test_ball_staggered import make_regular_ball_system
from analytic import vector_calculus as vc
from analytic.manufactured import (
    _add_bcs, _err, _fd_grid, _stag_solve, _stag_state, _stag_values, manufactured,
)


def maxdiff(X, Y):
    D = X - Y
    return abs(D).max() if D.nnz else 0.0


def hand_gradlnrho_strain(grids, m, a_r_fn, order=2):
    """Hand-built staggered contraction Dot(a, S) with a = a_r(r) r̂ and
    the symmetric strain S = ∇v + (∇v)ᵀ (no trace), an oracle independent
    of the Hodge maps of ``_CGridOps``.

    For a radial background only the r-row of S enters:
        (Dot(a,S))_r = a_r · S_rr = a_r · 2 ∂_r v_r
        (Dot(a,S))_θ = a_r · S_rθ = a_r · [ r ∂_r(v_θ/r) + (1/r) ∂_θ v_r ]
        (Dot(a,S))_φ = a_r · S_rφ = a_r · [ (im/(r sinθ)) v_r + r ∂_r(v_φ/r) ]

    Each output component is returned on the v_j sub-grid: r-output on RC
    (r_edge, θ_center), θ-output on CE (r_center, θ_edge), φ-output on CC.
    Radial derivatives are centered 2-point on the same grid with one-sided
    wall rows; the v_r regrid onto CE/CC is a row-normalized edge→center
    average.

    Returns a dict of sparse blocks keyed by (out_component, src_component):
        ('r','r'), ('t','r'), ('t','t'), ('p','r'), ('p','p').
    """
    r_edge, r_center = grids.r_edge, grids.r_center
    th_edge, th_center = grids.theta_edge, grids.theta_center
    Nr_e, Nr_c = len(r_edge), len(r_center)
    Nt_e, Nt_c = len(th_edge), len(th_center)
    hr = r_edge[1] - r_edge[0]
    ht = th_center[1] - th_center[0] if Nt_c > 1 else np.pi
    im = 1j * m
    I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
    I_te = sp.eye(Nt_e, format="csr", dtype=complex)

    def diag_on(rv, tv, fn):
        return sp.diags(np.asarray(fn(rv[:, None], tv[None, :])).ravel()
                        .astype(complex), format="csr")

    # a_r on each output sub-grid
    a_r_RC = diag_on(r_edge, th_center, a_r_fn)        # r-output grid
    a_r_CE = diag_on(r_center, th_edge, a_r_fn)        # θ-output grid
    a_r_CC = diag_on(r_center, th_center, a_r_fn)      # φ-output grid

    blocks = {}

    # r-output on RC: 2 a_r ∂_r v_r, ∂_r edge→edge (centered; one-sided at
    # the walls, whose v_r rows are pinned or eliminated downstream)
    Dr_ee = np.zeros((Nr_e, Nr_e), dtype=complex)
    for i in range(1, Nr_e - 1):
        Dr_ee[i, i - 1] = -1.0 / (2 * hr)
        Dr_ee[i, i + 1] = 1.0 / (2 * hr)
    Dr_ee[0, 0] = -1.0 / hr; Dr_ee[0, 1] = 1.0 / hr
    Dr_ee[-1, -2] = -1.0 / hr; Dr_ee[-1, -1] = 1.0 / hr
    dr_vr = sp.kron(sp.csr_matrix(Dr_ee), I_tc, format="csr")  # RC→RC
    blocks[("r", "r")] = (2.0 * a_r_RC @ dr_vr).tocsr()

    # θ-output on CE, term A: r ∂_r(v_θ/r), ∂_r center→center
    inv_r_CE = diag_on(r_center, th_edge, lambda r, t: 1.0 / r + 0 * t)
    r_CE = diag_on(r_center, th_edge, lambda r, t: r + 0 * t)
    Dr_cc = np.zeros((Nr_c, Nr_c), dtype=complex)
    for i in range(1, Nr_c - 1):
        Dr_cc[i, i - 1] = -1.0 / (2 * hr)
        Dr_cc[i, i + 1] = 1.0 / (2 * hr)
    Dr_cc[0, 0] = -1.0 / hr; Dr_cc[0, 1] = 1.0 / hr
    Dr_cc[-1, -2] = -1.0 / hr; Dr_cc[-1, -1] = 1.0 / hr
    dr_ce = sp.kron(sp.csr_matrix(Dr_cc), I_te, format="csr")  # CE→CE
    termA_tt = r_CE @ dr_ce @ inv_r_CE                          # CE→CE

    # θ-output on CE, term B: (1/r) ∂_θ v_r. ∂_θ center→edge on the RC
    # field gives (r_edge, θ_edge); then average r_edge→r_center.
    Dt_c2e = np.zeros((Nt_e, Nt_c), dtype=complex)
    for j in range(1, Nt_e - 1):
        Dt_c2e[j, j - 1] = -1.0 / ht
        Dt_c2e[j, j] = 1.0 / ht
    avg_e2c = _avg_1d(Nr_e, Nr_c).conj().T  # (Nr_c × Nr_e) unnormalized
    avg_e2c = _row_normalize(avg_e2c)
    inv_r_CE_b = diag_on(r_center, th_edge, lambda r, t: 1.0 / r + 0 * t)
    dtheta_RC_to_REe = sp.kron(sp.eye(Nr_e, format="csr", dtype=complex),
                               sp.csr_matrix(Dt_c2e), format="csr")  # RC→(r_edge,θ_edge)
    avg_e2c_te = sp.kron(avg_e2c, I_te, format="csr")  # (r_edge,θ_edge)→(r_center,θ_edge)
    termB_tr = inv_r_CE_b @ avg_e2c_te @ dtheta_RC_to_REe          # RC→CE
    blocks[("t", "t")] = (a_r_CE @ termA_tt).tocsr()
    blocks[("t", "r")] = (a_r_CE @ termB_tr).tocsr()

    # φ-output on CC, term C: (im/(r sinθ)) v_r, v_r averaged r_edge→r_center
    coef_CC = diag_on(r_center, th_center,
                      lambda r, t: im / (r * np.sin(t)))
    avg_e2c_tc = sp.kron(avg_e2c, I_tc, format="csr")  # RC→CC
    termC_pr = coef_CC @ avg_e2c_tc                                # RC→CC

    # φ-output on CC, term D: r ∂_r(v_φ/r), ∂_r center→center
    inv_r_CC = diag_on(r_center, th_center, lambda r, t: 1.0 / r + 0 * t)
    r_CC = diag_on(r_center, th_center, lambda r, t: r + 0 * t)
    dr_cc_tc = sp.kron(sp.csr_matrix(Dr_cc), I_tc, format="csr")    # CC→CC
    termD_pp = r_CC @ dr_cc_tc @ inv_r_CC                          # CC→CC
    blocks[("p", "r")] = (a_r_CC @ termC_pr).tocsr()
    blocks[("p", "p")] = (a_r_CC @ termD_pp).tocsr()

    return blocks


def _row_normalize(op):
    """Normalize each row of a sparse interpolation to sum to 1 (partition
    of unity), leaving all-zero rows untouched."""
    op = op.tolil()
    for i in range(op.shape[0]):
        s = sum(op.data[i]) if op.data[i] else 0.0
        if abs(s) > 1e-30:
            op.data[i] = [v / s for v in op.data[i]]
    return op.tocsr()


def grid_for(shell, N_r=9, N_theta=12):
    if shell:
        return make_shell_grid(N_r=N_r, N_theta=N_theta, r_inner=0.4)
    return make_ball_grid(N_r=N_r, N_theta=N_theta)


def walls_of(shell):
    return (Boundary.R_INNER, Boundary.R_OUTER) if shell else (Boundary.R_OUTER,)


# ---------------------------------------------------------------------------
# Systems
# ---------------------------------------------------------------------------

def zhang_system(Ek=0.0, wall="impenetrable", shell=True, gauge=(), with_dt=True):
    """The rotating incompressible pair; ``with_dt`` False spells the
    momentum equation as the eigenvalue operator itself (the Zhang form),
    which classifies with an implicit unit mass on v."""
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    ez = Background.Vector("ez", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                           v_phi=smp.S.Zero)
    mom = 2 * Cross(ez, v) + Grad(p)
    if with_dt:
        mom = Dt(v) + mom
    if Ek:
        mom = mom - smp.S(Ek) * Lap(v)
    bcs = []
    for w in walls_of(shell):
        if wall == "stress_free":
            bcs += list(ImpenetrableStressFree(v, boundary=w))
        elif wall == "noslip":
            bcs += [BoundaryCondition(w, Dot(u, v)) for u in (r_hat, theta_hat, phi_hat)]
        else:
            bcs.append(BoundaryCondition(w, Dot(r_hat, v)))
    return EquationSystem([Equation(mom, name="momentum"),
                           Equation(Div(v), name="continuity")], bcs,
                          gauge_unknowns=list(gauge))


def zhang_problem(Ek=0.0, wall="impenetrable", shell=True):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if Ek:
        pb.add_parameter("Ek", Ek)
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - Ek*lap(v) = 0",
                        name="momentum")
    else:
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in (("inner", "outer") if shell else ("outer",)):
        if wall == "stress_free":
            pb.add_stress_free("v", at=w)
        elif wall == "noslip":
            for u in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({u}, v) = 0", at=w)
        else:
            pb.add_bc("dot(r_hat, v) = 0", at=w)
    return pb


def boussinesq_problem(Ra=1e3, Ek=1e-2, Pr=1.0, wall="stress_free", shell=True,
                       coriolis=True, gravity="const", r_inner=0.5,
                       viscous="lap(v)"):
    """dt(v) + 2 ez x v + grad(p) - Ra T g - Ek (viscous) = 0,
    dt(T) + v . grad_T0 - (Ek/Pr) lap(T) = 0, div(v) = 0, T = 0 walls."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("T", "scalar")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if gravity == "const":
        pb.add_background("grad_T0", v_r=-1)
        buoy = "Ra*T*r_hat"
    else:
        pb.add_background("grad_T0", v_r="-r")
        pb.add_background("g", v_r="r")
        buoy = "Ra*T*g"
    for name, value in (("Ra", Ra), ("Ek", Ek), ("Pr", Pr)):
        pb.add_parameter(name, value, symbolic=False)
    cor = " + 2*cross(ez, v)" if coriolis else ""
    visc = f" - Ek*({viscous})" if Ek else ""
    pb.add_equation(f"dt(v){cor} + grad(p) - {buoy}{visc} = 0", name="momentum")
    pb.add_equation("dt(T) + dot(v, grad_T0) - (Ek/Pr)*lap(T) = 0", name="energy")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in (("inner", "outer") if shell else ("outer",)):
        if wall == "stress_free":
            pb.add_stress_free("v", at=w)
        elif wall == "noslip":
            for u in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({u}, v) = 0", at=w)
        else:
            pb.add_bc("dot(r_hat, v) = 0", at=w)
        pb.add_bc("T = 0", at=w)
    pb.r_inner = r_inner
    return pb


def anelastic_problem(Ek=1e-3, delta=1e-6):
    """The solar-eigenproblem polytrope with Div(rho0 v) = 0 in place of continuity."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("S", "scalar")
    pb.add_unknown("p", "scalar")
    ex = polytrope_exprs(pb.namespace["r"])
    pb.add_background("rho0", expr=ex["rho0"])
    pb.add_background("gacc", expr=ex["gacc"])
    pb.add_background("invHp", expr=-ex["dlnp"])
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("er_", v_r=1)
    pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_parameter("delta", delta, symbolic=False)
    pb.add_equation("dt(v) + grad(p) - gacc*S*er_ + 2*cross(ez, v) - Ek*lap(v) = 0",
                    name="momentum")
    pb.add_equation("dt(S) - delta*invHp*dot(er_, v) - Ek*lap(S) = 0", name="energy")
    pb.add_equation("div(rho0*v) = 0", name="continuity")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="inner")
    pb.add_bc("dot(r_hat, grad(S)) = 0", at="outer")
    return pb


def reduced_norm(system, cp, grid, m):
    """P^H W P: the volume norm of the assembled (wall-reduced) state, with
    the reduction of the compiled pencil."""
    sg = StaggeredGrids.from_grid(grid)
    ops = _CGridOps(sg, m, 2)
    record = cp.constraints
    full = _layout_for(system, record, ops)
    parts = []
    for f in full.fields:
        loc = {"center": "CC", "r_edge": "RC", "theta_edge": "CE"}[f.location]
        parts.append(ops.W[loc])
    W = sp.diags(np.concatenate(parts)).tocsr()
    face = [u for u, p in record["placement"].items() if p == "face"][0]
    if record["viscous"][face]:
        W, _, _ = _reduce_velocity_dofs(W, W, full, sg, m, face)
    return W.tocsr(), ops


# ---------------------------------------------------------------------------
# The saddle-point path on the rotating incompressible pair
# ---------------------------------------------------------------------------

CASES = [(0.0, "impenetrable"), (1e-2, "stress_free"), (1e-2, "noslip")]


class TestInertialSaddlePoint:

    @pytest.mark.parametrize("Ek,wall", CASES, ids=[c[1] for c in CASES])
    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_classification_and_layout(self, shell, m, Ek, wall):
        """Div(v) = 0 is the saddle-point constraint of the multiplier p,
        v takes the faces and p the centres, and a Lap(v) selects the
        symmetric wall reduction over the inviscid row overwrite."""
        grid = grid_for(shell)
        ops = _CGridOps(StaggeredGrids.from_grid(grid), m, 2)
        cp = compile_staggered_general(zhang_system(Ek, wall, shell), grid, m)
        assert cp.layout.placements == {"v": "face", "p": "center"}
        assert [(f.name, f.location) for f in cp.layout.fields] == [
            ("v_r", "r_edge"), ("v_t", "theta_edge"), ("v_p", "center"),
            ("p", "center")]
        assert cp.layout.field("p").size == ops.n["CC"]
        assert cp.constraints["equations"]["continuity"]["class"] == "saddle_point"
        assert cp.constraints["equations"]["continuity"]["multiplier"] == "p"
        assert cp.constraints["viscous"]["v"] == bool(Ek)
        n_v = cp.layout.field("p").offset
        B = cp.B_total.tocsr()
        assert (B - sp.diags(B.diagonal())).nnz == 0
        assert B[n_v:].nnz == 0, "the multiplier rows carry no mass"
        mass = B.diagonal()[:n_v]
        assert np.count_nonzero(mass) >= n_v - 2 * ops.n["CC"]
        assert np.allclose(mass[mass != 0].imag, 0.0) and mass[mass != 0].real.min() > 0

    @pytest.mark.parametrize("Ek,wall", CASES, ids=[c[1] for c in CASES])
    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_objects_and_strings_are_the_same_pencil(self, shell, m, Ek, wall):
        """``Problem.compile(discretization='staggered')`` on the strings
        and the IR objects of the same system give one pencil, entry for
        entry, on both geometries and every wall closure."""
        grid = grid_for(shell)
        cp = zhang_problem(Ek, wall, shell).compile(
            grid, m, discretization="staggered", order_r=2, order_theta=2)
        obj = compile_staggered_general(zhang_system(Ek, wall, shell), grid, m)
        assert cp.constraints["multipliers"] == {"p": "continuity"}
        assert maxdiff(cp.A_total, obj.A_total) == 0.0
        assert maxdiff(cp.B_total, obj.B_total) == 0.0

    def test_dt_less_form_carries_the_implicit_mass(self):
        """With no Dt anywhere the momentum equation is the eigenvalue
        operator itself: v gets the identity in B, exactly as a unit Dt(v)
        would, and the rest of the pencil is unchanged."""
        grid = grid_for(True)
        with_dt = compile_fd2_staggered(zhang_system(), grid, 1)
        without = compile_fd2_staggered(zhang_system(with_dt=False), grid, 1)
        assert general_path_reason(zhang_system(with_dt=False), 1) is None
        assert with_dt.constraints["equations"]["momentum"] == {
            "class": "evolution", "unknown": "v"}
        assert without.constraints["equations"]["momentum"] == {
            "class": "evolution", "unknown": "v", "implicit_mass": True}
        assert maxdiff(with_dt.A_total, without.A_total) == 0.0
        assert maxdiff(with_dt.B_total, without.B_total) == 0.0

    def test_integral_gauge_when_declared(self):
        """At m = 0 the declared gauge unknown takes the integral gauge
        row (the volume weights) instead of the point gauge."""
        grid = grid_for(True)
        cp = compile_staggered_general(zhang_system(gauge=["p"]), grid, 0)
        row = cp.layout.field("p").offset
        A = cp.A_total.tolil()
        ops = _CGridOps(StaggeredGrids.from_grid(grid), 0, 2)
        assert A.rows[row] == list(range(row, row + ops.n["CC"]))
        assert np.allclose(A.data[row], ops.W["CC"])
        assert cp.constraints["equations"]["continuity"]["gauge"] == "integral"
        cp0 = compile_staggered_general(zhang_system(), grid, 0)
        assert cp0.constraints["equations"]["continuity"]["gauge"] == "point"


# ---------------------------------------------------------------------------
# Boussinesq convection
# ---------------------------------------------------------------------------

class TestBoussinesq:

    @pytest.mark.parametrize("wall", ["stress_free", "noslip"])
    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_structure(self, shell, m, wall):
        """Saddle point with multiplier p; the buoyancy block and the
        advection block are exact W-adjoints (W_v B = Ra (W_c S)^T on the
        reduced pencil); no row is empty in both A and B."""
        grid = grid_for(shell)
        pb = boussinesq_problem(wall=wall, shell=shell)
        system = pb.build()
        cp = compile_staggered_general(system, grid, m, params=pb.params)
        rec = cp.constraints
        assert {k: v["class"] for k, v in rec["equations"].items()} == {
            "momentum": "evolution", "energy": "evolution",
            "continuity": "saddle_point"}
        assert rec["equations"]["continuity"]["multiplier"] == "p"
        assert rec["equations"]["continuity"]["weight"] is None
        assert rec["walls"]["vectors"]["v"]["tangential"] == wall
        assert rec["walls"]["scalars"]["T"]["closure"] == (
            ("dirichlet", "dirichlet") if shell else ("neumann", "dirichlet"))
        A, B = cp.A_total.tocsr(), cp.B_total.tocsr()
        empty = (np.diff(A.indptr) == 0) & (np.diff(B.indptr) == 0)
        assert not empty.any()
        W, ops = reduced_norm(system, cp, grid, m)
        lay = cp.layout
        n_v = lay.field("T").offset
        sT = lay.field("T").slice
        X = (W[:n_v, :n_v] @ A[:n_v, sT]).toarray()
        Y = (sp.diags(ops.W["CC"]) @ A[sT, :n_v]).toarray().T
        assert np.abs(X - 1e3 * Y).max() <= 1e-14 * np.abs(X).max()

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_coriolis_exactly_skew(self, shell, m):
        """On the inviscid system the velocity block is the Coriolis
        operator alone; away from the overwritten wall rows and the
        weightless DOFs it is W-skew to rounding."""
        grid = grid_for(shell)
        pb = boussinesq_problem(Ek=0.0, wall="impenetrable", shell=shell)
        cp = compile_staggered_general(pb.build(), grid, m, params=pb.params)
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, m, 2)
        W = ops.face_norm()
        Nt, Nr_e = len(sg.theta_center), len(sg.r_edge)
        keep = np.abs(W) > 0
        keep[:Nt] = False
        keep[(Nr_e - 1) * Nt:] = False
        n_v = cp.layout.field("T").offset
        C = cp.A_total.tocsr()[:n_v, :n_v][keep][:, keep].toarray()
        S = np.diag(W[keep]) @ C
        assert np.abs(S + S.T).max() <= 1e-14 * np.abs(S).max()

    def test_onset_eigenvalue_against_dedalus(self):
        """Ra = 2.5, Ek = 1e-2, Pr = 1, m = 2, stress-free shell of radius
        ratio 0.5, just above onset: the least-damped eigenvalue of the
        C-grid converges at second order to the Dedalus value on the same
        strings (lambda_D = -0.02510814 + 0.04842610i, stable to 1e-8
        between N_r = 16 and 24; C-grid errors 1.12e-3, 2.77e-4, 6.89e-5
        at 12x24, 24x48, 48x96)."""
        pytest.importorskip("dedalus")
        from spdee.backends.dedalus import DedalusGrid
        pb = boussinesq_problem(Ra=2.5, Ek=1e-2, wall="stress_free", shell=True)
        target = -0.0251 + 0.0484j
        cpd = pb.compile(DedalusGrid(N_r=16, N_theta=32, r_inner=0.5, r_outer=1.0),
                         m=2, backend="dedalus", gauge_unknowns=["p"])
        rd = cpd.solve_evp(k=6, sigma=target)
        ld = rd.eigenvalues[np.isfinite(rd.eigenvalues)]
        lam_D = ld[np.argmin(np.abs(ld - target))]
        assert abs(lam_D - (-0.02510814 + 0.04842610j)) < 1e-6
        errs = []
        for N in (12, 24):
            grid = make_shell_grid(N_r=N + 1, N_theta=2 * N, r_inner=0.5, r_outer=1.0)
            cp = pb.compile(grid, 2, discretization="staggered", order_r=2, order_theta=2)
            K, M = cp.assemble_evp_matrices()
            v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
            lam, _ = shift_invert(K, M, lam_D + 1e-3, 8, return_vectors=True,
                                  tol=1e-10, v0=v0)
            errs.append(abs(lam[np.argmin(np.abs(lam - lam_D))] - lam_D))
        assert errs[1] < 5e-4
        assert errs[1] < 0.3 * errs[0], errs

    def test_general_gravity_pair_is_adjoint_to_second_order(self):
        """With g(r) = r and dT0/dr = -r the backgrounds sit on the faces
        (buoyancy) and on the centres (advection): the pair is adjoint to
        the interpolation error, first order entrywise and second order on
        smooth fields."""
        devs = []
        for N_r in (9, 17):
            grid = make_shell_grid(N_r=N_r, N_theta=12, r_inner=0.5)
            pb = boussinesq_problem(gravity="r", shell=True)
            system = pb.build()
            cp = compile_staggered_general(system, grid, 2, params=pb.params)
            W, ops = reduced_norm(system, cp, grid, 2)
            lay = cp.layout
            n_v = lay.field("T").offset
            sT = lay.field("T").slice
            A = cp.A_total.tocsr()
            X = (W[:n_v, :n_v] @ A[:n_v, sT]).toarray()
            Y = (sp.diags(ops.W["CC"]) @ A[sT, :n_v]).toarray().T
            devs.append(np.abs(X - 1e3 * Y).max() / np.abs(X).max())
        assert devs[1] < 0.6 * devs[0]


# ---------------------------------------------------------------------------
# Anelastic
# ---------------------------------------------------------------------------

class TestAnelastic:

    @pytest.mark.parametrize("m", [0, 2])
    def test_rho0_weighted_saddle_point(self, m):
        pb = anelastic_problem()
        system = pb.build()
        rec = classify_constraints(system, walk_equation_system(system, m))
        info = rec["equations"]["continuity"]
        assert info["class"] == "saddle_point" and info["multiplier"] == "p"
        assert info["weight"] is system.backgrounds["rho0"].component
        grid = make_shell_grid(N_r=13, N_theta=16, r_inner=0.71, r_outer=0.985)
        cp = compile_staggered_general(system, grid, m, params=pb.params)
        assert cp.layout.placements == {"v": "face", "S": "center", "p": "center"}
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, m, 2)
        rho = system.backgrounds["rho0"].component
        rho_v = np.concatenate([ops.evaluate(rho, k) for k in ops.face_locs()])
        Wv = sp.diags(rho_v * ops.face_norm()).tocsr()
        Wc = sp.diags(ops.W["CC"])
        G, D = ops.gradient(), ops.divergence(rho)
        ref = abs(G.conj().T @ Wv).max()
        assert abs(Wc @ D + G.conj().T @ Wv).max() <= 1e-14 * ref
        full = _layout_for(system, cp.constraints, ops)
        W_all = sp.diags(np.concatenate([rho_v * ops.face_norm(), ops.W["CC"],
                                         ops.W["CC"]])).tocsr()
        W_red, _, _ = _reduce_velocity_dofs(W_all, W_all, full, sg, m, "v")
        n_v = cp.layout.field("S").offset
        Wv_red = W_red[:n_v, :n_v]
        sp_ = cp.layout.field("p").slice
        A = cp.A_total.tocsr()
        D_red, G_red = A[sp_, :n_v], A[:n_v, sp_]
        rows = np.arange(A.shape[0])[sp_]
        gauge = cp.constraints["equations"]["continuity"]["gauge"]
        if gauge is not None:
            rows = rows[1:]
            D_red = A[rows, :n_v]
            Wc = sp.diags(ops.W["CC"][1:])
            ref = abs(G_red[:, 1:].conj().T @ Wv_red).max()
            assert abs(Wc @ D_red + G_red[:, 1:].conj().T @ Wv_red).max() <= 1e-14 * ref
        else:
            ref = abs(G_red.conj().T @ Wv_red).max()
            assert abs(Wc @ D_red + G_red.conj().T @ Wv_red).max() <= 1e-14 * ref


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

GRID = make_shell_grid(N_r=7, N_theta=8, r_inner=0.5)


def _base_problem():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    return pb


def _add_walls(pb, bc="dot(r_hat, v) = 0"):
    pb.add_bc(bc, at="inner")
    pb.add_bc(bc, at="outer")
    return pb


class TestRefusals:

    def test_algebraic_relation_without_multiplier(self):
        pb = _base_problem()
        pb.add_unknown("T", "scalar")
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_equation("T - p = 0", name="relation")
        _add_walls(pb)
        with pytest.raises(NotImplementedError) as info:
            compile_staggered_general(pb.build(), GRID, 1)
        msg = str(info.value)
        assert "'relation'" in msg and "unrecognised" in msg
        assert "identity_scalar(T) + identity_scalar(p)" in msg
        assert "Div(M u) = 0" in msg and "involution" in msg

    def test_divergence_without_grad_or_curl(self):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_equation("dt(v) + 2*cross(ez, v) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        _add_walls(pb)
        with pytest.raises(NotImplementedError) as info:
            classify_constraints(pb.build(), walk_equation_system(pb.build(), 1))
        msg = str(info.value)
        assert "'continuity'" in msg and "unrecognised" in msg
        assert "found div_vector_scalar(v)" in msg
        assert "no Curl evolution" in msg
        assert general_path_reason(pb.build(), 1) == msg

    def test_tensor_unknown_without_tensor_equation(self):
        pb = _base_problem()
        pb.add_unknown("S", "tensor")
        pb.add_equation("dt(v) + grad(p) + div(S) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        _add_walls(pb)
        with pytest.raises(NotImplementedError, match="tensor unknown 'S'") as info:
            compile_staggered_general(pb.build(), GRID, 1)
        assert "section 2b" in str(info.value)

    def test_lap_and_grad_of_a_tensor(self):
        """Rank-3 gradients and tensor Laplacians are refused by the IR
        itself, before any assembler sees them."""
        pb = _base_problem()
        pb.add_unknown("S", "tensor")
        with pytest.raises(TypeError, match="Lap requires a scalar or vector"):
            pb.add_equation("dt(S) - lap(S) = 0", name="bad")
        with pytest.raises(TypeError, match="Grad requires a scalar or vector"):
            pb.add_equation("dt(v) + div(grad(S)) = 0", name="bad")

    def test_tensor_term_with_background_tensor(self):
        """Div of a background tensor times u is not a Grad(u) combination."""
        pb = _base_problem()
        pb.add_background("T0", components={("r", "r"): "r", ("theta", "theta"): "1"},
                          symmetric=True)
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - div(T0*div(v)) = 0",
                        name="momentum") if False else None
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - div(grad(v) + T0) = 0",
                        name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        _add_walls(pb)
        with pytest.raises((NotImplementedError, TypeError)):
            compile_staggered_general(pb.build(), GRID, 1)

    def test_composite_row_of_another_shape(self):
        """A composite row that is neither Robin, Navier slip nor the
        normal stress is refused naming the term."""
        pb = boussinesq_problem()
        pb.add_bc("dot(r_hat, grad(T)) + dot(r_hat, v) = 0", at="outer", name="odd")
        with pytest.raises(NotImplementedError) as info:
            compile_staggered_general(pb.build(), GRID, 1, params=pb.params)
        msg = str(info.value)
        assert "'odd'" in msg and "is not one of these" in msg
        assert "Robin on a centre scalar" in msg

    def test_varying_scale_of_constraint(self):
        pb = _base_problem()
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
        pb.add_equation("r*div(v) = 0", name="continuity")
        _add_walls(pb)
        with pytest.raises(NotImplementedError, match="Div\\(M v\\)"):
            compile_staggered_general(pb.build(), GRID, 1)

    def test_problem_compile_reports_the_classification(self):
        """``Problem.compile(discretization='staggered')`` reaches the same
        refusal: an algebraic relation between two scalars pairs with
        nothing."""
        pb = _base_problem()
        pb.add_unknown("T", "scalar")
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_equation("T - p = 0", name="relation")
        _add_walls(pb)
        with pytest.raises(NotImplementedError, match="is algebraic \\(no Dt\\)"):
            compile_problem(pb.build(), GRID, 1, discretization="staggered")


# ---------------------------------------------------------------------------
# Stage 3: the compressible system through the dispatcher
# ---------------------------------------------------------------------------

def solar_named_system(rho_form, Ek, delta, pressure="product"):
    """The solar-eigenproblem hydro system with the background names the hand assembler
    reads, built from the same polytrope expressions as the strings.
    ``rho_form`` 'true' carries rho0 (the viscous Gram weight) and 'unit'
    carries rho0 = 1 with d ln rho0 / dr as the mass-flux stratification
    (the rhat = rho1 / rho0 continuity row of the strings).  ``pressure``
    'product' spells the pressure gradient by the product rule with the
    background gradients, as ``make_polytrope_system`` does; 'grouped'
    spells it Grad(cs2 rho1) + Grad(cs2 s1), the form the hand assembler
    builds as G diag(cs2) and the walker expands into the tagged
    product-rule pair."""
    ex = polytrope_exprs(r)

    def S(name, e):
        return Background.Scalar(name, smp.sympify(e))

    def Vr(name, e):
        return Background.Vector(name, v_r=smp.sympify(e), v_theta=0, v_phi=0)
    rho0 = ex["rho0"] if rho_form == "true" else smp.S(1)
    drho = smp.diff(ex["rho0"], r) if rho_form == "true" else ex["dlnrho"]
    cs2_over_rho0, cs2_over_cp = S("cs2_over_rho0", ex["Cs2"]), S("cs2_over_cp", ex["Cs2"])
    grad_cs2 = Vr("grad_cs2_over_rho0", smp.diff(ex["Cs2"], r))
    grad_cs2_cp = Vr("grad_cs2_over_cp", smp.diff(ex["Cs2"], r))
    g_over_cp_vec = Vr("g_over_cp_vec", ex["gacc"])
    poly_rho, poly_grad_rho = S("poly_rho", rho0), Vr("poly_grad_rho", drho)
    grad_s0 = Vr("grad_s0_optA", delta * ex["dlnp"])
    grad_ln_rho, grad_ln_P0 = Vr("grad_ln_rho", ex["dlnrho"]), Vr("grad_ln_P0", ex["dlnp"])
    Om = Background.Vector("poly_Omega_vec", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                           v_phi=0)
    v, rho1, s1 = Unknown.Vector("v"), Unknown.Scalar("rho1"), Unknown.Scalar("s1")
    if pressure == "grouped":
        pgrad = Grad(cs2_over_rho0 * rho1) + Grad(cs2_over_cp * s1)
    else:
        pgrad = (grad_cs2 * rho1 + cs2_over_rho0 * Grad(rho1)
                 + grad_cs2_cp * s1 + cs2_over_cp * Grad(s1))
    mom = Dt(v) + pgrad - g_over_cp_vec * s1 + 2 * Cross(Om, v)
    cont = Dt(rho1) + poly_rho * Div(v) + Dot(poly_grad_rho, v)
    en = Dt(s1) + Dot(grad_s0, v)
    if Ek > 0:
        st = Grad(v) + Transpose(Grad(v))
        mom = (mom - smp.S(Ek) * Lap(v) - smp.Rational(1, 3) * smp.S(Ek) * Grad(Div(v))
               - smp.S(Ek) * Dot(grad_ln_rho, st)
               + smp.Rational(2, 3) * smp.S(Ek) * grad_ln_rho * Div(v))
        en = en - smp.S(Ek) * Lap(s1) - smp.S(Ek) * Dot(grad_ln_P0, Grad(s1))
    bcs = []
    for w in (Boundary.R_INNER, Boundary.R_OUTER):
        bcs += list(ImpenetrableStressFree(v, boundary=w))
        bcs.append(BoundaryCondition(w, Dot(r_hat, Grad(s1))))
    return EquationSystem([Equation(mom, name="momentum"), Equation(cont, name="continuity"),
                           Equation(en, name="energy")], bcs)


def block_deviations(A, lay_a, B, lay_b, rename=()):
    """{(row field, column field): (max |A - B|, max |A|, |B|)} per block."""
    rename = dict(rename)
    out = {}
    for fi in lay_a.fields:
        for fj in lay_a.fields:
            gi = lay_b.field(rename.get(fi.name, fi.name)).slice
            gj = lay_b.field(rename.get(fj.name, fj.name)).slice
            X, Y = A[fi.slice, fj.slice], B[gi, gj]
            D = X - Y
            out[(fi.name, fj.name)] = (
                abs(D).max() if D.nnz else 0.0,
                max(abs(X).max() if X.nnz else 0.0, abs(Y).max() if Y.nnz else 0.0))
    return out


SHELL_CASES = [
    (12, 48, 8, 2e12, 1e-6, "center"), (12, 48, 8, 2e12, 1e-6, "solar_eigenproblem"),
    (12, 48, 8, 0.0, 1e-6, "center"), (8, 16, 8, 2e12, 0.0, "center"),
    (8, 16, 0, 2e12, 1e-6, "center"), (8, 16, 1, 2e12, 1e-6, "solar_eigenproblem"),
    (8, 16, 1, 0.0, 0.0, "center"), (8, 16, 0, 0.0, 1e-6, "solar_eigenproblem"),
]


class TestCompressibleSpellings:

    @pytest.mark.parametrize("Nr,Nt,m,nu,delta,pairing", SHELL_CASES,
                             ids=[f"{c[0]}x{c[1]}-m{c[2]}-nu{c[3]:g}-d{c[4]:g}-{c[5]}"
                                  for c in SHELL_CASES])
    def test_strings_and_objects_are_the_same_pencil(self, Nr, Nt, m, nu, delta, pairing):
        """``solar_hydro_problem`` (Problem strings, rhat = rho1 / rho0)
        and ``solar_named_system('unit')`` (IR objects on the same
        polytrope expressions) are one pencil: the same layout, B exactly,
        A exactly when Ek = 0, and with Ek != 0 every block to 1e-14 of its
        own maximum -- the viscous Gram's density weight is recovered from
        d ln rho0 / dr by quadrature and the thermal coefficient arrives
        folded with Ekap, so both carry one rounding."""
        grid = make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=RMIN, r_outer=RMAX)
        pb = solar_hydro_problem(nu_cgs=nu, delta=delta)
        cp = compile_fd2_staggered(pb.build(), grid, m, coriolis_pairing=pairing)
        ref = compile_fd2_staggered(
            solar_named_system("unit", ekman(nu), delta, pressure="grouped"),
            grid, m, coriolis_pairing=pairing)
        rename = {"rhat": "rho1", "S": "s1"}
        assert [(rename.get(f.name, f.name), f.location, f.offset, f.size)
                for f in cp.layout.fields] == \
            [(f.name, f.location, f.offset, f.size) for f in ref.layout.fields]
        assert cp.constraints["viscous"]["v"] == (nu != 0.0)
        assert cp.constraints["r_power"] is None
        assert maxdiff(cp.B_total, ref.B_total) == 0.0
        A, A_ref = cp.A_total.tocsr(), ref.A_total.tocsr()
        if nu == 0.0:
            assert maxdiff(A, A_ref) == 0.0
            return
        for key, (d, scale) in block_deviations(A, cp.layout, A_ref, ref.layout,
                                                rename).items():
            assert d <= 1e-14 * scale, (key, d, scale)

    def test_classification_of_the_strings(self):
        grid = make_shell_grid(N_r=9, N_theta=16, r_inner=RMIN, r_outer=RMAX)
        system = solar_hydro_problem().build()
        assert general_path_reason(system, 8) is None
        cp = compile_fd2_staggered(system, grid, 8)
        assert cp.constraints is not None
        assert {k: v["class"] for k, v in cp.constraints["equations"].items()} == {
            "momentum": "evolution", "continuity": "evolution", "energy": "evolution"}
        assert cp.constraints["walls"]["scalars"]["S"]["closure"] == ("neumann", "neumann")
        assert cp.constraints["walls"]["vectors"]["v"]["tangential"] == "stress_free"
        assert cp.constraints["constraints"] == {}

    def test_entropy_dirichlet_from_the_bc_task(self):
        """The wall closure of the thermal operator comes from the BC on S,
        not from a name: s = 0 walls change the thermal block in the wall
        rows only, and mixed walls are refused."""
        grid = make_shell_grid(N_r=9, N_theta=16, r_inner=RMIN, r_outer=RMAX)
        pb = solar_hydro_problem()
        pb.add_bc("S = 0", at="inner")
        pb.add_bc("S = 0", at="outer")
        with pytest.raises(NotImplementedError, match="both neumann and dirichlet"):
            compile_staggered_general(pb.build(), grid, 8)
        neu = compile_staggered_general(solar_hydro_problem().build(), grid, 8)
        dir_ = compile_staggered_general(
            solar_hydro_problem(entropy_bc="dirichlet").build(), grid, 8)
        assert dir_.constraints["walls"]["scalars"]["S"]["closure"] == \
            ("dirichlet", "dirichlet")
        sl = neu.layout.field("S").slice
        D = abs((neu.A_total.tocsr()[sl, sl] - dir_.A_total.tocsr()[sl, sl]).tocoo())
        ops = _CGridOps(StaggeredGrids.from_grid(grid), 8, 2)
        n_t = len(ops.grids.theta_center)
        rows = set(np.unique(D.row[D.data != 0]) // n_t)
        assert rows == {0, len(ops.grids.r_center) - 1}


class TestCompressibleBall:

    @pytest.mark.parametrize("Ek,delta", [(0.0, 0.0), (1e-3, 2e-5)])
    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_ball_row_premultiply_and_origin(self, m, Ek, delta):
        """A pencil without multipliers is row-scaled by r^n per equation,
        n = max(1, the natural r-power of its terms) with the origin ring
        at the half-cell radius: B stays diagonal and carries that scaling
        on the centre fields (the record keeps the natural power, the
        scaling takes max(1, n)), and the origin v_r rows hold the
        |m| >= 2 Dirichlet row or the m = 0 / |m| = 1 profile anchor."""
        grid = make_ball_grid(N_r=9, N_theta=16)
        cp = compile_staggered_general(
            make_regular_ball_system(m, Ek=Ek, delta=delta, strain=True,
                                     pressure="grad"), grid, m)
        n = 2 if Ek else 1
        assert cp.constraints["r_power"] == {"momentum": n, "continuity": 1,
                                             "energy": 2 if Ek else 0}
        sg = StaggeredGrids.from_grid(grid)
        Nt_c = len(sg.theta_center)
        B = cp.B_total.tocsr()
        assert (B - sp.diags(B.diagonal())).nnz == 0
        for name, power in (("rho1", 1), ("s1", 2 if Ek else 1)):
            w = np.repeat(sg.r_center, Nt_c) ** power
            assert np.allclose(B.diagonal()[cp.layout.field(name).slice], w, rtol=1e-14)
        if Ek:
            return
        A = cp.A_total.tolil()
        if abs(m) >= 2:
            assert all(A.rows[i] == [i] for i in range(Nt_c))
            return
        prof = np.cos(sg.theta_center) if m == 0 else np.sin(sg.theta_center)
        assert all(A.rows[i] == [0, i] for i in range(1, Nt_c))
        assert np.allclose([A.data[i][0] for i in range(1, Nt_c)], -prof[1:] / prof[0])
        assert cp.B_total.tolil().rows[0] == [0], "the ring master keeps its mass"


# ---------------------------------------------------------------------------
# Stage 3: the tensor placement
# ---------------------------------------------------------------------------

def smooth_face_field(sg):
    re_, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
    vr = (re_[:, None]**2 * np.sin(tc[None, :])**2).ravel()
    vt = (rc[:, None]**2 * np.sin(te[None, :]) * np.cos(te[None, :])).ravel()
    vp = (rc[:, None]**2 * np.sin(tc[None, :])).ravel()
    return np.concatenate([vr, vt, vp]).astype(complex)


def deep_interior(nr, nt, dr=3, dt=2):
    """Away from the wall-adjacent centres, where the adjoint divergence
    is not a consistent stencil, and from the pole rows."""
    mk = np.ones((nr, nt), bool)
    mk[:dr, :] = mk[-dr:, :] = False
    mk[:, :dt] = mk[:, -dt:] = False
    return mk.ravel()


class TestTensorPlacement:

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_grad_then_div_is_the_strain_gram(self, shell, m):
        """Div(Grad v + Grad v^T) through the placement is the strain Gram
        with the deviatoric map replaced by 2I: the weighted Gram plus its
        trace part (2/3) W_v^-1 S^T W_T Tr^T Tr S, to rounding."""
        sg = StaggeredGrids.from_grid(grid_for(shell, 13, 16))
        ops = _CGridOps(sg, m, 2)
        M, place = ops.tensor_map(1.0, 1.0, "stress_free")
        assert place == "tensor"
        DG = ops.div_tensor(place, "stress_free") @ M
        S, WT, inv_Wv, _ = _strain_operator_and_norms(sg, m, 2, "stress_free")
        F = build_staggered_weighted_viscous_gram(
            sg, m, lambda rr, tt: np.ones_like(rr * tt), 2, wall_bc="stress_free")
        Tr = ops.trace("tensor")
        trace_part = -(sp.diags(inv_Wv) @ S.conj().T @ sp.diags(WT)
                       @ ((2.0 / 3.0) * (Tr.conj().T @ Tr)) @ S)
        assert maxdiff(DG, F + trace_part) <= 1e-14 * abs(F).max()

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_div_of_nonsymmetric_gradient_is_lap_plus_grad_div(self, shell, m):
        """Div(a Grad(v) + b Transpose(Grad(v))) with a != b is built from
        the vector identities, a Lap(v) + b Grad(Div(v)) on the face blocks,
        entrywise: Div(Transpose(Grad(v))) is the Grad(Div(v)) pencil and
        Div(2 Grad(v) - Transpose(Grad(v))) is 2 Lap(v) - Grad(Div(v)). The
        nine-component map is not divided (its theta-phi and phi-theta
        components carry finite pole limits that only their sum closes)."""
        def pencil(term):
            pb = Problem()
            pb.add_unknown("v", "vector")
            pb.add_equation(f"dt(v) - ({term}) = 0", name="momentum")
            for at in ("outer",) if not shell else ("inner", "outer"):
                for unit in ("r_hat", "theta_hat", "phi_hat"):
                    pb.add_bc(f"dot({unit}, v) = 0", at=at)
            return compile_staggered_general(pb.build(), grid_for(shell, 9, 12), m).A_total
        A_t = pencil("div(trans(grad(v)))")
        A_gd = pencil("grad(div(v))")
        assert maxdiff(A_t, A_gd) <= 1e-13 * abs(A_gd).max()
        A_mix = pencil("div(2*grad(v) - trans(grad(v)))")
        A_lap = pencil("lap(v)")
        assert maxdiff(A_mix, 2 * A_lap - A_gd) <= 1e-13 * abs(A_lap).max()

    def test_transpose_is_a_permutation(self):
        sg = StaggeredGrids.from_grid(grid_for(True, 9, 12))
        ops = _CGridOps(sg, 1, 2)
        P = ops.transpose("tensor9")
        assert set(np.unique(P.data).tolist()) == {1.0 + 0j}
        assert maxdiff(P @ P, sp.eye(P.shape[0], format="csr", dtype=complex)) == 0.0
        sl = ops.component_slices("tensor9")
        for a, b in (("rt", "tr"), ("rp", "pr"), ("tp", "pt")):
            (la, sa), (lb, sb) = sl[a], sl[b]
            assert la == lb
            x = np.zeros(P.shape[0], dtype=complex)
            x[sa] = np.arange(sa.stop - sa.start)
            assert np.array_equal((P @ x)[sb], x[sa])
        assert maxdiff(ops.transpose("tensor"),
                       sp.eye(ops.transpose("tensor").shape[0], format="csr")) == 0.0

    @pytest.mark.parametrize("m", [0, 2])
    def test_trace_of_grad_is_div_to_second_order(self, m):
        """Trace of the placement's gradient (nine components, and the
        strain's three diagonals) equals the norm-adjoint divergence on the
        centres to second order: the two are different consistent stencils
        of one identity (the flux form squares the edge radii, the strain
        averages v_r / r), so they agree to O(h^2), not to rounding."""
        errs9, errs6 = [], []
        for N in (12, 24, 48):
            sg = StaggeredGrids.from_grid(make_shell_grid(N_r=N + 1, N_theta=N, r_inner=0.4))
            ops = _CGridOps(sg, m, 2)
            v = smooth_face_field(sg)
            G9, _ = ops.grad9()
            M, _ = ops.tensor_map(1.0, 1.0, "stress_free")
            dv = ops.divergence(None) @ v
            mk = deep_interior(len(sg.r_center), len(sg.theta_center))
            ref = np.abs(dv[mk]).max()
            errs9.append(np.abs(ops.trace("tensor9") @ G9 @ v - dv)[mk].max() / ref)
            errs6.append(np.abs(ops.trace("tensor") @ (M / 2.0) @ v - dv)[mk].max() / ref)
        for errs in (errs9, errs6):
            assert errs[0] < 5e-3
            assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0, errs

    @pytest.mark.parametrize("m", [0, 2])
    def test_dot_er_matches_hand_gradlnrho_blocks_to_second_order(self, m):
        """Dot(r_hat, Grad v + Grad v^T) through the Hodge maps agrees with
        ``hand_gradlnrho_strain`` (centred same-grid
        derivatives, row-normalised averages) to second order on smooth
        fields; entrywise the stencils differ."""
        errs = []
        for N in (12, 24, 48):
            sg = StaggeredGrids.from_grid(make_shell_grid(N_r=N + 1, N_theta=N, r_inner=0.4))
            ops = _CGridOps(sg, m, 2)
            v = smooth_face_field(sg)
            M, _ = ops.tensor_map(1.0, 1.0, "stress_free")
            hand = hand_gradlnrho_strain(sg, m, lambda rr, tt: 1.0 + 0 * rr * tt, 2)
            Hand = sp.bmat([[hand[("r", "r")], None, None],
                            [hand[("t", "r")], hand[("t", "t")], None],
                            [hand[("p", "r")], None, hand[("p", "p")]]], format="csr")
            Dot_er = ops.dot_radial("tensor", "left") @ M
            mask = np.concatenate([
                deep_interior(len(sg.r_edge), len(sg.theta_center)),
                deep_interior(len(sg.r_center), len(sg.theta_edge)),
                deep_interior(len(sg.r_center), len(sg.theta_center))])
            hv = Hand @ v
            errs.append(np.abs(Dot_er @ v - hv)[mask].max() / np.abs(hv[mask]).max())
        assert errs[0] < 5e-3
        assert errs[0] / errs[1] > 3.0 and errs[1] / errs[2] > 3.0, errs

    def test_dot_bg_tensor_vec_symmetric_is_self_adjoint(self):
        """Dot(T0, v) with a symmetric background tensor is W-self-adjoint
        on the weightful DOFs (the off-diagonal pairs are Hodge pairs)."""
        pb = _base_problem()
        pb.add_background("T0", components={("r", "r"): "r", ("theta", "theta"): "1+r",
                                            ("phi", "phi"): "2", ("r", "theta"): "r*sin(theta)",
                                            ("theta", "r"): "r*sin(theta)",
                                            ("r", "phi"): "cos(theta)", ("phi", "r"): "cos(theta)",
                                            ("theta", "phi"): "r", ("phi", "theta"): "r"},
                          symmetric=True)
        pb.add_equation("dt(v) + grad(p) + dot(T0, v) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        _add_walls(pb)
        grid = grid_for(True)
        cp = compile_staggered_general(pb.build(), grid, 2)
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, 2, 2)
        W = ops.face_norm()
        Nt, Nr_e = len(sg.theta_center), len(sg.r_edge)
        keep = np.abs(W) > 0
        keep[:Nt] = False
        keep[(Nr_e - 1) * Nt:] = False
        n_v = cp.layout.field("p").offset
        C = cp.A_total.tocsr()[:n_v, :n_v][keep][:, keep].toarray()
        H = np.diag(W[keep]) @ C
        assert np.abs(H - H.conj().T).max() <= 1e-14 * np.abs(H).max()

    def test_dot_bg_grad_vec_radial_advection(self):
        """Dot(U0, Grad(v)) with a radial U0 is U0_r d_r v on the faces:
        second-order consistent on a smooth field."""
        errs = []
        for N in (12, 24):
            pb = _base_problem()
            pb.add_background("U0", v_r="r")
            pb.add_equation("dt(v) + grad(p) + dot(U0, grad(v)) = 0", name="momentum")
            pb.add_equation("div(v) = 0", name="continuity")
            _add_walls(pb)
            grid = make_shell_grid(N_r=N + 1, N_theta=N, r_inner=0.4)
            cp = compile_staggered_general(pb.build(), grid, 2)
            sg = StaggeredGrids.from_grid(grid)
            v = smooth_face_field(sg)
            n_v = cp.layout.field("p").offset
            got = cp.A_total.tocsr()[:n_v, :n_v] @ v
            re_, rc, te, tc = sg.r_edge, sg.r_center, sg.theta_edge, sg.theta_center
            want = np.concatenate([
                (re_[:, None] * 2 * re_[:, None] * np.sin(tc[None, :])**2).ravel(),
                (rc[:, None] * 2 * rc[:, None] * np.sin(te[None, :]) * np.cos(te[None, :])).ravel(),
                (rc[:, None] * 2 * rc[:, None] * np.sin(tc[None, :])).ravel()])
            mask = np.concatenate([deep_interior(len(re_), len(tc)),
                                   deep_interior(len(rc), len(te)),
                                   deep_interior(len(rc), len(tc))])
            errs.append(np.abs(got - want)[mask].max() / np.abs(want[mask]).max())
        assert errs[0] < 1e-2 and errs[0] / errs[1] > 3.0, errs

    def test_laph_scalar(self):
        """LapH on the centres: the angular part of the scalar Laplacian,
        W-self-adjoint, and -m^2 / (r^2 sin^2 theta) on a radial field."""
        pb = Problem()
        pb.add_unknown("s", "scalar")
        pb.add_equation("dt(s) - laph(s) = 0", name="heat")
        pb.add_bc("s = 0", at="inner")
        pb.add_bc("s = 0", at="outer")
        grid = grid_for(True)
        m = 2
        cp = compile_staggered_general(pb.build(), grid, m)
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, m, 2)
        L = build_staggered_angular_laplacian(sg, m, order=2)
        assert maxdiff(cp.A_total, -L) == 0.0
        WL = (sp.diags(ops.W["CC"]) @ L).toarray()
        assert np.abs(WL - WL.conj().T).max() <= 1e-14 * np.abs(WL).max()
        R, T = ops.mesh("CC")
        f = (R**2).ravel().astype(complex)
        assert np.allclose(L @ f, -(m**2) / (R**2 * np.sin(T)**2).ravel() * f)


# ---------------------------------------------------------------------------
# Stage 3: tensor unknowns and the strain-divergence viscous form
# ---------------------------------------------------------------------------

def viscous_problem(viscous, Ek=1e-2, dt_T=False):
    """dt(v) + 2 ez x v + grad(p) - Ek (viscous) = 0, div(v) = 0 with
    stress-free walls; a tensor unknown T when the viscous term uses it,
    defined by T - grad(v) - trans(grad(v)) = 0 (under Dt with ``dt_T``)."""
    pb = _base_problem()
    if "T" in viscous:
        pb.add_unknown("T", "tensor")
    pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_equation(f"dt(v) + 2*cross(ez, v) + grad(p) - Ek*({viscous}) = 0",
                    name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    if "T" in viscous:
        pb.add_equation(("dt(T) + " if dt_T else "") + "T - grad(v) - trans(grad(v)) = 0",
                        name="def")
    pb.add_stress_free("v", at="inner")
    pb.add_stress_free("v", at="outer")
    return pb


def spectrum_near(cp, sigma, radius=0.6):
    res = dense(cp.A_total, cp.B_total, sigma=sigma, backend="auto", vectors=True)
    lam = res.lam[res.resid < 1e-8]
    return np.sort_complex(lam[np.abs(lam - sigma) < radius])


class TestTensorUnknown:

    def test_definition_is_eliminated(self):
        """T - grad(v) - trans(grad(v)) = 0 classifies as a definition, T
        gets no DOFs, and the eliminated pencil is entrywise the
        strain-divergence pencil, with the same spectrum to rounding."""
        grid = grid_for(True, 13, 16)
        m = 2
        cpT = compile_staggered_general(viscous_problem("div(T)").build(), grid, m)
        cpS = compile_staggered_general(
            viscous_problem("div(grad(v) + trans(grad(v)))").build(), grid, m)
        rec = cpT.constraints
        assert rec["equations"]["def"]["class"] == "definition"
        assert rec["definitions"] == {"T": "def"}
        assert rec["placement"]["T"] == "tensor"
        assert cpT.layout.placements == {"v": "face", "p": "center"}
        assert rec["viscous"]["v"]
        assert maxdiff(cpT.A_total, cpS.A_total) <= 1e-14 * abs(cpS.A_total).max()
        assert maxdiff(cpT.B_total, cpS.B_total) == 0.0
        sT, sS = spectrum_near(cpT, -0.05 + 0.5j), spectrum_near(cpS, -0.05 + 0.5j)
        assert len(sT) == len(sS) >= 2
        assert np.abs(sT - sS).max() <= 1e-12

    def test_tensor_under_dt_keeps_its_dofs(self):
        """With Dt(T) the tensor stays in the state on the six symmetric
        blocks; its rows carry I on T and -2 S on v, and B is the identity
        on the T blocks."""
        grid = grid_for(True, 9, 12)
        m = 1
        cp = compile_staggered_general(viscous_problem("div(T)", dt_T=True).build(), grid, m)
        assert cp.layout.placements == {"v": "face", "T": "tensor", "p": "center"}
        assert cp.constraints["equations"]["def"]["class"] == "evolution"
        names = [f.name for f in cp.layout.fields]
        assert names[3:9] == ["T_rr", "T_tt", "T_pp", "T_rt", "T_rp", "T_tp"]
        assert [f.location for f in cp.layout.fields[3:9]] == [
            "center", "center", "center", "corner", "r_edge", "theta_edge"]
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, m, 2)
        full = _layout_for(viscous_problem("div(T)", dt_T=True).build(), cp.constraints, ops)
        M, _ = ops.tensor_map(1.0, 1.0, "stress_free")
        n_v, n_T = M.shape[1], M.shape[0]
        N = full.total_size
        X = sp.lil_matrix((N, N), dtype=complex)
        X[n_v:n_v + n_T, :n_v] = -M
        X[n_v:n_v + n_T, n_v:n_v + n_T] = sp.eye(n_T)
        Xr, _, _ = _reduce_velocity_dofs(X.tocsr(), X.tocsr(), full, sg, m, "v")
        t0 = cp.layout.field("T_rr").offset
        t1 = cp.layout.field("T_tp").offset + cp.layout.field("T_tp").size
        A = cp.A_total.tocsr()
        assert maxdiff(A[t0:t1, :], Xr.tocsr()[t0:t1, :]) == 0.0
        assert maxdiff(cp.B_total.tocsr()[t0:t1, t0:t1], sp.eye(t1 - t0, format="csr")) == 0.0

    def test_unsymmetrised_definition_uses_nine_components(self):
        pb = _base_problem()
        pb.add_unknown("T", "tensor")
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - 0.01*div(T) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_equation("dt(T) + T - grad(v) = 0", name="def")
        pb.add_stress_free("v", at="inner")
        pb.add_stress_free("v", at="outer")
        cp = compile_staggered_general(pb.build(), grid_for(True, 9, 12), 1)
        assert cp.layout.placements["T"] == "tensor9"
        assert [f.name for f in cp.layout.fields if f.name.startswith("T_")] == [
            "T_rr", "T_tt", "T_pp", "T_rt", "T_rp", "T_tp", "T_tr", "T_pr", "T_pt"]


class TestStrainDivergenceForm:

    @pytest.mark.parametrize("m", [0, 2])
    def test_identity_with_lap_plus_grad_div(self, m):
        """With the viscous term written as div(grad(v) + trans(grad(v)))
        the pencil differs from the lap(v) + grad(div(v)) one by
        (2/3) Ek (G Div - K_tr) on the velocity block, K_tr =
        -W_v^-1 S^T W_T Tr^T Tr S the trace Gram: G Div v vanishes exactly
        on the discrete constraint manifold and Tr S v = O(h^2) there, so
        the two pencils agree on the manifold to second order."""
        grid = grid_for(True, 13, 16)
        pb_S = boussinesq_problem(viscous="div(grad(v) + trans(grad(v)))")
        pb_L = boussinesq_problem(viscous="lap(v) + grad(div(v))")
        cpS = compile_staggered_general(pb_S.build(), grid, m, params=pb_S.params)
        cpL = compile_staggered_general(pb_L.build(), grid, m, params=pb_L.params)
        sg = StaggeredGrids.from_grid(grid)
        ops = _CGridOps(sg, m, 2)
        S, WT, inv_Wv, _ = _strain_operator_and_norms(sg, m, 2, "stress_free")
        Tr = ops.trace("tensor")
        K_tr = -(sp.diags(inv_Wv) @ S.conj().T @ sp.diags(WT) @ (Tr.conj().T @ Tr) @ S)
        GD = build_staggered_grad_div(sg, m, 2)
        X = 1e-2 * (2.0 / 3.0) * (GD - K_tr)
        full = _layout_for(pb_L.build(), cpL.constraints, ops)
        n_v, N = S.shape[1], full.total_size
        Xf = sp.bmat([[X, None], [None, sp.csr_matrix((N - n_v, N - n_v))]], format="csr")
        Xr, _, _ = _reduce_velocity_dofs(Xf, Xf, full, sg, m, "v")
        D = cpS.A_total - cpL.A_total
        assert maxdiff(D, Xr) <= 1e-13 * abs(cpS.A_total).max()
        n_v_red = cpL.layout.field("T").offset
        Div_red = cpL.A_total.tocsr()[cpL.layout.field("p").slice, :n_v_red]
        GD_red = Xr.tocsr()[:n_v_red, :n_v_red] * 0 + 1e-2 * (2.0 / 3.0) * \
            _reduce_velocity_dofs(
                sp.bmat([[GD, None], [None, sp.csr_matrix((N - n_v, N - n_v))]], format="csr"),
                Xf, full, sg, m, "v")[0].tocsr()[:n_v_red, :n_v_red]
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n_v_red) + 1j * rng.standard_normal(n_v_red)
        z, *_ = np.linalg.lstsq(Div_red.toarray(), Div_red @ y, rcond=None)
        w = y - z
        assert np.abs(Div_red @ w).max() <= 1e-9 * np.abs(w).max()
        assert np.abs(GD_red @ w).max() <= 1e-9 * abs(GD_red).max() * np.abs(w).max()

    def test_spectra_agree_to_second_order(self):
        errs = []
        for N in (12, 24):
            grid = make_shell_grid(N_r=N + 1, N_theta=N, r_inner=0.4)
            a = compile_staggered_general(
                viscous_problem("div(grad(v) + trans(grad(v)))").build(), grid, 2)
            b = compile_staggered_general(
                viscous_problem("lap(v) + grad(div(v))").build(), grid, 2)
            sa, sb = spectrum_near(a, -0.05 + 0.5j), spectrum_near(b, -0.05 + 0.5j)
            k = min(len(sa), len(sb), 2)
            assert k == 2
            errs.append(np.abs(sa[:k] - sb[:k]).max())
        assert errs[0] < 1e-5 and errs[0] / errs[1] > 3.0, errs


# ---------------------------------------------------------------------------
# Stage 4: constrained transport and the vector potential
# ---------------------------------------------------------------------------

CT_LE, CT_ETA = 2.0e-3, 2.0e-4


# ---------------------------------------------------------------------------
# Background advection, the nine-component divergence and the pole closure
# ---------------------------------------------------------------------------

def advection_problem(term, shell, wall="dot(r_hat, v) = 0"):
    """dt(v) + term + grad(p) = 0, div(v) = 0 with the solid-body flow
    U0 = s phi_hat and the axis ez."""
    pb = _base_problem()
    pb.add_background("U0", v_r=0, v_theta=0, v_phi="r*sin(theta)")
    pb.add_equation(f"dt(v) + {term} + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    for at in ("inner", "outer") if shell else ("outer",):
        pb.add_bc(wall, at=at)
    return pb.build()


def manufactured_solve(pb, geometry, m, n, source, pole_closure=False):
    """Max-norm relative error of the C-grid solution of the evolution
    form of ``pb`` driven by ``source`` (the exact image of the
    manufactured vector), against the manufactured vector."""
    u = manufactured(geometry, m, "vector")
    cm = compile_staggered_general(pb.build(), _fd_grid(geometry, n, 2), m,
                                   pole_closure=pole_closure)
    x, live = _stag_solve(cm, _stag_state(cm, {"u": source}, geometry, m))
    err = scale = 0.0
    for f in cm.layout.fields:
        sl = slice(f.offset, f.offset + f.size)
        exact = _stag_values(u, f, cm.grids, geometry, m)
        scale = max(scale, float(np.abs(exact).max()))
        err = max(err, _err(x[sl][live[sl]], exact[live[sl]]))
    return err / scale


def order_of(errs, ns):
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i]))
            for i in range(len(errs) - 1)]


class TestBackgroundAdvection:

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_solid_body_advection_is_doppler_plus_coriolis(self, shell, m):
        """Dot(s phi_hat, Grad(v)) = i m v + z_hat x v entrywise: the
        phi-row of the nine-component map is built from the Coriolis
        pairs, so the assembled advection block is the Coriolis block plus
        i m on the mass, to rounding (the Doppler shift of a rotating
        frame is then exact). On a ball the origin anchor row (m = 0,
        |m| = 1) is the exception: the contraction's r-row is the guarded
        zero there, where s phi_hat vanishes, while the Coriolis pair
        copies the first centre; the ring is receive-only, so that row is
        decoupled from the spectrum."""
        grid = grid_for(shell, 9, 12)
        adv = compile_staggered_general(advection_problem("dot(U0, grad(v))", shell), grid, m)
        cor = compile_staggered_general(advection_problem("cross(ez, v)", shell), grid, m)
        assert adv.layout.fields == cor.layout.fields
        D = (adv.A_total - cor.A_total - 1j * m * adv.B_total).tocsr()
        if not shell and abs(m) <= 1:
            assert abs(D[1:, :]).max() <= 1e-13 * abs(adv.A_total).max()
            assert abs(D[0, :]).max() > 0.1
        else:
            assert maxdiff(D, sp.csr_matrix(D.shape, dtype=complex)) <= 1e-13 * abs(adv.A_total).max()

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("geometry", ["shell", "ball"])
    def test_theta_dependent_background_is_second_order(self, geometry, m):
        """Dot(B0, Grad(u)) and Dot(B0, Grad(u) + Transpose(Grad(u))) with
        B0(r, theta) of all three components, as manufactured
        boundary-value problems with the Laplacian carrier, converge at
        second order on shell and ball."""
        sn = smp.sin(theta)
        B0 = vc.Vec(smp.cos(theta) * (1 + r ** 2 / 4) + r / 4, -sn * (1 + r ** 2 / 4), r * sn / 2)
        for probe in ("grad(u)", "grad(u) + trans(grad(u))"):
            errs = []
            for n in (12, 24):
                u = manufactured(geometry, m, "vector")
                img = vc.dot(B0, eval(probe, {"grad": lambda x: vc.grad(x, m),
                                              "trans": vc.transpose, "u": u}))
                pb = Problem()
                pb.add_unknown("u", "vector")
                pb.add_background("B0", v_r=B0[0], v_theta=B0[1], v_phi=B0[2])
                pb.add_equation(f"dt(u) - lap(u) - dot(B0, {probe}) = 0", name="carrier")
                _add_bcs(pb, "u", "vector", geometry)
                lap = vc.lap(u, m)
                errs.append(manufactured_solve(
                    pb, geometry, m, n, vc.Vec(lap[i] + img[i] for i in vc.AXES)))
            assert errs[0] < 5e-2 and order_of(errs, (12, 24))[0] > 1.8, (probe, errs)


class TestNineComponentDivergence:

    @pytest.mark.parametrize("m", [0, 1, 2])
    @pytest.mark.parametrize("geometry", ["shell", "ball"])
    def test_div_of_tensor_unknown_is_second_order(self, geometry, m):
        """dt(u) - div(T) = 0 with the non-symmetric definition T - grad(u)
        = 0 (nine components, no-slip walls) driven by Lap(u_manufactured)
        converges at second order: the pole rows of the map carry the
        parity limits of the theta-phi and phi-theta components and the
        divergence reads the pole share of the phi-theta one, and the wall
        rows carry the no-slip closure of the strain."""
        errs = []
        for n in (12, 24):
            pb = Problem()
            pb.add_unknown("u", "vector")
            pb.add_unknown("T", "tensor")
            pb.add_equation("dt(u) - div(T) = 0", name="carrier")
            pb.add_equation("T - grad(u) = 0", name="def")
            _add_bcs(pb, "u", "vector", geometry)
            errs.append(manufactured_solve(pb, geometry, m, n,
                                           vc.lap(manufactured(geometry, m, "vector"), m)))
        assert errs[0] < 5e-2 and order_of(errs, (12, 24))[0] > 1.8, errs

    def test_symmetric_placement_is_unchanged(self):
        """The six-component path (strain, norms, divergence) does not
        depend on the nine-component map: the strain Gram of the
        placement is the library's, entrywise, with the closure off."""
        sg = StaggeredGrids.from_grid(grid_for(True, 9, 12))
        for m in (0, 2):
            ops = _CGridOps(sg, m, 2)
            S, WT, inv_Wv, _ = _strain_operator_and_norms(sg, m, 2, "noslip")
            S2, WT2, inv_Wv2 = ops.strain("noslip")
            assert maxdiff(S, S2) == 0.0 and np.array_equal(WT, WT2)
            assert np.array_equal(inv_Wv, inv_Wv2)
            D = ops.div_tensor("tensor", "noslip")
            assert maxdiff(D, -(sp.diags(inv_Wv) @ S.conj().T @ sp.diags(WT))) == 0.0


class TestPoleClosure:

    @pytest.mark.parametrize("shell", [True, False], ids=["shell", "ball"])
    def test_closure_is_inert_at_m_0_and_1(self, shell):
        """The strain's pole rows vanish at m = 0 (no i m term) and |m| = 1
        (even tangential parity), so the closed Gram is the certified one
        there, entrywise."""
        sg = StaggeredGrids.from_grid(grid_for(shell, 9, 12))
        one = lambda rr, tt: np.ones_like(rr * tt)  # noqa: E731
        for m in (0, 1):
            F0 = build_staggered_weighted_viscous_gram(sg, m, one, 2, wall_bc="noslip")
            F1 = build_staggered_weighted_viscous_gram(sg, m, one, 2, wall_bc="noslip",
                                                       pole_closure=True)
            assert maxdiff(F0, F1) == 0.0

    def test_closure_touches_the_pole_adjacent_vphi_rows_only(self):
        """At |m| = 2 the closure adds, on the v_phi rows adjacent to the
        poles, the pole share of sigma_theta_phi read from the adjacent
        v_theta edge, and nothing else."""
        sg = StaggeredGrids.from_grid(grid_for(True, 9, 12))
        one = lambda rr, tt: np.ones_like(rr * tt)  # noqa: E731
        F0 = build_staggered_weighted_viscous_gram(sg, 2, one, 2, wall_bc="noslip")
        F1 = build_staggered_weighted_viscous_gram(sg, 2, one, 2, wall_bc="noslip",
                                                   pole_closure=True)
        D = (F1 - F0).tocoo()
        Nr_e, Nt_c, Nt_e = len(sg.r_edge), len(sg.theta_center), len(sg.theta_edge)
        n_vr, n_vt = Nr_e * Nt_c, len(sg.r_center) * Nt_e
        rows = set(D.row[np.abs(D.data) > 0].tolist())
        cols = set(D.col[np.abs(D.data) > 0].tolist())
        assert rows == {n_vr + n_vt + ir * Nt_c + j
                        for ir in range(len(sg.r_center)) for j in (0, Nt_c - 1)}
        assert cols <= {n_vr + ir * Nt_e + j
                        for ir in range(len(sg.r_center)) for j in (1, Nt_e - 2)}
        assert maxdiff(D, sp.csr_matrix(D.shape)) > 0

    @pytest.mark.slow
    @pytest.mark.parametrize("geometry", ["shell", "ball"])
    def test_lap_vector_at_m_2_stays_second_order(self, geometry):
        """Lap(vector) at |m| = 2 drops to first order between N = 24 and
        48 without the closure (the pole-adjacent v_phi ring is O(1 /
        dtheta) inconsistent) and stays at second order with it."""
        m, ns = 2, (24, 48)
        pb = Problem()
        pb.add_unknown("u", "vector")
        pb.add_equation("dt(u) - lap(u) = 0", name="carrier")
        _add_bcs(pb, "u", "vector", geometry)
        lap = vc.lap(manufactured(geometry, m, "vector"), m)
        closed = [manufactured_solve(pb, geometry, m, n, lap, pole_closure=True) for n in ns]
        opened = [manufactured_solve(pb, geometry, m, n, lap) for n in ns]
        assert order_of(closed, ns)[0] > 1.8, closed
        assert order_of(opened, ns)[0] < 1.5, opened


def axial_field():
    """The uniform axial field z_hat, regular at the origin."""
    return Background.Vector("B0_axial", v_r=smp.cos(theta), v_theta=-smp.sin(theta),
                             v_phi=0)


def dipole_field():
    """Unit-surface potential dipole (r_o / r)^3 (2 cos, sin, 0)."""
    f = (RMAX / r) ** 3
    return Background.Vector("B0_dipole", v_r=2 * smp.cos(theta) * f,
                             v_theta=smp.sin(theta) * f, v_phi=0)


def ct_system(Ek=2e-4, bc="conducting", pressure="grouped", formulation="b",
              delta=1e-6, Le=CT_LE):
    """The solar-eigenproblem hydro system of ``solar_named_system`` plus the dipole
    magnetic layer of ``add_magnetic_field``."""
    from spdee.applications.solar_eigenmodes import add_magnetic_field
    base = solar_named_system("true", Ek, delta, pressure=pressure)
    return add_magnetic_field(base, dipole_field(), Le=Le, eta=CT_ETA, bc=bc,
                              formulation=formulation)


def ct_grid(Nr, Nt):
    return make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=RMIN, r_outer=RMAX)


def layout_order(cp):
    return [(f.name, f.location, f.offset, f.size) for f in cp.layout.fields]


CT_CASES = [(12, 48, 8, 2e-4, "conducting"), (8, 16, 8, 2e-4, "conducting"),
            (8, 16, 1, 2e-4, "conducting"), (8, 16, 8, 0.0, "conducting"),
            (12, 48, 1, 2e-4, "radial"), (8, 16, 8, 2e-4, "radial")]


class TestConstrainedTransport:

    @pytest.mark.parametrize("Nr,Nt,m,Ek,bc", CT_CASES,
                             ids=[f"{c[0]}x{c[1]}-m{c[2]}-Ek{c[3]:g}-{c[4]}"
                                  for c in CT_CASES])
    def test_layout_and_involution_record(self, Nr, Nt, m, Ek, bc):
        """The dipole system compiles -- which asserts ||Div A_b|| = 0 on
        the flux rows before the walls are applied (``_check_div_curl``) --
        with b classified as an involution, placed on the faces and
        appended last, carrying unit mass off the pinned rings (the two
        b_r wall rings and the b_theta pole ring) and a flux divergence of
        the reduced state."""
        grid = ct_grid(Nr, Nt)
        cp = compile_fd2_staggered(ct_system(Ek=Ek, bc=bc), grid, m=m,
                                   order_r=2, order_theta=2)
        assert [f.name for f in cp.layout.fields] == [
            "v_r", "v_t", "v_p", "rho1", "s1", "b_r", "b_t", "b_p"]
        assert cp.constraints["constraints"] == {"b": "involution"}
        assert cp.constraints["placement"]["b"] == "face"
        assert np.all(np.isfinite(cp.A_total.tocoo().data))
        rows = slice(cp.layout.field("b_r").offset, cp.layout.total_size)
        B = cp.B_total.tocsr()[rows, rows]
        n_b = cp.layout.total_size - cp.layout.field("b_r").offset
        assert (B - sp.diags(B.diagonal())).nnz == 0
        assert np.count_nonzero(B.diagonal()) >= n_b - 2 * Nt - 2 * Nr
        D = cp.constraints["divergence"]["b"]
        assert D.shape[1] == cp.layout.total_size

    def test_pressure_spelling_changes_only_the_two_pressure_blocks(self):
        """Spelling the pressure gradient by the product rule instead of
        Grad(c q) changes the (v, rho1) and (v, s1) blocks and nothing
        else: constrained transport is untouched by it."""
        grid = ct_grid(8, 16)
        a = compile_fd2_staggered(ct_system(pressure="grouped"), grid, m=8)
        b = compile_fd2_staggered(ct_system(pressure="product"), grid, m=8)
        assert layout_order(a) == layout_order(b)
        dev = block_deviations(a.A_total.tocsr(), a.layout, b.A_total.tocsr(), b.layout)
        assert dev[("v_r", "rho1")][0] > 1.0 and dev[("v_r", "s1")][0] > 1.0
        for key, (d, _) in dev.items():
            if key[1] not in ("rho1", "s1") or key[0] not in ("v_r", "v_t", "v_p"):
                assert d == 0.0, key

    def test_pinned_wall_refused(self):
        system = ct_system(bc="pinned")
        with pytest.raises(NotImplementedError, match="pins its tangential components"):
            compile_fd2_staggered(system, ct_grid(8, 16), m=8)

    def test_the_involution_is_read_from_the_tasks(self):
        system = ct_system()
        assert general_path_reason(system, 8) is None
        cp = compile_fd2_staggered(system, ct_grid(8, 16), m=8)
        assert cp.constraints["involutions"] == {"b": "induction"}


class TestInvolution:

    def test_classification_and_layout(self):
        """``add_magnetic_field`` writes no constraint: the involution is
        read off the curl evolution, b keeps the face placement and its
        flux block is appended last, as the CT extension places it."""
        system = ct_system()
        record = classify_constraints(system, walk_equation_system(system, 8))
        assert record["constraints"] == {"b": "involution"}
        assert record["involutions"] == {"b": "induction"}
        assert record["placement"]["b"] == "face"
        assert set(record["principal"]) == {"momentum", "continuity", "energy",
                                            "induction"}
        ops = _CGridOps(StaggeredGrids.from_grid(ct_grid(8, 16)), 8, 2)
        names = [f.name for f in _layout_for(system, record, ops).fields]
        assert names == ["v_r", "v_t", "v_p", "rho1", "s1", "b_r", "b_t", "b_p"]

    def test_written_constraint_drops_its_row(self):
        """A written Div(b) = 0 with no multiplier classifies as an
        involution, its row is dropped and b gets no extra DOFs."""
        system = TestCTRefusals._mhd_problem()
        record = classify_constraints(system, walk_equation_system(system, 1))
        assert record["equations"]["solenoidal"] == {
            "class": "involution", "vector": "b", "evolution": "induction",
            "dropped": True}
        assert "solenoidal" not in record["principal"]
        assert record["constraints"] == {"v": "saddle_point", "b": "involution"}
        cp = compile_staggered_general(system, GRID, 1)
        assert [f.name for f in cp.layout.fields] == [
            "v_r", "v_t", "v_p", "p", "b_r", "b_t", "b_p"]

    @pytest.mark.parametrize("geometry,m", [("shell", 8), ("ball", 0), ("ball", 1),
                                            ("ball", 2), ("ball", 8)])
    def test_div_curl_is_exact(self, geometry, m):
        """Every flux tendency the CT builders produce is a discrete curl of
        edge circulations, so the flux divergence annihilates it exactly
        (the assembler asserts the same on the compiled block); on a ball
        the origin cell's degenerate edges carry zero circulation and the
        identity holds there without a closure value."""
        grid = ct_grid(8, 16) if geometry == "shell" else make_ball_grid(N_r=9, N_theta=16)
        B0 = dipole_field() if geometry == "shell" else axial_field()
        ops = _CGridOps(StaggeredGrids.from_grid(grid), m, 2)
        D = ops.flux_divergence()
        rows = cols = ("br", "bt", "bp")
        for blocks in (ct_induction_blocks(ops.ct(), B0, -1.0, True),
                       ct_induction_blocks(ops.ct(), B0, -1.0, False),
                       ct_resistive_blocks(ops.ct(), CT_ETA, True),
                       ct_resistive_blocks(ops.ct(), CT_ETA, False)):
            src = ("vr", "vt", "vp") if ("br", "vr") in blocks else cols
            M = ops.ct_matrix(blocks, rows, src, ops.ct_sizes())
            X = D @ M
            assert (abs(X).max() if X.nnz else 0.0) <= 1e-14 * abs(M).max()
            if geometry == "ball":
                ring = M[:ops.ct().Nt, :]
                assert ring.nnz == 0 or abs(ring).max() == 0.0

    def test_origin_average_reads_the_parity_reflection(self):
        """On a ball the r-averages onto the first centre (j_phi from the
        corners, j_theta and b_r from the r-edges) read their origin node
        as the parity reflection of the first interior node: the node
        itself at m = 0 and |m| = 1 (a field finite at the origin), half
        of it at |m| = 2 (a field vanishing there, the guard's zero
        average); the shell averages are the plain two-point ones and the
        other rows are untouched."""
        from spdee.backends.fd2.staggered_mhd import CTGrid
        Nr, Nt = 6, 8
        ball = StaggeredGrids.from_grid(make_ball_grid(N_r=Nr + 1, N_theta=Nt))
        shell = StaggeredGrids.from_grid(make_shell_grid(N_r=Nr + 1, N_theta=Nt,
                                                         r_inner=0.5))
        for m, w in ((0, 1.0), (1, 1.0), (2, 0.5), (3, 1.0)):
            ct = CTGrid(ball, m, pole_closure=True)
            first = ct.rf_to_c.tocsr()[:Nt, :].toarray()
            assert np.allclose(first[:, :Nt], 0.0)
            assert np.allclose(first[:, Nt:2 * Nt], w * np.eye(Nt))
            assert np.allclose(ct.rf_to_c.tocsr()[Nt:, :].toarray(),
                               CTGrid(shell, m).rf_to_c.tocsr()[Nt:, :].toarray())
            assert ct.K_to_vth.tocsr()[:Nt - 1, :].sum() == pytest.approx(w * (Nt - 1))
            assert list(ct.origin_ring()) == list(range(Nt))
        ct = CTGrid(shell, 1)
        assert np.allclose(ct.rf_to_c.tocsr()[:Nt, :2 * Nt].toarray(),
                           0.5 * np.hstack([np.eye(Nt), np.eye(Nt)]))
        assert ct.origin_ring().size == 0

    @pytest.mark.parametrize("m", [0, 1])
    def test_lorentz_force_of_a_uniform_current_on_the_first_ring(self, m):
        """The origin closure of the first-centre averages: a uniform
        current j (b = j x r / 2, linear, so every edge current is exact)
        in a background that makes F = j x B0 read the origin values
        j_theta(0, theta) (m = 0: j = z_hat, B0 = r r_hat, F_phi = r sin
        theta) or j_phi(0) and j_theta(0) (|m| = 1: j = x_hat, B0 = z_hat,
        F_theta = (i/2) cos theta, F_phi = -1/2 in the e^{i phi} component).
        With the parity closure the first ring is exact where the interior
        is (F_theta) and carries the interior's O(dtheta^2) elsewhere; the
        plain average with the guarded origin node would read half. (The
        wall ring reads the unclosed wall current and is left out.)"""
        from spdee.backends.fd2.staggered_mhd import CTGrid, ct_lorentz_blocks
        Nr, Nt = 8, 16
        sg = StaggeredGrids.from_grid(make_ball_grid(N_r=Nr + 1, N_theta=Nt))
        ops = _CGridOps(sg, m, 2)
        ct = CTGrid(sg, m, pole_closure=True)
        if m == 0:
            B0 = Background.Vector("B0", v_r=r, v_theta=0, v_phi=0)
            b_th = 0.0 * ct.rcT
            b_ph = 0.5 * ct.rcC * np.sin(ct.tcC)
            exact_th, exact_ph = None, ct.rcC * np.sin(ct.tcC)
        else:
            B0 = axial_field()
            b_th = 0.25j * ct.rcT
            b_ph = -0.25 * ct.rcC * np.cos(ct.tcC)
            exact_th, exact_ph = 0.5j * np.cos(ct.TfT), -0.5 * np.ones_like(ct.rcC)
        Phi = np.concatenate([np.zeros(ct.n_br), (ct.Area_th * b_th).ravel(),
                              (ct.Area_ph * b_ph).ravel()])
        pre = (np.ones((Nr - 1) * Nt), np.ones(Nr * (Nt - 1)), np.ones(Nr * Nt))
        M = ops.ct_matrix(ct_lorentz_blocks(ct, B0, pre), ("vr", "vt", "vp"),
                          ("br", "bt", "bp"), ops.ct_sizes())
        F = M @ Phi
        F_ph = F[ct.n_br + ct.n_bth:].reshape(Nr, Nt)
        ratio = (F_ph / exact_ph).real
        assert abs(ratio[1:-1] - 1.0).max() < 1e-2
        assert abs(ratio[0] - 1.0).max() < 1e-2, ratio[0]
        if m == 1:
            F_th = F[ct.n_br:ct.n_br + ct.n_bth].reshape(Nr, Nt + 1)[:-1, 1:-1]
            assert np.allclose(F_th, exact_th[:-1, 1:-1], atol=1e-13)
        weight = 0.25 * (3 + (-1) ** (max(abs(m) - 1, 0) % 2))
        assert ct.rf_to_c.tocsr()[0, Nt] == weight

    def test_divergence_helper_on_an_eigenvector(self):
        """``StaggeredCompiledMatrices.divergence`` returns the net flux out
        of each cell; an eigenvector with a nonzero eigenvalue is div-free
        because the involution is exact."""
        cp = compile_fd2_staggered(ct_system(), ct_grid(10, 24), m=8)
        K, M = cp.assemble_evp_matrices()
        v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
        lam, V = shift_invert(K, M, 1j * (-2 / 9), 6, return_vectors=True, v0=v0)
        j = int(np.argmin(np.abs(lam - 1j * (-2 / 9))))
        x = V[:, j] / np.linalg.norm(V[:, j])
        flux = np.concatenate([x[cp.layout.field(f"b_{c}").slice] for c in "rtp"])
        assert abs(cp.divergence(x)).max() <= 1e-11 * abs(flux).max()

    def test_lap_b_and_minus_curl_curl_b_are_the_same_pencil(self):
        """On the constraint manifold Lap(b) = -Curl(Curl(b)) (+ Grad(Div b),
        which vanishes there), so the two spellings of the resistive term
        assemble the same blocks; likewise for a vector potential."""
        def decay(spelling, formulation):
            pb = Problem()
            pb.add_unknown("b", "vector")
            term = "- 0.01*lap(b)" if spelling == "lap" else "+ 0.01*curl(curl(b))"
            if formulation == "A":
                pb.add_unknown("psi", "scalar")
                pb.add_equation(f"dt(b) {term} - grad(psi) = 0", name="induction")
                pb.add_equation("div(b) = 0", name="coulomb")
                walls = ("dot(theta_hat, b) = 0", "dot(phi_hat, b) = 0", "psi = 0")
            else:
                pb.add_equation(f"dt(b) {term} = 0", name="induction")
                pb.add_equation("div(b) = 0", name="solenoidal")
                walls = ("dot(r_hat, b) = 0", "dot(theta_hat, curl(b)) = 0",
                         "dot(phi_hat, curl(b)) = 0")
            for w in ("inner", "outer"):
                for bc in walls:
                    pb.add_bc(bc, at=w)
            return compile_staggered_general(pb.build(), GRID, 1)
        for formulation in ("b", "A"):
            a, b = decay("lap", formulation), decay("curl_curl", formulation)
            assert maxdiff(a.A_total, b.A_total) == 0.0
            assert maxdiff(a.B_total, b.B_total) == 0.0

    @staticmethod
    def _lap_problem(constrained):
        """dt(b) = 0.01 Lap(b) with no-slip walls, with or without the
        Coulomb-gauge saddle point Div(b) = 0."""
        pb = Problem()
        pb.add_unknown("b", "vector")
        term = ""
        if constrained:
            pb.add_unknown("psi", "scalar")
            term = " - grad(psi)"
        pb.add_equation(f"dt(b) - 0.01*lap(b){term} = 0", name="induction")
        if constrained:
            pb.add_equation("div(b) = 0", name="coulomb")
        for w in ("inner", "outer"):
            pb.add_bc("dot(r_hat, b) = 0", at=w)
            pb.add_bc("dot(theta_hat, b) = 0", at=w)
            pb.add_bc("dot(phi_hat, b) = 0", at=w)
            if constrained:
                pb.add_bc("psi = 0", at=w)
        return pb.build()

    def test_lap_without_a_divergence_constraint_is_a_plain_face_vector(self):
        """Lap(b) with no Div(b) = 0 is not a refusal: b is an ordinary face
        vector."""
        system = self._lap_problem(False)
        assert general_path_reason(system, 1) is None
        cp = compile_staggered_general(system, GRID, 1)
        assert cp.constraints["constraints"] == {}
        assert cp.constraints["involutions"] == {}
        assert cp.constraints["walls"]["vectors"]["b"]["tangential"] == "noslip"

    def test_the_divergence_constraint_selects_the_gram(self):
        """The strain Gram is Lap + (1/3) Grad Div, so it is Lap only on the
        constraint manifold: a face unknown carrying a divergence constraint
        keeps it (the certified inertial and CT identities live there), and
        one that carries none gets the true vector Laplacian, the Gram less
        (1/3) Grad Div. Both are read off the classification record, and the
        two blocks differ by exactly that operator."""
        free = compile_staggered_general(self._lap_problem(False), GRID, 1)
        held = compile_staggered_general(self._lap_problem(True), GRID, 1)
        assert held.constraints["constraints"] == {"b": "saddle_point"}
        n = free.layout.field("b_p").offset + free.layout.field("b_p").size
        A_free = free.A_total.tocsr()[:n, :n]
        A_held = held.A_total.tocsr()[:n, :n]
        sg = StaggeredGrids.from_grid(GRID)
        GD = build_staggered_grad_div(sg, 1, 2)
        full = _layout_for(self._lap_problem(False), free.constraints, _CGridOps(sg, 1, 2))
        N = full.total_size
        Xf = sp.bmat([[-0.01 / 3.0 * GD, None],
                      [None, sp.csr_matrix((N - GD.shape[0], N - GD.shape[0]))]],
                     format="csr")
        Xr = _reduce_velocity_dofs(Xf, Xf, full, sg, 1, "b")[0].tocsr()[:n, :n]
        assert maxdiff(A_held - A_free, Xr) <= 1e-14 * abs(A_held).max()


class TestCTRefusals:

    @staticmethod
    def _mhd_problem(with_multiplier=False, wall="conducting", ball=False):
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("b", "vector")
        pb.add_unknown("p", "scalar")
        pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        ind = "dt(b) - curl(cross(v, B0)) - 0.01*lap(b)"
        if with_multiplier:
            pb.add_unknown("psi", "scalar")
            ind += " + grad(psi)"
        pb.add_equation(ind + " = 0", name="induction")
        pb.add_equation("div(b) = 0", name="solenoidal")
        for w in (("outer",) if ball else ("inner", "outer")):
            pb.add_bc("dot(r_hat, v) = 0", at=w)
            pb.add_bc("dot(r_hat, b) = 0", at=w)
            if wall == "conducting":
                pb.add_bc("dot(theta_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
                pb.add_bc("dot(phi_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
            elif wall == "pinned":
                pb.add_bc("dot(theta_hat, b) = 0", at=w)
                pb.add_bc("dot(phi_hat, b) = 0", at=w)
        return pb.build()

    def test_both_readings_of_the_divergence_constraint(self):
        with pytest.raises(NotImplementedError) as info:
            compile_staggered_general(self._mhd_problem(with_multiplier=True), GRID, 1)
        msg = str(info.value)
        assert "'solenoidal'" in msg and "both readings apply" in msg
        assert "'psi' enters 'induction' through Grad alone" in msg
        assert "saddle point, psi the multiplier of Div(b)" in msg
        assert "curl_cross_unk_bg(v) (involution, Div Curl = 0 with no multiplier)" in msg

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_involution_on_a_ball_pins_the_origin_ring(self, m):
        """On a ball the origin ring of the face fluxes crosses faces of
        zero area: it is not a DOF at any m. Beside an inviscid velocity
        it carries a Dirichlet row with no mass, as the wall ring does;
        beside a viscous velocity both rings are dropped by the symmetric
        reduction. Every other b row carries unit mass off the pole ring,
        and the wall record is the shell's."""
        Nr, Nt = 6, 8
        ball = make_ball_grid(N_r=Nr + 1, N_theta=Nt)
        cp = compile_staggered_general(self._mhd_problem(ball=True), ball, m)
        assert cp.constraints["constraints"] == {"v": "saddle_point", "b": "involution"}
        assert cp.constraints["walls"]["vectors"]["b"]["kind"] == "conducting"
        f = cp.layout.field("b_r")
        assert f.size == (Nr + 1) * Nt
        A, B = cp.A_total.tocsr(), cp.B_total.tocsr()
        for ring in (slice(f.offset, f.offset + Nt),
                     slice(f.offset + Nr * Nt, f.offset + (Nr + 1) * Nt)):
            assert (A[ring, :] - sp.eye(Nt, A.shape[1], k=ring.start)).nnz == 0
            assert B[ring, :].nnz == 0
        rows = slice(f.offset, cp.layout.total_size)
        Bb = B[rows, rows]
        assert (Bb - sp.diags(Bb.diagonal())).nnz == 0
        assert np.count_nonzero(Bb.diagonal()) == Bb.shape[0] - 2 * Nt - 2 * Nr
        assert cp.constraints["divergence"]["b"].shape[1] == cp.layout.total_size
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("b", "vector")
        pb.add_unknown("p", "scalar")
        pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_equation("dt(v) + grad(p) - 0.01*lap(v) - cross(curl(b), B0) = 0",
                        name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_equation("dt(b) - curl(cross(v, B0)) - 0.01*lap(b) = 0", name="induction")
        for unit in ("r_hat", "theta_hat", "phi_hat"):
            pb.add_bc(f"dot({unit}, v) = 0", at="outer")
        pb.add_bc("dot(r_hat, b) = 0", at="outer")
        cp = compile_staggered_general(pb.build(), ball, m)
        assert cp.constraints["involutions"] == {"b": "induction"}
        assert cp.layout.field("b_r").size == (Nr + 1) * Nt - 2 * Nt
        assert cp.layout.field("v_r").size == (Nr + 1) * Nt - Nt - (Nt if m >= 2 else Nt - 1)

    def test_pinned_tangential_field_on_the_fluxes(self):
        with pytest.raises(NotImplementedError) as info:
            compile_staggered_general(self._mhd_problem(wall="pinned"), GRID, 1)
        msg = str(info.value)
        assert "the tangential condition 'noslip' on the flux unknown 'b'" in msg
        assert "over-determines the poloidal field" in msg
        assert "Dot(t_hat, eta Curl(b) - Cross(v, B0)) = 0" in msg

    def test_self_advected_vector_with_pinned_walls_is_not_a_flux(self):
        """dt(u) - lap(u) - curl(cross(u, B0)) = 0 with no-slip walls: the
        unknown is the velocity of its own induction term, so the no-slip
        closure is its own and it is a plain face unknown (the vector
        Laplacian, the incidence curl divided by the face areas, the wall
        circulations masked), not an over-determined flux; with radial
        walls alone it keeps the involution reading."""
        def problem(noslip, ball=False):
            pb = Problem()
            pb.add_unknown("u", "vector")
            pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
            pb.add_equation("dt(u) - lap(u) - curl(cross(u, B0)) = 0", name="carrier")
            for w in (("outer",) if ball else ("inner", "outer")):
                pb.add_bc("dot(r_hat, u) = 0", at=w)
                if noslip:
                    pb.add_bc("dot(theta_hat, u) = 0", at=w)
                    pb.add_bc("dot(phi_hat, u) = 0", at=w)
            return pb.build()
        cp = compile_staggered_general(problem(True), GRID, 1)
        assert cp.constraints["involutions"] == {} and cp.constraints["constraints"] == {}
        assert cp.constraints["viscous"] == {"u": True}
        cp = compile_staggered_general(problem(False), GRID, 1)
        assert cp.constraints["involutions"] == {"u": "carrier"}
        ball = make_ball_grid(N_r=7, N_theta=8)
        assert compile_staggered_general(problem(True, ball=True), ball, 1).constraints[
            "involutions"] == {}
        cp = compile_staggered_general(problem(False, ball=True), ball, 1)
        assert cp.constraints["involutions"] == {"u": "carrier"}
        assert cp.layout.field("u_r").size == 7 * 8 - 2 * 8

    def test_no_slip_velocity_with_conducting_or_radial_field(self):
        """A no-slip v beside a b with conducting or radial walls: the
        closure stays on v, b is the involution."""
        for wall in ("conducting", "radial"):
            pb = Problem()
            pb.add_unknown("v", "vector")
            pb.add_unknown("b", "vector")
            pb.add_unknown("p", "scalar")
            pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
            pb.add_equation("dt(v) + grad(p) - 0.01*lap(v) - cross(curl(b), B0) = 0",
                            name="momentum")
            pb.add_equation("div(v) = 0", name="continuity")
            pb.add_equation("dt(b) - curl(cross(v, B0)) - 0.01*lap(b) = 0", name="induction")
            for w in ("inner", "outer"):
                for unit in ("r_hat", "theta_hat", "phi_hat"):
                    pb.add_bc(f"dot({unit}, v) = 0", at=w)
                pb.add_bc("dot(r_hat, b) = 0", at=w)
                if wall == "conducting":
                    pb.add_bc("dot(theta_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
                    pb.add_bc("dot(phi_hat, 0.01*curl(b) - cross(v, B0)) = 0", at=w)
            cp = compile_staggered_general(pb.build(), GRID, 1)
            rec = cp.constraints
            assert rec["constraints"] == {"v": "saddle_point", "b": "involution"}
            assert rec["walls"]["vectors"]["v"]["tangential"] == "noslip"
            assert rec["walls"]["vectors"]["b"]["kind"] == (
                "conducting" if wall == "conducting" else "impenetrable")

    def test_electric_field_that_is_not_the_evolution_field(self):
        """A conducting wall whose tangential combination is not the
        electric field of the induction equation is refused."""
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("b", "vector")
        pb.add_unknown("p", "scalar")
        pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
        pb.add_equation("div(v) = 0", name="continuity")
        pb.add_equation("dt(b) - curl(cross(v, B0)) - 0.01*lap(b) = 0", name="induction")
        pb.add_equation("div(b) = 0", name="solenoidal")
        for w in ("inner", "outer"):
            pb.add_bc("dot(r_hat, v) = 0", at=w)
            pb.add_bc("dot(r_hat, b) = 0", at=w)
            pb.add_bc("dot(theta_hat, 0.5*curl(b) - cross(v, B0)) = 0", at=w)
            pb.add_bc("dot(phi_hat, 0.5*curl(b) - cross(v, B0)) = 0", at=w)
        with pytest.raises(NotImplementedError) as info:
            compile_staggered_general(pb.build(), GRID, 1)
        assert "is not the electric field" in str(info.value)

    @staticmethod
    def _free_problem(eta=0.0, ball=True):
        """Ideal MHD on the Malkus field with no wall condition on b."""
        pb = Problem()
        pb.add_unknown("v", "vector")
        pb.add_unknown("b", "vector")
        pb.add_unknown("p", "scalar")
        pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
        pb.add_background("B0", v_r=0, v_theta=0, v_phi="r*sin(theta)")
        pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - 0.01*cross(curl(b), B0) = 0",
                        name="momentum")
        diffusion = f" - {eta}*lap(b)" if eta else ""
        pb.add_equation(f"dt(b) - curl(cross(v, B0)){diffusion} = 0", name="induction")
        pb.add_equation("div(v) = 0", name="continuity")
        for w in (("outer",) if ball else ("inner", "outer")):
            pb.add_bc("dot(r_hat, v) = 0", at=w)
        return pb.build()

    @pytest.mark.parametrize("geometry", ["shell", "ball"])
    def test_free_wall_keeps_the_wall_fluxes_of_an_ideal_involution(self, geometry):
        """A flux unknown with no wall condition keeps its wall fluxes as
        DOFs (kind 'free'): they evolve with the unmasked wall
        circulations of Cross(v, B0), which vanish on the Malkus field
        with v_r = 0, so the wall rows are empty in A against unit mass;
        on a ball the origin ring is still pinned. With a resistive term
        the free wall is refused."""
        Nr, Nt = 6, 8
        ball = geometry == "ball"
        grid = make_ball_grid(N_r=Nr + 1, N_theta=Nt) if ball else \
            make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=0.5)
        cp = compile_staggered_general(self._free_problem(ball=ball), grid, 2)
        rec = cp.constraints
        assert rec["walls"]["vectors"]["b"] == {
            "normal": (), "tangential": None, "kind": "free", "electric": None,
            "open": {}}
        assert rec["involutions"] == {"b": "induction"}
        f = cp.layout.field("b_r")
        assert f.size == (Nr + 1) * Nt
        A, B = cp.A_total.tocsr(), cp.B_total.tocsr()
        outer = slice(f.offset + Nr * Nt, f.offset + (Nr + 1) * Nt)
        assert A[outer, :].nnz == 0
        assert np.allclose(B[outer, outer].diagonal(), 1.0)
        origin = slice(f.offset, f.offset + Nt)
        if ball:
            assert (A[origin, :] - sp.eye(Nt, A.shape[1], k=f.offset)).nnz == 0
            assert B[origin, :].nnz == 0
        else:
            assert A[origin, :].nnz == 0
        with pytest.raises(NotImplementedError, match="free wall keeps the wall fluxes"):
            compile_staggered_general(self._free_problem(eta=0.01, ball=ball), grid, 2)


# ---------------------------------------------------------------------------
# Stage 4: free decay in a conducting shell (closed form)
# ---------------------------------------------------------------------------

DECAY_RI, DECAY_RO = 0.5, 1.0


def _decay_det(family, l, k):
    """The characteristic determinant of free decay in a perfectly
    conducting shell.  With b = Curl(Curl(S r)) + Curl(T r) both potentials
    obey Dt(X) = Lap(X), so X = a j_l(k r) + b y_l(k r) and the rate is
    -k^2.  The walls carry b_r = 0 and a vanishing tangential electric
    field E = Curl(b): b_r = l(l+1) S / r gives S = 0, which also kills the
    toroidal part of E (proportional to Lap(S) = -k^2 S); the poloidal part
    of E is proportional to d(r T)/dr, so the toroidal potential closes on
    d(r T)/dr = 0."""
    from scipy.special import spherical_jn, spherical_yn

    def value(f, rr):
        if family == "pol":
            return f(l, k * rr)
        return f(l, k * rr) + k * rr * f(l, k * rr, derivative=True)
    return (value(spherical_jn, DECAY_RI) * value(spherical_yn, DECAY_RO)
            - value(spherical_jn, DECAY_RO) * value(spherical_yn, DECAY_RI))


def decay_rates(family, l, n=2, kmax=40.0):
    """The n gravest decay rates k^2 of ``_decay_det``."""
    from scipy.optimize import brentq
    ks = np.linspace(0.05, kmax, 8000)
    vals = np.array([_decay_det(family, l, k) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) \
                and np.sign(vals[i]) != np.sign(vals[i + 1]):
            k = brentq(lambda x: _decay_det(family, l, x), ks[i], ks[i + 1], xtol=1e-14)
            out.append(k * k)
            if len(out) == n:
                break
    return out


def decay_problem(formulation):
    """Free decay of b in a conducting shell, as the two formulations:
    'b' with the involution Div(b) = 0 and the conducting wall
    (b_r = 0, tangential Curl(b) = 0), 'A' with the Coulomb gauge as a
    saddle point (tangential A = 0 and psi = 0)."""
    pb = Problem()
    pb.add_unknown("b", "vector")
    if formulation == "A":
        pb.add_unknown("psi", "scalar")
        pb.add_equation("dt(b) - lap(b) - grad(psi) = 0", name="induction")
        pb.add_equation("div(b) = 0", name="coulomb")
        for w in ("inner", "outer"):
            pb.add_bc("dot(theta_hat, b) = 0", at=w)
            pb.add_bc("dot(phi_hat, b) = 0", at=w)
            pb.add_bc("psi = 0", at=w)
    else:
        pb.add_equation("dt(b) - lap(b) = 0", name="induction")
        pb.add_equation("div(b) = 0", name="solenoidal")
        for w in ("inner", "outer"):
            pb.add_bc("dot(r_hat, b) = 0", at=w)
            pb.add_bc("dot(theta_hat, curl(b)) = 0", at=w)
            pb.add_bc("dot(phi_hat, curl(b)) = 0", at=w)
    return pb.build()


def _toroidal_fraction(cp, x, formulation, ops):
    """The share of the magnetic energy in b_phi: 1 for a toroidal mode and
    0 for a poloidal one at m = 0.  The vector potential is curled first."""
    parts = [x[cp.layout.field(f"b_{c}").slice] for c in "rtp"]
    if formulation == "A":
        edge = ops.primal_curl("noslip") @ np.concatenate(parts)
        n_ce, n_rc = ops.n["CE"], ops.n["RC"]
        parts = [edge[:n_ce], edge[n_ce:n_ce + n_rc], edge[n_ce + n_rc:]]
    e = [float(np.linalg.norm(p)) ** 2 for p in parts]
    return e[2] / sum(e)


DECAY_TARGETS = [("tor", 1, 0), ("tor", 1, 1), ("pol", 1, 0), ("pol", 2, 0)]


class TestFreeDecay:
    """Free decay in a conducting shell against the Bessel conditions of
    ``_decay_det``, both formulations, at m = 0 where the toroidal and
    poloidal families decouple."""

    @pytest.mark.parametrize("formulation", ["b", "A"])
    def test_decay_rates_converge_at_second_order(self, formulation):
        system = decay_problem(formulation)
        assert general_path_reason(system, 0) is None
        errs = {}
        for Nr, Nt in ((12, 24), (24, 48)):
            grid = make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=DECAY_RI,
                                   r_outer=DECAY_RO)
            ops = _CGridOps(StaggeredGrids.from_grid(grid), 0, 2)
            cp = compile_staggered_general(system, grid, 0)
            K, M = cp.assemble_evp_matrices()
            v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
            for family, l, n in DECAY_TARGETS:
                rate = decay_rates(family, l, n + 1)[n]
                lam, V = shift_invert(K, M, rate + 0.0j, 12, return_vectors=True,
                                      tol=1e-10, v0=v0)
                found = None
                for j in np.argsort(np.abs(lam - rate)):
                    if not np.isfinite(lam[j]):
                        continue
                    frac = _toroidal_fraction(cp, V[:, j], formulation, ops)
                    if (frac > 0.9) if family == "tor" else (frac < 0.1):
                        found = lam[j]
                        break
                assert found is not None, (formulation, family, l, n, lam)
                errs.setdefault((family, l, n), []).append(abs(found - rate) / rate)
        for key, (coarse, fine) in errs.items():
            assert fine < 8e-3, (formulation, key, coarse, fine)
            assert 1.7 < np.log2(coarse / fine) < 2.4, (formulation, key, coarse, fine)

    def test_pole_closure_of_the_radial_current(self):
        """At m = 0 the r-face circulation of the edge current needs the
        pole closure of j_r; without it the toroidal rates do not converge
        (the poloidal family, which does not see j_r at the pole, is
        unaffected)."""
        from spdee.backends.fd2.staggered_mhd import CTGrid
        grid = make_shell_grid(N_r=25, N_theta=48, r_inner=DECAY_RI, r_outer=DECAY_RO)
        sg = StaggeredGrids.from_grid(grid)
        assert CTGrid(sg, 0, pole_closure=True).jr_bphi.nnz \
            > CTGrid(sg, 0, pole_closure=False).jr_bphi.nnz
        for m in (1, 8):
            on = CTGrid(sg, m, pole_closure=True).jr_bphi
            off = CTGrid(sg, m, pole_closure=False).jr_bphi
            assert maxdiff(on, off) == 0.0


# ---------------------------------------------------------------------------
# Stage 4: ideal Alfven modes on a uniform axial field
# ---------------------------------------------------------------------------

ALFVEN_RI, ALFVEN_RO = 0.5, 1.0
# gravest Alfven frequency of the shell: pi v_A / L with v_A = |B0| = 1 and
# L = 2 sqrt(R_o^2 - s^2) the longest field line, at the cylindrical radius
# s = R_i that grazes the inner sphere
ALFVEN_EDGE = np.pi / (2 * np.sqrt(ALFVEN_RO ** 2 - ALFVEN_RI ** 2))


def alfven_problem():
    """Ideal (eta = 0, nu = 0) MHD on a uniform axial field: the velocity is
    a saddle point with the pressure, b an involution driven by
    Curl(Cross(v, B0)) with the Lorentz force Cross(Curl(b), B0), and the
    walls are impenetrable and perfectly conducting (tangential
    Cross(v, B0) = 0, the ideal electric field)."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + grad(p) - cross(curl(b), B0) = 0", name="momentum")
    pb.add_equation("dt(b) - curl(cross(v, B0)) = 0", name="induction")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in ("inner", "outer"):
        pb.add_bc("dot(r_hat, v) = 0", at=w)
        pb.add_bc("dot(r_hat, b) = 0", at=w)
        pb.add_bc("dot(theta_hat, cross(v, B0)) = 0", at=w)
        pb.add_bc("dot(phi_hat, cross(v, B0)) = 0", at=w)
    return pb.build()


def alfven_spectrum(Nr, Nt, m, k=30):
    """The eigenfrequencies nearest the band edge and the compiled pencil."""
    from spdee.backends.fd2.compiled_problem import EvpResult
    grid = make_shell_grid(N_r=Nr + 1, N_theta=Nt, r_inner=ALFVEN_RI, r_outer=ALFVEN_RO)
    cp = compile_staggered_general(alfven_problem(), grid, m)
    K, M = cp.assemble_evp_matrices()
    v0 = np.random.default_rng(0).standard_normal(K.shape[0]).astype(complex)
    lam, V = shift_invert(K, M, 1j * ALFVEN_EDGE, k, return_vectors=True, v0=v0)
    om = EvpResult.eigenvalues_to_omega(lam)
    keep = np.isfinite(om) & (np.abs(om.real) > 1e-3)
    return cp, om[keep], V[:, keep]


class TestIdealAlfven:
    """The ideal axial-field spectrum is the Alfven continuum of the field
    lines, so single eigenvalues do not converge; what the C-grid must get
    right is that the band closes on the analytic edge from below, that the
    modes do not grow (the wall circulations of the ideal electric field
    Cross(v, B0) are masked, so the pencil loses no energy at the wall) and
    that the flux stays div-free."""

    def test_band_approaches_the_analytic_edge_and_the_flux_is_div_free(self):
        lows = []
        for Nr, Nt in ((12, 24), (24, 48)):
            cp, om, V = alfven_spectrum(Nr, Nt, 1)
            lows.append(float(np.abs(om.real).min()))
            assert np.abs(om.imag).max() < 1e-2 * ALFVEN_EDGE, (Nr, Nt, om)
            j = int(np.argmin(np.abs(np.abs(om.real) - ALFVEN_EDGE)))
            x = V[:, j] / np.linalg.norm(V[:, j])
            flux = np.concatenate([x[cp.layout.field(f"b_{c}").slice] for c in "rtp"])
            assert abs(cp.divergence(x)).max() < 1e-10 * abs(flux).max()
        assert lows[0] < lows[1] < ALFVEN_EDGE, lows

    def test_ideal_wall_masks_the_electric_field_of_the_involution(self):
        """The ideal conducting wall Dot(t_hat, Cross(v, B0)) = 0 names the
        velocity (there is no Curl in it), but it closes the induction
        equation, so it is read as the tangential electric field of the
        flux unknown and the velocity keeps an ordinary impenetrable wall."""
        system = alfven_problem()
        record = classify_constraints(system, walk_equation_system(system, 1))
        walls = _classify_walls(system, 1, record, _CGridOps(StaggeredGrids.from_grid(GRID), 1, 2))
        assert walls["vectors"]["b"]["kind"] == "conducting"
        assert set(walls["vectors"]["b"]["electric"]) == {Boundary.R_INNER,
                                                          Boundary.R_OUTER}
        assert walls["vectors"]["v"]["kind"] == "impenetrable"
        assert walls["vectors"]["v"]["tangential"] is None


# ---------------------------------------------------------------------------
# Composite wall rows on the shell
# ---------------------------------------------------------------------------

def _shell_problem(unknowns, equations, params, bcs):
    """``bcs`` maps a wall ('inner' / 'outer') to its rows."""
    pb = Problem()
    for name, kind in unknowns:
        pb.add_unknown(name, kind)
    for name, value in params.items():
        pb.add_parameter(name, value)
    for eq in equations:
        pb.add_equation(eq)
    for at, rows in bcs.items():
        for row in rows:
            pb.add_bc(row, at=at)
    return pb


def shell_robin_k(l, h_in, h_out, r_in, n=1):
    """p = a j_l(k r) + b y_l(k r) with d_r p + h p = 0 at both walls of
    [r_in, 1]: the roots k of the 2 x 2 determinant."""
    from scipy.special import spherical_jn, spherical_yn

    def row(k, r, h):
        return (k * spherical_jn(l, k * r, derivative=True) + h * spherical_jn(l, k * r),
                k * spherical_yn(l, k * r, derivative=True) + h * spherical_yn(l, k * r))

    def det(k):
        a, b = row(k, 1.0, h_out)
        c, d = row(k, r_in, h_in)
        return a * d - b * c
    from scipy.optimize import brentq
    ks = np.linspace(0.3, 30.0, 4000)
    v = np.array([det(k) for k in ks])
    out = []
    for i in np.flatnonzero(np.sign(v[:-1]) != np.sign(v[1:])):
        out.append(brentq(det, ks[i], ks[i + 1]))
    return out[:n]


def _nearest(cp, target, seed=0, k=8):
    A, B = cp.assemble_evp_matrices()
    v0 = np.random.default_rng(seed).standard_normal(A.shape[0]).astype(complex)
    lam, _ = shift_invert(A, B, target * 0.97, k, v0=v0)
    lam = lam[np.isfinite(lam)]
    return lam[np.argmin(np.abs(lam - target))]


STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"
STAG = dict(backend="fd2", discretization="staggered", order_r=2, order_theta=2)


class TestCompositeWallsShell:
    """The composite rows on the shell: the Robin closure against the
    two-wall Bessel determinant, its Dirichlet and Neumann limits, the
    Navier-slip limits and mixed walls, the open (normal-stress) wall's
    structure, and the refusals."""

    R_IN = 0.5

    @pytest.mark.parametrize("m", [0, 1, 2])
    def test_robin_decay_rates_second_order(self, m):
        """dt(p) - lap(p) = 0 on [1/2, 1] with d_r p + 3 p = 0 outside and
        d_r p - p = 0 inside (both walls absorb): lambda = k^2 of the
        determinant of shell_robin_k, l = m, m + 1, at second order."""
        h_in, h_out = -1.0, 3.0
        targets = {l: shell_robin_k(l, h_in, h_out, self.R_IN)[0]**2 for l in range(m, m + 2)}
        errs = {}
        for N in (12, 24):
            pb = _shell_problem([("p", "scalar")], ["dt(p) - lap(p) = 0"],
                                {"hi": h_in, "ho": h_out},
                                {"inner": ["dot(r_hat, grad(p)) + hi*p = 0"],
                                 "outer": ["dot(r_hat, grad(p)) + ho*p = 0"]})
            grid = make_shell_grid(N_r=N + 1, N_theta=2 * N, r_inner=self.R_IN)
            cp = pb.compile(grid, m=m, **STAG)
            assert cp.constraints["walls"]["scalars"]["p"]["closure"] == ("robin", "robin")
            for l, k2 in targets.items():
                errs.setdefault(l, []).append(abs(_nearest(cp, k2, l).real - k2) / k2)
        for l, (e1, e2) in errs.items():
            assert e2 < 3e-3, (l, e1, e2)
            assert order_of([e1, e2], [12, 24])[0] > 1.7, (l, e1, e2)

    def test_robin_limits(self):
        """h = 0 is the natural (Neumann) pencil to the entry; a vanishing
        derivative coefficient is the Dirichlet closure -2 p_0 / h to
        rounding; the ghost coefficients follow the closed form."""
        grid = make_shell_grid(N_r=9, N_theta=8, r_inner=self.R_IN)
        eqs = ["dt(p) - lap(p) = 0"]
        robin0 = _shell_problem([("p", "scalar")], eqs, {"h": 0.0},
                                {"outer": ["dot(r_hat, grad(p)) + h*p = 0"],
                                 "inner": ["dot(r_hat, grad(p)) + h*p = 0"]}
                                ).compile(grid, m=1, **STAG)
        neumann = _shell_problem([("p", "scalar")], eqs, {},
                                 {"outer": ["dot(r_hat, grad(p)) = 0"],
                                  "inner": ["dot(r_hat, grad(p)) = 0"]}).compile(grid, m=1, **STAG)
        assert maxdiff(robin0.A_total, neumann.A_total) == 0.0
        dirichlet_like = _shell_problem([("p", "scalar")], eqs, {"eps": 0.0},
                                        {"outer": ["eps*dot(r_hat, grad(p)) + p = 0"],
                                         "inner": ["eps*dot(r_hat, grad(p)) + 3*p = 0"]}
                                        ).compile(grid, m=1, **STAG)
        dirichlet = _shell_problem([("p", "scalar")], eqs, {},
                                   {"outer": ["p = 0"], "inner": ["p = 0"]}
                                   ).compile(grid, m=1, **STAG)
        scale = abs(dirichlet.A_total).max()
        assert maxdiff(dirichlet_like.A_total, dirichlet.A_total) < 1e-13 * scale
        sg = StaggeredGrids.from_grid(grid)
        hr = sg.r_edge[1] - sg.r_edge[0]
        cp = _shell_problem([("p", "scalar")], eqs, {},
                            {"outer": ["dot(r_hat, grad(p)) + 3*p = 0"],
                             "inner": ["2*dot(r_hat, grad(p)) - p = 0"]}).compile(grid, m=1, **STAG)
        robin = cp.constraints["walls"]["scalars"]["p"]["robin"]
        assert np.allclose(robin[Boundary.R_OUTER], -3.0 / (1.0 + 3.0 * hr / 2))
        assert np.allclose(robin[Boundary.R_INNER], 1.0 / (2.0 + 1.0 * hr / 2))

    def test_robin_refusals(self):
        grid = make_shell_grid(N_r=9, N_theta=8, r_inner=self.R_IN)
        pb = _shell_problem([("p", "scalar")], ["dt(p) - lap(p) = 0"], {},
                            {"outer": ["dot(r_hat, grad(p)) + p = 0"],
                             "inner": ["dot(r_hat, grad(p)) = 0"]})
        with pytest.raises(NotImplementedError, match="Robin closure of Lap\\(p\\) is order 2"):
            pb.compile(grid, m=1, backend="fd2", discretization="staggered",
                       order_r=4, order_theta=4)
        pb = zhang_problem(Ek=1e-3, wall="noslip")
        pb.add_bc("dot(r_hat, grad(p)) + p = 0", at="outer")
        with pytest.raises(NotImplementedError, match="multiplier 'p' by a Robin"):
            pb.compile(grid, m=1, **STAG)

    def test_navier_slip_limits_and_mixed_walls(self):
        """ls = 0 at both walls is the no-slip pencil bitwise, a vanishing
        velocity coefficient the stress-free one; no-slip inside and
        stress-free outside is one closure with ls = (0, inf), between the
        two."""
        grid = make_shell_grid(N_r=9, N_theta=8, r_inner=self.R_IN)
        eqs = ["dt(v) - Ek*lap(v) = 0"]
        slip = [f"dot({h}, v) + ls*dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
        free = [f"eps*dot({h}, v) + dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
        normal = ["dot(r_hat, v) = 0"]
        for m in (0, 1, 2):
            cp0 = _shell_problem([("v", "vector")], eqs, {"Ek": 0.01, "ls": 0.0},
                                 {"inner": normal + slip, "outer": normal + slip}
                                 ).compile(grid, m=m, **STAG)
            pinned = [f"dot({h}, v) = 0" for h in ("r_hat", "theta_hat", "phi_hat")]
            noslip = _shell_problem([("v", "vector")], eqs, {"Ek": 0.01},
                                    {w: pinned for w in ("inner", "outer")}
                                    ).compile(grid, m=m, **STAG)
            assert maxdiff(cp0.A_total, noslip.A_total) == 0.0
            assert cp0.A_total.nnz == noslip.A_total.nnz
            cpi = _shell_problem([("v", "vector")], eqs, {"Ek": 0.01, "eps": 0.0},
                                 {"inner": normal + free, "outer": normal + free}
                                 ).compile(grid, m=m, **STAG)
            sf = _shell_problem([("v", "vector")], eqs, {"Ek": 0.01},
                                {w: normal + [f"dot({h}, {STRAIN}) = 0"
                                              for h in ("theta_hat", "phi_hat")]
                                 for w in ("inner", "outer")}).compile(grid, m=m, **STAG)
            assert maxdiff(cpi.A_total, sf.A_total) == 0.0
            mixed = _shell_problem([("v", "vector")], eqs, {"Ek": 0.01},
                                   {"inner": pinned,
                                    "outer": normal + [f"dot({h}, {STRAIN}) = 0"
                                                       for h in ("theta_hat", "phi_hat")]}
                                   ).compile(grid, m=m, **STAG)
            wall = mixed.constraints["walls"]["vectors"]["v"]["tangential"]
            assert isinstance(wall, TangentialWall)
            assert wall.slip[0] == 0.0 and wall.slip[1] == np.inf
            assert maxdiff(mixed.A_total, noslip.A_total) > 0
            assert maxdiff(mixed.A_total, sf.A_total) > 0

    def test_normal_stress_structure(self):
        """The open wall keeps the wall v_r faces as DOFs (the closed wall
        loses its ring), the row is linear in the IR (the rows with -c p
        and +c p average to the row without p), the multiplier's m = 0
        gauge is lifted, the spectrum near zero is real and not growing."""
        grid = make_shell_grid(N_r=9, N_theta=8, r_inner=self.R_IN)
        Nt = 8
        eqs = ["dt(v) - lap(v) + grad(p) = 0", "div(v) = 0"]
        tangential = [f"dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
        cps = {}
        for sign in ("-", "+", ""):
            term = f"{sign} 2*p" if sign else ""
            pb = _shell_problem([("v", "vector"), ("p", "scalar")], eqs, {},
                                {"outer": [f"dot(r_hat, {STRAIN}) {term} = 0"] + tangential,
                                 "inner": ["dot(r_hat, v) = 0"] + tangential})
            cps[sign] = pb.compile(grid, m=0, **STAG)
        cp = cps["-"]
        rec = cp.constraints
        assert list(rec["walls"]["vectors"]["v"]["open"]) == [Boundary.R_OUTER]
        assert rec["walls"]["vectors"]["v"]["normal"] == (Boundary.R_INNER,)
        assert cp.layout.field("v_r").size == 9 * Nt - Nt
        assert rec["equations"]["eq2"]["gauge"] is None
        scale = abs(cp.A_total).max()
        assert maxdiff((cps["-"].A_total + cps["+"].A_total) / 2, cps[""].A_total) < 1e-13 * scale
        assert maxdiff(cps["-"].A_total, cps[""].A_total) > 1e-3 * scale
        A, B = cp.assemble_evp_matrices()
        lam, _ = shift_invert(A, B, -0.3, 8, v0=np.ones(A.shape[0], dtype=complex))
        lam = lam[np.isfinite(lam)]
        assert np.abs(lam.imag).max() < 1e-8 * np.abs(lam).max()
        assert lam.real.min() > -1e-6

    def test_normal_stress_refusals(self):
        grid = make_shell_grid(N_r=9, N_theta=8, r_inner=self.R_IN)
        tangential = [f"dot({h}, {STRAIN}) = 0" for h in ("theta_hat", "phi_hat")]
        closed = {"inner": ["dot(r_hat, v) = 0"] + tangential}
        pb = _shell_problem([("v", "vector"), ("p", "scalar")],
                            ["dt(v) - lap(v) + grad(p) = 0", "div(v) = 0"], {},
                            {"outer": [f"dot(r_hat, {STRAIN}) - p = 0", "dot(r_hat, v) = 0"]
                             + tangential, **closed})
        with pytest.raises(NotImplementedError, match="closed or open"):
            pb.compile(grid, m=1, **STAG)
        pb = _shell_problem([("v", "vector"), ("p", "scalar"), ("T", "scalar")],
                            ["dt(v) - lap(v) + grad(p) = 0", "div(v) = 0", "dt(T) - lap(T) = 0"],
                            {}, {"outer": [f"dot(r_hat, {STRAIN}) - T = 0"] + tangential
                                 + ["T = 0"], "inner": closed["inner"] + ["T = 0"]})
        with pytest.raises(NotImplementedError, match="not the multiplier of Div\\(v\\)"):
            pb.compile(grid, m=1, **STAG)
        pb = _shell_problem([("v", "vector"), ("p", "scalar")],
                            ["dt(v) + grad(p) = 0", "div(v) = 0"], {},
                            {"outer": [f"dot(r_hat, {STRAIN}) - p = 0"] + tangential, **closed})
        with pytest.raises(NotImplementedError, match="needs one Lap\\(v\\)"):
            pb.compile(grid, m=1, **STAG)
