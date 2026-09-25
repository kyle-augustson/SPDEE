"""The staggered (Arakawa-C) assembler, driven by the IR task list
(docs/staggered_general_design.md). The system is walked into
``OperatorTask`` records, every equation is classified as an evolution
equation, an algebraic constraint or a tensor definition (section 4),
every task kind is mapped to a mimetic block of the C-grid operator
library of ``staggered_compile`` and ``staggered_mhd`` (section 3), the
wall conditions are applied per section 5 and the ball origin per
section 6. The dispatcher carries no physics: signs, coefficients and
background operands come from the tasks and are evaluated on the grid at
the block's location.

Placement (sections 2a, 2b): every unknown carries a grade, ``'center'``
(one block on the cell centres), ``'face'`` (three blocks v_r on the
r-edges, v_theta on the theta-edges, v_phi on the centres), ``'tensor'``
(the six symmetric components at the strain locations CC / EE / RC / CE)
or ``'tensor9'`` (all nine components). The dispatcher and the block
builders read the placement from the classification record, never from
the unknown's kind.

Grade-changing products (Dot / Cross with a background, the Coriolis pairs,
the tensor-to-face contractions) are built from one base interpolation P
and its norm adjoint W^-1 P^T W, never from two independent averages, so
each component pair of the assembled operator is exactly W-adjoint or
W-skew. Grad(v) onto the tensor placement is the strain map S (symmetric
part) or the nine-component gradient map; Div(T) is the norm adjoint
-W_v^-1 S^T W_T back to the faces, which is how the viscous Gram is built.

Combinations the dispatcher reads from one equation's task list (section
3): Grad(M s) from its tagged product-rule pair as G diag(M); the viscous
force Lap + (1/3) Grad Div [+ a . (Grad v + Grad v^T) - (2/3) a Div(v)]
with a radial a as one strain Gram, density-weighted by
w = exp(integral a_r dr) when the stratification terms are present; the
thermal diffusion Lap(s) + a . Grad(s) as the w-weighted scalar Laplacian.
Tasks whose coefficient vanishes identically are dropped.

Sign convention: every A block is entered with the sign the term carries
in the equation ``Dt(u) + ... = 0``, so A = -A_phys with dt x = A_phys x
and omega = -i lambda.

Wall conditions (section 5) come from the walker's composite-aware BC
tasks (docs/ir_boundary_and_constraints.md): a single-shape row is one
of the C-grid closures, and a composite row is read as one of three
closures with its coefficients evaluated at the wall nodes: Robin on a
centre scalar as the mirror-ghost flux closure with the coefficient,
Navier slip as the strain closure with a slip length (``TangentialWall``,
whose ls = 0 and ls = inf limits are the no-slip and stress-free
closures), the normal stress as an open wall whose v_r faces stay
degrees of freedom. Any other composite shape is refused naming the
term.

Constrained transport (section 4, case 2): a face vector b whose
constraint Div(b) = 0 has no multiplier and whose evolution is a curl
(plus diffusion) keeps no constraint row; its DOFs are the face fluxes,
the induction, resistive and Lorentz blocks are the CT builders of
``staggered_mhd`` (Div Curl = 0 exactly, checked at assembly) and its
walls are b_r = 0 with the tangential electric field masked (conducting)
or unmasked (radial), or no condition at all on an ideal evolution (the
wall fluxes stay as DOFs). On a ball the origin ring of the fluxes
crosses faces of zero area and is pinned at every m; the Lorentz force
closes its first-centre averages by the vector origin parity
(docs/ball_staggered.md section 7). The vector potential (section 2a) is a saddle point
whose vector keeps its wall normal DOFs: tangential A = 0 closes the
curl stencils, psi = 0 closes the gradient wall rows with the half-cell
face weight, so the flux-form divergence stays the norm adjoint of the
gradient; Lap(A) is -Curl Curl A and the Lorentz force is
Cross(Curl(Curl(A)), B0) through the Hodge-paired cross block; on a ball
its origin ring takes the regular vector's fold.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional

import numpy as np
import scipy.sparse as sp
import sympy as smp
from scipy.integrate import quad

from spdee.backends.fd2.staggered_compile import (
    StaggeredCompiledMatrices, StaggeredField, StaggeredGrids, StaggeredLayout,
    _apply_ball_origin_vr, _assemble_curl_blocks, _avg_1d, _avg_c2e_zero_ends,
    _avg_e2c, _build_staggered_gradient_blocks, _origin_parity,
    _pole_vtheta_prolongation, _reduce_velocity_dofs, _robin_wall_term,
    _slip_face_weight, _slip_ratio, _stag_diff_1d, _stag_diff_1d_lagrange,
    _stag_diff_1d_origin_parity, _stag_diff_c2e_zero_ends, _stag_diff_e2c,
    _stag_interp_1d_lagrange, _stag_theta_diff_parity, _strain_operator_and_norms,
    _strain_pole_share, TangentialWall, build_staggered_angular_laplacian,
    build_staggered_grad_div, build_staggered_scalar_laplacian,
    build_staggered_weighted_scalar_laplacian, build_staggered_weighted_viscous_gram,
    normalize_wall_bc,
)
from spdee.backends.fd2.staggered_mhd import (
    CTGrid, ct_induction_blocks, ct_lorentz_blocks, ct_resistive_blocks,
)
from spdee.ir.constraints import classify, describe_tasks, tensor_terms
from spdee.ir.equations import (
    Boundary, BoundaryCondition, ImpenetrableStressFree, r_hat,
)
from spdee.ir.operators import Dot, Grad, Transpose
from spdee.ir.tasks import (
    BCTask, OperatorTask,
    BC_DIRICHLET_SCALAR, BC_DIRICHLET_VECTOR_PROJECTION, BC_LINEAR_COMBINATION,
    BC_NEUMANN_SCALAR_RADIAL, BC_PROJECTION_CURL_CROSS,
    BC_SCALAR_LINEAR_FROM_TENSOR,
    BC_UNIT_PHI_HAT, BC_UNIT_R_HAT, BC_UNIT_THETA_HAT,
    TARGET_A, TARGET_B, TARGET_RHS,
    TASK_BG_VEC_TIMES_DIV_UNK_VEC, TASK_BG_VEC_TIMES_UNK_SCALAR,
    TASK_CROSS_CURL_CURL_UNK_BG, TASK_CROSS_CURL_UNK_BG, TASK_CROSS_VECTOR,
    TASK_CURL_CROSS_UNK_BG, TASK_CURL_CURL_VECTOR,
    TASK_CURL_VECTOR, TASK_DIV_TENSOR_UNK, TASK_DIV_TENSOR_VECTOR,
    TASK_DIV_VECTOR_SCALAR, TASK_DOT_BG_GRAD_VEC, TASK_DOT_BG_TENSOR_VEC,
    TASK_DOT_BG_VEC_GRAD_SCALAR, TASK_DOT_BG_VEC_TENSOR_UNK,
    TASK_DOT_BG_VEC_UNK_TENSOR, TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
    TASK_GRAD_DIV_VECTOR, TASK_GRAD_SCALAR, TASK_IDENTITY_SCALAR,
    TASK_IDENTITY_TENSOR, TASK_IDENTITY_VECTOR, TASK_LAPH_SCALAR,
    TASK_LAP_SCALAR, TASK_LAP_VECTOR, TASK_SOURCE_SCALAR, TASK_SOURCE_TENSOR,
    TASK_SOURCE_VECTOR, TASK_TENSOR_OF_VECTOR, bind_bc_parameters,
    bind_parameters,
)
from spdee.ir.walker import (
    _parse_one_bc, walk_boundary_conditions, walk_equation_system,
)


# ---------------------------------------------------------------------------
# Placement vocabulary and task-kind coverage
# ---------------------------------------------------------------------------

# tensor component order: the six symmetric slots of build_staggered_strain
# first (rr, tt, pp, rt, rp, tp), then the transposed off-diagonals
TENSOR_COMPONENTS = ("rr", "tt", "pp", "rt", "rp", "tp", "tr", "pr", "pt")
TENSOR_LOCATIONS = ("center", "center", "center", "corner", "r_edge",
                    "theta_edge", "corner", "r_edge", "theta_edge")
TENSOR_INDEX = {"r": 0, "t": 1, "p": 2}

# placement -> ((field suffix, StaggeredField location), ...), the DOF blocks
# of an unknown in state order
PLACEMENTS = {
    "center": (("", "center"),),
    "face": (("_r", "r_edge"), ("_t", "theta_edge"), ("_p", "center")),
    "tensor": tuple(("_" + c, loc) for c, loc in
                    zip(TENSOR_COMPONENTS[:6], TENSOR_LOCATIONS[:6])),
    "tensor9": tuple(("_" + c, loc) for c, loc in
                     zip(TENSOR_COMPONENTS, TENSOR_LOCATIONS)),
}

# StaggeredField location -> sub-grid code (r nodes, theta nodes)
LOC_CODE = {"center": "CC", "r_edge": "RC", "theta_edge": "CE", "corner": "EE"}

FACE_COMPONENTS = ("v_r", "v_theta", "v_phi")

SUPPORTED_KINDS = frozenset({
    TASK_IDENTITY_SCALAR, TASK_IDENTITY_VECTOR, TASK_IDENTITY_TENSOR,
    TASK_GRAD_SCALAR, TASK_DIV_VECTOR_SCALAR, TASK_LAP_SCALAR,
    TASK_LAPH_SCALAR, TASK_LAP_VECTOR, TASK_GRAD_DIV_VECTOR,
    TASK_CROSS_VECTOR, TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
    TASK_BG_VEC_TIMES_UNK_SCALAR, TASK_DOT_BG_VEC_GRAD_SCALAR,
    TASK_BG_VEC_TIMES_DIV_UNK_VEC, TASK_DOT_BG_GRAD_VEC,
    TASK_DOT_BG_VEC_TENSOR_UNK, TASK_DIV_TENSOR_VECTOR, TASK_DOT_BG_TENSOR_VEC,
    TASK_TENSOR_OF_VECTOR, TASK_DIV_TENSOR_UNK, TASK_DOT_BG_VEC_UNK_TENSOR,
    TASK_SOURCE_SCALAR, TASK_SOURCE_VECTOR, TASK_SOURCE_TENSOR,
    TASK_CURL_VECTOR, TASK_CURL_CROSS_UNK_BG, TASK_CROSS_CURL_UNK_BG,
    TASK_CURL_CURL_VECTOR, TASK_CROSS_CURL_CURL_UNK_BG,
})

# natural r-power of a kind (the collocated table); the ball premultiply
# scales each evolution row by r^max(1, power)
R_POWER = {
    TASK_LAP_SCALAR: 2, TASK_LAPH_SCALAR: 2, TASK_LAP_VECTOR: 2,
    TASK_GRAD_DIV_VECTOR: 2, TASK_DIV_TENSOR_VECTOR: 2,
    TASK_CURL_CURL_VECTOR: 2, TASK_CROSS_CURL_CURL_UNK_BG: 2,
    TASK_GRAD_SCALAR: 1, TASK_DIV_VECTOR_SCALAR: 1, TASK_DOT_BG_GRAD_VEC: 1,
    TASK_DOT_BG_VEC_TENSOR_UNK: 1, TASK_DOT_BG_VEC_GRAD_SCALAR: 1,
    TASK_BG_VEC_TIMES_DIV_UNK_VEC: 1, TASK_TENSOR_OF_VECTOR: 1,
    TASK_DIV_TENSOR_UNK: 1, TASK_CURL_VECTOR: 1, TASK_CURL_CROSS_UNK_BG: 1,
    TASK_CROSS_CURL_UNK_BG: 1,
}


# ---------------------------------------------------------------------------
# Section 4: constraint classification
# ---------------------------------------------------------------------------

def classify_constraints(system, tasks: List[OperatorTask],
                         geometry: str = "shell") -> dict:
    """The C-grid view of ``spdee.ir.constraints.classify`` (mimetic, the
    'staggered' refusal prefix): the dict

        placement   : {unknown: 'center' | 'face' | 'tensor' | 'tensor9'}
        equations   : {equation: {'class': 'evolution' | 'saddle_point' |
                       'involution' | 'definition', ...}}
        principal   : {equation: unknown whose DOFs carry its rows}
        multipliers : {multiplier: equation}
        involutions : {vector: its curl evolution equation}
        constraints : {vector: 'saddle_point' | 'involution'}
        definitions : {tensor: equation} (eliminated tensor unknowns)

    of docs/staggered_general_design.md section 4; a written involution
    is dropped (constrained transport keeps the face fluxes).
    """
    return classify(system, tasks, geometry=geometry, mimetic=True,
                    core="staggered").staggered_view()


# ---------------------------------------------------------------------------
# Section 5: wall conditions
# ---------------------------------------------------------------------------

def _traction_shape(system, bct: BCTask, m: int):
    """('theta' | 'phi' | 'r', scale) when a tensor-projection BC is a
    traction row Dot(u_hat, Dot(strain, r_hat)) on the BC's unknown, the
    tangential ones of ``ImpenetrableStressFree`` or the normal stress,
    with the strain Grad(u) + Transpose(Grad(u)) (scale 1) or half of it
    (scale 1/2, the row is scale times the full traction); None else."""
    field = system.unknowns[bct.unknown_name]
    expr = bct.metadata["scalar_expr"]
    strain = Grad(field) + Transpose(Grad(field))
    canons = list(ImpenetrableStressFree(field, boundary=bct.location)[1:])
    canons.append(BoundaryCondition(bct.location, Dot(r_hat, Dot(strain, r_hat))))
    for canon in canons:
        ct = _parse_one_bc(canon, m)
        ref = ct.metadata["scalar_expr"]
        kind = {+1: "theta", -1: "phi", 0: "r"}[ct.metadata["host_sigma"]]
        if smp.expand(expr - ref) == 0:
            return kind, 1.0
        if smp.expand(2 * expr - ref) == 0:
            return kind, 0.5
    return None


def _electric_target(bct: BCTask, record: dict, named: str) -> str:
    """The flux unknown a tangential electric field closes.  The walker
    names the unknown under the Curl, or the first unknown of the
    combination when the field is ideal (E = -Cross(v, B0) alone); on the
    C-grid the condition masks the wall circulations of the induction
    equation, so it belongs to the involution unknown of that equation."""
    flux = record["involutions"]
    if named in flux:
        return named
    if len(flux) == 1:
        return next(iter(flux))
    raise NotImplementedError(
        f"staggered: the tangential electric field {bct.name!r} names {named!r}, "
        f"which is not a flux unknown, and the system carries {sorted(flux)} "
        f"involution unknowns; the condition masks the wall circulations of one "
        f"induction equation, so write it with the Curl of that unknown")


def _wall_values(ops: "_CGridOps", component, location: str, kind: str) -> np.ndarray:
    """A boundary coefficient on the wall nodes at ``location``: the theta
    centres (``kind`` 'centers', the wall row of the r-face grid) or the
    theta edges ('edges', of the corner grid)."""
    loc = "RC" if kind == "centers" else "EE"
    vals = ops.evaluate(component, loc).reshape(len(ops.grids.r_edge), -1)
    return vals[-1 if location == Boundary.R_OUTER else 0]


def _bound_row(bct: BCTask, params) -> BCTask:
    """A composite row with its parameters bound; unbound symbols are
    refused naming them."""
    try:
        (bound,) = bind_bc_parameters([bct], params or {})
    except ValueError as exc:
        raise NotImplementedError(
            f"staggered: boundary condition {bct.name!r} {exc}") from exc
    return bound


def _composite_closure(system, bct: BCTask, m: int, ops: "_CGridOps", record: dict,
                       params) -> tuple:
    """The C-grid reading of a BC_LINEAR_COMBINATION row (docs/
    ir_boundary_and_constraints.md): the coefficients are evaluated at the
    wall nodes and the row becomes one of

        ('robin', s, g)         c_N d_r s + c_D s = 0 on a centre scalar: the
                                mirror ghost s_g = s_0 (2 c_N - sigma c_D h) /
                                (2 c_N + sigma c_D h), sigma = +1 (outer) /
                                -1 (inner), so the wall flux is g s_0 with
                                g = -c_D / (c_N + sigma c_D h / 2) per theta
                                centre (c_N = 0 is the Dirichlet closure
                                -2 sigma s_0 / h, c_D = 0 the natural one)
        ('slip', u, tangent, ls)
                                c_D Dot(t_hat, u) + c_T scale Dot(t_hat,
                                Dot(strain, r_hat)) = 0: the Navier slip
                                closure with ls = c_T scale / c_D per theta
                                edge (theta_hat, e_rtheta) or centre
                                (phi_hat, e_rphi)
        ('normal_stress', u, p, kappa)
                                c_T scale Dot(r_hat, Dot(strain, r_hat)) +
                                c_p p = 0 on the multiplier p of Div(u) = 0:
                                the wall v_r faces stay degrees of freedom
                                with 2 e_rr = kappa p, kappa = -c_p /
                                (c_T scale) per theta centre

    and anything else is refused naming the term."""
    bct = _bound_row(bct, params)
    terms = bct.metadata["terms"]
    kinds = [t.kind for _c, t in terms]
    unknowns = bct.metadata["unknowns"]
    placement = record["placement"]
    w = bct.location

    def refuse(term, why):
        raise NotImplementedError(
            f"staggered: boundary condition {bct.name!r} combines "
            f"{[t.kind for _c, t in terms]}; the C-grid closes a composite row "
            f"as Robin on a centre scalar (Dot(r_hat, Grad(s)) + h s = 0), Navier "
            f"slip (Dot(t_hat, u) + ls Dot(t_hat, Dot(strain, r_hat)) = 0) or the "
            f"normal stress (Dot(r_hat, Dot(strain, r_hat)) - c p = 0); the term "
            f"{term.kind!r} on {term.unknown_name!r} {why}")

    if len(unknowns) == 1 and sorted(kinds) == sorted(
            [BC_NEUMANN_SCALAR_RADIAL, BC_DIRICHLET_SCALAR]):
        (s_,) = unknowns
        if placement[s_] != "center":
            refuse(terms[0][1], f"is not on a centre scalar ({placement[s_]!r})")
        c_n = c_d = None
        for c, t in terms:
            vals = _wall_values(ops, c, w, "centers")
            if t.kind == BC_NEUMANN_SCALAR_RADIAL:
                c_n = vals
            else:
                c_d = vals
        sigma = 1.0 if w == Boundary.R_OUTER else -1.0
        den = c_n + sigma * c_d * ops.hr / 2.0
        if np.any(den == 0):
            refuse(terms[0][1], "has a vanishing Robin denominator c_N + sigma c_D h / 2")
        return "robin", s_, -c_d / den
    if len(unknowns) == 1 and len(terms) == 2 and sorted(kinds) == sorted(
            [BC_DIRICHLET_VECTOR_PROJECTION, BC_SCALAR_LINEAR_FROM_TENSOR]):
        (u,) = unknowns
        proj = next(t for _c, t in terms if t.kind == BC_DIRICHLET_VECTOR_PROJECTION)
        tens = next(t for _c, t in terms if t.kind == BC_SCALAR_LINEAR_FROM_TENSOR)
        shape = _traction_shape(system, tens, m)
        if placement[u] != "face":
            refuse(proj, f"is not on a face unknown ({placement[u]!r})")
        if shape is None or proj.unit == BC_UNIT_R_HAT:
            refuse(tens, "is not the tangential traction Dot(t_hat, Dot(strain, r_hat)) "
                         "beside Dot(t_hat, u) on the same unit")
        kind, scale = shape
        if kind != {BC_UNIT_THETA_HAT: "theta", BC_UNIT_PHI_HAT: "phi"}[proj.unit]:
            refuse(tens, f"projects on {kind}_hat while the velocity term projects on "
                         f"{proj.unit}")
        nodes = "edges" if kind == "theta" else "centers"
        c_d = _wall_values(ops, next(c for c, t in terms if t is proj), w, nodes)
        c_t = _wall_values(ops, next(c for c, t in terms if t is tens), w, nodes) * scale
        if np.any(np.abs(c_d.imag) > 0) or np.any(np.abs(c_t.imag) > 0):
            refuse(tens, "has a complex slip length")
        with np.errstate(divide="ignore", invalid="ignore"):
            ls = np.where(c_d.real == 0, np.inf, c_t.real / c_d.real)
        if np.any(np.isnan(ls)):
            refuse(tens, "vanishes with the velocity term at a wall node")
        return "slip", u, kind, ls
    if len(unknowns) == 2 and len(terms) == 2 and sorted(kinds) == sorted(
            [BC_SCALAR_LINEAR_FROM_TENSOR, BC_DIRICHLET_SCALAR]):
        tens = next(t for _c, t in terms if t.kind == BC_SCALAR_LINEAR_FROM_TENSOR)
        pres = next(t for _c, t in terms if t.kind == BC_DIRICHLET_SCALAR)
        u, p = tens.unknown_name, pres.unknown_name
        shape = _traction_shape(system, tens, m)
        if shape is None or shape[0] != "r":
            refuse(tens, "is not the normal traction Dot(r_hat, Dot(strain, r_hat))")
        eq = record["multipliers"].get(p)
        if eq is None or record["equations"][eq]["vector"] != u:
            refuse(pres, f"is not the multiplier of Div({u}) = 0")
        c_t = _wall_values(ops, next(c for c, t in terms if t is tens), w, "centers")
        c_t = c_t * shape[1]
        c_p = _wall_values(ops, next(c for c, t in terms if t is pres), w, "centers")
        if np.any(c_t == 0):
            refuse(tens, "vanishes at a wall node")
        return "normal_stress", u, p, -c_p / c_t
    refuse(terms[0][1], "is not one of these")


def _classify_walls(system, m: int, record: dict, ops: "_CGridOps", params=None) -> dict:
    """Wall record from the BC tasks (composite rows included, docs/
    ir_boundary_and_constraints.md). Per face unknown: the walls carrying
    Dot(r_hat, u) = 0 (``normal``), the tangential closure ('stress_free',
    'noslip', a ``TangentialWall`` with the slip lengths of a Navier-slip
    or mixed closure, 'electric' for the tangential electric field of a
    conducting wall, or None), the ``open`` walls, {location: (kappa,
    multiplier)}, whose v_r faces stay degrees of freedom under the
    normal-stress row 2 e_rr = kappa p, and the ``kind``: 'impenetrable'
    (u_r = 0 on the closed wall faces, eliminated or overwritten),
    'conducting' (u_r = 0 and the wall circulations of the electric field
    masked; ``electric`` holds the terms per wall and unit) or 'potential'
    (tangential Dirichlet conditions only: the wall u_r DOFs stay, the
    tangential components close the curl stencils as 'noslip') or 'free'
    (an involution unknown with no wall condition at all: its wall fluxes
    stay as DOFs and evolve with the unmasked wall circulations of the
    electric field). Per center unknown the (inner, outer) flux closure
    ('dirichlet' / 'neumann' / 'robin', the natural Neumann closure when a
    wall has no condition) and the Robin ghost coefficients per wall; a
    multiplier may carry Dirichlet walls (the gradient wall rows of a
    potential). Anything else is refused."""
    is_ball = ops.is_ball
    walls = (Boundary.R_OUTER,) if is_ball else (Boundary.R_INNER, Boundary.R_OUTER)
    placement = record["placement"]
    try:
        bc_tasks = walk_boundary_conditions(system, m, composite=True)
    except NotImplementedError as exc:
        raise NotImplementedError(
            "staggered: a boundary condition is not one of the C-grid wall "
            "conditions (Dot(r_hat, u) = 0, Dot(theta_hat, u) = Dot(phi_hat, u) "
            "= 0, the stress-free tractions, Navier slip, the normal stress, the "
            "tangential electric field Dot(t_hat, c1 Curl(b) + c2 Cross(v, B0)) = "
            f"0, s = 0, Dot(r_hat, Grad(s)) = 0, Robin); the walker reports: {exc}"
        ) from exc
    normal = {u: set() for u, p in placement.items() if p == "face"}
    tangential = {u: {w: set() for w in walls} for u in normal}
    slip = {u: {w: {} for w in walls} for u in normal}
    open_ = {u: {} for u in normal}
    electric = {u: {w: {} for w in walls} for u in normal}
    closure = {u: {} for u, p in placement.items() if p == "center"}
    robin = {u: {} for u in closure}

    def scalar_closure(bct, u, kind, g=None):
        if u in record["multipliers"] and kind != "dirichlet":
            raise NotImplementedError(
                f"staggered: boundary condition {bct.name!r} constrains the "
                f"multiplier {u!r} by a {kind.title()} condition; a multiplier takes "
                f"the natural closure or a Dirichlet wall (the electric "
                f"potential of a vector potential)")
        prev = closure[u].get(bct.location)
        if prev is not None and prev != kind:
            raise NotImplementedError(
                f"staggered: {u!r} has both {prev} and {kind} conditions at "
                f"{bct.location!r}")
        closure[u][bct.location] = kind
        if g is not None:
            robin[u][bct.location] = g

    for bct in bc_tasks:
        u = bct.unknown_name
        if bct.location not in walls:
            raise NotImplementedError(
                f"staggered: boundary condition {bct.name!r} sits at "
                f"{bct.location!r}, which is not a wall of this "
                f"{'ball' if is_ball else 'shell'} (walls: {walls})")
        if bct.value is not None:
            raise NotImplementedError(
                f"staggered: boundary condition {bct.name!r} is inhomogeneous; "
                f"the C-grid wall closures are homogeneous")
        if placement.get(u, "").startswith("tensor"):
            raise NotImplementedError(
                f"staggered: boundary condition {bct.name!r} constrains the "
                f"tensor unknown {u!r}; tensor placements carry no wall condition")
        if bct.kind == BC_DIRICHLET_VECTOR_PROJECTION:
            if placement[u] != "face":
                raise NotImplementedError(
                    f"staggered: {bct.name!r} projects {u!r}, whose placement "
                    f"is {placement[u]!r}, not 'face'")
            if bct.unit == BC_UNIT_R_HAT:
                normal[u].add(bct.location)
            elif bct.unit in (BC_UNIT_THETA_HAT, BC_UNIT_PHI_HAT):
                tangential[u][bct.location].add(bct.unit)
        elif bct.kind == BC_SCALAR_LINEAR_FROM_TENSOR:
            shape = _traction_shape(system, bct, m)
            if shape is None:
                raise NotImplementedError(
                    f"staggered: boundary condition {bct.name!r} is a tensor "
                    f"projection other than the stress-free traction "
                    f"Dot(t_hat, Dot(Grad(u) + Transpose(Grad(u)), r_hat)); "
                    f"tensor projections beyond the strain are refused")
            kind = shape[0]
            if kind == "r":
                open_[u][bct.location] = (np.zeros(len(ops.grids.theta_center),
                                                   dtype=complex), None)
            else:
                tangential[u][bct.location].add(f"traction_{kind}")
        elif bct.kind == BC_PROJECTION_CURL_CROSS:
            u = _electric_target(bct, record, u)
            if placement[u] != "face":
                raise NotImplementedError(
                    f"staggered: {bct.name!r} projects the curl of {u!r}, whose "
                    f"placement is {placement[u]!r}, not 'face'")
            if bct.unit == BC_UNIT_R_HAT:
                raise NotImplementedError(
                    f"staggered: boundary condition {bct.name!r} is the radial "
                    f"component of an electric field; the C-grid masks the "
                    f"tangential wall circulations only (theta_hat and phi_hat)")
            terms = []
            for term in bct.metadata["terms"]:
                c = smp.sympify(term["coefficient"])
                if params:
                    c = c.subs(params)
                if c.free_symbols:
                    raise NotImplementedError(
                        f"staggered: boundary condition {bct.name!r} carries the "
                        f"unbound symbols {sorted(map(str, c.free_symbols))}")
                terms.append(dict(term, coefficient=complex(c)))
            electric[u][bct.location][bct.unit] = terms
        elif bct.kind in (BC_DIRICHLET_SCALAR, BC_NEUMANN_SCALAR_RADIAL):
            kind = "dirichlet" if bct.kind == BC_DIRICHLET_SCALAR else "neumann"
            scalar_closure(bct, u, kind)
        elif bct.kind == BC_LINEAR_COMBINATION:
            read = _composite_closure(system, bct, m, ops, record, params)
            if read[0] == "robin":
                scalar_closure(bct, read[1], "robin", read[2])
            elif read[0] == "slip":
                _, u, kind, ls = read
                tangential[u][bct.location].add(f"slip_{kind}")
                slip[u][bct.location][kind] = ls
            else:
                _, u, p, kappa = read
                open_[u][bct.location] = (kappa, p)
        else:
            raise NotImplementedError(
                f"staggered: boundary condition {bct.name!r} of kind "
                f"{bct.kind!r} has no C-grid treatment")
    out = {"vectors": {}, "scalars": {}}
    for u in normal:
        missing = [w for w in walls if w not in normal[u] and w not in open_[u]]
        both = {BC_UNIT_THETA_HAT, BC_UNIT_PHI_HAT}
        if missing and not normal[u] and not open_[u] \
                and all(tangential[u][w] == both for w in walls) \
                and not any(electric[u][w] for w in walls):
            out["vectors"][u] = {"normal": (), "tangential": "noslip",
                                 "kind": "potential", "electric": None, "open": {}}
            continue
        if missing and not normal[u] and not open_[u] and u in record["involutions"] \
                and not any(tangential[u][w] or electric[u][w] for w in walls):
            out["vectors"][u] = {"normal": (), "tangential": None,
                                 "kind": "free", "electric": None, "open": {}}
            continue
        if missing:
            raise NotImplementedError(
                f"staggered: face unknown {u!r} needs Dot(r_hat, {u}) = 0 at "
                f"every wall (or Dot(theta_hat, {u}) = Dot(phi_hat, {u}) = 0 at "
                f"every wall and no normal condition, a vector potential); "
                f"missing at {missing}")
        for w in open_[u]:
            if w in normal[u]:
                raise NotImplementedError(
                    f"staggered: {u!r} carries both Dot(r_hat, {u}) = 0 and the "
                    f"normal stress at {w!r}; the wall is closed or open")
        per_wall = {}
        for w in walls:
            got = tangential[u][w]
            el = electric[u][w]
            if el and got:
                raise NotImplementedError(
                    f"staggered: {u!r} carries both tangential Dirichlet / "
                    f"traction conditions {sorted(got)} and a tangential electric "
                    f"field at {w!r}")
            if el:
                if set(el) != both:
                    raise NotImplementedError(
                        f"staggered: the tangential electric field on {u!r} at "
                        f"{w!r} needs both theta_hat and phi_hat projections; "
                        f"got {sorted(el)}")
                per_wall[w] = "electric"
            elif not got:
                per_wall[w] = None
            elif got == both:
                per_wall[w] = "noslip"
            elif got == {"traction_theta", "traction_phi"}:
                per_wall[w] = "stress_free"
            else:
                lengths = {}
                for kind, unit in (("theta", BC_UNIT_THETA_HAT), ("phi", BC_UNIT_PHI_HAT)):
                    tags = got & {unit, f"traction_{kind}", f"slip_{kind}"}
                    if len(tags) != 1:
                        raise NotImplementedError(
                            f"staggered: tangential conditions {sorted(got)} on "
                            f"{u!r} at {w!r} are neither no-slip (theta_hat and "
                            f"phi_hat), stress-free (both tractions) nor one "
                            f"condition per tangential component (Dirichlet, "
                            f"traction or Navier slip on theta_hat and on phi_hat)")
                    (tag,) = tags
                    lengths[kind] = (0.0 if tag == unit else np.inf
                                     if tag.startswith("traction") else slip[u][w][kind])
                per_wall[w] = lengths
        strings = {k for k in per_wall.values() if not isinstance(k, dict)}
        mixed = [w for w, k in per_wall.items() if isinstance(k, dict)]
        if "electric" in strings and (len(strings) > 1 or mixed):
            raise NotImplementedError(
                f"staggered: mixed tangential closures {per_wall} on {u!r}; the "
                f"electric closure is one kind for both walls")
        if None in strings and (len(strings) > 1 or mixed or open_[u]):
            raise NotImplementedError(
                f"staggered: {u!r} carries a tangential closure at one wall and "
                f"none at the other; the wall closure is one kind for both walls")
        if len(strings) == 1 and not mixed and not open_[u]:
            kind = strings.pop()
        else:
            def lengths_of(w):
                k = per_wall[w]
                if isinstance(k, dict):
                    return (tuple(map(float, np.atleast_1d(k["theta"]))),
                            tuple(map(float, np.atleast_1d(k["phi"]))))
                return 0.0 if k == "noslip" else np.inf
            inner = lengths_of(Boundary.R_INNER) if not is_ball else 0.0
            kind = TangentialWall(
                slip=(inner, lengths_of(Boundary.R_OUTER)),
                open=(Boundary.R_INNER in open_[u], Boundary.R_OUTER in open_[u]))
        out["vectors"][u] = {
            "normal": tuple(w for w in walls if w not in open_[u]),
            "tangential": kind, "open": dict(open_[u]),
            "kind": "conducting" if kind == "electric" else "impenetrable",
            "electric": {w: electric[u][w] for w in walls} if kind == "electric" else None,
        }
    for s_ in closure:
        inner = closure[s_].get(Boundary.R_INNER, "neumann")
        outer = closure[s_].get(Boundary.R_OUTER, "neumann")
        out["scalars"][s_] = {
            "closure": (inner, outer),
            "natural": tuple(w for w in walls if w not in closure[s_]),
            "robin": dict(robin[s_]),
        }
    return out


def _demote_pinned_involutions(record: dict, walls: dict, by_eq: dict) -> None:
    """An implied involution (a face unknown whose evolution is a curl,
    with no written Div = 0) that is the velocity of its own induction
    term, Dt(u) = Curl(Cross(u, B0)) + ..., and carries a tangential
    Dirichlet or traction closure is a velocity with a no-slip or
    stress-free wall, not a flux: the closure belongs to the advected
    vector, which here is the unknown itself. It is read as a plain
    evolution unknown, Lap(u) the vector Laplacian and Curl(Cross(u, B0))
    the incidence curl divided by the face areas. A field transported by
    another unknown keeps the involution reading, so pinned tangential
    components on it are still refused (section 5), as is a written
    Div(b) = 0 with them."""
    written = {info["vector"] for info in record["equations"].values()
               if info["class"] == "involution"}
    for u, u_eq in list(record["involutions"].items()):
        tangential = walls["vectors"][u]["tangential"]
        if u in written or not (isinstance(tangential, TangentialWall)
                                or tangential in ("noslip", "stress_free")):
            continue
        if any(t.kind == TASK_CURL_CROSS_UNK_BG and t.unknown_name == u for t in by_eq[u_eq]):
            del record["involutions"][u]
            del record["constraints"][u]


def _check_involution_walls(record: dict, walls: dict, by_eq: dict):
    """The wall record of every involution unknown b: 'impenetrable' with
    no tangential closure (b_r = 0, the radial wall) or 'conducting' with
    the tangential electric field of its own induction equation at both
    walls (the same combination c_curl Curl(b) + c_cross Cross(v, B0) on
    theta_hat and phi_hat, proportional to the field a Cross(v, B0)
    - l Curl(b) the evolution Dt(b) + a Curl(Cross(v, B0)) + l Lap(b)
    carries), or 'free' (no wall condition) when the evolution is ideal:
    the wall fluxes then evolve with the wall circulations of
    Cross(v, B0), read from the wall-adjacent velocity, while a resistive
    wall current has no closure on the one-sided corner stencils. Pinned
    tangential components are refused."""
    for b, u_eq in record["involutions"].items():
        rec = walls["vectors"][b]
        if rec["kind"] == "impenetrable" and rec["tangential"] is None:
            continue
        if rec["kind"] == "free":
            diffusion = [t for t in by_eq[u_eq] if t.target == TARGET_A
                         and t.unknown_name == b
                         and t.kind in (TASK_LAP_VECTOR, TASK_CURL_CURL_VECTOR)]
            if diffusion:
                raise NotImplementedError(
                    f"staggered: the flux unknown {b!r} carries no wall condition "
                    f"and {u_eq!r} carries the diffusion {describe_tasks(diffusion)}; a "
                    f"free wall keeps the wall fluxes as DOFs, whose resistive "
                    f"tendency needs the wall current the one-sided corner "
                    f"stencils do not close. Write Dot(r_hat, {b}) = 0 (radial "
                    f"wall) or the tangential electric field (conducting wall)")
            continue
        if rec["kind"] != "conducting":
            raise NotImplementedError(
                f"staggered: the tangential condition {rec['tangential']!r} on the "
                f"flux unknown {b!r} pins its tangential components at the wall, "
                f"where the face fluxes carry them half a cell inside; a "
                f"Dirichlet condition on the tangential fluxes over-determines "
                f"the poloidal field of the involution. Constrained transport "
                f"takes Dot(r_hat, {b}) = 0 alone (radial wall) or with the "
                f"tangential electric field Dot(t_hat, eta Curl({b}) - "
                f"Cross(v, B0)) = 0 (conducting wall)")
        a = l = 0.0
        v_name = None
        for t in by_eq[u_eq]:
            if t.target != TARGET_A:
                continue
            c = _constant_coefficient(t)
            if t.kind == TASK_CURL_CROSS_UNK_BG:
                a += t.sign * c * (1.0 if t.metadata.get("inner_side") == "right" else -1.0)
                v_name = t.unknown_name
            elif t.kind == TASK_LAP_VECTOR:
                l += t.sign * c
            elif t.kind == TASK_CURL_CURL_VECTOR:
                l -= t.sign * c
        for w, units in rec["electric"].items():
            combos = []
            for unit, terms in units.items():
                c_curl = c_cross = 0.0
                for term in terms:
                    if term["kind"] == "curl" and term["unknown"] == b:
                        c_curl += term["coefficient"]
                    elif term["kind"] == "cross" and term["unknown"] == v_name:
                        c_cross += term["coefficient"] * (
                            1.0 if term["side"] == "right" else -1.0)
                    else:
                        raise NotImplementedError(
                            f"staggered: the tangential condition on {b!r} at {w!r} "
                            f"carries {term['kind']}({term['unknown']}), which is "
                            f"not part of the electric field of {u_eq!r}")
                combos.append((c_curl, c_cross))
            scale = max(abs(a), abs(l), 1e-300)
            for c_curl, c_cross in combos:
                ref = max(abs(c_curl), abs(c_cross), 1e-300)
                if abs(c_cross * l + c_curl * a) > 1e-12 * ref * scale:
                    raise NotImplementedError(
                        f"staggered: the tangential condition on {b!r} at {w!r}, "
                        f"{c_curl} Curl({b}) + {c_cross} Cross({v_name}, B0), is not "
                        f"the electric field {a} Cross({v_name}, B0) - {l} Curl({b}) "
                        f"of {u_eq!r}; the C-grid masks the total tangential "
                        f"electric field at a conducting wall")
            if combos[0] != combos[1]:
                raise NotImplementedError(
                    f"staggered: the theta_hat and phi_hat conditions on {b!r} at "
                    f"{w!r} are different combinations {combos}")


def _constant_coefficient(task: OperatorTask) -> complex:
    """The constant coefficient of a task (1 for None); a varying
    coefficient is refused."""
    if task.coefficient is None:
        return 1.0 + 0j
    expr = getattr(task.coefficient, "expr", None)
    if expr is not None and expr.is_number:
        return complex(expr)
    R = np.linspace(0.3, 0.9, 4)[:, None] + np.zeros((1, 3))
    T = np.linspace(0.4, 2.7, 3)[None, :] + np.zeros((4, 1))
    v = np.asarray(task.coefficient.evaluate_on_grid(R, T)) + 0.0 * R
    if not np.allclose(v, v.flat[0], rtol=1e-12, atol=0.0):
        raise NotImplementedError(
            f"staggered: the coefficient of {task.kind!r} on {task.unknown_name!r} "
            f"in {task.equation_name!r} varies in space; the wall check of the "
            f"electric field takes constant coefficients")
    return complex(v.flat[0])


# ---------------------------------------------------------------------------
# C-grid operator toolkit: norms, gradient, Hodge pairs, tensor maps
# ---------------------------------------------------------------------------

def _guard(x, zero=1e-14):
    x = np.asarray(x)
    nz = np.abs(x) > zero
    return np.where(nz, 1.0 / np.where(nz, x, 1.0), 0.0)


def _log_gradient_weight(grids: StaggeredGrids, a_fn, rule: str):
    """A positive weight w(r) with d ln w / dr = a_r, anchored at
    w(r_edge[0]) = 1, on the union of the edge and centre radii.
    ``rule`` 'trapezoid' is the second-order P0 recovery of the thermal
    Laplacian, which the entrywise certification against the seed
    operator pins (the weight enters only through edge / centre ratios);
    'quadrature' integrates each interval adaptively so the weight matches
    the exact density to rounding, which the viscous Gram reads."""
    r_all = np.unique(np.concatenate([grids.r_edge, grids.r_center]))
    half_pi = np.array([[np.pi / 2]])
    if rule == "trapezoid":
        a = np.asarray(a_fn(r_all[:, None], half_pi)).ravel().real
        lnw = np.concatenate([[0.0], np.cumsum(
            0.5 * (a[1:] + a[:-1]) * np.diff(r_all))])
    else:
        def f(x):
            return float(np.asarray(a_fn(np.array([[x]]), half_pi)).ravel().real[0])
        parts = [quad(f, r_all[i], r_all[i + 1], epsabs=0.0, epsrel=2e-14,
                      limit=200)[0] for i in range(len(r_all) - 1)]
        lnw = np.concatenate([[0.0], np.cumsum(parts)])

    def weight(R, T):
        R = np.asarray(R, float)
        ln = np.interp(R.ravel(), r_all, lnw).reshape(R.shape)
        return np.exp(ln) + 0.0 * np.asarray(T)
    return weight


class _CGridOps:
    """Sub-grids, volume norms, the gradient blocks, the interpolation
    pairs and the tensor maps of one (grids, m, order). ``hodge(src, dst)``
    returns the interpolation between two sub-grids: the base map for
    (CC, RC), (CC, CE), (CE, RC), (CE, EE), (RC, EE), (CC, EE) and the norm
    adjoint W_dst^-1 P^T W_src of the base map for the reverse directions,
    so a pair of blocks built from ``hodge(a, b)`` and ``hodge(b, a)`` is
    exactly W-adjoint. Every base map lands on RC, CE or EE, so the ball
    origin ring (weight 0) only ever receives a base interpolation and is
    never a source of an adjoint; the adjoints divide by the destination
    norm, whose weightless pole edges are guarded to zero (inert DOFs)."""

    def __init__(self, grids: StaggeredGrids, m: int, order: int,
                 pole_closure: bool = False):
        self.grids, self.m, self.order = grids, m, order
        self.pole_closure = bool(pole_closure)
        re, rc = grids.r_edge, grids.r_center
        te, tc = grids.theta_edge, grids.theta_center
        Nr_e, Nr_c, Nt_e, Nt_c = len(re), len(rc), len(te), len(tc)
        self.hr = re[1] - re[0]
        self.ht = tc[1] - tc[0] if Nt_c > 1 else np.pi
        self.nodes = {"CC": (rc, tc), "RC": (re, tc), "CE": (rc, te), "EE": (re, te)}
        self.n = {k: len(r) * len(t) for k, (r, t) in self.nodes.items()}
        self.W = {k: self._vol(r, t) for k, (r, t) in self.nodes.items()}
        self.is_ball = bool(np.isclose(re[0], 0.0))
        G_r, G_t, G_p = _build_staggered_gradient_blocks(grids, m, order)
        self.G = {"RC": G_r, "CE": G_t, "CC": G_p}
        I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
        I_te = sp.eye(Nt_e, format="csr", dtype=complex)
        I_rc = sp.eye(Nr_c, format="csr", dtype=complex)
        I_re = sp.eye(Nr_e, format="csr", dtype=complex)
        if order <= 2:
            Pr_c2e = _avg_1d(Nr_e, Nr_c)
            Pt_c2e = _avg_c2e_zero_ends(Nt_e)
            Pr_e2c = _avg_e2c(Nr_c)
            Pt_e2c = _avg_e2c(Nt_c)
        else:
            Pr_c2e = _stag_interp_1d_lagrange(rc, re, order)
            Pt_c2e = _stag_interp_1d_lagrange(tc, te, order)
            Pr_e2c = _stag_interp_1d_lagrange(re, rc, order)
            Pt_e2c = _stag_interp_1d_lagrange(te, tc, order)
        self.P1 = {"Pr_c2e": Pr_c2e, "Pt_c2e": Pt_c2e, "Pr_e2c": Pr_e2c,
                   "Pt_e2c": Pt_e2c}
        self._base = {
            ("CC", "RC"): sp.kron(Pr_c2e, I_tc, format="csr"),
            ("CC", "CE"): sp.kron(I_rc, Pt_c2e, format="csr"),
            ("CE", "RC"): sp.kron(Pr_c2e, Pt_e2c, format="csr"),
            ("CE", "EE"): sp.kron(Pr_c2e, I_te, format="csr"),
            ("RC", "EE"): sp.kron(I_re, Pt_c2e, format="csr"),
            ("CC", "EE"): sp.kron(Pr_c2e, Pt_c2e, format="csr"),
        }
        self.inv_W = {k: _guard(w, 0.0) for k, w in self.W.items()}
        self._avg = {
            ("RC", "CC"): sp.kron(Pr_e2c, I_tc, format="csr"),
            ("CE", "CC"): sp.kron(I_rc, Pt_e2c, format="csr"),
        }
        self._strain = {}
        self._grad9 = {}
        self._ct = None
        self._curl = {}

    def _vol(self, rv, tv):
        a = (rv**2)[:, None] * np.sin(tv)[None, :]
        return a.ravel().astype(complex) * self.hr * self.ht

    def hodge(self, src: str, dst: str):
        if src == dst:
            return sp.eye(self.n[src], format="csr", dtype=complex)
        if (src, dst) in self._base:
            return self._base[(src, dst)]
        P = self._base[(dst, src)]
        return sp.diags(self.inv_W[dst]) @ P.conj().T @ sp.diags(self.W[src])

    def average(self, src: str, dst: str):
        """Plain face-to-centre average (no adjoint partner): the centre
        value of a face field used where a block has no paired block."""
        return self._avg[(src, dst)]

    def mesh(self, loc: str):
        rv, tv = self.nodes[loc]
        R = rv[:, None] + 0.0 * tv[None, :]
        T = tv[None, :] + 0.0 * rv[:, None]
        return R, T

    def evaluate(self, component, loc: str) -> np.ndarray:
        """A background component on a sub-grid, flat (r outer, theta
        inner), complex; None evaluates to ones and a SymPy number to
        its exact value (lambdify prints 15 digits)."""
        if component is None:
            return np.ones(self.n[loc], dtype=complex)
        expr = getattr(component, "expr", None)
        if expr is not None and expr.is_number:
            return np.full(self.n[loc], complex(expr))
        R, T = self.mesh(loc)
        return np.asarray(component.evaluate_on_grid(R, T)).ravel().astype(complex)

    def face_locs(self):
        return [LOC_CODE[loc] for _, loc in PLACEMENTS["face"]]

    def tensor_locs(self, placement: str):
        return [LOC_CODE[loc] for _, loc in PLACEMENTS[placement]]

    def _wall_rows(self, closure):
        """(inner, outer) flags of the Dirichlet closure on the wall r-edges;
        the ball origin is not a wall."""
        if closure is None:
            return (False, False)
        inner, outer = closure
        return (inner == "dirichlet" and not self.is_ball, outer == "dirichlet")

    def gradient(self, closure=None):
        """The stacked centres-to-faces gradient in face component order.
        ``closure`` (inner, outer) 'dirichlet' replaces the wall rows of the
        radial block by the mirror-ghost closure of a scalar vanishing on
        the wall, -/+ 2 s_0 / hr over the half cell (the closure of
        ``_radial_flux_stencils``), which the wall rows of a potential's
        Grad(psi) carry; the plain one-sided wall rows otherwise."""
        G_r = self.G["RC"]
        inner, outer = self._wall_rows(closure)
        if inner or outer:
            Nr_e, Nr_c = len(self.grids.r_edge), len(self.grids.r_center)
            Nt_c = len(self.grids.theta_center)
            rows = G_r.tolil()
            for it in range(Nt_c):
                if inner:
                    rows.rows[it] = [it]
                    rows.data[it] = [2.0 / self.hr + 0j]
                if outer:
                    row = (Nr_e - 1) * Nt_c + it
                    rows.rows[row] = [(Nr_c - 1) * Nt_c + it]
                    rows.data[row] = [-2.0 / self.hr + 0j]
            G_r = rows.tocsr()
        return sp.vstack([G_r, self.G["CE"], self.G["CC"]], format="csr")

    def face_norm(self, closure=None):
        """The stacked face volume norm; with a Dirichlet ``closure`` the
        wall r-edges carry the half-cell weight, so the norm adjoint of
        the closed gradient is the flux-form divergence at the wall cells."""
        W_r = self.W["RC"]
        inner, outer = self._wall_rows(closure)
        if inner or outer:
            W_r = W_r.copy()
            Nt_c = len(self.grids.theta_center)
            if inner:
                W_r[:Nt_c] *= 0.5
            if outer:
                W_r[-Nt_c:] *= 0.5
        return np.concatenate([W_r, self.W["CE"], self.W["CC"]])

    def divergence(self, weight=None, closure=None):
        """Div_M = -W_c^-1 G^T (M W_v), the norm adjoint of the gradient
        under the M-weighted face norm (M = 1 without a weight); with a
        Dirichlet ``closure`` of the multiplier the pair (closed gradient,
        half-weighted norm) makes this the flux-form divergence, wall
        normal DOFs included."""
        G = self.gradient(closure)
        W_v = self.face_norm(closure)
        if weight is not None:
            W_v = np.concatenate([self.evaluate(weight, k) for k in self.face_locs()]) * W_v
        return -(sp.diags(1.0 / self.W["CC"]) @ G.conj().T @ sp.diags(W_v))

    # -- constrained transport and the curls ----------------------------------

    def ct(self) -> CTGrid:
        """The constrained-transport toolkit of this grid (face fluxes,
        edge circulations, the incidence curl and the edge curl of b)."""
        if self._ct is None:
            self._ct = CTGrid(self.grids, self.m, pole_closure=True)
        return self._ct

    def flux_divergence(self):
        """Net flux out of each cell from the stacked fluxes (r, theta, phi
        faces): the exact discrete divergence of an involution unknown."""
        return self.ct().divergence()

    def flux_scaling(self):
        """(area, 1 / area) diagonals on the stacked faces: a flux unknown
        is area times the physical component."""
        ct = self.ct()
        area = np.concatenate([ct.Area_r.ravel(), ct.Area_th.ravel(), ct.Area_ph.ravel()])
        inv = sp.block_diag([ct.iAr, ct.iAth, ct.iAph], format="csr")
        return sp.diags(area.astype(complex)), inv

    @staticmethod
    def ct_matrix(blocks, rows, cols, sizes):
        """One sparse matrix from CT blocks keyed (row, col) in the given
        row and column orders."""
        parts = []
        for i in rows:
            line = []
            for j in cols:
                blk = blocks.get((i, j))
                line.append(sp.csr_matrix((sizes[i], sizes[j]), dtype=complex)
                            if blk is None else sp.csr_matrix(blk, dtype=complex))
            parts.append(sp.hstack(line, format="csr"))
        return sp.vstack(parts, format="csr")

    def ct_sizes(self):
        ct = self.ct()
        return {"vr": ct.n_br, "vt": ct.n_bth, "vp": ct.n_bph,
                "br": ct.n_br, "bt": ct.n_bth, "bp": ct.n_bph}

    def primal_curl(self, wall_bc):
        """Faces to edges (curl_r on CE, curl_theta on RC, curl_phi on EE)
        with the ghost closure ``wall_bc`` of the tangential components; at
        |m| = 1 the pole theta-edge v_theta DOFs, pinned and not degrees of
        freedom, are read through ``_pole_vtheta_prolongation`` as the
        strain is, since the pole closure of curl_r needs the finite pole
        value (sin(theta) v_phi)' − i m v_theta = 0 there and the pole
        rows of curl_phi feed the dual curl."""
        key = ("primal", wall_bc)
        if key not in self._curl:
            Cp = _assemble_curl_blocks(self.grids, self.m, self.order,
                                       dual=False, wall_bc=wall_bc)
            if abs(int(self.m)) == 1:
                Cp = (Cp @ _pole_vtheta_prolongation(self.grids)).tocsr()
            self._curl[key] = Cp
        return self._curl[key]

    def dual_curl(self):
        """Edges back to the faces (the same differential curl on the dual
        grids)."""
        if "dual" not in self._curl:
            self._curl["dual"] = _assemble_curl_blocks(self.grids, self.m, self.order,
                                                       dual=True)
        return self._curl["dual"]

    def curl_curl(self, wall_bc):
        """Curl Curl u on the faces as the differential composition of the
        dual and the primal curl: pointwise consistent, wall rows of the
        normal component included (the vector potential keeps them)."""
        return (self.dual_curl() @ self.primal_curl(wall_bc)).tocsr()

    def curl_to_faces(self, wall_bc):
        """Curl u brought from the edges to the faces by the Hodge maps
        (curl_r CE -> RC, curl_theta RC -> CE, curl_phi EE -> CC): the
        standalone Curl(u) of a face equation."""
        Cp = self.primal_curl(wall_bc)
        n = {k: self.n[k] for k in ("RC", "CE", "EE", "CC")}
        P = sp.block_diag([self.hodge("CE", "RC"), self.hodge("RC", "CE"),
                           self.hodge("EE", "CC")], format="csr")
        assert P.shape[1] == n["CE"] + n["RC"] + n["EE"]
        return (P @ Cp).tocsr()

    # -- tensor placement -----------------------------------------------------

    def strain(self, wall_bc):
        """(S, W_T, inv_W_v): the six-component strain map e = (Grad v +
        Grad v^T) / 2 of ``build_staggered_strain`` with its wall closure,
        the tensor norm (off-diagonals twice) and the guarded inverse face
        norm, as the viscous Gram uses them."""
        if wall_bc not in self._strain:
            S, WT, inv_Wv, _ = _strain_operator_and_norms(
                self.grids, self.m, self.order, wall_bc, self.pole_closure)
            self._strain[wall_bc] = (S, WT, inv_Wv)
        return self._strain[wall_bc]

    def grad9(self, wall_bc=None, paired=False):
        """(G9, W_T9): the nine-component gradient map (Grad v)_ij at the
        locations of ``TENSOR_LOCATIONS`` (rr, tt, pp on CC; rt, tr on EE;
        rp, pr on RC; tp, pt on CE), the origin-parity stencil on a ball
        and the pole prolongation at |m| = 1, with its tensor norm (every
        component once), in one of two spellings of the connection terms.

        The placement spelling (``paired`` False) is the tensor
        placement's map, whose norm adjoint is Div(T): the connection
        terms v_r / r and cot(theta) v_theta / r on CC are the plain
        averages of the strain, so tp + pt is 2 e_theta_phi entrywise and
        the adjoint is consistent through the pole-adjacent ring and the
        ball's first cell; the wall rows of the radial shears rt, rp
        carry the closure ``wall_bc`` of the strain ('noslip': the mirror
        ghost 2 v_tan / h reading the wall cell only, with the wall faces
        at half weight; 'stress_free': d_r v_tan = v_tan / r from the wall
        cell; None: the plain one-sided rows) and the wall averages of
        tr, pr read a no-slip wall as zero. The theta-phi and phi-theta
        components have finite pole limits that only their sum, the
        strain's e_theta_phi, closes; their pole rows carry the parity
        limits, (1/r) d_theta v_phi as (1 - p) v_phi(theta_c) / (r dtheta)
        with p the tangential parity (-1)^(|m|+1), and (cot theta / r)
        v_phi, (i m / (r sin theta)) v_theta by L'Hopital, so a tensor
        unknown's pole DOFs hold the limits that ``div_tensor``'s pole
        share reads; the pole edges stay weightless.

        The paired spelling (``paired`` True) is the map of the
        contraction Dot(B0, Grad v), which takes no adjoint: the
        connection terms of the phi-row are the Hodge pairs of the
        Coriolis block with the metric factor on the pair's host grid,
        v_phi / r through the base map CC -> RC, cot(theta) v_phi / r as
        (1 / (r sin theta)) P(cos theta_c v_phi) through the base map
        CC -> CE, and v_r / r, cot(theta) v_theta / r on CC through the
        adjoint maps RC -> CC, CE -> CC, so that Dot(s phi_hat, Grad v) =
        i m v + z_hat x v holds entrywise against ``cross_vector``'s
        'center' pairing and the Doppler shift of a solid-body flow is
        exact to rounding. Those adjoint averages carry the Coriolis
        pair's O(1) value on the pole-adjacent ring (v_theta ~ sin theta
        at m = 0) and in the ball's first cell (v_r finite at the origin),
        harmless on a ring of one cell's volume under a background that
        vanishes there, but O(1 / dtheta) once divided, which is why the
        placement keeps the plain averages."""
        key = (wall_bc, bool(paired))
        if key in self._grad9:
            return self._grad9[key]
        g, m, order = self.grids, self.m, self.order
        re, rc, te, tc = g.r_edge, g.r_center, g.theta_edge, g.theta_center
        Nr_e, Nr_c, Nt_e, Nt_c = len(re), len(rc), len(te), len(tc)
        hr, ht = self.hr, self.ht
        im = 1j * m
        par_s = (-1) ** (abs(m) % 2)
        par_v = (-1) ** ((abs(m) + 1) % 2)
        I_tc = sp.eye(Nt_c, format="csr", dtype=complex)
        I_te = sp.eye(Nt_e, format="csr", dtype=complex)
        I_rc = sp.eye(Nr_c, format="csr", dtype=complex)
        I_re = sp.eye(Nr_e, format="csr", dtype=complex)

        def dg(loc, fn):
            R, T = self.mesh(loc)
            return sp.diags(np.asarray(fn(R, T)).ravel().astype(complex), format="csr")

        def inv_sin(t):
            return _guard(np.sin(t))

        def cot(t):
            return np.cos(t) * _guard(np.sin(t))

        if order <= 2:
            Dr_c2e = _stag_diff_1d(Nr_e, Nr_c, hr)
            Dr_e2c = _stag_diff_e2c(Nr_c, hr)

            def th_diff(src, dst, parity):
                if len(dst) == len(src) + 1:
                    return _stag_diff_c2e_zero_ends(len(dst), ht)
                return _stag_diff_e2c(len(dst), ht)
        else:
            Dr_c2e = _stag_diff_1d_lagrange(rc, re, order)
            Dr_e2c = _stag_diff_1d_lagrange(re, rc, order)

            def th_diff(src, dst, parity):
                return _stag_theta_diff_parity(src, dst, order, parity)
        if self.is_ball:
            Dr_c2e_v = _stag_diff_1d_origin_parity(rc, re, max(order, 2),
                                                   _origin_parity(m, "vector"))
        else:
            Dr_c2e_v = Dr_c2e
        Pr_e2c, Pr_c2e = self.P1["Pr_e2c"], self.P1["Pr_c2e"]
        Pt_e2c, Pt_c2e = self.P1["Pt_e2c"], self.P1["Pt_c2e"]
        # the wall closure of the radial shears and of the wall averages
        Dr_tan, Pr_tan = Dr_c2e_v.tolil(), Pr_c2e.tolil()
        walls = [Nr_e - 1] if self.is_ball else [0, Nr_e - 1]
        spec = normalize_wall_bc(wall_bc)
        closed = spec is not None and not paired
        if closed:
            for i in walls:
                Dr_tan.rows[i], Dr_tan.data[i] = [], []
        Dr_tan, Pr_tan = Dr_tan.tocsr(), Pr_tan.tocsr()

        def close_walls(D2, P2, kind, n_t):
            """The wall rows of the stacked d_r v_tan (D2) and wall average
            (P2) per theta node: the slip ghost v_g = rho v_0, derivative
            (1 - rho) / (r0 - rg) and average (1 + rho) / 2 (the no-slip
            mirror at ls = 0); ls = inf reads d_r v_tan = v_tan / r from the
            wall cell with the plain average."""
            if not closed:
                return D2, P2
            D2, P2 = D2.tolil(), P2.tolil()
            for i in walls:
                j0 = Nr_c - 1 if i == Nr_e - 1 else 0
                r0, rw = rc[j0], re[i]
                ls = spec.lengths(1 if i == Nr_e - 1 else 0, kind, n_t)
                rho = _slip_ratio(ls, rw, r0, hr)
                for j in range(n_t):
                    row, col = i * n_t + j, j0 * n_t + j
                    if np.isinf(ls[j]):
                        D2[row, col] = complex(1.0 / rw)
                        continue
                    D2[row, col] = complex((1.0 - rho[j]) / (r0 - (2.0 * rw - r0)))
                    P2.rows[row], P2.data[row] = [], []
                    avg = (1.0 + rho[j]) / 2.0
                    if avg != 0.0:
                        P2[row, col] = complex(avg)
            D2, P2 = D2.tocsr(), P2.tocsr()
            D2.sort_indices()
            P2.sort_indices()
            return D2, P2

        Dr_tan_te, Pr_tan_te = close_walls(sp.kron(Dr_tan, I_te, format="csr"),
                                           sp.kron(Pr_tan, I_te, format="csr"), "edges", Nt_e)
        Dr_tan_tc, Pr_tan_tc = close_walls(sp.kron(Dr_tan, I_tc, format="csr"),
                                           sp.kron(Pr_tan, I_tc, format="csr"), "centers",
                                           Nt_c)
        Dt_e2c = th_diff(te, tc, par_v)
        Dt_c2e_s = th_diff(tc, te, par_s)
        # centres to edges with the parity limits on the pole rows
        Dt_c2e_v = _stag_theta_diff_parity(tc, te, 2, par_v) if order <= 2 \
            else th_diff(tc, te, par_v)
        n_vr, n_vt, n_vp = self.n["RC"], self.n["CE"], self.n["CC"]

        def Z(loc, n):
            return sp.csr_matrix((self.n[loc], n), dtype=complex)

        inv_r_cc = dg("CC", lambda r, t: 1.0 / r + 0 * t)
        inv_r_ce = dg("CE", lambda r, t: 1.0 / r + 0 * t)
        inv_r_ee = dg("EE", lambda r, t: _guard(r) + 0 * t)
        inv_r_rc = dg("RC", lambda r, t: _guard(r) + 0 * t)
        # the pole rows of (cos theta v_phi) / sin theta and of
        # v_theta / sin theta by L'Hopital: (1 - p) / dtheta times the
        # adjacent centre / interior edge, oriented away from the pole
        pole_vp = sp.lil_matrix((Nt_e, Nt_c), dtype=complex)
        pole_vt = sp.lil_matrix((Nt_e, Nt_e), dtype=complex)
        pole_vp[0, 0] = (1 - par_v) / ht
        pole_vp[Nt_e - 1, Nt_c - 1] = -(1 - par_v) / ht
        pole_vt[0, 1] = (1 - par_v) / (2 * ht)
        pole_vt[Nt_e - 1, Nt_e - 2] = (1 - par_v) / (2 * ht)
        pt_vt = dg("CE", lambda r, t: im * inv_sin(t) / r) \
            + im * (inv_r_ce @ sp.kron(I_rc, pole_vt.tocsr(), format="csr"))
        if paired:
            pt_vp = -(dg("CE", lambda r, t: inv_sin(t) / r)
                      @ sp.kron(I_rc, Pt_c2e, format="csr")
                      @ dg("CC", lambda r, t: np.cos(t) + 0 * r))
            pp_vr = inv_r_cc @ self.hodge("RC", "CC")
            pp_vt = dg("CC", lambda r, t: cot(t) / r) @ self.hodge("CE", "CC")
        else:
            pt_vp = -(dg("CE", lambda r, t: cot(t) / r) @ sp.kron(I_rc, Pt_c2e, format="csr"))
            pp_vr = inv_r_cc @ sp.kron(Pr_e2c, I_tc, format="csr")
            pp_vt = dg("CC", lambda r, t: cot(t) / r) @ sp.kron(I_rc, Pt_e2c, format="csr")
        pt_vp = pt_vp - inv_r_ce @ sp.kron(I_rc, pole_vp.tocsr(), format="csr")
        comps = {
            "rr": [sp.kron(Dr_e2c, I_tc, format="csr"), Z("CC", n_vt), Z("CC", n_vp)],
            "tt": [inv_r_cc @ sp.kron(Pr_e2c, I_tc, format="csr"),
                   inv_r_cc @ sp.kron(I_rc, Dt_e2c, format="csr"), Z("CC", n_vp)],
            "pp": [pp_vr, pp_vt, dg("CC", lambda r, t: im * inv_sin(t) / r)],
            "rt": [Z("EE", n_vr), Dr_tan_te, Z("EE", n_vp)],
            "rp": [Z("RC", n_vr), Z("RC", n_vt), Dr_tan_tc],
            "tp": [Z("CE", n_vr), Z("CE", n_vt),
                   inv_r_ce @ sp.kron(I_rc, Dt_c2e_v, format="csr")],
            "tr": [inv_r_ee @ sp.kron(I_re, Dt_c2e_s, format="csr"),
                   -(inv_r_ee @ Pr_tan_te), Z("EE", n_vp)],
            "pr": [dg("RC", lambda r, t: im * inv_sin(t) * _guard(r)), Z("RC", n_vt),
                   -(inv_r_rc @ Pr_tan_tc)],
            "pt": [Z("CE", n_vr), pt_vt, pt_vp],
        }
        G9 = sp.vstack([sp.hstack(comps[c], format="csr") for c in TENSOR_COMPONENTS],
                       format="csr")
        if abs(int(m)) == 1:
            G9 = (G9 @ _pole_vtheta_prolongation(g)).tocsr()
        WT = []
        for c, loc in zip(TENSOR_COMPONENTS, self.tensor_locs("tensor9")):
            R, T = self.mesh(loc)
            w = self.W[loc].copy()
            w[np.abs(np.sin(T).ravel()) < 1e-12] = 0.0
            if closed and c in ("rt", "rp", "tr", "pr"):
                w = w.reshape(len(R), -1)
                for i in walls:
                    ls = spec.lengths(1 if i == Nr_e - 1 else 0,
                                      "edges" if loc == "EE" else "centers", w.shape[1])
                    w[i, :] *= _slip_face_weight(ls, hr)
                w = w.ravel()
            WT.append(w)
        self._grad9[key] = (G9, np.concatenate(WT))
        return self._grad9[key]

    def tensor_map(self, a: complex, b: complex, wall_bc, paired=False):
        """(M, placement): the map of a Grad(u) + b Transpose(Grad(u))
        from the faces onto the tensor placement: 2a S on the six
        symmetric components when a = b (the strain map with the wall
        closure), else the nine-component gradient map (the placement
        spelling, or the paired one for a contraction) and its
        transposition."""
        if a == b:
            S, _, _ = self.strain(wall_bc)
            return (2.0 * a) * S, "tensor"
        G9, _ = self.grad9(wall_bc, paired)
        return a * G9 + b * (self.transpose("tensor9") @ G9), "tensor9"

    def tensor_norms(self, placement: str, wall_bc):
        """(W_T, inv_W_v) of a tensor placement."""
        if placement == "tensor":
            _, WT, inv_Wv = self.strain(wall_bc)
            return WT, inv_Wv
        _, WT = self.grad9(wall_bc)
        _, _, inv_Wv = self.strain(wall_bc)
        return WT, inv_Wv

    def div_tensor(self, placement: str, wall_bc):
        """Div(T) = -W_v^-1 M^T W_T, the norm adjoint of the placement's
        gradient map, tensor components to the faces, plus the pole share
        of ``_strain_pole_share`` on the phi-theta component (always on
        the nine-component placement, whose pole rows carry the limits;
        on the six with ``pole_closure``)."""
        if placement == "tensor":
            S, WT, inv_Wv = self.strain(wall_bc)
            M = S
        else:
            M, WT = self.grad9(wall_bc)
            _, _, inv_Wv = self.strain(wall_bc)
        D = -(sp.diags(inv_Wv) @ M.conj().T @ sp.diags(WT))
        if placement == "tensor9" or self.pole_closure:
            comps = TENSOR_COMPONENTS[:len(self.tensor_locs(placement))]
            D = D + _strain_pole_share(self.grids, comps, "pt" if "pt" in comps else "tp")
        return D.tocsr()

    def transpose(self, placement: str):
        """The component permutation of Transpose (identity on the
        symmetric placement)."""
        locs = self.tensor_locs(placement)
        n = [self.n[k] for k in locs]
        off = np.concatenate([[0], np.cumsum(n)])
        comps = TENSOR_COMPONENTS[:len(locs)]
        rows, cols = [], []
        for i, c in enumerate(comps):
            j = comps.index(c[::-1]) if c[::-1] in comps else i
            rows.append(np.arange(off[i], off[i + 1]))
            cols.append(np.arange(off[j], off[j + 1]))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        return sp.csr_matrix((np.ones(len(rows), dtype=complex), (rows, cols)),
                             shape=(off[-1], off[-1]))

    def trace(self, placement: str):
        """Trace: the sum of the three CC diagonal components."""
        locs = self.tensor_locs(placement)
        n_cc = self.n["CC"]
        n_rest = sum(self.n[k] for k in locs[3:])
        eye = sp.eye(n_cc, format="csr", dtype=complex)
        return sp.hstack([eye, eye, eye, sp.csr_matrix((n_cc, n_rest), dtype=complex)],
                         format="csr")

    def component_slices(self, placement: str):
        locs = self.tensor_locs(placement)
        off = np.concatenate([[0], np.cumsum([self.n[k] for k in locs])])
        comps = TENSOR_COMPONENTS[:len(locs)]
        return {c: (locs[i], slice(int(off[i]), int(off[i + 1])))
                for i, c in enumerate(comps)}

    def dot_vector(self, placement: str, side: str, background):
        """The faces-valued contraction T -> Dot(B0, T) (side 'left', over
        the first index) or Dot(T, B0) ('right', over the second) of a
        tensor placement with a background vector: each component of T is
        multiplied by the background component at the tensor component's
        own location and brought to the face of the free index by the
        Hodge map from there (CC -> RC, EE -> CE, RC -> CC and their
        partners), so the block is the interpolation-paired product of
        the design. ``background(i, loc)`` returns the i-th background
        component ('r', 't', 'p') on sub-grid ``loc`` or None for a
        component that vanishes identically."""
        locs = self.tensor_locs(placement)
        comps = TENSOR_COMPONENTS[:len(locs)]
        axes = ["r", "t", "p"]
        parts = []
        for k, face in enumerate(self.face_locs()):
            row = []
            for c, loc in zip(comps, locs):
                block = None
                for i in axes:
                    comp = i + axes[k] if side == "left" else axes[k] + i
                    if comp not in comps:
                        comp = comp[::-1]
                    if comp != c:
                        continue
                    b = background(i, loc)
                    if b is None:
                        continue
                    part = self.hodge(loc, face) @ sp.diags(b)
                    block = part if block is None else block + part
                row.append(sp.csr_matrix((self.n[face], self.n[loc]), dtype=complex)
                           if block is None else block)
            parts.append(sp.hstack(row, format="csr"))
        return sp.vstack(parts, format="csr")

    def dot_radial(self, placement: str, side: str):
        """``dot_vector`` with the unit radial vector: the r-row (or
        r-column) of T brought to the faces."""
        return self.dot_vector(placement, side, lambda i, loc: np.ones(
            self.n[loc], dtype=complex) if i == "r" else None)


def _scaled(block, sign: int, values: Optional[np.ndarray]):
    """sign * coefficient * block, with a constant coefficient applied as
    one scalar multiply and a varying one as a diagonal on the rows."""
    if values is None:
        return block if sign > 0 else -block
    if np.all(values == values.flat[0]):
        c = complex(values.flat[0])
        if c == 1.0:
            return block if sign > 0 else -block
        return (sign * c) * block
    return sp.diags(sign * values) @ block


def _is_zero(values: Optional[np.ndarray]) -> bool:
    return values is not None and not np.any(values)


def _constant(values: Optional[np.ndarray]):
    """The constant value of an evaluated coefficient (1 for None), or
    None when it varies."""
    if values is None:
        return 1.0 + 0j
    if np.all(values == values.flat[0]):
        return complex(values.flat[0])
    return None


def _is_radial(field) -> bool:
    """True when the background vector has no theta and phi components."""
    R = np.linspace(0.4, 0.9, 3)[:, None] + np.zeros((1, 5))
    T = np.linspace(0.3, 2.8, 5)[None, :] + np.zeros((3, 1))
    for name in ("v_theta", "v_phi"):
        v = np.asarray(field.components[name].evaluate_on_grid(R, T)) + 0.0 * R
        if np.abs(v).max() > 0.0:
            return False
    return True


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

class _Assembly:
    """Accumulates the task blocks of one compile into A, B and b."""

    def __init__(self, system, tasks, ops: _CGridOps, layout: StaggeredLayout,
                 record: dict, walls: dict, coriolis_pairing: str):
        self.system, self.ops, self.layout = system, ops, layout
        self.record, self.walls = record, walls
        self.pairing = coriolis_pairing
        self.placement = record["placement"]
        self.N = layout.total_size
        self.by_eq = {eq.name: [] for eq in system.equations}
        for t in tasks:
            self.by_eq.setdefault(t.equation_name, []).append(t)
        self.A_parts, self.B_parts = [], []
        self.b = np.zeros(self.N, dtype=complex)
        self.has_source = False
        self.consumed = set()
        self.r_power = {eq.name: 0 for eq in system.equations}
        self.second_order = {u: False for u, p in self.placement.items() if p == "face"}
        self._defs = {}

    # -- DOF bookkeeping ----------------------------------------------------

    def blocks(self, name: str):
        """[(sub-grid code, offset, size)] of an unknown's DOF blocks."""
        out = []
        for suffix, loc in PLACEMENTS[self.placement[name]]:
            f = self.layout.field(name + suffix)
            out.append((LOC_CODE[loc], f.offset, f.size))
        return out

    def offset(self, name: str) -> int:
        return self.blocks(name)[0][1]

    def size(self, name: str) -> int:
        return sum(b[2] for b in self.blocks(name))

    def locs(self, name: str):
        return [LOC_CODE[loc] for _, loc in PLACEMENTS[self.placement[name]]]

    def coefficient_on(self, task: OperatorTask, name: str):
        """The task coefficient on the DOFs of ``name`` (None for 1)."""
        if task.coefficient is None:
            return None
        return np.concatenate([self.ops.evaluate(task.coefficient, loc)
                               for loc in self.locs(name)])

    def add(self, target: str, row: str, col: str, block):
        parts = self.B_parts if target == TARGET_B else self.A_parts
        parts.append((self.offset(row), self.offset(col), sp.csr_matrix(block)))

    def add_to(self, target: str, row: str, col: str, block):
        """A block on the columns of ``col``, composed with the definition
        when ``col`` is an eliminated tensor."""
        if col in self.record["definitions"]:
            Def, vec = self.definition(col)
            self.add(target, row, vec, block @ Def)
        else:
            self.add(target, row, col, block)

    def require(self, task, row, col, row_placement, col_placement):
        if (self.placement[row] != row_placement
                or self.placement[col] != col_placement):
            raise NotImplementedError(
                f"staggered: no block of kind {task.kind!r} from placement "
                f"{self.placement[col]!r} ({col!r}) to placement "
                f"{self.placement[row]!r} ({row!r}); built for "
                f"{col_placement!r} -> {row_placement!r}")

    def background(self, task: OperatorTask, loc: str, component: str):
        """coefficient * source_field component on a sub-grid."""
        vals = self.ops.evaluate(task.source_field.components[component], loc)
        if task.coefficient is not None:
            vals = vals * self.ops.evaluate(task.coefficient, loc)
        return vals

    def wall_bc(self, u: str, what: str):
        rec = self.walls["vectors"][u]
        if rec["kind"] == "potential":
            raise NotImplementedError(
                f"staggered: {what} is the strain Gram, whose wall rows of the "
                f"normal component are eliminated by Dot(r_hat, {u}) = 0; {u!r} "
                f"keeps its wall normal DOFs (a vector potential), so its "
                f"Laplacian is -Curl Curl {u} on the constraint manifold")
        wall_bc = rec["tangential"]
        if wall_bc is None or wall_bc == "electric":
            raise NotImplementedError(
                f"staggered: {what} needs a tangential wall condition on "
                f"{u!r} (the stress-free tractions or Dot(theta_hat, {u}) = "
                f"Dot(phi_hat, {u}) = 0) in addition to Dot(r_hat, {u}) = 0")
        return wall_bc

    def is_involution(self, u: str) -> bool:
        return u in self.record["involutions"]

    def is_potential(self, u: str) -> bool:
        return self.walls["vectors"].get(u, {}).get("kind") == "potential"

    def conducting(self, b: str, what: str) -> bool:
        """Whether the wall circulations of the electric field are masked
        for the flux unknown ``b``: 'conducting' walls; 'impenetrable'
        without a tangential closure is the radial wall and 'free' keeps
        the wall fluxes evolving."""
        rec = self.walls["vectors"][b]
        if rec["kind"] == "conducting":
            return True
        if rec["kind"] == "free" or (rec["kind"] == "impenetrable"
                                     and rec["tangential"] is None):
            return False
        raise NotImplementedError(
            f"staggered: {what} reads {b!r} as face fluxes, whose walls are "
            f"Dot(r_hat, {b}) = 0 (radial) or with the tangential electric field "
            f"(conducting); the tangential closure {rec['tangential']!r} pins "
            f"components the fluxes carry half a cell inside")

    def grad_closure(self, row: str, col: str):
        """(inner, outer) Dirichlet closure of the multiplier ``col`` in the
        gradient wall rows of ``row``: a potential keeps the wall rows that
        carry it, and an open (normal-stress) wall of ``row`` keeps its v_r
        faces, whose pressure force is the wall cell's value over the half
        cell, the same closure (``open_wall_rows``)."""
        if self.is_potential(row):
            return self.walls["scalars"].get(col, {}).get("closure")
        open_ = self.walls["vectors"].get(row, {}).get("open")
        if open_:
            return tuple("dirichlet" if w in open_ else "neumann"
                         for w in (Boundary.R_INNER, Boundary.R_OUTER))
        return None

    def open_wall_rows(self):
        """The normal-stress rows 2 e_rr = kappa p of the open walls. With
        the wall v_r faces kept, the strain Gram's half-cell rows carry the
        natural traction sigma_rr = 0 and the Dirichlet-closed gradient the
        pressure force of the wall cell's value, together the weak form of
        the traction-free wall of Dt(u) + c_g Grad(p) - c_l Lap(u) = 0; the
        residual traction (c_l kappa - c_g) p_w of the row is added on the
        wall faces over the half cell, sigma (2 / h) (c_l kappa - c_g) p_w
        with sigma = -1 (outer) / +1 (inner) and p_w the second-order
        extrapolation (3 p_c - p_c') / 2 from the two wall-most cells."""
        Nr_e, Nr_c = len(self.ops.grids.r_edge), len(self.ops.grids.r_center)
        Nt_c = len(self.ops.grids.theta_center)
        for u, rec in self.walls["vectors"].items():
            if not rec["open"]:
                continue
            eq = next(name for name, info in self.record["equations"].items()
                      if info["class"] == "evolution" and info["unknown"] == u)
            laps = [t for t in self.by_eq[eq] if t.target == TARGET_A
                    and t.kind == TASK_LAP_VECTOR and t.unknown_name == u]
            if len(laps) != 1:
                raise NotImplementedError(
                    f"staggered: the normal-stress wall of {u!r} needs one Lap({u}) "
                    f"in {eq!r}, the viscous stress the row constrains")
            for w, (kappa, p) in rec["open"].items():
                if p is None:
                    p = next(info["multiplier"] for info in self.record["equations"].values()
                             if info["class"] == "saddle_point" and info["vector"] == u)
                grads = [t for t in self.by_eq[eq] if t.target == TARGET_A
                         and t.kind == TASK_GRAD_SCALAR and t.unknown_name == p]
                if len(grads) != 1 or grads[0].metadata.get("grad_weight_role"):
                    raise NotImplementedError(
                        f"staggered: the normal-stress wall of {u!r} needs one plain "
                        f"Grad({p}) in {eq!r}")
                c_l = -laps[0].sign * _wall_values(self.ops, laps[0].coefficient, w,
                                                   "centers")
                c_g = grads[0].sign * _wall_values(self.ops, grads[0].coefficient, w,
                                                   "centers")
                sigma = -1.0 if w == Boundary.R_OUTER else 1.0
                vals = sigma * 2.0 / self.ops.hr * (c_l * kappa - c_g)
                i = Nr_e - 1 if w == Boundary.R_OUTER else 0
                j0, j1 = (Nr_c - 1, Nr_c - 2) if w == Boundary.R_OUTER else (0, 1)
                rows = np.arange(i * Nt_c, (i + 1) * Nt_c)
                block = sp.csr_matrix(
                    (np.concatenate([1.5 * vals, -0.5 * vals]),
                     (np.concatenate([rows, rows]),
                      np.concatenate([j0 * Nt_c + np.arange(Nt_c), j1 * Nt_c + np.arange(Nt_c)]))),
                    shape=(self.size(u), self.ops.n["CC"]), dtype=complex)
                self.add(TARGET_A, u, p, block)

    def physical(self, col: str, block):
        """A block reading the physical components of ``col``, composed
        with 1 / area when ``col`` is a flux unknown."""
        if self.is_involution(col):
            return block @ self.ops.flux_scaling()[1]
        return block

    def curl_wall_bc(self, u: str):
        rec = self.walls["vectors"][u]
        bc = rec["tangential"]
        if isinstance(bc, TangentialWall) or bc in ("noslip", "stress_free"):
            return bc
        return None

    def note_power(self, eq_name: str, power: int):
        self.r_power[eq_name] = max(self.r_power[eq_name], power)

    def tasks_of(self, eq_name):
        return self.by_eq[eq_name]

    def background_of(self, task: OperatorTask):
        """``background(i, loc)`` for ``_CGridOps.dot_vector``: coefficient
        times the task's background component on ``loc``, None where the
        component vanishes identically."""
        names = {"r": "v_r", "t": "v_theta", "p": "v_phi"}

        def background(i, loc):
            vals = self.background(task, loc, names[i])
            return None if _is_zero(vals) else vals
        return background

    # -- tensor definitions ---------------------------------------------------

    def definition(self, T: str):
        """(Def, u): the map from the DOFs of the vector u onto the
        components of the eliminated tensor T, solved from its definition
        c_I T + sum_k c_k tensor_k(u) = 0."""
        if T in self._defs:
            return self._defs[T]
        info = self.record["equations"][self.record["definitions"][T]]
        ident, u = info["identity"], info["vector"]
        placement = self.placement[T]
        c_I = np.concatenate([self.ops.evaluate(ident.coefficient, loc)
                              for loc in self.locs(T)]) * ident.sign
        if not np.all(np.abs(c_I) > 0):
            raise NotImplementedError(
                f"staggered: the definition of {T!r} has a vanishing "
                f"coefficient on {T!r}; it cannot be eliminated")
        wall_bc = self.wall_bc(u, f"the definition of {T!r} from Grad({u})")
        total = None
        for task, (a, b) in info["terms"]:
            M, place = self.ops.tensor_map(a, b, wall_bc)
            if place != placement:
                raise NotImplementedError(
                    f"staggered: the definition of {T!r} mixes symmetric and "
                    f"non-symmetric tensor terms")
            block = _scaled(M, task.sign, self.coefficient_on(task, T))
            total = block if total is None else total + block
        Def = -(sp.diags(1.0 / c_I) @ total)
        self._defs[T] = (Def.tocsr(), u)
        return self._defs[T]

    # -- dispatch -----------------------------------------------------------

    def dispatch(self, task: OperatorTask, row: str):
        if id(task) in self.consumed:
            return
        kind = task.kind
        handler = {
            TASK_IDENTITY_SCALAR: self.identity, TASK_IDENTITY_VECTOR: self.identity,
            TASK_IDENTITY_TENSOR: self.identity,
            TASK_GRAD_SCALAR: self.grad_scalar,
            TASK_DIV_VECTOR_SCALAR: self.div_vector_scalar,
            TASK_LAP_SCALAR: self.lap_scalar, TASK_LAPH_SCALAR: self.laph_scalar,
            TASK_LAP_VECTOR: self.viscous_force,
            TASK_GRAD_DIV_VECTOR: self.viscous_force,
            TASK_DOT_BG_VEC_TENSOR_UNK: self.viscous_force,
            TASK_BG_VEC_TIMES_DIV_UNK_VEC: self.viscous_force,
            TASK_CROSS_VECTOR: self.cross_vector,
            TASK_DOT_BG_VEC_UNK_VEC_SCALAR: self.dot_bg_vec_unk_vec_scalar,
            TASK_BG_VEC_TIMES_UNK_SCALAR: self.bg_vec_times_unk_scalar,
            TASK_DOT_BG_VEC_GRAD_SCALAR: self.dot_bg_vec_grad_scalar,
            TASK_DOT_BG_GRAD_VEC: self.dot_bg_grad_vec,
            TASK_DIV_TENSOR_VECTOR: self.div_tensor_vector,
            TASK_DOT_BG_TENSOR_VEC: self.dot_bg_tensor_vec,
            TASK_TENSOR_OF_VECTOR: self.tensor_of_vector,
            TASK_DIV_TENSOR_UNK: self.div_tensor_unk,
            TASK_DOT_BG_VEC_UNK_TENSOR: self.dot_bg_vec_unk_tensor,
            TASK_SOURCE_SCALAR: self.source, TASK_SOURCE_VECTOR: self.source,
            TASK_SOURCE_TENSOR: self.source,
            TASK_CURL_CROSS_UNK_BG: self.curl_cross_unk_bg,
            TASK_CROSS_CURL_UNK_BG: self.cross_curl_unk_bg,
            TASK_CURL_CURL_VECTOR: self.curl_curl_vector,
            TASK_CROSS_CURL_CURL_UNK_BG: self.cross_curl_curl_unk_bg,
            TASK_CURL_VECTOR: self.curl_vector,
        }.get(kind)
        if kind in (TASK_LAP_VECTOR, TASK_CURL_CURL_VECTOR) and task.unknown_name == row:
            if self.is_involution(row):
                handler = self.involution_diffusion
            elif self.is_potential(row):
                handler = self.potential_diffusion
        if handler is None:
            raise NotImplementedError(f"staggered: unknown task kind {kind!r}")
        if task.target != TARGET_RHS and _is_zero(self.coefficient_on(task, row)):
            self.consumed.add(id(task))
            return
        self.note_power(task.equation_name, R_POWER.get(kind, 0))
        handler(task, row)

    def identity(self, task, row):
        col = task.unknown_name
        if self.placement[row] != self.placement[col]:
            raise NotImplementedError(
                f"staggered: identity between {col!r} ({self.placement[col]}) "
                f"and the rows of {row!r} ({self.placement[row]}) needs equal "
                f"placements")
        vals = self.coefficient_on(task, col)
        block = _scaled(sp.eye(self.size(col), format="csr", dtype=complex),
                        task.sign, vals)
        self.add_to(task.target, row, col, block)

    def grad_scalar(self, task, row):
        col = task.unknown_name
        self.require(task, row, col, "face", "center")
        if task.metadata.get("grad_weight_role") == "grad":
            self.weighted_gradient(task, row)
            return
        G = self.ops.gradient(self.grad_closure(row, col))
        block = _scaled(G, task.sign, self.coefficient_on(task, row))
        self.add(TARGET_A, row, col, block)

    def weighted_gradient(self, task, row):
        """Grad(M s) from its tagged pair as G diag(M), M on the centres
        and the outer coefficient on the faces; the partner is consumed."""
        col = task.unknown_name
        weight = task.metadata["grad_weight"]
        partner = [t for t in self.tasks_of(task.equation_name)
                   if t.metadata.get("grad_weight") is weight and t is not task
                   and t.unknown_name == col]
        for t in partner:
            self.consumed.add(id(t))
        self.consumed.add(id(task))
        outer = task.metadata.get("grad_outer")
        vals = None if outer is None else np.concatenate(
            [self.ops.evaluate(outer, loc) for loc in self.locs(row)])
        G = self.ops.gradient(self.grad_closure(row, col))
        block = G @ sp.diags(self.ops.evaluate(weight, "CC"))
        self.add(TARGET_A, row, col, _scaled(block, task.sign, vals))

    def weighted_divergence(self, task, closure=None):
        """Div block of a plain or tagged (Div(M u)) task and its outer
        coefficient; the tagged partner task is consumed. ``closure`` is
        the Dirichlet closure of the multiplier for a potential."""
        role = task.metadata.get("div_weight_role")
        if role == "div":
            weight = task.metadata["div_weight"]
            partner = [t for t in self.tasks_of(task.equation_name)
                       if t.metadata.get("div_weight") is weight and t is not task]
            for t in partner:
                self.consumed.add(id(t))
            return self.ops.divergence(weight, closure), task.metadata.get("div_outer")
        return self.ops.divergence(None, closure), task.coefficient

    def div_vector_scalar(self, task, row):
        col = task.unknown_name
        self.require(task, row, col, "center", "face")
        Div, outer = self.weighted_divergence(task)
        vals = None if outer is None else self.ops.evaluate(outer, "CC")
        self.add(TARGET_A, row, col, _scaled(Div, task.sign, vals))

    # -- combinations read from the task list --------------------------------

    def log_gradient(self, task, c_l, sign_l, factor):
        """a_r(R, T) = factor * (coefficient * b_r) / c_l of a task with a
        radial background, the log-gradient of a weight; None when the
        ratio depends on theta. The background component itself is
        returned unscaled when the two coefficients are the same
        expression, so the recovered weight is bitwise the background's
        own profile."""
        b_r = task.source_field.components["v_r"]
        coef = task.coefficient
        same = coef is not None and type(coef).__name__ == "SymPyComponent" and (
            getattr(c_l, "expr", None) is not None and coef.expr == c_l.expr)
        ratio = task.sign * factor / (sign_l * (1.0 if same else _constant(
            self.ops.evaluate(c_l, "CC"))))
        if same:
            def a_fn(R, T):
                out = np.asarray(b_r.evaluate_on_grid(R, T)) + 0.0 * T
                return out if ratio == 1.0 else ratio * out
        else:
            def a_fn(R, T):
                out = np.asarray(b_r.evaluate_on_grid(R, T)) + 0.0 * T
                if coef is not None:
                    out = out * (np.asarray(coef.evaluate_on_grid(R, T)) + 0.0 * T)
                return ratio * out
        R, T = self.ops.mesh("CC")
        vals = np.asarray(a_fn(R, T))
        if np.abs(vals - vals[:, :1]).max() > 1e-12 * max(1.0, np.abs(vals).max()):
            return None
        return a_fn

    def viscous_force(self, task, row):
        """Lap(u), Grad(Div(u)), Dot(a, tensor(u)) and Div(u) a of one
        equation on one face unknown, read together.

        The strain Gram is Lap + (1/3) Grad Div, not Lap, so it stands in
        for Lap(u) only where the two agree: on a face unknown carrying a
        divergence constraint (the classification record's saddle point or
        involution, where Div(u) = 0 on the discrete manifold and the
        certified inertial and CT identities live), and where the equation
        writes the trace-free stress divergence itself, Lap(u) with
        (1/3) Grad(Div(u)) or the density-weighted form with the strain
        contraction a . (Grad u + Grad u^T) and its trace -(2/3) a Div(u)
        (a radial and proportional to the Lap coefficient c_l), which is
        the compressible viscous force and carries w = exp(integral a_r dr).
        Otherwise Lap(u) is the true vector Laplacian, the same Gram less
        (1/3) Grad Div (which keeps the Gram's wall closure and its |m| = 1
        pole prolongation; Grad Div - Curl Curl instead loses the tangential
        wall rows, whose error then grows like 1/h), and Grad(Div(u)) is
        built on its own with its own coefficient.
        With the Gram, Grad(Div(u)) of coefficient c_g adds the remainder
        (c_g - c_l/3) Grad Div on top. Terms that do not fit the
        combination are built on their own through the tensor placement."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        group = [t for t in self.tasks_of(task.equation_name)
                 if t.unknown_name == col and t.target == TARGET_A
                 and t.kind in (TASK_LAP_VECTOR, TASK_GRAD_DIV_VECTOR,
                                TASK_DOT_BG_VEC_TENSOR_UNK, TASK_BG_VEC_TIMES_DIV_UNK_VEC)
                 and id(t) not in self.consumed
                 and not _is_zero(self.coefficient_on(t, row))]
        for t in group:
            self.consumed.add(id(t))
        laps = [t for t in group if t.kind == TASK_LAP_VECTOR]
        gds = [t for t in group if t.kind == TASK_GRAD_DIV_VECTOR]
        tens = [t for t in group if t.kind == TASK_DOT_BG_VEC_TENSOR_UNK]
        traces = [t for t in group if t.kind == TASK_BG_VEC_TIMES_DIV_UNK_VEC]
        if not laps:
            for t in gds:
                self.grad_div_vector(t, row, 1.0)
            for t in tens:
                self.dot_bg_vec_tensor_unk(t, row)
            for t in traces:
                self.bg_vec_times_div_unk_vec(t, row)
            return
        self.second_order[col] = True
        self.note_power(task.equation_name, 2)
        if len(laps) > 1:
            raise NotImplementedError(
                f"staggered: several Lap({col}) terms in {task.equation_name!r}")
        lap = laps[0]
        c_vals = self.coefficient_on(lap, row)
        c_l = _constant(c_vals)
        wall_bc = self.wall_bc(col, f"Lap({col})")
        extra_gd = []
        for t in gds:
            c_g = _constant(self.coefficient_on(t, row))
            if c_l is None or c_g is None:
                raise NotImplementedError(
                    f"staggered: Lap({col}) and Grad(Div({col})) with varying "
                    f"coefficients in {task.equation_name!r}")
            shift = t.sign * c_g / (lap.sign * c_l) - 1.0 / 3.0
            if abs(shift) > 1e-12:
                extra_gd.append((t, shift))
        weight = None
        if c_l is not None and len(tens) == 1 and len(traces) == 1:
            a, b = tensor_terms(tens[0])
            if a == b and _is_radial(tens[0].source_field) \
                    and _is_radial(traces[0].source_field):
                a_fn = self.log_gradient(tens[0], lap.coefficient, lap.sign, a)
                t_fn = self.log_gradient(traces[0], lap.coefficient, lap.sign, -1.5)
                if a_fn is not None and t_fn is not None:
                    R, T = self.ops.mesh("CC")
                    va, vt = np.asarray(a_fn(R, T)), np.asarray(t_fn(R, T))
                    if np.abs(va - vt).max() <= 1e-10 * max(1.0, np.abs(va).max()):
                        weight = _log_gradient_weight(self.ops.grids, a_fn, "quadrature")
        gram = (col in self.record["constraints"] or weight is not None
                or (bool(gds) and not extra_gd))
        if weight is not None:
            F = build_staggered_weighted_viscous_gram(
                self.ops.grids, self.ops.m, weight, order=self.ops.order,
                wall_bc=wall_bc, pole_closure=self.ops.pole_closure)
        else:
            F, _ = self.lap_blocks(col, wall_bc, gram)
            for t in tens:
                self.dot_bg_vec_tensor_unk(t, row)
            for t in traces:
                self.bg_vec_times_div_unk_vec(t, row)
        self.add(TARGET_A, row, col, _scaled(F, lap.sign, c_vals))
        if gram:
            for t, shift in extra_gd:
                self.grad_div_vector(t, row, shift * (lap.sign * c_l)
                                     / (t.sign * _constant(self.coefficient_on(t, row))))
        else:
            for t in gds:
                self.grad_div_vector(t, row, 1.0)

    def lap_blocks(self, col, wall_bc, gram=None):
        """(Lap(col), Grad(Div(col))) on the faces: the unit-density strain
        Gram with the tangential wall closure, which is Lap + (1/3) Grad
        Div, less (1/3) Grad Div unless ``gram`` (a divergence-constrained
        unknown, where the two agree; read off the classification record
        when None), and the grad-div block."""
        if gram is None:
            gram = col in self.record["constraints"]
        GD = build_staggered_grad_div(self.ops.grids, self.ops.m, order=self.ops.order)
        F = build_staggered_weighted_viscous_gram(
            self.ops.grids, self.ops.m, lambda r, t: np.ones_like(r * t),
            order=self.ops.order, wall_bc=wall_bc, pole_closure=self.ops.pole_closure)
        if not gram:
            F = (F - GD / 3.0).tocsr()
        return F, GD

    def grad_div_vector(self, task, row, factor):
        col = task.unknown_name
        self.second_order[col] = True
        self.note_power(task.equation_name, 2)
        GD = build_staggered_grad_div(self.ops.grids, self.ops.m, order=self.ops.order)
        vals = self.coefficient_on(task, row)
        if factor != 1.0:
            vals = (np.ones(self.size(row), dtype=complex) if vals is None else vals) * factor
        self.add(TARGET_A, row, col, _scaled(GD, task.sign, vals))

    def lap_scalar(self, task, row):
        """Lap(s), read together with a Dot(B0, Grad(s)) of the same
        equation whose background is radial and whose coefficient is a
        radial multiple of the Lap coefficient: the weighted Laplacian
        (1/w) Div(w Grad s) with w = exp(integral a_r dr), the trapezoid
        recovery of ``_log_gradient_weight``; the plain Laplacian
        otherwise."""
        col = task.unknown_name
        self.require(task, row, col, "center", "center")
        self.consumed.add(id(task))
        rec = self.walls["scalars"][col]
        closure = tuple("neumann" if c == "robin" else c for c in rec["closure"])
        robin = {(1 if w == Boundary.R_OUTER else 0): g for w, g in rec["robin"].items()}
        if robin and self.ops.order > 2:
            raise NotImplementedError(
                f"staggered: the Robin closure of Lap({col}) is order 2 only; the "
                f"mirror ghost of a Robin wall is not a parity of the field, so the "
                f"reflected stencils of the higher orders do not carry it")
        c_vals = self.coefficient_on(task, row)
        partners = [t for t in self.tasks_of(task.equation_name)
                    if t.kind == TASK_DOT_BG_VEC_GRAD_SCALAR and t.unknown_name == col
                    and id(t) not in self.consumed
                    and not _is_zero(self.coefficient_on(t, row))]
        weight = None
        if _constant(c_vals) is not None and len(partners) == 1 \
                and _is_radial(partners[0].source_field):
            a_fn = self.log_gradient(partners[0], task.coefficient, task.sign, 1.0)
            if a_fn is not None:
                weight = _log_gradient_weight(self.ops.grids, a_fn, "trapezoid")
                self.consumed.add(id(partners[0]))
        if weight is None:
            L = build_staggered_scalar_laplacian(self.ops.grids, self.ops.m,
                                                 order=self.ops.order, wall_bc=closure)
        else:
            L = build_staggered_weighted_scalar_laplacian(
                self.ops.grids, self.ops.m, weight, order=self.ops.order,
                wall_bc=closure)
        if robin:
            L = (L + _robin_wall_term(self.ops.grids, robin, weight)).tocsr()
        self.add(TARGET_A, row, col, _scaled(L, task.sign, c_vals))

    def laph_scalar(self, task, row):
        col = task.unknown_name
        self.require(task, row, col, "center", "center")
        L = build_staggered_angular_laplacian(self.ops.grids, self.ops.m,
                                              order=self.ops.order)
        self.add(TARGET_A, row, col, _scaled(L, task.sign, self.coefficient_on(task, row)))

    # -- grade-changing products ---------------------------------------------

    def cross_vector(self, task, row):
        """Cross(B0, u) (side 'left') or Cross(u, B0) (side 'right') on the
        faces. Each component pair {i, k} couples through the third
        background component b_j with the Levi-Civita signs; the pair's
        host sub-grid H is the grid the base interpolation departs from
        (CC, v_phi's grid, for the pairs with v_phi; CE, v_theta's grid,
        for (v_r, v_theta), so the ball origin ring only receives).
        Pairing 'center': b_j on H in both blocks, P = hodge(H, O) and its
        adjoint hodge(O, H), so the pair is exactly W-skew. 'seed': b_j on
        each block's own output grid. 'adjoint': the base block with b_j on
        its output grid and the exact negative W-adjoint as partner."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.add(TARGET_A, row, col, self.physical(col, self.cross_matrix(task)))

    def cross_matrix(self, task):
        """sign * coefficient * Cross(B0, .) (side 'left') or Cross(., B0)
        (side 'right') on the faces, the coefficient folded into the
        background on the host grids."""
        if self.pairing not in ("center", "seed", "adjoint"):
            raise ValueError(
                f"coriolis_pairing must be 'center', 'seed' or 'adjoint'; got "
                f"{self.pairing!r}")
        side = task.metadata.get("side", "left")
        sign = task.sign if side == "left" else -task.sign
        locs = self.ops.face_locs()
        n = [self.ops.n[k] for k in locs]
        off = np.concatenate([[0], np.cumsum(n)])
        # (host component h, other component o, epsilon of block(o <- h))
        pairs = ((2, 0, +1, 1), (2, 1, -1, 0), (1, 0, -1, 2))
        blocks = {}
        for h, o, eps, j in pairs:
            H, O = locs[h], locs[o]
            b_H = self.background(task, H, FACE_COMPONENTS[j])
            P_ho, P_oh = self.ops.hodge(H, O), self.ops.hodge(O, H)
            if self.pairing == "center":
                blk_oh = eps * (P_ho @ sp.diags(b_H))
                blk_ho = -eps * (sp.diags(b_H) @ P_oh)
            else:
                b_O = self.background(task, O, FACE_COMPONENTS[j])
                blk_oh = eps * (sp.diags(b_O) @ P_ho)
                if self.pairing == "seed":
                    blk_ho = -eps * (sp.diags(b_H) @ P_oh)
                else:
                    blk_ho = -(sp.diags(self.ops.inv_W[H]) @ blk_oh.conj().T
                               @ sp.diags(self.ops.W[O]))
            blocks[(o, h)] = blk_oh
            blocks[(h, o)] = blk_ho
        M = self._assemble_blocks(blocks, off, off)
        return M if sign > 0 else -M

    # -- constrained transport and the vector potential ----------------------

    def _ct(self, blocks, rows, cols):
        return self.ops.ct_matrix(blocks, rows, cols, self.ops.ct_sizes())

    def curl_cross_unk_bg(self, task, row):
        """Curl(Cross(v, B0)) on the rows of a face unknown b: the electric
        field cE (v x B0) on the edges, its wall circulations masked for a
        conducting wall, and the incidence curl onto the face fluxes (the
        CT induction blocks); on a plain face unknown the flux tendency is
        divided by the face areas and the wall circulations follow v's
        wall, masked where v is no-slip (v = 0 on the wall) and read from
        the wall-adjacent v otherwise."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        if self.is_involution(row):
            conducting = self.conducting(row, "Curl(Cross(v, B0))")
        else:
            conducting = self.walls["vectors"][col]["tangential"] == "noslip"
        vals = self.coefficient_on(task, row)
        c = _constant(vals)
        side = 1.0 if task.metadata.get("inner_side") == "right" else -1.0
        cE = side if c is None else task.sign * c * side
        M = self._ct(ct_induction_blocks(self.ops.ct(), task.source_field, cE, conducting),
                     ("br", "bt", "bp"), ("vr", "vt", "vp"))
        if not self.is_involution(row):
            M = self.ops.flux_scaling()[1] @ M
        self.add(TARGET_A, row, col, -M if c is not None else _scaled(-M, task.sign, vals))

    def involution_diffusion(self, task, row):
        """Lap(b) = -Curl Curl b on the constraint manifold, or Curl(Curl(b)):
        j = curl b on the edges from the fluxes, eta j masked at a
        conducting wall, the incidence curl back onto the fluxes (the CT
        resistive blocks)."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.second_order[col] = True
        self.note_power(task.equation_name, 2)
        conducting = self.conducting(row, f"Lap({col})")
        vals = self.coefficient_on(task, row)
        c = _constant(vals)
        lap = task.kind == TASK_LAP_VECTOR
        if c is not None:
            eta = -(task.sign * c) if lap else task.sign * c
            M = self._ct(ct_resistive_blocks(self.ops.ct(), eta, conducting),
                         ("br", "bt", "bp"), ("br", "bt", "bp"))
            self.add(TARGET_A, row, col, -M)
            return
        M = self._ct(ct_resistive_blocks(self.ops.ct(), 1.0, conducting),
                     ("br", "bt", "bp"), ("br", "bt", "bp"))
        self.add(TARGET_A, row, col, _scaled(M if lap else -M, task.sign, vals))

    def cross_curl_unk_bg(self, task, row):
        """Cross(Curl(b), B0) (side 'right') / Cross(B0, Curl(b)) on the
        faces of a velocity row: j = curl b on the edges from the fluxes
        (a plain face unknown is first scaled to fluxes), brought to the
        faces and crossed with B0 (the CT Lorentz blocks, the plain
        averages of the constrained-transport recipe)."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        s_ = task.sign if task.metadata.get("side", "right") == "right" else -task.sign
        pre = []
        for loc, interior in (("RC", "r"), ("CE", "theta"), ("CC", None)):
            vals = self.ops.evaluate(task.coefficient, loc)
            pre.append(-(s_ * self._interior(vals, loc, interior)))
        M = self._ct(ct_lorentz_blocks(self.ops.ct(), task.source_field, tuple(pre)),
                     ("vr", "vt", "vp"), ("br", "bt", "bp"))
        if not self.is_involution(col):
            M = M @ self.ops.flux_scaling()[0]
        self.add(TARGET_A, row, col, -M)

    def _interior(self, vals, loc, interior):
        """A coefficient on a face grid restricted to the interior r-edges
        ('r') or interior theta-edges ('theta'), as the CT Lorentz builder
        indexes them."""
        rv, tv = self.ops.nodes[loc]
        v = np.asarray(vals).reshape(len(rv), len(tv))
        if interior == "r":
            v = v[1:-1, :]
        elif interior == "theta":
            v = v[:, 1:-1]
        return v.ravel()

    def potential_diffusion(self, task, row):
        """Lap(A) = -Curl Curl A on the Coulomb manifold, or Curl(Curl(A)),
        for a vector potential: the dual curl of the primal curl with the
        tangential Dirichlet closure, wall A_r rows included."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.note_power(task.equation_name, 2)
        J = self.ops.curl_curl(self.curl_wall_bc(col))
        block = -J if task.kind == TASK_LAP_VECTOR else J
        self.add(TARGET_A, row, col, _scaled(block, task.sign, self.coefficient_on(task, row)))

    def curl_curl_vector(self, task, row):
        """Curl(Curl(u)) of a plain face unknown on the faces (the
        differential composition, the unknown's tangential closure)."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.second_order[col] = True
        self.note_power(task.equation_name, 2)
        J = self.ops.curl_curl(self.curl_wall_bc(col))
        self.add(TARGET_A, row, col, _scaled(J, task.sign, self.coefficient_on(task, row)))

    def cross_curl_curl_unk_bg(self, task, row):
        """Cross(Curl(Curl(A)), B0): the current j = Curl Curl A returned to
        the faces, crossed with B0 through the Hodge-paired cross block
        (the coefficient on the host grids)."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        J = self.ops.curl_curl(self.curl_wall_bc(col))
        self.add(TARGET_A, row, col, self.cross_matrix(task) @ J)

    def curl_vector(self, task, row):
        """A standalone Curl(u) in a face equation: the edge curl brought to
        the faces by the Hodge maps."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        C = self.ops.curl_to_faces(self.curl_wall_bc(col))
        self.add(TARGET_A, row, col, _scaled(C, task.sign, self.coefficient_on(task, row)))

    @staticmethod
    def _assemble_blocks(blocks, row_off, col_off):
        rows, cols, data = [], [], []
        for (i, k), blk in blocks.items():
            coo = sp.coo_matrix(blk)
            rows.append(coo.row + row_off[i])
            cols.append(coo.col + col_off[k])
            data.append(coo.data)
        return sp.csr_matrix((np.concatenate(data),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(row_off[-1], col_off[-1]))

    def dot_bg_vec_unk_vec_scalar(self, task, row):
        """Dot(B0, u) on the centres: each face component brought to the
        centres by the adjoint of the centre-to-face interpolation, times
        the background component on the centres."""
        col = task.unknown_name
        self.require(task, row, col, "center", "face")
        parts = []
        for k, loc in enumerate(self.ops.face_locs()):
            b = self.background(task, "CC", FACE_COMPONENTS[k])
            parts.append(sp.diags(b) @ self.ops.hodge(loc, "CC"))
        block = self.physical(col, sp.hstack(parts, format="csr"))
        self.add(TARGET_A, row, col, block if task.sign > 0 else -block)

    def bg_vec_times_unk_scalar(self, task, row):
        """s B0 on the faces: the centre scalar interpolated to each face
        grid, times the background component there; the adjoint partner of
        ``dot_bg_vec_unk_vec_scalar``. The tagged partner of a weighted
        gradient is built with it."""
        col = task.unknown_name
        self.require(task, row, col, "face", "center")
        if task.metadata.get("grad_weight_role") == "weight":
            weight = task.metadata["grad_weight"]
            (partner,) = [t for t in self.tasks_of(task.equation_name)
                          if t.metadata.get("grad_weight") is weight and t is not task
                          and t.unknown_name == col]
            self.weighted_gradient(partner, row)
            return
        parts = []
        for k, loc in enumerate(self.ops.face_locs()):
            b = self.background(task, loc, FACE_COMPONENTS[k])
            parts.append(sp.diags(b) @ self.ops.hodge("CC", loc))
        block = sp.vstack(parts, format="csr")
        self.add(TARGET_A, row, col, block if task.sign > 0 else -block)

    def dot_bg_vec_grad_scalar(self, task, row):
        """B0 . Grad(s) on the centres: the gradient blocks averaged back to
        the centres (a plain average, there is no partner block) and
        contracted with the background on the centres."""
        col = task.unknown_name
        self.require(task, row, col, "center", "center")
        block = None
        for k, loc in enumerate(self.ops.face_locs()):
            b = self.background(task, "CC", FACE_COMPONENTS[k])
            G = self.ops.G[loc]
            if loc != "CC":
                G = self.ops.average(loc, "CC") @ G
            part = sp.diags(b) @ G
            block = part if block is None else block + part
        self.add(TARGET_A, row, col, block if task.sign > 0 else -block)

    def bg_vec_times_div_unk_vec(self, task, row):
        """Div(u) B0 on the faces: the divergence on the centres brought to
        each face grid by the base interpolation, times the background
        component there."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.consumed.add(id(task))
        Div = self.ops.divergence(None)
        parts = []
        for k, loc in enumerate(self.ops.face_locs()):
            b = self.background(task, loc, FACE_COMPONENTS[k])
            parts.append(sp.diags(b) @ self.ops.hodge("CC", loc) @ Div)
        block = sp.vstack(parts, format="csr")
        self.add(TARGET_A, row, col, block if task.sign > 0 else -block)

    def dot_bg_tensor_vec(self, task, row):
        """Dot(T0, u) on the faces: the diagonal components of T0 on their
        own face grids, each off-diagonal pair through the pair's host
        grid H (CE for (r, theta), CC for the pairs with phi) with the
        base interpolation one way and its norm adjoint the other, so a
        symmetric T0 gives a W-self-adjoint block."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        locs = self.ops.face_locs()
        off = np.concatenate([[0], np.cumsum([self.ops.n[k] for k in locs])])
        names = ("r", "theta", "phi")
        comps = task.source_field.components

        def T0(i, k, loc):
            vals = self.ops.evaluate(comps[(names[i], names[k])], loc)
            if task.coefficient is not None:
                vals = vals * self.ops.evaluate(task.coefficient, loc)
            return vals
        blocks = {}
        for i in range(3):
            blocks[(i, i)] = sp.diags(T0(i, i, locs[i]))
        for i, k, h in ((0, 1, 1), (0, 2, 2), (1, 2, 2)):
            H = locs[h]
            o = i if h == k else k
            O = locs[o]
            blocks[(o, h)] = self.ops.hodge(H, O) @ sp.diags(T0(o, h, H))
            blocks[(h, o)] = sp.diags(T0(h, o, H)) @ self.ops.hodge(O, H)
        M = self.physical(col, self._assemble_blocks(blocks, off, off))
        self.add(TARGET_A, row, col, M if task.sign > 0 else -M)

    # -- tensor placement -----------------------------------------------------

    def _dot_to_faces(self, task, row, M, placement, side):
        """sign * Dot(B0, M u) (or Dot(M u, B0)) on the faces, the
        coefficient and B0 at the tensor component locations."""
        block = self.ops.dot_vector(placement, side, self.background_of(task)) @ M
        self.add(TARGET_A, row, task.unknown_name, block if task.sign > 0 else -block)

    def dot_bg_grad_vec(self, task, row):
        """Dot(B0, Grad(u)): the nine-component gradient contracted with
        B0 at the component locations and brought to the faces."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        G9, _ = self.ops.grad9(paired=True)
        self._dot_to_faces(task, row, G9, "tensor9", "left")

    def dot_bg_vec_tensor_unk(self, task, row):
        """Dot(B0, a Grad(u) + b Transpose(Grad(u))) (side 'left') or
        Dot(tensor(u), B0) ('right'): the tensor map onto its placement,
        then the contraction of ``dot_vector``."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.consumed.add(id(task))
        a, b = tensor_terms(task)
        wall_bc = self.walls["vectors"][col]["tangential"]
        M, placement = self.ops.tensor_map(a, b, wall_bc, paired=True)
        self._dot_to_faces(task, row, M, placement, task.metadata.get("side", "left"))

    def div_tensor_vector(self, task, row):
        """Div(a Grad(u) + b Transpose(Grad(u))): for a = b the strain map
        onto the six components followed by the norm-adjoint divergence,
        the strain Gram with C = 2I; otherwise the vector identities
        Div(Grad(u)) = Lap(u) and Div(Transpose(Grad(u))) = Grad(Div(u)),
        a Lap + b Grad Div on the face blocks of ``lap_blocks``. The
        nine-component map is not divided here: its theta-phi and
        phi-theta components carry finite pole limits that only their sum,
        the strain, closes, so its norm-adjoint divergence is O(1/dtheta)
        inconsistent on the pole-adjacent ring at m = 0."""
        col = task.unknown_name
        self.require(task, row, col, "face", "face")
        self.second_order[col] = True
        self.note_power(task.equation_name, 2)
        a, b = tensor_terms(task)
        wall_bc = self.wall_bc(col, f"Div(tensor({col}))")
        if a == b:
            M, placement = self.ops.tensor_map(a, b, wall_bc)
            block = self.ops.div_tensor(placement, wall_bc) @ M
        else:
            F, GD = self.lap_blocks(col, wall_bc)
            block = (a * F + b * GD).tocsr()
        self.add(TARGET_A, row, col, _scaled(block, task.sign, self.coefficient_on(task, row)))

    def tensor_of_vector(self, task, row):
        """a Grad(u) + b Transpose(Grad(u)) on the rows of a tensor unknown
        with DOFs (an evolution equation of the tensor)."""
        col = task.unknown_name
        self.require(task, row, col, self.placement[row], "face")
        a, b = tensor_terms(task)
        wall_bc = self.walls["vectors"][col]["tangential"]
        M, placement = self.ops.tensor_map(a, b, wall_bc)
        if placement != self.placement[row]:
            raise NotImplementedError(
                f"staggered: the term {task.kind!r} in {task.equation_name!r} is "
                f"{'symmetric' if a == b else 'not symmetric'} but {row!r} has "
                f"placement {self.placement[row]!r}")
        self.add(TARGET_A, row, col, _scaled(M, task.sign, self.coefficient_on(task, row)))

    def div_tensor_unk(self, task, row):
        """Div(T) of a tensor unknown: the norm-adjoint divergence of its
        placement, composed with the definition when T is eliminated."""
        col = task.unknown_name
        self.require(task, row, col, "face", self.placement[col])
        vec = self.record["equations"].get(self.record["definitions"].get(col), {})
        u = vec.get("vector", row)
        wall_bc = self.wall_bc(u, f"Div({col})")
        self.second_order[u] = True
        if col in self.record["definitions"]:
            self.note_power(task.equation_name, 2)
        block = self.ops.div_tensor(self.placement[col], wall_bc)
        self.add_to(TARGET_A, row, col, _scaled(block, task.sign, self.coefficient_on(task, row)))

    def dot_bg_vec_unk_tensor(self, task, row):
        """Dot(B0, T) / Dot(T, B0) of a tensor unknown."""
        col = task.unknown_name
        self.require(task, row, col, "face", self.placement[col])
        block = self.ops.dot_vector(self.placement[col], task.metadata.get("side", "left"),
                                    self.background_of(task))
        self.add_to(TARGET_A, row, col, block if task.sign > 0 else -block)

    def source(self, task, row):
        self.has_source = True
        if task.kind == TASK_SOURCE_SCALAR:
            loc, off, _ = self.blocks(row)[0]
            self.b[off:off + self.ops.n[loc]] += task.sign * self.ops.evaluate(
                task.coefficient, loc)
            return
        if task.kind == TASK_SOURCE_VECTOR:
            for k, (loc, off, size) in enumerate(self.blocks(row)):
                self.b[off:off + size] += task.sign * self.background(
                    task, loc, FACE_COMPONENTS[k])
            return
        names = {"r": "r", "t": "theta", "p": "phi"}
        comps = task.source_field.components
        for c, (loc, off, size) in zip(TENSOR_COMPONENTS, self.blocks(row)):
            vals = self.ops.evaluate(comps[(names[c[0]], names[c[1]])], loc)
            if task.coefficient is not None:
                vals = vals * self.ops.evaluate(task.coefficient, loc)
            self.b[off:off + size] += task.sign * vals

    # -- totals -------------------------------------------------------------

    def _total(self, parts):
        rows, cols, data = [], [], []
        for r0, c0, blk in parts:
            coo = blk.tocoo()
            rows.append(coo.row + r0)
            cols.append(coo.col + c0)
            data.append(coo.data)
        if not parts:
            return sp.csr_matrix((self.N, self.N), dtype=complex)
        return sp.csr_matrix((np.concatenate(data),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(self.N, self.N), dtype=complex)

    def totals(self):
        return self._total(self.A_parts), self._total(self.B_parts)


def _layout_for(system, record: dict, ops: _CGridOps) -> StaggeredLayout:
    """Face unknowns first, then tensor unknowns with DOFs, then centre
    unknowns, each in equation order (a multiplier where its constraint
    equation sits), then the involution unknowns (the flux block appended
    to the hydrodynamic state, as the CT extension places it); eliminated
    tensors have no DOFs; the DOF blocks of an unknown follow
    ``PLACEMENTS``."""
    ordered = [record["principal"][eq.name] for eq in system.equations
               if eq.name in record["principal"]
               and record["principal"][eq.name] not in record["definitions"]]
    placement = record["placement"]
    flux = record["involutions"]
    names = ([u for u in ordered if placement[u] == "face" and u not in flux]
             + [u for u in ordered if placement[u].startswith("tensor")]
             + [u for u in ordered if placement[u] == "center"]
             + [u for u in ordered if u in flux])
    fields, off = [], 0
    for u in names:
        for suffix, loc in PLACEMENTS[placement[u]]:
            size = ops.n[LOC_CODE[loc]]
            fields.append(StaggeredField(u + suffix, loc, off, size))
            off += size
    return StaggeredLayout(fields=tuple(fields), total_size=off,
                           placements={u: placement[u] for u in names})


def _ball_premultiply(A, B, layout, ops: _CGridOps, record: dict, powers: dict):
    """Rows of every equation scaled by r^n, n = max(1, the natural
    r-power of its terms), the origin ring at the half-cell radius, for a
    pencil with B = I (no multiplier); the eigenvalues are unchanged by
    the row scaling."""
    w = np.ones(layout.total_size)
    for eq_name, u in record["principal"].items():
        if u in record["definitions"]:
            continue
        n = max(1, powers[eq_name])
        for suffix, loc in PLACEMENTS[record["placement"][u]]:
            f = layout.field(u + suffix)
            rv, tv = ops.nodes[LOC_CODE[loc]]
            rv = rv.copy()
            if np.isclose(rv[0], 0.0):
                rv[0] = ops.grids.r_center[0]
            w[f.slice] = np.repeat(rv, len(tv)).astype(float) ** n
    Drow = sp.diags(w.astype(complex), format="csr")
    return (Drow @ A).tocsr(), (Drow @ B).tocsr()


def _pole_face_rows(layout, ops: _CGridOps):
    """The theta-face rows of every face unknown at theta = 0 and theta =
    pi. The face has zero area and zero volume weight there, so the DOF is
    not one: the mimetic blocks leave its row empty (the strain Gram zeroes
    it through the norm guard, and reads the adjacent interior value
    through ``_pole_vtheta_prolongation`` at |m| = 1) or fill it with the
    1 / sin(theta) guard of a curl. Either way the row carries no
    equation."""
    Nt_e, Nr_c = len(ops.grids.theta_edge), len(ops.grids.r_center)
    rows = []
    for u, placement in layout.placements.items():
        if placement != "face":
            continue
        off = layout.field(u + "_t").offset
        rows += [off + ir * Nt_e + j for ir in range(Nr_c) for j in (0, Nt_e - 1)]
    return rows


def _apply_gauge_and_walls(A, B, layout, ops: _CGridOps, record: dict,
                           walls: dict, viscous: Dict[str, bool]):
    """The pole rings, the m = 0 multiplier gauge, then the wall rows of
    the normal components (inviscid: Dirichlet overwrite and the ball
    origin rows; viscous: the symmetric DOF reduction, the wall rows of the
    other impenetrable face unknowns dropped with it), in that order,
    which the seed certification pins. A potential keeps its wall rows
    and, on a ball, takes the regular vector's origin fold; a flux
    unknown's origin ring (zero-area faces) is pinned at every m; a free
    wall keeps its wall fluxes. Returns (A, B, layout, kept, pinned) with
    ``kept`` the full-state indices that survive and ``pinned`` the pole
    rings among them, as reduced-state indices."""
    grids, m = ops.grids, ops.m
    Nr_e, Nt_c = len(grids.r_edge), len(grids.theta_center)
    n_full = layout.total_size
    A, B = A.tolil(), B.tolil()

    def dirichlet_zero(row):
        A.rows[row] = [row]
        A.data[row] = [1.0 + 0j]
        B.rows[row] = []
        B.data[row] = []

    poles = _pole_face_rows(layout, ops)
    for row in poles:
        dirichlet_zero(row)
    for eq_name, info in record["equations"].items():
        if info["class"] != "saddle_point" or info["gauge"] is None:
            continue
        p = info["multiplier"]
        off = layout.field(p).offset
        if info["gauge"] == "point":
            dirichlet_zero(off)
        else:
            n = ops.n["CC"]
            A.rows[off] = list(range(off, off + n))
            A.data[off] = list(ops.W["CC"])
            B.rows[off] = []
            B.data[off] = []
    kept = np.arange(layout.total_size)

    def reduced(rows):
        """``rows`` (full-state) as reduced-state indices, the dropped ones
        left out."""
        where = -np.ones(n_full, dtype=int)
        where[kept] = np.arange(len(kept))
        out = where[np.asarray(sorted(rows), dtype=int)] if rows else np.empty(0, int)
        return out[out >= 0]
    face = [u for u, p in layout.placements.items() if p == "face"]
    flux = record["involutions"]
    regular = [u for u in face if u not in flux]
    if not face:
        return A.tocsr(), B.tocsr(), layout, kept, reduced(poles)
    lead, others = face[0], face[1:]
    if layout.field(lead + "_r").offset != 0:
        raise AssertionError("the leading face unknown must sit at offset 0")
    if ops.is_ball and len(regular) > 1:
        raise NotImplementedError(
            f"staggered: the face unknowns {regular[1:]} on a ball need the origin "
            f"ring of their normal component, which only the leading face "
            f"unknown {regular[0]!r} carries")

    def pinned_rows(u):
        """The rows of u's normal component that hold no DOF: the wall
        rings unless the walls are free or open or u is a potential, and
        on a ball the origin ring of a flux unknown (zero-area faces)."""
        off = layout.field(u + "_r").offset
        kind = walls["vectors"][u]["kind"]
        open_ = walls["vectors"][u]["open"]
        rows = []
        if kind not in ("free", "potential"):
            if Boundary.R_OUTER not in open_:
                rows += [off + (Nr_e - 1) * Nt_c + it for it in range(Nt_c)]
            if not ops.is_ball and Boundary.R_INNER not in open_:
                rows += [off + it for it in range(Nt_c)]
        if ops.is_ball and u in flux:
            rows += [off + it for it in range(Nt_c)]
        return rows
    extra = [r for u in others for r in pinned_rows(u)]
    kind = walls["vectors"][lead]["kind"]
    if viscous[lead] or (kind == "potential" and ops.is_ball):
        origin = "drop" if lead in flux else "regular"
        rings = kind != "potential"
        if rings and walls["vectors"][lead]["open"]:
            rings, extra = False, extra + pinned_rows(lead)
        A, B, red = _reduce_velocity_dofs(A.tocsr(), B.tocsr(), layout, grids, m, lead,
                                          extra_drop=extra, walls=rings, origin=origin)
        layout = dataclasses.replace(red, placements=layout.placements)
        drop = set(pinned_rows(lead)) | set(extra)
        if ops.is_ball and origin == "regular" and abs(m) >= 2:
            drop |= set(range(Nt_c))
        elif ops.is_ball and origin == "regular":
            drop |= set(range(1, Nt_c))
        kept = np.array([i for i in kept if i not in drop])
        return A, B, layout, kept, reduced(poles)
    for row in pinned_rows(lead):
        dirichlet_zero(row)
    if ops.is_ball and lead not in flux:
        _apply_ball_origin_vr(A, B, m, grids.theta_center, Nt_c)
    for row in extra:
        dirichlet_zero(row)
    return A.tocsr(), B.tocsr(), layout, kept, reduced(poles)


def _check_div_curl(A, asm: "_Assembly", record: dict):
    """The evolution rows of every involution unknown have a divergence-free
    image: ‖Div A_b‖ = 0 to rounding of the block maximum (Div Curl = 0 on
    the C-grid)."""
    D = asm.ops.flux_divergence()
    for b in record["involutions"]:
        rows = slice(asm.offset(b), asm.offset(b) + asm.size(b))
        Ab = A[rows, :]
        ref = abs(Ab).max() if Ab.nnz else 0.0
        X = D @ Ab
        dev = abs(X).max() if X.nnz else 0.0
        if dev > 1e-14 * ref:
            raise AssertionError(
                f"staggered: the evolution of the involution unknown {b!r} is not "
                f"divergence-free: max |Div A_b| = {dev:.3e} against the block "
                f"maximum {ref:.3e}")


def _divergence_operators(layout, kept, asm: "_Assembly", record: dict):
    """{b: D_red} with D_red the flux divergence (centres) of the involution
    unknown b read from the reduced state vector."""
    D = asm.ops.flux_divergence().tocsc()
    n_full = int(kept.max()) + 1 if len(kept) else 0
    n_full = max(n_full, asm.N)
    reduced_of = -np.ones(n_full, dtype=int)
    reduced_of[kept] = np.arange(len(kept))
    out = {}
    for b in record["involutions"]:
        off, size = asm.offset(b), asm.size(b)
        cols = np.arange(off, off + size)
        red = reduced_of[cols]
        ok = red >= 0
        P = sp.csr_matrix((np.ones(ok.sum()), (np.arange(size)[ok], red[ok])),
                          shape=(size, layout.total_size), dtype=complex)
        out[b] = (D @ P).tocsr()
    return out


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def _walk(system, m: int, params=None) -> List[OperatorTask]:
    tasks = walk_equation_system(system, m)
    if params:
        tasks = bind_parameters(tasks, params)
    return tasks


def general_path_reason(system, m: int, params=None) -> Optional[str]:
    """None when the assembler builds this system (the walker accepts it
    and every equation classifies as an evolution equation, a saddle-point
    constraint, an involution or a tensor definition), else the reason it
    does not."""
    try:
        tasks = _walk(system, m, params)
        classify_constraints(system, tasks)
    except NotImplementedError as exc:
        return str(exc)
    unknown = sorted({t.kind for t in tasks if t.kind not in SUPPORTED_KINDS})
    if unknown:
        return f"task kinds without a C-grid block: {unknown}"
    return None


def compile_staggered_general(system, grid, m: int, order_r: int = 2,
                              order_theta: int = 2, params=None,
                              coriolis_pairing: str = "center",
                              pole_closure: bool = False,
                              ) -> StaggeredCompiledMatrices:
    """Compile an EquationSystem on the C-grid through the task
    dispatcher.

    Walks the system (binding ``params``), classifies the constraints
    (``classify_constraints``), classifies the wall conditions, dispatches
    every task to its mimetic block (eliminated tensor definitions
    composed into the blocks that act on them), applies the ball
    premultiply of a B = I pencil, the m = 0 gauge, the wall rows and the
    ball origin, and returns ``StaggeredCompiledMatrices`` whose
    ``constraints`` attribute is the classification record extended with
    ``walls`` (the wall record), ``viscous`` (per face unknown, whether a
    second-order operator acts on it, which selects the symmetric wall
    reduction over the inviscid row overwrite), ``r_power`` (the ball
    premultiply per equation, None on a shell), ``divergence`` (per
    involution unknown, its flux divergence on the reduced state, the
    operator behind ``StaggeredCompiledMatrices.divergence``) and
    ``pinned`` (the reduced-state indices of the theta-face pole rings,
    which carry a Dirichlet row and no mass and hold no solution value).
    The evolution rows of an involution unknown are verified divergence-free
    at assembly. ``order_r`` / ``order_theta`` (even) set the stencil
    order; the lower of the two is used. ``coriolis_pairing`` selects the
    Coriolis / cross-product pairing; the Lorentz force of a flux unknown
    is always the constrained-transport recipe. ``pole_closure`` gives the
    strain Gram the pole closure of ``build_staggered_strain`` and
    ``_strain_pole_share`` (even |m| >= 2 only; off by default, since it
    moves the certified m = 8 pencil).
    """
    tasks = _walk(system, m, params)
    record = classify_constraints(system, tasks)
    order = min(order_r, order_theta)
    grids = StaggeredGrids.from_grid(grid)
    ops = _CGridOps(grids, m, order, pole_closure)
    by_eq: Dict[str, List[OperatorTask]] = {eq.name: [] for eq in system.equations}
    for t in tasks:
        by_eq.setdefault(t.equation_name, []).append(t)
    # the wall record decides which implied involutions are velocities
    walls = _classify_walls(system, m, record, ops, params)
    _demote_pinned_involutions(record, walls, by_eq)
    _check_involution_walls(record, walls, by_eq)
    for info in record["equations"].values():
        if info["class"] == "saddle_point" and info["gauge"] is not None:
            closure = walls["scalars"][info["multiplier"]]["closure"]
            if "dirichlet" in closure or walls["vectors"][info["vector"]]["open"]:
                info["gauge"] = None
    layout = _layout_for(system, record, ops)
    asm = _Assembly(system, tasks, ops, layout, record, walls, coriolis_pairing)
    for eq in system.equations:
        info = record["equations"][eq.name]
        if info["class"] in ("definition", "involution"):
            continue
        row = record["principal"][eq.name]
        if info["class"] == "saddle_point":
            (div_task,) = [t for t in asm.by_eq[eq.name]
                           if t.kind == TASK_DIV_VECTOR_SCALAR]
            closure = asm.grad_closure(info["vector"], info["multiplier"])
            Div, outer = asm.weighted_divergence(div_task, closure)
            vals = None if outer is None else ops.evaluate(outer, "CC")
            if vals is not None and not np.all(vals == vals.flat[0]):
                raise NotImplementedError(
                    f"staggered: the constraint {eq.name!r} scales Div({info['vector']}) "
                    f"by a varying background; write the weight inside the "
                    f"divergence, Div(M {info['vector']}), so the pair stays "
                    f"adjoint under the M-weighted norm")
            asm.add(TARGET_A, row, info["vector"], _scaled(Div, info["div_sign"], vals))
            continue
        for task in asm.by_eq[eq.name]:
            if task.target == TARGET_RHS:
                asm.source(task, row)
            else:
                asm.dispatch(task, row)
        if info.get("implicit_mass"):
            asm.add_to(TARGET_B, row, row,
                       sp.eye(asm.size(row), format="csr", dtype=complex))
    asm.open_wall_rows()
    A, B = asm.totals()
    _check_div_curl(A, asm, record)
    powers = None
    if ops.is_ball and not record["multipliers"] and not record["involutions"]:
        powers = dict(asm.r_power)
        A, B = _ball_premultiply(A, B, layout, ops, record, powers)
    viscous = dict(asm.second_order)
    A, B, layout, kept, pinned = _apply_gauge_and_walls(
        A, B, layout, ops, record, walls, viscous)
    div_ops = _divergence_operators(layout, kept, asm, record)
    record = dict(record, walls=walls, viscous=viscous, r_power=powers,
                  divergence=div_ops, pinned=pinned)
    return StaggeredCompiledMatrices(
        A_total=A, B_total=B, layout=layout, grids=grids, m=m,
        constraints=record, b_total=asm.b if asm.has_source else None)
