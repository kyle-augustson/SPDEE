"""spdee.backends.fd2.compiler — assemble collocated fd2 matrices from IR tasks.

Given an EquationSystem, a Grid and an azimuthal mode m, walk the
equations into OperatorTasks, pair each equation with a principal
unknown to fix the state layout, choose a per-equation mass
``diag(r^p)`` (p = largest natural r-power among the equation's
differential operators), and dispatch every task to its sparse block
builder, accumulating into A_total (non-Dt terms), B_total (Dt terms)
and b_total (sources).

Scope: scalar, vector and tensor equations built from the task kinds
listed in spdee.ir.tasks. Several operators may share one equation; each
block is scaled from its natural r-power to the equation's common
r-power, so no auxiliary unknowns are needed. A tensor unknown is stored
in its nine covariant spin components (tensor_unknown_sparse.py); a
tensor definition T - tensor(u) = 0 (spdee.ir.constraints, class
'definition') is eliminated at compile time by rewriting the tasks on T
into the equivalent tasks on u, and a written involution Div(b) = 0 with
no multiplier takes the classifier's divergence-cleaning prescription
(an auto-introduced multiplier, its Grad in b's evolution equation,
Dirichlet-zero walls and the mean-zero gauge; `implicit_bc_tasks`).
Boundary conditions are applied later by CompiledProblem
(compiled_problem.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from sympy import S as sp_S, Function as sp_Function, expand as sp_expand

from spdee.backends.common.field_eval import evaluate_vector_background_spin_2d
from spdee.backends.fd2.grid import Grid
from spdee.backends.fd2.discretization import (
    Discretization,
    DEFAULT_DISCRETIZATION,
    require_implemented,
)
from spdee.backends.fd2.layout import flatten_2d
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.state_layout import StateLayout
from spdee.ir.backgrounds import BackgroundComponent
from spdee.ir.equations import Equation, EquationSystem
from spdee.ir.expressions import Expression, FieldRef
from spdee.ir.fields import VectorField, _FieldBase
from spdee.ir.operators import Dt
from spdee.ir.tasks import (
    OperatorTask, BCTask, BC_DIRICHLET_SCALAR,
    TASK_IDENTITY_SCALAR, TASK_LAP_SCALAR, TASK_LAPH_SCALAR, TASK_SOURCE_SCALAR,
    TASK_IDENTITY_VECTOR, TASK_LAP_VECTOR, TASK_SOURCE_VECTOR,
    TASK_DIV_TENSOR_VECTOR, TASK_CURL_VECTOR, TASK_CROSS_VECTOR,
    TASK_CROSS_CURL_UNK_BG, TASK_CURL_CROSS_UNK_BG, TASK_CURL_CURL_VECTOR,
    TASK_CROSS_CURL_CURL_UNK_BG,
    TASK_GRAD_SCALAR, TASK_DIV_VECTOR_SCALAR, TASK_GRAD_DIV_VECTOR,
    TASK_DOT_BG_GRAD_VEC,
    TASK_DOT_BG_TENSOR_VEC, TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
    TASK_DOT_BG_VEC_GRAD_SCALAR,
    TASK_DOT_BG_VEC_TENSOR_UNK,
    TASK_BG_VEC_TIMES_UNK_SCALAR, TASK_BG_VEC_TIMES_DIV_UNK_VEC,
    TASK_IDENTITY_TENSOR, TASK_SOURCE_TENSOR, TASK_TENSOR_OF_VECTOR,
    TASK_DIV_TENSOR_UNK, TASK_DOT_BG_VEC_UNK_TENSOR,
    TARGET_A, TARGET_B,
)
from spdee.ir.walker import walk_equation_system


def _find_dt_unknown_in_expr(expr: Expression) -> Optional[str]:
    """Return the name of the first FieldRef found inside a Dt(...) node
    of `expr`, or None if no Dt appears."""
    if isinstance(expr, Dt):
        if isinstance(expr.operand, FieldRef):
            return expr.operand.field.name
        return _find_dt_unknown_in_expr(expr.operand)
    if isinstance(expr, FieldRef):
        return None
    for attr in ("operand", "a", "b", "scalar", "children"):
        if hasattr(expr, attr):
            v = getattr(expr, attr)
            if isinstance(v, Expression):
                found = _find_dt_unknown_in_expr(v)
                if found is not None:
                    return found
            elif isinstance(v, (tuple, list)):
                for c in v:
                    if isinstance(c, Expression):
                        found = _find_dt_unknown_in_expr(c)
                        if found is not None:
                            return found
    return None


def _principal_unknown_for_equation(
    eq: Equation,
    available_unknowns: Dict[str, "_FieldBase"],
) -> Optional[str]:
    """Identify the principal unknown of `eq`: the unknown whose column
    block is aligned with this equation's row block in the state layout.

    In priority order:
    1. An augmentation defining equation named ``__def_<aux>`` has
       principal unknown ``<aux>``.
    2. If the LHS contains ``Dt(FieldRef(X))``, X is the principal.
    3. Otherwise (a constraint such as Div(v) = 0) return None and let
       the caller pair by leftover matching.
    """
    name = eq.name or ""
    if name.startswith("__def_"):
        aux_name = name[len("__def_"):]
        if aux_name in available_unknowns:
            return aux_name
    return _find_dt_unknown_in_expr(eq.lhs)


# Task kinds that are differential operators and therefore carry a
# natural r-power. Identity, source and pointwise tasks (Cross, the
# bg·unk contractions) are excluded.
_DIFFERENTIAL_KINDS = {
    TASK_LAP_SCALAR, TASK_LAPH_SCALAR, TASK_LAP_VECTOR, TASK_DIV_TENSOR_VECTOR,
    TASK_CURL_VECTOR, TASK_GRAD_SCALAR, TASK_DIV_VECTOR_SCALAR,
    TASK_DOT_BG_GRAD_VEC, TASK_GRAD_DIV_VECTOR,
    TASK_DOT_BG_VEC_TENSOR_UNK,
    TASK_DOT_BG_VEC_GRAD_SCALAR,
    TASK_BG_VEC_TIMES_DIV_UNK_VEC,
    TASK_CROSS_CURL_UNK_BG, TASK_CURL_CROSS_UNK_BG, TASK_CURL_CURL_VECTOR,
    TASK_CROSS_CURL_CURL_UNK_BG, TASK_TENSOR_OF_VECTOR, TASK_DIV_TENSOR_UNK,
}

# Rows and columns of an unknown of each kind, in blocks of N_r·N_θ.
_BLOCKS_PER_KIND = {"scalar": 1, "vector": 3, "tensor": 9}


@dataclass(frozen=True)
class CompiledMatrices:
    """Output of compile_fd2.

    Attributes
    ----------
    A_total : scipy.sparse matrix, shape (N, N)
        Operator matrix; receives every non-Dt contribution.
    B_total : scipy.sparse matrix, shape (N, N)
        Eigenvalue-side matrix; receives Dt-marked contributions. Zero
        when no Dt term appears (pure BVP).
    b_total : np.ndarray, shape (N,)
        Source vector (source tasks multiplied by the equation mass).
    layout : StateLayout
        Unknown name → flat-state slice.
    grid : Grid
    m : int
        Azimuthal mode the problem is compiled for.
    equation_row_slices : dict
        Equation name → slice of rows in A_total/B_total.
    equation_mass_matrices : dict
        Equation name → the mass matrix ``diag(r^p)`` applied to that
        equation, or None when the mass is the identity.
    bundle : DerivativeBundle
        Cached derivative stencils, reused by BC application (e.g. the
        dense D1_θ used by tensor-derived BCs).
    paired_equation_of_unknown : dict
        Unknown name → the equation it is paired with. The origin rows,
        BC rows and gauge row of an unknown are written into the rows
        of its paired equation, whatever the column layout (see
        ball_origin.py and CompiledProblem); in the equation-order
        layout those rows start at the unknown's column offset.
    divergence_constraint_of_vector : dict
        Vector unknown name → the scalar equation that is an algebraic
        constraint on it alone (Div(v) = 0, Div(ρ₀ v) = 0: no Dt, no
        task on any other unknown). `CompiledProblem` may host the
        impenetrability condition Dot(r̂, v) = 0 on that equation's wall
        rows (`impenetrability_host`), keeping the wall-normal momentum
        row alive as the pressure's natural boundary condition.
    gauge_unknowns : tuple of str
        Scalar unknowns that receive an integral gauge constraint,
        the divergence-cleaning multipliers included.
    implicit_bc_tasks : tuple of BCTask
        Boundary rows the compiler adds itself (the Dirichlet-zero walls
        of a divergence-cleaning multiplier); `compile_problem` appends
        them to the walked ones.
    classification : ClassificationRecord or None
        The constraint classification (spdee.ir.constraints) when the
        compiler consulted it: a tensor unknown or a written involution.
    eliminated_tensors : dict
        Tensor unknown -> its defining equation, for the definitions
        eliminated at compile time (no DOFs in the layout).
    """
    A_total: sp.spmatrix
    B_total: sp.spmatrix
    b_total: np.ndarray
    layout: StateLayout
    grid: Grid
    m: int
    equation_row_slices: Dict[str, slice]
    equation_mass_matrices: Dict[str, Optional[sp.spmatrix]]
    bundle: DerivativeBundle
    paired_equation_of_unknown: Dict[str, str] = field(default_factory=dict)
    divergence_constraint_of_vector: Dict[str, str] = field(default_factory=dict)
    gauge_unknowns: Tuple[str, ...] = ()
    implicit_bc_tasks: Tuple[BCTask, ...] = ()
    classification: Optional[object] = None
    eliminated_tensors: Dict[str, str] = field(default_factory=dict)


# ===========================================================================
# Top-level compile entry point
# ===========================================================================

def compile_fd2(
    system: EquationSystem,
    grid: Grid,
    m: int,
    order_r: int = 4,
    order_theta: int = 4,
    discretization=DEFAULT_DISCRETIZATION,
    params: Optional[dict] = None,
    pre_walked_tasks=None,
    coriolis_pairing: str = "center",
    pole_closure: bool = False,
) -> CompiledMatrices:
    """Compile an EquationSystem into fd2 matrices.

    Parameters
    ----------
    system : EquationSystem
    grid : Grid
    m : int
        Azimuthal mode.
    order_r : int, default 4
        Even formal order ≥ 2 of the radial finite-difference stencils.
    order_theta : int, default 4
        Even formal order ≥ 2 of the angular finite-difference stencils,
        independent of `order_r`.
    discretization : Discretization or str, default COLLOCATED
        ``COLLOCATED`` is handled here. ``STAGGERED`` delegates to the
        Arakawa-C path in staggered_compile. ``SH_FD`` and ``DEDALUS``
        raise ``DiscretizationNotImplementedError``.
    params : dict, optional
        Values for symbolic parameters (e.g. Ek, Le) bound into the
        walked tasks.
    pre_walked_tasks : list of OperatorTask, optional
        Already-walked (and already parameter-bound) tasks supplied by
        a sweep driver; skips the walk.

    Raises
    ------
    NotImplementedError
        If an equation mixes differential operators of different output
        kinds, or contains an operator combination without a dispatcher.
    DiscretizationNotImplementedError
        If a stub discretization is selected.
    """
    method = Discretization.coerce(discretization)
    require_implemented(method)
    if method is Discretization.STAGGERED:
        from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
        return compile_fd2_staggered(
            system, grid, m, order_r=order_r, order_theta=order_theta,
            coriolis_pairing=coriolis_pairing, params=params, pole_closure=pole_closure,
        )
    # Walk the system into tasks (m folded in) and bind parameters. The
    # walk is parameter-independent, so a sweep may walk once and pass
    # the bound tasks through `pre_walked_tasks`.
    if pre_walked_tasks is not None:
        tasks = pre_walked_tasks
    else:
        tasks = walk_equation_system(system, m)
        if params:
            from spdee.ir.tasks import bind_parameters
            tasks = bind_parameters(tasks, params)
    kinds: Dict[str, str] = {name: fld.kind for name, fld in system.unknowns.items()}
    equations: List[Equation] = list(system.equations)
    gauge_unknowns = tuple(system.gauge_unknowns)
    implicit_bcs: List[BCTask] = []
    classification = None
    eliminated: Dict[str, str] = {}
    if any(kind == "tensor" for kind in kinds.values()):
        classification = _classify(system, tasks, grid)
        tasks, eliminated = _eliminate_definitions(tasks, classification, m)
        for name, eq_name in eliminated.items():
            del kinds[name]
            equations = [eq for eq in equations if eq.name != eq_name]
    tasks_by_equation: Dict[str, List[OperatorTask]] = {}
    for t in tasks:
        tasks_by_equation.setdefault(t.equation_name, []).append(t)

    # State layout. Every equation is paired with exactly one unknown of
    # the same kind. When any equation carries Dt(...) or auxiliary
    # unknowns are present, column blocks are ordered by equation so
    # that `entry.offset == equation_row_slices[paired_eq].start`;
    # otherwise unknowns are ordered alphabetically. The origin, BC and
    # gauge rows of an unknown always go to its paired equation's rows
    # (`paired_equation_of_unknown`), so the placement does not depend
    # on the column order.
    has_aux = any(name.startswith("__aux_") for name in kinds)
    has_dt = any(
        _find_dt_unknown_in_expr(eq.lhs) is not None
        for eq in equations
    )
    use_equation_order = has_aux or has_dt
    paired_unknown_of_equation: Dict[str, str] = {}
    used_unknowns: set = set()
    for eq in equations:
        principal = _principal_unknown_for_equation(eq, kinds)
        if principal is None:
            continue
        if principal not in kinds:
            raise ValueError(
                f"Equation {eq.name!r} principal unknown {principal!r} "
                f"is not in the system's unknown roster."
            )
        if principal in used_unknowns:
            raise ValueError(
                f"Equation {eq.name!r} principal unknown {principal!r} "
                f"is already paired with another equation. Each unknown "
                f"can only be the principal for one equation."
            )
        expected_kind = kinds[principal]
        if expected_kind != eq.kind:
            raise ValueError(
                f"Equation {eq.name!r} (kind {eq.kind}) is paired with "
                f"unknown {principal!r} (kind {expected_kind}); kinds "
                f"must match."
            )
        paired_unknown_of_equation[eq.name] = principal
        used_unknowns.add(principal)
    # Remaining equations take the remaining unknowns of matching kind,
    # alphabetically. A divergence constraint Div(b) = 0 left without a
    # scalar to host it is an involution: the classifier's
    # divergence-cleaning prescription introduces its multiplier.
    remaining_unknowns_by_kind: Dict[str, List[str]] = {
        kind: [] for kind in _BLOCKS_PER_KIND}
    for name, kind in sorted(kinds.items()):
        if name not in used_unknowns:
            remaining_unknowns_by_kind[kind].append(name)
    divergence_constraints = _divergence_constraints(kinds, equations, tasks_by_equation)
    for eq in equations:
        if eq.name in paired_unknown_of_equation:
            continue
        pool = remaining_unknowns_by_kind[eq.kind]
        if not pool and eq.name in divergence_constraints.values():
            if classification is None:
                classification = _classify(system, tasks, grid)
            vector = next(v for v, e in divergence_constraints.items() if e == eq.name)
            cleaning = classification.divergence_cleaning.get(vector)
            if cleaning is not None and cleaning.constraint == eq.name:
                multiplier = cleaning.multiplier
                kinds[multiplier] = "scalar"
                pool.append(multiplier)
                tasks_by_equation.setdefault(cleaning.evolution, []).append(OperatorTask(
                    kind=TASK_GRAD_SCALAR, target=TARGET_A,
                    equation_name=cleaning.evolution, unknown_name=multiplier,
                    metadata={"m": m},
                ))
                # the Dirichlet walls fix the multiplier's constant; on a
                # ball at m = 0 the collocated gradient also has a
                # checkerboard null vector in the multiplier, which the
                # mean-zero row pins at the cost of one spurious mode far
                # from the physical band
                if grid.is_ball and m == 0:
                    gauge_unknowns = gauge_unknowns + (multiplier,)
                implicit_bcs.extend(
                    BCTask(kind=BC_DIRICHLET_SCALAR, location=wall,
                           unknown_name=multiplier, name=f"{multiplier}_{wall}")
                    for wall in cleaning.walls)
        if not pool:
            raise ValueError(
                f"Equation {eq.name!r} (kind {eq.kind}) has no matching "
                f"unknown left to pair with. System may be under-"
                f"determined or pairings inconsistent."
            )
        principal = pool.pop(0)
        paired_unknown_of_equation[eq.name] = principal
        used_unknowns.add(principal)
    for name in kinds:
        if name not in used_unknowns:
            raise ValueError(
                f"Unknown {name!r} is not paired with any equation. "
                f"System may be over-determined."
            )
    primary: List[Tuple[str, str]] = []
    if use_equation_order:
        for eq in equations:
            name = paired_unknown_of_equation[eq.name]
            primary.append((name, kinds[name]))
    else:
        primary = sorted(kinds.items())
    layout = StateLayout(
        primary=primary,
        N_r=grid.N_r,
        N_theta=grid.N_theta,
    )

    # Equation row slices, in equation order: a scalar equation owns
    # N_r·N_θ rows, a vector equation 3·N_r·N_θ, a tensor one 9·N_r·N_θ.
    block_scalar = grid.N_r * grid.N_theta
    equation_row_slices: Dict[str, slice] = {}
    row_offset = 0
    for eq in equations:
        if not eq.name:
            raise ValueError(
                f"Equation must have a name for compiler dispatch: {eq!r}"
            )
        if eq.name in equation_row_slices:
            raise ValueError(f"Duplicate equation name {eq.name!r}")
        eq_rows = _BLOCKS_PER_KIND[eq.kind] * block_scalar
        equation_row_slices[eq.name] = slice(row_offset, row_offset + eq_rows)
        row_offset += eq_rows
    if row_offset != layout.size:
        raise ValueError(
            f"Number of equation rows ({row_offset}) does not match the "
            f"total unknown size ({layout.size}). The system must be "
            f"square. Check that you have one equation per unknown "
            f"with matching kinds."
        )

    # Per-equation mass diag(r^p), p = max natural r-power over the
    # equation's differential tasks (0 when there are none).
    bundle = DerivativeBundle(grid, order_r=order_r, order_theta=order_theta)
    equation_mass_matrices: Dict[str, Optional[sp.spmatrix]] = {}
    equation_r_power_target: Dict[str, int] = {}
    for eq_name, eq_tasks in tasks_by_equation.items():
        equation_mass_matrices[eq_name] = _determine_equation_mass(
            eq_name, eq_tasks, bundle, m
        )
        diff_tasks = [t for t in eq_tasks if t.kind in _DIFFERENTIAL_KINDS]
        if diff_tasks:
            equation_r_power_target[eq_name] = max(
                _natural_r_power(t) for t in diff_tasks
            )
        else:
            equation_r_power_target[eq_name] = 0

    # Dispatch every task into COO triple builders; assemble CSR once.
    N = layout.size
    A_total = _TripleBuilder((N, N), dtype=complex)
    B_total = _TripleBuilder((N, N), dtype=complex)
    b_total = np.zeros(N, dtype=complex)
    from spdee.backends.fd2.profiling import perf_now, record, is_enabled
    prof = is_enabled()
    for eq_name, eq_tasks in tasks_by_equation.items():
        row_slice = equation_row_slices[eq_name]
        A_eq = equation_mass_matrices[eq_name]
        r_power_target = equation_r_power_target[eq_name]
        for task in eq_tasks:
            if task.kind in _DIFFERENTIAL_KINDS:
                _attach_multiop_metadata(task, r_power_target, grid)
            if prof: t0 = perf_now()
            _dispatch_task(
                task=task,
                row_slice=row_slice,
                A_eq=A_eq,
                layout=layout,
                grid=grid,
                bundle=bundle,
                A_total=A_total,
                B_total=B_total,
                b_total=b_total,
            )
            if prof: record(f"compile/dispatch/{task.kind}", perf_now() - t0)

    if prof: t0 = perf_now()
    A_csr = A_total.to_csr()
    B_csr = B_total.to_csr()
    if prof: record("compile/final_csr_conversion", perf_now() - t0)

    return CompiledMatrices(
        A_total=A_csr,
        B_total=B_csr,
        b_total=b_total,
        layout=layout,
        grid=grid,
        m=m,
        equation_row_slices=equation_row_slices,
        equation_mass_matrices=equation_mass_matrices,
        bundle=bundle,
        paired_equation_of_unknown={
            uk: eq for eq, uk in paired_unknown_of_equation.items()
        },
        divergence_constraint_of_vector=divergence_constraints,
        gauge_unknowns=gauge_unknowns,
        implicit_bc_tasks=tuple(implicit_bcs),
        classification=classification,
        eliminated_tensors=eliminated,
    )


def _classify(system: EquationSystem, tasks: List[OperatorTask], grid: Grid):
    """The constraint classification of the system for this non-mimetic
    core (spdee.ir.constraints.classify with mimetic=False)."""
    from spdee.ir.constraints import classify
    return classify(system, tasks, geometry=grid.geometry, mimetic=False,
                    core="collocated")


def _eliminate_definitions(tasks: List[OperatorTask], record, m: int):
    """Eliminate every tensor definition T - tensor(u) = 0 of `record`
    (class 'definition'): the tasks of the defining equation go, and a
    task on T elsewhere becomes the equivalent task on u with the
    definition substituted, r T_{σ_a σ_b} = -(1/c) Σ_k r T_k[σ_a σ_b] for
    the identity coefficient c and the walker's r-scaled components of
    the tensor terms. Div(T) becomes the closed-form Div of a tensor
    expression (TASK_DIV_TENSOR_VECTOR) and Dot(B0, T) / Dot(T, B0) the
    contraction with the tensor expression (TASK_DOT_BG_VEC_TENSOR_UNK),
    so the eliminated pencil is the strain-form pencil. Returns the
    rewritten tasks and {tensor: defining equation}."""
    import dataclasses
    from spdee.ir.backgrounds import SymPyComponent
    from spdee.ir.tensor_reduce import r2_div_tensor_spin_components
    from spdee.backends.fd2.layout import SPIN_COMPONENTS_TENSOR
    from spdee.ir.fields import r as r_sym, theta as t_sym
    maps = {}
    for T, eq_name in record.definitions.items():
        info = record.equations[eq_name]
        identity = info["identity"]
        c = identity.coefficient
        if c is None:
            c_expr = sp_S(1)
        elif isinstance(c, SymPyComponent):
            c_expr = c.expr
        else:
            raise NotImplementedError(
                f"collocated: the coefficient of {T!r} in its definition "
                f"{eq_name!r} is a numeric background; a SymPy expression is needed "
                f"to eliminate the tensor")
        c_expr = identity.sign * c_expr
        if c_expr == 0:
            raise NotImplementedError(
                f"collocated: the coefficient of {T!r} in its definition {eq_name!r} "
                f"vanishes; the tensor cannot be eliminated")
        vector = info["vector"]
        if vector is None:
            raise NotImplementedError(
                f"collocated: the definition {eq_name!r} of {T!r} has no vector term")
        rT = {key: sp_S(0) for key in SPIN_COMPONENTS_TENSOR}
        v_placeholders = None
        for task, _ab in info["terms"]:
            v_placeholders = task.metadata["v_placeholders"]
            for key in SPIN_COMPONENTS_TENSOR:
                rT[key] = rT[key] + task.metadata["rT_components"][key]
        rT = {key: sp_expand(-rT[key] / c_expr) for key in SPIN_COMPONENTS_TENSOR}
        maps[T] = (eq_name, vector, v_placeholders, rT)
    out = []
    for t in tasks:
        if t.unknown_name not in maps:
            if t.equation_name in {eq for eq, *_ in maps.values()}:
                continue
            out.append(t)
            continue
        eq_name, vector, v_placeholders, rT = maps[t.unknown_name]
        if t.equation_name == eq_name:
            continue
        if t.kind == TASK_DIV_TENSOR_UNK:
            label = {-1: "m", 0: "0", +1: "p"}
            U = {key: sp_Function(f"U_div_{vector}_{label[key[0]]}_{label[key[1]]}")(
                r_sym, t_sym) for key in SPIN_COMPONENTS_TENSOR}
            div = r2_div_tensor_spin_components(U, m)
            out.append(dataclasses.replace(
                t, kind=TASK_DIV_TENSOR_VECTOR, unknown_name=vector,
                metadata={**t.metadata, "per_sigma_expr_in_U": {
                    s: sp_expand(div[s]) for s in (-1, 0, +1)},
                    "U_placeholders": U, "U_in_v": rT,
                    "v_placeholders": v_placeholders,
                    "grad_transpose_weights": None, "origin_factor": None},
            ))
        elif t.kind == TASK_DOT_BG_VEC_UNK_TENSOR:
            side = t.metadata.get("side", "left")
            comps = rT if side == "left" else {
                (a, b): rT[(b, a)] for (a, b) in SPIN_COMPONENTS_TENSOR}
            out.append(dataclasses.replace(
                t, kind=TASK_DOT_BG_VEC_TENSOR_UNK, unknown_name=vector,
                metadata={**t.metadata, "rT_components": comps,
                          "v_placeholders": v_placeholders,
                          "tensor_symmetric": None, "grad_transpose_weights": None},
            ))
        else:
            raise NotImplementedError(
                f"collocated: the {t.kind!r} term on the eliminated tensor "
                f"{t.unknown_name!r} in {t.equation_name!r} has no vector form; Div "
                f"and Dot with a background vector are supported")
    return out, {T: eq for T, (eq, *_rest) in maps.items()}


def _divergence_constraints(
    kinds: Dict[str, str],
    equations: List[Equation],
    tasks_by_equation: Dict[str, List[OperatorTask]],
) -> Dict[str, str]:
    """{vector unknown: scalar equation} for every scalar equation that
    is an algebraic divergence constraint on one vector unknown: a
    Div(v) task, no Dt, and no task on any other unknown (sources
    allowed). An evolution equation Dt(ρ) + Div(ρ₀ v) = 0 is not one."""
    out: Dict[str, str] = {}
    for eq in equations:
        if eq.kind != "scalar":
            continue
        tasks = tasks_by_equation.get(eq.name, [])
        names = {t.unknown_name for t in tasks if t.unknown_name is not None}
        if len(names) != 1:
            continue
        name = names.pop()
        if kinds[name] != "vector":
            continue
        if any(t.target == TARGET_B for t in tasks):
            continue
        if any(t.kind == TASK_DIV_VECTOR_SCALAR for t in tasks):
            out[name] = eq.name
    return out


# ===========================================================================
# Internal: determining the mass matrix of an equation
# ===========================================================================

def _determine_equation_mass(
    eq_name: str,
    eq_tasks: List[OperatorTask],
    bundle: DerivativeBundle,
    m: int,
) -> Optional[sp.spmatrix]:
    """Return the mass matrix A_eq of equation `eq_name`, or None when
    the equation has no differential task (mass = identity).

    The mass is ``diag(r^p)`` on each output block, p being the largest
    natural r-power among the equation's differential tasks. In ball
    mode the i_r=0 rows of the mass are identity placeholders.

    Natural r-powers:
      Identity, Cross, Source       : 0
      Grad, Curl, Div(vector)       : 1
      Lap, Div(tensor), Curl(Curl)  : 2
      tensor(vector) (r T)          : 1
    """
    diff_tasks = [t for t in eq_tasks if t.kind in _DIFFERENTIAL_KINDS]
    if not diff_tasks:
        return None
    max_power = max(_natural_r_power(t) for t in diff_tasks)
    output_kinds = set(_task_output_kind(t) for t in diff_tasks)
    if len(output_kinds) > 1:
        raise NotImplementedError(
            f"Equation {eq_name!r} has differential operators producing "
            f"mixed output kinds {output_kinds}. Multi-operator support "
            f"requires uniform output (all vector or all scalar)."
        )
    output_kind = output_kinds.pop()
    from spdee.backends.fd2.operators import (
        r1_vector_mass, r2_vector_mass, r1_scalar_mass,
    )
    if output_kind == "tensor":
        if max_power == 1:
            from spdee.backends.fd2.tensor_unknown_sparse import r1_tensor_mass
            return r1_tensor_mass(bundle)
        raise NotImplementedError(
            f"max_power={max_power} not supported for tensor equations"
        )
    if output_kind == "vector":
        if max_power == 1:
            return r1_vector_mass(bundle)
        if max_power == 2:
            return r2_vector_mass(bundle)
        raise NotImplementedError(
            f"max_power={max_power} not supported for vector equations"
        )
    elif output_kind == "scalar":
        if max_power == 1:
            return r1_scalar_mass(bundle)
        if max_power == 2:
            return _r2_scalar_mass(bundle)
        raise NotImplementedError(
            f"max_power={max_power} not supported for scalar equations"
        )
    else:
        raise AssertionError(f"unexpected output_kind {output_kind!r}")


def _natural_r_power(task: OperatorTask) -> int:
    """Return the natural r-power of a differential task: the smallest
    n ≥ 0 such that ``diag(r^n) · op(·)`` is polynomial in r near r=0
    (1/r factors need n=1, 1/r² factors need n=2)."""
    kind = task.kind
    if kind in (TASK_LAP_SCALAR, TASK_LAPH_SCALAR, TASK_LAP_VECTOR,
                TASK_DIV_TENSOR_VECTOR, TASK_GRAD_DIV_VECTOR,
                TASK_CURL_CURL_VECTOR, TASK_CROSS_CURL_CURL_UNK_BG,
                TASK_DIV_TENSOR_UNK):
        return 2
    if kind in (TASK_CURL_VECTOR, TASK_GRAD_SCALAR, TASK_DIV_VECTOR_SCALAR,
                TASK_DOT_BG_GRAD_VEC, TASK_DOT_BG_VEC_TENSOR_UNK,
                TASK_DOT_BG_VEC_GRAD_SCALAR,
                TASK_BG_VEC_TIMES_DIV_UNK_VEC,
                TASK_CROSS_CURL_UNK_BG, TASK_CURL_CROSS_UNK_BG,
                TASK_TENSOR_OF_VECTOR):
        return 1
    if kind in (TASK_CROSS_VECTOR, TASK_DOT_BG_TENSOR_VEC,
                TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
                TASK_BG_VEC_TIMES_UNK_SCALAR, TASK_DOT_BG_VEC_UNK_TENSOR):
        return 0
    raise NotImplementedError(
        f"Natural r-power unknown for task kind {kind!r}"
    )


def _task_output_kind(task: OperatorTask) -> str:
    """Return 'scalar', 'vector' or 'tensor' for the output of this
    task's operator."""
    kind = task.kind
    if kind in (TASK_LAP_SCALAR, TASK_LAPH_SCALAR, TASK_DIV_VECTOR_SCALAR,
                TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
                TASK_DOT_BG_VEC_GRAD_SCALAR):
        return "scalar"
    if kind in (
        TASK_LAP_VECTOR, TASK_DIV_TENSOR_VECTOR, TASK_CURL_VECTOR,
        TASK_GRAD_SCALAR, TASK_CROSS_VECTOR, TASK_DOT_BG_GRAD_VEC,
        TASK_DOT_BG_TENSOR_VEC, TASK_BG_VEC_TIMES_UNK_SCALAR,
        TASK_GRAD_DIV_VECTOR, TASK_DOT_BG_VEC_TENSOR_UNK,
        TASK_BG_VEC_TIMES_DIV_UNK_VEC,
        TASK_CROSS_CURL_UNK_BG, TASK_CURL_CROSS_UNK_BG, TASK_CURL_CURL_VECTOR,
        TASK_CROSS_CURL_CURL_UNK_BG, TASK_DIV_TENSOR_UNK,
        TASK_DOT_BG_VEC_UNK_TENSOR,
    ):
        return "vector"
    if kind == TASK_TENSOR_OF_VECTOR:
        return "tensor"
    raise NotImplementedError(
        f"Output kind unknown for task kind {kind!r}"
    )


def _r2_scalar_mass(bundle: DerivativeBundle) -> sp.spmatrix:
    """(N, N) mass ``diag(r²)`` for scalar equations with max_power=2,
    with identity placeholders on the i_r=0 rows in ball mode."""
    from spdee.backends.fd2.operators import diag_from_2d_array
    grid = bundle.grid
    N_r, N_theta = grid.N_r, grid.N_theta
    R = grid.r[:, None] * np.ones((1, N_theta))
    diag_r2 = diag_from_2d_array(R ** 2)
    A_op = sp.lil_matrix(diag_r2, dtype=complex)
    if grid.is_ball:
        for j in range(N_theta):
            row = j * N_r + 0
            A_op.rows[row] = [row]
            A_op.data[row] = [1.0]
    return A_op.tocsr()


def _operator_constant_factor(task: OperatorTask) -> float:
    """Constant factor baked into the operator's natural mass.

    Grad's natural mass is ``diag(r·√2)``, so its factor is √2;
    Dot(bg_vec, Grad(scalar)) is built from the same Grad block and
    inherits it. Every other operator has factor 1.
    """
    if task.kind in (TASK_GRAD_SCALAR, TASK_DOT_BG_VEC_GRAD_SCALAR):
        return float(np.sqrt(2.0))
    return 1.0


def _attach_multiop_metadata(
    task: OperatorTask, r_power_target: int, grid: "Grid",
) -> None:
    """Store in ``task.metadata['mass_conversion_diag_2d']`` the
    (N_r, N_θ) diagonal that scales the operator's natural block
    ``diag(r^p_op · k_op) · op`` to the equation mass ``diag(r^p_eq)``:
    ``r^(p_eq − p_op) / k_op`` at interior rows and ``1 / k_op`` at the
    ball-mode i_r=0 rows, which hold the operator's r→0 limit in the
    builder's k_op-scaled form.
    """
    if task.metadata is None:
        task.metadata = {}
    p_op = _natural_r_power(task)
    k_op = _operator_constant_factor(task)
    delta = r_power_target - p_op
    if delta < 0:
        raise AssertionError(
            f"r_power_target ({r_power_target}) < natural r-power "
            f"({p_op}) for task kind {task.kind!r}"
        )
    R = grid.r[:, None] * np.ones((1, grid.N_theta))
    conv = (R ** delta) / k_op
    if grid.is_ball:
        conv[0, :] = 1.0 / k_op
    task.metadata["mass_conversion_diag_2d"] = conv


def _apply_mass_conversion(
    contribution: sp.spmatrix,
    conv_2d: np.ndarray,
    output_kind: str,
    grid: "Grid",
) -> sp.spmatrix:
    """Left-multiply `contribution` ((N, N_in) for scalar output,
    (3N, N_in) for vector output) by the diagonal built from `conv_2d`,
    broadcast over the three spin blocks for vector output."""
    diag_flat = flatten_2d(conv_2d.astype(complex))
    if output_kind == "vector":
        diag_3N = np.concatenate([diag_flat, diag_flat, diag_flat])
        D = sp.diags(diag_3N, format="csr", dtype=complex)
    elif output_kind == "scalar":
        D = sp.diags(diag_flat, format="csr", dtype=complex)
    elif output_kind == "tensor":
        D = sp.diags(np.tile(diag_flat, 9), format="csr", dtype=complex)
    else:
        raise AssertionError(f"unexpected output_kind {output_kind!r}")
    return D @ contribution


def _finalize_contribution(
    contribution: sp.spmatrix,
    task: OperatorTask,
    A_eq: Optional[sp.spmatrix],
    grid: "Grid",
    *,
    output_kind: str,
    pointwise: bool,
) -> sp.spmatrix:
    """Apply origin limits, mass alignment, coefficient and sign to a
    builder's block.

    1. Origin limit (ball grids, differential blocks). The builders
       replace the i_r=0 rows of the natural block (0 = 0) by the
       operator's r→0 limit (ball_origin.install_origin_rows) so that,
       with the identity placeholders of the masses and the 1/k_op
       mass conversion at i_r=0, the i_r=0 rows of A_total hold the
       unscaled equation at the origin.
    2. Mass alignment. A differential builder returns its block in
       natural form (``diag(r^p_op·k_op) · op(v) = block · v``); it is
       scaled by ``metadata['mass_conversion_diag_2d']`` to the equation
       mass ``diag(r^p_eq)``. A pointwise block (Cross, Identity, the
       bg·unk contractions) has no natural r-power and is left-multiplied
       by A_eq directly (no mass when A_eq is None).
    3. Coefficient. A scalar background coefficient multiplies the row
       side; for vector output it broadcasts over the three spin blocks.
    4. Sign. ``task.sign`` is ±1, set by the walker from Negation nodes.
    """
    if pointwise:
        if A_eq is not None:
            contribution = A_eq @ contribution
    else:
        conv_2d = task.metadata.get("mass_conversion_diag_2d")
        if conv_2d is not None:
            contribution = _apply_mass_conversion(
                contribution, conv_2d, output_kind=output_kind, grid=grid,
            )
    if task.coefficient is not None:
        if output_kind == "vector":
            contribution = _apply_scalar_coefficient_vector(
                contribution, task.coefficient, grid,
            )
        elif output_kind == "tensor":
            contribution = _apply_scalar_coefficient_vector(
                contribution, task.coefficient, grid, blocks=9,
            )
        else:
            contribution = _apply_coefficient(
                contribution, task.coefficient, grid,
            )
    return task.sign * contribution


# ===========================================================================
# Internal: per-task dispatch
# ===========================================================================

def _dispatch_task(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Add one task's contribution to the totals, by task kind."""
    builder = _DISPATCHERS.get(task.kind)
    if builder is None:
        raise NotImplementedError(
            f"the collocated core has no builder for the task kind "
            f"{task.kind!r} (equation {task.equation_name!r}); the Dedalus, "
            f"sh_fd and staggered cores may support it"
        )
    builder(task, row_slice, A_eq, layout, grid, bundle, A_total, B_total, b_total)


def _dispatch_lap_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Lap(scalar_unknown). Differential, scalar output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"LAP_SCALAR task with target={task.target} not supported; "
            f"LAP is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    m = task.metadata.get("m", 0)
    from spdee.backends.fd2.lap_sparse import build_r2_lap_scalar_block
    contribution = build_r2_lap_scalar_block(bundle, m, task.unknown_name)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="scalar", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_laph_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """LapH(scalar_unknown) — horizontal (angular) Laplacian.
    Differential, scalar output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"LAPH_SCALAR task with target={task.target} not supported; "
            f"LapH is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    m = task.metadata.get("m", 0)
    from spdee.backends.fd2.lap_sparse import build_r2_laph_scalar_block
    contribution = build_r2_laph_scalar_block(bundle, m, task.unknown_name)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="scalar", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_identity_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Identity on a scalar unknown: ``sign · diag(coef) · A_eq`` (A_eq
    replaced by I when None), routed to A or B by the task target."""
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    if A_eq is None:
        block = grid.N_r * grid.N_theta
        contribution = sp.eye(block, dtype=complex, format="csr")
    else:
        contribution = A_eq.copy()
    if task.coefficient is not None:
        coef_diag = _coefficient_as_diag(task.coefficient, grid)
        contribution = coef_diag @ contribution
    contribution = task.sign * contribution
    if task.target == TARGET_A:
        _accumulate(A_total, row_slice, col_slice, contribution)
    elif task.target == TARGET_B:
        _accumulate(B_total, row_slice, col_slice, contribution)
    else:
        raise NotImplementedError(
            f"IDENTITY_SCALAR with target={task.target} not supported"
        )


def _dispatch_source_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Scalar source: the equation reads ``... + sign·coef = 0``, so the
    right-hand side receives ``A_eq @ (−sign · coef_grid_flat)`` (no
    mass when A_eq is None)."""
    assert task.coefficient is not None
    coef_flat = _coefficient_as_flat(task.coefficient, grid)
    contribution = (-task.sign) * coef_flat
    if A_eq is not None:
        contribution = A_eq @ contribution
    b_total[row_slice] = b_total[row_slice] + contribution


# ===========================================================================
# Vector dispatchers
# ===========================================================================

def _dispatch_lap_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Lap(vector_unknown). Differential, vector output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"LAP_VECTOR task with target={task.target} not supported; "
            f"LAP is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    m = task.metadata.get("m", 0)
    from spdee.backends.fd2.lap_sparse import build_r2_lap_vector_block
    walker_meta_for_lap = {
        "m": m,
        "v_placeholders": task.metadata["v_placeholders"],
    }
    contribution = build_r2_lap_vector_block(walker_meta_for_lap, bundle, m)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_grad_div_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Grad(Div(unknown_vec)), built as one composed block (no auxiliary
    unknown). Differential, vector output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"GRAD_DIV_VECTOR task with target={task.target} not supported; "
            f"Grad(Div(v)) is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    m = task.metadata.get("m", 0)
    from spdee.backends.fd2.grad_div_sparse import build_r2_grad_div_vector_block
    walker_meta_for_grad_div = {
        "m": m,
        "v_placeholders": task.metadata["v_placeholders"],
    }
    contribution = build_r2_grad_div_vector_block(
        walker_meta_for_grad_div, bundle, m,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_div_tensor_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Div(tensor_expr) → vector. Differential, vector output, r-power 2.
    The walker reduces the tensor expression to per-(σ_a, σ_b) covariant
    SymPy expressions in v-placeholders; the builder discretizes them."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DIV_TENSOR_VECTOR task with target={task.target} not supported; "
            f"Div is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.div_tensor_sparse import build_div_tensor_vector_block
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    contribution = build_div_tensor_vector_block(task.metadata, bundle, m_mode)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_curl_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Curl(vector_unknown). Differential, vector output, r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CURL_VECTOR task with target={task.target} not supported; "
            f"Curl is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import build_curl_vector_block
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    contribution = build_curl_vector_block(task.metadata, bundle, m_mode)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_curl_curl_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Curl(Curl(vector_unknown)), the Curl formula composed with itself
    and multiplied through by r². Differential, vector output, r-power 2
    (it equals Grad(Div(v)) − Lap(v) block for block)."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CURL_CURL_VECTOR task with target={task.target} not supported; "
            f"Curl(Curl(v)) is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import (
        build_r2_curl_curl_vector_block,
    )
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    contribution = build_r2_curl_curl_vector_block(task.metadata, bundle, m_mode)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_cross_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Cross(background_vector, vector_unknown) (``side`` selects which
    factor is the background). Pointwise, vector output, r-power 0."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CROSS_VECTOR task with target={task.target} not supported; "
            f"Cross is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import build_cross_vector_block
    m_mode = task.metadata.get("m", 0)
    side = task.metadata.get("side", "left")
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError("Cross task missing source_field (background vector)")
    R, T = grid.mesh()
    b_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    contribution = build_cross_vector_block(b_spin, bundle, m_mode, side=side)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=True,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_cross_curl_unk_bg(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Cross(Curl(unknown_vec), bg_vec) or Cross(bg_vec, Curl(unknown_vec)).

    Differential (inner Curl), vector output, r-power 1, built as the
    product X(B₀) @ C of the elementary Cross and Curl blocks with no
    auxiliary unknown. ``side='right'`` is Cross(Curl(B), B₀),
    ``side='left'`` is Cross(B₀, Curl(B)).
    """
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CROSS_CURL_UNK_BG task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import (
        build_cross_curl_unk_bg_block,
    )
    m_mode = task.metadata.get("m", 0)
    side = task.metadata.get("side", "right")
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "CROSS_CURL_UNK_BG task missing source_field (background vector)"
        )
    contribution = build_cross_curl_unk_bg_block(
        bg_field, bundle, grid, m_mode, side=side,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_curl_cross_unk_bg(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Curl(Cross(unknown_vec, bg_vec)) or Curl(Cross(bg_vec, unknown_vec)).

    Differential (outer Curl), vector output, r-power 1, built as the
    product C @ X(B₀) with no auxiliary unknown. ``inner_side='right'``
    is Curl(Cross(v, B₀)), ``inner_side='left'`` is Curl(Cross(B₀, v)).
    """
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CURL_CROSS_UNK_BG task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import (
        build_curl_cross_unk_bg_block,
    )
    m_mode = task.metadata.get("m", 0)
    inner_side = task.metadata.get("inner_side", "right")
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "CURL_CROSS_UNK_BG task missing source_field (background vector)"
        )
    contribution = build_curl_cross_unk_bg_block(
        bg_field, bundle, grid, m_mode, inner_side=inner_side,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_dot_bg_grad_vec(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(bg_vector, Grad(unknown_vec)) — advection (v0·∇)v.
    Differential, vector output, r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_GRAD_VEC task with target={task.target} not supported; "
            f"the operator is always on A"
        )
    assert task.unknown_name is not None
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError("DOT_BG_GRAD_VEC task missing source_field "
                         "(background vector)")
    from spdee.backends.fd2.advection_sparse import (
        build_r_advection_vector_block,
    )
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    R, T = grid.mesh()
    b_grid = evaluate_vector_background_spin_2d(bg_field, R, T)
    v_placeholders_dict = task.metadata.get("v_placeholders")
    if v_placeholders_dict is None:
        raise ValueError("DOT_BG_GRAD_VEC task missing v_placeholders metadata")
    v_placeholders = v_placeholders_dict[task.unknown_name]
    contribution = build_r_advection_vector_block(
        b_grid=b_grid,
        v_placeholders=v_placeholders,
        bundle=bundle,
        m=m_mode,
        v_unknown_name=task.unknown_name,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_dot_bg_vec_tensor_unk(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(bg_vec, tensor_expr(unknown_vec)). The walker reduces the
    tensor expression to per-(σ_a, σ_b) covariant spin components in
    v-placeholders, multiplied by r to clear 1/r factors; the builder
    contracts with the background vector as
        r·(Dot(a, T))^σ_out = Σ_σ_a a^σ_a · (r·T)_{σ_a, -σ_out}.
    Differential, vector output, r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_VEC_TENSOR_UNK task with target={task.target} not "
            f"supported; the operator is always on A"
        )
    assert task.unknown_name is not None
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "DOT_BG_VEC_TENSOR_UNK task missing source_field "
            "(background vector)"
        )
    from spdee.backends.fd2.dot_bg_vec_tensor_unk_sparse import (
        build_r_dot_bg_vec_tensor_unk_block,
    )
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    R, T = grid.mesh()
    a_grid = evaluate_vector_background_spin_2d(bg_field, R, T)
    v_placeholders_dict = task.metadata.get("v_placeholders")
    rT_components = task.metadata.get("rT_components")
    if v_placeholders_dict is None or rT_components is None:
        raise ValueError(
            "DOT_BG_VEC_TENSOR_UNK task missing v_placeholders or "
            "rT_components metadata"
        )
    v_placeholders = v_placeholders_dict[task.unknown_name]
    contribution = build_r_dot_bg_vec_tensor_unk_block(
        a_grid=a_grid,
        rT_components=rT_components,
        v_placeholders=v_placeholders,
        bundle=bundle,
        m=m_mode,
        v_unknown_name=task.unknown_name,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_dot_bg_tensor_vec(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(bg_tensor, unknown_vec). Pointwise matvec M(r,θ)·v with
    M's spin components diagonal-block-wise. Pointwise, vector output,
    r-power 0."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_TENSOR_VEC task with target={task.target} not "
            f"supported; this operator is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "DOT_BG_TENSOR_VEC task missing source_field (background tensor)"
        )
    from spdee.backends.common.field_eval import (
        evaluate_tensor_background_matvec_spin_2d,
    )
    R, T = grid.mesh()
    M_spin = evaluate_tensor_background_matvec_spin_2d(bg_field, R, T)
    spin_order = (-1, 0, +1)
    sub_blocks = {}
    from spdee.backends.fd2.operators_core import diag_from_2d_array
    for s_out in spin_order:
        for s_in in spin_order:
            sub_blocks[(s_out, s_in)] = diag_from_2d_array(
                M_spin[(s_out, s_in)]
            ).tocsr()
    contribution = sp.bmat([
        [sub_blocks[(-1, -1)], sub_blocks[(-1,  0)], sub_blocks[(-1, +1)]],
        [sub_blocks[( 0, -1)], sub_blocks[( 0,  0)], sub_blocks[( 0, +1)]],
        [sub_blocks[(+1, -1)], sub_blocks[(+1,  0)], sub_blocks[(+1, +1)]],
    ], format="csr", dtype=complex)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=True,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_dot_bg_vec_unk_vec_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(bg_vec, unknown_vec) → scalar. Pointwise inner product,
    scalar output, r-power 0.

    Spin contraction: ``s = a^0 b^0 + a^+ b^- + a^- b^+``."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_VEC_UNK_VEC_SCALAR task with target={task.target} "
            f"not supported; this operator is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "DOT_BG_VEC_UNK_VEC_SCALAR task missing source_field "
            "(background vector)"
        )
    R, T = grid.mesh()
    bg_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    from spdee.backends.fd2.operators_core import diag_from_2d_array
    sub_m = diag_from_2d_array(bg_spin[+1]).tocsr()  # coeff of unk σ=-1: a^+
    sub_0 = diag_from_2d_array(bg_spin[0]).tocsr()   # coeff of unk σ=0:  a^0
    sub_p = diag_from_2d_array(bg_spin[-1]).tocsr()  # coeff of unk σ=+1: a^-
    contribution = sp.bmat(
        [[sub_m, sub_0, sub_p]],
        format="csr", dtype=complex,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="scalar", pointwise=True,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_dot_bg_vec_grad_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(bg_vec, Grad(scalar_unknown)) → scalar. Differential, scalar
    output, natural r-power 1 with Grad's √2 factor.

    ``build_grad_scalar_block`` returns the (3N, N) block enforcing
    ``diag(r·√2)·(Grad p)^σ = block · p``, rows stacked (σ=-1, 0, +1).
    Contracting with the background vector a by the spin inner product
        a·∇p = a^0 (∇p)^0 + a^+ (∇p)^- + a^- (∇p)^+
    gives one (N, N) scalar block carrying the same r·√2 natural mass,
    which the mass conversion then removes exactly as for a bare Grad.
    """
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_VEC_GRAD_SCALAR task with target={task.target} "
            f"not supported; this operator is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "DOT_BG_VEC_GRAD_SCALAR task missing source_field "
            "(background vector)"
        )
    from spdee.backends.fd2.grad_div_sparse import build_grad_scalar_block
    from spdee.backends.fd2.operators_core import diag_from_2d_array
    m_mode = task.metadata.get("m", 0)
    grad_block = build_grad_scalar_block(bundle, m_mode, task.unknown_name)
    N_r, N_theta = grid.N_r, grid.N_theta
    block = N_r * N_theta
    gp_m = grad_block[0 * block:1 * block, :]   # (∇p)^{σ=-1} rows
    gp_0 = grad_block[1 * block:2 * block, :]   # (∇p)^{σ=0}  rows
    gp_p = grad_block[2 * block:3 * block, :]   # (∇p)^{σ=+1} rows
    R, T = grid.mesh()
    bg_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    a_m = diag_from_2d_array(bg_spin[-1]).tocsr()  # a^-
    a_0 = diag_from_2d_array(bg_spin[0]).tocsr()   # a^0
    a_p = diag_from_2d_array(bg_spin[+1]).tocsr()  # a^+
    contribution = (a_0 @ gp_0 + a_p @ gp_m + a_m @ gp_p).tocsr()
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="scalar", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_bg_vec_times_unk_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """unknown_scalar × bg_vector → vector. Pointwise scalar-vector
    product, vector output, r-power 0."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"BG_VEC_TIMES_UNK_SCALAR task with target={task.target} "
            f"not supported; this operator is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "BG_VEC_TIMES_UNK_SCALAR task missing source_field "
            "(background vector)"
        )
    R, T = grid.mesh()
    bg_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    from spdee.backends.fd2.operators_core import diag_from_2d_array
    sub_m = diag_from_2d_array(bg_spin[-1]).tocsr()  # row σ=-1
    sub_0 = diag_from_2d_array(bg_spin[0]).tocsr()   # row σ=0
    sub_p = diag_from_2d_array(bg_spin[+1]).tocsr()  # row σ=+1
    contribution = sp.bmat(
        [[sub_m], [sub_0], [sub_p]],
        format="csr", dtype=complex,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=True,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_bg_vec_times_div_unk_vec(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """bg_vec · Div(unknown_vec) → vector: the (N, 3N) ``r·Div(v)`` block
    stacked under a per-output-σ diagonal of the background spin
    components. Differential, vector output, r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"BG_VEC_TIMES_DIV_UNK_VEC task with target={task.target} "
            f"not supported; this operator is always on A"
        )
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    bg_field = task.source_field
    if bg_field is None:
        raise ValueError(
            "BG_VEC_TIMES_DIV_UNK_VEC task missing source_field "
            "(background vector)"
        )
    from spdee.backends.fd2.grad_div_sparse import (
        build_div_vector_scalar_block,
    )
    from spdee.backends.fd2.operators_core import diag_from_2d_array
    m_mode = task.metadata.get("m", 0)
    v_placeholders_dict = task.metadata.get("v_placeholders")
    if v_placeholders_dict is None:
        raise ValueError(
            "BG_VEC_TIMES_DIV_UNK_VEC task missing v_placeholders metadata"
        )
    r_div_v_block = build_div_vector_scalar_block(
        walker_meta={"m": m_mode, "v_placeholders": v_placeholders_dict},
        bundle=bundle,
        m=m_mode,
    )
    R, T = grid.mesh()
    bg_spin = evaluate_vector_background_spin_2d(bg_field, R, T)
    sub_m = (diag_from_2d_array(bg_spin[-1]) @ r_div_v_block).tocsr()
    sub_0 = (diag_from_2d_array(bg_spin[0]) @ r_div_v_block).tocsr()
    sub_p = (diag_from_2d_array(bg_spin[+1]) @ r_div_v_block).tocsr()
    contribution = sp.bmat(
        [[sub_m], [sub_0], [sub_p]],
        format="csr", dtype=complex,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_grad_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Grad(scalar_unknown). Differential, vector output, r-power 1
    with a √2 in the natural mass — handled by the mass-conversion
    overlay via the `_operator_constant_factor`."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"GRAD_SCALAR task with target={task.target} not supported; "
            f"Grad is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.grad_div_sparse import build_grad_scalar_block
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    contribution = build_grad_scalar_block(bundle, m_mode, task.unknown_name)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_div_vector_scalar(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Div(unknown_vec) → scalar. Differential, scalar output,
    r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DIV_VECTOR_SCALAR task with target={task.target} not supported; "
            f"Div is always on A"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.grad_div_sparse import build_div_vector_scalar_block
    m_mode = task.metadata.get("m", 0)
    col_slice = layout.field_slice(task.unknown_name)
    contribution = build_div_vector_scalar_block(task.metadata, bundle, m_mode)
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="scalar", pointwise=False,
    )
    _accumulate(A_total, row_slice, col_slice, contribution)


def _dispatch_identity_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Place an identity contribution for a vector unknown.

    Identity acts on the 3N column block. The equation's mass A_eq
    (size 3N x 3N, or None for identity) acts from the left:
      contribution = sign * coef * A_eq
    """
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    if A_eq is None:
        size_3N = 3 * grid.N_r * grid.N_theta
        contribution = sp.eye(size_3N, dtype=complex, format="csr")
    else:
        contribution = A_eq.copy()
    if task.coefficient is not None:
        contribution = _apply_scalar_coefficient_vector(
            contribution, task.coefficient, grid
        )
    contribution = task.sign * contribution
    if task.target == TARGET_A:
        _accumulate(A_total, row_slice, col_slice, contribution)
    elif task.target == TARGET_B:
        _accumulate(B_total, row_slice, col_slice, contribution)
    else:
        raise NotImplementedError(
            f"IDENTITY_VECTOR with target={task.target} not supported"
        )


def _dispatch_source_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Vector source: the background VectorField is evaluated to spin
    components, packed as a flat 3N vector in spin order (-1, 0, +1) and
    multiplied by the scalar coefficient (broadcast over spin blocks).
    As for the scalar source, ``... + sign·g = 0`` puts
    ``A_eq @ (−sign · g)`` on the right-hand side.
    """
    assert task.source_field is not None
    field = task.source_field
    if not isinstance(field, VectorField) or not field.is_background:
        raise TypeError(
            f"SOURCE_VECTOR task expects a background VectorField; "
            f"got {type(field).__name__}"
        )
    R, Theta = grid.mesh()
    spin = evaluate_vector_background_spin_2d(field, R, Theta)
    N = grid.N_r * grid.N_theta
    g_flat = np.empty(3 * N, dtype=complex)
    for k, sigma in enumerate((-1, 0, +1)):
        g_flat[k * N:(k + 1) * N] = flatten_2d(spin[sigma])
    if task.coefficient is not None:
        coef_flat = _coefficient_as_flat(task.coefficient, grid)
        coef_3N = np.tile(coef_flat, 3)
        g_flat = coef_3N * g_flat
    contribution = (-task.sign) * g_flat
    if A_eq is not None:
        contribution = A_eq @ contribution
    b_total[row_slice] = b_total[row_slice] + contribution


# ===========================================================================
# Vector potential and tensor-unknown dispatchers
# ===========================================================================

def _dispatch_cross_curl_curl_unk_bg(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Cross(Curl(Curl(A)), bg) (side 'right') or Cross(bg, Curl(Curl(A)))
    ('left'), the Lorentz force of the vector potential: the pointwise
    Cross block times the r² Curl(Curl) block. Differential, vector
    output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"CROSS_CURL_CURL_UNK_BG task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.curl_cross_sparse import (
        build_cross_curl_curl_unk_bg_block,
    )
    if task.source_field is None:
        raise ValueError(
            "CROSS_CURL_CURL_UNK_BG task missing source_field (background vector)"
        )
    contribution = build_cross_curl_curl_unk_bg_block(
        task.source_field, bundle, grid, task.metadata.get("m", 0),
        side=task.metadata.get("side", "right"),
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, layout.field_slice(task.unknown_name), contribution)


def _dispatch_identity_tensor(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Identity on the nine component blocks of a tensor unknown:
    ``sign · coef · A_eq`` (A_eq replaced by I when None), routed to A
    or B by the task target."""
    assert task.unknown_name is not None
    col_slice = layout.field_slice(task.unknown_name)
    if A_eq is None:
        contribution = sp.eye(9 * grid.N_r * grid.N_theta, dtype=complex, format="csr")
    else:
        contribution = A_eq.copy()
    if task.coefficient is not None:
        contribution = _apply_scalar_coefficient_vector(
            contribution, task.coefficient, grid, blocks=9,
        )
    contribution = task.sign * contribution
    if task.target == TARGET_A:
        _accumulate(A_total, row_slice, col_slice, contribution)
    elif task.target == TARGET_B:
        _accumulate(B_total, row_slice, col_slice, contribution)
    else:
        raise NotImplementedError(
            f"IDENTITY_TENSOR with target={task.target} not supported"
        )


def _dispatch_source_tensor(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Tensor source: the background's covariant spin components
    T_{σ_a σ_b} = T^{-σ_a, -σ_b} on the grid, packed in the tensor
    component order and put on the right-hand side as ``A_eq @ (-sign ·
    coef · g)``."""
    from spdee.backends.common.field_eval import evaluate_tensor_background_spin_2d
    from spdee.backends.fd2.layout import SPIN_COMPONENTS_TENSOR
    assert task.source_field is not None
    R, Theta = grid.mesh()
    spin = evaluate_tensor_background_spin_2d(task.source_field, R, Theta)
    N = grid.N_r * grid.N_theta
    g_flat = np.empty(9 * N, dtype=complex)
    for k, (a, b) in enumerate(SPIN_COMPONENTS_TENSOR):
        g_flat[k * N:(k + 1) * N] = flatten_2d(spin[(-a, -b)])
    if task.coefficient is not None:
        g_flat = np.tile(_coefficient_as_flat(task.coefficient, grid), 9) * g_flat
    contribution = (-task.sign) * g_flat
    if A_eq is not None:
        contribution = A_eq @ contribution
    b_total[row_slice] = b_total[row_slice] + contribution


def _dispatch_tensor_of_vector(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """A tensor expression of a vector unknown as a term of a tensor
    equation, from the walker's r T_{σ_a σ_b}. Differential, tensor
    output, r-power 1."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"TENSOR_OF_VECTOR task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.tensor_unknown_sparse import build_r_tensor_of_vector_block
    contribution = build_r_tensor_of_vector_block(
        task.metadata["rT_components"],
        task.metadata["v_placeholders"][task.unknown_name],
        bundle, task.metadata.get("m", 0), task.unknown_name,
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="tensor", pointwise=False,
    )
    _accumulate(A_total, row_slice, layout.field_slice(task.unknown_name), contribution)


def _dispatch_div_tensor_unk(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Div(T) of a tensor unknown kept in the state. Differential,
    vector output, r-power 2."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DIV_TENSOR_UNK task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.tensor_unknown_sparse import build_r2_div_tensor_unk_block
    contribution = build_r2_div_tensor_unk_block(
        task.unknown_name, bundle, task.metadata.get("m", 0),
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=False,
    )
    _accumulate(A_total, row_slice, layout.field_slice(task.unknown_name), contribution)


def _dispatch_dot_bg_vec_unk_tensor(
    task: OperatorTask,
    row_slice: slice,
    A_eq: Optional[sp.spmatrix],
    layout: StateLayout,
    grid: Grid,
    bundle: DerivativeBundle,
    A_total: _TripleBuilder,
    B_total: _TripleBuilder,
    b_total: np.ndarray,
) -> None:
    """Dot(B0, T) ('left') or Dot(T, B0) ('right') of a tensor unknown
    kept in the state. Pointwise, vector output, r-power 0."""
    if task.target != TARGET_A:
        raise NotImplementedError(
            f"DOT_BG_VEC_UNK_TENSOR task with target={task.target} not supported"
        )
    assert task.unknown_name is not None
    from spdee.backends.fd2.tensor_unknown_sparse import build_dot_bg_vec_unk_tensor_block
    if task.source_field is None:
        raise ValueError(
            "DOT_BG_VEC_UNK_TENSOR task missing source_field (background vector)"
        )
    R, T = grid.mesh()
    a_grid = evaluate_vector_background_spin_2d(task.source_field, R, T)
    contribution = build_dot_bg_vec_unk_tensor_block(
        a_grid, bundle, task.metadata.get("side", "left"),
    )
    contribution = _finalize_contribution(
        contribution, task, A_eq, grid,
        output_kind="vector", pointwise=True,
    )
    _accumulate(A_total, row_slice, layout.field_slice(task.unknown_name), contribution)


_DISPATCHERS = {
    TASK_LAP_SCALAR: _dispatch_lap_scalar,
    TASK_LAPH_SCALAR: _dispatch_laph_scalar,
    TASK_IDENTITY_SCALAR: _dispatch_identity_scalar,
    TASK_SOURCE_SCALAR: _dispatch_source_scalar,
    TASK_LAP_VECTOR: _dispatch_lap_vector,
    TASK_GRAD_DIV_VECTOR: _dispatch_grad_div_vector,
    TASK_IDENTITY_VECTOR: _dispatch_identity_vector,
    TASK_SOURCE_VECTOR: _dispatch_source_vector,
    TASK_DIV_TENSOR_VECTOR: _dispatch_div_tensor_vector,
    TASK_CURL_VECTOR: _dispatch_curl_vector,
    TASK_CURL_CURL_VECTOR: _dispatch_curl_curl_vector,
    TASK_CROSS_VECTOR: _dispatch_cross_vector,
    TASK_CROSS_CURL_UNK_BG: _dispatch_cross_curl_unk_bg,
    TASK_CURL_CROSS_UNK_BG: _dispatch_curl_cross_unk_bg,
    TASK_GRAD_SCALAR: _dispatch_grad_scalar,
    TASK_DIV_VECTOR_SCALAR: _dispatch_div_vector_scalar,
    TASK_DOT_BG_GRAD_VEC: _dispatch_dot_bg_grad_vec,
    TASK_DOT_BG_VEC_TENSOR_UNK: _dispatch_dot_bg_vec_tensor_unk,
    TASK_DOT_BG_TENSOR_VEC: _dispatch_dot_bg_tensor_vec,
    TASK_DOT_BG_VEC_UNK_VEC_SCALAR: _dispatch_dot_bg_vec_unk_vec_scalar,
    TASK_DOT_BG_VEC_GRAD_SCALAR: _dispatch_dot_bg_vec_grad_scalar,
    TASK_BG_VEC_TIMES_UNK_SCALAR: _dispatch_bg_vec_times_unk_scalar,
    TASK_BG_VEC_TIMES_DIV_UNK_VEC: _dispatch_bg_vec_times_div_unk_vec,
    TASK_CROSS_CURL_CURL_UNK_BG: _dispatch_cross_curl_curl_unk_bg,
    TASK_IDENTITY_TENSOR: _dispatch_identity_tensor,
    TASK_SOURCE_TENSOR: _dispatch_source_tensor,
    TASK_TENSOR_OF_VECTOR: _dispatch_tensor_of_vector,
    TASK_DIV_TENSOR_UNK: _dispatch_div_tensor_unk,
    TASK_DOT_BG_VEC_UNK_TENSOR: _dispatch_dot_bg_vec_unk_tensor,
}


# ===========================================================================
# Internal: coefficient helpers
# ===========================================================================

def _coefficient_as_flat(
    coef: BackgroundComponent, grid: Grid
) -> np.ndarray:
    """Evaluate a BackgroundComponent on the grid, flattened with r
    varying fastest (the state layout order)."""
    R, Theta = grid.mesh()
    arr2d = np.asarray(coef.evaluate_on_grid(R, Theta))
    if not np.all(np.isfinite(arr2d)):
        bad = np.argwhere(~np.isfinite(arr2d))
        i, j = bad[0]
        raise ValueError(
            f"Coefficient {getattr(coef, 'name', coef)!r} is not finite on "
            f"the grid at {len(bad)} node(s), e.g. (r, θ) = "
            f"({grid.r[i]:.6g}, {grid.theta[j]:.6g}). Backgrounds must be "
            f"finite at every node, including r = 0 on a ball."
        )
    return flatten_2d(arr2d)


def _coefficient_as_diag(
    coef: BackgroundComponent, grid: Grid
) -> sp.spmatrix:
    """Build a diagonal sparse matrix from a coefficient's grid values."""
    flat = _coefficient_as_flat(coef, grid)
    return sp.diags(flat, dtype=complex, format="csr")


def _apply_coefficient(
    M: sp.spmatrix, coef: BackgroundComponent, grid: Grid
) -> sp.spmatrix:
    """Left-multiply M by diag(coef): the coefficient multiplies the
    output (row) side of the operator."""
    diag = _coefficient_as_diag(coef, grid)
    return diag @ M


def _apply_scalar_coefficient_vector(
    M: sp.spmatrix, coef: BackgroundComponent, grid: Grid, blocks: int = 3,
) -> sp.spmatrix:
    """Left-multiply a (3N, ·) matrix (or (9N, ·) with ``blocks=9``) by a
    scalar coefficient evaluated on the grid and broadcast across the
    spin blocks (e.g. ``rho0 * Lap(v)`` scales every spin row by rho0)."""
    coef_flat = _coefficient_as_flat(coef, grid)
    coef_3N = np.tile(coef_flat, blocks)
    diag = sp.diags(coef_3N, dtype=complex, format="csr")
    return diag @ M


class _TripleBuilder:
    """Sparse matrix accumulator built from (row, col, value) triples.

    `add_block(row_slice, col_slice, contribution)` appends the COO
    entries of `contribution`, offset by the slice starts; `to_csr()`
    assembles one CSR matrix, summing duplicate (row, col) entries.
    """

    __slots__ = ("shape", "dtype", "_rows", "_cols", "_data")

    def __init__(self, shape, dtype=complex):
        self.shape = shape
        self.dtype = dtype
        self._rows: List[np.ndarray] = []
        self._cols: List[np.ndarray] = []
        self._data: List[np.ndarray] = []

    def add_block(self, row_slice: slice, col_slice: slice, contribution) -> None:
        """Add a contribution into [row_slice, col_slice] of the matrix."""
        coo = sp.coo_matrix(contribution)
        if coo.nnz == 0:
            return
        self._rows.append(coo.row + row_slice.start)
        self._cols.append(coo.col + col_slice.start)
        self._data.append(coo.data.astype(self.dtype, copy=False))

    def to_csr(self) -> sp.csr_matrix:
        """Assemble into a CSR matrix. Duplicates are summed."""
        if not self._rows:
            return sp.csr_matrix(self.shape, dtype=self.dtype)
        rows = np.concatenate(self._rows)
        cols = np.concatenate(self._cols)
        data = np.concatenate(self._data)
        return sp.coo_matrix(
            (data, (rows, cols)), shape=self.shape, dtype=self.dtype
        ).tocsr()


def _accumulate(
    target: _TripleBuilder,
    row_slice: slice,
    col_slice: slice,
    contribution: sp.spmatrix,
) -> None:
    """Accumulate `contribution` into target[row_slice, col_slice]."""
    target.add_block(row_slice, col_slice, contribution)
