"""spdee.ir.walker — traverse an EquationSystem and emit OperatorTask lists.

The walker is backend-agnostic: it walks each equation's expression
tree and produces OperatorTask records (see tasks.py) describing the
matrix contributions a backend assembles.

Supported equation form:

    sum of (background_coefficient * operator(unknown))
        + sum of (background_source) = 0

for scalar and vector equations. Tensor-kind nodes are never walked at
the top level; they appear only as operands of Div(...) and Dot(...),
where they are reduced to spin components by tensor_reduce.

Patterns and the tasks they emit
--------------------------------
    unknown scalar / vector           IDENTITY_SCALAR / IDENTITY_VECTOR
                                      (TARGET_B inside Dt, else TARGET_A)
    background scalar / vector        SOURCE_SCALAR / SOURCE_VECTOR (TARGET_RHS)
    Lap(s), Lap(v)                    LAP_SCALAR, LAP_VECTOR
    LapH(s)                           LAPH_SCALAR
    Grad(s)                           GRAD_SCALAR
    Grad(M s), M a background scalar  GRAD_SCALAR (M folded in) +
                                      BG_VEC_TIMES_UNK_SCALAR (s Grad M),
                                      tagged 'grad_weight' as a pair;
                                      Grad distributes over sums of such
    Grad(Div(v))                      GRAD_DIV_VECTOR
    Div(v)                            DIV_VECTOR_SCALAR
    Div(M v), M a background scalar   DIV_VECTOR_SCALAR (M folded in) +
                                      DOT_BG_VEC_UNK_VEC_SCALAR (Grad M . v),
                                      tagged 'div_weight' as a pair
    Div(tensor_expr(v))               DIV_TENSOR_VECTOR
    Curl(v)                           CURL_VECTOR
    Curl(Cross(v, B0)), Curl(Cross(B0, v))    CURL_CROSS_UNK_BG
    Curl(Curl(v))                     CURL_CURL_VECTOR
    Cross(B0, v), Cross(v, B0)        CROSS_VECTOR
    Cross(Curl(v), B0), Cross(B0, Curl(v))    CROSS_CURL_UNK_BG
    Cross(Curl(Curl(v)), B0), Cross(B0, Curl(Curl(v)))    CROSS_CURL_CURL_UNK_BG
    Dot(B0, Grad(v))                  DOT_BG_GRAD_VEC
    Dot(B0, tensor_expr(v))           DOT_BG_VEC_TENSOR_UNK
    Dot(tensor_expr(v), B0)           DOT_BG_VEC_TENSOR_UNK on the
                                      transposed expression
    Dot(T0, v)                        DOT_BG_TENSOR_VEC
    Dot(v, T0)                        DOT_BG_TENSOR_VEC on the transposed
                                      background
    Dot(B0, v), Dot(v, B0)            DOT_BG_VEC_UNK_VEC_SCALAR
    Dot(B0, Grad(s)), Dot(Grad(s), B0)        DOT_BG_VEC_GRAD_SCALAR
    s * B0                            BG_VEC_TIMES_UNK_SCALAR
    Div(v) * B0                       BG_VEC_TIMES_DIV_UNK_VEC
    Div(T), Dot(B0, T), Dot(T, B0)    DIV_TENSOR_UNK, DOT_BG_VEC_UNK_TENSOR
                                      (T an unknown tensor)

and, in a tensor equation (one whose lhs is a tensor),

    unknown tensor T                  IDENTITY_TENSOR (TARGET_B inside Dt)
    tensor_expr(v)                    TENSOR_OF_VECTOR (reduced components)
    background tensor                 SOURCE_TENSOR

where s / v are unknown scalars / vectors and B0 / T0 background
vectors / tensors. Sum descends, Negation flips the sign, a background
ScalarMultiple folds its factor into the accumulated coefficient, and
Dt marks its subtree. A background leaf inside Dt is dropped with a
warning (its time derivative is zero). Differential operators inside
Dt, and every other node combination, raise NotImplementedError.

A background carries no azimuthal dependence. The pure-background part
of a tensor expression under Div(...) or Dot(B0, ...) (a background
tensor leaf, Grad of a background vector) is therefore split off after
reduction, contracted at m = 0 and emitted as a SOURCE_VECTOR; the
background part of a tensor-projection boundary condition moves to the
BC value.

Boundary conditions are walked separately by `walk_boundary_conditions`
into BCTask records.
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional
import warnings

import numpy as np
import sympy as sp

from spdee.ir.backgrounds import (
    BackgroundComponent, NumericComponent, SymPyComponent,
    multiply_components,
)
from spdee.ir.expressions import (
    Expression, FieldRef, _BackgroundScalarValue,
    Sum, Negation, ScalarMultiple, AsSymmetric,
    background_scalar_component,
)
from spdee.ir.operators import Lap, LapH, Dt, Div, Curl, Cross, Grad, Dot, Transpose
from spdee.ir.equations import Equation, EquationSystem
from spdee.ir.fields import (
    FieldRole, ScalarField, TensorField, VectorField, r as _r_sym,
    theta as _t_sym,
)
from spdee.ir.tasks import (
    OperatorTask, BCTask,
    TASK_IDENTITY_SCALAR, TASK_LAP_SCALAR, TASK_LAPH_SCALAR, TASK_SOURCE_SCALAR,
    TASK_IDENTITY_VECTOR, TASK_LAP_VECTOR, TASK_SOURCE_VECTOR,
    TASK_DIV_TENSOR_VECTOR, TASK_CURL_VECTOR, TASK_CROSS_VECTOR,
    TASK_CROSS_CURL_UNK_BG, TASK_CURL_CROSS_UNK_BG,
    TASK_CURL_CURL_VECTOR, TASK_CROSS_CURL_CURL_UNK_BG,
    TASK_GRAD_SCALAR, TASK_DIV_VECTOR_SCALAR, TASK_GRAD_DIV_VECTOR,
    TASK_DOT_BG_GRAD_VEC,
    TASK_DOT_BG_TENSOR_VEC, TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
    TASK_DOT_BG_VEC_GRAD_SCALAR,
    TASK_DOT_BG_VEC_TENSOR_UNK,
    TASK_BG_VEC_TIMES_UNK_SCALAR, TASK_BG_VEC_TIMES_DIV_UNK_VEC,
    TASK_IDENTITY_TENSOR, TASK_TENSOR_OF_VECTOR, TASK_DIV_TENSOR_UNK,
    TASK_DOT_BG_VEC_UNK_TENSOR, TASK_SOURCE_TENSOR,
    TARGET_A, TARGET_B, TARGET_RHS,
    BC_DIRICHLET_SCALAR, BC_DIRICHLET_VECTOR_PROJECTION,
    BC_SCALAR_LINEAR_FROM_TENSOR, BC_NEUMANN_SCALAR_RADIAL,
    BC_PROJECTION_CURL_CROSS, BC_LINEAR_COMBINATION, BC_TANGENT_MIXED,
    BC_UNIT_R_HAT, BC_UNIT_THETA_HAT, BC_UNIT_PHI_HAT,
)
from spdee.ir.tensor_reduce import (
    reduce_tensor_expression, vector_unknown_spin_placeholders,
    r2_div_tensor_spin_components,
)


# ===========================================================================
# Node predicates
# ===========================================================================

def _is_field_ref(expr, kind: str, background: bool) -> bool:
    """True if expr is a FieldRef of the given kind and role."""
    return (
        isinstance(expr, FieldRef)
        and expr.field.kind == kind
        and expr.field.is_background == background
    )


def _is_bg_vector(expr) -> bool:
    return _is_field_ref(expr, "vector", True)


def _is_bg_tensor(expr) -> bool:
    return _is_field_ref(expr, "tensor", True)


def _is_unknown_scalar(expr) -> bool:
    return _is_field_ref(expr, "scalar", False)


def _is_unknown_vector(expr) -> bool:
    return _is_field_ref(expr, "vector", False)


def _is_unknown_tensor(expr) -> bool:
    return _is_field_ref(expr, "tensor", False)


def _is_curl_of_unknown_vector(expr) -> bool:
    return isinstance(expr, Curl) and _is_unknown_vector(expr.operand)


def _is_curl_curl_of_unknown_vector(expr) -> bool:
    return isinstance(expr, Curl) and _is_curl_of_unknown_vector(expr.operand)


def _is_div_of_unknown_vector(expr) -> bool:
    return isinstance(expr, Div) and _is_unknown_vector(expr.operand)


def _is_grad_of_unknown_vector(expr) -> bool:
    return isinstance(expr, Grad) and _is_unknown_vector(expr.operand)


def _is_grad_of_unknown_scalar(expr) -> bool:
    return isinstance(expr, Grad) and _is_unknown_scalar(expr.operand)


def _placeholders_for(field: VectorField) -> dict:
    """{field.name: spin placeholders} for one unknown vector."""
    return {field.name: vector_unknown_spin_placeholders(field)}


# ===========================================================================
# Walker state
# ===========================================================================

class _WalkContext:
    """Mutable context passed down one equation's expression tree.

    Tracks the accumulated sign, the accumulated background coefficient
    (a BackgroundComponent, initially the SymPy constant 1), and whether
    the walk is inside a Dt(...). ScalarMultiple factors are folded in
    with `multiply_components`, so SymPy and numeric coefficients
    compose without special-casing.
    """
    __slots__ = (
        "sign", "coefficient_component", "in_dt", "equation_name",
        "tasks", "m",
    )

    def __init__(self, equation_name: str, m: int):
        self.sign: int = +1
        self.coefficient_component: BackgroundComponent = SymPyComponent(sp.S.One)
        self.in_dt: bool = False
        self.equation_name: str = equation_name
        self.tasks: List[OperatorTask] = []
        self.m: int = m

    def emit_task(
        self,
        kind: str,
        unknown_name: Optional[str],
        *,
        target: str = TARGET_A,
        source_field=None,
        extra_metadata: Optional[dict] = None,
        coefficient_override: Optional[BackgroundComponent] = None,
    ) -> None:
        """Append an OperatorTask filled from the current context.

        Parameters
        ----------
        kind : str
            Task kind constant (TASK_*).
        unknown_name : str or None
            Unknown this task acts on (None for source tasks).
        target : str
            TARGET_A (default), TARGET_B, or TARGET_RHS.
        source_field : Field, optional
            Background field referenced by the task.
        extra_metadata : dict, optional
            Merged into the metadata on top of {"m": self.m}.
        coefficient_override : BackgroundComponent, optional
            Coefficient to use instead of the accumulated one.
        """
        metadata = {"m": self.m}
        if extra_metadata is not None:
            metadata.update(extra_metadata)
        if coefficient_override is None:
            coefficient = _materialize_coefficient(self.coefficient_component)
        else:
            coefficient = _materialize_coefficient(coefficient_override)
        self.tasks.append(OperatorTask(
            kind=kind,
            target=target,
            equation_name=self.equation_name,
            unknown_name=unknown_name,
            sign=self.sign,
            coefficient=coefficient,
            source_field=source_field,
            metadata=metadata,
        ))


def _materialize_coefficient(
    coef: BackgroundComponent,
) -> Optional[BackgroundComponent]:
    """The task-time coefficient: None when the accumulated coefficient
    is the SymPy constant 1, else the component itself."""
    if isinstance(coef, SymPyComponent) and coef.expr == sp.S.One:
        return None
    return coef


# ===========================================================================
# Walking individual expression nodes
# ===========================================================================

def _walk_expr(expr: Expression, ctx: _WalkContext) -> None:
    """Walk a scalar- or vector-kind node, emitting tasks into ctx.

    Sum descends into both children; Negation flips ctx.sign;
    ScalarMultiple by a background scalar folds the factor into
    ctx.coefficient_component; Dt sets ctx.in_dt. Each is restored
    after the subtree. Leaves and operators dispatch to the _emit_*
    functions.
    """
    if isinstance(expr, Sum):
        _walk_expr(expr.a, ctx)
        _walk_expr(expr.b, ctx)
        return
    if isinstance(expr, Negation):
        saved_sign = ctx.sign
        ctx.sign = -ctx.sign
        try:
            _walk_expr(expr.operand, ctx)
        finally:
            ctx.sign = saved_sign
        return
    if isinstance(expr, ScalarMultiple):
        _walk_scalar_multiple(expr, ctx)
        return
    if isinstance(expr, Dt):
        if expr.operand.is_background:
            raise NotImplementedError(
                "Dt(...) of a pure-background expression has no meaning in a "
                "linear eigenproblem; rejected."
            )
        saved_in_dt = ctx.in_dt
        ctx.in_dt = True
        try:
            _walk_expr(expr.operand, ctx)
        finally:
            ctx.in_dt = saved_in_dt
        return
    if isinstance(expr, _BackgroundScalarValue):
        # A bare scalar value as a term: pure-background source.
        if ctx.in_dt:
            _warn_dt_of_background(repr(expr))
            return
        total_coef = multiply_components(
            ctx.coefficient_component, SymPyComponent(expr.value),
        )
        ctx.emit_task(
            TASK_SOURCE_SCALAR,
            unknown_name=None,
            target=TARGET_RHS,
            coefficient_override=total_coef,
        )
        return
    if isinstance(expr, FieldRef):
        _emit_field_ref(expr, ctx)
        return
    if isinstance(expr, Lap):
        _emit_lap(expr, ctx)
        return
    if isinstance(expr, LapH):
        _emit_laph(expr, ctx)
        return
    if isinstance(expr, Div):
        _emit_div(expr, ctx)
        return
    if isinstance(expr, Curl):
        _emit_curl(expr, ctx)
        return
    if isinstance(expr, Cross):
        _emit_cross(expr, ctx)
        return
    if isinstance(expr, Grad):
        _emit_grad(expr, ctx)
        return
    if isinstance(expr, Dot):
        _emit_dot(expr, ctx)
        return
    raise NotImplementedError(
        f"Walker does not handle expression node {type(expr).__name__}: "
        f"{expr!r}"
    )


def _walk_scalar_multiple(expr: ScalarMultiple, ctx: _WalkContext) -> None:
    """ScalarMultiple(scalar, operand).

    Background scalar: fold it into the coefficient and walk the operand.

    Unknown scalar times background vector (s * B0, with background
    scalars around either factor folded into the coefficient, as in
    (g * s) * r_hat): emit TASK_BG_VEC_TIMES_UNK_SCALAR.

    Div(unknown vector) times background vector (Div(v) * B0), with
    background scalars around either factor folded into the coefficient:
    emit TASK_BG_VEC_TIMES_DIV_UNK_VEC.

    Any other non-background scalar raises NotImplementedError.
    """
    scalar_expr = expr.scalar
    operand = expr.operand
    if scalar_expr.is_background:
        scalar_component = _extract_background_scalar_component(scalar_expr)
        saved_coef = ctx.coefficient_component
        ctx.coefficient_component = multiply_components(
            ctx.coefficient_component, scalar_component,
        )
        try:
            _walk_expr(operand, ctx)
        finally:
            ctx.coefficient_component = saved_coef
        return
    unk, vec, coef = scalar_expr, operand, ctx.coefficient_component
    while isinstance(unk, ScalarMultiple) and unk.scalar.is_background:
        coef = multiply_components(
            coef, _extract_background_scalar_component(unk.scalar))
        unk = unk.operand
    while isinstance(vec, ScalarMultiple) and vec.scalar.is_background:
        coef = multiply_components(
            coef, _extract_background_scalar_component(vec.scalar))
        vec = vec.operand
    if _is_unknown_scalar(unk) and _is_bg_vector(vec):
        if ctx.in_dt:
            raise NotImplementedError(
                "Dt(unknown_scalar * background_vector) is not "
                "supported."
            )
        ctx.emit_task(
            TASK_BG_VEC_TIMES_UNK_SCALAR,
            unknown_name=unk.field.name,
            source_field=vec.field,
            coefficient_override=coef,
        )
        return
    if _is_div_of_unknown_vector(unk) and _is_bg_vector(vec):
        if ctx.in_dt:
            raise NotImplementedError(
                "Dt(Div(unknown_vec) * background_vector) is not "
                "supported."
            )
        unknown_vec_field = unk.operand.field
        ctx.emit_task(
            TASK_BG_VEC_TIMES_DIV_UNK_VEC,
            unknown_name=unknown_vec_field.name,
            source_field=vec.field,
            coefficient_override=coef,
            extra_metadata={
                "v_placeholders": _placeholders_for(unknown_vec_field),
            },
        )
        return
    raise NotImplementedError(
        f"Scalar coefficient is non-background: {scalar_expr!r}. "
        f"The walker only handles background coefficients."
    )


def _emit_field_ref(expr: FieldRef, ctx: _WalkContext) -> None:
    """A bare field reference as an equation term.

    Background scalar:  TASK_SOURCE_SCALAR (coefficient times component).
    Background vector:  TASK_SOURCE_VECTOR (source_field carries the
                        vector; coefficient the accumulated prefactor).
    Unknown scalar:     TASK_IDENTITY_SCALAR.
    Unknown vector:     TASK_IDENTITY_VECTOR.

    A background inside Dt(...) is dropped with a warning: d/dt of a
    background is zero.
    """
    field = expr.field
    if field.is_background:
        if ctx.in_dt:
            _warn_dt_of_background(field.name)
            return
        if field.kind == "scalar":
            assert isinstance(field, ScalarField)
            final_coef = multiply_components(
                ctx.coefficient_component, field.component,
            )
            ctx.emit_task(
                TASK_SOURCE_SCALAR,
                unknown_name=None,
                target=TARGET_RHS,
                coefficient_override=final_coef,
            )
        else:
            ctx.emit_task(
                TASK_SOURCE_VECTOR,
                unknown_name=None,
                target=TARGET_RHS,
                source_field=field,
            )
        return
    target = TARGET_B if ctx.in_dt else TARGET_A
    kind = TASK_IDENTITY_SCALAR if field.kind == "scalar" else TASK_IDENTITY_VECTOR
    ctx.emit_task(kind, unknown_name=field.name, target=target)


def _warn_dt_of_background(label: str) -> None:
    warnings.warn(
        f"Dt({label}): the time derivative of a background is zero; the "
        f"term is dropped",
        UserWarning, stacklevel=2,
    )


def _emit_lap(expr: Lap, ctx: _WalkContext) -> None:
    """Lap(scalar_unknown) -> TASK_LAP_SCALAR;
    Lap(vector_unknown) -> TASK_LAP_VECTOR.

    The operand must be a bare unknown FieldRef and the walk must not be
    inside Dt.
    """
    operand = expr.operand
    if not isinstance(operand, FieldRef):
        raise NotImplementedError(
            f"Lap requires a direct field reference; got Lap({operand!r}). "
            f"Composed operators (e.g. Lap of Lap, Lap of background-scaled "
            f"unknown) require state augmentation and aren't supported yet."
        )
    field = operand.field
    if field.is_background:
        raise NotImplementedError(
            "Lap of a background field at the top level would be a known "
            "quantity and should be evaluated outside the IR. Rejecting."
        )
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Lap(...)) is not supported; the eigenvalue side typically "
            "holds just unknowns, not differential operators."
        )
    if field.kind == "scalar":
        ctx.emit_task(TASK_LAP_SCALAR, unknown_name=field.name)
    else:
        ctx.emit_task(
            TASK_LAP_VECTOR,
            unknown_name=field.name,
            extra_metadata={"v_placeholders": _placeholders_for(field)},
        )


def _emit_laph(expr: LapH, ctx: _WalkContext) -> None:
    """LapH(scalar_unknown) -> TASK_LAPH_SCALAR, with the same operand
    and Dt restrictions as Lap."""
    operand = expr.operand
    if not isinstance(operand, FieldRef):
        raise NotImplementedError(
            f"LapH requires a direct field reference; got LapH({operand!r}). "
            f"Composed operators aren't supported yet."
        )
    field = operand.field
    if field.is_background:
        raise NotImplementedError(
            "LapH of a background field at the top level would be a known "
            "quantity and should be evaluated outside the IR. Rejecting."
        )
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(LapH(...)) is not supported."
        )
    ctx.emit_task(TASK_LAPH_SCALAR, unknown_name=field.name)


def _emit_div(expr: Div, ctx: _WalkContext) -> None:
    """Div(vector_unknown) -> TASK_DIV_VECTOR_SCALAR;
    Div(tensor_expr) -> TASK_DIV_TENSOR_VECTOR.

    For the tensor case the expression must reference exactly one
    unknown vector (background tensor leaves are allowed). It is reduced
    to covariant components T_{sigma_a, sigma_b} in that unknown's
    placeholders, and the task carries the r^2-scaled divergence in
    U = r T variables:

        'per_sigma_expr_in_U' : {sigma_1: r^2 (Div T)^{sigma_1}} in U placeholders
        'U_placeholders'      : {(sigma_a, sigma_b): U SymPy Function}
        'U_in_v'              : {(sigma_a, sigma_b): r T_{sigma_a, sigma_b}} in v placeholders
        'v_placeholders'      : {unknown_name: {sigma: v SymPy Function}}
        'origin_factor'       : see _compute_origin_factor_for_div (may be None)

    The backend lifts each U_in_v entry to a sparse (N, 3N) block on v,
    classifies each per-sigma formula on the U placeholders, and
    composes the two.
    """
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Div(tensor)) is not supported; the eigenvalue side typically "
            "holds just unknowns, not differential operators."
        )
    operand = expr.operand
    if operand.is_vector:
        weighted = _peel_background_scalar_weight(operand)
        if weighted is not None:
            _emit_weighted_div(weighted, ctx)
            return
        if not isinstance(operand, FieldRef):
            raise NotImplementedError(
                f"Div(vector) requires a direct vector unknown field reference; "
                f"got Div({operand!r}). Composed operators (e.g. Div of Curl) "
                f"require state augmentation and aren't supported yet."
            )
        field = operand.field
        if field.is_background:
            raise NotImplementedError(
                "Div of a background vector field at the top level would be a "
                "known quantity and should be evaluated outside the IR. Rejecting."
            )
        ctx.emit_task(
            TASK_DIV_VECTOR_SCALAR,
            unknown_name=field.name,
            extra_metadata={"v_placeholders": _placeholders_for(field)},
        )
        return
    if _is_unknown_tensor(operand):
        ctx.emit_task(TASK_DIV_TENSOR_UNK, unknown_name=operand.field.name)
        return
    unknown_vec_fields = _collect_unknown_vector_fields(operand)
    if not unknown_vec_fields:
        raise NotImplementedError(
            f"Div({operand!r}): the tensor expression contains no unknown "
            f"vector fields. Pure-background divergence should be computed "
            f"outside the IR."
        )
    if len(unknown_vec_fields) > 1:
        raise NotImplementedError(
            f"Div({operand!r}): the tensor expression references multiple "
            f"vector unknowns ({[f.name for f in unknown_vec_fields]}); "
            f"only single-unknown Div is supported."
        )
    host_unknown = unknown_vec_fields[0]
    v_placeholders = {
        f.name: vector_unknown_spin_placeholders(f) for f in unknown_vec_fields
    }
    reduced = reduce_tensor_expression(
        operand, m=ctx.m, unknown_spin_placeholders=v_placeholders,
    )
    bg_components, components = _split_background_part(
        reduced.components, v_placeholders)
    if bg_components is not None:
        U_bg = {key: _r_sym * val for key, val in bg_components.items()}
        r2_div_bg = r2_div_tensor_spin_components(U_bg, m=0)
        _emit_background_vector_source(
            {s: sp.cancel(r2_div_bg[s] / _r_sym ** 2) for s in (-1, 0, +1)},
            f"div_bg_{ctx.equation_name}", ctx,
        )
    U_placeholders = {}
    for s_a in (-1, 0, +1):
        for s_b in (-1, 0, +1):
            name = f"U_div_{host_unknown.name}_{_sigma_label(s_a)}_{_sigma_label(s_b)}"
            U_placeholders[(s_a, s_b)] = sp.Function(name)(_r_sym, _t_sym)
    r2_div_components = r2_div_tensor_spin_components(U_placeholders, m=ctx.m)
    per_sigma_expr = {s: sp.expand(r2_div_components[s]) for s in (-1, 0, +1)}
    U_in_v = {}
    for s_a in (-1, 0, +1):
        for s_b in (-1, 0, +1):
            U_in_v[(s_a, s_b)] = sp.expand(_r_sym * components[(s_a, s_b)])
    origin_factor = _compute_origin_factor_for_div(
        per_sigma_expr, U_placeholders, U_in_v, v_placeholders, m=ctx.m,
    )
    ctx.emit_task(
        TASK_DIV_TENSOR_VECTOR,
        unknown_name=host_unknown.name,
        extra_metadata={
            "per_sigma_expr_in_U": per_sigma_expr,
            "U_placeholders": U_placeholders,
            "U_in_v": U_in_v,
            "v_placeholders": v_placeholders,
            "grad_transpose_weights": _grad_transpose_weights(operand, host_unknown.name),
            "origin_factor": origin_factor,
        },
    )


def _peel_background_scalar_weight(operand):
    """(M, u) for an operand M u that is a chain of background-scalar
    ScalarMultiples around an unknown vector FieldRef, or None when the
    operand has another shape. M is the product of the peeled scalars."""
    weight = SymPyComponent(sp.S.One)
    e = operand
    peeled = False
    while isinstance(e, ScalarMultiple) and e.scalar.is_background:
        weight = multiply_components(
            weight, _extract_background_scalar_component(e.scalar))
        e = e.operand
        peeled = True
    if peeled and _is_unknown_vector(e):
        return weight, e.field
    return None


def _split_background_part(components, placeholders):
    """(background, unknown) parts of reduced tensor components: the
    additive terms free of every unknown placeholder form the background
    part. The background part is None when there is none, and the
    unknown part is then the input itself."""
    atoms = [fn for ph in placeholders.values() for fn in ph.values()]
    bg, unk = {}, {}
    for key, expr in components.items():
        terms = sp.Add.make_args(sp.expand(expr))
        bg_terms = [t for t in terms if not t.has(*atoms)]
        bg[key] = sp.Add(*bg_terms)
        unk[key] = sp.Add(*[t for t in terms if t.has(*atoms)]) if bg_terms else expr
    if all(val == 0 for val in bg.values()):
        return None, components
    return bg, unk


def _emit_background_vector_source(spin, label: str, ctx: _WalkContext) -> None:
    """TASK_SOURCE_VECTOR for a background vector given by its
    contravariant spin components {sigma: sp.Expr}."""
    sqrt2 = sp.sqrt(2)
    field = VectorField(
        name=label, role=FieldRole.BACKGROUND,
        v_r=spin[0],
        v_theta=(spin[+1] + spin[-1]) / sqrt2,
        v_phi=-sp.I * (spin[+1] - spin[-1]) / sqrt2,
    )
    ctx.emit_task(
        TASK_SOURCE_VECTOR, unknown_name=None, target=TARGET_RHS,
        source_field=field,
    )


_WEIGHT_GRADIENT_STEP = 1e-4


def _background_gradient_field(weight, label: str) -> VectorField:
    """(d_r M, (1/r) d_theta M, 0) of a background scalar M as a background
    vector: exact differentiation of a SymPy weight, central differences of
    step ``_WEIGHT_GRADIENT_STEP`` of a numeric one (the rule of
    ``spdee.ir.backgrounds`` for derivatives of numeric components)."""
    if isinstance(weight, SymPyComponent):
        m_expr = weight.expr
        return VectorField(
            name=f"grad({m_expr})", role=FieldRole.BACKGROUND,
            v_r=sp.diff(m_expr, _r_sym),
            v_theta=sp.diff(m_expr, _t_sym) / _r_sym,
            v_phi=sp.S.Zero,
        )
    h = _WEIGHT_GRADIENT_STEP

    def ev(R, T):
        return weight.evaluate_on_grid(R, T)

    def d_r(R, T):
        """Central difference; one-sided second order where r - h < 0."""
        R = np.asarray(R, dtype=float)
        T = np.asarray(T, dtype=float)
        inner = R < h
        Rs = np.where(inner, h, R)
        central = (ev(Rs + h, T) - ev(Rs - h, T)) / (2 * h)
        if not np.any(inner):
            return central
        Ri = np.where(inner, R, 0.0)
        onesided = (-3 * ev(Ri, T) + 4 * ev(Ri + h, T) - ev(Ri + 2 * h, T)) / (2 * h)
        return np.where(inner, onesided, central)

    def d_theta(R, T):
        """(1/r) d_theta M; at r < h the finite origin limit is extrapolated
        linearly from the values at r = h and r = 2h."""
        R = np.asarray(R, dtype=float)
        T = np.asarray(T, dtype=float)
        inner = R < h

        def at(Rv):
            return (ev(Rv, T + h) - ev(Rv, T - h)) / (2 * h * Rv)

        Rs = np.where(inner, h, R)
        value = at(Rs)
        if not np.any(inner):
            return value
        h_arr = np.full_like(R, h)
        limit = 2 * at(h_arr) - at(2 * h_arr)
        return np.where(inner, limit, value)

    return VectorField(
        name=f"grad({label})", role=FieldRole.BACKGROUND,
        v_r=NumericComponent(f=d_r, description=f"d_r {label}"),
        v_theta=NumericComponent(f=d_theta, description=f"(1/r) d_theta {label}"),
        v_phi=sp.S.Zero,
    )


def _emit_weighted_div(weighted, ctx: _WalkContext) -> None:
    """Div(M u) for a background scalar M and an unknown vector u, by the
    product rule: M Div(u) as TASK_DIV_VECTOR_SCALAR with M folded into the
    coefficient, plus Grad(M) . u as TASK_DOT_BG_VEC_UNK_VEC_SCALAR with the
    background vector of ``_background_gradient_field``. Both tasks carry
    metadata 'div_weight' (M), 'div_outer' (the accumulated coefficient
    outside the divergence) and 'div_weight_role' ('div' / 'grad'), so a
    backend with a weighted mimetic divergence can build Div(M u) as one
    block from the pair."""
    weight, field = weighted
    outer = _materialize_coefficient(ctx.coefficient_component)
    grad_field = _background_gradient_field(weight, field.name)
    tag = {"div_weight": weight, "div_outer": outer}
    ctx.emit_task(
        TASK_DIV_VECTOR_SCALAR,
        unknown_name=field.name,
        coefficient_override=multiply_components(
            ctx.coefficient_component, weight),
        extra_metadata={"v_placeholders": _placeholders_for(field),
                        "div_weight_role": "div", **tag},
    )
    ctx.emit_task(
        TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
        unknown_name=field.name,
        source_field=grad_field,
        extra_metadata={"div_weight_role": "grad", **tag},
    )


def _emit_curl(expr: Curl, ctx: _WalkContext) -> None:
    """Curl(v_unknown) -> TASK_CURL_VECTOR.

    Curl(Cross(v_unknown, B0)) and Curl(Cross(B0, v_unknown)) ->
    TASK_CURL_CROSS_UNK_BG with metadata 'inner_side' = 'right'
    (unknown x bg) or 'left' (bg x unknown).

    Other composed operands raise NotImplementedError.
    """
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Curl(...)) is not supported; the eigenvalue side typically "
            "holds just unknowns, not differential operators."
        )
    operand = expr.operand
    if isinstance(operand, Cross):
        a, b = operand.a, operand.b
        if _is_unknown_vector(a) and _is_bg_vector(b):
            unknown_field, bg_field, inner_side = a.field, b.field, "right"
        elif _is_bg_vector(a) and _is_unknown_vector(b):
            unknown_field, bg_field, inner_side = b.field, a.field, "left"
        else:
            unknown_field = None
        if unknown_field is not None:
            ctx.emit_task(
                TASK_CURL_CROSS_UNK_BG,
                unknown_name=unknown_field.name,
                source_field=bg_field,
                extra_metadata={
                    "v_placeholders": _placeholders_for(unknown_field),
                    "inner_side": inner_side,
                },
            )
            return
    if _is_curl_of_unknown_vector(operand):
        field = operand.operand.field
        ctx.emit_task(
            TASK_CURL_CURL_VECTOR,
            unknown_name=field.name,
            extra_metadata={"v_placeholders": _placeholders_for(field)},
        )
        return
    if not isinstance(operand, FieldRef):
        raise NotImplementedError(
            f"Curl requires a direct vector unknown field reference, "
            f"Curl(Cross(unknown_vec, bg_vec)) or Curl(Curl(unknown_vec)); got "
            f"Curl({operand!r}). Other composed operators require state "
            f"augmentation."
        )
    field = operand.field
    if field.is_background:
        raise NotImplementedError(
            "Curl of a background vector field at the top level would be a "
            "known quantity and should be evaluated outside the IR. Rejecting."
        )
    ctx.emit_task(
        TASK_CURL_VECTOR,
        unknown_name=field.name,
        extra_metadata={"v_placeholders": _placeholders_for(field)},
    )


def _emit_cross(expr: Cross, ctx: _WalkContext) -> None:
    """Cross(B0, v_unknown) -> TASK_CROSS_VECTOR, side='left';
    Cross(v_unknown, B0) -> TASK_CROSS_VECTOR, side='right'.

    Cross(Curl(v_unknown), B0) -> TASK_CROSS_CURL_UNK_BG, side='right';
    Cross(B0, Curl(v_unknown)) -> TASK_CROSS_CURL_UNK_BG, side='left';
    Cross(Curl(Curl(v_unknown)), B0) and the bg-first variant ->
    TASK_CROSS_CURL_CURL_UNK_BG with the same 'side'.

    Cross(unknown, unknown) is rejected at IR construction;
    Cross(background, background) is rejected here as a pure source.
    """
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Cross(...)) is not supported."
        )
    a, b = expr.a, expr.b
    if _is_curl_curl_of_unknown_vector(a) and _is_bg_vector(b):
        unknown_field, bg_field, side = a.operand.operand.field, b.field, "right"
    elif _is_bg_vector(a) and _is_curl_curl_of_unknown_vector(b):
        unknown_field, bg_field, side = b.operand.operand.field, a.field, "left"
    else:
        unknown_field = None
    if unknown_field is not None:
        ctx.emit_task(
            TASK_CROSS_CURL_CURL_UNK_BG,
            unknown_name=unknown_field.name,
            source_field=bg_field,
            extra_metadata={
                "v_placeholders": _placeholders_for(unknown_field),
                "side": side,
            },
        )
        return
    if _is_curl_of_unknown_vector(a) and _is_bg_vector(b):
        unknown_field, bg_field, side = a.operand.field, b.field, "right"
    elif _is_bg_vector(a) and _is_curl_of_unknown_vector(b):
        unknown_field, bg_field, side = b.operand.field, a.field, "left"
    else:
        unknown_field = None
    if unknown_field is not None:
        ctx.emit_task(
            TASK_CROSS_CURL_UNK_BG,
            unknown_name=unknown_field.name,
            source_field=bg_field,
            extra_metadata={
                "v_placeholders": _placeholders_for(unknown_field),
                "side": side,
            },
        )
        return
    if not (isinstance(a, FieldRef) and isinstance(b, FieldRef)):
        raise NotImplementedError(
            f"Cross requires either two bare vector FieldRefs or one "
            f"Curl-of-unknown plus one background vector; got "
            f"Cross({a!r}, {b!r}). Other composed operators require "
            f"state augmentation."
        )
    a_field, b_field = a.field, b.field
    if a_field.is_background and b_field.is_background:
        raise NotImplementedError(
            "Cross of two background vectors should be evaluated outside "
            "the IR; rejecting at the walker."
        )
    if a_field.is_background:
        bg_field, unknown_field, side = a_field, b_field, "left"
    else:
        bg_field, unknown_field, side = b_field, a_field, "right"
    ctx.emit_task(
        TASK_CROSS_VECTOR,
        unknown_name=unknown_field.name,
        source_field=bg_field,
        extra_metadata={
            "v_placeholders": _placeholders_for(unknown_field),
            "side": side,
        },
    )


def _emit_grad(expr: Grad, ctx: _WalkContext) -> None:
    """Grad(scalar_unknown) -> TASK_GRAD_SCALAR;
    Grad(Div(vector_unknown)) -> TASK_GRAD_DIV_VECTOR;
    Grad of a sum of background-weighted unknown scalars distributes,
    each weighted leaf Grad(M s) becoming the tagged product-rule pair of
    ``_emit_weighted_grad``.

    Grad(vector) is a tensor and only reaches the walker inside
    Div(...), Dot(...) or a tensor equation.
    """
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Grad(...)) is not supported; the eigenvalue side typically "
            "holds just unknowns, not differential operators."
        )
    operand = expr.operand
    if _is_div_of_unknown_vector(operand):
        inner_field = operand.operand.field
        ctx.emit_task(
            TASK_GRAD_DIV_VECTOR,
            unknown_name=inner_field.name,
            extra_metadata={"v_placeholders": _placeholders_for(inner_field)},
        )
        return
    leaves = _weighted_scalar_leaves(operand)
    if leaves is None:
        if isinstance(operand, FieldRef) and operand.field.is_background:
            raise NotImplementedError(
                "Grad of a background field at the top level would be a known "
                "quantity and should be evaluated outside the IR. Rejecting."
            )
        raise NotImplementedError(
            f"Grad requires a direct field reference or a sum of "
            f"background-weighted unknown scalars; got Grad({operand!r}). "
            f"Composed operators require state augmentation."
        )
    saved_sign = ctx.sign
    try:
        for sign, weight, field in leaves:
            ctx.sign = saved_sign * sign
            if weight is None:
                ctx.emit_task(TASK_GRAD_SCALAR, unknown_name=field.name)
            else:
                _emit_weighted_grad(weight, field, ctx)
    finally:
        ctx.sign = saved_sign


def _weighted_scalar_leaves(operand):
    """[(sign, M or None, field)] for an operand that is a Sum / Negation
    tree of background-weighted unknown scalars M s (M the product of the
    peeled background scalars, None when there is none); None for any
    other shape."""
    out = []

    def rec(e, sign, weight):
        while isinstance(e, ScalarMultiple) and e.scalar.is_background:
            w = _extract_background_scalar_component(e.scalar)
            weight = w if weight is None else multiply_components(weight, w)
            e = e.operand
        if isinstance(e, Negation):
            return rec(e.operand, -sign, weight)
        if isinstance(e, Sum):
            return rec(e.a, sign, weight) and rec(e.b, sign, weight)
        if _is_unknown_scalar(e):
            out.append((sign, weight, e.field))
            return True
        return False

    return out if rec(operand, +1, None) else None


def _emit_weighted_grad(weight, field, ctx: _WalkContext) -> None:
    """Grad(M s) for a background scalar M and an unknown scalar s, by the
    product rule: M Grad(s) as TASK_GRAD_SCALAR with M folded into the
    coefficient, plus s Grad(M) as TASK_BG_VEC_TIMES_UNK_SCALAR with the
    background vector of ``_background_gradient_field``. Both tasks carry
    metadata 'grad_weight' (M), 'grad_outer' (the accumulated coefficient
    outside the gradient) and 'grad_weight_role' ('grad' / 'weight'), so a
    backend with a mimetic gradient can build Grad(M s) as one block from
    the pair."""
    outer = _materialize_coefficient(ctx.coefficient_component)
    grad_field = _background_gradient_field(weight, field.name)
    tag = {"grad_weight": weight, "grad_outer": outer}
    ctx.emit_task(
        TASK_GRAD_SCALAR,
        unknown_name=field.name,
        coefficient_override=multiply_components(
            ctx.coefficient_component, weight),
        extra_metadata={"grad_weight_role": "grad", **tag},
    )
    ctx.emit_task(
        TASK_BG_VEC_TIMES_UNK_SCALAR,
        unknown_name=field.name,
        source_field=grad_field,
        extra_metadata={"grad_weight_role": "weight", **tag},
    )


def _transposed_background_tensor(field: TensorField) -> TensorField:
    """The background tensor with its physical components swapped,
    (T^T)_{ij} = T_{ji}; the field itself when it is symmetric."""
    if field.symmetric:
        return field
    return TensorField(
        name=f"__transpose_{field.name}", role=field.role,
        components={(j, i): c for (i, j), c in field.components.items()},
    )


def _emit_dot(expr: Dot, ctx: _WalkContext) -> None:
    """Dot at the equation level (BC Dots are parsed separately).

    Dot(B0, Grad(v))          -> TASK_DOT_BG_GRAD_VEC
    Dot(B0, T), Dot(T, B0)    -> TASK_DOT_BG_VEC_UNK_TENSOR (T an unknown
                                 tensor), metadata 'side'
    Dot(B0, tensor_expr(v))   -> TASK_DOT_BG_VEC_TENSOR_UNK, with
                                 metadata 'rT_components' =
                                 {(sigma_a, sigma_b): r T_{sigma_a, sigma_b}}
                                 in v placeholders and 'tensor_symmetric'
    Dot(T0, v)                -> TASK_DOT_BG_TENSOR_VEC
    Dot(B0, v), Dot(v, B0)    -> TASK_DOT_BG_VEC_UNK_VEC_SCALAR
    Dot(B0, Grad(s)), Dot(Grad(s), B0) -> TASK_DOT_BG_VEC_GRAD_SCALAR

    A background on the right of a rank-2 contraction is canonicalised
    to the left form, which is exact: Dot(X, B0)_i = X_ij B0_j =
    Dot(B0, Transpose(X))_i, and Dot(v, T0) = Dot(Transpose(T0), v)
    with the background transposed component-wise. Every core that
    builds the left form therefore builds the right one.

    Anything else raises NotImplementedError.
    """
    if ctx.in_dt:
        raise NotImplementedError(
            "Dt(Dot(...)) is not supported."
        )
    a, b = expr.a, expr.b
    if a.is_tensor and _is_bg_vector(b) and not isinstance(a, FieldRef):
        a, b = b, Transpose(a)
    elif _is_unknown_vector(a) and _is_bg_tensor(b):
        a, b = FieldRef(_transposed_background_tensor(b.field)), a
    if _is_bg_vector(a) and _is_grad_of_unknown_vector(b):
        unknown_field = b.operand.field
        ctx.emit_task(
            TASK_DOT_BG_GRAD_VEC,
            unknown_name=unknown_field.name,
            source_field=a.field,
            extra_metadata={"v_placeholders": _placeholders_for(unknown_field)},
        )
        return
    if _is_bg_vector(a) and b.is_tensor and not isinstance(b, FieldRef):
        unknown_vec_fields = _collect_unknown_vector_fields(b)
        if not unknown_vec_fields:
            raise NotImplementedError(
                f"Dot(bg_vec, tensor): tensor expression {b!r} contains "
                f"no unknown vector fields. A pure-background contraction "
                f"should be computed outside the IR."
            )
        if len(unknown_vec_fields) > 1:
            raise NotImplementedError(
                f"Dot(bg_vec, tensor): tensor expression references "
                f"multiple vector unknowns "
                f"({[f.name for f in unknown_vec_fields]}); only "
                f"single-unknown contractions are supported."
            )
        host_unknown = unknown_vec_fields[0]
        v_placeholders = {
            f.name: vector_unknown_spin_placeholders(f)
            for f in unknown_vec_fields
        }
        reduced = reduce_tensor_expression(
            b, m=ctx.m, unknown_spin_placeholders=v_placeholders,
        )
        bg_components, components = _split_background_part(
            reduced.components, v_placeholders)
        if bg_components is not None:
            a_spin = a.field.spin_components()
            _emit_background_vector_source(
                {s_out: sum(a_spin[s_in] * bg_components[(s_in, -s_out)]
                            for s_in in (-1, 0, +1))
                 for s_out in (-1, 0, +1)},
                f"dot_bg_{ctx.equation_name}", ctx,
            )
        rT_components = {
            (s_a, s_b): sp.expand(_r_sym * components[(s_a, s_b)])
            for s_a in (-1, 0, +1) for s_b in (-1, 0, +1)
        }
        ctx.emit_task(
            TASK_DOT_BG_VEC_TENSOR_UNK,
            unknown_name=host_unknown.name,
            source_field=a.field,
            extra_metadata={
                "rT_components": rT_components,
                "v_placeholders": v_placeholders,
                "tensor_symmetric": reduced.symmetric,
                "grad_transpose_weights": _grad_transpose_weights(b, host_unknown.name),
            },
        )
        return
    if _is_bg_tensor(a) and _is_unknown_vector(b):
        ctx.emit_task(
            TASK_DOT_BG_TENSOR_VEC,
            unknown_name=b.field.name,
            source_field=a.field,
        )
        return
    if _is_bg_vector(a) and _is_unknown_tensor(b):
        bg_field, tensor_field, side = a.field, b.field, "left"
    elif _is_unknown_tensor(a) and _is_bg_vector(b):
        bg_field, tensor_field, side = b.field, a.field, "right"
    else:
        bg_field = None
    if bg_field is not None:
        ctx.emit_task(
            TASK_DOT_BG_VEC_UNK_TENSOR,
            unknown_name=tensor_field.name,
            source_field=bg_field,
            extra_metadata={"side": side},
        )
        return
    if _is_bg_vector(a) and _is_unknown_vector(b):
        bg_field, unknown_field = a.field, b.field
    elif _is_unknown_vector(a) and _is_bg_vector(b):
        bg_field, unknown_field = b.field, a.field
    else:
        bg_field = None
    if bg_field is not None:
        ctx.emit_task(
            TASK_DOT_BG_VEC_UNK_VEC_SCALAR,
            unknown_name=unknown_field.name,
            source_field=bg_field,
        )
        return
    if _is_bg_vector(a) and _is_grad_of_unknown_scalar(b):
        bg_field, unknown_field = a.field, b.operand.field
    elif _is_grad_of_unknown_scalar(a) and _is_bg_vector(b):
        bg_field, unknown_field = b.field, a.operand.field
    else:
        bg_field = None
    if bg_field is not None:
        ctx.emit_task(
            TASK_DOT_BG_VEC_GRAD_SCALAR,
            unknown_name=unknown_field.name,
            source_field=bg_field,
        )
        return
    raise NotImplementedError(
        f"Dot expression at equation level: only "
        f"Dot(background_vector, Grad(unknown_vector)), "
        f"Dot(background_tensor, unknown_vector), "
        f"Dot(background_vector, unknown_vector) [→ scalar], and "
        f"Dot(background_vector, Grad(unknown_scalar)) [→ scalar] "
        f"are supported. Got Dot({a!r}, {b!r})."
    )


def _compute_origin_factor_for_div(
    per_sigma_expr_in_U,
    U_placeholders,
    U_in_v,
    v_placeholders,
    m: int,
):
    """Origin-regularity factor of a Div(tensor) operator.

    For the z-aligned axisymmetric l=0 mode v = f(r) z_hat with f even,
    the operator satisfies near r=0
        r^2 Div(T(v))^0 ~ factor * f''(0) * r^2 * cos(theta).
    The factor is found by substituting v = (r^2/2) z_hat (so f''(0)=1)
    and reading off the coefficient of r^2 cos(theta) in the sigma=0
    row. It is 3 for Div(Grad v) = Lap v, 4 for the divergence of the
    strain tensor, and the backend's ball-mode origin treatment uses it
    in place of the Laplacian default.

    Returns a float, or None when m != 0, when the placeholders do not
    describe a single unknown, or when the sigma=0 row is not a clean
    constant times r^2 cos(theta).
    """
    if m != 0:
        return None
    if len(v_placeholders) != 1:
        return None
    (v_ph,) = v_placeholders.values()
    r, th = _r_sym, _t_sym
    sub_v = {
        v_ph[0]: (r ** 2 / 2) * sp.cos(th),
        v_ph[+1]: -(r ** 2 / 2) * sp.sin(th) / sp.sqrt(2),
        v_ph[-1]: -(r ** 2 / 2) * sp.sin(th) / sp.sqrt(2),
    }
    U_eval = {}
    for k, expr in U_in_v.items():
        e = expr
        for sigma in (-1, 0, +1):
            fn = v_ph[sigma]
            e = e.replace(fn.func, sp.Lambda((r, th), sub_v[fn]))
        U_eval[k] = sp.simplify(e.doit())
    expr_0 = per_sigma_expr_in_U[0]
    for k, U_expr in U_eval.items():
        U_fn = U_placeholders[k]
        expr_0 = expr_0.replace(U_fn.func, sp.Lambda((r, th), U_expr))
    expr_0 = sp.simplify(expr_0.doit())
    factor_expr = sp.simplify(expr_0 / (r ** 2 * sp.cos(th)))
    if factor_expr.free_symbols & {r, th}:
        return None
    try:
        return float(factor_expr)
    except (TypeError, ValueError):
        return None


def _sigma_label(sigma: int) -> str:
    """Spin-index label used in placeholder names."""
    return {+1: "p", 0: "0", -1: "m"}[sigma]


# ===========================================================================
# Coefficient extraction
# ===========================================================================

def _extract_background_scalar_component(
    expr: Expression,
) -> BackgroundComponent:
    """BackgroundComponent of a background scalar Expression
    (`spdee.ir.expressions.background_scalar_component`)."""
    return background_scalar_component(expr)


# ===========================================================================
# Top-level entry points
# ===========================================================================

def walk_equation(eq: Equation, m: int) -> List[OperatorTask]:
    """Tasks of a single named scalar, vector or tensor Equation."""
    if not eq.name:
        raise ValueError(
            f"Equation must have a name for walker dispatch: {eq!r}"
        )
    ctx = _WalkContext(equation_name=eq.name, m=m)
    if eq.kind == "tensor":
        terms = []
        _walk_tensor_expr(eq.lhs, ctx, terms)
        _emit_tensor_of_vector(terms, ctx)
    else:
        _walk_expr(eq.lhs, ctx)
    return ctx.tasks


def _walk_tensor_expr(expr: Expression, ctx: _WalkContext, terms: list) -> None:
    """A term of a tensor equation: Sum, Negation, background
    ScalarMultiple and Dt as in ``_walk_expr``; an unknown tensor leaf ->
    TASK_IDENTITY_TENSOR; a background tensor leaf -> TASK_SOURCE_TENSOR;
    any other tensor expression must be linear in exactly one vector
    unknown (Grad, Transpose, sums, background multiples) and is collected
    into ``terms`` as (sign, coefficient, expression) for
    ``_emit_tensor_of_vector``."""
    if isinstance(expr, Sum):
        _walk_tensor_expr(expr.a, ctx, terms)
        _walk_tensor_expr(expr.b, ctx, terms)
        return
    if isinstance(expr, Negation):
        saved_sign = ctx.sign
        ctx.sign = -ctx.sign
        try:
            _walk_tensor_expr(expr.operand, ctx, terms)
        finally:
            ctx.sign = saved_sign
        return
    if isinstance(expr, ScalarMultiple) and expr.scalar.is_background:
        saved_coef = ctx.coefficient_component
        ctx.coefficient_component = multiply_components(
            ctx.coefficient_component,
            _extract_background_scalar_component(expr.scalar))
        try:
            _walk_tensor_expr(expr.operand, ctx, terms)
        finally:
            ctx.coefficient_component = saved_coef
        return
    if isinstance(expr, Dt):
        if expr.operand.is_background:
            raise NotImplementedError(
                "Dt(...) of a pure-background expression has no meaning in a "
                "linear eigenproblem; rejected."
            )
        saved_in_dt = ctx.in_dt
        ctx.in_dt = True
        try:
            _walk_tensor_expr(expr.operand, ctx, terms)
        finally:
            ctx.in_dt = saved_in_dt
        return
    if isinstance(expr, FieldRef):
        field = expr.field
        if field.is_background:
            if ctx.in_dt:
                _warn_dt_of_background(field.name)
                return
            ctx.emit_task(TASK_SOURCE_TENSOR, unknown_name=None,
                          target=TARGET_RHS, source_field=field)
            return
        if not isinstance(field, TensorField):
            raise NotImplementedError(
                f"tensor equation: bare unknown {field.name!r} is a "
                f"{field.kind}, not a tensor")
        ctx.emit_task(TASK_IDENTITY_TENSOR, unknown_name=field.name,
                      target=TARGET_B if ctx.in_dt else TARGET_A)
        return
    if ctx.in_dt:
        raise NotImplementedError(
            f"Dt of the tensor expression {expr!r} is not supported; the "
            f"eigenvalue side holds unknowns, not operators."
        )
    unknown_vec_fields = _collect_unknown_vector_fields(expr)
    if len(unknown_vec_fields) != 1 or _contains_unknown_tensor(expr):
        raise NotImplementedError(
            f"tensor equation: the term {expr!r} must be an unknown tensor, "
            f"a background tensor, or a tensor expression linear in exactly "
            f"one vector unknown (Grad, Transpose, sums, background "
            f"multiples)"
        )
    terms.append((ctx.sign, ctx.coefficient_component, expr))


def _grad_transpose_weights(expr, unknown_name: str):
    """(a, b) with expr = a Grad(u) + b Transpose(Grad(u)) for the vector
    unknown u, as SymPy expressions; None when expr is not such a
    combination (numeric coefficients, background tensors, other
    operators)."""
    if isinstance(expr, Sum):
        left = _grad_transpose_weights(expr.a, unknown_name)
        right = _grad_transpose_weights(expr.b, unknown_name)
        if left is None or right is None:
            return None
        return (left[0] + right[0], left[1] + right[1])
    if isinstance(expr, Negation):
        inner = _grad_transpose_weights(expr.operand, unknown_name)
        return None if inner is None else (-inner[0], -inner[1])
    if isinstance(expr, ScalarMultiple):
        if not expr.scalar.is_background:
            return None
        comp = _extract_background_scalar_component(expr.scalar)
        if not isinstance(comp, SymPyComponent):
            return None
        inner = _grad_transpose_weights(expr.operand, unknown_name)
        return None if inner is None else (comp.expr * inner[0], comp.expr * inner[1])
    if isinstance(expr, Transpose):
        inner = _grad_transpose_weights(expr.operand, unknown_name)
        return None if inner is None else (inner[1], inner[0])
    if isinstance(expr, AsSymmetric):
        inner = _grad_transpose_weights(expr.operand, unknown_name)
        if inner is None:
            return None
        half = (inner[0] + inner[1]) / 2
        return (half, half)
    if isinstance(expr, Grad) and _is_unknown_vector(expr.operand) \
            and expr.operand.field.name == unknown_name:
        return (sp.S.One, sp.S.Zero)
    return None


def _emit_tensor_of_vector(terms: list, ctx: _WalkContext) -> None:
    """One TASK_TENSOR_OF_VECTOR per vector unknown of a tensor equation:
    the reduced components r T_{sigma_a, sigma_b} of the signed,
    SymPy-weighted sum of its tensor terms in that unknown's placeholders,
    with 'tensor_symmetric' checked symbolically on the sum."""
    by_vector = {}
    for sign, coef, expr in terms:
        host = _collect_unknown_vector_fields(expr)[0]
        by_vector.setdefault(host.name, (host, []))[1].append((sign, coef, expr))
    for name, (host, parts) in by_vector.items():
        v_placeholders = {name: vector_unknown_spin_placeholders(host)}
        total = {(s_a, s_b): sp.S.Zero for s_a in (-1, 0, 1) for s_b in (-1, 0, 1)}
        for sign, coef, expr in parts:
            if not isinstance(coef, SymPyComponent):
                raise NotImplementedError(
                    f"tensor equation: the term {expr!r} carries a numeric "
                    f"background coefficient; SymPy-backed coefficients only")
            reduced = reduce_tensor_expression(
                expr, m=ctx.m, unknown_spin_placeholders=v_placeholders)
            for key in total:
                total[key] += sign * coef.expr * reduced.components[key]
        rT = {key: sp.expand(_r_sym * val) for key, val in total.items()}
        symmetric = all(sp.expand(rT[(a, b)] - rT[(b, a)]) == 0
                        for a in (-1, 0, 1) for b in (-1, 0, 1) if a < b)
        weights = (sp.S.Zero, sp.S.Zero)
        for sign, coef, expr in parts:
            ab = _grad_transpose_weights(expr, name)
            if ab is None or weights is None:
                weights = None
                break
            weights = (weights[0] + sign * coef.expr * ab[0],
                       weights[1] + sign * coef.expr * ab[1])
        saved_sign, saved_coef = ctx.sign, ctx.coefficient_component
        ctx.sign, ctx.coefficient_component = +1, SymPyComponent(sp.S.One)
        try:
            ctx.emit_task(
                TASK_TENSOR_OF_VECTOR,
                unknown_name=name,
                extra_metadata={
                    "rT_components": rT,
                    "v_placeholders": v_placeholders,
                    "tensor_symmetric": symmetric,
                    "grad_transpose_weights": weights,
                },
            )
        finally:
            ctx.sign, ctx.coefficient_component = saved_sign, saved_coef


def _contains_unknown_tensor(expr) -> bool:
    if isinstance(expr, FieldRef):
        return isinstance(expr.field, TensorField) and not expr.field.is_background
    return any(_contains_unknown_tensor(c) for c in getattr(expr, "children", ()))


def walk_equation_system(system: EquationSystem, m: int) -> List[OperatorTask]:
    """Tasks of all equations in a system, concatenated in equation
    order. Boundary conditions are walked by `walk_boundary_conditions`."""
    all_tasks: List[OperatorTask] = []
    for eq in system.equations:
        all_tasks.extend(walk_equation(eq, m))
    return all_tasks


# ===========================================================================
# Boundary-condition walker
# ===========================================================================

def walk_boundary_conditions(system: EquationSystem, m: int, *,
                             composite: bool = True) -> List:
    """BCTasks of all boundary conditions in a system.

    Each BC expression, after stripping outer negations, is either an
    unknown-side expression X (homogeneous, X = 0) or a Sum of X with
    a background scalar g (X = -g). The unknown side X may be:

        p                                   BC_DIRICHLET_SCALAR
        Dot(u_hat, v), Dot(v, u_hat)        BC_DIRICHLET_VECTOR_PROJECTION
        Dot(r_hat, Grad(s)), Dot(Grad(s), r_hat)   BC_NEUMANN_SCALAR_RADIAL
        Dot(u, Dot(T_expr, w)) and its
        Dot(Dot(u, T_expr), w) / left-mult variants  BC_SCALAR_LINEAR_FROM_TENSOR
        Dot(u_hat, X), X a constant-coefficient combination of
        Curl(v) and Cross(v, B0) / Cross(B0, v)     BC_PROJECTION_CURL_CROSS

    with p, s unknown scalars, v an unknown vector, u_hat / u / w the
    unit vectors r_hat, theta_hat, phi_hat, and T_expr a tensor
    expression in one unknown vector. With `composite`, a Sum /
    Negation / ScalarMultiple tree over several such shapes, with
    background coefficients and the unknown in more than one term
    (Robin, Navier slip, the normal stress), is a BC_LINEAR_COMBINATION
    of them, and Dot(u_hat, X) with X such a tree distributes the unit
    over it; without it such rows are refused as before, for the cores
    that consume the single-shape kinds only. Only radial boundaries
    are supported; anything else raises NotImplementedError.
    """
    out: List[BCTask] = []
    for bc in system.boundary_conditions:
        bc_task = _parse_one_bc(bc, m, composite=composite)
        out.append(bc_task)
    return out


def _parse_one_bc(bc, m: int, composite: bool = True):
    """Dispatch on a BC's expression structure and build its BCTask; with
    `composite` a row of several shapes is a BC_LINEAR_COMBINATION."""
    expr = bc.expr
    loc = bc.location
    if loc not in ("r=0", "r=1"):
        raise NotImplementedError(
            f"BC at {loc!r} not supported; only radial boundaries handled."
        )
    while isinstance(expr, Negation):
        expr = expr.operand
    if isinstance(expr, FieldRef):
        field = expr.field
        if field.is_background:
            raise NotImplementedError(
                f"BC {bc.name!r}: expression {expr!r} is a pure background. "
                f"A BC must constrain at least one unknown."
            )
        if field.kind == "scalar":
            return BCTask(
                kind=BC_DIRICHLET_SCALAR,
                location=loc,
                unknown_name=field.name,
                value=None,
                name=bc.name,
            )
        raise NotImplementedError(
            f"BC {bc.name!r}: bare unknown FieldRef must be scalar; "
            f"got {field.kind}. For vector unknowns, write "
            f"Dot(unit_vector, v) instead."
        )
    if isinstance(expr, Dot):
        try:
            return _parse_dot_bc(bc, expr, loc, m, value=None)
        except NotImplementedError:
            if not composite or _unit_dot_over_tree(expr) is None:
                raise
        return _parse_linear_combination_bc(bc, expr, loc, m)
    if composite and isinstance(expr, ScalarMultiple):
        return _parse_linear_combination_bc(bc, expr, loc, m)
    if isinstance(expr, Sum):
        a, b = expr.a, expr.b
        if composite and not _is_single_shape_sum(a, b):
            return _parse_linear_combination_bc(bc, expr, loc, m)
        unknown_side, bg_side = _split_unknown_and_background(a, b, bc.name)
        value_sympy = -_extract_signed_background_sympy(bg_side, bc.name)
        usign = +1
        u_expr = unknown_side
        while isinstance(u_expr, Negation):
            usign = -usign
            u_expr = u_expr.operand
        value_sympy = usign * value_sympy
        if value_sympy == sp.S.Zero:
            value_component = None
        else:
            value_component = SymPyComponent(value_sympy)
        if isinstance(u_expr, FieldRef):
            field = u_expr.field
            if field.is_background:
                raise NotImplementedError(
                    f"BC {bc.name!r}: unknown side has no unknown field."
                )
            if field.kind == "scalar":
                return BCTask(
                    kind=BC_DIRICHLET_SCALAR,
                    location=loc,
                    unknown_name=field.name,
                    value=value_component,
                    name=bc.name,
                )
            raise NotImplementedError(
                f"BC {bc.name!r}: bare FieldRef unknown must be scalar; "
                f"got {field.kind}. For vector unknowns, write "
                f"Dot(unit_vector, v)."
            )
        if isinstance(u_expr, Dot):
            return _parse_dot_bc(bc, u_expr, loc, m, value=value_component)
        raise NotImplementedError(
            f"BC {bc.name!r}: unknown-side expression {u_expr!r} not "
            f"recognized. Supported unknown-side forms are a scalar "
            f"FieldRef and Dot(unit_vector, v)."
        )
    raise NotImplementedError(
        f"BC {bc.name!r}: expression {expr!r} not recognized."
    )


def _is_single_shape_sum(a, b) -> bool:
    """True when a BC Sum has one unknown side, a bare scalar FieldRef
    or a Dot under negations, and one background side (or none): the
    single-shape rows; anything else is a linear combination."""
    a_has, b_has = _contains_unknown(a), _contains_unknown(b)
    if a_has and b_has:
        return False
    if not a_has and not b_has:
        return True
    u_expr = a if a_has else b
    while isinstance(u_expr, Negation):
        u_expr = u_expr.operand
    return isinstance(u_expr, (FieldRef, Dot))


def _unit_dot_over_tree(expr):
    """(unit FieldRef, operand, unit_first) when `expr` is Dot(u_hat, X)
    or Dot(X, u_hat) with X a Sum, Negation or background ScalarMultiple,
    over which the unit projection distributes; None otherwise."""
    if not isinstance(expr, Dot):
        return None
    for unit, operand, unit_first in ((expr.a, expr.b, True), (expr.b, expr.a, False)):
        if _is_unit_vector_ref(unit) and isinstance(
                operand, (Sum, Negation, ScalarMultiple)):
            if isinstance(operand, ScalarMultiple) and not operand.scalar.is_background:
                return None
            return unit, operand, unit_first
    return None


def _linear_bc_leaves(expr, coefficient=None):
    """[(coefficient, leaf)] of a Sum / Negation / background
    ScalarMultiple tree: every leaf with the BackgroundComponent product
    of the factors above it (None for 1)."""
    if isinstance(expr, Sum):
        return (_linear_bc_leaves(expr.a, coefficient)
                + _linear_bc_leaves(expr.b, coefficient))
    if isinstance(expr, Negation):
        minus = SymPyComponent(sp.S.NegativeOne)
        return _linear_bc_leaves(expr.operand, _times(coefficient, minus))
    if isinstance(expr, ScalarMultiple) and expr.scalar.is_background:
        factor = background_scalar_component(expr.scalar)
        return _linear_bc_leaves(expr.operand, _times(coefficient, factor))
    return [(coefficient, expr)]


def _times(a, b):
    """Product of two optional BackgroundComponents (None is 1)."""
    if a is None:
        return b
    if b is None:
        return a
    return multiply_components(a, b)


def _describe_bc_shapes() -> str:
    """The term shapes of a boundary row, for refusal messages."""
    return ("a scalar unknown p, Dot(u_hat, v), Dot(r_hat, Grad(s)), "
            "Dot(u, Dot(T_expr, w)) and Dot(u_hat, X) with X a combination of "
            "Curl(v) and Cross(v, B0), each under background coefficients")


def _bc_terms_of_leaf(bc, leaf, loc, m: int):
    """[(coefficient, homogeneous BCTask)] of one unknown-bearing leaf of
    a boundary row: a single shape as one term, Dot(u_hat, X) with X a
    tree distributed into the terms of its leaves. A term that carries
    a value (the background part of a tensor projection) keeps it for
    the caller to fold into the row's value."""
    if isinstance(leaf, FieldRef):
        return [(None, BCTask(kind=BC_DIRICHLET_SCALAR, location=loc,
                              unknown_name=leaf.field.name, name=bc.name))]
    if isinstance(leaf, Dot):
        try:
            return [(None, _parse_dot_bc(bc, leaf, loc, m, value=None))]
        except NotImplementedError as exc:
            split = _unit_dot_over_tree(leaf)
            if split is None:
                raise NotImplementedError(
                    f"BC {bc.name!r}: term {leaf!r} is none of the boundary shapes "
                    f"({_describe_bc_shapes()}); the walker reports: {exc}"
                ) from exc
        unit, operand, unit_first = split
        out = []
        for coefficient, sub in _regroup_curl_cross(bc, leaf, _linear_bc_leaves(operand)):
            node = Dot(unit, sub) if unit_first else Dot(sub, unit)
            for inner_coefficient, task in _bc_terms_of_leaf(bc, node, loc, m):
                out.append((_times(coefficient, inner_coefficient), task))
        return out
    raise NotImplementedError(
        f"BC {bc.name!r}: term {leaf!r} is none of the boundary shapes "
        f"({_describe_bc_shapes()})"
    )


def _regroup_curl_cross(bc, leaf, leaves):
    """The leaves of a projected tree with its Curl(u) / Cross(u, B0)
    leaves under SymPy coefficients gathered back into one combination
    at the place of the first, so that a distributed electric field is
    one BC_PROJECTION_CURL_CROSS term as when written projected."""
    out = []
    gathered = []
    slot = None
    for coefficient, sub in leaves:
        if sub.is_background:
            raise NotImplementedError(
                f"BC {bc.name!r}: term {leaf!r} projects the background {sub!r}; "
                f"write the background part of a row outside the projection"
            )
        sympy_coefficient = coefficient is None or isinstance(coefficient, SymPyComponent)
        if sympy_coefficient and _linear_curl_cross_terms(sub) is not None:
            if coefficient is not None:
                sub = ScalarMultiple(_BackgroundScalarValue(coefficient.expr), sub)
            gathered.append(sub)
            if slot is None:
                slot = len(out)
                out.append(None)
            continue
        out.append((coefficient, sub))
    if gathered:
        combined = gathered[0]
        for sub in gathered[1:]:
            combined = Sum(combined, sub)
        out[slot] = (None, combined)
    return out


def _sympy_of(component, bc_name: str, what: str) -> sp.Expr:
    if component is None:
        return sp.S.One
    if isinstance(component, SymPyComponent):
        return component.expr
    raise NotImplementedError(
        f"BC {bc_name!r}: {what} is non-SymPy ({type(component).__name__}); only "
        f"SymPy-backed BC values are supported"
    )


def _term_unknowns(task: BCTask) -> list:
    """The unknowns a single-shape term touches, first-seen order."""
    if task.kind == BC_PROJECTION_CURL_CROSS:
        names = [task.unknown_name]
        for term in task.metadata["terms"]:
            if term["unknown"] not in names:
                names.append(term["unknown"])
        return names
    return [task.unknown_name]


def _term_tangent(task: BCTask) -> Optional[str]:
    """The tangential unit a single-shape term projects on, or None."""
    if task.kind in (BC_DIRICHLET_VECTOR_PROJECTION, BC_PROJECTION_CURL_CROSS):
        return task.unit if task.unit in (BC_UNIT_THETA_HAT, BC_UNIT_PHI_HAT) else None
    if task.kind == BC_SCALAR_LINEAR_FROM_TENSOR:
        return {+1: BC_UNIT_THETA_HAT, -1: BC_UNIT_PHI_HAT}.get(task.metadata["host_sigma"])
    return None


def _parse_linear_combination_bc(bc, expr, loc, m: int):
    """BCTask of a boundary row that is a Sum / Negation / ScalarMultiple
    tree over the single shapes, sum_i c_i X_i + g = 0: the background
    leaves g move to the value, the unknown leaves become terms. One
    term with a SymPy coefficient (or no value) is normalized to its own
    single-shape kind, X = value / c; anything else is a
    BC_LINEAR_COMBINATION."""
    value_sympy = sp.S.Zero
    terms = []
    for coefficient, leaf in _linear_bc_leaves(expr):
        if leaf.is_background:
            g = _times(coefficient, background_scalar_component(leaf))
            value_sympy -= _sympy_of(g, bc.name, f"the value term {leaf!r}")
            continue
        for inner_coefficient, task in _bc_terms_of_leaf(bc, leaf, loc, m):
            c = _times(coefficient, inner_coefficient)
            if task.value is not None:
                c_sympy = _sympy_of(c, bc.name, f"the coefficient of {leaf!r}")
                value_sympy += c_sympy * _sympy_of(task.value, bc.name, "a term value")
                task = dataclasses.replace(task, value=None)
            terms.append((c, task))
    if not terms:
        raise NotImplementedError(
            f"BC {bc.name!r}: expression {expr!r} constrains no unknown."
        )
    value_sympy = sp.expand(value_sympy)
    if len(terms) == 1 and (value_sympy == 0 or terms[0][0] is None
                            or isinstance(terms[0][0], SymPyComponent)):
        c, task = terms[0]
        if value_sympy != 0 and c is not None:
            value_sympy = sp.expand(value_sympy / c.expr)
        value = None if value_sympy == 0 else SymPyComponent(value_sympy)
        return dataclasses.replace(task, value=value)
    unknowns = []
    for _c, task in terms:
        for name in _term_unknowns(task):
            if name not in unknowns:
                unknowns.append(name)
    rank = {"scalar": 0, "vector": 1, "tensor": 2}
    kinds = {}
    for _c, task in terms:
        for name in _term_unknowns(task):
            kinds.setdefault(name, _unknown_kind_of(expr, name))
    host = max(unknowns, key=lambda n: rank[kinds[n]])
    tangents = {_term_tangent(task) for _c, task in terms} - {None}
    if not tangents:
        tangent = None
    elif len(tangents) == 1:
        (tangent,) = tangents
    else:
        tangent = BC_TANGENT_MIXED
    if kinds[host] == "scalar":
        host_sigma = None
    else:
        host_sigma = {BC_UNIT_THETA_HAT: +1, BC_UNIT_PHI_HAT: -1, None: 0}.get(tangent, +1)
    coefficients = [SymPyComponent(sp.S.One) if c is None else c for c, _t in terms]
    return BCTask(
        kind=BC_LINEAR_COMBINATION,
        location=loc,
        unknown_name=host,
        value=None if value_sympy == 0 else SymPyComponent(value_sympy),
        name=bc.name,
        metadata={
            "terms": [(c, task) for c, (_c, task) in zip(coefficients, terms)],
            "unknowns": tuple(unknowns),
            "host_unknown": host,
            "host_sigma": host_sigma,
            "tangent": tangent,
            "pairing_key": boundary_pairing_key(bc),
        },
    )


def _unknown_kind_of(expr, name: str) -> str:
    """The kind of the unknown `name` referenced in `expr`."""
    if isinstance(expr, FieldRef):
        if not expr.field.is_background and expr.field.name == name:
            return expr.field.kind
        return None
    for child in getattr(expr, "children", ()):
        kind = _unknown_kind_of(child, name)
        if kind is not None:
            return kind
    return None


def boundary_pairing_key(bc) -> tuple:
    """(location, structure) of a boundary condition with theta_hat and
    phi_hat made alike: the rows of a theta / phi pair written on one
    expression share the key, and a spectral core imposes such a pair
    as one angular row (tasks.py, BC_LINEAR_COMBINATION)."""
    def key(e):
        if isinstance(e, FieldRef):
            if _is_unit_vector_ref(e) and e.field.name in ("theta_hat", "phi_hat"):
                return "tangent_hat"
            return repr(e)
        if not e.children:
            return repr(e)
        return f"{type(e).__name__}({', '.join(key(c) for c in e.children)})"
    return (bc.location, key(bc.expr))


def _parse_dot_bc(bc, dot_expr, loc, m: int, value):
    """BCTask for a Dot unknown side: radial Neumann, then tensor
    projection, then vector projection."""
    neumann_bc = _try_parse_neumann_scalar_radial(bc, dot_expr, loc, value=value)
    if neumann_bc is not None:
        return neumann_bc
    tensor_bc = _try_parse_tensor_projection_bc(bc, dot_expr, loc, m, value=value)
    if tensor_bc is not None:
        return tensor_bc
    curl_bc = _try_parse_curl_cross_projection_bc(bc, dot_expr, loc, value)
    if curl_bc is not None:
        return curl_bc
    return _parse_vector_projection_bc(bc, dot_expr, loc, value)


def _linear_curl_cross_terms(expr, coefficient=sp.S.One):
    """The terms of a constant-coefficient combination of Curl(u) and
    Cross(u, B0) / Cross(B0, u) of unknown vectors, as the dicts of
    BC_PROJECTION_CURL_CROSS; None when a term has another shape."""
    if isinstance(expr, Sum):
        left = _linear_curl_cross_terms(expr.a, coefficient)
        right = _linear_curl_cross_terms(expr.b, coefficient)
        return None if left is None or right is None else left + right
    if isinstance(expr, Negation):
        return _linear_curl_cross_terms(expr.operand, -coefficient)
    if isinstance(expr, ScalarMultiple):
        if not isinstance(expr.scalar, _BackgroundScalarValue):
            return None
        return _linear_curl_cross_terms(expr.operand, coefficient * expr.scalar.value)
    if _is_curl_of_unknown_vector(expr):
        return [{"kind": "curl", "coefficient": coefficient,
                 "unknown": expr.operand.field.name, "background": None, "side": None}]
    if isinstance(expr, Cross):
        a, b = expr.a, expr.b
        if _is_unknown_vector(a) and _is_bg_vector(b):
            return [{"kind": "cross", "coefficient": coefficient,
                     "unknown": a.field.name, "background": b.field, "side": "right"}]
        if _is_bg_vector(a) and _is_unknown_vector(b):
            return [{"kind": "cross", "coefficient": coefficient,
                     "unknown": b.field.name, "background": a.field, "side": "left"}]
    return None


def _try_parse_curl_cross_projection_bc(bc, dot_expr, loc, value):
    """BCTask of kind BC_PROJECTION_CURL_CROSS for Dot(u_hat, X) or
    Dot(X, u_hat) with X a constant-coefficient combination of Curl(v)
    and Cross(v, B0) terms of unknown vectors, or None."""
    a, b = dot_expr.a, dot_expr.b
    if _is_unit_vector_ref(a):
        unit, inner = a.field, b
    elif _is_unit_vector_ref(b):
        unit, inner = b.field, a
    else:
        return None
    terms = _linear_curl_cross_terms(inner)
    if not terms:
        return None
    curls = [t["unknown"] for t in terms if t["kind"] == "curl"]
    unit_tag = {"r_hat": BC_UNIT_R_HAT, "theta_hat": BC_UNIT_THETA_HAT,
                "phi_hat": BC_UNIT_PHI_HAT}[unit.name]
    return BCTask(
        kind=BC_PROJECTION_CURL_CROSS,
        location=loc,
        unknown_name=curls[0] if curls else terms[0]["unknown"],
        value=value,
        name=bc.name,
        unit=unit_tag,
        metadata={"terms": terms},
    )


# Names of the IR's unit-vector background fields (equations.py).
_UNIT_VECTOR_NAMES = frozenset({"r_hat", "theta_hat", "phi_hat"})


def _is_unit_vector_ref(expr) -> bool:
    return (
        isinstance(expr, FieldRef)
        and expr.field.is_background
        and expr.field.name in _UNIT_VECTOR_NAMES
    )


def _try_parse_tensor_projection_bc(bc, dot_expr, loc, m: int, value):
    """BCTask of kind BC_SCALAR_LINEAR_FROM_TENSOR for
    Dot(u, Dot(T_expr, w)) or Dot(u, Dot(w, T_expr)) (and the outer
    unit vector on either side), or None if `dot_expr` has a different
    shape.

    Dot(A, B) contracts A's last index with B's first, so in
    Dot(u, Dot(T, w)) the unit w contracts T's second index and u its
    first; in Dot(u, Dot(w, T)) the roles swap. The constraint is
    u^{sigma_1} T_{sigma_1, sigma_2} w^{sigma_2} in the covariant
    components of T.

    The host row is chosen by the (first-index, second-index) unit
    pair: the non-r_hat unit when exactly one of the two is not r_hat,
    else the first-index unit, mapped as r_hat -> sigma 0,
    theta_hat -> +1, phi_hat -> -1. Syntactically different forms of
    the same contraction therefore host on the same row.
    """
    a, b = dot_expr.a, dot_expr.b
    if _is_unit_vector_ref(a) and isinstance(b, Dot):
        outer_unit_field = a.field
        inner = b
    elif _is_unit_vector_ref(b) and isinstance(a, Dot):
        outer_unit_field = b.field
        inner = a
    else:
        return None
    ia, ib = inner.a, inner.b
    if ia.is_tensor and _is_unit_vector_ref(ib):
        tensor_expr = ia
        first_index_unit = outer_unit_field.name
        second_index_unit = ib.field.name
    elif ib.is_tensor and _is_unit_vector_ref(ia):
        tensor_expr = ib
        first_index_unit = ia.field.name
        second_index_unit = outer_unit_field.name
    else:
        return None
    unknown_vec_fields = _collect_unknown_vector_fields(tensor_expr)
    if not unknown_vec_fields:
        raise NotImplementedError(
            f"BC {bc.name!r}: tensor expression contains no unknown vector "
            f"fields; can't constrain anything."
        )
    placeholders = {
        f.name: vector_unknown_spin_placeholders(f) for f in unknown_vec_fields
    }
    reduced = reduce_tensor_expression(
        tensor_expr, m=m, unknown_spin_placeholders=placeholders,
    )
    bg_components, components = _split_background_part(
        reduced.components, placeholders)
    if bg_components is not None:
        # X + g = value becomes X = value - g.
        g = _project_tensor_to_scalar(
            bg_components, outer_unit=first_index_unit,
            inner_unit=second_index_unit,
        )
        if value is not None and not isinstance(value, SymPyComponent):
            raise NotImplementedError(
                f"BC {bc.name!r}: a background tensor part with a non-SymPy "
                f"value ({type(value).__name__}) is not supported."
            )
        shifted = sp.expand((sp.S.Zero if value is None else value.expr) - g)
        value = None if shifted == 0 else SymPyComponent(shifted)
    scalar_expr = sp.expand(_project_tensor_to_scalar(
        components,
        outer_unit=first_index_unit,
        inner_unit=second_index_unit,
    ))
    host_for_unit = {"r_hat": 0, "theta_hat": +1, "phi_hat": -1}
    first_is_r = (first_index_unit == "r_hat")
    second_is_r = (second_index_unit == "r_hat")
    if first_is_r and not second_is_r:
        host_sigma = host_for_unit[second_index_unit]
    else:
        host_sigma = host_for_unit[first_index_unit]
    if len(unknown_vec_fields) > 1:
        raise NotImplementedError(
            f"BC {bc.name!r}: tensor-projection BC involves multiple "
            f"unknown vector fields ({[f.name for f in unknown_vec_fields]}); "
            f"only single-unknown tensor BCs are supported."
        )
    host_unknown = unknown_vec_fields[0].name
    return BCTask(
        kind=BC_SCALAR_LINEAR_FROM_TENSOR,
        location=loc,
        unknown_name=host_unknown,
        value=value,
        name=bc.name,
        metadata={
            "scalar_expr": scalar_expr,
            "placeholders": placeholders,
            "host_unknown": host_unknown,
            "host_sigma": host_sigma,
        },
    )


def _project_tensor_to_scalar(
    T_components: dict,
    outer_unit: str,
    inner_unit: str,
) -> sp.Expr:
    """The scalar u^{sigma_1} T_{sigma_1, sigma_2} w^{sigma_2} for
    covariant components T (as produced by `reduce_tensor_expression`)
    and unit vectors u = outer_unit, w = inner_unit.

    The unit vectors enter with their contravariant spin components,
        r_hat:     (r^-, r^0, r^+) = (0, 1, 0)
        theta_hat: (t^-, t^0, t^+) = (1/sqrt2, 0, 1/sqrt2)
        phi_hat:   (p^-, p^0, p^+) = (-i/sqrt2, 0, i/sqrt2),
    so the sums run over matching indices with no sign flip.

    Parameters
    ----------
    T_components : dict {(sigma_1, sigma_2): sp.Expr}
    outer_unit, inner_unit : str
        'r_hat', 'theta_hat', or 'phi_hat'.
    """
    sqrt2 = sp.sqrt(2)
    I = sp.I
    unit_spin = {
        "r_hat":     {+1: sp.S.Zero, 0: sp.S.One,   -1: sp.S.Zero},
        "theta_hat": {+1: 1 / sqrt2,  0: sp.S.Zero, -1: 1 / sqrt2},
        "phi_hat":   {+1: I / sqrt2,  0: sp.S.Zero, -1: -I / sqrt2},
    }
    if outer_unit not in unit_spin:
        raise ValueError(f"Unknown outer unit vector {outer_unit!r}")
    if inner_unit not in unit_spin:
        raise ValueError(f"Unknown inner unit vector {inner_unit!r}")
    u = unit_spin[outer_unit]
    w = unit_spin[inner_unit]
    Tw = {}
    for s1 in (-1, 0, +1):
        Tw[s1] = (
            T_components[(s1, -1)] * w[-1]
            + T_components[(s1,  0)] * w[ 0]
            + T_components[(s1, +1)] * w[+1]
        )
    return u[-1] * Tw[-1] + u[0] * Tw[0] + u[+1] * Tw[+1]


def _collect_unknown_vector_fields(expr) -> list:
    """Unique unknown VectorFields referenced in an expression tree, in
    first-seen order."""
    found = []
    seen = set()

    def visit(node):
        if isinstance(node, FieldRef):
            field = node.field
            if (
                isinstance(field, VectorField)
                and not field.is_background
                and id(field) not in seen
            ):
                found.append(field)
                seen.add(id(field))
            return
        if hasattr(node, "children"):
            for child in node.children:
                visit(child)

    visit(expr)
    return found


def _try_parse_neumann_scalar_radial(bc, dot_expr, loc, value):
    """BCTask of kind BC_NEUMANN_SCALAR_RADIAL for Dot(r_hat, Grad(s))
    or Dot(Grad(s), r_hat) with s an unknown scalar, or None if
    `dot_expr` has a different shape."""
    a, b = dot_expr.a, dot_expr.b

    def _is_r_hat(node):
        return (
            isinstance(node, FieldRef)
            and node.field.is_background
            and node.field.name == "r_hat"
        )

    if _is_r_hat(a) and _is_grad_of_unknown_scalar(b):
        grad_node = b
    elif _is_r_hat(b) and _is_grad_of_unknown_scalar(a):
        grad_node = a
    else:
        return None
    return BCTask(
        kind=BC_NEUMANN_SCALAR_RADIAL,
        location=loc,
        unknown_name=grad_node.operand.field.name,
        value=value,
        name=bc.name,
    )


def _parse_vector_projection_bc(bc, dot_expr, loc, value):
    """BCTask of kind BC_DIRICHLET_VECTOR_PROJECTION for
    Dot(unit_vec, v) or Dot(v, unit_vec) with v an unknown vector."""
    a, b = dot_expr.a, dot_expr.b
    if not (isinstance(a, FieldRef) and isinstance(b, FieldRef)):
        raise NotImplementedError(
            f"BC {bc.name!r}: Dot arguments must be bare FieldRefs; "
            f"got Dot({a!r}, {b!r})."
        )
    a_field, b_field = a.field, b.field
    if (a_field.is_background and a_field.name in _UNIT_VECTOR_NAMES
            and not b_field.is_background):
        unit_field, v_field = a_field, b_field
    elif (b_field.is_background and b_field.name in _UNIT_VECTOR_NAMES
            and not a_field.is_background):
        unit_field, v_field = b_field, a_field
    else:
        raise NotImplementedError(
            f"BC {bc.name!r}: Dot must be between a unit vector "
            f"(r_hat, theta_hat, phi_hat) and an unknown vector. "
            f"Got Dot({a_field.name}, {b_field.name})."
        )
    unit_tag = {
        "r_hat": BC_UNIT_R_HAT,
        "theta_hat": BC_UNIT_THETA_HAT,
        "phi_hat": BC_UNIT_PHI_HAT,
    }[unit_field.name]
    return BCTask(
        kind=BC_DIRICHLET_VECTOR_PROJECTION,
        location=loc,
        unknown_name=v_field.name,
        value=value,
        name=bc.name,
        unit=unit_tag,
    )


def _split_unknown_and_background(a, b, bc_name):
    """(unknown_side, bg_side) of a BC Sum's two operands; exactly one
    side must reference an unknown."""
    a_has_unknown = _contains_unknown(a)
    b_has_unknown = _contains_unknown(b)
    if a_has_unknown and not b_has_unknown:
        return a, b
    if b_has_unknown and not a_has_unknown:
        return b, a
    if a_has_unknown and b_has_unknown:
        raise NotImplementedError(
            f"BC {bc_name!r}: both sides of the Sum contain unknowns; "
            f"only single-unknown Dirichlet supported."
        )
    raise NotImplementedError(
        f"BC {bc_name!r}: neither side of the Sum contains an unknown; "
        f"the BC must constrain at least one unknown."
    )


def _contains_unknown(expr) -> bool:
    """True if `expr` references any unknown field."""
    if isinstance(expr, FieldRef):
        return not expr.field.is_background
    for child in getattr(expr, "children", ()):
        if _contains_unknown(child):
            return True
    return False


def _extract_signed_background_sympy(expr, bc_name) -> sp.Expr:
    """SymPy value of a pure-background scalar BC side: bare values,
    SymPy-backed background ScalarField refs, and Negations thereof."""
    if isinstance(expr, Negation):
        inner = _extract_signed_background_sympy(expr.operand, bc_name)
        return -inner
    if isinstance(expr, _BackgroundScalarValue):
        return expr.value
    if isinstance(expr, FieldRef):
        field = expr.field
        if not field.is_background:
            raise NotImplementedError(
                f"BC {bc_name!r}: cannot use unknown as value side"
            )
        if not hasattr(field, "component") or field.component is None:
            raise NotImplementedError(
                f"BC {bc_name!r}: field {field.name!r} has no component"
            )
        comp = field.component
        if isinstance(comp, SymPyComponent):
            return comp.expr
        raise NotImplementedError(
            f"BC {bc_name!r}: field {field.name!r} is non-SymPy "
            f"({type(comp).__name__}). Only SymPy-backed BC values "
            f"are supported."
        )
    raise NotImplementedError(
        f"BC {bc_name!r}: cannot extract value from {type(expr).__name__}: "
        f"{expr!r}"
    )
