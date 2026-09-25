"""spdee.ir.tensor_reduce — reduce a tensor expression to per-(sigma_1, sigma_2)
scalar components.

Given a tensor expression built from `Grad(vector_expression)`,
`Transpose`, `Sum`, `Negation`, `ScalarMultiple`, `AsSymmetric`, and
background tensor `FieldRef` leaves, produce a `ReducedTensor` whose
`components` dict maps spin pairs (sigma_1, sigma_2) in {-1, 0, +1}^2
to scalar SymPy expressions in (r, theta) at fixed azimuthal mode m,
with e^{i m phi} factored out. Unknown vectors appear through SymPy
Function placeholders (see `vector_unknown_spin_placeholders`).

Convention (V19): the components are COVARIANT,

    T_{sigma_1, sigma_2} = (e_{sigma_1})_i (e_{sigma_2})_j T_{ij},

with e_+ = (e_theta - i e_phi)/sqrt(2), e_0 = e_r,
e_- = (e_theta + i e_phi)/sqrt(2), and T_{ij} the physical
(orthonormal spherical) components. The metric is
g_{sigma sigma'} = delta_{sigma, -sigma'}, so raising or lowering an
index maps sigma -> -sigma. Vector components elsewhere in spdee are
contravariant, v^+ = (v_theta + i v_phi)/sqrt(2), v^0 = v_r,
v^- = (v_theta - i v_phi)/sqrt(2).

Downstream consumers (Div(tensor), Dot(bg_vec, tensor), BC
projections) contract these components; no tensor expression survives
past the reducer.

Symmetry is detected structurally (`is_structurally_symmetric`); the
user can assert it with `.as_symmetric()`, which is checked
symbolically here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import sympy as sp

from spdee.ir.expressions import (
    Expression, FieldRef, Sum, Negation, ScalarMultiple, AsSymmetric,
    _BackgroundScalarValue,
)
from spdee.ir.operators import Grad, Transpose
from spdee.ir.fields import (
    ScalarField, VectorField, TensorField,
    is_sympy_component, get_sympy_expr, r as r_sym, theta as t_sym,
)

# Spin indices in V19 order and all 9 index pairs.
SPIN_INDICES: Tuple[int, int, int] = (-1, 0, +1)
SPIN_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    (s1, s2) for s1 in SPIN_INDICES for s2 in SPIN_INDICES
)


@dataclass
class ReducedTensor:
    """Reduced form of a tensor expression.

    Attributes
    ----------
    components : dict {(sigma_1, sigma_2): sp.Expr}
        Covariant components at fixed m. All 9 keys are present even
        when `symmetric` is True.
    symmetric : bool
        Whether the reducer concluded the tensor is symmetric, from
        structural matching or a checked `.as_symmetric()`.
    """
    components: Dict[Tuple[int, int], sp.Expr]
    symmetric: bool

    def __getitem__(self, key: Tuple[int, int]) -> sp.Expr:
        return self.components[key]


# =============================================================================
# Vector unknown helper: per-spin SymPy placeholder for v^sigma(r, theta)
# =============================================================================

def vector_unknown_spin_placeholders(
    field: VectorField,
) -> Dict[int, sp.Function]:
    """SymPy Function placeholders {sigma: v_sigma(r, theta)} for an
    unknown vector, named `{field_name}_sigma_{m|0|p}`."""
    if field.is_background:
        raise TypeError(
            f"vector_unknown_spin_placeholders requires an unknown VectorField; "
            f"{field.name!r} is a background"
        )
    name_map = {-1: "m", 0: "0", +1: "p"}
    return {
        s: sp.Function(f"{field.name}_sigma_{name_map[s]}")(r_sym, t_sym)
        for s in SPIN_INDICES
    }


# =============================================================================
# Grad(vector) and Div(tensor) formulas
# =============================================================================

def grad_vector_spin_components(
    v_spin: Dict[int, sp.Expr],
    m: int,
) -> Dict[Tuple[int, int], sp.Expr]:
    """Covariant components (Grad v)_{sigma_1, sigma_2} of the gradient
    of a vector with contravariant spin components v^sigma at mode m.

    Parameters
    ----------
    v_spin : dict {-1: v_minus, 0: v_zero, +1: v_plus}
        Contravariant spin components as SymPy expressions in (r, theta).
    m : int
        Azimuthal mode (d/dphi -> i m).

    Returns
    -------
    dict {(sigma_1, sigma_2): sp.Expr}
    """
    vm, v0, vp = v_spin[-1], v_spin[0], v_spin[+1]
    sqrt2 = sp.sqrt(2)
    r, t = r_sym, t_sym

    def dr(f):  return sp.diff(f, r)
    def dt(f):  return sp.diff(f, t)
    def cot(x): return 1 / sp.tan(x)
    def csc(x): return 1 / sp.sin(x)

    return {
        (-1, -1): sqrt2 * (-m * vp * csc(t) - vp * cot(t) + dt(vp)) / (2 * r),
        (-1,  0): (-sqrt2 * m * v0 * csc(t) - 2 * vp + sqrt2 * dt(v0)) / (2 * r),
        (-1, +1): (-sqrt2 * m * vm * csc(t) + 2 * v0 + sqrt2 * vm * cot(t)
                   + sqrt2 * dt(vm)) / (2 * r),
        ( 0, -1): dr(vp),
        ( 0,  0): dr(v0),
        ( 0, +1): dr(vm),
        (+1, -1): ( sqrt2 * m * vp * csc(t) + 2 * v0 + sqrt2 * vp * cot(t)
                   + sqrt2 * dt(vp)) / (2 * r),
        (+1,  0): ( sqrt2 * m * v0 * csc(t) - 2 * vm + sqrt2 * dt(v0)) / (2 * r),
        (+1, +1): sqrt2 * (m * vm * csc(t) - vm * cot(t) + dt(vm)) / (2 * r),
    }


def r2_div_tensor_spin_components(
    U_cov: Dict[Tuple[int, int], sp.Expr],
    m: int,
) -> Dict[int, sp.Expr]:
    """Contravariant components r^2 (Div T)^{sigma_1} of the divergence
    of a tensor, expressed in U_{sigma_a, sigma_b} = r T_{sigma_a, sigma_b}.

    Multiplying by r^2 and substituting T = U/r clears every 1/r factor;
    the result involves only U, d_r U, d_theta U with cot/csc(theta)
    coefficients, so the operator is regular at the origin and the
    equation's mass becomes diag(r^2).

    Parameters
    ----------
    U_cov : dict {(sigma_a, sigma_b): sp.Expr}
        Covariant components of U = r T, e.g.
        `{k: r * red.components[k] for k in red.components}`.
    m : int
        Azimuthal mode.

    Returns
    -------
    dict {sigma_1: sp.Expr}
    """
    r, t = r_sym, t_sym
    sqrt2 = sp.sqrt(2)

    def dr(f): return sp.diff(f, r)
    def dt(f): return sp.diff(f, t)
    def csc(x): return 1 / sp.sin(x)
    def cot(x): return 1 / sp.tan(x)

    return {
        -1: (
            r * dr(U_cov[(0, +1)])
            + sqrt2 * dt(U_cov[(-1, +1)]) / 2
            + sqrt2 * dt(U_cov[(+1, +1)]) / 2
            + sqrt2 * U_cov[(+1, +1)] * cot(t)
            + sqrt2 * m * U_cov[(-1, +1)] * csc(t) / 2
            - sqrt2 * m * U_cov[(+1, +1)] * csc(t) / 2
            + U_cov[(0, +1)]
            + U_cov[(+1, 0)]
        ),
        0: (
            -U_cov[(-1, +1)]
            - U_cov[(+1, -1)]
            + r * dr(U_cov[(0, 0)])
            + sqrt2 * dt(U_cov[(-1, 0)]) / 2
            + sqrt2 * dt(U_cov[(+1, 0)]) / 2
            + sqrt2 * U_cov[(-1, 0)] * cot(t) / 2
            + sqrt2 * U_cov[(+1, 0)] * cot(t) / 2
            + sqrt2 * m * U_cov[(-1, 0)] * csc(t) / 2
            - sqrt2 * m * U_cov[(+1, 0)] * csc(t) / 2
            + U_cov[(0, 0)]
        ),
        +1: (
            r * dr(U_cov[(0, -1)])
            + sqrt2 * dt(U_cov[(-1, -1)]) / 2
            + sqrt2 * dt(U_cov[(+1, -1)]) / 2
            + sqrt2 * U_cov[(-1, -1)] * cot(t)
            + sqrt2 * m * U_cov[(-1, -1)] * csc(t) / 2
            - sqrt2 * m * U_cov[(+1, -1)] * csc(t) / 2
            + U_cov[(-1, 0)]
            + U_cov[(0, -1)]
        ),
    }


# =============================================================================
# Structural symmetry detection
# =============================================================================

def _expressions_structurally_equal(a: Expression, b: Expression) -> bool:
    """Structural equality of two Expression trees: same node types and
    recursively equal children. FieldRef leaves compare by field
    identity, _BackgroundScalarValue leaves by SymPy value."""
    if type(a) is not type(b):
        return False
    if a is b:
        return True
    if isinstance(a, FieldRef):
        return a.field is b.field
    if isinstance(a, _BackgroundScalarValue):
        return a.value == b.value
    if len(a.children) != len(b.children):
        return False
    return all(
        _expressions_structurally_equal(ca, cb)
        for ca, cb in zip(a.children, b.children)
    )


def is_structurally_symmetric(expr: Expression) -> bool:
    """Structural test for symmetry of a tensor expression.

    Recognized patterns:
    - `AsSymmetric(X)`
    - `FieldRef(TensorField(symmetric=True))`
    - `Transpose(sym)`, `Negation(sym)`, `ScalarMultiple(c, sym)`
    - `Sum(sym, sym)`
    - `X + Transpose(X)` or `Transpose(X) + X` for any tensor X, using
      structural equality so separately constructed copies of X match.

    Anything else is reported as not symmetric.
    """
    if not isinstance(expr, Expression) or not expr.is_tensor:
        return False
    if isinstance(expr, AsSymmetric):
        return True
    if isinstance(expr, FieldRef):
        field = expr.field
        if isinstance(field, TensorField) and field.symmetric:
            return True
        return False
    if isinstance(expr, Transpose):
        return is_structurally_symmetric(expr.operand)
    if isinstance(expr, Negation):
        return is_structurally_symmetric(expr.operand)
    if isinstance(expr, ScalarMultiple):
        return is_structurally_symmetric(expr.operand)
    if isinstance(expr, Sum):
        a_sym = is_structurally_symmetric(expr.a)
        b_sym = is_structurally_symmetric(expr.b)
        if a_sym and b_sym:
            return True
        if (
            isinstance(expr.b, Transpose)
            and _expressions_structurally_equal(expr.b.operand, expr.a)
        ):
            return True
        if (
            isinstance(expr.a, Transpose)
            and _expressions_structurally_equal(expr.a.operand, expr.b)
        ):
            return True
        return False
    return False


# =============================================================================
# Reduction primitive
# =============================================================================

def reduce_tensor_expression(
    expr: Expression,
    m: int,
    *,
    unknown_spin_placeholders: Optional[Dict[str, Dict[int, sp.Function]]] = None,
) -> ReducedTensor:
    """Reduce a tensor expression to its covariant (sigma_1, sigma_2)
    components at azimuthal mode m.

    Parameters
    ----------
    expr : Expression with kind="tensor"
    m : int
        Azimuthal mode.
    unknown_spin_placeholders : dict, optional
        {unknown_name: {sigma: sp.Function}} placeholders to use for the
        unknown vectors (from `vector_unknown_spin_placeholders`). When
        omitted, fresh placeholders are created per unknown as they are
        met, so pass them explicitly when the result must be matched
        against placeholders held elsewhere.

    Raises
    ------
    TypeError
        Non-tensor input, or a node the reducer cannot handle.
    NotImplementedError
        Unknown tensor leaves, or scalar coefficients that are not
        background.
    ValueError
        An `.as_symmetric()` assertion that fails verification.
    """
    if not isinstance(expr, Expression):
        raise TypeError(f"reduce_tensor_expression requires an Expression; got {type(expr)}")
    if not expr.is_tensor:
        raise TypeError(
            f"reduce_tensor_expression requires a tensor expression; "
            f"got {expr.kind}"
        )
    placeholders = (
        dict(unknown_spin_placeholders)
        if unknown_spin_placeholders is not None
        else {}
    )

    def get_or_make_placeholders(field: VectorField) -> Dict[int, sp.Function]:
        if field.name not in placeholders:
            placeholders[field.name] = vector_unknown_spin_placeholders(field)
        return placeholders[field.name]

    def reduce(node: Expression) -> ReducedTensor:
        if isinstance(node, FieldRef):
            field = node.field
            if not isinstance(field, TensorField):
                raise TypeError(
                    f"Tensor FieldRef expected; got field {field!r} of "
                    f"kind {node.kind}"
                )
            if field.is_background:
                # spin_components() is contravariant; lowering both
                # indices maps sigma -> -sigma.
                spin = field.spin_components()
                return ReducedTensor(
                    components={(s1, s2): spin[(-s1, -s2)] for (s1, s2) in SPIN_PAIRS},
                    symmetric=field.symmetric,
                )
            raise NotImplementedError(
                f"Tensor unknowns are not supported in the reducer; field "
                f"{field.name!r} is an unknown tensor. Tensor expressions "
                f"can only contain vector unknowns and tensor backgrounds."
            )
        if isinstance(node, Grad):
            inner = node.operand
            if not inner.is_vector:
                raise TypeError(
                    f"reduce_tensor_expression: Grad operand must be a "
                    f"vector; got {inner.kind}"
                )
            # A background carries no e^{i m phi}: its gradient is taken
            # at m = 0, the unknown part at m.
            bg_spin, unk_spin = _vector_expr_to_spin(inner, get_or_make_placeholders)
            comps = grad_vector_spin_components(unk_spin, m)
            if any(bg_spin[s] != 0 for s in SPIN_INDICES):
                bg_comps = grad_vector_spin_components(bg_spin, 0)
                comps = {pair: comps[pair] + bg_comps[pair] for pair in SPIN_PAIRS}
            return ReducedTensor(components=comps, symmetric=False)
        if isinstance(node, Transpose):
            inner = reduce(node.operand)
            transposed = {
                (s1, s2): inner.components[(s2, s1)]
                for (s1, s2) in SPIN_PAIRS
            }
            return ReducedTensor(
                components=transposed,
                symmetric=inner.symmetric,
            )
        if isinstance(node, Sum):
            a = reduce(node.a)
            b = reduce(node.b)
            comps = {
                pair: sp.expand(a.components[pair] + b.components[pair])
                for pair in SPIN_PAIRS
            }
            sym = is_structurally_symmetric(node)
            return ReducedTensor(components=comps, symmetric=sym)
        if isinstance(node, Negation):
            inner = reduce(node.operand)
            comps = {
                pair: -inner.components[pair]
                for pair in SPIN_PAIRS
            }
            return ReducedTensor(components=comps, symmetric=inner.symmetric)
        if isinstance(node, ScalarMultiple):
            scalar = node.scalar
            if not scalar.is_background:
                raise NotImplementedError(
                    "ScalarMultiple with non-background scalar is not "
                    "supported in the tensor reducer"
                )
            scalar_sym = _scalar_expression_to_sympy(scalar)
            inner = reduce(node.operand)
            comps = {
                pair: sp.expand(scalar_sym * inner.components[pair])
                for pair in SPIN_PAIRS
            }
            return ReducedTensor(components=comps, symmetric=inner.symmetric)
        if isinstance(node, AsSymmetric):
            inner = reduce(node.operand)
            for s1 in SPIN_INDICES:
                for s2 in SPIN_INDICES:
                    if s1 >= s2:
                        continue
                    diff = sp.simplify(
                        inner.components[(s1, s2)] - inner.components[(s2, s1)]
                    )
                    if diff != 0:
                        raise ValueError(
                            f"as_symmetric() verification FAILED: "
                            f"the ({s1:+d}, {s2:+d}) and ({s2:+d}, {s1:+d}) "
                            f"components of the tensor expression differ "
                            f"by a non-zero amount.\n"
                            f"  T^({s1:+d},{s2:+d}) = "
                            f"{inner.components[(s1, s2)]}\n"
                            f"  T^({s2:+d},{s1:+d}) = "
                            f"{inner.components[(s2, s1)]}\n"
                            f"  Difference: {diff}"
                        )
            return ReducedTensor(components=inner.components, symmetric=True)
        raise NotImplementedError(
            f"reduce_tensor_expression: don't know how to reduce node "
            f"of type {type(node).__name__}: {node!r}"
        )

    return reduce(expr)


# =============================================================================
# Vector expression reduction (used by Grad)
# =============================================================================

_ZERO_SPIN = {s: sp.S.Zero for s in SPIN_INDICES}


def _vector_expr_to_spin(
    expr: Expression,
    get_or_make_placeholders,
) -> Tuple[Dict[int, sp.Expr], Dict[int, sp.Expr]]:
    """(background, unknown) contravariant spin components
    {sigma: sp.Expr} of a vector expression built from FieldRefs
    (unknowns become placeholders, backgrounds use their stored
    components), Sum, Negation, and ScalarMultiple by a background
    scalar. The two parts are kept apart because a background carries
    no azimuthal dependence."""
    if not isinstance(expr, Expression):
        raise TypeError(f"vector expression expected; got {type(expr)}")
    if not expr.is_vector:
        raise TypeError(
            f"_vector_expr_to_spin expects a vector expression; got {expr.kind}"
        )
    if isinstance(expr, FieldRef):
        field = expr.field
        if not isinstance(field, VectorField):
            raise TypeError(
                f"Vector FieldRef expected to wrap a VectorField; got {type(field)}"
            )
        if field.is_background:
            spin = field.spin_components()
            return {s: spin[s] for s in SPIN_INDICES}, dict(_ZERO_SPIN)
        return dict(_ZERO_SPIN), dict(get_or_make_placeholders(field))
    if isinstance(expr, Sum):
        a_bg, a_unk = _vector_expr_to_spin(expr.a, get_or_make_placeholders)
        b_bg, b_unk = _vector_expr_to_spin(expr.b, get_or_make_placeholders)
        return ({s: sp.expand(a_bg[s] + b_bg[s]) for s in SPIN_INDICES},
                {s: sp.expand(a_unk[s] + b_unk[s]) for s in SPIN_INDICES})
    if isinstance(expr, Negation):
        bg, unk = _vector_expr_to_spin(expr.operand, get_or_make_placeholders)
        return ({s: -bg[s] for s in SPIN_INDICES}, {s: -unk[s] for s in SPIN_INDICES})
    if isinstance(expr, ScalarMultiple):
        scalar = expr.scalar
        if not scalar.is_background:
            raise NotImplementedError(
                "ScalarMultiple with non-background scalar not supported in "
                "vector expression reducer"
            )
        scalar_sym = _scalar_expression_to_sympy(scalar)
        bg, unk = _vector_expr_to_spin(expr.operand, get_or_make_placeholders)
        return ({s: sp.expand(scalar_sym * bg[s]) for s in SPIN_INDICES},
                {s: sp.expand(scalar_sym * unk[s]) for s in SPIN_INDICES})
    raise NotImplementedError(
        f"_vector_expr_to_spin: don't know how to reduce node of "
        f"type {type(expr).__name__}: {expr!r}"
    )


# =============================================================================
# Scalar expression -> SymPy
# =============================================================================

def _scalar_expression_to_sympy(expr: Expression) -> sp.Expr:
    """SymPy value of a background scalar expression built from bare
    values, SymPy-backed ScalarField refs, Sum, Negation and
    ScalarMultiple."""
    if not isinstance(expr, Expression) or not expr.is_scalar:
        raise TypeError(f"scalar expression expected; got {expr!r}")
    if not expr.is_background:
        raise NotImplementedError(
            "Only background scalar expressions can be reduced to SymPy; "
            "an unknown scalar appeared as a coefficient in a tensor "
            "expression, which is not supported."
        )
    if isinstance(expr, _BackgroundScalarValue):
        return expr.value
    if isinstance(expr, FieldRef):
        field = expr.field
        if isinstance(field, ScalarField) and field.is_background:
            if is_sympy_component(field.component):
                return get_sympy_expr(field.component)
            raise NotImplementedError(
                f"Non-SymPy background ScalarField {field.name!r} not "
                f"supported in tensor expression reducer"
            )
        raise NotImplementedError(
            f"FieldRef of {type(field).__name__} not supported as scalar"
        )
    if isinstance(expr, Sum):
        return _scalar_expression_to_sympy(expr.a) + _scalar_expression_to_sympy(expr.b)
    if isinstance(expr, Negation):
        return -_scalar_expression_to_sympy(expr.operand)
    if isinstance(expr, ScalarMultiple):
        return (
            _scalar_expression_to_sympy(expr.scalar)
            * _scalar_expression_to_sympy(expr.operand)
        )
    raise NotImplementedError(
        f"_scalar_expression_to_sympy: don't know how to reduce node of "
        f"type {type(expr).__name__}: {expr!r}"
    )
