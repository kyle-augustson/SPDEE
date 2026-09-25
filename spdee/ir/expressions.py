"""spdee.ir.expressions — symbolic expression tree.

Every IR expression has:

- a `kind` in {"scalar", "vector", "tensor"} — the tensorial rank;
- a `linearity` — "background" (no unknowns), "linear" (linear in the
  unknowns), or "nonlinear" (rejected for the eigenproblem).

Expressions form a tree: leaves are field references (FieldRef) and
bare scalar values (_BackgroundScalarValue); internal nodes are
operators (Sum, Grad, Cross, ...).

Algebraic nodes are built via Python operator overloading on the
Expression base class:

    a + b        Sum
    a - b        Sum(a, Negation(b))
    -a           Negation
    c * a, a * c ScalarMultiple (c is a Python/SymPy scalar or a scalar
                 Expression; a background scalar factor is always the
                 coefficient, whichever side it is written on)
    a / c        ScalarMultiple(1/c, a) for a background scalar c: a bare
                 Python/SymPy scalar or a background scalar expression

Differential and tensor operators are built via the factories in
operators.py (Grad, Div, Curl, Dot, Cross, Lap, LapH, Dt, Transpose).

Conventions
-----------
- Expression nodes do not compute components; the IR tracks only rank
  and operator structure. Spin components are produced by the tensor
  reducer and the backends.
- Time enters via Dt(...). At assembly time a Dt-wrapped unknown
  contributes to the eigenvalue-side matrix, everything else to the
  operator-side matrix.

Every constructor checks rank compatibility, so e.g. Sum(scalar, vector)
raises TypeError.
"""

from __future__ import annotations

from typing import Dict, Tuple
import sympy as sp

from spdee.ir.backgrounds import (
    BackgroundComponent, SymPyComponent, multiply_components,
    reciprocal_component,
)
from spdee.ir.fields import FieldRole, ScalarField, _FieldBase


# =============================================================================
# Linearity tracking
# =============================================================================

class Linearity:
    """How an Expression depends on unknown fields.

    BACKGROUND: contains no unknowns.
    LINEAR:     linear in the unknowns.
    NONLINEAR:  contains a product of two or more unknowns; rejected.
    """
    BACKGROUND = "background"
    LINEAR = "linear"
    NONLINEAR = "nonlinear"

    @staticmethod
    def combine_product(a: str, b: str) -> str:
        """Linearity of (a * b): background acts as identity; linear times
        linear is nonlinear."""
        if a == Linearity.NONLINEAR or b == Linearity.NONLINEAR:
            return Linearity.NONLINEAR
        if a == Linearity.BACKGROUND:
            return b
        if b == Linearity.BACKGROUND:
            return a
        return Linearity.NONLINEAR

    @staticmethod
    def combine_sum(a: str, b: str) -> str:
        """Linearity of (a + b): background only if both are background."""
        if a == Linearity.NONLINEAR or b == Linearity.NONLINEAR:
            return Linearity.NONLINEAR
        if a == Linearity.BACKGROUND and b == Linearity.BACKGROUND:
            return Linearity.BACKGROUND
        return Linearity.LINEAR


# =============================================================================
# Base Expression class
# =============================================================================

class Expression:
    """Base class for IR expression tree nodes.

    Subclasses set:
      - self.kind        ('scalar' / 'vector' / 'tensor')
      - self.linearity   (Linearity.BACKGROUND / LINEAR / NONLINEAR)
      - self.children    (tuple of child Expressions, possibly empty)
    """

    kind: str
    linearity: str
    children: Tuple["Expression", ...]

    @property
    def is_scalar(self) -> bool:
        return self.kind == "scalar"

    @property
    def is_vector(self) -> bool:
        return self.kind == "vector"

    @property
    def is_tensor(self) -> bool:
        return self.kind == "tensor"

    @property
    def is_background(self) -> bool:
        return self.linearity == Linearity.BACKGROUND

    @property
    def is_linear(self) -> bool:
        return self.linearity == Linearity.LINEAR

    @property
    def is_nonlinear(self) -> bool:
        return self.linearity == Linearity.NONLINEAR

    def __add__(self, other) -> "Expression":
        return Sum(self, _wrap(other))

    def __radd__(self, other) -> "Expression":
        return Sum(_wrap(other), self)

    def __sub__(self, other) -> "Expression":
        return Sum(self, Negation(_wrap(other)))

    def __rsub__(self, other) -> "Expression":
        return Sum(_wrap(other), Negation(self))

    def __neg__(self) -> "Expression":
        return Negation(self)

    def __mul__(self, other) -> "Expression":
        return _make_product(self, _wrap(other))

    def __rmul__(self, other) -> "Expression":
        return _make_product(_wrap(other), self)

    def __truediv__(self, other) -> "Expression":
        """Division by a background scalar (a bare Python/SymPy scalar or
        a background scalar expression); represented as
        (1/other) * self with the reciprocal a background scalar leaf."""
        other = _wrap(other)
        if not other.is_scalar:
            raise TypeError(
                f"Cannot divide by a {other.kind}; only scalar division is "
                f"supported"
            )
        if not other.is_background:
            raise TypeError(
                "Cannot divide by an expression involving unknowns; the IR "
                "is for linear PDEs"
            )
        return _make_product(self, _reciprocal(other))

    def __rtruediv__(self, other) -> "Expression":
        return _wrap(other) / self

    def as_symmetric(self) -> "Expression":
        """Mark a tensor expression as symmetric.

        The tensor reducer verifies symbolically that the (sigma_1,
        sigma_2) and (sigma_2, sigma_1) components agree for every
        sigma_1 != sigma_2 and raises ValueError otherwise.

        This is the manual override for tensors whose symmetry the
        structural detector does not recognize. `X + Transpose(X)`,
        `Transpose(sym)`, `Sum(sym, sym)`, `ScalarMultiple(c, sym)` and
        `Negation(sym)` are detected automatically.

        Raises TypeError on a non-tensor expression.
        """
        if not self.is_tensor:
            raise TypeError(
                f"as_symmetric() requires a tensor expression; "
                f"got {self.kind}"
            )
        return AsSymmetric(self)


# =============================================================================
# Leaf nodes: field references
# =============================================================================

class FieldRef(Expression):
    """Leaf node wrapping a single field.

    The field's rank sets the kind; the field's role sets the linearity.
    """
    __slots__ = ("field",)

    def __init__(self, field_obj: _FieldBase):
        self.field = field_obj
        if field_obj.rank == 0:
            self.kind = "scalar"
        elif field_obj.rank == 1:
            self.kind = "vector"
        elif field_obj.rank == 2:
            self.kind = "tensor"
        else:
            raise ValueError(f"Unsupported field rank {field_obj.rank}")
        self.linearity = (
            Linearity.BACKGROUND if field_obj.is_background else Linearity.LINEAR
        )
        self.children = ()

    def __repr__(self) -> str:
        return f"FieldRef({self.field.name!r})"


def _wrap(x) -> Expression:
    """Promote a bare field or a Python/SymPy scalar to an Expression."""
    if isinstance(x, Expression):
        return x
    if isinstance(x, _FieldBase):
        return FieldRef(x)
    if isinstance(x, (int, float, complex)) or isinstance(x, sp.Expr):
        return _BackgroundScalarValue(sp.sympify(x))
    raise TypeError(
        f"Cannot use {type(x).__name__} {x!r} in a spdee expression; "
        f"must be a Field, an Expression, or a scalar (int/float/complex/SymPy)"
    )


# =============================================================================
# Algebraic nodes
# =============================================================================

class _BackgroundScalarValue(Expression):
    """A bare scalar value: a Python number or a SymPy expression in
    (r, theta). Represents coefficients such as the `2` in `2*v`."""
    __slots__ = ("value",)

    def __init__(self, value: sp.Expr):
        self.value = sp.sympify(value)
        self.kind = "scalar"
        self.linearity = Linearity.BACKGROUND
        self.children = ()

    def __repr__(self) -> str:
        return f"BgScalar({self.value!r})"


def _extract_background_scalar(expr: Expression) -> sp.Expr:
    """SymPy value of a _BackgroundScalarValue; raises for any other node."""
    if not isinstance(expr, _BackgroundScalarValue):
        raise TypeError("Expected a background scalar value")
    return expr.value


def background_scalar_component(expr: Expression) -> BackgroundComponent:
    """BackgroundComponent of a background scalar expression: a bare
    value, a background ScalarField reference, or a Sum, Negation or
    ScalarMultiple of such. Anything else raises NotImplementedError."""
    if isinstance(expr, _BackgroundScalarValue):
        return SymPyComponent(expr.value)
    if isinstance(expr, FieldRef):
        field = expr.field
        if not field.is_background:
            raise NotImplementedError(
                f"Cannot use unknown field {field.name!r} as a coefficient"
            )
        if getattr(field, "component", None) is None:
            raise NotImplementedError(
                f"Field {field.name!r} has no component"
            )
        return field.component
    if isinstance(expr, ScalarMultiple):
        return multiply_components(
            background_scalar_component(expr.scalar),
            background_scalar_component(expr.operand),
        )
    if isinstance(expr, Negation):
        return multiply_components(
            SymPyComponent(sp.S.NegativeOne),
            background_scalar_component(expr.operand),
        )
    if isinstance(expr, Sum):
        left = background_scalar_component(expr.a)
        right = background_scalar_component(expr.b)
        if isinstance(left, SymPyComponent) and isinstance(right, SymPyComponent):
            return SymPyComponent(left.expr + right.expr)
        raise NotImplementedError(
            "Sum of numeric background scalar components is not supported "
            "as a coefficient"
        )
    raise NotImplementedError(
        f"Cannot extract coefficient from {type(expr).__name__}: {expr!r}"
    )


# Reciprocal fields of numeric background scalar expressions, keyed by
# expression structure and field identity so that every division by the
# same expression yields the same field object.
_RECIPROCAL_FIELDS: Dict[tuple, ScalarField] = {}


def _structure_key(expr: Expression) -> tuple:
    if isinstance(expr, FieldRef):
        return ("field", id(expr.field), expr.field)
    if isinstance(expr, _BackgroundScalarValue):
        return ("value", expr.value)
    return (type(expr).__name__,) + tuple(_structure_key(c) for c in expr.children)


def _label(expr: Expression) -> str:
    if isinstance(expr, FieldRef):
        return expr.field.name
    if isinstance(expr, _BackgroundScalarValue):
        return str(expr.value)
    if isinstance(expr, Sum):
        return f"{_label(expr.a)} + {_label(expr.b)}"
    if isinstance(expr, Negation):
        return f"-({_label(expr.operand)})"
    return f"({_label(expr.scalar)})*({_label(expr.operand)})"


def _reciprocal(expr: Expression) -> Expression:
    """1/expr for a background scalar expression as a background scalar
    leaf: a bare value when the reciprocal is SymPy-backed, else a
    reference to a background ScalarField with a reciprocal component."""
    if isinstance(expr, _BackgroundScalarValue):
        return _BackgroundScalarValue(1 / expr.value)
    component = background_scalar_component(expr)
    if isinstance(component, SymPyComponent):
        return _BackgroundScalarValue(1 / component.expr)
    key = _structure_key(expr)
    if key not in _RECIPROCAL_FIELDS:
        _RECIPROCAL_FIELDS[key] = ScalarField(
            name=f"1/({_label(expr)})", role=FieldRole.BACKGROUND,
            expr=reciprocal_component(component),
        )
    return FieldRef(_RECIPROCAL_FIELDS[key])


class Sum(Expression):
    """Sum of two Expressions of the same kind."""
    __slots__ = ("a", "b")

    def __init__(self, a: Expression, b: Expression):
        if a.kind != b.kind:
            raise TypeError(
                f"Cannot add {a.kind} and {b.kind}; both arguments must "
                f"have the same rank"
            )
        self.a = a
        self.b = b
        self.kind = a.kind
        self.linearity = Linearity.combine_sum(a.linearity, b.linearity)
        self.children = (a, b)
        if self.is_nonlinear:
            raise TypeError(
                "Sum produces a nonlinear expression; the IR is for linear "
                "PDEs"
            )

    def __repr__(self) -> str:
        return f"Sum({self.a!r}, {self.b!r})"


class Negation(Expression):
    """Unary minus; preserves kind and linearity."""
    __slots__ = ("operand",)

    def __init__(self, operand: Expression):
        self.operand = operand
        self.kind = operand.kind
        self.linearity = operand.linearity
        self.children = (operand,)

    def __repr__(self) -> str:
        return f"Negation({self.operand!r})"


class AsSymmetric(Expression):
    """Wrapper marking a tensor expression as symmetric.

    Built by Expression.as_symmetric(). The tensor reducer verifies the
    symmetry symbolically when it reduces this node.
    """
    __slots__ = ("operand",)

    def __init__(self, operand: Expression):
        if not operand.is_tensor:
            raise TypeError(
                f"AsSymmetric requires a tensor operand; got {operand.kind}"
            )
        self.operand = operand
        self.kind = "tensor"
        self.linearity = operand.linearity
        self.children = (operand,)

    def __repr__(self) -> str:
        return f"AsSymmetric({self.operand!r})"


class ScalarMultiple(Expression):
    """Scalar Expression times an Expression of any kind.

    The result has the operand's kind and the product linearity.
    """
    __slots__ = ("scalar", "operand")

    def __init__(self, scalar: Expression, operand: Expression):
        if not scalar.is_scalar:
            raise TypeError(
                f"ScalarMultiple requires a scalar coefficient; got {scalar.kind}"
            )
        self.scalar = scalar
        self.operand = operand
        self.kind = operand.kind
        self.linearity = Linearity.combine_product(scalar.linearity, operand.linearity)
        self.children = (scalar, operand)
        if self.is_nonlinear:
            raise TypeError(
                "ScalarMultiple produces a nonlinear expression "
                "(product of two non-background terms); the IR is for "
                "linear PDEs"
            )

    def __repr__(self) -> str:
        return f"ScalarMultiple({self.scalar!r}, {self.operand!r})"


def _make_product(a: Expression, b: Expression) -> Expression:
    """Dispatch a * b on ranks.

    - scalar * scalar      -> ScalarMultiple with the background factor
                              as the coefficient (p * 2 is 2 * p)
    - scalar * anything    -> ScalarMultiple(a, b)
    - anything * scalar    -> ScalarMultiple(b, a)
    - vector/tensor pairs  -> error (use Dot or Cross explicitly)
    """
    if a.is_scalar and b.is_scalar and b.is_background and not a.is_background:
        return ScalarMultiple(b, a)
    if a.is_scalar:
        return ScalarMultiple(a, b)
    if b.is_scalar:
        return ScalarMultiple(b, a)
    raise TypeError(
        f"Cannot use '*' for {a.kind} * {b.kind}; use Dot(...) or Cross(...) "
        f"to disambiguate the operation"
    )
