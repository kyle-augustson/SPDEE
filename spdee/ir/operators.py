"""spdee.ir.operators — symbolic differential and tensor operators.

These build Expression tree nodes; evaluation is the backend's job.

Rank arithmetic:

    Grad(scalar)   -> vector
    Grad(vector)   -> tensor
    Grad(tensor)   -> error (rank 3 not supported)
    Div(vector)    -> scalar
    Div(tensor)    -> vector
    Div(scalar)    -> error
    Curl(vector)   -> vector
    Curl(other)    -> error
    Lap(scalar)    -> scalar
    Lap(vector)    -> vector
    Lap(tensor)    -> error
    LapH(scalar)   -> scalar
    LapH(other)    -> error
    Cross(v, w)    -> vector  (both vectors)
    Dot(v, w)      -> scalar  (both vectors)
    Dot(T, v)      -> vector
    Dot(v, T)      -> vector
    Dot(T, S)      -> tensor  (both rank 2)
    Transpose(T)   -> tensor
    Dt(anything)   -> same rank

Linearity rules:

  Differential operators, Transpose and Dt preserve linearity.
  Cross and Dot follow Linearity.combine_product; a nonlinear result
  (two unknown operands) raises TypeError.
"""

from __future__ import annotations

from spdee.ir.expressions import (
    Expression, Linearity, _wrap,
)


# =============================================================================
# Differential operators
# =============================================================================

class Grad(Expression):
    """Gradient of a scalar (-> vector) or vector (-> tensor) field."""
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if operand.is_scalar:
            result_kind = "vector"
        elif operand.is_vector:
            result_kind = "tensor"
        else:
            raise TypeError(
                f"Grad requires a scalar or vector argument; got {operand.kind}"
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = result_kind
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Grad({self.operand!r})"


class Div(Expression):
    """Divergence of a vector (-> scalar) or rank-2 tensor (-> vector)."""
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if operand.is_vector:
            result_kind = "scalar"
        elif operand.is_tensor:
            result_kind = "vector"
        else:
            raise TypeError(
                f"Div requires a vector or tensor argument; got {operand.kind}"
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = result_kind
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Div({self.operand!r})"


class Curl(Expression):
    """Curl of a vector field."""
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if not operand.is_vector:
            raise TypeError(
                f"Curl requires a vector argument; got {operand.kind}"
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = "vector"
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Curl({self.operand!r})"


class Lap(Expression):
    """Laplacian of a scalar or vector field."""
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if operand.is_scalar:
            result_kind = "scalar"
        elif operand.is_vector:
            result_kind = "vector"
        else:
            raise TypeError(
                f"Lap requires a scalar or vector argument; got {operand.kind}"
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = result_kind
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Lap({self.operand!r})"


class LapH(Expression):
    """Horizontal (angular) Laplacian of a scalar field.

    In spherical coordinates:
        ∇_H² f = (1/(r² sin θ)) ∂_θ(sin θ ∂_θ f)
                + (1/(r² sin² θ)) ∂²_φ f

    At fixed azimuthal mode m, ∂²_φ -> -m², so
        r² ∇_H² f = ∂²_θ f + cot θ · ∂_θ f - m²/sin²θ · f,
    the angular part of r² times the scalar Laplacian (natural
    r-power 2, matching Lap).

    Used by toroidal-scalar formulations of inertial-wave / Stokes-flow
    problems and by vorticity-streamfunction formulations.

    Scalar operand only; a vector angular Laplacian would need a
    component-specific convention and is not defined here.
    """
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if not operand.is_scalar:
            raise TypeError(
                f"LapH requires a scalar argument; got {operand.kind}. "
                f"Vector LapH isn't defined here — vectors require a "
                f"component-specific convention."
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = "scalar"
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"LapH({self.operand!r})"


class Dt(Expression):
    """Time-derivative marker.

    Dt(expr) puts the unknowns in expr on the eigenvalue side of the
    problem: at assembly time the eigenvalue multiplies these terms.
    Preserves rank and linearity.
    """
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if operand.is_nonlinear:
            raise TypeError("Cannot take Dt of a nonlinear expression")
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = operand.kind
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Dt({self.operand!r})"


# =============================================================================
# Tensor algebra
# =============================================================================

class Cross(Expression):
    """Cross product of two vectors."""
    __slots__ = ("a", "b")

    def __new__(cls, a, b):
        a, b = _wrap(a), _wrap(b)
        if not (a.is_vector and b.is_vector):
            raise TypeError(
                f"Cross requires two vector arguments; got {a.kind} and {b.kind}"
            )
        linearity = Linearity.combine_product(a.linearity, b.linearity)
        if linearity == Linearity.NONLINEAR:
            raise TypeError(
                "Cross(unknown, unknown) is nonlinear; the IR is for "
                "linear PDEs"
            )
        inst = super().__new__(cls)
        inst.a = a
        inst.b = b
        inst.kind = "vector"
        inst.linearity = linearity
        inst.children = (a, b)
        return inst

    def __repr__(self) -> str:
        return f"Cross({self.a!r}, {self.b!r})"


class Dot(Expression):
    """Dot product / single-index contraction.

    Dot(A, B) contracts the last index of A with the first index of B:

    - Dot(vector, vector) -> scalar
    - Dot(tensor, vector) -> vector
    - Dot(vector, tensor) -> vector
    - Dot(tensor, tensor) -> tensor
    """
    __slots__ = ("a", "b")

    def __new__(cls, a, b):
        a, b = _wrap(a), _wrap(b)
        if a.is_vector and b.is_vector:
            kind = "scalar"
        elif a.is_tensor and b.is_vector:
            kind = "vector"
        elif a.is_vector and b.is_tensor:
            kind = "vector"
        elif a.is_tensor and b.is_tensor:
            kind = "tensor"
        else:
            raise TypeError(
                f"Dot requires (vector,vector), (tensor,vector), "
                f"(vector,tensor), or (tensor,tensor); got "
                f"({a.kind}, {b.kind})"
            )
        linearity = Linearity.combine_product(a.linearity, b.linearity)
        if linearity == Linearity.NONLINEAR:
            raise TypeError(
                "Dot(unknown, unknown) is nonlinear; the IR is for "
                "linear PDEs"
            )
        inst = super().__new__(cls)
        inst.a = a
        inst.b = b
        inst.kind = kind
        inst.linearity = linearity
        inst.children = (a, b)
        return inst

    def __repr__(self) -> str:
        return f"Dot({self.a!r}, {self.b!r})"


class Transpose(Expression):
    """Transpose of a rank-2 tensor: (T^T)_{ij} = T_{ji}.

    In the spin basis the transpose swaps the spin index pair,
    (T^T)_{sigma_1, sigma_2} = T_{sigma_2, sigma_1}. Transpose is its
    own inverse and preserves kind and linearity.
    """
    __slots__ = ("operand",)

    def __new__(cls, operand):
        operand = _wrap(operand)
        if not operand.is_tensor:
            raise TypeError(
                f"Transpose requires a tensor argument; got {operand.kind}"
            )
        inst = super().__new__(cls)
        inst.operand = operand
        inst.kind = "tensor"
        inst.linearity = operand.linearity
        inst.children = (operand,)
        return inst

    def __repr__(self) -> str:
        return f"Transpose({self.operand!r})"
