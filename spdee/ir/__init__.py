"""spdee.ir — backend-agnostic intermediate representation."""

from spdee.ir.fields import (
    ScalarField, VectorField, TensorField,
    Background, Unknown, FieldRole,
    r, theta,
)
from spdee.ir.expressions import (
    Expression, FieldRef, Linearity, AsSymmetric,
)
from spdee.ir.operators import (
    Grad, Div, Curl, Cross, Dot, Lap, LapH, Dt, Transpose,
)
from spdee.ir.equations import (
    Equation, BoundaryCondition, EquationSystem,
    ImpenetrableStressFree,
)

__all__ = [
    "ScalarField", "VectorField", "TensorField",
    "Background", "Unknown", "FieldRole",
    "r", "theta",
    "Expression", "FieldRef", "Linearity", "AsSymmetric",
    "Grad", "Div", "Curl", "Cross", "Dot", "Lap", "LapH", "Dt", "Transpose",
    "Equation", "BoundaryCondition", "EquationSystem",
    "ImpenetrableStressFree",
]
