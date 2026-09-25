"""spdee — Spherical Partial Differential Equation Eigensolver.

A modular eigensolver for linear PDEs in spherical geometry, separable in
the azimuthal coordinate phi so that each Fourier mode m yields an
independent generalized eigenvalue problem L x = lambda R x.

Quickstart
----------

    >>> import sympy as sp
    >>> from spdee import (
    ...     Background, Unknown, r, theta,
    ...     Grad, Div, Curl, Cross, Dot, Lap, Dt,
    ...     Equation, BoundaryCondition, EquationSystem, Boundary,
    ...     r_hat,
    ...     compile_problem,
    ... )
    >>>
    >>> # Define fields
    >>> rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    >>> B0 = Background.Vector("B0", v_phi=r*sp.sin(theta))
    >>> v = Unknown.Vector("v")
    >>> p = Unknown.Scalar("p")
    >>>
    >>> # Write equations
    >>> momentum = Equation(
    ...     rho0 * Dt(v) + Grad(p) - Cross(B0, Curl(v)),
    ...     name="momentum",
    ... )
    >>>
    >>> # BCs
    >>> bc = BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="no-penetration")
    >>>
    >>> # Assemble system, dispatch to a backend
    >>> system = EquationSystem([momentum], [bc])
    >>> # compiled = compile_problem(system, backend='fd2', ...)

Architecture
------------
- `spdee.ir` — backend-agnostic intermediate representation.
- `spdee.backends.fd2` — explicit centered finite differences in r and
  theta, collocated or on the staggered Arakawa-C grid.
- `spdee.backends.sh_fd` — spin-weighted spherical harmonics in
  (theta, phi), finite differences in r.
- `spdee.backends.dedalus` — hand-off to Dedalus.
- `spdee.solvers` — generalized eigenvalue solvers.
"""

__version__ = "0.1.0"

from spdee.ir import (
    ScalarField, VectorField, TensorField,
    Background, Unknown,
    Grad, Div, Curl, Cross, Dot, Lap, LapH, Dt, Transpose,
    Equation, BoundaryCondition, EquationSystem,
    ImpenetrableStressFree,
    r, theta,
)
from spdee.ir.equations import Boundary, r_hat, theta_hat, phi_hat
from spdee.ir.backgrounds import (
    BackgroundComponent, SymPyComponent, NumericComponent, CompositeComponent,
    as_background_component, multiply_components,
    radial_interpolator, meridional_interpolator,
)
from spdee.backends import compile_problem, available_backends
from spdee.problem import Problem, parse_expression

__all__ = [
    # field types
    "ScalarField", "VectorField", "TensorField",
    "Background", "Unknown",
    # background components
    "BackgroundComponent", "SymPyComponent", "NumericComponent",
    "CompositeComponent", "as_background_component", "multiply_components",
    "radial_interpolator", "meridional_interpolator",
    # coordinates
    "r", "theta",
    # operators
    "Grad", "Div", "Curl", "Cross", "Dot", "Lap", "LapH", "Dt", "Transpose",
    # equation system
    "Equation", "BoundaryCondition", "EquationSystem", "Boundary",
    "ImpenetrableStressFree",
    # convenience unit vectors
    "r_hat", "theta_hat", "phi_hat",
    # backend dispatch
    "compile_problem", "available_backends",
    # string entry
    "Problem", "parse_expression",
]
