"""spdee.ir.equations — equations, boundary conditions, equation systems.

An Equation wraps an Expression and asserts it equals zero; its rank
(scalar/vector/tensor) is the expression's kind.

A BoundaryCondition is a location plus a scalar Expression that must
vanish there. Locations are r=0, r=1, theta=0, theta=pi. How the
condition is enforced (Dirichlet replacement, tau, ...) is the
backend's choice; the IR specifies only what to impose and where. A
vector component is targeted with a unit-vector Dot product, e.g.
Dot(r_hat, v).

An EquationSystem collects equations and BCs and reports which
background and unknown fields appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Tuple

import sympy as sp

from spdee.ir.fields import _FieldBase, Background, r as _R_SYM
from spdee.ir.expressions import Expression, FieldRef, _wrap


# -----------------------------------------------------------------------------
# Boundary locations
# -----------------------------------------------------------------------------

class Boundary:
    """Boundary location markers, e.g. BoundaryCondition(Boundary.R_OUTER, expr)."""
    R_INNER = "r=0"       # inner radial boundary of a shell
    R_ORIGIN = "r=0"      # alias for ball mode (same string)
    R_OUTER = "r=1"
    THETA_NORTH = "theta=0"
    THETA_SOUTH = "theta=pi"

    _VALID = frozenset({"r=0", "r=1", "theta=0", "theta=pi"})


# -----------------------------------------------------------------------------
# Unit vectors
# -----------------------------------------------------------------------------

# Physical unit vectors as background fields, for projecting vectors
# onto components in BCs and equations.
r_hat = Background.Vector("r_hat", v_r=1)
theta_hat = Background.Vector("theta_hat", v_theta=1)
phi_hat = Background.Vector("phi_hat", v_phi=1)


# -----------------------------------------------------------------------------
# Equation
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Equation:
    """A bulk PDE: lhs == 0.

    The equation's kind is the lhs kind: a scalar equation is one row
    block, a vector equation three (one per spin component). Bare
    fields and scalars are promoted to Expressions.

    ``weight`` (optional) is a radial function w(r), a SymPy expression in
    r, by which a projecting core multiplies the equation before adding its
    tau terms: the Dedalus core solves w lhs + taus = 0.  Pointwise cores
    (fd2, sh_fd, collocated) ignore it, since a row scaling by a function of
    r leaves their eigenvalues unchanged; on a tau core it changes the
    discrete problem (which residual the taus absorb), as in the reference
    implementation, whose momentum, continuity and entropy
    equations are multiplied by r^2 H_rho, H_rho and H_p.  The Dedalus core
    folds w into each term's background coefficient (w dlnrho as one NCC),
    so a weight that makes the coefficients polynomial removes their
    ncc_cutoff truncation.  Rebuild an equation with
    ``dataclasses.replace(eq, lhs=...)`` to keep it.
    """
    lhs: Expression
    name: str = ""
    weight: Any = None

    def __post_init__(self):
        if not isinstance(self.lhs, Expression):
            try:
                wrapped = _wrap(self.lhs)
            except TypeError as exc:
                raise TypeError(
                    f"Equation lhs must be an Expression, Field, or scalar; "
                    f"got {type(self.lhs).__name__}"
                ) from exc
            object.__setattr__(self, "lhs", wrapped)
        if self.weight is not None:
            w = sp.sympify(self.weight)
            if w.free_symbols - {_R_SYM}:
                raise ValueError(f"Equation weight {w} must be a function of r only")
            object.__setattr__(self, "weight", w)
        if self.lhs.is_nonlinear:
            raise TypeError(
                "Equation lhs is nonlinear; spdee only supports linear PDEs"
            )

    @property
    def kind(self) -> str:
        return self.lhs.kind

    @property
    def is_linear(self) -> bool:
        return self.lhs.is_linear

    def __repr__(self) -> str:
        if self.weight is not None:
            return f"Equation({self.name!r}: ({self.weight}) * ({self.lhs!r}) == 0)"
        if self.name:
            return f"Equation({self.name!r}: {self.lhs!r} == 0)"
        return f"Equation({self.lhs!r} == 0)"


# -----------------------------------------------------------------------------
# Boundary condition
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryCondition:
    """A single scalar condition `expr == 0` at a boundary location.

    Parameters
    ----------
    location : str
        One of Boundary.R_INNER, R_OUTER, THETA_NORTH, THETA_SOUTH.
    expr : Expression
        A scalar Expression. Vector conditions are written as several
        scalar BCs (e.g. one per unit-vector projection).
    name : str
        Optional label.
    """
    location: str
    expr: Expression
    name: str = ""

    def __post_init__(self):
        if self.location not in Boundary._VALID:
            raise ValueError(
                f"Unknown boundary location {self.location!r}; "
                f"valid choices: {sorted(Boundary._VALID)}"
            )
        if not isinstance(self.expr, Expression):
            try:
                wrapped = _wrap(self.expr)
            except TypeError as exc:
                raise TypeError(
                    f"BC expr must be an Expression, Field, or scalar; "
                    f"got {type(self.expr).__name__}"
                ) from exc
            object.__setattr__(self, "expr", wrapped)
        if not self.expr.is_scalar:
            raise TypeError(
                f"BC expr must be a scalar Expression; got {self.expr.kind}. "
                f"Decompose vector/tensor BCs into separate scalar BCs "
                f"(e.g., using Dot(r_hat, v))."
            )
        if self.expr.is_nonlinear:
            raise TypeError(
                "BC expression is nonlinear; spdee only supports linear BCs"
            )

    def __repr__(self) -> str:
        loc = self.location
        if self.name:
            return f"BC({self.name!r} at {loc}: {self.expr!r} == 0)"
        return f"BC(at {loc}: {self.expr!r} == 0)"


# -----------------------------------------------------------------------------
# EquationSystem
# -----------------------------------------------------------------------------

class EquationSystem:
    """A linear PDE eigenproblem: bulk equations, boundary conditions,
    and the rosters of background and unknown fields they reference.

    Construction checks that every equation is an Equation with a name
    unique in the system (the backends key row blocks by equation name),
    every BC a BoundaryCondition, and that `gauge_unknowns` names scalar
    unknowns of the system. Well-posedness (equation/unknown counts, BC
    sufficiency) is checked by the backend at assembly.
    """

    def __init__(
        self,
        equations: List[Equation],
        boundary_conditions: List[BoundaryCondition] = None,
        gauge_unknowns: List[str] = None,
    ):
        self.equations: Tuple[Equation, ...] = tuple(equations)
        self.boundary_conditions: Tuple[BoundaryCondition, ...] = tuple(
            boundary_conditions or []
        )
        # Scalar unknowns whose volume integral is constrained to zero.
        # These are pressure-like unknowns that appear only through their
        # gradient, so the global constant is a null mode; the backend
        # replaces one row of the unknown's constraint equation with the
        # discrete integral row.
        self.gauge_unknowns: Tuple[str, ...] = tuple(gauge_unknowns or ())
        names = set()
        for eq in self.equations:
            if not isinstance(eq, Equation):
                raise TypeError(
                    f"Equation list must contain Equation objects; got "
                    f"{type(eq).__name__}"
                )
            if eq.name and eq.name in names:
                raise ValueError(
                    f"Duplicate equation name {eq.name!r}; equation names "
                    f"must be unique within a system"
                )
            names.add(eq.name)
        for bc in self.boundary_conditions:
            if not isinstance(bc, BoundaryCondition):
                raise TypeError(
                    f"BC list must contain BoundaryCondition objects; got "
                    f"{type(bc).__name__}"
                )
        self._backgrounds, self._unknowns = self._collect_fields()
        for gname in self.gauge_unknowns:
            if gname not in self._unknowns:
                raise ValueError(
                    f"gauge_unknowns references {gname!r}, which is not an "
                    f"unknown in this system. Known unknowns: "
                    f"{sorted(self._unknowns)}"
                )
            if self._unknowns[gname].kind != "scalar":
                raise ValueError(
                    f"gauge_unknowns only supports scalar unknowns; "
                    f"{gname!r} is {self._unknowns[gname].kind}"
                )

    def _collect_fields(self) -> Tuple[
        Dict[str, _FieldBase], Dict[str, _FieldBase]
    ]:
        """Record every field referenced by the equations and BCs, by
        role. Two distinct field objects with the same name raise."""
        backgrounds: Dict[str, _FieldBase] = {}
        unknowns: Dict[str, _FieldBase] = {}

        def collect(expr: Expression):
            if isinstance(expr, FieldRef):
                fld = expr.field
                bucket = backgrounds if fld.is_background else unknowns
                if fld.name in bucket and bucket[fld.name] is not fld:
                    raise ValueError(
                        f"Two distinct fields share the name {fld.name!r}; "
                        f"give each a unique name"
                    )
                bucket[fld.name] = fld
            for child in expr.children:
                collect(child)

        for eq in self.equations:
            collect(eq.lhs)
        for bc in self.boundary_conditions:
            collect(bc.expr)
        return backgrounds, unknowns

    @property
    def backgrounds(self) -> Dict[str, _FieldBase]:
        """Mapping name -> background field."""
        return dict(self._backgrounds)

    @property
    def unknowns(self) -> Dict[str, _FieldBase]:
        """Mapping name -> unknown field."""
        return dict(self._unknowns)

    @property
    def unknown_names(self) -> FrozenSet[str]:
        return frozenset(self._unknowns.keys())

    @property
    def background_names(self) -> FrozenSet[str]:
        return frozenset(self._backgrounds.keys())

    def __repr__(self) -> str:
        return (
            f"EquationSystem({len(self.equations)} equations, "
            f"{len(self.boundary_conditions)} BCs, "
            f"{len(self._unknowns)} unknowns)"
        )

    def summary(self) -> str:
        """Multi-line human-readable description."""
        lines = ["EquationSystem:"]
        lines.append(f"  Unknowns ({len(self._unknowns)}):")
        for name, fld in sorted(self._unknowns.items()):
            lines.append(f"    {fld!r}")
        lines.append(f"  Backgrounds ({len(self._backgrounds)}):")
        for name, fld in sorted(self._backgrounds.items()):
            lines.append(f"    {fld!r}")
        lines.append(f"  Equations ({len(self.equations)}):")
        for eq in self.equations:
            lines.append(f"    {eq!r}")
        lines.append(f"  BCs ({len(self.boundary_conditions)}):")
        for bc in self.boundary_conditions:
            lines.append(f"    {bc!r}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Convenience BC constructors
# -----------------------------------------------------------------------------

def ImpenetrableStressFree(
    v,
    boundary: str = Boundary.R_OUTER,
    *,
    name_prefix: str = "stress_free",
) -> List[BoundaryCondition]:
    """The three boundary conditions of an impenetrable, stress-free wall
    on a vector unknown ``v`` at a radial boundary:

      1. Impenetrability: ``Dot(r_hat, v) = 0``.
      2. Stress-free theta traction:
         ``Dot(theta_hat, Dot(strain, r_hat)) = 0`` with
         ``strain = Grad(v) + Transpose(Grad(v))``.
      3. Stress-free phi traction: ``Dot(phi_hat, Dot(strain, r_hat)) = 0``.

    The normal stress T_rr is left unconstrained: at an impenetrable
    boundary it couples to the pressure and adjusts on its own.

    Parameters
    ----------
    v : Expression or _FieldBase
        Unknown vector field (or vector expression).
    boundary : str
        Boundary.R_INNER or Boundary.R_OUTER (default).
    name_prefix : str
        The emitted names are ``"<prefix>_v_r"``, ``"<prefix>_theta"``,
        ``"<prefix>_phi"``.

    Examples
    --------
    >>> from spdee import Unknown, EquationSystem, Lap, Equation
    >>> v = Unknown.Vector("v")
    >>> bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
    >>> system = EquationSystem([Equation(Lap(v), name="vec_lap")], bcs)
    """
    from spdee.ir.operators import Grad, Dot, Transpose
    v_expr = _wrap(v)
    if not v_expr.is_vector:
        raise TypeError(
            f"ImpenetrableStressFree requires a vector unknown; "
            f"got {v_expr.kind}"
        )
    if boundary not in (Boundary.R_INNER, Boundary.R_OUTER):
        raise ValueError(
            f"ImpenetrableStressFree only supported at radial boundaries; "
            f"got {boundary!r}"
        )
    strain = Grad(v_expr) + Transpose(Grad(v_expr))
    return [
        BoundaryCondition(
            boundary,
            Dot(r_hat, v_expr),
            name=f"{name_prefix}_v_r",
        ),
        BoundaryCondition(
            boundary,
            Dot(theta_hat, Dot(strain, r_hat)),
            name=f"{name_prefix}_theta",
        ),
        BoundaryCondition(
            boundary,
            Dot(phi_hat, Dot(strain, r_hat)),
            name=f"{name_prefix}_phi",
        ),
    ]
