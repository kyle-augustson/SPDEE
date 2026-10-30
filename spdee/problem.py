"""String entry of equations.

    >>> pb = spdee.Problem()
    >>> v = pb.add_unknown("v", "vector")
    >>> p = pb.add_unknown("p", "scalar")
    >>> pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)")
    >>> pb.add_parameter("Ek", 1e-4)
    >>> pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0", name="momentum")
    >>> pb.add_equation("div(v) = 0", name="continuity")
    >>> pb.add_bc("dot(r_hat, v) = 0", at="outer")
    >>> system = pb.build(gauge_unknowns=["p"])

Strings are evaluated in a namespace holding the declared fields, the IR
operators under capitalized and lowercase names, the coordinates r and
theta, the unit vectors, the SymPy elementary functions and the declared
parameters.  ``lhs = rhs`` becomes ``Equation(lhs - rhs)``; a bare
expression means ``expr = 0``.  Symbolic parameters are bound at compile
time through ``params``, so one walk of the system serves a sweep.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import sympy as sp

from spdee.ir import (
    Background, Unknown,
    Grad, Div, Curl, Cross, Dot, Lap, LapH, Dt, Transpose,
    Equation, BoundaryCondition, EquationSystem, ImpenetrableStressFree,
    r, theta,
)
from spdee.ir.equations import Boundary, r_hat, theta_hat, phi_hat
from spdee.ir.expressions import Expression, _wrap

__all__ = ["Problem", "parse_expression", "DEFAULT_NAMESPACE", "boundary_location"]

_OPERATORS = {
    "Grad": Grad, "Div": Div, "Curl": Curl, "Cross": Cross, "Dot": Dot,
    "Lap": Lap, "LapH": LapH, "Dt": Dt, "Transpose": Transpose,
    "grad": Grad, "div": Div, "curl": Curl, "cross": Cross, "dot": Dot,
    "lap": Lap, "laph": LapH, "lap_h": LapH, "dt": Dt,
    "trans": Transpose, "transpose": Transpose,
}
_SYMPY_FUNCS = {
    name: getattr(sp, name) for name in (
        "sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh",
        "Abs", "sign", "Rational", "Integer", "Float", "Symbol", "I", "pi", "E",
    )
}
DEFAULT_NAMESPACE: Dict[str, Any] = {
    **_OPERATORS, **_SYMPY_FUNCS,
    "r": r, "theta": theta,
    "r_hat": r_hat, "theta_hat": theta_hat, "phi_hat": phi_hat,
    "Background": Background, "Unknown": Unknown,
    "ImpenetrableStressFree": ImpenetrableStressFree,
}
_BOUNDARY_ALIASES = {
    "inner": Boundary.R_INNER, "r_inner": Boundary.R_INNER, "bottom": Boundary.R_INNER,
    "origin": Boundary.R_ORIGIN, "r_origin": Boundary.R_ORIGIN, "r=0": Boundary.R_INNER,
    "outer": Boundary.R_OUTER, "r_outer": Boundary.R_OUTER, "top": Boundary.R_OUTER,
    "surface": Boundary.R_OUTER, "r=1": Boundary.R_OUTER,
    "north": Boundary.THETA_NORTH, "theta_north": Boundary.THETA_NORTH, "theta=0": Boundary.THETA_NORTH,
    "south": Boundary.THETA_SOUTH, "theta_south": Boundary.THETA_SOUTH, "theta=pi": Boundary.THETA_SOUTH,
}


def boundary_location(at: str) -> str:
    """Normalize a boundary alias to the IR location string."""
    key = str(at).strip().lower().replace(" ", "")
    if key in _BOUNDARY_ALIASES:
        return _BOUNDARY_ALIASES[key]
    if at in Boundary._VALID:
        return at
    raise ValueError(f"unknown boundary {at!r}; use one of {sorted(set(_BOUNDARY_ALIASES))} "
                     f"or a spdee.Boundary constant")


def _split_equals(text: str):
    """Split 'lhs = rhs' on the single '=' (not '==', '<=', ...)."""
    parts = re.split(r"(?<![=!<>])=(?!=)", text)
    if len(parts) == 1:
        return parts[0], None
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"expected at most one '=' in {text!r}")


def _eval(text: str, namespace: Dict[str, Any]) -> Any:
    try:
        return eval(text, {"__builtins__": {}}, namespace)  # noqa: S307
    except NameError as exc:
        missing = str(exc).split("'")[1] if "'" in str(exc) else "?"
        known = sorted(k for k in namespace if not k.startswith("_"))
        raise NameError(f"unknown name {missing!r} in {text!r}; declare it with add_unknown / "
                        f"add_background / add_parameter or pass it in the namespace. "
                        f"Known names: {known}") from exc
    except SyntaxError as exc:
        raise SyntaxError(f"cannot parse {text!r}: {exc.msg}") from exc


def _is_zero(x: Any) -> bool:
    if isinstance(x, Expression):
        return False
    try:
        return complex(sp.sympify(x)) == 0
    except (sp.SympifyError, TypeError, ValueError):
        return False


def parse_expression(text: str, namespace: Dict[str, Any]) -> Expression:
    """Parse ``"lhs = rhs"`` or a bare expression into the IR Expression ``lhs - rhs``."""
    lhs_text, rhs_text = _split_equals(text)
    lhs = _eval(lhs_text.strip(), namespace)
    rhs = None if rhs_text is None else _eval(rhs_text.strip(), namespace)
    expr = lhs if (rhs is None or _is_zero(rhs)) else lhs - rhs
    if not isinstance(expr, Expression):
        expr = _wrap(expr)
    return expr


class Problem:
    """A linear spherical eigenproblem entered as strings.

    ``namespace`` adds names visible to the strings (fields from the object
    API, numbers, SymPy expressions, callables); ``locals()`` works.
    """

    def __init__(self, namespace: Optional[Dict[str, Any]] = None):
        self.namespace: Dict[str, Any] = dict(DEFAULT_NAMESPACE)
        if namespace:
            self.namespace.update({k: v for k, v in namespace.items() if not k.startswith("__")})
        self.equations: List[Equation] = []
        self.boundary_conditions: List[BoundaryCondition] = []
        self.unknowns: Dict[str, Any] = {}
        self.backgrounds: Dict[str, Any] = {}
        self.params: Dict[sp.Symbol, Any] = {}
        self._system: Optional[EquationSystem] = None

    def _bind(self, name: str, obj: Any) -> Any:
        if name in DEFAULT_NAMESPACE:
            raise ValueError(f"{name!r} is a reserved name")
        self.namespace[name] = obj
        self._system = None
        return obj

    def add_unknown(self, name: str, kind: str = "scalar"):
        """Declare an unknown field and return it."""
        factory = {"scalar": Unknown.Scalar, "vector": Unknown.Vector,
                   "tensor": Unknown.Tensor}.get(kind.lower())
        if factory is None:
            raise ValueError(f"kind must be scalar/vector/tensor, got {kind!r}")
        f = factory(name)
        self.unknowns[name] = f
        return self._bind(name, f)

    def add_background(self, name: str, expr=None, *, v_r=None, v_theta=None,
                       v_phi=None, components=None, symmetric=False):
        """Declare a background field.

        ``expr`` (scalar) and the vector components may be SymPy expressions
        in r and theta, numbers, strings parsed in the problem namespace, or
        the numeric interpolators of ``spdee.ir.backgrounds``.
        """
        def conv(x):
            return _eval(x, self.namespace) if isinstance(x, str) else x

        if components is not None:
            f = Background.Tensor(name, {k: conv(v) for k, v in components.items()},
                                  symmetric=symmetric)
        elif any(x is not None for x in (v_r, v_theta, v_phi)):
            f = Background.Vector(name, v_r=conv(v_r), v_theta=conv(v_theta), v_phi=conv(v_phi))
        else:
            if expr is None:
                raise ValueError("a scalar background needs expr=")
            value = conv(expr)
            if isinstance(value, (str, int, float, complex, sp.Basic)):
                value = sp.sympify(value)
            f = Background.Scalar(name, expr=value)
        self.backgrounds[name] = f
        return self._bind(name, f)

    def add_parameter(self, name: str, value=None, symbolic: bool = True):
        """Declare a scalar parameter.

        Symbolic parameters are SymPy symbols whose value is bound at compile
        time (``self.params``); with ``symbolic=False`` the number itself is
        substituted at parse time.
        """
        if symbolic:
            sym = sp.Symbol(name)
            if value is not None:
                self.params[sym] = value
            return self._bind(name, sym)
        if value is None:
            raise ValueError("a non-symbolic parameter needs a value")
        return self._bind(name, sp.sympify(value))

    def set_parameter(self, name: str, value) -> None:
        """Rebind the compile-time value of a symbolic parameter."""
        sym = self.namespace.get(name)
        if not isinstance(sym, sp.Symbol):
            raise KeyError(f"{name!r} is not a symbolic parameter")
        self.params[sym] = value

    def add_equation(self, text: str, name: Optional[str] = None, weight=None) -> Equation:
        """Add ``text = 0``; ``weight`` is an optional radial function (text
        in r or a SymPy expression) multiplying it on tau cores (`Equation`)."""
        if isinstance(weight, str):
            weight = sp.sympify(weight, locals={"r": self.namespace["r"]})
        eq = Equation(parse_expression(text, self.namespace),
                      name=name or f"eq{len(self.equations) + 1}", weight=weight)
        self.equations.append(eq)
        self._system = None
        return eq

    def add_bc(self, text: str, at: str, name: Optional[str] = None) -> BoundaryCondition:
        """Add one scalar boundary condition at ``at`` ('inner', 'outer', ...)."""
        bc = BoundaryCondition(boundary_location(at), parse_expression(text, self.namespace),
                               name=name or f"bc{len(self.boundary_conditions) + 1}")
        self.boundary_conditions.append(bc)
        self._system = None
        return bc

    def add_bcs(self, bcs: Iterable[BoundaryCondition]) -> List[BoundaryCondition]:
        """Add boundary conditions built with the object API."""
        out = list(bcs)
        self.boundary_conditions.extend(out)
        self._system = None
        return out

    def add_stress_free(self, vector: str, at: str, name_prefix: str = "stress_free"):
        """Impenetrable stress-free wall on a vector unknown (three conditions)."""
        return self.add_bcs(ImpenetrableStressFree(self.namespace[vector],
                                                   boundary=boundary_location(at),
                                                   name_prefix=name_prefix))

    def build(self, gauge_unknowns: Sequence[str] = ()) -> EquationSystem:
        """The EquationSystem, cached until the problem changes."""
        if self._system is None or tuple(gauge_unknowns) != self._system.gauge_unknowns:
            self._system = EquationSystem(list(self.equations), list(self.boundary_conditions),
                                          gauge_unknowns=list(gauge_unknowns))
        return self._system

    def compile(self, grid, m: int, *, backend: str = "fd2",
                gauge_unknowns: Sequence[str] = (), params: Optional[dict] = None, **kwargs):
        """Compile through a backend, forwarding the declared parameters.

        ``backend='fd2'`` calls the collocated/staggered compiler directly
        (``discretization='staggered'`` selects the Arakawa-C path); other
        backends go through the registry.
        """
        system = self.build(gauge_unknowns)
        bound = dict(self.params)
        if params:
            bound.update(params)
        missing = [s for s in bound if bound[s] is None]
        if missing:
            raise ValueError(f"parameters without a value: {missing}")
        if backend == "fd2":
            from spdee.backends.fd2.compiled_problem import compile_problem as _cp
            return _cp(system, grid, m, params=bound or None, **kwargs)
        from spdee.backends import compile_problem as _dispatch
        return _dispatch(system, backend=backend, grid=grid, m=m, params=bound or None, **kwargs)

    def __repr__(self) -> str:
        eqs = "\n".join(f"  {e!r}" for e in self.equations)
        bcs = "\n".join(f"  {b!r}" for b in self.boundary_conditions)
        return (f"Problem(unknowns={sorted(self.unknowns)}, backgrounds={sorted(self.backgrounds)}, "
                f"params={ {str(k): v for k, v in self.params.items()} })\n"
                f"equations:\n{eqs}\nboundary conditions:\n{bcs}")
