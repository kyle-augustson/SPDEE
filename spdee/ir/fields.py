"""spdee.ir.fields — field types.

Three field kinds (scalar, vector, rank-2 tensor) and two roles:
background (known, r/theta-dependent) and unknown (eigenproblem
variable).

Vector and tensor components use the V19 spin convention:

    v_+ = (v_theta + i v_phi) / sqrt(2)
    v_0 = v_r
    v_- = (v_theta - i v_phi) / sqrt(2)

with basis vectors e_pm = (e_theta -/+ i e_phi)/sqrt(2), e_0 = e_r.
The spin index sigma ranges over {-1, 0, +1}. A rank-2 tensor has
components T^{sigma_1, sigma_2} on the outer product of the rank-1
basis.

Background fields are specified by their physical components
(v_r, v_theta, v_phi); the IR stores each as a BackgroundComponent
(see spdee.ir.backgrounds), which may be an analytic SymPy expression
in (r, theta) or a numeric callable. Backgrounds may depend on r and
theta but not on phi (the problem is m-separable).

Arithmetic on fields returns Expression objects (see expressions.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import sympy as sp

from spdee.ir.backgrounds import (
    BackgroundComponent,
    as_background_component, is_sympy_component, get_sympy_expr,
)

# Coordinate symbols for background expressions.
r, theta = sp.symbols("r theta", real=True, positive=True)

# Spin indices in V19 order.
SPIN_INDICES = (-1, 0, +1)


# =============================================================================
# Field role
# =============================================================================

class FieldRole:
    """Whether a field is a background (known) or an unknown."""
    BACKGROUND = "background"
    UNKNOWN = "unknown"


# =============================================================================
# Field kinds
# =============================================================================

@dataclass(frozen=True)
class _FieldBase:
    """Common base for all field types.

    Attributes
    ----------
    name : str
        Display name (used in expressions and error messages).
    role : str
        FieldRole.BACKGROUND or FieldRole.UNKNOWN.
    """
    name: str
    role: str

    def __post_init__(self):
        if self.role not in (FieldRole.BACKGROUND, FieldRole.UNKNOWN):
            raise ValueError(
                f"Invalid role {self.role!r}; "
                f"must be 'background' or 'unknown'"
            )
        if not self.name:
            raise ValueError("Field must have a non-empty name")

    @property
    def rank(self) -> int:
        raise NotImplementedError

    @property
    def kind(self) -> str:
        raise NotImplementedError

    @property
    def is_background(self) -> bool:
        return self.role == FieldRole.BACKGROUND

    @property
    def is_unknown(self) -> bool:
        return self.role == FieldRole.UNKNOWN

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, role={self.role!r})"

    def _as_expr(self):
        # Local import: expressions.py imports this module.
        from spdee.ir.expressions import FieldRef
        return FieldRef(self)

    def __add__(self, other):
        return self._as_expr() + other

    def __radd__(self, other):
        return other + self._as_expr()

    def __sub__(self, other):
        return self._as_expr() - other

    def __rsub__(self, other):
        return other - self._as_expr()

    def __neg__(self):
        return -self._as_expr()

    def __mul__(self, other):
        return self._as_expr() * other

    def __rmul__(self, other):
        return other * self._as_expr()

    def __truediv__(self, other):
        return self._as_expr() / other

    def __rtruediv__(self, other):
        return other / self._as_expr()


@dataclass(frozen=True)
class ScalarField(_FieldBase):
    """Rank-0 (scalar) field.

    A background scalar is specified by `expr`: a SymPy expression in
    (r, theta), a callable f(R, Theta), or a BackgroundComponent. An
    unknown scalar is a labelled placeholder.

    Parameters
    ----------
    name : str
    role : str
    expr : SymPy expression, callable, BackgroundComponent, or None
        Value of a background field. Must be None for unknowns.

    Attributes (post-construction)
    ------------------------------
    component : BackgroundComponent or None
        Normalized value of a background field; None for unknowns.
    expr : sympy.Expr or None
        The SymPy expression when `component` is SymPy-backed, else
        None. Use `component.evaluate_on_grid(...)` for numerical values
        regardless of type.
    """
    expr: Optional[Any] = None
    component: Optional[BackgroundComponent] = None

    def __post_init__(self):
        super().__post_init__()
        if self.is_background:
            if self.expr is None:
                raise ValueError(
                    f"Background ScalarField {self.name!r} requires an expr"
                )
            _validate_background_expr(self.expr, self.name)
            comp = as_background_component(self.expr)
            object.__setattr__(self, "component", comp)
            if is_sympy_component(comp):
                object.__setattr__(self, "expr", get_sympy_expr(comp))
            else:
                object.__setattr__(self, "expr", None)
        else:
            if self.expr is not None:
                raise ValueError(
                    f"Unknown ScalarField {self.name!r} must not have an expr"
                )

    @property
    def rank(self) -> int:
        return 0

    @property
    def kind(self) -> str:
        return "scalar"


@dataclass(frozen=True)
class VectorField(_FieldBase):
    """Rank-1 (vector) field.

    A background vector is specified by its physical components
    (v_r, v_theta, v_phi), each a SymPy expression in (r, theta), a
    callable f(R, Theta), or a BackgroundComponent; omitted components
    are zero. An unknown vector is a labelled placeholder.

    Parameters
    ----------
    name : str
    role : str
    v_r, v_theta, v_phi : SymPy expr, callable, BackgroundComponent, or None
        Physical components of a background field. Must be None for
        unknowns.

    Attributes (post-construction)
    ------------------------------
    components : dict {"v_r": BackgroundComponent, "v_theta": ..., "v_phi": ...}
        Normalized components of a background field; None for unknowns.
    v_r, v_theta, v_phi : sympy.Expr or None
        The SymPy expression of each SymPy-backed component, else None.
        Use `components[...].evaluate_on_grid(...)` for numerical values
        regardless of type.
    """
    v_r: Optional[Any] = None
    v_theta: Optional[Any] = None
    v_phi: Optional[Any] = None
    components: Optional[Dict[str, BackgroundComponent]] = None

    def __post_init__(self):
        super().__post_init__()
        if self.is_background:
            if all(c is None for c in (self.v_r, self.v_theta, self.v_phi)):
                raise ValueError(
                    f"Background VectorField {self.name!r} must specify at "
                    f"least one physical component (v_r, v_theta, or v_phi)"
                )
            comps = {}
            for label, value in (
                ("v_r", self.v_r),
                ("v_theta", self.v_theta),
                ("v_phi", self.v_phi),
            ):
                raw = _zero_if_none(value)
                _validate_background_expr(raw, f"{self.name}.{label}")
                comps[label] = as_background_component(raw)
            object.__setattr__(self, "components", comps)
            for label, comp in comps.items():
                if is_sympy_component(comp):
                    object.__setattr__(self, label, get_sympy_expr(comp))
                else:
                    object.__setattr__(self, label, None)
        else:
            for label, comp in (
                ("v_r", self.v_r),
                ("v_theta", self.v_theta),
                ("v_phi", self.v_phi),
            ):
                if comp is not None:
                    raise ValueError(
                        f"Unknown VectorField {self.name!r} must not have "
                        f"physical components ({label} was specified)"
                    )

    @property
    def rank(self) -> int:
        return 1

    @property
    def kind(self) -> str:
        return "vector"

    def spin_components(self) -> Dict[int, sp.Expr]:
        """Contravariant spin components {-1: ..., 0: ..., +1: ...} as
        SymPy expressions:

            v^+ = (v_theta + i v_phi) / sqrt(2)
            v^0 = v_r
            v^- = (v_theta - i v_phi) / sqrt(2)

        Requires a background field with SymPy-backed components; numeric
        backgrounds are evaluated by the backend instead.
        """
        if not self.is_background:
            raise ValueError(
                f"spin_components() requires a background field; "
                f"{self.name!r} is an unknown"
            )
        for label, comp in self.components.items():
            if not is_sympy_component(comp):
                raise TypeError(
                    f"spin_components() requires SymPy-backed components; "
                    f"{self.name!r}.{label} is {type(comp).__name__}. "
                    f"Use the backend evaluator for non-SymPy backgrounds."
                )
        I = sp.I
        sqrt2 = sp.sqrt(2)
        v_r = get_sympy_expr(self.components["v_r"])
        v_th = get_sympy_expr(self.components["v_theta"])
        v_ph = get_sympy_expr(self.components["v_phi"])
        return {
            -1: (v_th - I * v_ph) / sqrt2,
             0: v_r,
            +1: (v_th + I * v_ph) / sqrt2,
        }


@dataclass(frozen=True)
class TensorField(_FieldBase):
    """Rank-2 tensor field.

    A background tensor is specified by physical components T_{i, j},
    i, j in ("r", "theta", "phi"), each a SymPy expression in
    (r, theta), a callable, or a BackgroundComponent; omitted entries
    are zero. An unknown tensor is a labelled placeholder.

    Parameters
    ----------
    name : str
    role : str
    components : dict, optional
        {(i, j): value} for a background field. Must be None for
        unknowns.
    symmetric : bool, default False
        Asserts T_{ij} = T_{ji}. For a background field with SymPy-backed
        components this is checked at construction; a mismatch raises
        ValueError. The tensor reducer uses the flag to mark the leaf as
        symmetric.

    Attributes (post-construction)
    ------------------------------
    components : dict {(i, j): BackgroundComponent}
        All 9 entries of a background field (missing entries filled
        with zero); None for unknowns.
    symmetric : bool
    """
    components: Optional[Dict[Tuple[str, str], Any]] = None
    symmetric: bool = False

    def __post_init__(self):
        super().__post_init__()
        valid_idx = ("r", "theta", "phi")
        if self.is_background:
            if not self.components:
                raise ValueError(
                    f"Background TensorField {self.name!r} must specify "
                    f"at least one component in `components`"
                )
            full: Dict[Tuple[str, str], BackgroundComponent] = {}
            for i in valid_idx:
                for j in valid_idx:
                    full[(i, j)] = as_background_component(sp.S.Zero)
            for key, value in self.components.items():
                if (
                    not isinstance(key, tuple)
                    or len(key) != 2
                    or key[0] not in valid_idx
                    or key[1] not in valid_idx
                ):
                    raise ValueError(
                        f"TensorField component key {key!r} invalid; "
                        f"must be (i, j) with i, j in {valid_idx}"
                    )
                _validate_background_expr(value, f"{self.name}{key}")
                full[key] = as_background_component(value)
            object.__setattr__(self, "components", full)
            if self.symmetric:
                # Only SymPy-backed pairs can be compared symbolically;
                # numeric components are taken on trust.
                for i_idx, i in enumerate(valid_idx):
                    for j_idx, j in enumerate(valid_idx):
                        if i_idx >= j_idx:
                            continue
                        c_ij = full[(i, j)]
                        c_ji = full[(j, i)]
                        if is_sympy_component(c_ij) and is_sympy_component(c_ji):
                            diff = sp.simplify(
                                get_sympy_expr(c_ij) - get_sympy_expr(c_ji)
                            )
                            if diff != 0:
                                raise ValueError(
                                    f"Background TensorField {self.name!r} "
                                    f"declared symmetric=True but the "
                                    f"({i}, {j}) and ({j}, {i}) components "
                                    f"differ by a non-zero amount: {diff}. "
                                    f"Either supply symmetric components or "
                                    f"set symmetric=False."
                                )
        else:
            if self.components is not None:
                raise ValueError(
                    f"Unknown TensorField {self.name!r} must not have "
                    f"physical components"
                )

    @property
    def rank(self) -> int:
        return 2

    @property
    def kind(self) -> str:
        return "tensor"

    def spin_components(self) -> Dict[Tuple[int, int], sp.Expr]:
        """Contravariant spin components {(sigma_1, sigma_2): expr} for
        all 9 index pairs:

            T^{sigma_1, sigma_2} = sum_{i, j} U(sigma_1, i) U(sigma_2, j) T_{i, j}

        where U is the rank-1 physical-to-spin matrix
        (`_U_physical_to_spin`). Requires a background field with
        SymPy-backed components.
        """
        if not self.is_background:
            raise ValueError(
                f"spin_components() requires a background field; "
                f"{self.name!r} is an unknown"
            )
        for key, comp in self.components.items():
            if not is_sympy_component(comp):
                raise TypeError(
                    f"spin_components() requires SymPy-backed components; "
                    f"{self.name!r}{key} is {type(comp).__name__}. "
                    f"Use the backend evaluator for non-SymPy backgrounds."
                )
        U = _U_physical_to_spin()
        idx_to_phys = {0: "r", 1: "theta", 2: "phi"}
        result: Dict[Tuple[int, int], sp.Expr] = {}
        for s1_idx, s1 in enumerate(SPIN_INDICES):
            for s2_idx, s2 in enumerate(SPIN_INDICES):
                expr = sp.S.Zero
                for i_idx in range(3):
                    for j_idx in range(3):
                        T_ij = get_sympy_expr(self.components[
                            (idx_to_phys[i_idx], idx_to_phys[j_idx])
                        ])
                        expr += U[s1_idx, i_idx] * U[s2_idx, j_idx] * T_ij
                result[(s1, s2)] = sp.simplify(expr)
        return result


# =============================================================================
# Convenience constructors
# =============================================================================

class Background:
    """Factory namespace: Background.Scalar(...), .Vector(...), .Tensor(...)."""

    @staticmethod
    def Scalar(name: str, expr: sp.Expr) -> ScalarField:
        return ScalarField(name=name, role=FieldRole.BACKGROUND, expr=expr)

    @staticmethod
    def Vector(
        name: str,
        v_r: Optional[sp.Expr] = None,
        v_theta: Optional[sp.Expr] = None,
        v_phi: Optional[sp.Expr] = None,
    ) -> VectorField:
        return VectorField(
            name=name,
            role=FieldRole.BACKGROUND,
            v_r=v_r, v_theta=v_theta, v_phi=v_phi,
        )

    @staticmethod
    def Tensor(
        name: str,
        components: Dict[Tuple[str, str], sp.Expr],
        symmetric: bool = False,
    ) -> TensorField:
        return TensorField(
            name=name,
            role=FieldRole.BACKGROUND,
            components=components,
            symmetric=symmetric,
        )


class Unknown:
    """Factory namespace: Unknown.Scalar(...), .Vector(...), .Tensor(...)."""

    @staticmethod
    def Scalar(name: str) -> ScalarField:
        return ScalarField(name=name, role=FieldRole.UNKNOWN)

    @staticmethod
    def Vector(name: str) -> VectorField:
        return VectorField(name=name, role=FieldRole.UNKNOWN)

    @staticmethod
    def Tensor(name: str) -> TensorField:
        return TensorField(name=name, role=FieldRole.UNKNOWN)


# =============================================================================
# Internal helpers
# =============================================================================

def _zero_if_none(expr: Optional[Any]) -> Any:
    """SymPy zero if `expr` is None, else `expr` unchanged."""
    return sp.S.Zero if expr is None else expr


def _validate_background_expr(value: Any, label: str) -> None:
    """Check that a SymPy-backed background component depends only on
    r and theta.

    Numeric components and raw callables cannot be inspected and pass
    through; values that fail to sympify also pass through (the later
    `as_background_component` call raises a clearer error for them).
    Raises ValueError on a disallowed symbol.
    """
    if isinstance(value, BackgroundComponent):
        if is_sympy_component(value):
            expr = get_sympy_expr(value)
        else:
            return
    elif callable(value) and not isinstance(value, sp.Basic):
        return
    else:
        try:
            expr = sp.sympify(value)
        except sp.SympifyError:
            return
    free = expr.free_symbols
    allowed = {r, theta}
    bad = free - allowed
    if bad:
        names = ", ".join(sorted(str(s) for s in bad))
        raise ValueError(
            f"Background expression for {label!r} contains disallowed "
            f"symbols: {{{names}}}. Backgrounds may depend only on r "
            f"and theta (phi-dependence breaks m-separability)."
        )


def _U_physical_to_spin() -> sp.Matrix:
    """Rank-1 unitary mapping (v_r, v_theta, v_phi) -> (v^{-1}, v^0, v^{+1}).

    Rows index spin in order (-1, 0, +1), columns index physical in
    order (r, theta, phi):

        v^- = (v_theta - i v_phi) / sqrt(2)   -> row -1 = [0, 1/sqrt2, -i/sqrt2]
        v^0 = v_r                             -> row 0  = [1, 0, 0]
        v^+ = (v_theta + i v_phi) / sqrt(2)   -> row +1 = [0, 1/sqrt2, +i/sqrt2]
    """
    s2 = 1 / sp.sqrt(2)
    I = sp.I
    return sp.Matrix([
        [0,  s2, -I * s2],
        [1,  0,   0     ],
        [0,  s2,  I * s2],
    ])
