"""spdee.ir.backgrounds — uniform representation of background-field components.

Background fields appear in IR equations as known coefficients. A
component's value may be an analytic SymPy expression in (r, theta) or
a numerical callable (typically an interpolator over tabulated data,
e.g. a solar reference model). Each physical component of a field is
wrapped in a BackgroundComponent, and the backend's evaluator
dispatches on the subclass. Derivatives of numeric components are
taken by applying the backend's finite-difference operators to the
gridded values.

Public API
----------
BackgroundComponent : abstract base class with .evaluate_on_grid(R, Theta).
SymPyComponent      : wraps a SymPy expression in (r, theta).
NumericComponent    : wraps a callable f(R, Theta) -> array.
CompositeComponent  : pointwise product of two components.

as_background_component(value) : SymPy expressions and numbers become
                                  SymPyComponent; callables become
                                  NumericComponent; BackgroundComponent
                                  values pass through.
multiply_components(a, b)       : product, collapsing SymPy x SymPy.
reciprocal_component(a)         : 1/a, a SymPyComponent when a is one.
radial_interpolator(...)        : NumericComponent from tabular f(r).
meridional_interpolator(...)    : NumericComponent from tabular f(r, theta).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import sympy as sp


# ===========================================================================
# Base class
# ===========================================================================

class BackgroundComponent(ABC):
    """A single physical component of a background field.

    Subclasses provide `evaluate_on_grid(R, Theta)`, returning a complex
    array with the shape of R.
    """

    @abstractmethod
    def evaluate_on_grid(self, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        """Values of the component on the (R, Theta) mesh, shaped like R
        and coercible to complex."""
        raise NotImplementedError


# ===========================================================================
# SymPy-backed component
# ===========================================================================

@dataclass(frozen=True)
class SymPyComponent(BackgroundComponent):
    """A component defined by a SymPy expression in (r, theta).

    Symbol validation happens at field construction
    (`_validate_background_expr` in spdee.ir.fields).

    Parameters
    ----------
    expr : sympy.Expr
        Expression in r and theta; may be a constant.
    """
    expr: sp.Expr

    def evaluate_on_grid(self, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        # Lambdify at call time: lambdified functions do not pickle, and
        # the component must stay picklable.
        from spdee.ir.fields import r as r_sym, theta as theta_sym
        fn = sp.lambdify((r_sym, theta_sym), self.expr, modules="numpy")
        val = fn(R, Theta)
        if np.isscalar(val):
            return np.full(R.shape, val, dtype=complex)
        arr = np.asarray(val)
        if arr.ndim == 0:
            return np.full(R.shape, arr, dtype=complex)
        return np.broadcast_to(arr, R.shape).astype(complex, copy=True)


# ===========================================================================
# Numeric (callable / interpolator) component
# ===========================================================================

@dataclass(frozen=True)
class NumericComponent(BackgroundComponent):
    """A component defined by a numerical callable.

    Parameters
    ----------
    f : callable
        ``f(R, Theta) -> np.ndarray`` returning values shaped like R.
    description : str, optional
        Label for error messages.

    The callable is not validated at construction; a shape mismatch is
    reported when the component is evaluated.
    """
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    description: str = "numeric component"

    def evaluate_on_grid(self, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        val = self.f(R, Theta)
        arr = np.asarray(val)
        if arr.shape != R.shape:
            raise ValueError(
                f"NumericComponent {self.description!r}: callable returned "
                f"array of shape {arr.shape}, expected {R.shape}"
            )
        return arr.astype(complex, copy=True)


# ===========================================================================
# Coercion and inspection helpers
# ===========================================================================

def as_background_component(value: Any) -> BackgroundComponent:
    """Coerce a user-supplied value to a BackgroundComponent.

    None and sympifiable values (SymPy expressions, numbers) become
    SymPyComponent; callables become NumericComponent; BackgroundComponent
    instances pass through. Raises TypeError otherwise.
    """
    if value is None:
        return SymPyComponent(expr=sp.S.Zero)
    if isinstance(value, BackgroundComponent):
        return value
    if callable(value) and not isinstance(value, sp.Basic):
        return NumericComponent(f=value)
    try:
        return SymPyComponent(expr=sp.sympify(value))
    except (sp.SympifyError, TypeError) as e:
        raise TypeError(
            f"Cannot interpret value {value!r} (type {type(value).__name__}) "
            f"as a BackgroundComponent. Expected SymPy expression, number, "
            f"callable, or BackgroundComponent instance."
        ) from e


def is_sympy_component(comp: BackgroundComponent) -> bool:
    """True if `comp` is a SymPyComponent."""
    return isinstance(comp, SymPyComponent)


def is_numeric_component(comp: BackgroundComponent) -> bool:
    """True if `comp` is a NumericComponent."""
    return isinstance(comp, NumericComponent)


def get_sympy_expr(comp: BackgroundComponent) -> sp.Expr:
    """SymPy expression of a SymPyComponent; raises TypeError otherwise."""
    if not isinstance(comp, SymPyComponent):
        raise TypeError(
            f"get_sympy_expr requires a SymPyComponent; got "
            f"{type(comp).__name__}"
        )
    return comp.expr


# ===========================================================================
# CompositeComponent — products of BackgroundComponents
# ===========================================================================

@dataclass(frozen=True)
class CompositeComponent(BackgroundComponent):
    """Pointwise product of two BackgroundComponents.

    Produced by `multiply_components` when at least one factor is
    numeric, so coefficient chains such as `rho0_tabular * 2 * Lap(v)`
    fold into a single component. SymPy x SymPy products collapse to a
    SymPyComponent instead.
    """
    left: BackgroundComponent
    right: BackgroundComponent

    def evaluate_on_grid(self, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        l = self.left.evaluate_on_grid(R, Theta)
        r = self.right.evaluate_on_grid(R, Theta)
        return l * r


def multiply_components(
    a: BackgroundComponent, b: BackgroundComponent
) -> BackgroundComponent:
    """Product of two BackgroundComponents.

    The SymPy constant 1 is the identity (the other operand is returned
    unchanged). SymPy x SymPy gives a SymPyComponent with the symbolic
    product; any other combination gives a CompositeComponent.
    """
    if isinstance(a, SymPyComponent) and a.expr == sp.S.One:
        return b
    if isinstance(b, SymPyComponent) and b.expr == sp.S.One:
        return a
    if isinstance(a, SymPyComponent) and isinstance(b, SymPyComponent):
        return SymPyComponent(expr=a.expr * b.expr)
    return CompositeComponent(left=a, right=b)


@dataclass(frozen=True)
class ReciprocalComponent(BackgroundComponent):
    """Pointwise reciprocal 1/inner of a BackgroundComponent, produced by
    `reciprocal_component` for a numeric divisor."""
    inner: BackgroundComponent

    def evaluate_on_grid(self, R: np.ndarray, Theta: np.ndarray) -> np.ndarray:
        return 1.0 / self.inner.evaluate_on_grid(R, Theta)


def reciprocal_component(a: BackgroundComponent) -> BackgroundComponent:
    """1/a: a SymPyComponent with the symbolic reciprocal when `a` is
    SymPy-backed, else a ReciprocalComponent."""
    if isinstance(a, SymPyComponent):
        return SymPyComponent(expr=1 / a.expr)
    return ReciprocalComponent(inner=a)


# ===========================================================================
# Interpolator factories — tabular data -> BackgroundComponent
# ===========================================================================

def radial_interpolator(
    r_table: np.ndarray,
    f_table: np.ndarray,
    kind: str = "cubic",
    boundary: str = "clamp",
    description: str = "radial interpolator",
) -> NumericComponent:
    """NumericComponent from tabular f(r) data, broadcast along theta.

    Parameters
    ----------
    r_table : 1D array of length N
        Radial sample points, strictly increasing.
    f_table : 1D array of length N
        f(r) values at each sample.
    kind : str
        'cubic' (scipy CubicSpline), 'pchip', 'akima', 'linear', or
        'quadratic' (the last two via scipy interp1d).
    boundary : str
        Handling of queries outside [r_table[0], r_table[-1]]:
        'clamp' (nearest endpoint value), 'extrapolate', or 'error'.
    description : str
        Label for error messages.
    """
    from scipy.interpolate import (
        CubicSpline, interp1d, PchipInterpolator, Akima1DInterpolator,
    )
    r_arr = np.asarray(r_table, dtype=float)
    f_arr = np.asarray(f_table, dtype=float)
    if r_arr.ndim != 1 or f_arr.ndim != 1:
        raise ValueError(
            f"radial_interpolator: r_table and f_table must be 1D; "
            f"got shapes {r_arr.shape}, {f_arr.shape}"
        )
    if r_arr.shape != f_arr.shape:
        raise ValueError(
            f"radial_interpolator: r_table and f_table must have the same "
            f"length; got {r_arr.size}, {f_arr.size}"
        )
    if not np.all(np.diff(r_arr) > 0):
        raise ValueError("radial_interpolator: r_table must be strictly increasing")
    if boundary not in ("clamp", "extrapolate", "error"):
        raise ValueError(
            f"radial_interpolator: boundary must be 'clamp', 'extrapolate', "
            f"or 'error'; got {boundary!r}"
        )

    if kind == "cubic":
        interp = CubicSpline(
            r_arr, f_arr,
            extrapolate=(boundary == "extrapolate"),
        )
    elif kind == "pchip":
        interp = PchipInterpolator(
            r_arr, f_arr,
            extrapolate=(boundary == "extrapolate"),
        )
    elif kind == "akima":
        interp = Akima1DInterpolator(r_arr, f_arr)
    elif kind in ("linear", "quadratic"):
        fill = ("extrapolate" if boundary == "extrapolate"
                else (f_arr[0], f_arr[-1]))
        interp = interp1d(
            r_arr, f_arr, kind=kind,
            bounds_error=(boundary == "error"),
            fill_value=fill,
        )
    else:
        raise ValueError(
            f"radial_interpolator: unsupported kind {kind!r}; "
            f"expected 'cubic', 'pchip', 'akima', 'linear', or 'quadratic'"
        )

    r_min, r_max = r_arr[0], r_arr[-1]

    def f_callable(R, Theta):
        R_arr = np.asarray(R)
        if boundary == "clamp":
            R_query = np.clip(R_arr, r_min, r_max)
        elif boundary == "error":
            if np.any(R_arr < r_min) or np.any(R_arr > r_max):
                raise ValueError(
                    f"radial_interpolator {description!r}: query out of "
                    f"range [{r_min}, {r_max}]"
                )
            R_query = R_arr
        else:
            R_query = R_arr
        return np.asarray(interp(R_query), dtype=float).reshape(R_arr.shape)

    return NumericComponent(f=f_callable, description=description)


def meridional_interpolator(
    r_table: np.ndarray,
    theta_table: np.ndarray,
    f_table: np.ndarray,
    kind: str = "linear",
    boundary: str = "clamp",
    description: str = "meridional interpolator",
) -> NumericComponent:
    """NumericComponent from tabular f(r, theta) data on a rectangular
    grid, via scipy RegularGridInterpolator.

    Parameters
    ----------
    r_table : 1D array, length N_r
        Radial sample points, strictly increasing.
    theta_table : 1D array, length N_theta
        Theta sample points in radians, strictly increasing.
    f_table : 2D array, shape (N_r, N_theta)
        f(r_i, theta_j) values.
    kind : str
        RegularGridInterpolator method: 'linear', 'nearest', 'cubic',
        or 'quintic'.
    boundary : str
        Out-of-range handling: 'clamp' (snap the query to the domain),
        'extrapolate', or 'error'.
    description : str
        Label for error messages.
    """
    from scipy.interpolate import RegularGridInterpolator
    r_arr = np.asarray(r_table, dtype=float)
    t_arr = np.asarray(theta_table, dtype=float)
    f_arr = np.asarray(f_table, dtype=float)
    if r_arr.ndim != 1 or t_arr.ndim != 1:
        raise ValueError(
            "meridional_interpolator: r_table and theta_table must be 1D"
        )
    if f_arr.shape != (r_arr.size, t_arr.size):
        raise ValueError(
            f"meridional_interpolator: f_table shape {f_arr.shape} "
            f"must equal (len(r_table)={r_arr.size}, "
            f"len(theta_table)={t_arr.size})"
        )
    if not (np.all(np.diff(r_arr) > 0) and np.all(np.diff(t_arr) > 0)):
        raise ValueError(
            "meridional_interpolator: r_table and theta_table must be "
            "strictly increasing"
        )
    if boundary not in ("clamp", "extrapolate", "error"):
        raise ValueError(
            f"meridional_interpolator: boundary must be 'clamp', "
            f"'extrapolate', or 'error'; got {boundary!r}"
        )

    if boundary == "error":
        rgi = RegularGridInterpolator(
            (r_arr, t_arr), f_arr, method=kind, bounds_error=True,
        )
    else:
        # 'extrapolate' relies on fill_value=None; 'clamp' clips the
        # query below so extrapolation never triggers.
        rgi = RegularGridInterpolator(
            (r_arr, t_arr), f_arr, method=kind,
            bounds_error=False, fill_value=None,
        )
    r_min, r_max = r_arr[0], r_arr[-1]
    t_min, t_max = t_arr[0], t_arr[-1]

    def f_callable(R, Theta):
        R_a = np.asarray(R)
        T_a = np.asarray(Theta)
        if R_a.shape != T_a.shape:
            raise ValueError(
                f"meridional_interpolator: R and Theta shapes must match; "
                f"got {R_a.shape} vs {T_a.shape}"
            )
        if boundary == "clamp":
            R_q = np.clip(R_a, r_min, r_max)
            T_q = np.clip(T_a, t_min, t_max)
        else:
            R_q = R_a
            T_q = T_a
        pts = np.stack([R_q.ravel(), T_q.ravel()], axis=-1)
        return rgi(pts).reshape(R_a.shape)

    return NumericComponent(f=f_callable, description=description)
