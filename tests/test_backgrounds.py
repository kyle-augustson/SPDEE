"""Tests for the BackgroundComponent hierarchy.

These tests verify that:
  1. SymPyComponent works as a thin wrapper around a SymPy expression.
  2. NumericComponent allows declaring a background via a callable, with
     the IR and backend evaluator both producing correct grid values.
  3. A vector or tensor field can mix SymPy and Numeric components.
  4. The IR's spin_components() method still works for all-SymPy fields
     and raises a clear error for non-SymPy fields.
  5. The backend evaluator gives consistent results across the two paths
     when they represent the same mathematical function.
"""

import numpy as np
import pytest
import sympy as sp

from spdee.ir.backgrounds import (
    SymPyComponent, NumericComponent, as_background_component,
)
from spdee.ir.fields import (
    Background, r as r_sym, theta as theta_sym,
)
from spdee.backends.common.field_eval import (
    evaluate_scalar_background_2d,
    evaluate_vector_background_spin_2d,
)
from spdee.backends.fd2.grid import make_shell_grid


# ===========================================================================
# Basic SymPyComponent and NumericComponent
# ===========================================================================

def test_sympy_component_evaluation():
    """SymPyComponent evaluates correctly on a (R, Theta) mesh."""
    comp = SymPyComponent(r_sym ** 2 * sp.sin(theta_sym))
    grid = make_shell_grid(N_r=10, N_theta=8, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = comp.evaluate_on_grid(R, Theta)
    expected = R ** 2 * np.sin(Theta)
    assert np.allclose(out, expected)
    assert out.dtype == complex


def test_sympy_component_constant():
    """A constant SymPy expression broadcasts to the full grid."""
    comp = SymPyComponent(sp.Integer(5))
    grid = make_shell_grid(N_r=4, N_theta=3, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = comp.evaluate_on_grid(R, Theta)
    assert out.shape == R.shape
    assert np.allclose(out, 5.0)


def test_sympy_component_zero():
    """Zero is the natural default and broadcasts correctly."""
    comp = SymPyComponent(sp.S.Zero)
    grid = make_shell_grid(N_r=4, N_theta=3, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = comp.evaluate_on_grid(R, Theta)
    assert out.shape == R.shape
    assert np.allclose(out, 0.0)


def test_numeric_component_callable():
    """NumericComponent wraps an arbitrary callable."""
    def f(R, Theta):
        return R ** 2 * np.sin(Theta)
    comp = NumericComponent(f, description="r^2 sin(theta)")
    grid = make_shell_grid(N_r=10, N_theta=8, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = comp.evaluate_on_grid(R, Theta)
    assert np.allclose(out, R ** 2 * np.sin(Theta))
    assert out.dtype == complex


def test_numeric_component_wrong_shape_raises():
    """A callable returning the wrong shape raises a clear error."""
    def f(R, Theta):
        return np.zeros(5)  # wrong shape
    comp = NumericComponent(f, description="bad shape")
    grid = make_shell_grid(N_r=10, N_theta=8, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    with pytest.raises(ValueError, match="shape"):
        comp.evaluate_on_grid(R, Theta)


def test_sympy_and_numeric_agree():
    """Two equivalent SymPy and Numeric components give the same values."""
    expr = r_sym ** 2 * sp.sin(theta_sym)
    sympy_comp = SymPyComponent(expr)
    numeric_comp = NumericComponent(lambda R, T: R ** 2 * np.sin(T))

    grid = make_shell_grid(N_r=16, N_theta=12, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()

    a = sympy_comp.evaluate_on_grid(R, Theta)
    b = numeric_comp.evaluate_on_grid(R, Theta)
    assert np.allclose(a, b)


# ===========================================================================
# as_background_component dispatching
# ===========================================================================

def test_as_background_component_sympy_expr():
    """SymPy expressions auto-wrap to SymPyComponent."""
    comp = as_background_component(r_sym ** 2)
    assert isinstance(comp, SymPyComponent)
    assert comp.expr == r_sym ** 2


def test_as_background_component_number():
    """Python numbers wrap to SymPyComponent via sympify."""
    comp = as_background_component(3.14)
    assert isinstance(comp, SymPyComponent)


def test_as_background_component_callable():
    """A callable wraps to NumericComponent."""
    comp = as_background_component(lambda R, T: R + T)
    assert isinstance(comp, NumericComponent)


def test_as_background_component_passthrough():
    """An existing BackgroundComponent is passed through unchanged."""
    inner = SymPyComponent(r_sym)
    outer = as_background_component(inner)
    assert outer is inner


def test_as_background_component_none():
    """None becomes SymPyComponent(0)."""
    comp = as_background_component(None)
    assert isinstance(comp, SymPyComponent)
    assert comp.expr == 0


# ===========================================================================
# IR field construction with BackgroundComponent / callable inputs
# ===========================================================================

def test_scalar_field_accepts_callable():
    """Background.Scalar accepts a callable as expr."""
    f = lambda R, T: np.exp(-R)
    field = Background.Scalar("rho0", expr=f)
    assert isinstance(field.component, NumericComponent)
    assert field.expr is None  # not SymPy-backed


def test_scalar_field_accepts_sympy():
    """Background.Scalar accepts a SymPy expression."""
    field = Background.Scalar("rho0", expr=sp.exp(-r_sym))
    assert isinstance(field.component, SymPyComponent)
    assert field.expr == sp.exp(-r_sym)


def test_scalar_field_accepts_explicit_component():
    """Background.Scalar accepts a BackgroundComponent directly."""
    comp = NumericComponent(lambda R, T: R, description="r")
    field = Background.Scalar("rho0", expr=comp)
    assert field.component is comp


def test_vector_field_accepts_callable_components():
    """Background.Vector accepts mixed SymPy and callable components."""
    field = Background.Vector(
        "v0",
        v_r=sp.cos(theta_sym),
        v_theta=lambda R, T: np.sin(T),  # numeric
        v_phi=None,  # zero
    )
    assert isinstance(field.components["v_r"], SymPyComponent)
    assert isinstance(field.components["v_theta"], NumericComponent)
    assert isinstance(field.components["v_phi"], SymPyComponent)
    assert field.v_r == sp.cos(theta_sym)
    assert field.v_theta is None  # numeric, no SymPy expr accessor
    assert field.v_phi == 0


def test_vector_field_spin_components_blocks_on_numeric():
    """vec.spin_components() raises if any component is non-SymPy."""
    field = Background.Vector(
        "v0",
        v_r=lambda R, T: np.ones_like(R),  # numeric
    )
    with pytest.raises(TypeError, match="SymPy-backed"):
        field.spin_components()


def test_vector_field_spin_components_works_for_sympy():
    """vec.spin_components() still works for all-SymPy fields."""
    field = Background.Vector("v0", v_r=sp.cos(theta_sym), v_theta=sp.sin(theta_sym))
    sc = field.spin_components()
    assert sc[0] == sp.cos(theta_sym)
    # v^- = (v_theta - i v_phi)/sqrt(2) = sin(theta) / sqrt(2)
    assert sp.simplify(sc[-1] - sp.sin(theta_sym) / sp.sqrt(2)) == 0


def test_tensor_field_accepts_callable_components():
    """Background.Tensor accepts callable component values."""
    field = Background.Tensor("T0", components={
        ("r", "r"): r_sym ** 2,
        ("theta", "theta"): lambda R, T: np.sin(T),
    })
    assert isinstance(field.components[("r", "r")], SymPyComponent)
    assert isinstance(field.components[("theta", "theta")], NumericComponent)
    assert isinstance(field.components[("phi", "phi")], SymPyComponent)  # zero default


# ===========================================================================
# Backend evaluator end-to-end (SymPy and Numeric paths)
# ===========================================================================

def test_evaluator_sympy_scalar():
    """Scalar field with SymPy expr evaluates correctly on a grid."""
    field = Background.Scalar("rho0", expr=r_sym ** 2 * sp.sin(theta_sym))
    grid = make_shell_grid(N_r=12, N_theta=10, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = evaluate_scalar_background_2d(field, R, Theta)
    assert np.allclose(out, R ** 2 * np.sin(Theta))


def test_evaluator_numeric_scalar():
    """Scalar field with callable expr evaluates correctly."""
    f = lambda R, T: R ** 2 * np.sin(T)
    field = Background.Scalar("rho0", expr=f)
    grid = make_shell_grid(N_r=12, N_theta=10, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()
    out = evaluate_scalar_background_2d(field, R, Theta)
    assert np.allclose(out, R ** 2 * np.sin(Theta))


def test_evaluator_sympy_and_numeric_scalar_agree():
    """Same mathematical function via SymPy and Numeric gives the same grid values."""
    grid = make_shell_grid(N_r=20, N_theta=14, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()

    sympy_field = Background.Scalar("f_sympy", expr=r_sym ** 3 + sp.cos(theta_sym))
    numeric_field = Background.Scalar(
        "f_numeric", expr=lambda R, T: R ** 3 + np.cos(T)
    )
    a = evaluate_scalar_background_2d(sympy_field, R, Theta)
    b = evaluate_scalar_background_2d(numeric_field, R, Theta)
    assert np.allclose(a, b)


def test_evaluator_mixed_vector():
    """A vector with mixed SymPy + Numeric components yields the right spin grids."""
    grid = make_shell_grid(N_r=20, N_theta=14, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()

    field = Background.Vector(
        "v0",
        v_r=sp.cos(theta_sym),                          # SymPy
        v_theta=lambda R, T: np.sin(T),                 # Numeric
        v_phi=NumericComponent(lambda R, T: R * np.cos(T)),  # explicit Numeric
    )
    spin = evaluate_vector_background_spin_2d(field, R, Theta)

    # Check against direct numerical evaluation:
    # v^- = (v_theta - i v_phi) / sqrt(2)
    # v^0 = v_r
    # v^+ = (v_theta + i v_phi) / sqrt(2)
    v_r = np.cos(Theta)
    v_th = np.sin(Theta)
    v_ph = R * np.cos(Theta)
    sqrt2 = np.sqrt(2.0)
    assert np.allclose(spin[0], v_r)
    assert np.allclose(spin[-1], (v_th - 1j * v_ph) / sqrt2)
    assert np.allclose(spin[+1], (v_th + 1j * v_ph) / sqrt2)


def test_evaluator_numeric_interpolator_style():
    """Common use case: callable mimicking scipy.interpolate.RegularGridInterpolator
    (which has a __call__ that takes a points array). Wrap it for our API."""
    grid = make_shell_grid(N_r=16, N_theta=12, r_inner=0.1, r_outer=1.0)
    R, Theta = grid.mesh()

    # Simulate an interpolator that wants a (..., 2) point array.
    # We adapt it with a small lambda.
    def my_interp_inner(points):
        # points shape (n, 2) with (r, theta)
        r_arr = points[..., 0]
        t_arr = points[..., 1]
        return np.exp(-r_arr) * np.sin(t_arr)

    def wrapper(R, T):
        pts = np.stack([R, T], axis=-1)
        return my_interp_inner(pts)

    field = Background.Scalar("rho0", expr=wrapper)
    out = evaluate_scalar_background_2d(field, R, Theta)
    assert np.allclose(out, np.exp(-R) * np.sin(Theta))


# ===========================================================================
# Validation still works
# ===========================================================================

def test_sympy_phi_dependence_rejected():
    """A SymPy expression with phi-dependence is still rejected."""
    phi = sp.Symbol("phi")
    with pytest.raises(ValueError, match="disallowed symbols"):
        Background.Scalar("bad", expr=sp.cos(phi))


def test_numeric_component_not_validated():
    """We can't statically validate a callable; it goes through."""
    # Reasonable user-facing behavior: trust the callable. If the callable
    # ends up using phi internally, that's the user's problem.
    field = Background.Scalar("rho0", expr=lambda R, T: np.ones_like(R))
    # No exception raised at construction time.
    assert isinstance(field.component, NumericComponent)
