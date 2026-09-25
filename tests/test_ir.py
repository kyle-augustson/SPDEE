"""Stage 1: IR type-system tests.

Verifies that:
- Fields construct correctly with proper roles
- Background fields validate their SymPy expressions (no phi)
- Operators enforce rank rules (Grad(scalar) -> vector, etc.)
- Algebraic operations track linearity (unknown * unknown rejected)
- Spin-component conversion uses the V19 convention
"""

import pytest
import sympy as sp

from spdee.ir.fields import (
    Background, Unknown, ScalarField, VectorField,
    FieldRole, r, theta,
)
from spdee.ir.operators import (
    Grad, Div, Curl, Lap, Dt, Cross, Dot,
)


# =============================================================================
# Field construction
# =============================================================================

def test_unknown_scalar_construction():
    p = Unknown.Scalar("p")
    assert p.rank == 0
    assert p.kind == "scalar"
    assert p.is_unknown
    assert not p.is_background
    assert p.expr is None


def test_unknown_vector_construction():
    v = Unknown.Vector("v")
    assert v.rank == 1
    assert v.kind == "vector"
    assert v.is_unknown


def test_unknown_tensor_construction():
    T = Unknown.Tensor("T")
    assert T.rank == 2
    assert T.kind == "tensor"
    assert T.is_unknown


def test_background_scalar_construction():
    rho = Background.Scalar("rho", expr=sp.exp(-r))
    assert rho.rank == 0
    assert rho.is_background
    assert rho.expr == sp.exp(-r)


def test_background_vector_construction_partial():
    """Only one physical component specified — others default to 0."""
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    assert B0.v_r == 0
    assert B0.v_theta == 0
    assert B0.v_phi == r * sp.sin(theta)


def test_background_tensor_construction():
    T0 = Background.Tensor("T0", components={
        ("r", "r"): r**2,
        ("theta", "theta"): sp.sin(theta),
    })
    # T0.components values are BackgroundComponent instances. For SymPy
    # values, the underlying expression is in .expr.
    assert T0.components[("r", "r")].expr == r**2
    assert T0.components[("theta", "theta")].expr == sp.sin(theta)
    assert T0.components[("phi", "phi")].expr == 0  # default


def test_background_scalar_rejects_phi():
    """Backgrounds can't depend on phi (would break m-separability)."""
    phi = sp.Symbol("phi")
    with pytest.raises(ValueError, match="disallowed symbols"):
        Background.Scalar("bad", expr=sp.cos(phi))


def test_background_vector_rejects_phi():
    phi = sp.Symbol("phi")
    with pytest.raises(ValueError, match="disallowed symbols"):
        Background.Vector("bad", v_r=phi)


def test_background_must_specify_expr():
    with pytest.raises(ValueError, match="requires an expr"):
        ScalarField(name="bad", role=FieldRole.BACKGROUND, expr=None)


def test_background_vector_must_specify_some_component():
    with pytest.raises(ValueError, match="at least one"):
        Background.Vector("bad")


def test_unknown_cannot_have_expr():
    with pytest.raises(ValueError, match="must not have"):
        ScalarField(name="bad", role=FieldRole.UNKNOWN, expr=r)


def test_unknown_vector_cannot_have_components():
    with pytest.raises(ValueError, match="must not have"):
        VectorField(name="bad", role=FieldRole.UNKNOWN, v_r=r)


def test_field_must_have_name():
    with pytest.raises(ValueError, match="non-empty name"):
        Background.Scalar("", expr=r)


def test_field_role_must_be_valid():
    with pytest.raises(ValueError, match="Invalid role"):
        ScalarField(name="bad", role="garbage", expr=r)


# =============================================================================
# Spin component conversion (V19 convention)
# =============================================================================

def test_vector_spin_components_phi_hat():
    """B0 = (r sin theta) phi_hat. V19 convention:
        v^- = (v_theta - i v_phi)/sqrt(2) = -i r sin(theta)/sqrt(2)
        v^0 = v_r = 0
        v^+ = (v_theta + i v_phi)/sqrt(2) = +i r sin(theta)/sqrt(2)
    """
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    sc = B0.spin_components()
    assert sp.simplify(sc[0]) == 0
    expected_minus = -sp.I * r * sp.sin(theta) / sp.sqrt(2)
    expected_plus = sp.I * r * sp.sin(theta) / sp.sqrt(2)
    assert sp.simplify(sc[-1] - expected_minus) == 0
    assert sp.simplify(sc[+1] - expected_plus) == 0


def test_vector_spin_components_r_hat():
    """B0 = f(r) r_hat. v_0 = f(r), v_+ = v_- = 0."""
    B0 = Background.Vector("B0", v_r=sp.exp(-r))
    sc = B0.spin_components()
    assert sp.simplify(sc[0] - sp.exp(-r)) == 0
    assert sp.simplify(sc[-1]) == 0
    assert sp.simplify(sc[+1]) == 0


def test_vector_spin_components_theta_hat():
    """B0 = g(theta) theta_hat. v_0 = 0, v_+ = v_- = g(theta)/sqrt(2)."""
    g = sp.cos(theta)
    B0 = Background.Vector("B0", v_theta=g)
    sc = B0.spin_components()
    assert sp.simplify(sc[0]) == 0
    assert sp.simplify(sc[-1] - g / sp.sqrt(2)) == 0
    assert sp.simplify(sc[+1] - g / sp.sqrt(2)) == 0


def test_tensor_spin_components_rr():
    """T = r^2 r_hat ⊗ r_hat. Only the (0,0) spin component is nonzero."""
    T = Background.Tensor("T", components={("r", "r"): r**2})
    sc = T.spin_components()
    assert sp.simplify(sc[(0, 0)] - r**2) == 0
    for s1 in (-1, 0, +1):
        for s2 in (-1, 0, +1):
            if (s1, s2) == (0, 0):
                continue
            assert sp.simplify(sc[(s1, s2)]) == 0


def test_spin_components_blocked_on_unknown():
    v = Unknown.Vector("v")
    with pytest.raises(ValueError, match="requires a background"):
        v.spin_components()


# =============================================================================
# Operator type rules
# =============================================================================

def test_grad_of_scalar_is_vector():
    p = Unknown.Scalar("p")
    g = Grad(p)
    assert g.is_vector


def test_grad_of_vector_is_tensor():
    v = Unknown.Vector("v")
    g = Grad(v)
    assert g.is_tensor


def test_grad_of_tensor_raises():
    T = Unknown.Tensor("T")
    with pytest.raises(TypeError, match="scalar or vector"):
        Grad(T)


def test_div_of_vector_is_scalar():
    v = Unknown.Vector("v")
    d = Div(v)
    assert d.is_scalar


def test_div_of_tensor_is_vector():
    T = Unknown.Tensor("T")
    d = Div(T)
    assert d.is_vector


def test_div_of_scalar_raises():
    p = Unknown.Scalar("p")
    with pytest.raises(TypeError, match="vector or tensor"):
        Div(p)


def test_curl_of_vector_is_vector():
    v = Unknown.Vector("v")
    c = Curl(v)
    assert c.is_vector


def test_curl_of_scalar_raises():
    p = Unknown.Scalar("p")
    with pytest.raises(TypeError, match="vector argument"):
        Curl(p)


def test_curl_of_tensor_raises():
    T = Unknown.Tensor("T")
    with pytest.raises(TypeError, match="vector argument"):
        Curl(T)


def test_lap_of_scalar_is_scalar():
    p = Unknown.Scalar("p")
    l = Lap(p)
    assert l.is_scalar


def test_lap_of_vector_is_vector():
    v = Unknown.Vector("v")
    l = Lap(v)
    assert l.is_vector


def test_dt_preserves_rank():
    p = Unknown.Scalar("p")
    v = Unknown.Vector("v")
    assert Dt(p).is_scalar
    assert Dt(v).is_vector


# =============================================================================
# Cross and Dot
# =============================================================================

def test_cross_vector_vector_is_vector():
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    c = Cross(B0, v)
    assert c.is_vector


def test_cross_requires_vectors():
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    with pytest.raises(TypeError, match="vector arguments"):
        Cross(p, v)


def test_cross_unknown_unknown_rejected():
    """Cross product of two unknowns is nonlinear; reject."""
    v = Unknown.Vector("v")
    w = Unknown.Vector("w")
    with pytest.raises(TypeError, match="nonlinear"):
        Cross(v, w)


def test_dot_vector_vector_is_scalar():
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    assert Dot(B0, v).is_scalar


def test_dot_tensor_vector_is_vector():
    T = Unknown.Tensor("T")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    assert Dot(T, B0).is_vector
    assert Dot(B0, T).is_vector


def test_dot_tensor_tensor_is_tensor():
    T = Unknown.Tensor("T")
    S = Background.Tensor("S", components={("r", "r"): r})
    assert Dot(T, S).is_tensor


def test_dot_scalar_raises():
    p = Unknown.Scalar("p")
    v = Unknown.Vector("v")
    with pytest.raises(TypeError, match="\\(scalar"):
        Dot(p, v)


# =============================================================================
# Algebra via operator overloading
# =============================================================================

def test_sum_same_rank():
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    s = v + B0
    assert s.is_vector
    assert s.is_linear


def test_sum_different_ranks_rejected():
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    with pytest.raises(TypeError, match="same rank"):
        v + p


def test_scalar_times_vector_overload():
    v = Unknown.Vector("v")
    result = 2 * v
    assert result.is_vector
    assert result.is_linear


def test_sympy_scalar_times_vector_overload():
    v = Unknown.Vector("v")
    result = r * v
    assert result.is_vector


def test_negation():
    v = Unknown.Vector("v")
    neg = -v
    assert neg.is_vector


def test_subtraction():
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    diff = v - B0
    assert diff.is_vector


def test_vector_times_vector_rejected():
    """v * w (raw multiplication) is ambiguous; user must use Dot or Cross."""
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    with pytest.raises(TypeError, match="Dot.*Cross"):
        v * B0


# =============================================================================
# Linearity tracking
# =============================================================================

def test_background_only_expression_is_background():
    rho = Background.Scalar("rho", expr=sp.exp(-r))
    e = 2 * rho
    assert e.is_background


def test_background_times_unknown_is_linear():
    rho = Background.Scalar("rho", expr=sp.exp(-r))
    v = Unknown.Vector("v")
    e = rho * v
    assert e.is_linear


def test_unknown_in_grad_is_linear():
    v = Unknown.Vector("v")
    e = Grad(v)
    assert e.is_linear


def test_cross_background_unknown_is_linear():
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    v = Unknown.Vector("v")
    assert Cross(B0, v).is_linear
    assert Cross(v, B0).is_linear


# =============================================================================
# Composite expressions (sanity)
# =============================================================================

def test_mhd_momentum_pattern():
    """Mock MHD momentum equation linear terms."""
    rho = Background.Scalar("rho", expr=sp.exp(-r))
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")

    # rho Dt(v) + Grad(p) - Cross(B0, Curl(v))
    momentum = rho * Dt(v) + Grad(p) - Cross(B0, Curl(v))
    assert momentum.is_vector
    assert momentum.is_linear


def test_continuity_pattern():
    """Continuity: Dt(rho_pert) + Div(rho * v) — linear in unknowns."""
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    rho_pert = Unknown.Scalar("rho_pert")
    v = Unknown.Vector("v")

    continuity = Dt(rho_pert) + Div(rho0 * v)
    assert continuity.is_scalar
    assert continuity.is_linear


# =============================================================================
# Operand order and division by background scalars
# =============================================================================

def test_background_scalar_factor_is_the_coefficient_on_either_side():
    """p * 2, S * N2 and Div(v) * rho0 build the same ScalarMultiple as
    the reversed order: the background factor is the coefficient."""
    from spdee.ir.expressions import ScalarMultiple, FieldRef
    p = Unknown.Scalar("p")
    v = Unknown.Vector("v")
    rho0 = Background.Scalar("rho0", 1 + r ** 2)
    for left, right in ((p * 2, 2 * p), (p * rho0, rho0 * p),
                        (Div(v) * rho0, rho0 * Div(v)), (Lap(p) * rho0, rho0 * Lap(p))):
        assert isinstance(left, ScalarMultiple) and isinstance(right, ScalarMultiple)
        assert left.scalar.is_background and right.scalar.is_background
        assert repr(left) == repr(right)
    # An unknown scalar times a background vector keeps its product form.
    B0 = Background.Vector("B0", v_r=r)
    e = p * B0
    assert isinstance(e, ScalarMultiple) and isinstance(e.scalar, FieldRef)
    assert e.scalar.field is p and e.operand.field is B0


def test_division_by_background_scalar_expression():
    """Division by a background scalar field or expression multiplies by
    its reciprocal, a background scalar leaf; 1/rho0 * x is accepted too."""
    from spdee.ir.expressions import ScalarMultiple, _BackgroundScalarValue
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", 1 + r ** 2)
    for e in (Grad(p) / rho0, Grad(p) / (2 * rho0), Grad(p) / (rho0 * 2),
              Grad(p) / (rho0 + 1), Grad(p) / (-rho0), Grad(p) / 2, 1 / rho0 * Grad(p),
              Grad(p) * (1 / rho0)):
        assert isinstance(e, ScalarMultiple) and e.scalar.is_background
        assert e.kind == "vector"
    e = Grad(p) / (2 * rho0)
    assert isinstance(e.scalar, _BackgroundScalarValue)
    assert sp.simplify(e.scalar.value - 1 / (2 * (1 + r ** 2))) == 0
    with pytest.raises(TypeError, match="involving unknowns"):
        Grad(p) / p
    with pytest.raises(TypeError, match="only scalar division"):
        Grad(p) / Background.Vector("B0", v_r=r)


def test_division_by_numeric_background_scalar_shares_one_reciprocal_field():
    """A numeric divisor becomes a reciprocal background field, the same
    object for every division by the same expression, so the system's
    background roster has one entry and its values are 1/rho0."""
    import numpy as np
    from spdee import Equation, EquationSystem
    from spdee.ir.backgrounds import radial_interpolator
    p = Unknown.Scalar("p")
    v = Unknown.Vector("v")
    rt = np.linspace(0.3, 1.2, 25)
    rho0 = Background.Scalar("rho0", radial_interpolator(rt, 1 + rt ** 2))
    system = EquationSystem([Equation(Grad(p) / rho0 + Lap(v) / rho0, name="a"),
                             Equation(Div(v) / (2 * rho0) + p / (2 * rho0), name="b")])
    names = sorted(system.background_names)
    assert names == ["1/((2)*(rho0))", "1/(rho0)"]
    R = np.array([[0.5, 1.0]])
    T = np.full_like(R, 0.3)
    comp = system.backgrounds["1/(rho0)"].component
    assert np.allclose(comp.evaluate_on_grid(R, T), 1 / (1 + R ** 2))
    comp2 = system.backgrounds["1/((2)*(rho0))"].component
    assert np.allclose(comp2.evaluate_on_grid(R, T), 0.5 / (1 + R ** 2))
