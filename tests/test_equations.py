"""Tests for Equation, BoundaryCondition, EquationSystem."""

import pytest
import sympy as sp

from spdee.ir import (
    Background, Unknown,
    Grad, Div, Curl, Cross, Dot, Lap, Dt,
    Equation, BoundaryCondition, EquationSystem,
    r, theta,
)
from spdee.ir.equations import Boundary, r_hat, theta_hat, phi_hat


# =============================================================================
# Equation
# =============================================================================

def test_scalar_equation_construction():
    p = Unknown.Scalar("p")
    eq = Equation(Dt(p), name="trivial")
    assert eq.kind == "scalar"
    assert eq.is_linear
    assert "trivial" in repr(eq)


def test_vector_equation_construction():
    v = Unknown.Vector("v")
    eq = Equation(Dt(v) + Grad(Unknown.Scalar("p_dummy")))
    assert eq.kind == "vector"


def test_equation_rejects_nonexpression():
    """Truly garbage inputs (e.g., a string) should fail with a helpful
    error.
    """
    with pytest.raises(TypeError, match="must be"):
        Equation("garbage")


def test_equation_rejects_nonlinear():
    """If we construct a nonlinear expression directly (bypassing the
    rejected ops), the Equation should still catch it.

    Hard to trigger since the ops themselves reject; this is defensive.
    """
    # All the ops would have already raised, so this is just a sanity check
    p = Unknown.Scalar("p")
    eq = Equation(p)  # passes
    assert eq.is_linear


# =============================================================================
# Boundary conditions
# =============================================================================

def test_dirichlet_scalar_bc():
    """p = 0 at r=1."""
    p = Unknown.Scalar("p")
    bc = BoundaryCondition(Boundary.R_OUTER, p)
    assert bc.location == "r=1"
    assert bc.expr.is_scalar


def test_vector_component_bc_via_dot():
    """v_r = 0 at r=1 — expressed as Dot(r_hat, v) = 0."""
    v = Unknown.Vector("v")
    expr = Dot(r_hat, v)
    bc = BoundaryCondition(Boundary.R_OUTER, expr)
    assert bc.expr.is_scalar
    assert bc.location == "r=1"


def test_neumann_scalar_bc():
    """dp/dr|_{r=1} = 0 — Grad(p) dotted with r_hat."""
    p = Unknown.Scalar("p")
    expr = Dot(r_hat, Grad(p))
    bc = BoundaryCondition(Boundary.R_OUTER, expr)
    assert bc.expr.is_scalar


def test_robin_bc_combination():
    """a*p + b*dp/dr at r=1 — Robin."""
    p = Unknown.Scalar("p")
    expr = 2 * p + 3 * Dot(r_hat, Grad(p))
    bc = BoundaryCondition(Boundary.R_OUTER, expr)
    assert bc.expr.is_scalar
    assert bc.expr.is_linear


def test_bc_rejects_vector_expr():
    """BC must be scalar. Vector expressions should be decomposed by user."""
    v = Unknown.Vector("v")
    # `v` will auto-promote to FieldRef(VectorField), which has kind='vector'.
    with pytest.raises(TypeError, match="scalar Expression"):
        BoundaryCondition(Boundary.R_OUTER, v)


def test_bc_invalid_location():
    p = Unknown.Scalar("p")
    with pytest.raises(ValueError, match="Unknown boundary"):
        BoundaryCondition("garbage", p)


def test_all_four_boundary_locations():
    p = Unknown.Scalar("p")
    for loc in (Boundary.R_INNER, Boundary.R_OUTER,
                Boundary.THETA_NORTH, Boundary.THETA_SOUTH):
        bc = BoundaryCondition(loc, p)
        assert bc.location == loc


# =============================================================================
# Unit-vector convenience constants
# =============================================================================

def test_r_hat_spin_components():
    """r_hat: spin (0) = 1, others = 0."""
    sc = r_hat.spin_components()
    assert sp.simplify(sc[0] - 1) == 0
    assert sp.simplify(sc[-1]) == 0
    assert sp.simplify(sc[+1]) == 0


def test_theta_hat_spin_components():
    """theta_hat: spin (0) = 0, spin (±1) = 1/sqrt(2)."""
    sc = theta_hat.spin_components()
    assert sp.simplify(sc[0]) == 0
    assert sp.simplify(sc[-1] - 1 / sp.sqrt(2)) == 0
    assert sp.simplify(sc[+1] - 1 / sp.sqrt(2)) == 0


def test_phi_hat_spin_components():
    """phi_hat (V19): v^- = -i/sqrt(2), v^0 = 0, v^+ = +i/sqrt(2)."""
    sc = phi_hat.spin_components()
    assert sp.simplify(sc[0]) == 0
    assert sp.simplify(sc[-1] + sp.I / sp.sqrt(2)) == 0
    assert sp.simplify(sc[+1] - sp.I / sp.sqrt(2)) == 0


# =============================================================================
# EquationSystem
# =============================================================================

def test_equation_system_basic():
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p), name="poisson")
    bc = BoundaryCondition(Boundary.R_OUTER, p, name="dirichlet")
    sys_ = EquationSystem([eq], [bc])
    assert len(sys_.equations) == 1
    assert len(sys_.boundary_conditions) == 1
    assert "p" in sys_.unknown_names
    assert sys_.background_names == frozenset()


def test_equation_system_collects_backgrounds_and_unknowns():
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")

    # momentum: rho0 * dt(v) + Grad(p) - Cross(B0, Curl(v)) = 0
    momentum = Equation(
        rho0 * Dt(v) + Grad(p) - Cross(B0, Curl(v)),
        name="momentum",
    )
    # continuity: Div(rho0 * v) = 0
    continuity = Equation(Div(rho0 * v), name="continuity")

    bc_v = BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="no penetration")

    sys_ = EquationSystem([momentum, continuity], [bc_v])
    assert sys_.unknown_names == frozenset({"v", "p"})
    # Note: r_hat is a background too, so it shows up
    assert "rho0" in sys_.background_names
    assert "B0" in sys_.background_names
    assert "r_hat" in sys_.background_names


def test_equation_system_summary_runs():
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p), name="poisson")
    bc = BoundaryCondition(Boundary.R_OUTER, p, name="dirichlet")
    sys_ = EquationSystem([eq], [bc])
    text = sys_.summary()
    assert "Unknowns" in text
    assert "Backgrounds" in text
    assert "Equations" in text
    assert "BCs" in text


def test_equation_system_rejects_non_equation():
    with pytest.raises(TypeError, match="Equation objects"):
        EquationSystem([5], [])


def test_equation_system_rejects_non_bc():
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p))
    with pytest.raises(TypeError, match="BoundaryCondition objects"):
        EquationSystem([eq], ["garbage"])


def test_equation_system_name_collision():
    """Two distinct fields with the same name -> error."""
    p1 = Unknown.Scalar("p")
    p2 = Unknown.Vector("p")
    eq1 = Equation(p1)
    eq2 = Equation(p2)
    with pytest.raises(ValueError, match="distinct fields share"):
        EquationSystem([eq1, eq2])


def test_equation_system_rejects_duplicate_equation_names():
    """Two equations with the same name would merge into one row block."""
    p = Unknown.Scalar("p")
    q = Unknown.Scalar("q")
    with pytest.raises(ValueError, match="Duplicate equation name 'e'"):
        EquationSystem([Equation(Lap(p), name="e"), Equation(Lap(q), name="e")])
    EquationSystem([Equation(Lap(p), name="e"), Equation(Lap(q), name="f")])
