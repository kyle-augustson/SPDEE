"""String entry of equations: a problem entered as strings compiles to
matrices bit-identical to the object API on the collocated and staggered
discretizations, with symbolic parameters bound at compile time.
"""

import numpy as np
import pytest
import sympy as sp

import spdee
from spdee import (
    Background, Unknown, Grad, Div, Cross, Dot, Lap, Dt,
    Equation, BoundaryCondition, EquationSystem, Boundary, r_hat, r, theta,
)
from spdee.problem import Problem, parse_expression, boundary_location
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


def _object_system(Ek_value=None):
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Om = Background.Vector("Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta),
                           v_phi=sp.S.Zero)
    mom = Dt(v) + 2 * Cross(Om, v) + Grad(p)
    if Ek_value is not None:
        mom = mom - Ek_value * Lap(v)
    eqs = [Equation(mom, name="momentum"), Equation(Div(v), name="continuity")]
    bcs = [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
           BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro")]
    return EquationSystem(eqs, bcs, gauge_unknowns=["p"])


def _string_problem(with_Ek=False):
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("Omega", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    if with_Ek:
        pb.add_parameter("Ek", 1e-3)
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) - Ek*lap(v) = 0",
                        name="momentum")
    else:
        pb.add_equation("dt(v) + 2*cross(Omega, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="inner", name="vri")
    pb.add_bc("dot(r_hat, v) = 0", at="outer", name="vro")
    return pb


def _maxdiff(A, B):
    return abs(A - B).max() if (A - B).nnz else 0.0


GRID = make_shell_grid(N_r=8, N_theta=12, r_inner=0.5, r_outer=1.0)


# ---------------------------------------------------------------------------

def test_parse_lhs_minus_rhs():
    pb = Problem()
    v = pb.add_unknown("v", "vector")
    e1 = parse_expression("lap(v) = dt(v)", pb.namespace)
    e2 = Lap(v) - Dt(v)
    assert repr(e1) == repr(e2)


def test_bare_expression_and_zero_rhs_are_identical():
    pb = Problem()
    pb.add_unknown("p", "scalar")
    a = parse_expression("lap(p)", pb.namespace)
    b = parse_expression("lap(p) = 0", pb.namespace)
    c = parse_expression("lap(p) = 0.0", pb.namespace)
    assert repr(a) == repr(b) == repr(c)


def test_unknown_name_is_reported():
    pb = Problem()
    pb.add_unknown("v", "vector")
    with pytest.raises(NameError, match="unknown name 'w'"):
        pb.add_equation("lap(w) = 0")


def test_nonlinear_rejected():
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    with pytest.raises(TypeError):
        pb.add_equation("p*lap(v) = 0")


def test_boundary_aliases():
    assert boundary_location("outer") == Boundary.R_OUTER
    assert boundary_location("r_inner") == Boundary.R_INNER
    assert boundary_location("top") == Boundary.R_OUTER
    assert boundary_location(Boundary.THETA_NORTH) == Boundary.THETA_NORTH
    with pytest.raises(ValueError):
        boundary_location("sideways")


def test_reserved_names_protected():
    pb = Problem()
    with pytest.raises(ValueError):
        pb.add_unknown("grad", "scalar")


# ---------------------------------------------------------------------------
# Bit-identical compilation, collocated and staggered
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("m", [1, 2])
def test_collocated_matrices_bit_identical(m):
    ref = compile_problem(_object_system(), GRID, m=m, order_r=2, order_theta=2)
    got = _string_problem().compile(GRID, m=m, gauge_unknowns=["p"],
                                    order_r=2, order_theta=2)
    assert _maxdiff(ref.matrices.A_total, got.matrices.A_total) == 0.0
    assert _maxdiff(ref.matrices.B_total, got.matrices.B_total) == 0.0


def test_staggered_matrices_bit_identical():
    ref = compile_problem(_object_system(), GRID, m=2, discretization="staggered")
    got = _string_problem().compile(GRID, m=2, gauge_unknowns=["p"],
                                    discretization="staggered")
    K0, M0 = ref.assemble_evp_matrices()
    K1, M1 = got.assemble_evp_matrices()
    assert _maxdiff(K0, K1) == 0.0
    assert _maxdiff(M0, M1) == 0.0


def test_symbolic_parameter_bound_at_compile():
    ref = compile_problem(_object_system(Ek_value=1e-3), GRID, m=2,
                          order_r=2, order_theta=2)
    pb = _string_problem(with_Ek=True)
    got = pb.compile(GRID, m=2, gauge_unknowns=["p"], order_r=2, order_theta=2)
    assert _maxdiff(ref.matrices.A_total, got.matrices.A_total) == 0.0
    # rebind and recompile: a different Ek gives a different operator
    pb.set_parameter("Ek", 2e-3)
    got2 = pb.compile(GRID, m=2, gauge_unknowns=["p"], order_r=2, order_theta=2)
    assert _maxdiff(got.matrices.A_total, got2.matrices.A_total) > 0.0


def test_rossby_mode_from_strings():
    """End to end: the string-entered inertial system has the m=2 sectoral
    Rossby mode at omega/Omega0 = -2/(m+1) = -2/3 on the collocated grid."""
    grid = make_shell_grid(N_r=12, N_theta=32, r_inner=0.71, r_outer=0.985)
    cp = _string_problem().compile(grid, m=2, gauge_unknowns=["p"])
    # omega = -i lambda  =>  the shift for a target omega is sigma = +i omega
    omega_target = -2.0 / 3.0
    res = cp.solve_evp(k=8, sigma=1j * omega_target + 0.02)
    om = res.eigenvalues_to_omega(res.eigenvalues)
    om = om[np.isfinite(om)]
    assert np.min(np.abs(om.real - omega_target)) < 0.04


def test_public_export():
    assert spdee.Problem is Problem


def _walk(pb, m=1):
    from spdee.ir.walker import walk_equation_system
    return walk_equation_system(pb.build(), m)


@pytest.mark.parametrize("text", [
    "grad(p)/rho0 = 0", "1/rho0*grad(p) = 0", "grad(p)*(1/rho0) = 0",
    "(1/rho0)*grad(p) = 0",
])
def test_string_division_by_background_scalar(text):
    """grad(p)/rho0 and its spellings fold 1/rho0 into the gradient's
    coefficient."""
    pb = Problem()
    pb.add_unknown("p"); pb.add_background("rho0", expr="1 + r**2")
    pb.add_equation(text, name="e")
    (task,) = _walk(pb)
    assert task.kind == "grad_scalar"
    assert sp.simplify(task.coefficient.expr - 1 / (1 + pb.namespace["r"] ** 2)) == 0


def test_string_division_by_background_scalar_expression():
    """A compound background scalar divisor (2*rho0, Ek*rho0) works, the
    parameter staying symbolic in the reciprocal."""
    pb = Problem()
    pb.add_unknown("p"); pb.add_background("rho0", expr="1 + r**2")
    pb.add_parameter("Ek", 1e-3)
    pb.add_equation("grad(p)/(2*rho0) = 0", name="a")
    pb.add_equation("grad(p)/(Ek*rho0) = 0", name="b")
    a, b = _walk(pb)
    rr, Ek = pb.namespace["r"], pb.namespace["Ek"]
    assert sp.simplify(a.coefficient.expr - 1 / (2 * (1 + rr ** 2))) == 0
    assert sp.simplify(b.coefficient.expr - 1 / (Ek * (1 + rr ** 2))) == 0


@pytest.mark.parametrize("text, kind", [
    ("p*2 = 0", "identity_scalar"), ("2*p = 0", "identity_scalar"),
    ("p*N2 = 0", "identity_scalar"), ("N2*p = 0", "identity_scalar"),
    ("div(v)*N2 = 0", "div_vector_scalar"), ("N2*div(v) = 0", "div_vector_scalar"),
    ("lap(p)*N2 = 0", "lap_scalar"), ("N2*lap(p) = 0", "lap_scalar"),
])
def test_string_product_order_is_commutative(text, kind):
    """Every order of a background scalar factor walks to the same task."""
    pb = Problem()
    pb.add_unknown("p"); pb.add_unknown("v", "vector")
    pb.add_background("N2", expr="r")
    pb.add_equation(text, name="e")
    (task,) = _walk(pb)
    assert task.kind == kind
    want = 2 if text[:3] in ("p*2", "2*p") else pb.namespace["r"]
    assert task.coefficient.expr == want


def test_string_division_compiles_like_the_reciprocal_coefficient():
    """grad(p)/rho0 and (1/rho0)*grad(p) as a SymPy coefficient give the
    same collocated operator."""
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", 1 + r ** 2)
    ref = EquationSystem([Equation((1 / (1 + r ** 2)) * Grad(p), name="mom"),
                          Equation(Div(v), name="cont")], gauge_unknowns=["p"])
    pb = Problem(namespace={"rho0": rho0, "v": v, "p": p})
    pb.add_equation("grad(p)/rho0 = 0", name="mom")
    pb.add_equation("div(v) = 0", name="cont")
    A_ref = compile_problem(ref, GRID, m=1, order_r=2, order_theta=2).matrices.A_total
    A_got = pb.compile(GRID, m=1, gauge_unknowns=["p"], order_r=2,
                       order_theta=2).matrices.A_total
    assert _maxdiff(A_ref, A_got) == 0.0
