"""Tests for spdee.ir.walker (scalar-only first cut).

Verifies that the walker produces correct OperatorTask lists for various
scalar equation shapes. These tests do NOT involve the backend; they
inspect the task list directly.
"""

import sympy as sp
import pytest

from spdee import (
    Background, Unknown, r, theta,
    Lap, Dt, Equation, EquationSystem,
)
from spdee.ir.tasks import (
    TASK_IDENTITY_SCALAR, TASK_LAP_SCALAR, TASK_SOURCE_SCALAR,
    TASK_LAP_VECTOR, TASK_SOURCE_VECTOR, TASK_IDENTITY_VECTOR,
    TARGET_A, TARGET_B, TARGET_RHS,
)
from spdee.ir.walker import walk_equation, walk_equation_system
from spdee.ir.backgrounds import SymPyComponent


# ===========================================================================
# Basic identity / source
# ===========================================================================

def test_walk_identity_unknown_scalar():
    """Eq: p = 0 should produce one identity task on A."""
    p = Unknown.Scalar("p")
    eq = Equation(p, name="trivial")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_IDENTITY_SCALAR
    assert t.target == TARGET_A
    assert t.equation_name == "trivial"
    assert t.unknown_name == "p"
    assert t.sign == +1
    assert t.coefficient is None  # trivial coefficient
    assert t.metadata == {"m": 0}


def test_walk_negation_flips_sign():
    p = Unknown.Scalar("p")
    eq = Equation(-p, name="neg")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    assert tasks[0].sign == -1


def test_walk_double_negation_restores_sign():
    p = Unknown.Scalar("p")
    eq = Equation(-(-p), name="double_neg")
    tasks = walk_equation(eq, m=0)
    assert tasks[0].sign == +1


def test_walk_scalar_multiple_background_constant():
    """Eq: 2 * p = 0 produces one identity task with coefficient = 2."""
    p = Unknown.Scalar("p")
    eq = Equation(2 * p, name="scaled")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_IDENTITY_SCALAR
    assert t.coefficient is not None
    assert isinstance(t.coefficient, SymPyComponent)
    assert t.coefficient.expr == 2


def test_walk_scalar_multiple_background_field():
    """Eq: rho0 * p = 0 produces one identity task with coefficient = rho0_expr."""
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    eq = Equation(rho0 * p, name="scaled_bg")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_IDENTITY_SCALAR
    assert t.coefficient is not None
    assert t.coefficient.expr == sp.exp(-r)


def test_walk_pure_background_source():
    """Eq: f = 0 where f is a background scalar -> SOURCE task on b."""
    f = Background.Scalar("f", expr=sp.sin(theta))
    eq = Equation(f, name="source_only")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_SOURCE_SCALAR
    assert t.target == TARGET_RHS
    assert t.unknown_name is None
    assert t.coefficient is not None
    assert t.coefficient.expr == sp.sin(theta)


# ===========================================================================
# Lap operator
# ===========================================================================

def test_walk_lap_unknown_scalar():
    """Eq: Lap(p) = 0 produces one LAP_SCALAR task."""
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p), name="laplace")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_LAP_SCALAR
    assert t.target == TARGET_A
    assert t.unknown_name == "p"
    assert t.sign == +1
    assert t.coefficient is None


def test_walk_poisson_structure():
    """Eq: Lap(p) - f = 0 produces two tasks: LAP_SCALAR on A, SOURCE on b.

    Note: the source has sign -1 because of the subtraction. (The
    assembler will move that to the other side as +f.)
    """
    p = Unknown.Scalar("p")
    f = Background.Scalar("f", expr=sp.sin(theta))
    eq = Equation(Lap(p) - f, name="poisson")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 2

    kinds = sorted(t.kind for t in tasks)
    assert kinds == sorted([TASK_LAP_SCALAR, TASK_SOURCE_SCALAR])

    lap_task = next(t for t in tasks if t.kind == TASK_LAP_SCALAR)
    src_task = next(t for t in tasks if t.kind == TASK_SOURCE_SCALAR)

    assert lap_task.target == TARGET_A
    assert lap_task.unknown_name == "p"
    assert lap_task.sign == +1

    assert src_task.target == TARGET_RHS
    assert src_task.unknown_name is None
    assert src_task.sign == -1  # -f
    assert src_task.coefficient.expr == sp.sin(theta)


def test_walk_lap_with_background_coefficient():
    """Eq: rho0 * Lap(p) = 0 emits ONE LAP_SCALAR task with rho0 as coefficient."""
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    eq = Equation(rho0 * Lap(p), name="weighted")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.kind == TASK_LAP_SCALAR
    assert t.coefficient is not None
    assert t.coefficient.expr == sp.exp(-r)


def test_walk_nested_coefficients_compose():
    """Eq: 3 * (rho0 * Lap(p)) emits ONE task with coefficient = 3*rho0_expr."""
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    eq = Equation(3 * (rho0 * Lap(p)), name="nested")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    assert sp.simplify(tasks[0].coefficient.expr - 3 * sp.exp(-r)) == 0


# ===========================================================================
# Dt marker -> goes to B
# ===========================================================================

def test_walk_dt_unknown():
    """Eq: Dt(p) -> identity task on B."""
    p = Unknown.Scalar("p")
    eq = Equation(Dt(p), name="time_only")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    assert tasks[0].kind == TASK_IDENTITY_SCALAR
    assert tasks[0].target == TARGET_B
    assert tasks[0].unknown_name == "p"


def test_walk_dt_with_background_coefficient():
    """Eq: rho0 * Dt(p) -> one identity task on B with coefficient rho0."""
    p = Unknown.Scalar("p")
    rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
    eq = Equation(rho0 * Dt(p), name="time_with_bg")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.target == TARGET_B
    assert t.kind == TASK_IDENTITY_SCALAR
    assert t.coefficient.expr == sp.exp(-r)


def test_walk_eigenproblem_structure():
    """Eq: Dt(p) + Lap(p) = 0 emits TWO tasks: identity on B, lap on A.

    This is the canonical scalar-Laplacian eigenproblem
        lambda * p + Lap(p) = 0    (or  Lap(p) = -lambda * p)
    """
    p = Unknown.Scalar("p")
    eq = Equation(Dt(p) + Lap(p), name="diffusion_eigen")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 2
    target_to_kind = {t.target: t.kind for t in tasks}
    assert target_to_kind[TARGET_B] == TASK_IDENTITY_SCALAR
    assert target_to_kind[TARGET_A] == TASK_LAP_SCALAR


# ===========================================================================
# Multi-equation system
# ===========================================================================

def test_walk_system_with_multiple_equations():
    """Two scalar equations -> tasks tagged with each eq's name."""
    p = Unknown.Scalar("p")
    q = Unknown.Scalar("q")
    eq1 = Equation(Lap(p), name="poisson_p")
    eq2 = Equation(Lap(q), name="poisson_q")
    system = EquationSystem([eq1, eq2], [])
    tasks = walk_equation_system(system, m=0)
    eq_names = sorted(t.equation_name for t in tasks)
    assert eq_names == ["poisson_p", "poisson_q"]


# ===========================================================================
# Error handling
# ===========================================================================

def test_walker_handles_vector_lap_equation():
    """Vector Lap equation: Lap(v) - g = 0.
    Should emit two tasks: LAP_VECTOR on A, SOURCE_VECTOR on b.
    """
    v = Unknown.Vector("v")
    g = Background.Vector("g_force", v_r=1)
    eq = Equation(Lap(v) - g, name="vector_poisson")
    tasks = walk_equation(eq, m=0)
    assert len(tasks) == 2

    kinds = sorted(t.kind for t in tasks)
    assert kinds == sorted([TASK_LAP_VECTOR, TASK_SOURCE_VECTOR])

    lap_task = next(t for t in tasks if t.kind == TASK_LAP_VECTOR)
    src_task = next(t for t in tasks if t.kind == TASK_SOURCE_VECTOR)

    assert lap_task.target == TARGET_A
    assert lap_task.unknown_name == "v"
    assert lap_task.sign == +1
    assert lap_task.coefficient is None

    assert src_task.target == TARGET_RHS
    assert src_task.unknown_name is None
    assert src_task.sign == -1   # -g
    assert src_task.source_field is g
    assert src_task.coefficient is None


def test_walker_tensor_equation():
    """A tensor equation walks into the identity of its unknown tensor and
    the reduced components of its vector-unknown terms."""
    from spdee.ir.operators import Grad, Transpose
    from spdee.ir.tasks import TASK_IDENTITY_TENSOR, TASK_TENSOR_OF_VECTOR
    T_unknown = Unknown.Tensor("T")
    v = Unknown.Vector("v")
    eq = Equation(T_unknown - Grad(v) - Transpose(Grad(v)), name="tensor_eq")
    tasks = walk_equation(eq, m=1)
    assert [t.kind for t in tasks] == [TASK_IDENTITY_TENSOR, TASK_TENSOR_OF_VECTOR]
    assert [t.sign for t in tasks] == [+1, +1]
    assert tasks[1].metadata["tensor_symmetric"]
    assert tasks[1].metadata["rT_components"][(0, 0)] != 0


def test_walker_rejects_unnamed_equation():
    """Equation must have a name to dispatch."""
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p))  # no name
    with pytest.raises(ValueError, match="name"):
        walk_equation(eq, m=0)


def test_walker_rejects_nested_lap():
    """Lap of Lap is not yet supported (requires state augmentation)."""
    p = Unknown.Scalar("p")
    eq = Equation(Lap(Lap(p)), name="biharmonic")
    with pytest.raises(NotImplementedError, match="composed|FieldRef|nested"):
        walk_equation(eq, m=0)


def test_walker_rejects_dt_of_lap():
    """Dt(Lap(p)) is not yet supported."""
    p = Unknown.Scalar("p")
    eq = Equation(Dt(Lap(p)), name="dt_of_lap")
    with pytest.raises(NotImplementedError):
        walk_equation(eq, m=0)


def test_walker_passes_m_through_metadata():
    """The m value flows into every task's metadata."""
    p = Unknown.Scalar("p")
    eq = Equation(Lap(p), name="m_test")
    for m in (0, 1, 5):
        tasks = walk_equation(eq, m=m)
        for t in tasks:
            assert t.metadata["m"] == m


# ===========================================================================
# A background on the right of a rank-2 Dot
# ===========================================================================

def test_walker_canonicalises_dot_tensor_expr_bg_vector():
    """Dot(X, B0) with X a tensor expression of v is Dot(B0, Transpose(X))
    exactly, so the walker emits the left-form task on the transposed
    expression: the reduced components and the Grad/Transpose(Grad)
    weights are swapped and the task kind is the existing one."""
    from spdee import Grad, Dot, Transpose
    from spdee.ir.tasks import TASK_DOT_BG_VEC_TENSOR_UNK
    v = Unknown.Vector("v")
    B0 = Background.Vector("B0", v_r=r, v_theta=sp.cos(theta), v_phi=sp.S.Zero)
    X = 2 * Grad(v) + Transpose(Grad(v))
    right = walk_equation(Equation(Dot(X, B0), name="e"), m=1)
    left = walk_equation(Equation(Dot(B0, Transpose(X)), name="e"), m=1)
    plain = walk_equation(Equation(Dot(B0, X), name="e"), m=1)
    assert [t.kind for t in right] == [TASK_DOT_BG_VEC_TENSOR_UNK]
    assert right[0].source_field is B0
    assert right[0].metadata["grad_transpose_weights"] == (1, 2)
    rT_right = right[0].metadata["rT_components"]
    rT_left = left[0].metadata["rT_components"]
    rT_plain = plain[0].metadata["rT_components"]
    assert all(sp.expand(rT_left[k] - rT_right[k]) == 0 for k in rT_left)
    assert any(sp.expand(rT_plain[k] - rT_right[k]) != 0 for k in rT_left)


def test_walker_canonicalises_dot_vector_bg_tensor():
    """Dot(v, T0) = Dot(Transpose(T0), v): the emitted task is the
    existing Dot(bg_tensor, v) kind on a background whose physical
    components are swapped; a symmetric T0 is passed through as is."""
    from spdee import Dot
    from spdee.ir.tasks import TASK_DOT_BG_TENSOR_VEC
    from spdee.ir.backgrounds import get_sympy_expr
    v = Unknown.Vector("v")
    T0 = Background.Tensor("T0", components={("r", "theta"): r, ("phi", "r"): 3})
    tasks = walk_equation(Equation(Dot(v, T0), name="e"), m=0)
    assert [t.kind for t in tasks] == [TASK_DOT_BG_TENSOR_VEC]
    tt = tasks[0].source_field
    assert tt.is_background and tt is not T0
    assert get_sympy_expr(tt.components[("theta", "r")]) == r
    assert get_sympy_expr(tt.components[("r", "phi")]) == 3
    assert get_sympy_expr(tt.components[("r", "theta")]) == 0
    S0 = Background.Tensor("S0", components={("r", "theta"): r, ("theta", "r"): r},
                           symmetric=True)
    assert walk_equation(Equation(Dot(v, S0), name="e"), m=0)[0].source_field is S0


# ===========================================================================
# Dt of a background, parameter binding
# ===========================================================================

def test_dt_of_background_inside_sum_is_dropped_with_a_warning():
    """Dt(v + B0): d/dt B0 = 0, so only the identity task on B remains;
    a scalar Dt(s + s0) and a bare value behave the same."""
    from spdee import Background as Bg
    v = Unknown.Vector("v")
    s = Unknown.Scalar("s")
    B0 = Bg.Vector("B0", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=r)
    s0 = Bg.Scalar("s0", r ** 2)
    cases = [(Dt(v + B0), [(TASK_IDENTITY_VECTOR, TARGET_B)]),
             (Dt(v) + Dt(B0 + v), [(TASK_IDENTITY_VECTOR, TARGET_B)] * 2),
             (Dt(s + s0), [(TASK_IDENTITY_SCALAR, TARGET_B)]),
             (Dt(s + 2), [(TASK_IDENTITY_SCALAR, TARGET_B)])]
    for lhs, want in cases:
        with pytest.warns(UserWarning, match="time derivative of a background is zero"):
            tasks = walk_equation(Equation(lhs, name="e"), m=1)
        assert [(t.kind, t.target) for t in tasks] == want
    with pytest.raises(NotImplementedError, match="pure-background"):
        walk_equation(Equation(Dt(B0), name="e"), m=1)


def test_bind_parameters_matches_symbols_by_name():
    """A coefficient written with Symbol('Ek', positive=True) binds through
    the string key 'Ek' or a plain Symbol('Ek'), in the coefficient and in
    the metadata of composed tasks."""
    from spdee import Grad, Transpose, Div
    from spdee.ir.tasks import bind_parameters
    v = Unknown.Vector("v")
    Ek = sp.Symbol("Ek", positive=True)
    nu = sp.Symbol("nu")
    eq = Equation(Dt(v) - Ek * Lap(v) - nu * Div(Grad(v) + Transpose(Grad(v))), name="e")
    tasks = walk_equation(eq, m=1)
    for key in ("Ek", sp.Symbol("Ek"), Ek):
        bound = bind_parameters(tasks, {key: 1e-3, "nu": 2})
        assert bound[0].coefficient is None
        assert bound[1].coefficient == SymPyComponent(sp.Float(1e-3))
        assert bound[2].coefficient == SymPyComponent(sp.S(2))
        assert not any(sp.sympify(w).free_symbols
                       for w in bound[2].metadata["grad_transpose_weights"])


def test_bind_parameters_rejects_unbound_symbols():
    """A free symbol other than r and theta left in any task after binding
    is an error naming the symbol and the term."""
    from spdee.ir.tasks import bind_parameters
    v = Unknown.Vector("v")
    Ek = sp.Symbol("Ek", positive=True)
    tasks = walk_equation(Equation(Dt(v) - Ek * Lap(v), name="mom"), m=1)
    with pytest.raises(ValueError, match=r"unbound symbols \['Ek'\].*'lap_vector'.*'mom'"):
        bind_parameters(tasks, {})
    with pytest.raises(ValueError, match="unbound symbols"):
        bind_parameters(tasks, {"Pr": 1.0})
    bound = bind_parameters(tasks, {"Ek": 1e-3})
    assert bound[1].coefficient == SymPyComponent(sp.Float(1e-3))
    # Coordinates are never parameters.
    tasks = walk_equation(Equation(Dt(v) - (1 + r ** 2) * Lap(v), name="mom"), m=1)
    assert bind_parameters(tasks, {}) == tasks


# ===========================================================================
# Background parts of tensor expressions against the vector calculus
# ===========================================================================

def _spin_of_source(task):
    from test_tensor_reduce import contravariant_spin
    from analytic import vector_calculus as vc
    f = task.source_field
    return contravariant_spin(vc.Vec(f.v_r, f.v_theta, f.v_phi))


@pytest.mark.parametrize("m", [0, 1, 2])
def test_div_of_background_tensor_plus_grad_splits_into_operator_and_source(m):
    """Div(T0 + Grad(v)) with a phi-carrying T0: the reduced Div task is
    linear in v and matches Div(Grad(v)) at m, and the background part
    Div(T0) at m = 0 is a SOURCE_VECTOR."""
    from analytic import vector_calculus as vc
    from test_tensor_reduce import (
        analytic_backgrounds, contravariant_spin, substitute_trial, assert_pointwise_equal,
    )
    from spdee import Div, Grad
    from spdee.ir.tasks import TASK_DIV_TENSOR_VECTOR
    T0, T0v, _B0, _B0v, trial = analytic_backgrounds()
    v = Unknown.Vector("v")
    tasks = walk_equation(Equation(-Div(T0 + Grad(v)), name="e"), m=m)
    assert [t.kind for t in tasks] == [TASK_SOURCE_VECTOR, TASK_DIV_TENSOR_VECTOR]
    src, div = tasks
    assert src.sign == -1 and div.sign == -1 and src.target == TARGET_RHS
    md = div.metadata
    ph = md["v_placeholders"]["v"]
    want = contravariant_spin(vc.div(vc.grad(trial, m), m))
    want_src = contravariant_spin(vc.div(T0v, 0))
    spin_src = _spin_of_source(src)
    for sigma in (-1, 0, 1):
        e = md["per_sigma_expr_in_U"][sigma]
        for key, U in md["U_placeholders"].items():
            e = e.replace(U.func, sp.Lambda((r, theta), md["U_in_v"][key]))
        got = substitute_trial(e.doit(), ph, contravariant_spin(trial)) / r ** 2
        assert_pointwise_equal(got, want[sigma], sigma)
        assert_pointwise_equal(spin_src[sigma], want_src[sigma], sigma)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_dot_bg_vector_with_background_tensor_plus_strain(m):
    """Dot(B0, T0 + strain(v)): the contraction task is linear in v and
    Dot(B0, T0) is a SOURCE_VECTOR with the accumulated coefficient."""
    from analytic import vector_calculus as vc
    from test_tensor_reduce import (
        analytic_backgrounds, contravariant_spin, substitute_trial, assert_pointwise_equal,
    )
    from spdee import Dot, Grad, Transpose
    from spdee.ir.tasks import TASK_DOT_BG_VEC_TENSOR_UNK
    T0, T0v, B0, B0v, trial = analytic_backgrounds()
    v = Unknown.Vector("v")
    strain = Grad(v) + Transpose(Grad(v))
    tasks = walk_equation(Equation(2 * Dot(B0, T0 + strain), name="e"), m=m)
    assert [t.kind for t in tasks] == [TASK_SOURCE_VECTOR, TASK_DOT_BG_VEC_TENSOR_UNK]
    src, dot = tasks
    assert src.coefficient == SymPyComponent(sp.S(2)) and src.sign == +1
    G = vc.grad(trial, m)
    want = contravariant_spin(vc.dot(B0v, G + vc.transpose(G)))
    want_src = contravariant_spin(vc.dot(B0v, T0v))
    a_spin = B0.spin_components()
    rT = dot.metadata["rT_components"]
    ph = dot.metadata["v_placeholders"]["v"]
    spin_src = _spin_of_source(src)
    for s_out in (-1, 0, 1):
        contracted = sum(a_spin[s_in] * rT[(s_in, -s_out)] for s_in in (-1, 0, 1))
        got = substitute_trial(contracted, ph, contravariant_spin(trial)) / r
        assert_pointwise_equal(got, want[s_out], s_out)
        assert_pointwise_equal(spin_src[s_out], want_src[s_out], s_out)


@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("unit", ["r_hat", "theta_hat", "phi_hat"])
def test_tensor_projection_bc_moves_background_part_to_the_value(m, unit):
    """Dot(u, Dot(T0 + strain, r_hat)) = 0 at the wall: the constraint
    is linear in v and its value is -(u . T0 . r_hat)."""
    from analytic import vector_calculus as vc
    from test_tensor_reduce import (
        analytic_backgrounds, contravariant_spin, substitute_trial, assert_pointwise_equal,
    )
    from spdee import Div, Dot, Grad, Transpose, BoundaryCondition, Boundary
    from spdee.ir import equations as eqmod
    from spdee.ir.walker import walk_boundary_conditions
    T0, T0v, _B0, _B0v, trial = analytic_backgrounds()
    u = getattr(eqmod, unit)
    uv = {"r_hat": vc.R_HAT, "theta_hat": vc.THETA_HAT, "phi_hat": vc.PHI_HAT}[unit]
    v = Unknown.Vector("v")
    strain = Grad(v) + Transpose(Grad(v))
    bc = BoundaryCondition(Boundary.R_OUTER, Dot(u, Dot(T0 + strain, eqmod.r_hat)) - 3)
    system = EquationSystem([Equation(Div(strain), name="e")], [bc])
    (task,) = walk_boundary_conditions(system, m)
    got = substitute_trial(task.metadata["scalar_expr"], task.metadata["placeholders"]["v"],
                           contravariant_spin(trial))
    G = vc.grad(trial, m)
    assert_pointwise_equal(got, vc.dot(uv, vc.dot(G + vc.transpose(G), vc.R_HAT)))
    assert_pointwise_equal(task.value.expr, 3 - vc.dot(uv, vc.dot(T0v, vc.R_HAT)))


# ===========================================================================
# Composite boundary rows (BC_LINEAR_COMBINATION)
# ===========================================================================

from spdee.ir.backgrounds import NumericComponent  # noqa: E402
from spdee.ir.tasks import (  # noqa: E402
    BCTask, BC_DIRICHLET_SCALAR, BC_DIRICHLET_VECTOR_PROJECTION,
    BC_LINEAR_COMBINATION, BC_NEUMANN_SCALAR_RADIAL, BC_PROJECTION_CURL_CROSS,
    BC_SCALAR_LINEAR_FROM_TENSOR, BC_TANGENT_MIXED, BC_UNIT_PHI_HAT,
    BC_UNIT_R_HAT, BC_UNIT_THETA_HAT, bind_bc_parameters,
)
from spdee.ir.walker import boundary_pairing_key, walk_boundary_conditions  # noqa: E402
from spdee.problem import Problem  # noqa: E402

STRAIN = "dot(grad(v) + trans(grad(v)), r_hat)"


def bc_problem(*bcs, unknowns=(("p", "scalar"),), backgrounds=(), at="outer"):
    """Diffusion of every unknown with the boundary rows `bcs` at `at`;
    h, ls, c, Bi are symbolic parameters."""
    pb = Problem()
    for name, kind in unknowns:
        pb.add_unknown(name, kind)
    for name in ("h", "ls", "c", "Bi"):
        pb.add_parameter(name, 1.0)
    for name, expr in backgrounds:
        pb.add_background(name, expr=expr)
    pb.add_background("B0", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    for name, _kind in unknowns:
        pb.add_equation(f"dt({name}) - lap({name}) = 0", name=f"eq_{name}")
    for i, bc in enumerate(bcs):
        pb.add_bc(bc, at=at, name=f"bc{i}")
    return pb


def walk_bcs(*bcs, m=1, composite=True, **kw):
    return walk_boundary_conditions(bc_problem(*bcs, **kw).build(), m, composite=composite)


def terms_of(task):
    return [(c.expr, t.kind, t.unknown_name, t.unit, t.metadata.get("host_sigma"))
            for c, t in task.metadata["terms"]]


def dump_systems():
    """(name, system) of the composite-boundary systems for the IR dump."""
    h = sp.Symbol("h")
    yield "robin", bc_problem("dot(r_hat, grad(p)) + h*p = 0").build()
    yield "robin_spellings", bc_problem(
        "r**2*dot(r_hat, grad(p)) + (2*r**2 + 1)*p = 0",
        "f0*p + dot(r_hat, grad(p)) = 0",
        "-(dot(r_hat, grad(p)) + 3*p) = 0",
        "dot(grad(p), r_hat) - (-3)*p = 0",
        "dot(r_hat, grad(p)) + Bi*p = 1 + r",
        backgrounds=(("f0", "2*r**2 + 1"),)).build()
    yield "navier_slip", bc_problem(
        "dot(r_hat, v) = 0",
        f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0",
        f"dot(phi_hat, v) + ls*dot(phi_hat, {STRAIN}) = 0",
        unknowns=(("v", "vector"),)).build()
    yield "navier_slip_distributed", bc_problem(
        "dot(theta_hat, v + ls*dot(grad(v) + trans(grad(v)), r_hat)) = 0",
        "dot(v + ls*dot(grad(v) + trans(grad(v)), r_hat), phi_hat) = 0",
        unknowns=(("v", "vector"),)).build()
    yield "normal_stress", bc_problem(
        f"dot(r_hat, {STRAIN}) - c*p = 0", f"dot(theta_hat, {STRAIN}) = 0",
        f"dot(phi_hat, {STRAIN}) = 0",
        unknowns=(("v", "vector"), ("p", "scalar"))).build()
    yield "electric_plus_field", bc_problem(
        "dot(theta_hat, 0.01*curl(b) - cross(v, B0)) + h*dot(theta_hat, b) = 0",
        "dot(theta_hat, 0.01*curl(b) - cross(v, B0) + h*b) = 0",
        unknowns=(("v", "vector"), ("b", "vector"))).build()
    yield "mixed_rows", bc_problem(
        "dot(theta_hat, v) + dot(phi_hat, v) = 0", "dot(theta_hat, v) + p = 0",
        "2*dot(r_hat, v) = 3*p", unknowns=(("v", "vector"), ("p", "scalar"))).build()
    yield "single_normalized", bc_problem(
        "p + 1 + r = 0", "2*p = 3", "-(h*dot(r_hat, grad(p))) = 0",
        f"2*dot(r_hat, grad(p)) = {h}*(1 - r)").build()


# ---- composite rows through the string interface

def test_robin_row():
    (task,) = walk_bcs("dot(r_hat, grad(p)) + h*p = 0")
    assert task.kind == BC_LINEAR_COMBINATION
    assert task.unknown_name == "p" and task.value is None and task.unit is None
    assert terms_of(task) == [(1, BC_NEUMANN_SCALAR_RADIAL, "p", None, None),
                              (sp.Symbol("h"), BC_DIRICHLET_SCALAR, "p", None, None)]
    meta = task.metadata
    assert meta["unknowns"] == ("p",) and meta["host_unknown"] == "p"
    assert meta["host_sigma"] is None and meta["tangent"] is None
    assert meta["pairing_key"] == boundary_pairing_key(
        bc_problem("dot(r_hat, grad(p)) + h*p = 0").boundary_conditions[0])
    for _c, term in meta["terms"]:
        assert isinstance(term, BCTask) and term.location == task.location == "r=1"
        assert term.value is None and term.name == task.name


@pytest.mark.parametrize("bc,coefficients", [
    ("r**2*dot(r_hat, grad(p)) + (2*r**2 + 1)*p = 0", [r ** 2, 2 * r ** 2 + 1]),
    ("f0*p + dot(r_hat, grad(p)) = 0", [1, 2 * r ** 2 + 1]),
    ("-(dot(r_hat, grad(p)) + 3*p) = 0", [1, 3]),
    ("dot(grad(p), r_hat) - (-3)*p = 0", [1, 3]),
    ("dot(r_hat, grad(p)) + Bi*p = 0", [1, sp.Symbol("Bi")]),
], ids=["sympy_r", "background", "negated", "reversed", "biot"])
def test_robin_coefficient_spellings(bc, coefficients):
    """Every spelling gives the Neumann and Dirichlet terms with the
    coefficient as written, a background scalar folded to its SymPy
    expression, the outer negation stripped, the double sign resolved."""
    (task,) = walk_bcs(bc, backgrounds=(("f0", "2*r**2 + 1"),))
    kinds = {t.kind: c for c, t in task.metadata["terms"]}
    got = [kinds[BC_NEUMANN_SCALAR_RADIAL].expr, kinds[BC_DIRICHLET_SCALAR].expr]
    assert [sp.expand(g - w) for g, w in zip(got, coefficients)] == [0, 0]
    assert task.value is None


def test_inhomogeneous_composite_row_keeps_the_value():
    """X + g = 0 becomes sum c_i X_i = -g; the row's value is SymPy."""
    (task,) = walk_bcs("dot(r_hat, grad(p)) + Bi*p = 1 + r")
    assert task.kind == BC_LINEAR_COMBINATION
    assert sp.expand(task.value.expr - (1 + r)) == 0
    (task,) = walk_bcs("-(dot(r_hat, grad(p)) + Bi*p - 2) = 0")
    assert sp.expand(task.value.expr - 2) == 0


@pytest.mark.parametrize("m", [0, 1])
def test_navier_slip_pair(m):
    """A theta row and a phi row on the same expression: two records
    with the tangent of their unit, hosted on sigma = +1 and -1, sharing
    the pairing key; the strain term is the stress-free traction."""
    rows = walk_bcs("dot(r_hat, v) = 0",
                    f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0",
                    f"dot(phi_hat, v) + ls*dot(phi_hat, {STRAIN}) = 0",
                    m=m, unknowns=(("v", "vector"),))
    assert rows[0].kind == BC_DIRICHLET_VECTOR_PROJECTION and rows[0].unit == BC_UNIT_R_HAT
    theta_row, phi_row = rows[1:]
    ls = sp.Symbol("ls")
    for row, unit, sigma in ((theta_row, BC_UNIT_THETA_HAT, +1), (phi_row, BC_UNIT_PHI_HAT, -1)):
        assert row.kind == BC_LINEAR_COMBINATION and row.unknown_name == "v"
        assert terms_of(row) == [(1, BC_DIRICHLET_VECTOR_PROJECTION, "v", unit, None),
                                 (ls, BC_SCALAR_LINEAR_FROM_TENSOR, "v", None, sigma)]
        assert row.metadata["tangent"] == unit and row.metadata["host_sigma"] == sigma
        assert row.metadata["unknowns"] == ("v",)
    assert theta_row.metadata["pairing_key"] == phi_row.metadata["pairing_key"]
    (_c, strain), = [ct for ct in theta_row.metadata["terms"]
                     if ct[1].kind == BC_SCALAR_LINEAR_FROM_TENSOR]
    (reference,) = walk_bcs(f"dot(theta_hat, {STRAIN}) = 0", m=m, unknowns=(("v", "vector"),))
    assert sp.expand(strain.metadata["scalar_expr"] - reference.metadata["scalar_expr"]) == 0


def test_unit_projection_distributes_over_a_tree():
    """Dot(t_hat, v + ls Dot(strain, r_hat)) is the same row as the
    written pair, on either side of the Dot, with the coefficient
    inside the projection folded into the term."""
    (row,) = walk_bcs("dot(theta_hat, v + ls*dot(grad(v) + trans(grad(v)), r_hat)) = 0",
                      unknowns=(("v", "vector"),))
    (reference,) = walk_bcs(f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0",
                            unknowns=(("v", "vector"),))
    assert terms_of(row) == terms_of(reference)
    (row,) = walk_bcs("dot(v + ls*dot(grad(v) + trans(grad(v)), r_hat), phi_hat) = 0",
                      unknowns=(("v", "vector"),))
    assert row.metadata["tangent"] == BC_UNIT_PHI_HAT
    assert [k for _e, k, *_ in terms_of(row)] == [BC_DIRICHLET_VECTOR_PROJECTION,
                                                  BC_SCALAR_LINEAR_FROM_TENSOR]
    (row,) = walk_bcs("dot(theta_hat, -(2*v) + 3*(v + v)) = 0", unknowns=(("v", "vector"),))
    assert [c for c, *_ in terms_of(row)] == [-2, 3, 3]


@pytest.mark.parametrize("sign", ["-", "+"])
def test_normal_stress_row_names_two_unknowns(sign):
    """dot(r_hat, dot(strain, r_hat)) -/+ c p = 0: the record names v and
    p, hosts on v (the higher rank) at sigma = 0 with no tangent."""
    (row,) = walk_bcs(f"dot(r_hat, {STRAIN}) {sign} c*p = 0",
                      unknowns=(("v", "vector"), ("p", "scalar")))
    c = sp.Symbol("c")
    assert row.kind == BC_LINEAR_COMBINATION and row.unknown_name == "v"
    assert terms_of(row) == [(1, BC_SCALAR_LINEAR_FROM_TENSOR, "v", None, 0),
                             (-c if sign == "-" else c, BC_DIRICHLET_SCALAR, "p", None, None)]
    assert row.metadata["unknowns"] == ("v", "p")
    assert row.metadata["host_unknown"] == "v" and row.metadata["host_sigma"] == 0
    assert row.metadata["tangent"] is None
    (row,) = walk_bcs(f"c*p - dot(r_hat, {STRAIN}) = 0",
                      unknowns=(("v", "vector"), ("p", "scalar")))
    assert row.metadata["unknowns"] == ("p", "v") and row.unknown_name == "v"


def test_tensor_projection_background_part_moves_to_the_row_value():
    """A background tensor inside a composite row's projection moves to
    the value scaled by the term's coefficient."""
    pb = bc_problem(unknowns=(("v", "vector"), ("p", "scalar")))
    pb.add_background("T0", components={("r", "r"): "r"}, symmetric=True)
    pb.add_bc("2*dot(r_hat, dot(T0 + grad(v) + trans(grad(v)), r_hat)) - c*p = 0", at="outer")
    (row,) = walk_boundary_conditions(pb.build(), 1, composite=True)
    assert row.kind == BC_LINEAR_COMBINATION
    assert sp.expand(row.value.expr + 2 * r) == 0
    assert all(t.value is None for _c, t in row.metadata["terms"])


def test_electric_field_plus_a_field_term():
    """A curl-cross projection beside a projection of the field itself:
    two terms, the unknowns of the cross term included, hosted on the
    first unknown seen; the distributed spelling gives the same terms."""
    (row, distributed) = walk_bcs(
        "dot(theta_hat, 0.01*curl(b) - cross(v, B0)) + h*dot(theta_hat, b) = 0",
        "dot(theta_hat, 0.01*curl(b) - cross(v, B0) + h*b) = 0",
        unknowns=(("v", "vector"), ("b", "vector")))
    assert [k for _c, k, *_ in terms_of(row)] == [BC_PROJECTION_CURL_CROSS,
                                                  BC_DIRICHLET_VECTOR_PROJECTION]
    assert row.metadata["unknowns"] == ("b", "v") and row.unknown_name == "b"
    assert row.metadata["tangent"] == BC_UNIT_THETA_HAT
    assert terms_of(distributed) == terms_of(row)


def test_mixed_tangential_and_scalar_rows():
    """theta_hat and phi_hat in one row is a mixed tangent hosted on
    sigma = +1; a tangential projection beside a scalar keeps its unit."""
    mixed, scalar = walk_bcs("dot(theta_hat, v) + dot(phi_hat, v) = 0",
                             "dot(theta_hat, v) + p = 0",
                             unknowns=(("v", "vector"), ("p", "scalar")))
    assert mixed.metadata["tangent"] == BC_TANGENT_MIXED and mixed.metadata["host_sigma"] == 1
    assert scalar.metadata["tangent"] == BC_UNIT_THETA_HAT
    assert scalar.metadata["unknowns"] == ("v", "p") and scalar.unknown_name == "v"


def test_pairing_key_makes_theta_and_phi_alike():
    pb = bc_problem(f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0",
                    f"dot(phi_hat, v) + ls*dot(phi_hat, {STRAIN}) = 0",
                    "dot(r_hat, v) = 0", "dot(theta_hat, v) = 0",
                    unknowns=(("v", "vector"),))
    keys = [boundary_pairing_key(bc) for bc in pb.boundary_conditions]
    assert keys[0] == keys[1] and keys[0][0] == "r=1"
    assert keys[2] != keys[3] and "tangent_hat" in keys[3][1]
    inner = bc_problem("dot(theta_hat, v) = 0", unknowns=(("v", "vector"),), at="inner")
    assert boundary_pairing_key(inner.boundary_conditions[0]) != keys[3]


def test_numeric_coefficient_is_carried():
    """A numeric background coefficient stays a component; the value
    must be SymPy."""
    pb = bc_problem(unknowns=(("p", "scalar"),))
    pb.add_background("g0", expr=NumericComponent(lambda R, T: 2 + 0 * R))
    pb.add_bc("dot(r_hat, grad(p)) + g0*p = 0", at="outer")
    (row,) = walk_boundary_conditions(pb.build(), 1, composite=True)
    assert isinstance(row.metadata["terms"][1][0], NumericComponent)
    pb.add_bc("dot(r_hat, grad(p)) + g0*p = g0", at="outer")
    with pytest.raises(NotImplementedError, match="non-SymPy"):
        walk_boundary_conditions(pb.build(), 1, composite=True)


def test_bind_bc_parameters():
    rows = walk_bcs("dot(r_hat, grad(p)) + h*p = h", "p = 0")
    bound = bind_bc_parameters(rows, {"h": 3.0})
    assert bound[1] is rows[1]
    assert bound[0].value.expr == 3.0
    assert bound[0].metadata["terms"][1][0].expr == 3.0
    assert bound[0].metadata["terms"][1][1] is rows[0].metadata["terms"][1][1]
    with pytest.raises(ValueError, match="unbound symbols \\['h'\\]"):
        bind_bc_parameters(rows, {})


# ---- single-shape rows keep their kinds

@pytest.mark.parametrize("bc,kind,value", [
    ("p = 0", BC_DIRICHLET_SCALAR, None),
    ("p = 2", BC_DIRICHLET_SCALAR, 2),
    ("-p - 1 = 0", BC_DIRICHLET_SCALAR, -1),
    ("p + 1 + r = 0", BC_DIRICHLET_SCALAR, -1 - r),
    ("2*p = 3", BC_DIRICHLET_SCALAR, sp.Rational(3, 2)),
    ("h*p = 0", BC_DIRICHLET_SCALAR, None),
    ("dot(r_hat, grad(p)) = 0", BC_NEUMANN_SCALAR_RADIAL, None),
    ("-(h*dot(r_hat, grad(p))) = 0", BC_NEUMANN_SCALAR_RADIAL, None),
    ("2*dot(r_hat, grad(p)) = h*(1 - r)", BC_NEUMANN_SCALAR_RADIAL,
     sp.Symbol("h") * (1 - r) / 2),
])
def test_single_shape_rows_keep_their_kind(bc, kind, value):
    """One unknown term, whatever wraps it, is its own kind with the
    value divided by the coefficient."""
    (task,) = walk_bcs(bc)
    assert task.kind == kind and task.metadata == {}
    if value is None:
        assert task.value is None
    else:
        assert sp.expand(task.value.expr - value) == 0


def test_single_vector_shapes_keep_their_kind():
    rows = walk_bcs("dot(r_hat, v) = 0", "2*dot(theta_hat, v) = 0",
                    f"dot(phi_hat, {STRAIN}) = 0", f"-dot(r_hat, {STRAIN}) = 0",
                    "dot(theta_hat, 0.01*curl(b) - cross(v, B0)) = 0",
                    unknowns=(("v", "vector"), ("b", "vector")))
    assert [t.kind for t in rows] == [
        BC_DIRICHLET_VECTOR_PROJECTION, BC_DIRICHLET_VECTOR_PROJECTION,
        BC_SCALAR_LINEAR_FROM_TENSOR, BC_SCALAR_LINEAR_FROM_TENSOR, BC_PROJECTION_CURL_CROSS]
    assert rows[1].unit == BC_UNIT_THETA_HAT and rows[1].value is None


def test_single_shape_rows_are_byte_identical():
    """Single-shape rows built with the object API, with and without the
    composite flag, give the same records field by field."""
    from spdee import Dot, Grad, Transpose, BoundaryCondition, Boundary, Lap
    from spdee.ir.equations import r_hat, theta_hat
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    strain = Grad(v) + Transpose(Grad(v))
    bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v) - 2, name="a"),
           BoundaryCondition(Boundary.R_OUTER, -(p + 3), name="b"),
           BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, Dot(strain, r_hat)), name="c"),
           BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, Grad(p)) + 1, name="d")]
    system = EquationSystem([Equation(Dt(v) - Lap(v), name="v"),
                             Equation(Dt(p) - Lap(p), name="p")], bcs)
    rows = walk_boundary_conditions(system, 1, composite=True)
    assert rows == walk_boundary_conditions(system, 1)
    assert [t.kind for t in rows] == [BC_DIRICHLET_VECTOR_PROJECTION, BC_DIRICHLET_SCALAR,
                                      BC_SCALAR_LINEAR_FROM_TENSOR, BC_NEUMANN_SCALAR_RADIAL]
    assert rows[0].value.expr == 2 and rows[1].value.expr == -3 and rows[3].value.expr == -1
    assert rows[2].value is None and rows[2].metadata["host_sigma"] == 1
    assert all(t.unit is None for t in rows[1:])


# ---- refusals

@pytest.mark.parametrize("bc,fragment", [
    ("dot(r_hat, grad(p)) + lap(p) = 0", "term Lap"),
    ("dot(theta_hat, curl(curl(v))) + p = 0", "term Dot"),
    ("h*lap(p) = 0", "term Lap"),
    ("dot(r_hat, v) + dot(B0, v) = 0", "term Dot(FieldRef('B0')"),
    ("dot(theta_hat, v + B0) = 0", "projects the background"),
])
def test_composite_row_refusals_name_the_term(bc, fragment):
    """A term that is none of the five shapes is refused naming it."""
    with pytest.raises(NotImplementedError, match="bc0") as info:
        walk_bcs(bc, unknowns=(("v", "vector"), ("p", "scalar")))
    assert fragment in str(info.value)


def test_row_without_an_unknown_is_refused():
    with pytest.raises(NotImplementedError, match="neither side"):
        walk_bcs("f0 + 1 = 0", backgrounds=(("f0", "r"),))
    with pytest.raises(NotImplementedError, match="pure background"):
        walk_bcs("f0 = 0", backgrounds=(("f0", "r"),))


def test_single_shape_refusals_are_unchanged():
    with pytest.raises(NotImplementedError, match="Dot arguments must be bare FieldRefs"):
        walk_bcs("dot(theta_hat, curl(curl(v))) = 0", unknowns=(("v", "vector"),))
    with pytest.raises(NotImplementedError, match="not recognized"):
        walk_bcs("h*r + 1 = 0")


@pytest.mark.parametrize("bc,fragment", [
    ("dot(r_hat, grad(p)) + h*p = 0", "both sides of the Sum contain unknowns"),
    (f"dot(theta_hat, v) + ls*dot(theta_hat, {STRAIN}) = 0", "both sides of the Sum"),
    ("dot(theta_hat, v + ls*dot(grad(v) + trans(grad(v)), r_hat)) = 0",
     "Dot arguments must be bare FieldRefs"),
    ("2*p = 0", "not recognized"),
])
def test_composite_rows_are_refused_without_the_flag(bc, fragment):
    """The cores that consume the single-shape kinds only walk without
    `composite` and see the refusals they saw before."""
    with pytest.raises(NotImplementedError, match="bc0") as info:
        walk_bcs(bc, composite=False, unknowns=(("v", "vector"), ("p", "scalar")))
    assert fragment in str(info.value)
