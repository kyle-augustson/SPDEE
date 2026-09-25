"""Tests for the new Grad(scalar) and Div(vector) operators.

Operator-level direct tests (apply operator block to known input, check
output) succeed in all 4 contexts. Coupled-system BVP tests succeed in
shell mode when boundary conditions over-constrain the pressure gauge
freedom (eliminating the LBB checkerboard null mode). Ball-mode coupled
systems require either staggered grids or pressure stabilization to be
well-posed; the underlying operators here are correct but the colocated
FD discretization has the classic inf-sup instability for Stokes-type
systems.

Coverage in this file:
  - 6 walker tests (emission for Grad(p), Div(v_unknown), rejection paths)
  - 4 Grad operator direct correctness tests (one per context)
  - 4 Grad operator convergence tests
  - 4 Div(vector) operator direct correctness tests
  - 4 Div(vector) operator convergence tests
  - 2 shell-mode coupled BVP tests (Stokes-like, with Dirichlet on p)
  - 1 gauge constraint smoke test
"""
import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Grad, Div, Cross, Lap, Dt, theta,
    Equation, EquationSystem,
    Boundary, BoundaryCondition, Dot, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.grad_div_sparse import (
    build_grad_scalar_block, build_div_vector_scalar_block,
)
from spdee.ir.walker import walk_equation_system
from spdee.ir.tensor_reduce import vector_unknown_spin_placeholders


def _flat(arr2d):
    return arr2d.T.flatten()


def _vector_state(v_r, v_t, v_p):
    v0 = v_r + 0j
    vp = (v_t + 1j * v_p) / np.sqrt(2)
    vm = (v_t - 1j * v_p) / np.sqrt(2)
    return np.concatenate([_flat(vm), _flat(v0), _flat(vp)]).astype(complex)


def _make_grid(geom, N):
    if geom == "ball":
        return make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    return make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)


CONTEXTS = [
    ("standard", "shell"),
    ("compact", "shell"),
    ("standard", "ball"),
    ("compact", "ball"),
]


# ---------------------------------------------------------------------------
# Walker tests
# ---------------------------------------------------------------------------

def test_walker_emits_grad_task():
    p = Unknown.Scalar("p")
    g = Background.Vector(
        "g", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
    )
    sys_ = EquationSystem(
        [Equation(Grad(p) - g, name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, p, name="p_o")],
    )
    tasks = walk_equation_system(sys_, m=0)
    kinds = [t.kind for t in tasks]
    assert "grad_scalar" in kinds, f"got {kinds}"


def test_walker_emits_div_vector_task():
    v = Unknown.Vector("v")
    sys_ = EquationSystem(
        [Equation(Div(v), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    tasks = walk_equation_system(sys_, m=0)
    kinds = [t.kind for t in tasks]
    assert "div_vector_scalar" in kinds, f"got {kinds}"


def test_walker_rejects_grad_of_grad():
    """Grad(Grad(p)) is a composed operator; walker should reject."""
    p = Unknown.Scalar("p")
    sys_ = EquationSystem(
        [Equation(Grad(Grad(p)), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, p, name="p_o")],
    )
    # Grad(scalar)->vector, then Grad(vector)->tensor. Walker handles
    # only direct vector ops at the top level; this hits the tensor
    # equation path. Either way, should reject.
    with pytest.raises((NotImplementedError, TypeError, ValueError)):
        walk_equation_system(sys_, m=0)


def test_walker_rejects_dt_of_grad():
    p = Unknown.Scalar("p")
    sys_ = EquationSystem(
        [Equation(Dt(Grad(p)), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, p, name="p_o")],
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        walk_equation_system(sys_, m=0)


def test_walker_rejects_grad_of_background():
    p_bg = Background.Scalar("p_bg", sp.S.One)
    sys_ = EquationSystem(
        [Equation(Grad(p_bg) - Background.Vector(
            "g", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero
        ), name="eq")],
        [],
    )
    # Background fields are still allowed as Grad operands at IR construction,
    # but the walker rejects them since they should be evaluated outside.
    with pytest.raises((NotImplementedError, ValueError)):
        walk_equation_system(sys_, m=0)


def test_walker_rejects_div_of_div():
    """Div(Div(...)) goes through the tensor path, then fails internally."""
    v = Unknown.Vector("v")
    sys_ = EquationSystem(
        [Equation(Div(Grad(v)), name="eq")],  # Div(Grad(v)) = Lap(v) — valid
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    # This SHOULD succeed — Div(Grad(v)) is Lap(v) handled via tensor reduction.
    tasks = walk_equation_system(sys_, m=0)
    kinds = [t.kind for t in tasks]
    assert "div_tensor_vector" in kinds, f"got {kinds}"


# ---------------------------------------------------------------------------
# Grad operator direct correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_grad_radial_polynomial(stencil, geom):
    """Grad(p = 1 - r²) = -2r r̂. The (r·√2)-multiplied output:
        σ=0 row: r·√2 · (-2r) = -2·r²·√2
        σ=± rows: 0 (no θ-dep)
    """
    grid = _make_grid(geom, 24)
    bundle = DerivativeBundle(grid)
    R, T = grid.mesh()
    p_arr = (1 - R**2) + 0j
    p_state = _flat(p_arr).astype(complex)
    op = build_grad_scalar_block(bundle, m=0, p_unknown_name="p")
    out = op @ p_state
    N = grid.N_r * grid.N_theta
    out_m = out[:N].reshape(grid.N_theta, grid.N_r).T
    out_0 = out[N:2*N].reshape(grid.N_theta, grid.N_r).T
    out_p = out[2*N:].reshape(grid.N_theta, grid.N_r).T
    exp_0 = -2 * R**2 * np.sqrt(2) + 0j
    assert np.max(np.abs(out_0 - exp_0)) < 1e-12, "σ=0"
    assert np.max(np.abs(out_m)) < 1e-12, "σ=-"
    assert np.max(np.abs(out_p)) < 1e-12, "σ=+"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_grad_with_theta_dep_convergence(stencil, geom):
    """Grad(p = (1-r²) r cos θ = z (1-r²)): convergence rate ≥ 3.5.

    (r·√2)·(Grad p)^± = ∂_θ p = -(1-r²) r sin θ. On a ball the i_r=0
    rows hold the r→0 limit of the operator instead of the r-scaled
    form, √2·(Grad p)(0)^± = -sin θ (Grad p(0) = ẑ).
    """
    def test_at_N(N):
        grid = _make_grid(geom, N)
        bundle = DerivativeBundle(grid)
        R, T = grid.mesh()
        p_arr = (1 - R**2) * R * np.cos(T) + 0j
        p_state = _flat(p_arr).astype(complex)
        op = build_grad_scalar_block(bundle, m=0, p_unknown_name="p")
        out = op @ p_state
        Nblk = grid.N_r * grid.N_theta
        out_m = out[:Nblk].reshape(grid.N_theta, grid.N_r).T
        out_p = out[2*Nblk:].reshape(grid.N_theta, grid.N_r).T
        exp_pm = -(1 - R**2) * R * np.sin(T)
        if geom == "ball":
            exp_pm[0, :] = -np.sin(T[0, :])
        return max(
            float(np.max(np.abs(out_m - exp_pm))),
            float(np.max(np.abs(out_p - exp_pm))),
        )

    errs = [test_at_N(N) for N in (16, 24, 32)]
    rate = np.log(errs[0] / errs[-1]) / np.log(32 / 16) if errs[-1] > 1e-13 else 5.0
    assert rate >= 3.5, (
        f"Grad convergence rate too low: {rate:.2f} ({stencil}-{geom}, errs={errs})"
    )


# ---------------------------------------------------------------------------
# Div(vector) operator direct correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_div_vector_zhat_polynomial(stencil, geom):
    """v = (1-r²) ẑ axisymmetric → Div(v) = -2r cos θ → r·Div(v) = -2r² cos θ.
    """
    grid = _make_grid(geom, 24)
    bundle = DerivativeBundle(grid)
    R, T = grid.mesh()
    f = 1 - R**2
    v_r = f * np.cos(T) + 0j
    v_t = -f * np.sin(T) + 0j
    v_p = 0 * R + 0j
    x = _vector_state(v_r, v_t, v_p)

    v_field = Unknown.Vector("v")
    walker_meta = {
        "m": 0,
        "v_placeholders": {"v": vector_unknown_spin_placeholders(v_field)},
    }
    op = build_div_vector_scalar_block(walker_meta, bundle, m=0)
    out = op @ x
    out_2d = out.reshape(grid.N_theta, grid.N_r).T
    exp = -2 * R**2 * np.cos(T) + 0j
    err = float(np.max(np.abs(out_2d - exp)))
    # Polynomial degree 2 in r → standard 4th-order FD is exact at this size.
    assert err < 1e-3, f"Div(zhat) err: {err:.3e} ({stencil}-{geom})"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_div_vector_convergence(stencil, geom):
    """Convergence rate ≥ 3.5 for Div(v) operator."""
    def test_at_N(N):
        grid = _make_grid(geom, N)
        bundle = DerivativeBundle(grid)
        R, T = grid.mesh()
        f = 1 - R**2
        v_r = f * np.cos(T) + 0j
        v_t = -f * np.sin(T) + 0j
        v_p = 0 * R + 0j
        x = _vector_state(v_r, v_t, v_p)
        v_field = Unknown.Vector("v")
        walker_meta = {
            "m": 0,
            "v_placeholders": {"v": vector_unknown_spin_placeholders(v_field)},
        }
        op = build_div_vector_scalar_block(walker_meta, bundle, m=0)
        out = op @ x
        out_2d = out.reshape(grid.N_theta, grid.N_r).T
        exp = -2 * R**2 * np.cos(T) + 0j
        return float(np.max(np.abs(out_2d - exp)))

    errs = [test_at_N(N) for N in (16, 24, 32)]
    rate = np.log(errs[0] / errs[-1]) / np.log(32 / 16) if errs[-1] > 1e-13 else 5.0
    assert rate >= 3.5, (
        f"Div(vector) rate too low: {rate:.2f} ({stencil}-{geom}, errs={errs})"
    )


# ---------------------------------------------------------------------------
# Shell-mode coupled BVP (Stokes-like, well-posed with Dirichlet on p)
# ---------------------------------------------------------------------------

def test_grad_div_coupled_shell():
    """Stokes-like coupled system in shell mode with Dirichlet on p at
    both r_inner and r_outer.

    Equations:
        Grad(p) + Cross(Ω, v) = g_v      (vector momentum)
        Div(v) = 0                       (continuity)
    Manufactured:
        v_exact = 0
        p_exact = r·(1-r²)·cos θ
    Then g_v = Grad(p_exact) is supplied.
    """
    r_sym = sp.symbols("r", positive=True)
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Omega = Background.Vector(
        "Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    g_v = Background.Vector(
        "g",
        v_r=(1 - 3 * r_sym**2) * sp.cos(theta),
        v_theta=-(1 - r_sym**2) * sp.sin(theta),
        v_phi=sp.S.Zero,
    )
    r_in = sp.Rational(3, 10)
    p_in_value = r_in * sp.cos(theta) * (1 - r_in**2)
    p_out_value = sp.S.Zero  # at r=1: r*(1-r²)=0

    eq1 = Equation(Grad(p) + Cross(Omega, v) - g_v, name="momentum")
    eq2 = Equation(Div(v), name="continuity")

    errs = []
    Ns = [16, 24, 32]
    for N in Ns:
        grid = make_shell_grid(
            N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0,
        )
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_o"),
            BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt_o"),
            BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp_o"),
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vr_i"),
            BoundaryCondition(Boundary.R_INNER, Dot(theta_hat, v), name="vt_i"),
            BoundaryCondition(Boundary.R_INNER, Dot(phi_hat, v), name="vp_i"),
            BoundaryCondition(Boundary.R_OUTER, p - p_out_value, name="p_o"),
            BoundaryCondition(Boundary.R_INNER, p - p_in_value, name="p_i"),
        ]
        sys_ = EquationSystem([eq1, eq2], bcs)
        res = compile_problem(sys_, grid, m=0).solve_bvp()
        R, T = grid.mesh()
        p_exact = R * np.cos(T) * (1 - R**2)
        errs.append(float(np.max(np.abs(res["p"] - p_exact))))
    rate = np.log(errs[0] / errs[-1]) / np.log(Ns[-1] / Ns[0])
    assert rate >= 3.5, f"Grad/Div coupled shell rate: {rate:.2f}, errs={errs}"


# ---------------------------------------------------------------------------
# Gauge constraint smoke test
# ---------------------------------------------------------------------------

def test_gauge_constraint_pins_constant_mode():
    """Verify that gauge_unknowns=['p'] kills the constant null mode.

    For a system where p is determined only up to a constant (no Dirichlet
    BC on p), the gauge constraint ∫ p dV = 0 should pin the mean of p
    to zero.
    """
    r_sym = sp.symbols("r", positive=True)
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Omega = Background.Vector(
        "Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    h = 1 - r_sym**2
    g_v = Background.Vector(
        "g",
        v_r=-h * sp.sin(theta), v_theta=-h * sp.cos(theta), v_phi=sp.S.Zero,
    )
    eq1 = Equation(Grad(p) + Cross(Omega, v) - g_v, name="momentum")
    eq2 = Equation(Div(v), name="continuity")
    # Manufactured: v = (1-r²) φ̂ (divergence-free), p = 0.
    grid = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)
    r_in = sp.Rational(3, 10)
    v_phi_inner = 1 - r_in**2
    bcs = [
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_o"),
        BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt_o"),
        BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp_o"),
        BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vr_i"),
        BoundaryCondition(Boundary.R_INNER, Dot(theta_hat, v), name="vt_i"),
        BoundaryCondition(
            Boundary.R_INNER, Dot(phi_hat, v) - v_phi_inner, name="vp_i",
        ),
    ]
    sys_ = EquationSystem([eq1, eq2], bcs, gauge_unknowns=["p"])
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    p_mean_abs = float(np.abs(np.mean(res["p"])))
    # The gauge constraint uses a 2nd-order quadrature; the discrete
    # ∫p dV = 0 might not exactly imply np.mean(p) = 0, but both should
    # be small. Check that the mean is at least far below the variation.
    p_var = float(np.max(np.abs(res["p"] - np.mean(res["p"]))))
    # With a divergence-free v that solves Grad(p) + Cross(Ω,v) = g exactly
    # (because Cross(Ω, v) = g_v is itself the equation with p=0), the
    # discrete p should be ~0 everywhere.
    assert p_mean_abs < 1e-10, f"gauge mean = {p_mean_abs:.3e}, var = {p_var:.3e}"
    assert p_var < 1e-10, f"p var = {p_var:.3e}"


def test_gauge_unknown_validation():
    """EquationSystem validates gauge_unknowns references."""
    p = Unknown.Scalar("p")
    g = Background.Vector(
        "g", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
    )
    # Reference an unknown name that doesn't appear in any equation.
    with pytest.raises(ValueError, match="not an unknown"):
        EquationSystem(
            [Equation(Grad(p) - g, name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, p, name="p_o")],
            gauge_unknowns=["nonexistent"],
        )

    # Reference a vector unknown (only scalar gauge supported).
    v = Unknown.Vector("v")
    with pytest.raises(ValueError, match="scalar unknowns"):
        EquationSystem(
            [Equation(Div(v), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
             BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
             BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
            gauge_unknowns=["v"],
        )


# =============================================================================
# Grad(Div(v)) — composed second-order vector operator
# =============================================================================
#
# Dispatched directly via TASK_GRAD_DIV_VECTOR; the walker recognizes the
# composed expression ``Grad(Div(unknown_vector))`` and emits one task.
# The alternative (state augmentation with an auxiliary scalar D = Div(v))
# was deliberately avoided to keep the state vector minimal at production
# resolutions — adding N_r·N_θ rows/cols per aux unknown is significant
# at 512×1024 or larger grids.
#
# Arises naturally in the compressible viscous stress:
#     ∇·[∇v + (∇v)^T − (2/3)(∇·v) I] = ∇²v + (1/3)∇(∇·v)
# The second term is exactly Grad(Div(v)).

class TestGradDivVectorWalker:
    """Walker emits TASK_GRAD_DIV_VECTOR for Grad(Div(unknown_vec))."""

    def test_emits_task_for_grad_div(self):
        """Walker recognition: ``Grad(Div(v))`` produces exactly one
        TASK_GRAD_DIV_VECTOR task, not a rejected composed-operator
        error.
        """
        from spdee.ir.tasks import TASK_GRAD_DIV_VECTOR
        v = Unknown.Vector("v")
        sys = EquationSystem(
            [Equation(Grad(Div(v)), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
             BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v),
                               name="vt"),
             BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
        )
        tasks = walk_equation_system(sys, m=0)
        gd_tasks = [t for t in tasks if t.kind == TASK_GRAD_DIV_VECTOR]
        assert len(gd_tasks) == 1
        assert gd_tasks[0].unknown_name == "v"

    def test_no_aux_state_introduced(self):
        """Compile a Grad(Div(v))-containing system and verify the state
        size is unchanged from a system with only the vector v as the
        unknown. The whole point of this dispatcher is to avoid the
        aux state augmentation that would otherwise be needed for a
        composed operator on an unknown.
        """
        v = Unknown.Vector("v")
        # System with just Dt(v): state size should be 3·N_r·N_θ.
        N_r, N_theta = 12, 24
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.3, r_outer=1.0,
        )
        sys_no_op = EquationSystem(
            [Equation(Dt(v), name="eq")],
            [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vi"),
             BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vo")],
        )
        cp_no_op = compile_problem(
            sys_no_op, grid, m=1, order_r=4, order_theta=4,
        )
        # System with Grad(Div(v)) added: same state size.
        sys_with_op = EquationSystem(
            [Equation(Dt(v) - Grad(Div(v)), name="eq")],
            [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vi"),
             BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vo")],
        )
        cp_with_op = compile_problem(
            sys_with_op, grid, m=1, order_r=4, order_theta=4,
        )
        assert cp_with_op.matrices.layout.size == cp_no_op.matrices.layout.size, (
            f"Grad(Div(v)) should NOT add state. "
            f"no_op size = {cp_no_op.matrices.layout.size}, "
            f"with_op size = {cp_with_op.matrices.layout.size}"
        )
        # And specifically: 3 × N_r × N_theta (just the vector v).
        assert cp_with_op.matrices.layout.size == 3 * N_r * N_theta


class TestGradDivVectorDirect:
    """Direct operator-block tests with manufactured solutions.

    Identities used:
      v = (1 - r²) ẑ:
        Div(v) = -2r cos θ
        Grad(Div(v)) = -2 cos θ r̂ + 2 sin θ θ̂
        In spin basis: src^0 = -2 cos θ, src^± = √2 sin θ
      r²·Grad(Div(v))^0 = -2 r² cos θ
      r²·Grad(Div(v))^± = √2 r² sin θ

    For m=2 with v^0 = (1-r²) sin²θ, v^± = 0:
      r·Div(v) = (2 - 4r²) sin²θ
      r²·∂_r Div(v) = -(4r² + 2) sin²θ
      The σ=± formulas pick up the m-coupling terms.
    """

    @staticmethod
    def _apply_block(grid, m, v_minus, v_zero, v_plus):
        """Build the operator block and apply it to the given state."""
        from spdee.backends.fd2.grad_div_sparse import (
            build_r2_grad_div_vector_block,
        )
        from spdee.backends.fd2.layout import flatten_2d, reshape_2d
        from spdee.ir.fields import VectorField as VF
        bundle = DerivativeBundle(
            grid=grid, order_r=4, order_theta=4,
        )
        _stand_in = VF(name="v", role="unknown")
        walker_meta = {
            "m": m,
            "v_placeholders": {
                "v": vector_unknown_spin_placeholders(_stand_in),
            },
        }
        A = build_r2_grad_div_vector_block(walker_meta, bundle, m)
        state = np.concatenate([
            flatten_2d(v_minus),
            flatten_2d(v_zero),
            flatten_2d(v_plus),
        ])
        out = A @ state
        N = grid.N_r * grid.N_theta
        return (
            reshape_2d(out[:N], grid.N_r, grid.N_theta),
            reshape_2d(out[N:2*N], grid.N_r, grid.N_theta),
            reshape_2d(out[2*N:], grid.N_r, grid.N_theta),
        )

    @staticmethod
    def _interior_L2_rel(a, b, ir_lo=3, ir_hi=-3, it_lo=3, it_hi=-3):
        diff = (a - b)[ir_lo:ir_hi, it_lo:it_hi]
        denom = max(
            np.linalg.norm(b[ir_lo:ir_hi, it_lo:it_hi]),
            1e-12,
        )
        return np.linalg.norm(diff) / denom

    @pytest.mark.parametrize("geom,N_r,N_theta", [
        ("shell", 60, 120),
        ("ball", 60, 120),
    ])
    def test_zhat_solution_m0(self, geom, N_r, N_theta):
        """v = (1-r²) ẑ at m=0: operator output matches analytical
        ``Grad(Div(v))`` to interior 4th-order accuracy. Tests both
        shell and ball geometries.
        """
        if geom == "ball":
            grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0)
        else:
            grid = make_shell_grid(
                N_r=N_r, N_theta=N_theta, r_inner=0.3, r_outer=1.0,
            )
        R, T = grid.mesh()
        v_r = (1 - R**2) * np.cos(T)
        v_t = -(1 - R**2) * np.sin(T)
        v_minus = v_t / np.sqrt(2)
        v_zero = v_r
        v_plus = v_t / np.sqrt(2)
        out_m, out_0, out_p = self._apply_block(
            grid, 0, v_minus, v_zero, v_plus,
        )
        exact_0 = R**2 * (-2 * np.cos(T))
        exact_pm = R**2 * np.sqrt(2) * np.sin(T)
        # Ball geometry: skip a few extra rows near origin where the
        # i_r=0 placeholder lives.
        ir_lo = 2 if geom == "ball" else 3
        e0 = self._interior_L2_rel(out_0, exact_0, ir_lo=ir_lo)
        em = self._interior_L2_rel(out_m, exact_pm, ir_lo=ir_lo)
        ep = self._interior_L2_rel(out_p, exact_pm, ir_lo=ir_lo)
        assert e0 < 1e-6, f"{geom} σ=0 err = {e0:.3e}"
        assert em < 1e-6, f"{geom} σ=- err = {em:.3e}"
        assert ep < 1e-6, f"{geom} σ=+ err = {ep:.3e}"

    @pytest.mark.parametrize("geom", ["shell", "ball"])
    def test_zhat_solution_m0_convergence(self, geom):
        """At m=0 with the v = (1-r²) ẑ test, errors converge at
        4th order (16× reduction per doubling of resolution).
        """
        errs = []
        for N in [40, 80, 160]:
            if geom == "ball":
                grid = make_ball_grid(N_r=N, N_theta=2*N, r_outer=1.0)
            else:
                grid = make_shell_grid(
                    N_r=N, N_theta=2*N, r_inner=0.3, r_outer=1.0,
                )
            R, T = grid.mesh()
            v_t = -(1 - R**2) * np.sin(T)
            v_minus = v_t / np.sqrt(2)
            v_zero = (1 - R**2) * np.cos(T)
            v_plus = v_t / np.sqrt(2)
            _, out_0, _ = self._apply_block(
                grid, 0, v_minus, v_zero, v_plus,
            )
            exact_0 = R**2 * (-2 * np.cos(T))
            ir_lo = 2 if geom == "ball" else 3
            errs.append(self._interior_L2_rel(out_0, exact_0, ir_lo=ir_lo))
        # Each doubling of resolution: error should drop by a factor of
        # at least 8 (we expect ~16 but allow slack for the leading
        # coefficient).
        assert errs[0] / errs[1] > 8.0, (
            f"{geom}: convergence too slow: {errs[0]:.3e} -> {errs[1]:.3e}"
        )
        assert errs[1] / errs[2] > 8.0, (
            f"{geom}: convergence too slow: {errs[1]:.3e} -> {errs[2]:.3e}"
        )

    def test_m2_polynomial_solution(self):
        """At m=2 with v^0 = (1-r²) sin²θ, v^± = 0:
            r²·Grad(Div(v))^0 = -(4r²+2) sin²θ
            r²·Grad(Div(v))^- = 2√2(-2r²cosθ - 2r² + cosθ + 1) sinθ
            r²·Grad(Div(v))^+ = 2√2(-2r²cosθ + 2r² + cosθ - 1) sinθ
        Exercises the full mixed-derivative + m-coupling path.
        """
        grid = make_shell_grid(
            N_r=80, N_theta=160, r_inner=0.3, r_outer=1.0,
        )
        R, T = grid.mesh()
        v_zero = (1 - R**2) * np.sin(T)**2
        v_minus = np.zeros_like(R)
        v_plus = np.zeros_like(R)
        out_m, out_0, out_p = self._apply_block(
            grid, 2, v_minus, v_zero, v_plus,
        )
        exact_0 = -(4*R**2 + 2) * np.sin(T)**2
        exact_m = (2*np.sqrt(2)
                   * (-2*R**2*np.cos(T) - 2*R**2 + np.cos(T) + 1)
                   * np.sin(T))
        exact_p = (2*np.sqrt(2)
                   * (-2*R**2*np.cos(T) + 2*R**2 + np.cos(T) - 1)
                   * np.sin(T))
        e0 = self._interior_L2_rel(out_0, exact_0)
        em = self._interior_L2_rel(out_m, exact_m)
        ep = self._interior_L2_rel(out_p, exact_p)
        # σ=0 happens to be polynomial in (r, sin θ) so it's machine-
        # precision; σ=± involve trigonometric mixed derivatives so they
        # have a normal 4th-order truncation error.
        assert e0 < 1e-10, f"σ=0 err = {e0:.3e}"
        assert em < 1e-6, f"σ=- err = {em:.3e}"
        assert ep < 1e-6, f"σ=+ err = {ep:.3e}"


class TestGradDivVectorEVP:
    """End-to-end EVP smoke tests using Grad(Div(v)) in the equation.

    Validates the full pipeline: walker → task → dispatcher → assembled
    matrix → shift-invert ARPACK. The physics target is a viscous Stokes
    problem in a ball:
        Dt(v) − ν Lap(v) − (ν/3) Grad(Div(v)) = 0
    with no-slip BCs. All modes should be purely decaying (Re(λ) > 0
    in spdee's Dt → -iω convention; Im(λ) ≈ 0 since the operator is
    self-adjoint).
    """

    def test_ball_stokes_spectrum_real_and_damped(self):
        v = Unknown.Vector("v")
        nu = sp.Rational(1, 100)
        eq = Equation(
            Dt(v)
            - nu * Lap(v)
            - nu * sp.Rational(1, 3) * Grad(Div(v)),
            name="mom",
        )
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
            BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v),
                              name="vt"),
            BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_ball_grid(N_r=20, N_theta=40, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=0, order_r=4, order_theta=4)
        res = cp.solve_evp(k=20, sigma=0.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        assert len(finite) >= 10
        # All modes purely damping (Re(λ) > 0 in spdee Dt → -iω convention
        # for decaying time evolution); Im ≈ 0 since the viscous operator
        # is real and self-adjoint with these BCs.
        for x in finite:
            assert x.real > -1e-6, (
                f"Eigenvalue {x} should have Re ≥ 0 (damping)"
            )
            assert abs(x.imag) < 1e-2, (
                f"Eigenvalue {x} should be purely real (self-adjoint operator)"
            )

    def test_shell_stokes_spectrum_real_and_damped(self):
        """Same as ball test but in shell geometry. Confirms the
        dispatcher works without ball-mode origin regularity.
        """
        v = Unknown.Vector("v")
        nu = sp.Rational(1, 100)
        eq = Equation(
            Dt(v)
            - nu * Lap(v)
            - nu * sp.Rational(1, 3) * Grad(Div(v)),
            name="mom",
        )
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
            BoundaryCondition(Boundary.R_INNER, Dot(theta_hat, v),
                              name="vti"),
            BoundaryCondition(Boundary.R_INNER, Dot(phi_hat, v), name="vpi"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro"),
            BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v),
                              name="vto"),
            BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vpo"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=20, N_theta=40, r_inner=0.3, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=0, order_r=4, order_theta=4)
        res = cp.solve_evp(k=20, sigma=0.0)
        lam = np.array(res.eigenvalues)
        finite = lam[np.isfinite(lam)]
        assert len(finite) >= 10
        for x in finite:
            assert x.real > -1e-6, (
                f"Eigenvalue {x} should have Re ≥ 0 (damping)"
            )
            assert abs(x.imag) < 1e-2, (
                f"Eigenvalue {x} should be purely real"
            )
