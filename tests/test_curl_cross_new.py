"""Tests for the new Cross and Curl operators across the four
(stencil, geometry) contexts.

Cross is pointwise (no derivatives), so the operator block should be
exact (machine precision) at all contexts.

Curl is first-order. The operator block alone is rate-4 accurate when
tested directly against manufactured solutions (the actual `Curl(v) = g`
BVP is rank-deficient because Curl has a non-trivial nullspace =
gradient fields; solving it requires a gauge condition or an additional
operator, which is multi-differential-operator territory not yet
supported by the compiler).

The walker correctly emits TASK_CURL_VECTOR / TASK_CROSS_VECTOR tasks
and rejects nonlinear / composed forms.
"""
import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Cross, Curl, Dt, theta,
    Equation, EquationSystem,
    Boundary, BoundaryCondition, Dot, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.curl_cross_sparse import (
    build_curl_vector_block, build_cross_vector_block,
)
from spdee.ir.walker import walk_equation_system
from spdee.ir.tensor_reduce import vector_unknown_spin_placeholders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(arr2d):
    """Flatten 2D (N_r, N_theta) → 1D, theta-major (Fortran)."""
    return arr2d.T.flatten()


def _vector_state(v_r, v_t, v_p):
    """Pack physical (r, θ, φ) components into V19 spin state."""
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

def test_walker_emits_curl_task():
    v = Unknown.Vector("v")
    g = Background.Vector(
        "g", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    sys_ = EquationSystem(
        [Equation(Curl(v) - g, name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    tasks = walk_equation_system(sys_, m=0)
    kinds = [t.kind for t in tasks]
    assert "curl_vector" in kinds, f"got kinds={kinds}"


def test_walker_emits_cross_task_left():
    v = Unknown.Vector("v")
    b = Background.Vector("b", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero)
    g = Background.Vector(
        "g", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
    )
    sys_ = EquationSystem(
        [Equation(Cross(b, v) - g, name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    tasks = walk_equation_system(sys_, m=0)
    cross_tasks = [t for t in tasks if t.kind == "cross_vector"]
    assert len(cross_tasks) == 1
    assert cross_tasks[0].metadata.get("side") == "left"


def test_walker_emits_cross_task_right():
    v = Unknown.Vector("v")
    b = Background.Vector("b", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero)
    g = Background.Vector(
        "g", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
    )
    sys_ = EquationSystem(
        [Equation(Cross(v, b) - g, name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    tasks = walk_equation_system(sys_, m=0)
    cross_tasks = [t for t in tasks if t.kind == "cross_vector"]
    assert len(cross_tasks) == 1
    assert cross_tasks[0].metadata.get("side") == "right"


def _curl_curl_system():
    v = Unknown.Vector("v")
    return EquationSystem(
        [Equation(Curl(Curl(v)), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )


def test_walker_emits_curl_of_curl():
    """Curl(Curl(v)) is the resistive term of an induction equation and the
    current of a vector potential; the walker emits it as one task."""
    from spdee.ir.tasks import TASK_CURL_CURL_VECTOR
    tasks = walk_equation_system(_curl_curl_system(), m=0)
    assert [t.kind for t in tasks] == [TASK_CURL_CURL_VECTOR]
    assert tasks[0].unknown_name == "v"


@pytest.mark.parametrize("geometry", ["shell", "ball"])
@pytest.mark.parametrize("m", [0, 1, 2])
def test_collocated_core_builds_curl_of_curl(geometry, m):
    """The collocated core builds Curl(Curl(v)) as the Curl formula
    composed with itself at r-power 2, and the block equals
    Grad(Div(v)) − Lap(v) entrywise: in the interior, at the walls and
    on the ball origin rows, since the three r²-multiplied spin
    expressions are identical term by term."""
    from spdee.backends.fd2.curl_cross_sparse import (
        build_r2_curl_curl_vector_block,
    )
    from spdee.backends.fd2.grad_div_sparse import build_r2_grad_div_vector_block
    from spdee.backends.fd2.lap_sparse import build_r2_lap_vector_block
    if geometry == "ball":
        grid = make_ball_grid(N_r=10, N_theta=12)
    else:
        grid = make_shell_grid(N_r=10, N_theta=12, r_inner=0.3, r_outer=1.0)
    bundle = DerivativeBundle(grid, order_r=4, order_theta=4)
    v = Unknown.Vector("v")
    meta = {"v_placeholders": {"v": vector_unknown_spin_placeholders(v)}}
    cc = build_r2_curl_curl_vector_block(meta, bundle, m)
    gd = build_r2_grad_div_vector_block(meta, bundle, m)
    lap = build_r2_lap_vector_block(meta, bundle, m)
    diff = (cc - (gd - lap)).tocsr()
    scale = float(abs(cc).max())
    assert (float(abs(diff).max()) if diff.nnz else 0.0) < 1e-14 * scale
    cp = compile_problem(_curl_curl_system(), grid, m=m)
    assert cp.matrices.A_total.nnz == cc.nnz


def test_walker_accepts_composed_cross_curl():
    """Cross(b, Curl(v)) is dispatched aux-free by the walker: it emits
    a TASK_CROSS_CURL_UNK_BG task rather than rejecting.
    """
    from spdee.ir.tasks import TASK_CROSS_CURL_UNK_BG
    v = Unknown.Vector("v")
    b = Background.Vector("b", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero)
    sys_ = EquationSystem(
        [Equation(Cross(b, Curl(v)), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    tasks = walk_equation_system(sys_, m=0)
    kinds = [t.kind for t in tasks]
    assert TASK_CROSS_CURL_UNK_BG in kinds, (
        f"Expected the composed task kind; got {kinds}"
    )


def test_walker_rejects_dt_of_curl():
    v = Unknown.Vector("v")
    sys_ = EquationSystem(
        [Equation(Dt(Curl(v)), name="eq")],
        [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
         BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt"),
         BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp")],
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        walk_equation_system(sys_, m=0)


# ---------------------------------------------------------------------------
# Cross operator direct correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_cross_zhat_shat_equals_phihat(stencil, geom):
    """Cross(ẑ, ŝ) = φ̂. Pointwise — should be exact at all contexts.
    ẑ = cos θ r̂ - sin θ θ̂.  ŝ = sin θ r̂ + cos θ θ̂.  φ̂ = φ̂.
    """
    grid = _make_grid(geom, 24)
    bundle = DerivativeBundle(grid)
    R, T = grid.mesh()

    # b = ẑ in V19 spin.
    b_spin = {
        -1: (-np.sin(T) - 1j * 0) / np.sqrt(2) + 0 * R,
         0: np.cos(T) + 0 * R,
        +1: (-np.sin(T) + 1j * 0) / np.sqrt(2) + 0 * R,
    }
    # Convert each to complex to match the helper signature.
    b_spin = {s: arr.astype(complex) for s, arr in b_spin.items()}

    op = build_cross_vector_block(b_spin, bundle, m=0, side="left")

    # v = ŝ.
    x = _vector_state(np.sin(T) + 0 * R, np.cos(T) + 0 * R, 0 * R + 0j)
    out = op @ x

    # Expected: φ̂ = (0, 0, 1) in physical, so v^0=0, v^±=±i/√2.
    exp = _vector_state(0 * R + 0j, 0 * R + 0j, 1 + 0 * R + 0j)
    err = float(np.max(np.abs(out - exp)))
    assert err < 1e-13, f"Cross(ẑ, ŝ) error: {err:.3e}"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_cross_bvp_zhat_parallel(stencil, geom):
    """Solve v + Cross(ẑ, v) = (1-r²) ẑ. Since ẑ × ẑ = 0, v_exact = (1-r²) ẑ.
    """
    v = Unknown.Vector("v")
    b = Background.Vector("b", v_r=sp.cos(theta), v_theta=-sp.sin(theta),
                          v_phi=sp.S.Zero)
    r_sym = sp.symbols("r", positive=True)
    g = Background.Vector(
        "g",
        v_r=(1 - r_sym**2) * sp.cos(theta),
        v_theta=-(1 - r_sym**2) * sp.sin(theta),
        v_phi=sp.S.Zero,
    )
    grid = _make_grid(geom, 24)
    sys_ = EquationSystem([Equation(v + Cross(b, v) - g, name="eq")], [])
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    R, T = grid.mesh()
    f_r = 1 - R**2
    v0_e = f_r * np.cos(T)
    vp_e = -f_r * np.sin(T) / np.sqrt(2)
    err = max(
        float(np.max(np.abs(res["v"][0] - v0_e))),
        float(np.max(np.abs(res["v"][+1] - vp_e))),
        float(np.max(np.abs(res["v"][-1] - vp_e))),
    )
    assert err < 1e-12, f"Cross BVP err: {err:.3e}"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_cross_bvp_nontrivial_coupling(stencil, geom):
    """Solve v + Cross(ŝ, v) = (1-r²)·(ẑ - φ̂). Cross(ŝ, ẑ) = -φ̂, so
    v_exact = (1-r²) ẑ.
    """
    v = Unknown.Vector("v")
    b = Background.Vector("b", v_r=sp.sin(theta), v_theta=sp.cos(theta),
                          v_phi=sp.S.Zero)
    r_sym = sp.symbols("r", positive=True)
    g = Background.Vector(
        "g",
        v_r=(1 - r_sym**2) * sp.cos(theta),
        v_theta=-(1 - r_sym**2) * sp.sin(theta),
        v_phi=-(1 - r_sym**2),
    )
    grid = _make_grid(geom, 24)
    sys_ = EquationSystem([Equation(v + Cross(b, v) - g, name="eq")], [])
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    R, T = grid.mesh()
    f_r = 1 - R**2
    v0_e = f_r * np.cos(T)
    vp_e = -f_r * np.sin(T) / np.sqrt(2)
    err = max(
        float(np.max(np.abs(res["v"][0] - v0_e))),
        float(np.max(np.abs(res["v"][+1] - vp_e))),
        float(np.max(np.abs(res["v"][-1] - vp_e))),
    )
    assert err < 1e-12, f"Cross-coupled BVP err: {err:.3e}"


# ---------------------------------------------------------------------------
# Curl operator direct correctness
# ---------------------------------------------------------------------------

def _make_curl_meta():
    v_field = Unknown.Vector("v")
    return {
        "m": 0,
        "v_placeholders": {"v": vector_unknown_spin_placeholders(v_field)},
    }


def _curl_apply(bundle, v_phys_fn):
    R, T = bundle.grid.mesh()
    v_r, v_t, v_p = v_phys_fn(R, T)
    x = _vector_state(v_r + 0j, v_t + 0j, v_p + 0j)
    op = build_curl_vector_block(_make_curl_meta(), bundle, m=0)
    return op @ x, R, T


def _expected_r_curl(R, T, curl_phys_fn):
    cr, ct, cp = curl_phys_fn(R, T)
    rcr = R * cr + 0j
    rct = R * ct + 0j
    rcp = R * cp + 0j
    rc0 = rcr
    rcp_e = (rct + 1j * rcp) / np.sqrt(2)
    rcm_e = (rct - 1j * rcp) / np.sqrt(2)
    return np.concatenate([_flat(rcm_e), _flat(rc0), _flat(rcp_e)]).astype(complex)


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_curl_zhat_polynomial(stencil, geom):
    """Curl((1-r²) ẑ) = 2r sin θ φ̂."""
    grid = _make_grid(geom, 24)
    bundle = DerivativeBundle(grid)

    def v_fn(R, T):
        f = 1 - R**2
        return (f * np.cos(T), -f * np.sin(T), 0 * R)

    def curl_fn(R, T):
        return (0 * R, 0 * R, 2 * R * np.sin(T))

    out, R, T = _curl_apply(bundle, v_fn)
    exp = _expected_r_curl(R, T, curl_fn)
    err = float(np.max(np.abs(out - exp)))
    # rate-5 convergence for symmetric polynomial input at N=24.
    assert err < 1e-4, f"Curl(zhat-poly) err: {err:.3e}"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_curl_shat_polynomial(stencil, geom):
    """v = (1-r²) ŝ. Curl_φ = -2r cos θ."""
    grid = _make_grid(geom, 24)
    bundle = DerivativeBundle(grid)

    def v_fn(R, T):
        f = 1 - R**2
        return (f * np.sin(T), f * np.cos(T), 0 * R)

    def curl_fn(R, T):
        return (0 * R, 0 * R, -2 * R * np.cos(T))

    out, R, T = _curl_apply(bundle, v_fn)
    exp = _expected_r_curl(R, T, curl_fn)
    err = float(np.max(np.abs(out - exp)))
    assert err < 1e-3, f"Curl(shat-poly) err: {err:.3e}"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_curl_convergence_rate(stencil, geom):
    """Verify rate ≥ 3.5 (theoretical 4) for the Curl operator using a
    smooth manufactured vector v = (1-r²)·ŝ."""

    def v_fn(R, T):
        f = 1 - R**2
        return (f * np.sin(T), f * np.cos(T), 0 * R)

    def curl_fn(R, T):
        return (0 * R, 0 * R, -2 * R * np.cos(T))

    errs = []
    Ns = [16, 24, 32]
    for N in Ns:
        grid = _make_grid(geom, N)
        bundle = DerivativeBundle(grid)
        out, R, T = _curl_apply(bundle, v_fn)
        exp = _expected_r_curl(R, T, curl_fn)
        errs.append(float(np.max(np.abs(out - exp))))
    rate = np.log(errs[0] / errs[-1]) / np.log(Ns[-1] / Ns[0])
    assert rate >= 3.5, (
        f"Curl convergence rate too low: {rate:.2f} "
        f"({stencil}-{geom}, errs={errs})"
    )
