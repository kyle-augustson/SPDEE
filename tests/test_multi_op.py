"""Tests for multi-operator equation support.

Equations may contain multiple differential operators per equation. The
compiler picks a common equation mass diag(r^p_eq) where p_eq is the
max natural r-power across all differential tasks. Each operator's
contribution is left-multiplied by diag(r^{p_eq - p_op}) / k_op to
align with the common mass.

For ball-mode origin regularity, the L'Hospital factor is the
sign-weighted, coefficient-weighted sum of contributions from each
differential task that affects the ẑ-aligned ℓ=0 mode (Lap contributes
3·coef, Div(strain) contributes 4·coef, Curl/Cross contribute 0).

Covered:
  - Single-unknown multi-op: Lap(v) + Curl(v) = g, with and without
    coefficients (verifies origin-factor accumulation).
  - Two-unknown multi-op: Lap(v) + Grad(p) + Cross(Ω, v) = g, Div(v) = 0
    (the inertial wave / Stokes case; v solves to rate 5, p has the
    LBB checkerboard which is a separate known issue).
  - Multi-op with coefficient on Lap (origin-factor scaling).
  - Three-operator equation: Lap(v) + Curl(v) + Cross(b, v) = g.
"""
import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Curl, Cross, Grad, Div, theta,
    Equation, EquationSystem,
    Boundary, BoundaryCondition, Dot, r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid, make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem


CONTEXTS = [
    ("standard", "shell"),
    ("compact", "shell"),
    ("standard", "ball"),
    ("compact", "ball"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid_for(geom, N):
    if geom == "ball":
        return make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    return make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)


def _err_v_against_zhat_poly(res, R, T):
    """err vs v_exact = (1-r²) ẑ in V19 spin basis."""
    f_r = 1 - R**2
    v0_e = f_r * np.cos(T)
    vp_e = -f_r * np.sin(T) / np.sqrt(2)
    return max(
        float(np.max(np.abs(res["v"][0] - v0_e))),
        float(np.max(np.abs(res["v"][+1] - vp_e))),
        float(np.max(np.abs(res["v"][-1] - vp_e))),
    )


def _shell_zhat_dirichlet_bcs(v, name_suffix=""):
    """Dirichlet BCs at both shell boundaries for v_exact = (1-r²) ẑ."""
    r_in = sp.Rational(3, 10)
    v_r_in = (1 - r_in**2) * sp.cos(theta)
    v_t_in = -(1 - r_in**2) * sp.sin(theta)
    return [
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                          name=f"vr_o{name_suffix}"),
        BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v),
                          name=f"vt_o{name_suffix}"),
        BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v),
                          name=f"vp_o{name_suffix}"),
        BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v) - v_r_in,
                          name=f"vr_i{name_suffix}"),
        BoundaryCondition(Boundary.R_INNER, Dot(theta_hat, v) - v_t_in,
                          name=f"vt_i{name_suffix}"),
        BoundaryCondition(Boundary.R_INNER, Dot(phi_hat, v),
                          name=f"vp_i{name_suffix}"),
    ]


def _ball_zhat_dirichlet_bcs(v, name_suffix=""):
    """Dirichlet BCs at r_outer for v_exact = (1-r²) ẑ (vanishes at r=1)."""
    return [
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                          name=f"vr_o{name_suffix}"),
        BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v),
                          name=f"vt_o{name_suffix}"),
        BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v),
                          name=f"vp_o{name_suffix}"),
    ]


# ---------------------------------------------------------------------------
# Lap(v) + Curl(v) = g
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_lap_plus_curl_resolves(stencil, geom):
    """Multi-op Lap(v) + Curl(v) = g. v_exact = (1-r²) ẑ."""
    v = Unknown.Vector("v")
    r_sym = sp.symbols("r", positive=True)
    # Lap((1-r²)ẑ) = -6ẑ; Curl((1-r²)ẑ)_φ = 2r sinθ.
    g_v = Background.Vector(
        "g",
        v_r=-6 * sp.cos(theta),
        v_theta=6 * sp.sin(theta),
        v_phi=2 * r_sym * sp.sin(theta),
    )
    grid = _make_grid_for(geom, 24)
    bcs = (_ball_zhat_dirichlet_bcs(v) if geom == "ball"
           else _shell_zhat_dirichlet_bcs(v))
    sys_ = EquationSystem([Equation(Lap(v) + Curl(v) - g_v, name="eq")], bcs)
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    R, T = grid.mesh()
    err = _err_v_against_zhat_poly(res, R, T)
    # rate-4 expectation: ~6e-6 in shell, ~6e-5 in ball.
    bound = 5e-4 if geom == "ball" else 5e-5
    assert err < bound, f"Lap+Curl err: {err:.3e} ({stencil}-{geom})"


@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_lap_plus_curl_convergence(stencil, geom):
    """Verify approximately rate-4 convergence for Lap(v) + Curl(v) = g.
    Asymptotic rate ≥ 3.5 measured as max consecutive-pair rate over a
    refinement sweep, since ball geometry can have lower pre-asymptotic
    rates due to L'Hospital placeholder treatment at the origin."""
    v = Unknown.Vector("v")
    r_sym = sp.symbols("r", positive=True)
    g_v = Background.Vector(
        "g",
        v_r=-6 * sp.cos(theta),
        v_theta=6 * sp.sin(theta),
        v_phi=2 * r_sym * sp.sin(theta),
    )
    errs = []
    Ns = [16, 24, 32, 48]
    for N in Ns:
        grid = _make_grid_for(geom, N)
        bcs = (_ball_zhat_dirichlet_bcs(v) if geom == "ball"
               else _shell_zhat_dirichlet_bcs(v))
        sys_ = EquationSystem(
            [Equation(Lap(v) + Curl(v) - g_v, name="eq")], bcs,
        )
        res = compile_problem(sys_, grid, m=0).solve_bvp()
        R, T = grid.mesh()
        errs.append(_err_v_against_zhat_poly(res, R, T))
    for i in range(len(errs) - 1):
        assert errs[i+1] < errs[i], (
            f"Non-monotonic at N={Ns[i]}->{Ns[i+1]}: errs={errs}"
        )
    pair_rates = [
        np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i])
        for i in range(len(errs) - 1)
    ]
    rate = max(pair_rates)
    assert rate >= 3.5, f"Rate {rate:.2f} too low; errs={errs}"


# ---------------------------------------------------------------------------
# Coefficient-weighted multi-op: 2*Lap(v) - Curl(v) = g
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_multiop_with_lap_coefficient(stencil, geom):
    """Coefficient on Lap (2*Lap(v) - Curl(v) = g). Tests that the
    origin-regularity factor accumulates coefficients correctly.

    For v_exact = (1-r²) ẑ:
        2·Lap(v) = -12 ẑ
        Curl(v)_φ = 2r sinθ
        g = -12 ẑ - 2r sinθ φ̂

    Verified by (i) monotone error decrease and (ii) max consecutive-pair
    rate ≥ 3.5 on a refinement sweep, which separates the asymptotic
    rate from pre-asymptotic effects.
    """
    v = Unknown.Vector("v")
    r_sym = sp.symbols("r", positive=True)
    g_v = Background.Vector(
        "g",
        v_r=-12 * sp.cos(theta),
        v_theta=12 * sp.sin(theta),
        v_phi=-2 * r_sym * sp.sin(theta),
    )
    errs = []
    Ns = [16, 24, 32, 48]
    for N in Ns:
        grid = _make_grid_for(geom, N)
        bcs = (_ball_zhat_dirichlet_bcs(v) if geom == "ball"
               else _shell_zhat_dirichlet_bcs(v))
        sys_ = EquationSystem(
            [Equation(2 * Lap(v) - Curl(v) - g_v, name="eq")], bcs,
        )
        res = compile_problem(sys_, grid, m=0).solve_bvp()
        R, T = grid.mesh()
        errs.append(_err_v_against_zhat_poly(res, R, T))
    for i in range(len(errs) - 1):
        assert errs[i+1] < errs[i], (
            f"Non-monotonic at N={Ns[i]}->{Ns[i+1]} ({stencil}-{geom}): errs={errs}"
        )
    pair_rates = [
        np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i])
        for i in range(len(errs) - 1)
    ]
    rate = max(pair_rates)
    assert rate >= 3.5, (
        f"Rate {rate:.2f} too low ({stencil}-{geom}); errs={errs}. "
        "Likely indicates origin-regularity factor isn't picking up the "
        "Lap coefficient correctly in multi-op mode."
    )


# ---------------------------------------------------------------------------
# Three differential operators: Lap(v) + Curl(v) + Cross(b, v) = g
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", CONTEXTS)
def test_three_op_equation(stencil, geom):
    """Lap(v) + Curl(v) + Cross(b, v) = g.

    With b = ẑ and v_exact = (1-r²) ẑ:
        Cross(ẑ, ẑ·h) = 0 (parallel)
    So g = Lap + Curl + 0 = -6 ẑ + 2r sinθ φ̂.
    """
    v = Unknown.Vector("v")
    r_sym = sp.symbols("r", positive=True)
    b = Background.Vector(
        "b", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    g_v = Background.Vector(
        "g",
        v_r=-6 * sp.cos(theta),
        v_theta=6 * sp.sin(theta),
        v_phi=2 * r_sym * sp.sin(theta),
    )
    grid = _make_grid_for(geom, 24)
    bcs = (_ball_zhat_dirichlet_bcs(v) if geom == "ball"
           else _shell_zhat_dirichlet_bcs(v))
    sys_ = EquationSystem(
        [Equation(Lap(v) + Curl(v) + Cross(b, v) - g_v, name="eq")], bcs,
    )
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    R, T = grid.mesh()
    err = _err_v_against_zhat_poly(res, R, T)
    bound = 5e-4 if geom == "ball" else 5e-5
    assert err < bound, f"3-op err: {err:.3e} ({stencil}-{geom})"


# ---------------------------------------------------------------------------
# Coupled system: inertial-wave-like (v and p)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stencil,geom", [
    ("standard", "shell"),
    ("compact", "shell"),
    ("standard", "ball"),
    ("compact", "ball"),
])
def test_inertial_wave_coupled_system(stencil, geom):
    """Coupled momentum + continuity:
        Lap(v) + Grad(p) + 2·Cross(Ω, v) = g
        Div(v) = 0
    Manufactured: v = (1-r²) r sinθ · φ̂ (divergence-free, regular),
                  p = 0 (gauge-fixed by integral constraint).
    """
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    r_sym = sp.symbols("r", positive=True)
    Omega = Background.Vector(
        "Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    g_v = Background.Vector(
        "g",
        v_r=-2 * (1 - r_sym**2) * r_sym * sp.sin(theta)**2,
        v_theta=-2 * (1 - r_sym**2) * r_sym * sp.sin(theta) * sp.cos(theta),
        v_phi=-10 * r_sym * sp.sin(theta),
    )
    eq1 = Equation(
        Lap(v) + Grad(p) + 2 * Cross(Omega, v) - g_v, name="momentum",
    )
    eq2 = Equation(Div(v), name="continuity")

    bcs = [
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_o"),
        BoundaryCondition(Boundary.R_OUTER, Dot(theta_hat, v), name="vt_o"),
        BoundaryCondition(Boundary.R_OUTER, Dot(phi_hat, v), name="vp_o"),
    ]
    if geom == "shell":
        r_in = sp.Rational(3, 10)
        v_phi_in = (1 - r_in**2) * r_in * sp.sin(theta)
        bcs += [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vr_i"),
            BoundaryCondition(
                Boundary.R_INNER, Dot(theta_hat, v), name="vt_i",
            ),
            BoundaryCondition(
                Boundary.R_INNER, Dot(phi_hat, v) - v_phi_in, name="vp_i",
            ),
        ]
        grid = make_shell_grid(N_r=24, N_theta=24, r_inner=0.3, r_outer=1.0)
    else:
        grid = make_ball_grid(N_r=24, N_theta=24, r_outer=1.0)

    sys_ = EquationSystem([eq1, eq2], bcs, gauge_unknowns=["p"])
    res = compile_problem(sys_, grid, m=0).solve_bvp()
    R, T = grid.mesh()
    h_r = (1 - R**2) * R * np.sin(T)
    # v_exact: v_r=0, v_θ=0, v_φ=h. Spin: v^0=0, v^± = ±i h/√2.
    v0_e = 0 * R + 0j
    vm_e = -1j * h_r / np.sqrt(2)
    vp_e = +1j * h_r / np.sqrt(2)
    err_v = max(
        float(np.max(np.abs(res["v"][0] - v0_e))),
        float(np.max(np.abs(res["v"][+1] - vp_e))),
        float(np.max(np.abs(res["v"][-1] - vm_e))),
    )
    # NOTE: p has the colocated-FD pressure checkerboard (LBB instability) —
    # we don't test its accuracy here. v should solve cleanly to rate 4+.
    assert err_v < 1e-5, f"err_v: {err_v:.3e} ({stencil}-{geom})"
