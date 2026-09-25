"""Tests for Div(tensor_expression) in vector equations.

Validates the two-stage composition pipeline:
- Walker emits TASK_DIV_TENSOR_VECTOR with Div-in-T and T-in-v payloads.
- Backend dispatcher composes (lift T from v) ∘ (apply Div on T) to
  build the (3N × 3N) Div(T) operator block.
"""

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Grad, Div, Transpose, Boundary, BoundaryCondition,
    Lap, Equation, EquationSystem, Dot, r_hat, theta_hat, phi_hat, theta,
)
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem


def _make_dirichlet_bcs(v, boundary=Boundary.R_OUTER):
    """Standard Dirichlet (= 0) BCs on all 3 spin components."""
    return [
        BoundaryCondition(boundary, Dot(r_hat, v), name="vr"),
        BoundaryCondition(boundary, Dot(theta_hat, v), name="vt"),
        BoundaryCondition(boundary, Dot(phi_hat, v), name="vp"),
    ]


# Suppress the divide-by-zero warnings from r=0 lambdify evaluations.
# These NaN/inf entries are masked at solve time by origin regularity,
# but the warnings noise up test output.
pytestmark = pytest.mark.filterwarnings(
    "ignore::RuntimeWarning"
)


# =============================================================================
# Walker: Div(tensor_expr) produces TASK_DIV_TENSOR_VECTOR
# =============================================================================

class TestDivTensorWalker:
    """Walker produces the right task structure for Div(tensor_expr)."""

    def test_div_grad_v_emits_div_tensor_task(self):
        from spdee.ir.tasks import TASK_DIV_TENSOR_VECTOR
        from spdee.ir.walker import walk_equation_system
        v = Unknown.Vector("v")
        sys = EquationSystem(
            [Equation(Div(Grad(v)), name="eq")],
            _make_dirichlet_bcs(v),
        )
        tasks = walk_equation_system(sys, m=0)
        div_tasks = [t for t in tasks if t.kind == TASK_DIV_TENSOR_VECTOR]
        assert len(div_tasks) == 1
        task = div_tasks[0]
        assert task.unknown_name == "v"
        # Payload contains the r²-multiplied form in U=r·T variables:
        # per_sigma_expr_in_U (first-order in U), U_in_v (first-order
        # in v). The two-stage composition picks up the second-
        # derivative behavior of Div(Grad(v)) at the dispatch level
        # without ever needing a second-order term classifier. The
        # r²-multiplication clears all 1/r factors symbolically.
        per_sigma = task.metadata["per_sigma_expr_in_U"]
        U_in_v = task.metadata["U_in_v"]
        for (s_a, s_b), expr in U_in_v.items():
            for d in expr.atoms(sp.Derivative):
                for _, order in d.variable_count:
                    assert order == 1, (
                        f"U_in_v[({s_a},{s_b})] has {d} with order {order} > 1"
                    )
        for s, expr in per_sigma.items():
            for d in expr.atoms(sp.Derivative):
                for _, order in d.variable_count:
                    assert order == 1, (
                        f"per_sigma_expr_in_U[{s}] has {d} order {order} > 1"
                    )

    def test_div_strain_emits_div_tensor_task(self):
        from spdee.ir.tasks import TASK_DIV_TENSOR_VECTOR
        from spdee.ir.walker import walk_equation_system
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        sys = EquationSystem(
            [Equation(Div(strain), name="eq")],
            _make_dirichlet_bcs(v),
        )
        tasks = walk_equation_system(sys, m=0)
        div_tasks = [t for t in tasks if t.kind == TASK_DIV_TENSOR_VECTOR]
        assert len(div_tasks) == 1


# =============================================================================
# End-to-end: Div(Grad(v)) = -source manufactured solutions
# =============================================================================

# Manufactured solution v = (1-r²) ẑ has Lap(v) = -6 ẑ.
# In spherical-physical components:
#   v_r = (1-r²) cos θ, v_θ = -(1-r²) sin θ, v_φ = 0
#   Lap(v) physical: v_r = -6 cos θ, v_θ = 6 sin θ, v_φ = 0


def _exact_zhat_solution(grid):
    """Exact (1-r²)ẑ spin-component arrays on the grid."""
    R, T = grid.mesh()
    v_r = (1 - R**2) * np.cos(T)
    v_t = -(1 - R**2) * np.sin(T)
    return {-1: v_t / np.sqrt(2), 0: v_r, +1: v_t / np.sqrt(2)}


class TestDivGradEndToEnd:
    """End-to-end: Div(Grad(v)) = source with manufactured (1-r²)ẑ
    solution. The Div dispatcher uses the r²-multiplied U=r·T
    formulation (compact-FD with parity-aware radial stencils at r=0
    in ball mode, compact θ-d1 in both modes); should converge to the
    same continuous solution as direct Lap(v) at rate ~4 or better."""

    def test_n32_max_err_acceptable(self):
        N = 32
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r=-6 * sp.cos(theta),
            v_theta=6 * sp.sin(theta),
            v_phi=sp.S.Zero,
        )
        sys = EquationSystem(
            [Equation(Div(Grad(v)) - g, name="eq")],
            _make_dirichlet_bcs(v),
        )
        res = compile_problem(sys, grid, m=0).solve_bvp()
        exact = _exact_zhat_solution(grid)
        for sigma in (-1, 0, +1):
            e = float(np.max(np.abs(res["v"][sigma] - exact[sigma])))
            # Sparse-banded Div builder (standard 4th-order FD) gives
            # rate ~3.8 for Div(Grad), measured ~1.9e-5 at N=32.
            # Threshold leaves a margin.
            assert e < 5e-5, f"σ={sigma}: max err = {e:.3e}"

    def test_convergence_rate(self):
        """Sparse-banded Div builder gives approximately rate-p convergence
        for Div(Grad) on a ball. Verified by: (i) absolute error decreases
        monotonically under refinement, (ii) the asymptotic rate (taken
        as the max consecutive-pair rate) is ≥ 3.5, allowing for the
        small order loss near the origin from L'Hospital placeholder
        treatment."""
        errs = []
        Ns = [12, 16, 24, 32, 48]
        for N in Ns:
            grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
            v = Unknown.Vector("v")
            g = Background.Vector(
                "g",
                v_r=-6 * sp.cos(theta),
                v_theta=6 * sp.sin(theta),
                v_phi=sp.S.Zero,
            )
            sys = EquationSystem(
                [Equation(Div(Grad(v)) - g, name="eq")],
                _make_dirichlet_bcs(v),
            )
            res = compile_problem(sys, grid, m=0).solve_bvp()
            exact = _exact_zhat_solution(grid)
            err = max(
                np.max(np.abs(res["v"][sigma] - exact[sigma]))
                for sigma in (-1, 0, +1)
            )
            errs.append(float(err))
        # Errors should decrease monotonically.
        for i in range(len(errs) - 1):
            assert errs[i + 1] < errs[i], (
                f"Non-monotonic at N={Ns[i]}->{Ns[i+1]}: {errs}"
            )
        # Asymptotic rate from the max consecutive-pair slope.
        pair_rates = [
            np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i])
            for i in range(len(errs) - 1)
        ]
        rate = max(pair_rates)
        assert rate >= 3.5, f"Convergence rate too low: {rate:.2f}; errs={errs}"

    def test_solves_same_problem_as_direct_Lap(self):
        """Div(Grad(v)) - g = 0 and Lap(v) - g = 0 should produce the
        SAME continuous solution. Numerical answers may differ at finite N
        because the two formulations use different discretizations
        (compact-FD Lap vs standard-FD Div composition), but the diff
        should shrink with N."""
        N = 48
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)

        v_lap = Unknown.Vector("v_lap")
        g1 = Background.Vector(
            "g1", v_r=-6 * sp.cos(theta), v_theta=6 * sp.sin(theta), v_phi=sp.S.Zero,
        )
        sys_lap = EquationSystem(
            [Equation(Lap(v_lap) - g1, name="eq")], _make_dirichlet_bcs(v_lap),
        )
        res_lap = compile_problem(sys_lap, grid, m=0).solve_bvp()

        v_div = Unknown.Vector("v_div")
        g2 = Background.Vector(
            "g2", v_r=-6 * sp.cos(theta), v_theta=6 * sp.sin(theta), v_phi=sp.S.Zero,
        )
        sys_div = EquationSystem(
            [Equation(Div(Grad(v_div)) - g2, name="eq")], _make_dirichlet_bcs(v_div),
        )
        res_div = compile_problem(sys_div, grid, m=0).solve_bvp()

        for sigma in (-1, 0, +1):
            d = float(np.max(np.abs(res_lap["v_lap"][sigma] - res_div["v_div"][sigma])))
            # Both methods produce the same continuous solution; finite-N
            # discretization difference should be small.
            assert d < 5e-5, f"σ={sigma}: Lap vs Div(Grad) diff at N=48 = {d:.3e}"


# =============================================================================
# Div(strain): manufactured solution with the strain identity
# =============================================================================

class TestDivStrainEndToEnd:
    """End-to-end test for Div(Grad(v) + Transpose(Grad(v))) on a
    manufactured solution v = (1-r²) ẑ.

    Identity: Div(strain) = Lap(v) + Grad(Div(v)).
    For v = (1-r²) ẑ:
      Lap(v) = -6 ẑ
      Div(v) = -2r cos θ  (computed in physical-spherical)
      Grad(Div(v)) = -2 cos θ r̂ + 2 sin θ θ̂
    So Div(strain) = (-6 - 2) cos θ r̂ + (6 + 2) sin θ θ̂
                  = -8 cos θ r̂ + 8 sin θ θ̂.
    Source f = Div(strain).
    """

    def test_div_strain_solves_correctly(self):
        N = 32
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        f = Background.Vector(
            "f",
            v_r=-8 * sp.cos(theta),
            v_theta=8 * sp.sin(theta),
            v_phi=sp.S.Zero,
        )
        strain = Grad(v) + Transpose(Grad(v))
        sys = EquationSystem(
            [Equation(Div(strain) - f, name="eq")],
            _make_dirichlet_bcs(v),
        )
        res = compile_problem(sys, grid, m=0).solve_bvp()
        exact = _exact_zhat_solution(grid)
        for sigma in (-1, 0, +1):
            e = float(np.max(np.abs(res["v"][sigma] - exact[sigma])))
            # Sparse-banded Div builder (standard 4th-order FD): rate
            # ~3.8 for Div(strain), measured ~3e-5 at N=32. Threshold
            # leaves a margin.
            assert e < 5e-5, f"Div(strain) σ={sigma}: err = {e:.3e}"

    def test_div_strain_convergence_rate(self):
        """Div(strain) with the U-form, θ-parity-folded stencils, and
        operator-aware origin regularity gives rate ~3.6. The remaining
        gap from rate-4 is the cot(θ_pole)×θ-stencil-error contribution
        at the corner cells.

        Verified by (i) monotone error decrease and (ii) max
        consecutive-pair rate ≥ 3.3.
        """
        errs = []
        Ns = [12, 16, 24, 32, 48]
        for N in Ns:
            grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
            v = Unknown.Vector("v")
            f = Background.Vector(
                "f", v_r=-8 * sp.cos(theta), v_theta=8 * sp.sin(theta), v_phi=sp.S.Zero,
            )
            strain = Grad(v) + Transpose(Grad(v))
            sys = EquationSystem(
                [Equation(Div(strain) - f, name="eq")], _make_dirichlet_bcs(v),
            )
            res = compile_problem(sys, grid, m=0).solve_bvp()
            exact = _exact_zhat_solution(grid)
            err = max(
                np.max(np.abs(res["v"][sigma] - exact[sigma]))
                for sigma in (-1, 0, +1)
            )
            errs.append(float(err))
        for i in range(len(errs) - 1):
            assert errs[i+1] < errs[i], (
                f"Non-monotonic at N={Ns[i]}->{Ns[i+1]}: {errs}"
            )
        pair_rates = [
            np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i])
            for i in range(len(errs) - 1)
        ]
        rate = max(pair_rates)
        assert rate >= 3.3, (
            f"Div(strain) convergence rate too low: {rate:.2f}; errs={errs}"
        )


# =============================================================================
# m != 0 test
# =============================================================================

class TestDivGradAtNonzeroM:
    """Verify Div(Grad(v)) at m != 0 with a manufactured solution."""

    def test_m_plus_one_manufactured(self):
        """v = (1-r²) ŷ at m=+1.
        Physical (m=+1 e^{iφ} coefficient):
          v_r = -i (1-r²) sin θ / 2
          v_θ = -i (1-r²) cos θ / 2
          v_φ =    (1-r²)         / 2
        Lap(v) = -6 ŷ at m=+1 → source f has the same shape as v, scaled:
          f_r = -i · (-6) · sin θ / 2 = 3i sin θ
          f_θ = -i · (-6) · cos θ / 2 = 3i cos θ
          f_φ =    (-6)            / 2 = -3
        """
        N = 32
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r=sp.I * 3 * sp.sin(theta),
            v_theta=sp.I * 3 * sp.cos(theta),
            v_phi=-3 * sp.S.One,
        )
        sys = EquationSystem(
            [Equation(Div(Grad(v)) - g, name="eq")],
            _make_dirichlet_bcs(v),
        )
        res = compile_problem(sys, grid, m=+1).solve_bvp()

        R, T = grid.mesh()
        v_r_exact = -1j * (1 - R**2) * np.sin(T) / 2
        v_t_exact = -1j * (1 - R**2) * np.cos(T) / 2
        v_phi_exact = (1 - R**2) / 2
        v0_exact = v_r_exact
        vp_exact = (v_t_exact + 1j * v_phi_exact) / np.sqrt(2)
        vm_exact = (v_t_exact - 1j * v_phi_exact) / np.sqrt(2)
        exact_map = {-1: vm_exact, 0: v0_exact, +1: vp_exact}
        for sigma in (-1, 0, +1):
            e = float(np.max(np.abs(res["v"][sigma] - exact_map[sigma])))
            # Sparse-banded Div builder (standard 4th-order FD):
            # rate ~3.8, measured ~7e-6 at N=32. Threshold leaves a margin.
            assert e < 5e-5, f"m=+1: σ={sigma}: err = {e:.3e}"
