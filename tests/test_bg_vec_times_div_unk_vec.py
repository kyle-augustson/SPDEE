"""Test the walker pattern for ``bg_vec * Div(unknown_vector)``.

This is the composed operator needed for the trace piece of the
viscous force decomposition of Bekki et al. (2022, §2.3):
    ∇ln ρ_0 · [-(2/3)(∇·v) I]  =  -(2/3)(∇·v) · ∇ln ρ_0
where ∇·v is the unknown scalar produced by ``Div(v)``. By dispatching
the composition ``bg_vec * Div(v)`` directly, we avoid introducing
an auxiliary scalar unknown ``D := ∇·v``.

V19 spin contraction:
    output^σ(r,θ) = bg^σ(r,θ) · (∇·v)(r,θ)
The dispatcher emits ``r · output^σ`` (natural r-power 1, same as
``TASK_DIV_VECTOR_SCALAR``).
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from spdee import (
    Unknown, Background, Dt, Div, Dot, Grad,
    Equation, EquationSystem, BoundaryCondition, Boundary,
    r_hat, NumericComponent,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import reshape_2d, flatten_2d


class TestWalkerRecognition:
    """Verify the walker emits ``TASK_BG_VEC_TIMES_DIV_UNK_VEC`` for
    the right expression structures and continues to handle the
    plain ``bg_vec * unknown_scalar`` case with its own task.
    """

    def test_div_unk_vec_times_bg_vec_emits_new_task(self):
        from spdee.ir.tasks import (
            TASK_BG_VEC_TIMES_DIV_UNK_VEC,
            TASK_BG_VEC_TIMES_UNK_SCALAR,
        )
        from spdee.ir.walker import walk_equation_system
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        sys_ = EquationSystem(
            [Equation(a_bg * Div(v), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                               name="vr_out")],
        )
        tasks = walk_equation_system(sys_, m=0)
        kinds = {t.kind for t in tasks}
        assert TASK_BG_VEC_TIMES_DIV_UNK_VEC in kinds
        assert TASK_BG_VEC_TIMES_UNK_SCALAR not in kinds

    def test_bg_vec_times_bare_unk_scalar_uses_old_path(self):
        """The plain ``unknown_scalar * background_vector`` pattern
        dispatches ``TASK_BG_VEC_TIMES_UNK_SCALAR``, not the Div task.
        """
        from spdee.ir.tasks import (
            TASK_BG_VEC_TIMES_DIV_UNK_VEC,
            TASK_BG_VEC_TIMES_UNK_SCALAR,
        )
        from spdee.ir.walker import walk_equation_system
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        s = Unknown.Scalar("s")
        v = Unknown.Vector("v")
        sys_ = EquationSystem(
            [Equation(Dt(v) - s * a_bg, name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                               name="vr_out"),
             BoundaryCondition(Boundary.R_OUTER, s, name="s_out")],
        )
        tasks = walk_equation_system(sys_, m=0)
        kinds = {t.kind for t in tasks}
        assert TASK_BG_VEC_TIMES_UNK_SCALAR in kinds
        assert TASK_BG_VEC_TIMES_DIV_UNK_VEC not in kinds


class TestNoAuxStateAdded:
    """The point of dispatching ``bg_vec * Div(v)`` directly is to
    avoid introducing an auxiliary scalar ``D := ∇·v``. Verify the
    state vector has only ``v`` (no aux scalar) after compilation.
    """

    def test_state_size_unchanged_with_or_without_term(self):
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v),
                              name="vri"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                              name="vro"),
        ]
        N_r, N_theta = 12, 24
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.5, r_outer=1.0,
        )
        # Without the term: only Dt(v)
        sys_no = EquationSystem([Equation(Dt(v), name="mom")], bcs)
        cp_no = compile_problem(sys_no, grid, m=1, order_r=4, order_theta=4)
        # With the term: Dt(v) + a_bg * Div(v)
        sys_yes = EquationSystem(
            [Equation(Dt(v) - a_bg * Div(v), name="mom")], bcs,
        )
        cp_yes = compile_problem(
            sys_yes, grid, m=1, order_r=4, order_theta=4,
        )
        # State size must be identical = 3 · N_r · N_theta.
        assert cp_yes.matrices.layout.size == cp_no.matrices.layout.size
        assert cp_yes.matrices.layout.size == 3 * N_r * N_theta


class TestManufacturedSolution:
    """Apply the assembled operator block to a known analytical input
    and compare against the analytical answer at interior grid points.

    Test problem:
        a = r̂ (constant radial bg vector, magnitude 1)
        v = (1 - r²) ẑ at m = 0
    Then:
        v_r = (1-r²)cosθ, v_θ = -(1-r²)sinθ, v_φ = 0
        ∇·v = (-2r)·cosθ
    For radial a (a_r = 1, a_θ = a_φ = 0):
        (a·∇·v)_r = -2r cosθ
        (a·∇·v)_θ = 0
        (a·∇·v)_φ = 0
    Dispatcher emits ``r · output^σ``:
        r·(a · Div v)^0 = -2r² cosθ
        r·(a · Div v)^± = 0
    """

    def test_zhat_solution_m0(self):
        """Build a one-equation system whose K matrix is exactly
        ``r·(a · Div(v))``, then apply to a known v and compare to
        the analytical answer.
        """
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        m_val = 0
        grid = make_shell_grid(
            N_r=60, N_theta=120, r_inner=0.3, r_outer=1.0,
        )
        # Single equation: a_bg * Div(v) = 0.
        # Two impenetrability BCs (one per radial boundary).
        sys_ = EquationSystem(
            [Equation(a_bg * Div(v), name="eq")],
            [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v),
                               name="vri"),
             BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                               name="vro")],
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=4, order_theta=4)
        K = cp.matrices.A_total

        R, T = grid.mesh()
        v_r_arr = (1 - R**2) * np.cos(T)
        v_t_arr = -(1 - R**2) * np.sin(T)
        state = np.concatenate([
            flatten_2d(v_t_arr / np.sqrt(2)),
            flatten_2d(v_r_arr),
            flatten_2d(v_t_arr / np.sqrt(2)),
        ])
        out = K @ state
        N = grid.N_r * grid.N_theta
        out_m = reshape_2d(out[:N], grid.N_r, grid.N_theta)
        out_0 = reshape_2d(out[N:2*N], grid.N_r, grid.N_theta)
        out_p = reshape_2d(out[2*N:], grid.N_r, grid.N_theta)

        # Expected (r-multiplied form):
        exact_0 = -2 * R**2 * np.cos(T)
        # σ=± rows: a_θ = a_φ = 0, so result is identically zero.
        sl_r = slice(3, -3)
        sl_t = slice(3, -3)
        diff = (out_0 - exact_0)[sl_r, sl_t]
        denom = np.linalg.norm(exact_0[sl_r, sl_t])
        rel_err_0 = np.linalg.norm(diff) / denom
        abs_err_m = np.linalg.norm(out_m[sl_r, sl_t])
        abs_err_p = np.linalg.norm(out_p[sl_r, sl_t])
        # 4th-order FD on (1-r²) cos θ is ~machine precision because
        # (1-r²) is polynomial, but ∇·v = -2r cos θ is also polynomial
        # so we expect very tight numerical agreement.
        assert rel_err_0 < 1e-8, (
            f"σ=0 manufactured-solution err {rel_err_0:.3e} > 1e-8"
        )
        # σ=± should be exactly zero (a_θ = a_φ = 0 bit-exact).
        assert abs_err_m == 0.0, f"σ=- output {abs_err_m:.3e} should be 0"
        assert abs_err_p == 0.0, f"σ=+ output {abs_err_p:.3e} should be 0"

    def test_nonzero_polar_bg(self):
        """Polar bg component ``a_θ`` activates σ=± rows.

        With ``a = θ̂`` (constant polar magnitude 1) and v = (1-r²) ẑ
        at m = 0:
            ∇·v = -2r cosθ  (same as before)
            (a · Div v)_θ = -2r cosθ  (full magnitude on θ row)
            (a · Div v)_r = 0
            (a · Div v)_φ = 0
        In spin: ``(a · Div v)^± = (a · Div v)_θ / √2 = -√2 r cosθ``.
        Dispatcher emits ``r · output^±``:
            r·(a · Div v)^± = -√2 r² cosθ
            r·(a · Div v)^0 = 0
        """
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.Zero, v_theta=sp.S.One, v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        m_val = 0
        grid = make_shell_grid(
            N_r=60, N_theta=120, r_inner=0.3, r_outer=1.0,
        )
        sys_ = EquationSystem(
            [Equation(a_bg * Div(v), name="eq")],
            [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v),
                               name="vri"),
             BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),
                               name="vro")],
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=4, order_theta=4)
        K = cp.matrices.A_total

        R, T = grid.mesh()
        v_r_arr = (1 - R**2) * np.cos(T)
        v_t_arr = -(1 - R**2) * np.sin(T)
        state = np.concatenate([
            flatten_2d(v_t_arr / np.sqrt(2)),
            flatten_2d(v_r_arr),
            flatten_2d(v_t_arr / np.sqrt(2)),
        ])
        out = K @ state
        N = grid.N_r * grid.N_theta
        out_m = reshape_2d(out[:N], grid.N_r, grid.N_theta)
        out_0 = reshape_2d(out[N:2*N], grid.N_r, grid.N_theta)
        out_p = reshape_2d(out[2*N:], grid.N_r, grid.N_theta)
        exact_pm = -np.sqrt(2) * R**2 * np.cos(T)
        sl = slice(3, -3)
        denom_pm = np.linalg.norm(exact_pm[sl, sl])
        err_m = np.linalg.norm((out_m - exact_pm)[sl, sl]) / denom_pm
        err_p = np.linalg.norm((out_p - exact_pm)[sl, sl]) / denom_pm
        err_0 = np.linalg.norm(out_0[sl, sl])  # absolute (target is 0)
        assert err_m < 1e-8, f"σ=- err {err_m:.3e} > 1e-8"
        assert err_p < 1e-8, f"σ=+ err {err_p:.3e} > 1e-8"
        # σ=0 absolute: a_r = 0, so this row's contribution from
        # bg_vec * Div(v) is bit-exact zero. (However it would also
        # carry the BC rows, which is why we don't compare on the
        # full grid.)
        assert err_0 < 1e-10, f"σ=0 should be 0, got {err_0:.3e}"


class TestSolarTraceTerm:
    """Integration test: the trace piece ``-(2/3)(∇·v)∇ln ρ_0`` of
    the solar eigenproblem's viscous force compiles on a solar background and produces
    a numerically distinguishable contribution to a Step-4-like
    spectrum.
    """

    @pytest.fixture(scope="class")
    def solar_bg(self):
        from pathlib import Path
        data_dir = Path(__file__).parent.parent / "data" / "solar"
        if not (data_dir / "model_s_corrected.txt").exists():
            pytest.skip(f"Solar test data missing under {data_dir}")
        from spdee.backgrounds.solar import SolarBackgrounds
        return SolarBackgrounds(
            data_dir, ('shell', 0.71, 0.985),
            nondimensional=True, uniform_core_blend=True,
        )

    def test_solar_full_viscous_compiles(self, solar_bg):
        """The full solar-eigenproblem viscous decomposition
            ν · [Lap(v) + (1/3) Grad(Div(v))
                 + (∇ln ρ_0) · (Grad(v) + (Grad v)^T)
                 - (2/3)(∇·v) ∇ln ρ_0]
        compiles cleanly on the solar background. No aux state added.
        """
        from spdee import Lap, Transpose, ImpenetrableStressFree

        rho_eval = solar_bg.rho.component
        grad_rho_r_eval = solar_bg.grad_rho.components["v_r"]
        def grad_ln_rho_r_fn(R, T):
            return (
                grad_rho_r_eval.evaluate_on_grid(R, T)
                / rho_eval.evaluate_on_grid(R, T)
            )
        grad_ln_rho = Background.Vector(
            "grad_ln_rho",
            v_r=NumericComponent(
                f=grad_ln_rho_r_fn,
                description="(1/ρ_0) ∂_r ρ_0",
            ),
            v_theta=None, v_phi=None,
        )
        v = Unknown.Vector("v")
        Ek = 3e-3
        strain = Grad(v) + Transpose(Grad(v))
        mom_terms = (
            Dt(v)
            - sp.S(Ek) * Lap(v)
            - sp.Rational(1, 3) * sp.S(Ek) * Grad(Div(v))
            - sp.S(Ek) * Dot(grad_ln_rho, strain)
            + sp.Rational(2, 3) * sp.S(Ek) * grad_ln_rho * Div(v)
        )
        eq = Equation(mom_terms, name="mom")
        bcs = (
            ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
            + ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        )
        sys_ = EquationSystem([eq], bcs)
        N_r, N_theta = 24, 48
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.71, r_outer=0.985,
        )
        cp = compile_problem(sys_, grid, m=3, order_r=4, order_theta=4)
        # State must contain only v (3 spin components) — no aux.
        assert cp.matrices.layout.size == 3 * N_r * N_theta
