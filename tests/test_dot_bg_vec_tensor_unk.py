"""Test the walker pattern for ``Dot(background_vector, tensor_expression_with_unknown_vector)``.

This is the operator needed for the compressible viscous force in Bekki
et al. (2022) §2.3:
    (1/ρ_0) ∇·(ρ_0 ν T)  =  ν [∇·T + ∇ln ρ_0 · T]
where T is the strain tensor.  The first term ``∇·T`` is handled by the
existing ``TASK_DIV_TENSOR_VECTOR``; this test exercises the second
term ``Dot(∇ln ρ_0, T)`` via the new ``TASK_DOT_BG_VEC_TENSOR_UNK``
walker pattern.

V19 contraction rule (covariant tensor from the reducer):
    (Dot(a, T))^{σ_out} = Σ_{σ_in} a^{σ_in} · T_{σ_in, -σ_out}
The ``-σ_out`` second-index pairing comes from the V19 raising/lowering
convention.

Verification strategy:
  1. **Walker recognition**: ``Dot(bg_vec, Grad(v))`` (bare FieldRef
     inside Grad) emits the specialized ``TASK_DOT_BG_GRAD_VEC``;
     tensor expressions emit the general ``TASK_DOT_BG_VEC_TENSOR_UNK``.
  2. **Cross-validation**: ``Dot(a, Transpose(Transpose(Grad(v))))``
     gives the same matrix as ``Dot(a, Grad(v))`` (both reduce to the
     same operator, but via different dispatchers).  This is a strong
     end-to-end check that the tensor path is consistent with the
     advection-like path.
  3. **Strain-tensor linearity**: ``Dot(a, Grad(v) + (Grad v)^T)``
     equals the sum of the two contractions for radial-only ``a``.
  4. **Manufactured solution**: for a radial-only background and a
     ẑ-aligned test vector, the operator output is computed
     analytically and compared to the discretized result.
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import pytest

from spdee import (
    Unknown, Background, Dt, Lap, Grad, Div, Dot, Transpose,
    Equation, EquationSystem, BoundaryCondition, Boundary,
    r_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import reshape_2d, flatten_2d


# ============================================================================
# Walker-level tests
# ============================================================================

class TestWalkerRecognition:
    """Verify the walker dispatches to the right task kind for each
    expression structure."""

    def test_bare_grad_still_uses_old_path(self):
        """``Dot(bg_vec, Grad(v))`` (bare FieldRef inside Grad) emits
        ``TASK_DOT_BG_GRAD_VEC``, the specialized dispatcher —
        not the general tensor-expression dispatcher.
        """
        from spdee.ir.tasks import (
            TASK_DOT_BG_GRAD_VEC, TASK_DOT_BG_VEC_TENSOR_UNK,
        )
        from spdee.ir.walker import walk_equation_system
        t_sym = sp.symbols("theta", positive=True)
        a_bg = Background.Vector(
            "a_bg", v_r=sp.cos(t_sym), v_theta=sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        sys_ = EquationSystem(
            [Equation(Dot(a_bg, Grad(v)), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")],
        )
        tasks = walk_equation_system(sys_, m=0)
        kinds = {t.kind for t in tasks}
        assert TASK_DOT_BG_GRAD_VEC in kinds
        assert TASK_DOT_BG_VEC_TENSOR_UNK not in kinds

    def test_transpose_grad_uses_new_path(self):
        """``Dot(bg_vec, Transpose(Grad(v)))`` emits the new task
        kind because the operand of Dot is not a bare ``Grad(FieldRef)``.
        """
        from spdee.ir.tasks import (
            TASK_DOT_BG_GRAD_VEC, TASK_DOT_BG_VEC_TENSOR_UNK,
        )
        from spdee.ir.walker import walk_equation_system
        t_sym = sp.symbols("theta", positive=True)
        a_bg = Background.Vector(
            "a_bg", v_r=sp.cos(t_sym), v_theta=sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        sys_ = EquationSystem(
            [Equation(Dot(a_bg, Transpose(Grad(v))), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")],
        )
        tasks = walk_equation_system(sys_, m=0)
        kinds = {t.kind for t in tasks}
        assert TASK_DOT_BG_VEC_TENSOR_UNK in kinds
        assert TASK_DOT_BG_GRAD_VEC not in kinds

    def test_strain_uses_new_path(self):
        """``Dot(bg_vec, Grad(v) + Transpose(Grad(v)))`` emits the
        new task kind.
        """
        from spdee.ir.tasks import TASK_DOT_BG_VEC_TENSOR_UNK
        from spdee.ir.walker import walk_equation_system
        t_sym = sp.symbols("theta", positive=True)
        a_bg = Background.Vector(
            "a_bg", v_r=sp.cos(t_sym), v_theta=sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        sys_ = EquationSystem(
            [Equation(Dot(a_bg, strain), name="eq")],
            [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")],
        )
        tasks = walk_equation_system(sys_, m=0)
        kinds = {t.kind for t in tasks}
        assert TASK_DOT_BG_VEC_TENSOR_UNK in kinds


# ============================================================================
# Cross-validation tests
# ============================================================================

class TestCrossValidationAgainstAdvection:
    """The tensor-expression dispatcher must produce identical numerical
    matrices to the ``DOT_BG_GRAD_VEC`` dispatcher when given an
    expression that simplifies to ``Grad(v)``.

    This is the strongest end-to-end check: two completely independent
    code paths through the IR / walker / dispatcher must yield the same
    sparse matrix.
    """

    @staticmethod
    def _compile_dot_with_grad_v(bg_vec, m_val):
        """Build a system with ``Dot(bg_vec, Grad(v))`` and return its K
        matrix."""
        v = Unknown.Vector("v")
        eq = Equation(Dt(v) + Dot(bg_vec, Grad(v)), name="mom")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=12, N_theta=24, r_inner=0.5, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=4, order_theta=4)
        return cp.matrices.A_total.toarray()

    @staticmethod
    def _compile_dot_with_transpose_transpose(bg_vec, m_val):
        """Build a system with ``Dot(bg_vec, Transpose(Transpose(Grad(v))))``
        — semantically identical to ``Dot(bg_vec, Grad(v))`` but goes
        through the tensor-expression dispatcher.
        """
        v = Unknown.Vector("v")
        op = Transpose(Transpose(Grad(v)))
        eq = Equation(Dt(v) + Dot(bg_vec, op), name="mom")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=12, N_theta=24, r_inner=0.5, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=4, order_theta=4)
        return cp.matrices.A_total.toarray()

    @pytest.mark.parametrize("m_val", [0, 1, 3, 5])
    def test_radial_bg(self, m_val):
        """Pure radial bg vector."""
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        K_old = self._compile_dot_with_grad_v(a_bg, m_val)
        K_new = self._compile_dot_with_transpose_transpose(a_bg, m_val)
        diff = np.max(np.abs(K_old - K_new))
        K_norm = max(np.max(np.abs(K_old)), 1e-12)
        assert diff / K_norm < 1e-12, (
            f"m={m_val}: new dispatcher disagrees with existing one. "
            f"max|K_old - K_new| = {diff:.3e}, "
            f"||K_old||_max = {K_norm:.3e}"
        )

    @pytest.mark.parametrize("m_val", [0, 2, 4])
    def test_general_bg(self, m_val):
        """Non-trivial bg vector with both radial and polar components."""
        t_sym = sp.symbols("theta", positive=True)
        a_bg = Background.Vector(
            "a_bg",
            v_r=sp.cos(t_sym),
            v_theta=sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        K_old = self._compile_dot_with_grad_v(a_bg, m_val)
        K_new = self._compile_dot_with_transpose_transpose(a_bg, m_val)
        diff = np.max(np.abs(K_old - K_new))
        K_norm = max(np.max(np.abs(K_old)), 1e-12)
        assert diff / K_norm < 1e-12, (
            f"m={m_val}: new dispatcher disagrees on general bg. "
            f"max|K_old - K_new| = {diff:.3e}"
        )


# ============================================================================
# Strain-tensor linearity test
# ============================================================================

class TestStrainTensorLinearity:
    """The strain ``T = Grad(v) + Transpose(Grad(v))`` is a Sum, and
    ``reduce_tensor_expression`` should handle it by summing the
    spin components of each piece. Therefore:
        Dot(a, Grad(v) + (Grad v)^T) = Dot(a, Grad(v)) + Dot(a, (Grad v)^T)
    by linearity of the contraction.

    We verify this by building each side separately, both through the
    tensor dispatcher (using ``Transpose(Transpose(Grad(v)))`` to force
    the tensor path for the bare-Grad piece), and checking matrix
    equality.
    """

    def test_strain_decomposes_linearly(self):
        t_sym = sp.symbols("theta", positive=True)
        a_bg = Background.Vector(
            "a_bg", v_r=sp.cos(t_sym), v_theta=sp.sin(t_sym),
            v_phi=sp.S.Zero,
        )
        grid = make_shell_grid(
            N_r=12, N_theta=24, r_inner=0.5, r_outer=1.0,
        )
        m_val = 2
        v = Unknown.Vector("v")

        # System A: Dot(a, strain)
        strain = Grad(v) + Transpose(Grad(v))
        eq_A = Equation(Dt(v) + Dot(a_bg, strain), name="mom")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro"),
        ]
        sys_A = EquationSystem([eq_A], bcs)
        cp_A = compile_problem(sys_A, grid, m=m_val, order_r=4, order_theta=4)
        K_A = cp_A.matrices.A_total.toarray()

        # System B: Dot(a, T^T^T) + Dot(a, T^T) where T = Grad(v).
        # Both go through the tensor dispatcher.
        eq_B = Equation(
            Dt(v)
            + Dot(a_bg, Transpose(Transpose(Grad(v))))
            + Dot(a_bg, Transpose(Grad(v))),
            name="mom",
        )
        sys_B = EquationSystem([eq_B], bcs)
        cp_B = compile_problem(sys_B, grid, m=m_val, order_r=4, order_theta=4)
        K_B = cp_B.matrices.A_total.toarray()

        diff = np.max(np.abs(K_A - K_B))
        norm = max(np.max(np.abs(K_A)), 1e-12)
        assert diff / norm < 1e-12, (
            f"Strain decomposition test failed: "
            f"max|K_A - K_B| = {diff:.3e}, ||K_A||_max = {norm:.3e}"
        )


# ============================================================================
# Manufactured-solution test (interior accuracy)
# ============================================================================

class TestManufacturedSolution:
    """Apply the assembled operator to a known analytical input and
    compare against the analytical answer at interior grid points.

    Test problem:
        a = r̂ (constant magnitude 1)
        v = (1 - r²) ẑ at m = 0
    
    For this combination the strain tensor T = ∇v + (∇v)^T has:
        T_{r,r} = 2 ∂_r v_r = -4 r cos θ
        T_{r,θ} = ∂_r v_θ + (1/r) ∂_θ v_r - v_θ/r = 2r sin θ
        T_{r,φ} = 0 (because v_φ ≡ 0 and m = 0)
    
    Then for radial a (a_r = 1, a_θ = a_φ = 0):
        (Dot(a, T))_r = a_r T_{r,r} = -4 r cos θ
        (Dot(a, T))_θ = a_r T_{r,θ} = 2r sin θ
        (Dot(a, T))_φ = 0
    
    Converting to V19 contravariant spin (multiplied by r since the
    dispatcher emits ``r·(Dot(a, T))^σ``):
        r·(Dot(a, T))^0 = r·(Dot(a, T))_r = -4 r² cos θ
        r·(Dot(a, T))^± = r·((Dot(a, T))_θ ± i (Dot(a, T))_φ)/√2
                        = r·(2r sin θ)/√2  (φ-piece zero)
                        = √2 r² sin θ
    """

    def test_zhat_solution_m0(self):
        # We need to extract just the tensor-dispatcher operator block.
        # Strategy: build a minimal system with Dt(v) + Dot(a, T) = 0,
        # extract A_total, peel off the Dt block, and what remains is
        # exactly the Dot(a, T) contribution (since A_eq for this
        # equation is diag(r) — Dot is r-power 1).
        #
        # Cleaner: use the operator builder directly.
        from spdee.backends.fd2.dot_bg_vec_tensor_unk_sparse import (
            build_r_dot_bg_vec_tensor_unk_block,
        )
        from spdee.backends.fd2.operators import DerivativeBundle
        from spdee.ir.tensor_reduce import (
            vector_unknown_spin_placeholders, reduce_tensor_expression,
        )
        from spdee.backends.common.field_eval import (
            evaluate_vector_background_spin_2d,
        )
        from spdee.ir.fields import r as r_sym

        v = Unknown.Vector("v")
        a_bg = Background.Vector(
            "a_bg", v_r=sp.S.One, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )

        m_val = 0
        grid = make_shell_grid(
            N_r=60, N_theta=120, r_inner=0.3, r_outer=1.0,
        )
        bundle = DerivativeBundle(grid=grid, order_r=4, order_theta=4)

        v_placeholders = {v.name: vector_unknown_spin_placeholders(v)}
        strain = Grad(v) + Transpose(Grad(v))
        reduced = reduce_tensor_expression(
            strain, m=m_val, unknown_spin_placeholders=v_placeholders,
        )
        rT_components = {
            (s_a, s_b): sp.expand(r_sym * reduced.components[(s_a, s_b)])
            for s_a in (-1, 0, +1) for s_b in (-1, 0, +1)
        }
        R, T = grid.mesh()
        a_grid = evaluate_vector_background_spin_2d(a_bg, R, T)

        block = build_r_dot_bg_vec_tensor_unk_block(
            a_grid=a_grid,
            rT_components=rT_components,
            v_placeholders=v_placeholders[v.name],
            bundle=bundle,
            m=m_val,
            v_unknown_name=v.name,
        )

        # State for v = (1 - r²) ẑ
        v_r_arr = (1 - R**2) * np.cos(T)
        v_t_arr = -(1 - R**2) * np.sin(T)
        v_minus = v_t_arr / np.sqrt(2)
        v_zero = v_r_arr
        v_plus = v_t_arr / np.sqrt(2)
        state = np.concatenate([
            flatten_2d(v_minus), flatten_2d(v_zero), flatten_2d(v_plus),
        ])
        out = block @ state

        # Expected (r-multiplied):
        #   r·(Dot)^0 = -4 r² cos θ
        #   r·(Dot)^± = √2 r² sin θ
        N = grid.N_r * grid.N_theta
        out_m = reshape_2d(out[:N], grid.N_r, grid.N_theta)
        out_0 = reshape_2d(out[N:2*N], grid.N_r, grid.N_theta)
        out_p = reshape_2d(out[2*N:], grid.N_r, grid.N_theta)
        exact_0 = -4 * R**2 * np.cos(T)
        exact_pm = np.sqrt(2) * R**2 * np.sin(T)

        def L2(a, b, sl_r=slice(3, -3), sl_t=slice(3, -3)):
            diff = (a - b)[sl_r, sl_t]
            denom = max(np.linalg.norm(b[sl_r, sl_t]), 1e-12)
            return np.linalg.norm(diff) / denom

        e0 = L2(out_0, exact_0)
        em = L2(out_m, exact_pm)
        ep = L2(out_p, exact_pm)
        # All three should be at 4th-order accuracy ~1e-6 at this grid.
        assert e0 < 1e-6, f"σ=0 err = {e0:.3e}"
        assert em < 1e-6, f"σ=- err = {em:.3e}"
        assert ep < 1e-6, f"σ=+ err = {ep:.3e}"

    def test_convergence_4th_order(self):
        """Refining the grid by 2× reduces interior error by at least
        8× (consistent with 4th-order FD convergence, allowing slack
        for the leading-order constant).

        We pick a manufactured solution with a NON-polynomial radial
        profile so that 4th-order FD has nontrivial truncation error
        (otherwise the test runs into machine precision on all grids
        and the ratio test is meaningless).
        """
        from spdee.backends.fd2.dot_bg_vec_tensor_unk_sparse import (
            build_r_dot_bg_vec_tensor_unk_block,
        )
        from spdee.backends.fd2.operators import DerivativeBundle
        from spdee.ir.tensor_reduce import (
            vector_unknown_spin_placeholders, reduce_tensor_expression,
        )
        from spdee.backends.common.field_eval import (
            evaluate_vector_background_spin_2d,
        )
        from spdee.ir.fields import r as r_sym

        v = Unknown.Vector("v")
        # Radial background a_r(r) = sin(2π r); non-trivial structure.
        a_bg = Background.Vector(
            "a_bg",
            v_r=sp.sin(2 * sp.pi * sp.symbols("r", positive=True)),
            v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        m_val = 2
        v_placeholders = {v.name: vector_unknown_spin_placeholders(v)}
        # Use just Grad(v) (no Transpose) — simpler reduction with
        # less aggressive cancellation.
        T_expr = Grad(v)
        reduced = reduce_tensor_expression(
            T_expr, m=m_val, unknown_spin_placeholders=v_placeholders,
        )
        rT_components = {
            (s_a, s_b): sp.expand(r_sym * reduced.components[(s_a, s_b)])
            for s_a in (-1, 0, +1) for s_b in (-1, 0, +1)
        }

        # Manufactured v with non-polynomial structure:
        #   v^0 = sin(2π r) · cos θ
        #   v^- = v^+ = 0
        errs = []
        for N_r in [40, 80, 160]:
            grid = make_shell_grid(
                N_r=N_r, N_theta=2 * N_r, r_inner=0.3, r_outer=1.0,
            )
            bundle = DerivativeBundle(grid=grid, order_r=4, order_theta=4)
            R, T = grid.mesh()
            a_grid = evaluate_vector_background_spin_2d(a_bg, R, T)
            block = build_r_dot_bg_vec_tensor_unk_block(
                a_grid=a_grid,
                rT_components=rT_components,
                v_placeholders=v_placeholders[v.name],
                bundle=bundle,
                m=m_val,
                v_unknown_name=v.name,
            )
            v0_arr = np.sin(2 * np.pi * R) * np.cos(T)
            state = np.concatenate([
                flatten_2d(np.zeros_like(R)),
                flatten_2d(v0_arr),
                flatten_2d(np.zeros_like(R)),
            ])
            out = block @ state
            # For Dot(a, Grad(v)) with a = a_r(r) r̂ and v^0 = f(r) g(θ),
            # v^- = v^+ = 0, the analytical r·(Dot)^0 = r·a_r·∂_r v^0 = 
            #   r · sin(2π r) · (2π cos(2π r)) · cos θ.
            # We just check the σ=0 component.
            exact_0 = (
                R * np.sin(2 * np.pi * R)
                * (2 * np.pi * np.cos(2 * np.pi * R))
                * np.cos(T)
            )
            N = grid.N_r * grid.N_theta
            out_0 = reshape_2d(out[N:2*N], grid.N_r, grid.N_theta)
            sl = slice(3, -3)
            diff = (out_0 - exact_0)[sl, sl]
            denom = max(np.linalg.norm(exact_0[sl, sl]), 1e-12)
            errs.append(np.linalg.norm(diff) / denom)
        # Each doubling should reduce err by at least factor 8 (we
        # expect ~16 for 4th order, allow margin).
        assert errs[0] / errs[1] > 8.0, (
            f"Convergence too slow at 40→80: {errs[0]:.3e} → {errs[1]:.3e}"
        )
        assert errs[1] / errs[2] > 8.0, (
            f"Convergence too slow at 80→160: {errs[1]:.3e} → {errs[2]:.3e}"
        )


# ============================================================================
# Solar-background integration test (the actual Bekki use case)
# ============================================================================

class TestSolarIntegration:
    """The motivating use case: ``Dot(∇ln ρ_0, strain)`` on a solar
    background should compile without error inside a system that also
    has ``Lap(v)`` and ``Grad(Div(v))``.
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

    def test_solar_viscous_term_compiles(self, solar_bg):
        """The full Bekki viscous force
            ν·[Lap(v) + (1/3)·Grad(Div(v)) + Dot(grad_ln_rho_0, strain)]
        compiles cleanly on a solar background, without introducing
        any auxiliary state.
        """
        # Build grad_ln_rho_0 as a Background.Vector (radial only).
        # ∇ln ρ_0 = (1/ρ_0) ∇ρ_0; the solar adapter provides ∇ρ_0
        # as ``solar.grad_rho``, so divide by ρ_0.
        rho_eval = solar_bg.rho.component
        grad_rho_r_eval = solar_bg.grad_rho.components["v_r"]

        def grad_ln_rho_r_fn(R, T):
            return (
                grad_rho_r_eval.evaluate_on_grid(R, T)
                / rho_eval.evaluate_on_grid(R, T)
            )
        from spdee import NumericComponent
        grad_ln_rho = Background.Vector(
            "grad_ln_rho",
            v_r=NumericComponent(
                f=grad_ln_rho_r_fn, description="∂_r ln ρ_0"),
            v_theta=None, v_phi=None,
        )

        v = Unknown.Vector("v")
        Ek = 3e-3
        # Strain tensor (with trace subtraction would need an I tensor;
        # for compile test we use just the symmetric part).
        strain = Grad(v) + Transpose(Grad(v))
        mom_terms = (
            Dt(v)
            - sp.S(Ek) * Lap(v)
            - sp.S(Ek) / 3 * Grad(Div(v))
            - sp.S(Ek) * Dot(grad_ln_rho, strain)
        )
        eq = Equation(mom_terms, name="mom")
        from spdee import ImpenetrableStressFree
        bcs = (
            ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
            + ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        )
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(
            N_r=24, N_theta=48, r_inner=0.71, r_outer=0.985,
        )
        cp = compile_problem(sys_, grid, m=3, order_r=4, order_theta=4)
        # State size = 3·N (vector v only, no aux unknowns)
        N = 24 * 48
        assert cp.matrices.layout.size == 3 * N
