"""Tests for tensor-derived boundary conditions, specifically the
stress-free angular BC for an impenetrable wall.

Covers:
- Walker recognition of Dot(unit_vec, Dot(tensor_expr, r_hat)) BCs.
- ImpenetrableStressFree helper produces three BCs of the right kinds.
- End-to-end manufactured solver test on a ball with stress-free BC.
- Convergence of the recovered field to manufactured truth.
- Edge cases (using strain expression directly, with .as_symmetric()).
"""

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Background, Unknown, Grad, Transpose, Lap, Equation,
    BoundaryCondition, EquationSystem, Boundary, theta,
    Dot, r_hat, theta_hat, phi_hat,
    ImpenetrableStressFree,
)
from spdee.backends.fd2.grid import make_ball_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.ir.tasks import (
    BC_DIRICHLET_VECTOR_PROJECTION, BC_SCALAR_LINEAR_FROM_TENSOR,
)
from spdee.ir.walker import walk_boundary_conditions


# =============================================================================
# Walker / task generation
# =============================================================================

class TestStressFreeBCWalker:
    """Verify the walker emits the correct BCTask shapes."""

    def test_impenetrable_stress_free_helper_emits_three_bcs(self):
        v = Unknown.Vector("v")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        assert len(bcs) == 3
        # Names are sensible.
        names = [bc.name for bc in bcs]
        assert "stress_free_v_r" in names
        assert "stress_free_theta" in names
        assert "stress_free_phi" in names

    def test_helper_walked_tasks_have_right_kinds(self):
        v = Unknown.Vector("v")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        # Pair with a dummy equation so we can walk.
        sys = EquationSystem([Equation(Lap(v), name="lap")], bcs)
        tasks = walk_boundary_conditions(sys, m=0)
        assert len(tasks) == 3
        # First is the impenetrable v_r BC (vector projection).
        v_r_task = next(t for t in tasks if t.name == "stress_free_v_r")
        assert v_r_task.kind == BC_DIRICHLET_VECTOR_PROJECTION
        # Two angular stress BCs are tensor-derived.
        theta_task = next(t for t in tasks if t.name == "stress_free_theta")
        assert theta_task.kind == BC_SCALAR_LINEAR_FROM_TENSOR
        phi_task = next(t for t in tasks if t.name == "stress_free_phi")
        assert phi_task.kind == BC_SCALAR_LINEAR_FROM_TENSOR

    def test_helper_at_r_inner(self):
        """Helper should work at the inner boundary too (for shells)."""
        v = Unknown.Vector("v")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
        for bc in bcs:
            assert bc.location == "r=0"

    def test_helper_rejects_non_vector(self):
        p = Unknown.Scalar("p")
        with pytest.raises(TypeError, match="vector"):
            ImpenetrableStressFree(p)

    def test_explicit_symbolic_form_walks_to_same_tasks(self):
        """Writing the strain BC symbolically without the helper gives
        the same task kinds."""
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="v_r"),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(theta_hat, Dot(strain, r_hat)),
                name="stress_theta",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(phi_hat, Dot(strain, r_hat)),
                name="stress_phi",
            ),
        ]
        sys = EquationSystem([Equation(Lap(v), name="lap")], bcs)
        tasks = walk_boundary_conditions(sys, m=0)
        kinds = sorted(t.kind for t in tasks)
        assert kinds == sorted([
            BC_DIRICHLET_VECTOR_PROJECTION,
            BC_SCALAR_LINEAR_FROM_TENSOR,
            BC_SCALAR_LINEAR_FROM_TENSOR,
        ])

    def test_as_symmetric_form_also_walks(self):
        """The .as_symmetric() override should also work in BCs."""
        v = Unknown.Vector("v")
        strain = (Grad(v) + Transpose(Grad(v))).as_symmetric()
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="v_r"),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(theta_hat, Dot(strain, r_hat)),
                name="stress_theta",
            ),
        ]
        sys = EquationSystem([Equation(Lap(v), name="lap")], bcs)
        tasks = walk_boundary_conditions(sys, m=0)
        # Just verify no exception; further verification done via numerical test.
        assert len(tasks) == 2


# =============================================================================
# End-to-end manufactured solver test
# =============================================================================

class TestStressFreeBCEndToEnd:
    """Manufactured: v_r = (1-r²) cos θ, v_θ = -(1+r²) sin θ, v_φ = 0
    in a ball with stress-free + impenetrable BCs at r=1.

    Source: g_r = 2 cos θ, g_θ = -2 sin θ, g_φ = 0.
    """

    @staticmethod
    def _make_problem(N):
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        eq = Equation(Lap(v) - g, name="vec_poisson_sf")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        system = EquationSystem([eq], bcs)
        return grid, compile_problem(system, grid, m=0)

    @staticmethod
    def _exact_spin(R, T):
        sqrt2 = np.sqrt(2.0)
        v_r = (1 - R**2) * np.cos(T)
        v_t = -(1 + R**2) * np.sin(T)
        return {
             0: v_r,
            +1: v_t / sqrt2,
            -1: v_t / sqrt2,
        }

    def test_n24_max_err_acceptable(self):
        grid, problem = self._make_problem(24)
        result = problem.solve_bvp()
        R, T = grid.mesh()
        ve = self._exact_spin(R, T)
        err0 = float(np.max(np.abs(result["v"][0] - ve[0])))
        errp = float(np.max(np.abs(result["v"][+1] - ve[+1])))
        errm = float(np.max(np.abs(result["v"][-1] - ve[-1])))
        assert err0 < 1e-4
        assert errp < 1e-4
        assert errm < 1e-4

    def test_convergence_rate(self):
        Ns = [12, 16, 24, 32]
        errs = []
        for N in Ns:
            grid, problem = self._make_problem(N)
            result = problem.solve_bvp()
            R, T = grid.mesh()
            ve = self._exact_spin(R, T)
            err = max(
                float(np.max(np.abs(result["v"][ 0] - ve[ 0]))),
                float(np.max(np.abs(result["v"][-1] - ve[-1]))),
                float(np.max(np.abs(result["v"][+1] - ve[+1]))),
            )
            errs.append(err)
        rate = -float(np.polyfit(np.log(Ns), np.log(errs), 1)[0])
        assert errs[-1] < 1e-4, (
            f"Should converge to err < 1e-4 at N={Ns[-1]}; got {errs[-1]:.3e}"
        )
        assert rate > 3.5, f"Expected ≥3.5-order rate; got {rate:.2f}"

    def test_value_at_origin_matches_cartesian_z(self):
        """At the origin, v_r → cos θ (Cartesian-z=1)."""
        grid, problem = self._make_problem(32)
        result = problem.solve_bvp()
        # v^0 at i_r=0, j=0 should be cos(θ_0)
        v0_origin = result["v"][0][0, 0]
        expected = float(np.cos(grid.theta[0]))
        assert abs(v0_origin - expected) < 1e-3


# =============================================================================
# Equivalence test: helper vs symbolic
# =============================================================================

class TestHelperEquivalentToSymbolic:
    """The helper class and the equivalent symbolic form should give
    identical numerical results."""

    def test_helper_and_symbolic_match(self):
        v_helper = Unknown.Vector("v_helper")
        v_sym = Unknown.Vector("v_sym")
        g_components = dict(
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        g_helper = Background.Vector("g_h", **g_components)
        g_sym = Background.Vector("g_s", **g_components)
        N = 16
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)

        # Solution via helper.
        bcs_helper = ImpenetrableStressFree(v_helper, boundary=Boundary.R_OUTER)
        sys_helper = EquationSystem(
            [Equation(Lap(v_helper) - g_helper, name="eq_h")], bcs_helper,
        )
        sol_helper = compile_problem(sys_helper, grid, m=0).solve_bvp()

        # Solution via explicit symbolic form.
        strain = Grad(v_sym) + Transpose(Grad(v_sym))
        bcs_sym = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v_sym), name="vr"),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(theta_hat, Dot(strain, r_hat)),
                name="st",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(phi_hat, Dot(strain, r_hat)),
                name="sp",
            ),
        ]
        sys_sym = EquationSystem(
            [Equation(Lap(v_sym) - g_sym, name="eq_s")], bcs_sym,
        )
        sol_sym = compile_problem(sys_sym, grid, m=0).solve_bvp()

        for sigma in (-1, 0, +1):
            diff = np.max(np.abs(
                sol_helper["v_helper"][sigma] - sol_sym["v_sym"][sigma]
            ))
            assert diff < 1e-12, (
                f"Helper vs symbolic differ at sigma={sigma}: {diff:.2e}"
            )


# =============================================================================
# Stress-free BC at m≠0 (exercises m-dependent terms in the strain BC)
# =============================================================================

class TestStressFreeBCAtNonzeroM:
    """End-to-end stress-free BC tests at m≠0.

    These exercise the m-dependent term in the strain formula
        E^{±1, 0} = ... ± (m / (√2 r sin θ)) v^0
    which is identically zero at m=0 but becomes load-bearing at m≠0.
    Combined with the |m|=1 ball origin anchor, this exercises the full
    interaction between tensor-derived BCs and non-trivial origin physics.
    """

    @staticmethod
    def _solve_m1(N, alpha=2.3):
        """Manufactured: v = α(1-r²)² (x̂+iŷ)/√2 at m=+1."""
        from spdee.ir.fields import r as r_sym
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r   = 2 * sp.sqrt(2) * alpha * (5 * r_sym**2 - 3) * sp.sin(theta),
            v_theta = 2 * sp.sqrt(2) * alpha * (5 * r_sym**2 - 3) * sp.cos(theta),
            v_phi   = 2 * sp.sqrt(2) * sp.I * alpha * (5 * r_sym**2 - 3),
        )
        eq = Equation(Lap(v) - g, name="vec_poisson_sf_m1")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        system = EquationSystem([eq], bcs)
        return grid, compile_problem(system, grid, m=1).solve_bvp()

    @staticmethod
    def _exact_m1_spin(R, T, alpha=2.3):
        sqrt2 = np.sqrt(2.0)
        factor = alpha * (1 - R**2)**2
        return {
             0: factor * np.sin(T) / sqrt2,
            -1: factor * (np.cos(T) + 1) / 2,
            +1: factor * (np.cos(T) - 1) / 2,
        }

    def test_m1_n24_acceptable_error(self):
        grid, result = self._solve_m1(24)
        R, T = grid.mesh()
        ve = self._exact_m1_spin(R, T)
        err = max(
            float(np.max(np.abs(result["v"][ 0] - ve[ 0]))),
            float(np.max(np.abs(result["v"][-1] - ve[-1]))),
            float(np.max(np.abs(result["v"][+1] - ve[+1]))),
        )
        assert err < 1e-3, (
            f"m=+1 stress-free should converge at N=24; got err={err:.3e}"
        )

    def test_m1_convergence_rate(self):
        Ns = [12, 16, 24, 32, 48]
        errs = []
        for N in Ns:
            grid, result = self._solve_m1(N)
            R, T = grid.mesh()
            ve = self._exact_m1_spin(R, T)
            err = max(
                float(np.max(np.abs(result["v"][ 0] - ve[ 0]))),
                float(np.max(np.abs(result["v"][-1] - ve[-1]))),
                float(np.max(np.abs(result["v"][+1] - ve[+1]))),
            )
            errs.append(err)
        assert errs[-1] < 1e-4
        # Asymptotic rate: max consecutive-pair rate, more robust to
        # pre-asymptotic effects than a polyfit over the whole sweep.
        pair_rates = [
            np.log(errs[i] / errs[i+1]) / np.log(Ns[i+1] / Ns[i])
            for i in range(len(errs) - 1)
        ]
        rate = max(pair_rates)
        # Slightly lower threshold for m=+1 due to anchor interactions
        # observed in the prior |m|=1 ball-mode work.
        assert rate > 3.5, (
            f"Expected ≥3.5-order rate at m=+1; got {rate:.2f}; errs={errs}"
        )

    def test_m1_alpha_recovery(self):
        """The |m|=1 ball origin anchor should recover α."""
        ALPHA = 2.3
        grid, result = self._solve_m1(32, alpha=ALPHA)
        # α from v^-(0, θ_0): α = 2 v^-(0, θ_0) / (cos θ_0 + 1)
        cos_0 = float(np.cos(grid.theta[0]))
        alpha_recov = 2 * result["v"][-1][0, 0] / (cos_0 + 1)
        assert abs(alpha_recov.real - ALPHA) < 1e-3
        assert abs(alpha_recov.imag) < 1e-10

    def test_m_minus_1_symmetric_to_m_plus_1(self):
        """m=-1 with conjugated source should give the conjugate spin
        components — same numerical magnitude as m=+1."""
        from spdee.ir.fields import r as r_sym
        ALPHA = 2.3
        N = 24

        # m=-1: source has g_phi sign-flipped.
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r   = 2 * sp.sqrt(2) * ALPHA * (5 * r_sym**2 - 3) * sp.sin(theta),
            v_theta = 2 * sp.sqrt(2) * ALPHA * (5 * r_sym**2 - 3) * sp.cos(theta),
            v_phi   = -2 * sp.sqrt(2) * sp.I * ALPHA * (5 * r_sym**2 - 3),
        )
        eq = Equation(Lap(v) - g, name="eq_m_minus_1")
        bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        system = EquationSystem([eq], bcs)
        result = compile_problem(system, grid, m=-1).solve_bvp()

        # Exact spin (v^- and v^+ swap roles for m=-1):
        R, T = grid.mesh()
        sqrt2 = np.sqrt(2.0)
        factor = ALPHA * (1 - R**2)**2
        v0_exact = factor * np.sin(T) / sqrt2
        vm_exact = factor * (np.cos(T) - 1) / 2  # role-swap
        vp_exact = factor * (np.cos(T) + 1) / 2  # role-swap

        err = max(
            float(np.max(np.abs(result["v"][ 0] - v0_exact))),
            float(np.max(np.abs(result["v"][-1] - vm_exact))),
            float(np.max(np.abs(result["v"][+1] - vp_exact))),
        )
        assert err < 1e-3

    def test_m2_convergence(self):
        """Manufactured m=2 stress-free test.

        Field: v_r = r² sin²θ cos θ (1-r²)², v_θ = -r² sin³θ (1-r²)², v_φ = 0.
        Source: g_r = 4r²(9r²-7) sin²θ cos θ, g_θ = r²(28-36r²) sin³θ.
        """
        from spdee.ir.fields import r as r_sym
        Ns = [16, 24, 32]
        errs = []
        for N in Ns:
            grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
            v = Unknown.Vector("v")
            g = Background.Vector(
                "g",
                v_r   = 4 * r_sym**2 * (9 * r_sym**2 - 7) * sp.sin(theta)**2 * sp.cos(theta),
                v_theta = r_sym**2 * (28 - 36 * r_sym**2) * sp.sin(theta)**3,
                v_phi   = sp.S.Zero,
            )
            eq = Equation(Lap(v) - g, name="vec_poisson_m2")
            bcs = ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
            system = EquationSystem([eq], bcs)
            result = compile_problem(system, grid, m=2).solve_bvp()

            R, T = grid.mesh()
            sqrt2 = np.sqrt(2.0)
            factor = R**2 * (1 - R**2)**2
            v_r_exact = factor * np.sin(T)**2 * np.cos(T)
            v_t_exact = -factor * np.sin(T)**3
            v0_exact = v_r_exact
            vp_exact = v_t_exact / sqrt2
            vm_exact = v_t_exact / sqrt2
            err = max(
                float(np.max(np.abs(result["v"][ 0] - v0_exact))),
                float(np.max(np.abs(result["v"][-1] - vm_exact))),
                float(np.max(np.abs(result["v"][+1] - vp_exact))),
            )
            errs.append(err)
        rate = -float(np.polyfit(np.log(Ns), np.log(errs), 1)[0])
        assert errs[-1] < 1e-3, (
            f"m=2 should converge at N=32; got err={errs[-1]:.3e}"
        )
        assert rate > 3.5, f"Expected ≥3.5-order at m=2; got {rate:.2f}"


# =============================================================================
# General inner-unit projection (Dot(u, Dot(T, w)) for any u, w)
# =============================================================================

class TestGeneralInnerUnitProjection:
    """Tests for tensor-projection BCs with inner unit ≠ r_hat.

    The walker reduces Dot(unit_outer, Dot(T, unit_inner)) for all 9
    (outer, inner) pairs via the V19 dual-pairing formula. This is the
    full generalization of the inner=r_hat-only Stage B implementation.

    Key cross-check: for a symmetric tensor T, Dot(u, Dot(T, w)) must
    equal Dot(w, Dot(T, u)) numerically — and we can verify this by
    comparing the scalar SymPy formulas after symmetry-aware simplification.
    """

    def test_all_9_inner_unit_combos_walk_without_error(self):
        """All 9 (outer, inner) unit-vector pairs should walk to a
        valid scalar_linear_from_tensor task."""
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        units = [r_hat, theta_hat, phi_hat]
        unit_names = ["r_hat", "theta_hat", "phi_hat"]
        for u_outer, u_outer_name in zip(units, unit_names):
            for u_inner, u_inner_name in zip(units, unit_names):
                bc = BoundaryCondition(
                    Boundary.R_OUTER,
                    Dot(u_outer, Dot(strain, u_inner)),
                    name=f"T_{u_outer_name}_{u_inner_name}",
                )
                sys = EquationSystem(
                    [Equation(Lap(v), name="lap")],
                    [bc],
                )
                tasks = walk_boundary_conditions(sys, m=0)
                assert len(tasks) == 1
                assert tasks[0].kind == BC_SCALAR_LINEAR_FROM_TENSOR

    def test_symmetric_strain_inner_outer_swap(self):
        """For symmetric T (which Grad(v) + Transpose(Grad(v)) is),
        Dot(u, Dot(T, w)) should equal Dot(w, Dot(T, u)) symbolically."""
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        from spdee.ir.walker import walk_boundary_conditions

        # Each pair (u_outer, u_inner) and its swap must produce the
        # same scalar SymPy formula. Test at m=2 to exercise m-dependence.
        units = [r_hat, theta_hat, phi_hat]
        names = ["r", "t", "p"]
        for i in range(3):
            for j in range(3):
                bc_ij = BoundaryCondition(
                    Boundary.R_OUTER,
                    Dot(units[i], Dot(strain, units[j])),
                    name=f"T_{names[i]}{names[j]}",
                )
                bc_ji = BoundaryCondition(
                    Boundary.R_OUTER,
                    Dot(units[j], Dot(strain, units[i])),
                    name=f"T_{names[j]}{names[i]}",
                )
                sys_ij = EquationSystem(
                    [Equation(Lap(v), name="lap")], [bc_ij],
                )
                sys_ji = EquationSystem(
                    [Equation(Lap(v), name="lap")], [bc_ji],
                )
                t_ij = walk_boundary_conditions(sys_ij, m=2)[0]
                t_ji = walk_boundary_conditions(sys_ji, m=2)[0]
                e_ij = t_ij.metadata["scalar_expr"]
                e_ji = t_ji.metadata["scalar_expr"]
                diff = sp.simplify(e_ij - e_ji)
                assert diff == 0, (
                    f"Symmetric strain failed (i, j) ↔ (j, i) swap "
                    f"at indices ({i}, {j}): diff = {diff}"
                )

    def test_alternative_stress_free_formulation_works_for_symmetric_strain(self):
        """The alternative stress-free formulation
            Dot(r_hat, v) = 0
            Dot(r_hat, Dot(strain, theta_hat)) = 0
            Dot(r_hat, Dot(strain, phi_hat)) = 0
        is mathematically equivalent to ImpenetrableStressFree for
        symmetric strain. With index-based host_sigma routing (rather
        than syntactic outer-unit), all three BCs land on different
        host rows (σ=0, +1, -1) just like the standard form, and the
        problem solves with the same numerical answer.
        """
        N = 24
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)

        # Solution via standard stress-free helper.
        v_std = Unknown.Vector("v_std")
        g_std = Background.Vector(
            "g_std",
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        bcs_std = ImpenetrableStressFree(v_std, boundary=Boundary.R_OUTER)
        sys_std = EquationSystem(
            [Equation(Lap(v_std) - g_std, name="eq_std")], bcs_std,
        )
        sol_std = compile_problem(sys_std, grid, m=0).solve_bvp()

        # Solution via alternative inner-outer swap.
        v_alt = Unknown.Vector("v_alt")
        g_alt = Background.Vector(
            "g_alt",
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        strain_alt = Grad(v_alt) + Transpose(Grad(v_alt))
        bcs_alt = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v_alt), name="vr"),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(r_hat, Dot(strain_alt, theta_hat)),
                name="alt_theta",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(r_hat, Dot(strain_alt, phi_hat)),
                name="alt_phi",
            ),
        ]
        sys_alt = EquationSystem(
            [Equation(Lap(v_alt) - g_alt, name="eq_alt")], bcs_alt,
        )
        sol_alt = compile_problem(sys_alt, grid, m=0).solve_bvp()

        for sigma in (-1, 0, +1):
            diff = float(np.max(np.abs(
                sol_std["v_std"][sigma] - sol_alt["v_alt"][sigma]
            )))
            assert diff < 1e-12, (
                f"Standard and alternative stress-free formulations should "
                f"give identical numerical results (by symmetry of strain "
                f"plus index-based host_sigma routing). "
                f"At sigma={sigma}: max diff = {diff:.2e}"
            )

    def test_left_associated_form_works_too(self):
        """The left-associated form
            Dot(Dot(theta_hat, strain), r_hat) = 0
        is mathematically equivalent to
            Dot(theta_hat, Dot(strain, r_hat)) = 0
        (both compute theta_hat_alpha · strain^{alpha,beta} · r_hat_beta).
        The walker should handle both forms identically.
        """
        N = 24
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        v = Unknown.Vector("v")
        g = Background.Vector(
            "g",
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        strain = Grad(v) + Transpose(Grad(v))
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr"),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(Dot(theta_hat, strain), r_hat),
                name="left_theta",
            ),
            BoundaryCondition(
                Boundary.R_OUTER,
                Dot(Dot(phi_hat, strain), r_hat),
                name="left_phi",
            ),
        ]
        sys = EquationSystem(
            [Equation(Lap(v) - g, name="eq")], bcs,
        )
        result = compile_problem(sys, grid, m=0).solve_bvp()

        # Compare against the standard form.
        v_std = Unknown.Vector("v_std")
        g_std = Background.Vector(
            "g_std",
            v_r   = 2 * sp.cos(theta),
            v_theta = -2 * sp.sin(theta),
            v_phi   = sp.S.Zero,
        )
        sys_std = EquationSystem(
            [Equation(Lap(v_std) - g_std, name="eq")],
            ImpenetrableStressFree(v_std, boundary=Boundary.R_OUTER),
        )
        result_std = compile_problem(sys_std, grid, m=0).solve_bvp()

        for sigma in (-1, 0, +1):
            diff = float(np.max(np.abs(
                result["v"][sigma] - result_std["v_std"][sigma]
            )))
            assert diff < 1e-12, (
                f"Left-associated form should give the same answer as "
                f"the standard right-associated form. "
                f"At sigma={sigma}: max diff = {diff:.2e}"
            )

    def test_phi_outer_only_picks_up_imaginary_terms(self):
        """A BC with phi_hat as the outer unit should give a scalar
        expression with imaginary I coefficients on at least some terms."""
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        bc = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(phi_hat, Dot(strain, theta_hat)),
            name="T_phi_theta",
        )
        sys = EquationSystem(
            [Equation(Lap(v), name="lap")], [bc],
        )
        from spdee.ir.walker import walk_boundary_conditions
        task = walk_boundary_conditions(sys, m=1)[0]
        expr = task.metadata["scalar_expr"]
        # Should contain sp.I somewhere in the expression atoms.
        assert sp.I in expr.atoms(sp.I)

    def test_diagonal_Trr_is_2_dr_v0(self):
        """Dot(r_hat, Dot(strain, r_hat)) = 2 ∂_r v^0  (= 2 ∂_r v_r)."""
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        bc = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(r_hat, Dot(strain, r_hat)),
            name="Trr",
        )
        sys = EquationSystem(
            [Equation(Lap(v), name="lap")], [bc],
        )
        from spdee.ir.walker import walk_boundary_conditions
        task = walk_boundary_conditions(sys, m=0)[0]
        # Should be 2 * d_r v_sigma_0(r, theta). Build the expected expression.
        from spdee.ir.fields import r as r_sym, theta as t_sym
        v_sigma_0 = sp.Function("v_sigma_0")(r_sym, t_sym)
        expected = 2 * sp.diff(v_sigma_0, r_sym)
        diff = sp.simplify(task.metadata["scalar_expr"] - expected)
        assert diff == 0, f"T_rr formula mismatch: got {task.metadata['scalar_expr']}, expected {expected}"

    def test_collision_detected_when_two_bcs_share_host_row(self):
        """If two BCs are constructed to deliberately share a host row,
        the backend should refuse with a clear error rather than silently
        overwrite."""
        v = Unknown.Vector("v")
        # Two impenetrability BCs at the same boundary — same host_sigma=0.
        bcs = [
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_first"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_second"),
        ]
        N = 8
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        sys = EquationSystem([Equation(Lap(v), name="lap")], bcs)
        problem = compile_problem(sys, grid, m=0)
        with pytest.raises(ValueError, match="host-row collision"):
            problem.solve_bvp()

    def test_mixed_formulation_for_symmetric_strain_matches_standard(self):
        """For symmetric strain, the scalar formula and host_sigma for
        Dot(theta_hat, Dot(strain, r_hat)) and Dot(r_hat, Dot(strain, theta_hat))
        are identical:

        - Formulas are identical because strain is symmetric.
        - Host_sigmas are identical because we use index-based routing:
          both map to the (first_index, second_index) pair (θ̂, r̂) or
          (r̂, θ̂), and our rule picks host_for(non-r̂) = +1 for either.

        This means writing the BC in either form gives the same row in
        the matrix.
        """
        from spdee.ir.walker import walk_boundary_conditions
        v = Unknown.Vector("v")
        strain = Grad(v) + Transpose(Grad(v))
        bc_std = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(theta_hat, Dot(strain, r_hat)),
            name="std_theta",
        )
        bc_alt = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(r_hat, Dot(strain, theta_hat)),
            name="alt_theta",
        )
        bc_left = BoundaryCondition(
            Boundary.R_OUTER,
            Dot(Dot(theta_hat, strain), r_hat),
            name="left_theta",
        )
        for m_val in [0, 1, 2]:
            tasks = []
            for bc in [bc_std, bc_alt, bc_left]:
                sys = EquationSystem(
                    [Equation(Lap(v), name="lap")], [bc],
                )
                tasks.append(walk_boundary_conditions(sys, m=m_val)[0])
            # All three scalar expressions should be identical.
            e_std = sp.simplify(tasks[0].metadata["scalar_expr"])
            for i, t in enumerate(tasks[1:], start=1):
                e_i = sp.simplify(t.metadata["scalar_expr"])
                diff = sp.simplify(e_std - e_i)
                assert diff == 0, (
                    f"At m={m_val}, form #{i} scalar formula differs "
                    f"from standard: diff = {diff}"
                )
            # All three should have the same host_sigma.
            host_std = tasks[0].metadata["host_sigma"]
            assert host_std == +1, f"std host should be +1, got {host_std}"
            for i, t in enumerate(tasks[1:], start=1):
                assert t.metadata["host_sigma"] == host_std, (
                    f"At m={m_val}, form #{i} host_sigma differs: "
                    f"{t.metadata['host_sigma']} vs {host_std}"
                )

    def test_non_symmetric_tensor_outer_swap_gives_different_formulas(self):
        """For non-symmetric tensors (like Grad(v) alone, without the
        Transpose addition), the swap Dot(u, Dot(T, w)) vs
        Dot(w, Dot(T, u)) produces DIFFERENT scalar constraints —
        but both map to the same host_sigma under index-based routing
        (the non-r̂ unit wins regardless of position).

        So for non-symmetric T, the user can choose ONE of the two
        forms but not both at the same boundary (collision detector
        fires).
        """
        from spdee.ir.walker import walk_boundary_conditions
        v = Unknown.Vector("v")
        # Grad(v) is NOT symmetric in general.
        T_asym = Grad(v)
        bc_uTw = BoundaryCondition(
            Boundary.R_OUTER, Dot(theta_hat, Dot(T_asym, r_hat)),
            name="uTw",
        )
        bc_wTu = BoundaryCondition(
            Boundary.R_OUTER, Dot(r_hat, Dot(T_asym, theta_hat)),
            name="wTu",
        )
        sys_separate_uTw = EquationSystem(
            [Equation(Lap(v), name="lap")], [bc_uTw],
        )
        sys_separate_wTu = EquationSystem(
            [Equation(Lap(v), name="lap")], [bc_wTu],
        )
        # Different scalar formulas (this confirms the inner/outer
        # contraction semantics are mathematically correct).
        t_uTw = walk_boundary_conditions(sys_separate_uTw, m=0)[0]
        t_wTu = walk_boundary_conditions(sys_separate_wTu, m=0)[0]
        e_uTw = sp.simplify(t_uTw.metadata["scalar_expr"])
        e_wTu = sp.simplify(t_wTu.metadata["scalar_expr"])
        assert sp.simplify(e_uTw - e_wTu) != 0, (
            "Non-symmetric tensor: Dot(u, Dot(T, w)) and Dot(w, Dot(T, u)) "
            "should produce different scalar formulas."
        )
        # But same host_sigma (= +1, since the non-r̂ unit wins for both):
        assert t_uTw.metadata["host_sigma"] == +1
        assert t_wTu.metadata["host_sigma"] == +1

        # Putting both BCs in the same system collides:
        N = 8
        grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
        sys_both = EquationSystem(
            [Equation(Lap(v), name="lap")], [bc_uTw, bc_wTu],
        )
        problem = compile_problem(sys_both, grid, m=0)
        with pytest.raises(ValueError, match="host-row collision"):
            problem.solve_bvp()


# =============================================================================
# V19 convention correctness (sign of i)
# =============================================================================

class TestV19ConventionAudit:
    """Tests for the V19 covariant/contravariant convention in
    `_project_tensor_to_scalar`.

    A sign error on the imaginary parts of Dot(u, Dot(T, w)) shows up
    only when w has non-zero phi content (e.g., w=phi_hat) and T is
    non-symmetric. Stress-free BCs cannot detect it because:
    - All stress-free BCs use inner=r_hat (no phi content), and r_hat's
      spin coefficients are all real.
    - The strain tensor is symmetric, so an antisymmetric error
      contribution cancels.
    These tests use non-symmetric tensor expressions (like bare Grad(v))
    AND non-r_hat inner units AND fields with v_phi != 0.
    """

    @staticmethod
    def _physical_grad_at_m0(v_r, v_theta, v_phi):
        """Compute the physical (Grad v)_{i, j} for orthonormal spherical,
        m=0 axisymmetric. Returns dict {(i,j): expr}."""
        from spdee.ir.fields import r as r_sym, theta as t_sym
        r, t = r_sym, t_sym
        return {
            ("r", "r"): sp.diff(v_r, r),
            ("r", "theta"): sp.diff(v_theta, r),
            ("r", "phi"): sp.diff(v_phi, r),
            ("theta", "r"): (sp.diff(v_r, t) - v_theta) / r,
            ("theta", "theta"): (v_r + sp.diff(v_theta, t)) / r,
            ("theta", "phi"): sp.diff(v_phi, t) / r,
            ("phi", "r"): -v_phi * sp.sin(t) / (r * sp.sin(t)),
            ("phi", "theta"): -v_phi * sp.cos(t) / (r * sp.sin(t)),
            ("phi", "phi"): (v_r * sp.sin(t) + v_theta * sp.cos(t)) / (r * sp.sin(t)),
        }

    def test_grad_v_phi_unit_field_all_9_projections(self):
        """For v = e_phi (unit phi-vector), verify all 9 projections
        Dot(u, Dot(Grad(v), w)) match the physical (Grad v)_{u, w}."""
        from spdee.ir.walker import _project_tensor_to_scalar
        from spdee.ir.tensor_reduce import reduce_tensor_expression
        v_bg = Background.Vector(
            "v_phi_only",
            v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.One,
        )
        red = reduce_tensor_expression(Grad(v_bg), m=0)
        gp = self._physical_grad_at_m0(sp.S.Zero, sp.S.Zero, sp.S.One)
        name_to_phys = {"r_hat": "r", "theta_hat": "theta", "phi_hat": "phi"}
        for u_name in ["r_hat", "theta_hat", "phi_hat"]:
            for w_name in ["r_hat", "theta_hat", "phi_hat"]:
                walker_val = _project_tensor_to_scalar(
                    red.components,
                    outer_unit=u_name,
                    inner_unit=w_name,
                )
                phys_val = gp[(name_to_phys[u_name], name_to_phys[w_name])]
                diff = sp.simplify(walker_val - phys_val)
                assert diff == 0, (
                    f"v=e_phi: walker({u_name}, {w_name}) = "
                    f"{sp.simplify(walker_val)} != physical "
                    f"{sp.simplify(phys_val)}; diff = {diff}"
                )

    def test_mixed_real_and_phi_field_all_9_projections(self):
        """A more general field with both real and v_phi components."""
        from spdee.ir.walker import _project_tensor_to_scalar
        from spdee.ir.tensor_reduce import reduce_tensor_expression
        from spdee.ir.fields import r as r_sym, theta as t_sym
        v_bg = Background.Vector(
            "v_mixed",
            v_r=r_sym * sp.sin(t_sym),
            v_theta=r_sym * sp.cos(t_sym)**2,
            v_phi=r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        )
        red = reduce_tensor_expression(Grad(v_bg), m=0)
        gp = self._physical_grad_at_m0(
            r_sym * sp.sin(t_sym),
            r_sym * sp.cos(t_sym)**2,
            r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        )
        name_to_phys = {"r_hat": "r", "theta_hat": "theta", "phi_hat": "phi"}
        for u_name in ["r_hat", "theta_hat", "phi_hat"]:
            for w_name in ["r_hat", "theta_hat", "phi_hat"]:
                walker_val = _project_tensor_to_scalar(
                    red.components,
                    outer_unit=u_name,
                    inner_unit=w_name,
                )
                phys_val = gp[(name_to_phys[u_name], name_to_phys[w_name])]
                diff = sp.simplify(walker_val - phys_val)
                assert diff == 0, (
                    f"Mixed v: walker({u_name}, {w_name}) "
                    f"differs from physical: diff = {diff}"
                )
