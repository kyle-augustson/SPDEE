"""Tests for tabular background interpolators.

Verifies:
  1. `radial_interpolator` reproduces f(r) on a 2D mesh, with correct
     broadcasting (no theta dependence).
  2. `meridional_interpolator` reproduces f(r, θ) on a 2D mesh.
  3. Boundary policies ('clamp', 'extrapolate', 'error') work.
  4. CompositeComponent multiplies two BackgroundComponents pointwise.
  5. multiply_components keeps the SymPy×SymPy fast path; mixed
     SymPy×Numeric produces a CompositeComponent.
  6. End-to-end: a tabular rho0 used as `rho0 * Dt(v)` produces an
     EVP whose eigenvalues match the closed-form rho0 version to
     interpolation accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Lap, Dt, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat,
    radial_interpolator, meridional_interpolator,
    multiply_components, CompositeComponent, SymPyComponent, NumericComponent,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


class TestRadialInterpolator:
    """Tabular f(r) → 2D-broadcastable BackgroundComponent."""

    def test_cubic_recovers_polynomial(self):
        """Cubic spline on a polynomial sample is exact (to machine eps)."""
        r_t = np.linspace(0.0, 1.5, 30)
        f_t = 1 + r_t ** 2  # polynomial of degree 2 ≤ cubic
        comp = radial_interpolator(r_t, f_t, kind="cubic")
        R = np.linspace(0.1, 1.4, 8).reshape(-1, 1) * np.ones((1, 5))
        T = np.ones_like(R) * np.linspace(0, 3, 5)
        expected = 1 + R ** 2
        result = comp.evaluate_on_grid(R, T)
        assert np.max(np.abs(result - expected)) < 1e-12

    def test_broadcasts_along_theta(self):
        """Radial interpolator ignores theta — result is constant in θ."""
        r_t = np.linspace(0.0, 1.0, 20)
        f_t = np.sin(r_t * 3)
        comp = radial_interpolator(r_t, f_t)
        R = np.full((4, 6), 0.5)
        T = np.linspace(0, np.pi, 6).reshape(1, -1) * np.ones((4, 1))
        result = comp.evaluate_on_grid(R, T)
        # All values should be the same (= f(0.5)) regardless of θ.
        assert np.max(result - result[0, 0]) < 1e-14

    def test_clamp_boundary(self):
        """Queries outside [r_min, r_max] clamp to nearest endpoint."""
        r_t = np.linspace(0.0, 1.0, 20)
        f_t = r_t ** 2  # f(0)=0, f(1)=1
        comp = radial_interpolator(r_t, f_t, boundary="clamp")
        R = np.array([[-0.5, 0.0], [1.0, 1.5], [2.5, 0.5]])
        T = np.zeros_like(R)
        result = comp.evaluate_on_grid(R, T)
        # Expected: f(-0.5) clamps to f(0)=0, f(1.5) and f(2.5) clamp to f(1)=1
        expected = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.25]])
        assert np.allclose(result, expected, atol=1e-10)

    def test_error_boundary_raises(self):
        """boundary='error' raises on out-of-range queries."""
        r_t = np.linspace(0.1, 0.9, 10)
        f_t = r_t ** 2
        comp = radial_interpolator(r_t, f_t, boundary="error")
        R_ok = np.array([[0.2, 0.5, 0.8]])
        T = np.zeros_like(R_ok)
        # In-range: fine.
        comp.evaluate_on_grid(R_ok, T)
        # Out-of-range: should raise.
        R_bad = np.array([[0.2, 0.5, 1.5]])
        with pytest.raises(ValueError):
            comp.evaluate_on_grid(R_bad, T)

    def test_invalid_input_raises(self):
        """Mismatched shapes or non-monotonic r raise at construction."""
        with pytest.raises(ValueError):
            radial_interpolator(np.array([1.0, 2.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            radial_interpolator(np.array([1.0, 0.0, 2.0]),
                                np.array([1.0, 0.0, 4.0]))


class TestMeridionalInterpolator:
    """Tabular f(r, θ) → 2D-mesh BackgroundComponent."""

    def test_linear_recovers_bilinear_function(self):
        """Linear interpolation is exact on a bilinear function."""
        r_t = np.linspace(0.0, 1.0, 30)
        t_t = np.linspace(0.0, np.pi, 40)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        f_t = 2 * R_t + 3 * T_t  # bilinear
        comp = meridional_interpolator(r_t, t_t, f_t, kind="linear")
        R = np.linspace(0.1, 0.9, 7).reshape(-1, 1) * np.ones((1, 5))
        T = np.ones((7, 1)) * np.linspace(0.5, 2.5, 5)
        expected = 2 * R + 3 * T
        result = comp.evaluate_on_grid(R, T)
        assert np.max(np.abs(result - expected)) < 1e-12

    def test_cubic_recovers_smooth_function(self):
        """Cubic interpolation recovers smooth f to high precision on a
        dense table."""
        r_t = np.linspace(0.0, 1.0, 60)
        t_t = np.linspace(0.0, np.pi, 80)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        f_t = np.sin(np.pi * R_t) * np.cos(T_t)
        comp = meridional_interpolator(r_t, t_t, f_t, kind="cubic")
        R = np.linspace(0.1, 0.9, 8).reshape(-1, 1) * np.ones((1, 6))
        T = np.ones((8, 1)) * np.linspace(0.3, 2.8, 6)
        expected = np.sin(np.pi * R) * np.cos(T)
        result = comp.evaluate_on_grid(R, T)
        # Cubic on a smooth function with N=60×80 should be much better
        # than linear; expect < 1e-5.
        assert np.max(np.abs(result - expected)) < 1e-5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            # f_table shape (M, N) vs r_table size N_r, t_table size N_t
            meridional_interpolator(
                np.linspace(0, 1, 5),
                np.linspace(0, np.pi, 7),
                np.zeros((6, 7)),  # wrong size
            )


class TestCompositeComponent:
    """CompositeComponent multiplies two BackgroundComponents pointwise."""

    def test_evaluates_as_product(self):
        a = SymPyComponent(sp.symbols("r", positive=True))
        b = SymPyComponent(sp.cos(sp.symbols("theta", positive=True)))
        comp = CompositeComponent(left=a, right=b)
        R, T = np.meshgrid(np.linspace(0.1, 1, 5),
                            np.linspace(0.3, 2.8, 4), indexing="ij")
        result = comp.evaluate_on_grid(R, T)
        expected = R * np.cos(T)
        assert np.max(np.abs(result - expected)) < 1e-14


class TestMultiplyComponents:
    """multiply_components keeps SymPy×SymPy on the fast path."""

    def test_sympy_times_sympy_collapses(self):
        r_sym = sp.symbols("r", positive=True)
        a = SymPyComponent(r_sym)
        b = SymPyComponent(1 + r_sym)
        out = multiply_components(a, b)
        assert isinstance(out, SymPyComponent)
        assert sp.simplify(out.expr - r_sym * (1 + r_sym)) == 0

    def test_identity_one_collapses(self):
        r_sym = sp.symbols("r", positive=True)
        one = SymPyComponent(sp.S.One)
        b = SymPyComponent(r_sym ** 2)
        assert multiply_components(one, b) is b
        assert multiply_components(b, one) is b

    def test_sympy_times_numeric_yields_composite(self):
        a = SymPyComponent(sp.cos(sp.symbols("theta", positive=True)))
        b = NumericComponent(f=lambda R, T: R * 0 + 1.5)
        out = multiply_components(a, b)
        assert isinstance(out, CompositeComponent)
        # Evaluate and check
        R = np.array([[0.5, 0.7]])
        T = np.array([[0.3, 0.8]])
        result = out.evaluate_on_grid(R, T)
        expected = np.cos(T) * 1.5
        assert np.max(np.abs(result - expected)) < 1e-14


class TestTabularDtCoefficientEndToEnd:
    """rho0(r) from a table, used in `rho0 * Dt(v) = Lap(v)`, gives
    eigenvalues consistent with the SymPy version of the same function."""

    def _eigenvalues(self, rho0_kind):
        r_sym = sp.symbols("r", positive=True)
        if rho0_kind == "sympy":
            rho0 = Background.Scalar("rho0", expr=1 + r_sym ** 2)
        elif rho0_kind == "tabular_dense":
            r_t = np.linspace(0.0, 1.5, 200)
            f_t = 1 + r_t ** 2
            rho0 = Background.Scalar(
                "rho0", expr=radial_interpolator(r_t, f_t, kind="cubic"),
            )
        else:
            raise ValueError(rho0_kind)

        v = Unknown.Vector("v")
        eq = Equation(rho0 * Dt(v) - Lap(v), name="strat")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="r_in"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="r_out"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(N_r=14, N_theta=20, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)
        res = cp.solve_evp(k=15, sigma=+5.0)
        lam = np.array(res.eigenvalues)
        pos = lam[np.isfinite(lam) & (lam.real > 0)]
        return sorted(pos, key=lambda y: y.real)[:5]

    def test_tabular_matches_sympy(self):
        l_sym = self._eigenvalues("sympy")
        l_tab = self._eigenvalues("tabular_dense")
        # Cubic spline of 1+r² on 200 points should match SymPy to ~1e-6
        # at this resolution.
        max_diff = max(abs(a - b) for a, b in zip(l_sym, l_tab))
        assert max_diff < 1e-6, (
            f"Tabular vs sympy mismatch: max diff {max_diff:.2e}\n"
            f"SymPy: {l_sym}\nTabular: {l_tab}"
        )


class TestMeridionalCoefficientEndToEnd:
    """Tabular Ω(r, θ) can be used in coefficient position (e.g. for
    differential-rotation profiles)."""

    def test_meridional_rho0_compiles_and_runs(self):
        # A spatially-varying rho0(r, θ) = (1 + r²)·(1 + 0.3 cos²θ)
        r_t = np.linspace(0.0, 1.5, 60)
        t_t = np.linspace(0.0, np.pi, 70)
        R_t, T_t = np.meshgrid(r_t, t_t, indexing="ij")
        rho_t = (1 + R_t ** 2) * (1 + 0.3 * np.cos(T_t) ** 2)
        rho0 = Background.Scalar(
            "rho0",
            expr=meridional_interpolator(r_t, t_t, rho_t, kind="cubic"),
        )

        v = Unknown.Vector("v")
        eq = Equation(rho0 * Dt(v) - Lap(v), name="strat")
        bcs = [
            BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="r_in"),
            BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="r_out"),
        ]
        sys_ = EquationSystem([eq], bcs)
        grid = make_shell_grid(N_r=14, N_theta=20, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4)

        # B_total should reflect the meridional structure.
        B = cp.matrices.B_total
        eq_mass = cp.matrices.equation_mass_matrices["strat"]
        N = grid.N_r * grid.N_theta
        ratio_sigma0 = B.diagonal()[N:2*N] / eq_mass.diagonal()[N:2*N]
        ratio_2d = ratio_sigma0.reshape((grid.N_theta, grid.N_r)).T

        R, T = grid.mesh()
        expected = (1 + R ** 2) * (1 + 0.3 * np.cos(T) ** 2)
        err = np.max(np.abs(ratio_2d - expected.astype(complex)))
        assert err < 1e-4, (
            f"Tabular rho0(r,θ) interpolation error too large: {err:.2e}"
        )

        # Just verify the EVP solves without error.
        res = cp.solve_evp(k=10, sigma=+5.0)
        lam = np.array(res.eigenvalues)
        assert len(lam[np.isfinite(lam)]) > 0
