"""Tests for the state-augmentation pass.

Currently exercises one pattern:
    Cross(Curl(unknown_vector), background_vector)
which compiles by introducing an auxiliary unknown
    aux ≡ Curl(unknown_vector)
and rewriting the term as Cross(aux, background_vector).

The test compares the augmented-system action to an independent
computation: spdee's standalone Curl operator applied to B, followed
by an analytic Cross with B0 pointwise. The two should match within
discretization error.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Background, Curl, Cross, Lap, LapH, Div, Grad, Dot, Dt, Equation,
    EquationSystem, r_hat, theta,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import split_vector_state


def _curl_only_block(B0_unused, m_val, grid):
    """Compile a system that's just `Curl(B)` (rebound to a vector
    equation with B as unknown). Used to compute Curl(B) numerically
    for cross-checks. Returns the compiled system and a closure to
    apply Curl to a given B state.
    """
    B = Unknown.Vector("B")
    eq = Equation(Curl(B), name="curl_only")
    sys_ = EquationSystem([eq], [])
    cp = compile_problem(sys_, grid, m=m_val, order_r=6, order_theta=6, prefer_aux_free=False)
    layout = cp.matrices.layout
    # The Curl operator has natural r-power 1; eq mass is diag(r). So
    # A @ state = diag(r) · (Curl B). Divide by r to recover Curl B.

    def apply_curl(B_data_dict):
        state = layout.pack({"B": B_data_dict})
        Av = cp.matrices.A_total @ state
        eq_slice = cp.matrices.equation_row_slices["curl_only"]
        block_curl = split_vector_state(
            Av[eq_slice], grid.N_r, grid.N_theta,
        )
        # Undo the r-mass to get Curl(B) directly.
        R, _ = grid.mesh()
        out = {}
        for sigma in (-1, 0, +1):
            out[sigma] = block_curl[sigma] / R.astype(complex)
        return out

    return apply_curl


def _phys_cross_with_bg(curl_spin_dict, B0_phys, grid):
    """Compute Cross(curl, B0) pointwise in physical components.

    curl_spin_dict : {σ: 2D array}  -- the vector (here, Curl(B))
    B0_phys : (B0_r, B0_theta, B0_phi) symbolic
    grid : Grid

    Returns: dict {σ: 2D array} for Cross(curl, B0) in spin components.

    Process: convert curl from spin to physical, do physical cross
    product, convert back to spin.
    """
    R, T = grid.mesh()
    # spin → physical
    cm = curl_spin_dict[-1]
    c0 = curl_spin_dict[0]
    cp_ = curl_spin_dict[+1]
    sqrt2 = np.sqrt(2.0)
    curl_phys_r = c0
    curl_phys_th = (cm + cp_) / sqrt2
    curl_phys_ph = 1j * (cm - cp_) / sqrt2

    # B0 on grid
    r_sym = sp.symbols("r", positive=True)
    t_sym = sp.symbols("theta", positive=True)
    B0_r_fn = sp.lambdify((r_sym, t_sym), B0_phys[0], modules="numpy")
    B0_th_fn = sp.lambdify((r_sym, t_sym), B0_phys[1], modules="numpy")
    B0_ph_fn = sp.lambdify((r_sym, t_sym), B0_phys[2], modules="numpy")
    B0_r = np.broadcast_to(np.asarray(B0_r_fn(R, T), dtype=complex), R.shape).copy()
    B0_th = np.broadcast_to(np.asarray(B0_th_fn(R, T), dtype=complex), R.shape).copy()
    B0_ph = np.broadcast_to(np.asarray(B0_ph_fn(R, T), dtype=complex), R.shape).copy()

    # Cross(curl, B0) in physical:
    #   (a × b)_r = a_θ b_φ - a_φ b_θ
    #   (a × b)_θ = a_φ b_r - a_r b_φ
    #   (a × b)_φ = a_r b_θ - a_θ b_r
    cross_r = curl_phys_th * B0_ph - curl_phys_ph * B0_th
    cross_th = curl_phys_ph * B0_r - curl_phys_r * B0_ph
    cross_ph = curl_phys_r * B0_th - curl_phys_th * B0_r

    # Physical → spin
    out = {
        -1: (cross_th - 1j * cross_ph) / sqrt2,
         0: cross_r,
        +1: (cross_th + 1j * cross_ph) / sqrt2,
    }
    return out


class TestAugmentationCompiles:
    """The augmented system compiles for Lorentz-style terms."""

    def test_lorentz_compiles(self):
        B = Unknown.Vector("B")
        B0 = Background.Vector(
            "B0",
            v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
        )
        eq = Equation(Cross(Curl(B), B0), name="lorentz")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(N_r=8, N_theta=12, r_inner=0.5, r_outer=1.0)
        cp = compile_problem(sys_, grid, m=1, order_r=4, order_theta=4, prefer_aux_free=False)
        # Both B and the aux are now in the layout.
        names = list(cp.matrices.layout.all_names)
        assert "B" in names
        assert any(n.startswith("__aux_curl_") for n in names)


class TestLorentzAgainstStandalone:
    """The Lorentz operator Cross(Curl(B), B0) via augmentation matches
    the direct computation (standalone Curl, then pointwise Cross with B0)."""

    def _run(self, B0_phys, B_data_sym, m_val, N_r=20, N_theta=32):
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        B0 = Background.Vector(
            "B0",
            v_r=B0_phys[0], v_theta=B0_phys[1], v_phi=B0_phys[2],
        )
        # Augmented Lorentz system
        B = Unknown.Vector("B")
        eq = Equation(Cross(Curl(B), B0), name="lorentz")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.3, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=6, order_theta=6, prefer_aux_free=False)

        # Build B state in spin
        R, T = grid.mesh()
        B_data = {}
        for sigma in (-1, 0, +1):
            fn = sp.lambdify((r_sym, t_sym), B_data_sym[sigma], modules="numpy")
            B_data[sigma] = np.broadcast_to(
                np.asarray(fn(R, T), dtype=complex), R.shape,
            ).copy()

        # ALSO need to provide aux_state = Curl(B) at each grid point. The
        # operator equation is `aux - Curl(B) = 0`, i.e., the defining-eq
        # row outputs zero when aux is set correctly. The Lorentz row
        # block outputs Cross(aux, B0) — and if aux is correct, that's
        # Cross(Curl(B), B0).
        #
        # We use spdee's own standalone Curl operator to compute the
        # correct aux values, so we're comparing one spdee path
        # (augmented) against another (direct Curl).
        apply_curl = _curl_only_block(B0, m_val, grid)
        aux_correct = apply_curl(B_data)

        # Find the aux name dynamically (depends on the counter).
        aux_name = [
            n for n in cp.matrices.layout.all_names
            if n.startswith("__aux_curl_B_")
        ][0]
        state = cp.matrices.layout.pack({"B": B_data, aux_name: aux_correct})

        # Apply A_total
        Av = cp.matrices.A_total @ state

        # Lorentz row block: should equal Cross(aux_correct, B0)
        lorentz_slice = cp.matrices.equation_row_slices["lorentz"]
        result = split_vector_state(
            Av[lorentz_slice], grid.N_r, grid.N_theta,
        )

        # Independent computation: Cross(aux_correct, B0) pointwise.
        expected = _phys_cross_with_bg(aux_correct, B0_phys, grid)

        return result, expected, grid

    def test_dipole_B0_with_smooth_B(self):
        """B0 = ẑ (rotation-axis-like), B = some smooth field."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        B0_phys = (sp.cos(t_sym), -sp.sin(t_sym), sp.S.Zero)
        B_data_sym = {
            -1: r_sym * sp.sin(t_sym),
             0: r_sym ** 2 * sp.cos(t_sym),
            +1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        }
        result, expected, grid = self._run(B0_phys, B_data_sym, m_val=1)
        for sigma in (-1, 0, +1):
            interior = np.zeros_like(result[sigma], dtype=bool)
            interior[3:-3, 4:-4] = True
            err = float(np.max(np.abs(result[sigma] - expected[sigma])[interior]))
            denom = max(float(np.max(np.abs(expected[sigma])[interior])), 1e-30)
            assert err / denom < 1e-10, (
                f"σ={sigma}: augmented Lorentz != standalone Curl+Cross; "
                f"rel = {err/denom:.2e}"
            )

    def test_zonal_B0(self):
        """B0 = sin(θ) φ̂ (azimuthal flow-like)."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        B0_phys = (sp.S.Zero, sp.S.Zero, sp.sin(t_sym))
        B_data_sym = {
            -1: r_sym ** 2 * sp.sin(t_sym),
             0: r_sym * sp.cos(t_sym),
            +1: r_sym * sp.sin(t_sym),
        }
        result, expected, grid = self._run(B0_phys, B_data_sym, m_val=2)
        for sigma in (-1, 0, +1):
            interior = np.zeros_like(result[sigma], dtype=bool)
            interior[3:-3, 4:-4] = True
            err = float(np.max(np.abs(result[sigma] - expected[sigma])[interior]))
            denom = max(float(np.max(np.abs(expected[sigma])[interior])), 1e-30)
            assert err / denom < 1e-10, (
                f"σ={sigma}: zonal-B0 Lorentz rel err {err/denom:.2e}"
            )


class TestInductionTermAgainstStandalone:
    """The induction operator Curl(Cross(v, B0)) via augmentation matches
    the direct computation: pointwise Cross(v, B0), then standalone Curl."""

    def _run(self, B0_phys, v_data_sym, m_val, N_r=20, N_theta=32):
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        B0 = Background.Vector(
            "B0",
            v_r=B0_phys[0], v_theta=B0_phys[1], v_phi=B0_phys[2],
        )
        v = Unknown.Vector("v")
        eq = Equation(Curl(Cross(v, B0)), name="ind")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.3, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=6, order_theta=6, prefer_aux_free=False)

        R, T = grid.mesh()
        v_data = {}
        for sigma in (-1, 0, +1):
            fn = sp.lambdify((r_sym, t_sym), v_data_sym[sigma], modules="numpy")
            v_data[sigma] = np.broadcast_to(
                np.asarray(fn(R, T), dtype=complex), R.shape,
            ).copy()

        # Compute the correct aux = Cross(v, B0) pointwise (independent
        # numerical computation).
        aux_correct = _phys_cross_with_bg(v_data, B0_phys, grid)
        # The aux name encodes both v and B0.
        aux_name = [
            n for n in cp.matrices.layout.all_names
            if n.startswith("__aux_cross_v_x_B0_")
        ][0]
        state = cp.matrices.layout.pack({"v": v_data, aux_name: aux_correct})
        Av = cp.matrices.A_total @ state

        # Induction row block = Curl(aux). Use standalone Curl to compute
        # the correct result.
        ind_slice = cp.matrices.equation_row_slices["ind"]
        result = split_vector_state(
            Av[ind_slice], grid.N_r, grid.N_theta,
        )

        # The Curl-eq mass is diag(r); A @ state on this row block gives
        # r · Curl(aux). To compare against a direct Curl, also build the
        # standalone Curl block and feed it aux_correct.
        # Use a separate Curl-only compile so we don't confuse layouts.
        apply_curl = _curl_only_block(B0, m_val, grid)
        curl_of_aux = apply_curl(aux_correct)
        # apply_curl already undoes the r-mass; we get bare Curl(aux).
        # Our induction-row output is r·Curl(aux). So expected is
        # r · curl_of_aux.
        expected = {sigma: R.astype(complex) * curl_of_aux[sigma] for sigma in (-1, 0, +1)}
        return result, expected, grid

    def test_zonal_v_simple_B0(self):
        """B0 = ẑ; v = some smooth field."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        B0_phys = (sp.cos(t_sym), -sp.sin(t_sym), sp.S.Zero)
        v_data_sym = {
            -1: r_sym * sp.sin(t_sym),
             0: r_sym ** 2 * sp.cos(t_sym),
            +1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        }
        result, expected, grid = self._run(B0_phys, v_data_sym, m_val=1)
        for sigma in (-1, 0, +1):
            interior = np.zeros_like(result[sigma], dtype=bool)
            interior[3:-3, 4:-4] = True
            err = float(np.max(np.abs(result[sigma] - expected[sigma])[interior]))
            denom = max(float(np.max(np.abs(expected[sigma])[interior])), 1e-30)
            assert err / denom < 1e-10, (
                f"σ={sigma}: induction-aux Curl rel err {err/denom:.2e}"
            )


class TestInductionFlippedOperandOrder:
    """The operand-flipped induction term Curl(Cross(bg, unknown))
    also gets augmented correctly (separately from Curl(Cross(unknown,
    bg))). Cross is anti-symmetric, so the result should equal
    -1 × the same-direction non-flipped computation.
    """

    def _run(self, bg_phys, unk_data_sym, m_val, N_r=20, N_theta=32):
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        bg = Background.Vector(
            "bg_field",
            v_r=bg_phys[0], v_theta=bg_phys[1], v_phi=bg_phys[2],
        )
        # Flipped form: Curl(Cross(bg, unknown))
        unk = Unknown.Vector("u")
        eq = Equation(Curl(Cross(bg, unk)), name="ind_flipped")
        sys_ = EquationSystem([eq], [])
        grid = make_shell_grid(
            N_r=N_r, N_theta=N_theta, r_inner=0.3, r_outer=1.0,
        )
        cp = compile_problem(sys_, grid, m=m_val, order_r=6, order_theta=6, prefer_aux_free=False)

        R, T = grid.mesh()
        unk_data = {}
        for sigma in (-1, 0, +1):
            fn = sp.lambdify((r_sym, t_sym), unk_data_sym[sigma], modules="numpy")
            unk_data[sigma] = np.broadcast_to(
                np.asarray(fn(R, T), dtype=complex), R.shape,
            ).copy()

        # The correct aux value is Cross(bg, unk) — pointwise.
        # _phys_cross_with_bg computes Cross(arg1, bg_phys) — i.e., it
        # treats bg as the SECOND argument. To get Cross(bg, unk),
        # we negate: Cross(bg, unk) = -Cross(unk, bg).
        aux_correct_neg = _phys_cross_with_bg(unk_data, bg_phys, grid)
        aux_correct = {sigma: -aux_correct_neg[sigma] for sigma in (-1, 0, +1)}

        aux_name = [
            n for n in cp.matrices.layout.all_names
            if n.startswith("__aux_cross_bg_field_x_u_")
        ][0]
        state = cp.matrices.layout.pack({"u": unk_data, aux_name: aux_correct})
        Av = cp.matrices.A_total @ state

        ind_slice = cp.matrices.equation_row_slices["ind_flipped"]
        result = split_vector_state(
            Av[ind_slice], grid.N_r, grid.N_theta,
        )

        # Independent: Curl(aux_correct) — same calculation as the
        # non-flipped induction test (just with negated aux).
        apply_curl = _curl_only_block(bg, m_val, grid)
        curl_of_aux = apply_curl(aux_correct)
        expected = {sigma: R.astype(complex) * curl_of_aux[sigma] for sigma in (-1, 0, +1)}
        return result, expected, grid

    def test_axial_bg_with_smooth_unknown(self):
        """bg = ẑ, unknown smooth field; flipped form Curl(Cross(bg, u))."""
        r_sym = sp.symbols("r", positive=True)
        t_sym = sp.symbols("theta", positive=True)
        bg_phys = (sp.cos(t_sym), -sp.sin(t_sym), sp.S.Zero)
        unk_data_sym = {
            -1: r_sym * sp.cos(t_sym),
             0: r_sym ** 2 * sp.sin(t_sym),
            +1: r_sym * sp.sin(t_sym) * sp.cos(t_sym),
        }
        result, expected, grid = self._run(bg_phys, unk_data_sym, m_val=1)
        for sigma in (-1, 0, +1):
            interior = np.zeros_like(result[sigma], dtype=bool)
            interior[3:-3, 4:-4] = True
            err = float(np.max(np.abs(result[sigma] - expected[sigma])[interior]))
            denom = max(float(np.max(np.abs(expected[sigma])[interior])), 1e-30)
            assert err / denom < 1e-10, (
                f"σ={sigma}: flipped-form Curl(Cross(bg,u)) rel err "
                f"{err/denom:.2e}"
            )


# ---------------------------------------------------------------------------
# Zero-background drop optimization
# ---------------------------------------------------------------------------

class TestZeroBackgroundDrop:
    """When a background vector is identically zero, the augmentation
    pass drops the entire composed-unknown term rather than creating
    spurious auxiliary unknowns. This keeps the eigenvalue spectrum
    clean (no spurious complex eigenvalues from aux constraint rows).
    """

    def test_cross_curl_with_zero_bg_drops_term(self):
        """Cross(Curl(B), B0=0) drops without creating aux unknowns."""
        from spdee.ir.augment import augment_equation_system
        B = Unknown.Vector("B")
        B0 = Background.Vector(
            "B0",
            v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        eq = Equation(Lap(B) - Cross(Curl(B), B0), name="ind")
        sys_ = EquationSystem([eq], [])
        sys_aug = augment_equation_system(sys_, prefer_aux_free=False)
        # No aux unknowns should be present.
        unknowns = list(sys_aug._unknowns.keys())
        assert not any(n.startswith("__aux_") for n in unknowns)
        # The augmented LHS should NOT contain the dropped Cross term.
        # Easiest check: only one equation (no defining equation added).
        assert len(sys_aug.equations) == 1

    def test_curl_cross_with_zero_bg_drops_term(self):
        """Curl(Cross(v, B0=0)) drops without creating aux unknowns."""
        from spdee.ir.augment import augment_equation_system
        v = Unknown.Vector("v")
        B0 = Background.Vector(
            "B0",
            v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero,
        )
        eq = Equation(Lap(v) - Curl(Cross(v, B0)), name="ind")
        sys_ = EquationSystem([eq], [])
        sys_aug = augment_equation_system(sys_, prefer_aux_free=False)
        unknowns = list(sys_aug._unknowns.keys())
        assert not any(n.startswith("__aux_") for n in unknowns)
        assert len(sys_aug.equations) == 1

    def test_nonzero_bg_still_augments(self):
        """Nonzero backgrounds produce auxiliary unknowns and defining
        equations.
        """
        from spdee.ir.augment import augment_equation_system
        B = Unknown.Vector("B")
        B0 = Background.Vector(
            "B0",
            v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
        )
        eq = Equation(Cross(Curl(B), B0), name="lorentz")
        sys_ = EquationSystem([eq], [])
        sys_aug = augment_equation_system(sys_, prefer_aux_free=False)
        unknowns = list(sys_aug._unknowns.keys())
        assert any(n.startswith("__aux_curl_B") for n in unknowns)
        assert len(sys_aug.equations) == 2  # original + defining

    @pytest.mark.parametrize("wrap", [
        lambda x: Div(Grad(x)), lambda x: Lap(x), lambda x: Dt(x), lambda x: Curl(x),
        lambda x: Dot(r_hat, x), lambda x: Dot(x, r_hat), lambda x: Grad(Dot(r_hat, x)),
        lambda x: Div(x) * r_hat, lambda x: 2 * x, lambda x: -x, lambda x: LapH(Dot(r_hat, x)),
    ])
    def test_drop_propagates_through_operators(self, wrap):
        """A dropped composed term stays dropped under Div, Grad, Lap, LapH,
        Dt, Curl, Dot and products, leaving the rest of the sum."""
        from spdee.ir.augment import augment_equation_system
        B = Unknown.Vector("B")
        B0 = Background.Vector("B0", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero)
        term = wrap(Cross(Curl(B), B0))
        rest = Lap(B) if term.is_vector else Lap(Unknown.Scalar("q"))
        sys_ = EquationSystem([Equation(rest + term, name="eq")], [])
        sys_aug = augment_equation_system(sys_, prefer_aux_free=False)
        assert len(sys_aug.equations) == 1
        assert sys_aug.equations[0].lhs is rest
        assert not any(n.startswith("__aux_") for n in sys_aug.unknown_names)

    def test_equation_that_vanishes_identically_is_an_error(self):
        """An equation whose every term drops is a user error, not an
        empty row."""
        from spdee.ir.augment import augment_equation_system
        B = Unknown.Vector("B")
        B0 = Background.Vector("B0", v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=sp.S.Zero)
        sys_ = EquationSystem([Equation(Cross(Curl(B), B0) - Curl(Cross(B, B0)), name="zero"),
                               Equation(Lap(B), name="ok")], [])
        with pytest.raises(ValueError, match="'zero' vanishes identically"):
            augment_equation_system(sys_, prefer_aux_free=False)
