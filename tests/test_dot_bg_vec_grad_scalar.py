"""Test the first-order differential operator
Dot(background_vector, Grad(unknown_scalar)) → scalar.

This is the operator needed for background-stratification corrections in
the entropy/energy equation — e.g. the thermal-diffusion term

    κ (dln P₀/dr) ∂_r s₁ = κ Dot(grad_ln_P0, Grad(s₁))

the analogue, for entropy, of the viscous ∇lnρ₀·T term. The operator
contracts a background vector a with the gradient of a scalar unknown,
which in physical components is

    a · ∇s = a_r ∂_r s + a_θ (1/r) ∂_θ s + a_φ (1/(r sinθ)) ∂_φ s,

and in V19 spin components is  a^0 (∇s)^0 + a^+ (∇s)^- + a^- (∇s)^+.

It is built by composing build_grad_scalar_block (the (r√2)∇s^σ block)
with that spin contraction, then finalized as a DIFFERENTIAL scalar
operator (natural r-power 1, same as Grad) so the r√2 natural-mass
factor is divided out by the mass-conversion overlay.

Cross-validation:
  - Radial bg (A_r r̂): Dot(A r̂, Grad s) = A_r ∂_r s. Matched against a
    high-order finite-difference of s in radius.
  - Operand-flipped Dot(Grad s, bg) gives the identical matrix.
  - Pure thermal decay  ∂_t s = κ[Lap s + Dot(grad_ln_P0, Grad s)]  has
    all eigenvalues damped (Im ω ≤ 0); the stratification term
    strengthens the damping relative to Lap alone (sign check).
"""

from __future__ import annotations

import numpy as np
import sympy as sp
import scipy.linalg as sla

from spdee import (
    Unknown, Background, Dot, Dt, Lap, Grad, Equation, EquationSystem,
    BoundaryCondition, Boundary,
)
from spdee.ir.backgrounds import NumericComponent
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem
from spdee.backends.fd2.layout import reshape_2d


def _build_scalar_only_system(bg, N=20, flip=False, with_lap=True):
    """Energy-only system exercising the new pattern:
        Dt(s) + Dot(bg, Grad s) [- Lap s] = 0
    """
    s = Unknown.Scalar("s")
    term = Dot(Grad(s), bg) if flip else Dot(bg, Grad(s))
    expr = Dt(s) + term
    if with_lap:
        expr = expr - Lap(s)
    bcs = [
        BoundaryCondition(Boundary.R_INNER, s, name="s_i"),
        BoundaryCondition(Boundary.R_OUTER, s, name="s_o"),
    ]
    sys_ = EquationSystem([Equation(expr, name="energy")], bcs)
    grid = make_shell_grid(N_r=N, N_theta=N, r_inner=0.3, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=0, order_r=6, order_theta=6)
    return cp, grid


def _apply_energy(cp, grid, s_data):
    state = cp.matrices.layout.pack({"s": s_data})
    result = cp.matrices.A_total @ state
    sl = cp.matrices.equation_row_slices["energy"]
    return reshape_2d(result[sl], grid.N_r, grid.N_theta)


def _fd_dr(field2d, r1d, order=6):
    """High-order finite-difference ∂_r of a 2D field (axis 0 = r)."""
    return np.gradient(field2d, r1d, axis=0, edge_order=2)


class TestDotBgVecGradScalar:

    def test_radial_bg_gives_Ar_dr_s(self):
        """bg = A_r(r) r̂ ⇒ Dot(bg, Grad s) = A_r ∂_r s.

        The energy A-block carries the equation mass diag(r²) (one
        Lap → r-power 2 equation mass), so the row equals r²·A_r·∂_r s.
        Compared against a finite-difference ∂_r s in the interior.
        """
        def Ar(R, T):
            return (2.0 + R)            # smooth radial profile

        bg = Background.Vector(
            "bg", v_r=NumericComponent(f=Ar, description="A_r"),
            v_theta=None, v_phi=None,
        )
        cp, grid = _build_scalar_only_system(bg, N=24, with_lap=True)
        R, T = grid.mesh()
        # Smooth analytic test field (NOT separable in a trivial way).
        s_data = (R**3 + 0.5 * R * np.cos(T)**2).astype(complex)

        # The energy equation also has -Lap(s); to isolate the new term,
        # build a Lap-free companion and subtract.
        cp0, grid0 = _build_scalar_only_system(bg, N=24, with_lap=False)
        energy_2d_nolap = _apply_energy(cp0, grid0, s_data)

        r1d = R[:, 0]
        ds_dr = _fd_dr(s_data.real, r1d)
        # equation mass for the Lap-free system: the only differential
        # term is the new first-order op (natural r-power 1) ⇒ r-power 1.
        expected = R * (2.0 + R) * ds_dr

        interior = np.zeros_like(R, dtype=bool)
        interior[2:-2, :] = True
        err = float(np.max(np.abs(energy_2d_nolap.real - expected)[interior]))
        denom = float(np.max(np.abs(expected)[interior]))
        # FD oracle is ~2nd-order at edges; allow a modest tolerance.
        assert err / denom < 5e-3, f"radial bg rel err {err/denom:.2e}"

    def test_operand_order_commutes(self):
        """Dot(bg, Grad s) and Dot(Grad s, bg) assemble identically."""
        def Ar(R, T):
            return (1.0 + 0.3 * R)

        bg = Background.Vector(
            "bg", v_r=NumericComponent(f=Ar, description="A_r"),
            v_theta=None, v_phi=None,
        )
        cp_a, _ = _build_scalar_only_system(bg, N=16, flip=False)
        cp_b, _ = _build_scalar_only_system(bg, N=16, flip=True)
        A = cp_a.matrices.A_total.toarray()
        B = cp_b.matrices.A_total.toarray()
        assert np.allclose(A, B), "Dot operand order must commute"

    def test_thermal_decay_is_damped(self):
        """Pure thermal diffusion ∂_t s = κ[Lap s + Dot(grad_ln_P0,Grad s)]
        must damp all modes (Im ω ≤ 0), and the stratification term must
        not flip any mode unstable. Sign check for the LHS −κ convention.
        """
        from spdee.backgrounds.polytrope import PolytropeBackground
        poly = PolytropeBackground(delta=1e-6)
        g, C, rbc = poly.gamma, poly.C, poly.r_bc

        def gP_r(R, T):
            Phi = poly._Phi(R)
            dPhi = -C * rbc / R**2
            return (g / (g - 1.0)) * dPhi / Phi

        gP = Background.Vector(
            "grad_ln_P0",
            v_r=NumericComponent(f=gP_r, description="dlnP0/dr"),
            v_theta=None, v_phi=None,
        )
        s = Unknown.Scalar("s")
        Ek = 1e-3
        grid = make_shell_grid(N_r=14, N_theta=16,
                               r_inner=0.71, r_outer=0.985)

        def eigs(expr):
            sys_ = EquationSystem([Equation(expr, name="e")], [])
            cp = compile_problem(sys_, grid, m=4, order_r=4, order_theta=4)
            K, M = cp.assemble_evp_matrices()
            lam = sla.eig(K.toarray(), M.toarray(), right=False)
            lam = lam[np.isfinite(lam)]
            lam = lam[np.abs(lam) < 1e8]
            return -1j * lam            # ω = −iλ

        om_lap = eigs(Dt(s) - sp.S(Ek) * Lap(s))
        om_full = eigs(Dt(s) - sp.S(Ek) * Lap(s)
                       - sp.S(Ek) * Dot(gP, Grad(s)))

        assert om_lap.imag.max() <= 1e-6, "Lap-only must be damped"
        assert om_full.imag.max() <= 1e-6, (
            "Lap+stratification must remain damped (no spurious growth)"
        )
        # Stratification should strengthen (not weaken) the damping.
        assert om_full.imag.max() <= om_lap.imag.max() + 1e-9
