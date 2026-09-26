"""Tests for the λ→ω sign convention (the DAMPING axis).

Background
---------
The compiler solves the generalized pencil ``K x = λ M x`` where ``M``
collects ``Dt(...)`` terms and ``K`` collects everything else. The
physical frequency is recovered via ``ω = −iλ`` (see
``EvpResult.eigenvalues_to_omega`` for the full derivation):

    Re(ω) = +Im(λ)     (oscillation frequency)
    Im(ω) = −Re(λ)     (growth rate; Im ω < 0 ⇒ damped)

The other EVP tests check only ``Re(ω)`` (frequency / branch
position). A neutral, sign-symmetric mode (e.g. the inviscid Rossby
mode) appears at both ``λ = ±i|ω|``, so a frequency-only check passes
under EITHER sign of the λ→ω map and is BLIND to a flipped damping
axis. Using ``ω = +iλ`` instead of
``ω = −iλ`` silently inverts growth↔decay — a viscous (dissipative)
operator then looks like it is pumping energy in.

These tests pin the damping sign with genuinely non-neutral modes:

1. Pure viscous decay (no rotation): every physical mode MUST damp,
   Im(ω) < 0.
2. The assembled vector Laplacian is negative-(semi)definite, so the
   pure-decay pencil ``Dt(v) − Ek·Lap(v) = 0`` returns λ with
   Re(λ) > 0, which maps to Im(ω) < 0 — the consistency check between
   the raw eigenvalue sign and the canonical map.
3. A viscously damped Rossby mode is BOTH retrograde (Re ω < 0) AND
   damped (Im ω < 0) under the single convention — the test that the
   frequency-only Rossby test could not provide.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from spdee import (
    Unknown, Dt, Lap, Grad, Div, Cross, Dot,
    Equation, EquationSystem, BoundaryCondition, Boundary,
    Background, r_hat, ImpenetrableStressFree,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem, EvpResult


# Small, fast grid — these tests are about SIGNS, not high accuracy.
_N_R, _N_THETA = 16, 24
_R_IN, _R_OUT = 0.71, 0.985


def _shell():
    return make_shell_grid(
        N_r=_N_R, N_theta=_N_THETA, r_inner=_R_IN, r_outer=_R_OUT,
    )


def test_omega_map_is_minus_i_lambda():
    """The canonical converter is ω = −iλ, not +iλ.

    Guards the single source of truth so a future 'cleanup' can't flip
    it back without this failing.
    """
    lam = np.array([1.0 + 0.0j, 0.0 + 2.0j, -3.0 + 4.0j])
    omega = EvpResult.eigenvalues_to_omega(lam)
    np.testing.assert_allclose(omega, -1j * lam)
    # Re(ω)=Im(λ), Im(ω)=−Re(λ):
    np.testing.assert_allclose(omega.real, lam.imag)
    np.testing.assert_allclose(omega.imag, -lam.real)


def test_pure_decay_is_damped():
    """Dt(v) − Ek·Lap(v) = 0: all physical modes must DAMP (Im ω < 0).

    A flipped damping axis shows up as Im(ω) > 0 ('all modes grow') for
    a purely dissipative operator.
    """
    import scipy.linalg as sla

    Ek = 1e-2
    v = Unknown.Vector("v")
    eq = Equation(Dt(v) - sp.S(Ek) * Lap(v), name="diffusion")
    bcs = (ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
           + ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    sys_ = EquationSystem([eq], list(bcs))
    cp = compile_problem(sys_, _shell(), m=2, order_r=4, order_theta=4)
    K, M = cp.assemble_evp_matrices()

    lam = sla.eig(K.toarray(), M.toarray(), right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam) < 1e6]
    omega = EvpResult.eigenvalues_to_omega(lam)

    # Pure decay is non-oscillatory: the genuine decay modes are
    # essentially purely imaginary ω with Im ω < 0. Restrict to those
    # (small |Re ω|) and assert every one is damped. (Large-|Re ω|
    # entries are BC/edge-row artifacts, not physical decay modes.)
    decay_band = omega[np.abs(omega.real) < 1.0]
    assert len(decay_band) > 0, "no low-frequency decay modes found"
    assert decay_band.imag.max() < 1e-6, (
        f"pure viscous decay produced a GROWING mode: "
        f"max Im(ω) = {decay_band.imag.max():+.3e} (must be < 0). "
        f"This is the flipped-damping-axis bug."
    )


def test_raw_lambda_sign_consistent_with_decay():
    """Cross-check: assembled pencil yields Re(λ) > 0 for pure decay.

    Lap is negative-definite, M = +I (from +Dt(v)), K = −Ek·Lap, so
    λ = −Ek·eig(Lap) > 0. Combined with ω = −iλ this gives Im(ω) < 0.
    If the assembled λ ever came out NEGATIVE here, the convention
    constant in eigenvalues_to_omega would need re-deriving — this test
    documents WHY the map is −i and not +i.
    """
    import scipy.linalg as sla

    Ek = 1e-2
    v = Unknown.Vector("v")
    eq = Equation(Dt(v) - sp.S(Ek) * Lap(v), name="diffusion")
    bcs = (ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
           + ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    sys_ = EquationSystem([eq], list(bcs))
    cp = compile_problem(sys_, _shell(), m=2, order_r=4, order_theta=4)
    K, M = cp.assemble_evp_matrices()

    lam = sla.eig(K.toarray(), M.toarray(), right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam) < 1e6]
    # The dominant bulk of the spectrum is the physical k² ladder with
    # Re(λ) > 0 (anti-damping in λ-space; correct decay in ω-space).
    assert np.median(lam.real) > 0, (
        f"expected median Re(λ) > 0 for the decay pencil; got "
        f"{np.median(lam.real):+.3e}"
    )


def _rossby_system(Ek=None):
    """Solar-eigenproblem retrograde-Rossby momentum + continuity; optionally viscous."""
    t_sym = sp.symbols("theta", positive=True)
    Omega = Background.Vector(
        "Omega", v_r=sp.cos(t_sym), v_theta=-sp.sin(t_sym), v_phi=sp.S.Zero,
    )
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    force = Dt(v) + 2 * Cross(Omega, v) + Grad(p)
    if Ek is not None:
        force = force - sp.S(Ek) * Lap(v)
    eq_mom = Equation(force, name="momentum")
    eq_cont = Equation(Div(v), name="continuity")
    bcs = [
        BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vr_in"),
        BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_out"),
    ]
    return EquationSystem([eq_mom, eq_cont], bcs, gauge_unknowns=["p"])


def _find_retrograde(cp, target, k=40):
    """Return the ω closest to the retrograde Rossby target (Re ω < 0)."""
    # Physical mode at frequency `target` sits at λ = +i·target, so shift
    # near there (small real nudge for shift-invert stability).
    sigma = 1j * target + 0.01
    res = cp.solve_evp(k=k, sigma=sigma)
    omega = EvpResult.eigenvalues_to_omega(np.array(res.eigenvalues))
    omega = omega[np.isfinite(omega)]
    band = omega[np.abs(omega.real - target) < 0.12]
    if len(band) == 0:
        return None
    return band[np.argmin(np.abs(band.real - target))]


@pytest.mark.parametrize("m", [2, 4])
def test_inviscid_rossby_retrograde_and_neutral(m):
    """Inviscid Rossby: Re(ω) = −2/(m+1) and Im(ω) ≈ 0 (neutral).

    Confirms the SAME ω = −iλ map that makes decay damp also places the
    retrograde frequency correctly and leaves it neutral.
    """
    cp = compile_problem(_rossby_system(), _shell(), m=m,
                         order_r=4, order_theta=4)
    target = -2.0 / (m + 1)
    w = _find_retrograde(cp, target)
    assert w is not None, f"m={m}: no retrograde mode near {target:.3f}"
    assert abs(w.real - target) < 0.03 * abs(target), (
        f"m={m}: Re(ω)={w.real:+.4f} vs target {target:+.4f}"
    )
    assert abs(w.imag) < 0.02, (
        f"m={m}: inviscid Rossby should be ~neutral, got Im(ω)={w.imag:+.4f}"
    )


@pytest.mark.parametrize("m", [2, 4])
def test_viscous_rossby_retrograde_and_damped(m):
    """Viscous Rossby is BOTH retrograde (Re ω < 0) AND damped (Im ω < 0).

    This is the decisive, non-neutral case the frequency-only Rossby
    test cannot provide: it fixes the damping sign. Under the wrong
    (ω = +iλ) map the retrograde mode would appear to GROW.
    """
    Ek = 1e-3
    cp = compile_problem(_rossby_system(Ek=Ek), _shell(), m=m,
                         order_r=4, order_theta=4)
    target = -2.0 / (m + 1)
    w = _find_retrograde(cp, target)
    assert w is not None, f"m={m}: no retrograde mode near {target:.3f}"
    assert w.real < 0, f"m={m}: expected retrograde Re(ω)<0, got {w.real:+.4f}"
    assert abs(w.real - target) < 0.05, (
        f"m={m}: Re(ω)={w.real:+.4f} drifted from {target:+.4f}"
    )
    assert w.imag < 1e-6, (
        f"m={m}: viscosity must DAMP the Rossby mode (Im ω < 0); got "
        f"Im(ω)={w.imag:+.4e}. Flipped-damping-axis bug."
    )
