"""spdee.backgrounds.polytrope — analytic adiabatic-polytrope background.

An analytic alternative to `SolarBackgrounds`: the adiabatically stratified
polytrope of Bekki & Yokoyama (2017), following Rempel (2005), parameterized
by base-of-convection-zone (bc) reference values and a constant
superadiabaticity δ that perturbs the adiabatic stratification just enough
to permit convective instability.

Background state (Bekki & Yokoyama 2017, eqs. 24-28)
----------------------------------------------------
With Φ(r) = 1 + ((γ-1)/γ) (r_bc/H_bc) (r_bc/r - 1):

    ρ₀(r) = ρ_bc · Φ(r)^{1/(γ-1)}
    p₀(r) = p_bc · Φ(r)^{γ/(γ-1)}
    T₀(r) = T_bc · Φ(r)
    g(r)  = g_bc · (r/r_bc)^{-2}

The pressure scale height H_p = -p₀/(dp₀/dr) has the closed form

    H_p(r) = Φ(r) · r² · H_bc / r_bc².

The superadiabaticity δ = ∇ - ∇_ad = -(H_p/c_p) ds₀/dr sets the entropy
gradient. Entropy is measured in units of c_p (S = s₁/c_p, the spelling of
the certified string system `spdee.applications.solar_strings`), so c_p ≡ 1
and

    ds₀/dr = -δ / H_p(r),      N² = (g/c_p) ds₀/dr = -|g| δ / H_p(r).

For δ > 0 the entropy decreases outward, which is convectively unstable and
produces the growing columnar convective modes of Bekki Fig. 2c. δ is a
small constant (default 1e-6); the radial structure of ds₀/dr comes from
H_p(r). The system builders use the same pair, so the buoyancy × source
loop of `make_polytrope_system`, `N2` here and the string system agree.

Reference values (Bekki & Yokoyama 2017, §3.2)
----------------------------------------------
  r_bc  = 0.71 R_sun         (base of convection zone)
  R_sun = 7e10 cm
  ρ_bc  = 0.2 g cm^-3
  p_bc  = 6e13 dyn cm^-2
  g_bc  = 5.2e4 cm s^-2
  γ     = 5/3
  ⇒ H_bc = p_bc/(ρ_bc g_bc) ≈ 5.77e9 cm ≈ 0.0824 R_sun
  ⇒ (γ-1)/γ · r_bc/H_bc ≈ 3.446   (the dimensionless polytrope coefficient)

Nondimensionalization
---------------------
All returned backgrounds are dimensionless: lengths in R_sun, and scalars
normalized to their base-of-CZ values (ρ → ρ/ρ_bc, etc.), so ρ₀(r_bc) = 1.
The Ω₀-based dynamical scalings of the buoyancy and pressure coefficients
are applied by the system builders (see `make_polytrope_system`).

Public interface
----------------
`PolytropeBackground` mirrors the subset of the `SolarBackgrounds`
attribute interface used by the Bekki system builders:

    .rho, .pressure, .sound_speed, .temperature, .gravity,
    .cp, .g_vec, .grad_rho, .grad_s0, .N2, .Omega, .Omega_vec
"""
from __future__ import annotations

import numpy as np
import sympy as sp

from spdee import Background
from spdee.ir.backgrounds import NumericComponent
from spdee.ir.fields import theta as _theta_sym


# Bekki & Yokoyama (2017) base-of-CZ reference values (cgs).
POLYTROPE_R_SUN = 7.0e10        # cm
POLYTROPE_R_BC = 0.71           # R_sun (base of convection zone)
POLYTROPE_RHO_BC = 0.2          # g cm^-3
POLYTROPE_P_BC = 6.0e13         # dyn cm^-2
POLYTROPE_G_BC = 5.2e4          # cm s^-2
POLYTROPE_GAMMA = 5.0 / 3.0


def _polytrope_coefficient(gamma: float, r_bc: float, H_bc: float) -> float:
    """The dimensionless coefficient C = (γ-1)/γ · r_bc/H_bc."""
    return (gamma - 1.0) / gamma * (r_bc / H_bc)


class PolytropeBackground:
    """Analytic adiabatic-polytrope background of Bekki & Yokoyama (2017).

    Parameters
    ----------
    r_inner, r_outer : float
        Shell bounds in R_sun. r_inner is typically the base of the
        convection zone (0.71); r_outer is near the surface (0.985).
    delta : float, default 1e-6
        Constant superadiabaticity δ = ∇ - ∇_ad. δ > 0 is convectively
        unstable. Sets the background entropy gradient ds₀/dr = -δ / H_p(r)
        (entropy in units of c_p).
    gamma : float, default 5/3
        Adiabatic exponent.
    r_bc, rho_bc, p_bc, g_bc : float
        Base-of-CZ reference values (cgs), defaulting to the Bekki & Yokoyama
        (2017) values. H_bc = p_bc/(ρ_bc g_bc) and the polytrope coefficient
        follow from these.
    R_sun : float
        Solar radius in cm (converts H_bc to R_sun units).
    """

    def __init__(
        self,
        r_inner: float = 0.71,
        r_outer: float = 0.985,
        delta: float = 1e-6,
        gamma: float = POLYTROPE_GAMMA,
        r_bc: float = POLYTROPE_R_BC,
        rho_bc: float = POLYTROPE_RHO_BC,
        p_bc: float = POLYTROPE_P_BC,
        g_bc: float = POLYTROPE_G_BC,
        R_sun: float = POLYTROPE_R_SUN,
    ):
        self.r_inner = float(r_inner)
        self.r_outer = float(r_outer)
        self.delta = float(delta)
        self.gamma = float(gamma)
        self.r_bc = float(r_bc)
        self.rho_bc = float(rho_bc)
        self.p_bc = float(p_bc)
        self.g_bc = float(g_bc)
        self.R_sun = float(R_sun)
        H_bc_cm = self.p_bc / (self.rho_bc * self.g_bc)
        self.H_bc = H_bc_cm / self.R_sun               # R_sun units
        self.C = _polytrope_coefficient(self.gamma, self.r_bc, self.H_bc)
        self.cs2_bc = self.gamma * self.p_bc / self.rho_bc   # cgs, c_s² = γ p/ρ

        # Domain validation. Φ(r) = 1 + C(r_bc/r - 1) decreases with r and, for
        # C > 1, crosses zero at r_zero = r_bc / (1 - 1/C), where the density
        # formally vanishes. Beyond r_zero the fractional powers Φ^{1/(γ-1)}
        # and Φ^{γ/(γ-1)} of a negative base evaluate to NaN in numpy, so the
        # whole domain, plus a margin for finite-difference stencils that probe
        # slightly outside [r_inner, r_outer], must stay inside r_zero.
        if self.C > 1.0:
            self.r_zero = self.r_bc / (1.0 - 1.0 / self.C)
        else:
            self.r_zero = float("inf")
        _fd_margin = 1e-3
        r_probe_max = self.r_outer + _fd_margin
        if r_probe_max >= self.r_zero:
            raise ValueError(
                f"PolytropeBackground: the polytrope density vanishes at "
                f"r_zero = {self.r_zero:.5f} R_sun (where Φ=0), but the "
                f"domain extends to r_outer = {self.r_outer} (+{_fd_margin} "
                f"FD margin = {r_probe_max:.5f}). Beyond r_zero the density "
                f"ρ₀ = ρ_bc·Φ^(1/(γ-1)) is a fractional power of a negative "
                f"number (→ NaN), which would corrupt the spectrum. "
                f"Reduce r_outer below {self.r_zero - _fd_margin:.5f}, or "
                f"adjust the base values (a larger H_bc / smaller "
                f"r_bc/H_bc pushes r_zero outward). Current "
                f"(γ-1)/γ·r_bc/H_bc = {self.C:.4f}."
            )

    # ----- core profile functions (numpy, dimensionless shape) -----

    def _Phi(self, R):
        """The polytrope base factor Φ(r) = 1 + C(r_bc/r - 1)."""
        return 1.0 + self.C * (self.r_bc / R - 1.0)

    def _Phi_pos(self, R):
        """Φ(r), raising ValueError if any sampled value is non-positive.

        Guards the fractional-power profiles (ρ, p, c_s) against a negative
        base, which numpy would silently turn into NaN.
        """
        Phi = self._Phi(R)
        if np.any(np.asarray(Phi) <= 0.0):
            bad = np.asarray(R)[np.asarray(Phi) <= 0.0]
            r_bad = float(np.asarray(bad).reshape(-1)[0]) if bad.size else float("nan")
            raise ValueError(
                f"PolytropeBackground: Φ(r) ≤ 0 at r ≈ {r_bad:.5f} "
                f"(r_zero = {self.r_zero:.5f}). A fractional power of a "
                f"non-positive base would yield NaN. This radius is "
                f"outside the validated domain [{self.r_inner}, "
                f"{self.r_outer}] — check for an oversized finite-"
                f"difference stencil or an evaluation outside the shell."
            )
        return Phi

    def _rho_ratio(self, R):
        """ρ₀(r)/ρ_bc."""
        return self._Phi_pos(R) ** (1.0 / (self.gamma - 1.0))

    def _p_ratio(self, R):
        """p₀(r)/p_bc."""
        return self._Phi_pos(R) ** (self.gamma / (self.gamma - 1.0))

    def _T_ratio(self, R):
        """T₀(r)/T_bc."""
        return self._Phi(R)

    def _g_ratio(self, R):
        """g(r)/g_bc = (r/r_bc)^-2."""
        return (R / self.r_bc) ** (-2.0)

    def _Hp(self, R):
        """Pressure scale height H_p(r) = Φ(r) · r² · H_bc / r_bc², in R_sun."""
        return self._Phi(R) * R**2 * self.H_bc / self.r_bc**2

    def _cs2_ratio(self, R):
        """c_s²(r)/c_s²_bc = Φ, since c_s² = γ p/ρ and p/ρ ∝ Φ."""
        return self._Phi_pos(R)

    # ----- spdee Background accessors -----

    @property
    def rho(self) -> Background:
        """Dimensionless density ρ₀(r)/ρ_bc."""
        return Background.Scalar(
            "poly_rho",
            NumericComponent(f=lambda R, T: self._rho_ratio(R),
                             description="ρ₀/ρ_bc"),
        )

    @property
    def pressure(self) -> Background:
        """Dimensionless pressure p₀(r)/p_bc."""
        return Background.Scalar(
            "poly_pressure",
            NumericComponent(f=lambda R, T: self._p_ratio(R),
                             description="p₀/p_bc"),
        )

    @property
    def temperature(self) -> Background:
        """Dimensionless temperature T₀(r)/T_bc."""
        return Background.Scalar(
            "poly_temperature",
            NumericComponent(f=lambda R, T: self._T_ratio(R),
                             description="T₀/T_bc"),
        )

    @property
    def sound_speed(self) -> Background:
        """Dimensionless sound speed c_s(r)/c_s_bc = sqrt(Φ).

        Returns c_s, not c_s², matching the SolarBackgrounds convention;
        callers square it.
        """
        return Background.Scalar(
            "poly_sound_speed",
            NumericComponent(f=lambda R, T: np.sqrt(self._cs2_ratio(R)),
                             description="c_s/c_s_bc"),
        )

    @property
    def gravity(self) -> Background:
        """Dimensionless gravity magnitude |g(r)|/g_bc."""
        return Background.Scalar(
            "poly_gravity",
            NumericComponent(f=lambda R, T: self._g_ratio(R),
                             description="|g|/g_bc"),
        )

    @property
    def cp(self) -> Background:
        """Specific heat c_p as the dimensionless constant 1.

        For the ideal-gas polytrope c_p = γ/(γ-1) · k_B/m is constant; its
        dimensional scale is folded into the buoyancy coefficient by the
        system builders, which use this same c_p in both s₁ terms of the
        momentum equation. Returning a Background keeps the interface parallel
        to SolarBackgrounds.
        """
        return Background.Scalar(
            "poly_cp",
            NumericComponent(f=lambda R, T: np.ones_like(R),
                             description="c_p (const)"),
        )

    @property
    def g_vec(self) -> Background:
        """Gravity vector g = -|g(r)|/g_bc r̂ (pointing inward)."""
        return Background.Vector(
            "poly_g_vec",
            v_r=NumericComponent(f=lambda R, T: -self._g_ratio(R),
                                 description="-|g|/g_bc"),
            v_theta=None, v_phi=None,
        )

    @property
    def grad_rho(self) -> Background:
        """Density gradient ∇ρ₀ = (dρ₀/dr) r̂, dimensionless.

        With ρ₀/ρ_bc = Φ^{1/(γ-1)} and Φ' = -C r_bc/r²:
        d(ρ₀/ρ_bc)/dr = [1/(γ-1)] Φ^{(2-γ)/(γ-1)} (-C r_bc/r²).
        """
        n = self.gamma
        C = self.C
        r_bc = self.r_bc

        def fn(R, T):
            Phi = self._Phi(R)
            dPhi = -C * r_bc / R**2
            return (1.0 / (n - 1.0)) * Phi ** ((1.0 / (n - 1.0)) - 1.0) * dPhi

        return Background.Vector(
            "poly_grad_rho",
            v_r=NumericComponent(f=fn, description="d(ρ₀/ρ_bc)/dr"),
            v_theta=None, v_phi=None,
        )

    @property
    def grad_s0(self) -> Background:
        """Background entropy gradient ∇s₀ = (ds₀/dr) r̂ = -δ / H_p(r) r̂ per
        R_sun, entropy in units of c_p. Negative for δ > 0."""
        delta = self.delta

        def fn(R, T):
            return -delta / self._Hp(R)

        return Background.Vector(
            "poly_grad_s0",
            v_r=NumericComponent(f=fn, description="ds₀/dr = -δ/H_p"),
            v_theta=None, v_phi=None,
        )

    @property
    def N2(self) -> Background:
        """Brunt-Väisälä frequency² N² = (g/c_p)(ds₀/dr) = -|g| δ / H_p with
        c_p ≡ 1 and g in units of g_bc, i.e. N² in units of g_bc/R_sun.
        Negative for δ > 0. The dynamical scale g_bc/(R_sun Ω₀²) is applied by
        the system builders."""
        delta = self.delta

        def fn(R, T):
            return -self._g_ratio(R) * delta / self._Hp(R)

        return Background.Scalar(
            "poly_N2",
            NumericComponent(f=fn, description="N² = -|g|δ/H_p"),
        )

    @property
    def Omega(self) -> Background:
        """Uniform rotation Ω(r,θ)/Ω₀ ≡ 1."""
        return Background.Scalar(
            "poly_Omega",
            NumericComponent(f=lambda R, T: np.ones_like(R),
                             description="Ω/Ω₀ = 1 (uniform)"),
        )

    @property
    def Omega_vec(self) -> Background:
        """Ω₀ ẑ = cos θ r̂ - sin θ θ̂ as a Background.Vector (|Ω| = 1)."""
        return Background.Vector(
            "poly_Omega_vec",
            v_r=sp.cos(_theta_sym),
            v_theta=-sp.sin(_theta_sym),
            v_phi=sp.S.Zero,
        )
