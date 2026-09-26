"""Linear eigensystems for solar inertial-mode studies, with and without a
background magnetic field.

Builders
--------
make_dipolar_B0, make_dipolar_B0_curl
    Divergence-free dipolar background field B₀(r, θ) and its curl J₀ = ∇×B₀.
make_solar_shell_system
    Model S shell with helioseismic Ω(r, θ) and viscosity (the Step 4 system of
    tests/test_solar_shell.py) plus the induction equation and Lorentz force.
make_polytrope_system
    Bekki & Yokoyama (2017) Fig. 2 setup on the adiabatic polytrope: uniform
    rotation, constant superadiabaticity δ, ν = κ = Ek; optional MHD layer.
make_solar_ball_system
    The same equations on a Model S ball with uniform rotation and the real
    solar N²(r).
add_magnetic_field
    Generic layer that adds an induction equation and the Lorentz force to any
    hydrodynamic EquationSystem (b or vector-potential formulation).

Magnetic physics
----------------
With a magnetic perturbation b and a static background B₀, the momentum
equation gains the Lorentz force

    (1/(μ₀ ρ₀)) [ (∇×b)×B₀ + (∇×B₀)×b ].

Nondimensionalizing with the Lehnert number Le = v_A / (Ω₀ R), where
v_A = B_ref / √(μ₀ ρ_ref), this becomes

    Le² (ρ_ref/ρ₀) [ (∇×b)×B̂₀ + (∇×B̂₀)×b ]

with B̂₀ the dimensionless field profile (|B̂₀| ~ 1 at the reference radius).
The (ρ_ref/ρ₀) prefactor carries the density stratification: the Lorentz
acceleration is larger where ρ₀ is smaller. Every builder writes this term on
the left-hand side as ``- Le² inv_rho_ratio [Cross(Curl(b), B0) + Cross(J0, b)]``
so that ∂_t v = +Le² (ρ_ref/ρ₀) (∇×b)×B̂₀.

The induction equation with magnetic diffusion η = Ek/Pm (Pm = ν/η) is

    ∂_t b - ∇×(v × B₀) - η ∇²b = 0.

Perfect-conductor walls pin all three components of b (make_solar_shell_system,
make_polytrope_system) or impose b_r = 0 and vanishing tangential electric
field (add_magnetic_field, bc='conducting').

Background field profile
------------------------
B₀ derives from the stream function ψ(r, θ) = f(r) sin²θ:

    B_r = (2/r²) f(r) cos θ,   B_θ = -(1/r) f'(r) sin θ,   B_φ = 0,

a divergence-free poloidal dipole. The radial profile is chosen so that the
equatorial |B_θ| decays exponentially from the inner to the outer boundary,

    B_eq(r) = exp(-α (r - r_inner)/(r_outer - r_inner)),   f'(r)/r = B_eq(r),

with α = ln(amplitude_ratio); the default ratio 30000 maps 30 kG at the
tachocline (r = 0.71) to 1 G at the surface (r = 0.985). On a ball the
profile must start at r_inner = 0, where it is the uniform field ẑ; the
builders raise otherwise, since f(0) ≠ 0 makes B_r ∝ 1/r² at the origin.

Nondimensionalization
---------------------
Lengths in R_sun, time in 1/Ω₀, velocities in Ω₀ R_sun. Le sets the overall
field amplitude; amplitude_ratio sets its radial distribution. For a solar
estimate, B = 30 kG and ρ ~ 200 kg/m³ at the tachocline give v_A ~ 6 m/s
against Ω₀ R ~ 290 m/s, i.e. Le ~ 0.02.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sympy as sp

from spdee import (
    Unknown, Background, Equation, EquationSystem,
    BoundaryCondition, Boundary, Dot, r_hat, theta_hat, phi_hat,
    Lap, Dt, Grad, Div, Curl, Cross, Transpose, ImpenetrableStressFree,
)
from spdee.ir.fields import r as r_sym, theta as t_sym
from spdee.ir.backgrounds import NumericComponent
from spdee.ir.expressions import FieldRef
from spdee.backgrounds.solar import SolarBackgrounds
from spdee.applications.solar_strings import RMIN, RMAX

# Carrington rotation rate in nHz; Ω₀ = 2π·OMEGA0_NHZ·1e-9 rad/s is the time
# unit of every builder (the Ω₀ of `spdee.applications.solar_strings`).
OMEGA0_NHZ = 456.0


def _require_ball_dipole_inner(geometry: str, r_inner: float) -> None:
    """A dipolar B₀ on a ball must start at the origin.

    The stream function's constant of integration is fixed at r_inner, so with
    r_inner > 0 on a ball B_r = 2 f(0)/r² cos θ diverges at the origin.
    """
    if geometry == "ball" and r_inner != 0.0:
        raise ValueError(
            f"a dipolar B0 on a ball needs r_inner = 0 (got {r_inner}); the "
            f"profile is singular at the origin otherwise")


# ---------------------------------------------------------------------------
# Background magnetic field
# ---------------------------------------------------------------------------

def _make_dipolar_B0_profile(
    r_inner: float,
    r_outer: float,
    amplitude_ratio: float,
) -> tuple:
    """Return SymPy expressions f(r), f'(r) for the stream function ψ = f sin²θ.

    With u = (r - r_inner)/(r_outer - r_inner) and α = ln(amplitude_ratio),
    f'(r) = r exp(-α u), so the equatorial |B_θ| = f'/r equals 1 at r_inner and
    1/amplitude_ratio at r_outer. f(r) = ∫_{r_inner}^{r} f'(r') dr' is computed
    in closed form; its constant of integration only shifts B_r.
    """
    if amplitude_ratio <= 1.0:
        raise ValueError(
            f"amplitude_ratio must be > 1 (got {amplitude_ratio}); "
            f"a ratio of 1 means uniform field — use a different profile."
        )
    L = r_outer - r_inner
    alpha = sp.log(sp.S(amplitude_ratio))
    u = (r_sym - r_inner) / L
    g = sp.exp(-alpha * u)
    fprime = r_sym * g

    # f(r) = L ∫_0^u (r_inner + L s) exp(-α s) ds, integrated symbolically.
    s_sym = sp.symbols("s", real=True)
    integrand = (r_inner + L * s_sym) * sp.exp(-alpha * s_sym)
    f_indef = sp.integrate(integrand, s_sym)
    f_at_u = f_indef.subs(s_sym, u)
    f_at_0 = f_indef.subs(s_sym, 0)
    f_expr = L * (f_at_u - f_at_0)
    f_expr = sp.simplify(f_expr)
    fprime_expr = sp.simplify(fprime)
    return f_expr, fprime_expr


# Radius below which a ball dipole takes its origin limit; the closed forms
# are 0/0 at r = 0 exactly, so only the origin node is affected.
_ORIGIN_EPS = 1e-8


def _origin_limit(expr, limit):
    """``expr`` with its removable singularity at r = 0 replaced by ``limit``."""
    return sp.Piecewise((limit, r_sym < _ORIGIN_EPS), (expr, True))


def make_dipolar_B0(
    name: str,
    r_inner: float,
    r_outer: float,
    amplitude: float = 1.0,
    amplitude_ratio: float = 30000.0,
) -> Background:
    """Divergence-free dipolar Background.Vector from ψ = f(r) sin²θ.

        B_r = (2 f / r²) cos θ,   B_θ = -(f' / r) sin θ,   B_φ = 0.

    The equatorial |B_θ| equals `amplitude` at r_inner and
    `amplitude / amplitude_ratio` at r_outer. r_inner and r_outer must match
    the grid used at compile time. With r_inner = 0 (a ball) f = r²/2 + O(r³)
    and the field is the uniform amplitude ẑ at the origin; the closed forms
    are 0/0 there, so the origin node takes that limit.
    """
    f, fprime = _make_dipolar_B0_profile(r_inner, r_outer, amplitude_ratio)
    B_r = sp.S(amplitude) * 2 * f / r_sym**2 * sp.cos(t_sym)
    B_theta = -sp.S(amplitude) * fprime / r_sym * sp.sin(t_sym)
    B_r, B_theta = sp.simplify(B_r), sp.simplify(B_theta)
    if r_inner == 0.0:
        B_r = _origin_limit(B_r, sp.S(amplitude) * sp.cos(t_sym))
        B_theta = _origin_limit(B_theta, -sp.S(amplitude) * sp.sin(t_sym))
    return Background.Vector(name, v_r=B_r, v_theta=B_theta, v_phi=sp.S.Zero)


def make_dipolar_B0_curl(
    name: str,
    r_inner: float,
    r_outer: float,
    amplitude: float = 1.0,
    amplitude_ratio: float = 30000.0,
) -> Background:
    """Curl of the dipolar B₀ of `make_dipolar_B0`, as a Background.Vector.

    The field is poloidal and axisymmetric, so its curl is purely toroidal:

        (∇×B)_φ = (1/r) ∂_r(r B_θ) - (1/r) ∂_θ B_r
                = sin θ [2 f(r)/r³ - f''(r)/r].

    With r_inner = 0 the origin node takes the limit (4α/3L) sin θ,
    L = r_outer, of the 0/0 closed form.
    """
    f, fprime = _make_dipolar_B0_profile(r_inner, r_outer, amplitude_ratio)
    fpp = sp.diff(fprime, r_sym)
    curl_phi = sp.simplify(sp.S(amplitude) * sp.sin(t_sym) * (
        2 * f / r_sym**3 - fpp / r_sym
    ))
    if r_inner == 0.0:
        alpha = sp.log(sp.S(amplitude_ratio))
        curl_phi = _origin_limit(
            curl_phi, sp.S(amplitude) * 4 * alpha / (3 * r_outer) * sp.sin(t_sym))
    return Background.Vector(name, v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=curl_phi)


@dataclass(frozen=True)
class ToroidalField:
    """A toroidal background field for `add_magnetic_field`: the unit-shape
    ``B0`` and its curl ``J0`` (Background vectors), the Lorentz prefactor
    ``inv_rho_ratio`` = rho0_ref / rho0(r), the Lehnert number ``Le`` and the
    reference density ``rho0_ref`` (g/cm^3) at ``r_ref``."""
    B0: Background
    J0: Background
    inv_rho_ratio: Background
    Le: float
    rho0_ref: float
    r_ref: float


def toroidal_eqpeak_field(B0_gauss: float, profile: str = "sqrtrho", a: float = 2.0,
                          shape: str = "eqpeak", r_inner: float = RMIN, r_outer: float = RMAX,
                          name: str = "B0") -> ToroidalField:
    """The solar-eigenproblem cross-validation project's toroidal band on the adiabatic
    polytrope of `spdee.applications.solar_strings`, B0 = B_phi(r, theta)
    phi_hat with unit peak at the base:

        B_phi = f(r) g(theta),
        g = sin^a(theta)                       (shape 'eqpeak', peak 1 at the equator)
          = sin^a(theta) cos(theta) / g_max    (shape 'eqnode', g_max = a^(a/2)/(a+1)^((a+1)/2))
        f = sqrt(rho0(r) / rho0(r_inner))      (profile 'sqrtrho', v_A independent of r)
          = (1 + 2x)(1 - x)^2, x = (r - r_inner)/(r_outer - r_inner)   (profile 'cubic')

    with rho0 = RHO_BC zeta^(1/(GAMMA-1)), zeta = 1 + (GAMMA-1)/GAMMA (R_BC/H_BC)
    (R_BC/r - 1), so that f = zeta^(1/(2(GAMMA-1))) = zeta^(3/4) for 'sqrtrho'
    (zeta(R_BC) = 1 and r_inner = R_BC = 0.71).  The background current is

        J0 = curl(B0):  J0_r = (a+1) f cos(theta) sin^(a-1)(theta) / r  ('eqpeak'),
                        J0_theta = -g (f/r + f'),  J0_phi = 0.

    The amplitude label ``B0_gauss`` is B_phi where f = 1, the base r_inner;
    the density reference sits at the same radius, rho0_ref = rho0(r_inner)
    = RHO_BC = 0.2 g/cm^3, and

        Le = v_A / (RSUN OMEGA_CARR),   v_A = B0_gauss / sqrt(4 pi rho0_ref)  [cgs],

    so the physical field is Le B0 and the Lorentz force per unit mass is
    Le^2 (rho0_ref / rho0) [curl(b) x B0 + J0 x b], ``inv_rho_ratio`` being
    rho0_ref / rho0(r) = zeta^(-1/(GAMMA-1)).
    """
    from spdee.applications.solar_strings import (
        RSUN, OMEGA_CARR, GAMMA, RHO_BC, R_BC, H_BC, polytrope_exprs)
    ex = polytrope_exprs(r_sym)
    zeta_ref = 1 + (GAMMA - 1) / GAMMA * (R_BC / H_BC) * (R_BC / r_inner - 1)
    rho_ref = sp.Float(RHO_BC * zeta_ref ** (1 / (GAMMA - 1)))
    a = sp.nsimplify(a) if float(a).is_integer() else sp.Float(a)
    if profile == "sqrtrho":
        f = sp.sqrt(ex["rho0"] / rho_ref)
    elif profile == "cubic":
        x = (r_sym - r_inner) / (r_outer - r_inner)
        f = (1 + 2 * x) * (1 - x) ** 2
    else:
        raise ValueError("profile must be 'sqrtrho' or 'cubic'")
    if shape == "eqpeak":
        g = sp.sin(t_sym) ** a
    elif shape == "eqnode":
        g_max = a ** (0.5 * a) / (a + 1.0) ** (0.5 * (a + 1.0))
        g = sp.sin(t_sym) ** a * sp.cos(t_sym) / g_max
    else:
        raise ValueError("shape must be 'eqpeak' or 'eqnode'")
    B_phi = f * g
    J_r = sp.diff(sp.sin(t_sym) * B_phi, t_sym) / (r_sym * sp.sin(t_sym))
    J_theta = -sp.diff(r_sym * B_phi, r_sym) / r_sym
    B0 = Background.Vector(name, v_r=sp.S.Zero, v_theta=sp.S.Zero, v_phi=B_phi)
    J0 = Background.Vector(f"J0_{name}" if name != "B0" else "J0", v_r=sp.simplify(J_r),
                           v_theta=sp.simplify(J_theta), v_phi=sp.S.Zero)
    inv_rho = Background.Scalar("inv_rho_ratio", expr=rho_ref / ex["rho0"])
    rho0_ref = float(rho_ref)
    vA = B0_gauss / np.sqrt(4 * np.pi * rho0_ref)
    Le = vA / (RSUN * OMEGA_CARR)
    return ToroidalField(B0=B0, J0=J0, inv_rho_ratio=inv_rho, Le=float(Le),
                         rho0_ref=rho0_ref, r_ref=float(r_inner))


def _make_inv_rho_ratio(solar, rho_ref: float) -> Background:
    """Background scalar ρ_ref / ρ₀(r, θ), the Lorentz-force prefactor."""
    rho_eval = solar.rho.component

    def fn(R, T):
        return float(rho_ref) / rho_eval.evaluate_on_grid(R, T)

    return Background.Scalar(
        "inv_rho_ratio",
        NumericComponent(f=fn, description="ρ_ref / ρ₀"),
    )


# ---------------------------------------------------------------------------
# Model S shell with differential rotation, viscosity and a dipolar B₀
# ---------------------------------------------------------------------------

def make_solar_shell_system(
    solar,
    m_val: int,
    Ek: float = 1e-4,
    Le: float = 0.0,
    Pm: float = 1.0,
    amplitude_ratio: float = 30000.0,
    rho_ref: Optional[float] = None,
    entropy_bc: str = "neumann",
    geometry: str = "shell",
    r_inner: float = 0.71,
    r_outer: float = 0.985,
):
    """Model S shell system with Ω(r, θ), viscosity, and magnetic couplings.

    Equations (unknowns v, b, rho1, s1):
      (1) ∂_t v + advection + pressure - Ek Lap(v) - (Ek/3) Grad(Div(v))
              - Le² (ρ_ref/ρ₀) [Cross(Curl(b), B̂₀) + Cross(Ĵ₀, b)] = 0
      (2) ∂_t b - Curl(Cross(v, B̂₀)) - (Ek/Pm) Lap(b) = 0
      (3) continuity, (4) energy, as in the hydrodynamic system.

    The hydrodynamic part is `TestCompressibleStep4._build` from
    tests/test_solar_shell.py (Model S + HMI Ω(r, θ) + viscosity, with
    with_grad_div=True and with_grad_lnrho_strain=False), imported at call
    time as the module ``test_solar_shell`` with the repository's tests/
    directory put on sys.path. The momentum equation is extended with the
    Lorentz terms and the boundary conditions are kept; perfect-conductor
    walls (b_r = b_θ = b_φ = 0) are added for b.

    With Le = 0 every magnetic coupling carries a zero coefficient, so the
    magnetic sector reduces to pure diffusion and the hydrodynamic spectrum is
    unchanged.

    Parameters
    ----------
    solar : SolarBackgrounds
    m_val : int
        Azimuthal order.
    Ek : float, default 1e-4
        Ekman number ν/(Ω₀ R²); the magnetic diffusivity is η = Ek/Pm.
    Le : float, default 0.0
        Lehnert number (solar estimate ~0.02 at the tachocline).
    Pm : float, default 1.0
        Magnetic Prandtl number ν/η.
    amplitude_ratio : float, default 30000.0
        |B₀| ratio between inner and outer boundary.
    rho_ref : float or None
        Reference density in the (ρ_ref/ρ₀) prefactor; None means ρ₀ at
        r = r_inner on the equator.
    entropy_bc : 'dirichlet' or 'neumann', default 'neumann'
    geometry : 'shell' or 'ball', default 'shell'
        Must match the grid passed at compile time.
    r_inner, r_outer : float
        Domain bounds; must match the grid.

    Returns
    -------
    EquationSystem
    """
    if geometry not in ("shell", "ball"):
        raise ValueError(f"geometry must be 'shell' or 'ball'; got {geometry!r}")
    _require_ball_dipole_inner(geometry, r_inner)

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tests"))
    try:
        from test_solar_shell import TestCompressibleStep4
    finally:
        sys.path.pop(0)
    base_system = TestCompressibleStep4._build(
        solar, m_val=m_val, Ek=Ek,
        with_grad_div=True, with_grad_lnrho_strain=False,
        with_tilt=False, entropy_bc=entropy_bc, geometry=geometry,
    )

    # The base system lists [momentum, continuity, energy]. v is taken from it
    # by name; a fresh Unknown.Vector("v") would be a distinct field object and
    # trip the IR's name-collision check.
    eq_mom_old = base_system.equations[0]
    eq_cont = base_system.equations[1]
    eq_energy = base_system.equations[2]
    v = base_system.unknowns["v"]
    v_ref = FieldRef(v)
    b = Unknown.Vector("b")

    B0 = make_dipolar_B0(
        "B0", r_inner=r_inner, r_outer=r_outer,
        amplitude=1.0, amplitude_ratio=amplitude_ratio,
    )
    J0 = make_dipolar_B0_curl(
        "J0", r_inner=r_inner, r_outer=r_outer,
        amplitude=1.0, amplitude_ratio=amplitude_ratio,
    )
    if rho_ref is None:
        rho_arr = solar.rho.component.evaluate_on_grid(
            np.array([[r_inner]]), np.array([[np.pi / 2]]),
        )
        rho_ref_val = float(np.real(np.asarray(rho_arr).reshape(-1)[0]))
    else:
        rho_ref_val = float(rho_ref)
    inv_rho_ratio = _make_inv_rho_ratio(solar, rho_ref_val)
    Le_sq = sp.S(Le) ** 2

    # Lorentz force on the LHS: ∂_t v = +Le² (ρ_ref/ρ₀) (∇×b)×B₀ + ...
    mom_terms_new = (
        eq_mom_old.lhs
        - Le_sq * inv_rho_ratio * Cross(Curl(b), B0)
        - Le_sq * inv_rho_ratio * Cross(J0, b)
    )
    eq_mom_new = Equation(mom_terms_new, name="momentum")

    eta = sp.S(Ek) / sp.S(Pm)
    ind_terms = Dt(b) - Curl(Cross(v_ref, B0)) - eta * Lap(b)
    eq_ind = Equation(ind_terms, name="induction")

    # Perfect-conductor walls: all three components of b vanish.
    b_bcs = []
    boundaries = [Boundary.R_OUTER]
    if geometry == "shell":
        boundaries.append(Boundary.R_INNER)
    for loc in boundaries:
        suffix = "out" if loc == Boundary.R_OUTER else "in"
        b_bcs.append(BoundaryCondition(
            loc, Dot(r_hat, b), name=f"b_r_{suffix}",
        ))
        b_bcs.append(BoundaryCondition(
            loc, Dot(theta_hat, b), name=f"b_t_{suffix}",
        ))
        b_bcs.append(BoundaryCondition(
            loc, Dot(phi_hat, b), name=f"b_p_{suffix}",
        ))
    new_bcs = list(base_system.boundary_conditions) + b_bcs

    return EquationSystem(
        [eq_mom_new, eq_ind, eq_cont, eq_energy],
        new_bcs,
    )


# ---------------------------------------------------------------------------
# Radial finite-difference gradient of a background scalar
# ---------------------------------------------------------------------------

def _radial_grad_bg_scalar(scalar_fn, name, h=1e-4, r_lo=None, r_hi=None):
    """Background.Vector (d/dr scalar_fn) r̂ by finite differences.

    `scalar_fn(R, T)` evaluates a radial background scalar. Centered
    differences of step `h` are used in the interior; within `h` of the data
    bounds [r_lo, r_hi] a second-order one-sided stencil is used instead, so no
    probe leaves the bounds.
    """
    def grad_fn(R, T):
        R = np.asarray(R, dtype=float)
        T = np.asarray(T, dtype=float)
        # Assemblers may pass meshgrid columns R = (Nr, 1), T = (1, Nt). The
        # gradient is purely radial, so T is replaced by a zero array of R's
        # shape; the caller's `+0.0*T` wrapper restores the θ broadcast.
        if T.shape != R.shape:
            T = np.zeros_like(R)
        probe = scalar_fn(R.ravel()[:1] if R.size else R,
                          T.ravel()[:1] if T.size else T)
        out = np.empty_like(R, dtype=np.asarray(probe).dtype)
        lo = -np.inf if r_lo is None else r_lo
        hi = np.inf if r_hi is None else r_hi
        ok_center = (R - h >= lo) & (R + h <= hi)
        ok_fwd = (~ok_center) & (R + 2 * h <= hi)
        ok_bwd = (~ok_center) & (~ok_fwd)
        if np.any(ok_center):
            out[ok_center] = (
                scalar_fn(R[ok_center] + h, T[ok_center])
                - scalar_fn(R[ok_center] - h, T[ok_center])
            ) / (2 * h)
        if np.any(ok_fwd):
            out[ok_fwd] = (
                -3 * scalar_fn(R[ok_fwd], T[ok_fwd])
                + 4 * scalar_fn(R[ok_fwd] + h, T[ok_fwd])
                - scalar_fn(R[ok_fwd] + 2 * h, T[ok_fwd])
            ) / (2 * h)
        if np.any(ok_bwd):
            out[ok_bwd] = (
                3 * scalar_fn(R[ok_bwd], T[ok_bwd])
                - 4 * scalar_fn(R[ok_bwd] - h, T[ok_bwd])
                + scalar_fn(R[ok_bwd] - 2 * h, T[ok_bwd])
            ) / (2 * h)
        return out

    return Background.Vector(
        name,
        v_r=NumericComponent(f=grad_fn, description=f"∂_r {name}"),
        v_theta=None, v_phi=None,
    )


# ---------------------------------------------------------------------------
# Bekki & Yokoyama (2017) Figure 2: polytrope, uniform rotation, constant δ
# ---------------------------------------------------------------------------

def make_polytrope_system(
    poly,
    m_val: int,
    Ek: float = 7e-6,
    Le: float = 0.0,
    Pm: float = 1.0,
    amplitude_ratio: float = 30000.0,
    rho_ref: Optional[float] = None,
    entropy_bc: str = "neumann",
    geometry: str = "shell",
    with_grad_lnrho_strain: bool = True,
    omega0_nHz: float = OMEGA0_NHZ,
):
    """Bekki & Yokoyama (2017) Fig. 2 system on a polytrope background.

    Uniform rotation Ω = Ω₀ ẑ, weakly superadiabatic stratification with the
    constant δ of `poly`, and constant diffusivities ν = κ = Ek (Pr = 1).
    Unknowns v, ρ₁, s₁ (plus b when Le > 0); equations momentum, continuity,
    energy (plus induction when Le > 0). The pressure gradient is written
    grouped, Grad(c q), the spelling of the reference implementation: the staggered core
    builds it as one mimetic gradient G diag(c) and the collocated core
    expands it by the product rule itself.

    Momentum:
        ∂_t v + Grad(cs²/ρ₀ ρ₁ + cs²/c_p s₁) - (|g|/c_p) s₁ r̂ + 2 Ω×v
              - Ek [Lap(v) + (1/3) Grad(Div(v))]
              - Ek [Dot(∇ln ρ₀, ∇v + (∇v)ᵀ) - (2/3) (∇·v) ∇ln ρ₀]   (optional)
    Continuity:
        ∂_t ρ₁ + ρ₀ Div(v) + Dot(∇ρ₀, v) = 0
    Energy:
        ∂_t s₁ + (ds₀/dr) v_r - Ek [Lap(s₁) + (dln P₀/dr) ∂_r s₁] = 0

    Nondimensionalization (lengths R_sun, time 1/Ω₀, Ω₀ = 2π·omega0_nHz):
      * cs² enters as the dynamical coefficient c_s²(r)/(R_sun Ω₀)², i.e. the
        polytrope ratio c_s²/c_s²_bc times CS2_DYN = c_s²_bc/(R_sun Ω₀)².
      * gravity enters as g(r)/(R_sun Ω₀²), i.e. the ratio g/g_bc times
        g_bc/(R_sun Ω₀²).
      * c_p ≡ 1 in both s₁ terms of the momentum equation (buoyancy g/c_p and
        pressure cs²/c_p), and the entropy source is ds₀/dr = -δ/H_p (the
        `poly.grad_s0` profile, entropy in units of c_p), so the buoyancy ×
        source loop gives N² = -g δ/H_p: the `poly.N2` profile times
        g_bc/(R_sun Ω₀²).
      * The dln P₀/dr coefficient in the thermal diffusion is a pure radial
        shape, [γ/(γ-1)] Φ'/Φ per R_sun, since κ = Ek and ∇s₁ are already
        nondimensional.

    For δ > 0, ds₀/dr < 0 and radial motions are destabilized, giving the
    growing convective modes (Im ω > 0) of Fig. 2c. The stratification
    correction to the viscous force affects mode linewidths, not the real
    frequencies.

    Parameters
    ----------
    poly : PolytropeBackground
    m_val : int
        Azimuthal order (Fig. 2 uses m = 8).
    Ek : float, default 7e-6
        Ekman number ν/(Ω₀R²); Fig. 2 uses ν = κ = 1e11 cm²/s. κ = ν.
    Le : float, default 0.0
        Lehnert number of the dipolar B₀; Le = 0 is the hydrodynamic system.
    Pm : float, default 1.0
        Magnetic Prandtl number ν/η (Le > 0 only).
    amplitude_ratio : float, default 30000.0
        B₀ inner/outer amplitude ratio (Le > 0 only).
    rho_ref : float or None
        Reference density for the Lorentz prefactor (Le > 0 only); None means
        ρ₀ at r_inner.
    entropy_bc : 'dirichlet' or 'neumann', default 'neumann'
        Entropy condition at the radial walls (the solar eigenproblem: insulating, Neumann).
    geometry : 'shell' or 'ball', default 'shell'
        Must match the grid.
    with_grad_lnrho_strain : bool, default True
        Include the -Ek ∇ln ρ₀·T correction to the viscous force.
    omega0_nHz : float, default OMEGA0_NHZ
        Rotation rate in nHz (Carrington).

    Returns
    -------
    EquationSystem
    """
    if geometry not in ("shell", "ball"):
        raise ValueError(f"geometry must be 'shell' or 'ball'; got {geometry!r}")
    r_lo = poly.r_inner
    r_hi = poly.r_outer
    if Le > 0:
        _require_ball_dipole_inner(geometry, r_lo)
    rho_eval = poly.rho.component
    cs_eval = poly.sound_speed.component   # c_s/c_s_bc (not squared)
    cp_eval = poly.cp.component
    OMEGA0 = 2.0 * np.pi * omega0_nHz * 1e-9   # rad/s
    CS2_DYN = poly.cs2_bc / (poly.R_sun * OMEGA0) ** 2   # ≈ 1.243e4
    g_dyn_norm = poly.g_bc / (poly.R_sun * OMEGA0**2)    # ≈ 1.0e5

    # ---- pressure coefficients cs²/ρ₀ and cs²/c_p (dynamical cs²) ----
    def cs2_over_rho0_fn(R, T):
        cs2 = cs_eval.evaluate_on_grid(R, T) ** 2 * CS2_DYN
        return cs2 / rho_eval.evaluate_on_grid(R, T)

    def cs2_over_cp_fn(R, T):
        cs2 = cs_eval.evaluate_on_grid(R, T) ** 2 * CS2_DYN
        return cs2 / cp_eval.evaluate_on_grid(R, T)

    cs2_over_rho0 = Background.Scalar(
        "cs2_over_rho0",
        NumericComponent(f=cs2_over_rho0_fn, description="c_s²/ρ₀"),
    )
    cs2_over_cp = Background.Scalar(
        "cs2_over_cp",
        NumericComponent(f=cs2_over_cp_fn, description="c_s²/c_p"),
    )

    # ---- buoyancy +(|g|/c_p) r̂ with poly.g_vec = -|g|/g_bc r̂, c_p = 1 ----
    g_r_eval = poly.g_vec.components["v_r"]

    def g_over_cp_r_fn(R, T):
        return (-g_r_eval.evaluate_on_grid(R, T)
                / cp_eval.evaluate_on_grid(R, T)) * g_dyn_norm

    g_over_cp_vec = Background.Vector(
        "g_over_cp_vec",
        v_r=NumericComponent(f=g_over_cp_r_fn,
                             description="|g|/c_p (c_p=1, dynamical ×g_bc/(R_sun Ω₀²))"),
        v_theta=None, v_phi=None,
    )

    # ---- density, entropy source ds₀/dr = -δ/H_p, uniform rotation ----
    rho0 = poly.rho
    grad_rho0 = poly.grad_rho
    _Hp_eval = poly._Hp
    _delta = poly.delta

    def _grad_s0_r_fn(R, T):
        return -_delta / _Hp_eval(R)

    grad_s0 = Background.Vector(
        "grad_s0_optA",
        v_r=NumericComponent(f=_grad_s0_r_fn, description="ds₀/dr = −δ/H_p"),
        v_theta=None, v_phi=None,
    )
    Omega_vec = poly.Omega_vec

    # ---- unknowns ----
    v = Unknown.Vector("v")
    rho1 = Unknown.Scalar("rho1")
    s1 = Unknown.Scalar("s1")

    # ---- momentum ----
    mom_terms = (
        Dt(v)
        + Grad(cs2_over_rho0 * rho1)
        + Grad(cs2_over_cp * s1)
        - g_over_cp_vec * s1
        + 2 * Cross(Omega_vec, v)
    )
    # Viscous force (1/ρ₀)∇·(ρ₀ ν T) = ν[∇·T + ∇ln ρ₀·T] with
    # T = ∇v + (∇v)ᵀ - (2/3)(∇·v)I and ∇·T = ∇²v + (1/3)∇(∇·v).
    if Ek > 0:
        mom_terms = (
            mom_terms
            - sp.S(Ek) * Lap(v)
            - sp.Rational(1, 3) * sp.S(Ek) * Grad(Div(v))
        )
        if with_grad_lnrho_strain:
            grad_rho_r_eval = poly.grad_rho.components["v_r"]

            def grad_ln_rho_r_fn(R, T):
                return (grad_rho_r_eval.evaluate_on_grid(R, T)
                        / rho_eval.evaluate_on_grid(R, T))

            grad_ln_rho = Background.Vector(
                "grad_ln_rho",
                v_r=NumericComponent(f=grad_ln_rho_r_fn,
                                     description="(1/ρ₀) ∂_r ρ₀"),
                v_theta=None, v_phi=None,
            )
            # -Ek ∇ln ρ₀·T = -Ek Dot(∇ln ρ₀, ∇v + (∇v)ᵀ) + (2/3) Ek (∇·v) ∇ln ρ₀
            strain = Grad(v) + Transpose(Grad(v))
            mom_terms = (
                mom_terms
                - sp.S(Ek) * Dot(grad_ln_rho, strain)
                + sp.Rational(2, 3) * sp.S(Ek) * grad_ln_rho * Div(v)
            )

    # ---- continuity ----
    cont_terms = Dt(rho1) + rho0 * Div(v) + Dot(grad_rho0, v)

    # ---- energy ----
    # Thermal diffusion (1/(ρ₀T₀))∇·(κρ₀T₀∇s₁) = κ[∇²s₁ + (dln P₀/dr) ∂_r s₁]
    # for an ideal gas (ρ₀T₀ ∝ P₀), with dln P₀/dr = [γ/(γ-1)] Φ'/Φ and
    # Φ' = -C r_bc/r².
    energy_terms = Dt(s1) + Dot(grad_s0, v)
    if Ek > 0:
        gamma_poly = poly.gamma
        C_poly = poly.C
        r_bc_poly = poly.r_bc

        def grad_ln_P0_r_fn(R, T):
            Phi = poly._Phi(R)
            dPhi = -C_poly * r_bc_poly / R**2
            return (gamma_poly / (gamma_poly - 1.0)) * dPhi / Phi

        grad_ln_P0 = Background.Vector(
            "grad_ln_P0",
            v_r=NumericComponent(f=grad_ln_P0_r_fn,
                                 description="dln P₀/dr = [γ/(γ−1)]Φ'/Φ"),
            v_theta=None, v_phi=None,
        )
        energy_terms = (
            energy_terms
            - sp.S(Ek) * Lap(s1)
            - sp.S(Ek) * Dot(grad_ln_P0, Grad(s1))
        )

    # ---- magnetic layer (Le > 0) ----
    if Le > 0:
        b = Unknown.Vector("b")
        B0 = make_dipolar_B0(
            "B0", r_inner=r_lo, r_outer=r_hi,
            amplitude=1.0, amplitude_ratio=amplitude_ratio,
        )
        J0 = make_dipolar_B0_curl(
            "J0", r_inner=r_lo, r_outer=r_hi,
            amplitude=1.0, amplitude_ratio=amplitude_ratio,
        )
        if rho_ref is None:
            rho_arr = rho_eval.evaluate_on_grid(
                np.array([[r_lo]]), np.array([[np.pi / 2]]))
            rho_ref_val = float(np.real(np.asarray(rho_arr).reshape(-1)[0]))
        else:
            rho_ref_val = float(rho_ref)
        inv_rho_ratio = _make_inv_rho_ratio(poly, rho_ref_val)
        Le_sq = sp.S(Le) ** 2
        mom_terms = (
            mom_terms
            - Le_sq * inv_rho_ratio * Cross(Curl(b), B0)
            - Le_sq * inv_rho_ratio * Cross(J0, b)
        )
        v_ref = FieldRef(v)
        eta = sp.S(Ek) / sp.S(Pm)
        ind_terms = Dt(b) - Curl(Cross(v_ref, B0)) - eta * Lap(b)

    # ---- equations ----
    eq_mom = Equation(mom_terms, name="momentum")
    eq_cont = Equation(cont_terms, name="continuity")
    eq_energy = Equation(energy_terms, name="energy")
    equations = [eq_mom, eq_cont, eq_energy]
    if Le > 0:
        equations.append(Equation(ind_terms, name="induction"))

    # ---- boundary conditions: impenetrable stress-free v, entropy, b ----
    if geometry == "ball":
        bcs = list(ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    else:
        bcs = list(
            ImpenetrableStressFree(v, boundary=Boundary.R_INNER)
            + ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
        )
    boundaries = [Boundary.R_OUTER]
    if geometry == "shell":
        boundaries.append(Boundary.R_INNER)
    if entropy_bc == "dirichlet":
        for loc in boundaries:
            suffix = "out" if loc == Boundary.R_OUTER else "in"
            bcs.append(BoundaryCondition(loc, s1, name=f"s1_{suffix}"))
    elif entropy_bc == "neumann":
        for loc in boundaries:
            suffix = "out" if loc == Boundary.R_OUTER else "in"
            bcs.append(BoundaryCondition(
                loc, Dot(r_hat, Grad(s1)), name=f"s1_{suffix}"))
    else:
        raise ValueError(
            f"entropy_bc must be 'dirichlet' or 'neumann'; got {entropy_bc!r}")
    if Le > 0:
        for loc in boundaries:
            suffix = "out" if loc == Boundary.R_OUTER else "in"
            bcs.append(BoundaryCondition(loc, Dot(r_hat, b), name=f"b_r_{suffix}"))
            bcs.append(BoundaryCondition(loc, Dot(theta_hat, b), name=f"b_t_{suffix}"))
            bcs.append(BoundaryCondition(loc, Dot(phi_hat, b), name=f"b_p_{suffix}"))

    return EquationSystem(equations, bcs)


# ---------------------------------------------------------------------------
# Model S ball with uniform rotation and the solar N²(r)
# ---------------------------------------------------------------------------

def make_solar_ball_system(
    solar,
    m_val: int,
    Ek: float = 7e-6,
    entropy_bc: str = "neumann",
    omega0_nHz: float = OMEGA0_NHZ,
    cp_ref: Optional[float] = None,
):
    """Solar-eigenproblem-style eigensystem on a Model S ball with uniform rotation.

    The equations of `make_polytrope_system` (momentum, continuity,
    energy; unknowns v, ρ₁, s₁; aux-free) with the background taken from
    tabulated Model S over the full ball r ∈ [0, r_outer], Ω = Ω₀ ẑ, and the
    solar stratification N²(r) in place of a constant δ. The background
    fields carry the same names as in the polytrope builder, and the
    pressure gradient is grouped, Grad(c q), as it is there.

    Nondimensionalization (lengths R_sun, time 1/Ω₀, Ω₀ = 2π·omega0_nHz):
      * cs² enters as c_s²(r)/(R_sun Ω₀)² and gravity as g(r)/(R_sun Ω₀²),
        applied to the dimensional (cgs) Model S profiles.
      * c_p(r) varies with radius and appears in both s₁ momentum terms,
        buoyancy (g/c_p) s₁ and pressure (cs²/c_p) ∇s₁, as the same shape
        c_p(r)/c_p_ref. The reference cancels between the 1/c_p in the
        buoyancy and the c_p in the entropy source.
      * The entropy source is c_p ds₀/dr with ds₀/dr = N²/g (both in
        dynamical units), so the buoyancy × source loop
        (g/c_p)(c_p ds₀/dr) = N²(r)/Ω₀²: negative (convective) in the
        convection zone, positive (stably stratified) in the radiative
        interior.
      * Density and pressure enter continuity and the dln P₀/dr thermal
        correction as ratios to their central values, so ρ₁ is in units of
        ρ₀(0) and the ρ₁ pressure coefficient is cs²/(ρ₀/ρ₀(0)): the product
        cs2_over_rho0 × poly_rho is the dynamical cs² everywhere, as in the
        polytrope builder (both ratios to ρ_bc there).

    Parameters
    ----------
    solar : SolarBackgrounds
        Must use geometry=('ball', r_outer). Its `data_dir` is reread with
        nondimensional=False to obtain the cgs profiles.
    m_val : int
        Azimuthal order.
    Ek : float, default 7e-6
        Ekman number ν/(Ω₀R²); κ = ν.
    entropy_bc : 'neumann' or 'dirichlet', default 'neumann'
    omega0_nHz : float, default OMEGA0_NHZ
        Rotation rate in nHz (Carrington).
    cp_ref : float or None
        c_p normalization (cgs); None means the central value c_p(0).

    Returns
    -------
    EquationSystem
    """
    geom = solar.geom
    if getattr(geom, "kind", None) != "ball":
        raise ValueError(
            "make_solar_ball_system requires a ball-geometry SolarBackgrounds "
            f"(geometry=('ball', r_outer)); got {geom!r}")
    r_outer = float(geom.r_max)
    data_dir = solar.data_dir
    solD = SolarBackgrounds(data_dir, ('ball', r_outer), nondimensional=False)
    R_sun = float(solD.normalization["R_sun"])     # cm
    OMEGA0 = 2.0 * np.pi * omega0_nHz * 1e-9        # rad/s
    CS2_FACTOR = 1.0 / (R_sun * OMEGA0) ** 2        # cs²_cgs → dynamical
    G_FACTOR = 1.0 / (R_sun * OMEGA0 ** 2)          # g_cgs → dynamical
    r_lo, r_hi = 0.0, r_outer
    cs_eval = solD.sound_speed.component            # c_s(r) cgs
    cp_eval = solD.cp.component                     # c_p(r) cgs
    rho_eval = solD.rho.component                   # ρ₀(r) cgs
    g_r_eval = solD.g_vec.components["v_r"]          # -|g(r)| cgs
    n2_eval = solD.N2.component                     # N²(r) rad²/s²
    p_eval = solD.pressure.component                # P₀(r) cgs
    if cp_ref is None:
        cp0 = cp_eval.evaluate_on_grid(np.array([[0.0]]), np.array([[np.pi / 2]]))
        cp_ref = float(np.real(np.asarray(cp0).reshape(-1)[0]))
    cp_ref = float(cp_ref)

    def cp_shape(R, T):
        return np.real(cp_eval.evaluate_on_grid(R, T)) / cp_ref

    rho0c = np.real(rho_eval.evaluate_on_grid(
        np.array([[0.0]]), np.array([[np.pi / 2]]))).reshape(-1)[0]

    def rho_ratio_fn(R, T):
        return np.real(rho_eval.evaluate_on_grid(R, T)) / rho0c

    # ---- pressure coefficients (dynamical cs², ρ₁ in units of ρ₀(0)) ----
    def cs2_over_rho0_fn(R, T):
        cs2 = np.real(cs_eval.evaluate_on_grid(R, T)) ** 2 * CS2_FACTOR
        return cs2 / rho_ratio_fn(R, T)

    def cs2_over_cp_fn(R, T):
        cs2 = np.real(cs_eval.evaluate_on_grid(R, T)) ** 2 * CS2_FACTOR
        return cs2 / cp_shape(R, T)

    cs2_over_rho0 = Background.Scalar(
        "cs2_over_rho0",
        NumericComponent(f=cs2_over_rho0_fn, description="c_s²/(ρ₀/ρ₀(0)) (dynamical)"))
    cs2_over_cp = Background.Scalar(
        "cs2_over_cp",
        NumericComponent(f=cs2_over_cp_fn, description="c_s²/c_p (dynamical)"))

    # ---- buoyancy |g|/c_p (dynamical g, variable c_p) ----
    def g_over_cp_r_fn(R, T):
        g = -np.real(g_r_eval.evaluate_on_grid(R, T)) * G_FACTOR
        return g / cp_shape(R, T)

    g_over_cp_vec = Background.Vector(
        "g_over_cp_vec",
        v_r=NumericComponent(f=g_over_cp_r_fn,
                             description="|g|/c_p (dynamical, variable c_p)"),
        v_theta=None, v_phi=None)

    # ---- density ratio ρ₀/ρ₀(0), its gradient, and ∇ln ρ₀ ----
    poly_rho = Background.Scalar(
        "poly_rho", NumericComponent(f=rho_ratio_fn, description="ρ₀/ρ₀(0)"))
    poly_grad_rho = _radial_grad_bg_scalar(
        rho_ratio_fn, "poly_grad_rho", r_lo=r_lo, r_hi=r_hi)

    def grad_ln_rho_r_fn(R, T):
        rr = rho_ratio_fn(R, T)
        gr = poly_grad_rho.components["v_r"].evaluate_on_grid(R, T)
        return np.real(gr) / rr

    grad_ln_rho = Background.Vector(
        "grad_ln_rho",
        v_r=NumericComponent(f=grad_ln_rho_r_fn, description="(1/ρ₀)∂_rρ₀"),
        v_theta=None, v_phi=None)

    # ---- entropy source c_p ds₀/dr with ds₀/dr = N²/g (dynamical units) ----
    def grad_s0_r_fn(R, T):
        g_dyn = -np.real(g_r_eval.evaluate_on_grid(R, T)) * G_FACTOR
        n2 = np.real(n2_eval.evaluate_on_grid(R, T))
        n2_dyn = n2 / OMEGA0 ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_g = np.where(np.abs(g_dyn) > 0, g_dyn, 1.0)
            ds0 = np.where(np.abs(g_dyn) > 0, n2_dyn / safe_g, 0.0)
        return ds0 * cp_shape(R, T)

    grad_s0 = Background.Vector(
        "grad_s0_optA",
        v_r=NumericComponent(f=grad_s0_r_fn,
                             description="c_p·N²/g (physical stratification)"),
        v_theta=None, v_phi=None)

    # ---- uniform rotation Ω₀ ẑ = cos θ r̂ - sin θ θ̂ (|Ω| = 1) ----
    poly_Omega_vec = Background.Vector(
        "poly_Omega_vec",
        v_r=NumericComponent(f=lambda R, T: np.cos(T) + 0 * R, description="Ω·r̂"),
        v_theta=NumericComponent(f=lambda R, T: -np.sin(T) + 0 * R, description="Ω·θ̂"),
        v_phi=None)

    # ---- unknowns ----
    v = Unknown.Vector("v")
    rho1 = Unknown.Scalar("rho1")
    s1 = Unknown.Scalar("s1")

    # ---- momentum ----
    mom_terms = (
        Dt(v)
        + Grad(cs2_over_rho0 * rho1)
        + Grad(cs2_over_cp * s1)
        - g_over_cp_vec * s1
        + 2 * Cross(poly_Omega_vec, v)
    )
    if Ek > 0:
        mom_terms = (
            mom_terms
            - sp.S(Ek) * Lap(v)
            - sp.Rational(1, 3) * sp.S(Ek) * Grad(Div(v))
        )
        strain = Grad(v) + Transpose(Grad(v))
        mom_terms = (
            mom_terms
            - sp.S(Ek) * Dot(grad_ln_rho, strain)
            + sp.Rational(2, 3) * sp.S(Ek) * grad_ln_rho * Div(v)
        )

    # ---- continuity ----
    cont_terms = Dt(rho1) + poly_rho * Div(v) + Dot(poly_grad_rho, v)

    # ---- energy: N² source and thermal diffusion with dln P₀/dr ----
    energy_terms = Dt(s1) + Dot(grad_s0, v)
    if Ek > 0:
        def p_ratio_fn(R, T):
            p0c = np.real(p_eval.evaluate_on_grid(
                np.array([[0.0]]), np.array([[np.pi / 2]]))).reshape(-1)[0]
            return np.real(p_eval.evaluate_on_grid(R, T)) / p0c
        grad_P0 = _radial_grad_bg_scalar(
            p_ratio_fn, "grad_P0_tmp", r_lo=r_lo, r_hi=r_hi)

        def grad_ln_P0_r_fn(R, T):
            pr = p_ratio_fn(R, T)
            gp = grad_P0.components["v_r"].evaluate_on_grid(R, T)
            return np.real(gp) / pr

        grad_ln_P0 = Background.Vector(
            "grad_ln_P0",
            v_r=NumericComponent(f=grad_ln_P0_r_fn, description="dlnP₀/dr"),
            v_theta=None, v_phi=None)
        energy_terms = (
            energy_terms
            - sp.S(Ek) * Lap(s1)
            - sp.S(Ek) * Dot(grad_ln_P0, Grad(s1))
        )

    # ---- equations and boundary conditions ----
    eq_mom = Equation(mom_terms, name="momentum")
    eq_cont = Equation(cont_terms, name="continuity")
    eq_energy = Equation(energy_terms, name="energy")
    bcs = list(ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    if entropy_bc == "dirichlet":
        bcs.append(BoundaryCondition(Boundary.R_OUTER, s1, name="s1_out"))
    elif entropy_bc == "neumann":
        bcs.append(BoundaryCondition(
            Boundary.R_OUTER, Dot(r_hat, Grad(s1)), name="s1_out"))
    else:
        raise ValueError(
            f"entropy_bc must be 'dirichlet' or 'neumann'; got {entropy_bc!r}")

    return EquationSystem([eq_mom, eq_cont, eq_energy], bcs)


# ---------------------------------------------------------------------------
# Generic magnetic layer on an existing hydrodynamic system
# ---------------------------------------------------------------------------

def add_magnetic_field(system, B0, *, Le, eta, J0=None, inv_rho_ratio=None,
                       b_name: str = "b", bc: str = "conducting",
                       lorentz_sign: float = -1.0, geometry: str = "shell",
                       solenoidal: bool = False, psi_name: str = "psi",
                       formulation: str = "b"):
    """Return a new EquationSystem = ``system`` + magnetic field ``b``.

    The momentum equation (the vector equation of ``system``) gains

        lorentz_sign * Le**2 * inv_rho_ratio * [Cross(Curl(b), B0) + Cross(J0, b)]

    (``inv_rho_ratio`` and ``J0`` are omitted when None) and the induction
    equation ``Dt(b) - Curl(Cross(v, B0)) - eta*Lap(b) = 0`` is appended. With
    ``lorentz_sign = -1`` (default) the left-hand side carries
    ``-Le² (ρ_ref/ρ₀)(∇×b)×B₀``, i.e. ∂_t v = +Le² (ρ_ref/ρ₀)(∇×b)×B₀, the
    physical Lorentz force.

    ``bc`` selects the wall conditions at each radial boundary:
    'conducting' -> b_r = 0 and the total tangential electric field
    E = eta*Curl(b) - Cross(v, B0) vanishes (the ideal part matters wherever
    B0_r != 0 at the wall); 'pinned' -> all three components of b vanish;
    'radial' -> b_r = 0 only; 'free' -> none.

    ``solenoidal=True`` adds a scalar multiplier ``psi`` (``+ Grad(psi)`` in
    the induction equation, ``Div(b) = 0`` as a constraint, ``psi`` gauged) so
    that div b = 0 is enforced on the spectral and collocated backends; the
    staggered constrained-transport backend has it built in and ignores the
    multiplier.

    ``formulation='A'`` (spectral backend) uses the vector potential instead:
    unknowns ``A`` (named ``b_name``) and the electric potential ``psi``,

        Dt(A) - Cross(v, B0) - eta*Lap(A) - Grad(psi) = 0,   Div(A) = 0

    (Coulomb gauge, psi gauged by integ(psi) = 0), with b = Curl(A) in the
    Lorentz force and the perfect-conductor walls of Lecoanet et al. (2019,
    eq. 26): tangential A = 0 and psi = 0 (b_r = 0 and tangential E = 0
    follow).
    """
    if formulation == "A":
        return _add_vector_potential(system, B0, Le=Le, eta=eta, J0=J0,
                                     inv_rho_ratio=inv_rho_ratio, A_name=b_name,
                                     psi_name=psi_name, lorentz_sign=lorentz_sign,
                                     geometry=geometry)
    if formulation != "b":
        raise ValueError("formulation must be 'b' or 'A'")
    v = next(f for f in system.unknowns.values() if f.kind == "vector")
    v_ref = FieldRef(v)
    b = Unknown.Vector(b_name)
    coeff = sp.S(lorentz_sign) * sp.S(Le) ** 2
    lor = Cross(Curl(b), B0)
    if inv_rho_ratio is not None:
        lor_term = coeff * inv_rho_ratio * lor
        cur_term = (coeff * inv_rho_ratio * Cross(J0, b)) if J0 is not None else None
    else:
        lor_term = coeff * lor
        cur_term = (coeff * Cross(J0, b)) if J0 is not None else None

    equations = []
    for eq in system.equations:
        if eq.kind == "vector":
            lhs = eq.lhs + lor_term
            if cur_term is not None:
                lhs = lhs + cur_term
            equations.append(Equation(lhs, name=eq.name))
        else:
            equations.append(eq)
    ind = Dt(b) - Curl(Cross(v_ref, B0)) - sp.S(eta) * Lap(b)
    gauge = list(system.gauge_unknowns)
    if solenoidal:
        psi = Unknown.Scalar(psi_name)
        ind = ind + Grad(psi)
        equations.append(Equation(ind, name="induction"))
        equations.append(Equation(Div(b), name="solenoidal"))
        gauge.append(psi_name)
    else:
        equations.append(Equation(ind, name="induction"))

    bcs = list(system.boundary_conditions)
    boundaries = [Boundary.R_OUTER] + ([Boundary.R_INNER] if geometry == "shell" else [])
    for loc in boundaries:
        suffix = "out" if loc == Boundary.R_OUTER else "in"
        if bc in ("conducting", "radial", "pinned"):
            bcs.append(BoundaryCondition(loc, Dot(r_hat, b), name=f"{b_name}_r_{suffix}"))
        if bc == "conducting":
            E = sp.S(eta) * Curl(b) - Cross(v_ref, B0)
            bcs.append(BoundaryCondition(loc, Dot(theta_hat, E), name=f"{b_name}_Et_{suffix}"))
            bcs.append(BoundaryCondition(loc, Dot(phi_hat, E), name=f"{b_name}_Ep_{suffix}"))
        elif bc == "pinned":
            bcs.append(BoundaryCondition(loc, Dot(theta_hat, b), name=f"{b_name}_t_{suffix}"))
            bcs.append(BoundaryCondition(loc, Dot(phi_hat, b), name=f"{b_name}_p_{suffix}"))
        elif bc not in ("radial", "free"):
            raise ValueError("bc must be 'conducting', 'pinned', 'radial' or 'free'")
    return EquationSystem(equations, bcs, gauge_unknowns=gauge)


def _add_vector_potential(system, B0, *, Le, eta, J0=None, inv_rho_ratio=None,
                          A_name="A", psi_name="psi", lorentz_sign=-1.0,
                          geometry="shell"):
    """Vector-potential form of `add_magnetic_field` (``formulation='A'``).

    Unknowns A and psi; b = Curl(A) in the Lorentz force; induction
    ``Dt(A) - Cross(v, B0) - eta*Lap(A) - Grad(psi) = 0`` with the Coulomb
    gauge ``Div(A) = 0``; walls: tangential A = 0 and psi = 0.
    """
    v = next(f for f in system.unknowns.values() if f.kind == "vector")
    v_ref = FieldRef(v)
    A = Unknown.Vector(A_name)
    psi = Unknown.Scalar(psi_name)
    b = Curl(A)
    coeff = sp.S(lorentz_sign) * sp.S(Le) ** 2
    lor = Cross(Curl(b), B0)
    cur = Cross(J0, b) if J0 is not None else None
    if inv_rho_ratio is not None:
        lor = coeff * inv_rho_ratio * lor
        cur = (coeff * inv_rho_ratio * cur) if cur is not None else None
    else:
        lor = coeff * lor
        cur = (coeff * cur) if cur is not None else None
    equations = []
    for eq in system.equations:
        if eq.kind == "vector":
            lhs = eq.lhs + lor
            if cur is not None:
                lhs = lhs + cur
            equations.append(Equation(lhs, name=eq.name))
        else:
            equations.append(eq)
    equations.append(Equation(Dt(A) - Cross(v_ref, B0) - sp.S(eta) * Lap(A) - Grad(psi),
                              name="induction"))
    equations.append(Equation(Div(A), name="coulomb_gauge"))
    bcs = list(system.boundary_conditions)
    boundaries = [Boundary.R_OUTER] + ([Boundary.R_INNER] if geometry == "shell" else [])
    for loc in boundaries:
        suffix = "out" if loc == Boundary.R_OUTER else "in"
        bcs.append(BoundaryCondition(loc, Dot(theta_hat, A), name=f"{A_name}_t_{suffix}"))
        bcs.append(BoundaryCondition(loc, Dot(phi_hat, A), name=f"{A_name}_p_{suffix}"))
        bcs.append(BoundaryCondition(loc, psi, name=f"{psi_name}_{suffix}"))
    gauge = list(system.gauge_unknowns) + [psi_name]
    return EquationSystem(equations, bcs, gauge_unknowns=gauge)
