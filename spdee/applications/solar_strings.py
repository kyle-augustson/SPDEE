"""The BCG22 / Bekki & Yokoyama (2017) hydro system entered as strings, in the
spelling of the solar-eigenproblem cross-validation project's Dedalus code.

Nondimensionalization: length Rsun, time 1/Omega0 (Omega0 = 2 pi 456 nHz),
velocity Rsun*Omega0; the eigenvalue is omega/Omega0.  Unknowns: v
(velocity), rhat = rho1/rho0, S = s1/c_p.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from spdee.problem import Problem

RSUN = 7.0e10
OMEGA_CARR = 2.0 * np.pi * 456.0e-9
GAMMA = 5.0 / 3.0
RHO_BC, P_BC, G_BC = 0.2, 6.0e13, 5.2e4
R_BC = 0.71
RMIN, RMAX = 0.71, 0.985
H_BC = P_BC / (RHO_BC * G_BC) / RSUN
U_CGS = RSUN * OMEGA_CARR


def ekman(nu_cgs: float) -> float:
    return nu_cgs / (RSUN ** 2 * OMEGA_CARR)


def polytrope_exprs(r):
    """SymPy expressions of the adiabatic polytrope in r (Rsun units)."""
    zeta = 1 + (GAMMA - 1) / GAMMA * (R_BC / H_BC) * (R_BC / r - 1)
    rho0 = RHO_BC * zeta ** (1 / (GAMMA - 1))
    p0 = P_BC * zeta ** (GAMMA / (GAMMA - 1))
    return dict(zeta=zeta, rho0=rho0, p0=p0,
                Cs2=(GAMMA * p0 / rho0) / U_CGS ** 2,
                gacc=(G_BC * (r / R_BC) ** -2) / (RSUN * OMEGA_CARR ** 2),
                dlnrho=sp.diff(sp.log(rho0), r), dlnp=sp.diff(sp.log(p0), r))


def solar_hydro_problem(nu_cgs: float = 2.0e12, kappa_cgs: float | None = None,
                        delta: float = 1.0e-6, rotation="uniform",
                        entropy_bc: str = "neumann", dr_amp: float = 1.0,
                        entropy_gradient: str = "none", tw_scale: float = 1.0,
                        frame_rate: float = 1.0, weights: str = "none") -> Problem:
    """The solar-eigenproblem hydro system as a Problem; nu, kappa in cm^2/s, delta
    the superadiabaticity, the entropy walls insulating ('neumann', as in BCG22) or
    isentropic ('dirichlet').

    ``rotation`` is 'uniform' (Omega0) or a differential profile: any object
    with vectorized callables ``Omega(r, theta)``, ``dOdr``, ``dOdth`` in
    nondimensional units (r in Rsun, Omega in Omega0).  The frame rotates at
    Omega0 and the profile enters as the mean zonal flow
    U0 = dr_amp (Omega/Omega0 - 1) r sin(theta) phi_hat, through
    U0.grad(v) + v.grad(U0) in momentum and U0.grad on rhat and S (BCG22
    Eqs. 6-10 written covariantly).  v.grad(U0) is entered as GU0T.v with
    the background tensor GU0T = (grad U0)^T, given analytically from Omega
    and its derivatives (regular at the poles), so that every core takes the
    same coefficients.  ``entropy_gradient='thermal_wind'`` adds
    the balancing latitudinal entropy gradient (BCG22 Eq. 27) as
    grad(S0).v in the entropy equation, grad(S0) = (1/r) dS0/dtheta theta_hat,
    (1/c_p) dS0/dtheta = tw_scale r^2 sin(theta) d(Omega_b^2)/dz / g with
    Omega_b = 1 + dr_amp (Omega - 1).  Both are meridional coefficient fields
    evaluated on the solver's grid, as in the reference implementation.

    ``frame_rate`` scales the Coriolis term, 2 frame_rate ez x v (1 = Omega0;
    the rigid-increment identity of the tests uses 1 + epsilon).

    ``weights='solar_eigenproblem'`` multiplies momentum, continuity and energy by
    r^2 H_rho, H_rho and H_p (H = -1 / dln/dr), the form of the reference
    implementation (`Equation` weight): the same PDE, but on the
    Dedalus core a different tau problem, and at nu = 1e12 with the
    thermal wind the m = 8 eigenvalues near -110 - 15i nHz differ by up
    to tens of nHz between the two forms at 64 x 192 and above (both
    resolution-converged); 'none' (default) is the unweighted form.
    Pointwise cores give the same eigenvalues either way."""
    if weights not in ("none", "solar_eigenproblem"):
        raise ValueError(f"weights {weights!r}: 'none' or 'solar_eigenproblem'")
    kappa_cgs = nu_cgs if kappa_cgs is None else kappa_cgs
    Ek, Ekap = ekman(nu_cgs), ekman(kappa_cgs)
    pb = Problem()
    pb.add_unknown("v", "vector"); pb.add_unknown("rhat", "scalar"); pb.add_unknown("S", "scalar")
    ex = polytrope_exprs(pb.namespace["r"])
    pb.add_background("Cs2", expr=ex["Cs2"])
    pb.add_background("gacc", expr=ex["gacc"])
    pb.add_background("dlnrho", expr=ex["dlnrho"])
    pb.add_background("dlnp", expr=ex["dlnp"])
    pb.add_background("invHp", expr=-ex["dlnp"])
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("er_", v_r=1)
    pb.add_parameter("Ek", Ek, symbolic=False)
    pb.add_parameter("Ekap", Ekap, symbolic=False)
    pb.add_parameter("delta", delta, symbolic=False)
    pb.add_parameter("frame_rate", frame_rate, symbolic=False)
    # momentum: dt v + grad Pi - g S er + 2 ez x v - Ek [div sigma + dlnrho (er . sigma)] = 0
    dr_mom = dr_cont = dr_ent = ""
    if not (isinstance(rotation, str) and rotation == "uniform"):
        if isinstance(rotation, str):
            raise ValueError(f"rotation {rotation!r}: pass 'uniform' or a profile object "
                             "with Omega, dOdr, dOdth (nondimensional)")
        _add_mean_flow(pb, rotation, dr_amp, entropy_gradient, tw_scale, ex)
        dr_mom = " + dot(U0, grad(v)) + dot(GU0T, v)"
        dr_cont = " + dot(U0, grad(rhat))"
        dr_ent = " + dot(U0, grad(S))" + (" + dot(gS0, v)" if entropy_gradient == "thermal_wind" else "")
    elif entropy_gradient != "none":
        raise ValueError("entropy_gradient needs a differential rotation profile")
    r = pb.namespace["r"]
    w_mom, w_cont, w_ent = ((r ** 2 * (-1 / ex["dlnrho"]), -1 / ex["dlnrho"], -1 / ex["dlnp"]) if weights == "solar_eigenproblem"
                            else (None, None, None))
    pb.add_equation("dt(v) + grad(Cs2*(rhat+S)) - gacc*S*er_ + 2*frame_rate*cross(ez, v)"
                    " - Ek*(lap(v) + (1/3)*grad(div(v)) + dlnrho*dot(er_, grad(v) + trans(grad(v)))"
                    " - (2/3)*dlnrho*div(v)*er_)" + dr_mom + " = 0", name="momentum", weight=w_mom)
    pb.add_equation("dt(rhat) + div(v) + dlnrho*dot(er_, v)" + dr_cont + " = 0", name="continuity", weight=w_cont)
    pb.add_equation("dt(S) - delta*invHp*dot(er_, v) - Ekap*(lap(S) + dlnp*dot(er_, grad(S)))"
                    + dr_ent + " = 0", name="energy", weight=w_ent)
    pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    s_wall = "dot(r_hat, grad(S)) = 0" if entropy_bc == "neumann" else "S = 0"
    pb.add_bc(s_wall, at="inner"); pb.add_bc(s_wall, at="outer")
    return pb


def _add_mean_flow(pb: Problem, rot, dr_amp: float, entropy_gradient: str, tw_scale: float, ex) -> None:
    """Declare the backgrounds U0 (and gS0 for the thermal wind) of a
    differential profile ``rot`` on ``pb``; see `solar_hydro_problem`."""
    from spdee.ir.backgrounds import NumericComponent
    if entropy_gradient not in ("none", "thermal_wind"):
        raise ValueError(f"entropy_gradient {entropy_gradient!r}: 'none' or 'thermal_wind'")
    a = float(dr_amp)
    gfun = sp.lambdify(pb.namespace["r"], ex["gacc"], modules="numpy")

    def u0(R, T):
        return a * (np.asarray(rot.Omega(R, T)) - 1.0) * R * np.sin(T)

    pb.add_background("U0", v_r=0, v_theta=0,
                      v_phi=NumericComponent(f=u0, description="U0_phi = a (Omega - 1) r sin(theta)"))

    # (grad U0)^T for U0 = U(r, theta) phi_hat: (v.grad U0)_r = -U v_phi / r,
    # (v.grad U0)_theta = -U cot(theta) v_phi / r, (v.grad U0)_phi = dU/dr v_r + (1/r) dU/dtheta v_theta
    def dudr(R, T):
        return a * (np.asarray(rot.dOdr(R, T)) * R + np.asarray(rot.Omega(R, T)) - 1.0) * np.sin(T)

    def dudth_r(R, T):
        return a * (np.asarray(rot.dOdth(R, T)) * np.sin(T) + (np.asarray(rot.Omega(R, T)) - 1.0) * np.cos(T))

    def m_u_r(R, T):
        return -a * (np.asarray(rot.Omega(R, T)) - 1.0) * np.sin(T)

    def m_ucot_r(R, T):
        return -a * (np.asarray(rot.Omega(R, T)) - 1.0) * np.cos(T)

    comps = {(i, j): sp.S.Zero for i in ("r", "theta", "phi") for j in ("r", "theta", "phi")}
    comps[("phi", "r")] = NumericComponent(f=dudr, description="dU0/dr")
    comps[("phi", "theta")] = NumericComponent(f=dudth_r, description="(1/r) dU0/dtheta")
    comps[("r", "phi")] = NumericComponent(f=m_u_r, description="-U0/r")
    comps[("theta", "phi")] = NumericComponent(f=m_ucot_r, description="-U0 cot(theta)/r")
    pb.add_background("GU0T", components=comps)
    if entropy_gradient == "thermal_wind":
        def gs0(R, T):
            Ob = 1.0 + a * (np.asarray(rot.Omega(R, T)) - 1.0)
            Or, Oth = a * np.asarray(rot.dOdr(R, T)), a * np.asarray(rot.dOdth(R, T))
            dS0 = float(tw_scale) * R ** 2 * np.sin(T) * 2.0 * Ob * (np.cos(T) * Or - np.sin(T) / R * Oth) / gfun(R)
            return dS0 / R

        pb.add_background("gS0", v_r=0, v_phi=0,
                          v_theta=NumericComponent(f=gs0, description="(1/r) dS0/dtheta (thermal wind)"))
