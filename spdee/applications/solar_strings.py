"""The BCG22 / Bekki & Yokoyama hydro system entered as strings, in the
spelling of the Bekki cross-validation project's Dedalus code.

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
                        delta: float = 1.0e-6, rotation: str = "uniform",
                        entropy_bc: str = "neumann") -> Problem:
    """The Bekki hydro eigenproblem as a Problem; nu, kappa in cm^2/s, delta
    the superadiabaticity, uniform rotation at Omega0, the entropy walls
    insulating ('neumann', Bekki) or isentropic ('dirichlet')."""
    if rotation != "uniform":
        raise NotImplementedError("solar_hydro_problem: only uniform rotation")
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
    # momentum: dt v + grad Pi - g S er + 2 ez x v - Ek [div sigma + dlnrho (er . sigma)] = 0
    pb.add_equation("dt(v) + grad(Cs2*(rhat+S)) - gacc*S*er_ + 2*cross(ez, v)"
                    " - Ek*(lap(v) + (1/3)*grad(div(v)) + dlnrho*dot(er_, grad(v) + trans(grad(v)))"
                    " - (2/3)*dlnrho*div(v)*er_) = 0", name="momentum")
    pb.add_equation("dt(rhat) + div(v) + dlnrho*dot(er_, v) = 0", name="continuity")
    pb.add_equation("dt(S) - delta*invHp*dot(er_, v) - Ekap*(lap(S) + dlnp*dot(er_, grad(S))) = 0",
                    name="energy")
    pb.add_stress_free("v", at="inner"); pb.add_stress_free("v", at="outer")
    s_wall = "dot(r_hat, grad(S)) = 0" if entropy_bc == "neumann" else "S = 0"
    pb.add_bc(s_wall, at="inner"); pb.add_bc(s_wall, at="outer")
    return pb
