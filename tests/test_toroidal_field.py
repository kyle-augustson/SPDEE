"""The toroidal band of `toroidal_eqpeak_field`: J0 is the curl of B0, the
Lorentz prefactor is the density ratio, and the amplitude label maps to the
Lehnert number through rho0 at the base (the solar-eigenproblem cross-validation
project's constants: rho0_ref = 0.2 g/cm^3 at r = 0.71 for B_phi(base) =
4534 G gives Le = 1.4260e-2).
"""

import numpy as np
import pytest
import sympy as sp

from spdee.applications.solar_eigenmodes import toroidal_eqpeak_field
from spdee.applications.solar_strings import RSUN, OMEGA_CARR, polytrope_exprs
from spdee.ir.fields import r, theta


def _curl_phi_only(B_phi):
    """curl of B_phi phi_hat in spherical coordinates, (r, theta) components."""
    return (sp.diff(sp.sin(theta) * B_phi, theta) / (r * sp.sin(theta)),
            -sp.diff(r * B_phi, r) / r)


@pytest.mark.parametrize("profile,shape", [("sqrtrho", "eqpeak"), ("cubic", "eqnode")])
def test_current_is_curl_of_field(profile, shape):
    F = toroidal_eqpeak_field(4534.0, profile=profile, shape=shape)
    assert F.B0.v_r == 0 and F.B0.v_theta == 0 and F.J0.v_phi == 0
    J_r, J_t = _curl_phi_only(F.B0.v_phi)
    R, T = np.meshgrid(np.linspace(0.71, 0.985, 9), np.linspace(0.2, 2.9, 7), indexing="ij")
    for got, want in ((F.J0.v_r, J_r), (F.J0.v_theta, J_t)):
        g = sp.lambdify((r, theta), got, "numpy")(R, T)
        w = sp.lambdify((r, theta), want, "numpy")(R, T)
        assert np.max(np.abs(g - w)) < 1e-12 * np.max(np.abs(w))


def test_sqrtrho_profile_and_prefactor():
    F = toroidal_eqpeak_field(4534.0)
    rho0 = polytrope_exprs(r)["rho0"]
    rr = np.linspace(0.71, 0.985, 9)
    f = sp.lambdify(r, F.B0.v_phi.subs(theta, sp.pi / 2), "numpy")(rr)
    rho = sp.lambdify(r, rho0, "numpy")(rr)
    assert np.allclose(f, np.sqrt(rho / F.rho0_ref), rtol=1e-12)
    assert abs(f[0] - 1.0) < 1e-12
    inv = sp.lambdify(r, F.inv_rho_ratio.expr, "numpy")(rr)
    assert np.allclose(inv, F.rho0_ref / rho, rtol=1e-12)


def test_lehnert_number():
    F = toroidal_eqpeak_field(4534.0)
    assert F.r_ref == 0.71
    assert abs(F.rho0_ref - 0.2) < 1e-12
    vA = 4534.0 / np.sqrt(4 * np.pi * 0.2)
    assert abs(F.Le - vA / (RSUN * OMEGA_CARR)) < 1e-15
    assert abs(F.Le - 1.4259977e-2) < 1e-9
