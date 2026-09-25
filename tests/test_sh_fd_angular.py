"""Analytic spin-weighted angular operators (spdee.backends.sh_fd.angular)."""

import numpy as np
import pytest

from spdee.backends.sh_fd.angular import AngularVasil, kcoef, sYlm_theta


@pytest.mark.parametrize("m,lmax", [(1, 10), (2, 12), (8, 24)])
def test_exact_identities_and_pointwise(m, lmax):
    a = AngularVasil(m, lmax)
    assert a.validate() < 1e-13
    assert a.validate_mult() < 1e-12


def test_coriolis_is_anti_hermitian_with_unit_spectrum():
    a = AngularVasil(2, 12)
    C = a.coriolis()
    assert np.abs(C + C.conj().T).max() < 1e-14
    ev = np.linalg.eigvals(C)
    assert np.abs(ev.real).max() < 1e-12
    assert abs(np.abs(ev).max() - 1.0) < 1e-12


def test_toroidal_rayleigh_quotient():
    """A toroidal mode t_pm = +-i k^pm psi has <t, ez x t>/<t,t> = -i m/(l(l+1))."""
    m, l = 2, 2
    a = AngularVasil(m, 12)
    C = a.coriolis()
    j = list(a.ls).index(l)
    t = np.zeros(3 * a.nl, dtype=complex)
    t[0 * a.nl + j] = +1j * kcoef(-1, l, 0)
    t[1 * a.nl + j] = -1j * kcoef(+1, l, 0)
    q = np.vdot(t, C @ t) / np.vdot(t, t)
    assert abs(q - (-1j * m / (l * (l + 1.0)))) < 1e-13


def test_m0_operators_and_absent_harmonics():
    """m = 0 carries l = 0..ell_max; the spin pm1 harmonics of l = 0 are zero
    and every operator gives them zero coefficients."""
    a = AngularVasil(0, 8)
    assert a.ls[0] == 0 and a.nl == 9
    assert a.validate() < 1e-13
    assert a.validate_mult() < 1e-12
    th = np.linspace(0.1, 3.0, 5)
    assert np.all(sYlm_theta(1, 0, 0, th) == 0) and np.all(sYlm_theta(-1, 0, 0, th) == 0)
    assert kcoef(+1, 0, -1) == 0 and kcoef(-1, 0, +1) == 0 and kcoef(+1, 0, 0) == 0
    for s in (-1, 1):
        assert np.abs(a.C(s)[0, :]).max() == 0 and np.abs(a.C(s)[:, 0]).max() == 0
    with pytest.raises(ValueError):
        AngularVasil(-1, 4)


@pytest.mark.parametrize("m", [0, 1, 2, 3])
def test_antipodal_identity(m):
    """sY_lm(pi - theta, phi + pi) = (-1)^l (-s)Y_lm(theta, phi): the identity
    behind the ball closure (docs/ball_sh_fd.md)."""
    th = np.linspace(0.05, 3.1, 9)
    for s in (-2, -1, 0, 1, 2):
        for l in range(max(abs(m), abs(s)), 8):
            lhs = sYlm_theta(s, l, m, np.pi - th) * (-1) ** m
            rhs = (-1) ** l * sYlm_theta(-s, l, m, th)
            assert np.abs(lhs - rhs).max() < 1e-12, (m, s, l)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_spin2_harmonics_and_ladder(m):
    """The |s| = 2 harmonics behind the strain rows and the tensor divergence:
    orthonormal under the quadrature, absent for l < 2, and the ladder
    k^-_{l,2} k^+_{l,1} = -(l-1)(l+2)/2 (eth-bar eth on spin 1)."""
    a = AngularVasil(m, 10)
    th, w = a.quadrature()
    for s in (-2, 2):
        Y = np.array([sYlm_theta(s, l, m, th) for l in a.ls])
        G = 2 * np.pi * (Y * w) @ Y.T
        live = a.ls >= 2
        assert np.abs(G[live][:, live] - np.eye(live.sum())).max() < 1e-12
        assert not np.any(Y[~live])
    for l in a.ls:
        if l < 2:
            assert kcoef(-1, l, 2) == 0 and kcoef(+1, l, -2) == 0
            continue
        assert abs(kcoef(-1, l, 2) * kcoef(+1, l, 1) + (l - 1) * (l + 2) / 2) < 1e-12
        assert abs(kcoef(+1, l, -2) * kcoef(-1, l, -1) + (l - 1) * (l + 2) / 2) < 1e-12

