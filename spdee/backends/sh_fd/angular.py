"""Analytic spin-weighted spherical-harmonic angular operators at fixed
azimuthal order m (Vasil et al. 2019, JCPX 3, 100013).

A vector is stored as spin components (-, +, r) = (0, 1, 2), each a
coefficient vector over l = m..ell_max; at m = 0 the spin pm1 entries of
l = 0 are absent (every operator gives them zero).

    scalar_lap        diag(-l(l+1))
    grad_scalar       k^mu_{l,0}: (grad f)_pm = (1/r) k^pm f
    div_vector_ang    (k^+_{l,-1}, k^-_{l,+1}): angular divergence
    C(s), Sp(s), Sm(s)  cos(theta) on spin s; sin(theta) spin s -> s +- 1
    coriolis          cross(ez, .) on (-, +, r)
    mult_matrix       multiplication by any g(theta) between spins by
                      Gauss-Legendre quadrature (exact for polynomial g)

``kcoef(mu, l, s)`` are the derivative coefficients used by the radial
compiler for grad, div, curl and the strain tensor.
"""

from __future__ import annotations

from math import lgamma

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_jacobi

__all__ = ["sYlm_theta", "AngularVasil", "kcoef"]


def sYlm_theta(s, l, m, theta):
    """Spin-weighted harmonic Y^s_{l,m}(theta, phi=0)."""
    l0 = max(abs(m), abs(s)); l1 = min(abs(m), abs(s))
    if l < l0:
        return np.zeros_like(theta)
    a = abs(m + s); b = abs(m - s); n = l - l0
    lognum = np.log(2 * l + 1.0) + lgamma(l + l0 + 1) + lgamma(l - l0 + 1)
    logden = np.log(4 * np.pi) + lgamma(l + l1 + 1) + lgamma(l - l1 + 1)
    N = (-1.0) ** max(m, -s) * np.exp(0.5 * (lognum - logden))
    return N * np.sin(theta / 2) ** a * np.cos(theta / 2) ** b * eval_jacobi(n, a, b, np.cos(theta))


def kcoef(mu, l, s):
    """Derivative coefficient k^mu_{l,s}."""
    num = (l - mu * s) * (l + mu * s + 1.0)
    return 0.0 if num < 0 else -mu * np.sqrt(num / 2.0)


class AngularVasil:
    """Analytic angular operators at fixed m over l = m..ell_max."""

    def __init__(self, m, ell_max):
        m = int(m)
        if m < 0:
            raise ValueError("m must be >= 0")
        if ell_max < m:
            raise ValueError("ell_max must be >= m")
        self.m = m; self.ell_max = int(ell_max)
        self.ls = np.arange(m, ell_max + 1)
        self.nl = len(self.ls)

    def _alpha(self, l, s):
        return 0.0 if l == 0 else -self.m * s / (l * (l + 1.0))

    def _beta(self, l, s):
        m = self.m
        if l < max(abs(m), abs(s)) or l == 0:
            return 0.0
        num = (l - m) * (l + m) * (l - s) * (l + s)
        if num <= 0:
            return 0.0
        return np.sqrt(num / ((2 * l - 1.0) * (2 * l + 1.0))) / l

    def _rho(self, l, s):
        num = (l - s) * (l + s + 1.0)
        if num <= 0 or l == 0:
            return 0.0
        return self.m * np.sqrt(num) / (l * (l + 1.0))

    def _kappa(self, l, s):
        m = self.m
        if l == 0:
            return 0.0
        num = (l - m) * (l + m) * (l - s - 1.0) * (l - s)
        if num <= 0:
            return 0.0
        return np.sqrt(num / ((2 * l - 1.0) * (2 * l + 1.0))) / l

    def C(self, s):
        """cos(theta) on spin s: symmetric tridiagonal."""
        nl = self.nl; M = np.zeros((nl, nl))
        for j, l in enumerate(self.ls):
            M[j, j] = self._alpha(l, s)
            if j + 1 < nl:
                M[j + 1, j] = self._beta(l + 1, s)
                M[j, j + 1] = self._beta(l + 1, s)
        return M

    def Sp(self, s):
        """sin(theta), spin s -> s+1."""
        nl = self.nl; M = np.zeros((nl, nl))
        for j, l in enumerate(self.ls):
            M[j, j] = self._rho(l, s)
            if j + 1 < nl:
                M[j + 1, j] = self._kappa(l + 1, -s - 1)
            if j - 1 >= 0:
                M[j - 1, j] = -self._kappa(l, s)
        return M

    def Sm(self, s):
        """sin(theta), spin s -> s-1."""
        nl = self.nl; M = np.zeros((nl, nl))
        for j, l in enumerate(self.ls):
            M[j, j] = self._rho(l, s - 1)
            if j + 1 < nl:
                M[j + 1, j] = -self._kappa(l + 1, s - 1)
            if j - 1 >= 0:
                M[j - 1, j] = self._kappa(l, -s)
        return M

    def k(self, mu, l, s):
        return kcoef(mu, l, s)

    def scalar_lap(self):
        return np.diag(-self.ls * (self.ls + 1.0)).astype(complex)

    def grad_scalar(self):
        """gm[spin, l]: (grad f)_spin = (1/r) gm[spin, l] f, spin (0, 1) = (-, +)."""
        gm = np.zeros((2, self.nl), dtype=complex)
        for j, l in enumerate(self.ls):
            gm[0, j] = kcoef(-1, l, 0)
            gm[1, j] = kcoef(+1, l, 0)
        return gm

    def div_vector_ang(self):
        """dv[spin, l]: div|_ang = (1/r)(dv[0, l] u_- + dv[1, l] u_+)."""
        dv = np.zeros((2, self.nl), dtype=complex)
        for j, l in enumerate(self.ls):
            dv[0, j] = kcoef(+1, l, -1)
            dv[1, j] = kcoef(-1, l, +1)
        return dv

    def coriolis(self):
        """cross(ez, .) as (3 nl x 3 nl) over (comp, l), comp (0, 1, 2) = (-, +, r)."""
        nl = self.nl
        Cm1 = self.C(-1); Cp1 = self.C(+1)
        Sp0 = self.Sp(0); Sm0 = self.Sm(0)
        Sm_p1 = self.Sm(+1); Sp_m1 = self.Sp(-1)
        i = 1j; s2 = np.sqrt(2.0)
        Z = np.zeros((nl, nl))
        return np.block([
            [-i * Cm1,          Z,                -(i / s2) * Sm0],
            [Z,                 +i * Cp1,         +(i / s2) * Sp0],
            [-(i / s2) * Sp_m1, +(i / s2) * Sm_p1, Z],
        ])

    def mult_matrix(self, gfun, s_in, s_out, nquad=None):
        """Multiplication by g(theta) mapping spin s_in -> s_out by Gauss-Legendre
        quadrature (exact for polynomial g of the right parity)."""
        n = nquad or (2 * self.ell_max + 16)
        x, w = leggauss(n)
        th = np.arccos(x)
        g = np.asarray(gfun(th), dtype=complex)
        Yin = np.array([sYlm_theta(s_in, l, self.m, th) for l in self.ls])
        Yout = np.array([sYlm_theta(s_out, l, self.m, th) for l in self.ls])
        return 2 * np.pi * (Yout * w) @ (g * Yin).T

    def quadrature(self, nquad=None):
        """(theta, weight) Gauss-Legendre nodes; weights include sin(theta) dtheta."""
        n = nquad or (2 * self.ell_max + 16)
        x, w = leggauss(n)
        return np.arccos(x), w

    def validate_mult(self):
        """Quadrature multiplication operators against the closed-form recursions."""
        dev = 0.0
        for s in (-1, 0, 1):
            dev = max(dev, np.abs(self.mult_matrix(np.cos, s, s) - self.C(s)).max())
        for s in (-1, 0):
            dev = max(dev, np.abs(self.mult_matrix(np.sin, s, s + 1) - self.Sp(s)).max())
        for s in (0, 1):
            dev = max(dev, np.abs(self.mult_matrix(np.sin, s, s - 1) - self.Sm(s)).max())
        return dev

    def validate(self, verbose=False):
        """Exact identities and pointwise synthesis; returns the maximum deviation."""
        dev = 0.0
        for s in (-1, 0, 1):
            dev = max(dev, np.abs(self.C(s) - self.C(s).T).max())
        for s in (0, 1):
            dev = max(dev, np.abs(self.Sm(s) - self.Sp(s - 1).T).max())
        for s in (-1, 0, 1):
            R = self.C(s) @ self.C(s) + self.Sp(s - 1) @ self.Sm(s)
            interior = slice(max(abs(s) - self.m, 0), self.nl - 1)
            dev = max(dev, np.abs((R - np.eye(self.nl))[interior, interior]).max())
        x, w = leggauss(2 * self.ell_max + 8)
        th = np.arccos(x)
        for s in (-1, 0, 1):
            for j, l in enumerate(self.ls[:-1]):
                Y = sYlm_theta(s, l, self.m, th)
                col = self.C(s)[:, j]
                rhs = sum(col[i2] * sYlm_theta(s, self.ls[i2], self.m, th)
                          for i2 in range(self.nl) if abs(col[i2]) > 1e-15)
                dev = max(dev, np.abs(np.cos(th) * Y - rhs).max())
                if s < 1:
                    col = self.Sp(s)[:, j]
                    rhs = sum(col[i2] * sYlm_theta(s + 1, self.ls[i2], self.m, th)
                              for i2 in range(self.nl) if abs(col[i2]) > 1e-15)
                    dev = max(dev, np.abs(np.sin(th) * Y - rhs).max())
                if s > -1:
                    col = self.Sm(s)[:, j]
                    rhs = sum(col[i2] * sYlm_theta(s - 1, self.ls[i2], self.m, th)
                              for i2 in range(self.nl) if abs(col[i2]) > 1e-15)
                    dev = max(dev, np.abs(np.sin(th) * Y - rhs).max())
        if verbose:
            print(f"AngularVasil.validate: max deviation = {dev:.2e}")
        return dev
