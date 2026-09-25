"""Closed-form eigenvalue benchmarks, one string problem per benchmark, run
on every core that compiles it.

Each benchmark is written once as a ``spdee.Problem`` from strings and is
handed to the four numerical cores -- the collocated fd2 core, the
staggered (Arakawa-C) core, the sh_fd core and the Dedalus core -- at two
resolutions.  The analytic value is derived in the test's docstring and in
docs/analytic_benchmarks.md; a core that cannot compile a problem asserts
its refusal instead of skipping, so the coverage matrix is part of the
suite rather than a comment.

Conventions.  Every core assembles the pencil ``K x = lambda M x`` with the
Dt terms in M and the rest in K, so the physical frequency is
``omega = -i lambda`` (``EvpResult.eigenvalues_to_omega``): a decay rate
gamma > 0 is ``lambda = +gamma`` and an oscillation of frequency omega is
``lambda = i omega``.  The azimuthal dependence is ``e^{i m phi}`` at fixed m.

Method.  Each (core, resolution) pair is solved once as a dense census
through the shift-invert resolvent (``spdee.solvers.dense``), and every
analytic target of that benchmark is read off the one census; ARPACK is not
used because the pencils carry large infinite-eigenvalue blocks on which its
restarts stagnate at some resolutions.  Resolutions are chosen so the census
stays under a few thousand unknowns.

Tolerances.  Every target is held to a tolerance at the finer resolution.
The order (4 for the collocated fd2 and sh_fd stencils used here, 2 for the
staggered C-grid) is measured for every target but asserted on the RMS error
over a benchmark's targets, and only above a core-dependent floor: below it
the radial, angular, wall and origin closures cancel against each other, the
eigenvalue error changes sign with resolution and a two-grid order is noise.
``pytest -s`` prints one ``BENCH`` row per (benchmark, core, target) that
fills the tables in the doc.  The module is 91 items and about 10 minutes on
three threads.
"""

from __future__ import annotations

import numpy as np
import importlib.util
import pytest
from scipy.optimize import brentq
from scipy.special import spherical_jn, spherical_yn

from spdee import Problem
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.sh_fd import ShFdGrid
from spdee.solvers import dense
from test_evp_zhang_2001 import (
    ZHANG_TARGETS_M1, ZHANG_TARGETS_M2, _zhang_pressure_basis, _polynomial_residual,
)

RI = 0.5
# |Im omega| above which a mode of a neutral (inviscid, ideal) problem is a
# spurious pressure mode of the collocated Grad/Div pair
NEUTRAL = 1e-6

# (coarse, fine) resolution per core; the fd2/staggered pair is (N_r, N_theta),
# sh_fd is (N_r, ell_max - m) and Dedalus is (N_r, N_theta)
_RES = {
    "fd2": ((12, 18), (18, 27)),
    "staggered": ((10, 16), (20, 32)),
    "sh_fd": ((24, 6), (48, 6)),
    "dedalus": ((10, 12), (14, 16)),
}
_RATIO = {"fd2": 1.5, "staggered": 2.0, "sh_fd": 2.0, "dedalus": 1.0}
_EXPECTED_ORDER = {"fd2": 4.0, "staggered": 2.0, "sh_fd": 4.0, "dedalus": None}
_ORDER_SLACK = {"fd2": 0.8, "staggered": 0.3, "sh_fd": 0.8, "dedalus": None}
# Below this relative error the radial, angular, wall and origin closures of a
# core cancel against each other -- the eigenvalue error changes sign with
# resolution -- and a two-grid order is not meaningful; the order is measured
# and printed everywhere but asserted only above the floor.
_ORDER_FLOOR = {"fd2": 3e-4, "staggered": 0.0, "sh_fd": 1e-5, "dedalus": None}
# relative tolerance of a diffusion decay rate at the finer resolution
_DIFFUSION_TOL = {"fd2": 2e-4, "staggered": 4e-3, "sh_fd": 5e-5, "dedalus": 1e-7}
CORES = tuple(_RES)


def _grid(core, geom, level, m, res=None):
    """The grid of one core at one resolution; ``geom`` is 'ball' or 'shell'."""
    a, b = (res or _RES)[core][level]
    ri = 0.0 if geom == "ball" else RI
    if core == "sh_fd":
        return ShFdGrid(N_r=a, ell_max=m + b, r_inner=ri, r_outer=1.0, fd_order=4)
    if core == "dedalus":
        from spdee.backends.dedalus import DedalusGrid
        return DedalusGrid(N_r=a, N_theta=b, r_inner=ri, r_outer=1.0)
    n_r = a + 1 if core == "staggered" else a
    if geom == "ball":
        return make_ball_grid(N_r=n_r, N_theta=b, r_outer=1.0)
    return make_shell_grid(N_r=n_r, N_theta=b, r_inner=ri, r_outer=1.0)


_HAVE_DEDALUS = importlib.util.find_spec("dedalus") is not None


def _compile(core, pb, geom, m, level, gauge=(), res=None):
    """Compile one string problem through one core."""
    if core == "dedalus" and not _HAVE_DEDALUS:
        pytest.skip("Dedalus is not installed")
    grid = _grid(core, geom, level, m, res)
    if core == "sh_fd":
        return pb.compile(grid, m=m, backend="sh_fd", gauge_unknowns=gauge)
    if core == "dedalus":
        return pb.compile(grid, m=m, backend="dedalus", gauge_unknowns=gauge)
    if core == "staggered":
        return pb.compile(grid, m=m, backend="fd2", gauge_unknowns=gauge,
                          discretization="staggered", order_r=2, order_theta=2)
    return pb.compile(grid, m=m, backend="fd2", gauge_unknowns=gauge,
                      order_r=4, order_theta=4)


def _census(core, pb, geom, m, level, sigma, gauge=(), res=None, resid=None,
            vectors=False):
    """The finite eigenvalues of one compiled pencil.

    ``resid`` gates the census on the pencil residual, which is how a
    saddle-point census is read; without it the eigenvectors are not computed.
    With ``vectors`` the eigenvectors (columns) and the compiled problem are
    returned as well, for identification by eigenfunction.
    """
    cp = _compile(core, pb, geom, m, level, gauge=gauge, res=res)
    K, M = cp.assemble_evp_matrices()
    out = dense(K, M, sigma=sigma, backend="auto", vectors=resid is not None or vectors,
                compute_resid=resid is not None, keep_vectors=vectors)
    keep = np.isfinite(out.lam)
    if resid is not None:
        keep &= out.resid < resid
    if vectors:
        return out.lam[keep], out.V[:, keep], cp
    return out.lam[keep]


def _spectra(core, pb, geom, m, sigma, gauge=(), res=None, resid=None):
    """{level: spectrum} for the two resolutions of one core."""
    return {lv: _census(core, pb, geom, m, lv, sigma, gauge=gauge, res=res, resid=resid)
            for lv in (0, 1)}


def _rate(lam, target):
    """The decay rate (real lambda) nearest ``target``."""
    return float(lam[np.argmin(np.abs(lam - target))].real)


def _omega(lam, target, neutral=None):
    """The frequency (omega = -i lambda) whose real part is nearest ``target``.

    ``neutral`` keeps only modes with |Im omega| below that bound, which is how
    the collocated core's non-neutral spurious pressure modes are dropped.
    """
    om = -1j * lam
    if neutral is not None:
        om = om[np.abs(om.imag) < neutral]
    return complex(om[np.argmin(np.abs(om.real - target))])


def _order(e_coarse, e_fine, ratio):
    if ratio <= 1.0 or e_fine <= 0.0 or e_coarse <= 0.0:
        return float("inf")
    return float(np.log(e_coarse / e_fine) / np.log(ratio))


def _row(bench, core, res, analytic, computed, errs, order):
    print(f"BENCH | {bench:36s} | {core:9s} | {res:7s} | analytic {analytic:+.8f} "
          f"| computed {computed:+.8f} | err {errs[0]:.2e} {errs[1]:.2e} "
          f"| order {order:.2f}")


def _run(bench, core, targets, spectra, pick, tol, scale=None, res=None):
    """Read every target off the two censuses and assert the value.

    ``scale`` normalizes the error by a fixed frequency scale instead of the
    target itself, which is how a near-zero target is measured.  The order is
    printed per target and asserted by ``_rms_order``.  Returns the
    (n_targets, 2) array of errors.
    """
    label = "x".join(str(n) for n in (res or _RES)[core][1])
    out = []
    for target in targets:
        vals = [pick(spectra[lv], target) for lv in (0, 1)]
        errs = [abs(v - target) / (scale or abs(target)) for v in vals]
        order = _order(errs[0], errs[1], _RATIO[core])
        _row(bench, core, label, target, vals[1], errs, order)
        assert errs[1] < tol, f"{bench} {core}: {target} -> {vals[1]} (rel {errs[1]:.2e})"
        out.append(errs)
    return np.array(out)


def _rms_order(bench, core, errs, min_order=None, floor=None, res=None):
    """The order of the RMS error over a benchmark's targets.

    Asserted instead of a per-target order wherever individual targets straddle
    the sign change of the eigenvalue error; skipped below ``floor``.
    """
    if _EXPECTED_ORDER[core] is None:
        return
    if min_order is None:
        min_order = _min_order(core)
    if floor is None:
        floor = _ORDER_FLOOR[core]
    rms = np.sqrt((errs ** 2).mean(axis=0))
    if rms[0] < floor:
        min_order = None
    order = _order(rms[0], rms[1], _RATIO[core])
    label = "x".join(str(n) for n in (res or _RES)[core][1])
    print(f"BENCH | {bench:36s} | {core:9s} | {label:7s} | rms error "
          f"{rms[0]:.2e} -> {rms[1]:.2e} | order {order:.2f}")
    if min_order is not None:
        assert order > min_order, f"{bench} {core}: rms order {order:.2f} {rms}"


def _zhang_omegas(targets):
    """Zhang's tabulated frequencies in the pencil convention.

    Zhang et al. write the modes as e^{i(m phi + 2 sigma t)} and tabulate
    omega_Z = 2 sigma; the pencil convention here is e^{i m phi - i omega t},
    so omega = -omega_Z.  The sectoral Rossby mode fixes the sign the same
    way: it sits at omega = -2/(m+1) on every core.
    """
    return [-w for _, w, _ in targets]


def _min_order(core):
    """The order the resolution pair must show, or None for a spectral core."""
    p = _EXPECTED_ORDER[core]
    return None if p is None else p - _ORDER_SLACK[core]


def _refuse(core, pb, geom, m, match, gauge=()):
    """Assert that a core refuses a problem, with the reason it gives.

    Some refusals land when the boundary rows are installed rather than when
    the operators are walked, so the pencil is assembled too.
    """
    with pytest.raises((NotImplementedError, ValueError), match=match):
        _compile(core, pb, geom, m, 0, gauge=gauge).assemble_evp_matrices()


# ---------------------------------------------------------------------------
# Radial eigenvalue conditions
# ---------------------------------------------------------------------------

def _wall_value(f, l, x, kind):
    """The wall functional of a radial solution ``f(l, x)``, x = k r.

    'dirichlet'   f = 0            (T or S vanishes at the wall)
    'neumann'     f' = 0           (radial derivative vanishes)
    'rT'          (x f)' = 0       (d(r T)/dr = 0)
    'T_over_r'    x f' - f = 0     (d(T/r)/dr = 0)
    """
    if kind == "dirichlet":
        return f(l, x)
    if kind == "neumann":
        return f(l, x, derivative=True)
    if kind == "rT":
        return f(l, x) + x * f(l, x, derivative=True)
    if kind == "T_over_r":
        return x * f(l, x, derivative=True) - f(l, x)
    raise ValueError(kind)


def _radial_det(kind, l, k, ri, ro):
    """The characteristic determinant of the radial problem in [ri, ro]."""
    if ri == 0.0:
        return _wall_value(spherical_jn, l, k * ro, kind)
    return (_wall_value(spherical_jn, l, k * ri, kind)
            * _wall_value(spherical_yn, l, k * ro, kind)
            - _wall_value(spherical_jn, l, k * ro, kind)
            * _wall_value(spherical_yn, l, k * ri, kind))


def _radial_roots(kind, l, n=1, ri=0.0, ro=1.0, kmax=60.0):
    """The n gravest positive k of ``_radial_det``."""
    ks = np.linspace(1e-3, kmax, 12000)
    v = np.array([_radial_det(kind, l, k, ri, ro) for k in ks])
    out = []
    for i in range(len(ks) - 1):
        if np.isfinite(v[i]) and np.isfinite(v[i + 1]) and v[i] * v[i + 1] < 0:
            out.append(brentq(lambda x: _radial_det(kind, l, x, ri, ro),
                              ks[i], ks[i + 1], xtol=1e-14))
            if len(out) == n:
                break
    return np.array(out)


# ---------------------------------------------------------------------------
# 1. Acoustic modes of a uniform gas sphere and shell
# ---------------------------------------------------------------------------

def acoustic_problem(cs, geometry):
    """Linearised Euler about a uniform gas at rest with rigid walls."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("P", "scalar")
    pb.add_parameter("cs2", cs ** 2, symbolic=False)
    pb.add_equation("dt(v) + grad(P) = 0", name="momentum")
    pb.add_equation("dt(P) + cs2*div(v) = 0", name="pressure")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    if geometry == "shell":
        pb.add_bc("dot(r_hat, v) = 0", at="inner")
    return pb


_ACOUSTIC_TOL = {"fd2": 3e-4, "staggered": 4e-3, "sh_fd": 2e-5, "dedalus": 1e-8}
_ACOUSTIC_CS = 2.0


@pytest.mark.parametrize("core", CORES)
def test_acoustic_ball(core):
    """Uniform gas ball, rigid wall, m = 1.

    With rho0 and c constant and P = p1/rho0 the linearised Euler pair is
    dt(v) = -Grad(P), dt(P) = -c^2 Div(v); eliminating v for
    v ~ e^{-i omega t} gives Lap(P) = -(omega/c)^2 P, and the rigid wall
    Dot(r_hat, v) = 0 is Neumann on P.  Regularity at the origin picks
    P = j_l(k r) Y_lm, so omega = c k with j_l'(k) = 0: k = 2.081576 (l = 1)
    and 3.342094 (l = 2).  The pair exercises Grad, Div and the
    pressure-velocity coupling with no rotation.
    """
    targets = [_ACOUSTIC_CS * _radial_roots("neumann", l, 1)[0] for l in (1, 2)]
    spectra = _spectra(core, acoustic_problem(_ACOUSTIC_CS, "ball"), "ball", 1,
                       sigma=1j * np.mean(targets) + 0.2)
    errs = _run("acoustic ball m=1", core, targets, spectra,
                lambda lam, t: _omega(lam, t).real, _ACOUSTIC_TOL[core])
    _rms_order("acoustic ball m=1", core, errs)


@pytest.mark.parametrize("core", CORES)
def test_acoustic_shell(core):
    """Uniform gas shell 0.5 <= r <= 1, rigid walls, m = 0.

    The same Helmholtz problem with Neumann walls at both radii, so
    P = a j_l(k r) + b y_l(k r) and omega = c k with
    j_l'(k/2) y_l'(k) - j_l'(k) y_l'(k/2) = 0: k = 1.840268 (l = 1),
    3.151176 (l = 2) and 6.572013 (l = 0, the gravest purely radial p-mode,
    near the half-wavelength estimate pi/(1 - 0.5) = 6.283).
    """
    targets = [_radial_roots("neumann", l, 1, ri=RI)[0] for l in (1, 2, 0)]
    spectra = _spectra(core, acoustic_problem(1.0, "shell"), "shell", 0,
                       sigma=1j * 3.0 + 0.2)
    errs = _run("acoustic shell m=0", core, targets, spectra,
                lambda lam, t: _omega(lam, t).real, _ACOUSTIC_TOL[core])
    _rms_order("acoustic shell m=0", core, errs)


# ---------------------------------------------------------------------------
# 2. Doppler shift under solid-body rotation
# ---------------------------------------------------------------------------

def _solid_body(pb, rate):
    """The solid-body flow U0 = rate * s e_phi (s = r sin theta)."""
    return pb.add_background("U0", v_r=0, v_theta=0, v_phi=f"{rate!r}*r*sin(theta)")


def doppler_scalar_problem(rate, kappa=0.05):
    """Advection-diffusion of a passive scalar by a solid-body flow."""
    pb = Problem()
    pb.add_unknown("s", "scalar")
    _solid_body(pb, rate)
    pb.add_parameter("kap", kappa, symbolic=False)
    pb.add_equation("dt(s) + dot(U0, grad(s)) - kap*lap(s) = 0", name="heat")
    pb.add_bc("s = 0", at="outer")
    return pb


def doppler_inertial_problem(rate, Omega0=1.0):
    """Inertial modes in a frame rotating at Omega0 - rate, so that the fluid
    still rotates at Omega0 and carries the mean flow U0 = rate * s e_phi.

    For rigid rotation U0 = rate * (z_hat x r) the two mean-flow terms of the
    linearised momentum equation are U0.Grad(v) + v.Grad(U0), and
    v.Grad(U0) = rate * z_hat x v exactly, so it is spelled as a Cross; the
    Coriolis coefficient of the new frame is 2*(Omega0 - rate).  The advection
    operator U0.Grad(v) is left in its general form: it is what the benchmark
    tests.
    """
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    _solid_body(pb, rate)
    pb.add_parameter("coriolis", 2.0 * Omega0 - rate, symbolic=False)
    pb.add_equation("dt(v) + dot(U0, grad(v)) + coriolis*cross(ez, v) + grad(p) = 0",
                    name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


_DOPPLER_RATES = (0.2, 0.5)
_SHIFT_TOL = 1e-10


def _shift_row(bench, core, level, target, got, tol):
    res = "x".join(str(n) for n in _RES[core][level])
    print(f"BENCH | {bench:36s} | {core:9s} | {res:7s} | shift {target:+.8f} "
          f"| computed {got.real:+.10f}{got.imag:+.2e}i | err {abs(got - target):.2e}")
    assert abs(got - target) < tol, f"{bench} {core} {res}: shift {got} != {target}"


@pytest.mark.parametrize("core", CORES)
def test_doppler_shift_scalar(core):
    """Passive scalar in a ball at m = 2: every eigenvalue shifts by m*rate.

    dt(s) + U0.Grad(s) = kappa Lap(s) with U0 = rate * s e_phi is
    dt(s) + i m rate s = kappa Lap(s) at fixed m, because U0.Grad = rate d/dphi
    on a scalar exactly.  With s = 0 at r = 1 the spectrum is
    lambda = kappa k^2 + i m rate over j_l(k) = 0, l >= m, so
    omega = m rate - i kappa k^2: the decay rates are those of the pure
    diffusion problem, every frequency is shifted by exactly m*rate and the
    eigenvectors are unchanged.  Two nonzero rates are compared, so that the
    advection operator is present in both censuses (an identically zero U0
    drops the term on every core); the shift is m*(rate2 - rate1) and
    carries no discretisation error, so it is asserted to rounding on every
    core.
    """
    m, kappa = 2, 0.05
    r1, r2 = _DOPPLER_RATES
    rates = [kappa * _radial_roots("dirichlet", l, 1)[0] ** 2 for l in (2, 3)]
    sig = float(np.mean(rates))
    slow = _spectra(core, doppler_scalar_problem(r1, kappa), "ball", m,
                    sigma=sig + 1j * m * r1)
    fast = _spectra(core, doppler_scalar_problem(r2, kappa), "ball", m,
                    sigma=sig + 1j * m * r2)
    errs = _run("doppler scalar decay m=2", core, rates, slow,
                lambda lam, t: _rate(lam, t + 1j * m * r1), _DIFFUSION_TOL[core])
    _rms_order("doppler scalar decay m=2", core, errs)
    for level in (0, 1):
        for gamma in rates:
            lam1 = slow[level][np.argmin(np.abs(slow[level] - gamma - 1j * m * r1))]
            lam2 = fast[level][np.argmin(np.abs(fast[level] - gamma - 1j * m * r2))]
            _shift_row("doppler scalar shift m=2", core, level, m * (r2 - r1),
                       -1j * (lam2 - lam1), _SHIFT_TOL)


@pytest.mark.parametrize("core", CORES)
def test_doppler_shift_inertial(core):
    """Inertial modes of the ball at m = 1 seen from two rotating frames:
    every frequency shifts by exactly m times the difference of the rates.

    For U0 = rate * (z_hat x r) and a vector field v,
    U0.Grad(v) = rate (d/dphi v + z_hat x v) and v.Grad(U0) = rate z_hat x v,
    so the mean-flow and Coriolis terms of the frame rotating at Omega0 - rate
    collapse to i m rate v + 2 Omega0 z_hat x v: the inertial-mode operator of
    the co-rotating frame plus i m rate.  Hence omega(rate) = omega_i + m rate
    with the eigenvector unchanged, where omega_i is the Zhang et al. (2001)
    frequency.  Both the absolute value omega(rate) - m rate = omega_i and the
    exact shift are asserted; the shift compares the same core at the same
    resolution and so is free of discretisation error, which makes it a
    rounding-level check of the advection operator against the Coriolis pair.
    """
    m = 1
    r1, r2 = _DOPPLER_RATES
    targets = _zhang_omegas(ZHANG_TARGETS_M1[2:5])
    slow = _spectra(core, doppler_inertial_problem(r1), "ball", m,
                    sigma=1j * (0.3 + m * r1) + 0.02, gauge=("p",))
    fast = _spectra(core, doppler_inertial_problem(r2), "ball", m,
                    sigma=1j * (0.3 + m * r2) + 0.02, gauge=("p",))
    _run("doppler inertial omega m=1", core, targets, slow,
         lambda lam, t: _omega(lam, t + m * r1, neutral=NEUTRAL).real - m * r1,
         _INERTIAL_TOL[core])
    for level in (0, 1):
        for omega in targets:
            w1 = _omega(slow[level], omega + m * r1, neutral=NEUTRAL)
            w2 = _omega(fast[level], omega + m * r2, neutral=NEUTRAL)
            _shift_row("doppler inertial shift m=1", core, level, m * (r2 - r1),
                       w2 - w1, _SHIFT_TOL)


# ---------------------------------------------------------------------------
# 3. Inertial modes of a full sphere (Greenspan; Zhang et al. 2001)
# ---------------------------------------------------------------------------

def inertial_problem():
    """Inviscid inertial modes of a rotating full sphere, impenetrable wall."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


_INVISCID_RESID = 1e-8
_INERTIAL_TOL = {"fd2": 8e-3, "staggered": 2e-2, "sh_fd": 1e-4, "dedalus": 1e-4}
# the curated targets are tabulated to five decimals, so an RMS error at this
# level is the tabulation, not the discretisation
_ZHANG_FLOOR = 3e-5
_INERTIAL_ORDER = {"fd2": 2.5, "staggered": 1.3, "sh_fd": 1.5, "dedalus": None}
# At m = 2 some collocated targets collide with a band neighbour into a
# complex pair (docs/analytic_benchmarks.md, finding 4), so there every target
# is read off the eigenvector of its Zhang class rather than off the nearest
# neutral eigenvalue, on the pair of censuses where the pressure-polynomial
# identification is unambiguous.
_RES_FD2_M2 = {"fd2": ((18, 27), (27, 40))}
_M2_CLASS = {-w: ("SYM" if "SYM" in label else "ANTI", int(label[2]))
             for label, w, _ in ZHANG_TARGETS_M2}
_IDENT_WINDOW = 0.05
# the pressure of the identified eigenvector lies in its Zhang polynomial
# space to this relative residual at the finer resolution
_IDENT_RESID = 0.2
# A core discriminates individual inertial modes only if it matches the curated
# targets much better than it matches random frequencies of the same band; the
# in-band spectrum of an inviscid ball is dense, so a nearest match on its own
# is not evidence.  None: no ratio is asserted (see the docstring).
_DISCRIMINATION = {"fd2": 4.0, "staggered": None, "sh_fd": 100.0, "dedalus": 100.0}
_CONTROLS = np.random.default_rng(3).uniform(-1.9, 1.9, 40)


def _discrimination(band, targets):
    """Median nearest-match error of the targets and of the control set."""
    hit = np.median([np.abs(band - t).min() for t in targets])
    miss = np.median([np.abs(band - c).min() for c in _CONTROLS])
    return float(hit), float(miss)


def _identify_by_pressure(lam, V, cp, m, target, parity, n_class):
    """The mode within ``_IDENT_WINDOW`` of ``target`` whose pressure lies
    closest to the Zhang (N, m, parity) polynomial space; returns
    (omega, residual).  Neutrality is not required: a colliding target is a
    complex pair whose real part is the frequency."""
    om = -1j * lam
    R, T = cp.matrices.grid.mesh()
    weight = (R ** 2 * np.sin(T)).ravel() + 1e-3
    basis = _zhang_pressure_basis(R, T, m, n_class, parity)
    cand = np.nonzero(np.abs(om.real - target) < _IDENT_WINDOW)[0]
    fits = [(_polynomial_residual(cp.matrices.layout.unpack(V[:, i], role="primary")["p"],
                                  basis, weight), i) for i in cand]
    resid, i = min(fits)
    return complex(om[i]), float(resid)


@pytest.mark.parametrize("m,targets", [(1, ZHANG_TARGETS_M1), (2, ZHANG_TARGETS_M2)],
                         ids=["m1", "m2"])
@pytest.mark.parametrize("core", CORES)
def test_inertial_modes_zhang_2001(core, m, targets):
    """The curated Zhang et al. (2001) N = 1, 2 targets of
    tests/test_evp_zhang_2001.py on every core.

    2 z_hat x v + Grad(p) = -dt(v), Div(v) = 0, Dot(r_hat, v) = 0 at r = 1 is
    Poincare's problem, whose eigenfunctions in a full sphere are polynomials
    in the cylindrical coordinates; Zhang et al. give the polynomial
    eigenvalue equations (their 2.15 and 2.16) whose roots the curated list
    holds.  Modes are neutral (Im omega = 0), so the collocated core's
    non-neutral spurious pressure modes are dropped by the NEUTRAL gate and
    the saddle-point censuses are read through the residual gate.

    Poincare's operator is not compact: its inviscid spectrum fills
    [-2, 2] densely, and a discretisation fills the band with as many modes
    as it has degrees of freedom.  A nearest match therefore proves nothing
    unless it is much sharper than the mean spacing, so each census is also
    matched against 40 random control frequencies of the same band and the
    ratio is asserted.  The staggered core carries no ratio: its in-band
    spectrum is dense enough that random controls are matched as well as the
    curated targets (see docs/analytic_benchmarks.md), so only the RMS error
    and its second-order decrease are asserted there.

    At m = 2 the collocated core's non-adjoint Grad/Div pair lets a target
    collide with a band neighbour into a complex pair at some resolutions, so
    the nearest neutral eigenvalue is then a neighbour at the local spacing.
    There every target is identified by its eigenvector instead: the mode
    within a window whose pressure lies closest to the polynomial space of
    its Zhang class (the projection of tests/test_evp_zhang_2001.py), on the
    18x27 / 27x40 pair where that identification is unambiguous.
    """
    tgts = _zhang_omegas(targets)
    bench = f"inertial zhang m={m}"
    identify = core == "fd2" and m == 2
    res = _RES_FD2_M2 if identify else None
    if identify:
        spectra, found = {}, {}
        for level in (0, 1):
            lam, V, cp = _census(core, inertial_problem(), "ball", m, level, 0.3j + 0.02,
                                 gauge=("p",), res=res, resid=_INVISCID_RESID, vectors=True)
            spectra[level] = lam
            found[level] = {t: _identify_by_pressure(lam, V, cp, m, t, *_M2_CLASS[t])
                            for t in tgts}
            for t, (w, fit) in found[level].items():
                print(f"BENCH | {bench} identified         | {core:9s} | "
                      f"{'x'.join(str(n) for n in res[core][level]):7s} | target {t:+.5f} "
                      f"| omega {w:+.6f} | polynomial residual {fit:.2e}")
        assert all(fit < _IDENT_RESID for _, fit in found[1].values()), found[1]
        errs = _run(bench, core, tgts, found, lambda f, t: f[t][0].real,
                    _INERTIAL_TOL[core], res=res)
    else:
        spectra = _spectra(core, inertial_problem(), "ball", m, sigma=0.3j + 0.02,
                           gauge=("p",), resid=_INVISCID_RESID)
        errs = _run(bench, core, tgts, spectra,
                    lambda lam, t: _omega(lam, t, neutral=NEUTRAL).real,
                    _INERTIAL_TOL[core])
    for level in (0, 1):
        om = -1j * spectra[level]
        band = om[np.abs(om.imag) < NEUTRAL].real
        band = band[np.abs(band) < 2.0]
        hit, miss = _discrimination(band, tgts)
        label = "x".join(str(n) for n in (res or _RES)[core][level])
        print(f"BENCH | {bench} spacing        | {core:9s} | {label:7s} | "
              f"{band.size:5d} modes in band | target {hit:.2e} | control {miss:.2e} "
              f"| ratio {miss / hit:.1f}")
        if _DISCRIMINATION[core] is not None and level == 1:
            assert miss / hit > _DISCRIMINATION[core], (
                f"{bench} {core} {label}: the curated targets are matched "
                f"{hit:.2e} and random controls {miss:.2e}; no discrimination")
    _rms_order(bench, core, errs, _INERTIAL_ORDER[core], floor=_ZHANG_FLOOR, res=res)


# ---------------------------------------------------------------------------
# 4. Free magnetic decay in a perfectly conducting shell
# ---------------------------------------------------------------------------

def decay_problem(formulation, eta=1.0):
    """Free decay of a magnetic field in a perfectly conducting shell.

    'b' evolves the field itself, with Div(b) = 0 written as an involution
    (no multiplier) and the conducting wall b_r = 0, tangential Curl(b) = 0.
    'A' evolves the vector potential with the Coulomb gauge as a saddle point
    (multiplier psi), tangential A = 0 and psi = 0 at the walls; b = Curl(A)
    then satisfies the same conducting conditions and obeys the same decay
    equation, so the two formulations share the analytic rates.
    """
    pb = Problem()
    pb.add_parameter("eta", eta, symbolic=False)
    if formulation == "A":
        pb.add_unknown("A", "vector")
        pb.add_unknown("psi", "scalar")
        pb.add_equation("dt(A) - eta*lap(A) - grad(psi) = 0", name="induction")
        pb.add_equation("div(A) = 0", name="coulomb")
        for wall in ("inner", "outer"):
            pb.add_bc("dot(theta_hat, A) = 0", at=wall)
            pb.add_bc("dot(phi_hat, A) = 0", at=wall)
            pb.add_bc("psi = 0", at=wall)
        return pb
    pb.add_unknown("b", "vector")
    pb.add_equation("dt(b) - eta*lap(b) = 0", name="induction")
    if formulation == "b":
        pb.add_equation("div(b) = 0", name="solenoidal")
    for wall in ("inner", "outer"):
        pb.add_bc("dot(r_hat, b) = 0", at=wall)
        pb.add_bc("dot(theta_hat, curl(b)) = 0", at=wall)
        pb.add_bc("dot(phi_hat, curl(b)) = 0", at=wall)
    return pb


def _decay_targets():
    """Free-decay rates k^2 of the conducting shell 0.5 <= r <= 1 at m = 0.

    With b = Curl(Curl(S r)) + Curl(T r) both potentials obey dt X = Lap X, so
    X = a j_l(k r) + b y_l(k r) and the rate is k^2.  b_r = l(l+1) S / r^2
    gives S = 0 at both walls for the poloidal family (which also kills the
    toroidal part of the wall electric field, proportional to Lap S = -k^2 S),
    while the poloidal part of the electric field is proportional to d(r T)/dr,
    so the toroidal family closes on d(r T)/dr = 0 at both walls.
    """
    tor = _radial_roots("rT", 1, 2, ri=RI)
    pol1 = _radial_roots("dirichlet", 1, 1, ri=RI)
    pol2 = _radial_roots("dirichlet", 2, 1, ri=RI)
    return [tor[0] ** 2, pol1[0] ** 2, pol2[0] ** 2, tor[1] ** 2]


_DECAY_TOL = {"fd2": 1e-2, "staggered": 1.5e-2, "sh_fd": 5e-4, "dedalus": 1e-6}
# the gravest decay rates sit at k ~ 7, which a spectral shell needs more radial
# modes to resolve than the frequencies of the other benchmarks
_DECAY_RES = {**_RES, "dedalus": ((14, 12), (20, 16))}
# the reason each core gives for refusing a formulation
_DECAY_REFUSAL = {
    ("b_free", "staggered"): "not a flux unknown",
}


@pytest.mark.parametrize("formulation", ["b", "A", "b_free"])
@pytest.mark.parametrize("core", CORES)
def test_magnetic_free_decay_shell(core, formulation):
    """Free decay in a perfectly conducting shell at m = 0, where the toroidal
    and poloidal families decouple, against the Bessel conditions of
    ``_decay_targets``: toroidal l = 1 (k = 1.984570 and 6.618465, from
    d(r T)/dr = 0) and poloidal l = 1, 2 (k = 6.572013 and 7.111576, from
    S = 0).

    Three spellings of the same physics.  'b' writes Div(b) = 0 with no
    multiplier: the staggered core reads it as an involution (constrained
    transport) and the other three clean the divergence through an
    auto-introduced multiplier, the saddle point the shared classifier
    prescribes.  'A' is the vector potential with the Coulomb gauge as a
    saddle point.  'b_free' drops the constraint and closes the vector
    Laplacian with the conducting wall alone through the tangential Curl(b)
    rows; the staggered core reads a tangential electric field only on a
    flux unknown and refuses it.
    """
    if (formulation, core) in _DECAY_REFUSAL:
        _refuse(core, decay_problem(formulation), "shell", 0,
                _DECAY_REFUSAL[(formulation, core)])
        return
    targets = _decay_targets()
    spectra = _spectra(core, decay_problem(formulation), "shell", 0,
                       sigma=float(np.mean(targets)), res=_DECAY_RES)
    errs = _run(f"free decay {formulation} shell m=0", core, targets, spectra, _rate,
                _DECAY_TOL[core], res=_DECAY_RES)
    _rms_order(f"free decay {formulation} shell m=0", core, errs, res=_DECAY_RES)


# ---------------------------------------------------------------------------
# 5. Malkus field hydromagnetic inertial modes
# ---------------------------------------------------------------------------

def malkus_problem(Le):
    """Ideal MHD on the Malkus field B0 = s e_phi in a rotating full sphere.

    The induction equation makes b a curl, hence solenoidal for omega != 0,
    so no Div(b) = 0 row and no wall condition on b is needed; the velocity
    keeps the impenetrable wall of the inertial-mode problem, which is also
    the vanishing tangential electric field of a perfectly conducting sphere
    because Cross(v, B0) has the tangential components (-s v_r, 0).
    """
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("b", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_background("ez", v_r="cos(theta)", v_theta="-sin(theta)", v_phi=0)
    pb.add_background("B0", v_r=0, v_theta=0, v_phi="r*sin(theta)")
    pb.add_parameter("Le2", Le ** 2, symbolic=False)
    pb.add_equation("dt(v) + 2*cross(ez, v) + grad(p) - Le2*cross(curl(b), B0) = 0",
                    name="momentum")
    pb.add_equation("dt(b) - curl(cross(v, B0)) = 0", name="induction")
    pb.add_equation("div(v) = 0", name="continuity")
    pb.add_bc("dot(r_hat, v) = 0", at="outer")
    return pb


def malkus_branches(omega_i, Le, m):
    """The two magneto-Coriolis frequencies grown from one inertial mode.

    For B0 = s e_phi = z_hat x r and a solenoidal v, Curl(Cross(v, B0)) =
    (B0.Grad)v - (v.Grad)B0 = (d/dphi v + z_hat x v) - z_hat x v = i m v, so
    the induction equation gives b = -(m/omega) v exactly.  The Lorentz force
    is Cross(Curl(b), B0) = i m b - Grad(b.B0) by the same two identities plus
    Curl(B0) = 2 z_hat, so the momentum equation becomes
    -i (omega - Le^2 m^2 / omega) v + 2 z_hat x v = -Grad(P): Poincare's
    operator with the eigenvalue omega_i = omega - Le^2 m^2 / omega and the
    inertial eigenvector unchanged.  Hence

        omega^2 - omega_i omega - Le^2 m^2 = 0,

    the fast branch omega+ -> omega_i and the slow branch
    omega- -> -Le^2 m^2 / omega_i as Le -> 0, with omega+ + omega- = omega_i
    and omega+ omega- = -Le^2 m^2 exactly.  This is the classical result for
    the Malkus field (Malkus 1967); the derivation above is self-contained and
    is what the test asserts, and the branch is checked independently by the
    sum and product relations and by the Le -> 0 limit.
    """
    disc = np.sqrt(omega_i ** 2 + 4.0 * Le ** 2 * m ** 2)
    return 0.5 * (omega_i + disc), 0.5 * (omega_i - disc)


# measured against the rotation frequency 2 Omega = 2, not against the target,
# because the slow branch sits near zero; the staggered value is set from the
# measured 20x32 errors (up to 2.8e-3 on the fast branch of the omega = 1.496
# inertial mode, the inertial error itself, at RMS order 2.2 to 2.3)
_MALKUS_TOL = {"fd2": 5e-3, "staggered": 5e-3, "sh_fd": 5e-5, "dedalus": 5e-5}


@pytest.mark.parametrize("Le", [0.1, 0.3], ids=["Le0.1", "Le0.3"])
@pytest.mark.parametrize("core", CORES)
def test_malkus_hydromagnetic_inertial_modes(core, Le):
    """Magneto-Coriolis modes of the Malkus field in a full sphere at m = 2.

    Every inertial frequency omega_i of the same sphere splits into the two
    roots of omega^2 - omega_i omega - Le^2 m^2 = 0 (``malkus_branches``); the
    inertial frequencies used are the exact Zhang et al. (2001) values, so the
    prediction is closed form.  Two independent checks of the branch: their
    sum is omega_i and their product is -Le^2 m^2, both asserted on the
    computed pair, and the fast branch approaches omega_i as Le falls.  The
    staggered core reads b as an involution unknown (constrained transport
    with the origin ring of the fluxes pinned, docs/ball_staggered.md section
    7) and, b carrying no wall condition, keeps its wall fluxes as DOFs
    evolving with the wall circulations of Cross(v, B0), which vanish on the
    Malkus field with v_r = 0; its product relation holds to 1e-6 because
    the discrete Curl(Cross(v, B0)) is i m v on the C-grid too.
    """
    m = 2
    inertial = _zhang_omegas(ZHANG_TARGETS_M2[2:5])
    targets = [w for wi in inertial for w in malkus_branches(wi, Le, m)]
    spectra = _spectra(core, malkus_problem(Le), "ball", m, sigma=0.5j + 0.02,
                       gauge=("p",), resid=_INVISCID_RESID)
    _run(f"malkus m=2 Le={Le}", core, targets, spectra,
         lambda lam, t: _omega(lam, t, neutral=NEUTRAL).real, _MALKUS_TOL[core],
         scale=1.0)
    for wi in inertial:
        fast, slow = malkus_branches(wi, Le, m)
        got = [_omega(spectra[1], w, neutral=NEUTRAL).real for w in (fast, slow)]
        res = "x".join(str(n) for n in _RES[core][1])
        print(f"BENCH | malkus branch m=2 Le={Le}          | {core:9s} | {res:7s} | "
              f"sum {got[0] + got[1]:+.6f} vs {wi:+.6f} | product "
              f"{got[0] * got[1]:+.6f} vs {-Le ** 2 * m ** 2:+.6f}")
        assert abs(got[0] + got[1] - wi) < 4 * _MALKUS_TOL[core]
        assert abs(got[0] * got[1] + Le ** 2 * m ** 2) < 4 * _MALKUS_TOL[core]


# ---------------------------------------------------------------------------
# 6. Gravity modes of a constant-N Boussinesq shell
# ---------------------------------------------------------------------------

def gravity_problem(N2=1.0):
    """Boussinesq gravity modes with a constant Brunt-Vaisala frequency."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("bu", "scalar")
    pb.add_unknown("p", "scalar")
    pb.add_background("er", v_r=1, v_theta=0, v_phi=0)
    pb.add_parameter("N2", N2, symbolic=False)
    pb.add_equation("dt(v) + grad(p) - bu*er = 0", name="momentum")
    pb.add_equation("dt(bu) + N2*dot(er, v) = 0", name="buoyancy")
    pb.add_equation("div(v) = 0", name="continuity")
    for wall in ("inner", "outer"):
        pb.add_bc("dot(r_hat, v) = 0", at=wall)
    return pb


def gravity_omega(l, n, N=1.0, ri=RI, ro=1.0):
    """The exact constant-N Boussinesq g-mode frequency of a shell.

    Eliminating v and the buoyancy bu from dt(v) = -Grad(p) + bu e_r,
    dt(bu) = -N^2 v_r, Div(v) = 0 for e^{-i omega t} gives
    v_h = Grad_h(p)/(i omega) and v_r = alpha p'/(i omega) with
    alpha = omega^2/(omega^2 - N^2); continuity is then
    alpha (r^2 P')' = l(l+1) P, an Euler equation whose solutions are r^a with
    a(a+1) = l(l+1)/alpha.  Rigid walls make P' vanish at both radii, which
    for a1 != a2 needs (ri/ro)^(a1 - a2) = 1, i.e. a1 - a2 = i q_n with
    q_n = 2 pi n / ln(ro/ri), n = 1, 2, ...  Then
    1 + 4 l(l+1)/alpha = -q_n^2 and

        omega^2 = 4 l(l+1) N^2 / (1 + q_n^2 + 4 l(l+1)).

    The thin-shell limit checks it: ln(ro/ri) -> d/r makes q_n -> 2 k_r r with
    k_r = n pi / d, and the formula becomes the plane-parallel Boussinesq
    dispersion omega^2 = N^2 k_h^2 / (k_h^2 + k_r^2) with k_h^2 = l(l+1)/r^2.
    The same elimination on a ball has no discrete spectrum: a is complex with
    real part -1/2, so every solution oscillates without bound at the origin.
    """
    q = 2.0 * np.pi * n / np.log(ro / ri)
    L = l * (l + 1)
    return float(N * np.sqrt(4.0 * L / (1.0 + q ** 2 + 4.0 * L)))


# the n = 2 g-mode carries three radial wavelengths in 0.5 <= r <= 1 and is
# only marginally resolved by the collocated core at N_r = 18
_GRAVITY_TOL = {"fd2": 1e-2, "staggered": 1e-2, "sh_fd": 2e-4, "dedalus": 1e-7}


@pytest.mark.parametrize("core", CORES)
def test_gravity_modes_constant_N_shell(core):
    """Boussinesq g-modes of a constant-N shell 0.5 <= r <= 1 at m = 1:
    omega = 0.296181 (l = 1, n = 1), 0.473202 (l = 2, n = 1) and 0.153924
    (l = 1, n = 2), derived in ``gravity_omega``.  The buoyancy coupling is
    the pair bu*er and N2*dot(er, v), so the benchmark fixes the relative sign
    of the buoyancy force and the entropy advection as well as the frequency.
    """
    targets = [gravity_omega(1, 1), gravity_omega(2, 1), gravity_omega(1, 2)]
    spectra = _spectra(core, gravity_problem(), "shell", 1, sigma=0.3j + 0.02,
                       gauge=("p",), resid=_INVISCID_RESID)
    errs = _run("gravity constant-N shell m=1", core, targets, spectra,
                lambda lam, t: _omega(lam, t, neutral=NEUTRAL).real,
                _GRAVITY_TOL[core])
    _rms_order("gravity constant-N shell m=1", core, errs)


# ---------------------------------------------------------------------------
# 7. Diffusion sanity set on the shell
# ---------------------------------------------------------------------------

def scalar_diffusion_problem(wall):
    """Scalar diffusion with Dirichlet or Neumann walls."""
    pb = Problem()
    pb.add_unknown("s", "scalar")
    pb.add_equation("dt(s) - lap(s) = 0", name="heat")
    text = "s = 0" if wall == "dirichlet" else "dot(r_hat, grad(s)) = 0"
    for w in ("inner", "outer"):
        pb.add_bc(text, at=w)
    return pb


def vector_diffusion_problem(wall):
    """Incompressible vector diffusion with no-slip or stress-free walls."""
    pb = Problem()
    pb.add_unknown("v", "vector")
    pb.add_unknown("p", "scalar")
    pb.add_equation("dt(v) - lap(v) + grad(p) = 0", name="momentum")
    pb.add_equation("div(v) = 0", name="continuity")
    for w in ("inner", "outer"):
        if wall == "noslip":
            for unit in ("r_hat", "theta_hat", "phi_hat"):
                pb.add_bc(f"dot({unit}, v) = 0", at=w)
        else:
            pb.add_stress_free("v", at=w)
    return pb


@pytest.mark.parametrize("wall", ["dirichlet", "neumann"])
@pytest.mark.parametrize("m", [0, 1, 2])
@pytest.mark.parametrize("core", CORES)
def test_scalar_diffusion_shell(core, m, wall):
    """dt(s) = Lap(s) in the shell 0.5 <= r <= 1 at m = 0, 1, 2.

    Separating s = R(r) Y_lm gives R = a j_l(k r) + b y_l(k r) and the decay
    rate k^2; a Dirichlet wall closes on R = 0 and a Neumann wall on R' = 0 at
    both radii, so k solves the corresponding 2x2 determinant.  The ball cases
    of the same problem are in tests/test_ball_collocated.py,
    tests/test_dedalus_ball.py, tests/test_sh_fd_ball.py and
    tests/test_ball_staggered.py; this is the shell half of that set on all
    four cores at once.
    """
    ells = range(max(m, 0), max(m, 0) + 2)
    targets = [_radial_roots(wall, l, 1, ri=RI)[0] ** 2 for l in ells]
    spectra = _spectra(core, scalar_diffusion_problem(wall), "shell", m,
                       sigma=float(np.mean(targets)))
    errs = _run(f"scalar {wall} shell m={m}", core, targets, spectra, _rate,
                _DIFFUSION_TOL[core])
    _rms_order(f"scalar {wall} shell m={m}", core, errs)


_VECTOR_DIFFUSION_TOL = {"fd2": 3e-4, "staggered": 8e-3, "sh_fd": 1e-4,
                         "dedalus": 1e-7}


@pytest.mark.parametrize("wall", ["noslip", "stress_free"])
@pytest.mark.parametrize("m", [1, 2])
@pytest.mark.parametrize("core", CORES)
def test_toroidal_diffusion_shell(core, m, wall):
    """Toroidal viscous decay in the shell 0.5 <= r <= 1 at m = 1, 2.

    A toroidal field v = Curl(T r) with T = R(r) Y_lm is divergence-free and
    pressure-free, and Lap(v) = Curl(Lap(T) r), so it decays at k^2 with
    R = a j_l(k r) + b y_l(k r).  No-slip walls need v = 0, i.e. R = 0 at both
    radii -- the same determinant as the Dirichlet scalar, and the same
    condition an insulating exterior puts on a freely decaying toroidal
    magnetic field.  Impenetrable stress-free walls need d(R/r)/dr = 0, i.e.
    k r R'(k r) = R(k r) at both radii.
    """
    kind = "dirichlet" if wall == "noslip" else "T_over_r"
    targets = [_radial_roots(kind, l, 1, ri=RI)[0] ** 2 for l in (m, m + 1)]
    spectra = _spectra(core, vector_diffusion_problem(wall), "shell", m,
                       sigma=float(np.mean(targets)), gauge=("p",))
    errs = _run(f"toroidal {wall} shell m={m}", core, targets, spectra, _rate,
                _VECTOR_DIFFUSION_TOL[core])
    _rms_order(f"toroidal {wall} shell m={m}", core, errs)


# ---------------------------------------------------------------------------
# 8. Cross-core consistency
# ---------------------------------------------------------------------------

def _acoustic_case():
    targets = [_radial_roots("neumann", l, 1, ri=RI)[0] for l in (1, 2, 0)]
    return (acoustic_problem(1.0, "shell"), "shell", 0, (), 1j * 3.0 + 0.2, None,
            targets, lambda lam, t: _omega(lam, t).real)


def _gravity_case():
    targets = [gravity_omega(1, 1), gravity_omega(2, 1)]
    return (gravity_problem(), "shell", 1, ("p",), 0.3j + 0.02, _INVISCID_RESID,
            targets, lambda lam, t: _omega(lam, t, neutral=NEUTRAL).real)


def _decay_case():
    targets = _decay_targets()
    return (scalar_diffusion_problem("dirichlet"), "shell", 1, (),
            float(np.mean(targets)), None,
            [_radial_roots("dirichlet", l, 1, ri=RI)[0] ** 2 for l in (1, 2)], _rate)


_CROSS_CORE = {"acoustic shell m=0": (_acoustic_case, 4e-3),
               "gravity shell m=1": (_gravity_case, 1e-2),
               "scalar dirichlet shell m=1": (_decay_case, 4e-3)}


@pytest.mark.parametrize("bench", sorted(_CROSS_CORE))
def test_cross_core_agreement(bench):
    """Three benchmarks read off all four cores at their finer resolution:
    every pair of cores agrees to the tolerance of the coarsest core, which is
    the staggered C-grid.  The spread is a stronger statement than each core's
    own tolerance, because a shared sign or convention error would move all
    four together while a discretisation error moves them apart.
    """
    factory, tol = _CROSS_CORE[bench]
    pb, geom, m, gauge, sigma, resid, targets, pick = factory()
    cores = [c for c in CORES if c != "dedalus" or _HAVE_DEDALUS]
    table = {}
    for core in cores:
        lam = _census(core, pb, geom, m, 1, sigma, gauge=gauge, resid=resid)
        table[core] = [pick(lam, t) for t in targets]
    for j, target in enumerate(targets):
        vals = np.array([table[c][j] for c in cores])
        spread = float(np.abs(vals - vals.mean()).max() / abs(target))
        print(f"BENCH | cross-core {bench:24s} | analytic {target:+.8f} | "
              + " ".join(f"{c} {table[c][j]:+.8f}" for c in cores)
              + f" | spread {spread:.2e}")
        assert spread < tol, f"cross-core {bench}: {target} -> {vals}"
        assert np.abs(vals - target).max() / abs(target) < tol
