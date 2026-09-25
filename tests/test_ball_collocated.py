"""Analytic benchmarks of the collocated core on the full sphere.

Every case runs at two resolutions and asserts the measured
convergence order; each test prints one table row per target
(analytic value, computed value, error, order) that `pytest -s`
collects into the benchmark table.

Conventions. `solve_evp` returns λ of ``K x = λ M x`` with K the
non-Dt terms and M the Dt terms (Path A) or the natural mass on the
named unknown (Path B, `mass_equations`). For ``c·Lap(u) = λ u``
(Path B) λ = −c κ²; for ``Dt(u) − Lap(u) = 0`` λ = +κ²; for
``Dt(v) − Ek Lap(v) = 0`` λ = +Ek k²; for the inviscid inertial system
``2 Cross(Ω, v) − Grad(p) = λ v`` the frequency is ω = −Im λ and a
physical mode is neutral, Re λ = 0.

Origin treatment: see spdee/backends/fd2/ball_origin.py and
docs/ball_collocated.md. The shell fingerprints at the end guard that
none of it touches shell grids.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sps
import sympy as sp
from scipy.optimize import brentq
from scipy.special import spherical_jn

from spdee import (
    Background, Unknown, Lap, Grad, Div, Curl, Cross, Dt, Dot,
    Equation, EquationSystem, BoundaryCondition, Boundary,
    r_hat, theta_hat, phi_hat, r, theta, ImpenetrableStressFree,
)
from spdee.backends.fd2.compiled_problem import (
    compile_problem, _gauge_quadrature_weights,
)
from spdee.backends.fd2.evp_filter import filter_spurious_eigenvalues
from spdee.backends.fd2.grad_div_sparse import build_div_vector_scalar_block
from spdee.backends.fd2.grid import make_ball_grid, make_shell_grid
from spdee.backends.fd2.layout import flatten_2d
from spdee.backends.fd2.operators import DerivativeBundle
from spdee.ir.walker import vector_unknown_spin_placeholders


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _roots(f, count, x0=0.3, step=0.2):
    """The first `count` positive roots of f found by bracket scanning."""
    roots, x = [], x0
    while len(roots) < count:
        if f(x) * f(x + step) < 0:
            roots.append(brentq(f, x, x + step))
        x += step
    return roots


def _bessel_zero(ell, n):
    """n-th positive zero of j_ell (Dirichlet)."""
    return _roots(lambda k: spherical_jn(ell, k), n + 1)[n]


def _bessel_prime_zero(ell, n):
    """n-th positive zero of j_ell' (Neumann)."""
    return _roots(lambda k: spherical_jn(ell, k, derivative=True), n + 1)[n]


def _stress_free_root(ell, n):
    """n-th positive root of k j_ell'(k) = j_ell(k): the stress-free
    condition d/dr (T/r) = 0 of a toroidal mode T(r) = j_ell(k r)."""
    f = lambda k: k * spherical_jn(ell, k, derivative=True) - spherical_jn(ell, k)
    return _roots(f, n + 1)[n]


def _order(err_coarse, err_fine, N_coarse, N_fine):
    if err_fine <= 0 or err_coarse <= 0:
        return float("inf")
    return float(np.log(err_coarse / err_fine) / np.log(N_fine / N_coarse))


def _row(case, analytic, computed, error, order):
    print(f"BENCH | {case:44s} | analytic {analytic:+.6f} | computed "
          f"{computed:+.6f} | error {error:.2e} | order {order:.2f}")


def _nearest(values, target):
    values = np.asarray(values)
    return values[int(np.argmin(np.abs(values - target)))]


# ---------------------------------------------------------------------------
# Scalar Helmholtz on the unit ball
# ---------------------------------------------------------------------------

_HELMHOLTZ_VARIANTS = {
    # name: (lhs builder, params, factor c with λ = c κ²)
    "Lap": (lambda u: Lap(u), None, -1.0),
    "-2*Lap": (lambda u: -2 * Lap(u), None, 2.0),
    "Ek*Lap": (lambda u: sp.Symbol("Ek") * Lap(u), {"Ek": 0.25}, -0.25),
    "Dt-Lap": (lambda u: Dt(u) - Lap(u), None, 1.0),
}


def _helmholtz_kappas(m, variant, bc_kind, N, order, targets):
    lhs, params, c = _HELMHOLTZ_VARIANTS[variant]
    u = Unknown.Scalar("u")
    if bc_kind == "dirichlet":
        bc = BoundaryCondition(Boundary.R_OUTER, u, name="bc")
    else:
        bc = BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, Grad(u)), name="bc")
    sys_ = EquationSystem([Equation(lhs(u), name="h")], [bc])
    grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=m, order_r=order, order_theta=order,
                         params=params)
    res = cp.solve_evp(k=10, sigma=c * targets[0] ** 2)
    lam = res.eigenvalues
    lam = lam[np.isfinite(lam)]
    kappa_sq = np.real(lam) / c
    return np.sqrt(kappa_sq[kappa_sq > 0.25])


@pytest.mark.parametrize("variant", sorted(_HELMHOLTZ_VARIANTS))
@pytest.mark.parametrize("m", [0, 1, 2])
def test_scalar_helmholtz_dirichlet(m, variant):
    """Lap(u) = −κ² u, u(1) = 0: κ with j_l(κ) = 0, l ≥ |m|. The
    coefficient and sign variants must give the same κ (they exercise
    the coefficient, sign and Dt weight of the origin anchor)."""
    ells = range(m, m + 3)
    targets = sorted([_bessel_zero(l, 0) for l in ells] + [_bessel_zero(m, 1)])[:3]
    Ns = (16, 32)
    kappas = {N: _helmholtz_kappas(m, variant, "dirichlet", N, 4, targets) for N in Ns}
    for t in targets:
        errs = [abs(_nearest(kappas[N], t) - t) / t for N in Ns]
        order = _order(errs[0], errs[1], *Ns)
        _row(f"helmholtz dirichlet m={m} {variant} kappa", t,
             float(_nearest(kappas[Ns[1]], t)), errs[1], order)
        assert errs[1] < 1e-4, f"m={m} {variant} κ={t}: rel err {errs[1]:.2e}"
        assert order >= 3.5, f"m={m} {variant} κ={t}: order {order:.2f}; errs={errs}"


@pytest.mark.parametrize("m", [0, 1, 2])
def test_scalar_helmholtz_neumann(m):
    """Lap(u) = −κ² u, ∂_r u(1) = 0: κ with j_l'(κ) = 0, l ≥ |m|
    (l=1: 2.081576, l=2: 3.342094, l=0: 4.493409, l=3: 4.514100)."""
    ells = range(max(m, 1), m + 3) if m > 0 else (1, 2, 0)
    targets = sorted(_bessel_prime_zero(l, 0) for l in ells)
    Ns = (16, 32)
    kappas = {N: _helmholtz_kappas(m, "Lap", "neumann", N, 4, targets) for N in Ns}
    for t in targets:
        errs = [abs(_nearest(kappas[N], t) - t) / t for N in Ns]
        order = _order(errs[0], errs[1], *Ns)
        _row(f"helmholtz neumann m={m} kappa", t,
             float(_nearest(kappas[Ns[1]], t)), errs[1], order)
        assert errs[1] < 1e-4, f"m={m} Neumann κ={t}: rel err {errs[1]:.2e}"
        assert order >= 3.5, f"m={m} Neumann κ={t}: order {order:.2f}; errs={errs}"


# ---------------------------------------------------------------------------
# Poisson BVP on a non-uniform radial grid
# ---------------------------------------------------------------------------

_U_SYM = sp.sin(r * sp.cos(theta)) * (1 - r ** 2)
_LAP_U_SYM = sp.simplify(
    (1 / r ** 2) * sp.diff(r ** 2 * sp.diff(_U_SYM, r), r)
    + (1 / (r ** 2 * sp.sin(theta)))
    * sp.diff(sp.sin(theta) * sp.diff(_U_SYM, theta), theta)
)
_U_FN = sp.lambdify((r, theta), _U_SYM, modules="numpy")


def _nonuniform_r(N):
    """Nodes clustered towards the outer boundary; r[0] = 0 exactly."""
    x = np.linspace(0.0, 1.0, N)
    rr = 0.6 * np.sin(0.5 * np.pi * x) + 0.4 * x
    rr[0] = 0.0
    rr[-1] = 1.0
    return rr


@pytest.mark.parametrize("order", [2, 4, 6])
def test_poisson_bvp_nonuniform_radial_grid(order):
    """Lap(u) = f with u = sin(z)(1 − r²) on a non-uniform radial grid:
    the origin rows use the actual node positions."""
    Ns = {2: (16, 32), 4: (16, 32), 6: (12, 24)}[order]
    errs = []
    for N in Ns:
        grid = make_ball_grid(N_r=N, N_theta=48, r=_nonuniform_r(N))
        u = Unknown.Scalar("u")
        f = Background.Scalar("f", expr=_LAP_U_SYM)
        bcs = [BoundaryCondition(Boundary.R_OUTER, u - _U_SYM.subs(r, 1), name="o")]
        sys_ = EquationSystem([Equation(Lap(u) - f, name="poisson")], bcs)
        cp = compile_problem(sys_, grid, m=0, order_r=order, order_theta=8)
        R, T = grid.mesh()
        errs.append(float(np.max(np.abs(cp.solve_bvp()["u"] - _U_FN(R, T)))))
    rate = _order(errs[0], errs[1], *Ns)
    _row(f"poisson nonuniform order={order} max|u-u_exact|", 0.0, errs[1], errs[1], rate)
    assert rate >= order - 0.5, f"order={order}: rate {rate:.2f}; errs={errs}"


# ---------------------------------------------------------------------------
# Toroidal decay modes with stress-free walls
# ---------------------------------------------------------------------------

_EK = 0.01


def _toroidal_problem(m, N):
    v = Unknown.Vector("v")
    eq = Equation(Dt(v) - _EK * Lap(v), name="mom")
    sys_ = EquationSystem([eq], ImpenetrableStressFree(v, boundary=Boundary.R_OUTER))
    grid = make_ball_grid(N_r=N, N_theta=N, r_outer=1.0)
    return v, grid, compile_problem(sys_, grid, m=m, order_r=6, order_theta=6)


@pytest.mark.parametrize("m", [0, 1, 2])
def test_toroidal_stress_free_decay_modes(m):
    """Dt(v) = Ek Lap(v) with impenetrable stress-free walls. A toroidal
    field v = ∇T × r with T = j_l(k r) Y_l^m has Lap(v) = ∇(Lap T) × r,
    so it decays at λ = Ek k²; v_r = 0 holds identically and the
    tangential stress ∂_r(v_θ/r) = ∂_r(v_φ/r) = 0 at r = 1 requires
    (T/r)' = 0, i.e. k j_l'(k) = j_l(k) (l=1: 5.763459, 9.095011;
    l=2: 2.501133, 7.136009; l=3: 3.864700). The fine-grid eigenvectors
    are checked to have v_r = 0 and to be divergence-free (r·Div v from
    the Div block, O(h^6) of |v|)."""
    Ns = (12, 24)
    ells = (max(m, 1), max(m, 1) + 1)
    for l in ells:
        k = _stress_free_root(l, 0)
        target = _EK * k ** 2
        errs = []
        for N in Ns:
            v, grid, cp = _toroidal_problem(m, N)
            res = cp.solve_evp(k=6, sigma=target, return_eigenvectors=True)
            lam = np.where(np.isfinite(res.eigenvalues), res.eigenvalues, np.inf)
            j = int(np.argmin(np.abs(lam - target)))
            errs.append(abs(lam[j].real - target) / target)
            assert abs(lam[j].imag) < 1e-10 * target
            if N != Ns[-1]:
                continue
            vec = res.eigenvectors[j]["v"]
            state = np.concatenate([flatten_2d(vec[-1]), flatten_2d(vec[0]),
                                    flatten_2d(vec[+1])])
            bundle = DerivativeBundle(grid, order_r=6, order_theta=6)
            meta = {"m": m, "v_placeholders": {"v": vector_unknown_spin_placeholders(v)}}
            div = build_div_vector_scalar_block(meta, bundle, m) @ state
            scale = float(np.abs(state).max())
            assert np.abs(vec[0]).max() < 1e-5 * scale, "v_r of a toroidal mode"
            assert np.abs(div).max() < 2e-3 * scale, (
                f"m={m} l={l} N={N}: |r Div v| = {np.abs(div).max():.2e} "
                f"vs |v| = {scale:.2e}"
            )
        order = _order(errs[0], errs[1], *Ns)
        _row(f"toroidal stress-free m={m} l={l} lambda=Ek k^2", target,
             target * (1 + errs[1]), errs[1], order)
        assert errs[1] < 1e-4, f"m={m} l={l}: rel err {errs[1]:.2e}"
        assert order >= 4.0, f"m={m} l={l}: order {order:.2f}; errs={errs}"


# ---------------------------------------------------------------------------
# Inviscid inertial modes (Zhang et al. 2001)
# ---------------------------------------------------------------------------

def _inertial_system():
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Omega = Background.Vector(
        "Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    eqs = [Equation(2 * Cross(Omega, v) - Grad(p), name="momentum"),
           Equation(Div(v), name="continuity")]
    bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")]
    return EquationSystem(eqs, bcs, gauge_unknowns=["p"])


# (ω, tolerance) from tests/test_evp_zhang_2001.py, the three lowest |ω|
# antisymmetric N=1,2 targets per m.
_INERTIAL_TARGETS = {
    1: [(-0.82001, 1e-4), (0.61198, 1e-4), (-0.53733, 1e-4)],
    2: [(-0.76334, 1e-4), (0.46690, 1e-4), (0.35954, 1e-4)],
}


@pytest.mark.parametrize("m", [1, 2])
def test_inertial_modes(m):
    """The curated Zhang targets are recovered by the twin-grid filter at
    the tolerances of test_evp_zhang_2001.py on 24×48, the error
    decreases from 16×32 (the measured order is reported; order ≥ 2 is
    asserted unless both errors are already under 1e-5, the floor set
    by the twin-grid tolerance and the ARPACK precision, where the
    order is noise) and the recovered mode is neutral (|Re λ| < 1e-8).

    Spurious modes: the collocated Grad/Div pair is not discretely
    adjoint, so the in-band spectrum also holds non-neutral eigenvalues
    (|Re λ| ~ 1e-3 .. 1e-2, drifting with N; see
    tests/test_checkerboard_probe.py). They are identified by
    |Re λ| > 1e-8 and are the only eigenvalues in a target's window that
    the filter flags; their count is reported.
    """
    grids = ((16, 32), (24, 48))
    for omega, tol in _INERTIAL_TARGETS[m]:
        errs = []
        for (N_r, N_theta) in grids:
            phys, spur_base, _ = filter_spurious_eigenvalues(
                _inertial_system(),
                grid_factory=lambda nr, nt: make_ball_grid(N_r=nr, N_theta=nt),
                N_r_base=N_r, N_theta_base=N_theta,
                m=m, k=8, sigma=-1j * omega - 0.001,
                mass_equations={"momentum": "v"},
                order_r=6, order_theta=6, dN=2, rel_tolerance=2e-3,
            )
            assert len(phys) > 0
            best = min(phys, key=lambda x: abs(-x.imag - omega))
            errs.append(abs(-best.imag - omega) / abs(omega))
            assert abs(best.real) < 1e-8, f"m={m} ω={omega}: not neutral, λ={best}"
            non_neutral = [x for x in spur_base if abs(x.real) > 1e-8]
            neutral_spurious = [x for x in spur_base if abs(x.real) <= 1e-8]
            print(f"BENCH | inertial m={m} omega={omega:+.5f} N=({N_r},{N_theta}): "
                  f"{len(non_neutral)} non-neutral spurious, "
                  f"{len(neutral_spurious)} neutral drifting")
        order = _order(errs[0], errs[1], grids[0][0], grids[1][0])
        _row(f"inertial m={m} omega", omega, omega * (1 + errs[1]), errs[1], order)
        assert errs[1] < tol, f"m={m} ω={omega}: rel err {errs[1]:.2e} > {tol}"
        assert order >= 2.0 or max(errs) < 1e-5, \
            f"m={m} ω={omega}: order {order:.2f}; errs={errs}"


# ---------------------------------------------------------------------------
# Gauge row and origin rows
# ---------------------------------------------------------------------------

def _p_rows(cp):
    cm = cp.matrices
    grid = cm.grid
    off = cm.layout.entry("p").offset
    N_r, N_theta = grid.N_r, grid.N_theta
    return off, N_r, N_theta


@pytest.mark.parametrize("m", [0, 2])
def test_gauge_row_and_origin_rows(m):
    """Dt(v) + Grad(p) = 0, Div(v) = 0 with p a gauge unknown. At m = 0
    the gauge row sits at (i_r = N_r−2, j = 0) of the continuity block
    (whose wall row hosts Dot(r̂, v) = 0) with the r² sin θ quadrature
    weights and no entry at r = 0, while the i_r = 0 rows of the p block
    are the N_θ−1 ties p(0, θ_j) = p(0, θ_0) plus the anchor
    Div v(0) = 0. At m = 2 no gauge row is installed (a constant is not
    an m ≠ 0 field) and the i_r = 0 rows are p(0, θ_j) = 0."""
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    eqs = [Equation(Dt(v) + Grad(p), name="momentum"), Equation(Div(v), name="continuity")]
    bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr")]
    grid = make_ball_grid(N_r=12, N_theta=10)
    cp = compile_problem(EquationSystem(eqs, bcs, gauge_unknowns=["p"]), grid, m=m)
    K, M = cp.assemble_evp_matrices()
    off, N_r, N_theta = _p_rows(cp)
    A = cp.matrices.A_total.tocsr()
    gauge_row = off + (N_r - 2)
    row = K[gauge_row]
    if m == 0:
        for j in range(N_theta):
            wall = K[off + j * N_r + N_r - 1]
            assert wall.nnz == 1 and wall.data[0] == 1.0, "r_hat.v not on continuity"
            assert wall.indices[0] == cp.matrices.layout.entry("v").offset \
                + N_r * N_theta + j * N_r + N_r - 1
        w_r, w_t = _gauge_quadrature_weights(grid.r, grid.theta)
        R, T = grid.mesh()
        w_2d = w_r[:, None] * w_t[None, :] * R ** 2 * np.sin(T)
        expected = flatten_2d(w_2d)
        got = np.zeros(N_r * N_theta)
        got[row.indices - off] = row.data.real
        assert np.allclose(got, expected, rtol=1e-13, atol=0)
        assert not np.any((row.indices - off) % N_r == 0), "gauge weight at r=0"
        assert M[gauge_row].nnz == 0
        for j in range(1, N_theta):
            tie = K[off + j * N_r]
            assert tie.nnz == 2 and set(tie.indices) == {off + j * N_r, off}
            assert M[off + j * N_r].nnz == 0
        anchor = K[off]
        assert anchor.nnz > 2 and off not in anchor.indices, "anchor overwritten"
        assert all(c < off for c in anchor.indices), "anchor is Div v(0) on v columns"
    else:
        assert (row != A[gauge_row]).nnz == 0, "gauge row installed at m != 0"
        for j in range(N_theta):
            rr = K[off + j * N_r]
            assert rr.nnz == 1 and rr.indices[0] == off + j * N_r and rr.data[0] == 1.0
            assert M[off + j * N_r].nnz == 0


@pytest.mark.parametrize("m,p_sym", [(0, r ** 2), (2, r ** 2 * sp.sin(theta) ** 2)])
def test_gauge_bvp(m, p_sym):
    """Lap(v) + Grad(p) = Grad(p_exact), Div(v) = 0, no-slip: v = 0 and
    p = p_exact. At m = 0 the returned p satisfies the discrete gauge
    Σ w p = 0 to round-off, is constant at r = 0 and equals p_exact up
    to its mean to round-off (the wall-normal momentum rows see the
    r-sawtooth, so the checkerboard is not a null mode); at m = 2 (no
    gauge row) v and p converge to the exact solution."""
    Ns = (12, 24)
    errs_v, errs_p = [], []
    for N in Ns:
        v = Unknown.Vector("v")
        p = Unknown.Scalar("p")
        g = Background.Vector(
            "g", v_r=sp.diff(p_sym, r), v_theta=sp.simplify(sp.diff(p_sym, theta) / r),
            v_phi=sp.simplify(sp.I * m * p_sym / (r * sp.sin(theta))),
        )
        eqs = [Equation(Lap(v) + Grad(p) - g, name="momentum"),
               Equation(Div(v), name="continuity")]
        bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(u, v), name=n)
               for u, n in ((r_hat, "vr"), (theta_hat, "vt"), (phi_hat, "vp"))]
        grid = make_ball_grid(N_r=N, N_theta=N)
        cp = compile_problem(EquationSystem(eqs, bcs, gauge_unknowns=["p"]), grid, m=m)
        res = cp.solve_bvp()
        R, T = grid.mesh()
        p_ex = sp.lambdify((r, theta), p_sym, modules="numpy")(R, T) + 0 * R
        errs_v.append(max(float(np.abs(res["v"][s]).max()) for s in (-1, 0, 1)))
        if m == 0:
            w_r, w_t = _gauge_quadrature_weights(grid.r, grid.theta)
            w_2d = w_r[:, None] * w_t[None, :] * R ** 2 * np.sin(T)
            assert abs((w_2d * res["p"]).sum()) < 1e-12 * np.abs(w_2d * res["p"]).sum()
            assert np.abs(res["p"][0, :] - res["p"][0, 0]).max() < 1e-12
            assert errs_v[-1] < 1e-9
            p_ex = p_ex - (w_2d * p_ex).sum() / w_2d.sum()
            errs_p.append(float(np.abs(res["p"] - p_ex).max()))
            assert errs_p[-1] < 1e-9, errs_p
        else:
            errs_p.append(float(np.abs(res["p"] - p_ex).max()))
    if m == 0:
        _row("gauge bvp m=0 max|v|", 0.0, errs_v[1], errs_v[1], float("inf"))
        _row("gauge bvp m=0 max|p-p_exact| (no checkerboard)", 0.0, errs_p[1],
             errs_p[1], float("inf"))
    else:
        order_v = _order(errs_v[0], errs_v[1], *Ns)
        order_p = _order(errs_p[0], errs_p[1], *Ns)
        _row("gauge bvp m=2 max|v|", 0.0, errs_v[1], errs_v[1], order_v)
        _row("gauge bvp m=2 max|p-p_exact|", 0.0, errs_p[1], errs_p[1], order_p)
        assert errs_v[1] < 1e-5 and errs_p[1] < 1e-4
        assert order_v >= 3.5 and order_p >= 3.5, (errs_v, errs_p)


# ---------------------------------------------------------------------------
# Shell grids are untouched
# ---------------------------------------------------------------------------

_SHELL_GRID = make_shell_grid(N_r=10, N_theta=12, r_inner=0.4, r_outer=1.0)


def _shell_scalar(coef):
    p = Unknown.Scalar("p")
    f = Background.Scalar("f", expr=sp.sin(theta) * r)
    eq = Equation(coef * Lap(p) - f, name="poisson")
    bcs = [BoundaryCondition(Boundary.R_INNER, p, name="in"),
           BoundaryCondition(Boundary.R_OUTER, p, name="out")]
    return EquationSystem([eq], bcs)


def _shell_scalar_dt():
    p = Unknown.Scalar("p")
    eq = Equation(Dt(p) - 2 * Lap(p) + p, name="heat")
    bcs = [BoundaryCondition(Boundary.R_INNER, p, name="in"),
           BoundaryCondition(Boundary.R_OUTER, p, name="out")]
    return EquationSystem([eq], bcs)


def _shell_vp():
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Om = Background.Vector("Omega", v_r=sp.cos(theta), v_theta=-sp.sin(theta),
                           v_phi=sp.S.Zero)
    mom = Dt(v) + 2 * Cross(Om, v) + Grad(p) - 1e-2 * Lap(v)
    eqs = [Equation(mom, name="momentum"), Equation(Div(v), name="continuity")]
    bcs = [BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
           BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro")]
    bcs += ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)[1:]
    return EquationSystem(eqs, bcs, gauge_unknowns=["p"])


def _shell_vec_helm():
    v = Unknown.Vector("v")
    eq = Equation(Lap(v) + Curl(v), name="eq")
    bcs = [BoundaryCondition(loc, Dot(u, v), name=f"{n}{i}")
           for i, loc in enumerate((Boundary.R_OUTER, Boundary.R_INNER))
           for u, n in ((r_hat, "a"), (theta_hat, "b"), (phi_hat, "c"))]
    return EquationSystem([eq], bcs)


def _fingerprint(M):
    M = sps.csr_matrix(M)
    M.sum_duplicates()
    co = M.tocoo()
    return (int(M.nnz), complex(co.data.sum()), float(np.abs(co.data).sum()),
            complex((co.data * (co.row + 2.0 * co.col + 1)).sum()))


# Fingerprints (nnz, Σ, Σ|·|, Σ weighted by row + 2 col + 1) of the shell
# matrices compiled before the ball origin treatment was rewritten; the
# K/M of the v-p system are those of the equation-hosted BC rows (r̂·v on
# the continuity wall rows, gauge row at i_r = 1). Path-B K/M of m ≠ 0
# gauge systems are not listed: their gauge row is no longer installed.
_SHELL_FINGERPRINTS = {
    ("scalar", 0, 2, "A"): (644, 2.8421709430404007e-13, 89108.32762766336, 5039.999999999884),
    ("scalar", 0, 2, "K"): (520, 24.000000000000753, 50448.213902731004, 8340.000000000064),
    ("scalar", 0, 2, "M"): (96, 49.280000000000015, 49.280000000000015, 8986.880000000001),
    ("scalar_neg", 2, 4, "A"): (1168, 11519.999999999953, 490194.7264632238, 2057759.9999999856),
    ("vp", 0, 2, "A"): (5236, 271.2000000000003, 8449.621550223399, 178453.74976503637 + 41085.223871986615j),
    ("vp", 0, 2, "B"): (360, 189.60000000000005, 189.60000000000005, 103120.80000000002),
    ("vp", 0, 2, "K"): (5053, 215.2340957819944 - 4.973799150320701e-14j, 8509.273745239036,
                        146890.35456675914 + 42557.82830681185j),
    ("vp", 0, 2, "M"): (336, 165.60000000000002, 165.60000000000005, 89848.80000000002),
    ("vp", 0, 4, "A"): (9272, 271.2 - 2.1316282072803006e-14j, 19438.345128485642, 178453.74976503613 + 41085.22387198661j),
    ("vp", 0, 4, "K"): (8861, 215.23409578199426 - 7.815970093361102e-14j, 18652.893415461876,
                        146890.354566759 + 42557.82830681186j),
    ("vec_helm", 1, 4, "A"): (5904, -7439.999999999965 - 8.526512829121202e-14j, 738181.0091090108, -4208203.55615011 + 60479.99999999988j),
    ("scalar_dt", 1, 2, "A"): (644, 2943.1999999999935, 180132.93881484523, 518501.59999999806),
    ("scalar_dt", 1, 2, "B"): (120, 63.20000000000002, 63.20000000000002, 11621.600000000002),
}
_SHELL_B_FINGERPRINTS = {
    ("scalar", 0, 2): (32.17744981726965, 32.17744981726964),
    ("scalar_neg", 2, 4): (32.17744981726965, 32.17744981726964),
}
_SHELL_SYSTEMS = {
    "scalar": (lambda: _shell_scalar(1), None),
    "scalar_neg": (lambda: _shell_scalar(-2), None),
    "scalar_dt": (_shell_scalar_dt, None),
    "vp": (_shell_vp, None),
    "vec_helm": (_shell_vec_helm, {"eq": "v"}),
}


@pytest.mark.parametrize("name,m,order", sorted({k[:3] for k in _SHELL_FINGERPRINTS}))
def test_shell_matrices_unchanged(name, m, order):
    """Compiled and assembled shell matrices match the pre-rewrite
    fingerprints: the ball treatment does not touch shell grids."""
    factory, mass_eq = _SHELL_SYSTEMS[name]
    cp = compile_problem(factory(), _SHELL_GRID, m=m, order_r=order, order_theta=order)
    cm = cp.matrices
    mats = {"A": cm.A_total, "B": cm.B_total}
    if any(k[3] in ("K", "M") for k in _SHELL_FINGERPRINTS if k[:3] == (name, m, order)):
        mats["K"], mats["M"] = cp.assemble_evp_matrices(mass_equations=mass_eq)
    for key, ref in _SHELL_FINGERPRINTS.items():
        if key[:3] != (name, m, order):
            continue
        got = _fingerprint(mats[key[3]])
        assert got[0] == ref[0], f"{key}: nnz {got[0]} != {ref[0]}"
        for a, b in zip(got[1:], ref[1:]):
            assert abs(a - b) <= 1e-12 * max(abs(b), 1.0), f"{key}: {got} != {ref}"
    if (name, m, order) in _SHELL_B_FINGERPRINTS:
        ref = _SHELL_B_FINGERPRINTS[(name, m, order)]
        b = cm.b_total
        assert abs(complex(b.sum()) - ref[0]) <= 1e-12 * abs(ref[0])
        assert abs(float(np.abs(b).sum()) - ref[1]) <= 1e-12 * abs(ref[1])
